"""
Utilities for running code in parallel, either via multiple Python threads or
multiple processes.
"""


# TODO: handle keyboard interrupt e.g. http://stackoverflow.com/a/11436603


from __future__ import division

from copy import copy
from functools import reduce
import queue
import threading
import time

from pisa import OMP_NUM_THREADS
from pisa.utils.log import logging, set_verbosity


__all__ = ['parallel_run']

__author__ = 'J.L. Lanfranchi'

__license__ = '''Copyright (c) 2014-2017, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


def wrapper(func, retvals_queue, chunk):
    """Wrapper function that calls `func` and places a dict containging the
    args, kwargs, and return value on a shared queue such that the returned
    values can be returned to the user.

    Parameters
    ----------
    func : callable
    retvals_queue : queue.Queue
    chunk : dict containing {'args': (...), 'kwargs': {...}}

    """
    return_value = func(*chunk['args'], **chunk['kwargs'])
    retvals_queue.put(dict(indices=chunk['indices'],
                           return_value=return_value))


def parallel_run(func, kind, num_parallel, scalar_func, divided_args_mask,
                 divided_kwargs_names, *args, **kwargs):
    """Run a function in parallel, either in separate threads or processes.

    Parameters
    ----------
    func : callable
        The function or method to be run in parallel.

    kind : string, either 'threads' or 'processes'
        Whether to parallelize with threads (via `threading` module) or
        processes (via `multiprocessing` module).

        'threads' is more lightweight and always useful if IO is the
        bottleneck. However, unless `func` can bypass Python's global
        interpreter lock (GIL), processing is _not_ run in parallel when using
        'threads'. (Note that, e.g., Numba can sidestep the GIL if compiled
        with `nogil=True`.)

        'processes' entails forking the application (including its memory
        contents) into separate processes, each of which is free to run
        entirely in parallel. Besides the expense of this operation, the other
        drawback to 'processes' is that all objects passed into and out of
        `func` must be pickle-able, as they are passed in and out after
        serializiing/deserializing using pickle.

    num_parallel : int or None
        Number of parallel threads or processes to start. If None, default is
        `pisa.OMP_NUM_THREADS` (settable via `OMP_NUM_THREADS` environment
        variable).

    scalar_func : bool
        If True, then func can only handle one of each divided arg and kwarg
        at a time.

    divided_args_mask : None or sequence of length(args) containing bools
        Positional arguments corresponding to `True` values are to be
        divided among workers; other arguments are passed as-is to workers.

    divided_kwargs_names : None or sequence of strings
        Names of keyword arguments that are to be divided among workers.
        `kwargs` not named are passed as-is to workers.

    *args
        Positional args passed (divided or not) to `func`. If
        `divided_args_mask` is not None, the number of `args` must be the same
        as the number of bools in `divided_args_mask`.

    **kwargs
        Keyword args passed (divided or not) to `func`.

    Returns
    -------
    out_queue : threading.Queue
        Outputs returned by the calls to `func`.

    Raises
    ------
    ValueError
        Illegal inputs; e.g., if `iivide_args_mask` or `divided_kwargs_names`
        are incompatible with the *args or **kwargs passed.

    """
    valid_kinds = ['threads', 'processes']
    if not kind in valid_kinds:
        raise ValueError(
            '`kind` must either be one of {%s}, but got "%s".'
            % (', '.join(['"%s"' % k for k in valid_kinds]), kind)
        )

    if divided_args_mask and len(args) != len(divided_args_mask):
        raise ValueError('Number of `divided_args_mask` elements (%d) does not'
                         ' match number of args (%d)'
                         % (len(divided_args_mask), len(args)))

    if divided_kwargs_names:
        diff = set(divided_kwargs_names).difference(kwargs.keys())
        if len(diff) > 0:
            raise ValueError('Excess names in `divided_kwargs_names`: %s'
                             % ', '.join(['"%s"' % s for s in sorted(diff)]))

    if num_parallel is None:
        num_parallel = OMP_NUM_THREADS

    if args:
        parg_lengths = [len(a) for m, a in zip(divided_args_mask, args) if m]
    else:
        parg_lengths = []

    if kwargs:
        pkwarg_lengths = {n: len(kwargs[n]) for n in divided_kwargs_names}
    else:
        pkwarg_lengths = {}

    unique_lengths = set(parg_lengths + pkwarg_lengths.values())
    if len(unique_lengths) != 1:
        for length in sorted(unique_lengths):
            parglist = []
            for argnum, (mask, arg) in enumerate(zip(divided_args_mask, args)):
                if not mask:
                    continue
                if len(arg) == length:
                    parglist.append(argnum)
            if len(parglist) > 0:
                pargstr = '`pargs` ' + ', '.join([str(i) for i in parglist])
            else:
                pargstr = ''

            pkwarglist = []
            for name in divided_kwargs_names:
                if len(kwargs[name]) == length:
                    pkwarglist.append(name)

            if len(pkwarglist) > 0:
                if pargstr > 0:
                    pkwargstr = ' and '
                pkwargstr += ', '.join(["'%s'" % key for key in pkwarglist])

            if len(parglist) + len(pkwarglist) > 1:
                has_str = 'all have'
            else:
                has_str = 'has'
            logging.error('%s%s %s length %d', pargstr, pkwargstr, has_str,
                          length)

        raise ValueError('Each divided arg and each divided keyword arg must'
                         ' have the same length as all the others.')

    num_items = unique_lengths.pop()

    if scalar_func:
        uniform_chunksize = 0
        singles = num_items
        batches = 1 + num_items // num_parallel
        # Selects out the (single) scalar argument from the sliced list (of
        # length one)
        subselector = 0
    else:
        uniform_chunksize, singles = divmod(num_items, num_parallel)
        batches = 1
        # Selects the full list of arguments
        subselector = slice(None)

    #
    # Divvy up the work (i.e. args and kwargs)
    #

    item_num = 0
    for batch_num in range(batches):
        if uniform_chunksize == 0 and singles == 0:
            break

        items_in_batch = 0
        chunks = []
        for worker in range(num_parallel):
            if uniform_chunksize == 0 and singles == 0:
                break

            n_in_chunk = uniform_chunksize
            if singles > 0:
                n_in_chunk += 1
                singles -= 1

            start_ind = item_num
            end_ind = item_num + n_in_chunk
            chunk_slice = slice(start_ind, end_ind)

            worker_args = []
            if args:
                for mask, arg in zip(divided_args_mask, args):
                    if mask:
                        worker_args.append(arg[chunk_slice][subselector])
                    else:
                        worker_args.append(arg)

            worker_kwargs = {}
            for name, val in kwargs.items():
                if name in divided_kwargs_names:
                    worker_kwargs[name] = val[chunk_slice][subselector]
                else:
                    worker_kwargs[name] = val

            chunk = dict(indices=list(range(start_ind, end_ind)),
                         args=tuple(worker_args),
                         kwargs=worker_kwargs)
            chunks.append(chunk)
            items_in_batch += n_in_chunk
            item_num += n_in_chunk

        if kind == 'threads':
            logging.trace(
                'Batch %4d: launching %d threads to process a total of %d'
                ' items', batch_num + 1, len(chunks), items_in_batch
            )

            return_values = queue.Queue()
            threads = []
            for chunk in chunks:
                thread = threading.Thread(
                    target=wrapper,
                    kwargs=dict(func=func, retvals_queue=return_values,
                                chunk=chunk)
                )
                thread.daemon = True
                threads.append(thread)

            for thread in threads:
                thread.start()

            while len(threads) > 0:
                time.sleep(0.5)
                for thread in copy(threads):
                    if not thread.isAlive():
                        threads.remove(thread)
                    time.sleep(0.01)

        elif kind == 'processes':
            raise NotImplementedError()

    return return_values


def test_parallel_run():
    """Unit test the parallel_run function"""
    def delay(sec):
        """delay test func"""
        if isinstance(sec, (float, int)):
            time.sleep(sec)
        else:
            for s in sec:
                time.sleep(s)

    times = [0.01 for i in range(100)]
    serial_time = reduce(lambda x, y: x + y, times, 0)
    parallel_time = times[-1]
    num_parallel = 4

    num_batches = len(times) // num_parallel + 1

    ideal_time = serial_time / num_parallel

    t0 = time.time()
    retval_queue = parallel_run(
        func=delay, kind='threads', num_parallel=num_parallel,
        divided_args_mask=None, divided_kwargs_names=['sec'],
        scalar_func=False, sec=times
    )
    logging.trace('entry pulled from queue: %s', retval_queue.get())
    runtime = time.time() - t0

    speedup = serial_time / runtime
    logging.trace('serial runtime = %.3f s', serial_time)
    logging.trace('ideal runtime  = %.3f s', ideal_time)
    logging.trace('actual runtime = %.3f s', runtime)

    logging.trace('ideal speedup  = %.3f', serial_time / ideal_time)
    logging.trace('actual speedup = %.3f', speedup)

    relative_speedup = ideal_time / runtime
    logging.trace('speedup/ideal = %.3f', relative_speedup)
    assert relative_speedup >= 0.3, 'rel speedup = %.4f' % relative_speedup
    logging.info('<< PASS : test_parallel_run >>')


if __name__ == '__main__':
    set_verbosity(1)
    test_parallel_run()
