#!/usr/bin/env python

"""
Multiple implementations of sum-of-gaussians for compatibility and speed
"""


from __future__ import absolute_import, division

from collections.abc import Iterable
from collections import OrderedDict
from math import exp, sqrt
import threading
from time import time

import numpy as np
from scipy import stats

from pisa import FTYPE, OMP_NUM_THREADS, NUMBA_CUDA_AVAIL, numba_jit
from pisa.utils.comparisons import recursiveEquality
from pisa.utils.log import logging, set_verbosity, tprofile


# TODO: if the Numba CUDA functions are defined, then other CUDA (e.g. pycuda)
# code doesn't run (possibly only when Nvidia driver is set to
# process-exclusive or thread-exclusive mode). Need to fix this behavior. (E.g.
# context that gets destroyed?)


__all__ = ['GAUS_IMPLEMENTATIONS', 'gaussians', 'test_gaussians']

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


GAUS_IMPLEMENTATIONS = ('singlethreaded', 'multithreaded')
if NUMBA_CUDA_AVAIL:
    GAUS_IMPLEMENTATIONS += ('cuda',)

PI = FTYPE(np.pi)
TWOPI = FTYPE(2*np.pi)
SQRTPI = FTYPE(sqrt(np.pi))
SQRT2PI = FTYPE(sqrt(2*np.pi))
PISQ = FTYPE(np.pi*np.pi)


def gaussians(x, mu, sigma, weights=None, implementation=None, **kwargs):
    """Sum of multiple Gaussian curves, normalized to have area of 1.

    Parameters
    ----------
    x : array
        Points at which to evaluate the sum of Gaussians

    mu : arrays
        Means of the Gaussians to accumulate

    sigma : array
        Standard deviations of the Gaussians to accumulate

    weights : array or None
        Weights given to each Gaussian

    implementation : None or string
        One of 'singlethreaded', 'multithreaded', or 'cuda'. Passing None, the
        function will try to determine which of the implementations is best to
        call.

    kwargs
        Passed on to the underlying implementation

    Returns
    -------
    outbuf : array
        Resulting sum of Gaussians

    Notes
    -----
    This function dynamically calls an appropriate implementation depending
    upon the problem size, the hardware available (multi-core CPU or a GPU),
    and whether the user specifies `implementation`.

    """
    # TODO: figure out which is the roughly the best function to call, if more
    # than one function is available to call (i.e., really small problems are
    # fastest in single core, really large `x` are probably best on GPU, and
    # if/where multithreaded CPU beats GPU is still up in the air
    if implementation is not None:
        implementation = implementation.strip().lower()
        if not implementation in GAUS_IMPLEMENTATIONS:
            raise ValueError('`implementation` must be one of %s'
                             % GAUS_IMPLEMENTATIONS)

    # Extract a 'threads' kwarg if it's present, or default to OMP_NUM_THREADS
    if 'threads' in kwargs:
        threads = kwargs.pop('threads')
    else:
        threads = OMP_NUM_THREADS

    # Convert all inputs to arrays of correct datatype
    if not isinstance(x, Iterable):
        x = [x]
    if not isinstance(mu, Iterable):
        mu = [mu]
    if not isinstance(sigma, Iterable):
        sigma = [sigma]
    if weights is None:
        use_weights = False
        weights = [-1]
    else:
        use_weights = True
        if not isinstance(weights, Iterable):
            weights = [weights]
    x = np.asarray(x, dtype=FTYPE)
    mu = np.asarray(mu, dtype=FTYPE)
    inv_sigma = 1/np.asarray(sigma, dtype=FTYPE)
    inv_sigma_sq = -0.5 * inv_sigma * inv_sigma
    weights = np.asarray(weights, dtype=FTYPE)

    n_points = len(x)
    n_gaussians = len(mu)

    # Normalization is computed here regardless of implementation
    if use_weights:
        norm = 1/(SQRT2PI * np.sum(weights))
    else:
        norm = 1/(SQRT2PI * n_gaussians)

    # Instantiate an empty output buffer
    outbuf = np.empty(shape=n_points, dtype=FTYPE)

    # Default to CUDA since it's generally fastest
    if implementation == 'cuda' or (implementation is None
                                    and NUMBA_CUDA_AVAIL):
        logging.trace('Using CUDA Gaussians implementation')
        _gaussians_cuda(outbuf, x, mu, inv_sigma, inv_sigma_sq, weights,
                        n_gaussians, **kwargs)

    # Use singlethreaded version if `threads` is 1
    elif (implementation == 'singlethreaded'
          or implementation is None and threads == 1):
        logging.trace('Using single-threaded Gaussians implementation')
        _gaussians_singlethreaded(
            outbuf=outbuf, x=x, mu=mu, inv_sigma=inv_sigma,
            inv_sigma_sq=inv_sigma_sq, weights=weights,
            n_gaussians=n_gaussians, start=0, stop=n_points, **kwargs
        )

    # Use multithreaded version otherwise
    elif implementation == 'multithreaded' or threads > 1:
        logging.trace('Using multi-threaded Gaussians implementation')
        _gaussians_multithreaded(
            outbuf=outbuf, x=x, mu=mu, inv_sigma=inv_sigma,
            inv_sigma_sq=inv_sigma_sq, weights=weights,
            n_gaussians=n_gaussians, threads=threads, **kwargs
        )

    else:
        raise ValueError(
            'Unhandled value(s): `implementation`="%s"; note: threads="%s"'
            ' and NUMBA_CUDA_AVAIL="%s"'
            % (implementation, threads, NUMBA_CUDA_AVAIL)
        )

    # Now apply the normalization
    return outbuf * norm


def _gaussians_multithreaded(outbuf, x, mu, inv_sigma, inv_sigma_sq, weights,
                             n_gaussians, threads=OMP_NUM_THREADS):
    """Sum of multiple Gaussians, optimized to be run in multiple threads. This
    dispatches the single-kernel threaded """
    n_points = len(x)
    chunklen = n_points // OMP_NUM_THREADS
    threads = []
    start = 0
    for i in range(OMP_NUM_THREADS):
        stop = n_points if i == (OMP_NUM_THREADS - 1) else start + chunklen
        thread = threading.Thread(
            target=_gaussians_singlethreaded,
            args=(outbuf, x, mu, inv_sigma, inv_sigma_sq, weights, n_gaussians,
                  start, stop)
        )
        thread.start()
        threads.append(thread)
        start += chunklen

    for thread in threads:
        thread.join()


@numba_jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _gaussians_singlethreaded(outbuf, x, mu, inv_sigma, inv_sigma_sq, weights,
                              n_gaussians, start, stop):
    """Sum of multiple Gaussians, optimized to be run in a single thread"""
    if weights[0] != -1:
        assert len(weights) == n_gaussians
        for i in range(start, stop):
            tot = 0.0
            for j in range(n_gaussians):
                xlessmu = x[i] - mu[j]
                tot += (
                    exp((xlessmu*xlessmu) * inv_sigma_sq[j])
                    * weights[j] * inv_sigma[j]
                )
            outbuf[i] = tot
    else:
        assert len(weights) == 1
        for i in range(start, stop):
            tot = 0.0
            for j in range(n_gaussians):
                xlessmu = x[i] - mu[j]
                tot += (
                    exp((xlessmu*xlessmu)*inv_sigma_sq[j])*inv_sigma[j]
                )
            outbuf[i] = tot


if NUMBA_CUDA_AVAIL:
    from numba import cuda

    def _gaussians_cuda(outbuf, x, mu, inv_sigma, inv_sigma_sq, weights,
                        n_gaussians):
        n_points = len(x)

        use_weights = True
        if weights[0] == -1:
            assert len(weights) == 1
            use_weights = False

        threads_per_block = 32
        blocks_per_grid = (
            (n_points + (threads_per_block - 1)) // threads_per_block
        )

        # Create empty array on GPU to store result
        d_outbuf = cuda.device_array(shape=n_points, dtype=FTYPE, stream=0)

        # Copy argument arrays to GPU
        d_x = cuda.to_device(x)
        d_mu = cuda.to_device(mu)
        d_inv_sigma = cuda.to_device(inv_sigma)
        d_inv_sigma_sq = cuda.to_device(inv_sigma_sq)
        if use_weights:
            assert len(weights) == n_gaussians
            d_weights = cuda.to_device(weights)
            func = _gaussians_weighted_cuda_kernel[blocks_per_grid, # pylint: disable=unsubscriptable-object
                                                   threads_per_block]
            func(d_outbuf, d_x, d_mu, d_inv_sigma, d_inv_sigma_sq, d_weights,
                 n_gaussians)
        else:
            d_weights = None
            func = _gaussians_cuda_kernel[blocks_per_grid, threads_per_block] # pylint: disable=unsubscriptable-object
            func(d_outbuf, d_x, d_mu, d_inv_sigma, d_inv_sigma_sq, n_gaussians)

        # Copy contents of GPU result to host's outbuf
        d_outbuf.copy_to_host(ary=outbuf, stream=0)

        del d_x, d_mu, d_inv_sigma, d_inv_sigma_sq, d_weights, d_outbuf

    @cuda.jit(inline=True, fastmath=True)
    def _gaussians_cuda_kernel(outbuf, x, mu, inv_sigma, inv_sigma_sq,
                               n_gaussians):
        pt_idx = cuda.grid(1) # pylint: disable=not-callable
        x_pt = x[pt_idx]
        tot = 0.0
        for g_idx in range(n_gaussians):
            xlessmu = x_pt - mu[g_idx]
            tot += (exp((xlessmu*xlessmu) * inv_sigma_sq[g_idx])
                    * inv_sigma[g_idx])
        outbuf[pt_idx] = tot

    @cuda.jit(inline=True, fastmath=True)
    def _gaussians_weighted_cuda_kernel(outbuf, x, mu, inv_sigma, inv_sigma_sq,
                                        weights, n_gaussians):
        pt_idx = cuda.grid(1) # pylint: disable=not-callable
        x_pt = x[pt_idx]
        tot = 0.0
        for g_idx in range(n_gaussians):
            xlessmu = x_pt - mu[g_idx]
            tot += (exp((xlessmu*xlessmu) * inv_sigma_sq[g_idx])
                    * weights[g_idx] * inv_sigma[g_idx])
        outbuf[pt_idx] = tot


def test_gaussians():
    """Test `gaussians` function"""
    n_gaus = [1, 10, 100, 1000, 10000]
    n_eval = int(1e4)

    x = np.linspace(-20, 20, n_eval)
    np.random.seed(0)
    mu_sigma_weight_sets = [(np.linspace(-50, 50, n), np.linspace(0.5, 100, n),
                             np.random.rand(n)) for n in n_gaus]

    timings = OrderedDict()
    for impl in GAUS_IMPLEMENTATIONS:
        timings[impl] = []

    for mus, sigmas, weights in mu_sigma_weight_sets:
        if not isinstance(mus, Iterable):
            mus = [mus]
            sigmas = [sigmas]
            weights = [weights]
        ref_unw = np.sum(
            [stats.norm.pdf(x, loc=m, scale=s) for m, s in zip(mus, sigmas)],
            axis=0
        )/len(mus)
        ref_w = np.sum(
            [stats.norm.pdf(x, loc=m, scale=s)*w
             for m, s, w in zip(mus, sigmas, weights)],
            axis=0
        )/np.sum(weights)
        for impl in GAUS_IMPLEMENTATIONS:
            t0 = time()
            test_unw = gaussians(x, mu=mus, sigma=sigmas, weights=None,
                                 implementation=impl)
            dt_unw = time() - t0
            t0 = time()
            test_w = gaussians(x, mu=mus, sigma=sigmas, weights=weights,
                               implementation=impl)
            dt_w = time() - t0
            timings[impl].append((np.round(dt_unw*1000, decimals=3),
                                  np.round(dt_w*1000, decimals=3)))
            err_msgs = []
            if not recursiveEquality(test_unw, ref_unw):
                err_msgs.append(
                    'BAD RESULT (unweighted), n_gaus=%d, implementation='
                    '"%s", max. abs. fract. diff.: %s'
                    %(len(mus), impl, np.max(np.abs((test_unw/ref_unw - 1))))
                )
            if not recursiveEquality(test_w, ref_w):
                err_msgs.append(
                    'BAD RESULT (weighted), n_gaus=%d, implementation="%s"'
                    ', max. abs. fract. diff.: %s'
                    %(len(mus), impl, np.max(np.abs((test_w/ref_w - 1))))
                )
            if err_msgs:
                for err_msg in err_msgs:
                    logging.error(err_msg)
                raise ValueError('\n'.join(err_msgs))

    tprofile.debug(
        'gaussians() timings (unweighted) (Note:OMP_NUM_THREADS=%d; evaluated'
        ' at %.0e points)', OMP_NUM_THREADS, n_eval
    )
    timings_str = '  '.join([format(t, '10d') for t in n_gaus])
    tprofile.debug(' '*30 + 'Number of gaussians'.center(59))
    tprofile.debug('         %15s       %s', 'impl.', timings_str)
    timings_str = '  '.join(['-'*10 for t in n_gaus])
    tprofile.debug('         %15s       %s', '-'*15, timings_str)
    for impl in GAUS_IMPLEMENTATIONS:
        # only report timings for unweighted case
        timings_str = '  '.join([format(t[0], '10.3f') for t in timings[impl]])
        tprofile.debug('Timings, %15s (ms): %s', impl, timings_str)
    logging.info('<< PASS : test_gaussians >>')


if __name__ == "__main__":
    set_verbosity(2)
    test_gaussians()
