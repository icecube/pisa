#! /usr/bin/env python

"""
Utilities to handle random numbers needed by PISA in a consistent and
reproducible manner.

"""


from __future__ import division

from collections.abc import Sequence

import numpy as np

from pisa.utils.log import set_verbosity


__all__ = ['get_random_state',
           'test_get_random_state']

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


def get_random_state(random_state, jumpahead=0):
    """Derive a `numpy.random.RandomState` object (usable to generate random
    numbers and distributions) from a flexible specification..

    Parameters
    ----------
    random_state : None, RandomState, string, int, state vector, or seq of int
        Note for all of the below cases, `jumpahead` is applied _after_ the
        RansomState is initialized using the `random_state` (except for
        `random_state` indicating a truly-random number, in which case
        `jumpahead` is ignored).
        * If instantiated RandomState object is passed, it is used directly
        * If string : must be either 'rand' or 'random'; random state is
          instantiated at random from either /dev/urandom or (if that is not
          present) the clock. This creates an irreproducibly-random number.
          `jumpahead` is ignored.
        * If int or sequence of lenth one: This is used as the `seed` value;
          must be in [0, 2**32).
        * If sequence of two integers: first must be in [0, 32768): 15
          most-significant bits. Second must be in [0, 131072): 17
          least-significant bits.
        * If sequence of three integers: first must be in [0, 4): 2
          most-significant bits. Second must be in [0, 8192): next 13
          (less-significant) bits. Third must be in [0, 131072): 17
          least-significant bits.
        * If a "state vector" (sequence of length five usable by
          `numpy.random.RandomState.set_state`), set the random state using
          this method.

    jumpahead : int >= 0
        Starting with the random state specified by `random_state`, produce
        `jumpahead` random numbers to move this many states forward in the
        random number generator's finite state machine. Note that this is
        ignored if `random_state`="random" since jumping ahead any number of
        states from a truly-random point merely yields another truly-random
        point, but takes additional computational time.

    Returns
    -------
    random_state : numpy.random.RandomState
        Object callable like `numpy.random` (e.g. `random_state.rand((10,10))`),
        but with __exclusively local__ state (whereas `numpy.random` has global
        state).

    """
    if random_state is None:
        new_random_state = np.random

    elif isinstance(random_state, np.random.RandomState):
        new_random_state = random_state

    elif isinstance(random_state, str):
        allowed_strings = ['rand', 'random']
        rs = random_state.lower().strip()
        if rs not in allowed_strings:
            raise ValueError(
                '`random_state`=%s not a valid string. Must be one of %s.'
                %(random_state, allowed_strings)
            )
        new_random_state = np.random.RandomState()
        jumpahead = 0

    elif isinstance(random_state, int):
        new_random_state = np.random.RandomState(seed=random_state)

    elif isinstance(random_state, Sequence):
        new_random_state = np.random.RandomState()
        if all([isinstance(x, int) for x in random_state]):
            if len(random_state) == 1:
                seed = random_state[0]
                assert seed >= 0 and seed < 2**32
            elif len(random_state) == 2:
                b0, b1 = 15, 17
                assert b0 + b1 == 32
                s0, s1 = random_state
                assert s0 >= 0 and s0 < 2**b0
                assert s1 >= 0 and s1 < 2**b1
                seed = (s0 << b1) + s1
            elif len(random_state) == 3:
                b0, b1, b2 = 1, 12, 19
                assert b0 + b1 + b2 == 32
                s0, s1, s2 = random_state
                assert s0 >= 0 and s0 < 2**b0
                assert s1 >= 0 and s1 < 2**b1
                assert s2 >= 0 and s2 < 2**b2
                seed = (s0 << b1+b2) + (s1 << b2) + s2
            else:
                raise ValueError(
                    '`random_state` sequence of int must be length 1-3'
                )
            new_random_state.seed(seed)
        elif len(random_state) == 5:
            new_random_state.set_state(random_state)
        else:
            raise ValueError(
                'Do not know what to do with `random_state` Sequence %s'
                %(random_state,)
            )
        return new_random_state

    else:
        raise ValueError('Unhandled `random_state` of type %s: %s'
                         %(type(random_state), random_state))

    if jumpahead > 0:
        new_random_state.rand(jumpahead)

    return new_random_state


def test_get_random_state():
    """Unit tests for get_random_state function"""
    raise NotImplementedError()


if __name__ == '__main__':
    set_verbosity(1)
    #test_get_random_state()
