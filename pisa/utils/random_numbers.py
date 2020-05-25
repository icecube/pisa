#! /usr/bin/env python

"""
Utilities to handle random numbers needed by PISA in a consistent and
reproducible manner.

"""


from __future__ import division

from collections.abc import Sequence

import numpy as np

from pisa.utils.log import Levels, logging, set_verbosity


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
        # FIXME: not a `numpy.random.RandomState` object
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
        raise TypeError(
            'Unhandled `random_state` of type %s: %s'
            %(type(random_state), random_state)
        )

    if jumpahead > 0:
        new_random_state.rand(jumpahead)

    return new_random_state


def test_get_random_state():
    """Unit tests for get_random_state function"""
    # Instantiate random states in all legal ways
    rstates = {
        0: get_random_state(None),
        1: get_random_state('rand'),
        2: get_random_state('random'),
        3: get_random_state(np.random.RandomState(0)),
        4: get_random_state(0),
        5: get_random_state([0,]),
        6: get_random_state([0, 0]),
        7: get_random_state([0, 0, 0]),
    }
    rstates[8] = get_random_state(rstates[4].get_state())

    # rs 4-8 should be identical
    ref_id, ref = None, None
    for rs_id, rs in rstates.items():
        if rs_id < 3:
            continue
        if ref is None:
            ref_id = rs_id
            ref = rs.rand(1000)
        else:
            test = rs.rand(1000)
            assert np.array_equal(test, ref), f'rs{rs_id} != rs{ref_id}'

    # Already generated 1k, so generating 2k more gets us 3k; pick off last 1k
    ref = rstates[ref_id].rand(2000)[1000:]
    test = get_random_state(random_state=0, jumpahead=2000).rand(1000)
    assert np.array_equal(test, ref), f'jumpahead=1k: rs != rs{ref_id}[2000:3000]'

    # Test stability of random number generator over time; following were
    # retrieved on 2020-03-19 using numpy 1.18.1 via .. ::
    #
    #   np.array2string(
    #       np.random.RandomState(0).rand(100), precision=20, separator=', '
    #   )
    #

    # pylint: disable=bad-whitespace
    ref = np.array(
        [
            0.5488135039273248  , 0.7151893663724195  , 0.6027633760716439  ,
            0.5448831829968969  , 0.4236547993389047  , 0.6458941130666561  ,
            0.4375872112626925  , 0.8917730007820798  , 0.9636627605010293  ,
            0.3834415188257777  , 0.7917250380826646  , 0.5288949197529045  ,
            0.5680445610939323  , 0.925596638292661   , 0.07103605819788694 ,
            0.08712929970154071 , 0.02021839744032572 , 0.832619845547938   ,
            0.7781567509498505  , 0.8700121482468192  , 0.978618342232764   ,
            0.7991585642167236  , 0.46147936225293185 , 0.7805291762864555  ,
            0.11827442586893322 , 0.6399210213275238  , 0.1433532874090464  ,
            0.9446689170495839  , 0.5218483217500717  , 0.4146619399905236  ,
            0.26455561210462697 , 0.7742336894342167  , 0.45615033221654855 ,
            0.5684339488686485  , 0.018789800436355142, 0.6176354970758771  ,
            0.6120957227224214  , 0.6169339968747569  , 0.9437480785146242  ,
            0.6818202991034834  , 0.359507900573786   , 0.43703195379934145 ,
            0.6976311959272649  , 0.06022547162926983 , 0.6667667154456677  ,
            0.6706378696181594  , 0.2103825610738409  , 0.1289262976548533  ,
            0.31542835092418386 , 0.3637107709426226  , 0.5701967704178796  ,
            0.43860151346232035 , 0.9883738380592262  , 0.10204481074802807 ,
            0.2088767560948347  , 0.16130951788499626 , 0.6531083254653984  ,
            0.2532916025397821  , 0.4663107728563063  , 0.24442559200160274 ,
            0.15896958364551972 , 0.11037514116430513 , 0.6563295894652734  ,
            0.1381829513486138  , 0.1965823616800535  , 0.3687251706609641  ,
            0.8209932298479351  , 0.09710127579306127 , 0.8379449074988039  ,
            0.09609840789396307 , 0.9764594650133958  , 0.4686512016477016  ,
            0.9767610881903371  , 0.604845519745046   , 0.7392635793983017  ,
            0.039187792254320675, 0.2828069625764096  , 0.1201965612131689  ,
            0.29614019752214493 , 0.11872771895424405 , 0.317983179393976   ,
            0.41426299451466997 , 0.06414749634878436 , 0.6924721193700198  ,
            0.5666014542065752  , 0.2653894909394454  , 0.5232480534666997  ,
            0.09394051075844168 , 0.5759464955561793  , 0.9292961975762141  ,
            0.31856895245132366 , 0.6674103799636817  , 0.13179786240439217 ,
            0.7163272041185655  , 0.2894060929472011  , 0.18319136200711683 ,
            0.5865129348100832  , 0.020107546187493552, 0.8289400292173631  ,
            0.004695476192547066,
        ]
    )
    test = np.random.RandomState(0).rand(100)
    assert np.array_equal(test, ref), 'random number generator changed!'

    logging.info('<< PASS : test_get_random_state >>')


if __name__ == '__main__':
    set_verbosity(Levels.INFO)
    test_get_random_state()
