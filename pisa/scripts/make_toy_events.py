#! /usr/bin/env python

"""
Using parameterisations defined for effective areas, resolution functions, and
PID classification efficiencies, generate MC toy data files for use with PISA.

Note that the random number generator is seeded by both number of events in a
set and the set number, so all MC sets generated should be identical across
invocations for the same (n_events, set_num) set while sets where either or
both of those differ should be statistically independent of one another (as
much as Numpy's Mersenne Twister with different seeds guarantees, at least).

Other parameters, such as coszen range, energy range, spectral index, etc. do
_not_ enter into the random state, and therefore sets that modify only these
parameters will **NOT be statistically independent** from one another.
"""


from __future__ import absolute_import, division

from argparse import ArgumentParser
from itertools import product
import os
import re

import numpy as np
from scipy.stats import norm

from pisa import FTYPE, ureg
from pisa.core.binning import basename
from pisa.core.events import Events
from pisa.stages.aeff.param import load_aeff_param
from pisa.stages.reco.param import load_reco_param
from pisa.stages.pid.param import load_pid_energy_param
from pisa.utils.comparisons import recursiveEquality
from pisa.utils.fileio import mkdir, to_file
from pisa.utils.format import format_num
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.random_numbers import get_random_state


__all__ = ['FNAME_TEMPLATE', 'FNAME_INFO_RE',
           'get_physical_bounds', 'sample_powerlaw', 'sample_truncated_dist',
           'generate_mc_events', 'populate_reco_observables',
           'populate_pid', 'mcgen_random_state', 'make_toy_events',
           'parse_args', 'main']

__author__ = 'T. Ehrhardt, J.L. Lanfranchi'

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


FNAME_TEMPLATE = (
    '{file_type}'
    '__{detector}'
    '__toy'
    '_{e_min}_to_{e_max}GeV'
    '_spidx{spectral_index}'
    '_cz{cz_min}_to_{cz_max}'
    '_{num_events}evts'
    '_set{set_index}'
    '__unjoined'
    '.{extension}'
)

FNAME_INFO_RE = re.compile(
    r'^(?P<file_type>[^_.]+)'
    r'__(?P<detector>[^_]+)'
    r'__toy'
    r'_(?P<e_min>[0-9e.+-]+)_to_(?P<e_max>[0-9e.+-]+)(?P<e_units>[^_]*)'
    r'_spidx(?P<spectral_index>[0-9e.+-]+)'
    r'_cz(?P<cz_min>[0-9e.+-]+)_to_(?P<cz_max>[0-9.+-]+)'
    r'_(?P<num_events>[0-9e.+-]+)evts'
    r'_set(?P<set_index>[0-9]+)'
    r'__(?P<unjoined_or_joined>[^_]+)'
    r'.(?P<extension>[^_]+)$'
    , re.IGNORECASE
)


def get_physical_bounds(dim):
    """Get physical bounds for a dimension. Works on the "base" dimension (e.g.
    true-energy and reco-energy both are treated as just "energy"). Ignores any
    artificially limits imposed by a user.

    Parameters
    ----------
    dim : string

    Returns
    -------
    lower_bound, upper_bound : pint.Quantity

    Raises
    ------
    ValueError
        If `dim` cannot resolve to either 'energy' or 'coszen'.

    """
    base_dim = basename(dim)
    if base_dim == 'energy':
        return 0 * ureg.GeV, np.inf * ureg.GeV
    if base_dim == 'coszen':
        return -1 * ureg.dimensionless, 1  * ureg.dimensionless

    raise ValueError('Unknown `dim` = %s' % dim)


def sample_powerlaw(gamma, x0, x1, size=1, random_state=None):
    r"""Sample from a power-law distribution of the form

    ..math::
        {\rm pdf}(x) \propto x^{-\gamma},  x \in [x_0, x_1] > 0

    Parameters
    ----------
    gamma : float >= 0
        Power law index; note that the power applied to x is the _negative_ of
        `gamma` (see above equation).

    x0, x1 : float > 0
        Mininmum and maximum for the domain of the distribution.

    Returns
    -------
    samples, weights : numpy.array
        Both have same size; `weights` are the inverse of the PDF evaluated at
        `samples`.

    Notes
    -----
    For the special case where `gamma` is 0, the uniform-random distribution is
    sampled from.

    For the special case where `gamma` is 1, inverse sampling is derived as
    follows. First, the PDF as function of x has the following form:

    ..math::
        {\rm pdf}(x) = C (x/x_0)^{-1}

    where C is a normalization constant such that the distribution integrates
    to 1. The integral is the CDF, whose value at a point x we'll call y:

    ..math::
        {\rm cdf}(x) &= \int C (x/x_0)^{-1}
                     &= C \log(x/x_0) + k \equiv y

    Forcing the integral from x0 to x1 to be 1 and setting the CDF to be
    0 at x = x0 and 1 at x = x1 fixes the constants C and k, giving:

    ..math::
        {\rm pdf}(x) = \frac{1}{\log(x_1/x_0} \cdot \frac{1}{x}
        {\rm cdf}(x) = y = \frac{\log(x/x_0)}{\log(x_1/x_0}

    Solving for the inverse of the CDF yields the PPF:

    ..math::
        {\rm ppf}(y) = x_0 (x_1/x_0)^y

    Sampling y from a uniform-random distribution in [0, 1] and transforming
    these y samples by the above PPF yeilds x-values drawn from the original
    PDF.

    For spectral indices `gamma` other than 0 or 1, code was adapted from
    http://stackoverflow.com/a/31117560

    """
    random_state = get_random_state(random_state)

    assert x1 > x0 > 0
    assert gamma >= 0

    if gamma == 0:
        samples = random_state.uniform(x0, x1, size)
        weights = np.full(size, fill_value=x1 - x0)

    elif gamma == 1:
        samples = x0 * (x1 / x0)**random_state.uniform(size=size)
        weights = samples * np.log(x1 / x0)

    else:
        g = 1 - gamma
        ag = x0**g
        bg = x1**g
        r = random_state.uniform(size=size)
        samples = (ag + r*(bg - ag))**(1/g)
        weights = (x1**g - x0**g) / (g * np.power(samples, g-1))

    return samples.astype(FTYPE), weights.astype(FTYPE)


def sample_truncated_dist(dist, size, trunc_low=-np.inf, trunc_high=np.inf,
                          random_state=None):
    """Sample from a truncated verison of a numpy/scipy distribution.

    Parameters
    ----------
    dist : scipy.stats distribution

    size : None or numpy shape spec
        Any valid shape specification for numpy arrays. E.g. an integer or
        tuple of integers.

        If `trunc_low` and `trunc_high` are arrays: `size` must either be None
        or match the shape of those arrays (i.e. `size == trunc_low.shape`).

    trunc_low, trunc_high : float or numpy.ndarrays of same shape
        Lower and upper limits of the distribution to sample from.

    random_state
        See pisa.utils.random_numbers.get_random_state

    Returns
    -------
    samples : numpy.array with dtype=pisa.FTYPE

    Notes
    -----
    Truncation limits should always specified with the same dtype (preferably
    max precision used in numpy, i.e. numpy.float64) such that resulting
    samples are as consistent as possible, and the effects of different
    precisions are only applied in the last step, truncating the precision of
    `samples` to pisa.FTYPE.

    """
    random_state = get_random_state(random_state)

    # Validation and normalization of args
    assert np.isscalar(trunc_low) == np.isscalar(trunc_high)
    if not np.isscalar(trunc_low):
        trunc_low = np.asarray(trunc_low)
        trunc_high = np.asarray(trunc_high)
        assert trunc_low.shape == trunc_high.shape
        if size is not None:
            if np.asarray(size) != trunc_low.shape:
                raise ValueError(
                    '`size` = %s does not match the shape of the passed'
                    ' `trunc_low` and `trunc_high` args (%s)'
                    % (size, trunc_low.shape)
                )

    cdf_low = dist.cdf(trunc_low)
    cdf_high = dist.cdf(trunc_high)

    uniform_rv = random_state.uniform(low=cdf_low, high=cdf_high, size=size)
    samples = dist.ppf(uniform_rv)

    return samples.astype(FTYPE)


def generate_mc_events(num_events, energy_range, coszen_range, spec_ind,
                       aeff_energy_param_source, aeff_coszen_param_source,
                       random_state=None):
    """Instantiates an Events object populated with toy Monte Carlo events.

    This generates MC-true variables ('true_energy' and 'true_coszen'), as well
    as effective area weights ('weighted_aeff'), just as Monte Carlo simulation
    would do.

    Parameters
    ----------
    num_events
    energy_range
    coszen_range
    spec_ind
    aeff_energy_param_source
    aeff_coszen_param_source
    random_state

    Returns
    -------
    mc_events : pisa.core.events.Events

    See Also
    --------
    populate_reco_observables
        Simulate the reconstruction process, populating the reco fields in the
        `mc_events` object.

    populate_pid
        Simulate the particle identification (classification) proecess,
        populating the 'pid' field in the `mc_events` object.

    """
    random_state = get_random_state(random_state)
    logging.info('  Generating mc events dict with true variables and'
                 ' effective area weights.')

    aeff_param = dict(energy=load_aeff_param(aeff_energy_param_source),
                      coszen=load_aeff_param(aeff_coszen_param_source))

    mc_events = Events()

    for flavint in mc_events.flavints:
        logging.debug('Processing flavint: %s', flavint)

        # TODO: should probably be generalised - if energy_range[0] smaller
        # than any of the charged leptons' rest masses, set lower energy limit
        # to the latter. But we're fine for typical lower limit of 1 GeV.

        if flavint in ['nutau_cc', 'nutaubar_cc']:
            # Take into account E_th = 3.5 GeV for nutau cc
            en_low = max(3.5, energy_range[0])
        else:
            en_low = energy_range[0]

        if flavint.cc and energy_range[0] <= 0.1:
            raise ValueError(
                'Min energy must be above muon rest mass (0.1 GeV).'
                ' Got %e instead.' % energy_range[0]
            )

        true_energies, energy_weights = sample_powerlaw(
            gamma=spec_ind,
            x0=en_low,
            x1=energy_range[1],
            size=num_events,
            random_state=random_state
        )

        cz_min, cz_max = np.min(coszen_range), np.max(coszen_range)
        true_coszens = random_state.uniform(
            low=cz_min,
            high=cz_max,
            size=num_events
        )

        mc_events[flavint] = dict(true_energy=true_energies.astype(FTYPE),
                                  true_coszen=true_coszens.astype(FTYPE))

        # Make sure that there will be exactly num_events events in the end, so
        # weight accordingly
        weights = np.full(num_events, fill_value=1/num_events)
        for dimension in ['energy', 'coszen']:
            logging.debug('Drawing %d samples from %s aeff',
                          num_events, dimension)

            aeff_func = None
            for flavintgroup, func in aeff_param[dimension].items():
                if flavint in flavintgroup:
                    aeff_func = func

            if aeff_func is None:
                raise ValueError('Did not find a %s aeff parameterization for'
                                 ' %s' % (dimension, flavint))

            if dimension == 'energy':
                weight_mod = aeff_func(true_energies)
                weights *= weight_mod

            # NOTE: Same eff. area modulation moving away to both sides
            # from horizontal to more inclined events
            elif dimension == 'coszen':
                weight_mod = aeff_func(true_coszens)
                # normalise
                weights *= weight_mod * num_events / np.sum(weight_mod)

        az_vol = 2*np.pi
        mc_events[flavint]['weighted_aeff'] = (
            weights * energy_weights * az_vol * (cz_max - cz_min)
        ).astype(FTYPE)

    return mc_events


def populate_reco_observables(mc_events, param_source, random_state=None):
    """Modify `mc_events` with the reconstructed variables derived from the
    true variables that must already be present.

    Note that modification is in-place.

    Parameters
    ----------
    mc_events : pisa.core.events.Events
    param_source : string
        Resource location from which to load parameterizations
    random_state
        Passed as argument to `pisa.utils.random_numbers.get_random_state`. See
        docs for that function for acceptable values.

    """
    random_state = get_random_state(random_state)
    logging.info('  Applying resolution functions')

    reco_params = load_reco_param(param_source)

    for flavint in mc_events.flavints:
        logging.debug('Processing %s.', flavint)

        all_dist_info = None
        for flavintgroup, info in reco_params.items():
            if flavint in flavintgroup:
                all_dist_info = info

        if all_dist_info is None:
            raise ValueError('Did not find reco parameterizations for'
                             ' %s' % flavint)

        true_energies = mc_events[flavint]['true_energy']
        num_events = len(true_energies)

        for true_dimension in list(mc_events[flavint]):
            if 'true' not in true_dimension:
                continue
            base_dim = basename(true_dimension)

            true_vals = mc_events[flavint][true_dimension]

            dist_info = all_dist_info[base_dim]
            if len(dist_info) != 1:
                raise NotImplementedError('Multiple distributions not'
                                          ' yet supported.')
            dist_info = dist_info[0]
            dist_class = dist_info['dist']
            dist_kwargs = {}
            for key, func in dist_info['kwargs'].items():
                dist_kwargs[key] = func(true_energies)
            #dist_frac = dist_info['fraction']
            reco_dist = dist_class(**dist_kwargs)

            logging.debug(
                'Drawing %d samples from res func. %s (only within'
                ' physical boundaries for )',
                num_events, dist_class
            )

            physical_min, physical_max = get_physical_bounds(
                base_dim,
            )
            error_min = physical_min.magnitude - true_vals
            error_max = physical_max.magnitude - true_vals

            error_samples = sample_truncated_dist(
                reco_dist,
                size=len(true_vals),
                trunc_low=error_min,
                trunc_high=error_max,
                random_state=random_state
            )

            reco_vals = true_vals + error_samples

            if base_dim == 'energy':
                min_reco_val = min(reco_vals)
                logging.trace('min reco energy = %s', min_reco_val)
                assert min_reco_val >= 0, format(min_reco_val, '%.15e')

            elif base_dim == 'coszen':
                min_reco_val = min(reco_vals)
                logging.trace('min reco coszen = %s', min_reco_val)
                assert min_reco_val >= -1, format(min_reco_val, '%.15e')
                max_reco_val = max(reco_vals)
                logging.trace('max reco coszen = %s', max_reco_val)
                assert max_reco_val <= +1, format(max_reco_val, '%.15e')

            mc_events[flavint]['reco_' + base_dim] = reco_vals

    return mc_events


def populate_pid(mc_events, param_source, cut_val=0, random_state=None,
                 dist='discrete', **dist_kwargs):
    """Construct a 'pid' field within the `mc_events` object.

    Parameters
    ----------
    mc_events : pisa.core.Events
    param_source
    cut_val
    random_state
    dist

    """
    random_state = get_random_state(random_state)
    logging.info('  Classifying events as tracks or cascades')

    dist_allowed = ('discrete', 'normal')
    assert dist in dist_allowed

    pid_param = load_pid_energy_param(param_source)

    for flavint in mc_events.flavints:
        pid_funcs = None
        for flavintgroup, funcs in pid_param.items():
            if flavint in flavintgroup:
                pid_funcs = funcs
        if pid_funcs is None:
            raise ValueError('Could not find pid param for %s' % flavint)

        reco_energies = mc_events[flavint]['reco_energy']
        track_pid_probs = pid_funcs['track'](reco_energies)
        cascade_pid_probs = pid_funcs['cascade'](reco_energies)
        assert np.all(
            np.isclose(track_pid_probs + cascade_pid_probs, 1)
        )
        if dist == 'discrete':
            logging.debug('  Drawing discrete PID values')
            rands = random_state.uniform(size=len(reco_energies))
            pid_vals = np.where(
                rands <= track_pid_probs, cut_val + 1, cut_val - 1
            )
        elif dist == 'normal':
            logging.debug('  Drawing normally distributed PID values')
            # cascades fall below `cut_val`, tracks above
            locs_shifted = cut_val - norm.ppf(cascade_pid_probs, **dist_kwargs)
            assert recursiveEquality(
                norm(loc=locs_shifted, **dist_kwargs).cdf(cut_val),
                cascade_pid_probs
            )
            rv = norm(loc=locs_shifted, **dist_kwargs)
            # size is important in the following, as otherwise all samples are
            # 100% correlated
            pid_vals = rv.rvs(size=len(reco_energies))
        mc_events[flavint]['pid'] = pid_vals.astype(FTYPE)

    return mc_events


def mcgen_random_state(num_events, set_index):
    """Seed and return a numpy.random.RandomState object.

    Parameters
    ----------
    num_events : int > 0
        Number of events in the set. See Notes below.

    set_index : int >= 0
        Set index. See Notes below.

    Returns
    -------
    random_state : numpy.random.RandomState

    Notes
    -----
    The seed is formulated from the arguments via:

    0. Up to two-digit `num_events` integer power-of-10. If the
       power-of-10 is either negative or exceeds 2 digits, an exception
       will be raised.

       E.g. 1289821 = 1.2... x 10^6 -> 6

    1. Up to three most-significant digits of num_events. If there are
       more significant figures, an exception will be raised (rounding
       will _not_ be performed)

       E.g. 1289821 -> ValueError
       but  1280000 -> 128
       and  1200000 ->  12

    2. The set_index, which can be from 0 to 32767

    (0) and (1) are combined together into a single number and then this and
    (2) are sent as a length-2 tuple to
    `pisa.utils.random_numbers.get_random_state` to retrieve a properly-seeded
    `numpy.random.RandomState` object.

    """
    assert num_events >= 1, str(num_events)
    power_of_10 = int(np.floor(np.log10(num_events)))
    assert np.log10(power_of_10) < 3, str(power_of_10)

    msd_power = max(power_of_10 - 2, 0)

    most_sig_digits = num_events // 10**(msd_power)
    if most_sig_digits * int(10**msd_power) != num_events:
        raise ValueError(
            'Only 3 significant figures in num_events can be provided.'
        )

    assert set_index >= 0, str(set_index)

    num_events_id = power_of_10 * int(1e3) + most_sig_digits

    logging.trace('power_of_10     = %d', power_of_10)
    logging.trace('msd_power       = %d', msd_power)
    logging.trace('most_sig_digits = %d', most_sig_digits)
    logging.trace('num_events_id   = %d', num_events_id)

    return get_random_state((num_events_id, set_index))


def make_toy_events(outdir, num_events, energy_range, spectral_index,
                    coszen_range, num_sets, first_set, aeff_energy_param,
                    aeff_coszen_param, reco_param, pid_param, pid_dist):
    """Make toy events and store to a file.

    Parameters
    ----------
    outdir : string
    num_events : int
    energy_range : 2-tuple of floats
    spectral_index : float
    coszen_range : 2-tuple of floats
    num_sets : int
    first_set : int
    aeff_energy_param : string
    aeff_coszen_param : string
    reco_param : string
    pid_param : string
    pid_dist : string

    Returns
    -------
    events : :class:`pisa.core.events.Events`

    """
    energy_range = sorted(energy_range)
    coszen_range = sorted(coszen_range)

    # Validation of args
    assert energy_range[0] > 0 and energy_range[1] < 1e9
    assert coszen_range[0] >= -1 and coszen_range[1] <= 1
    assert np.diff(energy_range)[0] > 0, str(energy_range)
    assert np.diff(coszen_range)[0] > 0, str(coszen_range)
    assert spectral_index >= 0, str(spectral_index)
    assert first_set >= 0, str(first_set)
    assert num_sets >= 1, str(first_set)

    # Make sure resources specified actually exist
    for arg in [aeff_energy_param, aeff_coszen_param, reco_param, pid_param]:
        find_resource(arg)

    mkdir(outdir, warn=False)

    set_indices = list(range(first_set, first_set + num_sets))

    # The following loop is for validation only
    for num, index in product(num_events, set_indices):
        mcgen_random_state(num_events=num, set_index=index)

    for num, set_index in product(num_events, set_indices):
        mcevts_fname = FNAME_TEMPLATE.format(
            file_type='events',
            detector='vlvnt',
            e_min=format_num(energy_range[0]),
            e_max=format_num(energy_range[1]),
            spectral_index=format_num(spectral_index, sigfigs=2,
                                      trailing_zeros=True),
            cz_min=format_num(coszen_range[0]),
            cz_max=format_num(coszen_range[1]),
            num_events=format_num(num, sigfigs=3, sci_thresh=(1, -1)),
            set_index=format_num(set_index, sci_thresh=(10, -10)),
            extension='hdf5'
        )
        mcevts_fpath = os.path.join(outdir, mcevts_fname)
        if os.path.isfile(mcevts_fpath):
            logging.warning('File already exists, skipping: "%s"', mcevts_fpath)
            continue

        logging.info('Working on set "%s"', mcevts_fname)

        # TODO: pass filepaths / resource locations via command line args

        # Create a single random state object to pass from function to function
        random_state = mcgen_random_state(num_events=num,
                                          set_index=set_index)

        mc_events = generate_mc_events(
            num_events=num,
            energy_range=energy_range,
            coszen_range=coszen_range,
            spec_ind=spectral_index,
            aeff_energy_param_source=aeff_energy_param,
            aeff_coszen_param_source=aeff_coszen_param,
            random_state=random_state
        )
        populate_reco_observables(
            mc_events=mc_events,
            param_source=reco_param,
            random_state=random_state
        )
        populate_pid(
            mc_events=mc_events,
            param_source=pid_param,
            random_state=random_state,
            dist=pid_dist
        )

        to_file(mc_events, mcevts_fpath)

        return mc_events


def parse_args(desc=__doc__):
    """Parse command line arguments"""

    parser = ArgumentParser(description=desc)

    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument(
        '--outdir', type=str, required=True,
        help='''Directory in which to place events files.'''
    )
    required_named.add_argument(
        '--num-events', metavar='N', type=str, nargs='+', required=True,
        help='''Space-sparated list of total number of events per set. The same
        number of sets (as specified by --num-sets) will be generated for
        *each* N specified here. Accepts numbers like 1.34e3 and 13400, but
        precision is limited to 3 significant figures.''',
    )
    required_named.add_argument(
        '--energy-range', metavar='LIM', type=float, nargs=2, required=True,
        help='''Limits of (true) energy range from with to draw samples, in
        GeV.'''
    )
    required_named.add_argument(
        '--spectral-index', metavar='GAMMA', type=float, required=True,
        help='''Energy spectral index (number >= 0) for power-law distribution.
        I.e., true-energies (E_true) are sampled proportional to
        E_true^(-GAMMA).'''
    )
    required_named.add_argument(
        '--coszen-range', metavar='LIM', type=float, nargs=2, required=True,
        help='''Range to draw true-cosine-zenity values from. Samples in this
        range are drawn from a uniform distribution. Specify as MIN MAX.'''
    )

    # The rest have defaults and therefore are not required

    parser.add_argument(
        '--num-sets', type=int, default=1,
        help='''No. of toy data sets for each amount of statistics
        (i.e., each value passed via --num-events) to produce.'''
    )
    parser.add_argument(
        '--first-set', metavar='SET_IDX', type=int, default=0,
        help='''Start by producing MC set index number. Must be >= 0.'''
    )
    parser.add_argument(
        '--aeff-energy-param', metavar='PATH',
        default='aeff/vlvnt_aeff_energy_param.json',
        help='''Resource location or file path to effective areas energy
        parameterization. (default: aeff/vlvnt_aeff_energy_param.json)'''
    )
    parser.add_argument(
        '--aeff-coszen-param', metavar='PATH',
        default='aeff/vlvnt_aeff_coszen_param.json',
        help='''Resource location or file path to effective areas coszen
        parameterization. (default: aeff/vlvnt_aeff_energy_param.json)'''
    )
    parser.add_argument(
        '--reco-param', metavar='PATH',
        default='reco/vlvnt_reco_param.json',
        help='''Resource location or file path to reco parameterization.
        (default: aeff/vlvnt_aeff_energy_param.json)'''
    )
    parser.add_argument(
        '--pid-param', metavar='PATH',
        default='pid/vlvnt_pid_energy_param.json',
        help='''Resource location or file path to pid parameterization.
        (default: aeff/vlvnt_aeff_energy_param.json)'''
    )
    parser.add_argument(
        '--pid-dist', choices=['discrete', 'normal'], type=str,
        default='discrete',
        help='''Whether PID values should be discrete or distributed. Specify
        either "discrete" or "normal". (default: "discrete")'''
    )
    parser.add_argument(
        '-v', action='count', default=1,
        help='''Set verbosity level. Repeat for increased verbosity. Note that
        default is level 1 (info), so specifying -v sets level 2 (debug+info)
        and -vv sets level 3 (trace+debug+info).'''
    )
    args = parser.parse_args()

    new_num_events = []
    for num_events in args.num_events:
        if int(float(num_events)) != float(num_events):
            raise ValueError('--num-events was passed "%s" which does not'
                             ' evaluate to an integer.' % num_events)
        new_num_events.append(int(float(num_events)))
    args.num_events = sorted(new_num_events)

    return args


def main():
    """Main"""
    args = parse_args()
    args_d = vars(args)
    set_verbosity(args_d.pop('v'))
    make_toy_events(**args_d)


if __name__ == '__main__':
    main()
