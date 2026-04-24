"""
Common tools for performing Bayesian analysis.
"""

import sys

import numpy as np

from pisa import FTYPE
from pisa.analysis.analysis import METRICS_TO_MAXIMIZE
from pisa.core.param import ParamSet
from pisa.utils.log import logging
from pisa.utils.random_numbers import get_random_state

__all__ = ['MCMC_sampling']

__author__ = 'J. Weldert, T. Ehrhardt'

__license__ = '''Copyright (c) 2014-2026, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''

def MCMC_sampling(data_dist, hypo_maker, *, metric, nwalkers, burnin, nsteps, # pylint: disable=invalid-name
                  pprint=True, return_burn_in=False, random_state=None,
                  sampling_algorithm=None):
    """Performs MCMC sampling. Only supports serial (single CPU) execution at the
    moment. See issue #830.

    Parameters
    ----------

    data_dist : Sequence of MapSets or MapSet
        Data distribution to be fit. Can be an actual-, Asimov-, or pseudo-data
        distribution (where the latter two are derived from simulation and so
        aren't technically "data").

    hypo_maker : Detectors or DistributionMaker
        Creates the per-bin expectation values per map based on its param values.
        Free params in the `hypo_maker` are modified by the minimizer to achieve a
        "best" fit.

    metric : string or iterable of strings
        Metric by which to evaluate the fit. See documentation of Map.

    nwalkers : int
        Number of walkers

    burnin : int
        Number of steps in burn in phase

    nSteps : int
        Number of steps after burn in

    pprint : bool (default: True)
        Whether to show live updates of the sampling progress.

    return_burn_in : bool (default: False)
        Also return the steps of the burn in phase.

    random_state : None or type accepted by utils.random_numbers.get_random_state (default: None)
        Random state of the walker starting points.

    sampling_algorithm : None or emcee.moves object (default: None)
        Sampling algorithm used by the emcee sampler. None means to use the default
        which is a Goodman & Weare “stretch move” with parallelization.
        See https://emcee.readthedocs.io/en/stable/user/moves/#moves-user to learn
        more about the emcee sampling algorithms.

    Returns
    -------

    scaled_chain : numpy array
        Array containing all points in the parameter space visited by each walker.
        It is sorted by steps, so all the first steps of all walkers come first.
        To for example get all values of the Nth parameter and the ith walker, use
        scaled_chain[i::nwalkers, N].

    scaled_chain_burnin : numpy array (optional)
        Same as scaled_chain, but for the burn in phase.

    """
    import emcee # pylint: disable=import-outside-toplevel

    assert 'llh' in metric or 'chi2' in metric, 'Use either a llh or chi2 metric'
    if 'chi2' in metric:
        logging.warning("You are using a chi2 metric for the MCMC sampling."
                        "The sampler will assume that llh=0.5*chi2.")

    ndim = len(hypo_maker.params.free)
    bounds = np.repeat([[0,1]], ndim, axis=0)
    rs = get_random_state(random_state)
    p0 = rs.random(ndim * nwalkers).reshape((nwalkers, ndim))

    def func(scaled_param_vals, bounds, data_dist, hypo_maker, metric):
        """Function called by the MCMC sampler. Similar to _minimizer_callable it
        returns the current metric value + prior penalties.

        """
        if (np.any(scaled_param_vals > np.array(bounds)[:, 1]) or
            np.any(scaled_param_vals < np.array(bounds)[:, 0])):
            return -np.inf
        sign = +1 if metric in METRICS_TO_MAXIMIZE else -1
        if 'llh' in metric:
            N = 1 # pylint: disable=invalid-name
        elif 'chi2' in metric:
            N = 0.5 # pylint: disable=invalid-name

        hypo_maker._set_rescaled_free_params(scaled_param_vals) # pylint: disable=protected-access
        hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)
        metric_val = (
            N * data_dist.metric_total( # pylint: disable=possibly-used-before-assignment
                expected_values=hypo_asimov_dist, metric=metric)
            + hypo_maker.params.priors_penalty(metric=metric)
        )
        return sign*metric_val

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, func,
        moves=sampling_algorithm,
        args=[bounds, data_dist, hypo_maker, metric]
    )

    if pprint:
        sys.stdout.write('Burn in')
        sys.stdout.flush()
    pos, prob, state = sampler.run_mcmc(p0, burnin, progress=pprint) # pylint: disable=unused-variable

    if return_burn_in:
        flatchain_burnin = sampler.flatchain
        scaled_chain_burnin = np.full_like(flatchain_burnin, np.nan, dtype=FTYPE)
        param_copy_burnin = ParamSet(hypo_maker.params.free)

        for s, sample in enumerate(flatchain_burnin):
            for dim, rescaled_val in enumerate(sample):
                param = param_copy_burnin[dim]
                param._rescaled_value = rescaled_val # pylint: disable=protected-access
                val = param.value.m
                scaled_chain_burnin[s, dim] = val

    sampler.reset()
    if pprint:
        sys.stdout.write('Main sampling')
        sys.stdout.flush()
    sampler.run_mcmc(pos, nsteps, progress=pprint)

    flatchain = sampler.flatchain
    scaled_chain = np.full_like(flatchain, np.nan, dtype=FTYPE)
    param_copy = ParamSet(hypo_maker.params.free)

    for s, sample in enumerate(flatchain):
        for dim, rescaled_val in enumerate(sample):
            param = param_copy[dim]
            param._rescaled_value = rescaled_val # pylint: disable=protected-access
            val = param.value.m
            scaled_chain[s, dim] = val

    if return_burn_in:
        return scaled_chain, scaled_chain_burnin
    return scaled_chain
