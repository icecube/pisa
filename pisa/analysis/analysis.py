"""
Common tools for performing an analysis collected into a single class
`Analysis` that can be subclassed by specific analyses.
"""


from __future__ import absolute_import, division

from collections.abc import Sequence
from collections import OrderedDict
from copy import deepcopy
import sys
import time

import numpy as np

from pisa import ureg
from pisa.core.detectors import Detectors
from pisa.core.map import Map, MapSet
from pisa.utils.fitting import apply_fit_settings
from pisa.utils.fisher_matrix import get_fisher_matrix
from pisa.utils.log import logging, set_verbosity
from pisa.utils.minimization import (
    LOCAL_MINIMIZERS_WITH_DEFAULTS,
    Counter, display_minimizer_header, minimizer_x0_bounds,
    _run_minimizer, set_minimizer_defaults, validate_minimizer_settings
)
from pisa.utils.pull_method import calculate_pulls
from pisa.utils.random_numbers import get_random_state
from pisa.utils.stats import (
    METRICS_TO_MAXIMIZE, METRICS_TO_MINIMIZE, it_got_better
)


__all__ = ['ANALYSIS_METHODS', 'Analysis', 't23_octant']

__author__ = 'J.L. Lanfranchi, P. Eller, S. Wren'

__license__ = '''Copyright (c) 2014-2020, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''

ANALYSIS_METHODS = ('minimize', 'scan', 'pull')
"""Allowed parameter fitting methods."""

# Define names that users can specify in configs such that the eval of those
# strings works.
numpy = np # pylint: disable=invalid-name
inf = np.inf # pylint: disable=invalid-name
units = ureg # pylint: disable=invalid-name

def t23_octant(fit_info):
    """Check that theta23 is in the first or second octant.

    Parameters
    ----------
    fit_info

    Returns
    -------
    octant_index : int

    Raises
    ------
    ValueError
        Raised if the theta23 value is not in first (`octant_index`=0) or
        second octant (`octant_index`=1)

    """
    valid_octant_indices = (0, 1)

    theta23 = fit_info['params'].theta23.value
    octant_index = int(
        ((theta23 % (360 * ureg.deg)) // (45 * ureg.deg)).magnitude
    )
    if octant_index not in valid_octant_indices:
        raise ValueError('Fitted theta23 value is not in the'
                         ' first or second octant.')
    return octant_index



def get_separate_t23_octant_params(hypo_maker, inflection_point):
    '''
    This function creates versions of the theta23 param that are confined to
    a single octant. It does this for both octant cases. This is used to allow
    fits to be done where only one of the octants is allowed. The fit can then
    be done for the two octant cases and compared to find the best fit.

    Parameters
    ----------
    hypo_maker : DistributionMaker or Detector
        The hypothesis maker being used by the fitter
    inflection_point : quantity
        Point distinguishing between the two octants, e.g. 45 degrees

    Returns
    -------
    theta23_orig : Param
        theta23 param as it was before applying the octant separation
    theta23_first_octant : Param
        theta23 param confined to first octant
    theta23_second_octant : Param
        theta23 param confined to second octant
    '''

    # Reset theta23 before starting
    theta23 = hypo_maker.params.theta23
    theta23.reset()

    # Store the original theat23 param before we mess with it
    theta23_orig = deepcopy(theta23)

    # Get the octant definition
    if (min(theta23.range[0], theta23.range[1]) > inflection_point or
            max(theta23.range[0], theta23.range[1]) < inflection_point):
        raise ValueError(
            "Range of theta23 needs to encompass both octants for separate-octant"
            " fits to work!"
        )
    octants = (
        (theta23.range[0], inflection_point), (inflection_point, theta23.range[1])
        )

    # If theta23 is very close to maximal (e.g. the transition between octants)
    # offset it slightly to be clearly in one octant (note that fit can still
    # move the value back to maximal)
    tolerance = 1. * ureg.degree
    if np.isclose(theta23.value.m_as("degree"), 45., atol=tolerance.m_as("degree")):
        theta23.value -= tolerance

    theta23_first_octant = deepcopy(theta23)
    theta23_second_octant = deepcopy(theta23)

    theta23_first_octant.range = octants[0]
    theta23_second_octant.range = octants[1]

    other_octant_value = 2 * inflection_point - theta23.value

    if theta23.value > inflection_point:
        # no need to set value of `theta23_second_octant`
        theta23_first_octant.value = other_octant_value
    else:
        # no need to set value of `theta23_first_octant`
        theta23_second_octant.value = other_octant_value

    return theta23_orig, theta23_first_octant, theta23_second_octant


class Analysis(object):
    """Major tools for performing "canonical" IceCube/DeepCore/PINGU analyses.

    * "Data" distribution creation (via passed `data_maker` object)
    * Asimov distribution creation (via passed `distribution_maker` object)
    * Minimizer Interface (via method `_minimizer_callable`)
        Interfaces to a minimizer for modifying the free parameters of the
        `distribution_maker` to fit its output (as closely as possible) to the
        data distribution is provided. See [minimizer_settings] for

    """
    def __init__(self):
        self._nit = 0
        self.counter = Counter()

    @staticmethod
    def _calculate_metric_val(data_dist, hypo_asimov_dist, hypo_maker,
                              metric, blind, external_priors_penalty=None):
        """
        Calculates the value of the metric given data and hypo.

        Should not be called externally.

        """
        try:
            if isinstance(hypo_maker, Detectors):
                # FIXME: what about the external priors in this case?
                # assertion for now
                assert external_priors_penalty is None

                metric_val = 0
                for i in range(len(hypo_maker._distribution_makers)): # pylint: disable=protected-access
                    metric_stats = data_dist[i].metric_total(
                        expected_values=hypo_asimov_dist[i], metric=metric[i]
                    )
                    metric_val += metric_stats

                # TODO: is this really just done silently? should be documented
                metric_priors = hypo_maker.params.priors_penalty(
                    metric=metric[0]
                ) # uses just the "first" metric for prior
                metric_val += metric_priors

            else: # DistributionMaker object
                metric_val = (
                    data_dist.metric_total(
                        expected_values=hypo_asimov_dist, metric=metric[0]
                    )
                    + hypo_maker.params.priors_penalty(metric=metric[0])
                )
                if external_priors_penalty is not None:
                    metric_val += external_priors_penalty(
                        hypo_maker=hypo_maker, metric=metric[0]
                    )
        except Exception as e:
            if blind:
                logging.error('Failed when computing metric.')
            else:
                logging.error(
                    'Failed when computing metric with free params %s',
                    hypo_maker.params.free
                )
                logging.error(str(e))
            raise

        return metric_val

    def fit_from_startpoints(
            self, data_dist, hypo_maker, hypo_param_selections,
            extra_param_selections, metric, startpoints=None,
            randomize_params=None, nstart=None, random_state=None,
            fit_settings=None, minimizer_settings=None, other_metrics=None,
            check_octant=True, blind=False, pprint=True, reset_free=False
    ):
        '''Rerun fit either from `nstart` random start points (seeds) or
        definite start points defined in `startpoints`.'''

        if not startpoints and not nstart:
            # covers cases such as None, empty list, 0 etc.
            raise ValueError(
                'Provide either list of start points or number of points!'
            )
        if startpoints and nstart:
            raise ValueError(
                'Either provide list of start points or number of points,'
                ' but not both!'
            )
        if startpoints:
            randomize = False
            if not isinstance(startpoints, Sequence):
                raise TypeError('`startpoints` needs to be a sequence.'
                                ' Got %s instead.' % type(startpoints))

        elif nstart:
            randomize = True
            if not np.issubdtype(type(nstart), np.int):
                raise TypeError('`nstart` needs to be an integer.'
                                ' Got %s instead.' % type(nstart))

        fit_infos = []
        start_t = time.time()
        nruns = nstart if randomize else len(startpoints)
        for irun in range(nruns):
            if randomize:
                # each run uses initial random state moved forward by irun
                if randomize_params is not None:
                    # just randomise specified parameters
                    for pname in randomize_params:
                        hypo_maker.params[pname].randomize(
                            random_state=get_random_state(random_state, jumpahead=irun)
                        )
                else:
                    # randomise all free
                    hypo_maker.params.randomize_free(
                        random_state=get_random_state(random_state, jumpahead=irun)
                    )
            else:
                if len(startpoints[irun]) != len(hypo_maker.params.free):
                    raise ValueError(
                        'You have to provide as many start points as there'
                        ' are free parameters!'
                    )
                for pname, pval in startpoints[irun]:
                    if not pname in hypo_maker.params.free:
                        raise ValueError(
                            'Param "%s" not among set of free hypothesis'
                            ' parameters!'
                        )
                    hypo_maker.params[pname].value = pval
            logging.info('Starting fit from point %s.' % hypo_maker.params.free)
            irun_fit_info = self.optimize_discrete_selections(
                data_dist=data_dist,
                hypo_maker=hypo_maker,
                hypo_param_selections=hypo_param_selections,
                extra_param_selections=extra_param_selections,
                metric=metric,
                fit_settings=fit_settings,
                reset_free=reset_free,
                check_octant=check_octant,
                minimizer_settings=minimizer_settings,
                other_metrics=other_metrics,
                blind=blind,
                pprint=pprint
            )[0]
            fit_infos.append(irun_fit_info)

        # TODO: find optimum, correctly report metadata
        end_t = time.time()
        multi_run_fit_t = end_t - start_t

        logging.info(
            'Total time to fit from all start points: %8.4f s.'
            % multi_run_fit_t
        )
        return fit_infos

    def optimize_discrete_selections(
            self, data_dist, hypo_maker, hypo_param_selections,
            extra_param_selections, metric, fit_settings=None, reset_free=True,
            check_octant=True, minimizer_settings=None, other_metrics=None,
            randomize_params=None, random_state=None,
            blind=False, pprint=True
    ):
        """Outermost optimization wrapper: multiple discrete selections.

        Parameters
        ----------
        #TODO
        """
        # let someone pass just a single extra param selection
        # (which could just as well be part of the regular
        # hypo param selections)
        if not isinstance(extra_param_selections, Sequence):
            extra_param_selections = [extra_param_selections]

        if not isinstance(hypo_param_selections, Sequence):
            hypo_param_selections = [hypo_param_selections]

        if not extra_param_selections:
            extra_param_selections = [None]

        start_t = time.time()

        # here we store the (best) fit(s) for each discrete selection
        fit_infos = []
        fit_metric_vals = []
        fit_num_dists = []
        fit_times = []
        for extra_param_selection in extra_param_selections:
            if (extra_param_selection is not None and
                extra_param_selection in hypo_param_selections):
                raise ValueError(
                    'Your extra parameter selection "%s" has already '
                    'been specified as one of the hypotheses but the '
                    'fit has been requested to minimize over it. These '
                    'are incompatible.' % extra_param_selection
                )
            # combine any previous param selection + the extra selection
            full_param_selections = hypo_param_selections
            full_param_selections.append(extra_param_selection)
            if extra_param_selection is not None:
                logging.info('Fitting discrete selection "%s".'
                             % extra_param_selection)

            # ignore alternate fits below (it's complicated enough with the various
            # discrete hypo best fits we have already)
            this_hypo_fits, _ = self.fit_hypo(
                data_dist=data_dist,
                hypo_maker=hypo_maker,
                hypo_param_selections=full_param_selections,
                metric=metric,
                fit_settings=fit_settings,
                reset_free=reset_free,
                check_octant=check_octant,
                randomize_params=randomize_params,
                random_state=random_state,
                minimizer_settings=minimizer_settings,
                other_metrics=other_metrics,
                blind=blind,
                pprint=pprint,
            )
            this_hypo_metric_vals = [
                hypo_fit['metric_val'] for hypo_fit in this_hypo_fits
            ]
            this_hypo_num_dists = [
                hypo_fit['num_distributions_generated'] for hypo_fit in this_hypo_fits
            ]
            this_hypo_times = [
                hypo_fit['fit_time'].m_as('second') for hypo_fit in this_hypo_fits
            ]

            fit_infos.append(this_hypo_fits)
            fit_metric_vals.append(this_hypo_metric_vals)
            fit_num_dists.append(this_hypo_num_dists)
            fit_times.append(this_hypo_times)

        # what's returned by fit_hypo can either be a full scan or just
        # a single point - in any case, for each point we now optimize
        # the extra selections manually
        if metric in METRICS_TO_MAXIMIZE:
            bf_dims = np.argmax(fit_metric_vals, axis=0)
        else:
            bf_dims = np.argmin(fit_metric_vals, axis=0)
        bf_num_dists = np.sum(fit_num_dists, axis=0)
        bf_fit_times = np.sum(fit_times, axis=0) * ureg.sec

        # select the fitting infos corresponding to these best metric values
        best_fit_infos = [fit_infos[dim][i] for i, dim in enumerate(bf_dims)]
        for num_dist, fit_time, bf_info in \
            zip(bf_num_dists, bf_fit_times, best_fit_infos):
            bf_info['num_distributions_generated'] = num_dist
            bf_info['fit_time'] = fit_time

        end_t = time.time()
        multi_hypo_fit_t = end_t - start_t

        if len(extra_param_selections) > 1:
            logging.info(
                'Total time to fit all discrete hypos: %8.4f s;'
                ' # of dists. generated: %6d',
                multi_hypo_fit_t, np.sum(bf_num_dists)
            )

        return best_fit_infos

    def optimize_t23_octant(self, best_fit_info, alternate_fits, data_dist,
                            hypo_maker, metric, minimizer_settings,
                            other_metrics, pprint, blind,
                            theta23_orig_and_other_octant=None,
                            external_priors_penalty=None):
        """Logic for optimizing octant of theta23.
        #TODO: docstring

        """
        if theta23_orig_and_other_octant is not None:
            if len(theta23_orig_and_other_octant) != 2:
                raise ValueError(
                    "Expecting original theta23 param and that for the fit"
                    " constrained to the second octant!"
                )
            theta23_orig, theta23_other = theta23_orig_and_other_octant
            hypo_maker.update_params(theta23_other)
        else:
            # Hop to other octant by reflecting about 45 deg
            old_octant = t23_octant(best_fit_info)
            theta23 = hypo_maker.params.theta23
            inflection_point = (45*ureg.deg).to(theta23.units)
            tgt = 2*inflection_point - theta23.value
            # the target value must not fall outside the range, so do something
            # about those cases
            if tgt > max(theta23.range):
                theta23.value = (max(theta23.range) -
                                 0.01 * (max(theta23.range)-min(theta23.range)))
            elif tgt < min(theta23.range):
                theta23.value = (min(theta23.range) +
                     0.01 * (max(theta23.range)-min(theta23.range)))
            else:
                theta23.value = tgt
            hypo_maker.update_params(theta23)

        # Re-run minimizer starting at new point
        new_fit_info = self._fit_hypo_inner(
            hypo_maker=hypo_maker,
            data_dist=data_dist,
            metric=metric,
            minimizer_settings=minimizer_settings,
            other_metrics=other_metrics,
            pprint=pprint,
            blind=blind,
            external_priors_penalty=external_priors_penalty
        )
        previous_history = best_fit_info['fit_history']
        rerun_history = new_fit_info['fit_history'][1:]
        total_history = previous_history + rerun_history

        # Check to make sure these two fits were either side of 45
        # degrees. May not be the case (unless enforced separate
        # octant fits)
        if theta23_orig_and_other_octant is None:
            old_octant = t23_octant(best_fit_info)
            new_octant = t23_octant(new_fit_info)

            # independent of whether the new octant is the same as the previous one:
            # compare fit metrics
            if it_got_better(
                new_metric_val=new_fit_info['metric_val'],
                old_metric_val=best_fit_info['metric_val'],
                metric=metric
            ):
            # Take the one with the best fit
                alternate_fits.append(best_fit_info)
                best_fit_info = new_fit_info
                if not blind:
                    logging.debug('Accepting other-octant fit')
            else:
                alternate_fits.append(new_fit_info)
                if not blind:
                    logging.debug('Accepting initial-octant fit')

            if old_octant == new_octant:
                # no harm in reporting this even in case of blindness, right?
                logging.warning(
                    'Checking other octant *might* not have been successful since'
                    ' both fits have resulted in the same octant. Fit will be'
                    ' tried again starting at a point further into the opposite'
                    ' octant.'
                )
                if old_octant > 0.0:
                    # either start at 55 deg or close to upper end of range
                    theta23.value = min(
                        (55.0*ureg.deg).to(theta23.units),
                        max(theta23.range) - 0.01 * (max(theta23.range)-min(theta23.range))
                    )
                else:
                    # either start at 35 deg or close to lower end of range
                    theta23.value = max(
                        (35.0*ureg.deg).to(theta23.units),
                        min(theta23.range) + 0.01 * (max(theta23.range)-min(theta23.range))
                    )
                hypo_maker.update_params(theta23)

                # Re-run minimizer starting at new point
                new_fit_info = self._fit_hypo_inner(
                    hypo_maker=hypo_maker,
                    data_dist=data_dist,
                    metric=metric,
                    minimizer_settings=minimizer_settings,
                    other_metrics=other_metrics,
                    pprint=pprint,
                    blind=blind,
                    external_priors_penalty=external_priors_penalty
                )
                # Make sure the new octant is sensible
                t23_octant(new_fit_info)
                total_history += new_fit_info['fit_history'][1:]

        # record the correct range for theta23
        # (we force its value when fitting the octants separately)
        else:
            # TODO: not sure why we'd need to deepcopy here
            best_fit_info['params'].theta23.range = deepcopy(theta23_orig.range)
            new_fit_info['params'].theta23.range = deepcopy(theta23_orig.range)
            # If changed the range of the theta23 param whilst checking octants
            # reset the range now.
            # Keep the final value though (is up to the reset_free param
            # to deal with resetting this)
            theta23_orig.value = hypo_maker.params.theta23.value
            hypo_maker.update_params(theta23_orig)

        if it_got_better(
            new_metric_val=new_fit_info['metric_val'],
            old_metric_val=best_fit_info['metric_val'],
            metric=metric
        ):
        # Take the one with the best fit
            alternate_fits.append(best_fit_info)
            best_fit_info = new_fit_info
            if not blind:
                logging.debug('Accepting last other-octant fit')
        else:
            alternate_fits.append(new_fit_info)
            if not blind:
                logging.debug('Sticking to previous best fit')

        best_fit_info['fit_history'] = total_history

        return best_fit_info

    # TODO: fix docstring (not just here)
    def fit_hypo(self, data_dist, hypo_maker, hypo_param_selections, metric,
                 fit_settings=None, reset_free=True, check_octant=True,
                 fit_octants_separately=False, randomize_params=None,
                 random_state=None, minimizer_settings=None,
                 other_metrics=None, blind=False, pprint=True,
                 external_priors_penalty=None):
        """Fitter "outer" loop: If `check_octant` is True, run
        `_fit_hypo_inner` starting in each octant of theta23 (assuming that
        is a param in the `hypo_maker`). Otherwise, just run the inner
        method once.

        Note that prior to running the fit, the `hypo_maker` has
        `hypo_param_selections` applied and its free parameters are reset to
        their nominal values.

        Parameters
        ----------
        data_dist : MapSet or List of MapSets
            Data distribution(s). These are what the hypothesis is tasked to
            best describe during the optimization process.

        hypo_maker : Detectors, DistributionMaker or instantiable thereto
            Generates the expectation distribution under a particular
            hypothesis. This typically has (but is not required to have) some
            free parameters which can be modified by the minimizer to optimize
            the `metric`.

        hypo_param_selections : None, string, or sequence of strings
            A pipeline configuration can have param selectors that allow
            switching a parameter among two or more values by specifying the
            corresponding param selector(s) here. This also allows for a single
            instance of a DistributionMaker to generate distributions from
            different hypotheses.

        metric : string or iterable of strings
            The metric to use for optimization. Valid metrics are found in
            `VALID_METRICS`. Note that the optimized hypothesis also has this
            metric evaluated and reported for each of its output maps.

        fit_settings : string or dict

        minimizer_settings : string or dict

        check_octant : bool
            If theta23 is a parameter to be used in the optimization (i.e.,
            free), the fit will be re-run in the second (first) octant if
            theta23 is initialized in the first (second) octant.

        reset_free : bool
            Resets all free parameters to values defined in stages when starting a fit

        fit_octants_separately : bool
            If 'check_octant' is set so that the two octants of theta23 are
            individually checked, this flag enforces that each theta23 can
            only vary within the octant currently being checked (e.g. the
            minimizer cannot swap octants).

        randomize_params : sequence of str
            Names of params whose start values are to be randomized

        random_state : random_state or instantiable thereto
            Initial random state for randomization of parameter start values

        other_metrics : None, string, or list of strings
            After finding the best fit, these other metrics will be evaluated
            for each output that contributes to the overall fit. All strings
            must be valid metrics, as per `VALID_METRICS`, or the
            special string 'all' can be specified to evaluate all
            VALID_METRICS..

        pprint : bool
            Whether to show live-update of minimizer progress.

        blind : bool
            Whether to carry out a blind analysis. This hides actual parameter
            values from display.

        external_priors_penalty : func
            User defined prior penalty function. Adds an extra penalty
            to the metric that is minimized, depending on the input function.


        Returns
        -------
        best_fit_info : OrderedDict (see _fit_hypo_inner method for details of
            `fit_info` dict)
        alternate_fits : list of `fit_info` from other fits run

        """
        start_t = time.time()
        # set up lists for storing the fits
        best_fits = []
        alternate_fits = []

        if isinstance(metric, str):
            metric = [metric]

        # reset the counter whenever we start a new hypo fit
        self.counter = Counter()

        if not check_octant and fit_octants_separately:
            raise ValueError(
                "If 'check_octant' is False, 'fit_octants_separately' must be False"
            )

        # Select the version of the parameters used for this hypothesis
        hypo_maker.select_params(hypo_param_selections)

        # only apply fit settings after the param selection has been applied
        if fit_settings is not None:
            fit_settings = apply_fit_settings(fit_settings, hypo_maker.params.free)

            minimize_params = fit_settings['minimize']['params']
            if minimize_params:
                # check if minimizer settings are passed into this method,
                # fall back to those given in fit settings
                if minimizer_settings is None:
                    # note: we assume these are parsed already!
                    minimizer_settings = {
                        'global': fit_settings['minimize']['global'],
                        'local': fit_settings['minimize']['local']
                    }
                else:
                    logging.warn(
                        'Minimizer settings provided as argument'
                        ' to `fit_hypo` used to override those in'
                        ' the fit settings!'
                    )
                if isinstance(randomize_params, Sequence):
                    excess = set(randomize_params).difference(set(minimize_params))
                    for pname in excess:
                        logging.warn(
                            "Parameter '%s''s start value cannot be"
                            " randomized as it is not among minimization"
                            " parameters. Request has no effect."
                        )
                        randomize_params.remove(pname)
            else:
                if check_octant:
                    logging.warn(
                        'Selecting "check_octant" only useful if theta23'
                        ' is among *minimization* parameters. No need or no'
                        ' point with any other fitting method.'
                        ' Request has no effect.'
                    )
                    check_octant = False

        else:
            # when there are no fit settings we want the default
            # behavior - just numerical minimization over all free
            # parameters: `_fit_hypo_inner` makes sure of this
            if hypo_maker.params.free and minimizer_settings is None:
                raise ValueError(
                    'You did not specify any fit settings, but there are free'
                    ' parameters which cannot be minimized over if there are'
                    ' no minimizer settings!'
                )

        # Reset free parameters to nominal values
        if reset_free:
            hypo_maker.reset_free()
        else:
            minimizer_start_params = hypo_maker.params

        # Determine if checking theta23 octant
        need_octant_check = (
            check_octant and 'theta23' in hypo_maker.params.free.names
        )

        #Determine inflection point, e.g. transition between octants
        if need_octant_check:
            inflection_point = (45. * ureg.deg).to(hypo_maker.params.theta23.units)
            if fit_octants_separately:
                # If fitting each theta23 octant separately, create distinct params
                # for theta23 confined to each of the two octants
                # (also store the original param so can reset later)
                theta23_orig, theta23_first_octant, theta23_second_octant = \
                    get_separate_t23_octant_params(hypo_maker, inflection_point)
                # start with the first octant
                hypo_maker.update_params(theta23_first_octant)

        # Perform the fit
        best_fit_info = self._fit_hypo_inner(
            hypo_maker=hypo_maker,
            data_dist=data_dist,
            metric=metric,
            fit_settings_inner=fit_settings,
            minimizer_settings=minimizer_settings,
            randomize_params=randomize_params,
            random_state=random_state,
            other_metrics=other_metrics,
            pprint=pprint,
            blind=blind,
            external_priors_penalty=external_priors_penalty
        )

        # Decide whether fit for other octant is necessary
        if need_octant_check:
            if ('global' in minimizer_settings and
                minimizer_settings['global'] is not None):
                logging.info(
                    'Checking other octant of theta23 might not be'
                    ' necessary with a global minimizer. Doing so'
                    ' anyway right now.'
                )
            logging.debug('Checking other octant of theta23.')
            if reset_free:
                hypo_maker.reset_free()
            else:
                for param in minimizer_start_params:
                    hypo_maker.params[param.name].value = param.value

            theta23_orig_and_other_octant = (
                (theta23_orig, theta23_second_octant) if fit_octants_separately
                else None
            )

            best_fit_info = self.optimize_t23_octant(
                best_fit_info=best_fit_info,
                alternate_fits=alternate_fits,
                data_dist=data_dist,
                hypo_maker=hypo_maker,
                metric=metric,
                minimizer_settings=minimizer_settings,
                other_metrics=other_metrics,
                pprint=pprint,
                blind=blind,
                theta23_orig_and_other_octant=theta23_orig_and_other_octant,
                external_priors_penalty=external_priors_penalty
            )
        # make sure the overall best fit contains the
        # overall number of distributions generated
        # across the whole fitting process for this point
        best_fit_info['num_distributions_generated'] = self.counter.count
        # append the best fit for this scan point
        best_fits.append(best_fit_info)

        end_t = time.time()
        fit_t = end_t - start_t

        logging.info(
            'Total time to fit hypo: %8.4f s;'
            ' # of dists generated: %6d',
            fit_t, self.counter.count,
        )

        return best_fit_info, alternate_fits

    def _fit_hypo_inner(self, data_dist, hypo_maker, metric,
                        fit_settings_inner=None, minimizer_settings=None,
                        randomize_params=None, random_state=None,
                        other_metrics=None, pprint=True, blind=False,
                        external_priors_penalty=None):
        """Fitter "inner" loop: decides on which fitting routine should be
        dispatched.

        Note that an "outer" loop can handle discrete scanning over e.g. the
        octant for theta23; for each discrete point the "outer" loop can make a
        call to this "inner" loop. One such "outer" loop is implemented in the
        `fit_hypo` method.

        Should not be called outside of `fit_hypo`


        Parameters
        ----------
        data_dist : MapSet or List of MapSets
            Data distribution(s)

        hypo_maker : Detectors, DistributionMaker or convertible thereto

        metric : string or iterable of strings

        fit_settings_inner : dict
            Already-processed fit settings, depending on free hypo_maker
            params

        minimizer_settings : dict

        other_metrics : None, string, or sequence of strings

        pprint : bool
            Whether to show live-update of minimizer progress.

        blind : bool

        external_priors_penalty : func
            User defined prior penalty function


        Returns
        -------
        fit_info : OrderedDict with details of the fit

        """
        if isinstance(metric, str):
            metric = [metric]

        if fit_settings_inner is not None:
            pull_params = fit_settings_inner['pull']['params']
            minimize_params = fit_settings_inner['minimize']['params']
        else:
            # the default: just minimizer over all free
            pull_params = []
            minimize_params = hypo_maker.params.free.names

        # dispatch correct fitting method depending on combination of
        # pull and minimize params

        # no parameters to fit
        if not len(pull_params) and not len(minimize_params):
            logging.debug("Nothing else to do. Calculating metric(s).")
            nofit_hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)
            self.counter += 1
            fit_info = self.nofit_hypo(
                data_dist=data_dist,
                hypo_maker=hypo_maker,
                hypo_asimov_dist=nofit_hypo_asimov_dist,
                metric=metric,
                other_metrics=other_metrics,
                blind=blind,
                external_priors_penalty=external_priors_penalty
           )

        # only parameters to optimize numerically
        elif len(minimize_params) and not len(pull_params):
            fit_info = self._fit_hypo_minimizer(
                data_dist=data_dist,
                hypo_maker=hypo_maker,
                minimizer_settings=minimizer_settings,
                randomize_params=randomize_params,
                random_state=random_state,
                metric=metric,
                other_metrics=other_metrics,
                blind=blind,
                pprint=pprint,
                external_priors_penalty=external_priors_penalty
            )

        # only parameters to fit with pull method
        elif len(pull_params) and not len(minimize_params):
            fit_info = self._fit_hypo_pull(
                data_dist=data_dist,
                hypo_maker=hypo_maker,
                pull_settings=fit_settings_inner['pull'],
                metric=metric,
                other_metrics=other_metrics,
                blind=blind,
                external_priors_penalty=external_priors_penalty
            )

        # parameters to optimize numerically and to fit with pull method
        else:
            raise NotImplementedError(
                "Combination of minimization and pull method not implemented yet!"
            )
        return fit_info

    def _fit_hypo_minimizer(self, data_dist, hypo_maker, metric, minimizer_settings,
                            randomize_params=None, random_state=None,
                            other_metrics=None, pprint=True, blind=False,
                            external_priors_penalty=None):
        """Fitter "inner" loop: Run an arbitrary scipy minimizer to modify
        hypo dist maker's free params until the data_dist is most likely to have
        come from this hypothesis.

        Should not be called externally.

        Parameters
        ----------
        data_dist : MapSet
            Data distribution(s)

        hypo_maker : DistributionMaker or convertible thereto

        metric : string or iterable of strings

        minimizer_settings : dict

        randomize_params : sequence of str or boolean
            list of param names or `True`/`False`

        random_state : random_state or instantiable thereto

        other_metrics : None, string, or sequence of strings

        pprint : bool
            Whether to show live-update of minimizer progress.

        blind : bool

        external_priors_penalty : func
            User defined prior penalty function


        Returns
        -------
        fit_info : OrderedDict with details of the fit with keys 'metric',
            'metric_val', 'params', 'detailed_metric_info', 'hypo_asimov_dist',
            'fit_metadata', 'fit_time', 'fit_history', 'num_distributions_generated'

        """
        if set(minimizer_settings.keys()) == set(('local', 'global')):
            # allow for an entry of `None`
            for minimizer_type in ['local', 'global']:
                try:
                    minimizer_type_settings =\
                        set_minimizer_defaults(minimizer_settings[minimizer_type])
                    validate_minimizer_settings(minimizer_type_settings)
                except:
                    minimizer_type_settings = None
                minimizer_settings[minimizer_type] = minimizer_type_settings
        else:
            # just try to interpret as "regular" local minimization
            method = minimizer_settings['method'].lower()
            if not method in LOCAL_MINIMIZERS_WITH_DEFAULTS:
                raise ValueError(
                    'Minimizer method "%s" could not be identified as'
                    ' corresponding to local minimization (valid methods: %s).'
                    ' If you desire to run a global minimizer pass in the'
                    ' config with explicit "global" and "local" keys.'
                    % (method, LOCAL_MINIMIZERS_WITH_DEFAULTS)
                )
            minimizer_settings = set_minimizer_defaults(minimizer_settings)
            validate_minimizer_settings(minimizer_settings)
            new_minimizer_settings = {
                'global': None,
                'local': minimizer_settings
            }
            minimizer_settings = new_minimizer_settings

        # Want to *maximize* e.g. log-likelihood but we're using a minimizer,
        # so flip sign of metric in those cases.
        if isinstance(metric, str):
            metric = [metric]
        sign = 0
        for m in metric:
            if m in METRICS_TO_MAXIMIZE and sign != +1:
                sign = -1
            elif m in METRICS_TO_MINIMIZE and sign != -1:
                sign = +1
            else:
                raise ValueError('Defined metrics are not compatible')

        # set starting values and bounds (bounds possibly modified depending
        # on whether the local minimizer uses gradients)
        x0, bounds = minimizer_x0_bounds(
            free_params=hypo_maker.params.free,
            randomize_params=randomize_params,
            random_state=random_state,
            minimizer_settings=minimizer_settings['local']
        )

        fit_history = []
        fit_history.append([metric] + [p.name for p in hypo_maker.params.free])

        if pprint and not blind:
            # display header if desired/allowed
            display_minimizer_header(
                free_params=hypo_maker.params.free,
                metric=metric
            )

        # reset number of iterations before each minimization
        self._nit = 0
        # also create a dedicated counter for this one
        # minimization process
        min_counter = Counter()

        # record start time
        start_t = time.time()

        logging.debug('Start minimization at point %s.' % hypo_maker.params.free)
        # this is the function that does the heavy lifting
        # TODO: external priors penalty?
        # TODO: deal with list of metrics
        optimize_result = _run_minimizer(
            fun=self._minimizer_callable,
            x0=x0,
            bounds=bounds,
            random_state=random_state,
            minimizer_settings=minimizer_settings,
            minimizer_callback=self._minimizer_callback,
            hypo_maker=hypo_maker,
            data_dist=data_dist,
            metric=metric,
            sign=sign,
            counter=min_counter,
            fit_history=fit_history,
            pprint=pprint,
            blind=blind,
            external_priors_penalty=external_priors_penalty
        )

        if pprint:
            # clear the line
            sys.stdout.write('\n\n')
            sys.stdout.flush()

        # Will not assume that the minimizer left the hypo maker in the
        # minimized state, so set the values now (also does conversion of
        # values from [0,1] back to physical range)
        rescaled_pvals = optimize_result.pop('x')
        hypo_maker._set_rescaled_free_params(rescaled_pvals) # pylint: disable=protected-access

        # Record the Asimov distribution with the optimal param values
        hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)
        min_counter += 1

        # update the global counter
        self.counter += min_counter.count

        # Get the best-fit metric value
        metric_val = sign * optimize_result.pop('fun')

        end_t = time.time()
        minimizer_time = end_t - start_t

        logging.info(
            'Total time to minimize: %8.4f s;'
            ' # of dists. generated: %6d;'
            ' avg. dist. gen. time: %10.4f ms',
            minimizer_time, min_counter.count,
            minimizer_time*1000./min_counter.count
        )

        # Record minimizer metadata (all info besides 'x' and 'fun')
        # Record all data even for blinded analysis
        metadata = OrderedDict()
        for k in sorted(optimize_result.keys()):
            #if blind and k in ['jac', 'hess', 'hess_inv']:
            #    continue
            metadata[k] = optimize_result[k]

        fit_info = OrderedDict()
        fit_info['metric'] = metric
        fit_info['metric_val'] = metric_val
        #if blind:
        #    hypo_maker.reset_free()
        fit_info['params'] = deepcopy(hypo_maker.params)
        fit_info['detailed_metric_info'] = self.get_detailed_metric_info(
            data_dist=data_dist, hypo_asimov_dist=hypo_asimov_dist,
            params=hypo_maker.params, metric=metric, other_metrics=other_metrics
        )
        fit_info['fit_time'] = minimizer_time * ureg.sec
        # store the no. of distributions for this minimization process
        fit_info['num_distributions_generated'] = min_counter.count
        fit_info['fit_metadata'] = metadata
        fit_info['fit_history'] = fit_history
        # If blind replace hypo_asimov_dist with none object
        #if blind:
        #    hypo_asimov_dist = None
        fit_info['hypo_asimov_dist'] = hypo_asimov_dist

        msg = optimize_result.message
        #if blind:
        #    msg = ''

        if hasattr(optimize_result, 'success'):
            if not optimize_result.success:
                raise ValueError('Optimization failed. Message: "%s"' % msg)
        else:
            logging.warn('Could not tell whether optimization was successful -'
                         ' most likely because global optimization was'
                         ' requested. Message: "%s"' % msg)

        return fit_info

    # TODO: external priors, pprint
    def _fit_hypo_pull(self, data_dist, hypo_maker, pull_settings, metric,
                       other_metrics=None, pprint=True, blind=False,
                       external_priors_penalty=None):
        """Fit a hypo to a data distribution via the pull method.

        Parameters
        ----------
        data_dist : MapSet
            Data distribution(s)

        hypo_maker : DistributionMaker or convertible thereto

        pull_settings : dict

        metric : string or iterable of strings

        other_metrics : None, string, or sequence of strings

        pprint : bool
            Whether to show live-update of fit progress.

        blind : bool

        external_priors_penalty : func
            User defined prior penalty function


        Should not be called externally.

        Returns
        -------
        fit_info : OrderedDict with details of the fit with keys 'metric',
            'metric_val', 'detailed_metric_info', 'params', 'fit_time',
            'num_distributions_generated', 'hypo_asimov_dist'
        """
        fit_info = OrderedDict()

        if isinstance(metric, str):
            metric = [metric]

        # currently only chi2 fit implemented
        if not all(m == "chi2" for m in metric):
            raise ValueError(
                "Only metric 'chi2' supported by pull method."
            )
        # TODO
        assert external_priors_penalty is None

        # record start time
        start_t = time.time()

        pull_counter = Counter()

        # main algorithm: calculate fisher matrix and parameter pulls
        # TODO: check this is indeed generated at the fiducial model
        test_vals = {pname: pull_settings['values'][i] for i, pname in
                     enumerate(pull_settings['params'])}

        fisher, gradient_maps, fid_hypo_asimov_dist, nonempty = get_fisher_matrix(
            hypo_maker=hypo_maker,
            test_vals=test_vals,
            counter=pull_counter
        )

        pulls = calculate_pulls(
            fisher=fisher,
            fid_maps_truth=data_dist,
            fid_hypo_asimov_dist=fid_hypo_asimov_dist,
            gradient_maps=gradient_maps,
            nonempty=nonempty
        )

        # update hypo maker params to best fit values
        for pname, pull in pulls:
            hypo_maker.params[pname].value = (
                hypo_maker.params[pname].nominal_value + pull
            )

        # generate the hypo distribution at the best fit
        best_fit_hypo_dist = hypo_maker.get_outputs(return_sum=True)
        pull_counter += 1
        self.counter += pull_counter.count

        # calculate the value of the metric at the best fit
        metric_val = self._calculate_metric_val(
            data_dist=data_dist, hypo_asimov_dist=best_fit_hypo_dist,
            hypo_maker=hypo_maker, metric=metric, blind=blind,
            external_priors_penalty=external_priors_penalty
        )

        # record stop time
        end_t = time.time()

        fit_info['metric'] = metric
        # store the metric value
        fit_info['metric_val'] = metric_val

        # store the fit duration
        fit_t = end_t - start_t

        logging.info(
            'Total time to compute pulls: %8.4f s;'
            ' # of dists. generated: %6d',
            fit_t, pull_counter.count,
        )

        fit_info['fit_time'] = fit_t * ureg.sec

        #if blind:
        #    hypo_maker.reset_free()
        #    fit_info['params'] = ParamSet()
        #else:
        fit_info['params'] = deepcopy(hypo_maker.params)

        # TODO: this logic by now should also not be duplicated everywhere
        if isinstance(hypo_maker, Detectors):
            detailed_metric_info = []
            for i in range(len(data_dist)):
                ith_detailed_metric_info = self.get_detailed_metric_info(
                    data_dist=data_dist[i],
                    hypo_asimov_dist=best_fit_hypo_dist[i],
                    params=hypo_maker._distribution_makers[i].params, # pylint: disable=protected-access
                    metric=metric[i], other_metrics=other_metrics,
                    detector_name=hypo_maker.det_names[i]
                )
                detailed_metric_info.append(ith_detailed_metric_info)
        else: # simple DistributionMaker object
            fit_info['detailed_metric_info'] = self.get_detailed_metric_info(
                data_dist=data_dist, hypo_asimov_dist=best_fit_hypo_dist,
                params=hypo_maker.params, metric=metric[0],
                other_metrics=other_metrics,
                detector_name=hypo_maker._detector_name # pylint: disable=protected-access
            )

        fit_info['num_distributions_generated'] = pull_counter.count
        #if blind:
        #    best_fit_hypo_dist = None
        fit_info['hypo_asimov_dist'] = best_fit_hypo_dist

        return fit_info

    # FIXME (TE): add hypo_param_selections back in?
    def nofit_hypo(self, data_dist, hypo_maker, hypo_asimov_dist,
                   metric, other_metrics=None, blind=False,
                   external_priors_penalty=None):
        """Fitting a hypo to Asimov distribution generated by its own
        distribution maker is unnecessary. In such a case, use this method
        (instead of `fit_hypo`) to still retrieve meaningful information for
        e.g. the match metrics.

        Parameters
        ----------
        data_dist : MapSet or List of MapSets
        hypo_maker : Detectors or DistributionMaker
        hypo_param_selections : None, string, or sequence of strings
        hypo_asimov_dist : MapSet or List of MapSets
        metric : string or iterable of strings
        other_metrics : None, string, or sequence of strings
        blind : bool
        external_priors_penalty : func


        """
        fit_info = OrderedDict()
        if isinstance(metric, str):
            metric = [metric]
        fit_info['metric'] = metric

        # record start time
        start_t = time.time()

        # Check number of used metrics
        if isinstance(hypo_maker, Detectors):
            if len(metric) == 1: # One metric for all detectors
                metric = list(metric) * len(hypo_maker._distribution_makers) # pylint: disable=protected-access
            elif len(metric) != len(hypo_maker._distribution_makers): # pylint: disable=protected-access
                raise ValueError(
                    "Number of defined metrics does not match with number"
                    " of detectors."
                )
        else: # DistributionMaker object
            if len(metric) > 1:
                raise ValueError(
                    "You're not using the `Detectors` class, so stick to a single"
                    " metric."
                )

        metric_val = self._calculate_metric_val(
            data_dist=data_dist, hypo_asimov_dist=hypo_asimov_dist,
            hypo_maker=hypo_maker, metric=metric, blind=blind,
            external_priors_penalty=external_priors_penalty
        )

        # record stop time
        end_t = time.time()
        # store the "fit" duration
        fit_t = end_t - start_t

        fit_info['metric_val'] = metric_val

        if blind:
            # Okay, if blind analysis is being performed, reset the values so
            # the user can't find them in the object
            hypo_maker.reset_free()
            # make it possible to find the best fit in the output
            # fit_info['params'] = ParamSet()

        # have all of this in here, whether blind or not
        fit_info['params'] = deepcopy(hypo_maker.params)

        if isinstance(hypo_maker, Detectors):
            detailed_metric_info = []
            for i in range(len(data_dist)):
                ith_detailed_metric_info = self.get_detailed_metric_info(
                    data_dist=data_dist[i],
                    hypo_asimov_dist=hypo_asimov_dist[i],
                    params=hypo_maker._distribution_makers[i].params, # pylint: disable=protected-access
                    metric=metric[i], other_metrics=other_metrics,
                    detector_name=hypo_maker.det_names[i]
                )
                detailed_metric_info.append(ith_detailed_metric_info)
        else: # simple DistributionMaker object
            fit_info['detailed_metric_info'] = self.get_detailed_metric_info(
                data_dist=data_dist, hypo_asimov_dist=hypo_asimov_dist,
                params=hypo_maker.params, metric=metric[0],
                other_metrics=other_metrics,
                detector_name=hypo_maker._detector_name # pylint: disable=protected-access
            )

        fit_info['fit_time'] = fit_t * ureg.sec
        fit_info['num_distributions_generated'] = 1
        fit_info['fit_metadata'] = OrderedDict()
        fit_info['hypo_asimov_dist'] = hypo_asimov_dist

        return fit_info

    @staticmethod
    def get_detailed_metric_info(data_dist, hypo_asimov_dist, params, metric,
                                 other_metrics=None, detector_name=None):
        """Get detailed fit information, including e.g. maps that yielded the
        metric.

        Parameters
        ----------
        data_dist
        hypo_asimov_dist
        params
        metric : str !
        other_metrics

        Returns
        -------
        detailed_metric_info : OrderedDict

        """
        if other_metrics is None:
            other_metrics = []
        elif isinstance(other_metrics, str):
            other_metrics = [other_metrics]
        all_metrics = sorted(set([metric] + other_metrics))
        detailed_metric_info = OrderedDict()
        if detector_name is not None:
            detailed_metric_info['detector_name'] = detector_name
        for m in all_metrics:
            name_vals_d = OrderedDict()
            name_vals_d['maps'] = data_dist.metric_per_map(
                expected_values=hypo_asimov_dist, metric=m
            )
            metric_hists = data_dist.metric_per_map(
                expected_values=hypo_asimov_dist, metric='binned_'+m
            )
            maps_binned = []
            for asimov_map, metric_hist in zip(hypo_asimov_dist, metric_hists):
                map_binned = Map(
                    name=asimov_map.name,
                    hist=np.reshape(metric_hists[metric_hist],
                                    asimov_map.shape),
                    binning=asimov_map.binning
                )
                maps_binned.append(map_binned)
            name_vals_d['maps_binned'] = MapSet(maps_binned)
            name_vals_d['priors'] = params.priors_penalties(metric=metric)
            detailed_metric_info[m] = name_vals_d
        return detailed_metric_info

    def _minimizer_callable(self, scaled_param_vals, hypo_maker, data_dist,
                            metric, sign, counter, fit_history, pprint, blind,
                            external_priors_penalty=None):
        """Simple callback for use by scipy.optimize minimizers.

        This should *not* in general be called by users, as `scaled_param_vals`
        are stripped of their units and scaled to the range [0, 1], and hence
        some validation of inputs is bypassed by this method.

        Parameters
        ----------
        scaled_param_vals : sequence of floats
            If called from a scipy.optimize minimizer, this sequence is
            provieded by the minimizer itself. These values are all expected to
            be in the range [0, 1] and be simple floats (no units or
            uncertainties attached, etc.). Rescaling the parameter values to
            their original (physical) ranges (including units) is handled
            within this method.

        hypo_maker : Detectors or DistributionMaker
            Creates the per-bin expectation values per map (aka Asimov
            distribution) based on its param values. Free params in the
            `hypo_maker` are modified by the minimizer to achieve a "best" fit.

        data_dist : Sequence of MapSets or MapSet
            Data distribution to be fit. Can be an actual-, Asimov-, or
            pseudo-data distribution (where the latter two are derived from
            simulation and so aren't technically "data").

        metric : iterable of strings
            Metric by which to evaluate the fit. See Map

        sign : +1 or -1
            sign with which to multipy overall metric value

        counter : Counter
            Mutable object to keep track--outside this method--of the number of
            times this method is called.

        pprint : bool
            Displays a single-line that updates live (assuming the entire line
            fits the width of your TTY).

        blind : bool

        external_priors_penalty : func
            User defined prior penalty function

        """
        # Set param values from the scaled versions the minimizer works with
        hypo_maker._set_rescaled_free_params(scaled_param_vals) # pylint: disable=protected-access

        # Get the Asimov map set
        try:
            hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)
        except Exception as e:
            if blind:
                logging.error('Failed to generate Asimov distribution.')
            else:
                logging.error(
                    'Failed to generate Asimov distribution with free'
                    ' params %s', hypo_maker.params.free
                )
                logging.error(str(e))
            raise

        # Check number of used metrics
        if isinstance(hypo_maker, Detectors):
            if len(metric) == 1: # One metric for all detectors
                metric = list(metric) * len(hypo_maker._distribution_makers) # pylint: disable=protected-access
            elif len(metric) != len(hypo_maker._distribution_makers): # pylint: disable=protected-access
                raise ValueError(
                    "Number of defined metrics does not match with number of"
                    " detectors."
                )
        else: # DistributionMaker object
            if len(metric) > 1:
                raise ValueError(
                    "You're not using the `Detectors` class, so stick to a single"
                    " metric."
                )

        metric_val = self._calculate_metric_val(
            data_dist=data_dist, hypo_asimov_dist=hypo_asimov_dist,
            hypo_maker=hypo_maker, metric=metric, blind=blind,
            external_priors_penalty=external_priors_penalty
        )

        # Report status of metric & params (except if blinded)
        if blind:
            msg = ('minimizer iteration: #%6d | function call: #%6d'
                   %(self._nit, counter.count))
        else:
            #msg = '%s=%.6e | %s' %(metric, metric_val, hypo_maker.params.free)
            msg = '%s %s %s | ' %(('%d'%self._nit).center(6),
                                  ('%d'%counter.count).center(10),
                                  format(metric_val, '0.5e').rjust(12))
            msg += ' '.join([('%0.5e'%p.value.m).rjust(12)
                             for p in hypo_maker.params.free])

        if pprint:
            sys.stdout.write('\r' + msg)
            sys.stdout.flush()
            # TODO: why again?
            sys.stdout.write('\b' * len(msg))
        else:
            logging.trace(msg)

        counter += 1

        #if not blind:
        # do record this
        fit_history.append(
            [metric_val] + [v.value.m for v in hypo_maker.params.free]
        )
            
        return sign*metric_val

    def _minimizer_callback(self, xk, *args): # pylint: disable=unused-argument
        """Passed as `callback` parameter to `optimize.minimize`, and is called
        after each iteration. Keeps track of number of iterations.

        Parameters
        ----------
        xk : list
            Parameter vector

        """
        self._nit += 1

# TODO: how to do something fast, that doesn't rely too much on external stuff?
def test_fitting():
    """Unit test for fitting routine(s).
    """
    return

if __name__ == "__main__":
    set_verbosity(1)
    test_fitting()

