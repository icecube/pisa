"""
Common tools for performing an analysis collected into a single class
`Analysis` that can be subclassed by specific analyses.
"""


from __future__ import absolute_import, division

from collections.abc import Sequence
from collections import OrderedDict, Mapping
from copy import deepcopy
import sys
import time

import numpy as np
from scipy.optimize import OptimizeWarning

from pisa import ureg
from pisa.core.detectors import Detectors
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.utils.config_parser import parse_minimizer_config
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

__author__ = 'J.L. Lanfranchi, P. Eller, S. Wren, E. Bourbeau'

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

ANALYSIS_METHODS = ('minimize', 'pull')
"""Allowed parameter fitting methods."""

# Define names that users can specify in configs such that the eval of those
# strings works.
numpy = np # pylint: disable=invalid-name
inf = np.inf # pylint: disable=invalid-name
units = ureg # pylint: disable=invalid-name

MINIMIZERS_USING_SYMM_GRAD = ('l-bfgs-b', 'slsqp')
"""Minimizers that use symmetrical steps on either side of a point to compute
gradients. See https://github.com/scipy/scipy/issues/4916"""

def merge_mapsets_together(mapset_list=None):
    '''Handle merging of multiple MapSets, when they come in
    the shape of a dict

    '''

    if isinstance(mapset_list[0], Mapping):    
        new_dict = OrderedDict()
        for S in mapset_list:
            for k,v in S.items():

                if k not in new_dict.keys():
                    new_dict[k] = [m for m in v.maps]
                else:
                    new_dict[k] += [m for m in v.maps]

        for k,v in new_dict.items():
            new_dict[k] = MapSet(v)

    else:
        raise TypeError('This function only works when mapsets are provided as dicts')

    return new_dict


# TODO: add Nelder-Mead, as it was used previously...
def set_minimizer_defaults(minimizer_settings):
    """Fill in default values for minimizer settings.

    Parameters
    ----------
    minimizer_settings : dict

    Returns
    -------
    new_minimizer_settings : dict

    """
    new_minimizer_settings = dict(
        method=dict(value='', desc=''),
        options=dict(value=dict(), desc=dict())
    )
    new_minimizer_settings.update(minimizer_settings)

    sqrt_ftype_eps = np.sqrt(np.finfo(FTYPE).eps)
    opt_defaults = {}
    method = minimizer_settings['method']['value'].lower()

    if method == 'l-bfgs-b' and FTYPE == np.float64:
        # From `scipy.optimize.lbfgsb._minimize_lbfgsb`
        opt_defaults.update(dict(
            maxcor=10, ftol=2.2204460492503131e-09, gtol=1e-5, eps=1e-8,
            maxfun=15000, maxiter=15000, iprint=-1, maxls=20
        ))
    elif method == 'l-bfgs-b' and FTYPE == np.float32:
        # Adapted to lower precision
        opt_defaults.update(dict(
            maxcor=10, ftol=sqrt_ftype_eps, gtol=1e-3, eps=1e-5,
            maxfun=15000, maxiter=15000, iprint=-1, maxls=20
        ))
    elif method == 'slsqp' and FTYPE == np.float64:
        opt_defaults.update(dict(
            maxiter=100, ftol=1e-6, iprint=0, eps=sqrt_ftype_eps,
        ))
    elif method == 'slsqp' and FTYPE == np.float32:
        opt_defaults.update(dict(
            maxiter=100, ftol=1e-4, iprint=0, eps=sqrt_ftype_eps
        ))
    else:
        raise ValueError('Unhandled minimizer "%s" / FTYPE=%s'
                         % (method, FTYPE))

    opt_defaults.update(new_minimizer_settings['options']['value'])

    new_minimizer_settings['options']['value'] = opt_defaults

    # Populate the descriptions with something
    for opt_name in new_minimizer_settings['options']['value']:
        if opt_name not in new_minimizer_settings['options']['desc']:
            new_minimizer_settings['options']['desc'] = 'no desc'

    return new_minimizer_settings

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


def get_separate_t23_octant_params(hypo_maker, inflection_point=45.0*ureg.deg,
                                   target_tolerance=0.1*ureg.deg):
    """This function creates versions of the theta23 param that are confined to
    a single octant. It does this for both octant cases. This is used to allow
    fits to be done where only one of the octants is allowed. The fit can then
    be done for the two octant cases and compared to find the best fit.

    Parameters
    ----------
    hypo_maker : DistributionMaker or Detector
        The hypothesis maker being used by the fitter

    inflection_point : quantity
        Point distinguishing between the two octants, e.g. 45 degrees
        (though it can be used for any arbitrary inflection point)

    target_tolerance : None or quantity
        Target for deviation of fit starting point from inflection point
        (e.g., to start clearly in one over the other octant)

    Returns
    -------
    theta23_orig : Param
        theta23 param as it was before applying the octant separation

    theta23_first_octant : Param
        theta23 param confined to first octant

    theta23_second_octant : Param
        theta23 param confined to second octant

    """

    # Reset theta23 before starting -> the octant in which the nominal value
    # of theta23 lies is what theta23 is set to first
    theta23 = hypo_maker.params.theta23
    theta23.reset()

    # Store the original parameter before we mess with it
    theta23_orig = deepcopy(theta23)

    # Get the octant definition (don't need to convert units beforehand)
    if (min(theta23.range) > inflection_point or
        max(theta23.range) < inflection_point):
        raise ValueError(
            "Range of theta23 needs to encompass both octants for"
            " separate-octant fits to work!"
        )

    octant_ranges = (
        (min(theta23.range), inflection_point),
        (inflection_point, max(theta23.range))
    )

    theta23_first_octant = deepcopy(theta23)
    theta23_second_octant = deepcopy(theta23)

    theta23_first_octant.range = octant_ranges[0]
    theta23_second_octant.range = octant_ranges[1]

    # If theta23 is very close to maximal (e.g. the transition between octants)
    # offset it slightly to be clearly in one octant (note that fit can still
    # move the value back to maximal).  The reason for this is that
    # otherwise checks on the parameter bounds (which include a margin for
    # minimizer tolerance) can throw an exception.
    if target_tolerance is not None:
        if np.isclose(
            theta23.value.m_as("degree"), inflection_point.m_as("degree"),
            atol=target_tolerance.m_as("degree")
        ):
            if theta23.value > inflection_point:
                tgt_val = inflection_point + target_tolerance
                # Set the value to the smaller among: target and upper bound.
                # If upper bound is smaller, use a 1% safety margin.
                theta23.value = min(
                    (tgt_val).to(theta23.units),
                    max(theta23.range) - 0.01 * (max(theta23.range) - inflection_point.to(theta23.units))
                )
            else:
                # i.e., if the current value of theta23 is maximal,
                # we target a seed in the lower octant
                tgt_val = inflection_point - target_tolerance
                # Set the value to the larger among: target and lower bound.
                # If lower bound is larger, use a 1% safety margin.
                theta23.value = max(
                    (tgt_val).to(theta23.units),
                    min(theta23.range) + 0.01 * (-min(theta23.range) + inflection_point.to(theta23.units))
                )

    other_octant_value = 2 * inflection_point - theta23.value

    if theta23.value > inflection_point:
        # no need to set value of `theta23_second_octant`
        theta23_first_octant.value = other_octant_value
    else:
        # no need to set value of `theta23_first_octant`
        theta23_second_octant.value = other_octant_value

    logging.debug(
        "Will probe two discrete theta23 ranges: %s and %s. "
        " The two corresponding seeds are: %s and %s"
        % (theta23_first_octant.range, theta23_second_octant.range,
           theta23_first_octant.value, theta23_second_octant.value)
    )
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
        """Calculates the value of the metric given data and hypo.

        Should not be called externally.

        Parameters
        ----------
        data_dist : Map, MapSet, or sequence of Map/MapSet

        hypo_asimov_dist : Map, MapSet, or sequence of Map/MapSet

        hypo_maker : DistributionMaker or Detectors

        metric : sequence of str

        blind : bool

        external_priors_penalty : func

        Returns
        -------
        metric_val : float

        """

        try:
            if isinstance(hypo_maker, Detectors):
                metric_val = 0
                for i in range(len(hypo_maker._distribution_makers)): # pylint: disable=protected-access
                    metric_stats = data_dist[i].metric_total(
                        expected_values=hypo_asimov_dist[i], metric=metric[i]
                    )
                    metric_val += metric_stats

                # TODO: is this really just done silently? document?
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

            # function assumed to work the same way for `Detectors` and
            # `DistributionMaker`
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

    # TODO: keep this or leave for individual analyser to implement?
    def fit_from_startpoints(
            self, data_dist, hypo_maker, hypo_param_selections,
            extra_param_selections, metric, startpoints=None,
            randomize_params=None, nstart=None, random_state=None,
            fit_settings=None, minimizer_settings=None, other_metrics=None,
            check_octant=True, fit_octants_separately=False,
            blind=False, pprint=True, reset_free=False
    ):
        """Rerun fit either from `nstart` random start points (seeds) or
        definite start points defined in `startpoints`. See
        `optimize_discrete_selections for explanations of the various
        parameters.

        Parameters
        ----------
        data_dist : MapSet or sequence of MapSets

        hypo_maker : Detectors, DistributionMaker or instantiable thereto

        hypo_param_selections : None, string, or sequence of strings

        extra_param_selections : None, string, or sequence of strings

        metric : string or iterable of strings

        startpoints : None or sequence of (string, quantity) tuples
            Each tuple must consist of a parameter name and an
            associated value. The parameter names must correspond
            to `hypo_maker.params.free.names`

        randomize_params : None or sequence of string
            Names of parameters to randomize (if used together with
            `nstart` > 0). Must be a subset of
            `hypo_maker.params.free.names`. All remaining free
            parameters' start values will not be randomized.

        nstart : None or int
            No. of different start points from which to run the fit.

        random_state : random_state or instantiable thereto

        fit_settings : string or dict

        minimizer_settings : string or dict

        other_metrics : None, string, or list of strings

        check_octant : bool

        fit_octants_separately : bool

        blind : bool

        pprint : bool

        reset_free : bool


        Returns
        -------
        fit_infos : sequence of dict
            One dictionary per fit, as returned by
            `optimize_discrete_selections`

        """

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
            logging.debug('Starting fit from point %s.' % hypo_maker.params.free)
            irun_fit_info = self.optimize_discrete_selections(
                data_dist=data_dist,
                hypo_maker=hypo_maker,
                hypo_param_selections=hypo_param_selections,
                extra_param_selections=extra_param_selections,
                metric=metric,
                fit_settings=fit_settings,
                reset_free=reset_free,
                check_octant=check_octant,
                fit_octants_separately=fit_octants_separately,
                minimizer_settings=minimizer_settings,
                other_metrics=other_metrics,
                blind=blind,
                pprint=pprint
            )
            fit_infos.append(irun_fit_info)

        end_t = time.time()
        multi_run_fit_t = end_t - start_t

        logging.info(
            'Total time to fit from all start points: %8.4f s.'
            % multi_run_fit_t
        )

        return fit_infos


    def optimize_discrete_selections(
            self, data_dist, hypo_maker, hypo_param_selections,
            extra_param_selections, metric, fit_settings=None,
            minimizer_settings=None, check_octant=True,
            fit_octants_separately=False, reset_free=True,
            randomize_params=None, random_state=None,
            other_metrics=None, blind=False, pprint=True,
            external_priors_penalty=None
    ):
        """Outermost optimization wrapper: multiple discrete selections.

        Parameters
        ----------
        data_dist : MapSet or sequence of MapSets
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

        extra_param_selections : None, string, or sequence of strings
            These will be combined with `hypo_param_selections` and
            optimized (e.g. for discrete hypotheses, such as NMO, or for
            testing only discrete values/ranges of some set of parameters).
            Note that none of these selections may be part of
            `hypo_param_selections`.

        metric : string or iterable of strings
            The metric to use for optimization. Valid metrics are found in
            `VALID_METRICS`. Note that the optimized hypothesis also has this
            metric evaluated and reported for each of its output maps.

        fit_settings : string or dict
            Location of fit settings config or parsed dictionary.

        minimizer_settings : string or dict
            Location of minimizer settings config or parsed dictionary.

        check_octant : bool
            If theta23 is a parameter to be used in the optimization (i.e.,
            free), the fit will be re-run in the second (first) octant if
            theta23 is initialized in the first (second) octant.

        fit_octants_separately : bool
            If 'check_octant' is set so that the two octants of theta23 are
            individually checked, this flag enforces that each theta23 can
            only vary within the octant currently being checked (e.g. the
            minimizer cannot swap octants).

        reset_free : bool
            Resets all free parameters to values defined in stages when
            starting a fit.

        randomize_params : sequence of str or bool
            Names of params whose start values are to be randomized or
            `True`/`False`

        random_state : random_state or instantiable thereto
            Initial random state for randomization of parameter start values.

        other_metrics : None, string, or list of strings
            After finding the best fit, these other metrics will be evaluated
            for each output that contributes to the overall fit. All strings
            must be valid metrics, as per `VALID_METRICS`, or the
            special string 'all' can be specified to evaluate all
            VALID_METRICS.

        blind : bool or int
            Whether to carry out a blind analysis. If True or 1, this hides actual
            parameter values from display and disallows these (as well as Jacobian,
            Hessian, etc.) from ending up in logfiles. If given an integer > 1, the
            fitted parameters are also prevented from being stored in fit info
            dictionaries.
        
        pprint : bool
            Whether to show live-update of minimizer progress.

        external_priors_penalty : func
            User defined prior penalty function. Adds an extra penalty
            to the metric that is fit, depending on the input function.


        Returns
        -------
        best_fit_info : OrderedDict (see _fit_hypo_inner method for details of
            `fit_info` dict)


        """

        if (
            isinstance(extra_param_selections, str) or not
            isinstance(extra_param_selections, Sequence)
        ):
            extra_param_selections = [extra_param_selections]

        # transform this into a sequence, too, in order to test
        # compatibility with `extra_param_selections` and to combine them
        # afterwards (not required to be a sequence when applied to pipeline)
        if (
            isinstance(hypo_param_selections, str) or not
            isinstance(hypo_param_selections, Sequence)
        ):
            hypo_param_selections = [hypo_param_selections]

        start_t = time.time()

        # here we store the (best) fit(s) for each discrete selection
        fit_infos = []
        fit_metric_vals = []
        fit_num_dists = []
        fit_times = []
        for extra_param_selection in extra_param_selections:
            # if it's `None`, we don't need to check anything further
            if (extra_param_selection is not None and
                extra_param_selection in hypo_param_selections):
                raise ValueError(
                    'Your extra parameter selection "%s" has already '
                    'been specified as one of the hypotheses but the '
                    'fit has been requested to minimize over it. These '
                    'are incompatible.' % extra_param_selection
                )
            # combine any previous param selection + the extra selection
            # for this fit
            full_param_selections = hypo_param_selections
            full_param_selections.append(extra_param_selection)
            if extra_param_selection is not None:
                logging.info('Fitting discrete selection "%s".'
                             % extra_param_selection)

            # ignore alternate fits below (it's complicated enough with the
            # various discrete hypo best fits we have already)
            this_hypo_fit, _ = self.fit_hypo(
                data_dist=data_dist,
                hypo_maker=hypo_maker,
                hypo_param_selections=full_param_selections,
                metric=metric,
                fit_settings=fit_settings,
                reset_free=reset_free,
                check_octant=check_octant,
                fit_octants_separately=fit_octants_separately,
                randomize_params=randomize_params,
                random_state=random_state,
                minimizer_settings=minimizer_settings,
                other_metrics=other_metrics,
                blind=blind,
                pprint=pprint,
                external_priors_penalty=external_priors_penalty
            )
            this_hypo_metric_val = this_hypo_fit['metric_val']
            this_hypo_num_dists = this_hypo_fit['num_distributions_generated']
            this_hypo_time = this_hypo_fit['fit_time'].m_as('second')

            fit_infos.append(this_hypo_fit)
            fit_metric_vals.append(this_hypo_metric_val)
            fit_num_dists.append(this_hypo_num_dists)
            fit_times.append(this_hypo_time)

        # optimize the extra selections manually and make sure the fit times
        # and no. of distributions are accurate
        if metric in METRICS_TO_MAXIMIZE:
            bf_dims = np.argmax(fit_metric_vals, axis=0)
        else:
            bf_dims = np.argmin(fit_metric_vals, axis=0)
        bf_num_dists = np.sum(fit_num_dists, axis=0)
        bf_fit_time = np.sum(fit_times, axis=0) * ureg.sec

        best_fit_info = fit_infos[bf_dims]
        best_fit_info['num_distributions_generated'] = bf_num_dists
        best_fit_info['fit_time'] = bf_fit_time

        end_t = time.time()
        multi_hypo_fit_t = end_t - start_t

        if len(extra_param_selections) > 1:
            logging.info(
                'Total time to fit all discrete hypos: %8.4f s;'
                ' # of dists. generated: %6d',
                multi_hypo_fit_t, np.sum(bf_num_dists)
            )

        return best_fit_info


    def _optimize_t23_octant(self, best_fit_info, alternate_fits, data_dist,
                             hypo_maker, metric, minimizer_settings,
                             other_metrics, pprint, blind,
                             theta23_orig_and_other_octant=None,
                             external_priors_penalty=None):
        """Logic for optimizing octant of theta23, which should not be called
        externally. The value of `theta23_orig_and_other_octant` determines
        the behaviour. If it is `None`, the outcome from a previous theta23
        fit in `best_fit_info` will be mirrored into the other octant to get
        the seed for the next fit. If the outcome is in the same octant as
        that in `best_fit_info`, another fit is run, starting even further
        into the other octant. If, however, `theta23_orig_and_other_octant`
        is a tuple of two "theta23" parameters, the theta23 fit will use
        the second parameter in there to run a single fit starting from that
        parameter's value and taking into account its range. No second fit is
        performed in this scenario.


        Parameters
        ----------
        best_fit_info : dict

        alternate_fits : dict

        data_dist : MapSet

        hypo_maker : Detectors or DistributionMaker

        metric : string or iterable of strings

        minimizer_settings : dict
            parsed minimizer configuration

        other_metrics : iterable of strings

        pprint : bool

        blind : bool

        theta23_orig_and_other_octant : 2-param tuple or None
            Determines how the octant of theta23 is treated

        external_priors_penalty : func

        Returns
        -------
        best_fit_info : dict
            Best fit dictionary across fits performed here and the results in
            `best_fit_info` passed in.

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

        # ensure that we record the *full* fit history
        previous_history = best_fit_info['fit_history']
        rerun_history = new_fit_info['fit_history'][1:]
        total_history = previous_history + rerun_history

        # Check to make sure these two fits were either side of 45
        # degrees, unless enforced separate octant fits.
        # If not the case, we run another fit to ensure there's really no
        # minimum in one of the octants.
        if theta23_orig_and_other_octant is None:
            old_octant = t23_octant(best_fit_info)
            new_octant = t23_octant(new_fit_info)

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
                if old_octant == 0:
                    # either start at 55 deg or close to upper end of range
                    theta23.value = min(
                        (55.0*ureg.deg).to(theta23.units),
                        max(theta23.range) - 0.01 * (max(theta23.range)-min(theta23.range))
                    )
                elif old_octant == 1:
                    # either start at 35 deg or close to lower end of range
                    theta23.value = max(
                        (35.0*ureg.deg).to(theta23.units),
                        min(theta23.range) + 0.01 * (max(theta23.range)-min(theta23.range))
                    )
                else:
                    raise ValueError("Octant index of %s unknown!" % old_octant)
                hypo_maker.update_params(theta23)

                # Re-run minimizer starting at new point
                # All parameters except for theta23 are at their outcomes
                # from the previous fit.
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

                # Check to make sure these two fits were either side of 45
                # degrees. May not be the case (unless enforced separate
                # octant fits)
                if not fit_octants_separately :
     
                    old_octant = check_t23_octant(best_fit_info)
                    new_octant = check_t23_octant(new_fit_info)

                    if old_octant == new_octant:
                        logging.warning(
                            'Checking other octant was NOT successful since both '
                            'fits have resulted in the same octant. Fit will be'
                            ' tried again starting at a point further into '
                            'the opposite octant.'
                        )
                        alternate_fits.append(new_fit_info)
                        if old_octant > 0.0:
                            theta23.value = (55.0*ureg.deg).to(theta23.units)
                        else:
                            theta23.value = (35.0*ureg.deg).to(theta23.units)
                        hypo_maker.update_params(theta23)

                        # Re-run minimizer starting at new point
                        # Note that we are overwriting the previous attempt 
                        # to flip octant
                        new_fit_info = self.fit_hypo_inner(
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
                        check_t23_octant(new_fit_info)

                # record the correct range for theta23 (we force its value when fitting the octants separately)
                # If we are at the strictest blindness level 2, no parameters are stored and the 
                # dict only contains an empty dict. Attempting to set a range would cause an eror.
                if fit_octants_separately and blind < 2:
                    best_fit_info['params'].theta23.range = deepcopy(theta23_orig.range)
                    new_fit_info['params'].theta23.range = deepcopy(theta23_orig.range)

                # Take the one with the best fit
                if metric[0] in METRICS_TO_MAXIMIZE:
                    it_got_better = (
                        new_fit_info['metric_val'] > best_fit_info['metric_val']
                    )
                else:
                    it_got_better = (
                        new_fit_info['metric_val'] < best_fit_info['metric_val']
                    )

                if it_got_better:

                    alternate_fits.append(best_fit_info)
                    best_fit_info = new_fit_info
                    if not blind:
                        logging.debug('Accepting last other-octant fit')
                else:
                    alternate_fits.append(new_fit_info)
                    if not blind:
                        logging.debug('Sticking to previous best fit')

        else:
            # we first need a check on the fit outcome for the separate octant
            # case
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

            # record the correct range for theta23
            # (we force its value when fitting the octants separately)
            # TODO: probably not necessary to deepcopy here?
            best_fit_info['params'].theta23.range = deepcopy(theta23_orig.range)
            new_fit_info['params'].theta23.range = deepcopy(theta23_orig.range)
            # If changed the range of the theta23 param whilst checking octants
            # reset the range now.
            # Keep the final value though (is up to the reset_free param
            # to deal with resetting this)
            theta23_orig.value = hypo_maker.params.theta23.value
            hypo_maker.update_params(theta23_orig)

        # finally, we make sure we report the full fit history in here
        best_fit_info['fit_history'] = total_history

        return best_fit_info


    def fit_hypo(self, data_dist, hypo_maker, hypo_param_selections, metric,
                 fit_settings=None, minimizer_settings=None, check_octant=True,
                 fit_octants_separately=False, reset_free=True,
                 randomize_params=None, random_state=None,
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
            Location of fit settings config or parsed dictionary. If none are
            provided despite the presence of free parameters in `hypo_maker`,
            need to supply `minimizer_settings`. Minimizer configuration can
            be overriden by providing `minimizer_settings` in any case.

        minimizer_settings : string or dict
            Location of minimizer settings config or parsed dictionary. These
            will either be used to override minimizer settings in `fit_settings`
            (if supplied), but can also be employed as standalone fit
            (=minimization) settings.

        check_octant : bool
            If theta23 is a parameter to be used in the optimization (i.e.,
            free), the fit will be re-run in the second (first) octant if
            theta23 is initialized in the first (second) octant.

        fit_octants_separately : bool
            If 'check_octant' is set so that the two octants of theta23 are
            individually checked, this flag enforces that each theta23 can
            only vary within the octant currently being checked (e.g. the
            minimizer cannot swap octants).

        reset_free : bool
            Resets all free parameters to values defined in stages when
            starting a fit.

        randomize_params : sequence of str or bool
            Names of params whose start values are to be randomized or
            `True`/`False`

        random_state : random_state or instantiable thereto
            Initial random state for randomization of parameter start values.

        other_metrics : None, string, or list of strings
            After finding the best fit, these other metrics will be evaluated
            for each output that contributes to the overall fit. All strings
            must be valid metrics, as per `VALID_METRICS`, or the
            special string 'all' can be specified to evaluate all
            VALID_METRICS.

        blind : bool
            Whether to carry out a blind analysis. This hides actual parameter
            values from display.

        pprint : bool
            Whether to show live-update of minimizer progress.

        external_priors_penalty : func
            User defined prior penalty function. Adds an extra penalty
            to the metric that is fit, depending on the input function.


        Returns
        -------
        best_fit_info : OrderedDict (see _fit_hypo_inner method for details of
            `fit_info` dict)

        alternate_fits : list of `fit_info` from other fits run

        """

        start_t = time.time()
        # set up list for storing all alternate fit outcomes
        alternate_fits = []

        if isinstance(metric, str):
            metric = [metric]

        # reset the counter whenever we start a new hypo fit
        self.counter = Counter()

        if not check_octant and fit_octants_separately:
            raise ValueError(
                "If 'check_octant' is False, 'fit_octants_separately' must"
                " be False!"
            )

        # Select the version of the parameters used for this hypothesis
        hypo_maker.select_params(hypo_param_selections)

        new_fit_settings = deepcopy(fit_settings)
        # only apply fit settings after the param selection has been applied
        if fit_settings is not None:
            new_fit_settings = apply_fit_settings(
                fit_settings=new_fit_settings,
                free_params=hypo_maker.params.free
            )

            minimize_params = new_fit_settings['minimize']['params']
            if minimize_params:
                # check if minimizer settings are passed into this method,
                # fall back to those given in fit settings
                if minimizer_settings is None:
                    # note: we assume these are parsed already!
                    parsed_minimizer_settings = {
                        'global': new_fit_settings['minimize']['global'],
                        'local': new_fit_settings['minimize']['local']
                    }
                else:
                    logging.warn(
                        'Minimizer settings provided as argument'
                        ' to `fit_hypo` used to override those in'
                        ' the fit settings!'
                    )
                    parsed_minimizer_settings = parse_minimizer_config(minimizer_settings)
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
            parsed_minimizer_settings = parse_minimizer_config(minimizer_settings)

        # Reset free parameters to nominal values
        if reset_free:
            hypo_maker.reset_free()
        else:
            # Save the current minimizer start values for the octant check.
            # deepcopy: mustn't be modified.
            minimizer_start_params = deepcopy(hypo_maker.params)

        # Determine if checking theta23 octant
        need_octant_check = (
            check_octant and 'theta23' in hypo_maker.params.free.names
        )

        # Determine inflection point, e.g. transition between octants
        if need_octant_check:
            if fit_octants_separately:
                # If fitting each theta23 octant separately, create distinct params
                # for theta23 confined to each of the two octants
                # (also store the original param so can reset later)
                theta23_orig, theta23_first_octant, theta23_second_octant = \
                    get_separate_t23_octant_params(hypo_maker)
                # start with the first octant
                hypo_maker.update_params(theta23_first_octant)

        # Perform the fit
        best_fit_info = self._fit_hypo_inner(
            hypo_maker=hypo_maker,
            data_dist=data_dist,
            metric=metric,
            fit_settings_inner=new_fit_settings,
            minimizer_settings=parsed_minimizer_settings,
            randomize_params=randomize_params,
            random_state=random_state,
            other_metrics=other_metrics,
            pprint=pprint,
            blind=blind,
            external_priors_penalty=external_priors_penalty
        )

        # Decide whether fit for other octant is necessary
        if need_octant_check:
            if ('global' in parsed_minimizer_settings and
                parsed_minimizer_settings['global']['method']):
                logging.info(
                    'Checking other octant of theta23 might not be'
                    ' necessary with a global minimizer. Doing so'
                    ' anyway right now.'
                )
            logging.info('Checking other octant of theta23.')
            if reset_free:
                # need to distinguish between the two possible cases
                if fit_octants_separately:
                    logging.debug(
                        "Resetting all free parameters to their nominal values"
                        " except for theta23."
                    )
                    # if fitting octants separately, the range of the theta23
                    # param is currently restricted to the first octant.
                    # Resetting to nominal is not what we want for theta23,
                    # but the other free ones we need to reset
                    hypo_maker.params.fix("theta23")
                    hypo_maker.reset_free()
                    hypo_maker.params.unfix("theta23")
                else:
                    # here, it's safe to reset theta23 also: the next
                    # optimization run will ensure to mirror theta23 from its
                    # value in `best_fit_info` into the other octant
                    logging.debug(
                        "Resetting all free parameters to their nominal values."
                    )
                    hypo_maker.reset_free()
            else:
                # Set all parameters to the values they had at the beginning
                # of the function. The same comments apply as in the case of
                # `reset_free` set to `True`.
                if fit_octants_separately:
                    logging.debug(
                        "Resetting all free parameters except for theta23"
                        " to the values they had when `fit_hypo` was called."
                    )
                    for param in minimizer_start_params:
                        if not "theta23" in param.name:
                            hypo_maker.params[param.name].value = param.value
                else:
                    logging.debug(
                        "Resetting all free parameters"
                        " to the values they had when `fit_hypo` was called."
                    )
                    for param in minimizer_start_params:
                        hypo_maker.params[param.name].value = param.value

            # We use this to pass in the original unmodified theta23 param
            # together with its second octant param (value and range modified)
            # in the case of separate octant fits, and otherwise we don't need
            # the two.
            theta23_orig_and_other_octant = (
                (theta23_orig, theta23_second_octant) if fit_octants_separately
                else None
            )

            best_fit_info = self._optimize_t23_octant(
                best_fit_info=best_fit_info,
                alternate_fits=alternate_fits,
                data_dist=data_dist,
                hypo_maker=hypo_maker,
                metric=metric,
                minimizer_settings=parsed_minimizer_settings,
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

        end_t = time.time()
        fit_t = end_t - start_t

        best_fit_info['fit_time'] = fit_t * ureg.sec

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
            params. Just used to determine the fitting methods to employ.

        minimizer_settings : dict

        randomize_params : sequence of str or bool
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


    def _fit_hypo_minimizer(self, data_dist, hypo_maker, metric,
                            minimizer_settings,
                            randomize_params=None, random_state=None,
                            other_metrics=None, pprint=True, blind=False,
                            external_priors_penalty=None):
        """Fitter "inner" loop: Run an arbitrary scipy minimizer to modify
        hypo dist maker's free params until the data_dist is most likely to have
        come from this hypothesis.

        Should not be called externally.

        Parameters
        ----------
        data_dist : MapSet or List of MapSets
            Data distribution(s)

        hypo_maker : DistributionMaker or convertible thereto

        metric : string or iterable of strings

        minimizer_settings : dict

        randomize_params : sequence of str or bool
            list of param names or `True`/`False`

        random_state : random_state or instantiable thereto

        other_metrics : None, string, or sequence of strings

        pprint : bool
            Whether to show live-update of minimizer progress.

        blind : bool or int

        external_priors_penalty : func
            User defined prior penalty function


        Returns
        -------
        fit_info : OrderedDict with details of the fit with keys 'metric',
            'metric_val', 'params', 'detailed_metric_info', 'hypo_asimov_dist',
            'fit_metadata', 'fit_time', 'fit_history', 'num_distributions_generated'

        """
        # don't modify the original dict
        new_minimizer_settings = deepcopy(minimizer_settings)

        if set(new_minimizer_settings.keys()) == set(('local', 'global')):
            # allow for an entry of `None`
            for minimizer_type in ['local', 'global']:
                if isinstance(new_minimizer_settings[minimizer_type]['method'], str):
                    minimizer_type_settings =\
                        set_minimizer_defaults(new_minimizer_settings[minimizer_type])
                    validate_minimizer_settings(minimizer_type_settings)
                else:
                    minimizer_type_settings = None
                new_minimizer_settings[minimizer_type] = minimizer_type_settings
        else:
            # just try to interpret as "regular" local minimization
            method = new_minimizer_settings['method'].lower()
            if not method in LOCAL_MINIMIZERS_WITH_DEFAULTS:
                raise ValueError(
                    'Minimizer method "%s" could not be identified as'
                    ' corresponding to local minimization (valid methods: %s).'
                    ' If you desire to run a global minimizer pass in the'
                    ' config with explicit "global" and "local" keys.'
                    % (method, LOCAL_MINIMIZERS_WITH_DEFAULTS)
                )
            new_minimizer_settings = set_minimizer_defaults(new_minimizer_settings)
            validate_minimizer_settings(new_minimizer_settings)
            new_minimizer_settings = {
                'global': None,
                'local': new_minimizer_settings
            }

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
            minimizer_settings=new_minimizer_settings['local']
        )

        fit_history = []
        fit_history.append(metric + [p.name for p in hypo_maker.params.free])

        logging.debug('Start minimization at point %s.' % hypo_maker.params.free)

        if pprint and not blind:
            # display header if desired/allowed
            # only show the first metric here (only part of info for Detectors analysis)
            display_minimizer_header(
                free_params=hypo_maker.params.free,
                metric=metric[0]
            )

        # reset number of iterations before each minimization
        self._nit = 0
        # also create a dedicated counter for this one minimization process
        min_counter = Counter()

        # record start time
        start_t = time.time()

        optimize_result = _run_minimizer(
            fun=self._minimizer_callable,
            x0=x0,
            bounds=bounds,
            random_state=random_state,
            minimizer_settings=new_minimizer_settings,
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
            if blind and k in ['jac', 'hess', 'hess_inv']:
                continue
            if k=='hess_inv':
                continue
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
        fit_info['num_distributions_generated'] = min_counter.count
        fit_info['fit_metadata'] = metadata
        fit_info['fit_history'] = fit_history
        fit_info['hypo_asimov_dist'] = hypo_asimov_dist

        msg = optimize_result.message

        if hasattr(optimize_result, 'success'):
            if not optimize_result.success:
                raise OptimizeWarning('Optimization failed. Message: "%s"' % msg)
        if blind > 1:  # only at stricter blindness level
            # Reset to starting value of the fit, rather than nominal values because
            # the nominal value might be out of range if this is inside an octant check.
            hypo_maker._set_rescaled_free_params(x0)
            fit_info['params'] = ParamSet()
        else:
            logging.warn('Could not tell whether optimization was successful -'
                         ' most likely because global optimization was'
                         ' requested.\nMessage: "%s"' % msg)

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
            fit_info['detailed_metric_info'] = [self.get_detailed_metric_info(
                data_dist=data_dist[i], hypo_asimov_dist=hypo_asimov_dist[i],
                params=hypo_maker._distribution_makers[i].params, metric=metric[i],
                other_metrics=other_metrics, detector_name=hypo_maker.det_names[i]
            ) for i in range(len(data_dist))]
        else: # DistributionMaker object

            if 'generalized_poisson_llh' == metric[0]:
                generalized_poisson_dist = hypo_maker.get_outputs(return_sum=False, force_standard_output=False)
                generalized_poisson_dist = merge_mapsets_together(mapset_list=generalized_poisson_dist)
            else:
                generalized_poisson_dist = None

            fit_info['detailed_metric_info'] = self.get_detailed_metric_info(
                data_dist=data_dist, hypo_asimov_dist=hypo_asimov_dist, generalized_poisson_hypo=generalized_poisson_dist,
                params=hypo_maker.params, metric=metric[0], other_metrics=other_metrics,
                detector_name=hypo_maker._detector_name
            )

        fit_info['num_distributions_generated'] = pull_counter.count
        #if blind:
        #    best_fit_hypo_dist = None
        fit_info['hypo_asimov_dist'] = best_fit_hypo_dist

        return fit_info


    def nofit_hypo(self, data_dist, hypo_maker, hypo_asimov_dist, metric,
                   other_metrics=None, blind=False,
                   external_priors_penalty=None, hypo_param_selections=None):
        """Fitting a hypo to Asimov distribution generated by its own
        distribution maker is unnecessary. In such a case, use this method
        (instead of `fit_hypo`) to still retrieve meaningful information for
        e.g. the match metrics.

        Parameters
        ----------
        data_dist : MapSet or List of MapSets
        hypo_maker : Detectors or DistributionMaker
        hypo_asimov_dist : MapSet or List of MapSets
        metric : string or iterable of strings
        other_metrics : None, string, or sequence of strings
        blind : bool
        external_priors_penalty : func
        hypo_param_selections : None, string, or sequence of strings


        Returns
        -------
        fit_info : OrderedDict

        """

        fit_info = OrderedDict()
        if isinstance(metric, str):
            metric = [metric]
        fit_info['metric'] = metric

        # record start time
        start_t = time.time()

        # NOTE: Select params but *do not* reset to nominal values to record
        # the current param values
        hypo_maker.select_params(hypo_param_selections)

        # Check number of used metrics
        if isinstance(hypo_maker, Detectors):
            if len(metric) == 1: # One metric for all detectors
                metric = list(metric) * len(hypo_maker._distribution_makers)
            elif len(metric) != len(hypo_maker._distribution_makers):
                raise IndexError('Number of defined metrics does not match with number of detectors.')
        else: # DistributionMaker object
            assert len(metric) == 1

        # Assess the fit: whether the data came from the hypo_asimov_dist
        try:
            if isinstance(hypo_maker, Detectors):
                metric_val = 0
                for i in range(len(hypo_maker._distribution_makers)):
                    data = data_dist[i].metric_total(expected_values=hypo_asimov_dist[i],metric=metric[i])
                    metric_val += data
                priors = hypo_maker.params.priors_penalty(metric=metric[0]) # uses just the "first" metric for prior
                metric_val += priors
            else: # DistributionMaker object

                if 'generalized_poisson_llh' == metric[0]:

                    hypo_asimov_dist = hypo_maker.get_outputs(return_sum=False, output_mode='binned', force_standard_output=False)
                    hypo_asimov_dist = merge_mapsets_together(mapset_list=hypo_asimov_dist)
                    data_dist = data_dist.maps[0] # Extract the map from the MapSet
                    metric_kwargs = {'empty_bins':hypo_maker.empty_bin_indices}
                else:
                    hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)
                    if isinstance(hypo_asimov_dist, OrderedDict):
                        hypo_asimov_dist = hypo_asimov_dist['weights']
                    metric_kwargs = {}

                metric_val = (
                    data_dist.metric_total(expected_values=hypo_asimov_dist,
                                           metric=metric[0], metric_kwargs=metric_kwargs)
                    + hypo_maker.params.priors_penalty(metric=metric[0])
                )
                if external_priors_penalty is not None:
                    metric_val += external_priors_penalty(hypo_maker=hypo_maker,metric=metric[0])
                    
        except Exception as e:
            if blind:
                logging.error('Minimizer failed')
            else :
                logging.error(
                    'Failed when computing metric with free params %s',
                    hypo_maker.params.free
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

            fit_info['detailed_metric_info'] = [self.get_detailed_metric_info(
                data_dist=data_dist[i], hypo_asimov_dist=hypo_asimov_dist[i],
                params=hypo_maker._distribution_makers[i].params, metric=metric[i],
                other_metrics=other_metrics, detector_name=hypo_maker.det_names[i]
            ) for i in range(len(data_dist))]
        else: # DistributionMaker object

            if 'generalized_poisson_llh' == metric[0]:
                generalized_poisson_dist = hypo_maker.get_outputs(return_sum=False, force_standard_output=False)
                generalized_poisson_dist = merge_mapsets_together(mapset_list=generalized_poisson_dist)
            else:
                generalized_poisson_dist = None

            fit_info['detailed_metric_info'] = self.get_detailed_metric_info(
                data_dist=data_dist, hypo_asimov_dist=hypo_asimov_dist, generalized_poisson_hypo=generalized_poisson_dist,
                params=hypo_maker.params, metric=metric[0], other_metrics=other_metrics,
                detector_name=hypo_maker._detector_name
            )

        fit_info['minimizer_time'] = 0 * ureg.sec
        fit_info['num_distributions_generated'] = 0
        fit_info['minimizer_metadata'] = OrderedDict()
        fit_info['hypo_asimov_dist'] = hypo_asimov_dist

        return fit_info

    @staticmethod
    def get_detailed_metric_info(data_dist, hypo_asimov_dist, params, metric,generalized_poisson_hypo=None,
                                 other_metrics=None, detector_name=None):
        """Get detailed fit information, including e.g. maps that yielded the
        metric.

        Parameters
        ----------
        data_dist
        hypo_asimov_dist
        params
        metric : str or sequence of str
        other_metrics

        Returns
        -------
        detailed_metric_info : OrderedDict

        """

        if other_metrics is None:
            other_metrics = []
        elif isinstance(other_metrics, str):
            other_metrics = [other_metrics]
        if isinstance(metric, str):
            metric = [metric]
        all_metrics = sorted(set(metric + other_metrics))
        detailed_metric_info = OrderedDict()
        if detector_name is not None:
            detailed_metric_info['detector_name'] = detector_name
        for m in all_metrics:
            name_vals_d = OrderedDict()

            # if the metric is not generalized poisson, but the distribution is a dict,
            # retrieve the 'weights' mapset from the distribution output
            if m == 'generalized_poisson_llh':
                name_vals_d['maps'] = data_dist.maps[0].generalized_poisson_llh(expected_values=generalized_poisson_hypo)
                llh_binned = data_dist.maps[0].generalized_poisson_llh(expected_values=generalized_poisson_hypo, binned=True)
                map_binned = Map(name=metric,
                                hist=np.reshape(llh_binned, data_dist.maps[0].shape),
                                binning=data_dist.maps[0].binning
                    )
                name_vals_d['maps_binned'] = MapSet(map_binned)
                name_vals_d['priors'] = params.priors_penalties(metric=metric)
                detailed_metric_info[m] = name_vals_d

            else:
                if isinstance(hypo_asimov_dist,OrderedDict):
                    hypo_asimov_dist = hypo_asimov_dist['weights']

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
            if metric[0] == 'generalized_poisson_llh':
                hypo_asimov_dist = hypo_maker.get_outputs(return_sum=False, output_mode='binned', force_standard_output=False)
                hypo_asimov_dist = merge_mapsets_together(mapset_list=hypo_asimov_dist)
                data_dist = data_dist.maps[0] # Extract the map from the MapSet
                metric_kwargs = {'empty_bins':hypo_maker.empty_bin_indices}
            else:
                hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)
                if isinstance(hypo_asimov_dist, OrderedDict):
                    hypo_asimov_dist = hypo_asimov_dist['weights']
                metric_kwargs = {}

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
                metric = list(metric) * len(hypo_maker._distribution_makers)
            elif len(metric) != len(hypo_maker._distribution_makers):
                raise IndexError('Number of defined metrics does not match with number of detectors.')
        else: # DistributionMaker object
            assert len(metric) == 1

        #
        # Assess the fit: whether the data came from the hypo_asimov_dist
        #
        try:
            if isinstance(hypo_maker, Detectors):
                metric_val = 0
                for i in range(len(hypo_maker._distribution_makers)):
                    data = data_dist[i].metric_total(expected_values=hypo_asimov_dist[i],
                                                  metric=metric[i], metric_kwargs=metric_kwargs)
                    metric_val += data
                priors = hypo_maker.params.priors_penalty(metric=metric[0]) # uses just the "first" metric for prior
                metric_val += priors
            else: # DistributionMaker object
                metric_val = (
                    data_dist.metric_total(expected_values=hypo_asimov_dist,
                                               metric=metric[0], metric_kwargs=metric_kwargs)
                        + hypo_maker.params.priors_penalty(metric=metric[0])
                    )
        except Exception as e:
            if blind:
                logging.error('Minimizer failed')
            else :
                logging.error(
                    'Failed when computing metric with free params %s',
                    hypo_maker.params.free
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
            msg = '%s %s %s | ' %(('%d'%self._nit).center(6),
                                  ('%d'%counter.count).center(10),
                                  format(metric_val, '0.5e').rjust(12))
            msg += ' '.join([('%0.5e'%p.value.m).rjust(12)
                             for p in hypo_maker.params.free])

        if pprint:
            sys.stdout.write('\r' + msg)
            sys.stdout.flush()
            sys.stdout.write('\b' * len(msg))
        else:
            logging.trace(msg)

        counter += 1

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

# --------------------------------------------------------------------------- #
def test_optimize_discrete_selections(
    pipeline_cfg='settings/pipeline/example.cfg'
):
    """Unit tests of `Analysis.optimize_discrete_selections`."""

    from pisa.utils.minimization import override_min_opt

    hypo_maker = DistributionMaker(pipeline_cfg)
    data_dist = hypo_maker.get_outputs(return_sum=True)

    discrete_selections = ['earth', 'lead']

    for i, param in enumerate(sorted(hypo_maker.params.free)):
        # just use two free parameters for testing
        if i > 1:
            param.is_fixed = True

    # use defaults for the chosen FTYPE
    local_minimizer_cfg = set_minimizer_defaults(
        dict(
            method='slsqp',
            options=dict()
        )
    )
    override_min_opt(local_minimizer_cfg, ('disp:False', ))

    a = Analysis()
    try:
        a.optimize_discrete_selections(
            data_dist=data_dist, hypo_maker=hypo_maker, hypo_param_selections='nh',
            extra_param_selections=discrete_selections, metric='chi2',
            fit_settings=None, minimizer_settings=local_minimizer_cfg,
            check_octant=True, reset_free=True,
            randomize_params=hypo_maker.params.free.names, random_state=18,
            other_metrics=None, blind=False, pprint=False,
            external_priors_penalty=None
        )
    except OptimizeWarning:
        logging.warn("Minimization was unsuccessful, but this is highly likely"
                     " to be a glitch related to the chosen minimizer config.")
# --------------------------------------------------------------------------- #
def test_fit_hypo_minimizer(
    pipeline_cfg='settings/pipeline/example.cfg',
    fit_cfg='settings/fit/example_basinhopping_lbfgsb.cfg'
):
    """Unit tests of `Analysis._fit_hypo_minimizer`."""

    from pisa.utils.minimization import override_min_opt

    hypo_maker = DistributionMaker(pipeline_cfg)
    data_dist = hypo_maker.get_outputs(return_sum=True)

    for i, param in enumerate(sorted(hypo_maker.params.free)):
        # just use two free parameters for testing
        if i > 1:
            param.is_fixed = True

    # use defaults for the chosen FTYPE
    local_minimizer_cfg = set_minimizer_defaults(
        dict(
            method='l-bfgs-b',
            options=dict()
        )
    )
    override_min_opt(local_minimizer_cfg, ('disp:0',))

    a = Analysis()
    try:
        a._fit_hypo_minimizer(
            data_dist=data_dist, hypo_maker=hypo_maker, metric='chi2',
            minimizer_settings=local_minimizer_cfg,
            randomize_params=True, random_state=18,
            other_metrics=['mod_chi2'], pprint=False, blind=False,
            external_priors_penalty=None
        )
    except OptimizeWarning:
        logging.warn("Minimization was unsuccessful, but this is highly likely"
                     " to be a glitch related to the chosen minimizer config.")

    fit_settings = apply_fit_settings(fit_cfg, hypo_maker.params.free)
    minimizer_settings = {
        'global': fit_settings['minimize']['global'],
        'local': local_minimizer_cfg
    }
    # ensure the fit finishes up quickly
    override_min_opt(minimizer_settings['global'], ('niter_success:2', 'niter:5'))
    try:
        a._fit_hypo_minimizer( # pylint: disable=protected-access
            data_dist=data_dist, hypo_maker=hypo_maker, metric='chi2',
            minimizer_settings=minimizer_settings,
            randomize_params=True, random_state=18,
            other_metrics=['mod_chi2'], pprint=False, blind=False,
            external_priors_penalty=None
        )
    except OptimizeWarning:
        logging.warn("Minimization was unsuccessful, but this is highly likely"
                     " to be a glitch related to the chosen minimizer config.")
# --------------------------------------------------------------------------- #
def test_fit_hypo_pull(
    param_variations=None,
    pipeline_cfg='settings/pipeline/example.cfg'
):
    """Unit tests of `Analysis._fit_hypo_pull`."""

    data_maker = DistributionMaker(pipelines=pipeline_cfg)
    hypo_maker = DistributionMaker(pipelines=pipeline_cfg)

    if param_variations is None:
        param_variations = {
            'aeff_scale': 0.05 * ureg.dimensionless,
            'nue_numu_ratio': -0.1 * ureg.dimensionless
        }
    else:
        for pname in param_variations:
            assert pname in hypo_maker.params.names
            hypo_maker.params[pname].is_fixed = False

    for pname, variation in param_variations.items():
        nominal = data_maker.params[pname].nominal_value
        data_maker.params[pname].value = nominal + variation.to(nominal.units)

    data_dist = data_maker.get_outputs(return_sum=True)

    # we want to test whether we can get back the parameters
    # varied away from nominal above
    for param in hypo_maker.params.free:
        if not param.name in param_variations:
            param.is_fixed = True

    pull_settings = {'params': [], 'values': []}
    for pname in param_variations:
        pull_settings['params'].append(pname)
        param = hypo_maker.params[pname]
        param.is_fixed = False
        # set sensible ranges over which difference quotients are computed
        # (don't have to include the true value of the parameter
        # to fit it back)
        if param.nominal_value.m > 0:
            test_vals = [0.95*param.nominal_value, 1.05*param.nominal_value]
        elif param.nominal_value.m == 0:
            test_vals = [-0.05*param.nominal_value.units,
                          0.05*param.nominal_value.units]
        else:
            test_vals = [1.05*param.nominal_value, 0.95*param.nominal_value]
        pull_settings['values'].append(test_vals)

    a = Analysis()
    fit_info = a._fit_hypo_pull( # pylint: disable=protected-access
        data_dist=data_dist,
        hypo_maker=hypo_maker,
        pull_settings=pull_settings,
        metric='chi2'
    )

    msg = 'fit vs. true parameter *pulls*:\n'
    for pname, variation in param_variations.items():
        true_pull = variation.to(hypo_maker.params[pname].nominal_value.units).m
        fit_pull = fit_info['params'][pname].value - data_maker.params[pname].nominal_value
        msg += ' '*12
        msg += '%s: %.5f (fit) vs. %.5f (truth)\n' % (pname, fit_pull, true_pull)
    logging.info(msg)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    set_verbosity(1)
    test_optimize_discrete_selections()
    test_fit_hypo_minimizer()
    test_fit_hypo_pull()
