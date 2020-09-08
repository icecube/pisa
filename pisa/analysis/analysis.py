"""
Common tools for performing an analysis collected into a single class
`Analysis` that can be subclassed by specific analyses.
"""


from __future__ import absolute_import, division

from collections.abc import Sequence
from collections import OrderedDict, Mapping
from copy import deepcopy
from itertools import product
import re
import sys
import time

import numpy as np
import scipy.optimize as optimize

from pisa import EPSILON, FTYPE, ureg
from pisa.core.detectors import Detectors
from pisa.core.map import Map, MapSet
from pisa.core.param import ParamSet
from pisa.utils.comparisons import recursiveEquality
from pisa.utils.log import logging
from pisa.utils.fileio import to_file
from pisa.utils.stats import METRICS_TO_MAXIMIZE, METRICS_TO_MINIMIZE


__all__ = ['MINIMIZERS_USING_SYMM_GRAD',
           'set_minimizer_defaults', 'validate_minimizer_settings',
           'Counter', 'Analysis']

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


# TODO: add Nelder-Mead, as it was used previously...
def validate_minimizer_settings(minimizer_settings):
    """Validate minimizer settings.

    See source for specific thresholds set.

    Parameters
    ----------
    minimizer_settings : dict

    Raises
    ------
    ValueError
        If any minimizer settings are deemed to be invalid.

    """
    ftype_eps = np.finfo(FTYPE).eps
    method = minimizer_settings['method']['value'].lower()
    options = minimizer_settings['options']['value']
    if method == 'l-bfgs-b':
        must_have = ('maxcor', 'ftol', 'gtol', 'eps', 'maxfun', 'maxiter',
                     'maxls')
        may_have = must_have + ('args', 'jac', 'bounds', 'disp', 'iprint',
                                'callback')
    elif method == 'slsqp':
        must_have = ('maxiter', 'ftol', 'eps')
        may_have = must_have + ('args', 'jac', 'bounds', 'constraints',
                                'iprint', 'disp', 'callback')

    missing = set(must_have).difference(set(options))
    excess = set(options).difference(set(may_have))
    if missing:
        raise ValueError('Missing the following options for %s minimizer: %s'
                         % (method, missing))
    if excess:
        raise ValueError('Excess options for %s minimizer: %s'
                         % (method, excess))

    eps_msg = '%s minimizer option %s(=%e) is < %d * %s_EPS(=%e)'
    eps_gt_msg = '%s minimizer option %s(=%e) is > %e'
    fp64_eps = np.finfo(np.float64).eps

    if method == 'l-bfgs-b':
        err_lim, warn_lim = 2, 10
        for s in ['ftol', 'gtol']:
            val = options[s]
            if val < err_lim * ftype_eps:
                raise ValueError(eps_msg % (method, s, val, err_lim, 'FTYPE',
                                            ftype_eps))
            if val < warn_lim * ftype_eps:
                logging.warning(eps_msg, method, s, val, warn_lim, 'FTYPE', ftype_eps)

        val = options['eps']
        err_lim, warn_lim = 1, 10
        if val < err_lim * fp64_eps:
            raise ValueError(eps_msg % (method, 'eps', val, err_lim, 'FP64',
                                        fp64_eps))
        if val < warn_lim * ftype_eps:
            logging.warning(eps_msg, method, 'eps', val, warn_lim, 'FTYPE', ftype_eps)

        err_lim, warn_lim = 0.25, 0.1
        if val > err_lim:
            raise ValueError(eps_gt_msg % (method, 'eps', val, err_lim))
        if val > warn_lim:
            logging.warning(eps_gt_msg, method, 'eps', val, warn_lim)

    if method == 'slsqp':
        err_lim, warn_lim = 2, 10
        val = options['ftol']
        if val < err_lim * ftype_eps:
            raise ValueError(eps_msg % (method, 'ftol', val, err_lim, 'FTYPE',
                                        ftype_eps))
        if val < warn_lim * ftype_eps:
            logging.warning(eps_msg, method, 'ftol', val, warn_lim, 'FTYPE', ftype_eps)

        val = options['eps']
        err_lim, warn_lim = 1, 10
        if val < err_lim * fp64_eps:
            raise ValueError(eps_msg % (method, 'eps', val, 1, 'FP64',
                                        fp64_eps))
        if val < warn_lim * ftype_eps:
            logging.warning(eps_msg, method, 'eps', val, warn_lim, 'FP64', fp64_eps)

        err_lim, warn_lim = 0.25, 0.1
        if val > err_lim:
            raise ValueError(eps_gt_msg % (method, 'eps', val, err_lim))
        if val > warn_lim:
            logging.warning(eps_gt_msg, method, 'eps', val, warn_lim)


def check_t23_octant(fit_info):
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



def get_separate_t23_octant_params(hypo_maker, inflection_point) :
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
    theta23_case1 : Param
        theta23 param confined to first octant
    theta23_case2 : Param
        theta23 param confined to second octant
    '''

    # Reset theta23 before starting
    theta23 = hypo_maker.params.theta23
    theta23.reset()

    # Store the original theat23 param before we mess with it
    theta23_orig = deepcopy(theta23)

    # Get the octant definition
    octants = (
        (theta23.range[0], inflection_point) ,
        (inflection_point, theta23.range[1])
        )

    # If theta23 is maximal (e.g. the transition between octants) or very close 
    # to it, offset it slightly to be clearly in one octant (note that fit can 
    # still move the value back to maximal). The reason for this is that 
    # otherwise checks on the parameter bounds (which include a margin for 
    # minimizer tolerance) can an throw exception.
    tolerance = 0.1 * ureg.degree
    dist_from_inflection = theta23.value - inflection_point 
    if np.abs(dist_from_inflection) < tolerance :
        sign = -1. if dist_from_inflection < 0. else +1. # Note this creates +ve shift also for theta == 45 (arbitary)
        theta23.value = inflection_point + (sign * tolerance)

    # Store the cases
    theta23_case1 = deepcopy(theta23)
    theta23_case2 = deepcopy(theta23)

    # Get case 1, e.g. the current octant
    case1_octant_index = 0 if theta23_case1.value < inflection_point else 1
    theta23_case1.range = octants[case1_octant_index]

    # Also get case 2, e.g. the other octant
    case2_octant_index = 0 if case1_octant_index == 1 else 1
    theta23_case2.value = 2*inflection_point - theta23_case2.value
    theta23_case2.range = octants[case2_octant_index]

    return theta23_orig, theta23_case1, theta23_case2


# TODO: move this to a central location prob. in utils
class Counter(object):
    """Simple counter object for use as a minimizer callback."""
    def __init__(self, i=0):
        self._count = i

    def __str__(self):
        return str(self._count)

    def __repr__(self):
        return str(self)

    def __iadd__(self, inc):
        self._count += inc

    def reset(self):
        """Reset counter"""
        self._count = 0

    @property
    def count(self):
        """int : Current count"""
        return self._count


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

    def fit_hypo(self, data_dist, hypo_maker, hypo_param_selections, metric,
                 minimizer_settings, reset_free=True, 
                 check_octant=True, fit_octants_separately=True,
                 check_ordering=False, other_metrics=None,
                 blind=False, pprint=True, external_priors_penalty=None):
        """Fitter "outer" loop: If `check_octant` is True, run
        `fit_hypo_inner` starting in each octant of theta23 (assuming that
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

        check_ordering : bool
            If the ordering is not in the hypotheses already being tested, the
            fit will be run in both orderings.

        other_metrics : None, string, or list of strings
            After finding the best fit, these other metrics will be evaluated
            for each output that contributes to the overall fit. All strings
            must be valid metrics, as per `VALID_METRICS`, or the
            special string 'all' can be specified to evaluate all
            VALID_METRICS..

        pprint : bool
            Whether to show live-update of minimizer progress.

        blind : bool or int
            Whether to carry out a blind analysis. If True or 1, this hides actual
            parameter values from display and disallows these (as well as Jacobian,
            Hessian, etc.) from ending up in logfiles. If given an integer > 1, the
            fitted parameters are also prevented from being stored in fit info
            dictionaries.

        external_priors_penalty : func
            User defined prior penalty function. Adds an extra penalty
            to the metric that is minimized, depending on the input function.


        Returns
        -------
        best_fit_info : OrderedDict (see fit_hypo_inner method for details of
            `fit_info` dict)
        alternate_fits : list of `fit_info` from other fits run

        """
        if isinstance(metric, str):
            metric = [metric]

        if ( not check_octant ) and fit_octants_separately :
            raise ValueError("If 'check_octant' is False, 'fit_octants_separately' must be False")

        if check_ordering:
            if 'nh' in hypo_param_selections or 'ih' in hypo_param_selections:
                raise ValueError('One of the orderings has already been '
                                 'specified as one of the hypotheses but the '
                                 'fit has been requested to check both. These '
                                 'are incompatible.')

            logging.info('Performing fits in both orderings.')
            extra_param_selections = ['nh', 'ih']
        else:
            extra_param_selections = [None]

        alternate_fits = []

        for extra_param_selection in extra_param_selections:
            if extra_param_selection is not None:
                full_param_selections = hypo_param_selections
                full_param_selections.append(extra_param_selection)
            else:
                full_param_selections = hypo_param_selections
            # Select the version of the parameters used for this hypothesis
            hypo_maker.select_params(full_param_selections)

            # Reset free parameters to nominal values
            if reset_free:
                hypo_maker.reset_free()
            else:
                # Saves the current minimizer start values for the octant check
                minimizer_start_params = hypo_maker.params

            # Determine if checking theta23 octant
            peforming_octant_check = check_octant and ( 'theta23' in hypo_maker.params.free.names )

            #Determine inflection point, e.g. transition between octants
            if peforming_octant_check :
                inflection_point = (45.*ureg.deg).to(hypo_maker.params.theta23.units)

            # If fitting each theta23 octant separately, create distinct params for the 
            # two cases (also store the original param so can reset later)
            # Start with the first case
            if peforming_octant_check and fit_octants_separately :
                theta23_orig,theta23_case1,theta23_case2 = get_separate_t23_octant_params(hypo_maker,inflection_point)
                hypo_maker.update_params(theta23_case1)

            # Perform the fit
            best_fit_info = self.fit_hypo_inner(
                hypo_maker=hypo_maker,
                data_dist=data_dist,
                metric=metric,
                minimizer_settings=minimizer_settings,
                other_metrics=other_metrics,
                pprint=pprint,
                blind=blind,
                external_priors_penalty=external_priors_penalty
            )
            

            # Decide whether fit for other octant is necessary
            if peforming_octant_check :
                logging.debug('checking other octant of theta23')
                if reset_free:
                    hypo_maker.reset_free()
                else:
                    for param in minimizer_start_params:
                        hypo_maker.params[param.name].value = param.value

                #Determine new values for theta23 parameter in the other octant
                if fit_octants_separately :
                    #Use with the second case
                    hypo_maker.update_params(theta23_case2)
                else :
                    # Hop to other octant by reflecting about 45 deg
                    theta23 = hypo_maker.params.theta23
                    theta23.value = 2*inflection_point - theta23.value
                    hypo_maker.update_params(theta23)

                # Re-run minimizer starting at new point
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
                        logging.debug('Accepting other-octant fit')
                else:
                    alternate_fits.append(new_fit_info)
                    if not blind:
                        logging.debug('Accepting initial-octant fit')

                # If changed the range of the theta23 param whilst checking octants
                # reset the range now.
                # Keep the final value though (is up to the reset_free param
                # to deal with resetting this)
                if fit_octants_separately :
                    theta23_orig.value = hypo_maker.params.theta23.value
                    hypo_maker.update_params(theta23_orig)

        return best_fit_info, alternate_fits

    def fit_hypo_inner(self, data_dist, hypo_maker, metric, minimizer_settings,
                       other_metrics=None, pprint=True, blind=False, external_priors_penalty=None):
        """Fitter "inner" loop: Run an arbitrary scipy minimizer to modify
        hypo dist maker's free params until the data_dist is most likely to have
        come from this hypothesis.

        Note that an "outer" loop can handle discrete scanning over e.g. the
        octant for theta23; for each discrete point the "outer" loop can make a
        call to this "inner" loop. One such "outer" loop is implemented in the
        `fit_hypo` method.


        Parameters
        ----------
        data_dist : MapSet or List of MapSets
            Data distribution(s)

        hypo_maker : Detectors, DistributionMaker or convertible thereto

        metric : string or iterable of strings

        minimizer_settings : dict

        other_metrics : None, string, or sequence of strings

        pprint : bool
            Whether to show live-update of minimizer progress.

        blind : bool or int

        external_priors_penalty : func
            User defined prior penalty function


        Returns
        -------
        fit_info : OrderedDict with details of the fit with keys 'metric',
            'metric_val', 'params', 'hypo_asimov_dist', and
            'minimizer_metadata'

        """
        minimizer_settings = set_minimizer_defaults(minimizer_settings)
        validate_minimizer_settings(minimizer_settings)

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

        # Get starting free parameter values
        x0 = hypo_maker.params.free._rescaled_values # pylint: disable=protected-access

        minimizer_method = minimizer_settings['method']['value'].lower()
        if minimizer_method in MINIMIZERS_USING_SYMM_GRAD:
            logging.warning(
                'Minimizer %s requires artificial boundaries SMALLER than the'
                ' user-specified boundaries (so that numerical gradients do'
                ' not exceed the user-specified boundaries).',
                minimizer_method
            )
            step_size = minimizer_settings['options']['value']['eps']
            bounds = [(0 + step_size, 1 - step_size)]*len(x0)
        else:
            bounds = [(0, 1)]*len(x0)

        clipped_x0 = []
        for param, x0_val, bds in zip(hypo_maker.params.free, x0, bounds):
            if x0_val < bds[0] - EPSILON:
                raise ValueError(
                    'Param %s, initial scaled value %.17e is below lower bound'
                    ' %.17e.' % (param.name, x0_val, bds[0])
                )
            if x0_val > bds[1] + EPSILON:
                raise ValueError(
                    'Param %s, initial scaled value %.17e exceeds upper bound'
                    ' %.17e.' % (param.name, x0_val, bds[1])
                )

            clipped_x0_val = np.clip(x0_val, a_min=bds[0], a_max=bds[1])
            clipped_x0.append(clipped_x0_val)

            if recursiveEquality(clipped_x0_val, bds[0]):
                logging.warning(
                    'Param %s, initial scaled value %e is at the lower bound;'
                    ' minimization may fail as a result.',
                    param.name, clipped_x0_val
                )
            if recursiveEquality(clipped_x0_val, bds[1]):
                logging.warning(
                    'Param %s, initial scaled value %e is at the upper bound;'
                    ' minimization may fail as a result.',
                    param.name, clipped_x0_val
                )

        x0 = tuple(clipped_x0)

        logging.debug('Running the %s minimizer...', minimizer_method)

        # Using scipy.optimize.minimize allows a whole host of minimizers to be
        # used.
        counter = Counter()
        
        fit_history = []
        fit_history.append(list(metric) + [v.name for v in hypo_maker.params.free])

        start_t = time.time()

        if pprint and not blind:
            free_p = hypo_maker.params.free

            # Display any units on top
            r = re.compile(r'(^[+0-9.eE-]* )|(^[+0-9.eE-]*$)')
            hdr = ' '*(6+1+10+1+12+3)
            unt = []
            for p in free_p:
                u = r.sub('', format(p.value, '~')).replace(' ', '')[0:10]
                if u:
                    u = '(' + u + ')'
                unt.append(u.center(12))
            hdr += ' '.join(unt)
            hdr += '\n'

            # Header names
            hdr += ('iter'.center(6) + ' ' + 'funcalls'.center(10) + ' ' +
                    metric[0][0:12].center(12) + ' | ')
            hdr += ' '.join([p.name[0:12].center(12) for p in free_p])
            hdr += '\n'

            # Underscores
            hdr += ' '.join(['-'*6, '-'*10, '-'*12, '+'] + ['-'*12]*len(free_p))
            hdr += '\n'

            sys.stdout.write(hdr)

        # reset number of iterations before each minimization
        self._nit = 0

        #
        # From that point on, optimize starts using the metric and 
        # iterates, no matter what you do 
        #
        optimize_result = optimize.minimize(
            fun=self._minimizer_callable,
            x0=x0,
            args=(hypo_maker, data_dist, metric, counter, fit_history, pprint,
                  blind, external_priors_penalty),
            bounds=bounds,
            method=minimizer_settings['method']['value'],
            options=minimizer_settings['options']['value'],
            callback=self._minimizer_callback
        )
        end_t = time.time()
        if pprint:
            # clear the line
            sys.stdout.write('\n\n')
            sys.stdout.flush()

        minimizer_time = end_t - start_t

        logging.info(
            'Total time to optimize: %8.4f s; # of dists generated: %6d;'
            ' avg dist gen time: %10.4f ms',
            minimizer_time, counter.count, minimizer_time*1000./counter.count
        )

        # Will not assume that the minimizer left the hypo maker in the
        # minimized state, so set the values now (also does conversion of
        # values from [0,1] back to physical range)
        rescaled_pvals = optimize_result.pop('x')
        hypo_maker._set_rescaled_free_params(rescaled_pvals) # pylint: disable=protected-access

        # Record the Asimov distribution with the optimal param values
        hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)

        # Get the best-fit metric value
        metric_val = sign * optimize_result.pop('fun')

        # Record minimizer metadata (all info besides 'x' and 'fun'; also do
        # not record some attributes if performing blinded analysis)
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
        if blind > 1:  # only at stricter blindness level
            # Reset to starting value of the fit, rather than nominal values because
            # the nominal value might be out of range if this is inside an octant check.
            hypo_maker._set_rescaled_free_params(x0)
            fit_info['params'] = ParamSet()
        else:
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
        fit_info['minimizer_time'] = minimizer_time * ureg.sec
        fit_info['num_distributions_generated'] = counter.count
        fit_info['minimizer_metadata'] = metadata
        fit_info['fit_history'] = fit_history
        # If blind replace hypo_asimov_dist with none object
        if blind:
            hypo_asimov_dist = None
        fit_info['hypo_asimov_dist'] = hypo_asimov_dist

        if not optimize_result.success:
            if blind:
                msg = ''
            else:
                msg = ' ' + optimize_result.message
            raise ValueError('Optimization failed.' + msg)

        return fit_info

    def nofit_hypo(self, data_dist, hypo_maker, hypo_param_selections,
                   hypo_asimov_dist, metric, other_metrics=None, blind=False, external_priors_penalty=None):
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

        # NOTE: Select params but *do not* reset to nominal values to record
        # the current (presumably already optimal) param values
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
                logging.error(str(e))
            raise

        fit_info['metric_val'] = metric_val

        if blind:
            # Okay, if blind analysis is being performed, reset the values so
            # the user can't find them in the object
            hypo_maker.reset_free()
            fit_info['params'] = ParamSet()
        else:
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
        metric
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
                            metric, counter, fit_history, pprint, blind, external_priors_penalty=None):
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

        metric : string or iterable of strings
            Metric by which to evaluate the fit. See Map

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
                logging.error('Minimizer failed')
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
                logging.error(str(e))
            raise

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
        else:
            logging.trace(msg)

        counter += 1

        if not blind:
            fit_history.append(
                [metric_val] + [v.value.m for v in hypo_maker.params.free]
            )
            
        return sign*metric_val

    def _minimizer_callback(self, xk): # pylint: disable=unused-argument
        """Passed as `callback` parameter to `optimize.minimize`, and is called
        after each iteration. Keeps track of number of iterations.

        Parameters
        ----------
        xk : list
            Parameter vector

        """
        self._nit += 1

    # TODO: move the complexity of defining a scan into a class with various
    # factory methods, and just pass that class to the scan method; we will
    # surely want to use scanning over parameters in more general ways, too:
    # * set (some) fixed params, then run (minimizer, scan, etc.) on free
    #   params
    # * set (some free or fixed) params, then check metric
    # where the setting of the params is done for some number of values.
    def scan(self, data_dist, hypo_maker, metric, hypo_param_selections=None,
             param_names=None, steps=None, values=None, only_points=None,
             outer=True, profile=True, minimizer_settings=None, outfile=None,
             debug_mode=1, **kwargs):
        """Set hypo maker parameters named by `param_names` according to
        either values specified by `values` or number of steps specified by
        `steps`, and return the `metric` indicating how well the data
        distribution is described by each Asimov distribution.

        Some flexibility in how the user can specify `values` is allowed, based
        upon the shapes of `param_names` and `values` and how the `outer` flag
        is set.

        Either `values` or `steps` must be specified, but not both.

        Parameters
        ----------
        data_dist : Sequence of MapSets or MapSet
            Data distribution(s). These are what the hypothesis is tasked to
            best describe during the optimization/comparison process.

        hypo_maker : Detectors, DistributionMaker or instantiable thereto
            Generates the expectation distribution under a particular
            hypothesis. This typically has (but is not required to have) some
            free parameters which will be modified by the minimizer to optimize
            the `metric` in case `profile` is set to True.

        hypo_param_selections : None, string, or sequence of strings
            A pipeline configuration can have param selectors that allow
            switching a parameter among two or more values by specifying the
            corresponding param selector(s) here. This also allows for a single
            instance of a DistributionMaker to generate distributions from
            different hypotheses.

        metric : string or iterable of strings
            The metric to use for optimization/comparison. Note that the
            optimized hypothesis also has this metric evaluated and reported for
            each of its output maps. Confer `pisa.core.map` for valid metrics.

        param_names : None, string, or sequence of strings
            If None, assume all parameters are to be scanned; otherwise,
            specifies only the name or names of parameters to be scanned.

        steps : None, integer, or sequence of integers
            Number of steps to take within the allowed range of the parameter
            (or parameters). Value(s) specified for `steps` must be >= 2. Note
            that the endpoints of the range are always included, and numbers of
            steps higher than 2 fill in between the endpoints.

            * If integer...
                  Take this many steps for each specified parameter.
            * If sequence of integers...
                  Take the coresponding number of steps within the allowed range
                  for each specified parameter.

        values : None, scalar, sequence of scalars, or sequence-of-sequences
          * If scalar...
                Set this value for the (one) param name in `param_names`.
          * If sequence of scalars...
              * if len(param_names) is 1, set its value to each number in the
                sequence.
              * otherwise, set each param in param_names to the corresponding
                value in `values`. There must be the same number of param names
                as values.
          * If sequence of (sequences or iterables)...
              * Each param name corresponds to one of the inner sequences, in
                the order that the param names are specified.
              * If `outer` is False, all inner sequences must have the same
                length, and there will be one Asimov distribution generated for
                each set of values across the inner sequences. In other words,
                there will be a total of len(inner sequence) Asimov
                distribution generated.
              * If `outer` is True, the lengths of inner sequences needn't be
                the same. This takes the outer product of the passed sequences
                to arrive at the permutations of the parameter values that will
                be used to produce Asimov distributions (essentially nested
                loops over each parameter). E.g., if two params are scanned,
                for each value of the first param's inner sequence, an Asimov
                distribution is produced for every value of the second param's
                inner sequence. In total, there will be
                ``len(inner seq0) * len(inner seq1) * ...``
                Asimov distributions produced.

        only_points : None, integer, or even-length sequence of integers
            Only select subset of points to be analysed by specifying their
            range of positions within the whole set (0-indexed, incremental).
            For the lazy amongst us...

        outer : bool
            If set to True and a sequence of sequences is passed for `values`,
            the points scanned are the *outer product* of the inner sequences.
            See `values` for a more detailed explanation.

        profile : bool
            If set to True, minimizes specified metric over all free parameters
            at each scanned point. Otherwise keeps them at their nominal values
            and only performs grid scan of the parameters specified in
            `param_names`.

        minimizer_settings : dict
            Dictionary containing the settings for minimization, which are
            only needed if `profile` is set to True. Hint: it has proven useful
            to sprinkle with a healthy dose of scepticism.

        outfile : string
            Outfile to store results to. Will be updated at each scan step to
            write out intermediate results to prevent loss of data in case
            the apocalypse strikes after all.

        debug_mode : int, either one of [0, 1, 2]
            If set to 2, will add a wealth of minimisation history and physics
            information to the output file. Otherwise, the output will contain
            the essentials to perform an analysis (0), or will hopefully be
            detailed enough for some simple debugging (1). Any other value for
            `debug_mode` will be set to 2.

        """

        if debug_mode not in (0, 1, 2):
            debug_mode = 2

        # Either `steps` or `values` must be specified, but not both (xor)
        assert (steps is None) != (values is None)

        if isinstance(param_names, str):
            param_names = [param_names]

        nparams = len(param_names)
        hypo_maker.select_params(hypo_param_selections)

        if values is not None:
            if np.isscalar(values):
                values = np.array([values])
                assert nparams == 1
            for i, val in enumerate(values):
                if not np.isscalar(val):
                    # no scalar here, need a corresponding parameter name
                    assert nparams >= i+1
                else:
                    # a scalar, can either have only one parameter or at least
                    # this many
                    assert nparams == 1 or nparams >= i+1
                    if nparams > 1:
                        values[i] = np.array([val])

        else:
            ranges = [hypo_maker.params[pname].range for pname in param_names]
            if np.issubdtype(type(steps), int):
                assert steps >= 2
                values = [np.linspace(r[0], r[1], steps)*r[0].units
                          for r in ranges]
            else:
                assert len(steps) == nparams
                assert np.all(np.array(steps) >= 2)
                values = [np.linspace(r[0], r[1], steps[i])*r[0].units
                          for i, r in enumerate(ranges)]

        if nparams > 1:
            steplist = [[(pname, val) for val in values[i]]
                        for (i, pname) in enumerate(param_names)]
        else:
            steplist = [[(param_names[0], val) for val in values[0]]]

        #Number of steps must be > 0
        assert len(steplist) > 0

        points_acc = []
        if only_points is not None:
            assert len(only_points) == 1 or len(only_points) % 2 == 0
            if len(only_points) == 1:
                points_acc = only_points
            for i in range(0, len(only_points)-1, 2):
                points_acc.extend(
                    list(range(only_points[i], 1 + only_points[i + 1]))
                )

        # Instead of introducing another multitude of tests above, check here
        # whether the lists of steps all have the same length in case `outer`
        # is set to False
        if nparams > 1 and not outer:
            assert np.all(len(steps) == len(steplist[0]) for steps in steplist)
            loopfunc = zip
        else:
            # With single parameter, can use either `zip` or `product`
            loopfunc = product

        params = hypo_maker.params

        # Fix the parameters to be scanned if `profile` is set to True
        params.fix(param_names)

        results = {'steps': {}, 'results': []}
        results['steps'] = {pname: [] for pname in param_names}
        for i, pos in enumerate(loopfunc(*steplist)):
            if points_acc and i not in points_acc:
                continue

            msg = ''
            for (pname, val) in pos:
                params[pname].value = val
                results['steps'][pname].append(val)
                if isinstance(val, float):
                    msg += '%s = %.2f '%(pname, val)
                elif isinstance(val, ureg.Quantity):
                    msg += '%s = %.2f '%(pname, val.magnitude)
                else:
                    raise TypeError("val is of type %s which I don't know "
                                    "how to deal with in the output "
                                    "messages."% type(val))
            logging.info('Working on point ' + msg)
            hypo_maker.update_params(params)

            # TODO: consistent treatment of hypo_param_selections and scanning
            if not profile or not hypo_maker.params.free:
                logging.info('Not optimizing since `profile` set to False or'
                             ' no free parameters found...')
                best_fit = self.nofit_hypo(
                    data_dist=data_dist,
                    hypo_maker=hypo_maker,
                    hypo_param_selections=hypo_param_selections,
                    hypo_asimov_dist=hypo_maker.get_outputs(return_sum=True),
                    metric=metric,
                    **{k: v for k,v in kwargs.items() if k not in ["pprint","reset_free","check_octant"]}
                )
            else:
                logging.info('Starting optimization since `profile` requested.')
                best_fit, _ = self.fit_hypo(
                    data_dist=data_dist,
                    hypo_maker=hypo_maker,
                    hypo_param_selections=hypo_param_selections,
                    metric=metric,
                    minimizer_settings=minimizer_settings,
                    **kwargs
                )
                # TODO: serialisation!
                for k in best_fit['minimizer_metadata']:
                    if k in ['hess', 'hess_inv']:
                        logging.debug("deleting %s", k)
                        del best_fit['minimizer_metadata'][k]

            best_fit['params'] = deepcopy(
                best_fit['params'].serializable_state
            )
            if isinstance(best_fit['hypo_asimov_dist'], Sequence):
                best_fit['hypo_asimov_dist'] = [deepcopy(
                    best_fit['hypo_asimov_dist'][i].serializable_state
                ) for i in range(len(best_fit['hypo_asimov_dist']))]
            else:
                best_fit['hypo_asimov_dist'] = deepcopy(
                    best_fit['hypo_asimov_dist'].serializable_state
                )

            # decide which information to retain based on chosen debug mode
            if debug_mode == 0 or debug_mode == 1:
                try:
                    del best_fit['fit_history']
                    del best_fit['hypo_asimov_dist']
                except KeyError:
                    pass

            if debug_mode == 0:
                # torch the woods!
                try:
                    del best_fit['minimizer_metadata']
                    del best_fit['minimizer_time']
                except KeyError:
                    pass

            results['results'].append(best_fit)
            if outfile is not None:
                # store intermediate results
                to_file(results, outfile)

        return results
