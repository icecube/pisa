"""
Common tools for performing an analysis collected into a single class
`Analysis` that can be subclassed by specific analyses.
"""


from __future__ import absolute_import, division

from collections import OrderedDict, Sequence
from copy import deepcopy
from itertools import product
import sys
import time

import numpy as np
import scipy.optimize as optimize

from pisa import EPSILON, FTYPE, ureg
from pisa.core.map import Map, MapSet
from pisa.core.param import ParamSet
from pisa.utils.fileio import to_file
from pisa.utils.log import logging
from pisa.utils.minimization import set_minimizer_defaults, _minimizer_x0_bounds,\
                                    validate_minimizer_settings,\
                                    display_minimizer_header, run_minimizer,\
                                    Counter, MINIMIZERS_USING_SYMM_GRAD
from pisa.utils.stats import METRICS_TO_MAXIMIZE


__all__ = ['Analysis']

__author__ = 'J.L. Lanfranchi, P. Eller, S. Wren'

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

ANALYSIS_METHODS = ('minimize', 'scan', 'pull')
"""Allowed parameter fitting methods."""

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

    def fit_hypo_new(self, data_dist, hypo_maker, hypo_param_selections, metric,
                     fit_settings=None, reset_free=False, minimizer_settings=None,
                     extra_hypo_param_selections=None, other_metrics=None,
                     blind=False, pprint=True):
        """require fit_settings to be dict like
        fit_settings = {'minimize': {'params': ['param3', ...],
                                     'seeds': ['seeds3', ....]}, ?
                        'scan':     {'params': ['param1', ...],
                                     'values': [steplist1, ...]},
                        'pull':     {'params': ['param2', ...],
                                     'values': [steplist2, ...]}
                       }
        """

        #validate_fit_settings(fit_settings)

        if not isinstance(extra_hypo_param_selections, Sequence):
            extra_hypo_param_selections = [extra_hypo_param_selections]

        for extra_param_selection in extra_hypo_param_selections:
            if extra_param_selection is not None:
                full_param_selections = hypo_param_selections
                full_param_selections.append(extra_param_selection)
            else:
                full_param_selections = hypo_param_selections
            # Select the version of the parameters used for this hypothesis
            hypo_maker.select_params(full_param_selections)

            assert set(fit_settings.keys()) == set(ANALYSIS_METHODS)
            minimize_params = fit_settings['minimize']['params']
            if minimize_params:
                assert minimizer_settings is not None

            scan_params = fit_settings['scan']['params']
            scan_vals = []
            for i,pname in enumerate(scan_params):
                scan_vals.append([(pname, val) for val in settings['scan']['values'][i]])

            pull_params = fit_settings['pull']['params']
            fit_settings.pop('scan')

            print minimize_params, scan_params, pull_params
            for pname in tuple(minimize_params) + tuple(scan_params) + tuple(pull_params):
                # require all params to be set to free initially
                assert pname in hypo_maker.params.free.names
            # TODO: excess, missing

            # the parameters to scan over need to be fixed
            hypo_maker.params.fix(scan_params)
            params = hypo_maker.params

            # TODO: if there are no scan_vals, we can just inject e.g. the nominal
            # value for each free parameter
            if not scan_vals:
                scan_vals = [[(pname, hypo_maker.params[pname].value)]
                              for pname in hypo_maker.params.free.names]

            for i, pos in enumerate(zip(*scan_vals)):
                print i, pos
                msg = ''
                sep = ', '
                for (pname, val) in pos:
                    params[pname].value = val
                    if isinstance(val, float) or isinstance(val, ureg.Quantity):
                        if msg:
                            msg += sep
                        msg += '%s = %s'%(pname, val)
                    else:
                        raise TypeError("val is of type %s which I don't know "
                                        "how to deal with in the output "
                                        "messages."% type(val))
                # Reset free parameters to nominal values
                if reset_free:
                    hypo_maker.reset_free()
                else:
                    # Saves the current minimizer start values (for the octant check)
                    optimizer_start_params = hypo_maker.params

                best_fit_info = self.fit_hypo_inner_new(
                    hypo_maker=hypo_maker,
                    data_dist=data_dist,
                    metric=metric,
                    fit_settings_inner=fit_settings,
                    minimizer_settings=minimizer_settings,
                    other_metrics=other_metrics,
                    pprint=pprint,
                    blind=blind
                )

        return best_fit_info


    def fit_hypo_inner_new(self, data_dist, hypo_maker, metric, fit_settings_inner,
                           minimizer_settings=None, other_metrics=None,
                           pprint=True, blind=False):

        pull_params = fit_settings_inner['pull']['params']
        minimize_params = fit_settings_inner['minimize']['params']

        # dispatch correct fitting method depending on combination of
        # pull and minimize params

        # no parameters to fit
        if not len(pull_params) and not len(minimize_params):
            logging.info("Nothing else to do. Calculating metric(s).")
            nofit_hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)
            fit_info = self.nofit_hypo(
                data_dist=data_dist,
                hypo_maker=hypo_maker,
                hypo_param_selections=None,
                hypo_asimov_dist=nofit_hypo_asimov_dist,
                metric=metric,
                other_metrics=other_metrics,
                blind=blind
           )

        # only parameters to optimize numerically
        elif len(minimize_params) and not len(pull_params):
            fit_info = self.fit_hypo_minimizer(
                data_dist=data_dist,
                hypo_maker=hypo_maker,
                minimizer_settings=minimizer_settings,
                metric=metric,
                other_metrics=other_metrics,
                blind=blind
            )

        # only parameters to fit with pull method
        elif len(pull_params) and not len(minimize_params):
            raise NotImplementedError("Pull method not implemented yet!")
            fit_info = self.fit_hypo_pull(
                data_dist=data_dist,
                hypo_maker=hypo_maker,
                metric=metric,
                other_metrics=other_metrics,
                blind=blind
            )
        # parameters to optimize numerically and to fit with pull method
        else:
            raise NotImplementedError(
                "Combination of minimization and pull method not implemented yet!"
            )
        return fit_info


    def fit_hypo(self, data_dist, hypo_maker, hypo_param_selections, metric,
                 minimizer_settings, reset_free=True, check_octant=True,
                 check_ordering=False, other_metrics=None,
                 blind=False, pprint=True):
        """Fitter "outer" loop: If `check_octant` is True, run
        `fit_hypo_inner` starting in each octant of theta23 (assuming that
        is a param in the `hypo_maker`). Otherwise, just run the inner
        method once.

        Note that prior to running the fit, the `hypo_maker` has
        `hypo_param_selections` applied and its free parameters are reset to
        their nominal values.

        Parameters
        ----------
        data_dist : MapSet
            Data distribution(s). These are what the hypothesis is tasked to
            best describe during the optimization process.

        hypo_maker : DistributionMaker or instantiable thereto
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

        metric : string
            The metric to use for optimization. Valid metrics are found in
            `VALID_METRICS`. Note that the optimized hypothesis also has this
            metric evaluated and reported for each of its output maps.

        minimizer_settings : string or dict

        check_octant : bool
            If theta23 is a parameter to be used in the optimization (i.e.,
            free), the fit will be re-run in the second (first) octant if
            theta23 is initialized in the first (second) octant.

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

        blind : bool
            Whether to carry out a blind analysis. This hides actual parameter
            values from display and disallows these (as well as Jacobian,
            Hessian, etc.) from ending up in logfiles.


        Returns
        -------
        best_fit_info : OrderedDict (see fit_hypo_inner method for details of
            `fit_info` dict)
        alternate_fits : list of `fit_info` from other fits run

        """

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

            best_fit_info = self.fit_hypo_inner(
                hypo_maker=hypo_maker,
                data_dist=data_dist,
                metric=metric,
                minimizer_settings=minimizer_settings,
                other_metrics=other_metrics,
                pprint=pprint,
                blind=blind
            )

            # Decide whether fit for other octant is necessary
            if check_octant and 'theta23' in hypo_maker.params.free.names:
                logging.debug('checking other octant of theta23')
                if reset_free:
                    hypo_maker.reset_free()
                else:
                    for param in minimizer_start_params:
                        hypo_maker.params[param.name].value = param.value

                # Hop to other octant by reflecting about 45 deg
                theta23 = hypo_maker.params.theta23
                inflection_point = (45*ureg.deg).to(theta23.units)
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
                    blind=blind
                )

                # Check to make sure these two fits were either side of 45
                # degrees.
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
                    new_fit_info = self.fit_hypo_inner(
                        hypo_maker=hypo_maker,
                        data_dist=data_dist,
                        metric=metric,
                        minimizer_settings=minimizer_settings,
                        other_metrics=other_metrics,
                        pprint=pprint,
                        blind=blind
                    )
                    # Make sure the new octant is sensible
                    check_t23_octant(new_fit_info)

                # Take the one with the best fit
                if metric in METRICS_TO_MAXIMIZE:
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

        return best_fit_info, alternate_fits


    def fit_hypo_minimizer(self, data_dist, hypo_maker, metric, minimizer_settings,
                           other_metrics=None, pprint=True, blind=False):
        """Fitter "inner" loop: Run an arbitrary scipy minimizer to modify
        hypo dist maker's free params until the data_dist is most likely to have
        come from this hypothesis.

        Note that an "outer" loop can handle discrete scanning over e.g. the
        octant for theta23; for each discrete point the "outer" loop can make a
        call to this "inner" loop. One such "outer" loop is implemented in the
        `fit_hypo` method.


        Parameters
        ----------
        data_dist : MapSet
            Data distribution(s)

        hypo_maker : DistributionMaker or convertible thereto

        metric : string

        minimizer_settings : dict

        other_metrics : None, string, or sequence of strings

        pprint : bool
            Whether to show live-update of minimizer progress.

        blind : bool


        Returns
        -------
        fit_info : OrderedDict with details of the fit with keys 'metric',
            'metric_val', 'params', 'hypo_asimov_dist', and
            'minimizer_metadata'

        """
        use_global_minimizer = False
        # allow for an entry of `None` but also no entry at all
        try:
            minimizer_settings_global = minimizer_settings['global']
        except:
            minimizer_settings_global = None

        if minimizer_settings_global is not None:
            minimizer_settings_global =\
                set_minimizer_defaults(minimizer_settings_global)
            use_global_minimizer = True
            validate_minimizer_settings(minimizer_settings_global)
        minimizer_settings['global'] = minimizer_settings_global

        use_local_minimizer = False
        try:
            minimizer_settings_local = minimizer_settings['local']
        except:
            minimizer_settings_local = None

        # TODO: only require this for now
        assert minimizer_settings_local is not None

        if minimizer_settings_local is not None:
            minimizer_settings_local =\
                set_minimizer_defaults(minimizer_settings_local)
            use_local_minimizer = True
            validate_minimizer_settings(minimizer_settings_local)
        minimizer_settings['local'] = minimizer_settings_local

        sign = -1 if metric in METRICS_TO_MAXIMIZE else +1

        # TODO: bounds handling in case global minimization requested
        # should they only depend on local minimimizer?
        x0, bounds = _minimizer_x0_bounds(
            free_params=hypo_maker.params.free,
            minimizer_settings=minimizer_settings_local
        )

        # Using scipy.optimize.minimize allows a whole host of minimizers to be
        # used.
        counter = Counter()
        
        fit_history = []
        fit_history.append( [metric] + [v.name for v in hypo_maker.params.free])

        if pprint and not blind:
            # display header if desired/allowed
            display_minimizer_header(free_params=hypo_maker.params.free,
                                     metric=metric)

        # reset number of iterations before each minimization
        self._nit = 0

        # record start time
        start_t = time.time()

        # this is the function that does the heavy lifting
        optimize_result = run_minimizer(
            fun=self._minimizer_callable,
            x0=x0,
            bounds=bounds,
            minimizer_settings=minimizer_settings,
            minimizer_callback=self._minimizer_callback,
            hypo_maker=hypo_maker,
            data_dist=data_dist,
            metric=metric,
            counter=counter,
            fit_history=fit_history,
            pprint=pprint,
            blind=blind
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
            metadata[k] = optimize_result[k]

        fit_info = OrderedDict()
        fit_info['metric'] = metric
        fit_info['metric_val'] = metric_val
        if blind:
            hypo_maker.reset_free()
            fit_info['params'] = ParamSet()
        else:
            fit_info['params'] = deepcopy(hypo_maker.params)
        fit_info['detailed_metric_info'] = self.get_detailed_metric_info(
            data_dist=data_dist, hypo_asimov_dist=hypo_asimov_dist,
            params=hypo_maker.params, metric=metric, other_metrics=other_metrics
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
                   hypo_asimov_dist, metric, other_metrics=None, blind=False):
        """Fitting a hypo to Asimov distribution generated by its own
        distribution maker is unnecessary. In such a case, use this method
        (instead of `fit_hypo`) to still retrieve meaningful information for
        e.g. the match metrics.

        Parameters
        ----------
        data_dist : MapSet
        hypo_maker : DistributionMaker
        hypo_param_selections : None, string, or sequence of strings
        hypo_asimov_dist : MapSet
        metric : string
        other_metrics : None, string, or sequence of strings
        blind : bool

        """
        fit_info = OrderedDict()
        fit_info['metric'] = metric

        # NOTE: Select params but *do not* reset to nominal values to record
        # the current (presumably already optimal) param values
        hypo_maker.select_params(hypo_param_selections)

        # Assess the fit: whether the data came from the hypo_asimov_dist
        try:
            metric_val = (
                data_dist.metric_total(expected_values=hypo_asimov_dist,
                                       metric=metric)
                + hypo_maker.params.priors_penalty(metric=metric)
            )
        except:
            if not blind:
                logging.error(
                    'Failed when computing metric with free params %s',
                    hypo_maker.params.free
                )
            raise

        fit_info['metric_val'] = metric_val

        if blind:
            # Okay, if blind analysis is being performed, reset the values so
            # the user can't find them in the object
            hypo_maker.reset_free()
            fit_info['params'] = ParamSet()
        else:
            fit_info['params'] = deepcopy(hypo_maker.params)
        fit_info['detailed_metric_info'] = self.get_detailed_metric_info(
            data_dist=data_dist, hypo_asimov_dist=hypo_asimov_dist,
            params=hypo_maker.params, metric=metric, other_metrics=other_metrics
        )
        fit_info['minimizer_time'] = 0 * ureg.sec
        fit_info['num_distributions_generated'] = 0
        fit_info['minimizer_metadata'] = OrderedDict()
        fit_info['hypo_asimov_dist'] = hypo_asimov_dist
        return fit_info

    @staticmethod
    def get_detailed_metric_info(data_dist, hypo_asimov_dist, params, metric,
                                 other_metrics=None):
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
        elif isinstance(other_metrics, basestring):
            other_metrics = [other_metrics]
        all_metrics = sorted(set([metric] + other_metrics))
        detailed_metric_info = OrderedDict()
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
                            metric, counter, fit_history, pprint, blind):
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

        hypo_maker : DistributionMaker
            Creates the per-bin expectation values per map (aka Asimov
            distribution) based on its param values. Free params in the
            `hypo_maker` are modified by the minimizer to achieve a "best" fit.

        data_dist : MapSet
            Data distribution to be fit. Can be an actual-, Asimov-, or
            pseudo-data distribution (where the latter two are derived from
            simulation and so aren't technically "data").

        metric : str
            Metric by which to evaluate the fit. See Map

        counter : Counter
            Mutable object to keep track--outside this method--of the number of
            times this method is called.

        pprint : bool
            Displays a single-line that updates live (assuming the entire line
            fits the width of your TTY).

        blind : bool

        """
        # Want to *maximize* e.g. log-likelihood but we're using a minimizer,
        # so flip sign of metric in those cases.
        sign = -1 if metric in METRICS_TO_MAXIMIZE else +1

        # Set param values from the scaled versions the minimizer works with
        hypo_maker._set_rescaled_free_params(scaled_param_vals) # pylint: disable=protected-access

        # Get the Asimov map set
        try:
            hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)
        except:
            if not blind:
                logging.error(
                    'Failed to generate Asimov distribution with free'
                    ' params %s', hypo_maker.params.free
                )
            raise

        # Assess the fit: whether the data came from the hypo_asimov_dist
        try:
            metric_val = (
                data_dist.metric_total(expected_values=hypo_asimov_dist,
                                       metric=metric)
                + hypo_maker.params.priors_penalty(metric=metric)
            )
        except:
            if not blind:
                logging.error(
                    'Failed when computing metric with free params %s',
                    hypo_maker.params.free
                )
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
            sys.stdout.write(msg)
            sys.stdout.flush()
            sys.stdout.write('\b' * len(msg))
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
        data_dist : MapSet
            Data distribution(s). These are what the hypothesis is tasked to
            best describe during the optimization/comparison process.

        hypo_maker : DistributionMaker or instantiable thereto
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

        metric : string
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

        if isinstance(param_names, basestring):
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
