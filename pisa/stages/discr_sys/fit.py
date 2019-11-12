"""
The purpose of this stage is to reweight an event sample to include effects of
so called "discrete" systematics.

This service in particular is intended to follow a `weight` service
which takes advantage of the Data object being passed as an output of the
Stage.
"""


from collections import OrderedDict
from copy import deepcopy

from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit

import numpy as np
from uncertainties import unumpy as unp
import pint

from pisa import FTYPE
from pisa import ureg
from pisa.core.binning import OneDimBinning
from pisa.core.events import Data
from pisa.core.map import Map, MapSet
from pisa.core.param import ParamSet
from pisa.core.stage import Stage
from pisa.core.pipeline import Pipeline
from pisa.utils.flavInt import ALL_NUFLAVINTS
from pisa.utils.flavInt import NuFlavIntGroup
from pisa.utils.fileio import from_file
from pisa.utils.format import text2tex
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile


__all__ = ['fit']

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


class fit(Stage):
    """discr_sys service to rewight an event sample to take into account
    discrete systematics.

    This type of systematic has been fluctuated at the MC level, so
    separate samples exists with variations of some systematic
    parameter. Since the generation of the alternative samples fix the
    amount of variation, a given sample will represent an individual
    value of the variation. To get a continous spectrum of the
    variations the systematics parameter causes in it's avaliable phase
    space, a curve is fit to the variations of the simulated samples.

    Parameters
    ----------
    params: ParamSet of sequence with which to instantiate a ParamSet
        Parameters which set everything besides the binning

        Parameters required by this service are
            * pipeline_config : filepath
                Filepath to pipeline config

            * discr_sys_sample_config : filepath
                Filepath to event sample configuration

            * stop_after_stage : int
                Extract templates up to this stage for the fitting

            * poly_degree : int
                Polynominal degree to use when fitting

            * force_through_nominal : bool
                Force the polynominal to pass through the nominal sample

            * smoothing : str
                Option to apply smoothing once fit coefficients have been
                calculated

            * Neutrino discrete systematics
                - nu_dom_eff
                - nu_hole_ice

            * Muon discrete systematics
                - mu_dom_eff
                - mu_hole_ice

            * cache_fit: bool
                Flag to specifiy whether to cache the fit values if
                calculated inside this service to a file specified by
                `disk_cache`.

    input_names : string
        Specifies the string representation of the NuFlavIntGroup(s) that
        belong in the Data object passed to this service.

    output_binning : MultiDimBinning or convertible thereto
        The binning desired for the output maps.
        NOTE: this is binning with which the curves will be fitted at.

    output_names : string
        Specifies the string representation of the NuFlavIntGroup(s) which will
        be produced as an output.

    output_events : bool
        Flag to specify whether the service output returns a MapSet
        or the full information about each event

    error_method : None, bool, or string
        If None, False, or empty string, the stage does not compute errors for
        the transforms and does not apply any (additional) error to produce its
        outputs. (If the inputs already have errors, these are propagated.)

    debug_mode : None, bool, or string
        If None, False, or empty string, the stage runs normally.
        Otherwise, the stage runs in debug mode. This disables caching (forcing
        recomputation of any nominal transforms, transforms, and outputs).

    transforms_cache_depth
    outputs_cache_depth : int >= 0

    """
    def __init__(self, params, output_binning, input_names, output_names,
                 output_events=True, error_method=None, debug_mode=None,
                 disk_cache=None, memcache_deepcopy=True,
                 outputs_cache_depth=20):
        self.sample_hash = None
        """Hash of input event sample."""
        self.weight_hash = None
        """Hash of event sample."""
        self.fit_hash = None
        """Hash of fit sample."""
        self.fitcoeffs_hash = None
        """Hash of fit coefficients."""
        self.fitcoeffs_cache_hash = None
        """Hash of cached fit coefficients."""

        self.fit_params = (
            'pipeline_config', 'discr_sys_sample_config', 'stop_after_stage',
            'poly_degree', 'force_through_nominal', 'smoothing'
        )

        self.nu_params = (
            'nu_dom_eff', 'nu_hole_ice'
        )

        self.mu_params = (
            'mu_dom_eff', 'mu_hole_ice'
        )

        self.other_params = (
            'cache_fit',
        )

        expected_params = self.fit_params + self.other_params
        if ('all_nu' in input_names) or ('neutrinos' in input_names):
            expected_params += self.nu_params
        if 'muons' in input_names:
            expected_params += self.mu_params

        self.neutrinos = False
        self.muons = False
        self.noise = False

        if input_names != output_names:
            raise AssertionError(
                'Input names must match output names for this '
                'stage\n{0}(input names) != {1}(output '
                'names)'.format(input_names, output_names)
            )

        output_names = output_names.replace(' ', '').split(',')
        clean_outnames = []
        self._output_nu_groups = []
        for name in output_names:
            if 'muons' in name:
                self.muons = True
                clean_outnames.append(name)
            elif 'noise' in name:
                self.noise = True
                clean_outnames.append(name)
            elif 'all_nu' in name:
                self.neutrinos = True
                self._output_nu_groups = \
                    [NuFlavIntGroup(f) for f in ALL_NUFLAVINTS]
            else:
                self.neutrinos = True
                self._output_nu_groups.append(NuFlavIntGroup(name))

        if self.neutrinos:
            clean_outnames += [str(f) for f in self._output_nu_groups]

        if not isinstance(output_events, bool):
            raise AssertionError(
                'output_events must be of type bool, instead it is supplied '
                'with type {0}'.format(type(output_events))
            )
        self.fit_binning = deepcopy(output_binning)
        if output_events:
            output_binning = None
        self.output_events = output_events

        super().__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            input_names=clean_outnames,
            output_names=clean_outnames,
            error_method=error_method,
            debug_mode=debug_mode,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning
        )

        if self.params['smoothing'].value is not None:
            if self.params['smoothing'].value != 'gauss':
                raise AssertionError(
                    'Parameter "smoothing" accepts "none" or "gauss" as '
                    'input, instead got {0} as '
                    'input'.format(self.params['smoothing'].value)
                )

        self.include_attrs_for_hashes('sample_hash')

    @profile
    def _compute_outputs(self, inputs=None):
        """Compute histograms for output channels."""
        logging.debug('Entering fit._compute_outputs')
        if not isinstance(inputs, Data):
            raise AssertionError('inputs is not a Data object, instead is '
                                 'type {0}'.format(type(inputs)))
        self.weight_hash = deepcopy(inputs.metadata['weight_hash'])
        logging.trace('{0} fit weight_hash = '
                      '{1}'.format(inputs.metadata['name'], self.weight_hash))
        logging.trace('{0} fit fit_hash = '
                      '{1}'.format(inputs.metadata['name'], self.fit_hash))
        self._data = inputs
        self.reweight()

        if self.output_events:
            return self._data

        outputs = []
        if self.neutrinos:
            trans_nu_data = self._data.transform_groups(
                self._output_nu_groups
            )
            for fig in trans_nu_data.keys():
                outputs.append(
                    trans_nu_data.histogram(
                        kinds=fig,
                        binning=self.output_binning,
                        weights_col='pisa_weight',
                        errors=True,
                        name=str(NuFlavIntGroup(fig)),
                    )
                )

        if self.muons:
            outputs.append(
                self._data.histogram(
                    kinds='muons',
                    binning=self.output_binning,
                    weights_col='pisa_weight',
                    errors=True,
                    name='muons',
                    tex=text2tex('muons')
                )
            )

        if self.noise:
            outputs.append(
                self._data.histogram(
                    kinds='noise',
                    binning=self.output_binning,
                    weights_col='pisa_weight',
                    errors=True,
                    name='noise',
                    tex=text2tex('noise')
                )
            )

        return MapSet(maps=outputs, name=self._data.metadata['name'])

    def reweight(self):
        """Main rewighting function."""
        this_hash = hash_obj(
            [self.weight_hash, self.params.values_hash],
            full_hash=self.full_hash
        )
        if this_hash == self.fit_hash:
            return

        fit_coeffs = self.calculate_fit_coeffs()

        sample_config = from_file(self.params['discr_sys_sample_config'].value)
        degree = int(self.params['poly_degree'].value)
        force_through_nominal = self.params['force_through_nominal'].value

        if force_through_nominal:
            def fit_func(vals, *poly_coeffs):
                return np.polynomial.polynomial.polyval(
                    vals, [1.] + list(poly_coeffs)
                )
        else:
            def fit_func(vals, *poly_coeffs):
                return np.polynomial.polynomial.polyval(
                    vals, list(poly_coeffs)
                )
            # add free param for constant term
            degree += 1

        def parse(string):
            return string.replace(' ', '').split(',')

        if self.neutrinos:
            sys_list = parse(sample_config.get('neutrinos', 'sys_list'))

            for fig in self._data.keys():
                self._data[fig]['fit_weight'] = \
                    deepcopy(self._data[fig]['weight_weight'])

            for sys in sys_list:
                nominal = sample_config.get('neutrinos|' + sys, 'nominal')
                for fig in self._data.keys():
                    fit_map = unp.nominal_values(fit_coeffs[sys][fig].hist)

                    if self.params['smoothing'].value == 'gauss':
                        # TODO(shivesh): new MapSet functions?
                        for d in range(degree):
                            fit_map[..., d] = gaussian_filter(
                                fit_map[..., d], sigma=1
                            )

                    shape = self.fit_binning.shape
                    transform = np.ones(shape)
                    sys_offset = self.params['nu_'+sys].value.m-float(nominal)
                    for idx in np.ndindex(shape):
                        transform[idx] *= fit_func(sys_offset, *fit_map[idx])

                    hist_idxs = self._data.digitize(
                        kinds   = fig,
                        binning = self.fit_binning,
                    )

                    # Discrete systematics reweighting
                    # TODO(shivesh): speedup this
                    for idx, wght in enumerate(
                        np.nditer(self._data[fig]['fit_weight'],
                                  op_flags=['readwrite'])
                    ):
                        idx_slice = tuple(hist_idxs[idx])
                        if shape[0] == 0 or shape[1] == 0 or \
                           idx_slice[0] > shape[0] or idx_slice[1] > shape[1]:
                            # Outside binning range
                            wght *= 0
                        else:
                            wght *= transform[tuple([x-1 for x in idx_slice])]

            for fig in self._data.keys():
                self._data[fig]['pisa_weight'] = \
                    deepcopy(self._data[fig]['fit_weight'])

        if self.muons:
            sys_list = parse(sample_config.get('muons', 'sys_list'))

            self._data['muons']['fit_weight'] = \
                deepcopy(self._data['muons']['weight_weight'])

            for sys in sys_list:
                fit_map = unp.nominal_values(fit_coeffs[sys]['muons'].hist)

                if self.params['smoothing'].value == 'gauss':
                    # TODO(shivesh): new MapSet functions?
                    for d in range(degree):
                        fit_map[..., d] = gaussian_filter(
                            fit_map[..., d], sigma=1
                        )

                shape = self.fit_binning.shape
                transform = np.ones(shape)
                for idx in np.ndindex(shape):
                    transform[idx] *= fit_func(
                        self.params['mu_'+sys].value, *fit_map[idx]
                    )

                hist_idxs = self._data.digitize(
                    kinds   = 'muons',
                    binning = self.fit_binning,
                )

                # Discrete systematics reweighting
                for idx, wght in enumerate(self._data['muons']['fit_weight']):
                    idx_slice = tuple(hist_idxs[idx])
                    if shape[0] == 0 or shape[1] == 0 or \
                       idx_slice[0] > shape[0] or idx_slice[1] > shape[1]:
                        # Outside binning range
                        wght *= 0
                    else:
                        wght *= transform[tuple([x-1 for x in idx_slice])]

                self._data['muons']['pisa_weight'] = \
                    deepcopy(self._data['muons']['fit_weight'])

        self.fit_hash = this_hash
        self._data.metadata['fit_hash'] = self.fit_hash
        self._data.update_hash()

    def calculate_fit_coeffs(self):
        """
        Calculate the fit coefficients for each systematic, flavint, bin
        for a polynomial.
        """
        this_hash = hash_obj(
            [self.fit_binning.hash, self.weight_hash] +
            [self.params[name].value for name in self.fit_params],
            full_hash=self.full_hash
        )
        if self.fitcoeffs_hash == this_hash:
            return self._fit_coeffs

        if self.neutrinos:
            nu_params = self.nu_params
        else:
            nu_params = None
        if self.muons:
            mu_params = self.mu_params
        else:
            mu_params = None

        if self.params['cache_fit'].value:
            this_cache_hash = hash_obj(
                [self._data.metadata['name'], self._data.metadata['sample'],
                 self._data.metadata['cuts'], self.fit_binning.hash] +
                [self.params[name].value for name in self.fit_params],
                full_hash=self.full_hash
            )

            if self.fitcoeffs_cache_hash == this_cache_hash:
                fit_coeffs = deepcopy(self._cached_fc)
            elif this_cache_hash in self.disk_cache:
                logging.info('Loading fit coefficients from cache.')
                self._cached_fc = self.disk_cache[this_cache_hash]
                fit_coeffs = deepcopy(self._cached_fc)
                self.fitcoeffs_cache_hash = this_cache_hash
            else:
                fit_coeffs = self._calculate_fit_coeffs(
                    self._data, ParamSet(p for p in self.params
                                         if p.name in self.fit_params),
                    self.fit_binning, nu_params, mu_params
                )
        else:
            fit_coeffs = self._calculate_fit_coeffs(
                self._data, ParamSet(p for p in self.params
                                     if p.name in self.fit_params),
                self.fit_binning, nu_params, mu_params
            )

        if self.params['cache_fit'].value:
            if this_cache_hash not in self.disk_cache:
                logging.info('Caching fit coefficients values to disk.')
                self.disk_cache[this_cache_hash] = fit_coeffs

        self.fitcoeffs_hash = this_hash
        self._fit_coeffs = fit_coeffs
        return fit_coeffs

    @staticmethod
    def _calculate_fit_coeffs(data, params, fit_binning, nu_params=None,
                              mu_params=None):
        """
        Calculate the fit coefficients for each systematic, flavint,
        bin for a polynomial.
        """
        logging.debug('Calculating fit coefficients')

        config = from_file(params['discr_sys_sample_config'].value)

        degree = int(params['poly_degree'].value)
        force_through_nominal = params['force_through_nominal'].value

        if force_through_nominal:
            def fit_func(vals, *poly_coeffs):
                return np.polynomial.polynomial.polyval(
                    vals, [1.] + list(poly_coeffs)
                )
        else:
            def fit_func(vals, *poly_coeffs):
                return np.polynomial.polynomial.polyval(
                    vals, list(poly_coeffs)
                )
            # add free param for constant term
            degree += 1

        template_maker = Pipeline(params['pipeline_config'].value)
        dataset_param = template_maker.params['dataset']

        def parse(string):
            return string.replace(' ', '').split(',')

        sys_fit_coeffs = OrderedDict()
        if nu_params is not None:
            sys_list = parse(config.get('neutrinos', 'sys_list'))
            nu_params = deepcopy(map(lambda x: x[3:], nu_params))

            if set(nu_params) != set(sys_list):
                raise AssertionError(
                    'Systematics list listed in the sample config file does '
                    'not match the params in the pipeline config file\n {0} '
                    '!= {1}'.format(set(nu_params), set(sys_list))
                )

            for sys in sys_list:
                ev_sys = 'neutrinos|' + sys
                runs = parse(config.get(ev_sys, 'runs')[1: -1])
                nominal = config.get(ev_sys, 'nominal')

                mapset_dict = OrderedDict()
                flavint_groups = None
                for run in runs:
                    logging.info('Loading run {0} of systematic '
                                 '{1}'.format(run, sys))
                    dataset_param.value = ev_sys + '|' + run
                    template_maker.update_params(dataset_param)
                    template = template_maker.get_outputs(
                        idx=int(params['stop_after_stage'].m)
                    )
                    if not isinstance(template, Data):
                        raise AssertionError(
                            'Template output is not a Data object, instead is '
                            'type {0}'.format(type(template))
                        )
                    if flavint_groups is None:
                        flavint_groups = template.flavint_groups
                    else:
                        if set(flavint_groups) != set(template.flavint_groups):
                            raise AssertionError(
                                'Mismatch of flavint_groups - ({0}) does not '
                                'match flavint_groups '
                                '({1})'.format(flavint_groups,
                                               template.flavint_groups)
                            )

                    outputs = []
                    for fig in template.keys():
                        outputs.append(template.histogram(
                            kinds       = fig,
                            binning     = fit_binning,
                            weights_col = 'pisa_weight',
                            errors      = False,
                            name        = str(NuFlavIntGroup(fig))
                        ))
                    mapset_dict[run] = MapSet(outputs, name=run)

                nom_mapset = mapset_dict[nominal]
                fracdiff_mapset_dict = OrderedDict()
                for run in runs:
                    mapset = []
                    for flavintg_map in mapset_dict[run]:
                        # TODO(shivesh): error propagation?
                        flavintg = flavintg_map.name
                        mask = ~(nom_mapset[flavintg].hist == 0.)
                        div = np.zeros(flavintg_map.shape)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            div[mask] = \
                                unp.nominal_values(flavintg_map.hist[mask]) /\
                                unp.nominal_values(nom_mapset[flavintg].hist[mask])
                        mapset.append(Map(
                            name=flavintg, binning=flavintg_map.binning,
                            hist=div
                        ))
                    fracdiff_mapset_dict[run] = MapSet(mapset)

                delta_runs = np.array([float(x) for x in runs])-float(nominal)

                coeff_binning = OneDimBinning(
                    name='coeff', num_bins=degree, is_lin=True, domain=[-1, 1]
                )
                combined_binning = fit_binning + coeff_binning

                params_mapset = []
                for fig in template.keys():
                    # TODO(shivesh): Fix numpy warning on this line
                    pvals_hist = np.empty(map(int, combined_binning.shape),
                                          dtype=object)
                    hists = [fracdiff_mapset_dict[run][fig].hist for run in runs]
                    zip_hists = np.dstack(hists)
                    for idx in np.ndindex(fit_binning.shape):
                        y_values = []
                        y_sigma = []
                        for run in fracdiff_mapset_dict:
                            y_values.append(unp.nominal_values(fracdiff_mapset_dict[run][fig].hist[idx]))
                            y_sigma.append(unp.std_devs(fracdiff_mapset_dict[run][fig].hist[idx]))

                        if np.any(y_sigma):
                            popt, pcov = curve_fit(
                                fit_func, delta_runs, y_values, sigma=y_sigma,
                                p0=np.ones(degree)
                            )
                        else:
                            popt, pcov = curve_fit(
                                fit_func, delta_runs, y_values,
                                p0=np.ones(degree)
                            )
                        # perr = np.sqrt(np.diag(pcov))
                        # pvals = unp.uarray(popt, perr)
                        pvals_hist[idx] = popt
                    pvals_hist = np.array(pvals_hist.tolist())
                    params_mapset.append(Map(
                        name=fig, binning=combined_binning, hist=pvals_hist
                    ))
                params_mapset = MapSet(params_mapset, name=sys)

                if sys in sys_fit_coeffs:
                    sys_fit_coeffs[sys] = MapSet(
                        [sys_fit_coeffs[sys], params_mapset]
                    )
                else:
                    sys_fit_coeffs[sys] = params_mapset

        if mu_params is not None:
            sys_list = parse(config.get('muons', 'sys_list'))
            mu_params = deepcopy(map(lambda x: x[3:], mu_params))

            if set(mu_params) != set(sys_list):
                raise AssertionError(
                    'Systematics list listed in the sample config file does '
                    'not match the params in the pipeline config file\n {0} '
                    '!= {1}'.format(set(mu_params), set(sys_list))
                )

            for sys in sys_list:
                ev_sys = 'muons|' + sys
                runs = parse(config.get(ev_sys, 'runs')[1: -1])
                nominal = config.get(ev_sys, 'nominal')

                map_dict = OrderedDict()
                flavint_groups = None
                for run in runs:
                    logging.info('Loading run {0} of systematic '
                                 '{1}'.format(run, sys))
                    dataset_param.value = ev_sys + '|' + run
                    template_maker.update_params(dataset_param)
                    template = template_maker.get_outputs(
                        idx=int(params['stop_after_stage'].m)
                    )
                    if not isinstance(template, Data):
                        raise AssertionError(
                            'Template output is not a Data object, instead is '
                            'type {0}'.format(type(template))
                        )
                    if not template.contains_muons:
                        raise AssertionError(
                            'Template output does not contain muons'
                        )

                    output = template.histogram(
                        kinds       = 'muons',
                        binning     = fit_binning,
                        # NOTE: weights cancel in fraction
                        weights_col = None,
                        errors      = False,
                        name        = 'muons'
                    )
                    map_dict[run] = output

                nom_map = map_dict[nominal]
                fracdiff_map_dict = OrderedDict()
                for run in runs:
                    mask = ~(nom_map.hist == 0.)
                    div = np.zeros(nom_map.shape)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        div[mask] = \
                            unp.nominal_values(map_dict[run].hist[mask]) /\
                            unp.nominal_values(nom_map.hist[mask])
                    fracdiff_map_dict[run] = Map(
                        name='muons', binning = nom_map.binning, hist=div
                    )

                delta_runs = np.array([float(x) for x in runs])-float(nominal)

                coeff_binning = OneDimBinning(
                    name='coeff', num_bins=degree, is_lin=True, domain=[-1, 1]
                )
                combined_binning = fit_binning + coeff_binning

                pvals_hist = np.empty(map(int, combined_binning.shape),
                                      dtype=object)
                hists = [fracdiff_map_dict[run].hist for run in runs]
                zip_hists = np.dstack(hists)
                for idx in np.ndindex(fit_binning.shape):
                    y_values = [] 
                    y_sigma = []
                    for run in fracdiff_mapset_dict:
                        y_values.append(unp.nominal_values(fracdiff_mapset_dict[run][fig].hist[idx]))
                        y_sigma.append(unp.std_devs(fracdiff_mapset_dict[run][fig].hist[idx]))
                    if np.any(y_sigma):
                        popt, pcov = curve_fit(
                            fit_func, delta_runs, y_values, sigma=y_sigma,
                            p0=np.ones(degree)
                        )
                    else:
                        popt, pcov = curve_fit(
                            fit_func, delta_runs, y_values,
                            p0=np.ones(degree)
                        )
                    # perr = np.sqrt(np.diag(pcov))
                    # pvals = unp.uarray(popt, perr)
                    pvals_hist[idx] = popt
                pvals_hist = np.array(pvals_hist.tolist())
                params_map = Map(
                    name='muons', binning=combined_binning, hist=pvals_hist
                )
                if sys in sys_fit_coeffs:
                    sys_fit_coeffs[sys] = MapSet(
                        [sys_fit_coeffs[sys], params_map]
                    )
                else:
                    sys_fit_coeffs[sys] = params_map

        return sys_fit_coeffs

    def validate_params(self, params):
        pq = pint.quantity._Quantity
        param_types = [
            ('pipeline_config', str),
            ('discr_sys_sample_config', str),
            ('stop_after_stage', pq),
            ('poly_degree', pq),
            ('force_through_nominal', bool),
        ]
        if self.neutrinos:
            param_types.extend([
                ('nu_dom_eff', pq),
                ('nu_hole_ice', pq)
            ])
        if self.muons:
            param_types.extend([
                ('mu_dom_eff', pq),
                ('mu_hole_ice', pq)
            ])
        if not isinstance(params['smoothing'].value, str) \
           and params['smoothing'].value is not None:
            raise TypeError(
                'Param "smoothing" must be type str or NoneType but is '
                '{0} instead'.format(type(params['smoothing'].value))
            )
        for p, t in param_types:
            val = params[p].value
            if not isinstance(val, t):
                raise TypeError(
                    'Param "%s" must be type %s but is %s instead'
                    % (p, type(t), type(val))
                )
