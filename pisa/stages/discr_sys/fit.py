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
import pint

from pisa import FTYPE
from pisa import ureg
from pisa.core.events import Data
from pisa.core.map import Map, MapSet
from pisa.core.param import ParamSet
from pisa.core.stage import Stage
from pisa.core.pipeline import Pipeline
from pisa.utils.comparisons import normQuant
from pisa.utils.flavInt import ALL_NUFLAVINTS
from pisa.utils.flavInt import NuFlavInt, NuFlavIntGroup
from pisa.utils.fileio import from_file
from pisa.utils.format import text2tex
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.resources import open_resource


__all__ = ['fit']


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
            * discr_sys_sample_config : filepath
                Filepath to event sample configuration

            * stop_after_stage : int
                Extract templates up to this stage for the fitting

            * output_events_discr_sys : bool
                Flag to specify whether the service output returns a
                MapSet or the Data

            * nu_dom_eff

            * nu_hole_ice

    input_names : string
        Specifies the string representation of the NuFlavIntGroup(s) that
        belong in the Data object passed to this service.

    output_binning : MultiDimBinning or convertible thereto
        The binning desired for the output maps.
        NOTE: this is binning with which the curves will be fitted at.

    output_names : string
        Specifies the string representation of the NuFlavIntGroup(s) which will
        be produced as an output.

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
                 error_method=None, debug_mode=None, disk_cache=None,
                 memcache_deepcopy=True, outputs_cache_depth=20):
        self.fit_params = (
            'pipeline_config', 'discr_sys_sample_config', 'stop_after_stage',
            'output_events_discr_sys'
        )

        self.nu_params = (
            'nu_dom_eff', 'nu_hole_ice'
        )

        self.atm_muon_params = (
            'mu_dom_eff', 'mu_hole_ice'
        )

        expected_params = self.fit_params
        if ('all_nu' in input_names) or ('neutrinos' in input_names):
            expected_params += self.nu_params
        if 'muons' in input_names:
            expected_params += self.atm_muon_params

        self.neutrinos = False
        self.muons = False
        self.noise = False

        input_names = input_names.replace(' ', '').split(',')
        clean_innames = []
        for name in input_names:
            if 'muons' in name:
                clean_innames.append(name)
            elif 'noise' in name:
                clean_innames.append(name)
            elif 'all_nu' in name:
                clean_innames = [str(NuFlavIntGroup(f))
                                 for f in ALL_NUFLAVINTS]
            else:
                clean_innames.append(str(NuFlavIntGroup(name)))

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

        super(self.__class__, self).__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            input_names=clean_innames,
            output_names=clean_outnames,
            error_method=error_method,
            debug_mode=debug_mode,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning
        )

    @profile
    def _compute_outputs(self, inputs=None):
        """Compute histograms for output channels."""
        if not isinstance(inputs, Data):
            raise AssertionError('inputs is not a Data object, instead is '
                                 'type {0}'.format(type(inputs)))
        self._data = deepcopy(inputs)
        config = from_file(self.params['discr_sys_sample_config'].value)

        degree = config.getint('general', 'poly_degree')
        smoothing = config.get('general', 'smoothing')
        force_through_nominal = config.getboolean(
            'general', 'force_through_nominal'
        )

        if force_through_nominal:
            def fit_func(vals, poly_coeffs):
                return np.polynomial.polynomial.polyval(
                    vals, [1.] + list(poly_coeffs)
                )
        else:
            def fit_func(vals, poly_coeffs):
                return np.polynomial.polynomial.polyval(
                    vals, list(poly_coeffs)
                )
            # add free param for constant term
            degree += 1

        template_maker = Pipeline(self.params['pipeline_config'].value)
        dataset_param = template_maker.params['dataset']

        def parse(string):
            return string.replace(' ', '').split(',')

        if self.neutrinos:
            sys_list = parse(config.get('neutrinos', 'sys_list'))
            nu_params = map(lambda x: x[3:], self.nu_params)

            if set(nu_params) != set(sys_list):
                raise AssertionError(
                    'Systematics list listed in the sample config file does '
                    'not match the params in the pipeline config file\n {0} '
                    '!= {1}'.format(set(nu_params), set(sys_list))
                )

            for sys in sys_list:
                ev_sys = 'neutrinos:' + sys
                runs = parse(config.get(ev_sys, 'runs')[1: -1])
                nominal = config.get(ev_sys, 'nominal')

                mapset_dict = OrderedDict()
                for run in runs:
                    logging.info('Loading run {0} of systematic '
                                 '{1}'.format(run, sys))
                    dataset_param.value = ev_sys + ':' + run
                    template_maker.update_params(dataset_param)
                    template = template_maker.get_outputs(
                        idx=int(self.params['stop_after_stage'].m)
                    )
                    if not isinstance(template, Data):
                        raise AssertionError(
                            'template output is not a Data object, instead is '
                            'type {0}'.format(type(inputs))
                        )

                    outputs = []
                    for fig in template.iterkeys():
                        outputs.append(template.histogram(
                            kinds       = fig,
                            binning     = self.output_binning,
                            weights_col = 'pisa_weight',
                            errors      = True,
                            name        = str(NuFlavIntGroup(fig)),
                        ))
                    mapset_dict[run] = MapSet(outputs, name=run)

                nominal_mapset = mapset_dict[nominal]
                delta_mapset_dict = OrderedDict()
                for run in mapset_dict.iterkeys():
                    # TODO(shivesh): 0's?
                    mask = (nominal_mapset.hist == 0.)
                    div = unp.uarray(np.zeros(nominal_mapset.hist.shape),
                                     np.zeros(nominal_mapset.hist.shape))
                    div[~mask] = mapset_dict[run][~mask] / nominal_mapset[~mask]
                    delta_mapset_dict[run] = div

                delta_runs = np.array([float(x) for x in runs]) - float(nominal)
                print delta_runs

    def validate_params(self, params):
        pq = pint.quantity._Quantity
        param_types = [
            ('pipeline_config', basestring),
            ('discr_sys_sample_config', basestring),
            ('stop_after_stage', pq),
            ('output_events_discr_sys', bool),
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
        for p, t in param_types:
            val = params[p].value
            if not isinstance(val, t):
                raise TypeError(
                    'Param "%s" must be type %s but is %s instead'
                    % (p, type(t), type(val))
                )

