"""
The purpose of this stage is to reweight an event sample to include effects of
oscillation and various systematics.

This service in particular is intended to follow a `data` service which takes
advantage of the Data object being passed as a sideband in the Stage.

"""


from __future__ import absolute_import, division

from collections import OrderedDict
from copy import deepcopy

from scipy.interpolate import interp1d

import numpy as np

from pisa import FTYPE, ureg
from pisa.core.events import Data
from pisa.core.map import Map, MapSet
from pisa.core.param import ParamSet
from pisa.core.stage import Stage
from pisa.utils.flavInt import ALL_NUFLAVINTS
from pisa.utils.flavInt import NuFlavInt, NuFlavIntGroup
from pisa.utils.flux_weights import load_2D_table, calculate_flux_weights
from pisa.utils.format import text2tex
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.resources import open_resource


__all__ = ['weight']


class weight(Stage):
    """mc service to reweight an event sample taking into account atmospheric
    fluxes, neutrino oscillations and various other systematics.

    Parameters
    ----------
    params: ParamSet of sequence with which to instantiate a ParamSet
        Parameters which set everything besides the binning

        Parameters required by this service are
            * livetime : ureg.Quantity
                Desired lifetime.

            * Cross-section related parameters:
                - nu_diff_DIS
                - nu_diff_norm
                - nubar_diff_DIS
                - nubar_diff_norm
                - hadron_DIS

            * Flux related parameters:
                For more information see `$PISA/pisa/stages/flux/honda.py`
                - flux_reweight : bool
                    Flag to specifiy whether to reweight the flux.
                - flux_file
                - atm_delta_index
                - nue_numu_ratio
                - nu_nubar_ratio
                - norm_numu
                - norm_nutau
                - norm_nc
                - cache_flux : bool
                    Flag to specifiy whether to cache the flux values if
                    calculated inside this service to a file specified
                    by `disk_cache`.

            * Oscillation related parameters:
                For more information see `$PISA/pisa/stage/osc/prob3gpu.py`
                - oscillate : bool
                    Flag to specifiy whether to include the effects of neutrino
                    oscillation. `flux_reweight` option must be set to "True"
                    for oscillations reweighting.
                - earth_model
                - YeI
                - YeM
                - YeO
                - detector_depth
                - prop_height
                - deltacp
                - deltam21
                - deltam31
                - theta12
                - theta13
                - theta23
                - no_nc_osc : bool
                    Flag to turn off oscillations for the neutral current
                    interactions.
                - true_e_scale

    input_names : string
        Specifies the string representation of the NuFlavIntGroup(s) that
        belong in the Data object passed to this service.

    output_binning : MultiDimBinning or convertible thereto
        The binning desired for the output maps.

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
        """Hash of reweighted event sample."""
        self.xsec_hash = None
        """Hash of reweighted xsec values."""
        self.flux_cache_hash = None
        """Hash of cached flux values."""
        self.flux_hash = None
        """Hash of reweighted flux values."""
        self.osc_hash = None
        """Hash of reweighted osc flux values."""

        self.weight_params = (
            'kde_hist',
            'livetime',
        )

        self.nu_params = (
            'oscillate',
            'flux_reweight',
            'cache_flux'
        )

        self.xsec_params = (
            'nu_diff_DIS',
            'nu_diff_norm',
            'nubar_diff_DIS',
            'nubar_diff_norm',
            'hadron_DIS'
        )

        self.flux_params = (
            'flux_file',
            'atm_delta_index',
            'nu_nubar_ratio',
            'nue_numu_ratio',
            'norm_numu',
            'norm_nutau',
            'norm_nc'
        )

        self.osc_params = (
            'earth_model',
            'YeI',
            'YeO',
            'YeM',
            'detector_depth',
            'prop_height',
            'theta12',
            'theta13',
            'theta23',
            'deltam21',
            'deltam31',
            'deltacp',
            'no_nc_osc',
            'true_e_scale'
        )

        self.atm_muon_params = (
            'atm_muon_scale',
            'delta_gamma_mu_file',
            'delta_gamma_mu_spline_kind',
            'delta_gamma_mu_variable',
            'delta_gamma_mu'
        )

        self.noise_params = (
            'norm_noise',
        )

        expected_params = self.weight_params
        if ('all_nu' in input_names) or ('neutrinos' in input_names):
            # Allows muons to be passed through this stage on a CPU machine
            expected_params += self.nu_params
            expected_params += self.xsec_params
            expected_params += self.flux_params
            expected_params += self.osc_params
        if 'muons' in input_names:
            expected_params += self.atm_muon_params
        if 'noise' in input_names:
            expected_params += self.noise_params

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

        if not isinstance(output_events, bool):
            raise AssertionError(
                'output_events must be of type bool, instead it is supplied '
                'with type {0}'.format(type(output_events))
            )
        if output_events:
            output_binning = None
        self.output_events = output_events

        super(weight, self).__init__(
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

        if self.params['kde_hist'].value:
            raise ValueError(
                'The KDE option is currently not working properly. Please '
                'disable this in your configuration file by setting kde_hist '
                'to False.'
            )
            if self.output_events:
                logging.warn(
                    'Warning - You have selected to apply KDE smoothing to '
                    'the output histograms but have also selected that the '
                    'output is an Events object rather than a MapSet (where '
                    'the histograms would live.'
                )
            else:
                from pisa.utils.kde_hist import kde_histogramdd
                self.kde_histogramdd = kde_histogramdd

        if self.muons:
            self.prim_unc_spline = self.make_prim_unc_spline()

        self.include_attrs_for_hashes('sample_hash')

    @profile
    def _compute_outputs(self, inputs=None):
        """Compute histograms for output channels."""
        logging.debug('Entering weight._compute_outputs')
        if not isinstance(inputs, Data):
            raise AssertionError('inputs is not a Data object, instead is '
                                 'type {0}'.format(type(inputs)))
        self.sample_hash = deepcopy(inputs.metadata['sample_hash'])
        logging.trace('{0} weight sample_hash = '
                      '{1}'.format(inputs.metadata['name'], self.sample_hash))
        logging.trace('{0} weight weight_hash = '
                      '{1}'.format(inputs.metadata['name'], self.weight_hash))
        self._data = inputs
        self.reweight()

        if self.output_events:
            return self._data

        outputs = []
        if self.neutrinos:
            trans_nu_data = self._data.transform_groups(
                self._output_nu_groups
            )
            for fig in trans_nu_data.iterkeys():
                if self.params['kde_hist'].value:
                    coszen_name = None
                    for bin_name in self.output_binning.names:
                        if 'coszen' in bin_name:
                            coszen_name = bin_name
                    if coszen_name is None:
                        raise ValueError("Did not find coszen in binning. KDE "
                                         "will not work correctly.")
                    kde_hist = self.kde_histogramdd(
                        sample=np.array([
                            trans_nu_data[bin_name] for bin_name in
                            self.output_binning.names]).T,
                        binning=self.output_binning,
                        weights=trans_nu_data['pisa_weight'],
                        coszen_name=coszen_name,
                        use_cuda=False,
                        bw_method='silverman',
                        alpha=0.3,
                        oversample=10,
                        coszen_reflection=0.5,
                        adaptive=True
                    )
                    outputs.append(
                        Map(
                            name=fig,
                            hist=kde_hist,
                            error_hist=np.sqrt(kde_hist),
                            binning=self.output_binning,
                            tex=text2tex(fig)
                        )
                    )
                else:
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
            if self.params['kde_hist'].value:
                for bin_name in self.output_binning.names:
                    if 'coszen' in bin_name:
                        coszen_name = bin_name
                kde_hist = self.kde_histogramdd(
                    sample=np.array([
                        self._data['muons'][bin_name] for bin_name in \
                        self.output_binning.names]).T,
                    binning=self.output_binning,
                    weights=self._data['muons']['pisa_weight'],
                    coszen_name=coszen_name,
                    use_cuda=False,
                    bw_method='silverman',
                    alpha=0.3,
                    oversample=10,
                    coszen_reflection=0.5,
                    adaptive=True
                )
                outputs.append(
                    Map(
                        name='muons',
                        hist=kde_hist,
                        error_hist=np.sqrt(kde_hist),
                        binning=self.output_binning,
                        tex=text2tex('muons')
                    )
                )
            else:
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
            if self.params['kde_hist'].value:
                for bin_name in self.output_binning.names:
                    if 'coszen' in bin_name:
                        coszen_name = bin_name
                kde_hist = self.kde_histogramdd(
                    sample=np.array([
                        self._data['noise'][bin_name] for bin_name in \
                        self.output_binning.names]).T,
                    binning=self.output_binning,
                    weights=self._data['noise']['pisa_weight'],
                    coszen_name=coszen_name,
                    use_cuda=False,
                    bw_method='silverman',
                    alpha=0.3,
                    oversample=10,
                    coszen_reflection=0.5,
                    adaptive=True
                )
                outputs.append(
                    Map(
                        name='noise',
                        hist=kde_hist,
                        error_hist=np.sqrt(kde_hist),
                        binning=self.output_binning,
                        tex=text2tex('noise')
                    )
                )
            else:
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
            [self.sample_hash, self.params.values_hash],
            full_hash = self.full_hash
        )
        if this_hash == self.weight_hash:
            return

        if self.neutrinos:
            for fig in self._data.iterkeys():
                self._data[fig]['weight_weight'] = \
                    deepcopy(self._data[fig]['sample_weight'])

            # XSec reweighting
            xsec_weights = self.compute_xsec_weights()
            for fig in self._data.iterkeys():
                self._data[fig]['weight_weight'] *= xsec_weights[fig]

            # Flux reweighting
            if not self.params['flux_reweight'].value and \
               self.params['oscillate'].value:
                raise AssertionError(
                    '`oscillate` flag is set to "True" when `flux_reweight` '
                    'flag is set to "False". Oscillations reweighting requires '
                    'the `flux_reweight` flag to be set to "True".'
                )
            if self.params['flux_reweight'].value:
                flux_weights = self.compute_flux_weights(attach_units=True)
                if not self.params['oscillate'].value:
                    # No oscillations
                    for fig in self._data.iterkeys():
                        flav_pdg = NuFlavInt(fig).flavCode()
                        p_reweight = self._data[fig]['weight_weight']
                        if flav_pdg == 12:
                            p_reweight *= flux_weights[fig]['nue_flux']
                        elif flav_pdg == 14:
                            p_reweight *= flux_weights[fig]['numu_flux']
                        elif flav_pdg == -12:
                            p_reweight *= flux_weights[fig]['nuebar_flux']
                        elif flav_pdg == -14:
                            p_reweight *= flux_weights[fig]['numubar_flux']
                        elif abs(flav_pdg) == 16:
                            # attach units of flux from nue
                            p_reweight *= 0. * flux_weights[fig]['nue_flux'].u
                else:
                    # Oscillations
                    osc_weights = self.compute_osc_weights(flux_weights)
                    for fig in self._data.iterkeys():
                        self._data[fig]['weight_weight'] *= osc_weights[fig]

            # Livetime reweighting
            livetime = self.params['livetime'].value
            for fig in self._data.iterkeys():
                logging.debug(
                    'Rate for {0} = '.format(fig).ljust(25) +
                    r'{0:.3f}{1:~}'.format(
                        np.sum(self._data[fig]['weight_weight'].m_as('mHz')),
                        ureg('mHz')
                    ).rjust(6)
                )
                self._data[fig]['weight_weight'] *= livetime
                self._data[fig]['weight_weight'].ito('dimensionless')

            for fig in self._data.iterkeys():
                self._data[fig]['pisa_weight'] = \
                    deepcopy(self._data[fig]['weight_weight'])

        if self.muons:
            self._data.muons['weight_weight'] = \
                deepcopy(self._data.muons['sample_weight'])

            # Livetime reweighting
            livetime = self.params['livetime'].value
            self._data.muons['weight_weight'] *= livetime
            self._data.muons['weight_weight'].ito('dimensionless')

            # Scaling
            atm_muon_scale = self.params['atm_muon_scale'].value
            self._data.muons['weight_weight'] *= atm_muon_scale

            # Primary CR systematic
            cr_rw_scale = self.params['delta_gamma_mu'].value
            rw_variable = self.params['delta_gamma_mu_variable'].value
            rw_array = self.prim_unc_spline(self._data.muons[rw_variable])

            # Reweighting term is positive-only by construction, so normalise
            # it by shifting the whole array down by a normalisation factor
            norm = sum(rw_array)/len(rw_array)
            cr_rw_array = rw_array-norm
            self._data.muons['weight_weight'] *= (1+cr_rw_scale*cr_rw_array)

            self._data.muons['pisa_weight'] = \
                deepcopy(self._data.muons['weight_weight'])

        if self.noise:
            # TODO(shivesh): not working properly
            self._data.noise['weight_weight'] = \
                deepcopy(self._data.noise['sample_weight'])

            # Livetime reweighting
            livetime = self.params['livetime'].value
            self._data.noise['weight_weight'] *= livetime
            self._data.noise['weight_weight'].ito('dimensionless')

            # Scaling
            norm_noise = self.params['norm_noise'].value
            self._data.noise['weight_weight'] *= norm_noise

            self._data.noise['pisa_weight'] = \
                deepcopy(self._data.noise['weight_weight'])

        self.weight_hash = this_hash
        self._data.metadata['weight_hash'] = self.weight_hash
        self._data.update_hash()

    def compute_xsec_weights(self):
        """Reweight to take into account xsec systematics."""
        this_hash = hash_obj(
            [self.params[name].value for name in self.xsec_params] +
            [self.sample_hash], full_hash=self.full_hash
        )
        if self.xsec_hash == this_hash:
            return self._xsec_weights

        xsec_weights = self._compute_xsec_weights(
            self._data, ParamSet(p for p in self.params
                                 if p.name in self.xsec_params)
        )

        self.xsec_hash = this_hash
        self._xsec_weights = xsec_weights
        return xsec_weights

    def compute_flux_weights(self, attach_units=False):
        """Neutrino fluxes via `honda` service."""
        this_hash = hash_obj(
            [self.params[name].value for name in self.flux_params] +
            [self.sample_hash], full_hash=self.full_hash
        )
        out_units = ureg('1 / (GeV s m**2 sr)')
        if self.flux_hash == this_hash:
            if attach_units:
                flux_weights = OrderedDict()
                for fig in self._flux_weights.iterkeys():
                    flux_weights[fig] = OrderedDict()
                    for flav in self._flux_weights[fig].iterkeys():
                        flux_weights[fig][flav] = \
                                self._flux_weights[fig][flav]*out_units
                return flux_weights
            return self._flux_weights

        data_contains_flux = all(
            ['nue_flux' in fig and 'numu_flux' in fig and 'nuebar_flux' in fig
             and 'numubar_flux' in fig for fig in self._data.itervalues()]
        )
        if data_contains_flux:
            logging.info('Loading flux values from data.')
            flux_weights = OrderedDict()
            for fig in self._data.iterkeys():
                d = OrderedDict()
                d['nue_flux'] = self._data[fig]['nue_flux']
                d['numu_flux'] = self._data[fig]['numu_flux']
                d['nuebar_flux'] = self._data[fig]['nuebar_flux']
                d['numubar_flux'] = self._data[fig]['numubar_flux']
                flux_weights[fig] = d
        elif self.params['cache_flux'].value:
            this_cache_hash = hash_obj(
                [self._data.metadata['name'], self._data.metadata['sample'],
                 self._data.metadata['cuts'], self.params['flux_file'].value],
                full_hash=self.full_hash
            )

            if self.flux_cache_hash == this_cache_hash:
                flux_weights = deepcopy(self._cached_fw)
            elif this_cache_hash in self.disk_cache:
                logging.info('Loading flux values from cache.')
                self._cached_fw = self.disk_cache[this_cache_hash]
                flux_weights = deepcopy(self._cached_fw)
                self.flux_cache_hash = this_cache_hash
            else:
                flux_weights = self._compute_flux_weights(
                    self._data, ParamSet(p for p in self.params
                                         if p.name in self.flux_params)
                )
        else:
            flux_weights = self._compute_flux_weights(
                self._data, ParamSet(p for p in self.params
                                     if p.name in self.flux_params)
            )

        if self.params['cache_flux'].value:
            if this_cache_hash not in self.disk_cache:
                logging.info('Caching flux values to disk.')
                self.disk_cache[this_cache_hash] = flux_weights

        # TODO(shivesh): Barr flux systematics
        for fig in flux_weights:
            nue_flux = flux_weights[fig]['nue_flux']
            numu_flux = flux_weights[fig]['numu_flux']
            nuebar_flux = flux_weights[fig]['nuebar_flux']
            numubar_flux = flux_weights[fig]['numubar_flux']

            norm_nc = 1.0
            if 'nc' in fig:
                norm_nc = self.params['norm_nc'].m
            norm_numu = self.params['norm_numu'].m
            atm_index = np.power(
                self._data[fig]['energy'], self.params['atm_delta_index'].m
            )
            nue_flux *= atm_index * norm_nc
            numu_flux *= atm_index * norm_nc * norm_numu
            nuebar_flux *= atm_index * norm_nc
            numubar_flux *= atm_index * norm_nc * norm_numu

            nue_flux, nuebar_flux = self.apply_ratio_scale(
                nue_flux, nuebar_flux, self.params['nu_nubar_ratio'].m
            )
            numu_flux, numubar_flux = self.apply_ratio_scale(
                numu_flux, numubar_flux, self.params['nu_nubar_ratio'].m
            )
            nue_flux, numu_flux = self.apply_ratio_scale(
                nue_flux, numu_flux, self.params['nue_numu_ratio'].m
            )
            nuebar_flux, numubar_flux = self.apply_ratio_scale(
                nuebar_flux, numubar_flux, self.params['nue_numu_ratio'].m
            )

            flux_weights[fig]['nue_flux'] = nue_flux
            flux_weights[fig]['numu_flux'] = numu_flux
            flux_weights[fig]['nuebar_flux'] = nuebar_flux
            flux_weights[fig]['numubar_flux'] = numubar_flux

        self.flux_hash = this_hash
        self._flux_weights = flux_weights
        if attach_units:
            fw_units = OrderedDict()
            for fig in flux_weights.iterkeys():
                fw_units[fig] = OrderedDict()
                for flav in flux_weights[fig].iterkeys():
                    fw_units[fig][flav] = flux_weights[fig][flav]*out_units
            return fw_units
        return flux_weights

    def compute_osc_weights(self, flux_weights):
        """Neutrino oscillations calculation via Prob3."""
        this_hash = hash_obj(
            [self.params[name].value for name in self.flux_params +
             self.osc_params] + [self.sample_hash], full_hash=self.full_hash
        )
        if self.osc_hash == this_hash:
            return self._osc_weights
        osc_weights = self._compute_osc_weights(
            self._data, ParamSet(p for p in self.params
                                 if p.name in self.osc_params), flux_weights
        )

        for fig in osc_weights:
            if 'tau' in fig:
                osc_weights[fig] *= self.params['norm_nutau'].m

        self.osc_hash = this_hash
        self._osc_weights = osc_weights
        return self._osc_weights

    @staticmethod
    def _compute_xsec_weights(nu_data, params):
        """Reweight to take into account xsec systematics."""
        logging.debug('Reweighting xsec systematics')

        xsec_weights = OrderedDict()
        for fig in nu_data.iterkeys():
            # Differential xsec systematic
            if 'bar' not in fig:
                nu_diff_DIS = params['nu_diff_DIS'].m
                nu_diff_norm = params['nu_diff_norm'].m
            else:
                nu_diff_DIS = params['nubar_diff_DIS'].m
                nu_diff_norm = params['nubar_diff_norm'].m

            with np.errstate(divide='ignore', invalid='ignore'):
                xsec_weights[fig] = (
                    (1 - nu_diff_norm * nu_diff_DIS) *
                    np.power(nu_data[fig]['GENIE_x'], -nu_diff_DIS)
                )
            xsec_weights[fig][~np.isfinite(xsec_weights[fig])] = 0.

            # High W hadronization systematic
            hadron_DIS = params['hadron_DIS'].m
            if hadron_DIS != 0.:
                xsec_weights[fig] *= (
                    1. / (1 + (2*hadron_DIS * np.exp(
                        -nu_data[fig]['GENIE_y'] / hadron_DIS
                    )))
                )
        return xsec_weights

    @staticmethod
    def _compute_flux_weights(nu_data, params):
        """Neutrino fluxes via integral preserving spline."""
        logging.debug('Computing flux values')
        spline_dict = load_2D_table(params['flux_file'].value)

        flux_weights = OrderedDict()
        for fig in nu_data.iterkeys():
            flux_weights[fig] = OrderedDict()
            logging.debug('Computing flux values for flavour {0}'.format(fig))
            flux_weights[fig]['nue_flux'] = calculate_flux_weights(
                nu_data[fig]['energy'], nu_data[fig]['coszen'],
                spline_dict['nue']
            )
            flux_weights[fig]['numu_flux'] = calculate_flux_weights(
                nu_data[fig]['energy'], nu_data[fig]['coszen'],
                spline_dict['numu']
            )
            flux_weights[fig]['nuebar_flux'] = calculate_flux_weights(
                nu_data[fig]['energy'], nu_data[fig]['coszen'],
                spline_dict['nuebar']
            )
            flux_weights[fig]['numubar_flux'] = calculate_flux_weights(
                nu_data[fig]['energy'], nu_data[fig]['coszen'],
                spline_dict['numubar']
            )

        return flux_weights

    @staticmethod
    def _compute_osc_weights(nu_data, params, flux_weights):
        """Neutrino oscillations calculation via Prob3."""
        # Import oscillations calculator only if needed
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pisa.stages.osc.prob3gpu import prob3gpu
        logging.debug('Computing oscillation weights')
        # Read parameters in, convert to the units used internally for
        # computation, and then strip the units off. Note that this also
        # enforces compatible units (but does not sanity-check the numbers).
        theta12 = params['theta12'].m_as('rad')
        theta13 = params['theta13'].m_as('rad')
        theta23 = params['theta23'].m_as('rad')
        deltam21 = params['deltam21'].m_as('eV**2')
        deltam31 = params['deltam31'].m_as('eV**2')
        deltacp = params['deltacp'].m_as('rad')
        true_e_scale = params['true_e_scale'].m_as('dimensionless')

        osc = prob3gpu(
            params=params,
            input_binning=None,
            output_binning=None,
            error_method=None,
            memcache_deepcopy=False,
            transforms_cache_depth=0,
            outputs_cache_depth=0
        )

        osc_data = OrderedDict()
        for fig in nu_data.iterkeys():
            if 'nc' in fig and params['no_nc_osc'].value:
                continue
            osc_data[fig] = OrderedDict()
            energy_array = nu_data[fig]['energy'].astype(FTYPE)
            coszen_array = nu_data[fig]['coszen'].astype(FTYPE)
            n_evts = np.uint32(len(energy_array))
            osc_data[fig]['n_evts'] = n_evts

            device = OrderedDict()
            device['true_energy'] = energy_array
            device['prob_e'] = np.zeros(n_evts, dtype=FTYPE)
            device['prob_mu'] = np.zeros(n_evts, dtype=FTYPE)
            out_layers_n = ('numLayers', 'densityInLayer', 'distanceInLayer')
            out_layers = osc.calc_layers(coszen_array)
            device.update(dict(zip(out_layers_n, out_layers)))

            osc_data[fig]['device'] = OrderedDict()
            for key in device.iterkeys():
                osc_data[fig]['device'][key] = (
                    cuda.mem_alloc(device[key].nbytes)
                )
                cuda.memcpy_htod(osc_data[fig]['device'][key], device[key])

        osc.update_MNS(theta12, theta13, theta23, deltam21, deltam31, deltacp)

        osc_weights = OrderedDict()
        for fig in nu_data.iterkeys():
            flavint = NuFlavInt(fig)
            pdg = abs(flavint.flav.code)
            kNuBar = 1 if flavint.particle else -1
            p = '' if flavint.particle else 'bar'
            if pdg == 12:
                kFlav = 0
            elif pdg == 14:
                kFlav = 1
            elif pdg == 16:
                kFlav = 2

            if 'nc' in fig and params['no_nc_osc'].value:
                if kFlav == 0:
                    osc_weights[fig] = flux_weights[fig]['nue'+p+'_flux']
                elif kFlav == 1:
                    osc_weights[fig] = flux_weights[fig]['numu'+p+'_flux']
                elif kFlav == 2:
                    osc_weights[fig] = 0.
                continue

            osc.calc_probs(
                kNuBar, kFlav, osc_data[fig]['n_evts'], true_e_scale,
                **osc_data[fig]['device']
            )

            prob_e = np.zeros(osc_data[fig]['n_evts'], dtype=FTYPE)
            prob_mu = np.zeros(osc_data[fig]['n_evts'], dtype=FTYPE)
            cuda.memcpy_dtoh(prob_e, osc_data[fig]['device']['prob_e'])
            cuda.memcpy_dtoh(prob_mu, osc_data[fig]['device']['prob_mu'])

            for key in osc_data[fig]['device']:
                osc_data[fig]['device'][key].free()

            osc_weights[fig] = (flux_weights[fig]['nue'+p+'_flux']*prob_e
                                + flux_weights[fig]['numu'+p+'_flux']*prob_mu)

        return osc_weights

    @staticmethod
    def apply_ratio_scale(flux_a, flux_b, ratio_scale):
        """Apply a ratio systematic to the flux weights."""
        orig_ratio = flux_a / flux_b
        orig_sum = flux_a + flux_b

        scaled_b = orig_sum / (1 + ratio_scale*orig_ratio)
        scaled_a = ratio_scale*orig_ratio * scaled_b
        return scaled_a, scaled_b

    def make_prim_unc_spline(self):
        """
        Create the spline which will be used to re-weight muons based on the
        uncertainties arising from cosmic rays.

        Notes
        -----

        Details on this work can be found here -

        https://wiki.icecube.wisc.edu/index.php/DeepCore_Muon_Background_Systematics

        This work was done for the GRECO sample but should be reasonably
        generic. It was found to pretty much be a negligible systemtic. Though
        you should check both if it seems reasonable and it is still negligible
        if you use it with a different event sample.
        """
        # TODO(shivesh): "energy"/"coszen" on its own is taken to be the truth
        # TODO(shivesh): what does "true" muon correspond to - the deposited muon?
        # if 'true' not in self.params['delta_gamma_mu_variable'].value:
        #     raise ValueError(
        #         'Variable to construct spline should be a truth variable. '
        #         'You have put %s in your configuration file.'
        #         % self.params['delta_gamma_mu_variable'].value
        #     )

        bare_variable = self.params['delta_gamma_mu_variable']\
                            .value.split('true_')[-1]
        if not bare_variable == 'coszen':
            raise ValueError(
                'Muon primary cosmic ray systematic is currently only '
                'implemented as a function of cos(zenith). %s was set in the '
                'configuration file.'
                % self.params['delta_gamma_mu_variable'].value
            )
        if bare_variable not in self.params['delta_gamma_mu_file'].value:
            raise ValueError(
                'Variable set in configuration file is %s but the file you '
                'have selected, %s, does not make reference to this in its '
                'name.' % (self.params['delta_gamma_mu_variable'].value,
                           self.params['delta_gamma_mu_file'].value)
            )

        unc_data = np.genfromtxt(
            open_resource(self.params['delta_gamma_mu_file'].value)
        ).T

        # Need to deal with zeroes that arise due to a lack of MC. For example,
        # in the case of the splines as a function of cosZenith, there are no
        # hoirzontal muons. Current solution is just to replace them with their
        # nearest non-zero values.
        while 0.0 in unc_data[1]:
            zero_indices = np.where(unc_data[1] == 0)[0]
            for zero_index in zero_indices:
                unc_data[1][zero_index] = unc_data[1][zero_index+1]

        # Add dummpy points for the edge of the zenith range
        xvals = np.insert(unc_data[0], 0, 0.0)
        xvals = np.append(xvals, 1.0)
        yvals = np.insert(unc_data[1], 0, unc_data[1][0])
        yvals = np.append(yvals, unc_data[1][-1])

        muon_uncf = interp1d(
            xvals,
            yvals,
            kind=self.params['delta_gamma_mu_spline_kind'].value
        )

        return muon_uncf

    def validate_params(self, params):
        pq = ureg.Quantity
        param_types = [
            ('kde_hist', bool),
            ('livetime', pq)
        ]
        if self.neutrinos:
            param_types.extend([
                ('flux_reweight', bool),
                ('oscillate', bool),
                ('cache_flux', bool),
                ('nu_diff_DIS', pq),
                ('nu_diff_norm', pq),
                ('nubar_diff_DIS', pq),
                ('nubar_diff_norm', pq),
                ('hadron_DIS', pq),
                ('flux_file', basestring),
                ('atm_delta_index', pq),
                ('nu_nubar_ratio', pq),
                ('nue_numu_ratio', pq),
                ('norm_numu', pq),
                ('norm_nutau', pq),
                ('norm_nc', pq),
                ('earth_model', basestring),
                ('YeI', pq),
                ('YeO', pq),
                ('YeM', pq),
                ('detector_depth', pq),
                ('prop_height', pq),
                ('theta12', pq),
                ('theta13', pq),
                ('theta23', pq),
                ('deltam21', pq),
                ('deltam31', pq),
                ('deltacp', pq),
                ('no_nc_osc', bool),
                ('true_e_scale', pq)
            ])
        if self.muons:
            param_types.extend([
                ('atm_muon_scale', pq),
                ('delta_gamma_mu_file', basestring),
                ('delta_gamma_mu_spline_kind', basestring),
                ('delta_gamma_mu_variable', basestring),
                ('delta_gamma_mu', pq)
            ])
        if self.noise:
            param_types.extend([
                ('norm_noise', pq)
            ])

        for p, t in param_types:
            val = params[p].value
            if not isinstance(val, t):
                raise TypeError(
                    'Param "%s" must be type %s but is %s instead'
                    % (p, type(t), type(val))
                )
