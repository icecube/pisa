"""
The purpose of this stage is to reweight an event sample to include effects of
oscillation and various systematics.

This service in particular is intended to follow a `data` service which takes
advantage of the Data object being passed as a sideband in the Stage.
"""
from copy import deepcopy

import numpy as np
import pint

import pycuda.driver as cuda
import pycuda.autoinit

from pisa import FTYPE
from pisa import ureg, Q_
from pisa.core.stage import Stage
from pisa.core.events import Data
from pisa.core.map import MapSet
from pisa.core.param import ParamSet
from pisa.utils.flavInt import ALL_NUFLAVINTS
from pisa.utils.flavInt import NuFlavInt, NuFlavIntGroup
from pisa.utils.comparisons import normQuant
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile

from pisa.utils.flux_weights import load_2D_table, calculate_flux_weights
from pisa.stages.osc.prob3gpu import prob3gpu


class weight(Stage):
    """mc service to reweight an event sample taking into account atmospheric
    fluxes, neutrino oscillations and various other systematics.

    Parameters
    ----------
    params: ParamSet of sequence with which to instantiate a ParamSet
        Parameters which set everything besides the binning

        Parameters required by this service are
            * output_events_mc : bool
                Flag to specify whether the service output returns a
                MapSet or the Data

            * livetime : ureg.Quantity
                Desired lifetime.

            * Cross-section related parameters:
                - nu_dis_a
                - nu_dis_b
                - nubar_dis_a
                - nubar_dis_b

            * Flux related parameters:
                For more information see `$PISA/pisa/stages/flux/honda.py`
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
                    oscillation.
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

    input_names : string
        Specifies the string representation of the NuFlavIntGroup(s) that
        belong in the Data object passed to this service.

    output_binning : MultiDimBinning or convertible thereto
        The binning desired for the output maps.

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
        self.sample_hash = None
        """Hash of event sample"""

        self.weight_params = (
            'output_events_mc',
            'livetime',
            'oscillate',
            'cache_flux'
        )

        self.xsec_params = (
            'nu_dis_a',
            'nu_dis_b',
            'nubar_dis_a',
            'nubar_dis_b'
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
            'no_nc_osc'
        )

        self.atmmu_params = (
            'norm_atmmu',
        )

        self.noise_params = (
            'norm_noise',
        )

        expected_params = self.weight_params + self.xsec_params + \
                self.flux_params + self.osc_params + self.atmmu_params + \
                self.noise_params

        self.neutrino = False
        self.muons = False
        self.noise = False

        self.xsec_hash = None
        self.flux_hash = None
        self.osc_hash = None

        input_names = input_names.replace(' ', '').split(',')
        clean_innames = []
        for name in input_names:
            if 'muons' in name:
                clean_innames.append(name)
            elif 'noise' in name:
                clean_innames.append(name)
            elif 'all_nu' in name:
                clean_innames = [str(NuFlavIntGroup(f)) for f in ALL_NUFLAVINTS]
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
                self.neutrino = True
                self._output_nu_groups = \
                    [NuFlavIntGroup(f) for f in ALL_NUFLAVINTS]
            else:
                self.neutrino = True
                self._output_nu_groups.append(NuFlavIntGroup(name))

        if self.neutrino:
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

        if disk_cache is not None and self.params['cache_flux'].value:
            self.instantiate_disk_cache()

        self.include_attrs_for_hashes('sample_hash')

    @profile
    def _compute_outputs(self, inputs=None):
        """Compute histograms for output channels."""
        if not isinstance(inputs, Data):
            raise AssertionError('inputs is not a Data object, instead is '
                                 'type {0}'.format(type(inputs)))
        self._data = deepcopy(inputs)

        # TODO(shivesh): muons + noise reweighting
        if self.neutrino:
            # XSec reweighting
            xsec_weights = self.compute_xsec_weights()
            for fig in self._data.iterkeys():
                self._data[fig]['pisa_weight'] *= xsec_weights[fig]

            # Flux reweighting
            flux_weights = self.compute_flux_weights(attach_units=True)
            if not self.params['oscillate'].value:
                # no oscillations
                for fig in self._data.iterkeys():
                    flav_pdg = NuFlavInt(fig).flavCode()
                    weight = self._data[fig]['pisa_weight']
                    if flav_pdg == 12:
                        weight *= flux_weights[fig]['nue_flux']
                    elif flav_pdg == 14:
                        weight *= flux_weights[fig]['numu_flux']
                    elif flav_pdg == -12:
                        weight *= flux_weights[fig]['nuebar_flux']
                    elif flav_pdg == -14:
                        weight *= flux_weights[fig]['numubar_flux']
                    elif abs(flav_pdg) == 16:
                        # attach units of flux from nue
                        weight *= 0. * flux_weights[fig]['nue_flux'].u
            else:
                # oscillations
                osc_weights = self.compute_osc_weights(flux_weights)
                for fig in self._data.iterkeys():
                    self._data[fig]['pisa_weight'] *= osc_weights[fig]

            # Livetime reweighting
            livetime = self.params['livetime'].value
            for fig in self._data.iterkeys():
                self._data[fig]['pisa_weight'] *= livetime
                self._data[fig]['pisa_weight'].ito('dimensionless')

        self._data.metadata['params_hash'] = self.params.values_hash
        self._data.update_hash()
        self.sample_hash = self._data.hash

        if self.params['output_events_mc'].value:
            return self._data

        outputs = []
        if self.neutrino:
            trans_nu_data = self._data.transform_groups(
                self._output_nu_groups
            )
            for fig in trans_nu_data.iterkeys():
                outputs.append(trans_nu_data.histogram(
                    kinds       = fig,
                    binning     = self.output_binning,
                    weights_col = 'pisa_weight',
                    errors      = True,
                    name        = str(NuFlavIntGroup(fig)),
                ))

        if self.muons:
            outputs.append(self._data.histogram(
                kinds       = 'muons',
                binning     = self.output_binning,
                weights_col = 'pisa_weight',
                errors      = True,
                name        = 'muons',
                tex         = r'\rm{muons}'
            ))

        return MapSet(maps=outputs, name=self._data.metadata['name'])

    def compute_xsec_weights(self):
        """Reweight to take into account xsec systematics."""
        this_hash = normQuant([self.params[name].value
                               for name in self.xsec_params])
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
        this_hash = normQuant([self.params[name].value
                               for name in self.flux_params])
        out_units = ureg('1 / (GeV s m**2 sr)')
        if self.flux_hash == this_hash:
            if attach_units:
                flux_weights = {}
                for fig in self._flux_weights.iterkeys():
                    flux_weights[fig] = {}
                    for flav in self._flux_weights[fig].iterkeys():
                        flux_weights[fig][flav] = \
                                self._flux_weights[fig][flav]*out_units
                return flux_weights
            else:
                return self._flux_weights

        data_contains_flux = all(
            ['nue_flux' in fig and 'numu_flux' in fig and 'nuebar_flux' in fig
             and 'numubar_flux' in fig for fig in self._data.itervalues()]
        )
        if data_contains_flux:
            logging.info('Loading flux values from data.')
            flux_weights = {}
            for fig in self._data.iterkeys():
                flux_weights[fig] = {}
                flux_weights[fig]['nue_flux'] = self._data[fig]['nue_flux']
                flux_weights[fig]['numu_flux'] = self._data[fig]['numu_flux']
                flux_weights[fig]['nuebar_flux'] = self._data[fig]['nuebar_flux']
                flux_weights[fig]['numubar_flux'] = self._data[fig]['numubar_flux']
        elif self.params['cache_flux'].value:
            this_cache_hash = normQuant([self._data.metadata['name'],
                                         self._data.metadata['sample'],
                                         self._data.metadata['cuts'],
                                         self.params['flux_file'].value])
            this_cache_hash = hash_obj(this_cache_hash)

            if this_cache_hash in self.disk_cache:
                logging.info('Loading flux values from cache.')
                flux_weights = self.disk_cache[this_cache_hash]
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
            if not this_cache_hash in self.disk_cache:
                logging.info('Caching flux values to disk.')
                self.disk_cache[this_cache_hash] = flux_weights

        # TODO(shivesh): Barr flux systematics
        for fig in flux_weights:
            nue_flux     = flux_weights[fig]['nue_flux']
            numu_flux    = flux_weights[fig]['numu_flux']
            nuebar_flux  = flux_weights[fig]['nuebar_flux']
            numubar_flux = flux_weights[fig]['numubar_flux']

            norm_nc = 1.0
            if 'nc' in fig:
                norm_nc = self.params['norm_nc'].m
            norm_numu = self.params['norm_numu'].m
            atm_index = np.power(
                self._data[fig]['energy'], self.params['atm_delta_index'].m
            )
            nue_flux     *= atm_index * norm_nc
            numu_flux    *= atm_index * norm_nc * norm_numu
            nuebar_flux  *= atm_index * norm_nc
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

        self.flux_hash = this_hash
        self._flux_weights = flux_weights
        if attach_units:
            fw_units = {}
            for fig in flux_weights.iterkeys():
                fw_units[fig] = {}
                for flav in flux_weights[fig].iterkeys():
                    fw_units[fig][flav] = flux_weights[fig][flav]*out_units
            return fw_units
        else:
            return flux_weights

    def compute_osc_weights(self, flux_weights):
        """Neutrino oscillations calculation via Prob3."""
        this_hash = normQuant([self.params[name].value
                               for name in self.flux_params + self.osc_params])
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

        xsec_weights = {}
        for fig in nu_data.iterkeys():
            if 'bar' not in fig:
                dis_a = params['nu_dis_a'].m
                dis_b = params['nu_dis_b'].m
            else:
                dis_a = params['nubar_dis_a'].m
                dis_b = params['nubar_dis_b'].m
            xsec_weights[fig] = dis_b * np.power(nu_data[fig]['GENIE_x'], -dis_a)
        return xsec_weights

    @staticmethod
    def _compute_flux_weights(nu_data, params):
        """Neutrino fluxes via integral preserving spline."""
        logging.debug('Computing flux values')
        spline_dict = load_2D_table(params['flux_file'].value)

        flux_weights = {}
        for fig in nu_data.iterkeys():
            flux_weights[fig] = {}
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

        osc = prob3gpu(
            params = params,
            input_binning = None,
            output_binning = None,
            error_method = None,
            memcache_deepcopy = False,
            transforms_cache_depth = 0,
            outputs_cache_depth = 0
        )

        osc_data = {}
        for fig in nu_data.iterkeys():
            if 'nc' in fig and params['no_nc_osc'].value:
                continue
            osc_data[fig] = {}
            energy_array = nu_data[fig]['energy'].astype(FTYPE)
            coszen_array = nu_data[fig]['coszen'].astype(FTYPE)
            n_evts = np.uint32(len(energy_array))
            osc_data[fig]['n_evts'] = n_evts

            device = {}
            device['true_energy'] = energy_array
            device['prob_e'] = np.zeros(n_evts, dtype=FTYPE)
            device['prob_mu'] = np.zeros(n_evts, dtype=FTYPE)
            out_layers_n = ('numLayers', 'densityInLayer', 'distanceInLayer')
            out_layers = osc.calc_layers(coszen_array)
            device.update(dict(zip(out_layers_n, out_layers)))

            osc_data[fig]['device'] = {}
            for key in device.iterkeys():
                osc_data[fig]['device'][key] = cuda.mem_alloc(device[key].nbytes)
                cuda.memcpy_htod(osc_data[fig]['device'][key], device[key])

        osc.update_MNS(theta12, theta13, theta23, deltam21, deltam31, deltacp)

        osc_weights = {}
        for fig in nu_data.iterkeys():
            flavint = NuFlavInt(fig)
            pdg = abs(flavint.flavCode())
            kNuBar = 1 if flavint.isParticle() else -1
            p = '' if flavint.isParticle() else 'bar'
            if pdg == 12: kFlav = 0
            elif pdg == 14: kFlav = 1
            elif pdg == 16: kFlav = 2

            if 'nc' in fig and params['no_nc_osc'].value:
                if kFlav == 0:
                    osc_weights[fig] = flux_weights[fig]['nue'+p+'_flux']
                elif kFlav == 1:
                    osc_weights[fig] = flux_weights[fig]['numu'+p+'_flux']
                elif kFlav == 2: osc_weights[fig] = 0.
                continue

            osc.calc_probs(
                kNuBar, kFlav, osc_data[fig]['n_evts'], **osc_data[fig]['device']
            )

            prob_e = np.zeros(osc_data[fig]['n_evts'], dtype=FTYPE)
            prob_mu = np.zeros(osc_data[fig]['n_evts'], dtype=FTYPE)
            cuda.memcpy_dtoh(prob_e, osc_data[fig]['device']['prob_e'])
            cuda.memcpy_dtoh(prob_mu, osc_data[fig]['device']['prob_mu'])

            for key in osc_data[fig]['device']:
                osc_data[fig]['device'][key].free()

            osc_weights[fig] = flux_weights[fig]['nue'+p+'_flux']*prob_e + \
                    flux_weights[fig]['numu'+p+'_flux']*prob_mu

        return osc_weights

    @staticmethod
    def apply_ratio_scale(flux_a, flux_b, ratio_scale):
        """Apply a ratio systematic to the flux weights."""
        orig_ratio = flux_a / flux_b
        orig_sum = flux_a + flux_b

        scaled_a = orig_sum / (1 + ratio_scale*orig_ratio)
        scaled_b = ratio_scale*orig_ratio * scaled_a
        return scaled_a, scaled_b

    def validate_params(self, params):
        pq = pint.quantity._Quantity
        assert isinstance(params['output_events_mc'].value, bool)
        assert isinstance(params['livetime'].value, pq)
        assert isinstance(params['oscillate'].value, bool)
        assert isinstance(params['cache_flux'].value, bool)
        assert isinstance(params['nu_dis_a'].value, pq)
        assert isinstance(params['nu_dis_b'].value, pq)
        assert isinstance(params['nubar_dis_a'].value, pq)
        assert isinstance(params['nubar_dis_b'].value, pq)
        assert isinstance(params['flux_file'].value, basestring)
        assert isinstance(params['atm_delta_index'].value, pq)
        assert isinstance(params['nu_nubar_ratio'].value, pq)
        assert isinstance(params['nue_numu_ratio'].value, pq)
        assert isinstance(params['norm_numu'].value, pq)
        assert isinstance(params['norm_nutau'].value, pq)
        assert isinstance(params['norm_nc'].value, pq)
        assert isinstance(params['earth_model'].value, basestring)
        assert isinstance(params['YeI'].value, pq)
        assert isinstance(params['YeO'].value, pq)
        assert isinstance(params['YeM'].value, pq)
        assert isinstance(params['detector_depth'].value, pq)
        assert isinstance(params['prop_height'].value, pq)
        assert isinstance(params['theta12'].value, pq)
        assert isinstance(params['theta13'].value, pq)
        assert isinstance(params['theta23'].value, pq)
        assert isinstance(params['deltam21'].value, pq)
        assert isinstance(params['deltam31'].value, pq)
        assert isinstance(params['deltacp'].value, pq)
        assert isinstance(params['no_nc_osc'].value, bool)
        assert isinstance(params['norm_atmmu'].value, pq)
        assert isinstance(params['norm_noise'].value, pq)
