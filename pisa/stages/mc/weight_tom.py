"""
The purpose of this stage is to reweight an event sample to include effects of
oscillation and various systematics.

This service in particular is intended to follow a `data` service which takes
advantage of the Data object being passed as a sideband in the Stage.

"""


from __future__ import absolute_import, division

from collections import OrderedDict
from copy import deepcopy
import time

from scipy.interpolate import interp1d

import numpy as np

from pisa import FTYPE, ureg
from pisa.core.events import Data, Events
from pisa.core.map import Map, MapSet
from pisa.core.param import ParamSet
from pisa.core.stage import Stage
from pisa.utils.flavInt import ALL_NUFLAVINTS, NuFlavInt, NuFlavIntGroup
from pisa.utils.flux_weights import load_2D_table, calculate_flux_weights
from pisa.utils.format import text2tex
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.resources import open_resource
from pisa.utils.comparisons import normQuant
from pisa.scripts.make_events_file import CMSQ_TO_MSQ

__all__ = ['weight_tom']


class weight_tom(Stage):
    """mc service to reweight an event sample taking into account atmospheric
    fluxes, neutrino oscillations and various other systematics.

    Parameters
    ----------
    params: ParamSet of sequence with which to instantiate a ParamSet
        Parameters which set everything besides the binning

        #TODO Update the descriptions...

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
                - nutau_norm

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
                 outputs_cache_depth=20, use_gpu=False):

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
            'kde_hist', #TODO kwarg?
            'livetime',
            "aeff_scale",
            "hist_e_scale",
            'hist_pid_scale',
            "nutau_cc_norm",
            'reco_e_res_raw',
            'reco_e_scale_raw',
            'reco_cz_res_raw',
        )

        self.nu_params = (
            'oscillate',
            'flux_reweight',
            'cache_flux'
        )

        '''
        self.xsec_params = (
            'nu_diff_DIS',
            'nu_diff_norm',
            'nubar_diff_DIS',
            'nubar_diff_norm',
            'hadron_DIS'
        )
        '''
        self.xsec_params = (
            'Genie_Ma_QE',
            'Genie_Ma_RES',
        )

        self.flux_params = (
            'flux_file',
            'atm_delta_index',
            'nu_nubar_ratio',
            'nue_numu_ratio',
            'Barr_uphor_ratio',
            'Barr_nu_nubar_ratio',
#            'norm_numu',
#            'norm_nc'
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
            'true_e_scale',
            'nutau_norm', #TODO nutau_cc_norm somewhere???
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

        self.use_gpu = use_gpu

        #Get the names of all expected inputs
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

        super(weight_tom, self).__init__(
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

        if self.params.kde_hist.value:
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


    def _compute_nominal_outputs(self):

        #This function is called during initialisation, and is used to compute/init stuff (and hash it)
        #that is needed during compute_outputs but does not change at each new computation

        #
        # Hashing
        #

        #TODO This is from Philipp, merge properly with Shivesh's stuff

        # Store hashes for caching that is done inside the stage
        self.osc_hash = None #TODO???
        self.flux_hash = None #TODO???

        # Reset fixed errors
        self.fixed_error = None #TODO remove once shift to latest histogramming tools???


        #
        # Prepare oscillation calculation tools
        #

        if self.neutrinos:

            #Determine whether to use GPU or CPU for the oscillations calculations
            #User can specify whether to run the general re-weighting using GPU or CPU 
            #via the 'use_gpu' flag, so in general use this choice for the oscillations
            #A special case however is that if no Earth model is provided (e.g. 
            #oscillations are in vacuum), then the oscillations are performed using a 
            #CPU (everything else is still done using a GPU) #TODO why is this???
            self.use_gpu_for_osc = self.use_gpu and ( self.params.earth_model.value is not None )

            # Get param subset wanted for oscillations class
            osc_params_subset = []
            for param in self.params:
                if param.name in self.osc_params :
                    #There are a few params that probgpu wants that prob3cpu does not
                    if not self.use_gpu: 
                        if param.name == 'no_nc_osc' : continue
                        if param.name == 'true_e_scale' : continue
                    osc_params_subset.append(param)
            osc_params_subset = ParamSet(osc_params_subset)

            #Import prob3 implementation based on CPU vs GPU choice, then instantiate
            if self.use_gpu_for_osc: from pisa.stages.osc.prob3gpu import prob3gpu as prob3
            else : from pisa.stages.osc.prob3cpu import prob3cpu as prob3
            self.osc = prob3(
                params=osc_params_subset,
                input_binning=None,
                output_binning=None,
                error_method=None,
                memcache_deepcopy=False,
                transforms_cache_depth=0,
                outputs_cache_depth=0,
            )


        #
        # Prepare weight calculator
        #

        # Instantiate weight calculator (differs depending on whether using CPU or GPU code) #TODO Make into a common class
        if self.use_gpu: from pisa.stages.mc.GPUWeight import CPUWeight as WeightCalculator
        else : from pisa.stages.mc.CPUWeight import CPUWeight as WeightCalculator
        self.weight_calc = WeightCalculator()


        #
        # Prepare histogramming tools
        #

        # Binning
        self.bin_names = self.output_binning.names
        self.bin_edges = []

        for i,name in enumerate(self.bin_names):
            if 'energy' in name:
                bin_edges = self.output_binning[name].bin_edges.to('GeV').magnitude.astype(FTYPE)
                self.e_bin_number = i
            else:
                bin_edges = self.output_binning[name].bin_edges.magnitude.astype(FTYPE)
            if 'pid' in name:
                self.pid_bin_number = i
            self.bin_edges.append(bin_edges)

        #Grab histogramming tools and check user inputs
        if self.params.kde_hist.value:
            assert self.error_method == None
        else:
            if self.use_gpu: 
                # GPU histogramer
                bin_edges = deepcopy(self.bin_edges)
                bin_edges[self.e_bin_number] *= FTYPE(self.params.hist_e_scale.value.m_as('dimensionless'))
                if 'pid' in self.bin_names:
                    bin_edges[self.pid_bin_number][1] *= FTYPE(self.params.hist_pid_scale.value.m_as('dimensionless'))
                self.histogrammer = self.GPUHist(*bin_edges)
            #TODO What should happen when using CPU mode and KDE is false????
            #TODO Or alternatively, does CPU offer some mode (perhaps non-hist) that GPU does not currently???



    @profile
    def _compute_outputs(self, inputs=None):

        """Compute histograms for output channels."""

        #
        # Get input data
        #

        ''' #TODO UPDATE
        #Two cases:
        #  1) Receive the input 'Data' instance from an upstream stage via the 'inputs' argument to this function
        #  2) User has provided a file containing data for the legacy 'Events' class (maintained for backwards compatibility)

        input_data = None

        if self.legacy_events_file :

            #Using a legacy file, open it and convert to the desired 'Data' format...

            #Check there is no other data being provided
            if inputs is not None :
                raise AssertionError("There are inputs to this stage from an upstream stage, but user has also specified an events file. Must be one or the other." )

            #Convert 'Events' instance to 'Data' instance #TODO factor out into a dedicated function
            self._data = Events(self.legacy_events_file)


        else :

            #Expect data from an upstream stage
            if inputs is None :
                raise AssertionError("No data found in inputs. Perhaps no upstream 'data' stage was run?" )
        '''

        #Check the input data
        if not isinstance(inputs, Data):
            raise AssertionError('inputs is not a Data object, instead is '
                                 'type {0}'.format(type(inputs)))

        #Get the data #TODO Should we actually bother updating this if the hash is the same? Don't want to overwrite changes,e g. added fluxes
        new_data = inputs

        #Check input data has changed
        new_sample_hash = deepcopy(new_data.metadata['sample_hash'])
        if new_sample_hash != self.sample_hash :

            #Found new data
            self._data = inputs

            if self.neutrinos :

                #Add flux to neutrino events if required #TODO Maybe handle this more like Shivesh has, but for now keeping things as cloe to Philipp as possible
                self.add_bar_fluxes_to_events()

                #Calculate weighted effective area if required
                self.add_weighted_aeff_to_events()

                #If no other PID provided, use tracklength
                self.add_tracklength_pid_to_events()

                #Format the data ready for use in the re-weighting code #TODO Enforce once only, maybe my moving to better way of copying to GPU
                self.format_nu_data_for_processing() #TODO Don't redo this every time

        #Store the latest hash
        self.sample_hash = new_sample_hash
        logging.trace('{0} weight sample_hash = '
                      '{1}'.format(inputs.metadata['name'], self.sample_hash))
        logging.trace('{0} weight weight_hash = '
                      '{1}'.format(inputs.metadata['name'], self.weight_hash))

        #Perform the event-by-event re-weighting
        self.reweight()

        #Return the events themselves if requested by the user, otherwise fill Maps from the events and return these
        if self.output_events:
            return self._data
        else :
            return self.get_maps_from_data()



    def add_weighted_aeff_to_events(self) : 

        #The weighting code (GPU/CPUWeight.calc_weight) uses the a parameter 'weighted_aeff'
        #to determine the weight (taking into account the flux, oscillations and xsec effects)
        #This is derived from OneWeight as: weighted_aeff = one_weight / (N_events_per_file * N_file) #TODO nu.nubar generation ratio???
        #When inout was a PISA hdf5 file produced using 'make_events_file.py', this has already
        #been calculated (e.g. DRAGON), but when using the GRECO input we need to calculate this
        #ourselves from the pickle files

        var = "weighted_aeff"

        #Loop over flavor-interaction combinations
        for flavint in self._data.keys() :

            #Check if weighted effective area already calculated
            if var not in self._data[flavint] :

                #Need to calculate it...

                #Report (once only)
                if flavint == self._data.keys()[0] : logging.info("Adding %s to neutrino events" % var) 

                #For GRECO, the one_weight variable in the pickle files have already been normalised by (N_files*N_events) #TODO What about nu/nubar gen ration???
                if str(self._data.metadata["name"]).lower() == "greco" :
                    self._data[flavint][var] = deepcopy(self._data[flavint]["sample_weight"]) * CMSQ_TO_MSQ #onvert from cm^2 to m^2


    def add_tracklength_pid_to_events(self) :

        #GRECO uses the reconstructed tracklength of an event for PID

        if str(self._data.metadata["name"]).lower() == "greco" :

            #Loop over flavor-interaction combinations
            for flavint in self._data.keys() :

                #Write tracklength to the PID variable
                if 'pid' not in self._data[flavint] :
                    self._data[flavint]['pid'] = self._data[flavint]['tracklength']
                else :
                    raise Exception("'pid' variable already exists, which is not expected for GRECO")   


    def add_bar_fluxes_to_events(self) : #TODO Merge with compute_flux_weights

        #This is used to add flux values to events, in the format required by our 
        #implementation of the Barr 2006 flux uncertainties

        '''
        #Define the variables
        event_flux_variables = [
            'neutrino_nue_flux',
            'neutrino_numu_flux',
            'neutrino_oppo_nue_flux',
            'neutrino_oppo_numu_flux',
        ]

        #TODO Use cacheing here to only do the lookup once...

        #Check if the flux variables have already been added
        event_variables_found = self._data[self._data.keys()[0]].keys()
        if not np.any( [ var in event_variables_found for var in event_flux_variables ] ) :

            #No flux variables found, add them...

            logging.info( "Adding flux variables to events : Using flux file '%s'" % self.params['flux_file'].value )

            #Load a flux table
            flux_table = load_2D_table(self.params['flux_file'].value)

            #Add the fluxes to the events
            for fig in self._data.iterkeys():
                true_e = self._data[fig]['true_energy']
                true_cz = self._data[fig]['true_coszen']
                isbar = 'bar' if 'bar' in fig else ''
                nue_flux = calculate_flux_weights(true_e, true_cz, flux_table['nue'+isbar])
                numu_flux = calculate_flux_weights(true_e, true_cz, flux_table['numu'+isbar])
                # the opposite flavor fluxes( used only in the nu_nubar_ratio systematic)
                oppo_isbar = '' if 'bar' in fig else 'bar'
                oppo_nue_flux = calculate_flux_weights(true_e, true_cz, flux_table['nue'+oppo_isbar])
                oppo_numu_flux = calculate_flux_weights(true_e, true_cz, flux_table['numu'+oppo_isbar]) 
                self._data[fig]['neutrino_nue_flux'] = nue_flux
                self._data[fig]['neutrino_numu_flux'] = numu_flux
                self._data[fig]['neutrino_oppo_nue_flux'] = oppo_nue_flux
                self._data[fig]['neutrino_oppo_numu_flux'] = oppo_numu_flux
        '''

        #Check if data already contains flux
        data_contains_flux = all(
            ['neutrino_nue_flux' in fig and 'neutrino_numu_flux' in fig and 'neutrino_oppo_nue_flux' in fig
             and 'neutrino_oppo_numu_flux' in fig for fig in self._data.itervalues()]
        )

        if not data_contains_flux:

            #No flux, need to get either get it from cache or calculate it

            #If user has enabled caching, grab any cached data or else calculate new data
            if self.params['cache_flux'].value:
                this_cache_hash = hash_obj(
                    [self._data.metadata['name'], self._data.metadata['sample'],
                     self._data.metadata['cuts'], self.params['flux_file'].value],
                    full_hash=self.full_hash
                )

                if self.flux_cache_hash == this_cache_hash:
                    flux_weights = deepcopy(self._cached_fw)
                elif this_cache_hash in self.disk_cache:
                    logging.info('Loading flux values from disk cache.')
                    self._cached_fw = self.disk_cache[this_cache_hash]
                    flux_weights = deepcopy(self._cached_fw)
                    self.flux_cache_hash = this_cache_hash
                else:
                    flux_weights = self._compute_flux_weights_barr(
                        self._data, ParamSet(p for p in self.params
                                             if p.name in self.flux_params)
                    )

            #If not caching, need to calculate the fluxes
            else:
                flux_weights = self._compute_flux_weights_barr(
                self._data, ParamSet(p for p in self.params
                                     if p.name in self.flux_params)
            )

            #Store the flux to the disk cache
            if self.params['cache_flux'].value:
                if this_cache_hash not in self.disk_cache:
                    logging.info('Caching flux values to disk.')
                    self.disk_cache[this_cache_hash] = flux_weights

            #Add the fluxes to the events
            logging.info( "Adding flux variables to neutrino events" )
            for fig in self._data.iterkeys():
                self._data[fig]['neutrino_nue_flux'] = flux_weights[fig]['neutrino_nue_flux']
                self._data[fig]['neutrino_numu_flux'] = flux_weights[fig]['neutrino_numu_flux']
                self._data[fig]['neutrino_oppo_nue_flux'] = flux_weights[fig]['neutrino_oppo_nue_flux']
                self._data[fig]['neutrino_oppo_numu_flux'] = flux_weights[fig]['neutrino_oppo_numu_flux']



    @staticmethod
    def _compute_flux_weights_barr(nu_data, params):

        """Neutrino fluxes via integral preserving spline."""
        logging.info('Computing flux values in the format required for the Barr parameterisation (may take some time...)')

        logging.debug("Loading flux table : %s" % params['flux_file'].value)
        spline_dict = load_2D_table(params['flux_file'].value)
        logging.debug("Finished loading flux table")

        flux_weights = OrderedDict()
        for fig in nu_data.iterkeys():
            flux_weights[fig] = OrderedDict()
            true_e = nu_data[fig]['true_energy']
            true_cz = nu_data[fig]['true_coszen']
            logging.debug( "Calculating fluxes for %i '%s' events" % (len(true_e),fig) )
            isbar = 'bar' if 'bar' in fig else ''
            nue_flux = calculate_flux_weights(true_e, true_cz, spline_dict['nue'+isbar])
            numu_flux = calculate_flux_weights(true_e, true_cz, spline_dict['numu'+isbar])
            # the opposite flavor fluxes( used only in the Barr nu_nubar_ratio systematic)
            oppo_isbar = '' if 'bar' in fig else 'bar'
            oppo_nue_flux = calculate_flux_weights(true_e, true_cz, spline_dict['nue'+oppo_isbar])
            oppo_numu_flux = calculate_flux_weights(true_e, true_cz, spline_dict['numu'+oppo_isbar]) 
            flux_weights[fig]['neutrino_nue_flux'] = nue_flux
            flux_weights[fig]['neutrino_numu_flux'] = numu_flux
            flux_weights[fig]['neutrino_oppo_nue_flux'] = oppo_nue_flux
            flux_weights[fig]['neutrino_oppo_numu_flux'] = oppo_numu_flux

        return flux_weights


    def format_nu_data_for_processing(self) :

        #TODO This is very clunky, and only really used to shoehorn Philipp's stuff into here, I prefer the way it is done in Shivesh's osc reweighting function, update to be more like this...

        #
        # Define data format
        #

        if self.neutrinos:

            #Define the variables that events are expected to contain for the weighting code to follow
            required_event_variables = [
                'true_energy',
                'true_coszen',
                'reco_energy',
                'reco_coszen',
                'neutrino_nue_flux',
                'neutrino_numu_flux',
                'neutrino_oppo_nue_flux',
                'neutrino_oppo_numu_flux',
                'weighted_aeff',
            ]
            optional_event_variables = [
                'linear_fit_MaCCQE',
                'quad_fit_MaCCQE',
                'linear_fit_MaCCRES',
                'quad_fit_MaCCRES',
                'pid', #TODO Should this be mandatory???
            ]

            #Define variables that will be filled during the re-weighting
            #Initially these will be created as empty arrays (e.g. on the GPUs) and filled as we go
            output_event_variables = [
                'prob_e',
                'prob_mu',
                'weight',
                'scaled_nue_flux',
                'scaled_numu_flux',
                'scaled_nue_flux_shape',
                'scaled_numu_flux_shape'
            ]

            #Also add error method to this if relevent
            if self.error_method in ['sumw2', 'fixed_sumw2']:
                output_event_variables += ['sumw2']

            #TODO can pribably remove this and just directly call Data.keys wherever
            self.flavint_strings = self._data.keys()


            #
            # Create data arrays on CPU and/or GPU containing the neutrino data
            #

            # This creates the data arrays that will be processed by the oscillation code (prob3)
            # This will include the event data, nu flavor, and possibly Earth layers
            # Here the arrays created are referred to as the 'host' arrays, and reside on this machine
            # If running on a CPU, the 'host' copy of the arrays will be the only copy, and will be used 
            # for processing
            # If running on a GPU, will also make a 'device' copy (copied from the 'device' copy) shortly

            # setup all arrays that need to be put on GPU
            logging.debug('read in events and copy to GPU')
            start_t = time.time()
            self.nu_events_processing_dict = {}
            for flavint in self.flavint_strings :

                #Get the prib3 flavor and nu/nubar codes for this neutrino flavor
                kFlav,kNuBar = NuFlavInt(flavint).flav.prob3_codes

                self.nu_events_processing_dict[flavint] = {}
                # neutrinos: 1, anti-neutrinos: -1
                self.nu_events_processing_dict[flavint]['kNuBar'] = kNuBar
                # electron: 0, muon: 1, tau: 2
                self.nu_events_processing_dict[flavint]['kFlav'] = kFlav

                # Create fill arrays on the 'host', e.g. this machine (the 'device')
                self.nu_events_processing_dict[flavint]['host'] = {}
                for var in required_event_variables:
                    if var not in self._data[flavint] :
                        raise KeyError("Required variable '%s' missing for '%s'" % (var,flavint))
                    self.nu_events_processing_dict[flavint]['host'][var] = ( self._data[flavint][var].astype(FTYPE) )
                for var in optional_event_variables:
                    if var in self._data[flavint] :
                        self.nu_events_processing_dict[flavint]['host'][var] = ( self._data[flavint][var].astype(FTYPE) )
                    else :
                        # If variable doesn't exist (e.g. axial mass coeffs, just fill in ones) only warn first time
                        if flavint == self.flavint_strings[0]:
                            logging.warning('replacing variable %s by ones'%var)
                        self.nu_events_processing_dict[flavint]['host'][var] = np.ones_like(
                            self._data[flavint]['true_energy'],
                            dtype=FTYPE
                        )

                self.nu_events_processing_dict[flavint]['n_evts'] = np.uint32(
                    len(self.nu_events_processing_dict[flavint]['host'][required_event_variables[0]])
                )
                for var in output_event_variables:
                    if (self.params.no_nc_osc and
                            ((flavint in ['nue_nc', 'nuebar_nc'] and var == 'prob_e')
                             or (flavint in ['numu_nc', 'numubar_nc']
                                 and var == 'prob_mu'))):
                        # In case of not oscillating NC events, we can set the
                        # probabilities of nue->nue and numu->numu at 1, and
                        # nutau->nutau at 0
                        self.nu_events_processing_dict[flavint]['host'][var] = np.ones(
                            self.nu_events_processing_dict[flavint]['n_evts'], dtype=FTYPE
                        )
                    else:
                        self.nu_events_processing_dict[flavint]['host'][var] = np.zeros(
                            self.nu_events_processing_dict[flavint]['n_evts'], dtype=FTYPE
                        )

                # Calulate the layers of the Earth (every particle crosses a number of layers in the
                # earth with different densities, and for a given length these depend only on the earth
                # model (PREM) and the true coszen of an event. Therefore we can calculate these for 
                # once and are done
                # Note: prob3cpu handles this in a different way (entirely internally, no need to call
                # this function), so only do this in GPU mode
                if self.use_gpu_for_osc: 
                        nlayers, dens, dist = self.osc.calc_layers(
                            self.nu_events_processing_dict[flavint]['host']['true_coszen']
                        )
                        self.nu_events_processing_dict[flavint]['host']['numLayers'] = nlayers
                        self.nu_events_processing_dict[flavint]['host']['densityInLayer'] = dens
                        self.nu_events_processing_dict[flavint]['host']['distanceInLayer'] = dist

            end_t = time.time()
            logging.debug( 'Output data formatted%s in %.4f ms'% ( (" and Earth layers calculated" if self.use_gpu else "") , (end_t - start_t) * 1000) )


            #
            # Copy data arrays to GPU
            #

            #If using GPU, copy the data arrays across to it
            if self.use_gpu: 
                import pycuda.driver as cuda
                start_t = time.time()
                for flav in self.flavint_strings:
                    #Copy all data from the local ('host') events array on this machine to the GPU ('device') copy of the array
                    self.nu_events_processing_dict[flav]['device'] = {} 
                    for key, val in self.nu_events_processing_dict[flav]['host'].items():
                        self.nu_events_processing_dict[flav]['device'][key] = cuda.mem_alloc(val.nbytes)
                        cuda.memcpy_htod(self.nu_events_processing_dict[flav]['device'][key], val)
                end_t = time.time()
                logging.debug('copy of events to GPU device done in %.4f ms'%((end_t - start_t) * 1000))


        #
        # Apply raw reco sys
        #

        self.apply_reco()


    def apply_reco(self):
        """Apply raw reco systematics (to use as inputs to polyfit stage)"""
        for flav in self.flavint_strings:
            # Apply energy reco sys
            f = self.params.reco_e_res_raw.value.m_as('dimensionless')
            g = self.params.reco_e_scale_raw.value.m_as('dimensionless')
            self.nu_events_processing_dict[flav]['host']['reco_energy'] = (
                g * ((1.-f) * self.nu_events_processing_dict[flav]['host']['true_energy']
                     + f * self.nu_events_processing_dict[flav]['host']['reco_energy'])
            ).astype(FTYPE)

            # Apply coszen reco sys
            f = self.params.reco_cz_res_raw.value.m_as('dimensionless')
            self.nu_events_processing_dict[flav]['host']['reco_coszen'] = (
                (1.-f) * self.nu_events_processing_dict[flav]['host']['true_coszen']
                + f * self.nu_events_processing_dict[flav]['host']['reco_coszen']
            ).astype(FTYPE)

            # Make sure everything is within -1 <= coszen <= 1, otherwise
            # reflect
            reco_cz = self.nu_events_processing_dict[flav]['host']['reco_coszen']
            lt_m1_mask = reco_cz < -1
            gt_p1_mask = reco_cz > 1
            while np.any(lt_m1_mask + gt_p1_mask):
                reco_cz[gt_p1_mask] = 2 - reco_cz[gt_p1_mask]
                reco_cz[lt_m1_mask] = -2 - reco_cz[lt_m1_mask]
                lt_m1_mask = reco_cz < -1
                gt_p1_mask = reco_cz > 1

            #If using GPU, write these reco values to the data arrays on the GPU
            if self.use_gpu: 
                self.update_device_arrays(flav, 'reco_energy')
                self.update_device_arrays(flav, 'reco_coszen')


    def update_device_arrays(self, flav, var):
        """Helper function to update device arrays from the host arrays"""
        import pycuda.driver as cuda
        self.nu_events_processing_dict[flav]['device'][var].free()
        self.nu_events_processing_dict[flav]['device'][var] = cuda.mem_alloc(
            self.nu_events_processing_dict[flav]['host'][var].nbytes
        )
        cuda.memcpy_htod(
            self.nu_events_processing_dict[flav]['device'][var],
            self.nu_events_processing_dict[flav]['host'][var]
        )


    def get_device_arrays(self, variables=['weight']):
        """Copy back event by event information from the device dict into the host dict"""
        import pycuda.driver as cuda
        for flav in self.flavint_strings:
            for var in variables:
                buff = np.full(self.nu_events_processing_dict[flav]['n_evts'],
                               fill_value=np.nan, dtype=FTYPE)
                cuda.memcpy_dtoh(buff, self.nu_events_processing_dict[flav]['device'][var])
                assert np.all(np.isfinite(buff))
                self.nu_events_processing_dict[flav]['host'][var] = buff


    def sum_array(self, x, n_evts):
        """Helper function to compute the sum over a device array"""
        import pycuda.driver as cuda
        out = np.array([0.], dtype=FTYPE)
        d_out = cuda.mem_alloc(out.nbytes)
        cuda.memcpy_htod(d_out, out)
        self.weight_calc.calc_sum(n_evts, x, d_out)
        cuda.memcpy_dtoh(out, d_out)
        return out[0]

    def reweight(self):

        """Main rewighting function."""

        #Check hash to see if we need to do anything (e.g. has anything changed?)
        this_hash = hash_obj(
            [self.sample_hash, self.params.values_hash],
            full_hash = self.full_hash
        )
        if this_hash == self.weight_hash:
            return

        #Perform the re-weighting for each class of event
        if self.neutrinos:
            self.reweight_neutrinos()

        if self.muons:
            self.reweight_muons()

        if self.noise:
            self.reweight_noise()

        #Update hash
        self.weight_hash = this_hash
        self._data.metadata['weight_hash'] = self.weight_hash
        self._data.update_hash()


    def reweight_neutrinos(self):

        #TODO Check if nu are present

        #TODO Merge properly
        self.reweight_neutrinos_philipp()
        #self.reweight_neutrinos_shivesh()


    def reweight_neutrinos_philipp(self) : #TODO Merge properly

        #print "+++ Philipp re-weight start"

        #
        # Determine what needs to be re-calculated
        #

        # Get hash to decide whether expensive stuff needs to be recalculated
        osc_param_vals = [self.params[name].value for name in self.osc_params]
        gpu_flux_vals = [self.params[name].value for name in self.flux_params]
        
        #true_params_vals = [self.params[name].value for name in self.true_params]
        #osc_param_vals += true_params_vals
        #gpu_flux_vals += true_params_vals #TODO required???

        if self.full_hash:
            osc_param_vals = normQuant(osc_param_vals)
            gpu_flux_vals = normQuant(gpu_flux_vals)
        osc_hash = hash_obj(osc_param_vals, full_hash=self.full_hash)
        flux_hash = hash_obj(gpu_flux_vals, full_hash=self.full_hash)

        recalc_osc = not (osc_hash == self.osc_hash)
        recalc_flux = not (flux_hash == self.flux_hash)

        #print "+++ Philipp re-weight : recalc_osc = %s" % recalc_osc
        #print "+++ Philipp re-weight : recalc_flux = %s" % recalc_flux



        #
        # Calculate weights
        #

        #Here the effects of flux and oscillations are calculated and combined to weight the MC events

        #First, get any params required...

        livetime = self.params.livetime.value.m_as('seconds')
        aeff_scale = self.params.aeff_scale.value.m_as('dimensionless')
        Genie_Ma_QE = self.params.Genie_Ma_QE.value.m_as('dimensionless')
        Genie_Ma_RES = self.params.Genie_Ma_RES.value.m_as('dimensionless')
        true_e_scale = self.params.true_e_scale.value.m_as('dimensionless')

        if recalc_flux:
            nue_numu_ratio = self.params.nue_numu_ratio.value.m_as('dimensionless')
            nu_nubar_ratio = self.params.nu_nubar_ratio.value.m_as('dimensionless')
            atm_delta_index = self.params.atm_delta_index.value.m_as('dimensionless')
            Barr_uphor_ratio = self.params.Barr_uphor_ratio.value.m_as('dimensionless')
            Barr_nu_nubar_ratio = self.params.Barr_nu_nubar_ratio.value.m_as('dimensionless')

        if recalc_osc:
            if self.use_gpu_for_osc: #TODO This functionality is only used in the GPU version (no update_MNS function, and nothing done here in cpu.py). Is there really nothing to do here, or is this a bug???
                theta12 = self.params.theta12.value.m_as('rad')
                theta13 = self.params.theta13.value.m_as('rad')
                theta23 = self.params.theta23.value.m_as('rad')
                deltam21 = self.params.deltam21.value.m_as('eV**2')
                deltam31 = self.params.deltam31.value.m_as('eV**2')
                deltacp = self.params.deltacp.value.m_as('rad')
                self.osc.update_MNS(theta12, theta13, theta23,
                                    deltam21, deltam31, deltacp)

        #Now loop over neutrino flavor/interactions and re-calculate flux and/or oscillations as required...
        self.num_events_processed = 0
        start_t = time.time()
        for flav in self.flavint_strings:

            #Specify which data array should be used in the calculations ('host' if running on CPU, 'device' if on GPU)
            data_array = self.nu_events_processing_dict[flav]['device'] if self.use_gpu else self.nu_events_processing_dict[flav]['host']

            # Calculate osc probs, filling the device arrays with probabilities
            if recalc_osc:
                if not (self.params.no_nc_osc.value and flav.endswith('_nc')): #Don't bother oscillation NC interactions if user requested it
                    if self.use_gpu_for_osc :
                        #Running on GPU case, use the appropriate args
                        self.osc.calc_probs(
                            kNuBar=self.nu_events_processing_dict[flav]['kNuBar'],
                            kFlav=self.nu_events_processing_dict[flav]['kFlav'],
                            n_evts=self.nu_events_processing_dict[flav]['n_evts'],
                            true_e_scale=true_e_scale,
                            **self.nu_events_processing_dict[flav]['device'] #Use data array on device
                        )
                    else :
                        #Running on CPU case, use the appropriate args
                        self.osc.calc_probs(
                            kNuBar=self.nu_events_processing_dict[flav]['kNuBar'],
                            kFlav=self.nu_events_processing_dict[flav]['kFlav'],
                            true_e_scale=true_e_scale,
                            **self.nu_events_processing_dict[flav]['host'] #Use data array on host
                        )

                        #If running on a GPU in general but using CPU for the oscillations part, need to update the device arrays
                        #TODO doe the values need copying to the host BEFORE? Maybe not because it is first here, but this is super future-proof...
                        if self.use_gpu :
                            self.update_device_arrays(flav, 'prob_e')
                            self.update_device_arrays(flav, 'prob_mu')

            #print "+++ Philipp re-weight : after osc re-calc : prob_e = %s" % data_array["prob_e"][:5]


            # Calculate weights
            if recalc_flux:
                # Calculate the flux weights
                self.weight_calc.calc_flux(
                    n_evts=self.nu_events_processing_dict[flav]['n_evts'],
                    nue_numu_ratio=nue_numu_ratio,
                    nu_nubar_ratio=nu_nubar_ratio,
                    kNuBar=self.nu_events_processing_dict[flav]['kNuBar'],
                    delta_index=atm_delta_index,
                    Barr_uphor_ratio=Barr_uphor_ratio,
                    Barr_nu_nubar_ratio=Barr_nu_nubar_ratio,
                    true_e_scale=true_e_scale,
                    **data_array
                )


            #print "+++ Philipp re-weight : after flux re-calc : scaled_nue_flux = %s" % data_array["scaled_nue_flux"][:5]

            # Calculate global scales for flux normalizations
            #nue_flux_norm_n = self.sum_array(data_array['scaled_nue_flux'], self.nu_events_processing_dict[flav]['n_evts'])
            #nue_flux_norm_d = self.sum_array(data_array['scaled_nue_flux_shape'], self.nu_events_processing_dict[flav]['n_evts'])
            #nue_flux_norm = nue_flux_norm_n / nue_flux_norm_d
            nue_flux_norm = 1.
            #numu_flux_norm_n = self.sum_array(data_array['scaled_numu_flux'], self.nu_events_processing_dict[flav]['n_evts'])
            #numu_flux_norm_d = self.sum_array(data_array['scaled_numu_flux_shape'], self.nu_events_processing_dict[flav]['n_evts'])
            #numu_flux_norm = numu_flux_norm_n / numu_flux_norm_d
            numu_flux_norm = 1.

            # Calculate the event weights, from osc. probs and flux weights
            # global scaling factors for the nue and numu flux can be
            # given, for normalization purposes
            self.weight_calc.calc_weight(
                self.nu_events_processing_dict[flav]['n_evts'],
                livetime=livetime,
                nue_flux_norm=nue_flux_norm,
                numu_flux_norm=numu_flux_norm,
                aeff_scale=aeff_scale,
                kNuBar=self.nu_events_processing_dict[flav]['kNuBar'],
                Genie_Ma_QE=Genie_Ma_QE,
                Genie_Ma_RES=Genie_Ma_RES,
                true_e_scale=true_e_scale,
                **data_array
            )

            #print "+++ Philipp re-weight : after weight re-calc : weight = %s" % data_array["weight"][:5]

            # Calculate weights squared, for error propagation
            if self.error_method in ['sumw2', 'fixed_sumw2']:
                self.weight_calc.calc_sumw2(
                    n_evts=self.nu_events_processing_dict[flav]['n_evts'],
                    **data_array
                )

            self.num_events_processed += self.nu_events_processing_dict[flav]['n_evts']

        end_t = time.time()
        logging.debug('Flux and/or oscillation calcs done in %.4f ms for %s events'
                      %(((end_t - start_t) * 1000), self.num_events_processed))

        # Done with osc and flux calculations now, store the hashes ready fo comparison at the next call
        self.osc_hash = osc_hash
        self.flux_hash = flux_hash


    def reweight_neutrinos_shivesh(self) : #TODO Merge properly

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

    def reweight_muons(self):

        #TODO Check if mu are present

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


    def reweight_noise(self):

        #TODO Check if noise is present

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




    def compute_xsec_weights(self): #TODO Switch from Philipp's method to this one (Shivesh's)????
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
                osc_weights[fig] *= self.params['nutau_norm'].m

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


    def get_maps_from_data(self) :

        #TODO Safety checks that this is not called at the wrong time?

        #TODO merge
        return self.get_maps_from_data_philipp()
        #return self.get_maps_from_data_shivesh()


    def get_maps_from_data_shivesh(self) : #TODO merge

        #TODO Can the actual filling be turned into a single function that can be filled once for each event class (e.g. nu,mu,noise)

        outputs = []
        if self.neutrinos:
            trans_nu_data = self._data.transform_groups(
                self._output_nu_groups
            )
            for fig in trans_nu_data.iterkeys():
                if self.params.kde_hist.value:
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
            if self.params.kde_hist.value:
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
            if self.params.kde_hist.value:
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





    def get_maps_from_data_philipp(self) : #TODO merge

        #
        # Make histograms
        #

        #Create histograms from the weighted events
        #Making one histogram per flavor/interaction 

        #KDE case
        if self.params.kde_hist.value:
            start_t = time.time()

            #If using a GPU, copy back weights
            if self.use_gpu: 
                self.get_device_arrays(variables=['weight'])

            for flav in self.flavint_strings:
                # loop over pid bins and for every bin evaluate the KDEs
                # and put them together into a 3d array
                data = np.array([
                    self.nu_events_processing_dict[flav]['host'][self.bin_names[0]],
                    self.nu_events_processing_dict[flav]['host'][self.bin_names[1]],
                    self.nu_events_processing_dict[flav]['host'][self.bin_names[2]]
                ])
                weights = self.nu_events_processing_dict[flav]['host']['weight']
                hist = self.kde_histogramdd(
                        data.T,
                        weights=weights,
                        binning=self.output_binning,
                        coszen_name='reco_coszen',
                        use_cuda=True,
                        bw_method='silverman',
                        alpha=0.8,
                        oversample=1,
                        coszen_reflection=0.5,
                        adaptive=True
                    )
                self.nu_events_processing_dict[flav]['hist'] = hist
            end_t = time.time()
            logging.debug('KDE done in %.4f ms for %s events'
                          %(((end_t - start_t) * 1000), self.num_events_processed))

        #Regular histogram case
        else:

            # Scale the bin edges according to the energy and PID scale
            bin_edges = deepcopy(self.bin_edges)
            bin_edges[self.e_bin_number] *= FTYPE(self.params.hist_e_scale.value.m_as('dimensionless'))
            if 'pid' in self.bin_names:
                bin_edges[self.pid_bin_number][1] *= FTYPE(self.params.hist_pid_scale.value.m_as('dimensionless'))

            #Histogram on GPU case (use GPUHist class)
            if self.use_gpu:

                self.histogrammer.update_bin_edges(*bin_edges)

                start_t = time.time()
                # Histogram events and download fromm GPU, if either weights or
                # osc changed
                if len(self.bin_names) == 2:
                    for flav in self.flavint_strings:
                        hist = self.histogrammer.get_hist(
                            self.nu_events_processing_dict[flav]['n_evts'],
                            d_x = self.nu_events_processing_dict[flav]['device'][self.bin_names[0]],
                            d_y = self.nu_events_processing_dict[flav]['device'][self.bin_names[1]],
                            d_w = self.nu_events_processing_dict[flav]['device']['weight']
                        )
                        self.nu_events_processing_dict[flav]['hist'] = hist

                        if self.error_method in ['sumw2', 'fixed_sumw2']:
                            sumw2 = self.histogrammer.get_hist(
                                self.nu_events_processing_dict[flav]['n_evts'],
                                d_x=self.nu_events_processing_dict[flav]['device'][self.bin_names[0]],
                                d_y=self.nu_events_processing_dict[flav]['device'][self.bin_names[1]],
                                d_w=self.nu_events_processing_dict[flav]['device']['sumw2']
                            )
                            self.nu_events_processing_dict[flav]['sumw2'] = sumw2
                else:
                    for flav in self.flavint_strings:
                        hist = self.histogrammer.get_hist(
                            self.nu_events_processing_dict[flav]['n_evts'],
                            d_x=self.nu_events_processing_dict[flav]['device'][self.bin_names[0]],
                            d_y=self.nu_events_processing_dict[flav]['device'][self.bin_names[1]],
                            d_z=self.nu_events_processing_dict[flav]['device'][self.bin_names[2]],
                            d_w=self.nu_events_processing_dict[flav]['device']['weight']
                        )
                        self.nu_events_processing_dict[flav]['hist'] = hist

                        if self.error_method in ['sumw2', 'fixed_sumw2']:
                            sumw2 = self.histogrammer.get_hist(
                                self.nu_events_processing_dict[flav]['n_evts'],
                                d_x=self.nu_events_processing_dict[flav]['device'][self.bin_names[0]],
                                d_y=self.nu_events_processing_dict[flav]['device'][self.bin_names[1]],
                                d_z=self.nu_events_processing_dict[flav]['device'][self.bin_names[2]],
                                d_w=self.nu_events_processing_dict[flav]['device']['sumw2']
                            )
                            self.nu_events_processing_dict[flav]['sumw2'] = sumw2

                end_t = time.time()
                logging.debug('GPU hist done in %.4f ms for %s events'
                              %(((end_t - start_t) * 1000), self.num_events_processed))


            #Histogram on CPU case (use numpy N-dim histogram)
            else :

                # Histogram events
                #TODO only if something changed? Store hist?
                for flav in self.flavint_strings:

                    sample = []
                    for bin_name in self.bin_names:
                        sample.append(self.nu_events_processing_dict[flav]['host'][bin_name])
                    hist,_ = np.histogramdd(
                        sample=np.array(sample).T,
                        bins=self.bin_edges,
                        weights=self.nu_events_processing_dict[flav]['host']['weight']
                    )
                    self.nu_events_processing_dict[flav]['hist'] = hist #Store hist

                    if self.error_method in ['sumw2', 'fixed_sumw2']:
                        sumw2,_ = np.histogramdd(
                            sample=np.array(sample).T,
                            bins=self.bin_edges,
                            weights=self.nu_events_processing_dict[flav]['host']['sumw2']
                        )
                        self.nu_events_processing_dict[flav]['sumw2'] = sumw2 #Store sumw2 hist

                logging.debug('CPU hist done for %s events'%(self.num_events_processed))


        # Add histos together into output names, and apply nutau normalizations
        # errors (sumw2) are also added, while scales are applied in quadrature
        # of course
        out_hists = {}
        out_sumw2 = {}
        for name in self.output_names:
            for flav in self.flavint_strings:
                f = 1.0
                if flav in ['nutau_cc', 'nutaubar_cc']:
                    f *= self.params.nutau_cc_norm.value.m_as('dimensionless')
                if 'nutau' in flav:
                    f *= self.params.nutau_norm.value.m_as('dimensionless')
                if ('bar_nc' in flav and 'allbar_nc' in name) or ('_nc' in flav and 'all_nc' in name) or (flav in name):
                    if out_hists.has_key(name):
                        out_hists[name] += self.nu_events_processing_dict[flav]['hist'] * f
                        if self.error_method in ['sumw2', 'fixed_sumw2']:
                            out_sumw2[name] += self.nu_events_processing_dict[flav]['sumw2'] * f * f
                    else:
                        out_hists[name] = np.copy(self.nu_events_processing_dict[flav]['hist']) * f
                        if self.error_method in ['sumw2', 'fixed_sumw2']:
                            out_sumw2[name] = np.copy(self.nu_events_processing_dict[flav]['sumw2']) * f * f

        # Pack everything in a final PISA MapSet
        maps = []
        for name, hist in out_hists.items():
            if self.error_method == 'sumw2':
                maps.append(Map(name=name, hist=hist,
                                error_hist=np.sqrt(out_sumw2[name]),
                                binning=self.output_binning))
            # This is a special case where we always want the error to be the
            # same....so for the first Mapet it is taken from the calculation,
            # and every following time it is just euqal to the first one
            elif self.error_method == 'fixed_sumw2':
                if self.fixed_error == None:
                    self.fixed_error = {}
                if not self.fixed_error.has_key(name):
                    self.fixed_error[name] = np.sqrt(out_sumw2[name])
                maps.append(Map(name=name, hist=hist,
                                error_hist=self.fixed_error[name],
                                binning=self.output_binning))
            else:
                maps.append(Map(name=name, hist=hist,
                                binning=self.output_binning))

        return MapSet(maps, name='weight_tom_mc')






    def fill_map_for_species(self,species_events_dict) : #TOD better name

        #This function is used to create a map/histogram from the events of a
        #given species of events (e.g. numu_cc, muons, noise, etc)
        #It handles three method of histogramming:
        #  1) KDE (CU or GPU)
        #  2) GPU histogramming
        #  3) CPU histogramming

        if self.params.kde_hist.value:

            #
            # KDE histogramming
            #

            start_t = time.time()

            #If using a GPU, copy back weights
            if self.use_gpu: 
                self.get_device_arrays(variables=['weight'])

            for flav in self.flavint_strings:
                # loop over pid bins and for every bin evaluate the KDEs
                # and put them together into a 3d array
                data = np.array([
                    self.nu_events_processing_dict[flav]['host'][self.bin_names[0]],
                    self.nu_events_processing_dict[flav]['host'][self.bin_names[1]],
                    self.nu_events_processing_dict[flav]['host'][self.bin_names[2]]
                ])
                weights = self.nu_events_processing_dict[flav]['host']['weight']
                hist = self.kde_histogramdd(
                        data.T,
                        weights=weights,
                        binning=self.output_binning,
                        coszen_name='reco_coszen',
                        use_cuda=True, #TODO CPU option...
                        bw_method='silverman',
                        alpha=0.8,
                        oversample=1,
                        coszen_reflection=0.5,
                        adaptive=True
                    )
                self.nu_events_processing_dict[flav]['hist'] = hist

            end_t = time.time()

            logging.debug('KDE done in %.4f ms for %s events'
                          %(((end_t - start_t) * 1000), self.num_events_processed))


        else:

            #
            # Regular histogramming
            #

            #Do anything that is common to both CPU and GPU histgramming here first, thn specialise after 

            # Scale the bin edges according to the energy and PID scale
            bin_edges = deepcopy(self.bin_edges)
            bin_edges[self.e_bin_number] *= FTYPE(self.params.hist_e_scale.value.m_as('dimensionless'))
            if 'pid' in self.bin_names:
                bin_edges[self.pid_bin_number][1] *= FTYPE(self.params.hist_pid_scale.value.m_as('dimensionless'))

            #Histogram on GPU case (use GPUHist class)
            if self.use_gpu:

                #
                # Use GPU for histogram
                #

                self.histogrammer.update_bin_edges(*bin_edges)

                start_t = time.time()
                # Histogram events and download fromm GPU, if either weights or
                # osc changed
                if len(self.bin_names) == 2:
                    for flav in self.flavint_strings:
                        hist = self.histogrammer.get_hist(
                            self.nu_events_processing_dict[flav]['n_evts'],
                            d_x = self.nu_events_processing_dict[flav]['device'][self.bin_names[0]],
                            d_y = self.nu_events_processing_dict[flav]['device'][self.bin_names[1]],
                            d_w = self.nu_events_processing_dict[flav]['device']['weight']
                        )
                        self.nu_events_processing_dict[flav]['hist'] = hist

                        if self.error_method in ['sumw2', 'fixed_sumw2']:
                            sumw2 = self.histogrammer.get_hist(
                                self.nu_events_processing_dict[flav]['n_evts'],
                                d_x=self.nu_events_processing_dict[flav]['device'][self.bin_names[0]],
                                d_y=self.nu_events_processing_dict[flav]['device'][self.bin_names[1]],
                                d_w=self.nu_events_processing_dict[flav]['device']['sumw2']
                            )
                            self.nu_events_processing_dict[flav]['sumw2'] = sumw2
                else:
                    for flav in self.flavint_strings:
                        hist = self.histogrammer.get_hist(
                            self.nu_events_processing_dict[flav]['n_evts'],
                            d_x=self.nu_events_processing_dict[flav]['device'][self.bin_names[0]],
                            d_y=self.nu_events_processing_dict[flav]['device'][self.bin_names[1]],
                            d_z=self.nu_events_processing_dict[flav]['device'][self.bin_names[2]],
                            d_w=self.nu_events_processing_dict[flav]['device']['weight']
                        )
                        self.nu_events_processing_dict[flav]['hist'] = hist

                        if self.error_method in ['sumw2', 'fixed_sumw2']:
                            sumw2 = self.histogrammer.get_hist(
                                self.nu_events_processing_dict[flav]['n_evts'],
                                d_x=self.nu_events_processing_dict[flav]['device'][self.bin_names[0]],
                                d_y=self.nu_events_processing_dict[flav]['device'][self.bin_names[1]],
                                d_z=self.nu_events_processing_dict[flav]['device'][self.bin_names[2]],
                                d_w=self.nu_events_processing_dict[flav]['device']['sumw2']
                            )
                            self.nu_events_processing_dict[flav]['sumw2'] = sumw2

                end_t = time.time()
                logging.debug('GPU hist done in %.4f ms for %s events'
                              %(((end_t - start_t) * 1000), self.num_events_processed))


            else :

                #
                # Use CPU for histogram
                #

                #Histogram on CPU case (use numpy N-dim histogram) #TOD Use Shivesh's function in Data???

                # Histogram events
                #TODO only if something changed? Store hist?
                for flav in self.flavint_strings:

                    sample = []
                    for bin_name in self.bin_names:
                        sample.append(self.nu_events_processing_dict[flav]['host'][bin_name])
                    hist,_ = np.histogramdd(
                        sample=np.array(sample).T,
                        bins=self.bin_edges,
                        weights=self.nu_events_processing_dict[flav]['host']['weight']
                    )
                    self.nu_events_processing_dict[flav]['hist'] = hist #Store hist

                    if self.error_method in ['sumw2', 'fixed_sumw2']:
                        sumw2,_ = np.histogramdd(
                            sample=np.array(sample).T,
                            bins=self.bin_edges,
                            weights=self.nu_events_processing_dict[flav]['host']['sumw2']
                        )
                        self.nu_events_processing_dict[flav]['sumw2'] = sumw2 #Store sumw2 hist

                logging.debug('CPU hist done for %s events'%(self.num_events_processed))


        # Add histos together into output names, and apply nutau normalizations
        # errors (sumw2) are also added, while scales are applied in quadrature
        # of course
        out_hists = {}
        out_sumw2 = {}
        for name in self.output_names:
            for flav in self.flavint_strings:
                f = 1.0
                if flav in ['nutau_cc', 'nutaubar_cc']:
                    f *= self.params.nutau_cc_norm.value.m_as('dimensionless')
                if 'nutau' in flav:
                    f *= self.params.nutau_norm.value.m_as('dimensionless')
                if ('bar_nc' in flav and 'allbar_nc' in name) or ('_nc' in flav and 'all_nc' in name) or (flav in name):
                    if out_hists.has_key(name):
                        out_hists[name] += self.nu_events_processing_dict[flav]['hist'] * f
                        if self.error_method in ['sumw2', 'fixed_sumw2']:
                            out_sumw2[name] += self.nu_events_processing_dict[flav]['sumw2'] * f * f
                    else:
                        out_hists[name] = np.copy(self.nu_events_processing_dict[flav]['hist']) * f
                        if self.error_method in ['sumw2', 'fixed_sumw2']:
                            out_sumw2[name] = np.copy(self.nu_events_processing_dict[flav]['sumw2']) * f * f

        # Pack everything in a final PISA MapSet
        maps = []
        for name, hist in out_hists.items():
            if self.error_method == 'sumw2':
                maps.append(Map(name=name, hist=hist,
                                error_hist=np.sqrt(out_sumw2[name]),
                                binning=self.output_binning))
            # This is a special case where we always want the error to be the
            # same....so for the first Mapet it is taken from the calculation,
            # and every following time it is just euqal to the first one
            elif self.error_method == 'fixed_sumw2':
                if self.fixed_error == None:
                    self.fixed_error = {}
                if not self.fixed_error.has_key(name):
                    self.fixed_error[name] = np.sqrt(out_sumw2[name])
                maps.append(Map(name=name, hist=hist,
                                error_hist=self.fixed_error[name],
                                binning=self.output_binning))
            else:
                maps.append(Map(name=name, hist=hist,
                                binning=self.output_binning))

        return MapSet(maps, name='weight_tom_mc')





    def validate_params(self, params):
        pq = ureg.Quantity
        param_types = [
            ('kde_hist', bool),
            ('livetime', pq),
            ('aeff_scale', pq),
            ('hist_e_scale', pq),
            ('hist_pid_scale', pq),
            ('nutau_cc_norm',pq),
            ('reco_e_res_raw',pq),
            ('reco_e_scale_raw',pq),
            ('reco_cz_res_raw',pq),
        ]
        if self.neutrinos:
            param_types.extend([
                ('flux_reweight', bool),
                ('oscillate', bool),
                ('cache_flux', bool),
#                ('nu_diff_DIS', pq),
#                ('nu_diff_norm', pq),
#                ('nubar_diff_DIS', pq),
#                ('nubar_diff_norm', pq),
#                ('hadron_DIS', pq),
                ('Genie_Ma_QE', pq),
                ('Genie_Ma_RES', pq),
                ('flux_file', basestring),
                ('atm_delta_index', pq),
                ('nu_nubar_ratio', pq),
                ('nue_numu_ratio', pq),
#                ('norm_numu', pq),
#                ('norm_nc', pq),
                ('Barr_uphor_ratio', pq),
                ('Barr_nu_nubar_ratio', pq),
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
                ('nutau_norm', pq),
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



