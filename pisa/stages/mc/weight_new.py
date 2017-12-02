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
from pisa.utils.flavInt import ALL_NUFLAVINTS, NuFlav, NuFlavInt, NuFlavIntGroup
from pisa.utils.flux_weights import load_2D_table, calculate_flux_weights
from pisa.utils.format import text2tex
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.resources import open_resource
from pisa.utils.comparisons import normQuant
from pisa.scripts.make_events_file import CMSQ_TO_MSQ
from pisa.stages.data.sample import parse_event_type_names
from pisa.stages.osc.prob3_new import prob3wrapper #TODO (Tom) prob3new -> prob3
from pisa.stages.osc.decoherence import decoherence
from pisa.stages.osc.osc_spline import OscSpline
#from pisa.stages.reco.hist_new import calc_true_to_reco_transforms #TODO (Tom) hist_new -> hist
from pisa.utils.flavInt import flavintGroupsFromString
from pisa.utils.fileio import mkdir, from_file, to_file

__all__ = ['weight_new']


class weight_new(Stage):
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
                - flux_file
                - atm_delta_index
                - nue_numu_ratio
                - nu_nubar_ratio
                - norm_numu
                - norm_nc
                    Flag to specifiy whether to cache the flux values if
                    calculated inside this service to a file specified
                    by `disk_cache`.

            * Oscillation related parameters:
                For more information see `$PISA/pisa/stage/osc/prob3gpu.py`
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
                 outputs_cache_depth=20, use_gpu=False, cache_flux=True,
                 kde_hist=False, decoherence=False,
                 spline_osc_probs=False, osc_spline_binning=None,
                 output_events_file=None):

        self.sample_hash = None #Input event sample hash
        self.weight_hash = None
        self.flux_cache_hash = None #flux calculation hash
        self.xsec_hash = None #xsec weights calc inputs hash
        self.flux_hash = None #weight weights calc inputs hash
        self.osc_hash = None #oscillation probability calc inputs hash
        self.muon_hash = None #atmospheric muon weight calc inputs hash
        self.noise_hash = None #noiseweight calc inputs hash
        self.weight_calc_hash = None #weights calc inputs hash

        self.weight_params = (
            'livetime',
            "aeff_scale",
            'true_e_scale',
            "nutau_cc_norm",
            'nu_nc_norm',
            'reco_e_res_raw',
            'reco_e_scale_raw',
            'reco_cz_res_raw',
        )

        self.hist_params = (
            "hist_e_scale",
            'hist_pid_scale',
        )

        self.xsec_params = (
            'nu_diff_DIS',
            'nu_diff_norm',
            'nubar_diff_DIS',
            'nubar_diff_norm',
            'hadron_DIS',
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
            'YeM', 
            'YeO',
            'detector_depth', 
            'prop_height',
            'deltacp', 
            'deltam21', 
            'deltam31',
            'theta12', 
            'theta13', 
            'theta23',
            'nutau_norm',
            'no_nc_osc',
        )

        self.atm_muon_params = (
            'atm_muon_scale',
            'delta_gamma_mu_file', #TODO (Tom) -> kwarg?
            'delta_gamma_mu_spline_kind', #TODO (Tom) -> kwarg?
            'delta_gamma_mu_variable', #TODO (Tom) -> kwarg?
            'delta_gamma_mu'
        )

        self.noise_params = (
            'norm_noise',
        )

        #Store kwargs
        self.use_gpu = use_gpu
        self.cache_flux = cache_flux
        self.kde_hist = kde_hist
        self.decoherence = decoherence
        self.spline_osc_probs = spline_osc_probs
        self.osc_spline_binning = osc_spline_binning
        self.output_events_file = output_events_file

        #Handle nuances when decoherence is included in oscillations model
        if self.decoherence :
            self.osc_params = tuple( list(self.osc_params) + ['gamma21','gamma31','gamma32'] ) #New params to consider
            if self.use_gpu : raise Exception("No GPU implementation for decoherence") #TODO numba

        #Get the names of all expected inputs and outputs
        input_names,self.muons,self.noise,self.neutrinos = parse_event_type_names(input_names,return_flags=True)
        output_names = parse_event_type_names(output_names,return_flags=False)

        #Define expected parameters based on types of input events
        expected_params = self.weight_params + self.hist_params
        if self.neutrinos :
            expected_params += self.xsec_params
            expected_params += self.flux_params
            expected_params += self.osc_params
        if self.muons :
            expected_params += self.atm_muon_params
        if self.noise :
            expected_params += self.noise_params

        if not isinstance(output_events, bool):
            raise AssertionError(
                'output_events must be of type bool, instead it is supplied '
                'with type {0}'.format(type(output_events))
            )
        if output_events:
            output_binning = None
        self.output_events = output_events

        super(weight_new, self).__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            debug_mode=debug_mode,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning
        )

        if self.muons:
            self.muon_prim_unc_spline = self.make_muon_prim_unc_spline()

        #If using GPUs, init CUDA
        if self.use_gpu:
            import pycuda.autoinit #This performs initialisation tasks #TODO Is it OK that this goes out of scope???

        #Grab histgramming tools of choice
        if self.kde_hist:
            #Use a histogram smoothed by a KDE
            if output_events : raise ValueError("Cannot use 'kde_hist' when selecting 'output_events'")
            from pisa.utils.kde_hist import kde_histogramdd
            self.kde_histogramdd = kde_histogramdd
        else:
            #Use a regular histogram, either computed on a GPU or locally
            if self.use_gpu:
                from pisa.utils.gpu_hist import GPUHist
                self.GPUHist = GPUHist

        #Register attributes that will be used for hashes
        self.include_attrs_for_hashes('use_gpu')
        self.include_attrs_for_hashes('spline_osc_probs')
        self.include_attrs_for_hashes('sample_hash')


    def _compute_nominal_outputs(self):

        #This function is called during initialisation, and is used to compute/init stuff (and hash it)
        #that is needed during compute_outputs but does not change at each new computation

        #
        # Hashing
        #

        #Reset reweighting hashes
        self.osc_hash = None
        self.flux_hash = None
        self.weight_calc_hash = None
        self.muon_hash = None
        self.noise_hash = None

        # Reset fixed errors
        self.fixed_error = None #TODO (Tom) remove once shift to latest histogramming tools???


        #
        # Prepare weight calculator
        #

        # Instantiate weight calculator (differs depending on whether using CPU or GPU code) #TODO (Tom) Merge these into a common class
        if self.use_gpu: from pisa.stages.mc.GPUWeight import GPUWeight as WeightCalculator
        else : from pisa.stages.mc.CPUWeight import CPUWeight as WeightCalculator
        self.weight_calc = WeightCalculator()


        #
        # Prepare histogramming tools
        #

        #Useing KDE case
        if self.kde_hist:

            #Cannot povide error method if using a KDE
            assert self.error_method == None

            #Check that coz(zenith) is one of the output binning dimensions
            self.kde_coszen_dim = None
            for dim_name in self.output_binning.names :
                if "coszen" in dim_name.lower() :
                    self.kde_coszen_dim = dim_name
                    break
            if self.kde_coszen_dim is None :
                raise ValueError("Did not find coszen in binning. KDE will not work correctly.")

        #Using regular histogram case
        else:

            #If using a GPU for histogramming, instantiate the helper class here
            if self.use_gpu: 
                self.gpu_histogrammer = self.GPUHist(*self.get_scaled_bin_edges())



    @profile
    def _compute_outputs(self, inputs=None):

        """Compute histograms for output channels."""

        #
        # Get input data
        #

        #Check the input data
        if not isinstance(inputs, Data):
            raise AssertionError('inputs is not a Data object, instead is '
                                 'type {0}'.format(type(inputs)))

        #Check if input data has changed
        new_sample_hash = deepcopy(inputs.metadata['sample_hash'])
        if new_sample_hash != self.sample_hash :

            #Found new data
            self._data = inputs

            #
            # Prepare oscillation calculation tools
            #

            if self.neutrinos:

                #Instantiate oscillation probability calculator
                if self.decoherence :
                    self.osc = decoherence(
                        earth_model=self.params.earth_model.value,
                        detector_depth=self.params.detector_depth.value,
                        prop_height=self.params.prop_height.value,
                        YeI=self.params.YeI.value,
                        YeO=self.params.YeO.value,
                        YeM=self.params.YeM.value,
                    )
                else :
                    self.osc = prob3wrapper(
                        earth_model=self.params.earth_model.value,
                        detector_depth=self.params.detector_depth.value,
                        prop_height=self.params.prop_height.value,
                        YeI=self.params.YeI.value,
                        YeO=self.params.YeO.value,
                        YeM=self.params.YeM.value,
                        use_gpu=self.use_gpu,
                    )

                #Instantiate oscillation spline tools if required
                if self.spline_osc_probs :
                    self.osc_spline = OscSpline(binning=self.osc_spline_binning,osc_calculator=self.osc,use_gpu=self.use_gpu)
                    logging.info("Using spline for oscillation calculations")


            #
            # Pre-process the data
            #

            #If no other PID provided, use reconstructed tracklength
            self._add_tracklength_pid_to_events()

            if self.neutrinos :

                #Add flux to neutrino events if required #TODO Maybe handle this more like Shivesh has, but for now keeping things as cloe to Philipp as possible
                self._add_flux_to_events()

                #Calculate weighted effective area if required (used by neutrino weight calculation)
                self._add_weighted_aeff_to_events()

            #TODO store new sample hash now thing shave been changed... Or does "Data.update_hash" already handle this????



            #
            # Handle event-by-event to bin-by-bin toggle (fast mode)
            #

            #TODO (Tom) this needs finishing

            '''
            self.bin_by_bin_mode = False #TODO kwarg

            if self.bin_by_bin_mode :
                self._convert_to_bin_by_bin_mode()
            '''


            #
            # Prepare data arrays
            #

            #Write the nu variable relevent to re-weighting calculations to data arrays on the host (CPU) 
            #and if required the device (GPU) 
            #TODO Enforce once only, maybe by moving to better way of copying to GPU
            #TODO Also for muons and noise?
            self._populate_nu_data_arrays()


        #Store the latest hash
        self.sample_hash = new_sample_hash
        self._data.metadata['sample_hash'] = self.sample_hash #Note that sample itself uis being changed above (adding flux, etc), so restoring the hash

        logging.trace('{0} weight sample_hash = {1}'.format(self._data.metadata['name'], self.sample_hash))
        logging.trace('{0} weight weight_hash = {1}'.format(self._data.metadata['name'], self.weight_hash)) #TODO Do I need to update the weight hash somewhere??


        #
        # Compute the outputs
        #

        #Perform the event-by-event re-weighting
        self.reweight()

        #Dump events to a file #TODO (Tom) remove this, is a hack for now just to get access to the events until we put something better in
        if self.output_events_file is not None :
            print "Dumping events to file : %s" % self.output_events_file
            output_events = {}
            event_type_keys = self._data.keys()
            if self.muons : event_type_keys.append("muons")
            if self.noise : event_type_keys.append("noise")
            variables = ["true_energy","true_coszen","reco_energy","reco_coszen","pisa_weight"]
            for event_type in event_type_keys :
                output_events[event_type] = {}
                for var in variables :
                    if var in self._data[event_type] :
                        output_events[event_type][var] = self._data[event_type][var]
            to_file(output_events,self.output_events_file)

        #Return the events themselves if requested by the user, otherwise fill Maps from the events and return these
        if self.output_events:
            return self._data

        else :

            #Compute maps
            output_maps =  self.get_output_maps()

            ''' #TODO (Tom) this needs finishing
            #If using bin-by-bin fast mode, apply true->reco transforms to the maps
            if self.bin_by_bin_mode :
                output_maps = self.reco_transforms.apply(output_maps)
            '''

            return output_maps



    def _add_weighted_aeff_to_events(self) : 

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


    def _add_tracklength_pid_to_events(self) :

        #GRECO uses the reconstructed tracklength of an event for PID
        if str(self._data.metadata["name"]).lower() == "greco" : #TODO make a variable in the greco.cfg file for this instead

            logging.info("Using 'tracklength' for 'pid'")

            #Loop over ALL events types, e.g. all nu flavor-interaction combinations, plus muons and noise
            for event_type in self._data.metadata['flavints_joined'] : #TODO is there somewhere better to get all present types from?

                #Write tracklength to the PID variable
                #Check if events already contain PID
                if 'pid' not in self._data[event_type] :

                    #No PID, so use track length
                    if 'tracklength' in self._data[event_type] :
                        self._data[event_type]['pid'] = self._data[event_type]['tracklength']
                    else :
                        raise ValueError("'tracklength' variable missing from '%s' events, cannot use as 'pid'" % event_type )   

                else :
                    raise ValueError("'pid' variable already exists in '%s' events, which is not expected for GRECO" % event_type )   


    def _add_flux_to_events(self) :

        # This is used to add flux values to events
        # In some cases theflux has already have been added upstream (e.g. 
        # in a hdf5 file), in which case nothing will be done here
        # The flux is being added in a format that is compatible with our 
        # implementation of the Barr 2006 flux uncertainties

        #Check if data already contains flux
        data_contains_flux = all(
            ['neutrino_nue_flux' in fig and 'neutrino_numu_flux' in fig and 'neutrino_oppo_nue_flux' in fig
             and 'neutrino_oppo_numu_flux' in fig for fig in self._data.itervalues()]
        )

        if not data_contains_flux:

            #No flux, need to get either get it from cache or calculate it

            #If user has enabled caching, grab any cached data or else calculate new data
            if self.cache_flux :
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
                    flux_weights = self._compute_flux_weights(
                        self._data, ParamSet(p for p in self.params
                                             if p.name in self.flux_params)
                    )

            #If not caching, need to calculate the fluxes
            else:
                flux_weights = self._compute_flux_weights(
                self._data, ParamSet(p for p in self.params
                                     if p.name in self.flux_params)
            )

            #Store the flux to the disk cache
            if self.cache_flux :
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


    '''
    #TODO (Tom) this needs finishing
    def _convert_to_bin_by_bin_mode(self) :

        #TODO document...

        #
        # Calculate true -> reco space transforms
        #

        #Not merging any groups (note: formatting according to pisa.utils.flavInt.flavintGroupsFromString)
        #transform_groups = self.input_names#", ".join(self.input_names)
        #transform_groups = flavintGroupsFromString(", ".join(self.input_names))
        #transform_groups = flavintGroupsFromString("nue_cc+nuebar_cc, numu_cc+numubar_cc, nutau_cc+nutaubar_cc, nuall_nc+nuallbar_nc")
        transform_groups = flavintGroupsFromString("") #Don't join anything (note: formatting according to pisa.utils.flavInt.flavintGroupsFromString)

        self.reco_transforms = calc_true_to_reco_transforms(input_data=self._data,
                                                            transform_groups=transform_groups,
                                                            input_names=self.input_names, #TODO what if input_names and output-names are not the same?
                                                            output_names=self.output_names, #TODO Do these need any thought?
                                                            input_binning=self.output_binning, #Use output binning for both
                                                            output_binning=self.output_binning,
                                                            e_res_scale=1.,# e_res_scale, #TODO
                                                            cz_res_scale=1., #cz_res_scale, #TODO
                                                            e_reco_bias=0., #*ureg["GeV"], #e_reco_bias, #TODO
                                                            cz_reco_bias=0., #*ureg["dimensionless"], #cz_reco_bias, #TODO
                                                            reco_weights_name=None, #TODO
                                                            res_scale_ref="zero",
                                                            sum_grouped_flavints=False, #TODO?
                                                            error_method=self.error_method )

        #
        # Create a single "event" for each histogram bin
        #

        #TODO
        
    '''


    @staticmethod
    def _compute_flux_weights(nu_data, params) : #TODO (Tom) Merge with `compute_flux_weights`, basically doing the same thing

        """Neutrino fluxes via integral preserving spline."""

        logging.info('Computing flux values for events (may take some time...)')

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

        #Check if either the sample of ANY of the input parameters have changed
        #If nothing has changed, then no need to re-calculate the weights
        #Note that individual re-weighting subfunctions called within this function
        #each check whether they need to be recalculated indivdually
        this_hash = self._get_params_hash(self.params)
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


    def reweight_neutrinos(self) :


        #
        # Calculate all contributions to re-weighting
        #

        #For each case, grab the subset of the stage parameters required for the calculation
        #Doing this here and passing to the function rather than directly accessing the 
        #member params class from  the subfunctions. This is so that each subfunction can 
        #easily check if its inputs have changed at each call (so they know whether to bother
        #recalculating)

        # XSec reweighting
        '''
        xsec_params = ParamSet( p for p in self.params if p.name in self.xsec_params )
        xsec_recalculated = self.compute_nu_xsec_weights_tom(xsec_params)
        #xsec_weights = self.compute_nu_xsec_weights_tom()
        #for fig in self._data.iterkeys():
        #    self._data[fig]['weight_weight'] *= xsec_weights[fig]
        '''

        #Flux systematics calculation
        flux_params = ParamSet( p for p in self.params if p.name in (list(self.flux_params)+["true_e_scale"]) )
        flux_recalculated = self.compute_nu_flux_weights(flux_params)

        #Oscillations calculation
        osc_params = ParamSet( p for p in self.params if p.name in (list(self.osc_params)+["true_e_scale"]) )
        osc_recalculated = self.compute_nu_osc_probabilities(osc_params)

        #DIS cross-section reweighting
        #Note that this function actually modifies the weights directly, whereas the the weight modifications for the 
        #osc and flux functions are handled below in `compute_weights`
        dis_xsec_params = ParamSet( p for p in self.params if p.name in ["nu_diff_DIS","nu_diff_norm","nubar_diff_DIS","nubar_diff_norm","hadron_DIS"] ) 
        xsec_recalculated = self.compute_nu_dis_xsec_weights(dis_xsec_params)

        
        #
        # Calculate new weights
        #

        #TODO (Tom) Use Shivesh style calculation here...

        #Only if any of the preceeding steps werre performed
        if flux_recalculated or osc_recalculated or xsec_recalculated : #TODO (Tom) hash data_arrays output variables instead?

            #Perform calculation
            weight_params = ParamSet( p for p in self.params if p.name in ["livetime","aeff_scale","Genie_Ma_QE","Genie_Ma_RES"] )
            self.compute_weights(weight_params)

            #Write final weight to data structure
            for fig in self._data.iterkeys():
                if self.use_gpu : self.update_host_arrays(fig) #Copy data back from GPU to here...
                self._data[fig]['pisa_weight'] = self._data_arrays[fig]['host']["weight"] #.ito('dimensionless') #TODO (Tom) attach units



    def compute_nu_osc_probabilities(self,params) :

        #This is where the probabilities for a given nu flavor state are calculate for each event
        #This function includes handling for the CPU vs GPU cases
        #Note that nothing is returned, instead the results can be found in self._data_arrays in
        #the variables listed in self.osc_output_variables


        #
        # Check if anything to do
        #

        #Get current state of all inputs to this calculation
        this_hash = self._get_params_hash(params)

        #If nothing has changed, then there is nothing to do
        if self.osc_hash == this_hash:
            return False


        #
        # Set oscillation parameters
        #

        #Grab required params
        true_e_scale = params.true_e_scale.value.m_as('dimensionless')

        #Update the params in the oscillation calculator
        '''
        theta12 = params.theta12.value.m_as('rad')
        theta13 = params.theta13.value.m_as('rad')
        theta23 = params.theta23.value.m_as('rad')
        deltam21 = params.deltam21.value.m_as('eV**2')
        deltam31 = params.deltam31.value.m_as('eV**2')
        deltacp = params.deltacp.value.m_as('rad')
        self.osc.set_params(theta12, theta13, theta23, deltam21, deltam31, deltacp)
        '''
        if self.decoherence :
                self.osc.set_params( theta12=params.theta12.value, 
                            theta13=params.theta13.value, 
                            theta23=params.theta23.value, 
                            deltam21=params.deltam21.value, 
                            deltam31=params.deltam31.value, 
                            deltacp=params.deltacp.value,
                            gamma21=params.gamma21.value,
                            gamma31=params.gamma31.value,
                            gamma32=params.gamma32.value )
        else :
            self.osc.set_params( theta12=params.theta12.value, 
                                theta13=params.theta13.value, 
                                theta23=params.theta23.value, 
                                deltam21=params.deltam21.value, 
                                deltam31=params.deltam31.value, 
                                deltacp=params.deltacp.value )

        #
        # Get oscillation splines
        #

        #Calculate oscillation splines for this new set of oscillation parameters
        #(if the user requested to use splines)
        if self.spline_osc_probs :
            self.osc_spline.generate_splines(true_e_scale=true_e_scale) 


        #
        # Calculate oscillation probabilities
        #

        #Loop over flavor/interaction combinations
        for fig in self._data.keys() :

            #If not calculating oscillations for NC interactions, skip this
            #(leaving prob_e and prob_mu as the default value of 1. in the data arrays)
            if 'nc' in fig and params.no_nc_osc.value:
                 continue

            #Get flav-int from string TODO use FlavInt directly as the map key
            flavint = NuFlavInt(fig)

            #Get data array on either host or device, depending on whether using GPU or CPU
            data_array = self._data_arrays[fig]["device"] if self.use_gpu else self._data_arrays[fig]["host"]

            #Spline case
            if self.spline_osc_probs :

                #Evaluate the spline at the E,coszen value for each event
                self.osc_spline.eval(
                    flav=flavint.flav,
                    true_e_scale=true_e_scale,
                    **data_array
                )

            #Event-by-event calculation case
            else :

                #Get the number of events
                n_evts = np.uint32(self.get_num_events(fig))

                #Get the prob3 flavor and nu/nubar codes for this neutrino flavor
                kFlav,kNuBar = flavint.flav.prob3_codes

                #Calculate the oscillation probabilities
                self.osc.calc_probs(
                    kNuBar=kNuBar,
                    kFlav=kFlav,
                    n_evts=n_evts,
                    true_e_scale=true_e_scale,
                    **data_array
                )

            #If running on a GPU in general but using CPU for the oscillations part, need to update the device arrays
            #This can be the case for vacuum oscillations
            if self.use_gpu and not self.osc.use_gpu :
                self.update_device_arrays(fig, 'prob_e')
                self.update_device_arrays(fig, 'prob_mu')


        #Store the new hash
        self.osc_hash = this_hash

        #Done. Let the calling function know a new calculation has been performed
        return True


    def compute_nu_flux_weights(self,params) :

        #This function is where the modifications to the event weight due to flux systematics
        #are calculated
        #This function includes handling for the CPU vs GPU cases
        #Note that nothing is returned, instead the results can be found in self._data_arrays in
        #the variables listed in self.flux_output_variables

        #TODO (Tom) attach_units????

        #
        # Check if anything to do
        #

        #Get current state of all inputs to this calculation
        this_hash = self._get_params_hash(params)

        #If nothing has changed, nothing to do
        if self.flux_hash == this_hash:
            return False


        #
        # Calculate flux weight modifications
        #

        #Grab required params
        true_e_scale = params.true_e_scale.value.m_as('dimensionless')
        nue_numu_ratio = params.nue_numu_ratio.value.m_as('dimensionless')
        nu_nubar_ratio = params.nu_nubar_ratio.value.m_as('dimensionless')
        atm_delta_index = params.atm_delta_index.value.m_as('dimensionless')
        Barr_uphor_ratio = params.Barr_uphor_ratio.value.m_as('dimensionless')
        Barr_nu_nubar_ratio = params.Barr_nu_nubar_ratio.value.m_as('dimensionless')

        #Loop over flavor/interaction combinations
        for fig in self._data.keys() :

            #Get the number of events
            n_evts = np.uint32(self.get_num_events(fig))

            #Get the prob3 flavor and nu/nubar codes for this neutrino flavor
            kFlav,kNuBar = NuFlavInt(fig).flav.prob3_codes

            #Use either host or device data array depending on whether runnong on GPU or CPU
            data_array = self._data_arrays[fig]['device'] if self.use_gpu else self._data_arrays[fig]['host']

            #Calculate the flux weights modifications
            self.weight_calc.calc_flux(
                n_evts=n_evts,
                nue_numu_ratio=nue_numu_ratio,
                nu_nubar_ratio=nu_nubar_ratio,
                kNuBar=kNuBar,
                delta_index=atm_delta_index,
                Barr_uphor_ratio=Barr_uphor_ratio,
                Barr_nu_nubar_ratio=Barr_nu_nubar_ratio,
                true_e_scale=true_e_scale,
                **data_array
            )

        #Store the new hash
        self.flux_hash = this_hash

        #Done. Let the calling function know a new calculation has been performed
        return True



    def compute_nu_dis_xsec_weights(self,params) :

        #Compute weight modifications due to DIS cross-section uncertainty

        #Note that uncertainties due to GENIE axial mass (quasi-elastic or
        #resonance) are handled separately by CPU/GPUWeight.calc_weight
        #TODO (Tom) Merge these in a more consistent way

        #TODO (Tom) Make a GPU version of this code?


        #
        # Check if anything to do
        #

        #Get current state of all inputs to this calculation
        this_hash = self._get_params_hash(params)

        #If nothing has changed, then there is nothing to do
        if self.xsec_hash == this_hash:
            return False


        #
        # Calculate DIS cross-section weight modifications
        #

        #Get param values
        nu_diff_DIS = params.nu_diff_DIS.value.m
        nu_diff_norm = params.nu_diff_norm.value.m
        nubar_diff_DIS = params.nubar_diff_DIS.value.m
        nubar_diff_norm = params.nubar_diff_norm.value.m
        hadron_DIS = params.hadron_DIS.value.m

        #Loop over neutrino flavour-interaction combinations
        for fig in self._data.keys() :

            # Differential xsec systematic
            if 'bar' not in fig:
                nu_diff_DIS_to_use = nu_diff_DIS
                nu_diff_norm_to_use = nu_diff_norm
            else:
                nu_diff_DIS_to_use = nubar_diff_DIS
                nu_diff_norm_to_use = nubar_diff_norm

            with np.errstate(divide='ignore', invalid='ignore'):
                xsec_weights = (
                    (1 - nu_diff_norm_to_use * nu_diff_DIS_to_use) *
                    np.power(self._data[fig]['GENIE_x'], -nu_diff_DIS_to_use)
                )
            xsec_weights[~np.isfinite(xsec_weights)] = 0. #TODO (Tom) REPORT WHEN THIS HAPPENS

            # High W hadronization systematic
            if hadron_DIS != 0.:
                xsec_weights *= (
                    1. / (1 + (2*hadron_DIS * np.exp(
                        -self._data[fig]['GENIE_y'] / hadron_DIS
                    )))
                )

            #Update event weight
            #This is running on host (CPU), so need to handle copy from/to GPU is using one
            if self.use_gpu : self.update_host_arrays(fig,"weight") #Get weight from GPU
            self._data_arrays[fig]['host']["weight"] * xsec_weights #Apply weight modifications
            if self.use_gpu : self.update_device_arrays(fig,"weight") #Copy new weight to GPU


        #Store the new hash
        self.xsec_hash = this_hash

        #Done. Let the calling function know a new calculation has been performed
        return True


    def compute_weights(self,params) :

        #
        # Check if anything to do
        #

        #Get current state of all inputs to this calculation
        this_hash = self._get_params_hash(params)

        #If nothing has changed, nothing to do
        if self.weight_calc_hash == this_hash:
            return False

       
        #
        # Calculate the weights
        #
            
        #Get params
        livetime = params.livetime.value.m_as('seconds')
        aeff_scale = params.aeff_scale.value.m_as('dimensionless')
        Genie_Ma_QE = params.Genie_Ma_QE.value.m_as('dimensionless')
        Genie_Ma_RES = params.Genie_Ma_RES.value.m_as('dimensionless')
        true_e_scale = params.Genie_Ma_RES.value.m_as('dimensionless')
        nue_flux_norm = 1.
        numu_flux_norm = 1.

        #Combine all information to calculate new weights #TODO (Tom) Move to Shivesh's style here
        for fig in self._data.iterkeys():

            data_array = self._data_arrays[fig]['device'] if self.use_gpu else self._data_arrays[fig]['host'] #Choose data based on CPU vs GPU selection
            n_evts = np.uint32(self.get_num_events(fig))
            self.weight_calc.calc_weight(
                n_evts=n_evts,
                livetime=livetime,
                nue_flux_norm=nue_flux_norm,
                numu_flux_norm=numu_flux_norm,
                aeff_scale=aeff_scale,
                Genie_Ma_QE=Genie_Ma_QE,
                Genie_Ma_RES=Genie_Ma_RES,
                true_e_scale=true_e_scale, #TODO Why is this used by GPU case but not CPU? Overlap with flux/prob calc?
                **data_array
            )

        #Store the new hash
        self.weight_calc_hash = this_hash

        #Done. Let the calling function know a new calculation has been performed
        return True


    def reweight_muons(self) :

        # This function calculates the reweighting for atmospheric muons
        # Unlike 'reweight_neutrinos', no data array is used as part of this
        # calculation, as currently there is not GPU implementation 

        #Get the params subset for the muon reweighting calculation
        params = ParamSet( p for p in self.params if p.name in (list(self.atm_muon_params)+["livetime"]) )

        #
        # Check if anything to do
        #

        #Get current state of all inputs to this calculation
        this_hash = self._get_params_hash(params)

        #If nothing has changed, then there is nothing to do
        if self.muon_hash == this_hash:
            return False


        #
        # Calculate weights
        #

        #TODO (Tom) GPU implementation?

        #Start from sample weight
        weights = deepcopy(self._data.muons['sample_weight'])

        #Get params
        livetime = params.livetime.value.m_as('seconds')
        atm_muon_scale = params.atm_muon_scale.value.m_as('dimensionless')

        # Livetime reweighting
        weights *= livetime

        # Scaling
        weights *= atm_muon_scale

        # Primary CR systematic
        cr_rw_scale = params.delta_gamma_mu.value.m_as('dimensionless')
        rw_variable = str(params.delta_gamma_mu_variable.value)
        rw_array = self.muon_prim_unc_spline(self._data.muons[rw_variable])

        # Reweighting term is positive-only by construction, so normalise
        # it by shifting the whole array down by a normalisation factor
        norm = sum(rw_array)/len(rw_array)
        cr_rw_array = rw_array-norm
        weights *= (1+cr_rw_scale*cr_rw_array)

        #Set weights as dimensionless
        #weights.ito('dimensionless') #TODO (Tom) ?

        #Copy weights to events
        self._data.muons['pisa_weight'] = deepcopy(weights)

        #Store new hash
        self.muon_hash = this_hash

        return True


    def reweight_noise(self) :

        # This function calculates the reweighting for events triggered by noise
        # Unlike 'reweight_neutrinos', no data array is used as part of this
        # calculation, as currently there is not GPU implementation 

        #Get the params subset for the noise reweighting calculation
        params = ParamSet( p for p in self.params if p.name in (list(self.noise_params)+["livetime"]) )

        #
        # Check if anything to do
        #

        #Get current state of all inputs to this calculation
        this_hash = self._get_params_hash(params)

        #If nothing has changed, then there is nothing to do
        if self.noise_hash == this_hash:
            return False


        #
        # Calculate weights
        #

        #TODO (Tom) GPU implementation?

        #Start from sample weight
        weights = deepcopy(self._data.noise['sample_weight'])

        #Get params
        livetime = params.livetime.value.m_as('seconds')
        norm_noise = params.norm_noise.value.m_as('dimensionless')

        # Livetime reweighting
        weights *= livetime

        # Nornalisation
        weights *= norm_noise

        #Set weights as dimensionless
        #weights.ito('dimensionless') #TODO?

        #Copy weights to events
        self._data.noise['pisa_weight'] = deepcopy(weights)

        #Store new hash
        self.noise_hash = this_hash

        return True


    def _populate_nu_data_arrays(self) :

        # This function fills arrays with the data required for all the reweighting calculations
        # This includes input data required from the events, as well as placeholders for output
        # data that will be filled
        # 
        # All the data is stored as an array per vairable with one entry per event
        #
        # The 'host' data represents the data as stored on the CPU
        # If running on a GPU, there will also be 'device' data which is the data on the GPU
        #
        # The data is also divided into inputs (which are never modified) and outputs (which
        # are modified during the calculation)
        #
        # Data will be copied from host to device before a calculation, then back to the host 
        # after
        #
        # The array is populated only once at the start of calculation for efficiency reasons,
        # but the outputs are overwritten at each calculation

        #
        # Determine which variables are required
        #

        #Oscillation variables
        self.osc_input_variables = [
                'true_energy',
                'true_coszen',
            ]
        self.osc_output_variables = [
                'prob_e',
                'prob_mu',
            ]

        #Flux variables
        self.flux_input_variables = [
                'true_energy',
                'true_coszen',
                'neutrino_nue_flux',
                'neutrino_numu_flux',
                'neutrino_oppo_nue_flux',
                'neutrino_oppo_numu_flux',
            ]
        self.flux_output_variables = [
                'scaled_nue_flux',
                'scaled_numu_flux',
                'scaled_nue_flux_shape',
                'scaled_numu_flux_shape',
            ]

        #Final weight calculation variables
        self.weight_input_variables = [
                'weighted_aeff',
            ]
        self.weight_output_variables = [
                'weight',
            ]

        #Also define some optional input variables
        self.optional_input_variables = [
                'linear_fit_MaCCQE',
                'quad_fit_MaCCQE',
                'linear_fit_MaCCRES',
                'quad_fit_MaCCRES',
            ]

        #Gather all variables required by the various re-weighting functions
        self.input_variables = self.osc_input_variables + self.flux_input_variables + self.weight_input_variables
        self.output_variables = self.osc_output_variables + self.flux_output_variables + self.weight_output_variables
        #TODO Check no overlap...
        #TODO reco variables??

        #Add the output binning variables
        self.input_variables = self.input_variables + self.output_binning.names

        #Trim duplicates (note, order is not preserved by set)
        self.input_variables = list( set(self.input_variables) )
        self.output_variables = list( set(self.output_variables) )


        #
        # Fill the host arrays
        #

        self._data_arrays = {}

        #Loop over neutrino flavor/interaction combinations
        for i_fig,fig in enumerate(self._data.keys()) :

            #Create arrays to fill
            self._data_arrays[fig] = {}
            host_array = self._data_arrays[fig]['host'] = {}

            #Get number of events (this is just the length of any of the variables)
            n_evts = self.get_num_events(fig)

            #Copy input variables from events to the data arrays
            for var in self.input_variables :
                host_array[var] = self._data[fig][var].astype(FTYPE) #'astype' returns a copy

            #Add placeholders on host for output variables
            #Default value is '1.' for numu/nue oscillation probabilities (e.g. if don't oscillate this is the value),
            #or '0.' for everything else
            for var in self.output_variables :
                if var in self.osc_output_variables :
                    host_array[var] = np.ones(n_evts, dtype=FTYPE)
                else :
                    host_array[var] = np.zeros(n_evts, dtype=FTYPE)

            #For optional variables, either copy them if they exists or fill with 0 if not
            self.optional_input_variables
            for var in self.optional_input_variables :
                if var in self._data[fig] :
                    host_array[var] = self._data[fig][var].astype(FTYPE) #'astype' returns a copy
                else :
                    if i_fig == 0 : logging.warning("Optional event input variable '%s' not found, values set to 0."%var) #Warn the first time
                    host_array[var] = np.zeros(n_evts, dtype=FTYPE)

            # Calulate the layers of the Earth (every particle crosses a number of layers in the
            # earth with different densities, and for a given length these depend only on the earth
            # model (PREM) and the true coszen of an event. Therefore we can calculate these for 
            # once and are done
            if not self.decoherence :
                nlayers, density, dist = self.osc.calc_path_layers(host_array['true_coszen'])
                if nlayers is not None : #TODO Just store anyway?
                    host_array['numLayers'] = nlayers
                    host_array['densityInLayer'] = density
                    host_array['distanceInLayer'] = dist


        #
        # Fill the device arrays
        #

        #If using GPU, copy the data arrays across to it #TODO (Tom) be selective if only some re-weighting done on GPU???
        if self.use_gpu : 
            import pycuda.driver as cuda
            start_t = time.time()
            for fig in self._data_arrays.keys() :
                    self._data_arrays[fig]['device'] = {}
                    for var, val in self._data_arrays[fig]['host'].items():
                        self._data_arrays[fig]['device'][var] = cuda.mem_alloc(val.nbytes)
                        cuda.memcpy_htod( self._data_arrays[fig]['device'][var], val )
            end_t = time.time()
            logging.debug('Copying data arrays to the GPU device took %.4f ms'%((end_t - start_t) * 1000))

        #
        # Apply raw reco sys
        #

        #TODO (Tom) Is this functionality used in event-by-event mode???



    def update_device_arrays(self,fig,variable=None) :
        #Copy data from 'host' array to 'device' array
        #Can provided a specific variable, or copy all 
        import pycuda.driver as cuda
        variables = self._data_arrays[fig]['device'].keys() if variable is None else [variable]
        for var in variables :
            self._data_arrays[fig]['device'][var].free()
            self._data_arrays[fig]['device'][var] = cuda.mem_alloc(self._data_arrays[fig]['host'][var].nbytes)
            cuda.memcpy_htod( self._data_arrays[fig]['device'][var], self._data_arrays[fig]['host'][var] )


    def update_host_arrays(self,fig,variable=None) :
        #Copy data from 'device' array to 'hist' array
        #Can provided a specific variable, or copy all
        import pycuda.driver as cuda
        variables = self._data_arrays[fig]['device'].keys() if variable is None else [variable]
        for var in variables :
            buff = np.full( self.get_num_events(fig), fill_value=np.nan, dtype=FTYPE )
            cuda.memcpy_dtoh( buff, self._data_arrays[fig]['device'][var] )
            assert np.all(np.isfinite(buff))
            self._data_arrays[fig]['host'][var] = buff



    def get_output_maps(self) :

        output_maps = []

        #TODO (Tom) Handling of the error hsit could be nicer...

        #
        # Neutrinos
        #

        if self.neutrinos :


            #
            # Fill neutrino hists
            #

            #Using data from the data arrays used during the calculations, as this avoids uncessecary data 
            #copies between the device and host (e.g. if using a GPU, do everything on the GPU in series with 
            #no copying to/from the CPU)
            
            #Check error hist type
            sumw2_errors = self.error_method is not None and "sumw2" in self.error_method.lower()

            nu_hists = {}
            if sumw2_errors : nu_sumw2 = {}
            if self.kde_hist : nu_err_hists = {}

            #The weight variable in the data arrays is named 'weight'
            weight_var = 'weight'

            #Loop over flavor/interaction combinations
            for fig in self._data_arrays.keys() :

                #Fill histogram based using the method quested by the user
                #Make sure to pass the correct data array (e.g. either the one on the CPU or the GPU)
                #Using data from the data arrays used during the calculations #TODO (Tom) Use event data (weight is written back)?
                if self.kde_hist:
                    hist,err_hist = self.fill_kde_hist(self._data_arrays[fig]["host"],weight_var=weight_var)
                    nu_err_hists[fig] = err_hist
                else :
                    if self.use_gpu:
                        hist,sumw2 = self.fill_hist_using_gpu(self._data_arrays[fig]["device"],n_events=self.get_num_events(fig),weight_var=weight_var)
                    else :
                        hist,sumw2 = self.fill_hist_using_cpu(self._data_arrays[fig]["host"],weight_var=weight_var)
                    if sumw2_errors : nu_sumw2[fig] = sumw2
                nu_hists[fig] = hist


            #
            # Apply normalisations
            #

            #Loop over hists
            for fig in self._data_arrays.keys() :

                scale = 1.0

                #Apply nutau normalisation params #TODO Instead apply nornalisation to event weights??
                if fig in ['nutau_cc', 'nutaubar_cc']:
                    scale *= self.params.nutau_cc_norm.value.m_as('dimensionless')
                if 'nutau' in fig:
                    scale *= self.params.nutau_norm.value.m_as('dimensionless') #TODO (Tom) Is this also being passed to prob3??? Is this then applied twice?

                #Apply NC norm
                if '_nc' in fig:
                    scale *= self.params.nu_nc_norm.value.m_as('dimensionless')

                #Scale the histogram according to the total norm
                nu_hists[fig] *= scale
                if sumw2_errors : nu_sumw2[fig] *= (scale*scale)


            #
            # Write maps
            #

            for fig in self._data_arrays.keys() :

                #Check if this is a quested output
                if fig in self.output_names : #TODO Handle special case strings like "all_nc", etc, but in a less hacky way than it was the original code

                    #Get the error histogram
                    if sumw2_errors : err_hist = np.sqrt(nu_sumw2[fig])
                    elif self.kde_hist : err_hist = nu_err_hists[fig]
                    else : err_hist = None

                    #Create the map and add it to the outputs list
                    output_maps.append(
                        Map(name=fig,
                            hist=nu_hists[fig],
                            error_hist=err_hist,
                            binning=self.output_binning,
                            tex=text2tex(fig)
                        )
                    )


        #
        # Muons
        #

        if self.muons :

            #For muons, using data directly from the events, as don't use dedicated data
            #arrays as currently no GPU implementation
            #Note that the calculated weights are written to the events in the 'pisa_weight' 
            #variable
            weight_var='pisa_weight'

            #Use the specified histgramming method (and error method) with muon data
            if self.kde_hist:
                hist,err_hist = self.fill_kde_hist(self._data['muons'],weight_var=weight_var)
            else :
                ''' #TODO (Tom) muons are currently entirely handled on CPUs (even hists, as no data arrays are populated), but should add GPU capability in future (including more generic histogramming tools, possibly using numba)
                if self.use_gpu:
                    hist,sumw2 = self.fill_hist_using_gpu(self._data['muons'],n_events=len(self._data['muons']),weight_var=weight_var)
                else :
                    hist,sumw2 = self.fill_hist_using_cpu(self._data['muons'],weight_var=weight_var)
                '''
                hist,sumw2 = self.fill_hist_using_cpu(self._data['muons'],weight_var=weight_var)
                err_hist = np.sqrt(sumw2) if sumw2_errors else None
            #Store as a map
            output_maps.append(
                Map(name='muons',
                    hist=hist,
                    error_hist=err_hist,
                    binning=self.output_binning,
                    tex=text2tex('muons')
                )
            )

        #
        # Noise
        #

        if self.noise :

            #TODO functionalise histogramming and map creation further so can avoid code reproduction with e.g. muons above?

            #For noise, using data directly from the events, as don't use dedicated data
            #arrays as currently no GPU implementation
            #Note that the calculated weights are written to the events in the 'pisa_weight' 
            #variable
            weight_var='pisa_weight'

            #Use the specified histgramming method (and error method) with muon data
            if self.kde_hist:
                hist,err_hist = self.fill_kde_hist(self._data['noise'],weight_var=weight_var)
            else :
                '''
                if self.use_gpu: #TODO (Tom) noise is currently entirely handled on CPUs (even hists, as no data arrays are populated), but should add GPU capability in future (including more generic histogramming tools, possibly using numba)
                    hist,sumw2 = self.fill_hist_using_gpu(self._data['noise'],n_events=len(self._data['noise']),weight_var=weight_var)
                else :
                    hist,sumw2 = self.fill_hist_using_cpu(self._data['noise'],weight_var=weight_var)
                '''
                hist,sumw2 = self.fill_hist_using_cpu(self._data['noise'],weight_var=weight_var)
                err_hist = np.sqrt(sumw2) if sumw2_errors else None

            #Store as a map
            output_maps.append(
                Map(name='noise',
                    hist=hist,
                    error_hist=err_hist,
                    binning=self.output_binning,
                    tex=text2tex('noise')
                )
            )

        #
        # Done
        #

        #Return all the maps produced as a single mapset
        return MapSet(maps=output_maps, name=self._data.metadata['name'])



    def fill_kde_hist(self,data_array,weight_var="weight") :

        #TODO Hist E and PID scaling

        #Grab the data for each histogram dimension form the data array
        sample = []
        for bin_name in self.output_binning.names :
            if bin_name not in data_array :
                raise AssertionError("Error filling KDE histogram : Output dimension '%s' not found in data array, instead found %s" % (bin_name,data_array.keys())  )
            sample.append(data_array[bin_name])

        kde_hist = self.kde_histogramdd(
            sample=np.array(sample).T,
            binning=self.output_binning,
            weights=data_array[weight_var],
            coszen_name=self.kde_coszen_dim,
            use_cuda=False,
            bw_method='silverman',
            alpha=0.8,
            oversample=10,
            coszen_reflection=0.5,
            adaptive=True
        )

        err_hist = np.sqrt(kde_hist)

        return hist,err_hist


    def fill_hist_using_gpu(self,data_array,n_events,weight_var="weight") :

        start_t = time.time()

        # Update bin edges according to the current scaling params
        self.gpu_histogrammer.update_bin_edges(*self.get_scaled_bin_edges())

        # Grab the bin names
        bin_names = self.output_binning.names

        sumw2 = None

        n_events = np.int32(n_events)

        # Handle 2D vs 3D cases
        if len(bin_names) == 2 :

                hist = self.gpu_histogrammer.get_hist(
                    n_events = n_events,
                    d_x = data_array[bin_names[0]],
                    d_y = data_array[bin_names[1]],
                    d_w = data_array[weight_var]
                )

                if self.error_method in ['sumw2', 'fixed_sumw2']:
                    sumw2 = self.gpu_histogrammer.get_hist(
                        n_events = n_events,
                        d_x=data_array[bin_names[0]],
                        d_y=data_array[bin_names[1]],
                        d_w=data_array[weight_var]
                    )

        elif len(bin_names) == 3 :

                hist = self.gpu_histogrammer.get_hist(
                    n_events = n_events,
                    d_x=data_array[bin_names[0]],
                    d_y=data_array[bin_names[1]],
                    d_z=data_array[bin_names[2]],
                    d_w=data_array[weight_var]
                )

                if self.error_method in ['sumw2', 'fixed_sumw2']:
                    sumw2 = self.gpu_histogrammer.get_hist(
                        n_events = n_events,
                        d_x=data_array[bin_names[0]],
                        d_y=data_array[bin_names[1]],
                        d_z=data_array[bin_names[2]],
                        d_w=data_array['sumw2']
                    )

        else :
            raise AssertionError( "GPU histogramming tools only support 2D or 3D histograms, found %i dimensions %s" % (len(bin_names),bin_names) )

        end_t = time.time()

        logging.debug( 'GPU hist done in %.4f ms' % ((end_t-start_t)*1000.) )

        return hist,sumw2


    def fill_hist_using_cpu(self,data_array,weight_var="weight") :

        #Histogram on CPU case (use numpy N-dim histogram) 
        #TOD Use Shivesh's function Data.histogram???

        #Grab the data for each histogram dimension form the data array
        sample = []
        for bin_name in self.output_binning.names :
            if bin_name not in data_array :
                raise AssertionError("Error filling histogram : Output dimension '%s' not found in data array, instead found %s" % (bin_name,data_array.keys())  )
            sample.append(data_array[bin_name])

        #Grab the bin edges for plotting
        bin_edges = self.get_scaled_bin_edges()

        #Fill a histogram from these event data
        hist,_ = np.histogramdd(
            sample=np.array(sample).T,
            bins=bin_edges,
            weights=data_array[weight_var]
        )

        #Also return the errors for the hitogram (stored as a separate histogram)
        if self.error_method in ['sumw2', 'fixed_sumw2']:
            sumw2,_ = np.histogramdd(
                sample=np.array(sample).T,
                bins=bin_edges,
                weights=data_array['sumw2']
            )
        else :
            sumw2 = None

        return hist,sumw2


    def get_num_events(self,fig) :
        #Get the number of events for this fig (use the length of the array for any variable, as each is filled once per event)
        return len(self._data[fig]["true_energy"])


    def get_scaled_bin_edges(self) :

        # Get the output binning for this stage, apply any scalings defined in the 
        # params, and return the bin edges for use in external histogramming tools
        # This also takes care of units, and returns striped down information 
        # relative to the overall binning class (e.g. just lists of edge magnitudes)

        scaled_bin_edges = []

        #Loop over binning dimensions
        for dim in self.output_binning.dimensions :

            #Get the edges for this dimension
            bin_edges = deepcopy(dim.bin_edges)

            #Get dimension in appropriates units, and strip off the units to give bare magnitude
            if 'energy' in dim.name.lower() :
                bin_edges = bin_edges.to('GeV').magnitude.astype(FTYPE) #Enforce GeV as our base unit
            else :
                bin_edges = bin_edges.magnitude.astype(FTYPE)

            #Perform scaling
            if 'energy' in dim.name.lower() :
                bin_edges *= FTYPE(self.params.hist_e_scale.value.m_as('dimensionless'))
            elif 'pid' in dim.name.lower() :
                bin_edges *= FTYPE(self.params.hist_pid_scale.value.m_as('dimensionless'))

            scaled_bin_edges.append(bin_edges)

        #Check got something
        assert len(scaled_bin_edges) > 0

        #Return the edges (note that the order of the dimensins is preserved)
        return scaled_bin_edges
        #return tuple(scaled_bin_edges)


    @staticmethod
    def apply_ratio_scale(flux_a, flux_b, ratio_scale):
        """Apply a ratio systematic to the flux weights."""
        orig_ratio = flux_a / flux_b
        orig_sum = flux_a + flux_b

        scaled_b = orig_sum / (1 + ratio_scale*orig_ratio)
        scaled_a = ratio_scale*orig_ratio * scaled_b
        return scaled_a, scaled_b


    def make_muon_prim_unc_spline(self):
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

        #Define the params for which 'None' is a valid value
        params_that_can_be_none = ["earth_model","YeI","YeO","YeM"]

        #Define th expected types for all params
        pq = ureg.Quantity
        param_types = [
            ('livetime', pq),
            ('aeff_scale', pq),
            ('hist_e_scale', pq),
            ('hist_pid_scale', pq),
            ('nutau_cc_norm',pq),
            ('nu_nc_norm',pq),
            ('reco_e_res_raw',pq),
            ('reco_e_scale_raw',pq),
            ('reco_cz_res_raw',pq),
        ]
        if self.neutrinos:
            param_types.extend([
                ('nu_diff_DIS', pq),
                ('nu_diff_norm', pq),
                ('nubar_diff_DIS', pq),
                ('nubar_diff_norm', pq),
                ('hadron_DIS', pq),
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
            if self.decoherence :
                param_types.extend([
                    ('gamma21', pq),
                    ('gamma31', pq),
                    ('gamma32', pq),
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
        param_types = dict(param_types)

        #Loop over params to check each one
        for p in params :

            #Check for None params
            if p.value is None :
                if p.name not in params_that_can_be_none :
                    raise TypeError('Param "%s" value is \'None\'')    
                continue

            #Check param types
            if p.name in param_types :
                if not isinstance(p.value,param_types[p.name]):
                    raise TypeError('Param "%s" has type \'%s\', but expected type \'%s\'' % (type(p.val),param_types[p.name]) )    

        #Code assumes that detector location and Earth model aren't changing
        #(this assumption is baked into the fact that we pre-compute layers for a given set of zenith angles)
        for pname in ["prop_height","detector_depth","earth_model"] : #TODO (Tom) same in prob3.py validate_params
            if not params[pname].is_fixed :
                raise TypeError("Param '%s' must be fixed" % pname)


    def _get_params_hash(self,params) :
        #Create a hash of the param values, as well as the sample hash
        #TODO (Tom) should params.normalize values be set to True such that "nearly identical" values are treated as the same in the hash?
        return hash_obj( [self.sample_hash, self.params.values_hash], full_hash = self.full_hash) #TODO (Tom) weight hash? maybe not, as this is an output, not an input
