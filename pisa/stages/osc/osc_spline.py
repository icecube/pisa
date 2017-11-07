# Create a spline of oscillation probabilities in [E,coszen] space
# Can use instead of event-by-event calculation for significant speed up
# Tom Stuttard (6th Nov 2017)


from __future__ import division

import os, sys, copy

import numpy as np

from scipy.interpolate import RectBivariateSpline

from pisa import FTYPE, C_FTYPE, C_PRECISION_DEF
from pisa.core.binning import MultiDimBinning
from pisa.stages.osc.prob3_new import ArrayVariable,prob3base,prob3wrapper
from pisa.utils.flavInt import ALL_NUFLAVINTS, NuFlav

__all__ = ['OscSpline']


class OscSpline :

    def __init__(self,binning,osc_calculator,flavors=None,use_gpu=False) :

        #Store args
        self.binning = binning
        self.osc_calculator = osc_calculator
        self.use_gpu = use_gpu
        self.flavors = flavors

        #Use default flavors if none provided
        if self.flavors is None :
          self.flavors = ALL_NUFLAVINTS.flavs

        #TODO Need to implement GPU side of things still
        if self.use_gpu :
          raise Exception("Splining oscillation probabilities not yet implemented for GPUs")

        #Check the provided oscillation calculator
        #TODO Add option not to provide an osc calculator, in which case on must manually supply the prob_e/mu values (needs a function)
        self.using_prob3 = False
        if issubclass(type(self.osc_calculator),prob3base) or isinstance(self.osc_calculator,prob3wrapper) :
          self.using_prob3 = True #TODO Need to consistently handle GPU usage here...
        else :
          raise Exception("Unknown oscillation calculator provided : %s" % type(self.osc_calculator) )


        #Initialise
        self._init_splines()


    def _init_splines(self) :

        #Initialise everything we are going to need to generate oscillation prbability splines

        #TODO enforce energy bins > 0 (get NaN for E=0)
        #TODO enforce grid dims > spline k

        #
        # Check inputs
        #

        #Check the binning
        for dim_name in ["true_energy","true_coszen"] :
          if dim_name not in self.binning.names :
            raise Exception("Oscillation spline binning must include '%s' dimension" % dim_name)


        #
        # Create arrays
        #

        #Get [E,coszen] grid
        true_energy_grid, true_coszen_grid = np.meshgrid(self.binning["true_energy"].bin_edges.astype(FTYPE),
                                                        self.binning["true_coszen"].bin_edges.astype(FTYPE),
                                                        indexing='ij')

        #Flatten the grid (prob3 GPU code needs 1D arrays), and store in array class #TODO Think about performance
        self.energy_grid_unscaled = ArrayVariable( host_array=true_energy_grid.flatten().astype(FTYPE) )
        self.coszen_grid = ArrayVariable( host_array=true_coszen_grid.flatten().astype(FTYPE) )

        #Will need a scale energy grid too later
        self.energy_grid_scaled = ArrayVariable( host_array=copy.deepcopy(self.energy_grid_unscaled.host_array) )

        #Create empty probability grids to fill (default to NaN so get errors if not filled correctly)
        self.prob_e_grid = ArrayVariable( host_array=np.full( self.coszen_grid.shape, np.NaN, dtype=FTYPE ) )
        self.prob_mu_grid = ArrayVariable( host_array=np.full( self.coszen_grid.shape, np.NaN, dtype=FTYPE ) )

        #For GPU case, need to copy arrays to device
        if self.use_gpu :
          self.energy_grid_unscaled.to_device()
          self.energy_grid_scaled.to_device()
          self.coszen_grid.to_device()
          self.prob_e_grid.to_device()
          self.prob_mu_grid.to_device()


        #
        # Init spline containers
        #

        #Create containers for splines that we are about to generate
        self.prob_e_splines = dict([ (flav,None) for flav in self.flavors ])
        self.prob_mu_splines = dict([ (flav,None) for flav in self.flavors ])


        #
        # Prob3-specific pre-computing
        #

        if self.using_prob3 :

          #Calculate the layers for the coszen values and store in array
          #Only required if using the GPU version of prob3, and obviously not at all if only 
          #considering vacuum oscillation
          if self.osc_calculator.earth_model is not None and self.use_gpu :

              #Perform the calculations
              num_layers,density_in_layer,distance_in_layer = self.osc_calculator.calc_path_layers(self.coszen_grid.host_array)
              self.num_layers = ArrayVariable( host_array=num_layers.astype(FTYPE) )
              self.density_in_layer = ArrayVariable( host_array=density_in_layer.astype(FTYPE) )
              self.distance_in_layer = ArrayVariable( host_array=distance_in_layer.astype(FTYPE) )

              #Write arrays to device
              self.num_layers.to_device()
              self.density_in_layer.to_device()
              self.distance_in_layer.to_device()


    def generate_splines(self,true_e_scale) :

        #This function is used to generate the splines
        #It must be recalled each time the physics parameters in the underlying osc_calculator are changed

        #TODO (Tom) Add logic in this class to determine whether splines need regenerating? Currently is the resonsiblity of the calling code (such as weight.py)

        #
        # Scale true energy (if required)
        #

        #TODO Could we just handle scaling in the splien evaluation?

        #TODO (Tom) Handle true_e_scale
        if not np.isclose(true_e_scale,1.) : raise NotImplementedError("generate_splines cannot currently handle true_e_scale != 1., this needs implementing")

        #Scale energy parameters
        #TODO (Tom) test this
        self.energy_grid_scaled.set_host_array( true_e_scale * self.energy_grid_scaled.host_array )
        if self.use_gpu : self.energy_grid_scaled.to_device()
        scaled_true_e_bins = true_e_scale * self.binning["true_energy"].bin_edges


        #
        # Generate splines
        #

        #Loop over flavors
        for flav in self.flavors :

            #
            # Calculate probabilities (prob3 case)
            #

            if self.using_prob3 :

                #Get the prob3 flavor and nu/nubar codes for this neutrino flavor
                kFlav,kNuBar = flav.prob3_codes

                #Handle args only required in certain modes
                extra_args = {}
                if self.use_gpu :
                  extra_args["numLayers"] = self.num_layers.device_array
                  extra_args["densityInLayer"] = self.density_in_layer.device_array
                  extra_args["distanceInLayer"] = self.distance_in_layer.device_array

                #Calculate probabilites for [E,coszen] grid
                self.osc_calculator.calc_probs( 
                    kNuBar=kNuBar, 
                    kFlav=kFlav, 
                    n_evts=np.uint32( len(self.energy_grid_scaled) ),
                    true_e_scale=true_e_scale, #TODO (Tom) Is this being applied twice?
                    true_energy=self.energy_grid_scaled.get(device=self.use_gpu), 
                    true_coszen=self.coszen_grid.get(device=self.use_gpu), #THis is only used in GPU mode 
                    prob_e=self.prob_e_grid.get(device=self.use_gpu), 
                    prob_mu=self.prob_mu_grid.get(device=self.use_gpu),
                    **extra_args
                )

                #If using a GPU, need to copy the results back to the host
                if self.use_gpu :
                  self.prob_e_grid.from_device()
                  self.prob_mu_grid.from_device()


            #
            # Create spline
            #

            #Spline it for smoothing (note the need to reshape the probability values to a 2D array)
            shape = (len(scaled_true_e_bins),len(self.binning["true_coszen"].bin_edges))
            self.prob_e_splines[flav] = RectBivariateSpline( scaled_true_e_bins, self.binning["true_coszen"].bin_edges, self.prob_e_grid.host_array.reshape(shape) )
            self.prob_mu_splines[flav] = RectBivariateSpline( scaled_true_e_bins, self.binning["true_coszen"].bin_edges, self.prob_mu_grid.host_array.reshape(shape) )



    def eval(self, prob_e, prob_mu, flav, true_e_scale, true_energy, true_coszen, **kw) :

        #Evaluate the spline fo the given E,coszen values
        #Using kwargs for arg hiding (makes it easier to call smae function but for different cases

        #TODO Make work with GPUs

        #Check spline has been produced
        if self.prob_e_splines[flav] is None :
            raise Exception( "Cannot calculate probabilty using spline for '%s' : Spline has not been generated yet" % str(flav) )

        #Extract probabilities for these events from the spline
        np.copyto( prob_e, self.prob_e_splines[flav].ev(true_e_scale*true_energy,true_coszen) ) #Use copyto to fill np array from another (using '=' sets the pointer reference without changing the underlying object)
        np.copyto( prob_mu, self.prob_mu_splines[flav].ev(true_e_scale*true_energy,true_coszen) ) 


