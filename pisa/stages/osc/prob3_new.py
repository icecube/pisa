# authors: T.Arlen, J.Lanfranchi, P.Eller, T.Stuttard
# date:   Oct 23, 2017


from __future__ import division

import os, sys, copy

import numpy as np

from pisa import FTYPE, C_FTYPE, C_PRECISION_DEF
from pisa.core.binning import MultiDimBinning
from pisa.core.param import ParamSet, ParamSelector
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.resources import find_resource
from pisa.stages.osc.prob3cc.BargerPropagator import BargerPropagator
from pisa.utils.comparisons import normQuant
from pisa.utils.profiler import profile
from pisa.utils.log import logging
from pisa.stages.osc.layers import Layers
from pisa.stages.osc.osc_params import OscParams


__all__ = ['prob3','prob3cpu','prob3gpu','prob3wrapper','ArrayVariable']


SIGFIGS = 12 
"""Significant figures for determining if numbers and quantities normalised
(using pisa.utils.comparisons.normQuant) are equal. Make sure this is less than
the numerical precision that calculations are being performed in to have the
desired effect that "essentially equal" things evaluate to be equal."""








#TODO Turn into usage instructions
'''
#GPU case

osc = prob3wrapper()

osc.set_params()
num_layers,density_in_layer,distance_in_layer = osc.calc_path_layers(coszen_array)

osc.calc_probs(prob_e,prob_mu,flav,nubar,energy_array,num_layers,density_in_layer,distance_in_layer)


#CPU case

osc = prob3wrapper()

osc.set_params()

osc.calc_probs(prob_e,prob_mu,flav,nubar,energy_array,coszen_array)


#GPU + spline

osc = prob3wrapper()

osc.set_params()

grid = ()

osc_spline = prob3spline(prob3=osc,grid=grid)
'''



#
# Array variable class
#


#TODO Description, own class
class ArrayVariable :

  #TODO WOuld be nice to handle units

  def __init__(self,host_array=None,to_device=False) :
    self._host_array = host_array
    self._device_array = None
    if to_device :
        if self._host_array is None :
            raise Exception("Cannot use 'to_device' constructor arg if no host array provided")
        else :
            self.to_device()

  @property
  def host_array(self) :
    if self._host_array is None :
      raise Exception("Cannot get host array, has not yet been set")
    return self._host_array

  @property
  def device_array(self) :
    if self._device_array is None :
      raise Exception("Cannot get device array, has not yet been set")
    return self._device_array

  def set_host_array(self,new_array) :
    if self._host_array is not None :
      if new_array.shape != self.shape :
        raise Exception("Cannot set host array, new array has different shape")
      if new_array.dtype != self.dtype : #TODO (Tom) handle casting safety...
        self._host_array = new_array.astype(self.dtype)
      else :
        self._host_array = new_array
    else :
      self._host_array = new_array

  def to_device(self) :
    import pycuda.driver as cuda
    if self._device_array is None :
      self._device_array = cuda.mem_alloc(self._host_array.nbytes)
    cuda.memcpy_htod(self._device_array,self._host_array)

  def from_device(self) :
    #TODO (Tom) Length check?
    import pycuda.driver as cuda
    if self._device_array is None :
      raise Exception("Cannot copy from device array, the device array has not yet been created")
    if self._host_array is None :
      raise Exception("Cannot copy from device array, there is no host array to copy to")
    buff = np.full( len(self), fill_value=np.nan, dtype=self.dtype )
    cuda.memcpy_dtoh(buff,self._device_array)
    self._host_array = buff

  #This is a useful way of getting either the host or device array beased on a flag
  def get(self,device) :
    if device : return self.device_array
    else : return self.host_array

  def __len__(self) :
    if self._host_array is None :
      raise Exception("Cannot get length, host array has not yet been set")
    return len(self._host_array)

  @property
  def shape(self) :
    if self._host_array is None :
      raise Exception("Cannot get shape, host array has not yet been set")
    return self._host_array.shape

  @property
  def dtype(self) :
    if self._host_array is None :
      raise Exception("Cannot get dtype, host array has not yet been set")
    return self._host_array.dtype

  def free(self) :
    if self._device_array is not None :
      self._device_array.free()





#
# Base class for a prob3 calculator
#


class prob3base(object) :

    def __init__(self, earth_model, detector_depth, prop_height, 
                      YeI=None, YeO=None, YeM=None) :

        #Store args
        self.earth_model = earth_model
        self.detector_depth = detector_depth
        self.prop_height = prop_height
        self.earth_model = earth_model
        self.YeI = YeI
        self.YeO = YeO
        self.YeM = YeM

        #Check matter inputs
        if self.earth_model :
          if self.YeI is None : raise Exception("Must provide 'YeI' when using an Earth model")
          if self.YeO is None : raise Exception("Must provide 'YeO' when using an Earth model")
          if self.YeM is None : raise Exception("Must provide 'YeM' when using an Earth model")

        #Flag indicating whether oscillation params have been set
        self.params_are_set = False

        #Flag indicating whether layers have been calculated
        self.layers_are_computed = False


    def set_params(self, theta12, theta13, theta23,
                   deltam21, deltam31, deltacp):

        #
        # Set the params
        #

        self.params_are_set = True

        self.theta12 = theta12
        self.theta13 = theta13
        self.theta23 = theta23
        self.deltam21 = deltam21
        self.deltam31 = deltam31
        self.deltacp = deltacp



    #Calculate path through sphere to detector as function of zenith angle
    #Returns the path length
    def calc_path_length(self, coszen) :

        """
        Calculates the path through a spherical body of radius rdetector for
        a neutrino coming in with at coszen from prop_height to a detector
        at depth.
        """

        prop_height = self.prop_height.m_as('km')
        depth = self.detector_depth.m_as('km')
        rdetector = 6371.0 - depth

        if coszen < 0:
            pathlength = np.sqrt(
                (rdetector + prop_height + depth) * \
                (rdetector + prop_height + depth) - \
                (rdetector*rdetector)*(1 - coszen*coszen)
            ) - rdetector*coszen
        else:
            kappa = (depth + prop_height)/rdetector
            pathlength = rdetector * np.sqrt(
                coszen*coszen - 1 + (1 + kappa)*(1 + kappa)
            ) - rdetector*coszen

        return pathlength



    #Calculate path through Earth model to detector as function of zenith angle,
    #Returns infromation about the layers, densities, etc
    def calc_path_layers(self, coszen) :

        if self.earth_model is None :
          raise Exception("Cannot calculate path layers if no Earth model supplied")

        #Get params
        YeI = self.YeI.m_as('dimensionless')
        YeO = self.YeO.m_as('dimensionless')
        YeM = self.YeM.m_as('dimensionless')
        prop_height = self.prop_height.m_as('km')
        detector_depth = self.detector_depth.m_as('km')

        #Instantiate layers helper class and set parameters
        layers = Layers(self.earth_model, detector_depth, prop_height)
        layers.setElecFrac(YeI, YeO, YeM)

        #Calculate the layers for each coszen track
        layers.calcLayers(coszen)

        #Record the max layers
        #Note that max layers depends on earth model only (e.g. do not need to store a different value per flav-int)
        self.max_layers = self.layers.max_layers

        #Extract the layers information we actually need, taking care with types so can place nicely with the GPU arrays
        num_layers = self.layers.n_layers.astype(np.int32)
        density_in_layer = self.layers.density.astype(FTYPE)
        distance_in_layer = self.layers.distance.astype(FTYPE)

        self.layers_are_computed = True

        return num_layers, density_in_layer, distance_in_layer



#
# Prob3 calculator for CPUs
#

class prob3cpu(prob3base) :

    def __init__(self, earth_model, detector_depth, prop_height, 
                      YeI=None, YeO=None, YeM=None) :

        #Call base class contructor
        super(self.__class__, self).__init__(
            earth_model=earth_model, 
            detector_depth=detector_depth, 
            prop_height=prop_height, 
            YeI=YeI, 
            YeO=YeO, 
            YeM=YeM
          )


    def _setup_barger_propagator(self):

        #TODO (Tom) do we actually need these "_barger_*" versions of the params?

        # If already instantiated with same parameters, don't instantiate again  #TODO (Tom) Think about how to manage this logic, given that current we get these value in the class constructor
        if (hasattr(self, 'barger_propagator') 
                and hasattr(self, '_barger_earth_model')
                and hasattr(self, '_barger_detector_depth')
                and (normQuant(self._barger_detector_depth, sigfigs=SIGFIGS)
                     == normQuant(self.detector_depth.m_as('km'),
                                  sigfigs=SIGFIGS))
                and self.earth_model == self._barger_earth_model):
            return

        # Some private variables to keep track of the state of the barger
        # propagator that has been instantiated, so if it is requested to be
        # instantiated again with equivalent parameters, this step can be
        # skipped (see checks above).
        self._barger_detector_depth = self.detector_depth.m_as('km')
        self._barger_earth_model = self.earth_model

        # TODO: can we pass kwargs to swig-ed C++ code?
        if self._barger_earth_model is not None:
            earth_model = find_resource(self._barger_earth_model)
            self.barger_propagator = BargerPropagator(
                earth_model.encode('ascii'),
                self._barger_detector_depth
            )
        else:
            # Initialise with the 12 layer model that should be there. All
            # calculations will use the GetVacuumProb so what we define here
            # doesn't matter.
            self.barger_propagator = BargerPropagator(
                find_resource('osc/PREM_12layer.dat'),
                self._barger_detector_depth
            )
        self.barger_propagator.UseMassEigenstates(False)



    @property
    def deltamatm(self) : #Atmospheric mass splitting  #TODO -> OscParams?
        # Comment BargerPropagator.cc::SetMNS()
        # "For the inverted Hierarchy, adjust the input
        # by the solar mixing (should be positive)
        # to feed the core libraries the correct value of m32."
        #if mAtm < 0.0: mAtm -= deltam21;
        if not self.params_are_set :
          raise Exception("Cannot get deltamatm : User must provide params using 'set_params' method")
        if self.deltam31.magnitude < 0.0:
            return self.deltam31
        else:
            return self.deltam31 - self.deltam21


    def _get_vacuum_prob_maps(self) :

        """
        Calculate oscillation probabilities in the case of vacuum oscillations
        Here we use Prob3 but only because it has already implemented the 
        vacuum oscillations and so makes life easier.
        """

        # Set up oscillation parameters needed to initialise MNS matrix
        kSquared = True
        theta12 = self.params.theta12.m_as('rad')
        theta13 = self.params.theta13.m_as('rad')
        theta23 = self.params.theta23.m_as('rad')
        deltam21 = self.params.deltam21.m_as('eV**2')
        deltam31 = self.params.deltam31.m_as('eV**2')
        deltacp = self.params.deltacp.m_as('rad')
        prop_height = self.params.prop_height.m_as('km')
        sin2th12Sq = np.sin(theta12)**2
        sin2th13Sq = np.sin(theta13)**2
        sin2th23Sq = np.sin(theta23)**2
        deltamatm = self.deltamatm

        # Initialise objects to look over for neutrino and antineutrino flavours
        # 1 - nue, 2 - numu, 3 - nutau
        nuflavs = [1,2,3]
        nubarflavs = [-1,-2,-3]
        prob_list = []

        # Set up the distance to the detector. Radius of Earth is 6371km and
        # we then account for the depth of the detector in the Earth.
        depth = self.detector_depth.m_as('km')
        rdetector = 6371.0 - depth

        # Probability is separately calculated for each energy and zenith bin
        # center as well as every initial and final neutrno flavour.
        for e_cen in self.e_centers:
            for cz_cen in self.cz_centers:
                # Neutrinos are calculated for first
                kNuBar = 1
                for alpha in nuflavs:
                    for beta in nuflavs:
                        path = self.calc_path_length(
                            coszen=cz_cen,
                            rdetector=rdetector,
                            prop_height=prop_height,
                            depth=depth
                        )
                        self.barger_propagator.SetMNS(
                            sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,
                            deltamatm,deltacp,e_cen,kSquared,kNuBar
                        )
                        prob_list.append(
                            self.barger_propagator.GetVacuumProb(
                                alpha, beta, e_cen, path
                            )
                        )
                # Then antineutrinos. With this, the layout of this prob_list
                # matches the output of the matter oscillations calculation.
                kNuBar = -1
                for alpha in nubarflavs:
                    for beta in nubarflavs:
                        path = self.calc_path_length(
                            coszen=cz_cen,
                            rdetector=rdetector,
                            prop_height=prop_height,
                            depth=depth
                        )
                        self.barger_propagator.SetMNS(
                            sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,
                            deltamatm,deltacp,e_cen,kSquared,kNuBar
                        )
                        prob_list.append(
                            self.barger_propagator.GetVacuumProb(
                                alpha, beta, e_cen, path
                            )
                        )

        return prob_list


    def calc_probs(self, prob_e, prob_mu, kNuBar, kFlav, true_e_scale, true_energy, true_coszen) :

        """
        Calculate oscillation probabilities event-by-event
        Both vacuum oscillations and propagation through matter are handled
        Both event-by-event calculation and pre-computed splines can be used
        """

        #Check that params have been set
        if not self.params_are_set :
          raise Exception("Cannot calculate probabilities : User must provide params using 'set_params' method")

        #Check inputs
        if prob_e.shape != true_energy.shape :
          raise Exception("energy and prob_e arrays must be the same shape : %s != %s" % (true_energy.shape,prob_e.shape,) )
        if prob_mu.shape != true_energy.shape :
          raise Exception("energy and prob_mu arrays must be the same shape : %s != %s" % (true_energy.shape,prob_mu.shape,) )
        if true_coszen.shape != true_energy.shape :
          raise Exception("energy and coszen arrays must be the same shape : %s != %s" % (true_energy.shape,true_coszen.shape,) )

        #Init the propagator
        self._setup_barger_propagator()

        # Set up oscillation parameters needed to initialise MNS matrix
        kSquared = True
        theta12 = self.theta12.m_as('rad')
        theta13 = self.theta13.m_as('rad')
        theta23 = self.theta23.m_as('rad')
        deltam21 = self.deltam21.m_as('eV**2')
        deltamatm = self.deltamatm.m_as('eV**2')
        deltacp = self.deltacp.m_as('rad')
        sin2th12Sq = np.sin(theta12)**2
        sin2th13Sq = np.sin(theta13)**2
        sin2th23Sq = np.sin(theta23)**2

        #
        # Vacuum case
        #

        if self._barger_earth_model is None:

            logging.debug("Calculating vacuum oscillations")

            # Probability is separately calculated for each event
            for i, (en, cz) in enumerate(zip(true_energy,true_coszen)) :

                en *= true_e_scale

                pathlength = self.calc_path_length(coszen=cz)
            
                self.barger_propagator.SetMNS(
                    sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,
                    deltamatm,deltacp,en,kSquared,kNuBar
                )

                # kFlav is zero-start indexed (as the GPU Prob3 version wants it), whereas Prob3 CPU wants it from 1
                prob_e[i] = self.barger_propagator.GetVacuumProb(1, kFlav+1, en, pathlength)
                prob_mu[i] = self.barger_propagator.GetVacuumProb(2, kFlav+1, en, pathlength)


        #
        # Matter effects cases
        #

        else:

            logging.debug("Calculating matter oscillations")
            YeI = self.YeI.m_as('dimensionless')
            YeO = self.YeO.m_as('dimensionless')
            YeM = self.YeM.m_as('dimensionless')
            depth = self.detector_depth.m_as('km')
            prop_height = self.prop_height.m_as('km')

            # Probability is separately calculated for each event
            #for i, (en, cz) in enumerate(zip(true_energy,true_coszen)):
            for i in np.ndindex(true_energy.shape):

                en = true_energy[i] * true_e_scale
                cz = true_coszen[i]

                self.barger_propagator.SetMNS(
                    sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,
                    deltamatm,deltacp,en,kSquared,kNuBar
                )

                self.barger_propagator.DefinePath(
                    float(cz), prop_height, YeI, YeO, YeM
                )

                self.barger_propagator.propagate(kNuBar)
                prob_e[i] = self.barger_propagator.GetProb(0, kFlav)
                prob_mu[i] = self.barger_propagator.GetProb(1, kFlav)


        #TODO nutau_norm?



#
# Prob3 GPU calculator
#

class prob3gpu :

    # Define CUDA kernel
    KERNEL_TEMPLATE = '''//CUDA//
    #define %(C_PRECISION_DEF)s
    #define fType %(C_FTYPE)s

    #include "mosc.cu"
    #include "mosc3.cu"
    #include "cuda_utils.h"
    #include <stdio.h>

    /* If we use some kind of oversampling then we need the original
     * binning with nebins and nczbins. In the current version we use a
     * fine binning for the first stages and do not need any
     * oversampling.
     */
    __global__ void propagateGrid(fType* d_smooth_maps,
                                  fType d_dm[3][3], fType d_mix[3][3][2],
                                  const fType* const d_ecen_fine,
                                  const fType* const d_czcen_fine,
                                  const int nebins_fine, const int nczbins_fine,
                                  const int nebins, const int nczbins,
                                  const int maxLayers,
                                  const int* const d_numberOfLayers,
                                  const fType* const d_densityInLayer,
                                  const fType* const d_distanceInLayer) {
      const int2 thread_2D_pos = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
                                           blockIdx.y*blockDim.y + threadIdx.y);

      // ensure we don't access memory outside of bounds!
      if (thread_2D_pos.x >= nczbins_fine || thread_2D_pos.y >= nebins_fine)
        return;

      int eidx = thread_2D_pos.y;
      int czidx = thread_2D_pos.x;

      int kNuBar;
      //if (threadIdx.z == 0)
      //  kNuBar = 1;
      if (blockIdx.z == 0)
        kNuBar = 1;
      else
        kNuBar=-1;

      bool kUseMassEstates = false;

      fType TransitionMatrix[3][3][2];
      fType TransitionProduct[3][3][2];
      fType TransitionTemp[3][3][2];
      fType RawInputPsi[3][2];
      fType OutputPsi[3][2];
      fType Probability[3][3];

      clear_complex_matrix(TransitionMatrix);
      clear_complex_matrix(TransitionProduct);
      clear_complex_matrix(TransitionTemp);
      clear_probabilities(Probability);

      //int layers = 1;//*(d_numberOfLayers + czidx);
      int layers = d_numberOfLayers[czidx];

      fType energy = d_ecen_fine[eidx];
      //fType coszen = d_czcen_fine[czidx];
      for (int i=0; i<layers; i++) {
        //fType density = 0.5;//*(d_densityInLayer + czidx*maxLayers + i);
        fType density = d_densityInLayer[czidx*maxLayers + i];
        //fType distance = 100.;//*(d_distanceInLayer + czidx*maxLayers + i);
        fType distance = d_distanceInLayer[czidx*maxLayers + i];

        get_transition_matrix(kNuBar,
                              energy,
                              density,
                              distance,
                              TransitionMatrix,
                              0.0,
                              d_mix,
                              d_dm);

        if (i==0) {
          copy_complex_matrix(TransitionMatrix, TransitionProduct);
        } else {
          clear_complex_matrix(TransitionTemp);
          multiply_complex_matrix(TransitionMatrix, TransitionProduct, TransitionTemp);
          copy_complex_matrix(TransitionTemp, TransitionProduct);
        }
      } // end layer loop

      // loop on neutrino types, and compute probability for neutrino i:
      // We actually don't care about nutau -> anything since the flux there is zero!
      for (unsigned i=0; i<2; i++) {
        for (unsigned j = 0; j < 3; j++) {
          RawInputPsi[j][0] = 0.0;
          RawInputPsi[j][1] = 0.0;
        }

        if (kUseMassEstates)
          convert_from_mass_eigenstate(i+1, kNuBar, RawInputPsi, d_mix);
        else
          RawInputPsi[i][0] = 1.0;

        multiply_complex_matvec(TransitionProduct, RawInputPsi, OutputPsi);
        Probability[i][0] += OutputPsi[0][0]*OutputPsi[0][0] + OutputPsi[0][1]*OutputPsi[0][1];
        Probability[i][1] += OutputPsi[1][0]*OutputPsi[1][0] + OutputPsi[1][1]*OutputPsi[1][1];
        Probability[i][2] += OutputPsi[2][0]*OutputPsi[2][0] + OutputPsi[2][1]*OutputPsi[2][1];
      } // end of neutrino loop

      int efctr = nebins_fine/nebins;
      int czfctr = nczbins_fine/nczbins;
      int eidx_smooth = eidx/efctr;
      int czidx_smooth = czidx/czfctr;
      fType scale = fType(efctr*czfctr);
      for (int i=0;i<2;i++) {
        int iMap = 0;
        if (kNuBar == 1)
          iMap = i*3;
        else
          iMap = 6 + i*3;

        for (unsigned to_nu=0; to_nu<3; to_nu++) {
          int k = (iMap+to_nu);
          fType prob = Probability[i][to_nu];
          atomicAdd_custom((d_smooth_maps + k*nczbins*nebins +
              eidx_smooth*nczbins + czidx_smooth), prob/scale);
        }
      }
    }

    __global__ void propagateArray(fType* d_prob_e,
                                   fType* d_prob_mu,
                                   fType d_dm[3][3],
                                   fType d_mix[3][3][2],
                                   const int n_evts,
                                   const int kNuBar,
                                   const int kFlav,
                                   const int maxLayers,
                                   fType true_e_scale,
                                   const fType* const d_energy,
                                   const int* const d_numberOfLayers,
                                   const fType* const d_densityInLayer,
                                   const fType* const d_distanceInLayer)
    {
      const int idx = blockIdx.x*blockDim.x + threadIdx.x;
      // ensure we don't access memory outside of bounds!
      if(idx >= n_evts) return;
      bool kUseMassEstates = false;
      fType TransitionMatrix[3][3][2];
      fType TransitionProduct[3][3][2];
      fType TransitionTemp[3][3][2];
      fType RawInputPsi[3][2];
      fType OutputPsi[3][2];
      fType Probability[3][3];
      clear_complex_matrix( TransitionMatrix );
      clear_complex_matrix( TransitionProduct );
      clear_complex_matrix( TransitionTemp );
      clear_probabilities( Probability );
      int layers = *(d_numberOfLayers + idx);
      fType energy = d_energy[idx] * true_e_scale;
      for( int i=0; i<layers; i++) {
        fType density = *(d_densityInLayer + idx*maxLayers + i);
        fType distance = *(d_distanceInLayer + idx*maxLayers + i);
        get_transition_matrix(kNuBar,
                              energy,
                              density,
                              distance,
                              TransitionMatrix,
                              0.0,
                              d_mix,
                              d_dm);
        if(i==0) {
          copy_complex_matrix(TransitionMatrix, TransitionProduct);
        } else {
          clear_complex_matrix( TransitionTemp );
          multiply_complex_matrix( TransitionMatrix, TransitionProduct, TransitionTemp );
          copy_complex_matrix( TransitionTemp, TransitionProduct );
        }
      } // end layer loop

      // loop on neutrino types, and compute probability for neutrino i:
      // We actually don't care about nutau -> anything since the flux there is zero!
      for( unsigned i=0; i<2; i++) {
        for ( unsigned j = 0; j < 3; j++ ) {
          RawInputPsi[j][0] = 0.0;
          RawInputPsi[j][1] = 0.0;
        }

        if( kUseMassEstates )
          convert_from_mass_eigenstate(i+1, kNuBar, RawInputPsi, d_mix);
        else
          RawInputPsi[i][0] = 1.0;

        // calculate 'em all here, from legacy code...
        multiply_complex_matvec( TransitionProduct, RawInputPsi, OutputPsi );
        Probability[i][0] +=OutputPsi[0][0]*OutputPsi[0][0]+OutputPsi[0][1]*OutputPsi[0][1];
        Probability[i][1] +=OutputPsi[1][0]*OutputPsi[1][0]+OutputPsi[1][1]*OutputPsi[1][1];
        Probability[i][2] +=OutputPsi[2][0]*OutputPsi[2][0]+OutputPsi[2][1]*OutputPsi[2][1];
      }

      d_prob_e[idx] = Probability[0][kFlav];
      d_prob_mu[idx] = Probability[1][kFlav];

    }
    '''


    def __init__(self, earth_model, detector_depth, prop_height, 
                      YeI, YeO, YeM ) :


        #Call base class contructor
        super(self.__class__, self).__init__(
            earth_model=earth_model, 
            detector_depth=detector_depth, 
            prop_height=prop_height, 
            YeI=YeI, 
            YeO=YeO, 
            YeM=YeM
          )

        #Vacuum oscillations are only implemented in prob3 CPU
        #Enforce CPU usage in this case
        if self.earth_model is None :
            raise Exception("No GPU prob3 implementation for vacuum oscillations")

        #Initialize CUDA usage
        self._initialize_kernel()

        #Initilize data arrays
        self._initialize_arrays()


    def _initialize_kernel(self):

        #Perform CUDA imports here (so that we don't need CUDA when using CPUs)
        import pycuda.autoinit
        import pycuda.compiler 

        """Initialize 1) the grid_propagator class, 2) the device arrays that
        will be passed to the `propagateGrid()` kernel, and 3) the kernel
        module.

        """
        # Path relative to `resources` directory
        include_dirs = [
            os.path.abspath(find_resource('../stages/osc/prob3cuda')),
            os.path.abspath(find_resource('../utils'))
        ]
        logging.debug('  pycuda INC PATH: %s' %include_dirs)
        logging.debug('  pycuda FLAGS: %s' %pycuda.compiler.DEFAULT_NVCC_FLAGS)

        kernel_code = (self.KERNEL_TEMPLATE
                       %dict(C_PRECISION_DEF=C_PRECISION_DEF, C_FTYPE=C_FTYPE))

        self.module = pycuda.compiler.SourceModule(
            kernel_code, include_dirs=include_dirs, keep=True
        )
        self.propArray = self.module.get_function('propagateArray') #TODO (Tom) only get the one we are currently expecting to use?
        self.propGrid = self.module.get_function('propagateGrid')


    def _initialize_arrays(self) :

        #Allocate memory for the dm and mix matrices
        #These will be filled (and copied to device) when the mixing params are defined, so for now are empty
        self.dm_mat = ArrayVariable( host_array=np.zeros( OscParams.M_mass_shape(), dtype=FTYPE) ) #TODO NaN?
        self.mix_mat = ArrayVariable( host_array=np.zeros( OscParams.M_pmns_shape(), dtype=FTYPE) ) #TODO NaN?


    def set_params(self, theta12, theta13, theta23,
                   deltam21, deltam31, deltacp) :

        #Call the base class functoon
        super(self.__class__, self).set_params(theta12=theta12, 
                                                theta13=theta13, 
                                                theta23=theta23,
                                                deltam21=deltam21, 
                                                deltam31=deltam31, 
                                                deltacp=deltacp )

        #Determine the corrsponding matrices
        sin2th12Sq = np.sin(self.theta12.m_as("rad"))**2
        sin2th13Sq = np.sin(self.theta13.m_as("rad"))**2
        sin2th23Sq = np.sin(self.theta23.m_as("rad"))**2
        deltam21 = self.deltam21.m_as("eV**2")
        deltamatm = self.deltamatm.m_as("eV**2")
        deltacp = self.deltacp.m_as('rad')

        osc = OscParams( deltam21, deltamatm, sin2th12Sq, sin2th13Sq, sin2th23Sq, deltacp )

        self.dm_mat.set_host_array( osc.M_mass )
        self.mix_mat.set_host_array( osc.M_pmns )

        self.dm_mat.to_device()
        self.mix_mat.to_device()

        logging.debug("dm_mat: \n %s"%str(self.dm_mat.host_array))
        logging.debug("mix[re]: \n %s"%str(self.mix_mat.host_array[:,:,0]))


    def calc_probs(self, prob_e, prob_mu, kNuBar, kFlav, n_evts, true_e_scale, true_energy, num_layers, density_in_layer, distance_in_layer ):

        """
        Calculate oscillation probabilities event-by-event
        #Note that this function assumed 'calc_path_layers' has been called already to pre-compute the
        #layers that each track passed through
        #If the input coszen array has changed since then it is the users reponsibility to re-compute
        #the layers. THis sysytem is a little awkward, but means layers only need to be computed once
        #in general.
        """

        #TODO Add option to create device arrays on here if passed host arrays
        #TODO Add option to calculate layers here if none provided
        #TODO These two options are not the highest performance cases, but would make it easy to call this code externally for e.g. oscillogram plotting

        #Check that params have been set
        if not self.params_are_set :
          raise Exception("Cannot calculate probabilities : User must provide params using 'set_params' method")

        #Check that layers have been computed
        if not self.layers_are_computed :
          raise Exception("Cannot calculate probabilities : User must calculate layers using 'calc_path_layers' method")

        #TODO Check inputs are GPU arrays

        #Some GPU stuff
        bdim = (32, 1, 1)
        dx, mx = divmod(n_evts, bdim[0])
        gdim = ((dx + (mx > 0)) * bdim[0], 1)

        #Callc the CUDA function doing the prob3 calculation
        self.propArray(
            prob_e, #Array to fill
            prob_mu, #Array to fill
            self.dm_mat.device_array,
            self.mix_mat.device_array,
            n_evts,
            np.int32(kNuBar),
            np.int32(kFlav),
            np.int32(self.max_layers),
            FTYPE(true_e_scale),
            true_energy,
            num_layers,
            density_in_layer,
            distance_in_layer,
            block=bdim,
            grid=gdim
        )



#
# Prob 3 wrapper class (covers CPU and GPU cases)
#

#This basically just exposes one single interface to the user, with a toggle
#to choose between CPU and GPU for the engine
#This is made a litle ticky by the quite different calculation architecture 
#in thw two implementations

class prob3wrapper(object) :

    def __init__(self, earth_model, detector_depth, prop_height, 
                      YeI=None, YeO=None, YeM=None,
                      use_gpu=False) :

        #Store args
        self.use_gpu = use_gpu

        #Vacuum oscillations are only implemented in prob3 CPU
        #Enforce CPU usage in this case
        if earth_model is None and self.use_gpu :
            logging.warn("Not GPU prob3 implementation for vacuum oscillations. Will use a CPU")
            self.use_gpu = False

        #Create the underlying calculator
        if self.use_gpu : 
            self.prob3 = prob3gpu(
                earth_model=earth_model, 
                detector_depth=detector_depth, 
                prop_height=prop_height, 
                YeI=YeI, 
                YeO=YeO, 
                YeM=YeM
            )
        else :
            self.prob3 = prob3cpu(
                earth_model=earth_model, 
                detector_depth=detector_depth, 
                prop_height=prop_height, 
                YeI=YeI, 
                YeO=YeO, 
                YeM=YeM
            )


    #Call the 'calc_probs' function of the underlying calculator
    def set_params(self, theta12, theta13, theta23,
                   deltam21, deltam31, deltacp) :

        return self.prob3.set_params(theta12=theta12, 
                                    theta13=theta13, 
                                    theta23=theta23,
                                    deltam21=deltam21, 
                                    deltam31=deltam31, 
                                    deltacp=deltacp )


    #Call the 'calc_probs' function of the underlying calculator
    #Need handle the different args between teh two, and as well as arg hiding for surplus args
    def calc_probs(self, **kw ) :

        #A handy function for grabbing an expected arg from kwargs, with clear error message
        def get_kwarg(name) :
            if name in kw : 
                return kw.pop(name)
            else :
                raise Exception( "Cannot calculate probabilities : Expected argument '%s' was not provided" % name )


        #
        # GPU case
        #

        if self.use_gpu :

            #Get the require args
            prob_e = get_kwarg("prob_e")
            prob_mu = get_kwarg("prob_mu")
            kNuBar = get_kwarg("kNuBar")
            kFlav = get_kwarg("kFlav")
            n_evts = get_kwarg("n_evts")
            true_e_scale = get_kwarg("true_e_scale")
            true_energy = get_kwarg("true_energy")
            num_layers = get_kwarg("num_layers")
            density_in_layer = get_kwarg("density_in_layer")
            distance_in_layer = get_kwarg("distance_in_layer")

            #Call the underlying function
            self.prob3.calc_probs(
                prob_e=prob_e, 
                prob_mu=prob_mu, 
                kNuBar=kNuBar, 
                kFlav=kFlav, 
                n_evts=n_evts, 
                true_e_scale=true_e_scale, 
                true_energy=true_energy,
                num_layers=num_layers, 
                density_in_layer=density_in_layer, 
                distance_in_layer=distance_in_layer )


        #
        # CPU case
        #

        else :

            #Get the require args
            prob_e = get_kwarg("prob_e")
            prob_mu = get_kwarg("prob_mu")
            kNuBar = get_kwarg("kNuBar")
            kFlav = get_kwarg("kFlav")
            true_e_scale = get_kwarg("true_e_scale")
            true_energy = get_kwarg("true_energy")
            true_coszen = get_kwarg("true_coszen")

            #Call the underlying function
            self.prob3.calc_probs(
                prob_e=prob_e, 
                prob_mu=prob_mu, 
                kNuBar=kNuBar, 
                kFlav=kFlav, 
                true_e_scale=true_e_scale,
                true_energy=true_energy,
                true_coszen=true_coszen )



    def calc_path_length(self, coszen) :
        return self.prob3.calc_path_length(coszen)


    def calc_path_layers(self, coszen) :
        return self.prob3.calc_path_layers(coszen)




#TODO the stage....

class prob3(Stage):

    """Neutrino oscillations calculation via Prob3.

    Parameters
    ----------
    params : ParamSet
        All of the following param names (and no more) must be in `params`.
        Earth parameters:
            * earth_model : str (resource location with earth model file)
            * YeI : float (electron fraction, inner core)
            * YeM : float (electron fraction, mantle)
            * YeO : float (electron fraction, outer core)
        Detector parameters:
            * detector_depth : float >= 0
            * prop_height
        Oscillation parameters:
            * deltacp
            * deltam21
            * deltam31
            * theta12
            * theta13
            * theta23
        Nutau (and nutaubar) normalization (binned transform mode only):
            * nutau_norm

    input_binning : MultiDimBinning
    output_binning : MultiDimBinning
    transforms_cache_depth : int >= 0
    outputs_cache_depth : int >= 0
    debug_mode : bool
    use_gpu : Use a GPU to perform the oscillation calculation (otherwise use CPU)
    gpu_id : If running on a system with multiple GPUs, it will choose
             the one with gpu_id. Otherwise, defaults to 0
    use_spline : Instead of calculating probability for individual events, generate a probability spline in E-coszen space and sample from that (faster, less precise)
    spline_binning : The binning to use for the spline, must have "true_energy" and "true_coszen" dimensions

    Input Names
    -----------
    The `inputs` container must include objects with `name` attributes:
      * 'nue'
      * 'numu'
      * 'nuebar'
      * 'numubar'

    Output Names
    ------------
    The `outputs` container generated by this service will be objects with the
    following `name` attribute:
      * 'nue'
      * 'numu'
      * 'nutau'
      * 'nuebar'
      * 'numubar'
      * 'nutaubar'

    """


    def __init__(self, params, input_binning, output_binning,
                 memcache_deepcopy, error_method, transforms_cache_depth,
                 outputs_cache_depth, debug_mode=None, 
                 use_spline=False, spline_binning=None,
                 use_gpu=False, gpu_id=None):

        # If no binning provided then we want to use this to calculate
        # probabilities for events instead of transforms for maps.
        # Set up this self.calc_binned_transforms to use as an assert on the
        # appropriate functions.
        self.calc_binned_transforms = (input_binning is not None and output_binning is not None)
        if ( input_binning is None or output_binning is None ) and ( input_binning != output_binning ):
            raise ValueError('Input and output binning must either both be'
                             ' defined or both be none, but not a mixture.'
                             ' Something is wrong here.')

        # Define the names of objects that are required by this stage (objects
        # will have the attribute `name`: i.e., obj.name)
        input_names = (
            'nue', 'numu', 'nuebar', 'numubar'
        )

        # Define the names of objects that get produced by this stage
        output_names = (
            'nue', 'numu', 'nutau', 'nuebar', 'numubar', 'nutaubar'
        )

        # Invoke the init method from the parent class (Stage), which does a
        # lot of work (caching, providing public interfaces, etc.)
        super(self.__class__, self).__init__(
            use_transforms=True,
            params=params,
            expected_params=self.get_expected_params(self.calc_binned_transforms),
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            disk_cache=None,
            outputs_cache_depth=outputs_cache_depth,
            memcache_deepcopy=memcache_deepcopy,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

        #Instantiate prob3 calculator #TODO THis needs updating
        self.prob3 = prob3wrapper( use_gpu=use_gpu,
                                earth_model=self.params.earth_model)
        #Report usage options
        if self.calc_binned_transforms :
            logging.debug('User has selected to use prob3 %s to produce binned oscillation transforms' % ("GPU" if self.prob3.use_gpu else "CPU") )
        else:
            logging.debug('User has selected to use prob3 %s to calculate event probabilties' % ("GPU" if self.prob3.use_gpu else "CPU"))


    def _derive_nominal_transforms_hash(self):
        """No nominal transforms implemented for this service."""
        return


    @profile
    def _compute_transforms(self):

        #
        # GPU case
        #

        if self.use_gpu :

            """Compute oscillation transforms using grid_propagator GPU code."""

            # Read parameters in, convert to the units used internally for
            # computation, and then strip the units off. Note that this also
            # enforces compatible units (but does not sanity-check the numbers).
            theta12 = self.params.theta12.m_as('rad')
            theta13 = self.params.theta13.m_as('rad')
            theta23 = self.params.theta23.m_as('rad')
            deltam21 = self.params.deltam21.m_as('eV**2')
            deltam31 = self.params.deltam31.m_as('eV**2')
            deltacp = self.params.deltacp.m_as('rad')
            YeI = self.params.YeI.m_as('dimensionless')
            YeO = self.params.YeO.m_as('dimensionless')
            YeM = self.params.YeM.m_as('dimensionless')
            prop_height = self.params.prop_height.m_as('km')

            sin2th12Sq = np.sin(theta12)**2
            sin2th13Sq = np.sin(theta13)**2
            sin2th23Sq = np.sin(theta23)**2

            mAtm = deltam31 if deltam31 < 0.0 else (deltam31 - deltam21)

            self.osc = OscParams(deltam21, mAtm, sin2th12Sq, sin2th13Sq,
                                 sin2th23Sq, deltacp)

            #Write all variables to be used in the probabilities calculation to host arrays
            #TODO (Tom) should the arrays ony be created (e.g. memory allocated) once? Probably, but old scheme did this at every call so doing the same here for now, but should be revisited
            self.host_arrays = { 
              "e_centers":self.e_centers,
              "cz_centers":self.cz_centers,
              "dm_mat":self.osc.M_mass.astype(FTYPE),
              "mix_mat":self.osc.M_pmns.astype(FTYPE)
            }

            self.host_arrays["numLayers"], self.host_arrays["densityInLayer"], self.host_arrays["distanceInLayer"] = self.calc_path_layers(self.cz_centers)

            logging.trace('dm_mat: \n %s' %str(self.host_arrays["dm_mat"]))
            logging.trace('mix[re]: \n %s' %str(self.host_arrays["mix_mat"][:,:,0]))

            ne_bin_centers = np.int32(len(self.e_centers)) #TODO (Tom) use self.num_ebins ??
            ncz_bin_centers = np.int32(len(self.cz_centers)) #TODO (Tom) use self.num_czbins ??

            # Earlier versions had self.e_centers*energy_scale but energy_scale is
            # not used anymore
            self._copy_to_device_arrays(variable="e_centers") #TODO (Tom) I don't see any need for this, should it be removed?

            #Create an array to fill with the results
            self.host_arrays["smooth_maps"] = np.zeros((ncz_bin_centers*ne_bin_centers*12), dtype=FTYPE)

            block_size = (16, 16, 1)
            grid_size = (
                ncz_bin_centers // block_size[0] + 1,
                ne_bin_centers // block_size[1] + 1,
                2
            )

            #Copy arrays to device
            self.device_arrays = self._create_device_arrays(self.host_arrays)

            #Perform the calculation
            self.propGrid(self.device_arrays["smooth_maps"],
                          self.device_arrays["dm_mat"], 
                          self.device_arrays["mix_mat"],
                          self.device_arrays["e_centers"], 
                          self.device_arrays["cz_centers"],
                          ne_bin_centers, ncz_bin_centers,
                          ne_bin_centers, ncz_bin_centers,
                          np.int32(self.maxLayers),
                          self.device_arrays["numLayers"], 
                          self.device_arrays["densityInLayer"],
                          self.device_arrays["distanceInLayer"],
                          block=block_size, grid=grid_size)
                          #shared=16384)

            #Copy the results back from the GPU to the CPU
            self._copy_from_device_arrays(variable="smooth_maps")

            # Return TransformSet
            smooth_maps = np.reshape(self.host_arrays["smooth_maps"], (12, ne_bin_centers, ncz_bin_centers))
            # Slice up the transform arrays into views to populate each transform
            dims = ['true_energy', 'true_coszen']
            xform_dim_indices = [0, 1]
            users_dim_indices = [self.input_binning.index(d) for d in dims]
            xform_shape = [2] + [self.input_binning[d].num_bins for d in dims]
            transforms = []
            for out_idx, output_name in enumerate(self.output_names):
                xform = np.empty(xform_shape)
                if out_idx < 3:
                    # Neutrinos
                    xform[0] = smooth_maps[out_idx]
                    xform[1] = smooth_maps[out_idx+3]
                    input_names = self.input_names[0:2]
                else:
                    # Antineutrinos
                    xform[0] = smooth_maps[out_idx+3]
                    xform[1] = smooth_maps[out_idx+6]
                    input_names = self.input_names[2:4]

                xform = np.moveaxis(
                    xform,
                    source=[0] + [i+1 for i in xform_dim_indices],
                    destination=[0] + [i+1 for i in users_dim_indices]
                )
                transforms.append(
                    BinnedTensorTransform(
                        input_names=input_names,
                        output_name=output_name,
                        input_binning=self.input_binning,
                        output_binning=self.input_binning,
                        xform_array=xform
                    )
                )


        #
        # CPU case
        #

        else :

            """Compute oscillation transforms using Prob3 CPU code."""
            self._setup_barger_propagator()

            # Read parameters in, convert to the units used internally for
            # computation, and then strip the units off. Note that this also
            # enforces compatible units (but does not sanity-check the numbers).
            theta12 = self.params.theta12.m_as('rad')
            theta13 = self.params.theta13.m_as('rad')
            theta23 = self.params.theta23.m_as('rad')
            deltam21 = self.params.deltam21.m_as('eV**2')
            deltam31 = self.params.deltam31.m_as('eV**2')
            deltacp = self.params.deltacp.m_as('rad')
            prop_height = self.params.prop_height.m_as('km')
            nutau_norm = self.params.nutau_norm.m_as('dimensionless')

            # The YeX will not be in params if the Earth model is None
            if self._barger_earth_model is not None:
                YeI = self.params.YeI.m_as('dimensionless')
                YeO = self.params.YeO.m_as('dimensionless')
                YeM = self.params.YeM.m_as('dimensionless')

                total_bins = int(len(self.e_centers)*len(self.cz_centers))
                # We use 18 since we have 3*3 possible oscillations for each of
                # neutrinos and antineutrinos.
                prob_list = np.empty(total_bins*18, dtype='double')
                
                # The 1.0 was energyscale from earlier versions. Perhaps delete this
                # if we no longer want energyscale.
                prob_list, evals, czvals = self.barger_propagator.fill_osc_prob_c(
                    self.e_centers, self.cz_centers, 1.0,
                    deltam21, deltam31, deltacp,
                    prop_height,
                    YeI, YeO, YeM,
                    total_bins*18, total_bins, total_bins,
                    theta12, theta13, theta23
                )
            else:
                # Code copied from BargerPropagator.cc but fill_osc_prob_c but
                # pythonised and modified to use the python binding to
                # GetVacuumProb.
                prob_list = self._get_vacuum_prob_maps(
                    deltam21, deltam31, deltacp,
                    prop_height,
                    theta12, theta13, theta23
                )

            # Slice up the transform arrays into views to populate each transform
            dims = ['true_energy', 'true_coszen']
            xform_dim_indices = [0, 1]
            users_dim_indices = [self.input_binning.index(d) for d in dims]
            xform_shape = [2] + [self.input_binning[d].num_bins for d in dims]

            # TODO: populate explicitly by flavor, don't assume any particular
            # ordering of the outputs names!
            transforms = []
            for out_idx, output_name in enumerate(self.output_names):
                xform = np.empty(xform_shape)
                if out_idx < 3:
                    # Neutrinos
                    xform[0] = np.array([
                        prob_list[out_idx + 18*i*self.num_czbins
                                  : out_idx + 18*(i+1)*self.num_czbins
                                  : 18]
                        for i in range(0, self.num_ebins)
                    ])
                    xform[1] = np.array([
                        prob_list[out_idx+3 + 18*i*self.num_czbins
                                  : out_idx+3 + 18*(i+1)*self.num_czbins
                                  : 18]
                        for i in range(0, self.num_ebins)
                    ])
                    input_names = self.input_names[0:2]

                else:
                    # Antineutrinos
                    xform[0] = np.array([
                        prob_list[out_idx+6 + 18*i*self.num_czbins
                                  : out_idx+6 + 18*(i+1)*self.num_czbins
                                  : 18]
                        for i in range(0, self.num_ebins)
                    ])
                    xform[1] = np.array([
                        prob_list[out_idx+9 + 18*i*self.num_czbins
                                  : out_idx+9 + 18*(i+1)*self.num_czbins
                                  : 18]
                        for i in range(0, self.num_ebins)
                    ])
                    input_names = self.input_names[2:4]

                xform = np.moveaxis(
                    xform,
                    source=[0] + [i+1 for i in xform_dim_indices],
                    destination=[0] + [i+1 for i in users_dim_indices]
                )
                if nutau_norm != 1 and output_name in ['nutau', 'nutaubar']:
                    xform *= nutau_norm
                transforms.append(
                    BinnedTensorTransform(
                        input_names=input_names,
                        output_name=output_name,
                        input_binning=self.input_binning,
                        output_binning=self.output_binning,
                        xform_array=xform
                    )
                )

        return TransformSet(transforms=transforms)


    #Static function for getting expected params
    #Implemented like this so external code can access this information (for example oscilogram plotters which may instantiate this class)
    @staticmethod
    def get_expected_params(calc_binned_transforms) :

        #TODO Is this

        #Define expected params and return them as a list
        #Depends on whether performing PISA map or event-by-event calculation
        #In the current implementation, this does not depend on whether using a CPU or GPU implementation of prob3

        #Start by defining the params common to all cases
        expected_params = [
            'earth_model', 'YeI', 'YeM', 'YeO',
            'detector_depth', 'prop_height',
            'deltacp', 'deltam21', 'deltam31',
            'theta12', 'theta13', 'theta23',
            'nutau_norm',
            ]

        return tuple(expected_params)


    def validate_params(self, params):
        if params['earth_model'].value is None:
            if params['YeI'].value is not None:
                raise ValueError("A none Earth model has been set but the YeI "
                                 "value is set to %s. Set this to none."
                                 %params['YeI'].value)
            if params['YeO'].value is not None:
                raise ValueError("A none Earth model has been set but the YeO "
                                 "value is set to %s. Set this to none."
                                 %params['YeO'].value)
            if params['YeM'].value is not None:
                raise ValueError("A none Earth model has been set but the YeM "
                                 "value is set to %s. Set this to none."
                                 %params['YeM'].value)
        pass


