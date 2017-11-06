# authors: T.Stuttard
# date:   Nov 3, 2017


from __future__ import division

import os, sys, copy

import numpy as np

from pisa import FTYPE, C_FTYPE, C_PRECISION_DEF
from pisa.core.binning import MultiDimBinning
from pisa.utils.resources import find_resource
from pisa.utils.log import logging
from pisa.stages.osc.prob3_new import prob3base
from pisa import ureg


__all__ = ['prob3','prob3cpu','prob3gpu','prob3wrapper','ArrayVariable']

class decoherence(prob3base) :

    def __init__(self, earth_model, detector_depth, prop_height, 
                      YeI=None, YeO=None, YeM=None) :

        #Haven't implemented matter effects yet
        if earth_model is not None :
            raise Exception("Matter effects not yet implemented for decoherence")

        #Warn about simple 2 flavour approximation
        #TODO

        #Call base class contructor
        super(self.__class__, self).__init__(
            earth_model=earth_model, 
            detector_depth=detector_depth, 
            prop_height=prop_height, 
            YeI=YeI, 
            YeO=YeO, 
            YeM=YeM
          )


    def set_params(self, theta12, theta13, theta23,
                   deltam21, deltam31, deltacp,
                   gamma21, gamma31, gamma32):

        #Not yet hadling delta CP
        if not np.isclose(deltacp.m_as("rad"),0.) :
            raise Exception("Decoherence calculator cannot yet handle deltacp != 0")

        #Set decoherence gamma params
        self.gamma21 = gamma21
        self.gamma31 = gamma31
        self.gamma32 = gamma32

        #Enforce >= 0. for decoherence parameters
        if self.gamma21.m_as("GeV") < 0. : raise Exception("Decoherence parameter gamma21 must be >= 0.") 
        if self.gamma31.m_as("GeV") < 0. : raise Exception("Decoherence parameter gamma31 must be >= 0.") 
        if self.gamma32.m_as("GeV") < 0. : raise Exception("Decoherence parameter gamma32 must be >= 0.") 

        #Otherwise use base class function for all standard oscillation params
        return super(self.__class__, self).set_params(theta12=theta12, 
                                                        theta13=theta13, 
                                                        theta23=theta23,
                                                        deltam21=deltam21, 
                                                        deltam31=deltam31, 
                                                        deltacp=deltacp )




    def calc_probs(self, prob_e, prob_mu, kNuBar, kFlav, true_e_scale, true_energy, true_coszen, **kw) :

        #Note that kwargs are not used, but included to hide args such that this function can be called
        #interchangeably with prob3

        #Get baseline from zenith angle #TODO rewrite calc_path_length to accept an array, and do this only once...
        L = np.full(true_coszen.shape,np.NaN) 
        for index in np.ndindex(true_coszen.shape) : 
            L[index] = self.calc_path_length(true_coszen[index])
        L *= ureg["km"] #calc_layers returns results in [km]

        #Note: Ignoring nu/nubar distinction here, as matter effects and CP violation phase are not implemented

        #Get deltam32
        #TODO Do this properly
        #TODO Do this properly
        #TODO Do this properly
        #TODO Do this properly
        #TODO Do this properly
        #TODO Do this properly
        deltam32 = self.deltam31


        #Electron neutrino case
        if kFlav == 0 :

            #For nu_e case, in this 2-flavor approiximation we are essential neglecting nu_e oscillations
            #If a particle starts as a nu_e, it stays as a nu_e
            prob_e.fill(1.)
            prob_mu.fill(0.)


        #Muon neutrino case
        elif kFlav == 1 :

            #For nu_mu case, in this 2-flavor approximation there is zero probability of becoming a nu_e
            prob_e.fill(0.)

            #Calculate numu survival probability
            numu_survial_prob = 1. - self._calc_numu_disappearance_prob(theta23=self.theta23,
                                                                        deltam32=deltam32,
                                                                        gamma32=self.gamma32,
                                                                        E=true_e_scale*true_energy,
                                                                        L=L)
            np.copyto(prob_mu,numu_survial_prob)


        #Tau neutrino case
        elif kFlav == 2 :

            #For nu_tau case, in this 2-flavor approximation there is zero probability of becoming a nu_e
            prob_e.fill(0.)

            #In 2-flavor approx, numu appearance is due to nutau disappearance
            nutau_disappearance_prob = 1. - self._calc_numu_disappearance_prob(theta23=self.theta23,
                                                                        deltam32=deltam32,
                                                                        gamma32=self.gamma32,
                                                                        E=true_e_scale*true_energy,
                                                                        L=L)
            np.copyto(prob_mu,nutau_disappearance_prob)

        else :
            raise Exception( "Unrecognised kFlav value %i" % kFlav )



    def _calc_numu_disappearance_prob(self,theta23,deltam32,gamma32,E,L) : #TODO staticmethod?

        #Define two-flavor decoherence approximation equation
        #Eq. 2 from arxiv:1702.04738

        #This line is a standard oscillations (no decoherence) 2 flavour approximation, can use for debugging
        #return np.sin(2.*theta23.m_as("rad"))**2 * np.square(np.sin(1.27*deltam32.m_as("eV**2")*L/E))

        #Assume units if none provided for main input arrays
        #Would prefer to get units but is not always the case
        E = E if hasattr(E,"units") else E * ureg["GeV"]
        L = L if hasattr(L,"units") else L * ureg["km"]

        #Calculate normalisation term
        norm_term = 0.5 * ( np.sin( 2. * theta23.m_as("rad") )**2 )

        #Calculate decoherence term
        decoh_term = np.exp( -gamma32.m_as("eV") * ( L.m_as("m")/1.97e-7 ) ) #Convert L from [m] to natural units

        #Calculate oscillation term
        osc_term = np.cos( ( 2. * 1.27 * deltam32.m_as("eV**2") * L.m_as("km") ) / ( E.m_as("GeV") ) )

        return norm_term * ( 1. - (decoh_term*osc_term) )

