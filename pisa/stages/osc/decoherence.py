# authors: T.Stuttard
# date:   Nov 3, 2017


from __future__ import division

import os, sys, copy

import math, cmath
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
                      YeI=None, YeO=None, YeM=None,
                      two_flavor=False) :

        #Haven't implemented matter effects yet
        if earth_model is not None :
            raise Exception("Matter effects not yet implemented for decoherence")

        #Warn about simple 2 flavour approximation
        self.two_flavor = two_flavor
        if self.two_flavor : 
            logging.warn("Using 2-flavor approximation for decoherence oscillation probability calcualtion")

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

        #Enforce >= 0. for decoherence parameters #TODO Think abou this...
        #if self.gamma21.m_as("GeV") < 0. : raise Exception("Decoherence parameter gamma21 must be >= 0.") 
        #if self.gamma31.m_as("GeV") < 0. : raise Exception("Decoherence parameter gamma31 must be >= 0.") 
        #if self.gamma32.m_as("GeV") < 0. : raise Exception("Decoherence parameter gamma32 must be >= 0.") 

        #Otherwise use base class function for all standard oscillation params
        super(self.__class__, self).set_params(theta12=theta12, 
                                                theta13=theta13, 
                                                theta23=theta23,
                                                deltam21=deltam21, 
                                                deltam31=deltam31, 
                                                deltacp=deltacp )

        #Get deltam32
        #TODO Do this properly
        self.deltam32 = self.deltam31



    def calc_probs(self, prob_e, prob_mu, kNuBar, kFlav, true_e_scale, true_energy, true_coszen, **kw) :

        #Note that kwargs are not used, but included to hide args such that this function can be called
        #interchangeably with prob3

        #Get baseline from zenith angle #TODO rewrite calc_path_length to accept an array, and do this only once...
        L = np.full(true_coszen.shape,np.NaN) 
        for index in np.ndindex(true_coszen.shape) : 
            L[index] = self.calc_path_length(true_coszen[index])
        L = L * ureg["km"] #calc_layers returns results in [km] (warning, don't use '*=', doesn't do what you'd expect with some pint versions)

        #Note: Ignoring nu/nubar distinction here, as matter effects and CP violation phase are not implemented

        #Electron neutrino case
        if kFlav == 0 :

            #For nu_e case, in this approiximation we are essential neglecting nu_e oscillations
            #If a particle starts as a nu_e, it stays as a nu_e
            prob_e.fill(1.)
            prob_mu.fill(0.)


        #Muon neutrino case
        elif kFlav == 1 :

            #For nu_mu case, in this approximation there is zero probability of becoming a nu_e
            prob_e.fill(0.)

            #Calculate numu survival probability
            numu_survival_prob = self.calc_numu_disappearance_prob( E=true_e_scale*true_energy, L=L )
            np.copyto(prob_mu,numu_survival_prob)

        #Tau neutrino case
        elif kFlav == 2 :

            #For nu_tau case, in this approximation there is zero probability of becoming a nu_e
            prob_e.fill(0.)

            #Currently just using the inverse of the numu case
            nutau_disappearance_prob = self.calc_numu_disappearance_prob( E=true_e_scale*true_energy, L=L )
            np.copyto(prob_mu,nutau_disappearance_prob)
         
        else :
            raise Exception( "Unrecognised kFlav value %i" % kFlav )




    def calc_numu_disappearance_prob(self,E,L) :
        if self.two_flavor : return self._calc_numu_disappearance_prob_2flav(E,L)
        else : return self._calc_numu_disappearance_prob_3flav(E,L)


    def _calc_numu_disappearance_prob_2flav(self,E,L) :

        #Define two-flavor decoherence approximation equation
        #Eq. 2 from arxiv:1702.04738

        #This line is a standard oscillations (no decoherence) 2 flavour approximation, can use for debugging
        #return np.sin(2.*theta23.m_as("rad"))**2 * np.square(np.sin(1.27*deltam32.m_as("eV**2")*L/E))

        #Assume units if none provided for main input arrays
        #Would prefer to get units but is not always the case
        E = E if hasattr(E,"units") else E * ureg["GeV"]
        L = L if hasattr(L,"units") else L * ureg["km"]

        #Calculate normalisation term
        norm_term = 0.5 * ( np.sin( 2. * self.theta23.m_as("rad") )**2 )

        #Calculate decoherence term
        decoh_term = np.exp( -self.gamma32.m_as("eV") * ( L.m_as("m")/1.97e-7 ) ) #Convert L from [m] to natural units

        #Calculate oscillation term
        osc_term = np.cos( ( 2. * 1.27 * self.deltam32.m_as("eV**2") * L.m_as("km") ) / ( E.m_as("GeV") ) )

        return norm_term * ( 1. - (decoh_term*osc_term) )


    def _updateMatrix(self, theta12, theta13, theta23):

        """Updates the PMNS matrix and its complex conjugate.
        
        Must be called by the class each time one of the PMNS matrix parameters are changed.
        """

        zero = 0.0
        c12  =  math.cos( theta12.m_as("rad") )
        c13  =  math.cos( theta13.m_as("rad") )
        c23  =  math.cos( theta23.m_as("rad") )
        s12  =  math.sin( theta12.m_as("rad") )
        s13  =  math.sin( theta13.m_as("rad") )
        s23  =  math.sin( theta23.m_as("rad") )
        eid  = 0.0 # e^( i * delta_cp)
        emid = 0.0 # e^(-i * delta_cp)
        
        matrix      = [[zero,zero,zero],[zero,zero,zero],[zero,zero,zero]]
        anti_matrix = [[zero,zero,zero],[zero,zero,zero],[zero,zero,zero]]
        
        matrix[0][0] = c12 * c13
        matrix[0][1] = s12 * c13
        matrix[0][2] = s13 * emid
        
        matrix[1][0] = (zero - s12*c23 ) - ( c12*s23*s13*eid )
        matrix[1][1] = ( c12*c23 ) - ( s12*s23*s13*eid )
        matrix[1][2] = s23*c13
        
        matrix[2][0] = ( s12*s23 ) - ( c12*c23*s13*eid)
        matrix[2][1] = ( zero - c12*s23 ) - ( s12*c23*s13*eid )
        matrix[2][2] = c23*c13
        
        for i in range(3):
            for j in range(3):
                anti_matrix[i][j] = matrix[i][j].conjugate()

        return matrix, anti_matrix


    #def _calc_numu_disappearance_prob_3flav(self, theta12, theta13, theta23, deltam21, deltam31, deltam32, gamma21, gamma31, gamma32, E, L):
    def _calc_numu_disappearance_prob_3flav(self, E, L):
            
        #Returns the oscillation probability.
            
        # Oscillations with E = 0 or don't really make sense,
        # but if you're plotting graphs these are the values you'll want.

        E = E if hasattr(E,"units") else E * ureg["GeV"]
        L = L if hasattr(L,"units") else L * ureg["km"]

        U = self._updateMatrix(self.theta12, self.theta13, self.theta23)[0] # Use PMNS matrix
        
        # No decoherence 
        #s = complex(0.0,0.0)

        #for x in range(3):
        #   s += U[a][x].conjugate() * U[b][x] * cmath.exp( (-i*m2[x]*L)/(2.0*E) )
        
        #return pow(abs(s), 2)
        
        
        # Decoherence
        Gamma = np.zeros([3,3])
        Gamma[1][0] = self.gamma21.m_as("GeV")
        Gamma[2][0] = self.gamma31.m_as("GeV")
        Gamma[2][1] = self.gamma32.m_as("GeV")
                    
                
        #if( E[i_e][i_l] == 0.0 or L[i_e][i_l] == 0.0 ):
        #   return 1.0          
        delta_jk = np.zeros([3,3])
        delta_jk[1][0] = self.deltam21.m_as("eV**2") 
        delta_jk[2][0] = self.deltam31.m_as("eV**2")
        delta_jk[2][1] = self.deltam32.m_as("eV**2")

        prob_dec = np.zeros(np.shape(E))
        
        for i_j in range(3):
            for i_k in range(3):
                if i_j > i_k:
                    prob_dec += abs(U[2][i_j])**2 * abs(U[2][i_k])**2 * (1.0 - np.exp( - Gamma[i_j][i_k] * L.m_as("km") * 5.07e+18) * np.cos(delta_jk[i_j][i_k] * 1.0e-18 / (2.0 * E.m_as("GeV")) * L.m_as("km") * 5.07e+18))
        prob_array = 2.0 * prob_dec.real    
        
        return prob_array

