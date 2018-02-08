# authors: T. Stuttard, M. Jensen
# date:   Nov 3, 2017

# Implementing an environmentally-induced decoherence model for neutrino oscillations
# Based on reference [1], which is seeking to explain theta23 tension between NOvA and T2K
#
# References:
# [1] arxiv:1702.04738



from __future__ import absolute_import, print_function, division

import math
import numpy as np
from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.stages.osc.pi_osc_params import OscParams
from pisa.stages.osc.layers import Layers
from pisa.stages.osc.prob3numba.numba_osc import propagate_array, fill_probs
from pisa.utils.numba_tools import WHERE
from pisa.utils.resources import find_resource
from pisa import ureg


__all__ = ['DecoherenceParams', 'calc_decoherence_probs', "decoherence"]


#
# Decoherence calculation helpers
#

#Container for decoherence oscillation params
class DecoherenceParams :

    def __init__(self, deltam21, deltam31, theta12, theta13, theta23, deltacp, gamma21, gamma31, gamma32):

        #Store args
        self.deltam21 = deltam21
        self.deltam31 = deltam31
        self.theta12 = theta12
        self.theta13 = theta13
        self.theta23 = theta23
        self.deltacp = deltacp
        self.gamma21 = gamma21
        self.gamma31 = gamma31
        self.gamma32 = gamma32

        #Get deltam32 (this is what is used in [1])
        self.deltam32 = self.deltam31 - self.deltam21


#Probability calculator function
def calc_decoherence_probs(decoh_params, flav, energy, baseline, prob_e, prob_mu, prob_tau, two_flavor=False):

    #Electron neutrino case
    #For nu_e case, in this approiximation we are essential neglecting nu_e oscillations
    #If a particle starts as a nu_e, it stays as a nu_e
    if flav.startswith("nue") :
        prob_e.fill(1.)
        prob_mu.fill(0.)

    #Muon neutrino case
    #For nu_mu case, in this approximation there is zero probability of becoming a nu_e
    #Use numu disappearance calculation to get numu/tau probs
    elif flav.startswith("numu") :
        prob_e.fill(0.)
        numu_disappearance_func = _calc_numu_disappearance_prob_2flav if two_flavor else _calc_numu_disappearance_prob_3flav
        numu_survival_prob = 1. - numu_disappearance_func( decoh_params=decoh_params, E=energy, L=baseline )
        np.copyto(src=numu_survival_prob,dst=prob_mu) #TODO avoid this wasted data copy 

    else :
        raise ValueError( "Input flavor '%s' not supported" % flav)

    #Assume unitarity
    np.copyto(src=1.-prob_e-prob_mu,dst=prob_tau)

    '''
    print("---------------")
    print("%s" % flav)
    print("prob_e   = %s" % prob_e[:5])
    print("prob_mu  = %s" % prob_mu[:5])
    print("prob_tau = %s" % prob_tau[:5])
    print("---------------")
    '''


#Calculate numju disppearance in 2-flavor model
#Don't call tihs directly, use calc_decoherence_probs
def _calc_numu_disappearance_prob_2flav(decoh_params,E,L) :

    #Define two-flavor decoherence approximation equation
    #Eq. 2 from [1]

    #This line is a standard oscillations (no decoherence) 2 flavour approximation, can use for debugging
    #return np.sin(2.*theta23.m_as("rad"))**2 * np.square(np.sin(1.27*deltam32.m_as("eV**2")*L/E))

    #Assume units if none provided for main input arrays
    #Would prefer to get units but is not always the case
    E = E if hasattr(E,"units") else E * ureg["GeV"]
    L = L if hasattr(L,"units") else L * ureg["km"]

    #Calculate normalisation term
    norm_term = 0.5 * ( np.sin( 2. * decoh_params.theta23.m_as("rad") )**2 )

    #Calculate decoherence term
    decoh_term = np.exp( -decoh_params.gamma32.m_as("eV") * ( L.m_as("m")/1.97e-7 ) ) #Convert L from [m] to natural units

    #Calculate oscillation term
    osc_term = np.cos( ( 2. * 1.27 * decoh_params.deltam32.m_as("eV**2") * L.m_as("km") ) / ( E.m_as("GeV") ) )

    return norm_term * ( 1. - (decoh_term*osc_term) )


#Helper function used by _calc_numu_disappearance_prob_3flav
def _update_matrix(theta12, theta13, theta23):

    """
    Updates the PMNS matrix and its complex conjugate.
    
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


#Calculate numju disppearance in 2-flavor model
#Don't call tihs directly, use calc_decoherence_probs
def _calc_numu_disappearance_prob_3flav(decoh_params, E, L):
        
    #Returns the oscillation probability.
        
    # Oscillations with E = 0 or don't really make sense,
    # but if you're plotting graphs these are the values you'll want.

    E = E if hasattr(E,"units") else E * ureg["GeV"]
    L = L if hasattr(L,"units") else L * ureg["km"]

    U = _update_matrix(decoh_params.theta12, decoh_params.theta13, decoh_params.theta23)[0] # Use PMNS matrix
    
    # No decoherence 
    #s = complex(0.0,0.0)

    #for x in range(3):
    #   s += U[a][x].conjugate() * U[b][x] * cmath.exp( (-i*m2[x]*L)/(2.0*E) )
    
    #return pow(abs(s), 2)
    
    # Decoherence matrix
    Gamma = np.zeros([3,3])
    Gamma[1][0] = decoh_params.gamma21.m_as("GeV")
    Gamma[2][0] = decoh_params.gamma31.m_as("GeV")
    Gamma[2][1] = decoh_params.gamma32.m_as("GeV")
                
            
    #if( E[i_e][i_l] == 0.0 or L[i_e][i_l] == 0.0 ):
    #   return 1.0          
    delta_jk = np.zeros([3,3])
    delta_jk[1][0] = decoh_params.deltam21.m_as("eV**2") 
    delta_jk[2][0] = decoh_params.deltam31.m_as("eV**2")
    delta_jk[2][1] = decoh_params.deltam32.m_as("eV**2")

    prob_dec = np.zeros(np.shape(E))
    
    for i_j in range(3):
        for i_k in range(3):
            if i_j > i_k:
                prob_dec += abs(U[2][i_j])**2 * abs(U[2][i_k])**2 * (1.0 - np.exp( - Gamma[i_j][i_k] * L.m_as("km") * 5.07e+18) * np.cos(delta_jk[i_j][i_k] * 1.0e-18 / (2.0 * E.m_as("GeV")) * L.m_as("km") * 5.07e+18))
    prob_array = 2.0 * prob_dec.real    
    
    return prob_array



#
# Decoherence stage
#

class decoherence(PiStage):
    """
    prob3 osc PISA Pi class

    Paramaters
    ----------
    detector_depth : float
    earth_model : PREM file path
    prop_height : quantity (dimensionless)
    YeI : quantity (dimensionless)
    YeO : quantity (dimensionless)
    YeM : quantity (dimensionless)
    theta12 : quantity (angle)
    theta13 : quantity (angle)
    theta23 : quantity (angle)
    deltam21 : quantity (mass^2)
    deltam31 : quantity (mass^2)
    deltacp : quantity (angle)

    None

    Notes
    -----

    """
    def __init__(self,
                 data=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                ):

        expected_params = ('detector_depth',
                           'earth_model',
                           'prop_height',
                           'YeI',
                           'YeO',
                           'YeM',
                           'theta12',
                           'theta13',
                           'theta23',
                           'deltam21',
                           'deltam31',
                           'deltacp',
                           'gamma21',
                           'gamma31',
                           'gamma32',
                          )

        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_apply_keys = ('weights',
                            'sys_flux',
                           )
        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ('prob_e',
                            'prob_mu',
                           )
        # what keys are added or altered for the outputs during apply
        output_apply_keys = ('weights',
                      )

        # init base class
        super(decoherence, self).__init__(data=data,
                                       params=params,
                                       expected_params=expected_params,
                                       input_names=input_names,
                                       output_names=output_names,
                                       debug_mode=debug_mode,
                                       input_specs=input_specs,
                                       calc_specs=calc_specs,
                                       output_specs=output_specs,
                                       input_apply_keys=input_apply_keys,
                                       output_calc_keys=output_calc_keys,
                                       output_apply_keys=output_apply_keys,
                                      )

        #Have not yet implemented matter effects
        if self.params.earth_model.value is not None :
            raise ValueError("Matter effects not yet implemented for decoherence, must set 'earth_model' to None")

        assert self.input_mode is not None
        assert self.calc_mode is not None
        assert self.output_mode is not None

        self.layers = None

        #Toggle between 2-flavor and 3-flavor models
        self.two_flavor = False


    def setup_function(self):

        # setup Earth model
        if self.params.earth_model.value is not None:
            earth_model = find_resource(self.params.earth_model.value)
            YeI = self.params.YeI.value.m_as('dimensionless')
            YeO = self.params.YeO.value.m_as('dimensionless')
            YeM = self.params.YeM.value.m_as('dimensionless')
        else :
            earth_model = None

        # setup the layers
        prop_height = self.params.prop_height.value.m_as('km')
        detector_depth = self.params.detector_depth.value.m_as('km')
        self.layers = Layers(earth_model, detector_depth, prop_height)
        if earth_model is not None :
            self.layers.setElecFrac(YeI, YeO, YeM)

        # set the correct data mode
        self.data.data_specs = self.calc_specs

        # --- calculate the layers ---
        if self.calc_mode == 'binned':
            # speed up calculation by adding links
            # as layers don't care about flavour
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc',
                                             'nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                             'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])

        for container in self.data:
            if self.params.earth_model.value is not None:
                self.layers.calcLayers(container['true_coszen'].get('host'))
                container['densities'] = self.layers.density.reshape((container.size, self.layers.max_layers))
                container['distances'] = self.layers.distance.reshape((container.size, self.layers.max_layers))
            else :
                self.layers.calcPathLength(container['true_coszen'].get('host'))
                container['distances'] = self.layers.distance

        # don't forget to un-link everything again
        self.data.unlink_containers()

        # --- setup empty arrays ---
        if self.calc_mode == 'binned':
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc'])
            self.data.link_containers('nubar', ['nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                                'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])
        for container in self.data:
            container['probability'] = np.empty((container.size, 3, 3), dtype=FTYPE)
        self.data.unlink_containers()

        # setup more empty arrays
        for container in self.data:
            container['prob_e'] = np.empty((container.size), dtype=FTYPE)
            container['prob_mu'] = np.empty((container.size), dtype=FTYPE)


    @profile
    def compute_function(self):

        # set the correct data mode
        self.data.data_specs = self.calc_specs

        if self.calc_mode == 'binned':
            # speed up calculation by adding links
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc'])
            self.data.link_containers('nubar', ['nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                                'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])

        # --- update params ---
        self.decoh_params = DecoherenceParams(deltam21=self.params.deltam21.value,
                                            deltam31=self.params.deltam31.value,
                                            theta12=self.params.theta12.value,
                                            theta13=self.params.theta13.value,
                                            theta23=self.params.theta23.value,
                                            deltacp=self.params.deltacp.value,
                                            gamma21=self.params.gamma21.value,
                                            gamma31=self.params.gamma31.value,
                                            gamma32=self.params.gamma32.value)

        # Calculate oscillation probabilities
        for container in self.data:
            self.calc_probs(container['nubar'],
                            container['true_energy'],
                            #container['densities'],
                            container['distances'],
                            out=container['probability'],
                           )

        # the following is flavour specific, hence unlink
        self.data.unlink_containers()

        for container in self.data:
            # initial electrons (0)
            fill_probs(container['probability'].get(WHERE),
                       0,
                       container['flav'],
                       out=container['prob_e'].get(WHERE),
                      )
            # initial muons (1)
            fill_probs(container['probability'].get(WHERE),
                       1,
                       container['flav'],
                       out=container['prob_mu'].get(WHERE),
                      )

            container['prob_e'].mark_changed(WHERE)
            container['prob_mu'].mark_changed(WHERE)

            '''
            print("---------------")
            print("%s" % container.name)
            print("prob_e   = %s" % container['prob_e'].get(WHERE)[:5])
            print("prob_mu  = %s" % container['prob_mu'].get(WHERE)[:5])
            print("---------------")
            '''


    @profile
    def apply_function(self):

        # update the outputted weights
        for container in self.data:
            apply_probs(container['sys_flux'].get(WHERE),
                        container['prob_e'].get(WHERE),
                        container['prob_mu'].get(WHERE),
                        out=container['weights'].get(WHERE))
            container['weights'].mark_changed(WHERE)


    def calc_probs(self, nubar, e_array, len_array, out):

        #Get the probability values output array
        prob_array = out.get(WHERE)

        #Attach units
        L = len_array.get(WHERE) * ureg["km"]
        E = e_array.get(WHERE) * ureg["GeV"]

        #nue
        calc_decoherence_probs( decoh_params=self.decoh_params, flav="nue", energy=E, baseline=L, prob_e=prob_array[:,0,0], prob_mu=prob_array[:,0,1], prob_tau=prob_array[:,0,2], two_flavor=self.two_flavor )

        #numu
        calc_decoherence_probs( decoh_params=self.decoh_params, flav="numu", energy=E, baseline=L, prob_e=prob_array[:,1,0], prob_mu=prob_array[:,1,1], prob_tau=prob_array[:,1,2], two_flavor=self.two_flavor )

        #nutau (basically just the inverse of the numu case)
        np.copyto(dst=prob_array[:,2,0],src=prob_array[:,1,0])
        np.copyto(dst=prob_array[:,2,1],src=prob_array[:,1,2])
        np.copyto(dst=prob_array[:,2,2],src=prob_array[:,1,1])

        '''
        print("---------------")
        print("nue")
        print("prob_e   = %s" % prob_array[:5,0,0])
        print("prob_mu  = %s" % prob_array[:5,0,1])
        print("prob_tau = %s" % prob_array[:5,0,2])
        print("---------------")

        print("---------------")
        print("numu")
        print("prob_e   = %s" % prob_array[:5,1,0])
        print("prob_mu  = %s" % prob_array[:5,1,1])
        print("prob_tau = %s" % prob_array[:5,1,2])
        print("---------------")

        print("---------------")
        print("nutau")
        print("prob_e   = %s" % prob_array[:5,2,0])
        print("prob_mu  = %s" % prob_array[:5,2,1])
        print("prob_tau = %s" % prob_array[:5,2,2])
        print("---------------")
        '''

        #Register that arrays have changed
        out.mark_changed(WHERE)



# vectorized function to apply (flux * prob)
# must be outside class
if FTYPE == np.float64:
    signature = '(f8[:], f8, f8, f8[:])'
else:
    signature = '(f4[:], f4, f4, f4[:])'
@guvectorize([signature], '(d),(),()->()', target=TARGET)
def apply_probs(flux, prob_e, prob_mu, out):
    out[0] *= (flux[0] * prob_e) + (flux[1] * prob_mu)
