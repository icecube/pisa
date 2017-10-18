import numpy as np
from  propy.osc import *
from pisa.stages.osc.osc_params import *


def propagateArray(d_prob_e,
                   d_prob_mu,
                   d_dm,
                   d_mix,
                   d_nsi_eps,
                   n_evts,
                   kNuBar,
                   kFlav,
                   maxLayers,
                   true_e_scale,
                   d_energy,
                   d_numberOfLayers,
                   d_densityInLayer,
                   d_distanceInLayer):

    kUseMassEstates = False

    #/*
    #TODO: * ensure convention below is respected in MC reweighting
    #          (kNuBar > 0 for nu, < 0 for anti-nu)
    #        * kNuBar is passed in, so could already pass in the correct form
    #          of mixing matrix, i.e., possibly conjugated
    #*/
    if (kNuBar > 0):
        # in this case the mixing matrix is left untouched
        #copy_complex_matrix(d_mix, mixNuType)
        mixNuType = d_mix
    
    else:
        # here we need to complex conjugate all entries
        # (note that this only changes calculations with non-zero deltacp)
        #conjugate_complex_matrix(d_mix, mixNuType)
        mixNuType = d_mix.conj().T

    HVac2Enu = np.zeros((3,3)) + np.zeros((3,3)) * 1j

    getHVac2Enu(mixNuType, d_dm, HVac2Enu)

    #define those
    #TransitionMatrix = np.zeros((3,3)) + np.zeros((3,3)) * 1j
    #TransitionProduct = np.zeros((3,3)) + np.zeros((3,3)) * 1j
    #TransitionTemp = np.zeros((3,3)) + np.zeros((3,3)) * 1j

    RawInputPsi = np.zeros((3)) + np.zeros((3)) * 1j
    Probability = np.zeros((3,3))

    layers = d_numberOfLayers
    energy = d_energy * true_e_scale
    for i in range(layers):
        density = d_densityInLayer[i]
        distance = d_distanceInLayer[i]
        TransitionMatrix = get_transition_matrix(kNuBar,
                               energy,
                               density,
                               distance,
                               0.0,
                               mixNuType, d_nsi_eps, HVac2Enu,
                               d_dm)
        #print TransitionMatrix
        if (i==0):
            #copy_complex_matrix(TransitionMatrix, TransitionProduct)
            TransitionProduct = TransitionMatrix
        else:
            TransitionProduct = TransitionMatrix * TransitionProduct 
            #clear_complex_matrix( TransitionTemp )
            #multiply_complex_matrix( TransitionMatrix, TransitionProduct, TransitionTemp )
            #copy_complex_matrix( TransitionTemp, TransitionProduct )
        
      # loop on neutrino types, and compute probability for neutrino i:
      # We actually don't care about nutau -> anything since the flux there is zero!
    for i in range(2):
        for j in range(3):
            RawInputPsi[j] = 0.0 + 0.0j

        if( kUseMassEstates ):
            convert_from_mass_eigenstate(i+1, RawInputPsi, mixNuType)
        else:
            RawInputPsi[i] = 1.0

        #// calculate 'em all here, from legacy code...
        OutputPsi = TransitionProduct.dot(RawInputPsi)
        Probability[i][0] += abs(OutputPsi[0])
        Probability[i][1] += abs(OutputPsi[1])
        Probability[i][2] += abs(OutputPsi[2])

    d_prob_e = Probability[0][kFlav]
    d_prob_mu = Probability[1][kFlav]
    print 'prob_e: ', d_prob_e
    print 'prob_mu: ', d_prob_mu


d_prob_e = np.zeros(10)
d_prob_mu = np.ones(10)


OP = OscParams(0.002, 0.02, 0.3, 0.1, 0.5, 0)

d_mix = OP.mix_matrix[:,:,0] + OP.mix_matrix[:,:,1] * 1j
d_dm = OP.dm_matrix

d_nsi_eps = np.zeros((3,3))

n_evts = 10

kNuBar = 1

kFlav = 1

maxLayers = 1

true_e_scale = 1.

d_energy = np.linspace(1,10,10)

d_numberOfLayers = np.ones((10), dtype=np.int)

# needs to be 2d
d_densityInLayer = np.ones((10,1))
d_distanceInLayer = np.ones((10,1))

i=0

propagateArray(d_prob_e[i],
                   d_prob_mu[i],
                   d_dm,
                   d_mix,
                   d_nsi_eps,
                   n_evts,
                   kNuBar,
                   kFlav,
                   maxLayers,
                   true_e_scale,
                   d_energy[i],
                   d_numberOfLayers[i],
                   d_densityInLayer[i],
                   d_distanceInLayer[i])

#print d_prob_e
#print d_prob_mu

