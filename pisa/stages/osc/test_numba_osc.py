from __future__ import print_function
import numpy as np
from  propy.osc import *
from pisa.stages.osc.osc_params import *
from numba import jit, vectorize, guvectorize, float64, complex64, int32, float32, complex128

nopython=False

#@guvectorize([(float64[:,:], complex128[:,:], float64[:,:], int32, int32, int32, float64, int32[:], float64[:], float64[:], float64[:,:])], '(a,b),(c,d),(e,f),(),(),(),(),(),(g),(h)->(a,b)', nopython=nopython)#, target='parallel')A
@guvectorize([(float64[:,:], complex128[:,:], float64[:,:], int32, int32, int32, float64, int32[:], float64[:], float64[:], float64[:,:])], '(a,b),(c,d),(e,f),(),(),(),(),(),(g),(h)->(a,b)', nopython=nopython)#, target='parallel')A
def propagateArray(dm,
                   mix,
                   nsi_eps,
                   kNuBar,
                   kFlav,
                   maxLayers,
                   energy,
                   numberOfLayers,
                   densityInLayer,
                   distanceInLayer,
                   Probability):

    kUseMassEstates = False
    #print('knubar = ',kNuBar)
    #print('mix = ',mix)

    #TODO: * ensure convention below is respected in MC reweighting
    #          (kNuBar > 0 for nu, < 0 for anti-nu)
    #        * kNuBar is passed in, so could already pass in the correct form
    #          of mixing matrix, i.e., possibly conjugated
    if (kNuBar > 0):
        # in this case the mixing matrix is left untouched
        mixNuType = mix
    
    else:
        # here we need to complex conjugate all entries
        # (note that this only changes calculations with non-zero deltacp)
        #print('mix: ',mix)
        mixNuType = np.conjugate(mix).T

    HVac2Enu = np.zeros((3,3)) + np.zeros((3,3)) * 1j

    getHVac2Enu(mixNuType, dm, HVac2Enu)

    RawInputPsi = np.zeros((3)) + np.zeros((3)) * 1j

    layers = numberOfLayers
    #print('layers = ',layers)
    for i in range(layers):
        density = densityInLayer[i]
        distance = distanceInLayer[i]
        TransitionMatrix = get_transition_matrix(kNuBar,
                               energy,
                               density,
                               distance,
                               0.0,
                               mixNuType,
                               nsi_eps,
                               HVac2Enu,
                               dm)
        if (i==0):
            TransitionProduct = TransitionMatrix
        else:
            TransitionProduct = TransitionMatrix * TransitionProduct 
        
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

    #prob_e = Probability[0][kFlav]
    #prob_mu = Probability[1][kFlav]
    #print 'prob_e: ', prob_e
    #print 'prob_mu: ', prob_mu



OP = OscParams(0.002, 0.02, 0.3, 0.1, 0.5, 0)

mix = OP.mix_matrix[:,:,0] + OP.mix_matrix[:,:,1] * 1j
dm = OP.dm_matrix

nsi_eps = np.zeros((3,3))

nevts = 1000

# input arrays
# nu /nu-bar
kNuBar = np.ones(nevts, dtype=np.int32)
# flavours
kFlav = np.ones(nevts, dtype=np.int32)

energy = np.linspace(1,10,nevts)

# Layers
maxLayers = np.int32(1)
numberOfLayers = np.ones((nevts), dtype=np.int32)
densityInLayer = np.ones((nevts,1))
distanceInLayer = np.ones((nevts,1))

# empty arrays to be filled
#prob_e = np.zeros(nevts)
#prob_mu = np.ones(nevts)
Probability = np.zeros((nevts,3,3))

i=0

propagateArray(
               dm,
               mix,
               nsi_eps,
               kNuBar,
               kFlav,
               maxLayers,
               energy,
               numberOfLayers,
               densityInLayer,
               distanceInLayer,
               Probability)

print(Probability)

