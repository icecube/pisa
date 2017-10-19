from __future__ import print_function
import numpy as np
from propy.osc import *
from pisa.stages.osc.osc_params import OscParams
from pisa.stages.osc.layers import Layers
from numba import jit, vectorize, guvectorize, float64, complex64, int32, float32, complex128
import time

#nopython=False
nopython=True

@guvectorize([(float64[:,:], complex128[:,:], complex128[:,:], int32, int32, float64, int32, float64[:], float64[:], float64[:,:])], '(a,b),(c,d),(e,f),(),(),(),(),(g),(h)->(a,b)', nopython=nopython, target='parallel', cache=True)
def propagateArray(dm,
                   mix,
                   nsi_eps,
                   kNuBar,
                   kFlav,
                   energy,
                   numberOfLayers,
                   densityInLayer,
                   distanceInLayer,
                   Probability):

    #print('dm',dm)
    #print('mix',mix)
    #print('nsi',nsi_eps)
    #print('knubar',kNuBar)
    #print('kflav',kFlav)
    #print('E',energy)
    #print('nlayers',numberOfLayers)
    #print('density',densityInLayer)
    #print('dist',distanceInLayer)
    #print('prob',Probability)

    kUseMassEstates = False

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
        mixNuType = np.conjugate(mix).T

    HVac2Enu = getHVac2Enu(mixNuType, dm)

    RawInputPsi = np.zeros((3)) + np.zeros((3)) * 1j

    for i in range(numberOfLayers):
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
            TransitionProduct = np.dot(TransitionMatrix,TransitionProduct)
        
    # loop on neutrino types, and compute probability for neutrino i:
    # We actually don't care about nutau -> anything since the flux there is zero!
    for i in range(2):
        for j in range(3):
            RawInputPsi[j] = 0.0 + 0.0j

        if( kUseMassEstates ):
            convert_from_mass_eigenstate(i+1, RawInputPsi, mixNuType)
        else:
            RawInputPsi[i] = 1.0 + 0.0j

        #// calculate 'em all here, from legacy code...
        OutputPsi = np.dot(TransitionProduct,RawInputPsi)
        Probability[i][0] += OutputPsi[0].real**2 + OutputPsi[0].imag**2
        Probability[i][1] += OutputPsi[1].real**2 + OutputPsi[1].imag**2
        Probability[i][2] += OutputPsi[2].real**2 + OutputPsi[2].imag**2


OP = OscParams(7.5e-5, 2.524e-3, np.sqrt(0.306), np.sqrt(0.02166), np.sqrt(0.441), 261/180.*np.pi)

mix = OP.mix_matrix[:,:,0] + OP.mix_matrix[:,:,1] * 1j
dm = OP.dm_matrix

nsi_eps = np.zeros((3,3)) + np.zeros((3,3)) * 1j

nevts = int(1e3)

# input arrays
# nu /nu-bar
kNuBar = np.ones(nevts, dtype=np.int32)
# flavours
kFlav = np.ones(nevts, dtype=np.int32)

energy = np.logspace(0,3,nevts)

myLayers = Layers('/home/peller/pisa/pisa/resources/osc/PREM_4layer.dat', 2, 20)
myLayers.setElecFrac(0.4656, 0.4656, 0.4957)

cz = -np.ones((nevts))
myLayers.calcLayers(cz)

#for now just preten they are all the same dimension
numberOfLayers = myLayers.n_layers
densityInLayer = myLayers.density.reshape((nevts,numberOfLayers[0]+1))
distanceInLayer = myLayers.distance.reshape((nevts,numberOfLayers[0]+1))
#print('nlayers: ',myLayers.n_layers)
#print('density: ',myLayers.density)
#print('distance: ',myLayers.distance)
#print('max: ',myLayers.max_layers)

# Layers
#nlay = 1
#numberOfLayers = nlay * np.ones((nevts), dtype=np.int32)
#densityInLayer = np.ones((nevts,nlay))
#distanceInLayer = 12700 * np.ones((nevts,nlay))
#print(distanceInLayer)

# empty arrays to be filled
Probability = np.zeros((nevts,3,3))

i=0
start_t = time.time()
propagateArray(
               dm,
               mix,
               nsi_eps,
               kNuBar,
               kFlav,
               energy,
               numberOfLayers,
               densityInLayer,
               distanceInLayer,
               Probability)
end_t = time.time()
print ('%.2f s for %i events'%((end_t-start_t),nevts))


import matplotlib as mpl
# Headless mode; must set prior to pyplot import
mpl.use('Agg')
from matplotlib import pyplot as plt

ax = plt.gca()
ax.plot(energy, Probability[:,1,0], color='g')
ax.plot(energy, Probability[:,1,1], color='b')
ax.plot(energy, Probability[:,1,2], color='r')
ax.set_xscale('log')
ax.set_ylim((0,1))
plt.savefig('osc_test.png')
