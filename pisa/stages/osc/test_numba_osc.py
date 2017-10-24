from __future__ import print_function
import numpy as np
from numba_osc import *
from pisa.stages.osc.osc_params import OscParams
from pisa.stages.osc.layers import Layers
import time

OP = OscParams(7.5e-5, 2.524e-3, np.sqrt(0.306), np.sqrt(0.02166), np.sqrt(0.441), 261/180.*np.pi)

mix = OP.mix_matrix[:,:,0] + OP.mix_matrix[:,:,1] * 1j
dm = OP.dm_matrix
nsi_eps = np.zeros((3,3)) + np.zeros((3,3)) * 1j

points = 1000
nevts = points**2

# input arrays
# nu /nu-bar
kNuBar = np.ones(nevts, dtype=np.int32)
# flavours
kFlav = np.ones(nevts, dtype=np.int32)
energy_points = np.logspace(0,3,points)
cz_points = np.linspace(-1,1,points)
energy, cz = np.meshgrid(energy_points, cz_points)
energy = energy.ravel()
cz = cz.ravel()

earth_model = '/home/peller/cake/pisa/resources/osc/PREM_12layer.dat'
det_depth = 2
atm_height = 20
myLayers = Layers(earth_model, det_depth, atm_height)
myLayers.setElecFrac(0.4656, 0.4656, 0.4957)
myLayers.calcLayers(cz)

numberOfLayers = myLayers.n_layers
densityInLayer = myLayers.density.reshape((nevts,myLayers.max_layers))
distanceInLayer = myLayers.distance.reshape((nevts,myLayers.max_layers))

# empty arrays to be filled
Probability = np.zeros((nevts,3,3))

@guvectorize([(float64[:,:], complex128[:,:], complex128[:,:], int32, int32, float64, int32, float64[:], float64[:], float64[:,:])], '(a,b),(c,d),(e,f),(),(),(),(),(g),(h)->(a,b)', target=target)
def propagate_array(dm, mix, nsi_eps, kNuBar, kFlav, energy, numberOfLayers, densityInLayer, distanceInLayer, Probability):
    propagate_array_kernel(dm, mix, nsi_eps, kNuBar, kFlav, energy, numberOfLayers, densityInLayer, distanceInLayer, Probability)

start_t = time.time()
propagate_array(dm,
               mix,
               nsi_eps,
               kNuBar,
               kFlav,
               energy,
               numberOfLayers,
               densityInLayer,
               distanceInLayer,
               out=Probability)
end_t = time.time()
print ('%.2f s for %i events'%((end_t-start_t),nevts))

time.sleep(2)

#print(Probability[:,1,1])

# do the same with Ol' Bargy
from pisa.stages.osc.prob3.BargerPropagator import BargerPropagator
prob_e = []
prob_mu = []

barger_propagator = BargerPropagator(earth_model, det_depth)
barger_propagator.UseMassEigenstates(False)
start_t= time.time()
for c,e,kNu,kF in zip(cz,energy,kNuBar,kFlav):
    barger_propagator.SetMNS(
                        0.306,0.02166,0.441,7.5e-5,
                        2.524e-3,261/180.*np.pi,e,True,int(kNu)
                    )
    barger_propagator.DefinePath(
        c, atm_height, 0.4656, 0.4656, 0.4957
    )
    barger_propagator.propagate(int(kNu))
    prob_e.append(barger_propagator.GetProb(
        0, int(kF)
    ))
    prob_mu.append(barger_propagator.GetProb(
        1, int(kF)
    ))
end_t = time.time()
print ('%.2f s for %i events'%((end_t-start_t),nevts))
prob_mu = np.array(prob_mu)

# plot maps
import matplotlib as mpl
# Headless mode; must set prior to pyplot import
mpl.use('Agg')
from matplotlib import pyplot as plt
pmap = Probability[:,1,1].reshape((points, points))
pmap2 = prob_mu.reshape((points, points))
# numba
pcol = plt.pcolormesh(energy_points, cz_points, pmap,
                                        cmap='RdBu', linewidth=0, rasterized=True)
ax = plt.gca()
ax.set_xscale('log')
plt.savefig('osc_test_map.png')
# barger
pcol = plt.pcolormesh(energy_points, cz_points, pmap2,
                                        vmin=0, vmax=1, cmap='RdBu', linewidth=0, rasterized=True)
ax = plt.gca()
ax.set_xscale('log')
plt.savefig('osc_test_map_barger.png')
# diff map
pcol = plt.pcolormesh(energy_points, cz_points, pmap2-pmap,
                                        cmap='RdBu', linewidth=0, rasterized=True)
print('max diff = ',np.max(np.abs(pmap2-pmap)))
ax = plt.gca()
ax.set_xscale('log')
plt.savefig('osc_test_map_diff.png')
