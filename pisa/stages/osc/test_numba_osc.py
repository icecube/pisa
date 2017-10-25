from __future__ import print_function

import time
import numpy as np
from numba import guvectorize

from pisa import FTYPE
from pisa.stages.osc.osc_params import OscParams
from pisa.stages.osc.layers import Layers
from prob3numba.numba_osc import *
from prob3numba.numba_tools import *

# Set up some mixing parameters
OP = OscParams(7.5e-5, 2.524e-3, np.sqrt(0.306), np.sqrt(0.02166), np.sqrt(0.441), 261/180.*np.pi)
mix = OP.mix_matrix_complex
dm = OP.dm_matrix
H_vac = OP.H_vac
nsi_eps = np.zeros_like(mix)


# number of points for E x CZ grid
points = 300
nevts = points**2

# input arrays
# nu /nu-bar
nubar = np.ones(nevts, dtype=np.int32)

energy_points = np.logspace(0,3,points, dtype=FTYPE)
cz_points = np.linspace(-1,1,points, dtype=FTYPE)
energy, cz = np.meshgrid(energy_points, cz_points)
energy = energy.ravel()
cz = cz.ravel()

# calc layers
earth_model = '/home/peller/cake/pisa/resources/osc/PREM_12layer.dat'
det_depth = 2
atm_height = 20
myLayers = Layers(earth_model, det_depth, atm_height)
myLayers.setElecFrac(0.4656, 0.4656, 0.4957)
myLayers.calcLayers(cz)
numberOfLayers = myLayers.n_layers
densityInLayer = myLayers.density.reshape((nevts,myLayers.max_layers))
distanceInLayer = myLayers.distance.reshape((nevts,myLayers.max_layers))

# empty array to be filled
Probability = np.zeros((nevts,3,3), dtype=FTYPE)

if FTYPE == np.float64:
    signature = '(f8[:,:], c16[:,:], c16[:,:], c16[:,:], i4, f8, f8[:], f8[:], f8[:,:])'
else:
    signature = '(f4[:,:], c8[:,:], c8[:,:], c8[:,:], i4, f4, f4[:], f4[:], f4[:,:])'

@guvectorize([signature], '(a,b),(c,d),(e,f),(g,h),(),(),(i),(j)->(a,b)', target=target)
def propagate_array(dm, mix, H_vac, nsi_eps, nubar, energy, densityInLayer, distanceInLayer, Probability):
    propagate_array_kernel(dm, mix, H_vac, nsi_eps, nubar, energy, densityInLayer, distanceInLayer, Probability)

start_t = time.time()
propagate_array(dm,
               mix,
               H_vac,
               nsi_eps,
               nubar,
               energy,
               densityInLayer,
               distanceInLayer,
               out=Probability)
end_t = time.time()
numba_time = end_t - start_t
print ('%.2f s for %i events'%(numba_time,nevts))

# add some sleep because of timing inconsistency
time.sleep(2)

# do the same with good Ol' Bargy
from pisa.stages.osc.prob3.BargerPropagator import BargerPropagator
prob_e = []
prob_mu = []

barger_propagator = BargerPropagator(earth_model, det_depth)
barger_propagator.UseMassEigenstates(False)
prob_e = []
prob_mu = []
start_t = time.time()
for c,e,kNu in zip(cz,energy,nubar):
    barger_propagator.SetMNS(
                        0.306,
                        0.02166,
                        0.441,7.5e-5,
                        2.524e-3,
                        261/180.*np.pi,
                        float(e),
                        True,
                        int(kNu)
                    )
    barger_propagator.DefinePath(
        float(c), atm_height, 0.4656, 0.4656, 0.4957
    )
    barger_propagator.propagate(int(kNu))
    # e -> mu
    prob_e.append(barger_propagator.GetProb(
        0, 1
    ))
    # mu -> mu
    prob_mu.append(barger_propagator.GetProb(
        1, 1
    ))
end_t = time.time()
cpp_time = end_t - start_t
print ('%.2f s for %i events'%(cpp_time,nevts))
print ('ratio numba/cpp: %.3f'%(numba_time/cpp_time))

prob_mu = np.array(prob_mu)
pmap = Probability[:,1,1].reshape((points, points))
pmap2 = prob_mu.reshape((points, points))
print('max diff = ',np.max(np.abs(pmap2-pmap)))

# plot isome maps
import matplotlib as mpl
# Headless mode; must set prior to pyplot import
mpl.use('Agg')
from matplotlib import pyplot as plt
# numba map
pcol = plt.pcolormesh(energy_points, cz_points, pmap,
                                        cmap='RdBu', linewidth=0, rasterized=True)
ax = plt.gca()
ax.set_xscale('log')
plt.savefig('osc_test_map_numba.png')
# barger
pcol = plt.pcolormesh(energy_points, cz_points, pmap2,
                                        cmap='RdBu', linewidth=0, rasterized=True)
plt.savefig('osc_test_map_barger.png')
# diff map
pcol = plt.pcolormesh(energy_points, cz_points, pmap2-pmap,
                                        cmap='RdBu', linewidth=0, rasterized=True)
plt.savefig('osc_test_map_diff.png')
