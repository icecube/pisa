'''
Test functions used during developpement of the osc. code
Please ignore unless you are the author
'''
from __future__ import print_function

import time
import numpy as np
from numba import guvectorize, SmartArray

from pisa import FTYPE
from pisa.stages.osc.osc_params import OscParams
from pisa.stages.osc.layers import Layers
from prob3numba.numba_osc import *
from prob3numba.numba_tools import *

# Set up some mixing parameters
OP = OscParams(7.5e-5, 2.524e-3, np.sqrt(0.306), np.sqrt(0.02166), np.sqrt(0.441), 261/180.*np.pi)
mix = OP.mix_matrix_complex
dm = OP.dm_matrix
nsi_eps = np.zeros_like(mix)



# calc layers
earth_model = '/home/peller/cake/pisa/resources/osc/PREM_59layer.dat'
det_depth = 2
atm_height = 20
myLayers = Layers(earth_model, det_depth, atm_height)
myLayers.setElecFrac(0.4656, 0.4656, 0.4957)
myLayers.calcLayers(cz)
numberOfLayers = myLayers.n_layers
densities = myLayers.density.reshape((nevts,myLayers.max_layers))
densities = SmartArray(densities)
distances = myLayers.distance.reshape((nevts,myLayers.max_layers))
distances = SmartArray(distances)

# empty array to be filled
probability = np.zeros((nevts,3,3), dtype=FTYPE)
probability = SmartArray(probability)
probability_vacuum = np.zeros((nevts,3,3), dtype=FTYPE)

if FTYPE == np.float64:
    signature = '(f8[:,:], c16[:,:], c16[:,:], i4, f8, f8[:], f8[:], f8[:,:])'
    signature_vac = '(f8[:,:], c16[:,:], i4, f8, f8[:], f8[:,:])'
else:
    signature = '(f4[:,:], c8[:,:], c8[:,:], i4, f4, f4[:], f4[:], f4[:,:])'
    signature_vac = '(f4[:,:], c8[:,:], i4, f4, f4[:], f4[:,:])'

@guvectorize([signature], '(a,b),(c,d),(e,f),(),(),(g),(h)->(a,b)', target=target)
def propagate_array(dm, mix, nsi_eps, nubar, energy, densities, distances, probability):
    osc_probs_layers_kernel(dm, mix, nsi_eps, nubar, energy, densities, distances, probability)

@guvectorize([signature_vac], '(a,b),(c,d),(),(),(i)->(a,b)', target=target)
def propagate_array_vacuum(dm, mix, nubar, energy, distances, probability):
    osc_probs_vacuum_kernel(dm, mix, nubar, energy, distances, probability)


if target == 'cuda':
    where='gpu'
else:
    where='host'
# first loop
start_t = time.time()
propagate_array(dm,
               mix,
               nsi_eps,
               nubar,
               energy.get(where),
               densities.get(where),
               distances.get(where),
               out=probability.get(where))
end_t = time.time()
numba_time = end_t - start_t
print ('%.2f s for %i events first loop'%(numba_time,nevts))
probability.mark_changed(where)
# second loop
start_t = time.time()
propagate_array(dm,
               mix,
               nsi_eps,
               nubar,
               energy.get(where),
               densities.get(where),
               distances.get(where),
               out=probability.get(where))
probability.mark_changed(where)
end_t = time.time()

numba_time = end_t - start_t
print ('%.2f s for %i events'%(numba_time,nevts))

start_t = time.time()
propagate_array_vacuum(dm,
               mix,
               nubar,
               energy,
               distances,
               out=probability_vacuum)
end_t = time.time()
numba_vac_time = end_t - start_t
print ('%.2f s for %i events'%(numba_vac_time,nevts))

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
pmap = probability.get('host')[:,1,1].reshape((points, points))
pmap_vac = probability_vacuum[:,1,1].reshape((points, points))
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
# vacuum
pcol = plt.pcolormesh(energy_points, cz_points, pmap_vac,
                                        cmap='RdBu', linewidth=0, rasterized=True)
plt.savefig('osc_test_map_numba_vacuum.png')
# matter effects
pcol = plt.pcolormesh(energy_points, cz_points, pmap - pmap_vac,
                                        cmap='RdBu', linewidth=0, rasterized=True)
plt.savefig('osc_test_map_numba_matter_vs_vacuum.png')
# barger
pcol = plt.pcolormesh(energy_points, cz_points, pmap2,
                                        cmap='RdBu', linewidth=0, rasterized=True)
plt.savefig('osc_test_map_barger.png')
# diff map
pcol = plt.pcolormesh(energy_points, cz_points, pmap2-pmap,
                                        cmap='RdBu', linewidth=0, rasterized=True)
plt.savefig('osc_test_map_diff.png')
