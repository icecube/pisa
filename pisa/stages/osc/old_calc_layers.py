import numpy as np
from pisa.stages.osc.grid_propagator.GridPropagator import GridPropagator
from pisa.utils.fileio import findFiles
from calc_layers import Layers, OscParams

# test params
cz = np.linspace(-1,1,100)

dm_s = 5e-5
dm_a = 2e-3
s12 = 0.5
s13 = 0.1
s23 = 0.7
dcp = 1

grid_prop = GridPropagator(
    'pisa/resources/osc/PREM_12layer.dat',
    cz.astype(np.float64),
    1.0
)
grid_prop.SetEarthDensityParams(2., 0.4656, 0.4656, 0.4957)
grid_prop.SetMNS(dm_s, dm_a, s12, s13, s23, dcp)

maxLayers = grid_prop.GetMaxLayers()
nczbins_fine = len(cz)
numLayers = np.zeros(nczbins_fine, dtype=np.int32)
densityInLayer = np.zeros((nczbins_fine*maxLayers),
              dtype=np.float64)
distanceInLayer = np.zeros((nczbins_fine*maxLayers),
               dtype=np.float64)

grid_prop.GetNumberOfLayers(numLayers)
grid_prop.GetDensityInLayer(densityInLayer)
grid_prop.GetDistanceInLayer(distanceInLayer)
n = maxLayers
m = len(cz)
densityInLayer = densityInLayer.reshape(m,n)
distanceInLayer = distanceInLayer.reshape(m,n)

# NEW
layer = Layers('osc/PREM_12layer.dat',1.,2.)
layer.SetElecFrac( 0.4656, 0.4656, 0.4957)
layer.ComputeMinLengthToLayers()
n_layers = []
density = []
distance = []
for coszen in cz.astype(np.float32):
    pathLength = layer.DefinePath(coszen)
    layer.SetDensityProfile(coszen, pathLength)
    n_layers.append(layer.Layers)
    density.append(layer.TraverseRhos * layer.TraverseElectronFrac)
    distance.append(layer.TraverseDistance)
new_numLayers = np.array(n_layers)
new_densityInLayer = np.vstack(density)
new_distanceInLayer = np.vstack(distance)

print 'maximum differences n layers'
print max(numLayers - new_numLayers)
print 'maximum differences density'
print np.nanmax(np.divide(np.abs(densityInLayer - new_densityInLayer),densityInLayer))
print 'maximum differences distance'
print np.nanmax(np.divide(np.abs(distanceInLayer - new_distanceInLayer),distanceInLayer))

# new osc
osc = OscParams(dm_s, dm_a, s12, s13, s23, dcp)


#compare

m_dm = np.zeros((3,3))
m_pmns = np.zeros((3,3,2))
grid_prop.Get_dm_mat(m_dm)
grid_prop.Get_mix_mat(m_pmns)

print 'maximum differences mass matrix'
print np.max(m_dm - osc.M_mass)
print 'maximum differences PMNS matrix'
print np.max(m_pmns - osc.M_pmns)

