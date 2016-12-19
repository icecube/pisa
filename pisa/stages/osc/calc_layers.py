from collections import Iterable, Iterator, Mapping, OrderedDict, Sequence

import numpy as np

from pisa.utils.fileio import from_file
from pisa.utils.log import logging, set_verbosity


class Layers(object):
    def __init__(self, prem_file, DetectorDepth=1., prop_height=2.):
        prem = from_file(prem_file, as_array=True)
        self.default_elec_frac = 0.5
        self.prop_height = prop_height
        self.DetectorDepth = DetectorDepth
        rEarth = prem[-1][0]
        self.rDetector = rEarth - DetectorDepth
        self.Rhos = prem[...,1][::-1]
        self.Radii = prem[...,0][::-1]
        self.MinDetectorDepth = 1.0e-3 # <-- Why? // [km] so min is ~ 1 m
        MaxDepth = len(self.Radii) - 1
        self.YeOuterRadius = np.array([1121.5, 3480.0, self.rDetector])
        self.max_layers = 2 * MaxDepth + 1
        if (self.DetectorDepth >= self.MinDetectorDepth):
            self.Radii[0] -= DetectorDepth
            self.max_layers += 1

    def SetElecFrac(self, YeI, YeO, YeM):
        self.YeFrac = np.array([YeI, YeO, YeM])

    def DefinePath(self, coszen):
        if coszen < 0:
            return np.sqrt(np.square((self.rDetector + self.prop_height + self.DetectorDepth))
                                  - np.square(self.rDetector) * (1 - np.square(coszen))) - self.rDetector * coszen
        else:
            kappa = (self.DetectorDepth + self.prop_height)/self.rDetector
            return self.rDetector * np.sqrt(np.square(coszen) - 1 + np.square(1 + kappa)) - self.rDetector * coszen

    def ComputeMinLengthToLayers(self):
        self.coszen_limit = OrderedDict()
        #first element of self.Radii is largest radius!
        for i,rad in enumerate(self.Radii):
            # Using a cosine threshold instead!
            if i == 0:
                x = 0
            else:
                x = - np.sqrt(1 - (np.square(rad) / np.square(self.rDetector)))
            self.coszen_limit[rad] = x

    def SetDensityProfile(self, coszen, PathLength):

        TotalEarthLength = -2.0 * coszen * self.rDetector

        self.TraverseRhos = np.zeros(self.max_layers)
        self.TraverseDistance = np.zeros(self.max_layers)
        self.TraverseElectronFrac = np.zeros(self.max_layers)

        if coszen >= 0:
            #Path through the air:
            kappa = self.DetectorDepth/self.rDetector
            lam = coszen + np.sqrt(np.square(coszen) - 1 + (1 + kappa) * (1 + kappa))
            lam *= self.rDetector
            pathThroughAtm = (self.prop_height * (self.prop_height + 2. * self.DetectorDepth +
                                    2.0*self.rDetector))/(PathLength + lam)
            pathThroughOuterLayer = PathLength - pathThroughAtm
            self.TraverseRhos[0] = 0.0
            self.TraverseDistance[0] = pathThroughAtm
            self.TraverseElectronFrac[0] = self.default_elec_frac
            self.Layers = 1

            if self.DetectorDepth > self.MinDetectorDepth:
                self.TraverseRhos[1] = self.Rhos[0]
                self.TraverseDistance[1] = pathThroughOuterLayer
                self.TraverseElectronFrac[1] = self.YeFrac[-1]
                self.Layers+=1

        else:
            # path through air
            self.TraverseRhos[0] = 0.
            self.TraverseDistance[0] = self.prop_height * (self.prop_height + self.DetectorDepth +
                                                     2. * self.rDetector) / PathLength
            self.TraverseElectronFrac[0] = self.default_elec_frac
            iTrav = 1

            # path through the final layer above the detector (if necessary)
            # Note: outer top layer is assumed to be the same as the next layer inward.
            if (self.DetectorDepth > self.MinDetectorDepth):
                self.TraverseRhos[1] = self.Rhos[0]
                self.TraverseDistance[1] = PathLength - TotalEarthLength - self.TraverseDistance[0]
                self.TraverseElectronFrac[1] = self.YeFrac[-1]
                iTrav += 1

            self.Layers = 0
            for val in self.coszen_limit.values():
                if coszen < val:
                    self.Layers += 1

            MaxLayer = self.Layers

            # the zeroth layer is the air!
            # and the first layer is the top layer (if detector is not on surface)
            for i in range(MaxLayer):
                self.TraverseRhos[i+iTrav] = self.Rhos[i]
                # ToDo why default?
                self.TraverseElectronFrac[i+iTrav] = self.default_elec_frac
                for iRad in range(len(self.YeOuterRadius)):
                    # why 1.001?
                    if self.Radii[i] < (self.YeOuterRadius[iRad] * 1.001):
                        self.TraverseElectronFrac[i+iTrav] = self.YeFrac[iRad]
                        break

                c2 = np.square(coszen)
                R2 = np.square(self.rDetector)
                s1 = np.square(self.Radii[i]) - R2*(1 -c2)
                s2 = np.square(self.Radii[i+1]) - R2*(1 -c2)
                CrossThis = 2. * np.sqrt(s1)
                if i < MaxLayer - 1:
                    CrossNext = 2. * np.sqrt(s2)
                    self.TraverseDistance[i+iTrav]  =  0.5 * (CrossThis - CrossNext)
                else:
                    self.TraverseDistance[i+iTrav]  =  CrossThis

                #assumes azimuthal symmetry
                if 0 < i and i < MaxLayer:
                    index = 2 * MaxLayer - i + iTrav - 1
                    self.TraverseRhos[index] = self.TraverseRhos[i+iTrav-1]
                    self.TraverseDistance[index] = self.TraverseDistance[i+iTrav-1]
                    self.TraverseElectronFrac[index] = self.TraverseElectronFrac[i+iTrav-1]

            self.Layers = 2 * MaxLayer + iTrav - 1


class OscParams(object):
    def __init__(self, dm_solar, dm_atm, x12, x13, x23, deltacp):
        """
          Expects dm_solar and dm_atm to be in [eV^2], and x_{ij} to be
          sin^2(theta_{ij})

          params:
            * xij - sin^2(theta_{ij}) values to use in oscillation calc.
            * dm_solar - delta M_{21}^2 value [eV^2]
            * dm_atm - delta M_{32}^2 value [eV^2] if Normal hierarchy, or
                      delta M_{31}^2 value if Inverted Hierarchy (following
                      BargerPropagator class).
            * deltacp - \delta_{cp} value to use.
        """
        assert x12 <= 1
        assert x13 <= 1
        assert x23 <= 1
        self.sin12 = np.sqrt(x12)
        self.sin13 = np.sqrt(x13)
        self.sin23 = np.sqrt(x23)

        self.deltacp = deltacp

        # Comment BargerPropagator.cc:
        # "For the inverted Hierarchy, adjust the input
        # by the solar mixing (should be positive)
        # to feed the core libraries the correct value of m32."
        self.dm_solar = dm_solar
        if dm_atm < 0.0:
            self.dm_atm = dm_atm - dm_solar
        else:
            self.dm_atm = dm_atm

    @property
    def M_pmns(self):

        # real part [...,0]
        # imaginary part [...,1]
        Mix = np.zeros((3,3,2))

        sd = np.sin( self.deltacp )
        cd = np.cos( self.deltacp )

        c12 = np.sqrt(1.0-self.sin12*self.sin12)
        c23 = np.sqrt(1.0-self.sin23*self.sin23)
        c13 = np.sqrt(1.0-self.sin13*self.sin13)

        Mix[0][0][0] =  c12*c13
        Mix[0][0][1] =  0.0
        Mix[0][1][0] =  self.sin12*c13
        Mix[0][1][1] =  0.0
        Mix[0][2][0] =  self.sin13*cd
        Mix[0][2][1] = -self.sin13*sd
        Mix[1][0][0] = -self.sin12*c23-c12*self.sin23*self.sin13*cd
        Mix[1][0][1] =         -c12*self.sin23*self.sin13*sd
        Mix[1][1][0] =  c12*c23-self.sin12*self.sin23*self.sin13*cd
        Mix[1][1][1] =         -self.sin12*self.sin23*self.sin13*sd
        Mix[1][2][0] =  self.sin23*c13
        Mix[1][2][1] =  0.0
        Mix[2][0][0] =  self.sin12*self.sin23-c12*c23*self.sin13*cd
        Mix[2][0][1] =         -c12*c23*self.sin13*sd
        Mix[2][1][0] = -c12*self.sin23-self.sin12*c23*self.sin13*cd
        Mix[2][1][1] =         -self.sin12*c23*self.sin13*sd
        Mix[2][2][0] =  c23*c13
        Mix[2][2][1] =  0.0

        return Mix

    @property
    def M_mass(self):
        dmVacVac = np.zeros((3,3))
        mVac = np.zeros(3)
        delta=5.0e-9

        mVac[0] = 0.0
        mVac[1] = self.dm_solar
        mVac[2] = self.dm_solar+self.dm_atm

        # Break any degeneracies
        if self.dm_solar == 0.0:
            mVac[0] -= delta
        if self.dm_atm == 0.0:
            mVac[2] += delta

        dmVacVac[0][0] = 0.
        dmVacVac[1][1] = 0.
        dmVacVac[2][2] = 0.
        dmVacVac[0][1] = mVac[0]-mVac[1]
        dmVacVac[1][0] = -dmVacVac[0][1]
        dmVacVac[0][2] = mVac[0]-mVac[2]
        dmVacVac[2][0] = -dmVacVac[0][2]
        dmVacVac[1][2] = mVac[1]-mVac[2]
        dmVacVac[2][1] = -dmVacVac[1][2]

        return dmVacVac


if __name__ == '__main__':
    layer = Layers('osc/PREM_4layer.dat')
    layer.SetElecFrac( 0.4656, 0.4656, 0.4957)
    cz = np.linspace(-1,1,11)
    layer.ComputeMinLengthToLayers()
    n_layers = []
    density = []
    distance = []
    for coszen in cz:
        pathLength = layer.DefinePath(coszen)
        layer.SetDensityProfile(coszen, pathLength)
        n_layers.append(layer.Layers)
        density.append(layer.TraverseRhos * layer.TraverseElectronFrac)
        distance.append(layer.TraverseDistance)
    logging.debug(str(np.array(n_layers)))
    logging.debug(str(np.vstack(density)))
    logging.debug(str(np.vstack(distance)))
