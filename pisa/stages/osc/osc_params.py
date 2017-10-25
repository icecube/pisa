# author: T. Ehrhardt
# date:   June 29, 2017
"""
OscParams: Characterize neutrino oscillation parameters
           (mixing angles, Dirac-type CP-violating phase, mass splittings)
"""

from __future__ import division

import numpy as np

from pisa import FTYPE

class OscParams(object):
    """
    Holds neutrino oscillation parameters, i.e., mixing angles, squared-mass
    differences, and a Dirac-type CPV phase. The neutrino mixing (PMNS) matrix
    constructed from these parameters is given in the standard parameterization.

    Parameters
    ----------
    dm_solar : float
        "Solar" mass splitting (delta M^2_{21}) expected to be given in [eV^2]

    dm_atm : float
        "Atmospheric" mass splitting (delta M^2_{3i}) expected to be given in
        [eV^2]. Note that i=2 if dm_atm > 0 (normal mass ordering), but i=1
        if dm_atm < 0 (inverted mass ordering). This follows the convention
        employed by the core libraries.

    sin12, sin13, sin23 : float
        1-2, 1-3 and 2-3 mixing angles, interpreted as sin(theta_{ij})

    deltacp : float
        Value of CPV phase in [rad]


    Attributes
    ----------
    dm_solar, dm_atm : float
        Cf. parameters

    sin12, sin13, sin23 : float
        Cf. parameters

    deltacp : float
        Cf. parameters

    mix_matrix : 3d float array of shape (3, 3, 2)
        Neutrino mixing (PMNS) matrix in standard parameterization. The third
        dimension holds the real and imaginary parts of each matrix element.

    dm_matrix : 2d float array of shape (3, 3)
        Antisymmetric matrix of squared-mass differences in vacuum

    """
    def __init__(self, dm_solar, dm_atm, sin12, sin13, sin23, deltacp):
        """Set oscillation parameters and mass splittings"""
        self.sin12 = sin12
        self.sin13 = sin13
        self.sin23 = sin23

        self.deltacp = deltacp

        # Comment BargerPropagator.cc:
        # "For the inverted Hierarchy, adjust the input
        # by the solar mixing (should be positive)
        # to feed the core libraries the correct value of m32."
        # TODO: Should we enforce `dm_solar` be positive, or warn at least?
        self.dm_solar = dm_solar
        if dm_atm < 0.0:
            self.dm_atm = dm_atm - dm_solar
        else:
            self.dm_atm = dm_atm

    @property
    def sin12(self):
        """Sine of 1-2 mixing angle"""
        return self._sin12

    @sin12.setter
    def sin12(self, value):
        assert (value*value <= 1)
        self._sin12 = value

    @property
    def sin13(self):
        """Sine of 1-3 mixing angle"""
        return self._sin13

    @sin13.setter
    def sin13(self, value):
        assert (value*value <= 1)
        self._sin13 = value

    @property
    def sin23(self):
        """Sine of 2-3 mixing angle"""
        return self._sin23

    @sin23.setter
    def sin23(self, value):
        assert (value*value <= 1)
        self._sin23 = value

    @property
    def deltacp(self):
        """CPV phase"""
        return self._deltacp

    @deltacp.setter
    def deltacp(self, value):
        self._deltacp = value

    @property
    def dm_solar(self):
        """'Solar' mass splitting"""
        return self._dm_solar

    @dm_solar.setter
    def dm_solar(self, value):
        self._dm_solar = value

    @property
    def dm_atm(self):
        """'Atmospheric' mass splitting"""
        return self._dm_atm

    @dm_atm.setter
    def dm_atm(self, value):
        self._dm_atm = value

    @property
    def mix_matrix(self):
        """Neutrino mixing matrix"""
        mix = np.zeros((3,3,2), dtype=FTYPE)

        sd = np.sin(self._deltacp)
        cd = np.cos(self._deltacp)

        c12 = np.sqrt(1.0-self._sin12*self._sin12)
        c23 = np.sqrt(1.0-self._sin23*self._sin23)
        c13 = np.sqrt(1.0-self._sin13*self._sin13)

        mix[0][0][0] = c12*c13
        mix[0][0][1] = 0.0
        mix[0][1][0] = self._sin12*c13
        mix[0][1][1] = 0.0
        mix[0][2][0] = self._sin13*cd
        mix[0][2][1] = -self._sin13*sd
        mix[1][0][0] = -self._sin12*c23-c12*self._sin23*self._sin13*cd
        mix[1][0][1] = -c12*self._sin23*self._sin13*sd
        mix[1][1][0] = c12*c23-self._sin12*self._sin23*self._sin13*cd
        mix[1][1][1] = -self._sin12*self._sin23*self._sin13*sd
        mix[1][2][0] = self._sin23*c13
        mix[1][2][1] = 0.0
        mix[2][0][0] = self._sin12*self._sin23-c12*c23*self._sin13*cd
        mix[2][0][1] = -c12*c23*self._sin13*sd
        mix[2][1][0] = -c12*self._sin23-self._sin12*c23*self._sin13*cd
        mix[2][1][1] = -self._sin12*c23*self._sin13*sd
        mix[2][2][0] = c23*c13
        mix[2][2][1] = 0.0

        return mix

    @property
    def mix_matrix_complex(self):
        ''' mixing matrix as complex 2-d array'''
        return self.mix_matrix[:,:,0] + self.mix_matrix[:,:,1] * 1.j

    @property
    def dm_matrix(self):
        """Neutrino mass splitting matrix in vacuum"""
        dmVacVac = np.zeros((3,3), dtype=FTYPE)
        mVac = np.zeros(3, dtype=FTYPE)
        delta = 5.0e-9

        mVac[0] = 0.0
        mVac[1] = self._dm_solar
        mVac[2] = self._dm_solar+self._dm_atm

        # Break any degeneracies
        if self._dm_solar == 0.0:
            mVac[0] -= delta
        if self._dm_atm == 0.0:
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
