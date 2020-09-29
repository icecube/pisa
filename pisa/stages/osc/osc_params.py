# author: T. Ehrhardt
# date:   2018
"""
OscParams: Characterize neutrino oscillation parameters
           (mixing angles, Dirac-type CP-violating phase, mass splittings)

changed by Elisa Lohfink (ellohfin; elohfink@icecube.wisc.edu) 
to include NSI changes made by Thomas Ehrhardt on his branch:  
original version can be found in thehrh/pisa nsi_reparameterisation branch 
"""

from __future__ import division

import numpy as np

from pisa import FTYPE

__all__ = ['OscParams']


class OscParams(object):
    """
    Holds neutrino oscillation parameters, i.e., mixing angles, squared-mass
    differences, and a Dirac-type CPV phase. The neutrino mixing (PMNS) matrix
    constructed from these parameters is given in the standard
    3x3 parameterization. Also holds the generalised matter potential matrix
    (divided by the matter potential a), i.e. diag(1, 0, 0) for the standard
    case.

    Parameters
    ----------
    dm21, dm31, dm41 : float
        Mass splittings (delta M^2_{21,31,41}) expected to be given in [eV^2]

    sin12, sin13, sin23 : float
        1-2, 1-3 and 2-3 mixing angles, interpreted as sin(theta_{ij})

    deltacp : float
        Value of CPV phase in [rad]


    Attributes
    ----------
    dm21, dm31, dm41 : float
        Cf. parameters

    sin12, sin13, sin23, sin14 : float
        Cf. parameters

    theta12, theta13, theta23, theta14 : float
        Mixing angles (corresponding to sinXY)

    deltacp : float
        Cf. parameters

    mix_matrix : 3d float array of shape (3, 3, 2)
        Neutrino mixing (PMNS) matrix in standard parameterization. The third
        dimension holds the real and imaginary parts of each matrix element.

    mix_matrix_complex : 3d complex array

    mix_matrix_reparam : 3d float array of shape (3, 3, 2)
        Reparameterized neutrino mixing matrix, such that CPT invariance
        of vacuum propagation implemented by 3 simultaneous osc. param.
        transformations.

    mix_matrix_reparam_complex : 3d complex array

    dm_matrix : 2d float array of shape (3, 3)
        Antisymmetric matrix of squared-mass differences in vacuum

    """
    def __init__(self):

        self._sin12 = 0.
        self._sin13 = 0.
        self._sin23 = 0.
        self._sin14 = 0.
        self._deltacp = 0.
        self.dm21 = 0.
        self.dm31 = 0.
        self.dm41 = 0.
        self.gamma21 = 0. # TODO Add full 3x3 matrix option, TODO update docs, TODO getters/setters to enforce values ranges?
        self.gamma31 = 0.
        self.gamma32 = 0.

    # --- theta12 ---
    @property
    def sin12(self):
        """Sine of 1-2 mixing angle"""
        return self._sin12

    @sin12.setter
    def sin12(self, value):
        assert (abs(value) <= 1)
        self._sin12 = value

    @property
    def theta12(self):
        return np.arcsin(self.sin12)

    @theta12.setter
    def theta12(self, value):
        self.sin12 = np.sin(value)

    # --- theta13 ---
    @property
    def sin13(self):
        """Sine of 1-3 mixing angle"""
        return self._sin13

    @sin13.setter
    def sin13(self, value):
        assert (abs(value) <= 1)
        self._sin13 = value

    @property
    def theta13(self):
        return np.arcsin(self.sin13)

    @theta13.setter
    def theta13(self, value):
        self.sin13 = np.sin(value)

    # --- theta23 ---
    @property
    def sin23(self):
        """Sine of 2-3 mixing angle"""
        return self._sin23

    @sin23.setter
    def sin23(self, value):
        assert (abs(value) <= 1)
        self._sin23 = value

    @property
    def theta23(self):
        return np.arcsin(self.sin23)

    @theta23.setter
    def theta23(self, value):
        self.sin23 = np.sin(value)

    # --- theta14 ---
    @property
    def sin14(self):
        """Sine of 1-4 mixing angle"""
        return self._sin14

    @sin14.setter
    def sin14(self, value):
        assert (abs(value) <= 1)
        self._sin14 = value

    @property
    def theta14(self):
        return np.arcsin(self.sin14)

    @theta14.setter
    def theta14(self, value):
        self.sin14 = np.sin(value)

    # --- deltaCP ---
    @property
    def deltacp(self):
        """CPV phase"""
        return self._deltacp

    @deltacp.setter
    def deltacp(self, value):
        assert value >= 0. and value <= 2*np.pi
        self._deltacp = value

    @property
    def mix_matrix(self):
        """Neutrino mixing matrix in its 'standard' form"""
        mix = np.zeros((3, 3, 2), dtype=FTYPE)

        sd = np.sin(self.deltacp)
        cd = np.cos(self.deltacp)

        c12 = np.sqrt(1. - self.sin12**2)
        c23 = np.sqrt(1. - self.sin23**2)
        c13 = np.sqrt(1. - self.sin13**2)

        mix[0, 0, 0] = c12 * c13
        mix[0, 0, 1] = 0.
        mix[0, 1, 0] = self.sin12 * c13
        mix[0, 1, 1] = 0.
        mix[0, 2, 0] = self.sin13 * cd
        mix[0, 2, 1] = - self.sin13 * sd
        mix[1, 0, 0] = - self.sin12 * c23 - c12 * self.sin23 * self.sin13 * cd
        mix[1, 0, 1] = - c12 * self.sin23 * self.sin13 * sd
        mix[1, 1, 0] = c12 * c23 - self.sin12 * self.sin23 * self.sin13 * cd
        mix[1, 1, 1] = - self.sin12 * self.sin23 * self.sin13 * sd
        mix[1, 2, 0] = self.sin23 * c13
        mix[1, 2, 1] = 0.
        mix[2, 0, 0] = self.sin12 * self.sin23 - c12 * c23 * self.sin13 * cd
        mix[2, 0, 1] = - c12 * c23 * self.sin13 * sd
        mix[2, 1, 0] = - c12 * self.sin23 - self.sin12 * c23 * self.sin13 * cd
        mix[2, 1, 1] = - self.sin12 * c23 * self.sin13 * sd
        mix[2, 2, 0] = c23 * c13
        mix[2, 2, 1] = 0.

        return mix

    @property
    def mix_matrix_complex(self):
        """Mixing matrix as complex 2-d array"""
        mix = self.mix_matrix
        return mix[:, :, 0] + mix[:, :, 1] * 1.j

    @property
    def mix_matrix_reparam(self):
        """
        Neutrino mixing matrix reparameterised in a way
        such that the CPT trafo Hvac -> -Hvac*  is exactly implemented by
        the simultaneous transformations
            * deltamsq31 -> -deltamsq32
            * theta12 -> pi/2 - theta12
            * deltacp -> pi - deltacp

        which hence leave vacuum propagation invariant.

        This representation follows from the standard form U
        as diag(exp(i*deltacp), 0, 0) * U * diag(exp(-i*deltacp), 0, 0).

        """
        mix = np.zeros((3, 3, 2), dtype=FTYPE)

        sd = np.sin(self.deltacp)
        cd = np.cos(self.deltacp)

        c12 = np.sqrt(1. - self.sin12**2)
        c23 = np.sqrt(1. - self.sin23**2)
        c13 = np.sqrt(1. - self.sin13**2)

        mix[0, 0, 0] = c12 * c13
        mix[0, 0, 1] = 0.
        mix[0, 1, 0] = self.sin12 * c13 * cd
        mix[0, 1, 1] = self.sin12 * c13 * sd
        mix[0, 2, 0] = self.sin13
        mix[0, 2, 1] = 0.
        mix[1, 0, 0] = - self.sin12 * c23 * cd - c12 * self.sin23 * self.sin13
        mix[1, 0, 1] = self.sin12 * c23 * sd
        mix[1, 1, 0] = c12 * c23 - self.sin12 * self.sin23 * self.sin13 * cd
        mix[1, 1, 1] = - self.sin12 * self.sin23 * self.sin13 * sd
        mix[1, 2, 0] = self.sin23 * c13
        mix[1, 2, 1] = 0.
        mix[2, 0, 0] = self.sin12 * self.sin23 * cd - c12 * c23 * self.sin13
        mix[2, 0, 1] = - self.sin12 * self.sin23 * sd
        mix[2, 1, 0] = - c12 * self.sin23 - self.sin12 * c23 * self.sin13 * cd
        mix[2, 1, 1] = - self.sin12 * c23 * self.sin13 * sd
        mix[2, 2, 0] = c23 * c13
        mix[2, 2, 1] = 0.

        return mix

    @property
    def mix_matrix_reparam_complex(self):
        """Reparameterised mixing matrix as complex 2-d array"""
        mix_reparam = self.mix_matrix_reparam
        return mix_reparam[:, :, 0] + mix_reparam[:, :, 1] * 1.j

    @property
    def dm_matrix(self):
        """Neutrino mass splitting matrix in vacuum"""
        dmVacVac = np.zeros((3, 3), dtype=FTYPE)
        mVac = np.zeros(3, dtype=FTYPE)
        delta = 5.e-9

        mVac[0] = 0.
        mVac[1] = self.dm21
        mVac[2] = self.dm31

        # Break any degeneracies
        if mVac[1] == 0.:
            mVac[0] -= delta
        if mVac[2] == 0.:
            mVac[2] += delta

        dmVacVac[0, 0] = 0.
        dmVacVac[1, 1] = 0.
        dmVacVac[2, 2] = 0.
        dmVacVac[0, 1] = mVac[0] - mVac[1]
        dmVacVac[1, 0] = - dmVacVac[0, 1]
        dmVacVac[0, 2] = mVac[0] - mVac[2]
        dmVacVac[2, 0] = - dmVacVac[0, 2]
        dmVacVac[1, 2] = mVac[1] - mVac[2]
        dmVacVac[2, 1] = - dmVacVac[1, 2]

        return dmVacVac


def test_osc_params():
    """
    # TODO: implement me!
    """
    pass


if __name__=='__main__':
    from pisa import TARGET
    from pisa.utils.log import set_verbosity, logging
    assert TARGET == 'cpu', "Cannot test functions on GPU, set PISA_TARGET to 'cpu'"
    set_verbosity(1)
    test_osc_params()

