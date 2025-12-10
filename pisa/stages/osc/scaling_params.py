"""
Author : Sharmistha Chattopadhyay
Date : August 10, 2023
"""

import numpy as np
from pisa import FTYPE

__all__ = [
    'Mass_scaling', 'Core_scaling_w_constrain', 'Core_scaling_wo_constrain',
    'FIVE_LAYER_RADII', 'FIVE_LAYER_RHOS'
]

FIVE_LAYER_RADII = np.array([0.0, 1221.50, 3480.00, 5701.00, 6151.0, 6371.00], dtype=FTYPE)
"""Radii (km) of five Earth layers to assume for the last two types of tomography."""
FIVE_LAYER_RHOS = np.array([13.0, 13.0, 10.96, 5.03, 3.7, 2.5], dtype=FTYPE)
"""Matter densities (g/cm^3) of five Earth layers to assume for the last two types of tomography."""


class Mass_scaling():
    """
    Uses a single positive scaling factor for all the layers.
    """
    def __init__(self):
        self._density_scale = 0.

    @property
    def density_scale(self):
        return self._density_scale

    @density_scale.setter
    def density_scale(self, value):
        assert value >= 0.
        self._density_scale = value


class Core_scaling_w_constrain():
    """
    Returns scaling factors for inner mantle and middle mantle by taking scaling
    factor of inner core and outer core as input.
    Scaling factor of inner and outer core = core_density_scale (alpha).
    Scaling factor of inner mantle = beta.
    Scaling factor of middle mantle = gamma.
    Outer mantle not scaled.
    This function solves the equations for two constraints: mass of earth and
    moment of inertia, by taking core_density_scale as an independent
    parameter, and returns scaling factors for inner and middle mantle.

    """
    def __init__(self):
        self._core_density_scale = 0.

    @property
    def core_density_scale(self):
        return self._core_density_scale

    @core_density_scale.setter
    def core_density_scale(self, value):
        self._core_density_scale = value

    @property
    def scaling_array(self):
        radii_cm = FIVE_LAYER_RADII * 10**5
        rho = FIVE_LAYER_RHOS

        a1 = (4*np.pi/3)*(rho[1]*radii_cm[1]**3)
        a2 = (8*np.pi/15)*(rho[1]*radii_cm[1]**5)
        b1 = (4*np.pi/3)*(rho[2]*(radii_cm[2]**3 - radii_cm--[1]**3))
        b2 = (8*np.pi/15)*(rho[2]*(radii_cm[2]**5 - radii_cm[1]**5))
        c1 = (4*np.pi/3)*(rho[3]*(radii_cm[3]**3 - radii_cm[2]**3))
        c2 = (8*np.pi/15)*(rho[3]*(radii_cm[3]**5 - radii_cm[2]**5))
        d1 = (4*np.pi/3)*(rho[4]*(radii_cm[4]**3 - radii_cm[3]**3))
        d2 = (8*np.pi/15)*(rho[4]*(radii_cm[4]**5 - radii_cm[3]**5))
        e1 = (4*np.pi/3)*(rho[5]*(radii_cm[5]**3 - radii_cm[4]**3))
        e2 = (8*np.pi/15)*(rho[5]*(radii_cm[5]**5 - radii_cm[4]**5))

        I = a2 + b2 + c2 + d2 + e2
        M = a1 + b1 + c1 + d1 + e1

        alpha = self.core_density_scale
        gamma = ((I*c1 - M*c2) - alpha*(c1*a2 - c2*a1) - alpha*(c1*b2 - b1*c2)-(c1*e2 - e1*c2))/(c1*d2 - d1*c2)
        beta = (I - alpha*a2 - alpha*b2 - gamma*d2 - e2)/c2

        # density scaling factors need to be positive
        assert (np.asarray([alpha, beta, gamma], dtype=FTYPE) >= 0).all()

        tmp_array = np.ones(6, dtype=FTYPE)
        tmp_array[1] = gamma
        tmp_array[2] = beta
        tmp_array[3] = alpha
        tmp_array[4] = alpha
        tmp_array[5] = alpha

        return tmp_array

class Core_scaling_wo_constrain():
    """
    Stores independent scaling factors for core, inner mantle and outer mantle.

    """
    def __init__(self):
        self._core_density_scale = 0.
        self._innermantle_density_scale = 0.
        self._middlemantle_density_scale = 0.

    @property
    def core_density_scale(self):
        return self._core_density_scale

    @core_density_scale.setter
    def core_density_scale(self, value):
        self._core_density_scale = value

    @property
    def innermantle_density_scale(self):
        return self._innermantle_density_scale

    @innermantle_density_scale.setter
    def innermantle_density_scale(self, value):
        self._innermantle_density_scale = value

    @property
    def middlemantle_density_scale(self):
        return self._middlemantle_density_scale

    @middlemantle_density_scale.setter
    def middlemantle_density_scale(self, value):
        self._middlemantle_density_scale = value

    @property
    def scaling_factor_array(self):
        tmp_array = np.ones(6, dtype=FTYPE)
        tmp_array[1] = self.middlemantle_density_scale
        tmp_array[2] = self.innermantle_density_scale
        tmp_array[3] = self.core_density_scale
        tmp_array[4] = self.core_density_scale
        tmp_array[5] = self.core_density_scale

        return tmp_array


def test_scaling_params():
    #TODO
    pass

if __name__=='__main__':
    test_scaling_params()
