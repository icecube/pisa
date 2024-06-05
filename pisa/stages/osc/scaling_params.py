"""
Author : Sharmistha Chattopadhyay
Date : August 10, 2023
"""

from __future__ import absolute_import,division

import numpy as np
import os
from pisa import FTYPE
import numba

if numba is None:
    class jit(object):
        """Decorator class to mimic Numba's `jit` when Numba is missing"""
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args):
            return args[0]
else:
    jit = numba.jit
    ftype = numba.typeof(FTYPE(1))


__all__ = ['Mass_scaling','Core_scaling_w_constrain','Core_scaling_wo_constrain']

class Mass_scaling():
    """
    Uses a single scaling factor for all the layers. Scaling factor can be only positive.
    """
    def __init__(self):
        self._density_scale = 0.

    @property
    def density_scale(self):
        
        return self._density_scale
    
    @density_scale.setter
    def density_scale(self, value):
        self._density_scale = value
    

class Core_scaling_w_constrain(object):
    """
    Returns scaling factors for inner mantle and middle mantle by taking scaling factor of inner core and outer core as input.
    Scaling factor of inner and outer core = core_density_scale (alpha)
    Scaling factor of inner mantle = beta
    Scaling factor of middle mantle = gamma
    Outer mantle not scaled
    This function solves the equations for two constraints: mass of earth and moment of inertia, by taking core_density_scale as an independent 
    parameter, and returns scaling factor factors for inner and middle mantle.
    
    """
    def __init__(self):
        self._core_density_scale = 0.

    @property
    def core_density_scale(self):
        
        return self._core_density_scale
    
    @core_density_scale.setter
    def core_density_scale(self, value):
        self._core_density_scale = value

    def is_positive(self,lst):
        for i in range(len(lst)):
            if lst[i] < 0:
                return False
        return True    
    
    def is_descending(self,lst):
        for i in range(len(lst) - 1):
            if lst[i] < lst[i + 1]:
                return False
        return True

    @property
    def scaling_array(self):

        radius= np.array([0.0, 1221.50, 3480.00, 5701.00, 6151.0, 6371.00])
        R = radius * 10**5

        rho = np.array([13.0, 13.0, 10.96, 5.03, 3.7, 2.5])

        a1 = (4*np.pi/3)*(rho[1]* R[1]**3)
        a2 = (8*np.pi/15)*(rho[1]* R[1]**5)
        b1 = (4*np.pi/3)*(rho[2]* (R[2]**3 - R[1]**3))
        b2 = (8*np.pi/15)*(rho[2]* (R[2]**5 - R[1]**5))
        c1 = (4*np.pi/3)*(rho[3]* (R[3]**3 - R[2]**3))
        c2 = (8*np.pi/15)*(rho[3]* (R[3]**5 - R[2]**5))
        d1 = (4*np.pi/3)*(rho[4]* (R[4]**3 - R[3]**3))
        d2 = (8*np.pi/15)*(rho[4]* (R[4]**5 - R[3]**5))
        e1 = (4*np.pi/3)*(rho[5]* (R[5]**3 - R[4]**3))
        e2 = (8*np.pi/15)*(rho[5]* (R[5]**5 - R[4]**5))

        I = a2 + b2 +c2 + d2 + e2
        M = a1 + b1 +c1 + d1 + e1

        alpha = self.core_density_scale
        

        new_rho = np.zeros(6, dtype=FTYPE)
        gamma = ((I*c1-M*c2)-alpha*(c1*a2 - c2*a1)- alpha*(c1*b2-b1*c2)-(c1*e2 - e1*c2))/(c1*d2-d1*c2)
        beta = (I - alpha * a2 - alpha * b2 - gamma*d2 - e2)/(c2)


        new_rho[0] = alpha * rho[0]
        new_rho[1] = alpha * rho[1]
        new_rho[2] = alpha * rho[2]
        new_rho[3] = beta * rho[3]
        new_rho[4] = gamma * rho[4]
        new_rho[5] = rho[5]

        tmp_array = np.ones(6,dtype=FTYPE)
        if self.is_positive(new_rho):   # and self.is_descending(new_rho): ##turn this on if you want to put hydrostatic equilibrium condition
            tmp_array[1] = gamma
            tmp_array[2] = beta
            tmp_array[3] = alpha
            tmp_array[4] = alpha
            tmp_array[5] = alpha
            
        return tmp_array

class Core_scaling_wo_constrain(object):
    """
    Takes scaling factors for core, inner mantle and outer mantle from pipeline and stores them in an array
    
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

        tmp_array = np.ones(6,dtype=FTYPE)
        tmp_array[1] = self.middlemantle_density_scale
        tmp_array[2] = self.innermantle_density_scale
        tmp_array[3] = self.core_density_scale
        tmp_array[4] = self.core_density_scale
        tmp_array[5] = self.core_density_scale
            
        return tmp_array


def test_scaling_params():
    pass

if __name__=='__main__':
    from pisa import TARGET
    from pisa.utils.log import set_verbosity, logging
    assert TARGET == 'cpu', "Cannot test functions on GPU, set PISA_TARGET to 'cpu'"
    set_verbosity(1)
    test_scaling_params()





