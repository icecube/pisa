"""
LRI Params: Charecterize Long Range Interaction mediator

Developed by following osc_params.py and nsi_params.py
"""

from __future__ import absolute_import, division

import numpy as np
from pisa import CTYPE, FTYPE
from pisa.utils.comparisons import ALLCLOSE_KW, isscalar, recursiveEquality
from pisa.utils.log import Levels, set_verbosity, logging

__all__ = ['LRIParams']

class LRIParams(object):
    """
    Holds the mediator information of long range interaction:z'
    Assumed three anamoly free symmetries, Le_mu, Le_tau, Lmu_tau(by mixing z and z')).
    
    Attributes
    ----------
    potential matrix : Three 2d float array of shape (3,3), one for each symmetry
    
    Potential matrix holding the potential term of three different symmetris, which is a 
    function of mediator mass, and the coupling constant of the interaction.
    
    
    """
    
    def __init__(self):
        
        self._v_lri = 0.
        self._potential_matrix_emu = np.zeros((3, 3), dtype=FTYPE)
        self._potential_matrix_etau = np.zeros((3, 3), dtype=FTYPE)
        self._potential_matrix_mutau = np.zeros((3, 3), dtype=FTYPE)
    
    # --- LRI potential ---
    
    @property
    def v_lri(self):
        """Potential term of symmetry e mu"""
        return self._v_lri
    
    @v_lri.setter
    def v_lri(self, value):
        assert value <1.
        self._v_lri = value
    
    @property
    def potential_matrix_emu(self):
        """LRI matter interaction potential matrix e mu symmetry"""
        
        v_matrix = np.zeros((3, 3), dtype=FTYPE)
        
        v_matrix[0, 0] = self.v_lri
        v_matrix[0, 1] = 0.
        v_matrix[0, 2] = 0.
        v_matrix[1, 0] = 0.
        v_matrix[1, 1] = - self.v_lri
        v_matrix[1, 2] = 0.
        v_matrix[2, 0] = 0.
        v_matrix[2, 1] = 0.
        v_matrix[2, 2] = 0.

        assert np.allclose(v_matrix, v_matrix.conj().T, **ALLCLOSE_KW)
        
        return v_matrix
    
    @property
    def potential_matrix_etau(self):
        """LRI matter interaction potential matrix e tau symmetry"""
        
        v_matrix = np.zeros((3, 3), dtype=FTYPE)
        
        v_matrix[0, 0] = self.v_lri
        v_matrix[0, 1] = 0.
        v_matrix[0, 2] = 0.
        v_matrix[1, 0] = 0.
        v_matrix[1, 1] = 0.
        v_matrix[1, 2] = 0.
        v_matrix[2, 0] = 0.
        v_matrix[2, 1] = 0.
        v_matrix[2, 2] = - self.v_lri

        assert np.allclose(v_matrix, v_matrix.conj().T, **ALLCLOSE_KW)
        
        return v_matrix
    
    @property
    def potential_matrix_mutau(self):
        """LRI matter interaction potential matrix mu tau symmetry"""
        
        v_matrix = np.zeros((3, 3), dtype=FTYPE)
        
        v_matrix[0, 0] = 0.
        v_matrix[0, 1] = 0.
        v_matrix[0, 2] = 0.
        v_matrix[1, 0] = 0.
        v_matrix[1, 1] = self.v_lri
        v_matrix[1, 2] = 0.
        v_matrix[2, 0] = 0.
        v_matrix[2, 1] = 0.
        v_matrix[2, 2] = - self.v_lri

        assert np.allclose(v_matrix, v_matrix.conj().T, **ALLCLOSE_KW)
        
        return v_matrix
    
    
def test_lri_params():
    """
    # TODO: implement me!
    """
    pass

if __name__=='__main__':
    from pisa import TARGET
    from pisa.utils.log import set_verbosity, logging
    set_verbosity(1)
    test_lri_params()
