# author: Anil Kumar (anil.kumar@desy.de)
# date:   2023

"""
DecayParams: Characterize neutrino decay parameters 
           (alpha3)
"""

from __future__ import division

import numpy as np

from pisa import FTYPE, CTYPE

__all__ = ['DecayParams']

class DecayParams(object):
    """
    Holds neutrino decay parameters, i.e., alpha

    Parameters
    ----------
    decay_alpha3: float
        expected to be given in [eV^2]


    Attributes
    ----------
    decay_alpha3 : float
        Cf. parameters
    decay_matrix : 3d complex array
    
    """
    def __init__(self):

        self._decay_alpha3 = 0.
        self._decay_matrix = np.zeros((3, 3), dtype=CTYPE)

    # --- theta12 ---
    @property
    def decay_alpha3(self):
        """alpha3"""
        return self._decay_alpha3

    @decay_alpha3.setter
    def decay_alpha3(self, value):
        self._decay_alpha3 = value
        
    @property
    def decay_matrix(self):
        """Neutrino decay matrix"""
        decay_mat = np.zeros((3, 3), dtype=CTYPE)

        decay_mat[2, 2] = 0 -self.decay_alpha3*1j

        return decay_mat
    
def test_decay_params():
    """
    # TODO: implement me!
    """
    pass


if __name__=='__main__':
    from pisa import TARGET
    from pisa.utils.log import set_verbosity, logging
    assert TARGET == 'cpu', "Cannot test functions on GPU, set PISA_TARGET to 'cpu'"
    set_verbosity(1)
    test_decay_params()