# author: T. Ehrhardt
# date:   June 28, 2017
"""
NSIParams: Characterize non-standard neutrino interaction coupling strengths
"""

from __future__ import division

import numpy as np

class NSIParams(object):
    """
    Holds non-standard neutrino interaction parameters of neutral current type
    for propagating neutrinos, interacting with a 1st generation Standard Model
    quark (u or d). The NSI matrix is assumed to be symmetric
    (real-valued coupling strengths in addition to Hermiticity of matter NSIs).

    Parameters
    ----------
    eps_ee, eps_emu, eps_etau, eps_mumu, eps_mutau, eps_tautau : float
        Coupling parameters describing strengths of NSI transitions between
        the two specified neutrino flavors, via NC-type interaction with one 1st
        generation SM quark.


    Attributes
    ----------
    eps_ee, eps_emu, eps_etau, eps_mumu, eps_mutau, eps_tautau
        Cf. parameters

    eps_matrix : 2d float array of shape (3, 3)
        Symmetric NSI matrix holding the epsilon parameters.

    """

    def __init__(self, eps_ee, eps_emu, eps_etau, eps_mumu, eps_mutau,
                 eps_tautau):
        """Set NSI parameters."""
        self._eps_matrix = np.zeros((3, 3))
        self._eps_type = (float,)
        self.eps_ee = eps_ee
        self.eps_emu = eps_emu
        self.eps_etau = eps_etau
        self.eps_mumu = eps_mumu
        self.eps_mutau = eps_mutau
        self.eps_tautau = eps_tautau

    @property
    def eps_type(self):
        """Allowed numerical type of NSI coupling parameters."""
        return self._eps_type

    @property
    def eps_matrix(self):
        """Symmetric matrix of NSI coupling parameters."""
        return self._eps_matrix

    @property
    def eps_ee(self):
        """nue-nue NSI coupling parameter"""
        return self._eps_ee

    @eps_ee.setter
    def eps_ee(self, value):
        assert isinstance(value, self.eps_type)
        self._eps_matrix[0][0] = self._eps_ee = value

    @property
    def eps_emu(self):
        """nue-numu NSI coupling parameter"""
        return self._eps_emu

    @eps_emu.setter
    def eps_emu(self, value):
        assert isinstance(value, self.eps_type)
        self._eps_matrix[1][0] = self._eps_matrix[0][1] = self._eps_emu = value

    @property
    def eps_etau(self):
        """nue-nutau NSI coupling parameter"""
        return self._eps_etau

    @eps_etau.setter
    def eps_etau(self, value):
        assert isinstance(value, self.eps_type)
        self._eps_matrix[2][0] = self._eps_matrix[0][2] = self._eps_etau = value

    @property
    def eps_mumu(self):
        """numu-numu NSI coupling parameter"""
        return self._eps_mumu

    @eps_mumu.setter
    def eps_mumu(self, value):
        assert isinstance(value, self.eps_type)
        self._eps_matrix[1][1] = self._eps_mumu = value

    @property
    def eps_mutau(self):
        """numu-nutau NSI coupling parameter"""
        return self._eps_mutau

    @eps_mutau.setter
    def eps_mutau(self, value):
        assert isinstance(value, self.eps_type)
        self._eps_matrix[1][2] = self._eps_matrix[2][1] = self._eps_mutau = value

    @property
    def eps_tautau(self):
        """nutau-nutau NSI coupling parameter"""
        return self._eps_tautau

    @eps_tautau.setter
    def eps_tautau(self, value):
        assert isinstance(value, self.eps_type)
        self._eps_matrix[2][2] = self._eps_tautau = value

