
from __future__ import division

import numpy as np

class NSIParams(object):
    def __init__(self, eps_ee, eps_emu, eps_etau, eps_mumu, eps_mutau,
                 eps_tautau):
        """

        params:
          * eps_ij: NSI coupling strengths to use in oscillation calculation
                    (assumed to be real)

        """
        self.eps_ee = eps_ee
        self.eps_emu = eps_emu
        self.eps_etau = eps_etau
        self.eps_mumu = eps_mumu
        self.eps_mutau = eps_mutau
        self.eps_tautau = eps_tautau

    @property
    def M_eps_nsi(self):
        """Set matrix of non-standard coupling parameters. Only real, no
           imaginary part for now."""

        M_eps = np.zeros((3,3))

        M_eps[0][0] = self.eps_ee
        M_eps[1][0] = M_eps[0][1] = self.eps_emu
        M_eps[2][0] = M_eps[0][2] = self.eps_etau
        M_eps[1][1] = self.eps_mumu
        M_eps[2][2] = self.eps_tautau
        M_eps[1][2] = M_eps[2][1] = self.eps_mutau

        return M_eps
