import numpy as np

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

        sd = np.sin(self.deltacp)
        cd = np.cos(self.deltacp)

        c12 = np.sqrt(1.0-self.sin12*self.sin12)
        c23 = np.sqrt(1.0-self.sin23*self.sin23)
        c13 = np.sqrt(1.0-self.sin13*self.sin13)

        Mix[0][0][0] = c12*c13
        Mix[0][0][1] = 0.0
        Mix[0][1][0] = self.sin12*c13
        Mix[0][1][1] = 0.0
        Mix[0][2][0] = self.sin13*cd
        Mix[0][2][1] = -self.sin13*sd
        Mix[1][0][0] = -self.sin12*c23-c12*self.sin23*self.sin13*cd
        Mix[1][0][1] = -c12*self.sin23*self.sin13*sd
        Mix[1][1][0] = c12*c23-self.sin12*self.sin23*self.sin13*cd
        Mix[1][1][1] = -self.sin12*self.sin23*self.sin13*sd
        Mix[1][2][0] = self.sin23*c13
        Mix[1][2][1] = 0.0
        Mix[2][0][0] = self.sin12*self.sin23-c12*c23*self.sin13*cd
        Mix[2][0][1] = -c12*c23*self.sin13*sd
        Mix[2][1][0] = -c12*self.sin23-self.sin12*c23*self.sin13*cd
        Mix[2][1][1] = -self.sin12*c23*self.sin13*sd
        Mix[2][2][0] = c23*c13
        Mix[2][2][1] = 0.0

        return Mix

    @property
    def M_mass(self):
        dmVacVac = np.zeros((3,3))
        mVac = np.zeros(3)
        delta = 5.0e-9

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
