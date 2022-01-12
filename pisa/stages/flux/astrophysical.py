"""
stage to implement getting the contribution to fluxes from astrophysical neutrino sources
"""
import numpy as np
from numba import guvectorize, cuda

from pisa.utils.profiler import profile 
from pisa import FTYPE, TARGET
from pisa.core.stage import Stage

class astro_simple(Stage):
    """
    Stage to apply power law astrophysical fluxes 

    Parameters
    ----------
    params
        Expected params are .. ::
            astro_delta : quantity (dimensionless)
            astro_norm : quantity (dimensionless)

    TODO: flavor ratio quantity? 
    """

    def __init__(self, **std_kwargs):
        expected_params = ("astro_delta",
                           "astro_norm")

        super(astro_simple, self).__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

    def setup_function(self):
        self.data.representation = self.calc_mode
        for container in self.data:
            container["astro_weights"] = np.empty((container.size, 2), dtype=FTYPE)


    @profile
    def compute_function(self):
        self.data.representation = self.calc_mode

        delta = self.params.astro_delta.value.m_as("dimensionless")
        norm = self.params.astro_norm.value.m_as("astro_norm")

        for container in self.data:
            pass

        