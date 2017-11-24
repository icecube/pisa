import numpy as np
from numba import guvectorize, SmartArray

from pisa import *
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import WHERE, multiply_and_scale
from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet


class genie_sys(PiStage):
    """
    stage to apply pre-calculated Genie sys

    Paramaters
    ----------
    Genie_Ma_QE : quantity (dimensionless)
    Genie_Ma_RES : quantity (dimensionless)

    Notes
    -----


    requires the following event keys
    linear_fit_maccqe : Genie CC quasi elastic linear coefficient
    quad_fit_maccqe : Genie CC quasi elastic quadratic coefficient
    linear_fit_maccres : Genie CC resonance linear coefficient
    quad_fit_maccres : Genie CC resonance quadratic coefficient


    """
    def __init__(self,
                 data=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                 ):

        expected_params = ('Genie_Ma_QE',
                           'Genie_Ma_RES',
                           )
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_keys = ('linear_fit_maccqe',
                      'quad_fit_maccqe',
                      'linear_fit_maccres',
                      'quad_fit_maccres',
                      )
        # what are keys added or altered in the calculation used during apply 
        calc_keys = ()
        # what keys are added or altered for the outputs during apply
        output_keys = ('weights',
                       )

        # init base class
        super(genie_sys, self).__init__(data=data,
                                       params=params,
                                       expected_params=expected_params,
                                       input_names=input_names,
                                       output_names=output_names,
                                       debug_mode=debug_mode,
                                       input_specs=input_specs,
                                       calc_specs=calc_specs,
                                       output_specs=output_specs,
                                       input_keys=input_keys,
                                       calc_keys=calc_keys,
                                       output_keys=output_keys,
                                       )

        assert self.input_mode is not None
        assert self.calc_mode is None
        assert self.output_mode is not None

    @profile
    def apply_function(self):

        Genie_Ma_QE = self.params.Genie_Ma_QE.m_as('dimensionless')
        Genie_Ma_RES = self.params.Genie_Ma_RES.m_as('dimensionless')

        for container in self.data:
            apply_genie_sys(Genie_Ma_QE,
                            container['linear_fit_maccqe'].get(WHERE),
                            container['quad_fit_maccqe'].get(WHERE),
                            Genie_Ma_RES,
                            container['linear_fit_maccres'].get(WHERE),
                            container['quad_fit_maccres'].get(WHERE),
                            out=container['weights'].get(WHERE),
                            )
            container['weights'].mark_changed(WHERE)


if FTYPE == np.float64:
    signature = '(f8, f8, f8, f8, f8, f8, f8[:])'
else:
    signature = '(f4, f4, f4, f4, f4, f4, f4[:])'
@guvectorize([signature], '(),(),(),(),(),()->()', target=TARGET)
def apply_genie_sys(Genie_Ma_QE, linear_fit_maccqe, quad_fit_maccqe, Genie_Ma_RES, linear_fit_maccres, quad_fit_maccres, out):
    out[0] *= 1. + (linear_fit_maccqe + quad_fit_maccqe * Genie_Ma_QE) * Genie_Ma_QE
    out[0] *= 1. + (linear_fit_maccres + quad_fit_maccres * Genie_Ma_RES) * Genie_Ma_RES
