"""
Stage to apply pre-calculated Genie uncertainties
"""

from __future__ import absolute_import, print_function, division

__all__ = ["genie_sys", "SIGNATURE", "apply_genie_sys"]

import numpy as np
from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.profiler import profile, line_profile
from pisa.utils.numba_tools import WHERE
from pisa.utils.log import logging

class genie_sys(Stage): # pylint: disable=invalid-name
    """
    Stage to apply pre-calculated Genie systematics.

    Parameters
    ----------
    params
        Must contain ::

            Genie_Ma_QE : quantity (dimensionless)
            Genie_Ma_RES : quantity (dimensionless)

    Notes
    -----
    Requires the events have the following keys ::

        linear_fit_maccqe
            Genie CC quasi elastic linear coefficient
        quad_fit_maccqe
            Genie CC quasi elastic quadratic coefficient
        linear_fit_maccres
            Genie CC resonance linear coefficient
        quad_fit_maccres
            Genie CC resonance quadratic coefficient

    """
    def __init__(
        self,
        **std_kwargs,
    ):
        expected_params = (
            'Genie_Ma_QE',
            'Genie_Ma_RES',
        )

        # init base class
        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )


    def setup_function(self):
        '''
        Check the range of the axial masses parameter
        in the analysis. Send a warning if these are beyond +- 2sigma
        '''
        if self.params['Genie_Ma_QE'].range[0]<-2. or self.params['Genie_Ma_QE'].range[1]>2.:
            logging.warn('Genie_Ma_QE parameter bounds have been set larger than the range used to produce interpolation points ([-2.,2]). This will void the warranty...')
        if self.params['Genie_Ma_RES'].range[0]<-2. or self.params['Genie_Ma_RES'].range[1]>2.:
            logging.warn('Genie_Ma_RES parameter bounds have been set larger than the range used to produce interpolation points ([-2.,2]). This will void the warranty...')




    def apply_function(self):
        genie_ma_qe = self.params.Genie_Ma_QE.m_as('dimensionless')
        genie_ma_res = self.params.Genie_Ma_RES.m_as('dimensionless')

        for container in self.data:
            apply_genie_sys(
                genie_ma_qe,
                container['linear_fit_maccqe'],
                container['quad_fit_maccqe'],
                genie_ma_res,
                container['linear_fit_maccres'],
                container['quad_fit_maccres'],
                out=container['weights'],
            )

            #
            # In cases where the axial mass is extrapolated outside
            # the range of the points used in the interpolation, some 
            # weights become negative. These are floored at 0.
            #
            container.mark_changed('weights')



if FTYPE == np.float64:
    SIGNATURE = '(f8, f8, f8, f8, f8, f8, f8[:])'
else:
    SIGNATURE = '(f4, f4, f4, f4, f4, f4, f4[:])'
@guvectorize([SIGNATURE], '(),(),(),(),(),()->()', target=TARGET)
def apply_genie_sys(
    genie_ma_qe,
    linear_fit_maccqe,
    quad_fit_maccqe,
    genie_ma_res,
    linear_fit_maccres,
    quad_fit_maccres,
    out,
):
    out[0] *= max(0, (
        (1. + (linear_fit_maccqe + quad_fit_maccqe * genie_ma_qe) * genie_ma_qe)
        * (1. + (linear_fit_maccres + quad_fit_maccres * genie_ma_res) * genie_ma_res)
    ))
