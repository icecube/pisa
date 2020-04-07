"""
Stage to apply pre-calculated Genie uncertainties
"""

from __future__ import absolute_import, print_function, division

__all__ = ["genie_sys", "SIGNATURE", "apply_genie_sys"]

import numpy as np
from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import WHERE


class genie_sys(PiStage): # pylint: disable=invalid-name
    """
    Stage to apply pre-calculated Genie systematics.

    Parameters
    ----------
    data
    params
        Must contain ::

            Genie_Ma_QE : quantity (dimensionless)
            Genie_Ma_RES : quantity (dimensionless)

    input_names
    output_names
    debug_mode
    input_specs
    calc_specs
    output_specs

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
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        input_specs=None,
        calc_specs=None,
        output_specs=None,
    ):
        expected_params = (
            'Genie_Ma_QE',
            'Genie_Ma_RES',
        )
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_apply_keys = (
            'linear_fit_maccqe',
            'quad_fit_maccqe',
            'linear_fit_maccres',
            'quad_fit_maccres',
        )
        # what keys are added or altered for the outputs during apply
        output_apply_keys = (
            'weights',
        )

        # init base class
        super().__init__(
            data=data,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            input_specs=input_specs,
            calc_specs=calc_specs,
            output_specs=output_specs,
            input_apply_keys=input_apply_keys,
            output_apply_keys=output_apply_keys,
        )

        assert self.input_mode is not None
        assert self.calc_mode is None
        assert self.output_mode is not None

    @profile
    def apply_function(self):
        genie_ma_qe = self.params.Genie_Ma_QE.m_as('dimensionless')
        genie_ma_res = self.params.Genie_Ma_RES.m_as('dimensionless')

        for container in self.data:
            apply_genie_sys(
                genie_ma_qe,
                container['linear_fit_maccqe'].get(WHERE),
                container['quad_fit_maccqe'].get(WHERE),
                genie_ma_res,
                container['linear_fit_maccres'].get(WHERE),
                container['quad_fit_maccres'].get(WHERE),
                out=container['weights'].get(WHERE),
            )
            container['weights'].mark_changed(WHERE)


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
    out[0] *= (
        (1. + (linear_fit_maccqe + quad_fit_maccqe * genie_ma_qe) * genie_ma_qe)
        * (1. + (linear_fit_maccres + quad_fit_maccres * genie_ma_res) * genie_ma_res)
    )
