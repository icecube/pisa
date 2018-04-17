"""
Stage to apply pre-calculated Genie uncertainties
"""
from __future__ import absolute_import, print_function, division

import numpy as np
from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import WHERE


class genie_sys(PiStage):
    """
    stage to apply pre-calculated Genie sys

    Paramaters
    ----------
    Genie_Ma_QE : quantity (dimensionless)
    Genie_Ma_RES : quantity (dimensionless)
    Genie_AHTBY : quantity (dimensionless)
    Genie_BHTBY : quantity (dimensionless)
    Genie_CV1UBY : quantity (dimensionless)
    Genie_CV2UBY : quantity (dimensionless)

    Notes
    -----

    requires the following event keys
    linear_fit_maccqe : Genie CC quasi elastic linear coefficient
    quad_fit_maccqe : Genie CC quasi elastic quadratic coefficient
    linear_fit_maccres : Genie CC resonance linear coefficient
    quad_fit_maccres : Genie CC resonance quadratic coefficient
    linear_fit_ahtby : GENIE NC/CC DIS higher-twist Bodek-Yang A linear coeff.
    quad_fit_ahtby : GENIE NC/CC DIS HT BY A quad. coeff.
    linear_fit_bhtby : GENIE NC/CC DIS HT BY B linear coeff.
    quad_fit_bhtby : GENIE NC/CC HT BY B quad. coeff.
    linear_fit_cv1uby : GENIE NC/CC DIS BY CV1U linear coeff.
    quad_fit_cv1uby : GENIE  NC/CC DIS BY CV1U quad. coeff.
    linear_fit_cv2uby : GENIE NC/CC DIS BY CV2U linear coeff.
    quad_fit_cv2uby : GENIE NC/CC DIS BY CV2U quad. coeff.


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
                           'Genie_AHTBY',
                           'Genie_BHTBY',
                           'Genie_CV1UBY',
                           'Genie_CV2UBY'
                          )
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_apply_keys = (
                      'linear_fit_maccqe',
                      'quad_fit_maccqe',
                      'linear_fit_maccres',
                      'quad_fit_maccres',
                      'linear_fit_ahtby',
                      'quad_fit_ahtby',
                      'linear_fit_bhtby',
                      'quad_fit_bhtby',
                      'linear_fit_cv1uby',
                      'quad_fit_cv1uby',
                      'linear_fit_cv2uby',
                      'quad_fit_cv2uby'
                     )
        # what keys are added or altered for the outputs during apply
        output_apply_keys = ('weights',
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
        genie_ahtby = self.params.Genie_AHTBY.m_as('dimensionless')
        genie_bhtby = self.params.Genie_BHTBY.m_as('dimensionless')
        genie_cv1uby = self.params.Genie_CV1UBY.m_as('dimensionless')
        genie_cv2uby = self.params.Genie_CV2UBY.m_as('dimensionless')

        for container in self.data:
            apply_genie_sys(genie_ma_qe,
                            container['linear_fit_maccqe'].get(WHERE),
                            container['quad_fit_maccqe'].get(WHERE),
                            genie_ma_res,
                            container['linear_fit_maccres'].get(WHERE),
                            container['quad_fit_maccres'].get(WHERE),
                            genie_ahtby,
                            container['linear_fit_ahtby'].get(WHERE),
                            container['quad_fit_ahtby'].get(WHERE),
                            genie_bhtby,
                            container['linear_fit_bhtby'].get(WHERE),
                            container['quad_fit_bhtby'].get(WHERE),
                            genie_cv1uby,
                            container['linear_fit_cv1uby'].get(WHERE),
                            container['quad_fit_cv1uby'].get(WHERE),
                            genie_cv2uby,
                            container['linear_fit_cv2uby'].get(WHERE),
                            container['quad_fit_cv2uby'].get(WHERE),
                            out=container['weights'].get(WHERE),
                           )
            container['weights'].mark_changed(WHERE)


if FTYPE == np.float64:
    signature = '(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8[:])'
else:
    signature = '(f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4[:])'

@guvectorize([signature], '(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()->()', target=TARGET)
def apply_genie_sys(genie_ma_qe, linear_fit_maccqe, quad_fit_maccqe,
                    genie_ma_res, linear_fit_maccres, quad_fit_maccres,
                    genie_ahtby, linear_fit_ahtby, quad_fit_ahtby,
                    genie_bhtby, linear_fit_bhtby, quad_fit_bhtby,
                    genie_cv1uby, linear_fit_cv1uby, quad_fit_cv1uby,
                    genie_cv2uby, linear_fit_cv2uby, quad_fit_cv2uby,
                    out):
    out[0] *= ((1. + (linear_fit_maccqe + quad_fit_maccqe * genie_ma_qe) * genie_ma_qe) *
               (1. + (linear_fit_maccres + quad_fit_maccres * genie_ma_res) * genie_ma_res) *
               (1. + (linear_fit_ahtby + quad_fit_ahtby * genie_ahtby) * genie_ahtby) *
               (1. + (linear_fit_bhtby + quad_fit_bhtby * genie_bhtby) * genie_bhtby) *
               (1. + (linear_fit_cv1uby + quad_fit_cv1uby * genie_cv1uby) * genie_cv1uby) *
               (1. + (linear_fit_cv2uby + quad_fit_cv2uby * genie_cv2uby) * genie_cv2uby))
