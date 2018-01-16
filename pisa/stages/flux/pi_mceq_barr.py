# pylint: disable=not-callable
"""
Stage to implement the old PISA/oscfit flux systematics

The `nominal_flux` and `nominal_opposite_flux` is something that realy should
not be done. That needs to be changed. We simply want to calcualte nu and nubar
fluxes insetad!

"""
from __future__ import absolute_import, print_function, division

import math
import numpy as np
from numba import guvectorize, cuda
import cPickle as pickle
from bz2 import BZ2File
from scipy.interpolate import RectBivariateSpline

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import WHERE, myjit, ftype
from pisa.utils.resources import find_resource


class pi_mceq_barr(PiStage):
    """
    stage to apply Barr style flux uncertainties, obtained from tables
    created with MCeq, these store the derivateives for each of the 12 (24)
    barr parameters, separately

    Paramaters
    ----------

    nue_numu_ratio : quantity (dimensionless)
    nu_nubar_ratio : quantity (dimensionless)
    delta_index : quantity (dimensionless)
    Barr_uphor_ratio : quantity (dimensionless)
    Barr_nu_nubar_ratio : quantity (dimensionless)

    Notes
    -----

    """
    # TODO: get rid of this _oppo_flux stuff!!!
    # Just replace with nu and nubar flux!!!

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

        expected_params = ('table_file',
                           'barr_a',
                           'barr_b',
                           'barr_c',
                          )
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_calc_keys = ('weights',
                           'nominal_flux',
                           'nominal_opposite_flux',
                          )
        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ('sys_flux',
                           )
        # what keys are added or altered for the outputs during apply
        output_apply_keys = ('sys_flux',
                            )

        # init base class
        super(pi_mceq_barr, self).__init__(data=data,
                                        params=params,
                                        expected_params=expected_params,
                                        input_names=input_names,
                                        output_names=output_names,
                                        debug_mode=debug_mode,
                                        input_specs=input_specs,
                                        calc_specs=calc_specs,
                                        output_specs=output_specs,
                                        input_calc_keys=input_calc_keys,
                                        output_calc_keys=output_calc_keys,
                                        output_apply_keys=output_apply_keys,
                                       )

        assert self.input_mode is not None
        assert self.calc_mode is not None
        assert self.output_mode is not None

    def setup_function(self):


        # load MCeq tables
        table = pickle.load(BZ2File(find_resource(self.params.table_file.value)))

        self.data.data_specs = self.calc_specs

        for container in self.data:
            container['sys_flux'] = np.empty((container.size, 2), dtype=FTYPE)

    @profile
    def compute_function(self):

        self.data.data_specs = self.calc_specs




        nue_numu_ratio = self.params.nue_numu_ratio.value.m_as('dimensionless')
        nu_nubar_ratio = self.params.nu_nubar_ratio.value.m_as('dimensionless')
        delta_index = self.params.delta_index.value.m_as('dimensionless')
        Barr_uphor_ratio = self.params.Barr_uphor_ratio.value.m_as('dimensionless')
        #Barr_uphor_ratio2 = self.params.Barr_uphor_ratio2.value.m_as('dimensionless')
        Barr_nu_nubar_ratio = self.params.Barr_nu_nubar_ratio.value.m_as('dimensionless')
        #Barr_nu_nubar_ratio2 = self.params.Barr_nu_nubar_ratio2.value.m_as('dimensionless')

        for container in self.data:

            apply_sys_vectorized(container['true_energy'].get(WHERE),
                                 container['true_coszen'].get(WHERE),
                                 container['nominal_flux'].get(WHERE),
                                 container['nominal_opposite_flux'].get(WHERE),
                                 container['nubar'],
                                 nue_numu_ratio,
                                 nu_nubar_ratio,
                                 delta_index,
                                 Barr_uphor_ratio,
                                 Barr_nu_nubar_ratio,
                                 out=container['sys_flux'].get(WHERE),
                                )
            container['sys_flux'].mark_changed(WHERE)



