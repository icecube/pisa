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

    table_file : str
        pointing to spline table obtained from MCEq
    barr_a : quantity (dimensionless)

    Notes
    -----

    The table containe for each barr parameter 8 splines, these are:
    flux nue, derivative nue, flux nuebar, derivative nuebar, flux numu, ...

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

        expected_params = ('table_file',
                           'barr_a',
                           #'barr_b',
                           #'barr_c',
                          )
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_calc_keys = ('weights',
                           'nominal_nu_flux',
                           'nominal_nubar_flux',
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
        spline_tables_dict = pickle.load(BZ2File(find_resource(self.params.table_file.value)))

        self.data.data_specs = self.calc_specs

        for container in self.data:
            container['sys_flux'] = np.empty((container.size, 2), dtype=FTYPE)

        if self.calc_mode == 'binned':
            # speed up calculation by adding links
            # as layers don't care about flavour
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc',
                                             'nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                             'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])

        for container in self.data:
            # evaluate the splines (flux and deltas) for each E/CZ point
            # at the moment this is done on CPU, therefore we force 'host'
            for key in spline_tables_dict.keys():
                logging.info('Evaluating MCEq splines for %s for Barr parameter %s'%(container.name, key))
                container['barr_'+key] = np.empty((container.size, 8), dtype=FTYPE)
                self.eval_spline(container['true_energy'].get('host'),
                                 container['true_coszen'].get('host'),
                                 spline_tables_dict[key],
                                 out=container['barr_'+key].get('host'))
                container['barr_'+key].mark_changed('host')
        self.data.unlink_containers()

    def eval_spline(self, true_energy, true_coszen, splines, out):
        '''
        dumb function to iterate trhouh all E, CZ values
        and evlauate all 8 Barr splines at these points
        '''
        for i in xrange(len(true_energy)):
            abs_cos = abs(true_coszen[i])
            log_e = np.log(true_energy[i])
            for j in xrange(len(splines)):
                out[i,j] = splines[j](abs_cos, log_e)[0,0]


    @profile
    def compute_function(self):

        self.data.data_specs = self.calc_specs

        barr_a = self.params.barr_a.value.m_as('dimensionless')

        for container in self.data:

            apply_barr_vectorized(container['nominal_nu_flux'].get(WHERE),
                                  container['nominal_nubar_flux'].get(WHERE),
                                  container['nubar'],
                                  container['barr_a+'].get(WHERE),
                                  container['barr_a-'].get(WHERE),
                                  barr_a,
                                  out=container['sys_flux'].get(WHERE),
                                 )
            container['sys_flux'].mark_changed(WHERE)



# vectorized function to apply
# must be outside class
if FTYPE == np.float64:
    signature = '(f8, f8, f8[:], f8[:], i4, f8, f8, f8, f8, f8, f8[:])'
else:
    signature = '(f4[:], f4[:], i4, f4[:], f4[:], f4, f4[:])'
@guvectorize([signature], '(d),(d),(),(c),(c),()->(d)', target=TARGET)
def apply_barr_vectorized(nominal_nu_flux,
                          nominal_nubar_flux,
                          nubar,
                          barr_a_pos,
                          barr_a_neg,
                          barr_a,
                          out):
    if nubar > 0:
        out[0] = nominal_nu_flux[0] * (1. + barr_a * ( (barr_a_pos[1] / barr_a_pos[0]) + (barr_a_neg[1] / barr_a_neg[0])))
        out[1] = nominal_nu_flux[1] * (1. + barr_a * ( (barr_a_pos[5] / barr_a_pos[4]) + (barr_a_neg[5] / barr_a_neg[4])))
    else:
        out[0] = nominal_nubar_flux[0] * (1. + barr_a * ( (barr_a_pos[3] / barr_a_pos[2]) + (barr_a_neg[3] / barr_a_neg[2])))
        out[1] = nominal_nubar_flux[1] * (1. + barr_a * ( (barr_a_pos[7] / barr_a_pos[6]) + (barr_a_neg[7] / barr_a_neg[6])))
