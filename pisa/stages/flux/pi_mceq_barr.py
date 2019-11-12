# pylint: disable=not-callable
"""
Stage to implement the the systematic flux variations based on the
Barr scheme and evaluate with MCEq

It requires spline tables created by the `$PISA/scripts/create_barr_sys_tables_mceq.py`

this stage is highly experimental and not yet validated or finished!
"""
from __future__ import absolute_import, print_function, division

import math
import numpy as np
from numba import guvectorize, cuda
import pickle
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
    barr_* : quantity (dimensionless)

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
                           'barr_b',
                           'barr_c',
                           'barr_d',
                           'barr_e',
                           'barr_f',
                           'barr_g',
                           'barr_h',
                           'barr_i',
                           'barr_w',
                           'barr_y',
                           'barr_z',
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
        for i in range(len(true_energy)):
            abs_cos = abs(true_coszen[i])
            log_e = np.log(true_energy[i])
            for j in range(len(splines)):
                out[i,j] = splines[j](abs_cos, log_e)[0,0]


    @profile
    def compute_function(self):

        self.data.data_specs = self.calc_specs

        barr_a = self.params.barr_a.value.m_as('dimensionless')
        barr_b = self.params.barr_b.value.m_as('dimensionless')
        barr_c = self.params.barr_c.value.m_as('dimensionless')
        barr_d = self.params.barr_d.value.m_as('dimensionless')
        barr_e = self.params.barr_e.value.m_as('dimensionless')
        barr_f = self.params.barr_f.value.m_as('dimensionless')
        barr_g = self.params.barr_g.value.m_as('dimensionless')
        barr_h = self.params.barr_h.value.m_as('dimensionless')
        barr_i = self.params.barr_i.value.m_as('dimensionless')
        barr_w = self.params.barr_w.value.m_as('dimensionless')
        barr_y = self.params.barr_y.value.m_as('dimensionless')
        barr_z = self.params.barr_z.value.m_as('dimensionless')

        for container in self.data:

            apply_barr_vectorized(container['nominal_nu_flux'].get(WHERE),
                                  container['nominal_nubar_flux'].get(WHERE),
                                  container['nubar'],
                                  container['barr_a+'].get(WHERE), container['barr_a-'].get(WHERE), barr_a,
                                  container['barr_b+'].get(WHERE), container['barr_b-'].get(WHERE), barr_b,
                                  container['barr_c+'].get(WHERE), container['barr_c-'].get(WHERE), barr_c,
                                  container['barr_d+'].get(WHERE), container['barr_d-'].get(WHERE), barr_d,
                                  container['barr_e+'].get(WHERE), container['barr_e-'].get(WHERE), barr_e,
                                  container['barr_f+'].get(WHERE), container['barr_f-'].get(WHERE), barr_f,
                                  container['barr_g+'].get(WHERE), container['barr_g-'].get(WHERE), barr_g,
                                  container['barr_h+'].get(WHERE), container['barr_h-'].get(WHERE), barr_h,
                                  container['barr_i+'].get(WHERE), container['barr_i-'].get(WHERE), barr_i,
                                  container['barr_w+'].get(WHERE), container['barr_w-'].get(WHERE), barr_w,
                                  container['barr_y+'].get(WHERE), container['barr_y-'].get(WHERE), barr_y,
                                  container['barr_z+'].get(WHERE), container['barr_z-'].get(WHERE), barr_z,
                                  out=container['sys_flux'].get(WHERE),
                                 )
            container['sys_flux'].mark_changed(WHERE)


@myjit
def delta(param, pos_f, pos_d, neg_f, neg_d):
    ''' return fractional delta given a barr parameter and
    the postitive and negative fluxes and derivateives
    '''
    return param * ((pos_d / pos_f) + (neg_d / neg_f))

@myjit
def mod_factor(idx,
               barr_a_pos, barr_a_neg, barr_a,
               barr_b_pos, barr_b_neg, barr_b,
               barr_c_pos, barr_c_neg, barr_c,
               barr_d_pos, barr_d_neg, barr_d,
               barr_e_pos, barr_e_neg, barr_e,
               barr_f_pos, barr_f_neg, barr_f,
               barr_g_pos, barr_g_neg, barr_g,
               barr_h_pos, barr_h_neg, barr_h,
               barr_i_pos, barr_i_neg, barr_i,
               barr_w_pos, barr_w_neg, barr_w,
               barr_y_pos, barr_y_neg, barr_y,
               barr_z_pos, barr_z_neg, barr_z,
               ):
    '''
    calculate the modification factor for the flux

    Parameters
    ----------

    idx : int
        which Barr splines to use
        for nue: 0
        for nuebar: 2
        for numu: 4
        for numubar : 6

    barr_*_pos : array of length 8
        (nue_flux, nue_derivative, nuebar...)

    barr_*_neg : array of length 8
        same as barr_*_pos
    
    barr_* : float
        systematics value

    '''
    return 1. + (
                 delta(barr_a, barr_a_pos[idx], barr_a_pos[idx+1], barr_a_neg[idx], barr_a_neg[idx+1])
                 + delta(barr_b, barr_b_pos[idx], barr_b_pos[idx+1], barr_b_neg[idx], barr_b_neg[idx+1])
                 + delta(barr_c, barr_c_pos[idx], barr_c_pos[idx+1], barr_c_neg[idx], barr_c_neg[idx+1])
                 + delta(barr_d, barr_d_pos[idx], barr_d_pos[idx+1], barr_d_neg[idx], barr_d_neg[idx+1])
                 + delta(barr_e, barr_e_pos[idx], barr_e_pos[idx+1], barr_e_neg[idx], barr_e_neg[idx+1])
                 + delta(barr_f, barr_f_pos[idx], barr_f_pos[idx+1], barr_f_neg[idx], barr_f_neg[idx+1])
                 + delta(barr_g, barr_g_pos[idx], barr_g_pos[idx+1], barr_g_neg[idx], barr_g_neg[idx+1])
                 + delta(barr_h, barr_h_pos[idx], barr_h_pos[idx+1], barr_h_neg[idx], barr_h_neg[idx+1])
                 + delta(barr_i, barr_i_pos[idx], barr_i_pos[idx+1], barr_i_neg[idx], barr_i_neg[idx+1])
                 + delta(barr_w, barr_w_pos[idx], barr_w_pos[idx+1], barr_w_neg[idx], barr_w_neg[idx+1])
                 + delta(barr_y, barr_y_pos[idx], barr_y_pos[idx+1], barr_y_neg[idx], barr_y_neg[idx+1])
                 + delta(barr_z, barr_z_pos[idx], barr_z_pos[idx+1], barr_z_neg[idx], barr_z_neg[idx+1])
                 )

# vectorized function to apply
# must be outside class
if FTYPE == np.float64:
    signature = '(f8[:], f8[:], i4, \
                  f8[:], f8[:], f8, \
                  f8[:], f8[:], f8, \
                  f8[:], f8[:], f8, \
                  f8[:], f8[:], f8, \
                  f8[:], f8[:], f8, \
                  f8[:], f8[:], f8, \
                  f8[:], f8[:], f8, \
                  f8[:], f8[:], f8, \
                  f8[:], f8[:], f8, \
                  f8[:], f8[:], f8, \
                  f8[:], f8[:], f8, \
                  f8[:], f8[:], f8, \
                  f8[:])'
else:
    signature = '(f4[:], f4[:], i4, \
                  f4[:], f4[:], f4, \
                  f4[:], f4[:], f4, \
                  f4[:], f4[:], f4, \
                  f4[:], f4[:], f4, \
                  f4[:], f4[:], f4, \
                  f4[:], f4[:], f4, \
                  f4[:], f4[:], f4, \
                  f4[:], f4[:], f4, \
                  f4[:], f4[:], f4, \
                  f4[:], f4[:], f4, \
                  f4[:], f4[:], f4, \
                  f4[:], f4[:], f4, \
                  f4[:])'
@guvectorize([signature], '(d),(d),(),\
                           (c),(c),(),\
                           (c),(c),(),\
                           (c),(c),(),\
                           (c),(c),(),\
                           (c),(c),(),\
                           (c),(c),(),\
                           (c),(c),(),\
                           (c),(c),(),\
                           (c),(c),(),\
                           (c),(c),(),\
                           (c),(c),(),\
                           (c),(c),()\
                           ->(d)', target=TARGET)
def apply_barr_vectorized(nominal_nu_flux,
                          nominal_nubar_flux,
                          nubar,
                          barr_a_pos, barr_a_neg, barr_a,
                          barr_b_pos, barr_b_neg, barr_b,
                          barr_c_pos, barr_c_neg, barr_c,
                          barr_d_pos, barr_d_neg, barr_d,
                          barr_e_pos, barr_e_neg, barr_e,
                          barr_f_pos, barr_f_neg, barr_f,
                          barr_g_pos, barr_g_neg, barr_g,
                          barr_h_pos, barr_h_neg, barr_h,
                          barr_i_pos, barr_i_neg, barr_i,
                          barr_w_pos, barr_w_neg, barr_w,
                          barr_y_pos, barr_y_neg, barr_y,
                          barr_z_pos, barr_z_neg, barr_z,
                          out):
    if nubar > 0:
        out[0] = nominal_nu_flux[0] * mod_factor(0,
                                                 barr_a_pos, barr_a_neg, barr_a,
                                                 barr_b_pos, barr_b_neg, barr_b,
                                                 barr_c_pos, barr_c_neg, barr_c,
                                                 barr_d_pos, barr_d_neg, barr_d,
                                                 barr_e_pos, barr_e_neg, barr_e,
                                                 barr_f_pos, barr_f_neg, barr_f,
                                                 barr_g_pos, barr_g_neg, barr_g,
                                                 barr_h_pos, barr_h_neg, barr_h,
                                                 barr_i_pos, barr_i_neg, barr_i,
                                                 barr_w_pos, barr_w_neg, barr_w,
                                                 barr_y_pos, barr_y_neg, barr_y,
                                                 barr_z_pos, barr_z_neg, barr_z,
                                                 )
        out[1] = nominal_nu_flux[1] * mod_factor(4,
                                                 barr_a_pos, barr_a_neg, barr_a,
                                                 barr_b_pos, barr_b_neg, barr_b,
                                                 barr_c_pos, barr_c_neg, barr_c,
                                                 barr_d_pos, barr_d_neg, barr_d,
                                                 barr_e_pos, barr_e_neg, barr_e,
                                                 barr_f_pos, barr_f_neg, barr_f,
                                                 barr_g_pos, barr_g_neg, barr_g,
                                                 barr_h_pos, barr_h_neg, barr_h,
                                                 barr_i_pos, barr_i_neg, barr_i,
                                                 barr_w_pos, barr_w_neg, barr_w,
                                                 barr_y_pos, barr_y_neg, barr_y,
                                                 barr_z_pos, barr_z_neg, barr_z,
                                                 )
    else:
        out[0] = nominal_nubar_flux[0] * mod_factor(2,
                                                    barr_a_pos, barr_a_neg, barr_a,
                                                    barr_b_pos, barr_b_neg, barr_b,
                                                    barr_c_pos, barr_c_neg, barr_c,
                                                    barr_d_pos, barr_d_neg, barr_d,
                                                    barr_e_pos, barr_e_neg, barr_e,
                                                    barr_f_pos, barr_f_neg, barr_f,
                                                    barr_g_pos, barr_g_neg, barr_g,
                                                    barr_h_pos, barr_h_neg, barr_h,
                                                    barr_i_pos, barr_i_neg, barr_i,
                                                    barr_w_pos, barr_w_neg, barr_w,
                                                    barr_y_pos, barr_y_neg, barr_y,
                                                    barr_z_pos, barr_z_neg, barr_z,
                                                    )
        out[1] = nominal_nubar_flux[1] * mod_factor(6,
                                                    barr_a_pos, barr_a_neg, barr_a,
                                                    barr_b_pos, barr_b_neg, barr_b,
                                                    barr_c_pos, barr_c_neg, barr_c,
                                                    barr_d_pos, barr_d_neg, barr_d,
                                                    barr_e_pos, barr_e_neg, barr_e,
                                                    barr_f_pos, barr_f_neg, barr_f,
                                                    barr_g_pos, barr_g_neg, barr_g,
                                                    barr_h_pos, barr_h_neg, barr_h,
                                                    barr_i_pos, barr_i_neg, barr_i,
                                                    barr_w_pos, barr_w_neg, barr_w,
                                                    barr_y_pos, barr_y_neg, barr_y,
                                                    barr_z_pos, barr_z_neg, barr_z,
                                                    )
