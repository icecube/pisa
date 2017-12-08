# pylint: disable=not-callable
"""
Stage to implement the old PISA/odcfit flux systematics

The `nominal_flux` and `nominal_opposite_flux` is something that realy should
not be done. That needs to be changed. We simply want to calcualte nu and nubar
fluxes insetad!

"""
from __future__ import absolute_import, print_function, division

import math
import numpy as np
from numba import guvectorize, cuda

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import WHERE, myjit, ftype


class pi_simple(PiStage):
    """
    stage to apply Barr style flux uncertainties

    Paramaters
    ----------

    nue_numu_ratio : quantity (dimensionless)
    nu_nubar_ratio : quantity (dimensionless)
    delta_index : quantity (dimensionless)
    Barr_uphor_ratio : quantity (dimensionless)
    Barr_nu_nubar_ratio : quantity (dimensionless)

    Notes
    -----

    TODO: get rid of this _oppo_flux stuff!!!
    Just replace with nu and nubar flux!!!

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

        expected_params = ('nue_numu_ratio',
                           'nu_nubar_ratio',
                           'delta_index',
                           'Barr_uphor_ratio',
                           'Barr_nu_nubar_ratio',
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
        super(pi_simple, self).__init__(data=data,
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
        Barr_nu_nubar_ratio = self.params.Barr_nu_nubar_ratio.value.m_as('dimensionless')

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



@myjit
def apply_ratio_scale(ratio_scale, sum_constant, in1, in2, out):
    ''' apply ratio scale to flux values

    Paramters
    ---------

    ratio_scale : float

    sum_constant : bool
        if Ture, then the sum of the new flux will be identical to the old flux

    in1 : float

    in2 : float

    out : array

    '''
    if in1 == 0. and in2 == 0.:
        out[0] = 0.
        out[1] = 0.
        return

    if sum_constant:
        orig_ratio = in1 / in2
        orig_sum = in1 + in2
        new = orig_sum / (1. + ratio_scale * orig_ratio)
        out[0] = ratio_scale * orig_ratio * new
        out[1] = new
    else:
        out[0] = ratio_scale * in1
        out[1] = in2

@myjit
def spectral_index_scale(true_energy, egy_pivot, delta_index):
    ''' calculate spectral index scale '''
    return math.pow((true_energy/egy_pivot), delta_index)

@myjit
def sign(val):
    ''' signum '''
    if val >= 0:
        return 1.
    return -1.

@myjit
def LogLogParam(true_energy, y1, y2, x1, x2, use_cutoff, cutoff_value):
        # oscfit function
    nu_nubar = sign(y2)
    y1 = sign(y1) * math.log10(abs(y1) + 0.0001)
    y2 = math.log10(abs(y2 + 0.0001))
    modification = nu_nubar * math.pow(10., (((y2 - y1) / (x2 - x1)) * (math.log10(true_energy) - x1) + y1 - 2.))
    if use_cutoff:
        modification *= math.exp(-1. * true_energy / cutoff_value)
    return modification

@myjit
def norm_fcn(x, A, sigma):
    # oscfit function
    return A / math.sqrt(2 * math.pi * math.pow(sigma, 2)) * math.exp(-math.pow(x, 2) / (2 * math.pow(sigma, 2)))

@myjit
def ModFlux(flav, true_energy, true_coszen, e1mu, e2mu, z1mu, z2mu, e1e, e2e, z1e, z2e):
    # These parameters are obtained from fits to the paper of Barr
    # E dependent ratios, max differences per flavor (Fig.7)
    e1max_mu = 3.
    e2max_mu = 43
    e1max_e = 2.5
    e2max_e = 10
    # Evaluated at
    x1e = 0.5
    x2e = 3.

    # Zenith dependent amplitude, max differences per flavor (Fig. 9)
    z1max_mu = 0.6
    z2max_mu = 5.
    z1max_e = 0.3
    z2max_e = 5.
    nue_cutoff = 650.
    numu_cutoff = 1000.
    # Evaluated at
    x1z = 0.5
    x2z = 2.
    # oscfit function
    if flav == 1:
        A_ave = LogLogParam(true_energy, e1max_mu*e1mu, e2max_mu*e2mu, x1e, x2e, False, 0)
        A_shape = 2.5*LogLogParam(true_energy, z1max_mu*z1mu, z2max_mu*z2mu, x1z, x2z, True, numu_cutoff)
        return A_ave - (norm_fcn(true_coszen, A_shape, 0.36) - 0.6 * A_shape)
    if flav == 0:
        A_ave = LogLogParam(true_energy, e1max_mu * e1mu + e1max_e * e1e, e2max_mu * e2mu + e2max_e * e2e, x1e, x2e, False, 0)
        A_shape = 1. * LogLogParam(true_energy, z1max_mu * z1mu + z1max_e * z1e, z2max_mu * z2mu + z2max_e * z2e, x1z, x2z, True, nue_cutoff)
        return A_ave - (1.5*norm_fcn(true_coszen, A_shape, 0.36) - 0.7 * A_shape)

@myjit
def modRatioUpHor(flav, true_energy, true_coszen, uphor):
    # Zenith dependent amplitude, max differences per flavor (Fig. 9)
    z1max_mu = 0.6
    z2max_mu = 5.
    z1max_e = 0.3
    z2max_e = 5.
    nue_cutoff = 650.
    # Evaluated at
    x1z = 0.5
    x2z = 2.
    # oscfit function
    if flav == 0:
        A_shape = 1. * abs(uphor) * LogLogParam(true_energy, (z1max_e + z1max_mu), (z2max_e + z2max_mu), x1z, x2z, True, nue_cutoff)
        return 1 - 0.3 * sign(uphor) * norm_fcn(true_coszen, A_shape, 0.35)
    if flav == 1:
        return 1.

@myjit
def modRatioNuBar(nubar, flav, true_energy, true_coszen, nubar_sys):
    # oscfit function
    modfactor = nubar_sys * ModFlux(flav, true_energy, true_coszen, 1., 1., 1., 1., 1., 1., 1., 1.)
    if nubar < 0:
        return max(0., 1. / (1 + 0.5 * modfactor))
    if nubar > 0:
        return max(0., 1. + 0.5 * modfactor)




@myjit
def apply_sys_kernel(true_energy,
                     true_coszen,
                     nominal_flux,
                     nominal_opposite_flux,
                     nubar,
                     nue_numu_ratio,
                     nu_nubar_ratio,
                     delta_index,
                     Barr_uphor_ratio,
                     Barr_nu_nubar_ratio,
                     out):
    # nue/numu ratio
    new_flux = cuda.local.array(shape=(2), dtype=ftype)
    new_opposite_flux = cuda.local.array(2, dtype=ftype)
    apply_ratio_scale(nue_numu_ratio, True, nominal_flux[0], nominal_flux[1], new_flux)
    apply_ratio_scale(nue_numu_ratio, True, nominal_opposite_flux[0], nominal_opposite_flux[1], new_opposite_flux)

    #apply flux systematics
    # spectral idx
    idx_scale = spectral_index_scale(true_energy, 24.0900951261, delta_index)
    new_flux[0] *= idx_scale
    new_flux[1] *= idx_scale
    new_opposite_flux[0] *= idx_scale
    new_opposite_flux[1] *= idx_scale

    # nu/nubar ratio
    new_nue_flux = cuda.local.array(2, dtype=ftype)
    new_numu_flux = cuda.local.array(2, dtype=ftype)
    if nubar < 0:
        apply_ratio_scale(nu_nubar_ratio, True, new_opposite_flux[0], new_flux[0], new_nue_flux)
        apply_ratio_scale(nu_nubar_ratio, True, new_opposite_flux[1], new_flux[1], new_numu_flux)
        out[0] = new_nue_flux[1]
        out[1] = new_numu_flux[1]
    else:
        apply_ratio_scale(nu_nubar_ratio, True, new_flux[0], new_opposite_flux[0], new_nue_flux)
        apply_ratio_scale(nu_nubar_ratio, True, new_flux[1], new_opposite_flux[1], new_numu_flux)
        out[0] = new_nue_flux[0]
        out[1] = new_numu_flux[0]

    # Barr flux
    out[0] *= modRatioNuBar(nubar, 0, true_energy, true_coszen, Barr_nu_nubar_ratio)
    out[1] *= modRatioNuBar(nubar, 1, true_energy, true_coszen, Barr_nu_nubar_ratio)

    out[0] *= modRatioUpHor(0, true_energy, true_coszen, Barr_uphor_ratio)
    out[1] *= modRatioUpHor(1, true_energy, true_coszen, Barr_uphor_ratio)


# vectorized function to apply
# must be outside class
if FTYPE == np.float64:
    signature = '(f8, f8, f8[:], f8[:], i4, f8, f8, f8, f8, f8, f8[:])'
else:
    signature = '(f4, f4, f4[:], f4[:], i4, f4, f4, f4, f4, f4, f4[:])'
@guvectorize([signature], '(),(),(d),(d),(),(),(),(),(),()->(d)', target=TARGET)
def apply_sys_vectorized(true_energy,
                         true_coszen,
                         nominal_flux,
                         nominal_opposite_flux,
                         nubar,
                         nue_numu_ratio,
                         nu_nubar_ratio,
                         delta_index,
                         Barr_uphor_ratio,
                         Barr_nu_nubar_ratio,
                         out):
    apply_sys_kernel(true_energy,
                     true_coszen,
                     nominal_flux,
                     nominal_opposite_flux,
                     nubar,
                     nue_numu_ratio,
                     nu_nubar_ratio,
                     delta_index,
                     Barr_uphor_ratio,
                     Barr_nu_nubar_ratio,
                     out)
