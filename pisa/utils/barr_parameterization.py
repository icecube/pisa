# pylint: disable=not-callable

"""
External PISA file to be kept in private location
Containing the Barr parameterizations for flux modifications

originaly developped by Juan Pablo Yanez
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import math

from pisa.utils.numba_tools import myjit


@myjit
def sign(val):
    ''' signum function'''
    if val == 0:
        return 0
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
        # pre-fix (wrong)
        #return A_ave - (norm_fcn(true_coszen, A_shape, 0.32) - 0.75 * A_shape)
        # fixed (correct)
        return A_ave - (norm_fcn(true_coszen, A_shape, 0.36) - 0.6 * A_shape)
    if flav == 0:
        A_ave = LogLogParam(true_energy, e1max_mu * e1mu + e1max_e * e1e, e2max_mu * e2mu + e2max_e * e2e, x1e, x2e, False, 0)
        A_shape = 1. * LogLogParam(true_energy, z1max_mu * z1mu + z1max_e * z1e, z2max_mu * z2mu + z2max_e * z2e, x1z, x2z, True, nue_cutoff)
        # pre-fix (wrong)
        #return A_ave - (1.5*norm_fcn(true_coszen, A_shape, 0.4) - 0.7 * A_shape)
        # fixed (correct)
        return A_ave - (1.5*norm_fcn(true_coszen, A_shape, 0.36) - 0.7 * A_shape)

@myjit
def modRatioUpHor(flav, true_energy, true_coszen, uphor):
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
    if flav == 0:
        A_shape = 1. * abs(uphor) * LogLogParam(true_energy, (z1max_e + z1max_mu), (z2max_e + z2max_mu), x1z, x2z, True, nue_cutoff)
        # correct:
        return 1 - 0.3 * sign(uphor) * norm_fcn(true_coszen, A_shape, 0.35)
    if flav == 1:
        # pre-fix (wrong)
        #A_shape = 1. * abs(uphor) * LogLogParam(true_energy, z1max_mu, z2max_mu, x1z, x2z, True, numu_cutoff)
        #return 1 - 0.3 * sign(uphor) * norm_fcn(true_coszen, A_shape, 0.35)
        # fixed (correct)
        return 1.
    # wrong:
    #return 1 - 3.5 * sign(uphor) * norm_fcn(true_coszen, A_shape, 0.35)

@myjit
def modRatioNuBar(nubar, flav, true_energy, true_coszen, nubar_sys):
    # oscfit function
    modfactor = nubar_sys * ModFlux(flav, true_energy, true_coszen, 1., 1., 1., 1., 1., 1., 1., 1.)
    if nubar < 0:
        return max(0., 1. / (1 + 0.5 * modfactor))
    if nubar > 0:
        return max(0., 1. + 0.5 * modfactor)
