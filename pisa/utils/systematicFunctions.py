"""
Detection uncertainties
"""


from __future__ import division

from copy import deepcopy
from itertools import product

import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy.interpolate import interp1d


__all__ = ['getSystematicFunctions', 'axialMassVar', 'norm_fcn', 'LogLogParam',
           'ModNuMuFlux', 'ModNuEFlux', 'modRatioNuBar', 'modRatioUpHor']


# TODO: this module needs lots of work and review!


def getSystematicFunctions(y_values, y_valuesw2, xpoints, xi_nominal=None,
                           poly_degree=2, use_interp=False, legacy_mode=False):
    """Given a list of histograms produced with varying a single variable X, it
    fits a polynomial function at every bin. The function returns the matrices,
    x and y values, as well as the normalization applied to isolate shape
    effects.

    Parameters
    ----------
        hist_list: list of histograms (numpy 2D arrays)

        hist_list_w2: list of histograms with squared weights (error estimate)

        xpoints: point in x at which the histograms are evaluated

        poly_degree: degree of the polinomial used in the fit

        free_norm: removes normalization, fit only the shapes (set to False to reproduce MC)

    """

    # Going over all indices (except in the last dimension, which is the systematic change)
    # Built to support N dimensional histograms
    # Pushing the systematic change to the last dimension
    y_values    = np.rollaxis(y_values, 0, len(y_values.shape))
    y_valuesw2  = np.rollaxis(y_valuesw2, 0, len(y_valuesw2.shape))
    # Defining the matrix of the functions
    poly_matrix = np.zeros(y_values.shape[:-1], dtype=object)

    indices      = np.array(list(y_values.shape)[:-1])
    all_indices  = np.array(product(*[range(0, x) for x in indices]), dtype=FTYPE)

    for one_index in all_indices:
        yvalues_bin = np.array(y_values[[[x]for x in one_index]][0])
        werrors     = np.sqrt(np.array(y_valuesw2[[[x]for x in one_index]][0]))/yvalues_bin

        # Express y as a variation of the nominal value
        # If the nominal is zero, then keep the bin at zero (for now)
        if yvalues_bin[xi_nominal] == 0.:
            one_fit = np.poly1d([1])
        else:
            if not legacy_mode:
                yvalues_bin  /= yvalues_bin[xi_nominal]
                werrors      *= yvalues_bin

            if use_interp:
                ### Interpolation solution ###
                # No need to take care of outliers, fits, anything
                # Simple linear transition between MC sets
                one_fit = interp1d(xpoints, yvalues_bin,
                                   bounds_error = False, fill_value = 1.)
            else:
                ### Polynomial solution ###
                # Need to remove outliers, and prevent bad functions (bins with few MC events)
                # Zero errors?
                werrors[werrors==0.] = np.mean(werrors)
                if np.sum(werrors) == 0:
                    werrors = np.ones_like(werrors)

                # For the legacy mode that is enough:
                if legacy_mode:
                    one_fit = np.poly1d(polyfit(xpoints, yvalues_bin,
                                                deg = poly_degree,
                                                rcond = None,
                                                w = 1./werrors)[::-1])
                else:
                    bad_bins = np.abs(1 - yvalues_bin) > 1.
                    mean_y   = yvalues_bin[~bad_bins].mean()
                    # Is the typical error larger than 50% or are there not enough good points?
                    if werrors[~bad_bins].mean() > 0.5 or (len(xpoints)-np.sum(bad_bins)) <= poly_degree:
                        one_fit = np.poly1d([mean_y])
                    else: # Fit without outliers
                        one_fit = np.poly1d(polyfit(xpoints, yvalues_bin,
                                                    deg = poly_degree,
                                                    rcond = None,
                                                    w = 1./werrors)[::-1])

                    # Is no-change a good fit or are the errors too large to tell?
                    # Large errors happen in bins where very few events are expected (like NuTau)
                    nochange_chi2 = np.sum((yvalues_bin[~bad_bins] - 1.)**2/werrors[~bad_bins]**2)/\
                        (1.*len(xpoints))
                    fit_chi2      = (np.sum((yvalues_bin[~bad_bins]-
                                             one_fit(xpoints[~bad_bins]))**2/werrors[~bad_bins]**2)/
                                     (1.*len(xpoints[~bad_bins] - poly_degree)))
                    if nochange_chi2 <= fit_chi2:
                        one_fit = np.poly1d([mean_y])

        poly_matrix[tuple(one_index)]=one_fit

    return poly_matrix, y_values, y_valuesw2, all_indices

############
# Cross section uncertainties
#############

def axialMassVar(coeff = np.zeros(2), Ma = 0.):
    # The 1 is summed because I simplified the formula to save space
    return 1 + coeff[:,0]*Ma**2 + coeff[:,1]*Ma

############
# Sub-leading atmospheric flux uncertainties
#############

# See the ipython notebook for the details of how these functions were derived
# There are two basic functions: a gaussian bell and a Log-Log parameterization


def norm_fcn(x, A, sigma = 0.3):
    #x *= 0.9
    return A/np.sqrt(2*np.pi*sigma**2) * np.exp(-x**2/(2*sigma**2))

def LogLogParam(energy = 1., y1 = 1., y2 = 1.,
                x1=0.5, x2=3.,
                cutoff_value = False):
    nu_nubar = np.sign(y2)
    if nu_nubar == 0.0:
        nu_nubar = 1.
    y1 = np.sign(y1)*np.log10(np.abs(y1)+0.0001)
    y2 = np.log10(np.abs(y2+0.0001))
    modification = nu_nubar*10**(((y2-y1)/(x2-x1))*(np.log10(energy)-x1)+y1-2.)
    if cutoff_value:
        modification *= np.exp(-1.*energy/cutoff_value)
    return modification

# These parameters are obtained from fits to the paper of Barr
# E dependent ratios, max differences per flavor (Fig.7)
e1max_mu = 3.
e2max_mu = 43
e1max_e  = 2.5
e2max_e  = 10
e1max_mu_e = 0.62
e2max_mu_e = 11.45
# Evaluated at
x1e = 0.5
x2e = 3.

# Zenith dependent amplitude, max differences per flavor (Fig. 9)
z1max_mu = 0.6
z2max_mu = 5.
z1max_e  = 0.3
z2max_e  = 5.
nue_cutoff  = 650.
numu_cutoff = 1000.
# Evaluated at
x1z = 0.5
x2z = 2.


# This is the neutrino/antineutrino ratio modification for NuMu
def ModNuMuFlux(energy, czenith,
                e1=1., e2=1., z1=1., z2=1.):
    A_ave = LogLogParam(energy=energy,
                        y1=e1max_mu*e1,
                        y2=e2max_mu*e2,
                        x1=x1e, x2=x2e)
    A_shape = 2.5*LogLogParam(energy=energy,
                              y1=z1max_mu*z1,
                              y2=z2max_mu*z2,
                              x1=x1z, x2=x2z,
                              cutoff_value = numu_cutoff)
    return A_ave - (norm_fcn(czenith, A_shape, 0.32) - 0.75*A_shape)

# The NuE flux modification requires that you know the NuMu parameters
# The uncertainty is added to that of NuMu. Assuming they are correlated
def ModNuEFlux(energy, czenith,
               e1mu=1., e2mu=1., z1mu=1., z2mu=1.,
               e1e=1., e2e=1., z1e=1., z2e=1.):

    A_ave = LogLogParam(energy=energy,
                        y1=e1max_mu*e1mu + e1max_e*e1e,
                        y2=e2max_mu*e2mu + e2max_e*e2e,
                        x1=x1e, x2=x2e)
    A_shape = 1.*LogLogParam(energy=energy,
                             y1=z1max_mu*z1mu + z1max_e*z1e,
                             y2=z2max_mu*z2mu + z2max_e*z2e,
                             x1=x1z, x2=x2z,
                             cutoff_value = nue_cutoff)
    return A_ave - (1.5*norm_fcn(czenith, A_shape, 0.4) - 0.7*A_shape)

def modRatioNuBar(prim, true_e, true_cz, nu_nubar, nubar_sys):
    #not sure what nu_nubar is, only found this line in the documentation:
    # +1 applies the change to neutrinos, 0 to antineutrinos. Anything in between is shared
    modfactor = np.ones(len(true_e))
    if 'nue' in prim:
        modfactor = nubar_sys*ModNuEFlux(true_e, true_cz ,
                           1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0)
    elif 'numu' in prim:
        modfactor = nubar_sys * ModNuMuFlux(true_e, true_cz,
                            1.0, 1.0,1.0,1.0)
    else:
        raise ValueError('Function modRatioNuBar not valid for %s '% prim)

    #weightmod = 1. + modfactor*nu_nubar
    #antineutrinos = particle_list['ptype']<0
    #weightmod[antineutrinos] = 1./(1+(1-nu_nubar)*modfactor[antineutrinos])

    if 'bar' in prim:
        return 1./(1+(1-nu_nubar)*modfactor)
    else:
        return 1. + modfactor*nu_nubar

# This I put for completeness. One could do without it.
# Don't use. Has no effect.
#def modRatioMuE(energy, elow, ehigh):
#    return 1.+LogLogParam(energy,
#                          elow*e1max_mu_e, ehigh*e2max_mu_e,
#                          x1e, x2e, 1000.)

def modRatioUpHor(prim, true_e, true_cz, uphor):
    if 'nue' in prim:
        A_shape   = 1.*np.abs(uphor)*LogLogParam(energy=true_e,
                               y1=(z1max_e+z1max_mu),
                               y2=(z2max_e+z2max_mu),
                               x1=x1z, x2=x2z,
                               cutoff_value = nue_cutoff)
    elif 'numu' in prim:
        A_shape   = 1.*np.abs(uphor)*LogLogParam(energy=true_e,
                               y1=z1max_mu,
                               y2=z2max_mu,
                               x1=x1z, x2=x2z,
                               cutoff_value = numu_cutoff)
    else:
        raise ValueError('Function modRatioUpHor not valid for %s '% prim)

    return 1-3.5*np.sign(uphor)*norm_fcn(true_cz, A_shape, 0.35)
