"""
Statistical functions
"""


from __future__ import absolute_import, division

import numpy as np
from scipy.special import gammaln
from uncertainties import unumpy as unp

from pisa import FTYPE
from pisa.utils.comparisons import FTYPE_PREC, isbarenumeric
from pisa.utils.log import logging
from pisa.utils import likelihood_functions

__all__ = ['SMALL_POS', 'CHI2_METRICS', 'LLH_METRICS', 'ALL_METRICS',
           'maperror_logmsg',
           'chi2', 'llh', 'log_poisson', 'log_smear', 'conv_poisson',
           'norm_conv_poisson', 'conv_llh', 'barlow_llh', 'mod_chi2', 'mcllh_mean', 'mcllh_eff']

__author__ = 'P. Eller, T. Ehrhardt, J.L. Lanfranchi'

__license__ = '''Copyright (c) 2014-2017, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


SMALL_POS = 1e-10 #if FTYPE == np.float64 else FTYPE_PREC
"""A small positive number with which to replace numbers smaller than it"""

CHI2_METRICS = ['chi2', 'mod_chi2']
"""Metrics defined that result in measures of chi squared"""

LLH_METRICS = ['llh', 'conv_llh', 'barlow_llh', 'mcllh_mean', 'mcllh_eff']
"""Metrics defined that result in measures of log likelihood"""

ALL_METRICS = LLH_METRICS + CHI2_METRICS
"""All metrics defined"""

METRICS_TO_MAXIMIZE = LLH_METRICS
"""Metrics that must be maximized to obtain a better fit"""

METRICS_TO_MINIMIZE = CHI2_METRICS
"""Metrics that must be minimized to obtain a better fit"""


# TODO(philippeller):
# * unit tests to ensure these don't break


def maperror_logmsg(m):
    """Create message with thorough info about a map for logging purposes"""
    with np.errstate(invalid='ignore'):
        msg = ''
        msg += '    min val : %s\n' %np.nanmin(m)
        msg += '    max val : %s\n' %np.nanmax(m)
        msg += '    mean val: %s\n' %np.nanmean(m)
        msg += '    num < 0 : %s\n' %np.sum(m < 0)
        msg += '    num == 0: %s\n' %np.sum(m == 0)
        msg += '    num > 0 : %s\n' %np.sum(m > 0)
        msg += '    num nan : %s\n' %np.sum(np.isnan(m))
    return msg


def chi2(actual_values, expected_values):
    """Compute the chi-square between each value in `actual_values` and
    `expected_values`.

    Parameters
    ----------
    actual_values, expected_values : numpy.ndarrays of same shape

    Returns
    -------
    chi2 : numpy.ndarray of same shape as inputs
        chi-squared values corresponding to each pair of elements in the inputs

    Notes
    -----
    * Uncertainties are not propagated through this calculation.
    * Values in each input are clipped to the range [SMALL_POS, inf] prior to
      the calculation to avoid infinities due to the divide function.

    """
    if actual_values.shape != expected_values.shape:
        raise ValueError(
            'Shape mismatch: actual_values.shape = %s,'
            ' expected_values.shape = %s'
            % (actual_values.shape, expected_values.shape)
        )

    # Convert to simple numpy arrays containing floats
    if not isbarenumeric(actual_values):
        actual_values = unp.nominal_values(actual_values)
    if not isbarenumeric(expected_values):
        expected_values = unp.nominal_values(expected_values)

    with np.errstate(invalid='ignore'):
        # Mask off any nan expected values (these are assumed to be ok)
        actual_values = np.ma.masked_invalid(actual_values)
        expected_values = np.ma.masked_invalid(expected_values)

        # TODO: this check (and the same for `actual_values`) should probably
        # be done elsewhere... maybe?
        if np.any(actual_values < 0):
            msg = ('`actual_values` must all be >= 0...\n'
                   + maperror_logmsg(actual_values))
            raise ValueError(msg)

        if np.any(expected_values < 0):
            msg = ('`expected_values` must all be >= 0...\n'
                   + maperror_logmsg(expected_values))
            raise ValueError(msg)

        # TODO: Is this okay to do? Mathematically suspect at best, and can
        #       still destroy a minimizer's hopes and dreams...

        # Replace 0's with small positive numbers to avoid inf in division
        np.clip(actual_values, a_min=SMALL_POS, a_max=np.inf,
                out=actual_values)
        np.clip(expected_values, a_min=SMALL_POS, a_max=np.inf,
                out=expected_values)

        delta = actual_values - expected_values

    if np.all(np.abs(delta) < 5*FTYPE_PREC):
        return np.zeros_like(delta, dtype=FTYPE)

    assert np.all(actual_values > 0), str(actual_values)
    #chi2_val = np.square(delta) / actual_values
    chi2_val = np.square(delta) / expected_values
    assert np.all(chi2_val >= 0), str(chi2_val[chi2_val < 0])
    return chi2_val


def llh(actual_values, expected_values):
    """Compute the log-likelihoods (llh) that each count in `actual_values`
    came from the the corresponding expected value in `expected_values`.

    Parameters
    ----------
    actual_values, expected_values : numpy.ndarrays of same shape

    Returns
    -------
    llh : numpy.ndarray of same shape as the inputs
        llh corresponding to each pair of elements in `actual_values` and
        `expected_values`.

    Notes
    -----
    * Uncertainties are not propagated through this calculation.
    * Values in `expected_values` are clipped to the range [SMALL_POS, inf]
      prior to the calculation to avoid infinities due to the log function.

    """
    assert actual_values.shape == expected_values.shape

    # Convert to simple numpy arrays containing floats
    if not isbarenumeric(actual_values):
        actual_values = unp.nominal_values(actual_values)
    if not isbarenumeric(expected_values):
        expected_values = unp.nominal_values(expected_values)

    with np.errstate(invalid='ignore'):
        # Mask off any nan expected values (these are assumed to be ok)
        actual_values = np.ma.masked_invalid(actual_values)
        expected_values = np.ma.masked_invalid(expected_values)

        # Check that new array contains all valid entries
        if np.any(actual_values < 0):
            msg = ('`actual_values` must all be >= 0...\n'
                   + maperror_logmsg(actual_values))
            raise ValueError(msg)

        # TODO: How should we handle nan / masked values in the "data"
        # (actual_values) distribution? How about negative numbers?

        # Make sure actual values (aka "data") are valid -- no infs, no nans,
        # etc.
        if np.any((actual_values < 0) | ~np.isfinite(actual_values)):
            msg = ('`actual_values` must be >= 0 and neither inf nor nan...\n'
                   + maperror_logmsg(actual_values))
            raise ValueError(msg)

        # Check that new array contains all valid entries
        if np.any(expected_values < 0.0):
            msg = ('`expected_values` must all be >= 0...\n'
                   + maperror_logmsg(expected_values))
            raise ValueError(msg)

        # Replace 0's with small positive numbers to avoid inf in log
        np.clip(expected_values, a_min=SMALL_POS, a_max=np.inf,
                out=expected_values)

    llh_val = actual_values*np.log(expected_values) - expected_values

    # Do following to center around 0
    llh_val -= actual_values*np.log(actual_values) - actual_values

    return llh_val

def mcllh_mean(actual_values, expected_values):
    """Compute the log-likelihood (llh) based on LMean in table 2 - https://doi.org/10.1007/JHEP06(2019)030
    accounting for finite MC statistics.
    This is the second most recommended likelihood in the paper.

    Parameters
    ----------
    actual_values, expected_values : numpy.ndarrays of same shape

    Returns
    -------
    llh : numpy.ndarray of same shape as the inputs
        llh corresponding to each pair of elements in `actual_values` and
        `expected_values`.

    Notes
    -----
    * 
    """
    assert actual_values.shape == expected_values.shape

    # Convert to simple numpy arrays containing floats
    actual_values = unp.nominal_values(actual_values).ravel()
    sigma = unp.std_devs(expected_values).ravel()
    expected_values = unp.nominal_values(expected_values).ravel() 
    
    with np.errstate(invalid='ignore'):
        # Mask off any nan expected values (these are assumed to be ok)
        actual_values = np.ma.masked_invalid(actual_values)
        expected_values = np.ma.masked_invalid(expected_values)

        # Check that new array contains all valid entries
        if np.any(actual_values < 0):
            msg = ('`actual_values` must all be >= 0...\n'
                   + maperror_logmsg(actual_values))
            raise ValueError(msg)

        # TODO: How should we handle nan / masked values in the "data"
        # (actual_values) distribution? How about negative numbers?

        # Make sure actual values (aka "data") are valid -- no infs, no nans,
        # etc.
        if np.any((actual_values < 0) | ~np.isfinite(actual_values)):
            msg = ('`actual_values` must be >= 0 and neither inf nor nan...\n'
                   + maperror_logmsg(actual_values))
            raise ValueError(msg)

        # Check that new array contains all valid entries
        if np.any(expected_values < 0.0):
            msg = ('`expected_values` must all be >= 0...\n'
                   + maperror_logmsg(expected_values))
            raise ValueError(msg)

    llh_val = likelihood_functions.poisson_gamma(actual_values, expected_values, sigma**2, a=0, b=0)
    return llh_val


def mcllh_eff(actual_values, expected_values):
    """Compute the log-likelihood (llh) based on eq. 3.16 - https://doi.org/10.1007/JHEP06(2019)030
    accounting for finite MC statistics.
    This is the most recommended likelihood in the paper.

    Parameters
    ----------
    actual_values, expected_values : numpy.ndarrays of same shape

    Returns
    -------
    llh : numpy.ndarray of same shape as the inputs
        llh corresponding to each pair of elements in `actual_values` and
        `expected_values`.

    Notes
    -----
    * 
    """
    assert actual_values.shape == expected_values.shape

    # Convert to simple numpy arrays containing floats
    actual_values = unp.nominal_values(actual_values).ravel()
    sigma = unp.std_devs(expected_values).ravel()
    expected_values = unp.nominal_values(expected_values).ravel() 
    
    with np.errstate(invalid='ignore'):
        # Mask off any nan expected values (these are assumed to be ok)
        actual_values = np.ma.masked_invalid(actual_values)
        expected_values = np.ma.masked_invalid(expected_values)

        # Check that new array contains all valid entries
        if np.any(actual_values < 0):
            msg = ('`actual_values` must all be >= 0...\n'
                   + maperror_logmsg(actual_values))
            raise ValueError(msg)

        # TODO: How should we handle nan / masked values in the "data"
        # (actual_values) distribution? How about negative numbers?

        # Make sure actual values (aka "data") are valid -- no infs, no nans,
        # etc.
        if np.any((actual_values < 0) | ~np.isfinite(actual_values)):
            msg = ('`actual_values` must be >= 0 and neither inf nor nan...\n'
                   + maperror_logmsg(actual_values))
            raise ValueError(msg)

        # Check that new array contains all valid entries
        if np.any(expected_values < 0.0):
            msg = ('`expected_values` must all be >= 0...\n'
                   + maperror_logmsg(expected_values))
            raise ValueError(msg)

    llh_val = likelihood_functions.poisson_gamma(actual_values, expected_values, sigma**2, a=1, b=0)
    return llh_val



def log_poisson(k, l):
    r"""Calculate the log of a poisson pdf

    .. math::
        p(k,l) = \log\left( l^k \cdot e^{-l}/k! \right)

    Parameters
    ----------
    k : float
    l : float

    Returns
    -------

    log of poisson

    """
    return k*np.log(l) -l - gammaln(k+1)


def log_smear(x, sigma):
    r"""Calculate the log of a normal pdf

    .. math::
        p(x, \sigma) = \log\left( (\sigma \sqrt{2\pi})^{-1} \exp( -x^2 / 2\sigma^2 ) \right)

    Parameters
    ----------
    x : float
    sigma : float

    Returns
    -------
    log of gaussian

    """
    return (
        -np.log(sigma) - 0.5*np.log(2*np.pi) - x**2 / (2*sigma**2)
    )


def conv_poisson(k, l, s, nsigma=3, steps=50):
    r"""Poisson pdf

    .. math::
        p(k,l) = l^k \cdot e^{-l}/k!

    Parameters
    ----------
    k : float
    l : float
    s : float
        sigma for smearing term (= the uncertainty to be accounted for)
    nsigma : int
        The ange in sigmas over which to do the convolution, 3 sigmas is > 99%,
        so should be enough
    steps : int
        Number of steps to do the intergration in (actual steps are 2*steps + 1,
        so this is the steps to each side of the gaussian smearing term)

    Returns
    -------
    float
        convoluted poissson likelihood

    """
    # Replace 0's with small positive numbers to avoid inf in log
    l = max(SMALL_POS, l)
    st = 2*(steps + 1)
    conv_x = np.linspace(-nsigma*s, +nsigma*s, st)[:-1]+nsigma*s/(st-1.)
    conv_y = log_smear(conv_x, s)
    f_x = conv_x + l
    #f_x = conv_x + k
    # Avoid zero values for lambda
    idx = np.argmax(f_x > 0)
    f_y = log_poisson(k, f_x[idx:])
    #f_y = log_poisson(f_x[idx:], l)
    if np.isnan(f_y).any():
        logging.error('`NaN values`:')
        logging.error('idx = %d', idx)
        logging.error('s = %s', s)
        logging.error('l = %s', l)
        logging.error('f_x = %s', f_x)
        logging.error('f_y = %s', f_y)
    f_y = np.nan_to_num(f_y)
    conv = np.exp(conv_y[idx:] + f_y)
    norm = np.sum(np.exp(conv_y))
    return conv.sum()/norm


def norm_conv_poisson(k, l, s, nsigma=3, steps=50):
    """Convoluted poisson likelihood normalized so that the value at k=l
    (asimov) does not change

    Parameters
    ----------
    k : float
    l : float
    s : float
        sigma for smearing term (= the uncertainty to be accounted for)
    nsigma : int
        The range in sigmas over which to do the convolution, 3 sigmas is >
        99%, so should be enough
    steps : int
        Number of steps to do the intergration in (actual steps are 2*steps + 1,
        so this is the steps to each side of the gaussian smearing term)

    Returns
    -------
    likelihood
        Convoluted poisson likelihood normalized so that the value at k=l
        (asimov) does not change

    """
    cp = conv_poisson(k, l, s, nsigma=nsigma, steps=steps)
    n1 = np.exp(log_poisson(l, l))
    n2 = conv_poisson(l, l, s, nsigma=nsigma, steps=steps)
    return cp*n1/n2


def conv_llh(actual_values, expected_values):
    """Compute the convolution llh using the uncertainty on the expected values
    to smear out the poisson PDFs

    Parameters
    ----------
    actual_values, expected_values : numpy.ndarrays of same shape

    Returns
    -------
    total log of convoluted poisson likelihood

    """
    actual_values = unp.nominal_values(actual_values).ravel()
    sigma = unp.std_devs(expected_values).ravel()
    expected_values = unp.nominal_values(expected_values).ravel()
    triplets = np.array([actual_values, expected_values, sigma]).T
    norm_triplets = np.array([actual_values, actual_values, sigma]).T
    total = 0
    for i in range(len(triplets)):
        total += np.log(max(SMALL_POS, norm_conv_poisson(*triplets[i])))
        total -= np.log(max(SMALL_POS, norm_conv_poisson(*norm_triplets[i])))
    return total

def barlow_llh(actual_values, expected_values):
    """Compute the Barlow LLH taking into account finite statistics.
    The likelihood is described in this paper: https://doi.org/10.1016/0010-4655(93)90005-W
    Parameters
    ----------
    actual_values, expected_values : numpy.ndarrays of same shape

    Returns
    -------
    barlow_llh: numpy.ndarray

    """
     
    actual_values = unp.nominal_values(actual_values).ravel()
    sigmas = unp.std_devs(expected_values).ravel()
    expected_values = unp.nominal_values(expected_values).ravel()

    with np.errstate(invalid='ignore'):
        # Mask off any nan expected values (these are assumed to be ok)
        actual_values = np.ma.masked_invalid(actual_values)
        expected_values = np.ma.masked_invalid(expected_values)

        # Check that new array contains all valid entries
        if np.any(actual_values < 0):
            msg = ('`actual_values` must all be >= 0...\n'
                   + maperror_logmsg(actual_values))
            raise ValueError(msg)

        # TODO: How should we handle nan / masked values in the "data"
        # (actual_values) distribution? How about negative numbers?

        # Make sure actual values (aka "data") are valid -- no infs, no nans,
        # etc.
        if np.any((actual_values < 0) | ~np.isfinite(actual_values)):
            msg = ('`actual_values` must be >= 0 and neither inf nor nan...\n'
                   + maperror_logmsg(actual_values))
            raise ValueError(msg)

        # Check that new array contains all valid entries
        if np.any(expected_values < 0.0):
            msg = ('`expected_values` must all be >= 0...\n'
                   + maperror_logmsg(expected_values))
            raise ValueError(msg)
    
    # TODO(tahmid): Run checks in case expected_values and/or corresponding sigma == 0
    # and handle these appropriately. If sigma/ev == 0 the code below will fail.
    unweighted = np.array([(ev/s)**2 for ev, s in zip(expected_values, sigmas)])
    weights = np.array([s**2/ev for ev, s in zip(expected_values, sigmas)])

    llh = likelihood_functions.barlowLLH(actual_values, unweighted, weights)
    return llh

def mod_chi2(actual_values, expected_values):
    """Compute the chi-square value taking into account uncertainty terms
    (incl. e.g. finite stats)

    Parameters
    ----------
    actual_values, expected_values : numpy.ndarrays of same shape

    Returns
    -------
    m_chi2 : numpy.ndarray of same shape as inputs
        Modified chi-squared values corresponding to each pair of elements in
        the inputs

    """
    # Replace 0's with small positive numbers to avoid inf in log
    np.clip(expected_values, a_min=SMALL_POS, a_max=np.inf,
            out=expected_values)
    actual_values = unp.nominal_values(actual_values).ravel()
    sigma = unp.std_devs(expected_values).ravel()
    expected_values = unp.nominal_values(expected_values).ravel()
    m_chi2 = (
        (actual_values - expected_values)**2 / (sigma**2 + expected_values)
    )
    return m_chi2
