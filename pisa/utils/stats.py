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
           'norm_conv_poisson', 'conv_llh', 'barlow_llh', 'mod_chi2',
           'mcllh_mean', 'mcllh_eff', 'signed_sqrt_mod_chi2', 'generalized_poisson_llh']

__author__ = 'P. Eller, T. Ehrhardt, J.L. Lanfranchi, E. Bourbeau'

__license__ = '''Copyright (c) 2014-2020, The IceCube Collaboration

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

LLH_METRICS = ['llh', 'conv_llh', 'barlow_llh', 'mcllh_mean', 
'mcllh_eff', 'generalized_poisson_llh']
"""Metrics defined that result in measures of log likelihood"""

ALL_METRICS = LLH_METRICS + CHI2_METRICS
"""All metrics defined"""

METRICS_TO_MAXIMIZE = LLH_METRICS
"""Metrics that must be maximized to obtain a better fit"""

METRICS_TO_MINIMIZE = CHI2_METRICS
"""Metrics that must be minimized to obtain a better fit"""


# TODO(philippeller):
# * unit tests to ensure these don't break

def it_got_better(new_metric_val, old_metric_val, metric):
    """Compare metric values and report whether improvement found.
    """
    to_maximize = is_metric_to_maximize(metric)
    if to_maximize:
        got_better = new_metric_val > old_metric_val
    else:
        got_better = new_metric_val < old_metric_val
    return got_better

def is_metric_to_maximize(metric):
    """Check whether the resulting metric has to be maximized or minimized.
    """
    if isinstance(metric, str):
        metric = [metric]
    if all(m in METRICS_TO_MAXIMIZE for m in metric):
        return True
    if all(m in METRICS_TO_MINIMIZE for m in metric):
        return False
    raise ValueError('Defined metrics %s are not compatible' % metric)

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
    * Values in expectation are clipped to the range [SMALL_POS, inf] prior to
      the calculation to avoid infinities due to the divide function.
    * actual_values are allowed to be = 0, since they don't com up in the denominator
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
        np.clip(expected_values, a_min=SMALL_POS, a_max=np.inf,
                out=expected_values)


        delta = actual_values - expected_values

    if np.all(np.abs(delta) < 5*FTYPE_PREC):
        return np.zeros_like(delta, dtype=FTYPE)

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

    #
    # natural logarith m of the Poisson probability
    # (uses Stirling's approximation to estimate ln(k!) ~ kln(k)-k)
    #
    llh_val = actual_values*np.log(expected_values) - expected_values
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

    llh_val = likelihood_functions.poisson_gamma(
        data=actual_values, sum_w=expected_values, sum_w2=sigma**2, a=0, b=0
        )

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

    llh_val = likelihood_functions.poisson_gamma(
        data=actual_values, sum_w=expected_values, sum_w2=sigma**2, a=1, b=0
    )
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
        total += np.log(max(SMALL_POS, norm_conv_poisson(*triplets[i]))) # FIXME? (cf. pylint)
        total -= np.log(max(SMALL_POS, norm_conv_poisson(*norm_triplets[i]))) # FIXME? (cf. pylint)
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

    llh_val = likelihood_functions.barlowLLH(actual_values, unweighted, weights)
    return llh_val

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

def signed_sqrt_mod_chi2(actual_values, expected_values):
    """Compute a (signed) pull value taking into account uncertainty terms.

    Parameters
    ----------
    actual_values, expected_values : numpy.ndarrays of same shape

    Returns
    -------
    m_pull : numpy.ndarray of same shape as inputs
        Pull values corresponding to each pair of elements in
        the inputs

    """
    # Replace 0's with small positive numbers to avoid inf in log
    np.clip(expected_values, a_min=SMALL_POS, a_max=np.inf,
            out=expected_values)
    actual_values = unp.nominal_values(actual_values).ravel()
    sigma = unp.std_devs(expected_values).ravel()
    expected_values = unp.nominal_values(expected_values).ravel()
    m_pull = (
        (actual_values - expected_values) / np.sqrt(sigma**2 + expected_values)
    )
    return m_pull
      
#
# Generalized Poisson-gamma llh from 1902.08831
#
def generalized_poisson_llh(actual_values, expected_values=None, empty_bins=None):
    '''Compute the generalized Poisson likelihood as formulated in https://arxiv.org/abs/1902.08831


    Note that unlike the other likelihood functions, expected_values
    is expected to be a ditribution maker

    inputs:
    ------

    actual_values: flattened hist of a Map object

    expected_values: OrderedDict of MapSets

    empty_bins: None, list or np.ndarray (list the bin indices that are empty)

    returns:
    --------
    llh_per_bin : bin-wise llh values, in a numpy array

    '''
    from collections import OrderedDict

    
    assert isinstance(expected_values, OrderedDict), 'ERROR: expected_values must be an OrderedDict of MapSet objects'
    assert 'weights' in expected_values.keys(), 'ERROR: expected_values need a key named "weights"'
    assert 'llh_alphas' in expected_values.keys(), 'ERROR: expected_values need a key named "llh_alphas"'
    assert 'llh_betas' in expected_values.keys(), 'ERROR: expected_values need a key named "llh_betas"'

    num_bins = actual_values.flatten().shape[0]
    llh_per_bin = np.zeros(num_bins)
    actual_values = unp.nominal_values(actual_values).ravel()

    # If no empty bins are specified, we assume that all of them should be included
    if empty_bins is None:
        empty_bins = []

    for bin_i in range(num_bins):

        # TODO: sometimes the histogram spits out uncertainty objects, sometimes not. 
        #       Not sure why.
        data_count = actual_values.astype(np.int64)[bin_i]

        # Automatically add a huge number if a bin has non zero data count
        # but completely empty MC
        if bin_i in empty_bins:
            if data_count > 0:
                llh_per_bin[bin_i] = np.log(SMALL_POS)
            continue

        # Make sure that no weight sum is negative. Crash if there are
        weight_sum = np.array([m.hist.flatten()[bin_i] for m in expected_values['weights'].maps])
        if (weight_sum<0).sum()>0:
            logging.debug('\n\n\n')
            logging.debug('weights that are causing problem: ')
            logging.debug(weight_sum[weight_sum<0])
            logging.debug((weight_sum<0).sum())
            logging.debug('\n\n\n')
        assert np.all(weight_sum >= 0), 'ERROR: negative weights detected'

        #
        # If the number of MC events is high, compute a normal poisson probability
        #
        n_mc_events = np.array([m.hist.flatten()[bin_i] for m in expected_values['n_mc_events'].maps])
        if np.all(n_mc_events>100):

            logP = data_count*np.log(weight_sum.sum())-weight_sum.sum()-(data_count*np.log(data_count)-data_count)
            llh_per_bin[bin_i] = logP
            
        else:
            from pisa.utils.llh_defs.poisson import fast_pgmix

            alphas = np.array([m.hist.flatten()[bin_i] for m in expected_values['llh_alphas'].maps])
            betas  = np.array([m.hist.flatten()[bin_i] for m in expected_values['llh_betas'].maps])

            # Remove the NaN's 
            mask = np.isfinite(alphas)*np.isfinite(betas)

            # Check that the alpha and betas make sense
            assert np.all(alphas[mask] > 0), 'ERROR: detected alpha values <=0'
            assert np.all(betas[mask] > 0 ), 'ERROR: detected beta values <=0'


            llh_of_bin = fast_pgmix(data_count, alphas[mask], betas[mask])
            llh_per_bin[bin_i] = llh_of_bin

    return llh_per_bin
    

def approximate_poisson_normal(data_count, alphas=None, betas=None, use_c=False):
    '''
    Compute the likelihood of a marginalized poisson-gamma
    function, using a single normal distribution instead of
    the convolution of gamma function

    This formula can be used when the MC counts are really
    high, and where the gamma function throws infinite values

    '''
    from scipy.integrate import quad
    import numpy as np

    gamma_mean = np.sum(alphas/betas)
    gamma_sigma = np.sqrt(np.sum(alphas/betas**2.))

    #
    # Define integration range as +- 5 sigma
    #
    lower = max(0,gamma_mean-5*gamma_sigma)
    upper = gamma_mean+5*gamma_sigma

    #
    # integrate over the boundaries
    #
    if use_c:

        import os, ctypes
        import numpy as np
        from scipy import integrate, LowLevelCallable

        lib = ctypes.CDLL(os.path.abspath('/groups/icecube/bourdeet/pisa/pisa/utils/poisson_normal.so'))
        lib.approximate_gamma_poisson_integrand.restype = ctypes.c_double
        lib.approximate_gamma_poisson_integrand.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

        # Define the parameters
        params = (ctypes.c_double*3)()

        params[0] = data_count
        params[1] = gamma_mean
        params[2] = gamma_sigma

        user_data = ctypes.cast(params, ctypes.c_void_p)
        func = LowLevelCallable(lib.approximate_gamma_poisson_integrand, user_data)
        LH = quad(func, lower, upper)[0]
        #print('lower ',lower,' upper: ',upper,' LH: ',LH)
    else:

        LH = quad(approximate_poisson_normal_python, lower, upper, args=(data_count, gamma_mean, gamma_sigma))[0]
        #print('lower ',lower,' upper: ',upper,' data_count: ',data_count,' mean: ', gamma_mean, ' sigma: ',gamma_sigma, ' LH: ',LH)

    LH = max(SMALL_POS,LH) 
    return np.log(LH)



def approximate_poisson_normal_python(lamb, k, A, B):

    from scipy.stats import norm
    
    normal_term = norm.pdf(lamb, loc=A, scale=B)
    normal_poisson = norm.pdf(k, loc=lamb, scale=np.sqrt(lamb))

    return normal_term*normal_poisson
