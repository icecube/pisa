"""
This script contains functions to compute Barlow-Beeston Likelihood, as well as
an implementation of the Poisson-Gamma mixture.

These likelihood implementations take into account uncertainties due to
finite Monte Carlo statistics.

The functions are called in stats.py to apply them to histograms.

Note that these likelihoods are NOT centered around 0 (i.e. if data == expectation, LLH != 0)
"""
from __future__ import print_function

import numpy as np
from scipy import special
from scipy import optimize

__author__ = "Ahnaf Tahmid"
__email__ = "tahmid@ualberta.ca"
__date__ = "2019-08-15"

def poisson_gamma(data, sum_w, sum_w2, a=1, b=0):
    """
    Log-likelihood based on the poisson-gamma mixture. This is a Poisson likelihood using a Gamma prior.
    This implementation is based on the implementation of Austin Schneider (aschneider@icecube.wisc.edu)
    -- Input variables --
    data = data histogram
    sum_w = MC histogram
    sum_w2 = Uncertainty map (sum of weights squared in each bin)
    a, b = hyperparameters of gamma prior for MC counts; default values of a = 1 and b = 0 corresponds to LEff (eq 3.16) https://doi.org/10.1007/JHEP06(2019)030
           a = 0 and b = 0 corresponds to LMean (Table 2) https://doi.org/10.1007/JHEP06(2019)030

    -- Output --
    llh = LLH values in each bin

    -- Notes --
    Shape of data, sum_w, sum_w2 and llh are identical
    """

    llh = np.ones(data.shape) * -np.inf # Binwise LLH values

    bad_bins = np.logical_or(sum_w <= 0, sum_w2 < 0) # Bins where the MC is 0 or less than 0

    # LLH would be 0 for the bad bins if the data is 0
    zero_llh = np.logical_and(data == 0, bad_bins)
    llh[zero_llh] = 0 # Zero LLH for these bins if data is also 0

    good_bins = ~bad_bins
    poisson_bins = np.logical_and(sum_w2 == 0, good_bins) # In the limit that sum_w2 == 0, the llh converges to poisson

    llh[poisson_bins] = poissonLLH(data[poisson_bins], sum_w[poisson_bins]) # Poisson LLH since limiting case

   # Calculate hyperparameters for the gamma posterior for MC counts
    regular_bins = np.logical_and(good_bins, ~poisson_bins) # Bins on which the poisson_gamma LLH would be evaluated
    alpha = sum_w[regular_bins]**2./sum_w2[regular_bins] + a
    beta = sum_w[regular_bins]/sum_w2[regular_bins] + b

    k = data[regular_bins]
    # Poisson-gamma LLH
    L = alpha*np.log(beta) + special.loggamma(k+alpha).real - special.loggamma(k+1.0).real - (k+alpha)*np.log1p(beta) - special.loggamma(alpha).real
    llh[regular_bins] = L

    return llh

def poissonLLH(data, mc):
    """
    Standard poisson likelihood
    -- Input variables --
    data = data histogram
    mc = MC histogram

    -- Output --
    LLH values in each bin

    -- Notes --
    Shape of data, mc are identical
    """
    return data*np.log(mc) - mc - special.loggamma(data + 1)

def barlowLLH(data, unweighted_mc, weights):
    """
    Barlow-Beeston log-likelihood (constant terms not omitted)
    Link to paper: https://doi.org/10.1016/0010-4655(93)90005-W
    -- Input variables --
    data = data histogram
    mc = weighted MC histogram
    unweighted_mc = unweighted MC histogream
    weights = weight of each bin

    -- Output --
    llh = LLH values in each bin

    -- Notes --
    Shape of data, mc, unweighted_mc, weights and llh must be identical
    """

    # The actual barlow LLH
    def llh(A_, k, w, a):
        SMALL_VAL = 1.e-10

        f = w*A_

        # Takes care of log(0) problems
        if not len(A_) > 1:
            f = max((f, SMALL_VAL))
            A_ = max((A_, SMALL_VAL))

        # The loggamma() terms takes care of the log(value!) for non-integer values
        return -1.*(k*np.log(f) - f + a*np.log(A_) - A_ - special.loggamma(k+1) - special.loggamma(a+1))

    A = np.array(unweighted_mc) # Expected unweighted counts in a bin
    # For each bin the appropriate 'A' will now be found using the current counts as a seed
    # This will be done by using a minimiser to ensure that the value of 'A' found minimises the -LLH

    # Find values of A such that llh in each bin is a maximum
    for i, val in enumerate(A):
        # If the unweighted MC counts in the bin is 0, A = 0
        if val == 0:
            continue
        # Otherwise, find the value of A
        arg = (data[i], weights[i], unweighted_mc[i])
        # Powell works fast and is fine for our purposes
        result = optimize.minimize(fun=llh, x0=val, args=arg, method='Powell')

        # Check that the minimisation ran properly
        if result.success:
            A[i] = result.x
        else:
            print("Something went wrong...")
            print("Minimiser message: ")
            print("------------------")
            print(result.message)
            return -np.inf

    LLH = llh(A, data, weights, unweighted_mc)

    return -1*LLH # Return LLH (not negative LLH)


