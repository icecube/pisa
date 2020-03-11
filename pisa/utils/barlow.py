#!/usr/bin/python
"""
`Likelihoods` class for computing Poisson and Barlow likelihoods.

The Poisson likelihood assumes the template being compared against is the
perfect expectation, while the Barlow likelihood accounts for the template
being imperfect due to being generated from finite Monte Carlo statistics.
"""


from __future__ import absolute_import, division, print_function

from copy import copy
import sys

import numpy as np
from scipy.optimize import minimize

__all__ = ['Likelihoods']
__author__ = 'Michael Larson'
__email__ = 'mlarson@nbi.ku.dk'
__date__ = '2016-03-14'

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


#------------------------------------------------------------------------------
# The following is adapted from user Alfe, https://stackoverflow.com/a/16787722
def handle_uncaught_exception(exctype, value, trace):
    """Exception hook that allows for exiting with a specified exit code.

    Replaces sys.exithook; see :class:`ExitCodeException`
    """
    OLD_EXCEPTHOOK(exctype, value, trace)
    if hasattr(value, 'exitcode'):
        sys.exit(value.exitcode)

sys.excepthook, OLD_EXCEPTHOOK = handle_uncaught_exception, sys.excepthook
#------------------------------------------------------------------------------


class ShapeError(Exception):
    exitcode = 100

class NaNValueError(Exception):
    exitcode = 101

class ArgValueError(Exception):
    exitcode = 102


class Likelihoods(object):
    """
A class to handle the likelihood calculations in OscFit. It can
be used in other poisson binned-likelihood calculators as well.
The class includes two likelihood spaces: the Poisson and the
Barlow LLH.

The Poisson likelihood is the ideal case and assumes infinite
statistical accuracy on the Monte Carlo distributions. This is
the simple case and has been used in the past for analyses, but
should not be used if there's a significant statistical error on
one of the samples.

The Barlow likelihood is an implementation of the likelihood model
given in doi:10.1016/0010-4655(93)90005-W ("Fitting using finite
Monte Carlo samples" by Barlow and Beeston). This requires the
unweighted distributions of the MC and involves assuming that each
MC samples is drawn from an underlying "true" expectation distribution
which should be fit. This gives (number of bins)x(number of samples)
new parameters to fit and allows the shape of the distributions to
fluctuate in order to fit both the observed MC and the observed
data. This gives a more informed result and allows one to estimate
the effect of the finite MC samples.

To use this, you need to run set_data, set_mc, and the set_unweighted
functions. 

.. important:: the `set_mc` function takes a histogram of the average
    weight per event for each bin! not the total weighted histogram!

Simply calling the get_llh function after these will return the
best fit LLH for your chosen likelihood function. The function takes
the name of the likelihood space ("Poisson" or "Barlow"). You can
retrieve the best fit weighted plots using the get_plot (total best-fit
histogram including all samples) and the get_single_plots (the list
of best-fit weighted histograms for each sample).
    """
    mc_histograms = None
    unweighted_histograms = None
    data_histogram = None
    shape = None
    bestfit_plots = None
    current_bin = None

    def __init__(self):
        """Instantiate and set all of the defaults"""
        self.mc_histograms = None
        self.unweighted_histograms = None
        self.data_histogram = None
        self.shape = None
        self.bestfit_plots = None
        self.current_bin = None

    def reset(self):
        """Re-instantiate so that we can reuse the object"""
        self.__init__()

    def set_data(self, data_histogram):
        """Set up the data histogram. This histogram is flattened in order to
        make the looping later on a bit simpler to handle."""
        if not self.shape:
            self.shape = data_histogram.shape

        if data_histogram.shape != self.shape:
            raise ShapeError(
                "Data histogram has shape {} but expected histogram shape {}"
                .format(data_histogram.shape, self.shape),
                exitcode=100
            )

        self.data_histogram = data_histogram.flatten()

    def set_mc(self, mc_histograms):
        """Set up the MC histogram. Each histogram is flattened in order to
        make the looping later on a bit simpler to handle. The values in each
        bin should correspond to the weight-per-event in that bin, NOT the
        total weight in that bin!

        Make sure you don't have any nulls here.

        """
        if not self.shape:
            self.shape = mc_histograms.values()[0].shape

        if np.any(np.isnan(mc_histograms)):
            raise NaNValueError(
                "At least one bin in your MC histogram is NaN! Take a look"
                " and decide how best to handle this",
                exitcode=101
            )

        flat_histograms = []
        for j in range(mc_histograms.shape[0]):
            if not self.shape == mc_histograms[j].shape:
                raise ShapeError(
                    "MC Histogram {} has shape {} but expected shape {}"
                    .format(j, mc_histograms[j].shape, self.shape)
                )

            flat_histograms.append(mc_histograms[j].flatten())

        self.mc_histograms = np.array(flat_histograms)

    def set_unweighted(self, unweighted_histograms):
        """Save the unweighted distributions in the MC. These can include 0s."""
        if not self.shape:
            self.shape = unweighted_histograms.values()[0].shape

        flat_histograms = []
        for j in range(unweighted_histograms.shape[0]):
            if not self.shape == unweighted_histograms[j].shape:
                raise ShapeError(
                    "Unweighted histogram {} has shape {} but expected shape"
                    " {}"
                    .format(j, unweighted_histograms[j].shape, self.shape)
                )
            flat_histograms.append(unweighted_histograms[j].flatten())

            self.unweighted_histograms = np.array(flat_histograms)

    def get_plot(self):
        """Get the total weighted best-fit histogram post-fit."""
        if self.bestfit_plots is None:
            return

        result = np.sum(self.get_single_plots(), axis=0)
        return result

    def get_single_plots(self):
        """Get the individual weighted best-fit histograms post-fit."""
        if self.bestfit_plots is None:
            return None
        result = np.multiply(self.mc_histograms, self.bestfit_plots)
        target_shape = result.shape[0], self.shape[0], self.shape[1]
        return np.reshape(result, target_shape)

    def get_llh(self, llh_type):
        """Calculate the likelihood given the data, average weights, and the
        unweighted histograms. You can choose between "Poisson" and "Barlow"
        likelihoods at the moment.

        If using the "Barlow" LLH, note that the method is picked to be Powell
        with 25 iterations maximum per step. This is not optimized at all and
        was explicitly chosen simply because it "worked". This doesn't work
        with the bounds set in the normal way, so the positive-definiteness of
        the rates is enforced in the get_llh_barlow_bin method.

        """
        llh_type = llh_type.lower()
        self.bestfit_plots = copy(self.unweighted_histograms)
        self.current_bin = 0

        # The simplest case: the Poisson binned likelihood
        if llh_type == "poisson":
            poisson_llh = self.get_llh_poisson()
            return poisson_llh

        # The more complicated case: The Barlow LLH
        # This requires a separate minimization step in each bin to estimate
        #  the expected rate in each bin from each MC sample using constraints
        #  from the data and the observed MC distribution.
        elif llh_type == "barlow":
            llh = 0
            for bin_n in range(len(self.data_histogram)):
                self.current_bin = bin_n
                bin_result = minimize(
                    fun=self.get_llh_barlow_bin,
                    x0=self.bestfit_plots[:, bin_n],
                    method="Powell",
                    options={'maxiter': 25, 'disp': False}
                )
                self.bestfit_plots[:, bin_n] = bin_result.x
                llh += bin_result.fun

            self.current_bin = None
            return llh

        raise ArgValueError(
            'Unknown `llh_type` "{}". Choose either "Poisson" (ideal) or'
            ' "Barlow" (including MC statistical errors).'
            .format(llh_type)
        )

    def get_llh_barlow_bin(self, a_i):
        """The Barlow LLH finds the best-fit "expected" MC distribution using
        both the data and observed MC as constraints. Each bin is independent
        in this calculation, since the assumption is that there are no
        correlations between bins. This likely leads to a somewhat worse limit
        than you might get otherwise, but at least its conservative.

        If you have a good idea for how to extend this to include bin-to-bin,
        let me know and I can help implement it.

        """
        if any([a_i[j] < 0 for j in range(len(a_i))]):
            return 1e10
        i = self.current_bin
        di = self.data_histogram[i]
        fi = np.sum(np.multiply(self.mc_histograms[:, i], a_i))
        ai = self.unweighted_histograms[:, i]

        llh = 0

        # This is the standard Poisson LLH comparing data to the total weighted
        # MC
        if fi > 0:
            llh += di * np.log(fi) - fi
        if di > 0:
            llh -= di * np.log(di) - di

        # The heart of the Barlow LLH. Fitting the a_i (expected number of
        # events in the MC sample for this bin) using the observed MC events as
        # a constraint.
        cut = a_i > 0
        #a_i[a_i <= 0] = 10e-10
        #llh += np.sum(np.dot(ai, np.log(a_i)) - np.sum(a_i))
        llh += np.sum(np.dot(ai[cut], np.log(a_i[cut])) - np.sum(a_i[cut]))

        # This is simply a normalization term that helps by centering the LLH
        # near 0
        # It's an expansion of Ln(ai!) using the Sterling expansion
        cut = ai > 0
        llh -= np.sum(np.dot(ai[cut], np.log(ai[cut])) - np.sum(ai[cut]))

        return -llh

    def get_llh_poisson(self):
        """The standard binned-poisson likelihood comparing the weighted MC
        distribution to the data, ignoring MC statistical uncertainties."""
        di = self.data_histogram
        fi = np.sum(np.multiply(self.mc_histograms,
                                self.unweighted_histograms), axis=0)
        llh = 0

        # The normal definition of the LLH, dropping the Ln(di!) constant term
        cut = fi > 0
        llh += np.sum(di[cut] * np.log(fi[cut]) - fi[cut])

        # This is simply a normalization term that helps by centering the LLH
        # near 0
        # It's an expansion of Ln(di!) using the Sterling expansion
        cut = di > 0
        llh -= np.sum(di[cut] * np.log(di[cut]) - di[cut])

        return -llh
