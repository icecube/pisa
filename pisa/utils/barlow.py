#!/usr/bin/python

"""
likelihoods.py
Michael Larson (mlarson@nbi.ku.dk)
14 March 2016

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

To use this, you need to run  SetData, SetMC, and the SetUnweighted
 functions. THE SetMC FUNCTION TAKES A HISTOGRAM OF THE AVERAGE
 WEIGHT PER EVENT FOR EACH BIN! NOT THE TOTAL WEIGHTED HISTOGRAM!

 Simply calling the GetLLH function after these will return the
 best fit LLH for your chosen likelihood function. The function takes
 the name of the likelihood space ("Poisson" or "Barlow"). You can
 retrieve the best fit weighted plots using the GetPlot (total best-fit
 histogram including all samples) and the GetSinglePlots (the list
 of best-fit weighted histograms for each sample).
"""


from __future__ import division

from copy import copy
import numpy
import sys

from scipy.optimize import minimize


__all__ = ['likelihoods']


# TODO: pep8 this mofo
class likelihoods():
    mc_histograms = None
    unweighted_histograms = None
    data_histogram = None
    shape = None
    bestfit_plots = None
    current_bin = None

    def init():
        """Instantiate and set all of the defaults"""
        self.mc_histograms = None
        self.unweighted_histograms = None
        self.data_histogram = None
        self.shape = None
        self.bestfit_plots = None
        self.current_bin = None
        return

    def Reset(self):
        """Re-instantiate so that we can reuse the object"""
        self.init()

    def SetData(self,data_histogram):
        """Set up the data histogram. This histogram is flattened in order to
        make the looping later on a bit simpler to handle."""
        if not self.shape: self.shape = data_histogram.shape

        if not data_histogram.shape == self.shape:
            print "Data histogram has shape ", data_histogram.shape, \
                "but expected histogram shape", self.shape
            sys.exit(100)

        self.data_histogram = data_histogram.flatten()

    def SetMC(self,mc_histograms):
        """Set up the MC histogram. Each histogram is flattened in order to
        make the looping later on a bit simpler to handle. The values in each
        bin should correspond to the weight-per-event in that bin, NOT the
        total weight in that bin!

        Make sure you don't have any nulls here.

        """
        if not self.shape:
            self.shape = mc_histograms.values()[0].shape

        if numpy.any(numpy.isnan(mc_histograms)):
            print "At least one bin in your MC histogram is NaN!"\
                " Take a look and decide how best to handle this"
            sys.exit(100)

        flat_histograms = []
        for j in range(mc_histograms.shape[0]):
            if not self.shape == mc_histograms[j].shape:
                print "MC Histogram", j, \
                    "has shape ", mc_histograms[j].shape, \
                    "but expected shape", self.shape
                sys.exit(101)
            flat_histograms.append(mc_histograms[j].flatten())

        self.mc_histograms = numpy.array(flat_histograms)
        return

    def SetUnweighted(self,unweighted_histograms):
        """Save the unweighted distributions in the MC. These can include 0s."""
        if not self.shape:
            self.shape = unweighted_histograms.values()[0].shape

        flat_histograms = []
        for j in range(unweighted_histograms.shape[0]):
            if not self.shape == unweighted_histograms[j].shape:
                print "Unweighted Histogram", j, \
                    "has shape ", unweighted_histograms[j].shape, \
                    "but expected shape", self.shape
                sys.exit(100)
            flat_histograms.append(unweighted_histograms[j].flatten())

            self.unweighted_histograms = numpy.array(flat_histograms)

    def GetPlot(self):
        """Get the total weighted best-fit histogram post-fit."""
        if self.bestfit_plots == None: return None

        result = numpy.sum( self.GetSinglePlots(), axis=0 )
        return result

    def GetSinglePlots(self):
        """Get the individual weighted best-fit histograms post-fit."""
        if self.bestfit_plots == None: return None
        result = numpy.multiply(self.mc_histograms,
                                self.bestfit_plots)

        return numpy.reshape(result, (result.shape[0], self.shape[0], self.shape[1]))

    def GetLLH(self, llh_type):
        """Calculate the likelihood given the data, average weights, and the
        unweighted histograms. You can choose between "Poisson" and "Barlow"
        likelihoods at the moment.

        If using the "Barlow" LLH, note that the method is picked to be Powell
        with 25 iterations maximum per step. This is not optimized at all and
        was explicitly chosen simply because it "worked". This doesn't work
        with the bounds set in the normal way, so the positive-definiteness of
        the rates is enforced in the getLLH_barlow_bin method.

        """
        self.bestfit_plots = copy(self.unweighted_histograms)
        self.current_bin = 0

        # The simplest case: the Poisson binned likelihood
        if llh_type.lower() == "poisson":
            poisson_LLH = self.getLLH_poisson()
            return poisson_LLH

        # The more complicated case: The Barlow LLH
        # This requires a separate minimization step in each bin to estimate
        #  the expected rate in each bin from each MC sample using constraints
        #  from the data and the observed MC distribution.
        elif llh_type.lower() == "barlow":
            LLH = 0
            for bin in range(len(self.data_histogram)):
                self.current_bin = bin
                bin_result = minimize(fun=self.getLLH_barlow_bin,
                                      x0=self.bestfit_plots[:,bin],
                                      method="Powell",
                                      options={'maxiter':25,
                                               'disp':False},
                                      )
                self.bestfit_plots[:,bin] = bin_result.x
                LLH += bin_result.fun

            self.current_bin = None
            return LLH

        # I don't know what you want to use.
        else:
            print "Unknown LLH space: %s. Choose either \"Poisson\" (ideal) or"\
                " \"Barlow\" (including MC statistical errors)."
            sys.exit(300)

    def getLLH_barlow_bin(self, Ai):
        """The Barlow LLH finds the best-fit "expected" MC distribution using
        both the data and observed MC as constraints. Each bin is independent
        in this calculation, since the assumption is that there are no
        correlations between bins. This likely leads to a somewhat worse limit
        than you might get otherwise, but at least its conservative.

        If you have a good idea for how to extend this to include bin-to-bin,
        let me know and I can help implement it.

        """
        if any([Ai[j] < 0 for j in range(len(Ai))]): return 1e10
        i = self.current_bin
        di = self.data_histogram[i]
        fi = numpy.sum( numpy.multiply(self.mc_histograms[:,i], Ai) )
        ai = self.unweighted_histograms[:,i]

        LLH = 0

        # This is the standard Poisson LLH comparing data to the total weighted
        # MC
        if fi > 0: LLH += di * numpy.log(fi) - fi
        if di > 0: LLH -= di * numpy.log(di) - di

        # The heart of the Barlow LLH. Fitting the Ai (expected number of events
        #  in the MC sample for this bin) using the observed MC events as a
        #  constraint.
        cut = Ai > 0
        #Ai[Ai <= 0] = 10e-10
        #LLH += numpy.sum(numpy.dot(ai, numpy.log(Ai)) - numpy.sum(Ai))
        LLH += numpy.sum(numpy.dot(ai[cut], numpy.log(Ai[cut])) - numpy.sum(Ai[cut]))

        # This is simply a normalization term that helps by centering the LLH
        # near 0
        # It's an expansion of Ln(ai!) using the Sterling expansion
        cut = ai > 0
        LLH -= numpy.sum(numpy.dot(ai[cut], numpy.log(ai[cut])) - numpy.sum(ai[cut]))

        return -LLH

    def getLLH_poisson(self):
        """The standard binned-poisson likelihood comparing the weighted MC
        distribution to the data, ignoring MC statistical uncertainties."""
        i = self.current_bin
        di = self.data_histogram
        fi = numpy.sum(numpy.multiply(self.mc_histograms,
                                      self.unweighted_histograms), axis=0)
        LLH = 0

        # The normal definition of the LLH, dropping the Ln(di!) constant term
        cut = fi > 0
        LLH += numpy.sum(di[cut] * numpy.log(fi[cut]) - fi[cut])

        # This is simply a normalization term that helps by centering the LLH
        # near 0
        # It's an expansion of Ln(di!) using the Sterling expansion
        cut = di > 0
        LLH -= numpy.sum(di[cut] * numpy.log(di[cut]) - di[cut])

        return -LLH
