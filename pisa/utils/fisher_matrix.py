"""
A Fisher matrix class definition.
"""
#TODO: fix, adapt, clean up

from __future__ import absolute_import, division

import copy
import itertools
import json
import operator
import sys

import numpy as np
from scipy.stats import chi2

from pisa import FTYPE
from pisa.utils.fileio import from_file, to_file
from pisa.utils.log import logging


__all__ = ['FisherMatrix']

__author__ = 'L. Schulte, S. Boeser, T. Ehrhardt'

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


def build_fisher_matrix(gradient_hist_flat_d, fiducial_hist, fiducial_params):
    # fix the ordering of parameters
    params = sorted(gradient_hist_flat_d.keys())

    # find non-empty bins in flattened map
    fiducial_hist_flat = fiducial_hist.nominal_values['total'].flatten()
    nonempty = np.nonzero(fiducial_hist_flat)
    logging.debug("Using %u non-empty bins of %u" %
                  (len(nonempty[0]), len(fiducial_hist_flat)))

    # get gradients as calculated above for non-zero bins
    gradients = np.array(
        [gradient_hist_flat_d[par][nonempty] for par in params], dtype=FTYPE
    )

    # get error estimate from best-fit bin count for non-zero bins
    # TODO: these are not variances
    variances = fiducial_hist['total'].std_devs.flatten()[nonempty]

    # Loop over all parameters per bin (simple transpose) and calculate Fisher
    # matrix by getting the outer product of all gradients in a bin.
    # Result is sum of matrix for all bins.
    fmatrix = np.zeros((len(params), len(params)), dtype=FTYPE)
    for bin_gradients, bin_var in zip(gradients.T, variances):
        fmatrix += np.outer(bin_gradients, bin_gradients)/bin_var

    # construct the fisher matrix object
    fisher = FisherMatrix(
        matrix=fmatrix,
        parameters=params,  #order is important here!
        best_fits=fiducial_params.nominal_values, # TODO: fix order (in the sense of making it definite?)
        priors=None, #FIXME: support priors
    )

    return fisher, nonempty


def get_fisher_matrix(hypo_maker, test_vals, counter):
    """Compute Fisher matrices at fiducial hypothesis given data.
    """
    from pisa.utils.pull_method import get_gradients
    hypo_params = hypo_maker.params.free

    #fisher = {'total': {}}
    fid_hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)
    counter += 1

    pmaps = {'total': {}}
    gradient_maps = {'total': {}}

    for pname in hypo_params.names:
        logging.trace("Computing binwise gradients for parameter '%s'." % pname)
        tpm, gm = get_gradients(
            param=pname,
            hypo_maker=hypo_maker,
            test_vals=test_vals[pname],
        )
        counter += len(test_vals[pname])
        # the maps corresponding to variations of
        # a single param are not flattened
        pmaps['total'][pname] = tpm
        # these are flattened, which is also what the
        # method below assumes
        gradient_maps['total'][pname] = gm

    # hypo param values are not at nominal anymore,
    # but we don't use their values here

    fisher, nonempty = build_fisher_matrix(
        gradient_hist_flat_d=gradient_maps['total'],
        fiducial_hist=fid_hypo_asimov_dist,
        fiducial_params=hypo_params
    )
    return fisher, gradient_maps, fid_hypo_asimov_dist, nonempty


### FISHER MATRIX CLASS DEFINITION ###

class FisherMatrix:

    def __init__(self, matrix, parameters, best_fits, priors=None,
                 labels=None):
        """
        Construct a fisher matrix object
        Arguments
        ---------
        matrix : len(parameters) x len(parameters) matrix or 2D array
            Matrix values
        parameters : sequence of str
            Identifiers used for each parameter
        best_fits : sequence of numbers
            Best-fit values for each parameter
        priors : sequence of Prior objects or None
            Priors for each parameter; only accepts Gaussian or uniform priors,
            otherwise raises TypeError. Note that uniform priors are
            functionally equivalent to no prior. If None, uniform priors are
            used for all parameters (i.e., sigma=np.inf).
        labels : sequence of str or None
            Pretty-print labels for the parameters. If None, use `paramaters`
            strings as labels.
        """

        self.matrix = np.matrix(matrix)
        self.parameters = list(parameters)
        self.best_fits = list(best_fits)
        if priors is None:
            self.priors = [np.inf for p in self.parameters]
        else:
            self.priors = [self.translatePrior(prior) for prior in priors]
        self.labels = list(labels) if labels is not None else parameters
        
        self.checkConsistency()
        self.calculateCovariance()


    @classmethod
    def fromFile(cls, filename):
        """
        Load a Fisher matrix from a json file
        """
        
        return cls(**from_file(filename))
    
    
    @classmethod
    def fromPaPAFile(cls, filename):
        """
        Load Fisher matrix from json file
        """
        
        loaded_dict = json.load(open(filename, 'r'))
        
        matrix = np.matrix(loaded_dict.pop('matrix'))
        parameters = loaded_dict.pop('parameters')
        best_fits = loaded_dict.pop('best_fits')
        labels = loaded_dict.pop('labels')
        
        new_fm = cls(matrix=matrix, parameters=parameters, 
                     best_fits=best_fits, labels=labels)
        
        while not len(loaded_dict)==0:
            
            par, prior_dict = loaded_dict.popitem()
            
            #self.parameters.append(par)
            for value in prior_dict.itervalues():
                new_fm.addPrior(par, value)
        
        new_fm.checkConsistency()
        new_fm.calculateCovariance()
        
        return new_fm
    
    
    def __add__(self, other):
        
        # merge parameter lists
        new_params = list(set(self.parameters+other.parameters))
        
        new_best_fits = []
        new_labels = []
        for param in new_params:
            try:
                value = self.getBestFit(param)
                lbl = self.getLabel(param)
            except IndexError:
                value = other.getBestFit(param)
                lbl = other.getLabel(param)
            
            new_best_fits.append(value)
            new_labels.append(lbl)
        
        # generate blank matrix
        new_matrix = np.matrix( np.zeros((len(new_params), len(new_params))) )
        
        # fill new matrix
        for (i,j) in itertools.product(range(len(new_params)), range(len(new_params))):
            
            for summand in [self, other]:
                try:
                    i_sum = summand.getParameterIndex(new_params[i])
                    j_sum = summand.getParameterIndex(new_params[j])
                except IndexError:
                    continue
                
                new_matrix[i,j] += summand.matrix[i_sum, j_sum]
        
        # create new FisherMatrix object
        new_object = FisherMatrix(matrix=new_matrix, parameters=new_params, 
                                  best_fits=new_best_fits, labels=new_labels)
        new_object.calculateCovariance()
        
        # fill in priors
        for par in new_object.parameters:
            
            for summand in [self, other]:
                
                try:
                    prior_dict = summand.getPriorDict()
                except IndexError:
                    continue
                
                for par, sigma in prior_dict.items():
                    new_object.addPrior(par, sigma)
        
        # ...and we're done!
        return new_object
    
    
    def checkConsistency(self):
        """
        Check whether number of parameters matches dimension of matrix; 
        matrix is symmetrical; parameter names are unique; and number of
        best_fits, labels, and priors all match number of parameters.
        """
        
        if not len(self.parameters) == np.shape(self.matrix)[1]:
            raise IndexError('Number of parameters does not match dimension of Fisher matrix! [%i, %i]' \
                %(len(self.parameters), len(self.matrix)) )
        
        if not np.all(self.matrix.T == self.matrix):
            raise ValueError('Fisher matrix not symmetric!')
        
        if not len(self.parameters) == len(set(self.parameters)):
            raise ValueError('Parameter names not unique! %s' \
                %(np.array2string(np.array(self.parameters))) )
        
        if not len(self.parameters) == len(self.best_fits) == len(self.labels) == len(self.priors):
            raise ValueError('Parameters, best_fits, labels, and priors must all have same length! (lengths = %d, %d, %d, %d)' \
                %(len(self.parameters), len(self.best_fits), len(self.labels), len(self.priors)) )
        
        return True
    
    
    def saveFile(self, filename):
        """
        Write Fisher matrix to json file
        """
        
        dict_to_write = {}
        dict_to_write['matrix'] = self.matrix
        dict_to_write['parameters'] = self.parameters
        dict_to_write['best_fits'] = self.best_fits
        dict_to_write['labels'] = self.labels
        dict_to_write['priors'] = self.priors
        
        to_file(dict_to_write, filename)
    
    
    def getParameterIndex(self, par):
        """
        Whether par is already existing in parameter list
        """
        
        if not par in self.parameters:
            raise IndexError('%s not found in parameter list %s'\
                %(par, np.array2string(np.array(self.parameters)) ) )
        return self.parameters.index(par)
    
    
    ## -> Why on earth would we ever want to do that?
    def renameParameter(self, fromname, toname):
        """
        Rename a parameter
        """
        
        idx = self.getParameterIndex(fromname)
        
        if toname in self.parameters[self.parameters!=fromname]:
            raise ValueError('%s already in parameter list %s'\
                %(toname, np.array2string(np.array(self.parameters)) ) )
        
        self.parameters[idx] = toname
    
    
    def calculateCovariance(self):
        """
        Calculate covariance matrix from Fisher matrix (i.e. invert including priors).
        """
        
        if np.linalg.det(self.matrix) == 0:
            raise ValueError('Fisher Matrix is singular, cannot be inverted!')
        
        self.covariance = np.linalg.inv(
            self.matrix + np.diag([1./self.getPrior(p)**2 for p in self.parameters])
        )
    
    
    def getBestFit(self, par):
        """
        Get best fit value for given parameter
        """
        
        idx = self.getParameterIndex(par)
        
        return self.best_fits[idx]
    
    
    def getLabel(self, par):
        """
        Get pretty-print label for given parameter
        """
        
        idx = self.getParameterIndex(par)
        
        return self.labels[idx]
    
    
    def setLabel(self, par, newlabel):
        """
        Change the pretty-print label for given parameter
        """
        
        idx = self.getParameterIndex(par)
        
        self.labels[idx] = newlabel
    
    
    def removeParameter(self, par):
        """
        Remove par from Fisher matrix and recalculate covariance
        """
        
        idx = self.getParameterIndex(par)
        
        # drop from parameter, best fit, and prior list
        self.parameters.pop(idx)
        self.best_fits.pop(idx)
        self.labels.pop(idx)
        self.priors.pop(idx)
        
        # drop from matrix (first row, then column)
        self.matrix = np.delete(np.delete(self.matrix, idx, axis=0), idx, axis=1)
        
        self.checkConsistency()
        self.calculateCovariance()
    
    
    @staticmethod
    def translatePrior(prior):
        """
        Translates a Prior object, numeric, or None to the simplistic prior
        format used internally (a value for sigma).
        
        Arguments
        ---------
        prior : Prior object (gaussian or uniform), float, or None
        Returns
        -------
        sigma : Standard deviation of prior (np.inf for uniform Prior or None)
        
        """
        if np.isscalar(prior):
            return float(prior)
        
        if prior is None:
            return np.inf
        
        # TODO: debug following check, which fails even when types are "same";
        # multiple import of Prior?
        # if not isinstance(prior, Prior):
        #    raise TypeError('prior must be Prior object, numeric, or None; got `%s` instead' % type(prior))
        
        if prior.kind == 'uniform':
            return np.inf
        elif prior.kind == 'gaussian':
            return prior.sigma
        else:
            raise TypeError('Prior object must be of either gaussian or uniform kind; got kind `'+str(prior.kind)+'` instead')
    
    
    def setPrior(self, par, sigma):
        """
        Set prior for parameter 'par' to value sigma. If sigma is None, no
        prior is assumed
        """
        
        idx = self.getParameterIndex(par)
        self.priors[idx] = sigma
        self.calculateCovariance()
    
    
    def addPrior(self, par, sigma):
        """
        Add a prior of value sigma to the existing one for par (in quadrature)
        """
        
        idx = self.getParameterIndex(par)
        self.priors[idx] = 1./np.sqrt(1./self.priors[idx]**2 + 1./sigma**2)
        self.calculateCovariance()
    
    
    def removeAllPriors(self):
        """
        Remove *all* priors from this Fisher Matrix
        """
        
        self.priors = [np.inf for p in self.parameters]
        self.calculateCovariance()
    
    
    def getPrior(self, par):
        """
        List the prior (sigma value) for par
        """
        
        idx = self.getParameterIndex(par)
        return self.priors[idx]
    
    
    def getPriorDict(self):
        """
        List priors of all parameters (sigma values)
        """
        
        return dict(zip(self.parameters, self.priors))
    
    
    def getCovariance(self, par1, par2):
        """
        Returns the covariance of par1 and par2
        """
        
        # Return the respective element
        idx1, idx2 = self.getParameterIndex(par1), self.getParameterIndex(par2)
        return self.covariance[idx1, idx2]
    
    
    def getVariance(self, par):
        """
        Returns full variance of par
        """
        
        return self.getCovariance(par,par)
    
    
    def getSigma(self, par):
        """
        Returns full standard deviation of par,
        marginalized over all other parameters
        """
        
        return np.sqrt(self.getVariance(par))
    
    
    def getSigmaNoPriors(self, par):
        """
        Returns standard deviation of par, marginalized over all other 
        parameters, but ignoring priors on this parameter
        """
        
        idx = self.getParameterIndex(par)
        
        # make temporary priors with the ones corresponding to par removed
        temp_priors = copy.deepcopy(self.priors)
        temp_priors[idx] = np.inf
        
        # calculate covariance with these priors
        temp_covariance = np.linalg.inv(
            self.matrix + np.diag([1./s**2 for s in temp_priors])
        )
        
        return np.sqrt(temp_covariance[idx,idx])
    
    
    def getSigmaStatistical(self, par):
        """
        Returns standard deviation of par,
        if all other parameters are fixed (i.e. known infinitely well)
        """
        
        idx = self.getParameterIndex(par)
        return 1./np.sqrt(self.matrix[idx,idx])
    
    
    def getSigmaSystematic(self, par):
        """
        Returns standard deviation of par for infinite statistics
        (i.e. systematic error)
        """
        
        return np.sqrt(self.getSigmaNoPriors(par)**2 -
                       self.getSigmaStatistical(par)**2)
    
    
    def getErrorEllipse(self, par1, par2, confLevel=0.6827):
        """
        Returns a, b, tan(2 theta) of confLevel error ellipse 
        in par1-par2-plane with:
        
        a: large half axis
        b: small half axis
        tan(2 theta): tilt angle, has to be divided by the aspect
                      ratio of the actual plot before taking arctan
        
        Formulae taken from arXiv:0906.4123
        """
        
        sigma1, sigma2 = self.getSigma(par1), self.getSigma(par2)
        cov = self.getCovariance(par1, par2)
        
        #for this we need sigma1 > sigma2, otherwise just swap parameters
        if sigma1 > sigma2:
            a_sq = (sigma1**2 + sigma2**2)/2. + np.sqrt((sigma1**2 - sigma2**2)**2/4. + cov**2)
            b_sq = (sigma1**2 + sigma2**2)/2. - np.sqrt((sigma1**2 - sigma2**2)**2/4. + cov**2)
        else:
            a_sq = (sigma2**2 + sigma1**2)/2. - np.sqrt((sigma2**2 - sigma1**2)**2/4. + cov**2)
            b_sq = (sigma2**2 + sigma1**2)/2. + np.sqrt((sigma2**2 - sigma1**2)**2/4. + cov**2)

        #Note: this has weird dimensions (actual size of the plot)!
        tan_2_th = 2.*cov / (sigma1**2 - sigma2**2)
        
        # we are dealing with a 2D error ellipse here
        scaling = np.sqrt(chi2.ppf(confLevel, 2))
        
        return scaling*np.sqrt(a_sq), scaling*np.sqrt(b_sq), tan_2_th
    
    
    def getCorrelation(self, par1, par2):
        """
        Returns correlation coefficient between par1 and par2
        """
        
        return self.getCovariance(par1, par2)/(self.getSigma(par1)*self.getSigma(par2))
    
    
    def printResults(self, parameters=None, file=None):
        """
        Prints statistical, systematic errors, priors, best fits 
        for specified (default: all) parameters
        """
        
        pars = parameters if parameters is not None else copy.deepcopy(self.parameters)
        pars.sort()
        
        if file is not None:    # redirect stdout
            orig_stdout = sys.stdout
            sys.stdout = open(file, 'w')
        
        param_width = max([max([len(name) for name in pars]), len('parameters')])
        header = (param_width, 'parameter', 'best fit', 'full', 'stat', 'syst', 'priors')
        print('%*s     %9s     %9s     %9s     %9s     %9s' %header)
        print('-'*(70+param_width))
        
        for par in pars:
            result = (param_width, par, self.getBestFit(par), self.getSigma(par),
                      self.getSigmaStatistical(par), self.getSigmaSystematic(par),
                      self.getPrior(par))
            par_str = '%*s    %10.3e     %.3e     %.3e     %.3e     %.3e'%result
            par_str = par_str.replace('inf', 'free')
            print(par_str)
        
        """
        # needed for PINGU only:
        if 'hierarchy' in pars: 
            
            # calculate proper significance according to arXiv:1305.5150
            sigma_gauss = 1./self.getSigma('hierarchy')
            sigma_bin = conv_sigma_to_bin_sigma(sigma_gauss)
            
            print '\nSignificance of hierarchy measurement: %.2f sigma' %sigma_bin
        """

        if file is not None:    # switch stdout back
            sys.stdout = orig_stdout
    
    
    def printResultsSorted(self, par, file=None, latex=False):
        """
        Prints statistical, systematic errors, priors, best fits 
        sorted by parameter par
        """
        
        if file is not None:    # redirect stdout
            orig_stdout = sys.stdout
            sys.stdout = open(file, 'w')
        
        if latex:
            # table header
            print('\\begin{tabular}{lrrrrrr} \n\\toprule')
            print('Parameter & Impact & Best Fit & Full & Stat. & Syst. & Prior \\\\ \n\\midrule')
        else:
            param_width = max([max([len(name) for name in self.parameters]), len('parameters')])
            header = (param_width, 'parameter', 'impact [%]','best fit', 'full', 'stat', 'syst', 'priors')
            
            print('%*s     %10s     %9s     %9s     %9s     %9s     %9s' %header)
            print('-'*(85+param_width))
        
        sortedp = self.sortByParam(par)
        
        for (par, impact) in sortedp:
        
            # print the line
            if latex:
                result = (self.getLabel(par), impact, self.getBestFit(par), self.getSigma(par),
                          self.getSigmaStatistical(par), self.getSigmaSystematic(par),
                          self.getPrior(par))
                par_str = '%s & %.1f & \\num{%.2e} & \\num{%.2e} & \\num{%.2e} & \\num{%.2e} & \\num{%.2e} \\\\'%result
                par_str = par_str.replace('\\num{inf}', 'free')
            else:
                result = (param_width, par, impact, self.getBestFit(par), self.getSigma(par),
                          self.getSigmaStatistical(par), self.getSigmaSystematic(par),
                          self.getPrior(par))
                par_str = '%*s          %5.1f    %10.3e     %.3e     %.3e     %.3e     %.3e'%result
                par_str = par_str.replace('inf', 'free')
            
            print(par_str)
        
        if latex:
            # table outro
            print('\\bottomrule \n\\end{tabular}')
        
        if file is not None:    # switch stdout back
            sys.stdout = orig_stdout
    
    
    def sortByParam(self, par):
        """
        Sorts the parameters by their impact on parameter par.
        Relevant quantity is covariance(par,i)/sigma_i.
        
        Returns sorted list of (parameters, impact), par first, 
        then ordered descendingly by impact.
        """
        
        # calculate impact
        impact = dict([[p, self.getCorrelation(p, par)**2 * 100] \
                        for p in self.parameters])
        
        # sort the dict by value
        # FIXME
        sorted_impact = sorted(impact.iteritems(), 
                               key=operator.itemgetter(1),
                               reverse=True)
        
        return sorted_impact
