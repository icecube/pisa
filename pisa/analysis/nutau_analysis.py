#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:    March 20, 2016

from collections import Sequence
import sys
import scipy.optimize as opt
from scipy.stats import chi2
import time
from uncertainties import unumpy as unp
import random
import os

from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.fileio import from_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.random_numbers import get_random_state
from pisa import ureg, Q_

class Analysis(object):
    """Major tools for performing "canonical" IceCube/DeepCore/PINGU analyses.

    * "Data" distribution creation (via passed `data_maker` object)
    * Template distribution creation (via passed `distribution_maker` object)

    * Minimizer Interface (via method `_minimizer_callable`)
        Interfaces to a minimizer for modifying the free parameters of the
        `distribution_maker` to fit its output (as closely as possible) to the
        data distribution is provided. See [minimizer_settings] for

    Parameters
    ----------
    data_maker : DistributionMaker
        Generates a (pseudo)data distribution. Parameters used for generating
        this distribution are not modified from their injected values during
        the analysis, although added fluctuations may be regenerated if
        `data_fluctuations='fluctuated'` is specified (see below for more
        explanation).

    template_maker : DistributionMaker
        Provides output templates to compare against the data-like template.
        The `template_maker` provides the interface to those parameters
        that can be modifed or studied for their effects during the analysis
        process.

    metric : str
        What metric to be used for likelihood calculations / optimization
        llh or chi2

    Attributes
    ----------
    data_maker
    template_maker

    Methods
    -------
    optimize
    scan
    profile : profile a given param (profile LLH)
    _minimizer_callable : private method indended to be called by a minimizer

    """
    def __init__(self, data_maker, template_maker, metric, blind=False):
        assert isinstance(data_maker, DistributionMaker)
        assert isinstance(template_maker, DistributionMaker)
        self.data_maker = data_maker
        self.template_maker = template_maker
        assert isinstance(metric, basestring)
        self.metric = metric.lower()
        self.minimizer_settings = None
        self.blind = blind

        # DOF as n_bins - n_free_params + n_gauss_priors
        template = self.template_maker.get_outputs(return_sum=True)
        template = template.combine_wildcard('*')

        n_bins = sum(map.binning.tot_num_bins for map in [template])
        self.n_free_params = len(self.template_maker.params.free)
        n_gauss_priors = 0
        for param in self.template_maker.params.free:
            if param.prior.kind == 'gaussian': n_gauss_priors +=1
        self.dof = n_bins - self.n_free_params + n_gauss_priors

        # Generate distribution
        data = self.data_maker.get_outputs(return_sum=True)#, sum_map_name='evts', sum_map_tex_name='evts')
        self.data = data.combine_wildcard('*')
        self.pseudodata_method = None
        self.pseudodata = None
        self.n_minimizer_calls = 0

    def generate_psudodata(self):
        #just to make it work right now
        data_random_state = get_random_state(random.randint(0,2**32))
        if self.pseudodata_method in ['asimov', 'data']:
            self.pseudodata = self.data
        elif self.pseudodata_method == 'poisson':
            self.pseudodata = self.data.fluctuate('poisson', random_state=data_random_state)
        elif self.pseudodata_method == 'gauss+poisson':
            self.pseudodata = self.data.fluctuate('gauss+poisson', random_state=data_random_state)
        elif self.pseudodata_method == 'gauss':
            self.pseudodata = self.data.fluctuate('gauss', random_state=data_random_state)
        else:
            raise Exception('unknown method %s'%method)
        self.N_data = sum([map.nominal_values.sum() for map in [self.pseudodata]])

    # TODO: move the complexity of defining a scan into a class with various
    # factory methods, and just pass that class to the scan method; we will
    # surely want to use scanning over parameters in more general ways, too:
    #   set (some) fixed params, then run (minimizer, scan, etc.) on free params
    #   set (some free or fixed) params, then check metric
    # where the setting of the params is done for some number of values.
    def scan(self, param_names=None, steps=None, values=None,
             outer=False):
        """Set template maker parameters named by `param_names` according to
        either values specified by `values` or number of steps specified by
        `steps`, and return the `metric` indicating how well each template
        matches the data distribution.

        Some flexibility in how the user can specify `values` is allowed, based
        upon the shapes of `param_names` and `values` and how the `outer` flag
        is set.

        Either `values` or `steps` must be specified, but not both.

        Parameters
        ----------
        param_names : None, string, or sequence of strings
            If None, assume all parameters are to be scanned; otherwise,
            specifies only the name or names of parameters to be scanned.

        steps : None, integer, or sequence of strings
            Number of steps to take within the allowed range of the parameter
            (or parameters). Value(s) specified for `steps` must be >= 2. Note
            that the endpoints of the range are always included, and numbers of
            steps higher than 2 fill in between the endpoints.
            * If integer...
                Take this many steps for each specified parameter.
            * If sequence of integers...
                Take the coresponding number of steps within the allowed range
                for each specified parameter.

        values : None, scalar, sequence of scalars, or sequence-of-sequences
          * If scalar...
                Set this value for the (one) param name in `param_names`.
          * If sequence of scalars...
              * if len(param_names) is 1, set its value to each number in the
                sequence.
              * otherwise, set each param in param_names to the corresponding
                value in `values`. There must be the same number of param names
                as values.
          * If sequence of (sequences or iterables)...
              * Each param name corresponds to one of the inner sequences, in
                the order that the param names are specified.
              * If `outer` is False, all inner sequences must have the same
                length, and there will be one template generated for each set
                of values across the inner sequences. In other words, there will
                be a total of len(inner sequenc) templates generated.
              * If `outer` is True, the lengths of inner sequences needn't be
                the same. This takes the outer product of the passed sequences
                to arrive at the permutations of the parameter values that will
                be used to produce templates (essentially nested loops over
                each parameter). E.g., if two params are scanned, for each
                value of the first param's inner sequence, a template is
                produced for every value of the second param's inner sequence.
                In total, there will be len(inner seq0) * len(inner seq1) * ...
                templates produced.

        outer : bool
            If set to True and a sequence of sequences is passed for `values`,
            the points scanned are the *outer product* of the inner sequences.
            See `values` for a more detailed explanation.

        """
        assert not (steps is not None and values is not None)
        if isinstance(param_names, basestring):
            param_names = [param_names]

        if values is not None and np.isscalar(values):
            values = np.array([values])
            nparams = len(param_names)

        metric_vals = []
        for val in values:
            fp = self.template_maker.params.free
            fp[param_names].value = val
            self.template_maker.update_params(fp)
            template = self.template_maker.get_outputs()
            template = [t.combine_wildcard('*') for t in template]
            template[0][0].name = 'total'
            metric_vals.append(self.pseudodata.metric_total(expected_values=template,
                                                      metric=self.metric))
        return metric_vals

    def randomize_free_params(self):
        n = len(self.template_maker.params.free)
        rand = np.random.rand(n)
        self.template_maker.params.free._rescaled_values = rand

    def _minimizer_callable(self, scaled_param_vals, pprint=True):
        """The callable interface to provide to simple minimzers e.g. those in
        scipy.optimize.

        This should *not* in general be called by users, as `scaled_param_vals`
        are stripped of their units and rescaled to [0, 1], and hence some
        validation of inputs is bypassed by this method.

        Parameters
        ----------
        scaled_param_vals : sequence of floats
            If called from a scipy.optimize minimizer, this sequence is provied
            by the minimizer itself. These values are all expected to be in the
            range [0, 1] and be simple floats (no units or uncertainties
            attached, etc.). Rescaling the parameter values to their
            original (physical) ranges (including units) is handled within this
            method.

        pprint
            Displays a single-line that updates live (assuming the entire line
            fits the width of your TTY).

        """
        sign = +1
        if self.metric in ['llh', 'conv_llh', 'barlow_llh']:
            # Want to *maximize* log-likelihood but we're using a minimizer
            sign = -1
        self.template_maker.params.free._rescaled_values = scaled_param_vals
        template = self.template_maker.get_outputs()
        template = [t.combine_wildcard('*') for t in template]
        template[0].name = 'total'
        #N_mc = sum([unp.nominal_values(map.hist).sum() for map in template])
        #scale = self.N_data/N_mc
        #scale=1.

        # Assess the fit of the template to the data distribution, and negate
        # if necessary
        #print [map.name for map in template[0]]
        #print [map.name for map in self.pseudodata]
        metric_val = (
            self.pseudodata.metric_total(expected_values=template, metric=self.metric)
            + template_maker.params.priors_penalty(metric=self.metric)
        )
        #print metric_val

        mod_chi2_val = (self.pseudodata.metric_total(expected_values=template, metric='mod_chi2')
            + template_maker.params.priors_penalty(metric='mod_chi2'))
        chi2_val = (self.pseudodata.metric_total(expected_values=template, metric='chi2')
            + template_maker.params.priors_penalty(metric='chi2'))
        # Report status of metric & params
        if self.blind:
            msg = '%s=%.6e | %s blinded parameters' %(self.metric, metric_val, self.n_free_params)
            msg += '    mod_chi2 = %.6e / %i DOF, p = %.4f' %(mod_chi2_val, self.dof, chi2.sf(mod_chi2_val, self.dof))
            msg += '    chi2 = %.6e / %i DOF, p = %.4f' %(chi2_val, self.dof, chi2.sf(chi2_val,self.dof))
        else:
            msg = '%s=%.6e , mod_chi2 = %.6e / %i DOF, p = %.4f, chi2 = %.6e / %i DOF, p = %.4f | %s' %(self.metric, metric_val,
                                    mod_chi2_val, self.dof, chi2.sf(mod_chi2_val, self.dof),
                                    chi2_val, self.dof, chi2.sf(chi2_val,self.dof),
                                    self.template_maker.params.free)
        if pprint:
            sys.stdout.write(msg)
            sys.stdout.flush()
            sys.stdout.write('\b' * len(msg))
        else:
            logging.debug(msg)

        self.n_minimizer_calls += 1
        return sign*metric_val

    def run_minimizer(self, pprint=True, skip=False):
        # Get initial values
        x0 = self.template_maker.params.free._rescaled_values

        # bfgs steps outside of given bounds by 1 epsilon to evaluate gradients
        try:
            epsilon = self.minimizer_settings['options']['value']['eps']
        except:
            epsilon = self.minimizer_settings['options']['value']['epsilon']
        bounds = [(0+epsilon, 1-epsilon)]*len(x0)
        logging.info('running the %s optimizer'%self.minimizer_settings['method']['value'])

        # Using scipy.opt.minimize allows a whole host of minimisers to be used
        # This set by the method value in your minimiser settings file
        self.n_minimizer_calls = 0
        if skip:
            best_fit_vals = x0
            metric_val = self._minimizer_callable(x0, False)
            dict_flags = {'warnflag':0, 'task':'skip', 'funcalls':0, 'nit':0, 'avg_tmp_time':0, 'n_minimizer_calls':0}
        else:
            start_t = time.time()
            minim_result = opt.minimize(fun=self._minimizer_callable,
                                        x0=x0,
                                        args=(pprint,),
                                        bounds=bounds,
                                        method = self.minimizer_settings['method']['value'],
                                        options = self.minimizer_settings['options']['value'])
            
            # get aditional metrics:
            end_t = time.time()
            if pprint:
                # clear the line
                print ''
            print '\naverage template generation time during minimizer run: %.4f ms'%((end_t - start_t) * 1000./self.n_minimizer_calls)
            avg_tmp_time = (end_t - start_t) * 1000./self.n_minimizer_calls
            best_fit_vals = minim_result.x
            metric_val = minim_result.fun
            template = self.template_maker.get_outputs()
            template = [t.combine_wildcard('*') for t in template]
            template[0].name = 'total'
            dict_flags = {}
            mod_chi2_val = (self.pseudodata.metric_total(expected_values=template, metric='mod_chi2')
                + template_maker.params.priors_penalty(metric='mod_chi2'))
            dict_flags['agreement_mod_chi2'] = mod_chi2_val
            dict_flags['warnflag'] = minim_result.status
            dict_flags['task'] = minim_result.message
            if minim_result.has_key('jac'):
                dict_flags['grad'] = minim_result.jac
            dict_flags['funcalls'] = minim_result.nfev
            dict_flags['nit'] = minim_result.nit
            dict_flags['avg_tmp_time'] = avg_tmp_time
            dict_flags['n_minimizer_calls'] = self.n_minimizer_calls
            if dict_flags['warnflag'] > 0:
                logging.warning(str(dict_flags))

        all_metrics = {}
        template = self.template_maker.get_outputs()
        template = [t.combine_wildcard('*') for t in template]
        template[0].name = 'total'
        #for metric in ['llh', 'conv_llh', 'barlow_llh','chi2', 'mod_chi2']:
        for metric in ['llh','chi2']:
            all_metrics[metric] = self.pseudodata.metric_total(expected_values=template, metric=metric) + template_maker.params.priors_penalty(metric=metric) 

        return best_fit_vals, metric_val, all_metrics, dict_flags

    def find_best_fit(self, check_octant=True, pprint=True, skip=False):
        """ find best fit points (max likelihood) for the free parameters and
            return likelihood + found parameter values.
        """
        # Reset free parameters to nominal values
        logging.info('resetting params')
        self.template_maker.params.reset_free()
        if not check_octant:
            logging.warning('Skipping octant check in fit!')

        best_fit_vals, metric_val, all_metrics, dict_flags = self.run_minimizer(pprint=pprint, skip=skip)
        best_fit = {}
        best_fit[self.metric] = metric_val
        best_fit['warnflag'] = dict_flags['warnflag']
        best_fit['avg_tmp_time'] = dict_flags['avg_tmp_time']
        best_fit['n_minimizer_calls'] = dict_flags['n_minimizer_calls']
        best_fit['funcalls'] = dict_flags['funcalls']
        best_fit['all_metrics'] = all_metrics
        if not self.blind:
            for pname in self.template_maker.params.free.names:
                best_fit[pname] = self.template_maker.params[pname].value

        # decide wether fit for second octant is necessary
        if 'theta23' in self.template_maker.params.free.names and not skip:
            if check_octant:
                logging.info('checking other octant of theta23')
                self.template_maker.params.reset_free()
                # changing to other octant
                theta23 = self.template_maker.params['theta23']
                inflection_point = 45 * ureg.degree
                theta23.value = 2*inflection_point.to(theta23.value.units) - theta23.value
                self.template_maker.update_params(theta23)
                best_fit_vals, metric_val, all_metrics, dict_flags = self.run_minimizer(pprint=pprint)

                # compare results a and b, take one with lower llh
                if metric_val < best_fit[self.metric]:
                    # accept these values
                    logging.info('Accepting other octant fit')
                    best_fit[self.metric] = metric_val
                    best_fit['warnflag'] = dict_flags['warnflag']
                    best_fit['all_metrics'] = all_metrics
                    if not self.blind:
                        for pname in self.template_maker.params.free.names:
                            best_fit[pname] = self.template_maker.params[pname].value
                    
                else:
                    logging.info('Accepting initial octant fit')

        return best_fit

    def profile(self, p_name, values, check_octant=True, pprint=True):
        """Run profile log likelihood method for param `p_name`.

        Parameters
        ----------
        p_name
        values to fix parameter to in conditional llh

        """
        # run numerator (conditional MLE)
        logging.info('fixing param %s'%p_name)
        self.template_maker.params.fix(p_name)
        condMLEs = {}
        for value in values:
            logging.info('scan point %s'%value)
            prm = template_maker.params[p_name]
            prm.value = value
            template_maker.update_params(prm)
            condMLE = self.find_best_fit(check_octant=check_octant, pprint=pprint)
            condMLE[p_name] = self.template_maker.params[p_name].value
            append_results(condMLEs,condMLE)
            # report MLEs and LLH
            # also add the fixed param
        ravel_results(condMLEs)
        # run denominator (global MLE)
        logging.info('unfixing param %s'%p_name)
        self.template_maker.params.unfix(p_name)
        if self.pseudodata_method == 'asimov': #and self.metric in ['llh', 'chi2', 'mod_chi2']:
            # in these cases we can skip minimization
            skip = True
        else:
            skip = False
        globMLE = self.find_best_fit(skip=skip, check_octant=check_octant, pprint=pprint)
        # report MLEs and LLH
        return [condMLEs, globMLE]

    def llr(self, template_makerA, template_makerB):
        """ Run loglikelihood ratio for two different template makers A and B
        """
        results = []
        for template_maker in [template_makerA, template_makerB]:
            self.template_maker = template_maker
            results.append(self.find_best_fit(check_octant=check_octant), pprint=pprint)
        return results


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    import numpy as np
    from pisa import ureg, Q_

    from pisa.utils.fileio import from_file, to_file
    from pisa.utils.config_parser import parse_quantity
    from pisa.utils.format import append_results, ravel_results

    parser = ArgumentParser()
    parser.add_argument('-d', '--data-settings', type=str,
                        metavar='configfile', default=None,
                        action='append',
                        help='settings for the generation of "data"')
    parser.add_argument('-t', '--template-settings',
                        metavar='configfile', required=True,
                        action='append',
                        help='''settings for the template generation''')
    parser.add_argument('-o', '--outfile', metavar='FILE',
                        type=str, action='store', default='out.json',
                        help='file to store the output')
    parser.add_argument('-v', action='count', default=None,
                        help='set verbosity level')
    parser.add_argument('-n', '--num-trials', type=int, default=1,
                        help='number of trials')
    parser.add_argument('-b', '--blind', action='store_true',
                        help='run blindly i.e. only reporting goodness of fit, no parameter values')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='more quiet')
    parser.add_argument('--no-check-octant', action='store_true',
                        help='Do not check the second octant of theta23. Careful with that axe Eugene!')
    parser.add_argument('-m', '--minimizer-settings', type=str,
                        metavar='JSONFILE', required=True,
                        help='''Settings related to the optimizer used in the
                        LLR analysis.''')
    parser.add_argument('-sp', '--set-param', type=str, default=None,
                        help='Set a param to a certain value for both hypo and data.', action='append')
    parser.add_argument('-spd', '--set-param-data', type=str, default='',
                        help='''Set a param to a certain value only for data ''')
    parser.add_argument('-fp', '--fix-param', type=str, default='',
                        help='''fix parameter''')
    parser.add_argument('-spf', '--fix-param-scan', type=str, default='',
                        help='''fix parameter for scan only in hypo''')
    parser.add_argument('-pd', '--pseudo-data', type=str, default='poisson',
                        choices=['poisson', 'gauss', 'gauss+poisson', 'asimov', 'data'], 
                        help='''Mode for pseudo data sampling''')
    parser.add_argument('--var', type=str, default='nutau_norm',
                        help='''param to be profiled''')
    parser.add_argument('--metric', type=str,
                        choices=['llh', 'chi2', 'conv_llh', 'mod_chi2', 'barlow_llh'], required=True,
                        help='''Settings related to the optimizer used in the
                        LLR analysis.''')
    parser.add_argument('--mode', type=str,
                        choices=['H0', 'scan', 'feldman_cousins'], default='H0',
                        help='''just run significance or whole scan''')
    parser.add_argument('--range', type=str, default='np.linspace(0,2,11)*ureg.dimensionless',
                        help=''' scanning range''')
    parser.add_argument('-f','--function', type=str,
                        choices=['profile','fit'], default='profile',
                        help='''what shpuld be executed''')
    args = parser.parse_args()

    set_verbosity(args.v)

    if os.path.isfile(args.outfile):
        print "Output file ", args.outfile, " already existed, delete or remove it."
    else:
        if args.blind:
            assert(args.function == 'fit')
            assert(args.pseudo_data == 'data')

        if args.data_settings is None:
            data_settings = args.template_settings
        else:
            data_settings = args.data_settings

        data_maker = DistributionMaker(data_settings)
        template_maker = DistributionMaker(args.template_settings)

        if not args.fix_param == '':
            template_maker.params.fix(args.fix_param)
        if args.set_param is not None:
            for one_set_param in args.set_param:
                p_name,value = one_set_param.split("=")
                print "set_parm ", p_name, " to  ", value
                value = parse_quantity(value)
                value = value.n * value.units
                prm = template_maker.params[p_name]
                print "old", p_name,".value for template= ", prm.value
                prm.value = value
                template_maker.update_params(prm)
                print "new ", p_name,".value for template= ", prm.value
                if p_name in data_maker.params.names:
                    prm = data_maker.params[p_name]
                    print "old", p_name,".value for data= ", prm.value
                    prm.value = value
                    print "new", p_name,".value for data= ", prm.value
                    data_maker.update_params(prm)
        if not args.fix_param_scan == '':
            p_name,value = args.fix_param_scan.split("=")
            print "p_name,value= ", p_name, " ", value
            value = parse_quantity(value)
            value = value.n * value.units
            prm = template_maker.params[p_name]
            prm.value = value
            template_maker.update_params(prm)
            template_maker.params.fix(p_name)

        data_fixed_param=None
        if not args.set_param_data == '':
            p_name,value = args.set_param_data.split("=")
            print "set param ", p_name, "to  ", value, "for data"
            value = parse_quantity(value)
            data_fixed_param={p_name:value.n}
            value = value.n * value.units
            prm = data_maker.params[p_name]
            prm.value = value
            data_maker.update_params(prm)
            data_maker.params.fix(p_name)

        analysis = Analysis(data_maker=data_maker,
                            template_maker=template_maker,
                            metric=args.metric,
                            blind=args.blind)

        analysis.minimizer_settings = from_file(args.minimizer_settings)
        analysis.pseudodata_method = args.pseudo_data
        
        #analysis.randomize_free_params()

        results = []

        for i in range(args.num_trials):
            logging.info('Running trial %i'%i)
            np.random.seed()
            analysis.generate_psudodata()
            
            if args.function == 'profile':
                if args.mode == 'H0':
                    results.append(analysis.profile(args.var,[0.]*ureg.dimensionless, check_octant=not args.no_check_octant, pprint=not args.quiet))
                elif args.mode == 'scan':
                    results.append(analysis.profile(args.var,eval(args.range), check_octant=not args.no_check_octant, pprint=not args.quiet))
                elif args.mode == 'feldman_cousins':
                    assert(data_fixed_param!=None)
                    p_name, value = data_fixed_param.items()[0][0], data_fixed_param.items()[0][1]
                    print "save the fixed_param_data to output: ", p_name, " ", value
                    return_result=analysis.profile(p_name,[value], check_octant=not args.no_check_octant, pprint=not args.quiet)
                    return_result.append({'data_%s'%p_name:value})
                    results.append(return_result)
            elif args.function == 'fit':
                best_fit_result=analysis.find_best_fit(check_octant=not args.no_check_octant, pprint=not args.quiet)
                if (data_fixed_param!=None):
                    p_name, value = data_fixed_param.items()[0][0], data_fixed_param.items()[0][1]
                    best_fit_result['data_'+p_name]=value
                    print "save the fixed_param_data to output: ", p_name, " ", value
                results.append(best_fit_result)

        to_file(results, args.outfile)
        logging.info('Done.')
