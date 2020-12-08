#!/usr/bin/env python
'''
Test script to compare the performances of
the generalized poisson llh with the other 
miminization metrics available in pisa

'''
from __future__ import absolute_import, print_function, division

__author__ = "Etienne Bourbeau (etienne.bourbeau@icecube.wisc.edu)"


#
# Standard python imports
#
import os
import pickle
from collections import OrderedDict
import copy
import numpy as np

#
# Font stuff
#
import matplotlib as mpl 
mpl.use('agg')
from matplotlib import rcParams
FONTSIZE=20
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 20
mpl.rc('text', usetex=True)
#mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]



#
# pisa tools and objects
#
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.config_parser import parse_pipeline_config
from pisa.core.param import Param, ParamSet
from pisa.analysis.analysis import Analysis

# debug tools
from pisa.utils.log import logging
from pisa.utils.profiler import line_profile
from pisa.utils.log import set_verbosity, Levels
#set_verbosity(Levels.TRACE)

##################################################################################

STANDARD_CONFIG = os.environ['PISA'] + \
    '/pisa/stages/data/super_simple_pipeline.cfg'
TRUE_MU = 20.
TRUE_SIGMA = 3.1
NBINS = 31

#
# Define formatting properties for all metrics
#
import seaborn as sns
COLORS = sns.color_palette("muted", 8)

LIKELIHOOD_FORMATTING = OrderedDict()
LIKELIHOOD_FORMATTING['llh'] = {'label':r'Poisson llh',
                                'marker':'s',
                                'color': COLORS[0]}

LIKELIHOOD_FORMATTING['mcllh_eff'] = {'label':r'Effective llh',
                                'color': COLORS[1]}

LIKELIHOOD_FORMATTING['mcllh_mean'] = {'label':r'Mean llh',
                                       'color': COLORS[2],
                                       'linestyle': '--'}

LIKELIHOOD_FORMATTING['generalized_poisson_llh'] = {'label':r'Generalized llh',
                                       'color': COLORS[7]}

LIKELIHOOD_FORMATTING['mod_chi2'] = {'label':r'Mod. $\chi^{2}$',
                                       'color': COLORS[3]}
################################################################################


class ToyMCllhParam:
    '''
    Class defining the parameters of the Toy MC
    '''

    def __init__(self):

        self.n_data = 0.                # Number of data points to bin
        self.signal_fraction = 1.       # fraction of those points that will constitute the signal
        self.true_mu = TRUE_MU          # True mean of the signal
        self.true_sigma = TRUE_SIGMA    # True width of the signal
        self.nbackground_low = 0.       # lowest value the background can take
        self.nbackground_high = 40.     # highest value the background can take
        self.stats_factor = 1.          # Statistical factor for the MC


        #
        # Binning
        #
        self.binning = None

    @property
    def nsig(self):
        '''
        number of data points that are part of the signal
        '''
        return int(self.n_data*self.signal_fraction)

    @property
    def nbkg(self):
        '''
        number of data points that are part of the background
        '''
        return self.n_data-self.nsig

    @property
    def nbins(self):
        '''
        number of bins in the binning
        '''
        assert self.binning is not None, 'ERROR: specify a binning first'
        return self.binning.tot_num_bins


def create_pseudo_data(toymc_params, seed=None):
    '''
    Create pseudo data consisting of a gaussian peak
    on top of a uniform background
    '''
    if seed is not None:
        np.random.seed(seed)

    binning = toymc_params.binning

    #
    # Gaussian signal peak
    #
    signal = np.random.normal(
        loc=toymc_params.mu, scale=toymc_params.sigma, size=toymc_params.nsig)

    #
    # Uniform background
    #
    background = np.random.uniform(
        high=toymc_params.nbackground_high, low=toymc_params.nbackground_low, size=toymc_params.nbkg)
    total_data = np.concatenate([signal, background])
    counts_data, _ = np.histogram(total_data, bins=binning.bin_edges[0].magnitude)

    # Convert data histogram into a pisa map
    data_map = Map(name='total', binning=binning, hist=counts_data)

    # Set the errors as the sqrt of the counts
    data_map.set_errors(error_hist=np.sqrt(counts_data))

    data_as_mapset = MapSet([data_map])

    return data_as_mapset


def create_mc_template(toymc_params, config_file=None, seed=None, keep_same_weight=True):
    '''
    Create MC template out of a pisa pipeline
    '''
    if seed is not None:
        np.random.seed(seed)

    Config = parse_pipeline_config(config_file)

    # Change binning
    Config[('data','pi_simple_signal')]['output_specs'] = toymc_params.binning
    Config[('likelihood','pi_generalized_llh_params')]['output_specs'] = toymc_params.binning

    # If keep_same_weight is True, turn off the mean adjust and pseudo weight of pi_generalized_llh
    if keep_same_weight:
        Config[('likelihood','pi_generalized_llh_params')]['with_mean_adjust'] = False
        Config[('likelihood','pi_generalized_llh_params')]['with_pseudo_weight'] = False
    else:
        Config[('likelihood','pi_generalized_llh_params')]['with_mean_adjust'] = True
        Config[('likelihood','pi_generalized_llh_params')]['with_pseudo_weight'] = True

    new_n_events_data = Param(
        name='n_events_data', value=toymc_params.n_data, prior=None, range=None, is_fixed=True)
    new_sig_frac = Param(name='signal_fraction', value=toymc_params.signal_fraction,
                         prior=None, range=None, is_fixed=True)
    new_stats_factor = Param(
        name='stats_factor', value=toymc_params.stats_factor, prior=None, range=None, is_fixed=True)

    # These should match the values of the config file, but we override them just in case we need to change these later
    new_mu = Param(name='mu', value=toymc_params.mu,
                   prior=None, range=[0, 100], is_fixed=False)
    new_sigma = Param(name='sigma', value=toymc_params.sigma,
                      prior=None, range=None, is_fixed=True)
    Config[('data', 'pi_simple_signal')]['params'].update(p=ParamSet(
        [new_n_events_data, new_sig_frac, new_stats_factor, new_mu, new_sigma]))

    MCtemplate = DistributionMaker(Config)

    return MCtemplate


##################################################################################

def run_llh_scans(metrics=[], mc_params=None, config_file=None, data_mapset=None, mc_seed=None, results=None):
    '''
    Perform Likelihood scans fover a range of injected mu values

    metrics: list of strings (names of the likelihood to run)

    mc_template: DistributionMaker

    data: MapSet

    '''

    assert isinstance(results, (dict, OrderedDict)
                      ), 'ERROR: results must be a dict'

    assert 'toymc_params' in results.keys(), 'ERROR: missing toymc_params'


    for metric in metrics:
        if metric not in results.keys():
            results[metric] = OrderedDict()
        results[metric]['llh_scan'] = OrderedDict()

    #
    # Create the mc template
    #
    mc_template = create_mc_template(mc_params, config_file=config_file, seed=mc_seed)

    #
    # Collect the llh value at the Truth
    #
    for metric in metrics:
        print(metric)

        mc_template.params['mu'].value = toymc_params.true_mu

        new_MC = mc_template.get_outputs(return_sum=True, force_standard_output=False)

        if metric == 'generalized_poisson_llh':
            llhval = data_mapset.maps[0].metric_total(new_MC, metric=metric, metric_kwargs={
                                                      'empty_bins': mc_template.empty_bin_indices})
            logging.trace('empty_bins: ', mc_template.empty_bin_indices)

        else:
            new_MC = new_MC['old_sum']
            llhval = data_mapset.metric_total(new_MC, metric=metric)

        results[metric]['llh_scan']['llh_at_truth'] = llhval

        results[metric]['llh_scan']['tested_mu'] = np.linspace(10., 30., 50)
        results[metric]['llh_scan']['scan_values'] = []

        #
        # Scan llh values around the true signal peak value
        #
        for tested_mu in results[metric]['llh_scan']['tested_mu']:

            #
            # Recompute the MC template with a new value of the mu parameter
            #
            mc_template.params['mu'].value = tested_mu
            new_MC = mc_template.get_outputs(return_sum=True, force_standard_output=False)

            if metric == 'generalized_poisson_llh':
                llhval = data_mapset.maps[0].metric_total(new_MC, metric=metric, metric_kwargs={
                                                          'empty_bins': mc_template.empty_bin_indices})
            else:
                new_MC = new_MC['old_sum']
                llhval = data_mapset.metric_total(new_MC, metric=metric)


            results[metric]['llh_scan']['scan_values'].append(llhval)


    return results


def plot_llh_scans(metrics=[], results=None, interactive=False, output_pdf=None, prefix='', save_individual_fig=False):
    '''
    Plot Likelihood scans
    '''

    fig, ax = plt.subplots(figsize=(7, 7))
    n = 0
    for llh_name in metrics:

        llhvals = results[llh_name]['llh_scan']['scan_values']
        tested_mu = results[llh_name]['llh_scan']['tested_mu']

        if 'chi2' in llh_name:
            TS = llhvals-np.amin(llhvals)
        else:
            TS = -2*(llhvals-np.amax(llhvals))

        ax.plot(tested_mu, TS, **LIKELIHOOD_FORMATTING[llh_name])
        n += 1
    ax.set_xlabel(r'injected $\mu$')
    ax.set_ylabel(r'Test Statistic(-2$\ln[L_{\mu}/L_{o}]$ or $\chi^{2}$)')
    ax.set_ylim([-10., 500])
    ax.plot([15.,25.],[0.,0.],'k')
    ax.set_title('MC factor = {}'.format(results['toymc_params'].stats_factor))
    #ax.set_title('Likelihood scans over mu')
    ax.legend()
    fig.tight_layout()

    if interactive:
        plt.show()

    if save_individual_fig:
        plt.savefig(prefix+'plot_llh_scan.png')

    if output_pdf is None:
        return fig
    else:
        output_pdf.savefig(fig)
        plt.close('all')
        del fig
        return 1


###################################################################################################
#@line_profile
def run_coverage_test(n_trials=100,
                      toymc_params=None,
                      mc_seed = None,
                      config_file=None,
                      metrics=None,
                      results=None,
                      output_stem='coverage_test'):
    '''
    Perform Coverage and bias tests

    We create n_trials pseudo-dataset, Fit them
    with each metric at various levels of statistics.
    and save the resulting llh values and fitted 
    parameters into a file

    n_trials: int (number of pseudo-experiment to run)

    toymc_params: ToyMC_LLh object (describe the parameters
                  of the experiment like signal_fraction and
                  stats_factor)


    mc_infinite_Stats: DistributionMaker 
                       (MC template made with an ideal level
                    of stats representing "infinite MC" precision)

    '''
    import time

    assert isinstance(results, (dict, OrderedDict)), 'ERROR: results must be a dict'
    assert isinstance(metrics, list), 'ERROR: must specify metrics as a list'
    assert 'toymc_params' in results.keys(), 'ERROR: missing toymc_params'


    results['toymc_params'] = toymc_params
    for metric in metrics:
        if metric not in results.keys():
            results[metric] = OrderedDict()
        results[metric]['coverage'] = []

    #
    # minimizer settings to pass into the pisa analysis class
    #
    minimizer_settings = {"method": {"value": "l-bfgs-b",  # "SLSQP",
                                     "desc": "The string to pass to scipy.optimize.minimize so it knows what to use"
                                     },
                          "options": {"value": {"disp": 0,
                                                "ftol": 1.0e-6,
                                                "eps": 1.0e-6,
                                                "maxiter": 100
                                                },
                                      "desc": {"disp": "Set to True to print convergence messages",
                                               "ftol": "Precision goal for the value of f in the stopping criterion",
                                               "eps": "Step size used for numerical approximation of the jacobian.",
                                               "maxiter": "Maximum number of iteration"
                                               }
                                      }
                          }


    #
    # Create the mc template
    #
    mc_template = create_mc_template(toymc_params, config_file=config_file, seed=mc_seed)

    #
    # Create a pseudo-infinite statistics template
    #
    infinite_toymc_params = copy.deepcopy(toymc_params)
    infinite_toymc_params.stats_factor = 100.
    mc_template_pseudo_infinite = create_mc_template(infinite_toymc_params, config_file=config_file, seed=mc_seed)

    #
    # Start pseudo trials
    #
    for metric in metrics_to_test:
        filename = output_stem+'_pseudo_exp_llh_%s.pckl' % metric
        if os.path.isfile(filename):
            results[metric]['coverage'] = pickle.load(open(filename,'rb'))

        else:

            logging.debug('minimizing: ', metric)
            to = time.time()

            trial_i = 0
            failed_fits = 0
            while trial_i < n_trials and failed_fits<2*n_trials:

                experiment_result = {}

                #
                # Create a pseudo-dataset
                #
                data_trial = create_pseudo_data(toymc_params=toymc_params, seed=None)

                #
                # Compute the truth llh value of this pseudo experiment
                # truth - if the truth comes from infinite stats MC
                #
                if metric == 'generalized_poisson_llh':
                    mc = mc_template_pseudo_infinite.get_outputs(return_sum=False, force_standard_output=False)[0]

                    llhval_true = data_trial.maps[0].metric_total(mc, 
                                                                  metric=metric, 
                                                                  metric_kwargs={
                                                                  'empty_bins': mc_template_pseudo_infinite.empty_bin_indices})
                else:
                    mc = mc_template_pseudo_infinite.get_outputs(return_sum=True)
                    llhval_true = data_trial.metric_total(mc, metric=metric)

                experiment_result['llh_infinite_stats'] = llhval_true

                #
                # truth if the truth comes from low stats MC
                #
                if metric == 'generalized_poisson_llh':
                    mc = mc_template.get_outputs(return_sum=False,
                                                 force_standard_output=False)[0]

                    llhval = data_trial.maps[0].metric_total(mc, 
                                                             metric=metric, 
                                                             metric_kwargs={
                                                             'empty_bins': mc_template.empty_bin_indices})
                else:
                    mc = mc_template.get_outputs(return_sum=True)
                    llhval = data_trial.metric_total(mc,
                                                     metric=metric)

                experiment_result['llh_lowstats'] = llhval

                # #
                # # minimized llh (high stats)
                # #
                # logging.debug('\nhigh stats fit:\n')
                # ana = Analysis()
                # result_pseudo_truth, _ = ana.fit_hypo(data_trial,
                #                                       mc_infinite_stats,
                #                                       metric=metric,
                #                                       minimizer_settings=minimizer_settings,
                #                                       hypo_param_selections=None,
                #                                       check_octant=False,
                #                                       fit_octants_separately=False,
                #                                       )
                # #except:
                # #    logging.trace('Failed Fit')
                # #    failed_fits += 1
                # #    continue
                # experiment_result['infinite_stats_opt'] = {'metric_val': result_pseudo_truth['metric_val'],
                #                                            'best_fit_param': result_pseudo_truth['params']['mu']}

                #
                # minimized llh (low stats)
                #
                logging.debug('\nlow stats fit:\n')
                ana = Analysis()

                try:
                    result_lowstats, _ = ana.fit_hypo(data_trial,
                                                      mc_template,
                                                      metric=metric,
                                                      minimizer_settings=minimizer_settings,
                                                      hypo_param_selections=None,
                                                      check_octant=False,
                                                      fit_octants_separately=False,
                                                      )
                except:
                    logging.debug('Failed Fit')
                    failed_fits += 1
                    continue

                experiment_result['lowstats_opt'] = {'metric_val': result_lowstats['metric_val'],
                                                     'best_fit_param': result_lowstats['params']['mu']}

                results[metric]['coverage'].append(experiment_result)
                trial_i += 1

            if trial_i==0:
                raise Exception('ERROR: no fit managed to converge after {} attempst'.format(failed_fits))

            t1 = time.time()
            logging.debug("Time for ", n_trials, " minimizations: ", t1-to, " s")
            logging.debug("Saving to file...")
            pickle.dump(results[metric]['coverage'], open(filename, 'wb'))
            logging.debug("Saved.")

    return results


def plot_coverage_test(output_pdf=None,
                       results=None,
                       metrics=None,
                       stats_factor=None,
                       output_stem=None,
                       n_trials=None,
                       prefix='',
                       save_individual_fig=False,
                       outname='test_coverage.pdf'):
    '''
    plot the results of the coverage test
    '''
    from utils.plotting.standard_modules import Figure

    assert isinstance(metrics, list), 'ERROR: must specify metrics as a list'
    from scipy.stats import chi2


    if output_pdf is None:
        output_pdf = PdfPages(outname)

    coverage_fig = Figure(figsize=(7, 7))

    #
    # produce an example chi2 distribution with d.o.f =1
    #
    # This will help us compare ts distribution directly
    sample_chi2_distrib = np.random.chisquare(size=n_trials, df=1)

    for llh_name in metrics:

        logging.trace('plotting %s'%llh_name)

        container_ts_truth_high = []
        container_ts_truth_low = []
        container_ts_lowstat = []
        container_ts_highstat = []
        llh_bias = []
        param_bias = []

        val_truth = TRUE_MU
        container_val_lowstat = []
        container_val_highstat = []

        # Retrieve data from the coverage test
        indata = results[llh_name]['coverage']

        if len(indata) < 1:
            print('No successful fits for metric: {}.skipping')

        for pseudo_exp in indata:

            val_low = pseudo_exp['lowstats_opt']['best_fit_param'].value.m
            llh_optimized_low = pseudo_exp['lowstats_opt']['metric_val']
            llh_truth_low = pseudo_exp['llh_lowstats']
            llh_truth_high = pseudo_exp['llh_infinite_stats']

            #
            # check that all elements of the comparison are finite
            #
            good_trial = np.isfinite(val_low)
            good_trial *= np.isfinite(llh_optimized_low)
            good_trial *= np.isfinite(llh_truth_low)
            good_trial *= np.isfinite(llh_truth_high)

            if good_trial:

                container_val_lowstat.append(val_low)
                container_ts_truth_high.append(llh_truth_high)
                container_ts_truth_low.append(llh_truth_low)

                ts_low = -2*(llh_optimized_low-llh_truth_low)

                # We take the absolute value here because we want to know how far
                # we are from the truth, and we can optimize to llh values above and below the truth
                container_ts_lowstat.append(np.abs(ts_low))

                param_bias.append((val_low-val_truth)/val_truth)

            else:
                continue

        #
        # First plot: TS distribution
        #
        fig_ts_distrib = Figure(figsize=(7,7), title=LIKELIHOOD_FORMATTING[llh_name]['label'])
        ts_binning = np.linspace(0, 25, 31)
        c,ts_edges = np.histogram(sample_chi2_distrib, bins=ts_binning)
        ts_x = ts_edges[:-1]+0.5*(ts_edges[1:]-ts_edges[:-1])
        fig_ts_distrib.get_ax().errorbar(ts_x, c, yerr=np.sqrt(c), drawstyle='steps-mid',
                                         linewidth=2., color='k', label=r'$\chi^{2}_{dof=1}$')

        fig_ts_distrib.get_ax().set_title('TS Distribution - {} x MC vs. data'.format(stats_factor))
        cmc, _ = np.histogram(container_ts_lowstat, bins=ts_binning)
        _, chi2_bins, _ = fig_ts_distrib.get_ax().errorbar(ts_x, cmc, yerr=np.sqrt(cmc), drawstyle='steps-mid', 
                                        linewidth=2., color='b', label='TS distribution')

        fig_ts_distrib.get_ax().set_xlabel(r'$\left |-2(LLH_{opt}-LLH_{truth}) \right |$ (Low statistics case)')
        fig_ts_distrib.get_ax().legend()
        if save_individual_fig:
            plt.savefig(prefix+llh_name+'plot_TS_distribution.png')

        #
        # Second plot: Bias distribution (absolute)
        #
        fig_bias = Figure(figsize=(7,7), title=LIKELIHOOD_FORMATTING[llh_name]['label'])
        fig_bias.get_ax().axvline(x=20, linewidth=2,
                                     color='k', ls='--', label=r'Truth ($\mu = 20$')
        fig_bias.get_ax().hist(container_val_lowstat, bins=20, histtype='step',
                                  linewidth=2., color='b', label=r'Best-fit $\mu_{opt}$')

        fig_bias.get_ax().set_xlabel(r'value')
        fig_bias.get_ax().set_title('Fitted Parameter Value - {} x MC vs. data'.format(stats_factor))
        if save_individual_fig:
            plt.savefig(prefix+llh_name+'plot_bias_abs.png')

        #
        # Third plot: Bias distribution (relative)
        #
        fig_pull = Figure(figsize=(7,7), title=LIKELIHOOD_FORMATTING[llh_name]['label'])
        fig_pull.get_ax().set_title('Parameter Bias in low stats')
        fig_pull.get_ax().hist(param_bias, bins=20)
        fig_pull.get_ax().set_xlabel(r'$\left( \frac{\mu_{opt}-\mu_{true}}{\mu_{true}}\right)$')
        fig_pull.get_ax().axvline(x=0., linewidth=2, color='k', ls='--')
        if save_individual_fig:
            plt.savefig(prefix+llh_name+'plot_bias_rel.png')

        #
        # Coverage test
        #
        coverage_y = []
        coverage_x = np.linspace(0.0, 1.0, 101)

        for percent_coverage in coverage_x:
            chi2_ts_value = chi2.ppf(percent_coverage, df=1)
            actual_coverage = sum(np.array(container_ts_lowstat) <= chi2_ts_value)

            if len(container_ts_lowstat)>0:
                actual_coverage/=float(len(container_ts_lowstat))
            else:
                actual_coverage=0.
            coverage_y.append(actual_coverage)

        coverage_fig.get_ax().plot(coverage_x, coverage_y, **LIKELIHOOD_FORMATTING[llh_name])

    coverage_fig.get_ax().set_xlabel('Expected Wilks coverage')
    coverage_fig.get_ax().set_ylabel('Actual Coverage (low statistics')
    coverage_fig.get_ax().legend()

    if output_pdf is not None:
        output_pdf.savefig(coverage_fig.fig)

    if save_individual_fig:
        plt.figure(coverage_fig.fig.number)
        plt.savefig(prefix+'plot_coverage.png')


def plot_data_and_mc(data_map=None,
                     config_file=STANDARD_CONFIG,
                     toymc_params=None,
                     interactive=False,
                     mc_seed=None,
                     output_pdf=None):
    '''
    plot the data, and the mc sets overlaid on top
    '''
    # ===============================================================
    #
    # Generate MC template using a pisa pipeline
    #
    # We first need to override the parameter values contained in the config file
    # before instantiating the pipeline
    print('Create the first template')
    mc_template = create_mc_template(toymc_params, config_file=config_file, seed=mc_seed)
    mc_map = sum(mc_template.get_outputs(return_sum=True, force_standard_output=False)['old_sum']) #old_sum = map without pseudo weights
    mc_generalized_map = sum(mc_template.get_outputs(return_sum=True, force_standard_output=False)['weights']) #weights = map with pseudo-weight
    mc_params = mc_template.params


    # =================================================================
    #
    # Produce a pseudo-infinite MC statistics template
    # Create a MC set with 10000 times more stats than data. will be used as the truth
    #
    print('creating the infinite template')
    infinite_toymc_params = copy.deepcopy(toymc_params)
    infinite_toymc_params.stats_factor = 1000.
    mc_template_pseudo_infinite = create_mc_template(infinite_toymc_params, config_file=config_file, seed=mc_seed)
    mc_map_pseudo_infinite = sum(mc_template_pseudo_infinite.get_outputs(return_sum=True, force_standard_output=False)['old_sum'])

    X = toymc_params.binning.midpoints[0].magnitude

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.errorbar(X, data_map.nominal_values, yerr=np.sqrt(data_map.nominal_values),
     label='data', fmt='-o', drawstyle='steps-mid', color='k')
    ax.set_xlabel('Arbitrary variable')
    ax.set_ylabel('Frequency (A.U.)')

    ax.text(0.65, 0.91, r'$\mu_{true}$ = '+'{}'.format(TRUE_MU),
            fontsize=20, transform=ax.transAxes)
    ax.text(0.65, 0.85, r'$\sigma_{true}$ = ' +
            '{}'.format(TRUE_SIGMA), fontsize=20, transform=ax.transAxes)
    ax.text(0.65, 0.79, r'$N_{signal}$ = '+'{}'.format(toymc_params.nsig),
            fontsize=20, transform=ax.transAxes)
    ax.text(0.65, 0.73, r'$N_{bkg}$ = '+'{}'.format(toymc_params.nbkg),
            fontsize=20, transform=ax.transAxes)
    ax.legend(loc='upper left')

    if interactive:
        plt.show()
    if output_pdf is not None:
        output_pdf.savefig(fig)

    #
    # Update the same plot with the low stats MC
    #
    ax.plot(X, mc_map.nominal_values, '-g', label='MC', drawstyle='steps-mid', zorder=10)
    ax.text(0.65, 0.67, r'$\mu_{MC}$ = '+'{}'.format(
        mc_params['mu'].value.m), color='g', fontsize=20, transform=ax.transAxes)
    ax.text(0.65, 0.61, r'$\sigma_{MC}$ = '+'{}'.format(
        mc_params['sigma'].value.m), color='g', fontsize=20, transform=ax.transAxes)
    ax.legend(loc='upper left')

    ax.set_title('MC factor = {}'.format(mc_params['stats_factor'].value.m))
    if interactive:
        plt.show()
    if output_pdf is not None:
        output_pdf.savefig(fig)

    #
    # Update the same plot with the low stats MC modified for the generalized Poisson
    #
    ax.plot(X, mc_generalized_map.nominal_values, ':b', label='MC - Generalized', drawstyle='steps-mid', zorder=10)
    ax.legend(loc='upper left')
    if interactive:
        plt.show()
    if output_pdf is not None:
        output_pdf.savefig(fig)

    #
    # Update with the pseudo-infinite MC set
    #
    ax.plot(X, mc_map_pseudo_infinite.nominal_values,
            '-r', label='MC (large statistics)', drawstyle='steps-mid', zorder=10)
    ax.legend(loc='upper left')

    if interactive:
        plt.show()
    if output_pdf is not None:
        output_pdf.savefig(fig)


##################################################################################
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        '1D Toy Monte Carlo to test various likelihoods / chi2 metrics')

    parser.add_argument(
        '-nd', '--ndata', help='total number of data points', type=int, default=200)
    parser.add_argument('-sf', '--signal-fraction',
                        help='fraction of the data in the signal dataset', type=float, default=1.)
    parser.add_argument('-s', '--stats-factor',
                        help='Defines how much MC weights to produce w.r.t data', type=float, default=1.)
    parser.add_argument(
        '-nt', '--ntrials', help='number of pseudo experiments in the bias study', type=int, default=200)

    parser.add_argument(
        '--make-llh-scan', help='if chosen, will run the likelihood scan for all llh', action='store_true')
    parser.add_argument('--run-coverage-test',
                        help='if chosen, run the pseudo-trials needed to perform coverage and bias tests', action='store_true')
    parser.add_argument(
        '-o', '--output', help='output stem files with plots', default='ToyMC_LLh')

    parser.add_argument(
        '--interactive', help='use interactive plots', action='store_true')

    args = parser.parse_args()

    #
    # Plotting tools
    #
    import matplotlib as mpl
    if not args.interactive:
        mpl.use('agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    output_pdf = PdfPages(args.output+'.pdf')

    #
    # Check some stuff
    #
    assert args.signal_fraction <= 1., 'ERROR: signal fraction ust be smaller or equal to 1'

    # ================================================================
    #
    # Parameters of the data
    #
    toymc_params = ToyMCllhParam()
    # Number of data points to bin
    toymc_params.n_data = args.ndata
    # fraction of those points that will constitute the signal
    toymc_params.signal_fraction = args.signal_fraction
    toymc_params.mu = TRUE_MU                        # True mean of the signal
    toymc_params.sigma = TRUE_SIGMA                  # True width of the signal
    toymc_params.nbackground_low = 0.                # lowest value the background can take
    toymc_params.nbackground_high = 40.              # highest value the background can take
    toymc_params.binning = MultiDimBinning( OneDimBinning(name='stuff', bin_edges=np.linspace(
        toymc_params.nbackground_low, toymc_params.nbackground_high, NBINS)))
    # Statistical factor for the MC
    toymc_params.stats_factor = args.stats_factor
    toymc_params.infinite_stats = 1000.


    metrics_to_test = ['llh', 'mcllh_eff','mod_chi2',
                       'mcllh_mean', 'generalized_poisson_llh']

    results = OrderedDict()
    results['toymc_params'] = toymc_params
    for metric in metrics_to_test:
        results[metric] = OrderedDict()

    # ==============================================================
    #
    # Generate a toy data set
    #
    data_as_mapset = create_pseudo_data(toymc_params=toymc_params, seed=564525)

   

    # =================================================================
    #
    # Plot the three graphs
    #
    plot_data_and_mc(data_map=data_as_mapset.maps[0],
                     config_file=STANDARD_CONFIG,
                     toymc_params=toymc_params,
                     mc_seed=564525,
                     interactive=args.interactive,
                     output_pdf=output_pdf)

    # ==================================================================
    #
    # Perform llh scans
    #
    if args.make_llh_scan:

        results = run_llh_scans(metrics=metrics_to_test,
                                mc_params=toymc_params,
                                config_file=STANDARD_CONFIG,
                                data_mapset=data_as_mapset,
                                mc_seed=564525,
                                results=results)

        plot_llh_scans(metrics=metrics_to_test, results=results,
                       save_individual_fig=True,
                       prefix=args.output,
                       interactive=args.interactive, output_pdf=output_pdf)

    # ==================================================================
    #
    # Perform bias and coverage test
    #
    if args.run_coverage_test:

        results = run_coverage_test(n_trials=args.ntrials,
                                    toymc_params=toymc_params,
                                    config_file=STANDARD_CONFIG,
                                    metrics=metrics_to_test,
                                    results=results,
                                    output_stem=args.output,
                                    )

        plot_coverage_test(output_pdf=output_pdf,
                           metrics=metrics_to_test,
                           results=results,
                           stats_factor=toymc_params.stats_factor,
                           output_stem=args.output,
                           save_individual_fig=True,
                           prefix=args.output,
                           n_trials=args.ntrials)

    output_pdf.close()
