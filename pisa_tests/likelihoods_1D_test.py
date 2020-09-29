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
import collections
import copy
import numpy as np


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
set_verbosity(Levels.TRACE)

##################################################################################

STANDARD_CONFIG = os.environ['PISA'] + \
    '/pisa/stages/data/super_simple_pipeline.cfg'
TRUE_MU = 20.
TRUE_SIGMA = 3.1
NBINS = 51

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
        # Value of what we consider an infinite MC stats factor
        self.infinite_stats = 10000

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
    counts_data, _ = np.histogram(total_data, bins=binning.bin_edges.magnitude)

    # Convert data histogram into a pisa map
    data_map = Map(name='total', binning=MultiDimBinning(
        [binning]), hist=counts_data)

    # Set the errors as the sqrt of the counts
    data_map.set_errors(error_hist=np.sqrt(counts_data))

    data_as_mapset = MapSet([data_map])

    return data_as_mapset


def create_mc_template(toymc_params, config_file=None, seed=None):
    '''
    Create MC template out of a pisa pipeline
    '''
    if seed is not None:
        np.random.seed(seed)

    Config = parse_pipeline_config(config_file)

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

def run_llh_scans(metrics=[], mc_template=None, data_mapset=None, results=None):
    '''
    Perform Likelihood scans fover a range of injected mu values

    metrics: list of strings (names of the likelihood to run)

    mc_template: DistributionMaker

    data: MapSet

    '''

    assert isinstance(results, (dict, collections.OrderedDict)
                      ), 'ERROR: results must be a dict'

    assert 'toymc_params' in results.keys(), 'ERROR: missing toymc_params'

    toymc_params = results['toymc_params']
    for metric in metrics:
        if metric not in results.keys():
            results[metric] = collections.OrderedDict()
        results[metric]['llh_scan'] = collections.OrderedDict()

    #
    # Collect the llh value at the Truth
    #
    for metric in metrics:

        mc_template.params['mu'].value = toymc_params.true_mu

        if metric == 'generalized_poisson_llh':
            new_MC = mc_template.get_outputs(
                return_sum=False, force_standard_output=False)[0]
            llhval = data_mapset.maps[0].metric_total(new_MC, metric=metric, metric_kwargs={
                                                      'empty_bins': mc_template.empty_bin_indices})
            logging.trace('empty_bins: ', mc_template.empty_bin_indices)
        else:
            new_MC = mc_template.get_outputs(return_sum=True)
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

            if metric == 'generalized_poisson_llh':
                new_MC = mc_template.get_outputs(
                    return_sum=False, force_standard_output=False)[0]
                llhval = data_mapset.maps[0].metric_total(new_MC, metric=metric, metric_kwargs={
                                                          'empty_bins': mc_template.empty_bin_indices})
            else:
                new_MC = mc_template.get_outputs(return_sum=True)
                llhval = data_mapset.metric_total(new_MC, metric=metric)

            if 'chi2' in metric:
                results[metric]['llh_scan']['scan_values'].append((llhval-results[metric]['llh_scan']['llh_at_truth']))
            else:
                results[metric]['llh_scan']['scan_values'].append(-2*(llhval-results[metric]['llh_scan']['llh_at_truth']))

    return results


def plot_llh_scans(metrics=[], results=None, interactive=False, output_pdf=None):
    '''
    Plot Likelihood scans
    '''

    fig, ax = plt.subplots(figsize=(9, 9))
    n = 0
    for llh_name in metrics:

        llhvals = results[llh_name]['llh_scan']['scan_values']
        tested_mu = results[llh_name]['llh_scan']['tested_mu']

        if llh_name == 'mcllh_eff':
            ax.plot(tested_mu, llhvals, 'o', color=COLORS[n], label=llh_name)
        else:
            ax.plot(tested_mu, llhvals, linewidth=2,
                    color=COLORS[n], label=llh_name)
        n += 1
    ax.set_xlabel(r'injected $\mu$')
    ax.set_ylabel(r'-2$\log[L_{\mu}/L_{o}]$')
    ax.set_ylim([0., 5000])
    ax.set_title('Likelihood scans over mu')
    ax.legend()

    if interactive:
        plt.show()

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
                      mc_template=None,
                      mc_infinite_stats=None,
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

    mc_template: DistributionMaker 
                 (MC template made with the level of stats you test)

    mc_infinite_Stats: DistributionMaker 
                       (MC template made with an ideal level
                    of stats representing "infinite MC" precision)

    '''
    import time

    assert isinstance(results, (dict, collections.OrderedDict)), 'ERROR: results must be a dict'
    assert isinstance(metrics, list), 'ERROR: must specify metrics as a list'
    assert 'toymc_params' in results.keys(), 'ERROR: missing toymc_params'

    toymc_params = results['toymc_params']

    for metric in metrics:
        if metric not in results.keys():
            results[metric] = collections.OrderedDict()
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
    # Start pseudo trials
    #
    for metric in metrics_to_test:
        filename = output_stem+'_pseudo_exp_llh_%s.pckl' % metric
        if os.path.isfile(filename):
            results[metric]['coverage'] = pickle.load(open(filename,'rb'))

        else:

            logging.trace('minimizing: ', metric)
            to = time.time()

            trial_i = 0
            failed_fits = 0
            while trial_i < n_trials and failed_fits<2*n_trials:

                experiment_result = {}

                #
                # Create a pseudo-dataset
                #
                data_trial = create_pseudo_data(
                    toymc_params=toymc_params, seed=None)

                #
                # Compute the truth llh value of this pseudo experiment
                # truth - if the truth comes from infinite stats MC
                #
                if metric == 'generalized_poisson_llh':
                    mc = mc_infinite_stats.get_outputs(
                        return_sum=False, force_standard_output=False)[0]
                    llhval_true = data_trial.maps[0].metric_total(mc, 
                                                                  metric=metric, 
                                                                  metric_kwargs={
                                                                  'empty_bins': mc_infinite_stats.empty_bin_indices})
                else:
                    mc = mc_infinite_stats.get_outputs(return_sum=True)
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

                #
                # minimized llh (high stats)
                #
                logging.trace('\nhigh stats fit:\n')
                ana = Analysis()
                result_pseudo_truth, _ = ana.fit_hypo(data_trial,
                                                      mc_infinite_stats,
                                                      metric=metric,
                                                      minimizer_settings=minimizer_settings,
                                                      hypo_param_selections=None,
                                                      check_octant=False,
                                                      fit_octants_separately=False,
                                                      )
                #except:
                #    logging.trace('Failed Fit')
                #    failed_fits += 1
                #    continue
                experiment_result['infinite_stats_opt'] = {'metric_val': result_pseudo_truth['metric_val'],
                                                           'best_fit_param': result_pseudo_truth['params']['mu']}

                #
                # minimized llh (low stats)
                #
                logging.trace('\nlow stats fit:\n')
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
                    logging.trace('Failed Fit')
                    failed_fits += 1
                    continue

                experiment_result['lowstats_opt'] = {'metric_val': result_lowstats['metric_val'],
                                                     'best_fit_param': result_lowstats['params']['mu']}

                results[metric]['coverage'].append(experiment_result)
                trial_i += 1

            if trial_i==0:
                raise Exception('ERROR: no fit managed to converge after {} attempst'.format(failed_fits))

            t1 = time.time()
            logging.trace("Time for ", n_trials, " minimizations: ", t1-to, " s")
            logging.trace("Saving to file...")
            pickle.dump(results[metric]['coverage'], open(filename, 'wb'))
            logging.trace("Saved.")

    return results


def plot_coverage_test(output_pdf=None,
                       results=None,
                       metrics=None,
                       stats_factor=None,
                       output_stem=None,
                       n_trials=None,
                       outname='test_coverage.pdf'):
    '''
    plot the results of the coverage test
    '''
    from utils.plotting.standard_modules import Figure

    assert isinstance(metrics, list), 'ERROR: must specify metrics as a list'
    from scipy.stats import chi2


    if output_pdf is None:
        output_pdf = PdfPages(outname)

    coverage_fig = Figure(figsize=(10, 10))

    #
    # produce an example chi2 distribution with d.o.f =1
    #
    # This will help us compare ts distribution directly
    sample_chi2_distrib = np.random.chisquare(size=n_trials, df=1)
    ts_binning = np.linspace(0, 50, 31)

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
            val_high = pseudo_exp['infinite_stats_opt']['best_fit_param'].value.m
            llh_optimized_low = pseudo_exp['lowstats_opt']['metric_val']
            llh_optimized_high = pseudo_exp['infinite_stats_opt']['metric_val']
            llh_truth_low = pseudo_exp['llh_lowstats']
            llh_truth_high = pseudo_exp['llh_infinite_stats']

            #
            # check that all elements of the comparison are finite
            #
            good_trial = np.isfinite(val_low)
            good_trial *= np.isfinite(val_high)
            good_trial *= np.isfinite(llh_optimized_low)
            good_trial *= np.isfinite(llh_optimized_high)
            good_trial *= np.isfinite(llh_truth_low)
            good_trial *= np.isfinite(llh_truth_high)

            if good_trial:

                container_val_lowstat.append(val_low)
                container_val_highstat.append(val_high)
                container_ts_truth_high.append(llh_truth_high)
                container_ts_truth_low.append(llh_truth_low)

                ts_low = -2*(llh_optimized_low-llh_truth_low)
                ts_high = -2*(llh_optimized_high-llh_truth_high)

                # We take the absolute value here because we want to know how far
                # we are from the truth, and we can optimize to llh values above and below the truth
                container_ts_lowstat.append(np.abs(ts_low))
                container_ts_highstat.append(np.abs(ts_high))

                llh_bias.append(-2*(ts_low-ts_high))
                param_bias.append((val_low-val_truth)/val_truth)

            else:
                continue

        fig = Figure(nx=2, ny=3, figsize=(20, 30), title=llh_name)
        fig.get_ax(x=0, y=0).set_title(
            'ts Distribution - 10000 x more MC than data')
        fig.get_ax(x=0, y=0).hist(container_ts_highstat, bins=ts_binning,
                                  histtype='step', linewidth=2., color='r', label='ts distribution')
        fig.get_ax(x=0, y=0).hist(sample_chi2_distrib, bins=ts_binning,
                                  histtype='step', linewidth=2., color='k', label=r'$\chi^{2}_{dof=1}$')
        fig.get_ax(x=0, y=0).set_xlabel(
            r'$\left |-2(LLH_{opt}-LLH_{truth}) \right |$ (High statistics case)')
        fig.get_ax(x=0, y=0).legend()

        fig.get_ax(x=0, y=1).set_title(
            'ts Distribution - {} x MC vs. data'.format(stats_factor))
        _, chi2_bins, _ = fig.get_ax(x=0, y=1).hist(
            container_ts_lowstat, bins=20, histtype='step', linewidth=2., color='b', label='ts distribution')
        fig.get_ax(x=0, y=1).hist(sample_chi2_distrib, bins=chi2_bins,
                                  histtype='step', linewidth=2., color='k', label=r'$\chi^{2}_{dof=1}$')
        fig.get_ax(x=0, y=1).set_xlabel(
            r'$\left |-2(LLH_{opt}-LLH_{truth}) \right |$ (Low statistics case)')
        fig.get_ax(x=0, y=1).legend()

        fig.get_ax(x=0, y=1).set_title(
            'Fitted Parameter Value - 10000 x MC vs. data')
        fig.get_ax(x=1, y=0).hist(container_val_highstat, bins=20, histtype='step',
                                  linewidth=2., color='r', label=r'Best-fit $\mu_{opt}$')
        fig.get_ax(x=1, y=0).axvline(x=20, linewidth=2,
                                     color='k', ls='--', label=r'Truth ($\mu = 20$')
        fig.get_ax(x=1, y=0).set_xlabel(r'value')
        fig.get_ax(x=1, y=0).legend()

        fig.get_ax(x=0, y=1).set_title(
            'Fitted Parameter Value - {} x MC vs. data'.format(stats_factor))
        fig.get_ax(x=1, y=1).hist(container_val_lowstat, bins=20, histtype='step',
                                  linewidth=2., color='b', label=r'Best-fit $\mu_{opt}$')
        fig.get_ax(x=1, y=1).axvline(x=20, linewidth=2,
                                     color='k', ls='--', label=r'Truth ($\mu = 20$')
        fig.get_ax(x=1, y=1).set_xlabel(r'value')
        fig.get_ax(x=1, y=1).legend()

        fig.get_ax(x=0, y=2).set_title('LLH Bias')
        fig.get_ax(x=0, y=2).hist(llh_bias, bins=20)
        fig.get_ax(x=0, y=2).set_xlabel(
            r'llh Bias $-2\left (ts_{low\ stats} -ts_{high\ stats} \right )$')
        fig.get_ax(x=0, y=2).axvline(x=0., linewidth=2, color='k', ls='--')

        fig.get_ax(x=1, y=2).set_title('Parameter Bias in low stats')
        fig.get_ax(x=1, y=2).hist(param_bias, bins=20)
        fig.get_ax(x=1, y=2).set_xlabel(
            r'$\left( \frac{\mu_{opt}-\mu_{true}}{\mu_{true}}\right)$')
        fig.get_ax(x=1, y=2).axvline(x=0., linewidth=2, color='k', ls='--')
        output_pdf.savefig(fig.fig)

        #
        # Coverage test
        #
        coverage_y = []
        coverage_x = np.linspace(0.0, 1.0, 101)

        for percent_coverage in coverage_x:
            chi2_ts_value = chi2.ppf(percent_coverage, df=1)
            actual_coverage = sum(np.array(
                container_ts_lowstat) <= chi2_ts_value)/float(len(container_ts_lowstat))
            coverage_y.append(actual_coverage)

        coverage_fig.get_ax().plot(coverage_x, coverage_y, label=llh_name)

    coverage_fig.get_ax().set_xlabel('Expected Wilks coverage')
    coverage_fig.get_ax().set_ylabel('Actual Coverage (low statistics')
    coverage_fig.get_ax().legend()
    output_pdf.savefig(coverage_fig.fig)


def plot_data_and_mc(data_map=None,
                     mc_map=None,
                     mc_params=None,
                     mc_map_pseudo_infinite=None, 
                     mc_params_pseudo_infinite=None,
                     toymc_params=None,
                     toymc_params_pseudo_infinite=None,
                     interactive=False,
                     output_pdf=None):
    '''
    plot the data, and the mc sets overlaid on top
    '''
    X = toymc_params.binning.midpoints

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.errorbar(X, data_map.nominal_values, yerr=np.sqrt(data_map.nominal_values),
     label='data', fmt='-o', drawstyle='steps-mid', color='k')
    ax.set_xlabel('Some variable')
    ax.set_ylabel('Some counts')
    ax.set_title('Pseudo data fed into the likelihoods')
    ax.text(0.65, 0.9, r'$\mu_{true}$ = '+'{}'.format(TRUE_MU),
            fontsize=12, transform=ax.transAxes)
    ax.text(0.65, 0.85, r'$\sigma_{true}$ = ' +
            '{}'.format(TRUE_SIGMA), fontsize=12, transform=ax.transAxes)
    ax.text(0.65, 0.8, r'$N_{signal}$ = '+'{}'.format(toymc_params.nsig),
            fontsize=12, transform=ax.transAxes)
    ax.text(0.65, 0.75, r'$N_{bkg}$ = '+'{}'.format(toymc_params.nbkg),
            fontsize=12, transform=ax.transAxes)
    ax.legend(loc='upper left')
    if interactive:
        plt.show()
    output_pdf.savefig(fig)

    #
    # Update the same plot with the low stats MC
    #
    ax.plot(X, mc_map.nominal_values, '-g', label='MC', drawstyle='steps-mid', zorder=10)
    ax.text(0.65, 0.7, r'$\mu_{MC}$ = '+'{}'.format(
        mc_params['mu'].value.m), color='g', fontsize=12, transform=ax.transAxes)
    ax.text(0.65, 0.65, r'$\sigma_{MC}$ = '+'{}'.format(
        mc_params['sigma'].value.m), color='g', fontsize=12, transform=ax.transAxes)
    ax.text(0.65, 0.6, 'MC factor = {}'.format(
        mc_params['stats_factor'].value.m), color='g', fontsize=12, transform=ax.transAxes)
    ax.legend(loc='upper left')
    if interactive:
        plt.show()
    output_pdf.savefig(fig)

    #
    # Update with the pseudo-infinite MC set
    #
    ax.plot(X, mc_map_pseudo_infinite.nominal_values,
            '-r', label='MC (large statistics)', drawstyle='steps-mid', zorder=10)
    ax.text(0.65, 0.55, 'Inf. MC factor = {}'.format(mc_params_pseudo_infinite['stats_factor'].value.m), 
            color='r', fontsize=12, transform=ax.transAxes)
    ax.legend(loc='upper left')
    if interactive:
        plt.show()
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
    COLORS = sns.color_palette("hls", 8)
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
    toymc_params.binning = OneDimBinning(name='stuff', bin_edges=np.linspace(
        toymc_params.nbackground_low, toymc_params.nbackground_high, NBINS))
    # Statistical factor for the MC
    toymc_params.stats_factor = args.stats_factor
    toymc_params.infinite_stats = 10000


    metrics_to_test = ['llh', 'mcllh_eff','mod_chi2',
                       'mcllh_mean', 'generalized_poisson_llh']

    results = collections.OrderedDict()
    results['toymc_params'] = toymc_params
    for metric in metrics_to_test:
        results[metric] = collections.OrderedDict()

    # ==============================================================
    #
    # Generate a toy data set
    #
    data_as_mapset = create_pseudo_data(toymc_params=toymc_params, seed=None)

    # ===============================================================
    #
    # Generate MC template using a pisa pipeline
    #
    # We first need to override the parameter values contained in the config file
    # before instantiating the pipeline
    mc_template = create_mc_template(
        toymc_params, config_file=STANDARD_CONFIG, seed=None)
    mc_map = mc_template.get_outputs(return_sum=True)

    # =================================================================
    #
    # Produce a pseudo-infinite MC statistics template
    # Create a MC set with 10000 times more stats than data. will be used as the truth
    #
    infinite_toymc_params = copy.deepcopy(toymc_params)
    infinite_toymc_params.stats_factor = toymc_params.infinite_stats
    mc_template_pseudo_infinite = create_mc_template(
        infinite_toymc_params, config_file=STANDARD_CONFIG, seed=None)
    mc_map_pseudo_infinite = mc_template_pseudo_infinite.get_outputs(
        return_sum=True)

    # =================================================================
    #
    # Plot the three graphs
    #
    plot_data_and_mc(data_map=data_as_mapset.maps[0],
                     mc_map=mc_map.maps[0],
                     mc_params=mc_template.params,
                     mc_map_pseudo_infinite=mc_map_pseudo_infinite.maps[0],
                     mc_params_pseudo_infinite=mc_template_pseudo_infinite.params,
                     toymc_params=toymc_params,
                     toymc_params_pseudo_infinite=infinite_toymc_params,
                     interactive=args.interactive,
                     output_pdf=output_pdf)

    # ==================================================================
    #
    # Perform llh scans
    #
    if args.make_llh_scan:

        results = run_llh_scans(metrics=metrics_to_test, mc_template=mc_template,
                                data_mapset=data_as_mapset, results=results)
        plot_llh_scans(metrics=metrics_to_test, results=results,
                       interactive=args.interactive, output_pdf=output_pdf)

    # ==================================================================
    #
    # Perform bias and coverage test
    #
    if args.run_coverage_test:

        results = run_coverage_test(n_trials=args.ntrials,
                                    toymc_params=toymc_params,
                                    mc_template=mc_template,
                                    mc_infinite_stats=mc_template_pseudo_infinite,
                                    metrics=metrics_to_test,
                                    results=results,
                                    output_stem=args.output,
                                    )

        plot_coverage_test(output_pdf=output_pdf,
                           metrics=metrics_to_test,
                           results=results,
                           stats_factor=toymc_params.stats_factor,
                           output_stem=args.output,
                           n_trials=args.ntrials)

    output_pdf.close()
