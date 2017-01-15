#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This script/module plots the ouput of the multiple fits performed to the "data" 
in hypo_testing_multifittests.py

"""

from __future__ import division

from argparse import ArgumentParser
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import numpy as np
import re

from scipy.stats import norm, spearmanr

from pisa.analysis.hypo_testing import Labels
from pisa.analysis.hypo_testing_postprocess import parse_pint_string
from pisa.core.param import Param, ParamSet
from pisa.utils.fileio import from_file, to_file, nsort
from pisa.utils.log import set_verbosity, logging
from pisa.utils.plotter import tex_axis_label


__all__ = ['extract_asimov_fits', 'extract_pseudo_fits',
           'extract_fit', 'parse_args', 'main']


def get_num_rows(data, omit_metric=False):
    '''
    Calculates the number of rows for multiplots based on the number of 
    systematics.
    '''
    if omit_metric:
        num_rows = int((len(data.keys())-1)/4)
    else:
        num_rows = int(len(data.keys())/4)
    if len(data.keys())%4 != 0:
        num_rows += 1
    return num_rows
    

def extract_asimov_fits(logdir, fluctuate_fid, fluctuate_data=False):
    """Extract and aggregate analysis results.

    Parameters
    ----------
    logdir : string
        Path to logging directory where files are stored. This should contain
        e.g. the "config_summary.json" file.

    fluctuate_fid : bool
        Whether the trials you're interested in applied fluctuations to the
        fiducial-fit Asimov distributions. `fluctuate_fid` False is equivalent
        to specifying an Asimov analysis (so long as the metric used was
        chi-squared).

    fluctuate_data : bool
        Whether the trials you're interested in applied fluctuations to the
        (toy) data. This is invalid if actual data was processed.

    Note that a single `logdir` can have different kinds of analyses run and
    results be logged within, so `fluctuate_fid` and `fluctuate_data` allows
    these to be separated from one another.

    """
    logdir = os.path.expanduser(os.path.expandvars(logdir))
    logdir_content = os.listdir(logdir)
    if 'data_sets.pckl' in logdir_content:
        logging.info('Found files I assume to be from a previous run of this '
                     'processing script. If this is incorrect please delete '
                     'the files: data_sets.pckl, minimiser_info.pckl and '
                     'starting_params.pckl from the logdir you have provided.')
        data_sets = from_file(os.path.join(logdir,
                                           'data_sets.pckl'))
        minimiser_info = from_file(os.path.join(logdir,
                                                'minimiser_info.pckl'))
        starting_params = from_file(os.path.join(logdir,
                                                 'starting_params.pckl'))
        labels = from_file(os.path.join(logdir,
                                        'labels.pckl'))
    elif 'config_summary.json' in logdir_content:
        config_summary_fpath = os.path.join(logdir, 'config_summary.json')
        cfg = from_file(config_summary_fpath)

        data_is_data = cfg['data_is_data']
        if data_is_data and fluctuate_data:
            raise ValueError('Analysis was performed on data, so '
                             '`fluctuate_data` is not supported.')

        # Get naming scheme
        labels = Labels(
            h0_name=cfg['h0_name'], h1_name=cfg['h1_name'],
            data_name=cfg['data_name'], data_is_data=data_is_data,
            fluctuate_data=fluctuate_data, fluctuate_fid=fluctuate_fid
        )

        # Find all relevant data dirs, and from each extract the fiducial fit(s)
        # information contained
        data_sets = OrderedDict()
        minimiser_info = OrderedDict()
        starting_params = OrderedDict()
        for basename in nsort(os.listdir(logdir)):
            m = labels.subdir_re.match(basename)
            if m is None:
                continue

            subdir = os.path.join(logdir, basename)
            data_name = 'toy_%s_asimov'%labels.dict['data_name'] 
            data_sets[data_name] = {}
            minimiser_info[data_name] = {}
            starting_params[data_name] = {}
            for fnum, fname in enumerate(nsort(os.listdir(subdir))):
                fit_num = fname.split('_')[-2]
                hypo = fname.split('_')[1]
                if 'hypo_%s'%hypo not in data_sets[data_name]:
                    data_sets[data_name]['hypo_%s'%hypo] = {}
                    minimiser_info[data_name]['hypo_%s'%hypo] = {}
                    starting_params[data_name]['hypo_%s'%hypo] = {}
                fpath = os.path.join(subdir, fname)
                data_sets[data_name]['hypo_%s'%hypo][fit_num] = extract_fit(
                    fpath,
                    ['metric', 'metric_val','params']
                )
                minimiser_info[data_name]['hypo_%s'%hypo][fit_num] = \
                    extract_fit(
                        fpath,
                        ['minimizer_time', 'minimizer_metadata']
                    )
                starting_params[data_name]['hypo_%s'%hypo][fit_num] = \
                    extract_fit(
                        fpath,
                        ['fit_history']
                    )
                starting_params[data_name]['hypo_%s'%hypo][fit_num][
                    'fit_history'] = starting_params[data_name][
                        'hypo_%s'%hypo][fit_num]['fit_history'][0]
        to_file(data_sets, os.path.join(logdir,'data_sets.pckl'))
        to_file(minimiser_info, os.path.join(logdir,'minimiser_info.pckl'))
        to_file(starting_params, os.path.join(logdir,'starting_params.pckl'))
        to_file(labels, os.path.join(logdir, 'labels.pckl'))
    else:
        raise ValueError('config_summary.json cannot be found in the specified'
                         ' logdir. It should have been created as part of the '
                         'output of hypo_testing.py and so this postprocessing'
                         ' cannot be performed.')
    return data_sets, minimiser_info, starting_params, labels


def extract_pseudo_fits(logdir, fluctuate_fid, fluctuate_data=False):
    """Extract and aggregate analysis results.

    Parameters
    ----------
    logdir : string
        Path to logging directory where files are stored. This should contain
        e.g. the "config_summary.json" file.

    fluctuate_fid : bool
        Whether the trials you're interested in applied fluctuations to the
        fiducial-fit Asimov distributions. `fluctuate_fid` False is equivalent
        to specifying an Asimov analysis (so long as the metric used was
        chi-squared).

    fluctuate_data : bool
        Whether the trials you're interested in applied fluctuations to the
        (toy) data. This is invalid if actual data was processed.

    Note that a single `logdir` can have different kinds of analyses run and
    results be logged within, so `fluctuate_fid` and `fluctuate_data` allows
    these to be separated from one another.

    """
    logdir = os.path.expanduser(os.path.expandvars(logdir))
    logdir_content = os.listdir(logdir)
    if 'data_sets.pckl' in logdir_content:
        logging.info('Found files I assume to be from a previous run of this '
                     'processing script. If this is incorrect please delete '
                     'the files: data_sets.pckl, minimiser_info.pckl and '
                     'starting_params.pckl from the logdir you have provided.')
        data_sets = from_file(os.path.join(logdir,
                                           'data_sets.pckl'))
        minimiser_info = from_file(os.path.join(logdir,
                                                'minimiser_info.pckl'))
        starting_params = from_file(os.path.join(logdir,
                                                 'starting_params.pckl'))
        labels = from_file(os.path.join(logdir,
                                        'labels.pckl'))
    elif 'config_summary.json' in logdir_content:
        config_summary_fpath = os.path.join(logdir, 'config_summary.json')
        cfg = from_file(config_summary_fpath)

        data_is_data = cfg['data_is_data']
        if data_is_data and fluctuate_data:
            raise ValueError('Analysis was performed on data, so '
                             '`fluctuate_data` is not supported.')

        # Get naming scheme
        labels = Labels(
            h0_name=cfg['h0_name'], h1_name=cfg['h1_name'],
            data_name=cfg['data_name'], data_is_data=data_is_data,
            fluctuate_data=fluctuate_data, fluctuate_fid=fluctuate_fid
        )

        # Find all relevant data dirs, and from each extract the fiducial fit(s)
        # information contained
        data_sets = OrderedDict()
        minimiser_info = OrderedDict()
        starting_params = OrderedDict()
        for basename in nsort(os.listdir(logdir)):
            m = labels.subdir_re.match(basename)
            if m is None:
                continue

            subdir = os.path.join(logdir, basename)
            data_name = 'toy_%s_asimov'%labels.dict['data_name'] 
            data_sets[data_name] = {}
            minimiser_info[data_name] = {}
            starting_params[data_name] = {}
            for fnum, fname in enumerate(nsort(os.listdir(subdir))):
                pseudo = fname.split('_')[-2]
                fit_num = fname.split('_')[-4]
                hypo = fname.split('_')[1]
                if pseudo not in data_sets[data_name]:
                    data_sets[data_name][pseudo] = {}
                    minimiser_info[data_name][pseudo] = {}
                    starting_params[data_name][pseudo] = {}
                if 'hypo_%s'%hypo not in data_sets[data_name][pseudo]:
                    data_sets[data_name][pseudo]['hypo_%s'%hypo] = {}
                    minimiser_info[data_name][pseudo]['hypo_%s'%hypo] = {}
                    starting_params[data_name][pseudo]['hypo_%s'%hypo] = {}
                fpath = os.path.join(subdir, fname)
                data_sets[data_name][pseudo]['hypo_%s'%hypo][fit_num] = \
                    extract_fit(
                        fpath,
                        ['metric', 'metric_val','params']
                    )
                minimiser_info[data_name][pseudo]['hypo_%s'%hypo][fit_num] = \
                    extract_fit(
                        fpath,
                        ['minimizer_time', 'minimizer_metadata']
                    )
                starting_params[data_name][pseudo]['hypo_%s'%hypo][fit_num] = \
                    extract_fit(
                        fpath,
                        ['fit_history']
                    )
                starting_params[data_name][pseudo]['hypo_%s'%hypo][fit_num][
                    'fit_history'] = starting_params[data_name][pseudo][
                        'hypo_%s'%hypo][fit_num]['fit_history'][0]
        to_file(data_sets, os.path.join(logdir,'data_sets.pckl'))
        to_file(minimiser_info, os.path.join(logdir,'minimiser_info.pckl'))
        to_file(starting_params, os.path.join(logdir,'starting_params.pckl'))
        to_file(labels, os.path.join(logdir, 'labels.pckl'))
    else:
        raise ValueError('config_summary.json cannot be found in the specified'
                         ' logdir. It should have been created as part of the '
                         'output of hypo_testing.py and so this postprocessing'
                         ' cannot be performed.')
    return data_sets, minimiser_info, starting_params, labels


def extract_fit(fpath, keys=None):
    """Extract fit info from a file.

    Parameters
    ----------
    fpath : string
        Path to the file

    keys : None, string, or iterable of strings
        Keys to extract. If None, all keys are extracted.

    """
    try:
        info = from_file(fpath)
    except:
        raise RuntimeError("Cannot read from file located at %s. Something is"
                           " potentially wrong with it. Please check."%fpath)
    if keys is None:
        return info
    if isinstance(keys, basestring):
        keys = [keys]
    for key in info.keys():
        if key not in keys:
            info.pop(key)
    return info


def plot_fit_information(minimiser_info, labels, detector,
                         selection, minimiser, outdir, pseudokey=None):
    '''
    Makes plots of the number of iterations and time taken with the 
    minimiser. This is a good cross-check that the minimiser did not end 
    abruptly since you would see significant pile-up if it did.
    '''
    outdir = os.path.join(outdir,'MinimiserPlots')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)
    MainTitle = '%s %s Event Selection %s Minimiser Information'%(detector,
                                                                  selection,
                                                                  minimiser)
    if pseudokey is not None:
        MainTitle += ' (Trial %s)'%pseudokey
    for fhkey in minimiser_info.keys():
        if minimiser_info[fhkey] is not None:
            hypo = fhkey.split('_')[1]
            minimiser_times = []
            minimiser_iterations = []
            minimiser_funcevals = []
            minimiser_status = []
            for trial in minimiser_info[fhkey].keys():
                bits = minimiser_info[fhkey][trial]['minimizer_time'].split(' ')
                minimiser_times.append(
                    float(bits[0])
                )
                minimiser_iterations.append(
                    int(minimiser_info[fhkey][trial][
                        'minimizer_metadata']['nit'])
                )
                minimiser_funcevals.append(
                    int(minimiser_info[fhkey][trial][
                        'minimizer_metadata']['nfev'])
                )
                minimiser_status.append(
                    int(minimiser_info[fhkey][trial][
                        'minimizer_metadata']['status'])
                )
                minimiser_units = bits[1]
            FitTitle = ("True %s, Hypothesis %s (%i Asimov Fits)"
                        %(labels['data_name'],
                          hypo,
                          len(minimiser_times)))
            plt.hist(minimiser_times, bins=10)
            plt.xlabel('Minimiser Time (seconds)')
            plt.ylabel('Number of Trials')
            plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
            if pseudokey is not None:
                SaveName = (
                    "true_%s_%s_%s_trial_%s_hypo_%s_minimiser_times.png"
                    %(labels['data_name'],
                      detector,
                      selection,
                      pseudokey,
                      hypo))
            else:
                SaveName = ("true_%s_%s_%s_hypo_%s_minimiser_times.png"
                            %(labels['data_name'],
                              detector,
                              selection,
                              hypo))
            plt.savefig(os.path.join(outdir,SaveName))
            plt.close()
            plt.hist(minimiser_iterations, bins=10)
            plt.xlabel('Minimiser Iterations')
            plt.ylabel('Number of Trials')
            plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
            if pseudokey is not None:
                SaveName = (
                    "true_%s_%s_%s_trial_%s_hypo_%s_minimiser_iterations.png"
                    %(labels['data_name'],
                      detector,
                      selection,
                      pseudokey,
                      hypo))
            else:
                SaveName = ("true_%s_%s_%s_hypo_%s_minimiser_iterations.png"
                            %(labels['data_name'],
                              detector,
                              selection,
                              hypo))
            plt.savefig(os.path.join(outdir,SaveName))
            plt.close()
            plt.hist(minimiser_funcevals, bins=10)
            plt.xlabel('Minimiser Function Evaluations')
            plt.ylabel('Number of Trials')
            plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
            if pseudokey is not None:
                SaveName = (
                    "true_%s_%s_%s_trial_%s_hypo_%s_minimiser_funcevals.png"
                    %(labels['data_name'],
                      detector,
                      selection,
                      pseudokey,
                      hypo))
            else:
                SaveName = ("true_%s_%s_%s_hypo_%s_minimiser_funcevals.png"
                            %(labels['data_name'],
                              detector,
                              selection,
                              hypo))
            plt.savefig(os.path.join(outdir,SaveName))
            plt.close()
            plt.hist(minimiser_status, bins=10)
            plt.xlabel('Minimiser Status')
            plt.ylabel('Number of Trials')
            plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
            if pseudokey is not None:
                SaveName = (
                    "true_%s_%s_%s_trial_%s_hypo_%s_minimiser_status.png"
                    %(labels['data_name'],
                      detector,
                      selection,
                      pseudokey,
                      hypo))
            else:
                SaveName = ("true_%s_%s_%s_hypo_%s_minimiser_status.png"
                            %(labels['data_name'],
                              detector,
                              selection,
                              hypo))
            plt.savefig(os.path.join(outdir,SaveName))
            plt.close()


def plot_fit_results(fit_results, labels, detector,
                     selection, minimiser, outdir, pseudokey=None):
    '''Makes histograms of the Asimov fit results. These should have basically 
    no variation if everything was fine.'''
    outdir = os.path.join(outdir,'FitResults')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)
    MainTitle = '%s %s Event Selection Fit Results (%s Minimiser)'%(detector,
                                                                    selection,
                                                                    minimiser)
    if pseudokey is not None:
        MainTitle += ' (Trial %s)'%pseudokey
    for fhkey in fit_results.keys():
        hypo = fhkey.split('_')[1]
        metric = []
        params = {}
        for fit in fit_results[fhkey].keys():
            metric_name = fit_results[fhkey][fit]['metric']
            metric.append(fit_results[fhkey][fit]['metric_val'])
            if len(params.keys()) == 0:
                for param in fit_results[fhkey][fit]['params'].keys():
                    params[param] = {}
                    params[param]['values'] = []
                    val, units = parse_pint_string(
                        pint_string=fit_results[fhkey][fit]['params'][param]
                    )
                    params[param]['units'] = units
            for param in fit_results[fhkey][fit]['params'].keys():
                params[param]['values'].append(
                    float(
                        fit_results[fhkey][fit]['params'][param].split(' ')[0]
                    )
                )
        FitTitle = ("True %s, Hypothesis %s (%i Asimov Fits)"
                    %(labels['data_name'],
                      hypo,
                      len(metric)))
        plt.hist(metric, bins=10)
        plt.xlabel(tex_axis_label(metric_name))
        plt.ylabel('Number of Trials')
        plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
        if pseudokey is not None:
            SaveName = ("true_%s_%s_%s_trial_%s_hypo_%s_%s_vals.png"
                        %(labels['data_name'],
                          detector,
                          selection,
                          pseudokey,
                          hypo,
                          metric_name))
        else:
            SaveName = ("true_%s_%s_%s_hypo_%s_%s_vals.png"
                        %(labels['data_name'],
                          detector,
                          selection,
                          hypo,
                          metric_name))
        plt.savefig(os.path.join(outdir,SaveName))
        plt.close()
        for param in params.keys():
            plt.hist(params[param]['values'], bins=10)
            if not params[param]['units'] == 'dimensionless':
                plt.xlabel(tex_axis_label(param) + ' ' +
                           tex_axis_label(params[param]['units']))
            else:
                plt.xlabel(tex_axis_label(param))
            plt.ylabel('Number of Trials')
            plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
            if pseudokey is not None:
                SaveName = ("true_%s_%s_%s_trial_%s_hypo_%s_%s_vals.png"
                            %(labels['data_name'],
                              detector,
                              selection,
                              pseudokey,
                              hypo,
                              param))
            else:
                SaveName = ("true_%s_%s_%s_hypo_%s_%s_vals.png"
                            %(labels['data_name'],
                              detector,
                              selection,
                              hypo,
                              param))
            plt.savefig(os.path.join(outdir,SaveName))
            plt.close()


def plot_starting_params(starting_params, fit_results, labels,
                         detector, selection, minimiser, outdir,
                         pseudokey=None):
    '''
    Makes histograms of the starting points for the minimiser.
    '''
    outdir = os.path.join(outdir,'StartingParams')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)
    MainTitle = '%s %s Event Selection %s Minimiser Start Params'%(detector,
                                                                   selection,
                                                                   minimiser)
    if pseudokey is not None:
        MainTitle += ' (Trial %s)'%pseudokey
    for fhkey in starting_params.keys():
        hypo = fhkey.split('_')[1]
        params = OrderedDict()
        num_trials = len(starting_params[fhkey].keys())
        for fit in starting_params[fhkey].keys():
            metric_name = fit_results[fhkey][fit]['metric']
            if len(params.keys()) == 0:
                params[metric_name] = {}
                params[metric_name]['values'] = []
                params[metric_name]['units'] = 'dimensionless'
                for param in fit_results[fhkey][fit]['params'].keys():
                    params[param] = {}
                    params[param]['values'] = []
                    val, units = parse_pint_string(
                        pint_string=fit_results[fhkey][fit]['params'][param]
                    )
                    params[param]['units'] = units
            for param,val in zip(params.keys(),
                                 starting_params[fhkey][fit]['fit_history']):
                params[param]['values'].append(float(val))
        FitTitle = ("True %s, Hypothesis %s (%i Asimov Fits)"
                    %(labels['data_name'],
                      hypo,
                      num_trials))
        for param in params.keys():
            plt.hist(params[param]['values'], bins=10)
            if not params[param]['units'] == 'dimensionless':
                plt.xlabel(tex_axis_label(param) + ' ' +
                           tex_axis_label(params[param]['units']))
            else:
                plt.xlabel(tex_axis_label(param))
            plt.ylabel('Number of Trials')
            plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
            if pseudokey is not None:
                SaveName = (
                    "true_%s_%s_%s_trial_%s_hypo_%s_%s_starting_vals.png"
                    %(labels['data_name'],
                      detector,
                      selection,
                      pseudokey,
                      hypo,
                      param)
                )
            else:
                SaveName = ("true_%s_%s_%s_hypo_%s_%s_starting_vals.png"
                            %(labels['data_name'],
                              detector,
                              selection,
                              hypo,
                              param))
            plt.savefig(os.path.join(outdir,SaveName))
            plt.close()

    
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '-d', '--dir', required=True,
        metavar='DIR', type=str,
        help='''Directory into which the results of hypo_testing_asimovtests.py
        was stored'''
    )
    parser.add_argument(
        '--detector',type=str,default='',
        help='''Name of detector to put in histogram titles.'''
    )
    parser.add_argument(
        '--selection',type=str,default='',
        help='''Name of selection to put in histogram titles.'''
    )
    parser.add_argument(
        '--minimiser',type=str,default='',
        help='''Name of minimiser to put in histogram titles.'''
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--data-is-asimov', action='store_true',
        help='''The multiple fits were performed on Asimov data.'''
    )
    group.add_argument(
        '--data-is-pseudo', action='store_true',
        help='''The multiple fits were performed on pseudo-data.'''
    )
    parser.add_argument(
        '--outdir', metavar='DIR', type=str, required=True,
        help="""Store all output plots to this directory. This will make
        further subdirectories, if needed, to organise the output plots."""
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    init_args_d = vars(args)

    # NOTE: Removing extraneous args that won't get passed to instantiate the
    # HypoTesting object via dictionary's `pop()` method.

    set_verbosity(init_args_d.pop('v'))

    detector = init_args_d.pop('detector')
    selection = init_args_d.pop('selection')
    minimiser = init_args_d.pop('minimiser')
    outdir = init_args_d.pop('outdir')

    data_is_asimov = init_args_d.pop('data_is_asimov')
    data_is_pseudo = init_args_d.pop('data_is_pseudo')

    if data_is_asimov:

        data_sets, minimiser_info, starting_params, labels = \
            extract_asimov_fits(
                logdir=args.dir,
                fluctuate_fid=True,
                fluctuate_data=False
            )

        for injkey in data_sets.keys():

            plot_starting_params(
                starting_params=starting_params[injkey],
                fit_results=data_sets[injkey],
                labels=labels.dict,
                detector=detector,
                selection=selection,
                minimiser=minimiser,
                outdir=outdir
            )

            plot_fit_results(
                fit_results=data_sets[injkey],
                labels=labels.dict,
                detector=detector,
                selection=selection,
                minimiser=minimiser,
                outdir=outdir
            )

            plot_fit_information(
                minimiser_info=minimiser_info[injkey],
                labels=labels.dict,
                detector=detector,
                selection=selection,
                minimiser=minimiser,
                outdir=outdir
            )

    elif data_is_pseudo:

        data_sets, minimiser_info, starting_params, labels = \
            extract_pseudo_fits(
                logdir=args.dir,
                fluctuate_fid=True,
                fluctuate_data=False
            )

        for injkey in data_sets.keys():
            for pseudokey in data_sets[injkey].keys():

                plot_starting_params(
                    starting_params=starting_params[injkey][pseudokey],
                    fit_results=data_sets[injkey][pseudokey],
                    labels=labels.dict,
                    detector=detector,
                    selection=selection,
                    minimiser=minimiser,
                    outdir=outdir,
                    pseudokey=pseudokey
                )

                plot_fit_results(
                    fit_results=data_sets[injkey][pseudokey],
                    labels=labels.dict,
                    detector=detector,
                    selection=selection,
                    minimiser=minimiser,
                    outdir=outdir,
                    pseudokey=pseudokey
                )

                plot_fit_information(
                    minimiser_info=minimiser_info[injkey][pseudokey],
                    labels=labels.dict,
                    detector=detector,
                    selection=selection,
                    minimiser=minimiser,
                    outdir=outdir,
                    pseudokey=pseudokey
                )

    else:

        raise ValueError('Data should be either Asimov or pseudo. Though, if '
                         'the argument parser is working correctly you should '
                         'never see this error')
                
        
if __name__ == '__main__':
    main()
