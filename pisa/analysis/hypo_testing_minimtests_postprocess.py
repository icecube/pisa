#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This script/module plots the ouput of the minimiser tests run on pseudo-trials 
by hypo_testing_minimtests.py

TODO:

1) Add units to plots

2) Add prior contribution to likelihood surface. Maybe as separate plots?

"""

from __future__ import division

from argparse import ArgumentParser
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from matplotlib import gridspec
import numpy as np
import re

from scipy.stats import norm, spearmanr

from pisa.analysis.hypo_testing import Labels
from pisa.analysis.hypo_testing_postprocess import parse_pint_string, extract_gaussian
from pisa.core.param import Param, ParamSet
from pisa.utils.fileio import from_file, to_file, nsort
from pisa.utils.log import set_verbosity, logging
from pisa.utils.plotter import tex_axis_label


__all__ = ['extract_trials', 'extract_fit', 'parse_args', 'main']


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
    

def extract_output(logdir, fluctuate_fid, fluctuate_data=False):
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
    if 'best_fits.pckl' in logdir_content:
        logging.info('Found files I assume to be from a previous run of this '
                     'processing script. If this is incorrect please delete '
                     'the files: best_fits.pckl, minimiser_info.pckl and '
                     'scans.pckl from the logdir you have provided.')
        best_fits = from_file(os.path.join(logdir,'best_fits.pckl'))
        minimiser_info = from_file(os.path.join(logdir,'minimiser_info.pckl'))
        scans = from_file(os.path.join(logdir,'scans.pckl'))
        all_params = from_file(os.path.join(logdir, 'all_params.pckl'))
        labels = from_file(os.path.join(logdir, 'labels.pckl'))
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

        all_params = {}
        h0_params_key = '%s_params'%labels.dict['h0_name']
        h1_params_key = '%s_params'%labels.dict['h1_name']
        all_params[h0_params_key] = {}
        all_params[h1_params_key] = {}
        parse_string = ('(.*)=(.*); prior=(.*),'
                        ' range=(.*), is_fixed=(.*),'
                        ' is_discrete=(.*); help="(.*)"')
        for param_string in cfg['h0_params']:
            bits = re.match(parse_string, param_string, re.M|re.I)
            if bits.group(5) == 'False':
                all_params[h0_params_key][bits.group(1)] = {}
                all_params[h0_params_key][bits.group(1)]['value'] \
                    = bits.group(2)
                all_params[h0_params_key][bits.group(1)]['prior'] \
                    = bits.group(3)
                all_params[h0_params_key][bits.group(1)]['range'] \
                    = bits.group(4)
        for param_string in cfg['h1_params']:
            bits = re.match(parse_string, param_string, re.M|re.I)
            if bits.group(5) == 'False':
                all_params[h1_params_key][bits.group(1)] = {}
                all_params[h1_params_key][bits.group(1)]['value'] \
                    = bits.group(2)
                all_params[h1_params_key][bits.group(1)]['prior'] \
                    = bits.group(3)
                all_params[h1_params_key][bits.group(1)]['range'] \
                    = bits.group(4)

        # Find all relevant data dirs, and from each extract the fiducial fit(s)
        # information contained
        best_fits = OrderedDict()
        minimiser_info = OrderedDict()
        scans = OrderedDict()
        trials_dir_name = 'toy_%s_asimov'%labels.dict['data_name']
        trials_dir = os.path.join(logdir, trials_dir_name)
        best_fits[trials_dir_name] = OrderedDict()
        minimiser_info[trials_dir_name] = OrderedDict()
        scans[trials_dir_name] = OrderedDict()
        fit_params = None
        for trials_file in os.listdir(trials_dir):
            if trials_dir_name not in trials_file:
                trial_num = trials_file.split('.json')[0].split('_')[-1]
                if trial_num not in best_fits[trials_dir_name].keys():
                    best_fits[trials_dir_name][trial_num] = OrderedDict()
                    minimiser_info[trials_dir_name][trial_num] = OrderedDict()
                    scans[trials_dir_name][trial_num] = OrderedDict()
                trial_type = trials_file.split('_pseudodata')[0]
                fpath = os.path.join(trials_dir, trials_file)
                best_fits[trials_dir_name][trial_num][trial_type] \
                    = extract_fit(
                        fpath,
                        ['metric', 'metric_val', 'params']
                    )
                if fit_params == None:
                    fit_params = best_fits[trials_dir_name][trial_num][
                        trial_type]['params'].keys()
                minimiser_info[trials_dir_name][trial_num][trial_type] \
                    = extract_fit(
                        fpath,
                        ['minimizer_time', 'minimizer_metadata']
                    )
        scans_dir_name = 'scans'
        scans_dir = os.path.join(logdir, scans_dir_name)
        for scans_file in os.listdir(scans_dir):
            trial_num = scans_file.split('.json')[0].split('_')[-1]
            if trial_num in scans[trials_dir_name].keys():
                fid_hypo = scans_file.split('_')[0]
                fit_hypo = scans_file.split('_')[2]
                trial_type = 'hypo_%s_fit_to_hypo_%s_fid'%(
                    labels.dict['%s_name'%fit_hypo],
                    labels.dict['%s_name'%fid_hypo]
                )
                if trial_type in best_fits[trials_dir_name][trial_num].keys():
                    if trial_type not in scans[trials_dir_name][
                            trial_num].keys():
                        scans[trials_dir_name][trial_num][trial_type] \
                            = OrderedDict() 
                    param_name = scans_file.split('_scan')[0].split('hypo_')[-1]
                    if param_name in fit_params:
                        fpath = os.path.join(scans_dir, scans_file)
                        scans[trials_dir_name][trial_num][trial_type][
                            param_name] = extract_fit(
                                fpath,
                                ['results']
                            )
                    else:
                        raise ValueError('Found a scanned parameter (%s) that '
                                         'is not in those varied in the '
                                         'psuedo-trials (%s)'%(
                                             param_name, fit_params))
                else:
                    raise ValueError('Found a fiducial/fit hypothesis (%s)'
                                     'combination in the scans that does not '
                                     'match those in the pseudo-trials (%s)'%(
                                         trial_type,
                                         best_fits[trials_dir_name][
                                             trial_num].keys()))
            else:
                raise ValueError('Found a scanned trial (%s) that does not '
                                 ' match a pseudo-trial'%trial_num)
        to_file(best_fits, os.path.join(logdir,'best_fits.pckl'))
        to_file(minimiser_info, os.path.join(logdir,'minimiser_info.pckl'))
        to_file(scans, os.path.join(logdir,'scans.pckl'))
        to_file(labels, os.path.join(logdir, 'labels.pckl'))
        to_file(all_params, os.path.join(logdir, 'all_params.pckl'))
    else:
        raise ValueError('config_summary.json cannot be found in the specified'
                         ' logdir. It should have been created as part of the '
                         'output of hypo_testing.py and so this postprocessing'
                         ' cannot be performed.')
    return best_fits, minimiser_info, scans, all_params, labels


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


def extract_scan_data(results, scan_param_name):
    '''Extracts the scan data from the data parsed from the output files in to 
    a format which is easy for plotting'''
    data = {}
    data['metric_vals'] = []
    for result in results:
        if result['metric_val'] == result['detailed_metric_info'][
                result['metric']]['maps']['total']:
            map_metric = result['detailed_metric_info'][
                result['metric']]['maps']['total']
            prior_contribution = sum(result['detailed_metric_info'][
                result['metric']]['priors'])
            data['metric_vals'].append(map_metric+prior_contribution)
        else:
            data['metric_vals'].append(result['metric_val'])
        for param_key in result['params'].keys():
            if param_key == scan_param_name:
                if param_key not in data.keys():
                    data[param_key] = {}
                    data[param_key]['vals'] = []
                    data[param_key]['units'] = \
                        result['params'][param_key]['prior']['units']
                data[param_key]['vals'].append(
                    result['params'][param_key]['value'][0]
                )
    return data


def plot_individual_scan(data, scan_param, hypo, all_params, metric_name,
                         best_fit_param_val, subplotnum=None, zoom=True):
    '''
    This function will use matplotlib to make a graph of the 1D likelihood 
    space given in data. This will also plot the best fit value and create axis
    labels and legends. The optional subplotnum argument can be given in the 
    combined case so that the y-axis labels only get put on when appropriate.
    '''
    # Get prior information to add to plot
    # TODO - Deal with non-gaussian priors
    wanted_params = all_params['%s_params'%hypo]
    if zoom:
        gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
        plt.subplot(gs[0])
    plt.plot(
        data[scan_param]['vals'],
        data['metric_vals'],
        linewidth=2
    )
    units = all_params['no_params'][scan_param]['value'].split(
        all_params['no_params'][scan_param]['value'].split(' ')[0]+' '
    )[-1]
    if subplotnum is not None:
        if (subplotnum-1)%4 == 0:
            plt.ylabel('Delta '+tex_axis_label(metric_name))
    else:
        plt.ylabel('Delta '+tex_axis_label(metric_name))
    plt.ylim(min(data['metric_vals']),plt.gca().get_ylim()[1])
    for param in wanted_params.keys():
        if param == scan_param:
            if 'gaussian' in wanted_params[param]['prior']:
                stddev, maximum = extract_gaussian(
                    prior_string = wanted_params[param]['prior'],
                    units = units
                )
                currentxlim = plt.xlim()
                if zoom:
                    if (stddev < 1e-2) and (stddev != 0.0):
                        priorlabel = (r'Gaussian Prior '
                                      '($%.3e\pm%.3e$)'%(maximum,stddev))
                    else:
                        priorlabel = (r'Gaussian Prior '
                                      '($%.3g\pm%.3g$)'%(maximum,stddev))
                else:
                    priorlabel = 'Gauss Prior'
                x = np.linspace(
                    min(data[scan_param]['vals']),
                    max(data[scan_param]['vals']),
                    100
                )
                priorsurface = ((x-maximum)/stddev)**2
                plt.plot(
                    x,
                    priorsurface,
                    linewidth=2,
                    linestyle='--',
                    color='r',
                    label=priorlabel
                )
            else:
                priorsurface = None
    if zoom:
        plt.axvline(
            best_fit_param_val,
            linestyle='--',
            color='g',
            linewidth=2,
            label = tex_axis_label(scan_param)+' Best Fit Value = %s'%(
                best_fit_param_val)
        )
    else:
        plt.axvline(
            best_fit_param_val,
            linestyle='--',
            color='g',
            linewidth=2,
            label = 'Best Fit'
        )
    plt.legend(
        loc='upper left',
        fontsize=16
    )
    if zoom:
        plt.subplot(gs[1])
        plt.plot(
            data[scan_param]['vals'],
            data['metric_vals'],
            linewidth=2
        )
        yrange = max(data['metric_vals'])-min(data['metric_vals'])
        plt.ylim(-0.01*yrange,0.01*yrange)
        systname = tex_axis_label(scan_param)
        if not units == 'dimensionless':
            systname += r' (%s)'%tex_axis_label(units)
        plt.xlabel(systname)
        if subplotnum is not None:
            if (subplotnum-1)%4 == 0:
                plt.ylabel('Delta '+tex_axis_label(metric_name))
        else:
            plt.ylabel('Delta '+tex_axis_label(metric_name))
        if priorsurface is not None:
            plt.plot(
                x,
                priorsurface,
                linewidth=2,
                linestyle='--',
                color='r',
                label=priorlabel
            )
        plt.axvline(
            best_fit_param_val,
            linestyle='--',
            color='g',
            linewidth=2
        )
        plt.axhline(
            0.0,
            linestyle='--',
            color='k',
            linewidth=1
        )
        plt.subplot(gs[0])
    else:
        systname = tex_axis_label(scan_param)
        if not units == 'dimensionless':
            systname += r' (%s)'%tex_axis_label(units)
        plt.xlabel(systname)
        if subplotnum is not None:
            if (subplotnum-1)%4 == 0:
                plt.ylabel('Delta '+tex_axis_label(metric_name))
        else:
            plt.ylabel('Delta '+tex_axis_label(metric_name))


def plot_individual_scans(scan_data, best_fits, all_params, labels,
                          detector, selection, outdir):
    '''
    Plots the 1D likelihood scans around the best fit to the pseudo trial. 
    This will show the best fit to that trial and so one can see if the true 
    minimum was found.
    '''
    outdir = os.path.join(outdir,'Scans')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)
    MainTitle = '%s %s Event Selection Best Fit 1D Scan'%(detector,selection)
    
    for trial_num in scan_data.keys():
        for fitfidhypo in scan_data[trial_num].keys():
            for scan_param in scan_data[trial_num][fitfidhypo].keys():
                results = scan_data[trial_num][fitfidhypo][
                    scan_param]['results']
                metric_name = results[0]['metric']
                data = extract_scan_data(
                    results=results,
                    scan_param_name=scan_param
                )
                best_fit_metric_val = best_fits[trial_num][fitfidhypo]['metric_val']
                best_fit_param_val, best_fit_param_units = parse_pint_string(
                    pint_string=best_fits[trial_num][fitfidhypo][
                        'params'][scan_param]
                )
                FitTitle = ("True %s, Fiducial Fit %s, Hypothesis %s (Trial %s)"
                            %(labels['data_name'],
                              fitfidhypo.split('_fid')[0].split('hypo_')[-1],
                              fitfidhypo.split('_fit')[0].split('hypo_')[-1],
                              trial_num))
                data['metric_vals'] = np.array(data['metric_vals'])
                data['metric_vals'] -= best_fit_metric_val
                if metric_name == 'llh':
                    data['metric_vals'] *= -1
                plot_individual_scan(
                    data=data,
                    scan_param=scan_param,
                    hypo=fitfidhypo.split('_fit')[0].split('hypo_')[-1],
                    all_params=all_params,
                    metric_name=metric_name,
                    best_fit_param_val=best_fit_param_val
                )
                plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
                savename = '%s_%s_1D_%s_trial_%s_scan.png'%(fitfidhypo,
                                                            scan_param,
                                                            metric_name,
                                                            trial_num)
                plt.savefig(os.path.join(outdir,savename))
                plt.close()


def plot_combined_scans(scan_data, best_fits, all_params, labels,
                        detector, selection, outdir):
    '''
    Plots the 1D likelihood scans around the best fit to the pseudo trial. 
    This will show the best fit to that trial and so one can see if the true 
    minimum was found. Will save once all of the scans have been plotted for
    a given combination of h0 and h1 on the same canvas.
    '''
    outdir = os.path.join(outdir,'CombinedScans')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)

    MainTitle = '%s %s Event Selection Best Fit 1D Scans'%(detector,selection)
    for trial_num in scan_data.keys():
        for fitfidhypo in scan_data[trial_num].keys():
            
            # Set up multi-plot
            num_rows = get_num_rows(scan_data[trial_num][fitfidhypo],
                                    omit_metric=False)
            plt.figure(figsize=(20,5*num_rows+2))
            subplotnum=1
            
            for scan_param in scan_data[trial_num][fitfidhypo].keys():
                
                results = scan_data[trial_num][fitfidhypo][
                    scan_param]['results']
                metric_name = results[0]['metric']
                data = extract_scan_data(
                    results=results,
                    scan_param_name=scan_param
                )
                best_fit_metric_val = best_fits[trial_num][fitfidhypo]['metric_val']
                best_fit_param_val, best_fit_param_units = parse_pint_string(
                    pint_string=best_fits[trial_num][fitfidhypo][
                        'params'][scan_param]
                )
                FitTitle = ("True %s, Fiducial Fit %s, Hypothesis %s (Trial %s)"
                            %(labels['data_name'],
                              fitfidhypo.split('_fid')[0].split('hypo_')[-1],
                              fitfidhypo.split('_fit')[0].split('hypo_')[-1],
                              trial_num))
                data['metric_vals'] = np.array(data['metric_vals'])
                data['metric_vals'] -= best_fit_metric_val
                if metric_name == 'llh':
                    data['metric_vals'] *= -1
                plt.subplot(num_rows,4,subplotnum)
                plot_individual_scan(
                    data=data,
                    scan_param=scan_param,
                    hypo=fitfidhypo.split('_fit')[0].split('hypo_')[-1],
                    all_params=all_params,
                    metric_name=metric_name,
                    best_fit_param_val=best_fit_param_val,
                    subplotnum=subplotnum,
                    zoom=False
                )
                subplotnum+=1
            plt.suptitle(MainTitle+r'\\'+FitTitle, fontsize=36)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            savename = '%s_1D_%s_trial_%s_scans.png'%(fitfidhypo,
                                                      metric_name,
                                                      trial_num)
            plt.savefig(os.path.join(outdir,savename))
            plt.close()

    
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '-d', '--dir', required=True,
        metavar='DIR', type=str,
        help='''Directory which contains the output of 
        hypo_testing_minimtests.py.'''
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
        '--outdir', metavar='DIR', type=str, required=True,
        help='''Store all output plots to this directory. This will make
        further subdirectories, if needed, to organise the output plots.'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='''set verbosity level'''
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
    outdir = init_args_d.pop('outdir')

    best_fits, minimiser_info, scans, all_params, labels = extract_output(
        logdir=args.dir,
        fluctuate_fid=True,
        fluctuate_data=False
    )

    for injkey in best_fits.keys():

        plot_individual_scans(
            scan_data=scans[injkey],
            best_fits=best_fits[injkey],
            all_params=all_params,
            labels=labels.dict,
            detector=detector,
            selection=selection,
            outdir=outdir
        )

        plot_combined_scans(
            scan_data=scans[injkey],
            best_fits=best_fits[injkey],
            all_params=all_params,
            labels=labels.dict,
            detector=detector,
            selection=selection,
            outdir=outdir
        )
                
        
if __name__ == '__main__':
    main()
