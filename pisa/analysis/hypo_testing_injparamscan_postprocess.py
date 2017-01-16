#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This script/module computes significances, etc. from the logfiles recorded by
the `hypo_testing_injparamscan.py` script. That is, give an Asimov sensitivity 
curve as a function of whichever injected parameter was scanned over.


"""

from __future__ import division

from argparse import ArgumentParser
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import numpy as np
import re

from scipy.special import erfc, erfcinv

from pisa.analysis.hypo_testing import Labels
from pisa.analysis.hypo_testing_postprocess import parse_pint_string
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
    

def extract_trials(logdir, fluctuate_fid, fluctuate_data=False):
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
    toy_names = []
    scan_variables = []
    for folder in logdir_content:
        if '.pckl' not in folder and 'Plots' not in folder:
            toy_names.append(folder.split('toy')[1].split('_')[1])
            scan_variables.append(folder.split('toy')[1].split('_')[2])
    toy_names = np.array(toy_names)
    scan_variables = np.array(scan_variables)
    # Require all to be the same injected truth model
    if not np.alltrue(toy_names == toy_names[0]):
        raise ValueError('Not all output is for the same injected truth '
                         'hypothesis. Got %s'%set(toy_names))
    # Require all to be scanning the same variable
    if not np.alltrue(scan_variables == scan_variables[0]):
        raise ValueError('Not all output is for the same scanned parameter. '
                         'Got %s'%set(scan_variables))
    
    if 'data_sets.pckl' in logdir_content:
        logging.info('Found files I assume to be from a previous run of this '
                     'processing script. If this is incorrect please delete '
                     'the files: data_sets.pckl, all_params.pckl and '
                     'labels.pckl from the logdir you have provided.')
        all_data = from_file(os.path.join(logdir, 'data_sets.pckl'))
        all_params = from_file(os.path.join(logdir, 'all_params.pckl'))
        all_labels = from_file(os.path.join(logdir, 'labels.pckl'))

    else:

        all_labels = {}
        all_params = {}
        all_data = {}
        for outputdir in logdir_content:
            outputdir = os.path.join(logdir,outputdir)
            outputdir_content = os.listdir(outputdir)
            if 'config_summary.json' in outputdir_content:
                config_summary_fpath = os.path.join(
                    outputdir,
                    'config_summary.json'
                )
                cfg = from_file(config_summary_fpath)

                data_is_data = cfg['data_is_data']
                if data_is_data:
                    raise ValueError('Analysis should NOT have been performed '
                                     'on data since this script should only '
                                     'process output from MC studies.')

                # Get naming scheme
                labels = Labels(
                    h0_name=cfg['h0_name'], h1_name=cfg['h1_name'],
                    data_name=cfg['data_name'], data_is_data=data_is_data,
                    fluctuate_data=fluctuate_data, fluctuate_fid=fluctuate_fid
                )
                injparam = '%s_%s'%(labels.dict['data_name'].split('_')[-2],
                                    labels.dict['data_name'].split('_')[-1])
                all_labels[injparam] = labels

                # Get injected parameters
                these_params = {}
                these_params['h0_params'] = {}
                these_params['h1_params'] = {}
                parse_string = ('(.*)=(.*); prior=(.*),'
                                ' range=(.*), is_fixed=(.*),'
                                ' is_discrete=(.*); help="(.*)"')
                for param_string in cfg['h0_params']:
                    bits = re.match(parse_string, param_string, re.M|re.I)
                    if bits.group(5) == 'False':
                        these_params['h0_params'][bits.group(1)] = {}
                        these_params['h0_params'][bits.group(1)]['value'] \
                            = bits.group(2)
                        these_params['h0_params'][bits.group(1)]['prior'] \
                            = bits.group(3)
                        these_params['h0_params'][bits.group(1)]['range'] \
                            = bits.group(4)
                for param_string in cfg['h1_params']:
                    bits = re.match(parse_string, param_string, re.M|re.I)
                    if bits.group(5) == 'False':
                        these_params['h1_params'][bits.group(1)] = {}
                        these_params['h1_params'][bits.group(1)]['value'] \
                            = bits.group(2)
                        these_params['h1_params'][bits.group(1)]['prior'] \
                            = bits.group(3)
                        these_params['h1_params'][bits.group(1)]['range'] \
                            = bits.group(4)

                all_params[injparam] = these_params

                # Find all relevant data dirs, and from each extract the
                # fiducial fit(s) information contained
                this_data = OrderedDict()
                for basename in nsort(os.listdir(outputdir)):
                    m = labels.subdir_re.match(basename)
                    if m is None:
                        continue

                    if fluctuate_data:
                        data_ind = int(m.groupdict()['data_ind'])
                        dset_label = data_ind
                    else:
                        dset_label = labels.data_prefix
                        if not labels.data_name in [None, '']:
                            dset_label += '_' + labels.data_name
                        if not labels.data_suffix in [None, '']:
                            dset_label += '_' + labels.data_suffix

                    lvl2_fits = OrderedDict()
                    lvl2_fits['h0_fit_to_data'] = None
                    lvl2_fits['h1_fit_to_data'] = None
                    
                    subdir = os.path.join(outputdir, basename)
                    for fnum, fname in enumerate(nsort(os.listdir(subdir))):
                        fpath = os.path.join(subdir, fname)
                        for x in ['0', '1']:
                            k = 'h{x}_fit_to_data'.format(x=x)
                            if fname == labels.dict[k]:
                                lvl2_fits[k] = extract_fit(fpath, 'metric_val')
                                break
                            # Also extract fiducial fits if needed
                            if 'toy' in dset_label:
                                ftest = ('hypo_%s_fit_to_%s.json'
                                         %(labels.dict['h{x}_name'.format(x=x)],
                                           dset_label))
                                if fname == ftest:
                                    k = 'h{x}_fit_to_{y}'.format(x=x,y=dset_label)
                                    lvl2_fits[k] = extract_fit(
                                        fpath,
                                        ['metric_val', 'params']
                                    )
                                    break
                            k = 'h{x}_fit_to_{y}'.format(x=x, y=dset_label)
                            for y in ['0','1']:
                                k = 'h{x}_fit_to_h{y}_fid'.format(x=x, y=y)
                                r = labels.dict[k + '_re']
                                m = r.match(fname)
                                if m is None:
                                    continue
                                if fluctuate_fid:
                                    fid_label = int(m.groupdict()['fid_ind'])
                                else:
                                    fid_label = labels.fid
                                if k not in lvl2_fits:
                                    lvl2_fits[k] = OrderedDict()
                                lvl2_fits[k][fid_label] = extract_fit(
                                    fpath,
                                    ['metric', 'metric_val','params']
                                )
                                break
                    this_data[dset_label] = lvl2_fits
                    this_data[dset_label]['params'] = extract_fit(
                        fpath,
                        ['params']
                    )['params']
                all_data[injparam] = this_data
        to_file(all_data, os.path.join(logdir, 'data_sets.pckl'))
        to_file(all_params, os.path.join(logdir, 'all_params.pckl'))
        to_file(all_labels, os.path.join(logdir, 'labels.pckl'))
        
    return all_data, all_params, all_labels


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


def extract_asimov_data(data_sets, labels):
    '''
    Takes the data sets returned by the extract_trials function and extracts 
    the data needed for the Asimov analysis.
    '''
    WO_to_TO_metrics = []
    TO_to_WO_metrics = []
    WO_to_TO_params = {}
    TO_to_WO_params = {}

    for injparam in sorted(data_sets.keys()):
        injlabels = labels[injparam].dict
        for injkey in data_sets[injparam].keys():
            h0_metric_val = data_sets[injparam][injkey][
                'h0_fit_to_toy_%s_asimov'
                %(injlabels['data_name'])]['metric_val']
            h1_metric_val = data_sets[injparam][injkey][
                'h1_fit_to_toy_%s_asimov'
                %(injlabels['data_name'])]['metric_val']
            if h1_metric_val > h0_metric_val:
                bestfit = 'h0'
                altfit = 'h1'
            else:
                bestfit = 'h1'
                altfit = 'h0'

            WO_to_TO_fit = data_sets[injparam][injkey][
                '%s_fit_to_%s_fid'%(altfit, bestfit)]['fid_asimov']
            TO_to_WO_fit = data_sets[injparam][injkey][
                '%s_fit_to_%s_fid'%(bestfit, altfit)]['fid_asimov']

            WO_to_TO_metrics.append(WO_to_TO_fit['metric_val'])
            TO_to_WO_metrics.append(TO_to_WO_fit['metric_val'])

            for systkey in WO_to_TO_fit['params'].keys():
                if systkey not in WO_to_TO_params.keys():
                    WO_to_TO_params[systkey] = []
                WO_to_TO_params[systkey].append(
                    WO_to_TO_fit['params'][systkey]
                )
            for systkey in TO_to_WO_fit['params'].keys():
                if systkey not in TO_to_WO_params.keys():
                    TO_to_WO_params[systkey] = []
                TO_to_WO_params[systkey].append(
                    TO_to_WO_fit['params'][systkey]
                )

    return WO_to_TO_metrics, TO_to_WO_metrics, WO_to_TO_params, TO_to_WO_params


def calculate_deltachi2_signifiances(WO_to_TO_metrics, TO_to_WO_metrics):
    '''
    Takes the true and wrong ordering fit metrics and combines them in to the 
    Asimov significance.
    '''
    significances = []
    num = WO_to_TO_metrics + TO_to_WO_metrics
    denom = 2 * np.sqrt(WO_to_TO_metrics)
    significances = num/denom
    return significances


def plot_significances(WO_to_TO_metrics, TO_to_WO_metrics, inj_param_vals,
                       inj_param_name, labels, detector, selection, outdir):
    '''
    Takes the two sets of metrics relevant to the Asimov-based analysis and 
    makes a plot of the significance as a function of the injected parameter.
    '''
    outdir = os.path.join(outdir, 'Significances')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)

    MainTitle = '%s %s Event Selection Asimov Analysis Significances'%(
        detector, selection)

    significances = calculate_deltachi2_signifiances(
        WO_to_TO_metrics=WO_to_TO_metrics,
        TO_to_WO_metrics=TO_to_WO_metrics
    )
    injlabels = labels['%s_%.4f'%(inj_param_name,inj_param_vals[0])].dict
    truth = injlabels['data_name'].split('_')[0]
    plt.plot(
        inj_param_vals,
        significances,
        linewidth=2,
        marker='o',
        color='r',
        label='True %s'%truth
    )
    yrange = max(significances)-min(significances)
    plt.ylim(min(significances)-0.1*yrange,max(significances)+0.1*yrange)
    plt.xlabel(tex_axis_label(inj_param_name))
    plt.ylabel(r'Significance ($\sigma$)')
    plt.title(MainTitle,fontsize=16)
    plt.legend(loc='best')
    SaveName = "true_%s_%s_%s_%s_asimov_significances.png"%(
        truth,
        detector,
        selection,
        inj_param_name
    )
    plt.savefig(os.path.join(outdir,SaveName))
    plt.close()


def plot_fit(inj_param_vals, inj_param_name, fit_param,
             TO_to_WO_param_vals, TO_to_WO_label,
             WO_to_TO_param_vals, WO_to_TO_label):
    '''
    This is the function which does the actual plotting of the best fit results.
    The rest just use it and change when the plot is saved.
    '''
    plt.plot(
        inj_param_vals,
        TO_to_WO_param_vals,
        linewidth=2,
        marker='o',
        color='darkviolet',
        label=TO_to_WO_label
    )
    plt.plot(
        inj_param_vals,
        WO_to_TO_param_vals,
        linewidth=2,
        marker='o',
        color='deepskyblue',
        label=WO_to_TO_label
    )
    ymax = max(max(TO_to_WO_param_vals),max(WO_to_TO_param_vals))
    ymin = min(min(TO_to_WO_param_vals),min(WO_to_TO_param_vals))
    yrange = ymax-ymin
    plt.ylim(ymin-0.1*yrange,ymax+0.1*yrange)
    plt.xlabel(tex_axis_label(inj_param_name))
    if fit_param == 'deltam31':
        plt.ylabel(r'$|$'+tex_axis_label(fit_param)+r'$|$')
    else:
        plt.ylabel(tex_axis_label(fit_param))
    plt.legend(loc='best')


def plot_individual_fits(WO_to_TO_params, TO_to_WO_params, inj_param_vals,
                         inj_param_name, labels, detector, selection, outdir):
    '''
    Takes the two sets of best fit parameters relevant to the Asimov-based 
    analysis and makes plots of them as a function of the injected parameter.
    This will use plot_fit and save each individual plot
    '''
    outdir = os.path.join(outdir, 'IndividualBestFits')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)

    injlabels = labels['%s_%.4f'%(inj_param_name,inj_param_vals[0])].dict
    truth = injlabels['data_name'].split('_')[0]
    h0 = injlabels['h0'].split('hypo_')[-1]
    h1 = injlabels['h1'].split('hypo_')[-1]
    if h0 == truth:
        TO = h0
        WO = h1
    else:
        TO = h1
        WO = h0

    MainTitle = '%s %s Event Selection Asimov Analysis'%(detector, selection)
    SubTitle = 'True %s Best Fit Parameters'%(truth)

    TO_to_WO_label = '%s fit to %s fid'%(TO,WO)
    WO_to_TO_label = '%s fit to %s fid'%(WO,TO)

    for param in WO_to_TO_params.keys():
        WO_to_TO_param_vals = []
        for param_val in WO_to_TO_params[param]:
            val, units = parse_pint_string(
                pint_string=param_val
            )
            WO_to_TO_param_units = units
            if param == 'deltam31':
                WO_to_TO_param_vals.append(np.abs(float(val)))
            else:
                WO_to_TO_param_vals.append(float(val))
        TO_to_WO_param_vals = []
        for param_val in TO_to_WO_params[param]:
            val, units = parse_pint_string(
                pint_string=param_val
            )
            TO_to_WO_param_units = units
            if param == 'deltam31':
                TO_to_WO_param_vals.append(np.abs(float(val)))
            else:
                TO_to_WO_param_vals.append(float(val))
        plot_fit(
            inj_param_vals=inj_param_vals,
            inj_param_name=inj_param_name,
            fit_param=param,
            TO_to_WO_param_vals=TO_to_WO_param_vals,
            TO_to_WO_label=TO_to_WO_label,
            WO_to_TO_param_vals=WO_to_TO_param_vals,
            WO_to_TO_label=WO_to_TO_label
        )
        plt.title(MainTitle+r'\\'+SubTitle,fontsize=16)
        SaveName = "true_%s_%s_%s_%s_%s_best_fit_values.png"%(
            truth,
            detector,
            selection,
            inj_param_name,
            param
        )
        plt.savefig(os.path.join(outdir,SaveName))
        plt.close()


def plot_combined_fits(WO_to_TO_params, TO_to_WO_params, inj_param_vals,
                       inj_param_name, labels, detector, selection, outdir):
    '''
    Takes the two sets of best fit parameters relevant to the Asimov-based 
    analysis and makes plots of them as a function of the injected parameter.
    This will use plot_fit and save once one for each parameter is on the 
    canvas.
    '''
    outdir = os.path.join(outdir, 'CombinedBestFits')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)

    injlabels = labels['%s_%.4f'%(inj_param_name,inj_param_vals[0])].dict
    truth = injlabels['data_name'].split('_')[0]
    h0 = injlabels['h0'].split('hypo_')[-1]
    h1 = injlabels['h1'].split('hypo_')[-1]
    if h0 == truth:
        TO = h0
        WO = h1
    else:
        TO = h1
        WO = h0

    MainTitle = '%s %s Event Selection Asimov Analysis'%(detector, selection)
    SubTitle = 'True %s Best Fit Parameters'%(truth)

    TO_to_WO_label = '%s fit to %s fid'%(TO,WO)
    WO_to_TO_label = '%s fit to %s fid'%(WO,TO)

    
    # Set up multi-plot
    num_rows = get_num_rows(WO_to_TO_params, omit_metric=False)
    plt.figure(figsize=(20,5*num_rows+2))
    subplotnum=1

    for param in WO_to_TO_params.keys():
        WO_to_TO_param_vals = []
        for param_val in WO_to_TO_params[param]:
            val, units = parse_pint_string(
                pint_string=param_val
            )
            WO_to_TO_param_units = units
            if param == 'deltam31':
                WO_to_TO_param_vals.append(np.abs(float(val)))
            else:
                WO_to_TO_param_vals.append(float(val))
        TO_to_WO_param_vals = []
        for param_val in TO_to_WO_params[param]:
            val, units = parse_pint_string(
                pint_string=param_val
            )
            TO_to_WO_param_units = units
            if param == 'deltam31':
                TO_to_WO_param_vals.append(np.abs(float(val)))
            else:
                TO_to_WO_param_vals.append(float(val))
        plt.subplot(num_rows,4,subplotnum)
        plot_fit(
            inj_param_vals=inj_param_vals,
            inj_param_name=inj_param_name,
            fit_param=param,
            TO_to_WO_param_vals=TO_to_WO_param_vals,
            TO_to_WO_label=TO_to_WO_label,
            WO_to_TO_param_vals=WO_to_TO_param_vals,
            WO_to_TO_label=WO_to_TO_label
        )
        subplotnum += 1
    plt.suptitle(MainTitle+r'\\'+SubTitle,fontsize=36)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    SaveName = "true_%s_%s_%s_%s_all_best_fit_values.png"%(
        truth,
        detector,
        selection,
        inj_param_name
    )
    plt.savefig(os.path.join(outdir,SaveName))
    plt.close()
    
    
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '-d', '--dir', required=True,
        metavar='DIR', type=str,
        help="""Directory into which the output of hypo_testing_injparamscan.py 
        was stored."""
    )
    parser.add_argument(
        '--detector',type=str,default='',
        help="""Name of detector to put in histogram titles."""
    )
    parser.add_argument(
        '--selection',type=str,default='',
        help="""Name of selection to put in histogram titles."""
    )
    parser.add_argument(
        '-IF', '--individual_fits', action='store_true', default=False,
        help='''Flag to make plots of all of the best fit parameters separated 
        by the fitted parameter.'''
    )
    parser.add_argument(
        '-CF', '--combined_fits', action='store_true', default=False,
        help='''Flag to make plots of all of the best fit parameters joined
        together.'''
    )
    parser.add_argument(
        '--outdir', metavar='DIR', type=str, required=True,
        help="""Store all output plots to this directory. This will make
        further subdirectories, if needed, to organise the output plots."""
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help="""set verbosity level"""
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
    ifits = init_args_d.pop('individual_fits')
    cfits = init_args_d.pop('combined_fits')
    outdir = init_args_d.pop('outdir')
    
    data_sets, all_params, labels = extract_trials(
        logdir=args.dir,
        fluctuate_fid=False,
        fluctuate_data=False
    )

    inj_params = data_sets.keys()
    inj_param_vals = []
    for inj_param in inj_params:
        inj_param_vals.append(float(inj_param.split('_')[-1]))
    inj_param_name = inj_params[0].split('_%.4f'%inj_param_vals[0])[0]
    inj_param_vals = sorted(inj_param_vals)

    WO_to_TO_metrics, TO_to_WO_metrics, WO_to_TO_params, TO_to_WO_params = \
        extract_asimov_data(data_sets, labels)

    plot_significances(
        WO_to_TO_metrics=np.array(WO_to_TO_metrics),
        TO_to_WO_metrics=np.array(TO_to_WO_metrics),
        inj_param_vals=inj_param_vals,
        inj_param_name=inj_param_name,
        labels=labels, 
        detector=detector,
        selection=selection,
        outdir=outdir
    )

    if cfits:

        plot_combined_fits(
            WO_to_TO_params=WO_to_TO_params,
            TO_to_WO_params=TO_to_WO_params,
            inj_param_vals=inj_param_vals,
            inj_param_name=inj_param_name,
            labels=labels, 
            detector=detector,
            selection=selection,
            outdir=outdir
        )

    if ifits:

        plot_individual_fits(
            WO_to_TO_params=WO_to_TO_params,
            TO_to_WO_params=TO_to_WO_params,
            inj_param_vals=inj_param_vals,
            inj_param_name=inj_param_name,
            labels=labels, 
            detector=detector,
            selection=selection,
            outdir=outdir
        )
                
        
if __name__ == '__main__':
    main()
