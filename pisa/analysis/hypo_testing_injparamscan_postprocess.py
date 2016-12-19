#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This script/module computes significances, etc. from the logfiles recorded by
the `hypo_testing_injparamscan.py` script. That is, give an Asimov sensitivity 
curve as a function of whichever injected parameter was scanned over.

TODO:

1) Once hypo_testing_injparamscan.py actually works, verify that this works.

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
        if '.pckl' not in folder:
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
                h0_injparam = '%s_%s'%(labels.dict['h0_name'].split('_')[-2],
                                       labels.dict['h0_name'].split('_')[-1])
                h1_injparam = '%s_%s'%(labels.dict['h1_name'].split('_')[-2],
                                       labels.dict['h1_name'].split('_')[-1])
                if not h0_injparam == h1_injparam:
                    raise ValueError('The same truth parameter should be '
                                     'injected regardles of hypothesis. Got '
                                     '%s and %s'
                                     %(h0_injparam, h1_injparam))
                all_labels[h0_injparam] = labels

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

                all_params[h0_injparam] = these_params

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
                all_data[h0_injparam] = this_data
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
    Asimov significance according to Blennow et al.

    References:
    ----------

        1. M. Blennow et al, JHEP 03, 028 (2014)
    '''
    significances = []
    num = WO_to_TO_metrics + TO_to_WO_metrics
    denom = np.sqrt(8 * TO_to_WO_metrics)
    alpha = 0.5 * erfc(num/denom)
    significances = np.sqrt(2.0) * erfcinv(alpha)
    return significances
                    

def make_llr_plots(data, fid_data, labels, detector, selection, outdir):
    '''
    Does what you think. Takes the data and makes LLR distributions. These are 
    then saved to the requested outdir within a folder labelled 
    "LLRDistributions".

    TODO:

    1) Currently the p-value is put on the LLR distributions as an annotation.
       This is probably fine, since the significances can just be calculated 
       from this after the fact.

    2) A method of quantifying the uncertainty due to finite statistics in the 
       pseudo-trials should be added. Possible ways are:
        a) Fitting the distributions and then drawing X new instances of Y 
           trials (where X and Y are large) and looking at the spread in the 
           p-values. This would require a good fit to the distributions, which
           is fine if they end up gaussian...
        b) Quantifying the uncertainty on the critical value used to determine 
           the significance. In the case of the median this could be achieved 
           by drawing subsets of the data and creating a distribution of 
           medians. Though, efforts to do this in the past have given results 
           that seemed to underestimate the true uncertainty.
    '''
    outdir = os.path.join(outdir,'LLRDistributions')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)

    h0_fid_metric = fid_data[
        'h0_fit_to_toy_%s_asimov'%labels['data_name']
    ][
        'metric_val'
    ]
    h1_fid_metric = fid_data[
        'h1_fit_to_toy_%s_asimov'%labels['data_name']
    ][
        'metric_val'
    ]

    if h1_fid_metric > h0_fid_metric:
        bestfit = 'h0'
        altfit = 'h1'
        critical_value = h0_fid_metric-h1_fid_metric
    else:
        bestfit = 'h1'
        altfit = 'h0'
        critical_value = h1_fid_metric-h0_fid_metric

    h0_fit_to_h0_fid_metrics = np.array(
        data['h0_fit_to_h0_fid']['metric_val']['vals']
    )
    h1_fit_to_h0_fid_metrics = np.array(
        data['h1_fit_to_h0_fid']['metric_val']['vals']
    )
    h0_fit_to_h1_fid_metrics = np.array(
        data['h0_fit_to_h1_fid']['metric_val']['vals']
    )
    h1_fit_to_h1_fid_metrics = np.array(
        data['h1_fit_to_h1_fid']['metric_val']['vals']
    )

    num_trials = len(h0_fit_to_h0_fid_metrics)
    metric_type = data['h0_fit_to_h0_fid']['metric_val']['type']
    metric_type_pretty = tex_axis_label(metric_type)

    # In the case of likelihood, the maximum metric is the better fit.
    # With chi2 metrics the opposite is true, and so we must multiply
    # everything by -1 in order to apply the same treatment.
    if 'chi2' in metric_type:
        logging.info("Converting chi2 metric to likelihood equivalent.")
        h0_fit_to_h0_fid_metrics *= -1
        h1_fit_to_h0_fid_metrics *= -1
        h0_fit_to_h1_fid_metrics *= -1
        h1_fit_to_h1_fid_metrics *= -1
        critical_value *= -1

    if bestfit == 'h0':
        LLRbest = h0_fit_to_h0_fid_metrics - h1_fit_to_h0_fid_metrics
        LLRalt = h0_fit_to_h1_fid_metrics - h1_fit_to_h1_fid_metrics
    else:
        LLRbest = h1_fit_to_h1_fid_metrics - h0_fit_to_h1_fid_metrics
        LLRalt = h1_fit_to_h0_fid_metrics - h0_fit_to_h0_fid_metrics

    minLLR = min(min(LLRbest), min(LLRalt))
    maxLLR = max(max(LLRbest), max(LLRalt))
    rangeLLR = maxLLR - minLLR
    binning = np.linspace(minLLR - 0.1*rangeLLR,
                          maxLLR + 0.1*rangeLLR,
                          int(num_trials/20))
    binwidth = binning[1]-binning[0]

    LLRbesthist, LLRbestbinedges = np.histogram(LLRbest,bins=binning)
    LLRalthist, LLRaltbinedges = np.histogram(LLRalt,bins=binning)

    LLRhistmax = max(max(LLRbesthist),max(LLRalthist))

    best_median = np.median(LLRbest)
    alt_median = np.median(LLRalt)

    inj_name = labels['data_name']
    best_name = labels['%s_name'%bestfit]
    alt_name = labels['%s_name'%altfit]

    # p value quantified by asking how much of the time the ALT hypothesis ends
    # up more convincing than the fiducial experiment.
    # The p value is simply this as the fraction of the total number of trials.
    crit_p_value = float(np.sum(LLRalt > critical_value))/len(LLRalt)
    # For the case of toy data we also look at the MEDIAN in order to quantify
    # the MEDIAN SENSITIVITY. THAT IS, THE CASE OF A MEDIAN EXPERIMENT.
    med_p_value = float(np.sum(LLRalt > best_median))/len(LLRalt)

    med_plot_labels = []
    med_plot_labels.append((r"Hypo %s median = %.4f"%(best_name,best_median)))
    med_plot_labels.append((r"Hypo %s median = %.4f"%(alt_name,alt_median)))
    med_plot_labels.append(
        (r"%s best fit - $\log\left[\mathcal{L}\left(\mathcal{H}_{%s}\right)/"
         r"\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"
         %(best_name, best_name, alt_name))
    )
    med_plot_labels.append(
        (r"%s best fit - $\log\left[\mathcal{L}\left(\mathcal{H}_{%s}\right)/"
         r"\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"
         %(alt_name, best_name, alt_name))
    )

    crit_plot_labels = []
    crit_plot_labels.append((r"Critical value = %.4f"%(critical_value)))
    crit_plot_labels.append(
        (r"%s best fit - $\log\left[\mathcal{L}\left(\mathcal{H}_{%s}\right)/"
         r"\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"
         %(best_name, best_name, alt_name))
    )
    crit_plot_labels.append(
        (r"%s best fit - $\log\left[\mathcal{L}\left(\mathcal{H}_{%s}\right)/"
         r"\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"
         %(alt_name, best_name, alt_name))
    )
    
    if metric_type == 'llh':
        plot_title = ("%s %s Event Selection "%(detector,selection)\
                      +r"\\"+" LLR Distributions for true %s (%i trials)"
                      %(inj_name,num_trials))
    else:
        plot_title = ("%s %s Event Selection "%(detector,selection)\
                      +r"\\"+" %s \"LLR\" Distributions for "
                      %(metric_type_pretty)\
                      +"true %s (%i trials)"%(inj_name,num_trials))

    # Factor with which to make everything visible
    plot_scaling_factor = 1.55

    plt.hist(LLRbest,bins=binning,color='r',histtype='step')
    plt.hist(LLRalt,bins=binning,color='b',histtype='step')
    plt.xlabel(r'Log-Likelihood Ratio')
    plt.ylabel(r'Number of Trials (per %.2f)'%binwidth)
    plt.ylim(0,plot_scaling_factor*LLRhistmax)
    plt.axvline(
        best_median,
        color='k',
        ymax=float(max(LLRbesthist))/float(plot_scaling_factor*LLRhistmax)
    )
    plt.axvline(
        alt_median,
        color='g',
        ymax=float(max(LLRalthist))/float(plot_scaling_factor*LLRhistmax)
    )
    plt.legend(med_plot_labels, loc='upper left')
    plt.title(plot_title)
    plt.figtext(
        0.15,
        0.60,
        r"p-value = $%.4f$"%(med_p_value),
        color='k',
        size='xx-large'
    )
    filename = 'true_%s_%s_%s_%s_LLRDistribution_median_%i_Trials.png'%(
        inj_name, detector, selection, metric_type, num_trials
    )
    plt.savefig(os.path.join(outdir,filename))
    plt.close()

    plt.hist(LLRbest,bins=binning,color='r',histtype='step')
    plt.hist(LLRalt,bins=binning,color='b',histtype='step')
    plt.xlabel(r'Log-Likelihood Ratio')
    plt.ylabel(r'Number of Trials (per %.2f)'%binwidth)
    plt.ylim(0,plot_scaling_factor*LLRhistmax)
    plt.axvline(
        critical_value,
        color='k',
        ymax=float(max(LLRbesthist))/float(plot_scaling_factor*LLRhistmax)
    )
    plt.legend(crit_plot_labels, loc='upper left')
    plt.title(plot_title)
    plt.figtext(
        0.15,
        0.60,
        r"p-value = $%.4f$"%(crit_p_value),
        color='k',
        size='xx-large'
    )
    filename = 'true_%s_%s_%s_%s_LLRDistribution_critical_%i_Trials.png'%(
        inj_name, detector, selection, metric_type, num_trials
    )
    plt.savefig(os.path.join(outdir,filename))
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
    outdir = init_args_d.pop('outdir')
    
    data_sets, all_params, labels = extract_trials(
        logdir=args.dir,
        fluctuate_fid=False,
        fluctuate_data=False
    )

    WO_to_TO_metrics, TO_to_WO_metrics, WO_to_TO_params, TO_to_WO_params = \
        extract_asimov_data(data_sets, labels)

    print TO_to_WO_params['theta23']

    significances = calculate_deltachi2_signifiances(
        np.array(WO_to_TO_metrics),
        np.array(TO_to_WO_metrics)
    )

    print significances
                
        
if __name__ == '__main__':
    main()
