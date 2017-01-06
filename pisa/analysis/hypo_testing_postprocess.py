#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This script/module computes significances, etc. from the logfiles recorded by
the `hypo_testing.py` script.

TODO:

1) Some of the "combined" plots currently make it impossible to read the axis 
   labels. Come up with a better way of doing this. Could involve making 
   legends and just labelling the axes alphabetically.

2) The important one - Figure out if this script generalises to the case of 
   analysing data. My gut says it doesn't...

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


def extract_injval(injparams, systkey, data_label, hypo_label, injlabel):
    '''
    Extracts the injected value and modifies it based on the 
    hypothesis/fiducial fit being considered. The label associated with this 
    is then modified accordingly.
    '''
    if systkey == 'deltam31':
        if hypo_label == data_label:
            injval = float(injparams[systkey].split(' ')[0])
        else:
            injval = -1*float(injparams[systkey].split(' ')[0])
            injlabel += r' ($\times-1$)'
            
    else:
        injval = float(injparams[systkey].split(' ')[0])

    if (injval < 1e-2) and (injval != 0.0):
        injlabel += ' = %.3e'%injval
    else:    
        injlabel += ' = %.3g'%injval

    return injval, injlabel


def extract_gaussian(prior_string, units):
    '''
    Parses the string for the Gaussian priors that comes from the config 
    summary file in the logdir. This should account for dimensions though is 
    only tested with degrees.
    '''
    if units == 'dimensionless':
        parse_string = ('gaussian prior: stddev=(.*)'
                        ' , maximum at (.*)')
        bits = re.match(
            parse_string,
            prior_string,
            re.M|re.I
        )
        stddev = float(bits.group(1))
        maximum = float(bits.group(2))
    else:
        parse_string = ('gaussian prior: stddev=(.*) (.*)'
                        ', maximum at (.*) (.*)')
        bits = re.match(
            parse_string,
            prior_string,
            re.M|re.I
        )
        stddev = float(bits.group(1))
        maximum = float(bits.group(3))

    return stddev, maximum
    

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
    if 'data_sets.pckl' in logdir_content:
        logging.info('Found files I assume to be from a previous run of this '
                     'processing script. If this is incorrect please delete '
                     'the files: data_sets.pckl, all_params.pckl and '
                     'labels.pckl from the logdir you have provided.')
        data_sets = from_file(os.path.join(logdir, 'data_sets.pckl'))
        all_params = from_file(os.path.join(logdir, 'all_params.pckl'))
        labels = from_file(os.path.join(logdir, 'labels.pckl'))
        minimiser_info = from_file(os.path.join(logdir, 'minimiser_info.pckl'))
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
        all_params['h0_params'] = {}
        all_params['h1_params'] = {}
        parse_string = ('(.*)=(.*); prior=(.*),'
                        ' range=(.*), is_fixed=(.*),'
                        ' is_discrete=(.*); help="(.*)"')
        for param_string in cfg['h0_params']:
            bits = re.match(parse_string, param_string, re.M|re.I)
            if bits.group(5) == 'False':
                all_params['h0_params'][bits.group(1)] = {}
                all_params['h0_params'][bits.group(1)]['value'] = bits.group(2)
                all_params['h0_params'][bits.group(1)]['prior'] = bits.group(3)
                all_params['h0_params'][bits.group(1)]['range'] = bits.group(4)
        for param_string in cfg['h1_params']:
            bits = re.match(parse_string, param_string, re.M|re.I)
            if bits.group(5) == 'False':
                all_params['h1_params'][bits.group(1)] = {}
                all_params['h1_params'][bits.group(1)]['value'] = bits.group(2)
                all_params['h1_params'][bits.group(1)]['prior'] = bits.group(3)
                all_params['h1_params'][bits.group(1)]['range'] = bits.group(4)

        # Find all relevant data dirs, and from each extract the fiducial fit(s)
        # information contained
        data_sets = OrderedDict()
        minimiser_info = OrderedDict()
        for basename in nsort(os.listdir(logdir)):
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
            minim_info = OrderedDict()
            minim_info['h0_fit_to_data'] = None
            minim_info['h1_fit_to_data'] = None

            # Account for failed jobs. Get the set of file numbers that exist
            # for all h0 an h1 combinations
            file_nums = OrderedDict()
            subdir = os.path.join(logdir, basename)
            for fnum, fname in enumerate(nsort(os.listdir(subdir))):
                fpath = os.path.join(subdir, fname)
                for x in ['0', '1']:
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
                        if k not in file_nums:
                            file_nums[k] = []
                        file_nums[k].append(fid_label)
                        break

            set_file_nums = []
            for hypokey in file_nums.keys():
                if len(set_file_nums) == 0:
                    set_file_nums = set(file_nums[hypokey])
                else:
                    set_file_nums = set_file_nums.intersection(file_nums[hypokey])

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
                            minim_info[k] = OrderedDict()
                        if fid_label in set_file_nums:
                            lvl2_fits[k][fid_label] = extract_fit(
                                fpath,
                                ['metric', 'metric_val','params']
                            )
                            minim_info[k][fid_label] = extract_fit(
                                fpath,
                                ['minimizer_metadata', 'minimizer_time']
                            )
                        break
            data_sets[dset_label] = lvl2_fits
            minimiser_info[dset_label] = minim_info
            data_sets[dset_label]['params'] = extract_fit(
                fpath,
                ['params']
            )['params']
        to_file(data_sets, os.path.join(logdir, 'data_sets.pckl'))
        to_file(all_params, os.path.join(logdir, 'all_params.pckl'))
        to_file(labels, os.path.join(logdir, 'labels.pckl'))
        to_file(minimiser_info, os.path.join(logdir, 'minimiser_info.pckl'))
    else:
        raise ValueError('config_summary.json cannot be found in the specified'
                         ' logdir. It should have been created as part of the '
                         'output of hypo_testing.py and so this postprocessing'
                         ' cannot be performed.')
    return data_sets, all_params, labels, minimiser_info


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


def extract_fid_data(data_sets):
    '''
    Takes the data sets returned by the extract_trials function and extracts 
    the data on the fiducial fits.

    TODO (?) - This works in the case of all MC, but I don't know about data.
    '''
    fid_values = {}
    for injkey in data_sets.keys():
        fid_values[injkey] = {}
        for datakey in data_sets[injkey]:
            if ('toy' in datakey) or ('data' in datakey):
                fid_values[injkey][datakey] \
                    = data_sets[injkey].pop(datakey)
    return fid_values


def extract_data(data):
    '''
    Takes the data sets returned by the extract_trials function and turns 
    them in to a format used by all of the plotting functions.
    '''
    values = {}
    for injkey in data.keys():
        values[injkey] = {}
        alldata = data[injkey]
        paramkeys = alldata['params'].keys()
        for datakey in alldata.keys():
            if not datakey == 'params':
                values[injkey][datakey] = {}
                values[injkey][datakey]['metric_val'] = {}
                values[injkey][datakey]['metric_val']['vals'] = []
                for paramkey in paramkeys:
                    values[injkey][datakey][paramkey] = {}
                    values[injkey][datakey][paramkey]['vals'] = []
                trials = alldata[datakey]
                for trial_num in trials.keys():
                    trial = trials[trial_num]
                    values[injkey][datakey]['metric_val']['vals'] \
                        .append(trial['metric_val'])
                    values[injkey][datakey]['metric_val']['type'] \
                        = trial['metric']
                    values[injkey][datakey]['metric_val']['units'] \
                        = 'dimensionless'
                    param_vals = trial['params']
                    for param_name in param_vals.keys():
                        val, units = parse_pint_string(
                            pint_string=param_vals[param_name]
                        )
                        values[injkey][datakey][param_name]['vals'] \
                            .append(float(val))
                        values[injkey][datakey][param_name]['units'] \
                            = units
    return values


def parse_pint_string(pint_string):
    '''
    Will return the value and units from a string with attached pint-style 
    units. i.e. the string "0.97 dimensionless" would return a value of 0.97 
    and units of dimensionless. Both will return as strings.
    '''
    val = pint_string.split(' ')[0]
    units = pint_string.split(val+' ')[-1]
    return val, units


def purge_failed_jobs(data, trial_nums, thresh=5.0):
    '''
    This function looks at the values of the metric and finds any it deems to 
    be from a failed job. That is, the value of the metric fell very far 
    outside of the rest of the values.

    References:
    ----------
        Taken from stack overflow:
        
            http://stackoverflow.com/questions/22354094/pythonic-way-\
            of-detecting-outliers-in-one-dimensional-observation-data

        which references:

            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect 
            and Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 

    Interestingly, I only saw a need for this with my true NO jobs, where I 
    attempted to run some jobs in fp32 mode. No jobs were needed to be removed 
    for true IO, where everything was run in fp64 mode.
    '''
    
    for fit_key in data.keys():
        points = np.array(data[fit_key]['metric_val']['vals'])
        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        modified_z_score = 0.6745 * diff / med_abs_deviation
        good_trials = modified_z_score < thresh
        if not np.all(good_trials):
            bad_trials = np.where(good_trials==False)[0]
            if len(bad_trials) == 1:
                logging.warning(
                    "Outlier detected for %s in trial %s. This will be "
                    "removed. If you think this should not happen, please "
                    "change the value of the threshold used for this decision "
                    "(currently set to %.2f)."
                    %(fit_key,trial_nums[bad_trials],thresh)
                )
            else:
                logging.warning(
                    "Outlier detected for %s in trials %s. These will be "
                    "removed. If you think this should not happen, please "
                    "change the value of the threshold used for this decision "
                    "(currently set to %.2f)."
                    %(fit_key,trial_nums[bad_trials],thresh)
                )
            for fitkey in data.keys():
                for param in data[fitkey].keys():
                    new_vals = np.delete(
                        np.array(data[fitkey][param]['vals']),
                        bad_trials
                    )
                    data[fitkey][param]['vals'] = new_vals


def plot_fit_information(minimiser_info, labels, detector, selection, outdir):
    '''Makes plots of the number of iterations and time taken with the 
    minimiser. This is a good cross-check that the minimiser did not end 
    abruptly since you would see significant pile-up if it did.'''
    outdir = os.path.join(outdir,'MinimiserPlots')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)
    MainTitle = '%s %s Event Selection Minimiser Information'%(detector,
                                                               selection)
    for fhkey in minimiser_info.keys():
        if minimiser_info[fhkey] is not None:
            hypo = fhkey.split('_')[0]
            fid = fhkey.split('_')[-2]
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
            FitTitle = ("True %s, Fiducial Fit %s, Hypothesis %s (%i Trials)"
                        %(labels['data_name'],
                          labels['%s_name'%fid],
                          labels['%s_name'%hypo],
                          len(minimiser_times)))
            plt.hist(minimiser_times, bins=10)
            plt.xlabel('Minimiser Time (seconds)')
            plt.ylabel('Number of Trials')
            plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
            SaveName = ("true_%s_%s_%s_fid_%s_hypo_%s_minimiser_times.png"
                        %(labels['data_name'],
                          detector,
                          selection,
                          labels['%s_name'%fid],
                          labels['%s_name'%hypo]))
            plt.savefig(os.path.join(outdir,SaveName))
            plt.close()
            plt.hist(minimiser_iterations, bins=10)
            plt.xlabel('Minimiser Iterations')
            plt.ylabel('Number of Trials')
            plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
            SaveName = ("true_%s_%s_%s_fid_%s_hypo_%s_minimiser_iterations.png"
                        %(labels['data_name'],
                          detector,
                          selection,
                          labels['%s_name'%fid],
                          labels['%s_name'%hypo]))
            plt.savefig(os.path.join(outdir,SaveName))
            plt.close()
            plt.hist(minimiser_funcevals, bins=10)
            plt.xlabel('Minimiser Function Evaluations')
            plt.ylabel('Number of Trials')
            plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
            SaveName = ("true_%s_%s_%s_fid_%s_hypo_%s_minimiser_funcevals.png"
                        %(labels['data_name'],
                          detector,
                          selection,
                          labels['%s_name'%fid],
                          labels['%s_name'%hypo]))
            plt.savefig(os.path.join(outdir,SaveName))
            plt.close()
            plt.hist(minimiser_status, bins=10)
            plt.xlabel('Minimiser Status')
            plt.ylabel('Number of Trials')
            plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
            SaveName = ("true_%s_%s_%s_fid_%s_hypo_%s_minimiser_status.png"
                        %(labels['data_name'],
                          detector,
                          selection,
                          labels['%s_name'%fid],
                          labels['%s_name'%hypo]))
            plt.savefig(os.path.join(outdir,SaveName))
            plt.close()
                    

def make_llr_plots(data, fid_data, labels, detector, selection, outdir):
    '''
    Does what you think. Takes the data and makes LLR distributions. These are 
    then saved to the requested outdir within a folder labelled 
    "LLRDistributions".

    TODO:

    1) Currently the p-value is put on the LLR distributions as an annotation.
       This is probably fine, since the significances can just be calculated 
       from this after the fact.

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
                          int(num_trials/40))
    binwidth = binning[1]-binning[0]
    bincens = np.linspace(binning[0]+binwidth/2.0,
                          binning[-1]-binwidth/2.0,
                          len(binning)-1)

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
    misid_c_trials = float(np.sum(LLRalt > critical_value))
    crit_p_value = misid_c_trials/num_trials
    unc_crit_p_value = np.sqrt(misid_c_trials*(1-crit_p_value))/num_trials
    # For the case of toy data we also look at the MEDIAN in order to quantify
    # the MEDIAN SENSITIVITY. THAT IS, THE CASE OF A MEDIAN EXPERIMENT.
    misid_m_trials = float(np.sum(LLRalt > best_median))
    med_p_value = misid_m_trials/num_trials
    # Quantify the uncertainty on the median by bootstrapping
    sampled_medians = []
    for i in range(0,1000):
        sampled_medians.append(
            np.median(
                np.random.choice(
                    LLRbest,
                    size=len(LLRbest),
                    replace=True
                )
            )
        )
    sampled_medians = np.array(sampled_medians)
    median_error = np.std(sampled_medians)/np.sqrt(num_trials)
    # Add relative errors in quadrature
    wdenom = misid_m_trials+median_error*median_error
    wterm = wdenom/(misid_m_trials*misid_m_trials)
    Nterm = 1.0/num_trials
    unc_med_p_value = med_p_value * np.sqrt(wterm + Nterm)

    med_plot_labels = []
    med_plot_labels.append((r"Hypo %s median = $%.4f\pm%.4f$"%(best_name,best_median,median_error)))
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

    # In case of median plot, draw both best and alt histograms
    plt.hist(LLRbest,bins=binning,color='r',histtype='step',lw=2)
    plt.hist(LLRalt,bins=binning,color='b',histtype='step',lw=2)
    plt.xlabel(r'Log-Likelihood Ratio')
    plt.ylabel(r'Number of Trials (per %.2f)'%binwidth)
    # Nicely scale the plot
    plt.ylim(0,plot_scaling_factor*LLRhistmax)
    # Add labels to show which side means what...
    xlim = plt.gca().get_xlim()
    plt.text(
        xlim[0]-0.05*(xlim[1]-xlim[0]),
        -0.11*plot_scaling_factor*LLRhistmax,
        r"$\leftarrow$ \\ prefers %s"%(alt_name),
        color='k',
        size='large'
    )
    plt.text(
        xlim[1]-0.05*(xlim[1]-xlim[0]),
        -0.11*plot_scaling_factor*LLRhistmax,
        r"$\rightarrow$ \\ prefers %s"%(best_name),
        color='k',
        size='large'
    )
    # Add the best hist median, since it is the "critical" value so that it
    # goes to the height of the best histogram
    plt.axvline(
        best_median,
        color='k',
        ymax=float(max(LLRbesthist))/float(plot_scaling_factor*LLRhistmax),
        lw=2
    )
    # Create an object so that a hatch can be drawn over the region of
    # interest to the p-value.
    finehist = np.repeat(LLRalthist,100)
    finebinning = np.linspace(binning[0],binning[-1],(len(binning)-1)*100+1)
    finebinwidth = finebinning[1]-finebinning[0]
    finebincens = np.linspace(finebinning[0]+finebinwidth/2.0,
                              finebinning[-1]-finebinwidth/2.0,
                              len(finebinning)-1)
    # Draw the hatch. This is between the x-axis (0) and the finehist object
    # made above. The "where" tells is to only draw above the critical value.
    # To make it just the hatch, color is set to none and hatch is set to X.
    # Also, so that it doesn't have a border we set linewidth to zero.
    plt.fill_between(
        finebincens,
        0,
        finehist,
        where=(finebincens>best_median),
        color='none',
        hatch='X',
        edgecolor="k",
        lw=0
    )
    plt.legend(med_plot_labels, loc='upper left')
    plt.title(plot_title)
    # Write the p-value on the plot
    plt.figtext(
        0.15,
        0.65,
        r"p-value = $%.4f\pm%.4f$"%(med_p_value,unc_med_p_value),
        color='k',
        size='xx-large'
    )
    filename = 'true_%s_%s_%s_%s_LLRDistribution_median_%i_Trials.png'%(
        inj_name, detector, selection, metric_type, num_trials
    )
    plt.savefig(os.path.join(outdir,filename))
    plt.close()

    # In case of critical plot, draw just alt histograms
    plt.hist(LLRalt,bins=binning,color='b',histtype='step',lw=2)
    plt.xlabel(r'Log-Likelihood Ratio')
    plt.ylabel(r'Number of Trials (per %.2f)'%binwidth)
    # Nicely scale the plot
    plt.ylim(0,plot_scaling_factor*LLRhistmax)
    # Add labels to show which side means what...
    xlim = plt.gca().get_xlim()
    plt.text(
        xlim[0]-0.05*(xlim[1]-xlim[0]),
        -0.11*plot_scaling_factor*LLRhistmax,
        r"$\leftarrow$ \\ prefers %s"%(alt_name),
        color='k',
        size='large'
    )
    plt.text(
        xlim[1]-0.05*(xlim[1]-xlim[0]),
        -0.11*plot_scaling_factor*LLRhistmax,
        r"$\rightarrow$ \\ prefers %s"%(best_name),
        color='k',
        size='large'
    )
    # Draw the critical value line on the plot so that it goes just
    # above the histogram
    plt.axvline(
        critical_value,
        color='k',
        ymax=float((max(LLRalthist)*1.1)/(plot_scaling_factor*LLRhistmax)),
        lw=2
    )
    # Create an object so that a hatch can be drawn over the region of
    # interest to the p-value.
    finehist = np.repeat(LLRalthist,100)
    finebinning = np.linspace(binning[0],binning[-1],(len(binning)-1)*100+1)
    finebinwidth = finebinning[1]-finebinning[0]
    finebincens = np.linspace(finebinning[0]+finebinwidth/2.0,
                              finebinning[-1]-finebinwidth/2.0,
                              len(finebinning)-1)
    # Draw the hatch. This is between the x-axis (0) and the finehist object
    # made above. The "where" tells is to only draw above the critical value.
    # To make it just the hatch, color is set to none and hatch is set to X.
    # Also, so that it doesn't have a border we set linewidth to zero.
    plt.fill_between(
        finebincens,
        0,
        finehist,
        where=(finebincens>critical_value),
        color='none',
        hatch='X',
        edgecolor="k",
        lw=0
    )
    plt.legend(crit_plot_labels, loc='upper left')
    plt.title(plot_title)
    # Write the p-value on the plot
    plt.figtext(
        0.15,
        0.70,
        r"p-value = $%.4f\pm%.4f$"%(crit_p_value,unc_crit_p_value),
        color='k',
        size='xx-large'
    )
    filename = 'true_%s_%s_%s_%s_LLRDistribution_critical_%i_Trials.png'%(
        inj_name, detector, selection, metric_type, num_trials
    )
    plt.savefig(os.path.join(outdir,filename))
    plt.close()


def plot_individual_posterior(data, injparams, altparams, all_params, labels,
                              injlabel, altlabel, systkey, fhkey,
                              subplotnum=None):
    '''
    This function will use matplotlib to make a histogram of the vals contained
    in data. The injected value will be plotted along with, where appropriate,
    the "wrong hypothesis" fiducial fit and the prior. The axis labels and the
    legend are taken care of in here. The optional subplotnum argument can be
    given in the combined case so that the y-axis label only get put on when 
    appropriate.
    '''

    if systkey == 'metric_val':
        metric_type = data['type']
    systvals = np.array(data['vals'])
    units = data['units']

    hypo = fhkey.split('_')[0]
    fid = fhkey.split('_')[-2]
                
    plt.hist(systvals, bins=10)

    # Add injected and alternate fit lines
    if not systkey == 'metric_val':
        injval, injlabelproper = extract_injval(
            injparams = injparams,
            systkey = systkey,
            data_label = labels['data_name'],
            hypo_label = labels['%s_name'%hypo],
            injlabel = injlabel
        )
        plt.axvline(
            injval,
            color='r',
            linewidth=2,
            label=injlabelproper
        )
        if not labels['%s_name'%fid] == labels['data_name']:
            altval, altlabelproper = extract_injval(
                injparams = altparams,
                systkey = systkey,
                data_label = labels['%s_name'%fid],
                hypo_label = labels['%s_name'%hypo],
                injlabel = altlabel
            )
            plt.axvline(
                altval,
                color='g',
                linewidth=2,
                label=altlabelproper
            )

    # Add shaded region for prior, if appropriate
    # TODO - Deal with non-gaussian priors
    wanted_params = all_params['%s_params'%hypo]
    for param in wanted_params.keys():
        if param == systkey:
            if 'gaussian' in wanted_params[param]['prior']:
                stddev, maximum = extract_gaussian(
                    prior_string = wanted_params[param]['prior'],
                    units = units
                )
                currentxlim = plt.xlim()
                if (stddev < 1e-2) and (stddev != 0.0):
                    priorlabel = (r'Gaussian Prior '
                                  '($%.3e\pm%.3e$)'%(maximum,stddev))
                else:
                    priorlabel = (r'Gaussian Prior '
                                  '($%.3g\pm%.3g$)'%(maximum,stddev))
                plt.axvspan(
                    maximum-stddev,
                    maximum+stddev,
                    color='k',
                    label=priorlabel,
                    ymax=0.1,
                    alpha=0.5
                )
                # Reset xlimits if prior makes it go far off
                if plt.xlim()[0] < currentxlim[0]:
                    plt.xlim(currentxlim[0],plt.xlim()[1])
                if plt.xlim()[1] > currentxlim[1]:
                    plt.xlim(plt.xlim()[0],currentxlim[1])

    # Make axis labels look nice
    if systkey == 'metric_val':
        systname = tex_axis_label(metric_type)
    else:
        systname = tex_axis_label(systkey)
    if not units == 'dimensionless':
        systname += r' (%s)'%tex_axis_label(units)
                
    plt.xlabel(systname)
    if subplotnum is not None:
        if (subplotnum-1)%4 == 0:
            plt.ylabel(r'Number of Trials')
    else:
        plt.ylabel(r'Number of Trials')
    plt.ylim(0,1.35*plt.ylim()[1])
    if not systkey == 'metric_val':
        plt.legend(loc='upper left')
    

def plot_individual_posteriors(data, fid_data, labels, all_params, detector,
                               selection, outdir):
    '''
    This function will make use of plot_individual_posterior and 
    save every time.
    '''

    outdir = os.path.join(outdir,'IndividualPosteriors')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)

    MainTitle = '%s %s Event Selection Posterior'%(detector, selection)

    if labels['data_name'] == labels['h0_name']:
        inj = 'h0'
        alt = 'h1'
    else:
        inj = 'h1'
        alt = 'h0'
    injparams = fid_data[
        ('%s_fit_to_toy_%s_asimov'
         %(inj,labels['data_name']))
    ]['params']
    altparams = fid_data[
        ('%s_fit_to_toy_%s_asimov'
         %(alt,labels['data_name']))
    ]['params']
    injlabel = 'Injected Value'
    altlabel = 'Alternate Fit'

    for fhkey in data.keys():
        for systkey in data[fhkey].keys():

            hypo = fhkey.split('_')[0]
            fid = fhkey.split('_')[-2]
            FitTitle = ("True %s, Fiducial Fit %s, Hypothesis %s (%i Trials)"
                        %(labels['data_name'],
                          labels['%s_name'%fid],
                          labels['%s_name'%hypo],
                          len(data[fhkey][systkey]['vals'])))

            plot_individual_posterior(
                data = data[fhkey][systkey],
                injparams = injparams,
                altparams = altparams,
                all_params = all_params,
                labels = labels,
                injlabel = injlabel,
                altlabel = altlabel,
                systkey = systkey,
                fhkey = fhkey
            )

            plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
            SaveName = ("true_%s_%s_%s_fid_%s_hypo_%s_%s_posterior.png"
                        %(labels['data_name'],
                          detector,
                          selection,
                          labels['%s_name'%fid],
                          labels['%s_name'%hypo],
                          systkey))
            plt.savefig(os.path.join(outdir,SaveName))
            plt.close()


def plot_combined_posteriors(data, fid_data, labels, all_params,
                             detector, selection, outdir):
    '''
    This function will make use of plot_individual_posterior but just save
    once all of the posteriors for a given combination of h0 and h1 have
    been plotted on the same canvas.
    '''

    outdir = os.path.join(outdir,'CombinedPosteriors')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)

    MainTitle = '%s %s Event Selection Posteriors'%(detector, selection)

    labels['MainTitle'] = MainTitle

    if labels['data_name'] == labels['h0_name']:
        inj = 'h0'
        alt = 'h1'
    else:
        inj = 'h1'
        alt = 'h0'
    injparams = fid_data[
        ('%s_fit_to_toy_%s_asimov'
         %(inj,labels['data_name']))
    ]['params']
    altparams = fid_data[
        ('%s_fit_to_toy_%s_asimov'
         %(alt,labels['data_name']))
    ]['params']
    injlabel = 'Injected Value'
    altlabel = 'Alternate Fit'

    for fhkey in data.keys():
        
        # Set up multi-plot
        num_rows = get_num_rows(data[fhkey], omit_metric=False)
        plt.figure(figsize=(20,5*num_rows+2))
        subplotnum=1
        
        for systkey in data[fhkey].keys():

            hypo = fhkey.split('_')[0]
            fid = fhkey.split('_')[-2]
            FitTitle = ("True %s, Fiducial Fit %s, Hypothesis %s (%i Trials)"
                        %(labels['data_name'],
                          labels['%s_name'%fid],
                          labels['%s_name'%hypo],
                          len(data[fhkey][systkey]['vals'])))

            plt.subplot(num_rows,4,subplotnum)

            plot_individual_posterior(
                data = data[fhkey][systkey],
                injparams = injparams,
                altparams = altparams,
                all_params = all_params,
                labels = labels,
                injlabel = injlabel,
                altlabel = altlabel,
                systkey = systkey,
                fhkey = fhkey,
                subplotnum = subplotnum
            )

            subplotnum += 1

        plt.suptitle(MainTitle+r'\\'+FitTitle, fontsize=36)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        SaveName = ("true_%s_%s_%s_fid_%s_hypo_%s_posteriors.png"
                    %(labels['data_name'],
                      detector,
                      selection,
                      labels['%s_name'%fid],
                      labels['%s_name'%hypo]))
        plt.savefig(os.path.join(outdir,SaveName))
        plt.close()


def plot_individual_scatter(xdata, ydata, labels, xsystkey, ysystkey,
                            subplotnum=None, num_rows=None, plot_cor=True):
    '''
    This function will use matplotlib to make a scatter plot of the vals
    contained in xdata and ydata. The correlation will be calculated and
    the plot will be annotated with this. Axis labels are done in here too. The 
    optional subplotnum argument can be given in the combined case so that the 
    y-axis label only get put on when appropriate.
    '''

    # Extract data and units
    xvals = np.array(xdata['vals'])
    xunits = xdata['units']
    yvals = np.array(ydata['vals'])
    yunits = ydata['units']

    # Make scatter plot
    plt.scatter(xvals, yvals)

    if plot_cor:
        # Calculate correlation and annotate
        if len(set(xvals)) == 1:
            logging.warn(("Parameter %s appears to not have been varied. i.e. all "
                          "of the values in the set are the same. This will "
                          "lead to NaN in the correlation calculation and so it "
                          "will not be done."%xsystkey))
        if len(set(yvals)) == 1:
            logging.warn(("Parameter %s appears to not have been varied. i.e. all "
                          "of the values in the set are the same. This will "
                          "lead to NaN in the correlation calculation and so it "
                          "will not be done."%ysystkey))
        if (len(set(xvals)) != 1) and (len(set(yvals)) != 1):
            rho, pval = spearmanr(xvals, yvals)
            if subplotnum is not None:
                row = int((subplotnum-1)/4)
                xtext = 0.25*0.25+((subplotnum-1)%4)*0.25
                ytext = 0.88-(1.0/num_rows)*0.9*row
                plt.figtext(
                    xtext,
                    ytext,
                    'Correlation = %.2f'%rho,
                    fontsize='large'
                )
            else:
                plt.figtext(
                    0.15,
                    0.85,
                    'Correlation = %.2f'%rho,
                    fontsize='large'
                )

    # Make plot range easy to look at
    Xrange = xvals.max() - xvals.min()
    Yrange = yvals.max() - yvals.min()
    if Xrange != 0.0:
        plt.xlim(xvals.min()-0.1*Xrange,
                 xvals.max()+0.1*Xrange)
    if Yrange != 0.0:
        plt.ylim(yvals.min()-0.1*Yrange,
                 yvals.max()+0.3*Yrange)
    
    # Make axis labels look nice
    xsystname = tex_axis_label(xsystkey)
    if not xunits == 'dimensionless':
        xsystname += r' (%s)'%tex_axis_label(xunits)
    ysystname = tex_axis_label(ysystkey)
    if not yunits == 'dimensionless':
        ysystname += r' (%s)'%tex_axis_label(yunits)

    plt.xlabel(xsystname)
    plt.ylabel(ysystname)
    
    
def plot_individual_scatters(data, labels, detector, selection, outdir):
    '''
    This function will make use of plot_individual_scatter and save every time.
    '''

    outdir = os.path.join(outdir,'IndividualScatterPlots')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)

    MainTitle = '%s %s Event Selection Correlation Plot'%(detector, selection)

    for fhkey in data.keys():
        for xsystkey in data[fhkey].keys():
            if not xsystkey == 'metric_val':
                for ysystkey in data[fhkey].keys():
                    if (ysystkey != 'metric_val') and (ysystkey != xsystkey):

                        hypo = fhkey.split('_')[0]
                        fid = fhkey.split('_')[-2]
                        FitTitle = ("True %s, Fiducial Fit %s, Hypothesis %s "
                                    "(%i Trials)"
                                    %(labels['data_name'],
                                      labels['%s_name'%fid],
                                      labels['%s_name'%hypo],
                                      len(data[fhkey][xsystkey]['vals'])))

                        plot_individual_scatter(
                            xdata = data[fhkey][xsystkey],
                            ydata = data[fhkey][ysystkey],
                            labels = labels,
                            xsystkey = xsystkey,
                            ysystkey = ysystkey
                        )

                        plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
                        SaveName = (("true_%s_%s_%s_fid_%s_hypo_%s_%s_%s"
                                     "_scatter_plot.png"
                                     %(labels['data_name'],
                                      detector,
                                      selection,
                                      labels['%s_name'%fid],
                                      labels['%s_name'%hypo],
                                      xsystkey,
                                      ysystkey)))
                        plt.savefig(os.path.join(outdir,SaveName))
                        plt.close()


def plot_combined_individual_scatters(data, labels, detector,
                                      selection, outdir):
    '''
    This function will make use of plot_individual_scatter and save once all of 
    the scatter plots for a single systematic with every other systematic have 
    been plotted on the same canvas for each h0 and h1 combination.
    '''

    outdir = os.path.join(outdir,'CombinedScatterPlots')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)

    MainTitle = '%s %s Event Selection Correlation Plot'%(detector, selection)

    for fhkey in data.keys():
        for xsystkey in data[fhkey].keys():
            if not xsystkey == 'metric_val':
                
                # Set up multi-plot
                num_rows = get_num_rows(data[fhkey], omit_metric=True)
                plt.figure(figsize=(20,5*num_rows+2))
                subplotnum=1
                
                for ysystkey in data[fhkey].keys():
                    if (ysystkey != 'metric_val') and (ysystkey != xsystkey):

                        hypo = fhkey.split('_')[0]
                        fid = fhkey.split('_')[-2]
                        FitTitle = ("True %s, Fiducial Fit %s, Hypothesis %s "
                                    "(%i Trials)"
                                    %(labels['data_name'],
                                      labels['%s_name'%fid],
                                      labels['%s_name'%hypo],
                                      len(data[fhkey][xsystkey]['vals'])))

                        plt.subplot(num_rows,4,subplotnum)

                        plot_individual_scatter(
                            xdata = data[fhkey][xsystkey],
                            ydata = data[fhkey][ysystkey],
                            labels = labels,
                            xsystkey = xsystkey,
                            ysystkey = ysystkey,
                            subplotnum = subplotnum,
                            num_rows = num_rows
                        )

                        subplotnum += 1

                plt.suptitle(MainTitle+r'\\'+FitTitle, fontsize=36)
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)
                SaveName = (("true_%s_%s_%s_fid_%s_hypo_%s_%s"
                             "_scatter_plot.png"
                             %(labels['data_name'],
                               detector,
                               selection,
                               labels['%s_name'%fid],
                               labels['%s_name'%hypo],
                               xsystkey)))
                plt.savefig(os.path.join(outdir,SaveName))
                plt.close()


def plot_combined_scatters(data, labels, detector, selection, outdir):
    '''
    This function will make use of plot_individual_scatter and save once every 
    scatter plot has been plotted on a single canvas for each of the h0 and h1 
    combinations.
    '''

    outdir = os.path.join(outdir,'CombinedScatterPlots')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)

    MainTitle = '%s %s Event Selection Correlation Plot'%(detector, selection)

    for fhkey in data.keys():
        # Systematic number is one less than number of keys since this also
        # contains the metric_val entry
        SystNum = len(data[fhkey].keys())-1
        # Set up multi-plot
        plt.figure(figsize=(3.5*(SystNum-1),3.5*(SystNum-1)))
        subplotnum=(SystNum-1)*(SystNum-1)+1
        # Set up container to know which correlations have already been plotted
        PlottedSysts = []
        for xsystkey in data[fhkey].keys():
            if not xsystkey == 'metric_val':
                PlottedSysts.append(xsystkey)
                for ysystkey in data[fhkey].keys():
                    if (ysystkey != 'metric_val') and (ysystkey != xsystkey):
                        subplotnum -= 1
                        if ysystkey not in PlottedSysts:

                            hypo = fhkey.split('_')[0]
                            fid = fhkey.split('_')[-2]
                            FitTitle = ("True %s, Fiducial Fit %s, Hypothesis "
                                        "%s (%i Trials)"
                                        %(labels['data_name'],
                                          labels['%s_name'%fid],
                                          labels['%s_name'%hypo],
                                          len(data[fhkey][xsystkey]['vals'])))
                            
                            plt.subplot(SystNum-1,SystNum-1,subplotnum)

                            plot_individual_scatter(
                                xdata = data[fhkey][xsystkey],
                                ydata = data[fhkey][ysystkey],
                                labels = labels,
                                xsystkey = xsystkey,
                                ysystkey = ysystkey,
                                plot_cor = False
                            )

        plt.suptitle(MainTitle+r'\\'+FitTitle, fontsize=120)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        SaveName = (("true_%s_%s_%s_fid_%s_hypo_%s_all"
                     "_scatter_plots.png"
                     %(labels['data_name'],
                       detector,
                       selection,
                       labels['%s_name'%fid],
                       labels['%s_name'%hypo])))
        plt.savefig(os.path.join(outdir,SaveName))
        plt.close()


def plot_correlation_matrices(data, labels, detector, selection, outdir):
    '''
    This will plot the correlation matrices since the individual scatter plots 
    are a pain to interpret on their own. This will plot them with a colour 
    scale and, if the user has the PathEffects module then it will also write 
    the values on the bins. If a number is invalid it will come up bright green.
    '''
    try:
        import matplotlib.patheffects as PathEffects
        logging.warn("PathEffects could be imported, so the correlation values"
                     " will be written on the bins. This is slow.")
        pe = True
    except:
        logging.warn("PathEffects could not be imported, so the correlation" 
                     " values will not be written on the bins.")
        pe = False

    outdir = os.path.join(outdir,'CorrelationMatrices')
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)

    MainTitle = ("%s %s Event Selection Correlation Coefficients"
                 %(detector, selection))
    Systs = []

    for fhkey in data.keys():
        # Systematic number is one less than number of keys since this also
        # contains the metric_val entry
        SystNum = len(data[fhkey].keys())-1
        # Set up array to hold lists of correlation values
        all_corr_lists = []
        for xsystkey in data[fhkey].keys():
            all_corr_values = []
            if not xsystkey == 'metric_val':
                if tex_axis_label(xsystkey) not in Systs:
                    Systs.append(tex_axis_label(xsystkey))
                for ysystkey in data[fhkey].keys():
                    if (ysystkey != 'metric_val'):
                        hypo = fhkey.split('_')[0]
                        fid = fhkey.split('_')[-2]
                        FitTitle = ("True %s, Fiducial Fit %s, Hypothesis "
                                    "%s (%i Trials)"
                                    %(labels['data_name'],
                                      labels['%s_name'%fid],
                                      labels['%s_name'%hypo],
                                      len(data[fhkey][xsystkey]['vals'])))

                        # Calculate correlation
                        xvals = np.array(data[fhkey][xsystkey]['vals'])
                        yvals = np.array(data[fhkey][ysystkey]['vals'])
                        if len(set(xvals)) == 1:
                            logging.warn(("Parameter %s appears to not have "
                                          "been varied. i.e. all of the values"
                                          " in the set are the same. This will"
                                          " lead to NaN in the correlation "
                                          "calculation and so it will not be "
                                          "done."%xsystkey))
                        if len(set(yvals)) == 1:
                            logging.warn(("Parameter %s appears to not have "
                                          "been varied. i.e. all of the values"
                                          " in the set are the same. This will"
                                          " lead to NaN in the correlation "
                                          "calculation and so it will not be "
                                          "done."%ysystkey))
                        if (len(set(xvals)) != 1) and (len(set(yvals)) != 1):
                            rho, pval = spearmanr(xvals, yvals)
                        else:
                            rho = np.nan
                        all_corr_values.append(rho)
                all_corr_lists.append(all_corr_values)

        all_corr_nparray = np.ma.masked_invalid(np.array(all_corr_lists))
        # Plot it!
        palette = plt.cm.RdBu
        palette.set_bad('lime',1.0)
        plt.imshow(
            all_corr_nparray,
            interpolation='none',
            cmap=plt.cm.RdBu,
            vmin=-1.0,
            vmax=1.0
        )
        plt.colorbar()
        # Add systematic names as x and y axis ticks
        plt.xticks(
            np.arange(len(Systs)),
            Systs,
            rotation=45,
            horizontalalignment='right'
        )
        plt.yticks(
            np.arange(len(Systs)),
            Systs,
            rotation=0
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.30,left=-0.30,right=1.05,top=0.9)
        plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
        SaveName = (("true_%s_%s_%s_fid_%s_hypo_%s_correlation_matrix.png"
                     %(labels['data_name'],
                       detector,
                       selection,
                       labels['%s_name'%fid],
                       labels['%s_name'%hypo])))
        plt.savefig(os.path.join(outdir,SaveName))
        if pe:
            for i in range(0,len(all_corr_nparray)):
                for j in range(0,len(all_corr_nparray[0])):
                    plt.text(i, j, '%.2f'%all_corr_nparray[i][j],
                             fontsize='7',
                             verticalalignment='center',
                             horizontalalignment='center',
                             color='w',
                             path_effects=[
                                 PathEffects.withStroke(
                                     linewidth=2.5,
                                     foreground='k'
                                 )
                             ])
        SaveName = (("true_%s_%s_%s_fid_%s_hypo_%s_correlation_matrix_"
                     "values.png"
                     %(labels['data_name'],
                       detector,
                       selection,
                       labels['%s_name'%fid],
                       labels['%s_name'%hypo])))
        plt.savefig(os.path.join(outdir,SaveName))
        plt.close()

    
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '-d', '--dir', required=True,
        metavar='DIR', type=str,
        help='''Directory containing output of hypo_testing.py.'''
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--asimov', action='store_true',
        help='''Analyze the Asimov trials in the specified directories.'''
    )
    group.add_argument(
        '--llr', action='store_true',
        help='''Analyze the LLR trials in the specified directories.'''
    )
    parser.add_argument(
        '--detector', type=str, default='',
        help='''Name of detector to put in histogram titles.'''
    )
    parser.add_argument(
        '--selection', type=str, default='',
        help='''Name of selection to put in histogram titles.'''
    )
    parser.add_argument(
        '-FM', '--fit_information', action='store_true', default=False,
        help='''Flag to make plots of the minimiser information i.e. status, 
        number of iterations, time taken etc.'''
    )
    parser.add_argument(
        '-IP', '--individual_posteriors', action='store_true', default=False,
        help='''Flag to plot individual posteriors.'''
    )
    parser.add_argument(
        '-CP', '--combined_posteriors', action='store_true', default=False,
        help='''Flag to plot combined posteriors for each h0 and h1 
        combination.'''
    )
    parser.add_argument(
        '-IS', '--individual_scatter', action='store_true', default=False,
        help='''Flag to plot individual 2D scatter plots of posteriors.'''
    )
    parser.add_argument(
        '-CIS', '--combined_individual_scatter',
        action='store_true', default=False,
        help='''Flag to plot all 2D scatter plots of one systematic with every 
        other systematic on one plot for each h0 and h1 combination.'''
    )
    parser.add_argument(
        '-CS', '--combined_scatter', action='store_true', default=False,
        help='''Flag to plot all 2D scatter plots on one plot for each 
        h0 and h1 combination.'''
    )
    parser.add_argument(
        '-CM', '--correlation_matrix', action='store_true', default=False,
        help='''Flag to plot the correlation matrices for each h0 and h1 
        combination.'''
    )
    parser.add_argument(
        '--threshold', type=float, default=5.0,
        help='''Sets the threshold for which to remove 'outlier' trials. 
        Ideally this will not be needed at all, but it is there in case of
        e.g. failed minimiser. The higher this value, the more outliers will 
        be included. Set it to 0 if you want to include ALL trials.'''
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
    fitinfo = init_args_d.pop('fit_information')
    iposteriors = init_args_d.pop('individual_posteriors')
    cposteriors = init_args_d.pop('combined_posteriors')
    iscatter = init_args_d.pop('individual_scatter')
    ciscatter = init_args_d.pop('combined_individual_scatter')
    cscatter = init_args_d.pop('combined_scatter')
    cmatrix = init_args_d.pop('correlation_matrix')
    threshold = init_args_d.pop('threshold')
    outdir = init_args_d.pop('outdir')

    if args.asimov:
        data_sets, all_params, labels, minimiser_info = extract_trials(
            logdir=args.dir,
            fluctuate_fid=False,
            fluctuate_data=False
        )
        od = data_sets.values()[0]
        #if od['h1_fit_to_h0_fid']['fid_asimov']['metric_val'] > od['h0_fit_to_h1_fid']['fid_asimov']['metric_val']:
        print np.sqrt(np.abs(od['h1_fit_to_h0_fid']['fid_asimov']['metric_val'] - od['h0_fit_to_h1_fid']['fid_asimov']['metric_val']))

    else:
        data_sets, all_params, labels, minimiser_info = extract_trials(
            logdir=args.dir,
            fluctuate_fid=True,
            fluctuate_data=False
        )

        trial_nums = data_sets[
            'toy_%s_asimov'%labels.dict['data_name']
        ][
            'h0_fit_to_h1_fid'
        ].keys()
        
        fid_values = extract_fid_data(data_sets)
        values = extract_data(data_sets)

        for injkey in values.keys():

            if threshold != 0.0:
                logging.info("Outlying trials will be removed with a "
                             "threshold of %.2f"%threshold)
                purge_failed_jobs(
                    data = values[injkey],
                    trial_nums = np.array(trial_nums),
                    thresh = threshold
                )
            else:
                logging.info("All trials will be included in the analysis.")
            
            make_llr_plots(
                data = values[injkey],
                fid_data = fid_values[injkey],
                labels = labels.dict,
                detector = detector,
                selection = selection,
                outdir = outdir
            )

            if fitinfo:
                
                plot_fit_information(
                    minimiser_info=minimiser_info[injkey],
                    labels = labels.dict,
                    detector = detector,
                    selection = selection,
                    outdir=outdir
                )

            if iposteriors:

                plot_individual_posteriors(
                    data = values[injkey],
                    fid_data = fid_values[injkey],
                    labels = labels.dict,
                    all_params = all_params,
                    detector = detector,
                    selection = selection,
                    outdir = outdir
                )

            if cposteriors:

                plot_combined_posteriors(
                    data = values[injkey],
                    fid_data = fid_values[injkey],
                    labels = labels.dict,
                    all_params = all_params,
                    detector = detector,
                    selection = selection,
                    outdir = outdir
                )

            if iscatter:

                plot_individual_scatters(
                    data = values[injkey],
                    labels = labels.dict,
                    detector = detector,
                    selection = selection,
                    outdir = outdir
                )

            if ciscatter:

                plot_combined_individual_scatters(
                    data = values[injkey],
                    labels = labels.dict,
                    detector = detector,
                    selection = selection,
                    outdir = outdir
                )

            if cscatter:

                plot_combined_scatters(
                    data = values[injkey],
                    labels = labels.dict,
                    detector = detector,
                    selection = selection,
                    outdir = outdir
                )

            if cmatrix:

                plot_correlation_matrices(
                    data = values[injkey],
                    labels = labels.dict,
                    detector = detector,
                    selection = selection,
                    outdir = outdir
                )
                
        
if __name__ == '__main__':
    main()
