#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This computes significances, etc. from the logfiles recorded by the
`hypo_testing.py` script, for either Asimov or LLR analysis. Plots and tables
are produced in the case of LLR analysis.
"""


# TODO:
#
# 1) Some of the "combined" plots currently make it impossible to read the axis
#    labels. Come up with a better way of doing this. Could involve making
#    legends and just labelling the axes alphabetically.
# 2) The important one - Figure out if this script generalises to the case of
#    analysing data. My gut says it doesn't...


from __future__ import division

from argparse import ArgumentParser
from collections import OrderedDict
import os
import re

import numpy as np
from scipy.stats import spearmanr

from pisa.analysis.hypo_testing import Labels
from pisa.utils.fileio import from_file, mkdir, to_file, nsort
from pisa.utils.log import set_verbosity, logging
from pisa.utils.postprocess import (tex_axis_label, parse_pint_string,
                                    get_num_rows, extract_gaussian,
                                    plot_colour, plot_style)


__all__ = ['extract_trials', 'extract_fit', 'parse_args', 'main']


def extract_paramval(injparams, systkey, fid_label=None, hypo_label=None,
                     paramlabel=None):
    """Extract a value from a set of parameters and modify it based on the
    hypothesis/fiducial fit being considered. The label associated with this
    is then modified accordingly."""
    paramval = float(injparams[systkey].split(' ')[0])
    if (fid_label is None) or (hypo_label is None) or (paramlabel is None):
        if not ((fid_label is None) and (hypo_label is None) and
                (paramlabel is None)):
            raise ValueError('Either all three labels must be None or they '
                             ' must all be specified.')
        return paramval
    else:
        if systkey == 'deltam31':
            if 'no' in hypo_label:
                if np.sign(paramval) != 1:
                    paramval = -1*float(injparams[systkey].split(' ')[0])
                    paramlabel += r' ($\times-1$)'
            elif 'io' in hypo_label:
                if np.sign(paramval) != -1:
                    paramval = -1*float(injparams[systkey].split(' ')[0])
                    paramlabel += r' ($\times-1$)'

        if (np.abs(paramval) < 1e-2) and (paramval != 0.0):
            paramlabel += ' = %.3e'%paramval
        else:
            paramlabel += ' = %.3g'%paramval

        return paramval, paramlabel


def get_set_file_nums(filedir, labels, fluctuate_fid):
    """This function returns the set of file numbers that exist for all h0 and 
    h1 combination. This is needed to account for any failed or non-transferred
    jobs. i.e. for trial X you may not have all of the necessary fit files so 
    it must be ignored."""
    file_nums = OrderedDict()
    for fnum, fname in enumerate(nsort(os.listdir(filedir))):
        fpath = os.path.join(filedir, fname)
        for x in ['0', '1']:
            for y in ['0', '1']:
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
            set_file_nums = set_file_nums.intersection(
                file_nums[hypokey]
            )
    return set_file_nums


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
    if 'config_summary.json' in logdir_content:
        # Look for the pickle files in the directory to indicate that this
        # data may have already been processed.
        expected_files = ['data_sets.pckl', 'all_params.pckl',
                          'labels.pckl', 'minimiser_info.pckl']
        if np.all(np.array([s in logdir_content for s in expected_files])):
            # Processed output files are there so make sure that there
            # have been no more trials run since this last processing.
            ## To do this, get the number of output files
            config_summary_fpath = os.path.join(logdir,
                                                'config_summary.json')
            cfg = from_file(config_summary_fpath)
            data_is_data = cfg['data_is_data']
            # Get naming scheme
            labels = Labels(
                h0_name=cfg['h0_name'], h1_name=cfg['h1_name'],
                data_name=cfg['data_name'], data_is_data=data_is_data,
                fluctuate_data=fluctuate_data, fluctuate_fid=fluctuate_fid
            )
            for basename in nsort(os.listdir(logdir)):
                m = labels.subdir_re.match(basename)
                if m is None:
                    continue
                # Here is the output directory which contains the files
                subdir = os.path.join(logdir, basename)
                # Account for failed jobs. Get the set of file numbers that
                # exist for all h0 and h1 combinations
                set_file_nums = get_set_file_nums(
                    filedir=os.path.join(logdir, basename),
                    labels=labels,
                    fluctuate_fid=fluctuate_fid
                )
                num_trials = len(set_file_nums)
                # Take one of the pickle files to see how many data
                # entries it has.
                data_sets = from_file(os.path.join(logdir,
                                                   'data_sets.pckl'))
                # Take the first data key and then the h0 fit to h0 fid
                # which should always exist. The length of this is then
                # the number of trials in the pickle files.
                pckl_trials = len(data_sets[data_sets.keys()[0]][
                    'h0_fit_to_h0_fid'].keys())
                # The number of pickle trials should match the number of
                # trials derived from the output directory.
                if num_trials == pckl_trials:
                    logging.info(
                        'Found files I assume to be from a previous run of'
                        ' this processing script containing %i trials. If '
                        'this seems incorrect please delete the files: '
                        'data_sets.pckl, all_params.pckl and labels.pckl '
                        'from the logdir you have provided.'%pckl_trials
                    )
                    data_sets = from_file(
                        os.path.join(logdir, 'data_sets.pckl')
                    )
                    all_params = from_file(
                        os.path.join(logdir, 'all_params.pckl')
                    )
                    labels = from_file(
                        os.path.join(logdir, 'labels.pckl')
                    )
                    minimiser_info = from_file(
                        os.path.join(logdir, 'minimiser_info.pckl')
                    )
                    genfiles = False
                else:
                    logging.info(
                        'Found files I assume to be from a previous run of'
                        ' this processing script containing %i trials. '
                        'However, based on the number of json files in the '
                        'output directory there should be %i trials in '
                        'these pickle files, so they will be regenerated.'%(
                            pckl_trials,num_trials)
                    )
                    genfiles = True
        else:
            logging.info(
                'Did not find all of the files - %s - expected to indicate '
                'this data has already been extracted.'
            )
            genfiles = True
        if genfiles:
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
            if not data_is_data:
                all_params['data_params'] = {}
                for param_string in cfg['data_params']:
                    bits = re.match(parse_string, param_string, re.M|re.I)
                    if bits.group(5) == 'False':
                        all_params['data_params'][bits.group(1)] = {}
                        all_params['data_params'][bits.group(1)]['value'] \
                            = bits.group(2)
                        all_params['data_params'][bits.group(1)]['prior'] \
                            = bits.group(3)
                        all_params['data_params'][bits.group(1)]['range'] \
                            = bits.group(4)
            else:
                all_params['data_params'] = None
            for param_string in cfg['h0_params']:
                bits = re.match(parse_string, param_string, re.M|re.I)
                if bits.group(5) == 'False':
                    all_params['h0_params'][bits.group(1)] = {}
                    all_params['h0_params'][bits.group(1)]['value'] \
                        = bits.group(2)
                    all_params['h0_params'][bits.group(1)]['prior'] \
                        = bits.group(3)
                    all_params['h0_params'][bits.group(1)]['range'] \
                        = bits.group(4)
            for param_string in cfg['h1_params']:
                bits = re.match(parse_string, param_string, re.M|re.I)
                if bits.group(5) == 'False':
                    all_params['h1_params'][bits.group(1)] = {}
                    all_params['h1_params'][bits.group(1)]['value'] \
                        = bits.group(2)
                    all_params['h1_params'][bits.group(1)]['prior'] \
                        = bits.group(3)
                    all_params['h1_params'][bits.group(1)]['range'] \
                        = bits.group(4)

            # Find all relevant data dirs, and from each extract the fiducial
            # fit(s) information contained
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

                # Account for failed jobs. Get the set of file numbers that
                # exist for all h0 an h1 combinations
                subdir = os.path.join(logdir, basename)
                set_file_nums = get_set_file_nums(
                    filedir=subdir,
                    labels=labels,
                    fluctuate_fid=fluctuate_fid
                )

                fnum = None
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
                                k = 'h{x}_fit_to_{y}'.format(x=x, y=dset_label)
                                lvl2_fits[k] = extract_fit(
                                    fpath,
                                    ['metric_val', 'params']
                                )
                                break
                        k = 'h{x}_fit_to_{y}'.format(x=x, y=dset_label)
                        for y in ['0', '1']:
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
                                    ['metric', 'metric_val', 'params']
                                )
                                minim_info[k][fid_label] = extract_fit(
                                    fpath,
                                    ['minimizer_metadata', 'minimizer_time']
                                )
                            break

                if fnum is None:
                    raise ValueError('No files?')

                data_sets[dset_label] = lvl2_fits
                minimiser_info[dset_label] = minim_info
                data_sets[dset_label]['params'] = extract_fit(
                    fpath,
                    ['params']
                )['params']
                to_file(data_sets, os.path.join(logdir, 'data_sets.pckl'))
                to_file(all_params, os.path.join(logdir, 'all_params.pckl'))
                to_file(labels, os.path.join(logdir, 'labels.pckl'))
                to_file(minimiser_info, os.path.join(logdir,
                                                     'minimiser_info.pckl'))
    else:
        raise ValueError(
            'config_summary.json cannot be found in the specified logdir. It '
            'should have been created as part of the output of hypo_testing.py'
            'and so this postprocessing cannot be performed.'
        )
    
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


# TODO (?) - This works in the case of all MC, but I don't know about data.
def extract_fid_data(data_sets):
    """Take the data sets returned by the `extract_trials` and extract the data
    on the fiducial fits."""
    fid_values = {}
    for injkey in data_sets.keys():
        fid_values[injkey] = {}
        for datakey in data_sets[injkey]:
            if ('toy' in datakey) or ('data' in datakey):
                fid_values[injkey][datakey] \
                    = data_sets[injkey].pop(datakey)
    return fid_values


def extract_data(data):
    """Take the data sets returned by `extract_trials` and turn them in to a
    format used by all of the plotting functions."""
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


def purge_failed_jobs(data, trial_nums, thresh=5.0):
    """Look at the values of the metric and find any deemed to be from a failed
    job. That is, the value of the metric falls very far outside of the rest of
    the values.

    Notes
    -----
    Interestingly, I only saw a need for this with my true NO jobs, where I
    attempted to run some jobs in fp32 mode. No jobs were needed to be removed
    for true IO, where everything was run in fp64 mode. So if there's a need
    for this function in your analysis it probably points at some more serious
    underlying problem.

    References:
    ----------
        Taken from stack overflow:

            http://stackoverflow.com/questions/22354094/pythonic-way-\
            of-detecting-outliers-in-one-dimensional-observation-data

        which references:

            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect
            and Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.

    """

    for fit_key in data.keys():
        points = np.array(data[fit_key]['metric_val']['vals'])
        if len(points.shape) == 1:
            points = points[:, None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        modified_z_score = 0.6745 * diff / med_abs_deviation
        good_trials = modified_z_score < thresh
        if not np.all(good_trials):
            bad_trials = np.where(good_trials == False)[0]
            logging.warn(
                'Outlier(s) detected for %s in trial(s) %s. Will be removed.'
                ' If you think this should not happen, please change the'
                ' value of the threshold used for the decision (currently set'
                ' to %.2e).',
                fit_key, trial_nums[bad_trials], thresh
            )
            for fitkey in data.keys():
                for param in data[fitkey].keys():
                    new_vals = np.delete(
                        np.array(data[fitkey][param]['vals']),
                        bad_trials
                    )
                    data[fitkey][param]['vals'] = new_vals

def save_plot(labels, fid, hypo, detector, selection, outdir, formats, end):
    """Save plot as each type of file format specified in `formats`"""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    save_name = ""
    if 'data_name' in labels.keys():
        save_name += "true_%s_"%labels['data_name']
    if detector is not None:
        save_name += "%s_"%detector
    if selection is not None:
        save_name += "%s_"%selection
    if fid is not None:
        save_name += "fid_%s_"%labels['%s_name'%fid]
    if hypo is not None:
        save_name += "hypo_%s_"%labels['%s_name'%hypo]
    save_name += end
    for fileformat in formats:
        full_save_name = save_name + '.%s'%fileformat
        plt.savefig(os.path.join(outdir, full_save_name))


def make_main_title(detector, selection, end):
    """Make main title accounting for detector and selection."""
    MainTitle = r"\begin{center}"
    if detector is not None:
        MainTitle += "%s "%detector
    if selection is not None:
        MainTitle += "%s Event Selection "%selection
    MainTitle += end
    return MainTitle


def make_fit_title(labels, fid, hypo, trials):
    """Make fit title to go with the main title"""
    FitTitle = ""
    if 'data_name' in labels.keys():
        FitTitle += "True %s, "%labels['data_name']
    if fid is not None:
        FitTitle += "Fiducial Fit %s, "%labels['%s_name'%fid]
    if hypo is not None:
        FitTitle += "Hypothesis %s "%labels['%s_name'%hypo]
    if trials is not None:
        FitTitle += "(%i Trials)"%trials
    FitTitle += r"\end{center}"
    return FitTitle


def make_fit_information_plot(fit_info, xlabel, title):
    """Make histogram of fit_info given with axis label and title"""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    plt.grid(axis='y', zorder=0)
    plt.hist(
        fit_info,
        bins=10,
        histtype='bar',
        color='darkblue',
        alpha=0.9,
        zorder=3
    )
    plt.xlabel(xlabel, size='24')
    plt.ylabel('Number of Trials', size='24')
    plt.title(title, fontsize=16)
    plt.subplots_adjust(left=0.10, right=0.90, top=0.9, bottom=0.11)


def plot_fit_information(minimiser_info, labels, detector,
                         selection, outdir, formats):
    """Make plots of the number of iterations and time taken with the
    minimiser. This is a good cross-check that the minimiser did not end
    abruptly since you would see significant pile-up if it did."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    outdir = os.path.join(outdir, 'MinimiserPlots')
    mkdir(outdir)
    MainTitle = make_main_title(detector, selection, 'Minimiser Information')
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
            FitTitle = make_fit_title(
                labels=labels,
                fid=fid,
                hypo=hypo,
                trials=len(minimiser_times)
            )
            # Minimiser times
            make_fit_information_plot(
                fit_info=minimiser_times,
                xlabel='Minimiser Time (seconds)',
                title=MainTitle+r'\\'+FitTitle
            )
            save_plot(
                labels=labels,
                fid=fid,
                hypo=hypo,
                detector=detector,
                selection=selection,
                outdir=outdir,
                formats=formats,
                end='minimiser_times'
            )
            plt.close()
            # Minimiser iterations
            make_fit_information_plot(
                fit_info=minimiser_iterations,
                xlabel='Minimiser Iterations',
                title=MainTitle+r'\\'+FitTitle
            )
            save_plot(
                labels=labels,
                fid=fid,
                hypo=hypo,
                detector=detector,
                selection=selection,
                outdir=outdir,
                formats=formats,
                end='minimiser_iterations'
            )
            plt.close()
            # Minimiser function evaluations
            make_fit_information_plot(
                fit_info=minimiser_funcevals,
                xlabel='Minimiser Function Evaluations',
                title=MainTitle+r'\\'+FitTitle
            )
            save_plot(
                labels=labels,
                fid=fid,
                hypo=hypo,
                detector=detector,
                selection=selection,
                outdir=outdir,
                formats=formats,
                end='minimiser_funcevals'
            )
            plt.close()
            # Minimiser status
            make_fit_information_plot(
                fit_info=minimiser_status,
                xlabel='Minimiser Status',
                title=MainTitle+r'\\'+FitTitle
            )
            save_plot(
                labels=labels,
                fid=fid,
                hypo=hypo,
                detector=detector,
                selection=selection,
                outdir=outdir,
                formats=formats,
                end='minimiser_status'
            )
            plt.close()


def add_extra_points(points, labels, ymax):
    """Add extra points specified by `points` and label them with `labels`"""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    linelist = []
    for point, label in zip(points, labels):
        if isinstance(point, basestring):
            if os.path.isfile(point):
                point = np.genfromtxt(point)
            try:
                point = eval(point)
            except:
                raise ValueError('Provided point, %s, was not either a '
                                 'path to a file or a string which could '
                                 'be parsed by eval()' % point)
        if not isinstance(point, float):
            raise ValueError('Expecting a single point here to add to the plot'
                             ' and got %s instead.' % point)
        line = plt.axvline(
            point,
            color=plot_colour(label),
            linestyle=plot_style(label),
            ymax=ymax,
            lw=2,
            label=label
        )
        linelist.append(label)
    return linelist


def calc_p_value(LLRdist, critical_value, num_trials, greater=True,
                 median_p_value=False, LLRbest=None):
    """Calculate the p-value for the given dataset based on the given critical
    value with an associated error.

    The calculation involves asking in how many trials the test statistic was
    "beyond" the critical value. The threshold of beyond will depend on whether
    the given distribution is the best fit or the alternate fit. The default is
    a "greater than" criterion, which can be switched by setting the "greater"
    argument to false.

    In the case of median_p_values the error calculation must also account for
    the uncertainty on the median, and so one must pass the distribution from
    which this was calculated so the error can be estimated with bootstrapping.

    """
    if greater:
        misid_trials = float(np.sum(LLRdist > critical_value))
    else:
        misid_trials = float(np.sum(LLRdist < critical_value))
    p_value = misid_trials/num_trials
    if median_p_value:
        # Quantify the uncertainty on the median by bootstrapping
        sampled_medians = []
        for i in range(0, 1000):
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
        wdenom = misid_trials+median_error*median_error
        wterm = wdenom/(misid_trials*misid_trials)
        Nterm = 1.0/num_trials
        unc_p_value = p_value * np.sqrt(wterm + Nterm)
        return p_value, unc_p_value, median_error
    else:
        unc_p_value = np.sqrt(misid_trials*(1-p_value))/num_trials
        return p_value, unc_p_value


def plot_LLR_histograms(LLRarrays, LLRhistmax, binning, colors, labels,
                        best_name, alt_name, critical_value, critical_label,
                        critical_height, LLRhist, critical_color='k',
                        plot_scaling_factor=1.55, greater=True, CLs=False):
    """Plot the LLR histograms. The `greater` argument is intended to be used
    the same as in the p value function above."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    for LLRarray, label, color in zip(LLRarrays, labels, colors):
        plt.hist(
            LLRarray,
            bins=binning,
            color=color,
            histtype='step',
            lw=2,
            label=label
        )
    plt.xlabel(r'Log-Likelihood Ratio', size='24', labelpad=22)
    plt.ylabel(r'Number of Trials (per %.2f)'%(binning[1]-binning[0]), size='24')
    # Nicely scale the plot
    plt.ylim(0, plot_scaling_factor*LLRhistmax)
    # Add labels to show which side means what...
    xlim = plt.gca().get_xlim()
    plt.text(
        xlim[0]-0.05*(xlim[1]-xlim[0]),
        -0.09*plot_scaling_factor*LLRhistmax,
        r'\begin{flushleft} $\leftarrow$ Prefers %s\end{flushleft}' % tex_axis_label(alt_name),
        color='k',
        size='large'
    )
    plt.text(
        xlim[1]+0.05*(xlim[1]-xlim[0]),
        -0.09*plot_scaling_factor*LLRhistmax,
        r'\begin{flushright} Prefers %s $\rightarrow$ \end{flushright}' % tex_axis_label(best_name),
        color='k',
        size='large',
        horizontalalignment='right'
    )
    # Add the critical value with the desired height and colour.
    plt.axvline(
        critical_value,
        color=critical_color,
        ymax=critical_height,
        lw=2,
        label=critical_label
    )
    if CLs:
        for hist, color in zip(LLRhist, colors):
            finehist = np.repeat(hist, 100)
            finebinning = np.linspace(binning[0],
                                      binning[-1],
                                      (len(binning)-1)*100+1)
            finebinwidth = finebinning[1]-finebinning[0]
            finebincens = np.linspace(finebinning[0]+finebinwidth/2.0,
                                      finebinning[-1]-finebinwidth/2.0,
                                      len(finebinning)-1)
            if color == 'r':
                where = (finebincens < critical_value)
            elif color == 'b':
                where = (finebincens > critical_value)
            plt.fill_between(
                finebincens,
                0,
                finehist,
                where=where,
                color='none',
                hatch='X',
                edgecolor=color,
                lw=0
            )
    else:
        # Create an object so that a hatch can be drawn over the region of
        # interest to the p-value.
        finehist = np.repeat(LLRhist, 100)
        finebinning = np.linspace(binning[0], binning[-1], (len(binning)-1)*100+1)
        finebinwidth = finebinning[1]-finebinning[0]
        finebincens = np.linspace(finebinning[0]+finebinwidth/2.0,
                                  finebinning[-1]-finebinwidth/2.0,
                                  len(finebinning)-1)
        # Draw the hatch. This is between the x-axis (0) and the finehist
        # object made above. The "where" tells is to only draw above the
        # critical value. To make it just the hatch, color is set to none and
        # hatch is set to X.
        # Also, so that it doesn't have a border we set linewidth to zero.
        if greater:
            where = (finebincens > critical_value)
        else:
            where = (finebincens < critical_value)
        plt.fill_between(
            finebincens,
            0,
            finehist,
            where=where,
            color='none',
            hatch='X',
            edgecolor='k',
            lw=0
        )

    plt.subplots_adjust(left=0.10, right=0.90, top=0.9, bottom=0.15)


def make_llr_plots(data, fid_data, labels, detector, selection, outdir,
                   formats, extra_points=None, extra_points_labels=None):
    """Make LLR plots.

    Takes the data and makes LLR distributions. These are then saved to the
    requested outdir within a folder labelled "LLRDistributions". The
    extra_points and extra_points_labels arguments can be used to specify extra
    points to be added to the plot for e.g. other fit LLR values.

    TODO:

    1) Currently the p-value is put on the LLR distributions as an annotation.
       This is probably fine, since the significances can just be calculated
       from this after the fact.

    """
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    outdir = os.path.join(outdir, 'LLRDistributions')
    mkdir(outdir)

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
        logging.info('Converting chi2 metric to likelihood equivalent.')
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
    # Special case for low numbers of trials. Here, the plot can't really be
    # interpreted but the numbers printed on it can still be useful, so we need
    # to make something.
    if num_trials < 100:
        binning = np.linspace(minLLR - 0.1*rangeLLR,
                              maxLLR + 0.1*rangeLLR,
                              10)
    else:
        binning = np.linspace(minLLR - 0.1*rangeLLR,
                              maxLLR + 0.1*rangeLLR,
                              int(num_trials/40))
    binwidth = binning[1]-binning[0]
    bincens = np.linspace(binning[0]+binwidth/2.0,
                          binning[-1]-binwidth/2.0,
                          len(binning)-1)

    LLRbesthist, LLRbestbinedges = np.histogram(LLRbest, bins=binning)
    LLRalthist, LLRaltbinedges = np.histogram(LLRalt, bins=binning)

    LLRhistmax = max(max(LLRbesthist), max(LLRalthist))

    best_median = np.median(LLRbest)
    alt_median = np.median(LLRalt)

    inj_name = labels['data_name']
    best_name = labels['%s_name'%bestfit]
    alt_name = labels['%s_name'%altfit]

    # Calculate p values
    ## First for the preferred hypothesis based on the fiducial fit
    crit_p_value, unc_crit_p_value = calc_p_value(
        LLRdist=LLRalt,
        critical_value=critical_value,
        num_trials=num_trials,
        greater=True
    )
    ## Then for the alternate hypothesis based on the fiducial fit
    alt_crit_p_value, alt_unc_crit_p_value = calc_p_value(
        LLRdist=LLRbest,
        critical_value=critical_value,
        num_trials=num_trials,
        greater=False
    )
    ## Combine these to give a CLs value based on arXiv:1407.5052
    cls_value = (1 - alt_crit_p_value) / (1 - crit_p_value)
    unc_cls_value = cls_value * np.sqrt(
        np.power(alt_unc_crit_p_value/alt_crit_p_value, 2.0) + \
        np.power(unc_crit_p_value/crit_p_value, 2.0)
    )
    ## Then for the preferred hypothesis based on the median. That is, the case
    ## of a median experiment from the distribution under the preferred
    ## hypothesis.
    med_p_value, unc_med_p_value, median_error = calc_p_value(
        LLRdist=LLRalt,
        critical_value=best_median,
        num_trials=num_trials,
        greater=True,
        median_p_value=True,
        LLRbest=LLRbest
    )

    if metric_type == 'llh':
        plot_title = (r"\begin{center}"\
                      +"%s %s Event Selection "%(detector, selection)\
                      +r"\\"+" LLR Distributions for true %s (%i trials)"%(
                          tex_axis_label(inj_name), num_trials)\
                      +r"\end{center}")

    else:
        plot_title = (r"\begin{center}"\
                      +"%s %s Event Selection "%(detector, selection)\
                      +r"\\"+" %s \"LLR\" Distributions for "
                      %(metric_type_pretty)\
                      +"true %s (%i trials)"%(tex_axis_label(inj_name),
                                              num_trials)\
                      +r"\end{center}")

    # Factor with which to make everything visible
    plot_scaling_factor = 1.55

    # In case of median plot, draw both best and alt histograms
    ## Set up the labels for the histograms
    LLR_labels = [
        r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
            tex_axis_label(best_name)) + \
        r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
            best_name, alt_name),
        r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
            tex_axis_label(alt_name)) + \
        r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
            best_name, alt_name)
    ]
    plot_LLR_histograms(
        LLRarrays=[LLRbest, LLRalt],
        LLRhistmax=LLRhistmax,
        binning=binning,
        colors=['r', 'b'],
        labels=LLR_labels,
        best_name=best_name,
        alt_name=alt_name,
        critical_value=best_median,
        critical_label=r"%s Median = $%.4f\pm%.4f$"%(
            tex_axis_label(best_name),
            best_median,
            median_error),
        critical_height=float(max(LLRbesthist))/float(
            plot_scaling_factor*LLRhistmax),
        LLRhist=LLRalthist,
        greater=True
    )
    plt.legend(loc='upper left')
    plt.title(plot_title)
    # Write the p-value on the plot
    plt.figtext(
        0.15,
        0.66,
        r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.4f\pm%.4f$"%(
            best_name, med_p_value, unc_med_p_value),
        color='k',
        size='xx-large'
    )
    save_plot(
        labels=labels,
        fid=None,
        hypo=None,
        detector=detector,
        selection=selection,
        outdir=outdir,
        formats=formats,
        end='%s_LLRDistribution_median_%i_Trials'%(metric_type, num_trials)
    )
    # Add the extra points if they exist
    if extra_points is not None:
        curleg = plt.gca().get_legend()
        linelist = add_extra_points(
            points=extra_points,
            labels=extra_points_labels,
            ymax=float(max(LLRbesthist))/float(plot_scaling_factor*LLRhistmax)
        )
        handles, labels = plt.gca().get_legend_handles_labels()
        newhandles = []
        for l, h in zip(labels, handles):
            if l in linelist:
                newhandles.append(h)
        if len(newhandles) > 13:
            newleg = plt.legend(
                handles=newhandles,
                loc='upper right',
                fontsize=11
            )
        else:
            newleg = plt.legend(handles=newhandles, loc='upper right')
        plt.gca().add_artist(newleg)
        plt.gca().add_artist(curleg)
        save_plot(
            labels=labels,
            fid=None,
            hypo=None,
            detector=detector,
            selection=selection,
            outdir=outdir,
            formats=formats,
            end='%s_LLRDistribution_median_w_extra_points_%i_Trials'%(
                metric_type, num_trials)
        )
    plt.close()

    # In case of critical plot, draw just alt histograms
    ## Set up the label for the histogram
    LLR_labels = [
        r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
            tex_axis_label(alt_name)) + \
        r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
            best_name, alt_name)
    ]
    plot_LLR_histograms(
        LLRarrays=[LLRalt],
        LLRhistmax=LLRhistmax,
        binning=binning,
        colors=['b'],
        labels=LLR_labels,
        best_name=best_name,
        alt_name=alt_name,
        critical_value=critical_value,
        critical_label=r"Critical Value = %.4f"%(critical_value),
        critical_height=float(max(LLRbesthist))/float(
            plot_scaling_factor*LLRhistmax),
        LLRhist=LLRalthist,
        greater=True
    )
    plt.legend(loc='upper left')
    plt.title(plot_title)
    # Write the p-value on the plot
    plt.figtext(
        0.15,
        0.70,
        r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.4f\pm%.4f$"%(
            best_name, crit_p_value, unc_crit_p_value),
        color='k',
        size='xx-large'
    )
    save_plot(
        labels=labels,
        fid=None,
        hypo=None,
        detector=detector,
        selection=selection,
        outdir=outdir,
        formats=formats,
        end='%s_LLRDistribution_critical_%i_Trials'%(metric_type, num_trials)
    )
    plt.close()

    # Make a second critical plot for the alt hypothesis, so we draw the
    # preferred hypothesis
    ## Set up the label for the histogram
    LLR_labels = [
        r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
            tex_axis_label(best_name)) + \
        r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
            best_name, alt_name)
    ]
    plot_LLR_histograms(
        LLRarrays=[LLRbest],
        LLRhistmax=LLRhistmax,
        binning=binning,
        colors=['r'],
        labels=LLR_labels,
        best_name=best_name,
        alt_name=alt_name,
        critical_value=critical_value,
        critical_label=r"Critical Value = %.4f"%(critical_value),
        critical_height=float(max(LLRbesthist))/float(
            plot_scaling_factor*LLRhistmax),
        LLRhist=LLRbesthist,
        greater=False
    )
    plt.legend(loc='upper left')
    plt.title(plot_title)
    # Write the p-value on the plot
    plt.figtext(
        0.15,
        0.70,
        r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.4f\pm%.4f$"%(
            alt_name, alt_crit_p_value, alt_unc_crit_p_value),
        color='k',
        size='xx-large'
    )
    save_plot(
        labels=labels,
        fid=None,
        hypo=None,
        detector=detector,
        selection=selection,
        outdir=outdir,
        formats=formats,
        end='%s_LLRDistribution_critical_alt_%i_Trials'%(
            metric_type, num_trials)
    )
    plt.close()

    # Lastly, show both exclusion regions and then the joined CLs value
    ## Set up the labels for the histograms
    LLR_labels = [
        r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
            tex_axis_label(best_name)) + \
        r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
            best_name, alt_name),
        r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
            tex_axis_label(alt_name)) + \
        r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
            best_name, alt_name)
    ]
    plot_LLR_histograms(
        LLRarrays=[LLRbest, LLRalt],
        LLRhistmax=LLRhistmax,
        binning=binning,
        colors=['r', 'b'],
        labels=LLR_labels,
        best_name=best_name,
        alt_name=alt_name,
        critical_value=critical_value,
        critical_label=r"Critical Value = %.4f"%(critical_value),
        critical_height=float(max(LLRbesthist))/float(
            plot_scaling_factor*LLRhistmax),
        LLRhist=[LLRbesthist, LLRalthist],
        CLs=True,
    )
    plt.legend(loc='upper left')
    plt.title(plot_title)
    # Write the p-values on the plot
    plt.figtext(
        0.50,
        0.66,
        r"$\mathrm{CL}_{s}\left(\mathcal{H}_{%s}\right) = %.4f\pm%.4f$"%(
            best_name, cls_value, unc_cls_value),
        horizontalalignment='center',
        color='k',
        size='xx-large'
    )
    plt.figtext(
        0.12,
        0.55,
        r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.2f\pm%.2f$"%(
            alt_name, alt_crit_p_value, alt_unc_crit_p_value),
        bbox=dict(facecolor='none', edgecolor='red', boxstyle='round'),
        horizontalalignment='left',
        color='k',
        size='x-large'
    )
    plt.figtext(
        0.88,
        0.55,
        r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.2f\pm%.2f$"%(
            best_name, crit_p_value, unc_crit_p_value),
        horizontalalignment='right',
        bbox=dict(facecolor='none', edgecolor='blue', boxstyle='round'),
        color='k',
        size='x-large'
    )
    save_plot(
        labels=labels,
        fid=None,
        hypo=None,
        detector=detector,
        selection=selection,
        outdir=outdir,
        formats=formats,
        end='%s_LLRDistribution_CLs_%i_Trials'%(metric_type, num_trials)
    )
    plt.close()


def write_latex_preamble(texfile):
    """Write latex preamble needed to make nice-looking tex files."""
    texfile.write("\n")
    texfile.write("\documentclass[a4paper,12pt]{article}\n")
    texfile.write("\usepackage{tabu}\n")
    texfile.write("\usepackage{booktabs}\n")
    texfile.write("\usepackage[font=small,labelsep=space]{caption} %specifies the caption formatting for the document\n")
    texfile.write("\usepackage[margin=2.5cm]{geometry}\n")
    texfile.write("\setlength{\\topmargin}{1.0cm}\n")
    texfile.write("\setlength{\\textheight}{22cm}\n")
    texfile.write("\usepackage{fancyhdr} %allows for headers and footers\n")
    texfile.write("\pagestyle{fancy}\n")
    texfile.write("\\fancyhf{}\n")
    texfile.write("\\fancyhead[R]{\leftmark}\n")
    texfile.write("\usepackage{multirow}\n")
    texfile.write("\n")
    texfile.write("\\begin{document}\n")
    texfile.write("\n")


def setup_latex_table(texfile, tabletype, injected=False):
    """Set up the beginning of the table for the tex output files. Currently
    will make tables for the output fiducial fit params and the chi2 values
    only."""
    texfile.write("\\renewcommand{\\arraystretch}{1.6}\n")
    texfile.write("\n")
    texfile.write("\\begin{table}[t!]\n")
    texfile.write("  \\begin{center}\n")
    if tabletype == 'fiducial_fit_params':
        if injected:
            texfile.write("    \\begin{tabu} to 1.0\\textwidth {| X[2.0,c] | X[1,c] | X[1,c] | X[1,c] | X[1,c] | X[1,c] | X[1,c] | X[1,c] |}\n")
            texfile.write("    \hline\n")
            texfile.write("    \multirow{2}{*}{\\textbf{Parameter}} & \multirow{2}{*}{\\textbf{Inj}} & \multicolumn{3}{c|}{h0} & \multicolumn{3}{c|}{h1} \\\\ \cline{3-8}")
            texfile.write("    & & Prior & Fit & \(\Delta\) & Prior & Fit & \(\Delta\) \\\\ \hline\n")
        else:
            texfile.write("    \\begin{tabu} to 1.0\\textwidth {| X[c] | X[c] | X[c] |}\n")
            texfile.write("    \hline\n")
            texfile.write("    Parameter & h0 & h1 \\\\ \hline\n")
    elif tabletype == 'fiducial_fit_metrics':
        texfile.write("    \\begin{tabu} to 1.0\\textwidth {| X[c] | X[c] | X[c] |}\n")
        texfile.write("    \hline\n")
        texfile.write("    h0 & h1 & $\Delta$ \\\\ \hline\n")
    else:
        raise ValueError("This function is only for making fit metric or fit "
                         "param tables in LaTeX.")


def end_latex_file(texfile, tabletype, detector, selection, h0, h1,
                   truth=None):
    """End the table and the whole document for the tex output files."""
    if tabletype == 'fiducial_fit_params':
        texfile.write("    \end{tabu}\n")
        texfile.write("  \end{center}\n")
        texfile.write("  \\vspace{-10pt}\n")
        if truth is not None:
            texfile.write("  \caption{shows the fiducial fit parameters obtained with the %s %s sample for h0 of %s and h1 of %s. The truth is %s.}\n"%(detector, selection, h0, h1, truth))
        else:
            texfile.write("  \caption{shows the fiducial fit parameters obtained with the %s %s sample for h0 of %s and h1 of %s.}\n"%(detector, selection, h0, h1))
        texfile.write("  \label{tab:%s%s%stable}\n"%(detector, selection, tabletype))
        texfile.write("\end{table}\n")
        texfile.write("\n")
        texfile.write("\end{document}\n")
    elif tabletype == 'fiducial_fit_metrics':
        texfile.write("    \end{tabu}\n")
        texfile.write("  \end{center}\n")
        texfile.write("  \\vspace{-10pt}\n")
        if truth is not None:
            texfile.write("  \caption{shows the fiducial fit metrics obtained with the %s %s sample for h0 of %s and h1 of %s. The truth is %s.}\n"%(detector, selection, h0, h1, truth))
        else:
            texfile.write("  \caption{shows the fiducial fit metrics obtained with the %s %s sample for h0 of %s and h1 of %s.}\n"%(detector, selection, h0, h1))
        texfile.write("  \label{tab:%s%s%stable}\n"%(detector, selection, tabletype))
        texfile.write("\end{table}\n")
        texfile.write("\n")
        texfile.write("\end{document}\n")


def format_table_line(val, dataval, stddev=None, maximum=None, last=False):
    """Formatting the numbers to look nice is awkard so do it in its own
    function"""
    line = ""
    if stddev is not None:
        if (np.abs(stddev) < 1e-2) and (stddev != 0.0):
            line += r'$%.3e\pm%.3e$ &'%(maximum, stddev)
        else:
            line += r'$%.3g\pm%.3g$ &'%(maximum, stddev)
    else:
        if maximum is not None:
            raise ValueError("Both stddev and maximum should be None or "
                             "specified")
        else:
            line += "-- &"
    if (np.abs(val) < 1e-2) and (val != 0.0):
        line += "%.3e"%val
    else:
        line += "%.3g"%val
    if dataval is not None:
        line += " &"
        if isinstance(dataval, basestring):
            line += "%s"%dataval
        else:
            delta = val - dataval
            if (np.abs(delta) < 1e-2) and (delta != 0.0):
                line += "%.3e"%delta
            else:
                line += "%.3g"%delta
    if not last:
        line += " &"
    return line


def make_fiducial_fits(data, fid_data, labels, all_params, detector,
                       selection, outdir):
    """Make tex files which can be then be compiled in to tables showing the
    two fiducial fits and, if applicable, how they compare to what was
    injected."""
    outdir = os.path.join(outdir, 'FiducialFits')
    mkdir(outdir)

    # Make output file to write to
    paramfilename = ("true_%s_%s_%s_fiducial_fit_params.tex"
                     %(labels['data_name'],
                       detector,
                       selection))
    paramfile = os.path.join(outdir, paramfilename)
    pf = open(paramfile, 'w')
    write_latex_preamble(texfile=pf)

    h0_params = fid_data[
        ('h0_fit_to_toy_%s_asimov'%labels['data_name'])
    ]['params']
    h1_params = fid_data[
        ('h1_fit_to_toy_%s_asimov'%labels['data_name'])
    ]['params']

    if 'data_params' in all_params.keys():
        data_params = {}
        for pkey in all_params['data_params'].keys():
            data_params[pkey] = all_params['data_params'][pkey]['value']
        injected = True
    else:
        data_params = None
        injected = False

    setup_latex_table(
        texfile=pf,
        tabletype='fiducial_fit_params',
        injected=injected
    )

    for param in h0_params.keys():
        
        # Get the units for this parameter
        val, param_units = parse_pint_string(
            pint_string=fid_data[
                'h0_fit_to_toy_%s_asimov'%labels['data_name']
            ]['params'][param]
        )
        # Get priors if they exists
        if 'gaussian' in all_params['h0_params'][param]['prior']:
            h0stddev, h0maximum = extract_gaussian(
                prior_string=all_params['h0_params'][param]['prior'],
                units=param_units
            )
        else:
            h0stddev = None
        if 'gaussian' in all_params['h1_params'][param]['prior']:
            h1stddev, h1maximum = extract_gaussian(
                prior_string=all_params['h1_params'][param]['prior'],
                units=param_units
            )
        else:
            h1stddev = None
        # Include injected parameter, fitted parameters and differences with
        # appropriate formatting.
        if data_params is not None:
            tableline = "      "
            tableline += "%s "%tex_axis_label(param)
            if param == 'deltam31':
                tableline += r" / $10^{-3}$ "
            if param_units != 'dimensionless':
                tableline += "(%s) &"%tex_axis_label(param_units)
            else:
                tableline += "&"
            if param in data_params.keys():
                dataval = extract_paramval(
                    injparams=data_params,
                    systkey=param
                )
                if param == 'deltam31':
                    dataval *= 1000.0
                if (np.abs(dataval) < 1e-2) and (dataval != 0.0):
                    tableline += "%.3e &"%dataval
                else:
                    tableline += "%.3g &"%dataval
            # If no injected parameter, show this and the deltas with a line
            else:
                dataval = '--'
                tableline += "%s &"%dataval
            h0val = extract_paramval(
                injparams=h0_params,
                systkey=param
            )
            if param == 'deltam31':
                h0val *= 1000.0
            if h0stddev is not None:
                tableline += format_table_line(val=h0val, dataval=dataval,
                                               stddev=h0stddev,
                                               maximum=h0maximum)
            else:
                tableline += format_table_line(val=h0val, dataval=dataval)
            h1val = extract_paramval(
                injparams=h1_params,
                systkey=param
            )
            if param == 'deltam31':
                h1val *= 1000.0
            if h1stddev is not None:
                tableline += format_table_line(val=h1val, dataval=dataval,
                                               stddev=h1stddev,
                                               maximum=h1maximum, last=True)
            else:
                tableline += format_table_line(val=h1val, dataval=dataval,
                                               last=True)
            tableline += " \\\\ \hline\n"
            pf.write(tableline)
        # If no injected parameters it's much simpler
        else:
            h0val = extract_paramval(
                injparams=h0_params,
                systkey=param
            )
            h1val = extract_paramval(
                injparams=h1_params,
                systkey=param
            )
            if (np.abs(h0val) < 1e-2) and (h0val != 0.0):
                pf.write("    %s & %.3e & %.3e\n"
                         % (tex_axis_label(param), h0val, h1val))
            else:
                pf.write("    %s & %.3g & %.3g\n"
                         % (tex_axis_label(param), h0val, h1val))

    end_latex_file(
        texfile=pf,
        tabletype='fiducial_fit_params',
        detector=detector,
        selection=selection,
        h0=tex_axis_label(labels['h0_name']),
        h1=tex_axis_label(labels['h1_name']),
        truth=tex_axis_label(labels['data_name'])
    )

    # Make output file to write to
    metricfilename = ("true_%s_%s_%s_fiducial_fit_metrics.tex"
                      %(labels['data_name'], detector, selection))
    metricfile = os.path.join(outdir, metricfilename)
    mf = open(metricfile, 'w')
    write_latex_preamble(texfile=mf)

    setup_latex_table(
        texfile=mf,
        tabletype='fiducial_fit_metrics'
    )

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

    metric_type = data['h0_fit_to_h0_fid']['metric_val']['type']
    metric_type_pretty = tex_axis_label(metric_type)
    # In the case of likelihood, the maximum metric is the better fit.
    # With chi2 metrics the opposite is true, and so we must multiply
    # everything by -1 in order to apply the same treatment.
    if 'chi2' not in metric_type:
        logging.info("Converting likelihood metric to chi2 equivalent.")
        h0_fid_metric *= -1
        h1_fid_metric *= -1

    # If truth is known, report the fits the correct way round
    if labels['data_name'] is not None:
        if labels['data_name'] in labels['h0_name']:
            delta = h1_fid_metric-h0_fid_metric
        elif labels['data_name'] in labels['h1_name']:
            delta = h0_fid_metric-h1_fid_metric
        else:
            logging.warning("Truth is known but could not be identified in "
                            "either of the hypotheses. The difference between"
                            " the best fit metrics will just be reported as "
                            "positive and so will not necessarily reflect if "
                            "the truth was recovered.")
            if h1_fid_metric > h0_fid_metric:
                delta = h0_fid_metric-h1_fid_metric
            else:
                delta = h1_fid_metric-h0_fid_metric
    # Else just report it as delta between best fits
    else:
        if h1_fid_metric > h0_fid_metric:
            delta = h0_fid_metric-h1_fid_metric
        else:
            delta = h1_fid_metric-h0_fid_metric
    # Write this in the file
    mf.write("    %.3g & %.3g & %.3g \\\\ \hline\n"%(h0_fid_metric, h1_fid_metric, delta))
    # Then end the file
    end_latex_file(
        texfile=mf,
        tabletype='fiducial_fit_metrics',
        detector=detector,
        selection=selection,
        h0=tex_axis_label(labels['h0_name']),
        h1=tex_axis_label(labels['h1_name']),
        truth=tex_axis_label(labels['data_name'])
    )


def plot_individual_posterior(data, data_params, h0_params, h1_params,
                              all_params, labels, h0label, h1label, systkey,
                              fhkey, subplotnum=None):
    """Use matplotlib to make a histogram of the vals contained in data. The
    injected value will be plotted along with, where appropriate, the "wrong
    hypothesis" fiducial fit and the prior. The axis labels and the legend are
    taken care of in here. The optional subplotnum argument can be given in the
    combined case so that the y-axis label only get put on when appropriate."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    if systkey == 'metric_val':
        metric_type = data['type']
    systvals = np.array(data['vals'])
    units = data['units']

    hypo = fhkey.split('_')[0]
    fid = fhkey.split('_')[-2]

    plt.grid(axis='y', zorder=0)
    plt.hist(
        systvals,
        bins=10,
        histtype='bar',
        color='darkblue',
        alpha=0.9,
        zorder=3
    )

    # Add injected and hypothesis fit lines
    if not systkey == 'metric_val':
        if data_params is not None:
            if systkey in data_params.keys():
                injval, injlabelproper = extract_paramval(
                    injparams=data_params,
                    systkey=systkey,
                    fid_label=labels['%s_name'%fid],
                    hypo_label=labels['%s_name'%hypo],
                    paramlabel='Injected Value'
                )
                plt.axvline(
                    injval,
                    color='r',
                    linewidth=2,
                    label=injlabelproper,
                    zorder=5
                )
            else:
                injval = None
        else:
            injval = None
        if fid == 'h0':
            fitval, fitlabelproper = extract_paramval(
                injparams=h0_params,
                systkey=systkey,
                fid_label=labels['%s_name'%fid],
                hypo_label=labels['%s_name'%hypo],
                paramlabel=h0label
            )
        elif fid == 'h1':
            fitval, fitlabelproper = extract_paramval(
                injparams=h1_params,
                systkey=systkey,
                fid_label=labels['%s_name'%fid],
                hypo_label=labels['%s_name'%hypo],
                paramlabel=h1label
            )
        else:
            raise ValueError("I got a hypothesis %s. Expected h0 or h1 only."
                             %fid)
        if injval is not None:
            if fitval != injval:
                plt.axvline(
                    fitval,
                    color='g',
                    linewidth=2,
                    label=fitlabelproper,
                    zorder=5
                )
        else:
            plt.axvline(
                fitval,
                color='g',
                linewidth=2,
                label=fitlabelproper,
                zorder=5
            )
    # Add shaded region for prior, if appropriate
    # TODO - Deal with non-gaussian priors
    wanted_params = all_params['%s_params'%hypo]
    for param in wanted_params.keys():
        if param == systkey:
            if 'gaussian' in wanted_params[param]['prior']:
                stddev, maximum = extract_gaussian(
                    prior_string=wanted_params[param]['prior'],
                    units=units
                )
                currentxlim = plt.xlim()
                if (np.abs(stddev) < 1e-2) and (stddev != 0.0):
                    priorlabel = (r'Gaussian Prior '
                                  '($%.3e\pm%.3e$)'%(maximum, stddev))
                else:
                    priorlabel = (r'Gaussian Prior '
                                  '($%.3g\pm%.3g$)'%(maximum, stddev))
                plt.axvspan(
                    maximum-stddev,
                    maximum+stddev,
                    color='k',
                    label=priorlabel,
                    ymax=0.1,
                    alpha=0.5,
                    zorder=5
                )
                # Reset xlimits if prior makes it go far off
                if plt.xlim()[0] < currentxlim[0]:
                    plt.xlim(currentxlim[0], plt.xlim()[1])
                if plt.xlim()[1] > currentxlim[1]:
                    plt.xlim(plt.xlim()[0], currentxlim[1])

    # Make axis labels look nice
    if systkey == 'metric_val':
        systname = tex_axis_label(metric_type)
    else:
        systname = tex_axis_label(systkey)
    if not units == 'dimensionless':
        systname += r' (%s)'%tex_axis_label(units)

    plt.xlabel(systname, size='24')
    if subplotnum is not None:
        if (subplotnum-1)%4 == 0:
            plt.ylabel(r'Number of Trials', size='24')
    else:
        plt.ylabel(r'Number of Trials', size='24')
    plt.ylim(0, 1.35*plt.ylim()[1])
    if not systkey == 'metric_val':
        plt.legend(loc='upper left')

    plt.subplots_adjust(left=0.10, right=0.90, top=0.9, bottom=0.11)


def plot_individual_posteriors(data, fid_data, labels, all_params, detector,
                               selection, outdir, formats):
    """Use `plot_individual_posterior` and save each time."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True


    outdir = os.path.join(outdir, 'IndividualPosteriors')
    mkdir(outdir)

    MainTitle = make_main_title(detector, selection, 'Posterior')

    h0_params = fid_data[
        ('h0_fit_to_toy_%s_asimov'%labels['data_name'])
    ]['params']
    h1_params = fid_data[
        ('h1_fit_to_toy_%s_asimov'%labels['data_name'])
    ]['params']

    if 'data_params' in all_params.keys():
        data_params = {}
        for pkey in all_params['data_params'].keys():
            data_params[pkey] = all_params['data_params'][pkey]['value']
    else:
        data_params = None

    for fhkey in data.keys():
        for systkey in data[fhkey].keys():

            hypo = fhkey.split('_')[0]
            fid = fhkey.split('_')[-2]
            FitTitle = make_fit_title(
                labels=labels,
                fid=fid,
                hypo=hypo,
                trials=len(data[fhkey][systkey]['vals'])
            )
            plot_individual_posterior(
                data=data[fhkey][systkey],
                data_params=data_params,
                h0_params=h0_params,
                h1_params=h1_params,
                all_params=all_params,
                labels=labels,
                h0label='%s Fiducial Fit'%labels['h0_name'],
                h1label='%s Fiducial Fit'%labels['h1_name'],
                systkey=systkey,
                fhkey=fhkey
            )

            plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
            save_plot(
                labels=labels,
                fid=fid,
                hypo=hypo,
                detector=detector,
                selection=selection,
                outdir=outdir,
                formats=formats,
                end='%s_posterior'%(systkey)
            )
            plt.close()


def plot_combined_posteriors(data, fid_data, labels, all_params,
                             detector, selection, outdir, formats):
    """Use `plot_individual_posterior` multiple times but just save once all of
    the posteriors for a given combination of h0 and h1 have been plotted on
    the same canvas."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    outdir = os.path.join(outdir, 'CombinedPosteriors')
    mkdir(outdir)

    MainTitle = make_main_title(detector, selection, 'Posteriors')

    h0_params = fid_data[
        ('h0_fit_to_toy_%s_asimov'%labels['data_name'])
    ]['params']
    h1_params = fid_data[
        ('h1_fit_to_toy_%s_asimov'%labels['data_name'])
    ]['params']

    if 'data_params' in all_params.keys():
        data_params = {}
        for pkey in all_params['data_params'].keys():
            data_params[pkey] = all_params['data_params'][pkey]['value']
    else:
        data_params = None

    labels['MainTitle'] = MainTitle

    for fhkey in data.keys():

        # Set up multi-plot
        num_rows = get_num_rows(data[fhkey], omit_metric=False)
        plt.figure(figsize=(20, 5*num_rows+2))
        subplotnum = 1

        for systkey in data[fhkey].keys():

            hypo = fhkey.split('_')[0]
            fid = fhkey.split('_')[-2]
            FitTitle = make_fit_title(
                labels=labels,
                fid=fid,
                hypo=hypo,
                trials=len(data[fhkey][systkey]['vals'])
            )
            plt.subplot(num_rows, 4, subplotnum)

            plot_individual_posterior(
                data=data[fhkey][systkey],
                data_params=data_params,
                h0_params=h0_params,
                h1_params=h1_params,
                all_params=all_params,
                labels=labels,
                h0label='%s Fiducial Fit'%labels['h0_name'],
                h1label='%s Fiducial Fit'%labels['h1_name'],
                systkey=systkey,
                fhkey=fhkey,
                subplotnum=subplotnum
            )

            subplotnum += 1

        plt.suptitle(MainTitle+r'\\'+FitTitle, fontsize=36)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        save_plot(
            labels=labels,
            fid=fid,
            hypo=hypo,
            detector=detector,
            selection=selection,
            outdir=outdir,
            formats=formats,
            end='posteriors'
        )
        plt.close()


def plot_individual_scatter(xdata, ydata, labels, xsystkey, ysystkey,
                            subplotnum=None, num_rows=None, plot_cor=True):
    """Use matplotlib to make a scatter plot of the vals contained in xdata and
    ydata. The correlation will be calculated and the plot will be annotated
    with this. Axis labels are done in here too. The optional subplotnum
    argument can be given in the combined case so that the y-axis label only
    get put on when appropriate."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

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
            logging.warn(
                'Parameter %s appears to not have been varied. i.e. all of the'
                ' values in the set are the same. This will lead to NaN in the'
                ' correlation calculation and so it will not be done.',
                xsystkey
            )
        if len(set(yvals)) == 1:
            logging.warn(
                'Parameter %s appears to not have been varied. i.e. all of the'
                ' values in the set are the same. This will lead to NaN in the'
                ' correlation calculation and so it will not be done.',
                ysystkey
            )
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


def plot_individual_scatters(data, labels, detector, selection,
                             outdir, formats):
    """Use `plot_individual_scatter` and save every time."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    outdir = os.path.join(outdir, 'IndividualScatterPlots')
    mkdir(outdir)

    MainTitle = make_main_title(detector, selection, 'Correlation Plot')

    for fhkey in data.keys():
        for xsystkey in data[fhkey].keys():
            if not xsystkey == 'metric_val':
                for ysystkey in data[fhkey].keys():
                    if (ysystkey != 'metric_val') and (ysystkey != xsystkey):

                        hypo = fhkey.split('_')[0]
                        fid = fhkey.split('_')[-2]
                        FitTitle = make_fit_title(
                            labels=labels,
                            fid=fid,
                            hypo=hypo,
                            trials=len(data[fhkey][xsystkey]['vals'])
                        )

                        plot_individual_scatter(
                            xdata=data[fhkey][xsystkey],
                            ydata=data[fhkey][ysystkey],
                            labels=labels,
                            xsystkey=xsystkey,
                            ysystkey=ysystkey
                        )

                        plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
                        save_plot(
                            labels=labels,
                            fid=fid,
                            hypo=hypo,
                            detector=detector,
                            selection=selection,
                            outdir=outdir,
                            formats=formats,
                            end='%s_%s_scatter_plot'%(xsystkey, ysystkey)
                        )
                        plt.close()


def plot_combined_individual_scatters(data, labels, detector,
                                      selection, outdir, formats):
    """Use `plot_individual_scatter` and save once all of the scatter plots for
    a single systematic with every other systematic have been plotted on the
    same canvas for each h0 and h1 combination."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    outdir = os.path.join(outdir, 'CombinedScatterPlots')
    mkdir(outdir)

    MainTitle = make_main_title(detector, selection, 'Correlation Plot')

    for fhkey in data.keys():
        for xsystkey in data[fhkey].keys():
            if not xsystkey == 'metric_val':

                # Set up multi-plot
                num_rows = get_num_rows(data[fhkey], omit_metric=True)
                plt.figure(figsize=(20, 5*num_rows+2))
                subplotnum = 1

                for ysystkey in data[fhkey].keys():
                    if (ysystkey != 'metric_val') and (ysystkey != xsystkey):

                        hypo = fhkey.split('_')[0]
                        fid = fhkey.split('_')[-2]
                        FitTitle = make_fit_title(
                            labels=labels,
                            fid=fid,
                            hypo=hypo,
                            trials=len(data[fhkey][xsystkey]['vals'])
                        )

                        plt.subplot(num_rows, 4, subplotnum)

                        plot_individual_scatter(
                            xdata=data[fhkey][xsystkey],
                            ydata=data[fhkey][ysystkey],
                            labels=labels,
                            xsystkey=xsystkey,
                            ysystkey=ysystkey,
                            subplotnum=subplotnum,
                            num_rows=num_rows
                        )

                        subplotnum += 1

                plt.suptitle(MainTitle+r'\\'+FitTitle, fontsize=36)
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)
                save_plot(
                    labels=labels,
                    fid=fid,
                    hypo=hypo,
                    detector=detector,
                    selection=selection,
                    outdir=outdir,
                    formats=formats,
                    end='%s_scatter_plot'%(xsystkey)
                )
                plt.close()


def plot_combined_scatters(data, labels, detector, selection, outdir, formats):
    """Use `plot_individual_scatter` and save once every scatter plot has been
    plotted on a single canvas for each of the h0 and h1 combinations."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    outdir = os.path.join(outdir, 'CombinedScatterPlots')
    mkdir(outdir)

    MainTitle = make_main_title(detector, selection, 'Correlation Plot')

    for fhkey in data.keys():
        # Systematic number is one less than number of keys since this also
        # contains the metric_val entry
        SystNum = len(data[fhkey].keys())-1
        # Set up multi-plot
        plt.figure(figsize=(3.5*(SystNum-1), 3.5*(SystNum-1)))
        subplotnum = (SystNum-1)*(SystNum-1)+1
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
                            FitTitle = make_fit_title(
                                labels=labels,
                                fid=fid,
                                hypo=hypo,
                                trials=len(data[fhkey][xsystkey]['vals'])
                            )

                            plt.subplot(SystNum-1, SystNum-1, subplotnum)

                            plot_individual_scatter(
                                xdata=data[fhkey][xsystkey],
                                ydata=data[fhkey][ysystkey],
                                labels=labels,
                                xsystkey=xsystkey,
                                ysystkey=ysystkey,
                                plot_cor=False
                            )

        plt.suptitle(MainTitle+r'\\'+FitTitle, fontsize=120)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        save_plot(
            labels=labels,
            fid=fid,
            hypo=hypo,
            detector=detector,
            selection=selection,
            outdir=outdir,
            formats=formats,
            end='all_scatter_plots'
        )
        plt.close()


def plot_correlation_matrices(data, labels, detector, selection,
                              outdir, formats):
    """Plot the correlation matrices since the individual scatter plots are a
    pain to interpret on their own. This will plot them with a colour scale
    and, if the user has the PathEffects module then it will also write the
    values on the bins. If a number is invalid it will come up bright green.
    """
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    try:
        import matplotlib.patheffects as PathEffects
        logging.warn('PathEffects could be imported, so the correlation values'
                     ' will be written on the bins. This is slow.')
        pe = True
    except ImportError:
        logging.warn('PathEffects could not be imported, so the correlation'
                     ' values will not be written on the bins.')
        pe = False

    outdir = os.path.join(outdir, 'CorrelationMatrices')
    mkdir(outdir)

    MainTitle = make_main_title(detector, selection, 'Correlation Coefficients')
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
                    if ysystkey != 'metric_val':
                        hypo = fhkey.split('_')[0]
                        fid = fhkey.split('_')[-2]
                        FitTitle = make_fit_title(
                            labels=labels,
                            fid=fid,
                            hypo=hypo,
                            trials=len(data[fhkey][xsystkey]['vals'])
                        )
                        # Calculate correlation
                        xvals = np.array(data[fhkey][xsystkey]['vals'])
                        yvals = np.array(data[fhkey][ysystkey]['vals'])
                        msg = ('Parameter %s appears to not have been varied.'
                               ' i.e. all of the values in the set are the'
                               ' same. This will lead to NaN in the'
                               ' correlation calculation and so it will not be'
                               ' done.')
                        if len(set(xvals)) == 1:
                            logging.warn(msg, xsystkey)
                        if len(set(yvals)) == 1:
                            logging.warn(msg, ysystkey)
                        if len(set(xvals)) != 1 and len(set(yvals)) != 1:
                            rho, pval = spearmanr(xvals, yvals)
                        else:
                            rho = np.nan
                        all_corr_values.append(rho)
                all_corr_lists.append(all_corr_values)

        all_corr_nparray = np.ma.masked_invalid(np.array(all_corr_lists))
        # Plot it!
        palette = plt.cm.RdBu
        palette.set_bad('lime', 1.0)
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
        plt.subplots_adjust(bottom=0.30, left=-0.30, right=1.05, top=0.9)
        plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
        save_plot(
            labels=labels,
            fid=fid,
            hypo=hypo,
            detector=detector,
            selection=selection,
            outdir=outdir,
            formats=formats,
            end='correlation_matrix'
        )
        if pe:
            for i in range(0, len(all_corr_nparray)):
                for j in range(0, len(all_corr_nparray[0])):
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
        save_plot(
            labels=labels,
            fid=fid,
            hypo=hypo,
            detector=detector,
            selection=selection,
            outdir=outdir,
            formats=formats,
            end='correlation_matrix_values'
        )
        plt.close()


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
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
        '--threshold', type=float, default=0.0,
        help='''Sets the threshold for which to remove 'outlier' trials.
        Ideally this will not be needed at all, but it is there in case of
        e.g. failed minimiser. The higher this value, the more outliers will
        be included. Do not set this parameter if you want all trials to be
        included.'''
    )
    parser.add_argument(
        '--extra-point', type=str, action='append',
        help='''Extra lines to be added to the LLR plots. This is useful, for
        example, when you wish to add specific LLR fit values to the plot for
        comparison. These should be supplied as a single value e.g. x1 or
        as a path to a file with the value provided in one column that can be
        intepreted by numpy genfromtxt. Repeat this argument in conjunction
        with the extra points label below to specify multiple (and uniquely
        identifiable) sets of extra points.'''
    )
    parser.add_argument(
        '--extra-point-label', type=str, action='append',
        help='''The label(s) for the extra points above.'''
    )
    parser.add_argument(
        '--outdir', metavar='DIR', type=str, default=None,
        help='''Store all output plots to this directory. This will make
        further subdirectories, if needed, to organise the output plots.'''
    )
    parser.add_argument(
        '--pdf', action='store_true',
        help='''Produce pdf plot(s).'''
    )
    parser.add_argument(
        '--png', action='store_true',
        help='''Produce png plot(s).'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='''set verbosity level'''
    )
    return parser.parse_args()


def main():
    """Main"""
    args = parse_args()

    # NOTE: Removing extraneous args that won't get passed to instantiate the
    # HypoTesting object via dictionary's `pop()` method.

    set_verbosity(args.v)

    extra_points = args.extra_point
    extra_points_labels = args.extra_point_label
    if extra_points is not None:
        if extra_points_labels is not None:
            if len(extra_points) != len(extra_points_labels):
                raise ValueError(
                    'You must specify at least one label for each set of extra'
                    ' points. Got %i label(s) for %s set(s) of extra points.'
                    % (len(extra_points), len(extra_points_labels))
                )
        else:
            raise ValueError(
                'You have specified %i set(s) of extra points but no labels to'
                ' go with them.' % len(extra_points)
            )
    else:
        if extra_points_labels is not None:
            raise ValueError(
                'You have specified %i label(s) for extra points but no set(s)'
                ' of extra points.' % len(extra_points_labels)
            )
    formats = []
    if args.png:
        formats.append('png')
    if args.pdf:
        formats.append('pdf')

    if args.asimov:
        data_sets, all_params, labels, minimiser_info = extract_trials(
            logdir=args.dir,
            fluctuate_fid=False,
            fluctuate_data=False
        )
        od = data_sets.values()[0]
        #if od['h1_fit_to_h0_fid']['fid_asimov']['metric_val'] > od['h0_fit_to_h1_fid']['fid_asimov']['metric_val']:
        print np.sqrt(np.abs(
            od['h1_fit_to_h0_fid']['fid_asimov']['metric_val'] -
            od['h0_fit_to_h1_fid']['fid_asimov']['metric_val']
        ))
        return

    # Otherwise: LLR analysis
    if args.outdir is None:
        raise ValueError('Must specify --outdir when processing LLR results.')

    if len(formats) > 0:
        logging.info('Files will be saved in format(s) %s', formats)
    else:
        raise ValueError('Must specify a plot file format, either --png or'
                         ' --pdf, when processing LLR results.')

    data_sets, all_params, labels, minimiser_info = extract_trials(
        logdir=args.dir,
        fluctuate_fid=True,
        fluctuate_data=False
    )

    trial_nums = data_sets['toy_%s_asimov'%labels.dict['data_name']]['h0_fit_to_h1_fid'].keys()

    fid_values = extract_fid_data(data_sets)
    values = extract_data(data_sets)

    for injkey in values.keys():
        if args.threshold != 0.0:
            logging.info('Outlying trials will be removed with a '
                         'threshold of %.2f', args.threshold)
            purge_failed_jobs(
                data=values[injkey],
                trial_nums=np.array(trial_nums),
                thresh=args.threshold
            )
        else:
            logging.info('All trials will be included in the analysis.')

        if len(trial_nums) != 1:
            make_llr_plots(
                data=values[injkey],
                fid_data=fid_values[injkey],
                labels=labels.dict,
                detector=args.detector,
                selection=args.selection,
                extra_points=extra_points,
                extra_points_labels=extra_points_labels,
                outdir=args.outdir,
                formats=formats
            )

        make_fiducial_fits(
            data=values[injkey],
            fid_data=fid_values[injkey],
            labels=labels.dict,
            all_params=all_params,
            detector=args.detector,
            selection=args.selection,
            outdir=args.outdir
        )

        if args.fit_information:
            plot_fit_information(
                minimiser_info=minimiser_info[injkey],
                labels=labels.dict,
                detector=args.detector,
                selection=args.selection,
                outdir=args.outdir,
                formats=formats
            )

        if args.individual_posteriors:
            plot_individual_posteriors(
                data=values[injkey],
                fid_data=fid_values[injkey],
                labels=labels.dict,
                all_params=all_params,
                detector=args.detector,
                selection=args.selection,
                outdir=args.outdir,
                formats=formats
            )

        if args.combined_posteriors:
            plot_combined_posteriors(
                data=values[injkey],
                fid_data=fid_values[injkey],
                labels=labels.dict,
                all_params=all_params,
                detector=args.detector,
                selection=args.selection,
                outdir=args.outdir,
                formats=formats
            )

        if args.individual_scatter:
            plot_individual_scatters(
                data=values[injkey],
                labels=labels.dict,
                detector=args.detector,
                selection=args.selection,
                outdir=args.outdir,
                formats=formats
            )

        if args.combined_individual_scatter:
            plot_combined_individual_scatters(
                data=values[injkey],
                labels=labels.dict,
                detector=args.detector,
                selection=args.selection,
                outdir=args.outdir,
                formats=formats
            )

        if args.combined_scatter:
            plot_combined_scatters(
                data=values[injkey],
                labels=labels.dict,
                detector=args.detector,
                selection=args.selection,
                outdir=args.outdir,
                formats=formats
            )

        if args.correlation_matrix:
            plot_correlation_matrices(
                data=values[injkey],
                labels=labels.dict,
                detector=args.detector,
                selection=args.selection,
                outdir=args.outdir,
                formats=formats
            )


if __name__ == '__main__':
    main()
