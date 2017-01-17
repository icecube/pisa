#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This script/module computes significances, etc. from the logfiles recorded by
the `hypo_testing_systtests.py` script. That is, looks at how the fits change for three different N-1 tests:

    1) Where one of the systematics is fixed to the baseline value
    2) Where one of the systematics is injected *off* baseline but fixed *on* 
       baseline in the hypotheses.
    3) Same as 2, but the systematic is not fixed and so the minimiser is 
       allowed to try correct for the incorrect hypothesis.

"""

from __future__ import division

from argparse import ArgumentParser
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import numpy as np
import re

from pisa.analysis.hypo_testing import Labels
from pisa.utils.fileio import from_file, to_file, nsort
from pisa.utils.log import set_verbosity, logging
from pisa.utils.postprocess import tex_axis_label


__all__ = ['extract_trials', 'extract_fit', 'parse_args', 'main']
    

def extract_tests(logdir, fluctuate_fid, fluctuate_data=False):
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
                if 'fixed' in labels.dict['h0_name']:
                    if 'inj' in labels.dict['data_name']:
                        testtype = 'fixwrong'
                        trueordering = labels.dict['data_name'].split('_')[0]
                        direction = labels.dict['data_name'].split('_')[-2]
                        systematic = labels.dict['data_name'].split(
                            '%s_inj_'%trueordering
                        )[-1].split('_%s_wrong'%direction)[0]
                    else:
                        testtype = 'nminusone'
                        trueordering = labels.dict['data_name'].split('_')[0]
                        hypo = labels.dict['h0_name'].split('_')[0]
                        direction = None
                        systematic = labels.dict['h0_name'].split(
                            '%s_fixed_'%hypo
                        )[-1].split('_baseline')[0]
                else:
                    testtype = 'fitwrong'
                    trueordering = labels.dict['data_name'].split('_')[0]
                    direction = labels.dict['data_name'].split('_')[-2]
                    systematic = labels.dict['data_name'].split(
                        '%s_inj_'%trueordering
                    )[-1].split('_%s_wrong'%direction)[0]
                trueordering = 'toy_%s_asimov'%trueordering
                if trueordering not in all_labels.keys():
                    all_labels[trueordering] = {}
                    all_params[trueordering] = {}
                    all_data[trueordering] = {}
                if systematic not in all_labels[trueordering].keys():
                    all_labels[trueordering][systematic] = {}
                    all_params[trueordering][systematic] = {}
                    all_data[trueordering][systematic] = {}
                if direction is not None:
                    if direction not in  all_labels[
                            trueordering][systematic].keys():
                        all_labels[trueordering][systematic][direction] = {}
                        all_params[trueordering][systematic][direction] = {}
                        all_data[trueordering][systematic][direction] = {}

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

                if direction is not None:
                    all_params[trueordering][systematic][direction] = \
                        these_params
                else:
                    all_params[trueordering][systematic] = these_params

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
                if direction is not None:
                    all_data[trueordering][systematic][direction] = this_data
                else:
                    all_data[trueordering][systematic] = this_data
        to_file(all_data, os.path.join(logdir, 'data_sets.pckl'))
        to_file(all_params, os.path.join(logdir, 'all_params.pckl'))
        to_file(all_labels, os.path.join(logdir, 'labels.pckl'))
        
    return all_data, all_params, all_labels, 


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


def extract_relevant_fit_data(fit_data, injkey):
    '''
    Function to extract relevant fit information from Asimov data
    '''
    # Find which hypothesis is the best fit. Since this is an MC study in
    # Asimov, this should also be the injected truth.
    h0_fit = fit_data['h0_fit_to_%s'%injkey]
    h1_fit = fit_data['h1_fit_to_%s'%injkey]
    if h0_fit['metric_val'] > h1_fit['metric_val']:
        bestfit = 'h1'
        altfit = 'h0'
    else:
        bestfit = 'h0'
        altfit = 'h1'
    # Extract the relevant fits
    TO_to_WO_fit = fit_data['%s_fit_to_%s_fid'%(bestfit,altfit)]['fid_asimov']
    WO_to_TO_fit = fit_data['%s_fit_to_%s_fid'%(altfit,bestfit)]['fid_asimov']
    relevant_fit_data = {}
    relevant_fit_data['TO_to_WO_fit'] = TO_to_WO_fit
    relevant_fit_data['WO_to_TO_fit'] = WO_to_TO_fit
    return relevant_fit_data


def make_plots(data, injkey, detector, selection, testtype, outdir):
    if testtype == 'nminusone':
        make_nminusone_plots(
            data=data,
            injkey=injkey,
            detector=detector,
            selection=selection,
            outdir=outdir
        )
    else:
        print "DUNNO yet m8"


def make_nminusone_plots(data, injkey, detector, selection, outdir):
    '''
    Make the N-1 test plot showing the importance of the systematics
    '''
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)
    MainTitle = '%s %s Event Selection N-1 Systematic Test for true %s'%(
        detector, selection, injkey.split('_')[1]
    )
    WO_to_TO_metrics = []
    TO_to_WO_metrics = []
    for testsyst in data.keys():
        WO_to_TO_metrics.append(data[testsyst]['WO_to_TO_fit']['metric_val'])
        TO_to_WO_metrics.append(data[testsyst]['TO_to_WO_fit']['metric_val'])
    WO_to_TO_metrics = np.array(WO_to_TO_metrics)
    TO_to_WO_metrics = np.array(TO_to_WO_metrics)
    significances = calculate_deltachi2_signifiances(
        WO_to_TO_metrics=WO_to_TO_metrics,
        TO_to_WO_metrics=TO_to_WO_metrics
    )
    systnames = []
    for systname in np.array(data.keys())[significances.argsort()]:
        systnames.append(tex_axis_label(systname))
    plt.plot(
        np.linspace(0.5,len(significances)-0.5,len(significances)),
        significances[significances.argsort()],
        linestyle='None',
        marker='x',
        markersize=10
    )
    plt.xticks(
        np.linspace(0.5,len(significances)-0.5,len(significances)),
        systnames,
        rotation=45,
        horizontalalignment='right'
    )
    plt.xlim(0,len(significances))
    plt.xlabel('Fixed Systematic')
    plt.ylabel(r'Asimov Significance ($\sigma$)')
    plt.title(MainTitle, fontsize=16)
    plt.tight_layout()
    SaveName = "true_%s_%s_%s_nminusone_systematic_test.png"%(
        injkey.split('_')[1], detector, selection
    )
    plt.savefig(os.path.join(outdir,SaveName))
    plt.close()


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
    
    data_sets, all_params, labels = extract_tests(
        logdir=args.dir,
        fluctuate_fid=False,
        fluctuate_data=False
    )

    for injkey in data_sets.keys():
        data = {}
        for testsyst in data_sets[injkey].keys():
            data[testsyst] = {}
            # This will be the case of doing systematic tests off baseline
            if len(data_sets[injkey][testsyst].keys()) == 2:
                print "Forthcoming"
            # Otherwise it's a standard N-1 test
            else:
                testtype = 'nminusone'
                data[testsyst] = extract_relevant_fit_data(
                    fit_data=data_sets[injkey][testsyst][injkey],
                    injkey=injkey
                )
        make_plots(
            data=data,
            injkey=injkey,
            detector=detector,
            selection=selection,
            testtype=testtype,
            outdir=outdir
        )
                
        
if __name__ == '__main__':
    main()
