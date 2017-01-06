#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This script/module will load the HypoTesting class from hypo_testing.py and
use it to do a minimiser study in pseudo-data. A pseudo-data trial will be 
created based on some set of injected parameters and it will be fit with both 
hypotheses. 1D scans will then be performed along all of the parameters to 
ensure that the true minimum was actually found.

"""


from argparse import ArgumentParser
import os

from pisa import ureg
from pisa.analysis.hypo_testing import HypoTesting, Labels
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.stats import ALL_METRICS


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '-d', '--logdir', required=True,
        metavar='DIR', type=str,
        help='Directory into which to store results and metadata.'
    )
    parser.add_argument(
        '-m', '--minimizer-settings',
        type=str, metavar='MINIMIZER_CFG', required=True,
        help='''Settings related to the optimizer used in the LLR analysis.'''
    )
    parser.add_argument(
        '--no-octant-check',
        action='store_true',
        help='''Disable fitting hypotheses in theta23 octant opposite initial
        octant.'''
    )
    parser.add_argument(
        '--h0-pipeline', required=True,
        type=str, action='append', metavar='PIPELINE_CFG',
        help='''Settings for the generation of hypothesis h0
        distributions; repeat this argument to specify multiple pipelines.'''
    )
    parser.add_argument(
        '--h0-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated (no spaces) list of param selectors to apply to
        hypothesis h0's distribution maker's pipelines.'''
    )
    parser.add_argument(
        '--h0-name',
        type=str, metavar='NAME', default=None,
        help='''Name for hypothesis h0. E.g., "NO" for normal
        ordering in the neutrino mass ordering analysis. Note that the name
        here has no bearing on the actual process, so it's important that you
        be careful to use a name that appropriately identifies the
        hypothesis.'''
    )
    parser.add_argument(
        '--h1-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated (no spaces) list of param selectors to apply to
        hypothesis h1 distribution maker's pipelines.'''
    )
    parser.add_argument(
        '--h1-name',
        type=str, metavar='NAME', default=None,
        help='''Name for hypothesis h1. E.g., "IO" for inverted
        ordering in the neutrino mass ordering analysis. Note that the name
        here has no bearing on the actual process, so it's important that you
        be careful to use a name that appropriately identifies the
        hypothesis.'''
    )
    parser.add_argument(
        '--data-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated list of param selectors to apply to the data
        distribution maker's pipelines. Pipeline always assumed to be 
        --h0-pipeline for this test. If --data-param-selections is not 
        specified it is copied from --h0-param-selections.'''
    )
    parser.add_argument(
        '--data-name',
        type=str, metavar='NAME', default=None,
        help='''Name for the data. E.g., "NO" for normal ordering in the
        neutrino mass ordering analysis. Note that the name here has no bearing
        on the actual process, so it's important that you be careful to use a
        name that appropriately identifies the hypothesis.'''
    )
    parser.add_argument(
        '--metric',
        type=str, default=None, metavar='METRIC',
        help='''Name of metric to use for optimizing the fit.'''
    )
    parser.add_argument(
        '--other-metric',
        type=str, default=None, metavar='METRIC', action='append',
        choices=['all'] + sorted(ALL_METRICS),
        help='''Name of another metric to evaluate at the best-fit point. Must
        be either "all" or a metric specified in ALL_METRICS. Repeat this
        argument (or use "all") to specify multiple metrics.'''
    )
    parser.add_argument(
        '--num-fid-trials',
        type=int, default=1,
        help='''Number of pseudo trials to run. Each one will be a minimisation
        on a different pseudo-experiment and a resulting scan to ensure the 
        true minimum was found. WARNING - This will likely be a lengthy 
        process, so you probably don't want to set this very high.'''
    )
    parser.add_argument(
        '--fid-start-ind',
        type=int, default=0,
        help='''Trial start index. Set this if you are saving files from 
        multiple runs in to the same log directory otherwise files may end up 
        being overwritten!'''
    )
    parser.add_argument(
        '--allow-dirty',
        action='store_true',
        help='''Warning: Use with caution. (Allow for run despite dirty
        repository.)'''
    )
    parser.add_argument(
        '--allow-no-git-info',
        action='store_true',
        help='''*** DANGER! Use with extreme caution! (Allow for run despite
        complete inability to track provenance of code.)'''
    )
    parser.add_argument(
        '--no-minimizer-history',
        action='store_true',
        help='''Do not store minimizer history (steps). This behavior is also
        enforced if --blind is specified.'''
    )
    parser.add_argument(
        '--pprint',
        action='store_true',
        help='''Live-updating one-line vew of metric and parameter values. (The
        latter are not displayed if --blind is specified.)'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    return parser.parse_args()


# TODO: make this work with Python package resources, not merely absolute
# paths! ... e.g. hash on the file or somesuch?
# TODO: move to a central loc prob. in utils
def normcheckpath(path, checkdir=False):
    normpath = find_resource(path)
    if checkdir:
        kind = 'dir'
        check = os.path.isdir
    else:
        kind = 'file'
        check = os.path.isfile

    if not check(normpath):
        raise IOError('Path "%s" which resolves to "%s" is not a %s.'
                      %(path, normpath, kind))
    return normpath


def main():
    args = parse_args()
    init_args_d = vars(args)

    # NOTE: Removing extraneous args that won't get passed to instantiate the
    # HypoTesting object via dictionary's `pop()` method.

    set_verbosity(init_args_d.pop('v'))
    num_trials = init_args_d['num_fid_trials']
    start_index = init_args_d['fid_start_ind']
    init_args_d['check_octant'] = not init_args_d.pop('no_octant_check')

    init_args_d['data_is_data'] = False

    init_args_d['store_minimizer_history'] = (
        not init_args_d.pop('no_minimizer_history')
    )

    init_args_d['reset_free'] = False

    other_metrics = init_args_d.pop('other_metric')
    if other_metrics is not None:
        other_metrics = [s.strip().lower() for s in other_metrics]
        if 'all' in other_metrics:
            other_metrics = sorted(ALL_METRICS)
        if init_args_d['metric'] in other_metrics:
            other_metrics.remove(init_args_d['metric'])
        if len(other_metrics) == 0:
            other_metrics = None
        else:
            logging.info('Will evaluate other metrics %s' %other_metrics)
        init_args_d['other_metrics'] = other_metrics

    # Normalize and convert `*_pipeline` filenames; store to `*_maker`
    # (which is argument naming convention that HypoTesting init accepts).
    for maker in ['h0', 'h1', 'data']:
        try:
            filenames = init_args_d.pop(maker + '_pipeline')
        except:
            filenames = None
        if filenames is not None:
            filenames = sorted(
                [normcheckpath(fname) for fname in filenames]
            )
        init_args_d[maker + '_maker'] = filenames

        ps_name = maker + '_param_selections'
        try:
            ps_str = init_args_d[ps_name]
        except:
            ps_str = None
        if ps_str is None:
            ps_list = None
        else:
            ps_list = [x.strip().lower() for x in ps_str.split(',')]
        init_args_d[ps_name] = ps_list

    init_args_d['fluctuate_data'] = False
    init_args_d['fluctuate_fid'] = True

    init_args_d['data_maker'] = init_args_d['h0_maker']
    init_args_d['h1_maker'] = init_args_d['h0_maker']

    init_args_d['h0_maker'] = DistributionMaker(init_args_d['h0_maker'])
    init_args_d['h1_maker'] = DistributionMaker(init_args_d['h1_maker'])
    init_args_d['h1_maker'].select_params(init_args_d['h1_param_selections'])
    init_args_d['data_maker'] = DistributionMaker(init_args_d['data_maker'])
    if init_args_d['data_param_selections'] is None:
        init_args_d['data_param_selections'] = \
            init_args_d['h0_param_selections']
    init_args_d['data_maker'].select_params(
        init_args_d['data_param_selections']
    )
    if init_args_d['data_name'] is None:
        init_args_d['data_name'] = init_args_d['h0_name']

    # Instantiate the analysis object
    hypo_testing = HypoTesting(**init_args_d)
    hypo_testing.setup_logging()
    hypo_testing.write_config_summary()
    hypo_testing.write_minimizer_settings()
    hypo_testing.write_run_info()
    # Run the initial fits
    hypo_testing.generate_data()
    hypo_testing.fit_hypos_to_data()

    for i in range(start_index,(start_index+num_trials)):
        # Run the fiducial fit
        hypo_testing.produce_fid_data()
        hypo_testing.fit_hypos_to_fid()
        # Do the 1D scans with the pseudo experiment as the data
        scanoutdir = os.path.join(
            hypo_testing.config_summary_fpath.split('config_summary.json')[0],
            'scans'
        )
        if not os.path.exists(scanoutdir):
            logging.info('Making output directory for scans %s'%scanoutdir)
            os.makedirs(scanoutdir)
        for param in hypo_testing.h0_maker.params.free:
            # Set the values of the parameters to those found in the
            # appropriate fiducial fit
            hypo_testing.h0_maker.params.free.set_values(
                new_params=hypo_testing.h0_fit_to_h0_fid['params'].free
            )
            h0_fid_h0_hypo_scan = hypo_testing.scan(
                data_dist=hypo_testing.h0_fid_dist,
                hypo_maker=hypo_testing.h0_maker,
                hypo_param_selections=init_args_d['h0_param_selections'],
                metric=init_args_d['metric'],
                param_names=param.name,
                steps=50,
                only_points=None,
                outer=True,
                profile=False,
                minimizer_settings=init_args_d['minimizer_settings'],
                outfile=os.path.join(scanoutdir,
                                     'h0_fid_h0_hypo_%s_scan_%i.json'
                                     %(param.name,i))
            )
            # Parameter is fixed in the scan without being unfixed, so do that
            # here now.
            hypo_testing.h0_maker.params.unfix(param)
            hypo_testing.h0_maker.params.free.set_values(
                new_params=hypo_testing.h0_fit_to_h1_fid['params'].free
            )
            h1_fid_h0_scan = hypo_testing.scan(
                data_dist=hypo_testing.h1_fid_dist,
                hypo_maker=hypo_testing.h0_maker,
                hypo_param_selections=init_args_d['h0_param_selections'],
                metric=init_args_d['metric'],
                param_names=param.name,
                steps=50,
                only_points=None,
                outer=True,
                profile=False,
                minimizer_settings=init_args_d['minimizer_settings'],
                outfile=os.path.join(scanoutdir,
                                     'h1_fid_h0_hypo_%s_scan_%i.json'
                                     %(param.name,i))
            )
            hypo_testing.h0_maker.params.unfix(param)
        for param in hypo_testing.h1_maker.params.free:
            hypo_testing.h1_maker.params.free.set_values(
                new_params=hypo_testing.h1_fit_to_h0_fid['params'].free
            )
            h0_fid_h1_scan = hypo_testing.scan(
                data_dist=hypo_testing.h0_fid_dist,
                hypo_maker=hypo_testing.h1_maker,
                hypo_param_selections=init_args_d['h1_param_selections'],
                metric=init_args_d['metric'],
                param_names=param.name,
                steps=50,
                only_points=None,
                outer=True,
                profile=False,
                minimizer_settings=init_args_d['minimizer_settings'],
                outfile=os.path.join(scanoutdir,
                                     'h0_fid_h1_hypo_%s_scan_%i.json'
                                     %(param.name,i))
            )
            hypo_testing.h1_maker.params.unfix(param)
            hypo_testing.h1_maker.params.free.set_values(
                new_params=hypo_testing.h1_fit_to_h1_fid['params'].free
            )
            h1_fid_h1_scan = hypo_testing.scan(
                data_dist=hypo_testing.h1_fid_dist,
                hypo_maker=hypo_testing.h1_maker,
                hypo_param_selections=init_args_d['h1_param_selections'],
                metric=init_args_d['metric'],
                param_names=param.name,
                steps=50,
                only_points=None,
                outer=True,
                profile=False,
                minimizer_settings=init_args_d['minimizer_settings'],
                outfile=os.path.join(scanoutdir,
                                     'h1_fid_h1_hypo_%s_scan_%i.json'
                                     %(param.name,i))
            )
            hypo_testing.h1_maker.params.unfix(param)
        # Need to advance the fid_ind by 1 here since I'm doing the
        # loop over the trials outside of the hypo_testing object
        hypo_testing.fid_ind += 1
        

if __name__ == '__main__':
    main()
