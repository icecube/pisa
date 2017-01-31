#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
This script/module will load the Analysis class from analysis.py and use it to
perform a blind fit. That is, the best fit point will be found between data and
our simulations with ONLY the fit metric being reported. This is for the NMO analysis, and the parameters for blind fit have been predetermined as:

    1) The fit metric is the only parameter to be unblinded with the DRAGON 
       sample. For the GRECO sample, the atmospheric muon scale may also be 
       unblinded in case of having to understand data/MC disagreement.

    2) The tau normalisation is to be fixed to 1. It is not a systematic to be 
       used in the final analysis and so should not significantly affect the 
       fits enough to be included here.

    3) Only the hypothesis of normal ordering is to be fit here. The delta-chi2
       between NO and IO is absolutely not going to be big enough that this 
       will give a problem with these blind fits.

    4) The threshold for a "good" fit is a 5% p-value.
"""


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Mapping, OrderedDict
import os
import pint

from pisa import ureg
from pisa.analysis.analysis import Analysis
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.fileio import from_file, to_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.random_numbers import get_random_state
from pisa.utils.resources import find_resource
from pisa.utils.stats import ALL_METRICS


def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='''Perform the LLR analysis for calculating the NMO
        sensitivity of the distribution made from data-settings compared with
        hypotheses generated from template-settings.

        Currently the output should be a json file containing the dictionary
        of best fit and likelihood values.'''
    )
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
        '--mc-pipeline', required=True,
        type=str, action='append', metavar='PIPELINE_CFG',
        help='''Settings for the generation of mc distributions; repeat this 
        argument to specify multiple pipelines.'''
    )
    parser.add_argument(
        '--data-pipeline',
        type=str, default=None, metavar='PIPELINE_CFG',
        help='''Settings for the generation of data distributions. This must be
        a singular argument, since here this is genuine data.'''
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
    
    init_args_d['check_octant'] = True
    init_args_d['data_is_data'] = True
    init_args_d['store_minimizer_history'] = False
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
    else:
        init_args_d['other_metrics'] = None

    # Normalize and convert `*_pipeline` filenames; store to `*_maker`
    # (which is argument naming convention that HypoTesting init accepts).
    for maker in ['mc', 'data']:
        try:
            filenames = init_args_d.pop(maker + '_pipeline')
        except:
            filenames = None
        if isinstance(filenames,str):
            filenames = [filenames]    
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

    mc_maker = DistributionMaker(init_args_d['mc_maker'])
    data_maker = DistributionMaker(init_args_d['data_maker'])

    # Read in minimizer settings
    if isinstance(init_args_d['minimizer_settings'], basestring):
        init_args_d['minimizer_settings'] = \
            from_file(init_args_d['minimizer_settings'])
    assert isinstance(init_args_d['minimizer_settings'], Mapping)

    # Instantiate the analysis object
    analysis = Analysis()

    best_fit_no_hypo, alt_fit_no_hypo = analysis.fit_hypo(
        data_dist = data_maker.get_outputs(return_sum=True),
        hypo_maker = mc_maker,
        hypo_param_selections = 'nh',
        metric = init_args_d['metric'],
        minimizer_settings = init_args_d['minimizer_settings'],
        check_octant = True,
        other_metrics = init_args_d['other_metrics'],
        blind = True,
        pprint = init_args_d['pprint']
    )
    # Immediately overwrite the alt_fit object just so as not to do
    # anything wrong...
    alt_fit_no_hypo = None

    if not os.path.exists(init_args_d['logdir']):
        logging.info('Making output directory %s'%init_args_d['logdir'])
        os.makedirs(init_args_d['logdir'])

    # Stealing log_fit function from hypo_testing
    serialize = ['metric', 'metric_val', 'params', 'minimizer_time',
                 'detailed_metric_info', 'minimizer_metadata']
    info = OrderedDict()
    for k, v in best_fit_no_hypo.iteritems():
        if k not in serialize:
            continue
        if k == 'params':
            d = OrderedDict()
            for param in v.free:
                d[param.name] = str(param.value)
            v = d
        if k == 'minimizer_metadata':
            if 'hess_inv' in v:
                try:
                    v['hess_inv'] = v['hess_inv'].todense()
                except AttributeError:
                    v['hess_inv'] = v['hess_inv']
        if isinstance(v, pint.quantity._Quantity):
            v = str(v)
        info[k] = v
    info['params'] = mc_maker.params.free.names
    to_file(
        info,
        os.path.join(
            init_args_d['logdir'],
            'nmo_blind_fit_result.json'
        )
    )


if __name__ == '__main__':
    main()
