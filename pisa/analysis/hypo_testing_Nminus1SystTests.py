#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This script/module will load the HypoTesting class from hypo_testing.py and
use it to do a systematic study in Asimov. This will take some input pipeline
configuration and then turn each one of the systematics off in turn, doing a new
hypothesis test each time. The user will have the option to fix this systematic
to either the baseline or some shifted value (+/- 1 sigma, or appropriate).

"""


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

from pisa import ureg
from pisa.analysis.hypo_testing import HypoTesting
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import VALID_METRICS
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource


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
        '--h1-pipeline',
        type=str, action='append', default=None, metavar='PIPELINE_CFG',
        help='''Settings for the generation of hypothesis h1 distributions;
        repeat this argument to specify multiple pipelines. If omitted, the
        same settings as specified for --h0-pipeline are used to generate
        hypothesis h1 distributions (and so you have to use the
        --h1-param-selections argument to generate a hypotheses distinct
        from hypothesis h0 but still use h0's distribution maker).'''
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
        '--metric',
        type=str, default=None, metavar='METRIC',
        help='''Name of metric to use for optimizing the fit.'''
    )
    parser.add_argument(
        '--other-metric',
        type=str, default=None, metavar='METRIC', action='append',
        choices=['all'] + sorted(VALID_METRICS),
        help='''Name of another metric to evaluate at the best-fit point. Must
        be either "all" or a metric specified in VALID_METRICS. Repeat this
        argument (or use "all") to specify multiple metrics.'''
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
        '--fix_wrong',
        action='store_true',
        help='''When the systematic is fixed, fix it to some systematically 
        wrong value. This will be either +/- 1 sigma or +/- 10%% if such a 
        definition is impossible.'''
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
    init_args_d['check_octant'] = not init_args_d.pop('no_octant_check')

    init_args_d['data_is_data'] = False

    init_args_d['store_minimizer_history'] = (
        not init_args_d.pop('no_minimizer_history')
    )

    other_metrics = init_args_d.pop('other_metric')
    if other_metrics is not None:
        other_metrics = [s.strip().lower() for s in other_metrics]
        if 'all' in other_metrics:
            other_metrics = sorted(VALID_METRICS)
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
    init_args_d['fluctuate_fid'] = False

    init_args_d['h0_maker'] = DistributionMaker(init_args_d['h0_maker'])

    fix_wrong = init_args_d.pop('fix_wrong')

    # Go through the parameters
    for param in init_args_d['h0_maker'].params:
        # Look for those that are nuisance parameters
        if not param.is_fixed:
            # Take the nominal values of this parameter and the hypotheses
            nominal_value = param.value
            nominal_h0_name = init_args_d['h0_name']
            nominal_h1_name = init_args_d['h1_name']
            # Can fix to a systematically wrong value
            if fix_wrong:
                # Calculate this wrong value based on the prior
                if hasattr(param, 'prior'):
                    # Gaussian priors are easy - just do 1 sigma
                    if param.prior.kind == 'gaussian':
                        param.value = param.value + param.prior.stddev
                    # Else do 10%
                    else:
                        param.value = 1.1 * param.value
                    # Names should reflect this change
                    init_args_d['h0_name'] = init_args_d['h0_name'] + \
                                             '_fixed_pvewrong_%s'%param.name
                    init_args_d['h1_name'] = init_args_d['h1_name'] + \
                                             '_fixed_pvewrong_%s'%param.name
                else:
                    raise TypeError('Param object should have a prior, even '
                                    'if it is uniform. No prior found for '
                                    'param %s'%param.name)
                
                param.is_fixed = True
                # Instantiate the analysis object
                hypo_testing = HypoTesting(**init_args_d)
                # Run the analysis
                hypo_testing.run_analysis()

                # Reset everything
                init_args_d['h0_name'] = nominal_h0_name
                init_args_d['h1_name'] = nominal_h1_name
                param.value = nominal_param_value
                param.is_fixed = False

                if hasattr(param, 'prior'):
                    if param.prior.kind == 'gaussian':
                        param.value = param.value - param.prior.stddev
                    else:
                        param.value = 0.9 * param.value
                    init_args_d['h0_name'] = init_args_d['h0_name'] + \
                                             '_fixed_nvewrong_%s'%param.name
                    init_args_d['h1_name'] = init_args_d['h1_name'] + \
                                             '_fixed_nvewrong_%s'%param.name
                else:
                    raise TypeError('Param object should have a prior, even '
                                    'if it is uniform. No prior found for '
                                    'param %s'%param.name)
                
                param.is_fixed = True
                # Instantiate the analysis object
                hypo_testing = HypoTesting(**init_args_d)
                # Run the analysis
                hypo_testing.run_analysis()

                # Reset everything
                init_args_d['h0_name'] = nominal_h0_name
                init_args_d['h1_name'] = nominal_h1_name
                param.value = nominal_param_value
                param.is_fixed = False
            # Or just to the baseline
            else:
                # Names should reflect this change
                init_args_d['h0_name'] = init_args_d['h0_name'] + \
                                         '_fixed_baseline_%s'%param.name
                init_args_d['h1_name'] = init_args_d['h1_name'] + \
                                         '_fixed_baseline_%s'%param.name

                param.is_fixed = True
                # Instantiate the analysis object
                hypo_testing = HypoTesting(**init_args_d)
                # Run the analysis
                hypo_testing.run_analysis()

                # Reset everything
                init_args_d['h0_name'] = nominal_h0_name
                init_args_d['h1_name'] = nominal_h1_name
                param.is_fixed = False
                


if __name__ == '__main__':
    main()
