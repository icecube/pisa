#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This script/module will load the HypoTesting class from hypo_testing.py and
use it to do an Asimov test across the space of one of the injected parameters.
The user will define the parameter and pass a numpy-interpretable string to 
set the range of values. For example, one could scan over the space of theta23 
by using a string such as `numpy.linspace(0.35,0.65,31)` which will then be 
evaluated to figure out a space of theta23 to inject and run Asimov tests.

TODO:

1) Make sure this actually works...

"""


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

import numpy as np

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
        '--param_name',
        type=str, metavar='NAME', required=True,
        help='''Name of param to scan over. This must be in the config files 
        defined above. One exception is that you can define this as 
        `sin2theta23` and it will be interpreted not as theta23 values but as 
        the square of the sine of theta23 values instead.'''
    )
    parser.add_argument(
        '--inj_vals',
        type=str, required=True,
        help='''List of values to inject as true points in the parameter 
        defined above. Must be something that numpy can interpret. In this 
        script, numpy is imported as np so please use np in your string. An 
        example would be np.linspace(0.35,0.65,31).'''
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
    init_args_d['h1_maker'] = DistributionMaker(init_args_d['h0_maker'])
    init_args_d['h1_maker'].select_params(init_args_d['h1_param_selections'])

    param_name = init_args_d.pop('param_name')
    inj_vals = eval(init_args_d.pop('inj_vals'))

    logging.info(
        'Scanning over %s between %.4f and %.4f with %i vals'
        %(param_name, min(inj_vals), max(inj_vals), len(inj_vals))
    )
    if param_name == 'sin2theta23':
        requested_vals = inj_vals
        inj_vals = np.arcsin(np.sqrt(inj_vals))
        logging.info(
            'Converting to theta23 values. Equivalent range is %.4f to %.4f '
            'radians, or %.4f to %.4f degrees'
            %(min(inj_vals), max(inj_vals),
              min(inj_vals)*180/np.pi, max(inj_vals)*180/np.pi)
        )
        test_name = 'theta23'
    else:
        test_name = param_name
        requested_vals = inj_vals

    rangediff = max(inj_vals) - min(inj_vals)
    rangemin = min(inj_vals) - 0.5*rangediff
    rangemax = max(inj_vals) + 0.5*rangediff

    for inj_val, requested_val in zip(inj_vals, requested_vals):
        for param in init_args_d['h0_maker'].params:
            if param.name == test_name:
                if hasattr(param, 'range'):
                    if param.range is not None:
                        if param_name=='sin2theta23' and param.units=='degree':
                            param.range = (rangemin * 180/np.pi * param.units,
                                           rangemax * 180/np.pi * param.units)
                            inj_val *= 180/np.pi
                        else:
                            param.range = (rangemin * param.units,
                                           rangemax * param.units)
                param.value = inj_val * param.units

                for alt_param in init_args_d['h1_maker'].params:
                    if alt_param.name == test_name:
                        if hasattr(alt_param, 'range'):
                            if alt_param.range is not None:
                                if param_name=='sin2theta23' \
                                   and alt_param.units=='degree':
                                    alt_param.range = (rangemin * 180/np.pi \
                                                   * alt_param.units,
                                                   rangemax * 180/np.pi \
                                                   * alt_param.units)
                                else:
                                    alt_param.range = (rangemin \
                                                       * alt_param.units,
                                                       rangemax \
                                                       * alt_param.units)
                                alt_param.value = inj_val * alt_param.units
                        
        nominal_h0_name = init_args_d['h0_name']
        nominal_h1_name = init_args_d['h1_name']
            
        # Make names reflect parameter value
        init_args_d['h0_name'] = init_args_d['h0_name'] + \
                                 '_%s_%.4f'%(param_name,
                                             requested_val)
        init_args_d['h1_name'] = init_args_d['h1_name'] + \
                                 '_%s_%.4f'%(param_name,
                                             requested_val)
                
        # Instantiate the analysis object
        hypo_testing = HypoTesting(**init_args_d)
        # Run the analysis
        hypo_testing.run_analysis()

        # Reset names
        init_args_d['h0_name'] = nominal_h0_name
        init_args_d['h1_name'] = nominal_h1_name


if __name__ == '__main__':
    main()
