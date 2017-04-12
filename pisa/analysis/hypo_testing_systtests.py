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
to either the baseline or some shifted value (+/- 1 sigma, or appropriate). One
also has the ability in the case of the latter to still fit with this 
systematically incorrect hypothesis.

"""


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

from pisa import ureg
from pisa.analysis.hypo_testing import HypoTesting, parse_args, normcheckpath
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.stats import ALL_METRICS


def main():
    args = parse_args(systtests=True)
    init_args_d = vars(args)

    # NOTE: Removing extraneous args that won't get passed to instantiate the
    # HypoTesting object via dictionary's `pop()` method.
    set_verbosity(init_args_d.pop('v'))
    init_args_d['check_octant'] = not init_args_d.pop('no_octant_check')
    init_args_d['store_minimizer_history'] = (
        not init_args_d.pop('no_minimizer_history')
    )
    inject_wrong = init_args_d.pop('inject_wrong')
    fit_wrong = init_args_d.pop('fit_wrong')
    if fit_wrong:
        if not inject_wrong:
            raise ValueError('You have specified to fit the systematically '
                             'wrong hypothesis but have not specified to '
                             'actually generate a systematically wrong '
                             'hypothesis. If you want to flag "fit_wrong" '
                             'please also flag "inject_wrong"')
        else:
            logging.info('Injecting a systematically wrong hypothesis while '
                         'also allowing the minimiser to attempt to correct '
                         'for it.')
    else:
        if inject_wrong:
            logging.info('Injecting a systematically wrong hypothesis but NOT ' 
                         'allowing the minimiser to attempt to correct for it. '
                         'Hypothesis maker will be FIXED at the baseline '
                         'value.')
        else:
            logging.info('A standard N-1 test will be performed where each '
                         'systematic is fixed to the baseline value '
                         'one-by-one.')

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

    init_args_d['h0_maker'] = DistributionMaker(init_args_d['h0_maker'])
    init_args_d['h1_maker'] = DistributionMaker(init_args_d['h1_maker'])
    init_args_d['h1_maker'].select_params(init_args_d['h1_param_selections'])
    init_args_d['data_maker'] = DistributionMaker(init_args_d['data_maker'])
    if init_args_d['data_param_selections'] is None:
        init_args_d['data_param_selections'] = \
            init_args_d['h0_param_selections']
        init_args_d['data_name'] = init_args_d['h0_name']
    init_args_d['data_maker'].select_params(
        init_args_d['data_param_selections']
    )

    # Instantiate the analysis object
    hypo_testing = HypoTesting(**init_args_d)
    # Everything is set up so do the tests
    hypo_testing.syst_tests(
        inject_wrong=inject_wrong,
        fit_wrong=fit_wrong
        **init_args_d
    )


if __name__ == '__main__':
    main()
