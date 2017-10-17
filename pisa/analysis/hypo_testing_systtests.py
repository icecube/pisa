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
from pisa.analysis.hypo_testing import HypoTesting, Labels
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.stats import ALL_METRICS


def systematic_wrong_analysis(data_param, hypo_testing, fit_wrong, direction,
                              h0_name, h1_name, data_name):
    '''
    This function will perform a modified version of the N-1 test on the 
    hypo_testing object specified in the arguments. This differs in that here 
    we do not assume the systematics take their baseline values but here see 
    instead what happens with something systematically wrong. So, the 
    data_param is shifted by 1 sigma or 10% off baseline. The direction of 
    this shift should be specified pve or nve in the direction argument 
    (meaning positive or negative). Then one can allow the minimiser to correct
    for this by specifying fit_wrong. If this is false then the hypothesis 
    maker will be fixed to the baseline in this parameter i.e. a systematically
    wrong hypothesis to what is injected. As with the N-1 test below it is 
    assumed that this function exists inside of a loop over the parameters in 
    the data_maker and this is for the systematic defined in data_param. This 
    function also expects h0_name, h1_name and data_name so that the labels of 
    hypo_testing can be redefined to make everything unique. 
    '''
    if direction != 'pve':
        if direction != 'nve':
            raise ValueError('Direction to shift systematic value must be '
                             'specified either as "pve" or "nve" for positive '
                             'and negative respectively')
    # Calculate this wrong value based on the prior
    if hasattr(data_param, 'prior'):
        # Gaussian priors are easy - just do 1 sigma
        if data_param.prior.kind == 'gaussian':
            if direction == 'pve':
                data_param.value \
                    = data_param.value + data_param.prior.stddev
            else:
                data_param.value \
                    = data_param.value - data_param.prior.stddev
        # Else do 10%
        else:
            if direction == 'pve':
                data_param.value = 1.1 * data_param.value
            else:
                data_param.value = 0.9 * data_param.value
    # If we are not allowing the fit to correct for this, it must be
    # fixed in the hypo makers.
    if not fit_wrong:
        for h0_param in hypo_testing.h0_maker.params.free:
            if h0_param.name == data_param.name:
                h0_param.is_fixed = True
        for h1_param in hypo_testing.h1_maker.params.free:
            if h1_param.name == data_param.name:
                h1_param.is_fixed = True
    # Set up labels so that each file comes out unique
    if fit_wrong:
        hypo_testing.labels = Labels(
            h0_name=h0_name,
            h1_name=h1_name,
            data_name=data_name + '_inj_%s_%s_wrong'%(
                data_param.name,direction),
            data_is_data=False,
            fluctuate_data=False,
            fluctuate_fid=False
        )
    else:
        hypo_testing.labels = Labels(
            h0_name=h0_name + '_fixed_%s_baseline'%data_param.name,
            h1_name=h1_name + '_fixed_%s_baseline'%data_param.name,
            data_name=data_name + '_inj_%s_%s_wrong'%(
                data_param.name,direction),
            data_is_data=False,
            fluctuate_data=False,
            fluctuate_fid=False
        )
    # Setup logging and things.
    hypo_testing.setup_logging(reset_params=False)
    hypo_testing.write_config_summary(reset_params=False)
    hypo_testing.write_minimizer_settings()
    hypo_testing.write_run_info()
    # Now do the fits
    hypo_testing.generate_data()
    hypo_testing.fit_hypos_to_data()
    hypo_testing.produce_fid_data()
    hypo_testing.fit_hypos_to_fid()


def nminusone_test(data_param, hypo_testing, h0_name, h1_name, data_name):
    '''
    This function will perform the standard N-1 test on the hypo_testing object
    specified in the arguments. It is assumed that this function exists inside 
    of a loop over the parameters in the data_maker and this is for the 
    systematic defined in data_param. This function also expects h0_name, 
    h1_name and data_name so that the labels of hypo_testing can be redefined 
    to make everything unique.
    '''
    # Set up labels so that each file comes out unique
    hypo_testing.labels = Labels(
        h0_name=h0_name + '_fixed_%s_baseline'%data_param.name,
        h1_name=h1_name + '_fixed_%s_baseline'%data_param.name,
        data_name=data_name,
        data_is_data=False,
        fluctuate_data=False,
        fluctuate_fid=False
    )
    # This is a standard N-1 test, so fix the parameter in the hypo makers.
    for h0_param in hypo_testing.h0_maker.params.free:
        if h0_param.name == data_param.name:
            h0_param.is_fixed = True
    for h1_param in hypo_testing.h1_maker.params.free:
        if h1_param.name == data_param.name:
            h1_param.is_fixed = True
    # Setup logging and things.
    hypo_testing.setup_logging(reset_params=False)
    hypo_testing.write_config_summary(reset_params=False)
    hypo_testing.write_minimizer_settings()
    hypo_testing.write_run_info()
    # Now do the fits
    hypo_testing.generate_data()
    hypo_testing.fit_hypos_to_data()
    hypo_testing.produce_fid_data()
    hypo_testing.fit_hypos_to_fid()


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
        hypothesis h1 distribution maker's pipelines. The h1 pipeline is forced
        to be the h0 pipeline in these tests.'''
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
        distribution maker's pipelines. If --data-param-selections are not
        specified then they are copied from --h0-param-selections. The data 
        pipeline is forced to be the h0 pipeline in these tests.'''
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
        '--inject_wrong',
        action='store_true',
        help='''Inject a parameter to some systematically wrong value. This 
        will be either +/- 1 sigma or +/- 10%% if such a definition is 
        impossible. By default this parameter will be fixed unless the 
        fit_wrong argument is also flagged.'''
    )
    parser.add_argument(
        '--fit_wrong',
        action='store_true',
        help='''In the case of injecting a systematically wrong hypothesis 
        setting this argument will get the minimiser to try correct for it. If 
        inject_wrong is set to false then this must also be set to false or 
        else the script will fail.'''
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

    init_args_d['fluctuate_data'] = False
    init_args_d['fluctuate_fid'] = False

    init_args_d['data_maker'] = init_args_d['h0_maker']
    init_args_d['h1_maker'] = init_args_d['h0_maker']

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

    # Go through the nuisance (systematic) parameters
    # Data maker must be constructed from same as h0/h1 maker in this script,
    # so we will loop over the data_maker "free" parameters here.
    for data_param in hypo_testing.data_maker.params.free:
        if inject_wrong:
            # First inject this wrong up by one sigma
            systematic_wrong_analysis(
                data_param=data_param,
                hypo_testing=hypo_testing,
                fit_wrong=fit_wrong,
                direction='pve',
                h0_name=init_args_d['h0_name'],
                h1_name=init_args_d['h1_name'],
                data_name=init_args_d['data_name']
            )
            # Then inject this wrong down by one sigma
            systematic_wrong_analysis(
                data_param=data_param,
                hypo_testing=hypo_testing,
                fit_wrong=fit_wrong,
                direction='nve',
                h0_name=init_args_d['h0_name'],
                h1_name=init_args_d['h1_name'],
                data_name=init_args_d['data_name']
            )
        else:
            # Just do the standard N-1 test
            nminusone_test(
                data_param=data_param,
                hypo_testing=hypo_testing,
                h0_name=init_args_d['h0_name'],
                h1_name=init_args_d['h1_name'],
                data_name=init_args_d['data_name']
            )

        # At the end, reset the parameters in the maker
        hypo_testing.data_maker.params.reset_free()
        hypo_testing.h0_maker.params.reset_free()
        hypo_testing.h1_maker.params.reset_free()
        # Also unfix the hypo maker parameters
        for h0_param in hypo_testing.h0_maker.params:
            if h0_param.name == data_param.name:
                h0_param.is_fixed = False
        for h1_param in hypo_testing.h1_maker.params:
            if h1_param.name == data_param.name:
                h1_param.is_fixed = False


if __name__ == '__main__':
    main()
