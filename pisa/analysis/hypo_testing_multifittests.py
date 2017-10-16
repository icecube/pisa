#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This script/module will load the HypoTesting class from hypo_testing.py and
use it to do a minimiser study in Asimov. A data distribution will be made at 
the injected values of the parameters and a hypothesis test will be run for 
both ordering hypothesis with seeded values randomly off truth.

The "data" can be Asimov (default) or can be set to a pseudo-experiment. In 
either case, the idea is that the fit is repeated to the same dataset multiple 
times to check the stability of the minimisation.

"""


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

from pisa import ureg
from pisa.analysis.hypo_testing import HypoTesting, Labels
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.fileio import normcheckpath
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
        '--num-trials',
        type=int, default=1,
        help='''Number of fits to run. The minimiser start point is 
        randomised in every case, and so running multiple is NOT the same 
        operation.'''
    )
    parser.add_argument(
        '--start-hypo-nominal',
        action='store_true', default=False,
        help='''Set this to fix the hypothesis maker to always start at the 
        same (nominal) values.'''
    )
    parser.add_argument(
        '--fix-muon-scale',
        action='store_true', default=False,
        help='''Set this to fix the muon scale to the nominal value. Use this 
        when wanting to evaluate "fixed" error for the muons so the final 
        returned chi2 doesn't inherently vary just because of this.'''
    )
    parser.add_argument(
        '--start-index',
        type=int, default=0,
        help='''Trial start index. Set this if you are saving files from 
        multiple runs in to the same log directory otherwise files may end up 
        being overwritten!'''
    )
    parser.add_argument(
        '--data-is-pseudo',
        action='store_true', default=False,
        help='''Set this to fluctuate the "data" distribution used in all of 
        the fits. If this is set to true you must also specify an index which 
        defines this random state with the --data-index argument.'''
    )
    parser.add_argument(
        '--data-index',
        type=int, default=0,
        help='''If --data-is-pseudo is set then this will be the index by 
        which the random state is defined.'''
    )
    parser.add_argument(
        '--print-metric-contributions',
        action='store_true', default=False,
        help='''Set this to print out some more details on the various 
        contributions to the final fit metric. Note that this is very crude so 
        is only meant for quick debugging.'''
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


def main():
    args = parse_args()
    init_args_d = vars(args)

    # NOTE: Removing extraneous args that won't get passed to instantiate the
    # HypoTesting object via dictionary's `pop()` method.

    set_verbosity(init_args_d.pop('v'))
    num_trials = init_args_d.pop('num_trials')
    start_index = init_args_d.pop('start_index')
    data_is_pseudo = init_args_d.pop('data_is_pseudo')
    data_index = init_args_d.pop('data_index')
    randomise_params = not init_args_d.pop('start_hypo_nominal')
    print_metric_contributions = init_args_d.pop('print_metric_contributions')
    if print_metric_contributions:
        from uncertainties import unumpy as unp
        import numpy as np
    if data_is_pseudo:
        if data_index == 0:
            logging.warning('You have requested the data be a pseudo-experiment'
                            ' but the data index by which the random state is'
                            ' defined has been left as the default value. '
                            'Please ensure you definitely wanted to do this.')
    else:
        if data_index != 0:
            logging.warning('You have not requested the data be a '
                            'pseudo-experiment but you have specified a value '
                            'for the data index by which the random state is '
                            'defined has been changed from the default value. '
                            'Please ensure you definitely wanted to do this.')
            
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
    init_args_d['fluctuate_fid'] = False

    init_args_d['data_maker'] = init_args_d['h0_maker']

    init_args_d['h0_maker'] = DistributionMaker(init_args_d['h0_maker'])
    init_args_d['h1_maker'] = DistributionMaker(init_args_d['h0_maker'])
    init_args_d['h1_maker'].select_params(init_args_d['h1_param_selections'])
    init_args_d['data_maker'] = DistributionMaker(init_args_d['data_maker'])
    if init_args_d['data_param_selections'] is None:
        init_args_d['data_param_selections'] = \
            init_args_d['h0_param_selections']
        init_args_d['data_name'] = init_args_d['h0_name']
    else:
        if init_args_d['data_name'] is None:
            raise ValueError('data_param_selections given but data_name was '
                             'not. Please be sure to specify both')
    init_args_d['data_maker'].select_params(
        init_args_d['data_param_selections']
    )

    fix_muon_scale = init_args_d.pop('fix_muon_scale')
    

    # Instantiate the analysis object
    hypo_testing = HypoTesting(**init_args_d)
    hypo_testing.setup_logging()
    hypo_testing.write_config_summary()
    hypo_testing.write_minimizer_settings()
    hypo_testing.write_run_info()

    if fix_muon_scale:
        if 'atm_muon_scale' in hypo_testing.h0_maker.params.names:
            nominal_muon_scale = hypo_testing.h0_maker.params[
                'atm_muon_scale'].value
        else:
            raise ValueError("You have requested to fix the atmospheric muon "
                             "scale but there doesn't appear to be atmospheric"
                             " muons in the files. Aborting.")

    hypo_testing.generate_data()
    if data_is_pseudo:
        data_random_state = get_random_state([0, data_index, 0])
        hypo_testing.data_dist = hypo_testing.data_dist.fluctuate(
            method='poisson', random_state=data_random_state
        )

    h0_nominal_free_params = hypo_testing.h0_maker.params.free
    h1_nominal_free_params = hypo_testing.h1_maker.params.free

    for i in range(start_index,(start_index+num_trials)):
        if randomise_params:
            # Randomise seeded parameters for hypotheses
            hypo_testing.h0_maker.randomize_free_params()
            hypo_testing.h1_maker.randomize_free_params()
            if fix_muon_scale:
                hypo_testing.h0_maker.params[
                    'atm_muon_scale'].value = nominal_muon_scale
                hypo_testing.h1_maker.params[
                    'atm_muon_scale'].value = nominal_muon_scale
        else:
            # Ensure at nominal
            hypo_testing.h0_maker.params.free.set_values(
                new_params=h0_nominal_free_params
            )
            hypo_testing.h1_maker.params.free.set_values(
                new_params=h1_nominal_free_params
            )
        # Create Labels dict to distinguish each of these Asimov "trials"
        if data_is_pseudo:
            hypo_testing.labels = Labels(
                h0_name=init_args_d['h0_name'],
                h1_name=init_args_d['h1_name'],
                data_name=init_args_d['data_name']+'_fit_%i_pseudo_%i'%(
                    i,data_index),
                data_is_data=init_args_d['data_is_data'],
                fluctuate_data=init_args_d['fluctuate_data'],
                fluctuate_fid=init_args_d['fluctuate_fid']
            )
        else:
            hypo_testing.labels = Labels(
                h0_name=init_args_d['h0_name'],
                h1_name=init_args_d['h1_name'],
                data_name=init_args_d['data_name']+'_fit_%i'%i,
                data_is_data=init_args_d['data_is_data'],
                fluctuate_data=init_args_d['fluctuate_data'],
                fluctuate_fid=init_args_d['fluctuate_fid']
            )
        # Run the fits
        hypo_testing.fit_hypos_to_data()
        hypo_asimov_dist = hypo_testing.h1_maker.get_outputs(
            return_sum=True
        )
        if print_metric_contributions:
            print "Evaluation %i"%i
            print "Map metric = %.11f"%hypo_testing.data_dist.metric_total(
                expected_values=hypo_asimov_dist,
                metric=init_args_d['metric']
            )
            print "Prior metric = %.11f"% \
                hypo_testing.h1_maker.params.priors_penalty(
                    metric=init_args_d['metric']
                )
            if init_args_d['metric'] == 'mod_chi2':
                print "Bare chi2 = %.11f"% \
                    np.sum(
                        (unp.nominal_values(
                            np.clip(
                                hypo_testing.data_dist.hist['total'],
                                a_min=1e-10,
                                a_max=np.inf,
                                out=hypo_testing.data_dist.hist['total']
                            )
                        ).ravel() - unp.nominal_values(
                            hypo_asimov_dist.hist['total']
                        ).ravel())**2)
                print "Sigma term = %.11f"% \
                    np.sum(unp.std_devs(hypo_asimov_dist.hist['total']).ravel())
        # Reset everything
        hypo_testing.h0_maker.reset_free()
        hypo_testing.h1_maker.reset_free()


if __name__ == '__main__':
    main()
