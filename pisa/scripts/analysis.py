#!/usr/bin/env python

"""
Test hypotheses
"""


from __future__ import absolute_import, division

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
from os.path import basename
import sys


from pisa.scripts.discrete_hypo_test import discrete_hypo_test
from pisa.scripts.inj_param_scan import inj_param_scan
from pisa.scripts.systematics_tests import systematics_tests
from pisa.utils.fileio import from_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.scripting import get_script, parse_command
from pisa.utils.stats import ALL_METRICS


__all__ = ['SCRIPT', 'parse_args', 'main']

__author__ = 'S. Wren, J.L. Lanfranchi'


SCRIPT = basename(get_script())


def parse_args(command, description):
    """Parse command line args.

    Returns
    -------
    init_args_d : dict

    """
    assert command in [discrete_hypo_test, inj_param_scan, systematics_tests]

    parser = ArgumentParser(
        description=description,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-d', '--logdir', required=True,
        metavar='DIR', type=str,
        help='Directory into which to store results and metadata.'
    )
    parser.add_argument(
        '--min-settings',
        type=str, metavar='MINIMIZER_CFG', default=None,
        help='''Minimizer settings config file.'''
    )
    parser.add_argument(
        '--min-method',
        type=str, default=None, choices=('l-bfgs-b', 'slsqp'),
        help='''Name of minimizer to use. Note that this takes precedence over
        the minimizer method specified via the --min-settings config
        file.'''
    )
    parser.add_argument(
        '--min-opt',
        type=str, metavar='OPTION:VALUE', nargs='+', default=None,
        help='''Minimizer option:value pair(s) (can specify multiple).
        Values specified here override any of the same name in the config file
        specified by --min-settings'''
    )
    parser.add_argument(
        '--no-octant-check',
        action='store_true',
        help='''Disable fitting hypotheses in theta23 octant opposite initial
        octant.'''
    )
    parser.add_argument(
        '--ordering-check',
        action='store_true',
        help='''Fit both ordering hypotheses. This should only be flagged if
        the ordering is NOT the discrete hypothesis being tested'''
    )

    if command == discrete_hypo_test:
        # Data cannot be data for MC studies e.g. injected parameter scans so
        # these arguments are redundant there.
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            '--data-is-data', action='store_true',
            help='''Data pipeline is based upon actual, measured data. The
            naming scheme for stored results is chosen accordingly.'''
        )
        group.add_argument(
            '--data-is-mc', action='store_true',
            help='''Data pipeline is based upon Monte Carlo simulation, and not
            actual data. The naming scheme for stored results is chosen
            accordingly. If this is selected, --fluctuate-data is forced off.'''
        )

        # For the MC tests (injected parameter scan, systematic tests etc.) you
        # must have the same pipeline for h0, h1 and data. So this argument is
        # instead replaced with a generic pipeline argument.
        parser.add_argument(
            '--h0-pipeline', required=True,
            type=str, action='append', metavar='PIPELINE_CFG',
            help='''Settings for the generation of hypothesis h0
            distributions; repeat this argument to specify multiple
            pipelines.'''
        )

    else:
        assert command in [inj_param_scan, systematics_tests]
        parser.add_argument(
            '--pipeline', required=True,
            type=str, action='append', metavar='PIPELINE_CFG',
            help='''Settings for the generation of h0, h1 and data
            distributions; repeat this argument to specify multiple
            pipelines.'''
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
    # For the MC tests (injected parameter scan, systematic tests etc.) you
    # must have the same pipeline for h0, h1 and data. So this argument is
    # hidden.
    if command not in (inj_param_scan, systematics_tests):
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
    # For the MC tests (injected parameter scan, systematic tests etc.) you
    # must have the same pipeline for h0, h1 and data. So this argument is
    # hidden.
    if command not in (inj_param_scan, systematics_tests):
        parser.add_argument(
            '--data-pipeline',
            type=str, action='append', default=None, metavar='PIPELINE_CFG',
            help='''Settings for the generation of "data" distributions; repeat
            this argument to specify multiple pipelines. If omitted, the same
            settings as specified for --h0-pipeline are used to generate data
            distributions (i.e., data is assumed to come from hypothesis h0.'''
        )
    parser.add_argument(
        '--data-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated list of param selectors to apply to the data
        distribution maker's pipelines. If neither --data-pipeline nor
        --data-param-selections are specified, *both* are copied from
        --h0-pipeline and --h0-param-selections, respectively. However,
        if --data-pipeline is specified while --data-param-selections is not,
        then the param selections in the pipeline config file(s) specified are
        used to produce data distributions.'''
    )
    parser.add_argument(
        '--data-name',
        type=str, metavar='NAME', default=None,
        help='''Name for the data. E.g., "NO" for normal ordering in the
        neutrino mass ordering analysis. Note that the name here has no bearing
        on the actual process, so it's important that you be careful to use a
        name that appropriately identifies the hypothesis.'''
    )
    # For the injected parameter scan and systematic studies, only the Asimov
    # analysis should be used, so these arguments are not needed.
    if command not in (inj_param_scan, systematics_tests):
        parser.add_argument(
            '--fluctuate-data',
            action='store_true',
            help='''Apply fluctuations to the data distribution. This should
            *not* be set for analyzing "real" (measured) data, and it is common
            to not use this feature even for Monte Carlo analysis. Note that if
            this is not set, --num-data-trials and --data-start-ind are forced
            to 1 and 0, respectively.'''
        )
        parser.add_argument(
            '--fluctuate-fid',
            action='store_true',
            help='''Apply fluctuations to the fiducaial distributions. If this
            flag is not set, --num-fid-trials and --fid-start-ind are forced to
            1 and 0, respectively.'''
        )
    parser.add_argument(
        '--metric',
        type=str, required=True, metavar='METRIC', choices=ALL_METRICS,
        help='''Name of metric to use for optimizing the fit. Must be one of
        %s.''' % (ALL_METRICS,)
    )
    parser.add_argument(
        '--other-metric',
        type=str, default=None, metavar='METRIC', action='append',
        choices=['all'] + sorted(ALL_METRICS),
        help='''Name of another metric to evaluate at the best-fit point. Must
        be either 'all' or one of %s. Repeat this argument (or use 'all') to
        specify multiple metrics.''' % (ALL_METRICS,)
    )
    if command not in (inj_param_scan, systematics_tests):
        parser.add_argument(
            '--num-data-trials',
            type=int, default=1,
            help='''When performing Monte Carlo analysis, set to > 1 to produce
            multiple pseudodata distributions from the data distribution maker's
            Asimov distribution. This is overridden if --fluctuate-data is not
            set (since each data distribution will be identical if it is not
            fluctuated). This is typically left at 1 (i.e., the Asimov
            distribution is assumed to be representative.'''
        )
        parser.add_argument(
            '--data-start-ind',
            type=int, default=0,
            help='''Fluctated data set index.'''
        )
        parser.add_argument(
            '--num-fid-trials',
            type=int, default=1,
            help='''Number of fiducial pseudodata trials to run. In our
            experience, it takes ~10^3-10^5 fiducial psuedodata trials to
            achieve low uncertainties on the resulting significance, though
            that exact number will vary based upon the details of an
            analysis.'''
        )
        parser.add_argument(
            '--fid-start-ind',
            type=int, default=0,
            help='''Fluctated fiducial data index.'''
        )
    # A blind analysis only makes sense when the possibility of actually
    # analysing data is available.
    if command not in (inj_param_scan, systematics_tests):
        parser.add_argument(
            '--blind',
            action='store_true',
            help='''Blinded analysis. Do not show parameter values or store to
            logfiles.'''
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
        '--no-min-history',
        action='store_true',
        help='''Do not store minimizer history (steps). This behavior is also
        enforced if --blind is specified.'''
    )
    # Add in the arguments specific to the injected parameter scan.
    if command == inj_param_scan:
        parser.add_argument(
            '--param_name',
            type=str, metavar='NAME', required=True,
            help='''Name of param to scan over. This must be in the config
            files defined above. One exception is that you can define this as
            `sin2theta23` and it will be interpreted not as theta23 values but
            as the square of the sine of theta23 values instead.'''
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
            '--inj_units',
            type=str, required=True,
            help='''A string to be able to deal with the units in the parameter
            scan and make sure that they match those in the config files. Even
            if the parameter is dimensionless this must be stated.'''
        )
        parser.add_argument(
            '--use-inj-prior', action='store_true',
            help='''Generally, one should not use a prior on the parameter of
            interest here since the Asimov analysis breaks down with the use of
            non-central prior i.e. injecting a truth that differs from the
            centre of the prior. Flag this to force the prior to be left on.'''
        )
    # Add in the arguments specific to the systematic tests.
    if command == systematics_tests:
        parser.add_argument(
            '--inject_wrong',
            action='store_true',
            help='''Inject a parameter to some systematically wrong value.
            This will be either +/- 1 sigma or +/- 10%% if such a definition
            is impossible. By default this parameter will be fixed unless
            the fit_wrong argument is also flagged.'''
        )
        parser.add_argument(
            '--fit_wrong',
            action='store_true',
            help='''In the case of injecting a systematically wrong hypothesis
            setting this argument will get the minimiser to try correct for it.
            If inject_wrong is set to false then this must also be set to
            false or else the script will fail.'''
        )
        parser.add_argument(
            '--only_syst', default=None,
            type=str, action='append', metavar='PARAM_NAME',
            help='''Specify the name of one of the systematics in the file to
            run the test for this systematic. Repeat this argument to specify
            multiple systematics. If none are provided, the test will be run
            over all systematics in the pipeline.'''
        )
        parser.add_argument(
            '--skip_baseline',
            action='store_true',
            help='''Skip the baseline systematic test i.e. the one where none
            of them are fixed and/or modified. In most cases you will want this
            for comparison but if you are only interested in the effect of
            shifting certain systematics then this step can be skipped.'''
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
    args = parser.parse_args(sys.argv[2:])
    assert args.min_settings is not None or args.min_method is not None
    init_args_d = vars(args)

    set_verbosity(init_args_d.pop('v'))

    min_settings_from_file = init_args_d.pop('min_settings')
    minimizer = init_args_d.pop('min_method')
    min_opt = init_args_d.pop('min_opt')

    # TODO: put this datastructure remnant from PISA 2 out of its misery...
    minimizer_settings = dict(
        method=dict(value='', desc='no desc'),
        options=dict(value=dict(), desc=dict())
    )

    if min_settings_from_file is not None:
        minimizer_settings.update(from_file(min_settings_from_file))

    if minimizer is not None:
        minimizer_settings['method'] = dict(value=minimizer, desc='no desc')

    if min_opt is not None:
        for opt_val_str in min_opt:
            opt, val_str = [s.strip() for s in opt_val_str.split(':')]
            try:
                val = int(val_str)
            except ValueError:
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str
            minimizer_settings['options']['value'][opt] = val
            minimizer_settings['options']['desc'][opt] = 'no desc'
    init_args_d['minimizer_settings'] = minimizer_settings

    init_args_d['check_octant'] = not init_args_d.pop('no_octant_check')
    init_args_d['check_ordering'] = init_args_d.pop('ordering_check')

    if command not in (inj_param_scan, systematics_tests):
        init_args_d['data_is_data'] = not init_args_d.pop('data_is_mc')
    else:
        init_args_d['data_is_data'] = False
        init_args_d['fluctuate_data'] = False
        init_args_d['fluctuate_fid'] = False

    init_args_d['store_minimizer_history'] = (
        not init_args_d.pop('no_min_history')
    )

    other_metrics = init_args_d.pop('other_metric')
    if other_metrics is not None:
        other_metrics = [s.strip().lower() for s in other_metrics]
        if 'all' in other_metrics:
            other_metrics = sorted(ALL_METRICS)
        if init_args_d['metric'] in other_metrics:
            other_metrics.remove(init_args_d['metric'])
        if not other_metrics:
            other_metrics = None
        else:
            logging.info('Will evaluate other metrics %s', other_metrics)
        init_args_d['other_metrics'] = other_metrics

    return init_args_d


MAIN_CMD_SPEC = dict(
    commands=OrderedDict([
        ('discrete_hypo', discrete_hypo_test),
        ('inj_param_scan', inj_param_scan),
        ('syst_tests', systematics_tests)
    ]),
    description='Perform hypothesis testing',
    usage='''{0} <command> [<args>]

The commands that can be issued are:

  discrete_hypo   Standard hypothesis testing analyses
  inj_param_scan  Scan over some injected parameter in the data
  syst_tests      Perform tests on the impact of systematics on the analysis

Run
  {0} <command> -h
to see the valid arguments for each commands.'''.format(SCRIPT)
)


def main(return_outputs=False):
    """main"""
    return parse_command(command_depth=0, return_outputs=return_outputs,
                         **MAIN_CMD_SPEC)


if __name__ == '__main__':
    outputs = main(return_outputs=True) # pylint: disable=invalid-name
