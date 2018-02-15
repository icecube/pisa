#!/usr/bin/env python

"""
Test hypotheses
"""


from __future__ import absolute_import, division

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
from collections import OrderedDict
from os.path import basename
import sys

# assign name to analysis wrapper scripts which they should be available by
from pisa.scripts.discrete_hypo_test import discrete_hypo_test as discrete_hypo
from pisa.scripts.inj_param_scan import inj_param_scan
from pisa.scripts.profile_scan import profile_scan
from pisa.scripts.systematics_tests import systematics_tests as syst_tests
from pisa.utils.log import logging, set_verbosity
from pisa.utils.scripting import get_script
from pisa.utils.stats import ALL_METRICS


__all__ = ['SCRIPT', 'AnalysisScript', 'main']

__author__ = 'S. Wren, J.L. Lanfranchi, T. Ehrhardt'

__license__ = '''Copyright (c) 2014-2017, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


SCRIPT = basename(get_script())


class AnalysisScript(object):

    def __init__(self):

        init_parser = ArgumentParser(
            description='Perform some analysis involving fitting/hypo testing',
            usage='''{0} <command> [<args>]

            The commands that can be issued are:

            discrete_hypo   Standard hypothesis testing analyses
            inj_param_scan  Scan over some injected parameter in the data
            profile_scan    Scan over some hypothesis parameters
            syst_tests      Perform tests on the impact of systematics on the analysis

            Run
            {0} <command> -h
            to see the valid arguments for each commands.'''.format(SCRIPT),
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        init_parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        init_args = init_parser.parse_args(sys.argv[1:2])

        # check whether the subcommand exists by looking for a method called
        # 'command_<command>'
        if not hasattr(self, 'command_' + init_args.command):
            raise ValueError('Unrecognized command: "%s"' % init_args.command)
            init_parser.print_help()
            exit(1)

        self.command = init_args.command

        # create all the available subparsers so they can be invoked
        # by any subcommand if desired
        self.setup_shared_arguments()

        # invoke subcommand
        analysis_parser = getattr(self, 'command_' + init_args.command)()

        # after the subcommand, so ignore the first TWO argvs: the command
        # (SCRIPT) and the subcommand (<command>)
        self.analysis_args = analysis_parser.parse_args(sys.argv[2:])

        # call validation of arguments on the subcommand
        self.validate_analysis_args()

        self.analysis_init_args_d = vars(self.analysis_args)
        set_verbosity(self.analysis_init_args_d.pop('v'))

        # make the dictionary of initialization arguments which will then
        # be passed to the analysis script wrapper living in some other
        # module
        self.make_analysis_init_args_d()


    ############### Setup of initialization argument dictionary ###############

    # TODO: this requires the existence of some of the subparsers below
    # --> make an analysis script parser include these always
    def make_analysis_init_args_d(self):
        """Creates a generically usable dictionary of init args for the
        various analysis scripts"""
        init_args_d = self.analysis_init_args_d
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

        if self.command not in ('inj_param_scan', 'syst_tests'):
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

        self.analysis_init_args_d = init_args_d


    ############### Argument validation ###############
    """
    Add your own by creating a function `validate_args_<command>`. This
    will be called automatically.
    """

    def validate_analysis_args(self):
        """Dispatch validation of arguments if a method is found with
        name determined by the chosen subcommand"""
        if hasattr(self, 'validate_args_' + self.command):
            getattr(self, 'validate_args_' + self.command)()
        else:
            return


    def validate_args_discrete_hypo(self):
        return


    def validate_args_inj_param_scan(self):
        return


    def validate_args_syst_tests(self):
        return


    def validate_args_profile_scan(self):
        """Validate parser arguments for profile scan."""
        if (self.analysis_args.min_settings is None and
            self.analysis_args.min_method is None and
            not self.analysis_args.no_profile):
            # only require minimization settings if some nuisance parameters
            # are supposed to be taken into account
            raise ArgumentError(
                None,
                'Invalid options: specify "--min-settings" or "--min-method"'
                ' or set "--no-profile".'
            )
        if self.analysis_args.fluctuate_fid:
            raise ArgumentError(
                None,
                'Invalid options: profile scan does not know how to deal with'
                ' --fluctuate-fid.'
            )


    ############### Subparser setup ###############

    def setup_shared_arguments(self):
        """Sets up all subparsers so these are available to the analysis
        specific ones.
        """
        # looks up all methods whose names begin with "parse..." and calls them
        # (do NOT rename this method to something beginning with "parse")
        for obj in dir(self):
            if obj.startswith("parse"):
                getattr(self, obj)()


    ############### Subparsers ###############
    """
    Add your own by creating a function `parse_<property_group>` and setting
    a class attribute which you can then use as parent parser to create an
    analysis specific parser (see methods <command> further down).
    """

    def parse_blindness(self):
        """Parser for blindness"""
        parser = ArgumentParser(
            add_help=False
        )
        parser.add_argument(
            '--blind',
            action='store_true',
            help='''Blinded analysis. Do not show parameter values or store to
            logfiles.'''
        )
        self.blindness_parser = parser


    def parse_data_fluct(self):
        """Parser for data fluctuations"""
        parser = ArgumentParser(
            add_help=False
        )
        parser.add_argument(
            '--fluctuate-data',
            action='store_true',
            help='''Apply fluctuations to the data distribution. This should
            *not* be set for analyzing "real" (measured) data, and it is common
            to not use this feature even for Monte Carlo analysis. Note that if
            this is not set, --num-data-trials and --data-start-ind are forced
            to 1 and 0, respectively.'''
        )
        self.data_fluct_parser = parser


    def parse_data_pipeline(self):
        """Parser for pipeline of data"""
        parser = ArgumentParser(
            add_help = False
        )
        parser.add_argument(
            '--data-pipeline',
            type=str, action='append', default=None, metavar='PIPELINE_CFG',
            help='''Settings for the generation of "data" distributions; repeat
            this argument to specify multiple pipelines. If omitted, the same
            settings as specified for --h0-pipeline are used to generate data
            distributions (i.e., data is assumed to come from hypothesis h0.'''
        )
        self.data_pipeline_parser = parser


    def parse_data_pipeline_properties(self):
        """Parser for properties of data"""
        parser = ArgumentParser(
            add_help = False
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
        self.data_prop_parser = parser


    def parse_data_trials(self):
        """Parser for data trials"""
        parser = ArgumentParser(
            add_help=False
        )
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
            help='''Fluctuated data set index.'''
        )
        self.data_trial_parser = parser


    def parse_data_type(self):
        """Parser for data type"""
        parser = ArgumentParser(
            add_help=False
        )
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
        self.data_type_parser = parser


    def parse_fid_fluct(self):
        """Parser for fiducial fluctuations"""
        parser = ArgumentParser(
            add_help=False
        )
        parser.add_argument(
            '--fluctuate-fid',
            action='store_true',
            help='''Apply fluctuations to the fiducial distributions. If this
            flag is not set, --num-fid-trials and --fid-start-ind are forced to
            1 and 0, respectively.'''
        )
        self.fid_fluct_parser = parser


    def parse_fid_trials(self):
        """Parser for fiducial data trials"""
        parser = ArgumentParser(
            add_help=False
        )
        parser.add_argument(
            '--num-fid-trials',
            type=int, default=1,
            help='''Number of fiducial pseudodata trials to run. In our
            experience, it takes ~10^3-10^5 fiducial pseudodata trials to
            achieve low uncertainties on the resulting significance, though
            that exact number will vary based upon the details of an
            analysis.'''
        )
        parser.add_argument(
            '--fid-start-ind',
            type=int, default=0,
            help='''Fluctuated fiducial data index.'''
        )
        self.fid_trial_parser = parser


    def parse_fit(self):
        """Parser for fit (methods)"""
        parser = ArgumentParser(
            add_help=False
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
        parser.add_argument(
            '--no-min-history',
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
        self.fit_parser = parser


    def parse_git_info(self):
        """Parser for git info"""
        parser = ArgumentParser(
            add_help=False
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
        self.git_info_parser = parser


    def parse_h0_pipeline(self):
        """Parser for pipeline of hypothesis h0"""
        parser = ArgumentParser(
            add_help=False
        )
        parser.add_argument(
            '--h0-pipeline', required=True,
            type=str, action='append', metavar='PIPELINE_CFG',
            help='''Settings for the generation of hypothesis h0
            distributions; repeat this argument to specify multiple
            pipelines.'''
        )
        self.h0_pipeline_parser = parser


    def parse_h0_properties(self):
        """Parser for properties of hypothesis h0"""
        parser = ArgumentParser(
            add_help=False
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
        self.h0_prop_parser = parser


    def parse_h1_pipeline(self):
        """Parser for pipeline of hypothesis h1"""
        parser = ArgumentParser(
            add_help=False
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
        self.h1_pipeline_parser = parser


    def parse_h1_properties(self):
        """Parser for properties of hypothesis h1"""
        parser = ArgumentParser(
            add_help=False
        )
        parser.add_argument(
            '--h1-param-selections',
            type=str, default=None, metavar='PARAM_SELECTOR_LIST',
            help='''Comma-separated (no spaces) list of param selectors to apply to
            hypothesis h0's distribution maker's pipelines.'''
        )
        parser.add_argument(
            '--h1-name',
            type=str, metavar='NAME', default=None,
            help='''Name for hypothesis h0. E.g., "NO" for normal
            ordering in the neutrino mass ordering analysis. Note that the name
            here has no bearing on the actual process, so it's important that you
            be careful to use a name that appropriately identifies the
            hypothesis.'''
        )
        self.h1_prop_parser = parser


    def parse_inj_param_scan(self):
        """Parser for an injected-parameter scan"""
        parser = ArgumentParser(
            add_help=False
        )
        parser.add_argument(
            '--param-name',
            type=str, metavar='NAME', required=True,
            help='''Name of param to scan over. This must be in the config
            files defined above. One exception is that you can define this as
            `sin2theta23` and it will be interpreted not as theta23 values but
            as the square of the sine of theta23 values instead.'''
        )
        parser.add_argument(
            '--inj-vals', #TODO: should work just like the profile scan (values with units)
            type=str, required=True,
            help='''List of values to inject as true points in the parameter
            defined above. Must be something that numpy can interpret. In this
            script, numpy is imported as np so please use np in your string. An
            example would be np.linspace(0.35,0.65,31).'''
        )
        parser.add_argument(
            '--inj-units', #TODO: cf. above
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
        self.inj_param_scan_parser = parser


    def parse_log(self):
        """Parser for logs"""
        parser = ArgumentParser(
            add_help=False
        )
        parser.add_argument(
            '-d', '--logdir', required=True,
            metavar='DIR', type=str,
            help='Directory into which to store results and metadata.'
        )
        self.log_parser = parser


    def parse_metric(self):
        """Parser for fit metric"""
        parser = ArgumentParser(
            add_help = False
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
        self.metric_parser = parser


    def parse_profile_scan(self):
        """Parser for a profile scan"""
        parser = ArgumentParser(
            add_help=False
        )
        # allow for several parameters to be scanned
        parser.add_argument(
            '--param-name',
            type=str, action='append', required=True,
            help='''Provide the name of a parameter to scan. Repeat
            for multiple.'''
        )
        parser.add_argument(
            '--scan-vals',
            type=str, action='append', required=True,
            help='''List of values to scan (interpretable by numpy)
            for a parameter, with units. Provide one for each parameter
            passed via '--param-name' (in the same order).
            Example: "np.linspace(35,55,10)*ureg.deg".'''
        )
        parser.add_argument(
            '--no-outer',
            action='store_true',
            help='''Do not scan points as outer product of inner sequences.'''
        )
        parser.add_argument(
            '--no-profile',
            action='store_true',
            help='''Just run scan (no profile), i.e. do not optimise over
            remaining free parameters at each point.'''
        )
        parser.add_argument(
            '--store-intermediate',
            action='store_true',
            help='Overwrite output after each fit.'
        )
        self.profile_scan_parser = parser


    def parse_single_pipeline(self):
        """Parser for a generic pipeline"""
        parser = ArgumentParser(
            add_help=False
        )
        parser.add_argument(
            '--pipeline', required=True,
            type=str, action='append', metavar='PIPELINE_CFG',
            help='''Settings for the generation of h0, h1 and data
            distributions; repeat this argument to specify multiple
            pipelines.'''
        )
        self.single_pipeline_parser = parser


    def parse_syst_tests(self):
        """Parser for systematics tests"""
        parser = ArgumentParser(
            add_help=False
        )
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
        self.syst_tests_parser = parser


    def parse_verbosity(self):
        """Parser for verbosity"""
        parser = ArgumentParser(
            add_help = False
        )
        parser.add_argument(
            '-v', action='count', default=None,
            help='set verbosity level'
        )
        self.verbosity_parser = parser


    ############### Analysis script parsers ###############
    """These are available to be called as a subcommand of this script
    by the part of the name following 'command_'.
    """

    def command_discrete_hypo(self):
        parser = ArgumentParser(
            description='Discrete hypothesis test',
            parents=[self.log_parser, self.git_info_parser,
                     self.fit_parser, self.metric_parser, self.blindness_parser,
                     self.data_type_parser, self.data_pipeline_parser,
                     self.data_prop_parser, self.data_fluct_parser,
                     self.data_trial_parser, self.fid_fluct_parser,
                     self.fid_trial_parser, self.h0_pipeline_parser,
                     self.h0_prop_parser, self.h1_pipeline_parser,
                     self.h1_prop_parser, self.verbosity_parser]
        )
        return parser


    def command_inj_param_scan(self):
        parser = ArgumentParser(
            description='Injected-parameter scan',
            parents=[self.log_parser, self.git_info_parser, self.fit_parser,
                     self.metric_parser, self.single_pipeline_parser,
                     self.data_prop_parser, self.h0_prop_parser,
                     self.h1_prop_parser, self.inj_param_scan_parser,
                     self.verbosity_parser]
        )
        return parser


    def command_profile_scan(self):
        parser = ArgumentParser(
            description='Profile scan',
            parents=[self.log_parser, self.git_info_parser, self.fit_parser,
                     self.metric_parser, self.blindness_parser,
                     self.data_type_parser, self.data_pipeline_parser,
                     self.data_prop_parser, self.data_fluct_parser,
                     self.data_trial_parser, self.fid_fluct_parser,
                     self.fid_trial_parser, self.h0_pipeline_parser,
                     self.h0_prop_parser, self.profile_scan_parser,
                     self.verbosity_parser]
        )
        return parser


    def command_syst_tests(self):
        parser = ArgumentParser(
            description='Systematics tests',
            parents=[self.log_parser, self.git_info_parser, self.fit_parser,
                     self.metric_parser, self.single_pipeline_parser,
                     self.data_prop_parser, self.h0_prop_parser,
                     self.h1_prop_parser, self.syst_tests_parser,
                     self.verbosity_parser]
        )
        return parser


def main(return_outputs=False):
    """Main function that fires up all the parsers and executes the analysis
    wrapper script"""
    # fire up the parsers
    analysis_script = AnalysisScript()
    init_args_d = analysis_script.analysis_init_args_d
    # get the analysis wrapper script to run
    run_script = globals()[analysis_script.command]
    # pass it the initialization arguments and run!
    run_script(init_args_d=init_args_d, return_outputs=return_outputs)


if __name__ == '__main__':
    outputs = main(return_outputs=True) # pylint: disable=invalid-name
