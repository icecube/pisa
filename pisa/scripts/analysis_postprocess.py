#!/usr/bin/env python

"""
Postprocess the outputs of a PISA analysis.
"""


from __future__ import absolute_import

from argparse import ArgumentParser
from collections import OrderedDict
from os.path import basename
import sys
import numpy as np

from pisa.utils.fileio import mkdir
from pisa.utils.log import logging, set_verbosity
from pisa.utils.postprocess import Postprocessor
from pisa.utils.scripting import get_script, parse_command


__all__ = ['SCRIPT', 'parse_args', 'postproc_profile_scan',
           'postproc_discrete_hypo', 'postproc_inj_param_scan',
           'postproc_syst_tests', 'parse_hypo_testing_subcommand', 'main']

__author__ = 'S. Wren, J.L. Lanfranchi'


SCRIPT = basename(get_script())


def parse_args(command, description):
    """Parse command line args, where `command` defines what args are to be
    displayed / accepted...

    Parameters
    ----------
    command : string
        Command passed to the script

    description : string
        Description line(s) to print to terminal in help message (i.e. if -h
        or --help is passed)

    Returns
    -------
    init_args_d : dict
        Command line arguments passed via the command line, in a dictionary

    """
    assert command in ['profile_scan', 'hypo_testing', 'inj_param_scan',
                       'syst_tests']

    parser = ArgumentParser(description=description)

    if command == 'inj_param_scan':
        parser.add_argument(
            '-d', '--dir', required=True,
            metavar='DIR', type=str, action='append',
            help='''Directory containing output of hypo_testing.py.
            Repeat this argument to plot multiple significance lines on
            the same plot. Note that if you do then none of the fits or
            the minimiser info will be plotted'''
        )
        parser.add_argument(
            '--dir-label', type=str, action='append',
            help="""A unique name from which to identify each the above
            directories can be identified. Repeat this argument for as
            many times as you have directories. If no labels are
            specified here they will be constructed using the truth
            information in the files. So either specify one for
            every directory or none at all."""
        )

    if command in ['hypo_testing', 'syst_tests']:
        parser.add_argument(
            '-d', '--dir', required=True,
            metavar='DIR', type=str,
            help='''Directory containing output of hypo_testing.py.'''
        )

    if command == 'hypo_testing':
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            '--asimov', action='store_true',
            help='''Analyze the Asimov trials in the specified
            directories.'''
        )
        group.add_argument(
            '--llr', action='store_true',
            help='''Analyze the LLR trials in the specified
            directories.'''
        )

    if command == 'profile_scan':
        parser.add_argument(
            '--infile', metavar='FILE', type=str, required=True,
            help='''Output file of profile_scan.py to processs.'''
        )
        parser.add_argument(
            '--best-fit-infile', metavar='FILE', type=str, default=None,
            help='''Output file of profile_scan.py containing the best
            fit to add to the plots, if available.'''
        )
        parser.add_argument(
            '--projection-infile', metavar='FILE',
            type=str, action='append', default=None,
            help='''If you want to add projections to your plots e.g. 1D
            projections to 2D plots you can specify them here. Repeat this
            argument to specify multiple projections.'''
        )
        parser.add_argument(
            '--other-contour', metavar='FILE',
            type=str, action='append', default=None,
            help='''If you want to add other contours to your plots e.g.
            Other experiments then specify them here. This is expected to
            be a json dictionary with the following keys: vars, contour,
            label, color, linestyle and (optionally) the best_fit point.'''
        )
        parser.add_argument(
            '--pseudo-experiments', metavar='DIR',
            type=str, default=None,
            help='''If you want to overlay pseudo experiment fits from
            the hypo_testing.py script on to the contours to check
            coverage, set the directory here. Note that this will overlay
            all of the hX_hypo_to_hY_fid fit results on to the contour
            so you can select the appropriate one after the script is run.'''
        )

    parser.add_argument(
        '--detector', type=str, default='',
        help='''Name of detector to put in histogram titles.'''
    )
    parser.add_argument(
        '--selection', type=str, default='',
        help='''Name of selection to put in histogram titles.'''
    )

    if command == 'hypo_testing':
        parser.add_argument(
            '--llr-plots', action='store_true', default=False,
            help='''Flag to make the LLR plots. This will give the
            actual analysis results.'''
        )
        parser.add_argument(
            '--fit-information', action='store_true', default=False,
            help='''Flag to make tex files containing the
            fiducial fit params and metric.'''
        )
        parser.add_argument(
            '--minim-information', action='store_true', default=False,
            help='''Flag to make plots of the minimiser information i.e. status,
            number of iterations, time taken etc.'''
        )
        parser.add_argument(
            '--individual-posteriors', action='store_true',
            default=False,
            help='''Flag to plot individual posteriors.'''
        )
        parser.add_argument(
            '--combined-posteriors', action='store_true', default=False,
            help='''Flag to plot combined posteriors for each h0 and h1
            combination.'''
        )
        parser.add_argument(
            '--individual-overlaid-posteriors', action='store_true',
            default=False,
            help='''Flag to plot individual overlaid posteriors. Overlaid
            here means that for a plot will be made with each of the h0
            and h1 returned values on the same plot for each of the
            fiducial h0 and h1 pseudos.'''
        )
        parser.add_argument(
            '--combined-overlaid-posteriors', action='store_true',
            default=False,
            help='''Flag to plot combined overlaid posteriors.'''
        )
        parser.add_argument(
            '--individual-scatter', action='store_true', default=False,
            help='''Flag to plot individual 2D scatter plots of posteriors.'''
        )
        parser.add_argument(
            '--combined-individual-scatter',
            action='store_true', default=False,
            help='''Flag to plot all 2D scatter plots of one systematic
            with every other systematic on one plot for each h0 and h1
            combination.'''
        )
        parser.add_argument(
            '--combined-scatter', action='store_true', default=False,
            help='''Flag to plot all 2D scatter plots on one plot for each
            h0 and h1 combination.'''
        )
        parser.add_argument(
            '--correlation-matrix', action='store_true', default=False,
            help='''Flag to plot the correlation matrices for each h0 and h1
            combination.'''
        )
        parser.add_argument(
            '--threshold', type=float, default=0.0,
            help='''Sets the threshold for which to remove 'outlier' trials.
            Ideally this will not be needed at all, but it is there in case 
            of e.g. failed minimiser. The higher this value, the more outliers
            will be included. Do not set this parameter if you want all trials
            to be included.'''
        )
        parser.add_argument(
            '--extra-points', type=str, action='append',
            help='''Extra lines to be added to the LLR plots. This is useful,
            for example, when you wish to add specific LLR fit values to the
            plot for comparison. These should be supplied as a single value
            e.g. x1 or as a path to a file with the value provided in one
            column that can be intepreted by numpy genfromtxt. Repeat this
            argument in conjunction with the extra points label below to
            specify multiple (and uniquely identifiable) sets of extra 
            points.'''
        )
        parser.add_argument(
            '--extra-points-labels', type=str, action='append',
            help='''The label(s) for the extra points above.'''
        )

    if command == 'inj_param_scan':
        parser.add_argument(
            '--inj-param-units', type=str, default=None,
            help="""If you know the units that you injected the parameter
            with and you expect that the script will not be able to find
            this by looking at the fit parameters in the config file
            (i.e. theta13 may be defined in degrees in the config file
            but you injected it in radians) then use this argument to
            explicitly set it for use in the plot labels."""
        )
        parser.add_argument(
            '--significances', action='store_true', default=False,
            help='''Flag to make the Asimov significance plots. This will
            give the actual results of the study.'''
        )
        parser.add_argument(
            '--minim-information', action='store_true', default=False,
            help='''Flag to make plots of the minimiser information i.e.
            status, number of iterations, time taken etc.'''
        )
        parser.add_argument(
            '--individual-fits', action='store_true', default=False,
            help='''Flag to make plots of all of the best fit parameters
            separated by the fitted parameter.'''
        )
        parser.add_argument(
            '--combined-fits', action='store_true', default=False,
            help='''Flag to make plots of all of the best fit parameters joined
            together.'''
        )
        parser.add_argument(
            '--extra-points', type=str, action='append', metavar='LIST',
            help='''Extra points to be added to the plots. This is useful,
            for example, when you wish to add LLR results to the plot.
            These should be supplied as a list of tuples e.g.
            "[(x1,y1),(x2,y2)]" or "[(x1,y1,y1err),(x2,y2,y2err)]" or 
            "[(x1,y1,y1uperr,y1downerr),(x2,y2,y2uperr,y2downerr)]" or
            as a path to a file with the values provided in columns that
            can be intepreted by numpy genfromtxt. Repeat this argument in
            conjunction with the extra points label below to specify
            multiple (and uniquely identifiable) sets of extra points.'''
        )
        parser.add_argument(
            '--extra-points-labels', type=str, action='append',
            help='''The label(s) for the extra points above.'''
        )

    parser.add_argument(
        '--outdir', metavar='DIR', type=str, default=None,
        help='''Store all output plots to this directory. This will make
        further subdirectories, if needed, to organise the output plots.'''
    )
    parser.add_argument(
        '--pdf', action='store_true',
        help='''Produce pdf plot(s).'''
    )
    parser.add_argument(
        '--png', action='store_true',
        help='''Produce png plot(s).'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='''set verbosity level'''
    )

    if command == 'profile_scan':
        args = parser.parse_args(sys.argv[2:])
    else: # inj_param_scan, syst_tests, and hypo_testing
        args = parser.parse_args(sys.argv[3:])
    init_args_d = vars(args)

    set_verbosity(init_args_d.pop('v'))

    init_args_d['formats'] = []
    if args.png:
        init_args_d['formats'].append('png')
    if args.pdf:
        init_args_d['formats'].append('pdf')

    if init_args_d['formats']:
        logging.info('Files will be saved in format(s) %s',
                     init_args_d['formats'])

    return init_args_d


def postproc_profile_scan(return_outputs=False):
    """Process the output files of profile_scan"""

    init_args_d = parse_args(
        description=postproc_profile_scan.__doc__,
        command='profile_scan'
    )

    if init_args_d['pseudo_experiments'] is not None:
        fluctuate_fid = True
        fluctuate_data = False
    else:
        fluctuate_fid = None
        fluctuate_data = None

    mkdir(init_args_d['outdir'])

    postprocessor = Postprocessor(
        analysis_type='profile_scan',
        detector=init_args_d['detector'],
        selection=init_args_d['selection'],
        outdir=init_args_d['outdir'],
        formats=init_args_d['formats'],
        scan_file=init_args_d['infile'],
        best_fit_file=init_args_d['best_fit_infile'],
        projection_files=init_args_d['projection_infile'],
        other_contours=init_args_d['other_contour'],
        pseudo_experiments=init_args_d['pseudo_experiments'],
        fluctuate_fid=fluctuate_fid,
        fluctuate_data=fluctuate_data
    )

    # 1D profile scans
    if len(postprocessor.all_bin_cens) == 1:
        postprocessor.plot_1d_scans()

    # 2D profile scans
    elif len(postprocessor.all_bin_cens) == 2:
        postprocessor.plot_2d_scans()

        if (postprocessor.all_bin_names[0] == 'theta23'
                and postprocessor.all_bin_names[1] == 'deltam31'):

            postprocessor.add_deltam32_sin2theta23()
            postprocessor.plot_2d_scans(
                xlabel='sin2theta23',
                xunits='dimensionless',
                ylabel='deltam32'
            )

    else:
        raise NotImplementedError(
            'Postprocessing of profile scans in anything other than 1D or '
            ' 2D not implemented in this script.'
        )

    if return_outputs:
        return postprocessor


def postproc_discrete_hypo(return_outputs=False):
    """Hypothesis testing: How do two hypotheses compare for
    describing MC or data?

    This computes significances, etc. from the logfiles recorded by the
    `hypo_testing.py` script, for either Asimov or llr analysis. Plots and
    tables are produced in the case of llr analysis."""

    # TODO:
    #
    # 1) Some of the "combined" plots currently make it impossible to read the
    #    axis labels. Come up with a better way of doing this. Could involve
    #    making legends and just labelling the axes alphabetically.

    init_args_d = parse_args(
        description=postproc_discrete_hypo.__doc__,
        command='hypo_testing'
    )

    if init_args_d['asimov']:
        # TODO - Something like the necessary function is there with
        # calculate_deltachi2_significances but exactly how to output
        # this should probably be thought about
        raise NotImplementedError(
            'Postprocessing of Asimov analysis not implemented yet.'
        )

    # Otherwise: llr analysis
    if init_args_d['outdir'] is None:
        raise ValueError('Must specify --outdir when processing llr results.')

    postprocessor = Postprocessor(
        analysis_type='hypo_testing',
        test_type='analysis',
        logdir=init_args_d['dir'],
        detector=init_args_d['detector'],
        selection=init_args_d['selection'],
        outdir=init_args_d['outdir'],
        formats=init_args_d['formats'],
        fluctuate_fid=True,
        fluctuate_data=False,
        extra_points=init_args_d['extra_points'],
        extra_points_labels=init_args_d['extra_points_labels']
    )

    trial_nums = postprocessor.data_sets[
        postprocessor.labels.dict['data']
    ]['h0_fit_to_h1_fid'].keys()

    if init_args_d['threshold'] != 0.0:
        logging.info('Outlying trials will be removed with a '
                     'threshold of %.2f', init_args_d['threshold'])
        postprocessor.purge_outlying_trials(
            trial_nums=np.array(trial_nums),
            thresh=init_args_d['threshold']
        )
    else:
        logging.info('All trials will be included in the analysis.')

    if init_args_d['llr_plots']:
        if len(trial_nums) != 1:
            postprocessor.make_llr_plots()
        else:
            raise ValueError(
                "llr plots were requested but only 1 trial "
                "was found in the logdir."
            )
    if init_args_d['fit_information']:
        postprocessor.make_fiducial_fit_files()
    if init_args_d['minim_information']:
        postprocessor.make_fit_information_plots()
    if init_args_d['individual_posteriors']:
        postprocessor.make_posterior_plots()
    if init_args_d['combined_posteriors']:
        postprocessor.make_posterior_plots(combined=True)
    if init_args_d['individual_overlaid_posteriors']:
        postprocessor.make_overlaid_posterior_plots()
    if init_args_d['combined_overlaid_posteriors']:
        postprocessor.make_overlaid_posterior_plots(combined=True)
    if init_args_d['individual_scatter']:
        postprocessor.make_scatter_plots()
    if init_args_d['combined_individual_scatter']:
        postprocessor.make_scatter_plots(combined=True, singlesyst=True)
    if init_args_d['combined_scatter']:
        postprocessor.make_scatter_plots(combined=True)
    if init_args_d['correlation_matrix']:
        postprocessor.make_scatter_plots(matrix=True)


def postproc_inj_param_scan(return_outputs=False):
    """Hypothesis testing: How do two hypotheses compare for
    describing MC or data?

    This computes significances, etc. from the logfiles recorded by the
    `hypo_testing.py` script for a scan over some injected parameter.
    The main result will be an Asimov sensitivity curve as a function of
    this inejcted parameter."""

    init_args_d = parse_args(
        description=postproc_inj_param_scan.__doc__,
        command='inj_param_scan'
    )

    postprocessor = Postprocessor(
        analysis_type='hypo_testing',
        test_type='injparamscan',
        logdir=init_args_d['dir'],
        detector=init_args_d['detector'],
        selection=init_args_d['selection'],
        outdir=init_args_d['outdir'],
        formats=init_args_d['formats'],
        fluctuate_fid=False,
        fluctuate_data=False,
        extra_points=init_args_d['extra_points'],
        extra_points_labels=init_args_d['extra_points_labels'],
        inj_param_units=init_args_d['inj_param_units']
    )

    if len(postprocessor.data_sets) == 1:
        if postprocessor.wh_to_th[0]['params'].keys() == ['bestfit', 'altit']:
            if init_args_d['individual_fits'] or init_args_d['combined_fits']:
                raise ValueError(
                    "You have requested to make plots of the best fit "
                    "points of the systematic parameters but this is "
                    "not possible snce there are none included in "
                    "this analysis."
                )
        if init_args_d['significances']:
            postprocessor.make_asimov_significance_plots()
        if init_args_d['minim_information']:
            postprocessor.make_fit_information_plots()
        if init_args_d['individual_fits']:
            postprocessor.make_asimov_fit_parameter_plots()
        if init_args_d['combined_fits']:
            postprocessor.make_asimov_fit_parameter_plots(combined=True)
    else:
        if (init_args_d['individual_fits'] or init_args_d['combned_fits']
                or init_args_d['minim_information']):
            raise ValueError(
                "You have specified multiple input directories but have "
                "also requested to make plots of the fit parameters or the "
                "minimiser information. Multiple input directories are "
                "only compatible with plotting the significances overlaid."
            )


def postproc_syst_tests(return_outputs=False):
    """Hypothesis testing: How do two hypotheses compare for
    describing MC or data?

    This script/module computes significances, etc. from the logfiles recorded
    by the `systematics_tests.py` script. That is, looks at how the fits
    change for three different N-1 tests:

      1) Where one of the systematics is fixed to the baseline value.
      2) Where one of the systematics is injected *off* baseline but fixed
         *on* baseline in the hypotheses.
      3) Same as 2, but the systematic is not fixed and so the minimiser is
         allowed to try correct for the incorrect hypothesis."""

    init_args_d = parse_args(
        description=postproc_syst_tests.__doc__,
        command='syst_tests'
    )

    postprocessor = Postprocessor(
        analysis_type='hypo_testing',
        test_type='systtests',
        logdir=init_args_d['dir'],
        detector=init_args_d['detector'],
        selection=init_args_d['selection'],
        outdir=init_args_d['outdir'],
        formats=init_args_d['formats'],
        fluctuate_fid=False,
        fluctuate_data=False
    )

    postprocessor.make_systtest_plots()


HYPO_TESTING_COMMANDS = OrderedDict([
    ('discrete_hypo', postproc_discrete_hypo),
    ('inj_param_scan', postproc_inj_param_scan),
    ('syst_tests', postproc_syst_tests)
])

HYPO_TESTING_DESCR = (
    'Process the outputs produced by pisa-analysis script'
)

HYPO_TESTING_SUBCOMMAND_STR = '\n'.join([
    '  {0:16s} Processes outputs of pisa-analysis {0} ...'.format(cmd)
    for cmd in HYPO_TESTING_COMMANDS.keys()
])


HYPO_TESTING_USAGE = '''{0} hypo_testing [<subcommand>] [<args>]

The subcommands that can be issued are:

{1}

Run
  {0} hypo_testing <subcommand> -h
to see the valid arguments for each of the above subcommands
'''.format(SCRIPT, HYPO_TESTING_SUBCOMMAND_STR)


def parse_hypo_testing_subcommand(return_outputs=False):
    """Parse command line args for hypo_testing subcommand"""
    return parse_command(command_depth=1,
                         commands=HYPO_TESTING_COMMANDS,
                         description=HYPO_TESTING_DESCR,
                         usage=HYPO_TESTING_USAGE,
                         return_outputs=return_outputs)


MAIN_CMD_SPEC = dict(
    commands=OrderedDict([
        ('hypo_testing', parse_hypo_testing_subcommand),
        ('profile_scan', postproc_profile_scan)
    ]),
    description='Postprocess outputs generated by a PISA analysis.',
    usage='''{0} <command> [<subcommand>] [<args>]

The commands that can be issued are:

  hypo_testing    Processes output from one of the hypo_testing commands.
  profile_scan    Processes output from profile_scan.

Run
  {0} <command> -h
to see the possible subcommands/arguments to each command.'''.format(SCRIPT)
)


def main(return_outputs=False):
    """main"""
    return parse_command(command_depth=0, return_outputs=return_outputs,
                         **MAIN_CMD_SPEC)


if __name__ == '__main__':
    outputs = main(return_outputs=True) # pylint: disable=invalid-name
