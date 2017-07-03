# author : S.Wren
#
# date   : July 2017
"""
A script for doing analysis postprocessing.

"""

from argparse import ArgumentParser
import sys
import numpy as np

from pisa.utils.postprocess import Postprocessor
from pisa.utils.fileio import mkdir
from pisa.utils.log import logging, set_verbosity

def parse_args(description=__doc__, profile_scan=False,
               injparamscan=False, systtests=False,
               hypo_testing_analysis=False):
    """Parse command line args. The booleans are used to specify what
    type of postprocessing is being done and so the correct args will
    be displayed.

    Returns
    -------
    init_args_d : dict

    """
    parser = ArgumentParser(description=description)

    if not profile_scan:
        if injparamscan:
            parser.add_argument(
                '-d', '--dir', required=True,
                metavar='DIR', type=str, action='append',
                help='''Directory containing output of hypo_testing.py.
                Repeat this argument to plot multiple significance lines on
                the same plot. Note that if you do then none of the fits or
                the minimiser info will be plotted'''
            )
            parser.add_argument(
                '-dl', '--dir_label', type=str, action='append',
                help="""A unique name from which to identify each the above
                directories can be identified. Repeat this argument for as
                many times as you have directories. If no labels are
                specified here they will be constructed using the truth
                information in the files. So either specify one for
                every directory or none at all."""
            )
        else:
            parser.add_argument(
                '-d', '--dir', required=True,
                metavar='DIR', type=str,
                help='''Directory containing output of hypo_testing.py.'''
            )
            if not systtests:
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
    else:
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
    if hypo_testing_analysis:
        parser.add_argument(
            '-LLR', '--llr_plots', action='store_true', default=False,
            help='''Flag to make the LLR plots. This will give the
            actual analysis results.'''
        )
        parser.add_argument(
            '-FM', '--fit_information', action='store_true', default=False,
            help='''Flag to make tex files containing the
            fiducial fit params and metric.'''
        )
        parser.add_argument(
            '-MM', '--minim_information', action='store_true', default=False,
            help='''Flag to make plots of the minimiser information i.e. status,
            number of iterations, time taken etc.'''
        )
        parser.add_argument(
            '-IP', '--individual_posteriors', action='store_true',
            default=False,
            help='''Flag to plot individual posteriors.'''
        )
        parser.add_argument(
            '-CP', '--combined_posteriors', action='store_true', default=False,
            help='''Flag to plot combined posteriors for each h0 and h1
            combination.'''
        )
        parser.add_argument(
            '-IOP', '--individual_overlaid_posteriors', action='store_true',
            default=False,
            help='''Flag to plot individual overlaid posteriors. Overlaid
            here means that for a plot will be made with each of the h0
            and h1 returned values on the same plot for each of the
            fiducial h0 and h1 pseudos.'''
        )
        parser.add_argument(
            '-COP', '--combined_overlaid_posteriors', action='store_true',
            default=False,
            help='''Flag to plot combined overlaid posteriors.'''
        )
        parser.add_argument(
            '-IS', '--individual_scatter', action='store_true', default=False,
            help='''Flag to plot individual 2D scatter plots of posteriors.'''
        )
        parser.add_argument(
            '-CIS', '--combined_individual_scatter',
            action='store_true', default=False,
            help='''Flag to plot all 2D scatter plots of one systematic
            with every other systematic on one plot for each h0 and h1
            combination.'''
        )
        parser.add_argument(
            '-CS', '--combined_scatter', action='store_true', default=False,
            help='''Flag to plot all 2D scatter plots on one plot for each
            h0 and h1 combination.'''
        )
        parser.add_argument(
            '-CM', '--correlation_matrix', action='store_true', default=False,
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
    elif injparamscan:
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
            '-SIG', '--significances', action='store_true', default=False,
            help='''Flag to make the Asimov significance plots. This will
            give the actual results of the study.'''
        )
        parser.add_argument(
            '-MM', '--minim_information', action='store_true', default=False,
            help='''Flag to make plots of the minimiser information i.e.
            status, number of iterations, time taken etc.'''
        )
        parser.add_argument(
            '-IF', '--individual_fits', action='store_true', default=False,
            help='''Flag to make plots of all of the best fit parameters
            separated by the fitted parameter.'''
        )
        parser.add_argument(
            '-CF', '--combined_fits', action='store_true', default=False,
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
    if profile_scan:
        args = parser.parse_args(sys.argv[2:])
    else:
        args = parser.parse_args(sys.argv[3:])
    init_args_d = vars(args)

    set_verbosity(init_args_d.pop('v'))

    init_args_d['formats'] = []
    if args.png:
        init_args_d['formats'].append('png')
    if args.pdf:
        init_args_d['formats'].append('pdf')

    if len(init_args_d['formats']) > 0:
        logging.info(
            "Files will be saved in format(s) %s"%init_args_d['formats']
        )
    else:
        raise ValueError('Must specify a plot file format, either --png or'
                         ' --pdf (or both), when processing llr results.')

    return init_args_d


class PostprocessArgParser(object):
    """
    Allows for clever usage of this script such that all of the
    postprocessing can be contained in this single script.
    """
    def __init__(self):
        parser = ArgumentParser(
            description="""This script contains all of the functionality for
            processing the output of analyses""",
            usage="""postprocess.py <command> [<subcommand>] [<args>]

            There are two commands that can be issued:

              hypo_testing    Processes output from some form of hypo_testing.
              profile_scan    Processes output from some form of profile_scan.

            Run postprocess.py <command> -h to see the different possible 
            subcommands/arguments to each of these commands."""
        )
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        expected_commands = ['hypo_testing', 'profile_scan']
        if not hasattr(self, args.command):
            raise ValueError(
                "The command issued, %s, was not one of the expected commands"
                " - %s."%(args.command, expected_commands)
            )
        else:
            getattr(self, args.command)()

    def hypo_testing(self):
        """Does the main when the user selects hypo_testing"""
        main_hypo_testing()

    def profile_scan(self):
        """Does the main when the user selects profile_scan"""
        main_profile_scan()


class PostprocessHypoTestingArgParser(object):
    """
    Allows for further clever usage of this script such that all of the
    hypo_testing postprocessing can be contained in this single script.
    """
    def __init__(self):
        parser = ArgumentParser(
            description="""This script contains all of the functionality for
            processing the output of hypo_testing analyses""",
            usage="""postprocess.py hypo_testing [<subcommand>] [<args>]

            There are three subcommands that can be issued:

              analysis        Processes output from the standard hypothesis 
                              testing analyses.
              injparamscan    Processes output from a scan over some injected
                              parameter in the data.
              systtests       Processes output from tests on the impact of
                              systematics on the analysis.

            Run postprocess.py hypo_testing <subcommand> -h to see the
            different possible arguments to each of these commands."""
        )
        parser.add_argument('subcommand', help='Subcommand to run')
        args = parser.parse_args(sys.argv[2:3])
        expected_commands = ['analysis', 'injparamscan', 'systtests']
        if not hasattr(self, args.subcommand):
            raise ValueError(
                "The command issued, %s, was not one of the expected commands"
                " - %s."%(args.subcommand, expected_commands)
            )
        else:
            getattr(self, args.subcommand)()

    def analysis(self):
        """Does the main when the user selects hypo_testing analysis"""
        main_analysis_postprocessing()

    def injparamscan(self):
        """Does the main when the user selects hypo_testing injparamscan"""
        main_injparamscan_postprocessing()

    def systtests(self):
        """Does the main when the user selects hypo_testing systtests"""
        main_systtests_postprocessing()


def main_hypo_testing():
    """Parses the args when the user selects hypo_testing"""
    PostprocessHypoTestingArgParser()


def main_profile_scan():
    description = """A script for processing the output files of
    profile_scan.py"""

    init_args_d = parse_args(description=description,
                             profile_scan=True)

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
        postprocessor.plot_1D_scans()

    # 2D profile scans
    elif len(postprocessor.all_bin_cens) == 2:
        postprocessor.plot_2D_scans()

        if (postprocessor.all_bin_names[0] == 'theta23') and \
           (postprocessor.all_bin_names[1] == 'deltam31'):

            postprocessor.add_deltam32_sin2theta23()
            postprocessor.plot_2D_scans(
                xlabel='sin2theta23',
                xunits='dimensionless',
                ylabel='deltam32'
            )

    else:
        raise NotImplementedError(
            "Postprocessing of profile scans in anything other than 1D or "
            " 2D not implemented in this script."
        )


def main_analysis_postprocessing():
    description = """Hypothesis testing: How do two hypotheses compare for
    describing MC or data?

    This computes significances, etc. from the logfiles recorded by the
    `hypo_testing.py` script, for either Asimov or llr analysis. Plots and
    tables are produced in the case of llr analysis."""

    # TODO:
    #
    # 1) Some of the "combined" plots currently make it impossible to read the
    #    axis labels. Come up with a better way of doing this. Could involve
    #    making legends and just labelling the axes alphabetically.

    init_args_d = parse_args(description=description,
                             hypo_testing_analysis=True)

    if init_args_d['asimov']:
        # TODO - Something like the necessary function is there with
        # calculate_deltachi2_significances but exactly how to output
        # this should probably be thought about
        raise NotImplementedError(
            "Postprocessing of Asimov analysis not implemented yet."
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


def main_injparamscan_postprocessing():
    description = """Hypothesis testing: How do two hypotheses compare for
    describing MC or data?

    This computes significances, etc. from the logfiles recorded by the
    `hypo_testing.py` script for a scan over some injected parameter.
    The main result will be an Asimov sensitivity curve as a function of
    this inejcted parameter."""

    init_args_d = parse_args(description=description,
                             injparamscan=True)

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
        if init_args_d['individual_fits'] or init_args_d['combned_fits'] or \
           init_args_d['minim_information']:
            raise ValueError(
                "You have specified multiple input directories but have "
                "also requested to make plots of the fit parameters or the "
                "minimiser information. Multiple input directories are "
                "only compatible with plotting the significances overlaid."
            )


def main_systtests_postprocessing():
    description = """Hypothesis testing: How do two hypotheses compare for
    describing MC or data?

    This script/module computes significances, etc. from the logfiles recorded
    by the `hypo_testing_systtests.py` script. That is, looks at how the fits
    change for three different N-1 tests:

      1) Where one of the systematics is fixed to the baseline value.
      2) Where one of the systematics is injected *off* baseline but fixed
         *on* baseline in the hypotheses.
      3) Same as 2, but the systematic is not fixed and so the minimiser is
         allowed to try correct for the incorrect hypothesis."""

    init_args_d = parse_args(description=description,
                             systtests=True)

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


if __name__ == '__main__':
    PostprocessArgParser()
