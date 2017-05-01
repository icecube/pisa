"""
A class for doing postprocessing.

"""

from argparse import ArgumentParser

from collections import OrderedDict
import os
import re
import sys
import numpy as np

from pisa import ureg
from pisa.analysis.hypo_testing import Labels
from pisa.utils.fileio import from_file, mkdir, nsort, to_file
from pisa.utils.log import logging, set_verbosity


def parse_args(description=__doc__, profile_scan=False,
               injparamscan=False, systtests=False,
               hypo_testing_analysis=False):
    """Parse command line args.

    Returns
    -------
    init_args_d : dict

    """
    parser = ArgumentParser(description=description)

    if not profile_scan:
        parser.add_argument(
            '-d', '--dir', required=True,
            metavar='DIR', type=str,
            help='''Directory containing output of hypo_testing.py.'''
        )
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            '--asimov', action='store_true',
            help='''Analyze the Asimov trials in the specified directories.'''
        )
        group.add_argument(
            '--llr', action='store_true',
            help='''Analyze the LLR trials in the specified directories.'''
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
        '--detector', type=str, default='',
        help='''Name of detector to put in histogram titles.'''
    )
    parser.add_argument(
        '--selection', type=str, default='',
        help='''Name of selection to put in histogram titles.'''
    )
    if hypo_testing_analysis:
        parser.add_argument(
            '-FM', '--fit_information', action='store_true', default=False,
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
            '--extra-point', type=str, action='append',
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
            '--extra-point-label', type=str, action='append',
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
    
    return init_args_d


class Postprocessingargparser(object):
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
        main_hypo_testing()

    def profile_scan(self):
        main_profile_scan()


class Hypotestingpostprocessingargparser(object):
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
        print args
        if not hasattr(self, args.subcommand):
            raise ValueError(
                "The command issued, %s, was not one of the expected commands"
                " - %s."%(args.subcommand, expected_commands)
            )
        else:
            getattr(self, args.subcommand)()

    def analysis(self):
        main_analysis_postprocessing()

    def injparamscan(self):
        main_injparamscan_postprocessing()

    def systtests(self):
        main_systtests_postprocessing()

class Postprocessor(object):
    """Class to contain all of the functions that are used by the various 
    postprocessing scripts.


    Parameters
    ----------
    analysis_type : string
        Name for the type of analysis this was run e.g. hypo_testing,
        profile_scan etc.
    test_type : string
        Name for the type of test then done. This may be none, but may also be
        something along the line of `systematic_tests` etc.
    logdir : string
        Path to logging directory where files are stored. This should 
        contain e.g. the "config_summary.json" file.
    fluctuate_fid : bool
        Whether the trials you're interested in applied fluctuations to the
        fiducial-fit Asimov distributions. `fluctuate_fid` False is
        equivalent to specifying an Asimov analysis (so long as the metric
        used was chi-squared).
    fluctuate_data : bool
        Whether the trials you're interested in applied fluctuations to the
        (toy) data. This is invalid if actual data was processed.

    Note that a single `logdir` can have different kinds of analyses run 
    and results be logged within, so `fluctuate_fid` and `fluctuate_data`
    allows these to be separated from one another.
    """

    def __init__(self, analysis_type, detector, selection, outdir, formats,
                 test_type=None, logdir=None, fluctuate_fid=None,
                 fluctuate_data=None, scan_file=None, best_fit_file=None):
        expected_analysis_types = ['hypo_testing', 'profile_scan']
        if analysis_type not in expected_analysis_types:
            raise ValueError(
                "Postprocessing only implemented for analyses of type %s "
                "but have been asked to process %s."%(
                    expected_analysis_types, analysis_type)
            )
        if analysis_type == 'hypo_testing':
            expected_test_types = ['analysis', 'injparamscan', 'systtests']
        elif analysis_type == 'profile_scan':
            expected_test_types = [None]
        if test_type not in expected_test_types:
            raise ValueError(
                "Postprocessing only implemented for %s analyses of test "
                "type %s but have been asked to process %s."%(
                    analysis_type, expected_test_types, test_type)
            )
        # Things to store for all postprocessing
        self.detector = detector
        self.selection = selection
        self.outdir = outdir
        self.formats = formats
        # Things to initialise for hypo_testing
        if analysis_type == 'hypo_testing':
            self.analysis_type = analysis_type
            self.test_type = test_type
            self.fluctuate_fid = fluctuate_fid
            self.fluctuate_data = fluctuate_data
            self.logdir = logdir
            self.extract_trials()
        elif analysis_type == 'profile_scan':
            self.scan_file_dict = from_file(scan_file)
            if best_fit_file is not None:
                self.best_fit_dict = from_file(best_fit_file)
            else:
                self.best_fit_dict = None
            self.get_scan_steps()
            self.get_scan_data()

    def store_extra_points(self, extra_points, extra_points_labels):
        """Stores the extra points to self"""
        if extra_points is not None:
            if extra_points_labels is not None:
                if len(extra_points) != len(extra_points_labels):
                    raise ValueError(
                        'You must specify at least one label for each set of '
                        'extra points. Got %i label(s) for %s set(s) of '
                        'extra points.'%(
                            len(extra_points), len(extra_points_labels)
                        )
                    )
            else:
                raise ValueError(
                    'You have specified %i set(s) of extra points but no '
                    'labels to go with them.'%len(extra_points)
                )
        else:
            if extra_points_labels is not None:
                raise ValueError(
                    'You have specified %i label(s) for extra points but no' 
                    ' set(s) of extra points.'%len(extra_points_labels)
                )
        self.extra_points = extra_points
        self.extra_points_labels = extra_points_labels

    def extract_trials(self):
        """Extract and aggregate analysis results."""
        self.logdir = os.path.expanduser(os.path.expandvars(self.logdir))
        logdir_content = os.listdir(self.logdir)
        if 'config_summary.json' in logdir_content:
            # Look for the pickle files in the directory to indicate that this
            # data may have already been processed.
            config_summary_fpath = os.path.join(
                self.logdir,
                'config_summary.json'
            )
            cfg = from_file(config_summary_fpath)
            self.data_is_data = cfg['data_is_data']
            # Get naming scheme
            self.labels = Labels(
                h0_name=cfg['h0_name'], h1_name=cfg['h1_name'],
                data_name=cfg['data_name'],
                data_is_data=self.data_is_data,
                fluctuate_data=self.fluctuate_data,
                fluctuate_fid=self.fluctuate_fid
            )
            pickle_there = self.check_pickle_files(logdir_content)
            if pickle_there:
                self.load_from_pickle()
            else:
                if self.data_is_data and self.fluctuate_data:
                    raise ValueError('Analysis was performed on data, so '
                                     '`fluctuate_data` is not supported.')
                # Get starting params
                self.get_starting_params(cfg=cfg)
                # Find all relevant data dirs, and from each extract the
                # fiducial fit(s) information contained
                self.get_data(cfg=cfg)
                self.pickle_data()
        else:
            raise ValueError(
                'config_summary.json cannot be found in the specified logdir. '
                'It should have been created as part of the output of '
                'hypo_testing.py and so this postprocessing cannot be '
                'performed.'
            )

    def extract_fit(self, fpath, keys=None):
        """Extract fit info from a file.

        Parameters
        ----------
        fpath : string
            Path to the file

        keys : None, string, or iterable of strings
            Keys to extract. If None, all keys are extracted.

        """
        try:
            info = from_file(fpath)
        except:
            raise RuntimeError("Cannot read from file located at %s."%fpath)
        if keys is None:
            return info
        if isinstance(keys, basestring):
            keys = [keys]
        for key in info.keys():
            if key not in keys:
                info.pop(key)
        return info

    def extract_paramval(self, injparams, systkey, fid_label=None,
                         hypo_label=None, paramlabel=None):
        """Extract a value from a set of parameters and modify it based on the
        hypothesis/fiducial fit being considered. The label associated with this
        is then modified accordingly."""
        paramval = float(injparams[systkey].split(' ')[0])
        if (fid_label is None) or (hypo_label is None) or (paramlabel is None):
            if not ((fid_label is None) and (hypo_label is None) and
                    (paramlabel is None)):
                raise ValueError('Either all three labels must be None or they '
                                 ' must all be specified.')
            return paramval
        else:
            if systkey == 'deltam31':
                if 'no' in hypo_label:
                    if np.sign(paramval) != 1:
                        paramval = -1*float(injparams[systkey].split(' ')[0])
                        paramlabel += r' ($\times-1$)'
                elif 'io' in hypo_label:
                    if np.sign(paramval) != -1:
                        paramval = -1*float(injparams[systkey].split(' ')[0])
                        paramlabel += r' ($\times-1$)'

            if (np.abs(paramval) < 1e-2) and (paramval != 0.0):
                paramlabel += ' = %.3e'%paramval
            else:
                paramlabel += ' = %.3g'%paramval

            return paramval, paramlabel

    # TODO (?) - This works in the case of all MC, but I don't know about data.
    def extract_fid_data(self):
        """Take the data sets returned by the `extract_trials` and extract the
        data on the fiducial fits."""
        fid_values = {}
        for injkey in self.data_sets.keys():
            fid_values[injkey] = {}
            for datakey in self.data_sets[injkey]:
                if ('toy' in datakey) or ('data' in datakey):
                    fid_values[injkey][datakey] \
                        = self.data_sets[injkey].pop(datakey)
        self.fid_values = fid_values

    def extract_data(self):
        """Take the data sets returned by `extract_trials` and turn them in to a
        format used by all of the plotting functions."""
        values = {}
        for injkey in self.data_sets.keys():
            values[injkey] = {}
            alldata = self.data_sets[injkey]
            paramkeys = alldata['params'].keys()
            for datakey in alldata.keys():
                if not datakey == 'params':
                    values[injkey][datakey] = {}
                    values[injkey][datakey]['metric_val'] = {}
                    values[injkey][datakey]['metric_val']['vals'] = []
                    for paramkey in paramkeys:
                        values[injkey][datakey][paramkey] = {}
                        values[injkey][datakey][paramkey]['vals'] = []
                    trials = alldata[datakey]
                    for trial_num in trials.keys():
                        trial = trials[trial_num]
                        values[injkey][datakey]['metric_val']['vals'] \
                            .append(trial['metric_val'])
                        values[injkey][datakey]['metric_val']['type'] \
                            = trial['metric']
                        values[injkey][datakey]['metric_val']['units'] \
                            = 'dimensionless'
                        param_vals = trial['params']
                        for param_name in param_vals.keys():
                            val, units = self.parse_pint_string(
                                pint_string=param_vals[param_name]
                            )
                            values[injkey][datakey][param_name]['vals'] \
                                .append(float(val))
                            values[injkey][datakey][param_name]['units'] \
                                = units
        self.values = values

    def extract_gaussian(self, prior_string, units):
        """Parses the string for the Gaussian priors that comes from the
        config summary file in the logdir. This should account for dimensions
        though has only been proven with "deg" and "ev ** 2"."""
        if units == 'dimensionless':
            parse_string = ('gaussian prior: stddev=(.*)'
                            ' , maximum at (.*)')
            bits = re.match(
                parse_string,
                prior_string,
                re.M|re.I
            )
            stddev = float(bits.group(1))
            maximum = float(bits.group(2))
        else:
            try:
                # This one works for deg and other single string units
                parse_string = ('gaussian prior: stddev=(.*) (.*)'
                                ', maximum at (.*) (.*)')
                bits = re.match(
                    parse_string,
                    prior_string,
                    re.M|re.I
                )
                stddev = float(bits.group(1))
                maximum = float(bits.group(3))
            except:
                # This one works for ev ** 2 and other triple string units
                parse_string = ('gaussian prior: stddev=(.*) (.*) (.*) (.*)'
                                ', maximum at (.*) (.*) (.*) (.*)')
                bits = re.match(
                    parse_string,
                    prior_string,
                    re.M|re.I
                )
                stddev = float(bits.group(1))
                maximum = float(bits.group(5))

        return stddev, maximum

    def purge_failed_jobs(self, trial_nums, thresh=5.0):
        """Look at the values of the metric and find any deemed to be from a
        failed job. That is, the value of the metric falls very far outside of
        the rest of the values.

        Notes
        -----
        Interestingly, I only saw a need for this with my true NO jobs, where I
        attempted to run some jobs in fp32 mode. No jobs were needed to be
        removed for true IO, where everything was run in fp64 mode. So if
        there's a need for this function in your analysis it probably points
        at some more serious underlying problem.

        References:
        ----------
        Taken from stack overflow:

            http://stackoverflow.com/questions/22354094/pythonic-way-\
            of-detecting-outliers-in-one-dimensional-observation-data

        which references:

            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect
            and Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.

        """
        for injkey in self.values.keys():
            for fit_key in self.values[injkey].keys():
                points = np.array(self.values[injkey][
                    fit_key]['metric_val']['vals'])
                if len(points.shape) == 1:
                    points = points[:, None]
                median = np.median(points, axis=0)
                diff = np.sum((points - median)**2, axis=-1)
                diff = np.sqrt(diff)
                med_abs_deviation = np.median(diff)
                modified_z_score = 0.6745 * diff / med_abs_deviation
                good_trials = modified_z_score < thresh
                if not np.all(good_trials):
                    bad_trials = np.where(good_trials == False)[0]
                    logging.warn(
                        'Outlier(s) detected for %s in trial(s) %s. Will be '
                        'removed. If you think this should not happen, please '
                        'change the value of the threshold used for the '
                        'decision (currently set to %.2e).'%(
                            fit_key, trial_nums[bad_trials], thresh
                        )
                    )
                    for fitkey in self.values[injkey].keys():
                        for param in self.values[injkey][fitkey].keys():
                            new_vals = np.delete(
                                np.array(self.values[injkey][
                                    fitkey][param]['vals']),
                                bad_trials
                            )
                            self.values[injkey][
                                fitkey][param]['vals'] = new_vals

    def make_fit_information_plot(fit_info, xlabel, title):
        """Make histogram of fit_info given with axis label and title"""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        plt.grid(axis='y', zorder=0)
        plt.hist(
            fit_info,
            bins=10,
            histtype='bar',
            color='darkblue',
            alpha=0.9,
            zorder=3
        )
        plt.xlabel(xlabel, size='24')
        plt.ylabel('Number of Trials', size='24')
        plt.title(title, fontsize=16)
        plt.subplots_adjust(left=0.10, right=0.90, top=0.9, bottom=0.11)

    def plot_fit_information(self, minimiser_info, labels, detector,
                             selection, outdir, formats):
        """Make plots of the number of iterations and time taken with the
        minimiser. This is a good cross-check that the minimiser did not end
        abruptly since you would see significant pile-up if it did."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True

        outdir = os.path.join(outdir, 'MinimiserPlots')
        mkdir(outdir)
        MainTitle = self.make_main_title(end='Minimiser Information')
        for fhkey in minimiser_info.keys():
            if minimiser_info[fhkey] is not None:
                hypo = fhkey.split('_')[0]
                fid = fhkey.split('_')[-2]
                minimiser_times = []
                minimiser_iterations = []
                minimiser_funcevals = []
                minimiser_status = []
                for trial in minimiser_info[fhkey].keys():
                    bits = minimiser_info[fhkey][
                        trial]['minimizer_time'].split(' ')
                    minimiser_times.append(
                        float(bits[0])
                    )
                    minimiser_iterations.append(
                        int(minimiser_info[fhkey][trial][
                            'minimizer_metadata']['nit'])
                    )
                    minimiser_funcevals.append(
                        int(minimiser_info[fhkey][trial][
                            'minimizer_metadata']['nfev'])
                    )
                    minimiser_status.append(
                        int(minimiser_info[fhkey][trial][
                            'minimizer_metadata']['status'])
                    )
                    minimiser_units = bits[1]
                FitTitle = make_fit_title(
                    labels=labels,
                    fid=fid,
                    hypo=hypo,
                    trials=len(minimiser_times)
                )
                data_to_plot = [
                    minimiser_times,
                    minimiser_iterations,
                    minimiser_funcevals,
                    minimiser_status
                ]
                data_to_plot_ends = [
                    'minimiser_times',
                    'minimiser_iterations',
                    'minimiser_funcevals',
                    'minimiser_status'
                ]
                for plot_data, plot_end in zip(data_to_plot,
                                               data_to_plot_ends):
                    make_fit_information_plot(
                        fit_info=plot_data,
                        xlabel=self.tex_axis_label(plot_end),
                        title=MainTitle+r'\\'+FitTitle
                    )
                    save_plot(
                        fid=fid,
                        hypo=hypo,
                        outdir=outdir,
                        formats=formats,
                        end=plot_end
                    )
                    plt.close()

    def add_extra_points(self, ymax):
        """Add extra points specified by self.extra_points and label them 
        with self.extra_points_labels`"""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True

        linelist = []
        for point, label in zip(self.extra_points, self.extra_points_labels):
            if isinstance(point, basestring):
                if os.path.isfile(point):
                    point = np.genfromtxt(point)
                try:
                    point = eval(point)
                except:
                    raise ValueError('Provided point, %s, was not either a '
                                     'path to a file or a string which could '
                                     'be parsed by eval()' % point)
            if not isinstance(point, float):
                raise ValueError('Expecting a single point here to add to the'
                                 ' plot and got %s instead.' % point)
            line = plt.axvline(
                point,
                color=self.plot_colour(label),
                linestyle=self.plot_style(label),
                ymax=ymax,
                lw=2,
                label=self.tex_axis_label(label)
            )
            linelist.append(self.tex_axis_label(label))
        return linelist

    def calc_p_value(self, LLRdist, critical_value, num_trials, greater=True,
                     median_p_value=False, LLRbest=None):
        """Calculate the p-value for the given dataset based on the given
        critical value with an associated error.

        The calculation involves asking in how many trials the test statistic
        was "beyond" the critical value. The threshold of beyond will depend
        on whether the given distribution is the best fit or the alternate fit.
        The default is a "greater than" criterion, which can be switched by
        setting the "greater" argument to false.

        In the case of median_p_values the error calculation must also account
        for the uncertainty on the median, and so one must pass the
        distribution from which this was calculated so the error can be
        estimated with bootstrapping."""
        if greater:
            misid_trials = float(np.sum(LLRdist > critical_value))
        else:
            misid_trials = float(np.sum(LLRdist < critical_value))
        p_value = misid_trials/num_trials
        if median_p_value:
            # Quantify the uncertainty on the median by bootstrapping
            sampled_medians = []
            for i in range(0, 1000):
                sampled_medians.append(
                    np.median(
                        np.random.choice(
                            LLRbest,
                            size=len(LLRbest),
                            replace=True
                        )
                    )
                )
            sampled_medians = np.array(sampled_medians)
            median_error = np.std(sampled_medians)/np.sqrt(num_trials)
            # Add relative errors in quadrature
            wdenom = misid_trials+median_error*median_error
            wterm = wdenom/(misid_trials*misid_trials)
            Nterm = 1.0/num_trials
            unc_p_value = p_value * np.sqrt(wterm + Nterm)
            return p_value, unc_p_value, median_error
        else:
            unc_p_value = np.sqrt(misid_trials*(1-p_value))/num_trials
            return p_value, unc_p_value

    def plot_LLR_histograms(self, LLRarrays, LLRhistmax, binning, colors,
                            labels, best_name, alt_name, critical_value,
                            critical_label, critical_height, LLRhist,
                            critical_color='k', plot_scaling_factor=1.55,
                            greater=True, CLs=False):
        """Plot the LLR histograms. The `greater` argument is intended to be
        used the same as in the p value function above."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True

        for LLRarray, label, color in zip(LLRarrays, labels, colors):
            plt.hist(
                LLRarray,
                bins=binning,
                color=color,
                histtype='step',
                lw=2,
                label=label
            )
        plt.xlabel(r'Log-Likelihood Ratio', size='24', labelpad=22)
        plt.ylabel(r'Number of Trials (per %.2f)'%(binning[1]-binning[0]),
                   size='24')
        # Nicely scale the plot
        plt.ylim(0, plot_scaling_factor*LLRhistmax)
        # Add labels to show which side means what...
        xlim = plt.gca().get_xlim()
        plt.text(
            xlim[0]-0.05*(xlim[1]-xlim[0]),
            -0.09*plot_scaling_factor*LLRhistmax,
            r'\begin{flushleft} $\leftarrow$ Prefers %s\end{flushleft}' % self.tex_axis_label(alt_name),
            color='k',
            size='large'
        )
        plt.text(
            xlim[1]+0.05*(xlim[1]-xlim[0]),
            -0.09*plot_scaling_factor*LLRhistmax,
            r'\begin{flushright} Prefers %s $\rightarrow$ \end{flushright}' % self.tex_axis_label(best_name),
            color='k',
            size='large',
            horizontalalignment='right'
        )
        # Add the critical value with the desired height and colour.
        if critical_value is not None:
            plt.axvline(
                critical_value,
                color=critical_color,
                ymax=critical_height,
                lw=2,
                label=critical_label
            )
            if LLRhist is not None:
                if CLs:
                    for hist, color in zip(LLRhist, colors):
                        finehist = np.repeat(hist, 100)
                        finebinning = np.linspace(binning[0],
                                                  binning[-1],
                                                  (len(binning)-1)*100+1)
                        finebinwidth = finebinning[1]-finebinning[0]
                        finebincens = np.linspace(
                            finebinning[0]+finebinwidth/2.0,
                            finebinning[-1]-finebinwidth/2.0,
                            len(finebinning)-1
                        )
                        if color == 'r':
                            where = (finebincens < critical_value)
                        elif color == 'b':
                            where = (finebincens > critical_value)
                        plt.fill_between(
                            finebincens,
                            0,
                            finehist,
                            where=where,
                            color='none',
                            hatch='X',
                            edgecolor=color,
                            lw=0
                        )
                else:
                    # Create an object so that a hatch can be drawn over the
                    # region of interest to the p-value.
                    finehist = np.repeat(LLRhist, 100)
                    finebinning = np.linspace(binning[0], binning[-1],
                                              (len(binning)-1)*100+1)
                    finebinwidth = finebinning[1]-finebinning[0]
                    finebincens = np.linspace(finebinning[0]+finebinwidth/2.0,
                                              finebinning[-1]-finebinwidth/2.0,
                                              len(finebinning)-1)
                    # Draw the hatch. This is between the x-axis (0) and the
                    # finehist object made above. The "where" tells is to only
                    # draw above the critical value. To make it just the hatch,
                    # color is set to none and hatch is set to X. Also, so that
                    # it doesn't have a border we set linewidth to zero.
                    if greater:
                        where = (finebincens > critical_value)
                    else:
                        where = (finebincens < critical_value)
                    plt.fill_between(
                        finebincens,
                        0,
                        finehist,
                        where=where,
                        color='none',
                        hatch='X',
                        edgecolor='k',
                        lw=0
                    )
        plt.subplots_adjust(left=0.10, right=0.90, top=0.9, bottom=0.15)

    def make_llr_plots(self):
        """Make LLR plots.

        Takes the data and makes LLR distributions. These are then saved to the
        requested outdir within a folder labelled "LLRDistributions". The
        extra_points and extra_points_labels arguments can be used to specify
        extra points to be added to the plot for e.g. other fit LLR values.

        TODO:

        1) Currently the p-value is put on the LLR distributions as an
           annotation. This is probably fine, since the significances can just
           be calculated from this after the fact.

        """
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True

        outdir = os.path.join(self.outdir, 'LLRDistributions')
        mkdir(outdir)

        for injkey in self.values.keys():

            data = self.values[injkey]
            h0_fid_metric = self.fid_values[injkey][
                'h0_fit_to_toy_%s_asimov'%self.labels.dict['data_name']
            ][
                'metric_val'
            ]
            h1_fid_metric = self.fid_values[injkey][
                'h1_fit_to_toy_%s_asimov'%self.labels.dict['data_name']
            ][
                'metric_val'
            ]

            if h1_fid_metric > h0_fid_metric:
                bestfit = 'h0'
                altfit = 'h1'
                critical_value = h0_fid_metric-h1_fid_metric
            else:
                bestfit = 'h1'
                altfit = 'h0'
                critical_value = h1_fid_metric-h0_fid_metric
                
            h0_fit_to_h0_fid_metrics = np.array(
                data['h0_fit_to_h0_fid']['metric_val']['vals']
            )
            h1_fit_to_h0_fid_metrics = np.array(
                data['h1_fit_to_h0_fid']['metric_val']['vals']
            )
            h0_fit_to_h1_fid_metrics = np.array(
                data['h0_fit_to_h1_fid']['metric_val']['vals']
            )
            h1_fit_to_h1_fid_metrics = np.array(
                data['h1_fit_to_h1_fid']['metric_val']['vals']
            )

        num_trials = len(h0_fit_to_h0_fid_metrics)
        metric_type = data['h0_fit_to_h0_fid']['metric_val']['type']
        metric_type_pretty = self.tex_axis_label(metric_type)

        # In the case of likelihood, the maximum metric is the better fit.
        # With chi2 metrics the opposite is true, and so we must multiply
        # everything by -1 in order to apply the same treatment.
        if 'chi2' in metric_type:
            logging.info('Converting chi2 metric to likelihood equivalent.')
            h0_fit_to_h0_fid_metrics *= -1
            h1_fit_to_h0_fid_metrics *= -1
            h0_fit_to_h1_fid_metrics *= -1
            h1_fit_to_h1_fid_metrics *= -1
            critical_value *= -1

        if bestfit == 'h0':
            LLRbest = h0_fit_to_h0_fid_metrics - h1_fit_to_h0_fid_metrics
            LLRalt = h0_fit_to_h1_fid_metrics - h1_fit_to_h1_fid_metrics
        else:
            LLRbest = h1_fit_to_h1_fid_metrics - h0_fit_to_h1_fid_metrics
            LLRalt = h1_fit_to_h0_fid_metrics - h0_fit_to_h0_fid_metrics

        minLLR = min(min(LLRbest), min(LLRalt))
        maxLLR = max(max(LLRbest), max(LLRalt))
        rangeLLR = maxLLR - minLLR
        # Special case for low numbers of trials. Here, the plot can't really
        # be interpreted but the numbers printed on it can still be useful, so
        # we need to make something.
        if num_trials < 100:
            binning = np.linspace(minLLR - 0.1*rangeLLR,
                                  maxLLR + 0.1*rangeLLR,
                                  10)
        else:
            binning = np.linspace(minLLR - 0.1*rangeLLR,
                                  maxLLR + 0.1*rangeLLR,
                                  int(num_trials/40))
        binwidth = binning[1]-binning[0]
        bincens = np.linspace(binning[0]+binwidth/2.0,
                              binning[-1]-binwidth/2.0,
                              len(binning)-1)

        LLRbesthist, LLRbestbinedges = np.histogram(LLRbest, bins=binning)
        LLRalthist, LLRaltbinedges = np.histogram(LLRalt, bins=binning)

        LLRhistmax = max(max(LLRbesthist), max(LLRalthist))

        best_median = np.median(LLRbest)
        alt_median = np.median(LLRalt)

        inj_name = self.labels.dict['data_name']
        best_name = self.labels.dict['%s_name'%bestfit]
        alt_name = self.labels.dict['%s_name'%altfit]

        # Calculate p values
        ## First for the preferred hypothesis based on the fiducial fit
        crit_p_value, unc_crit_p_value = self.calc_p_value(
            LLRdist=LLRalt,
            critical_value=critical_value,
            num_trials=num_trials,
            greater=True
        )
        ## Then for the alternate hypothesis based on the fiducial fit
        alt_crit_p_value, alt_unc_crit_p_value = self.calc_p_value(
            LLRdist=LLRbest,
            critical_value=critical_value,
            num_trials=num_trials,
            greater=False
        )
        ## Combine these to give a CLs value based on arXiv:1407.5052
        cls_value = (1 - alt_crit_p_value) / (1 - crit_p_value)
        unc_cls_value = cls_value * np.sqrt(
            np.power(alt_unc_crit_p_value/alt_crit_p_value, 2.0) + \
            np.power(unc_crit_p_value/crit_p_value, 2.0)
        )
        ## Then for the preferred hypothesis based on the median. That is, the
        ## case of a median experiment from the distribution under the
        ## preferred hypothesis.
        med_p_value, unc_med_p_value, median_error = self.calc_p_value(
            LLRdist=LLRalt,
            critical_value=best_median,
            num_trials=num_trials,
            greater=True,
            median_p_value=True,
            LLRbest=LLRbest
        )

        if metric_type == 'llh':
            plot_title = (r"\begin{center}"\
                          +"%s %s Event Selection "%(self.detector,
                                                     self.selection)\
                          +r"\\"+" LLR Distributions for true %s (%i trials)"%(
                              self.tex_axis_label(inj_name), num_trials)\
                          +r"\end{center}")

        else:
            plot_title = (r"\begin{center}"\
                          +"%s %s Event Selection "%(self.detector,
                                                     self.selection)\
                          +r"\\"+" %s \"LLR\" Distributions for "
                          %(metric_type_pretty)\
                          +"true %s (%i trials)"%(self.tex_axis_label(inj_name),
                                                  num_trials)\
                          +r"\end{center}")

        # Factor with which to make everything visible
        plot_scaling_factor = 1.55

        # In case of median plot, draw both best and alt histograms
        ## Set up the labels for the histograms
        LLR_labels = [
            r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
                self.tex_axis_label(best_name)) + \
            r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                best_name, alt_name),
            r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
                self.tex_axis_label(alt_name)) + \
            r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                best_name, alt_name)
        ]
        self.plot_LLR_histograms(
            LLRarrays=[LLRbest, LLRalt],
            LLRhistmax=LLRhistmax,
            binning=binning,
            colors=['r', 'b'],
            labels=LLR_labels,
            best_name=best_name,
            alt_name=alt_name,
            critical_value=best_median,
            critical_label=r"%s Median = $%.4f\pm%.4f$"%(
                self.tex_axis_label(best_name),
                best_median,
                median_error),
            critical_height=float(max(LLRbesthist))/float(
                plot_scaling_factor*LLRhistmax),
            LLRhist=LLRalthist,
            greater=True
        )
        plt.legend(loc='upper left')
        plt.title(plot_title)
        # Write the p-value on the plot
        plt.figtext(
            0.15,
            0.66,
            r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.4f\pm%.4f$"%(
                best_name, med_p_value, unc_med_p_value),
            color='k',
            size='xx-large'
        )
        self.save_plot(
            outdir=outdir,
            end='%s_LLRDistribution_median_%i_Trials'%(metric_type, num_trials)
        )
        # Add the extra points if they exist
        if self.extra_points is not None:
            plt.legend(loc='upper left', fontsize=11)
            curleg = plt.gca().get_legend()
            linelist = self.add_extra_points(
                ymax=float(max(LLRbesthist))/float(
                    plot_scaling_factor*LLRhistmax)
            )
            handles, labels = plt.gca().get_legend_handles_labels()
            newhandles = []
            for l, h in zip(labels, handles):
                if l in linelist:
                    newhandles.append(h)
            newleg = plt.legend(
                handles=newhandles,
                loc='upper right',
                fontsize=11
            )
            plt.gca().add_artist(newleg)
            plt.gca().add_artist(curleg)
            self.save_plot(
                fid=None,
                hypo=None,
                outdir=outdir,
                end='%s_LLRDistribution_median_w_extra_points_%i_Trials'%(
                    metric_type, num_trials)
            )
        plt.close()

        # Make some debugging plots
        ## Set up the labels for the histograms
        LLR_labels = [
            r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
                self.tex_axis_label(best_name)) + \
            r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                best_name, alt_name),
            r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
                self.tex_axis_label(alt_name)) + \
            r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                best_name, alt_name)
        ]
        self.plot_LLR_histograms(
            LLRarrays=[LLRbest, LLRalt],
            LLRhistmax=LLRhistmax,
            binning=binning,
            colors=['r', 'b'],
            labels=LLR_labels,
            best_name=best_name,
            alt_name=alt_name,
            critical_value=best_median,
            critical_label=r"%s Median = $%.4f\pm%.4f$"%(
                self.tex_axis_label(best_name),
                best_median,
                median_error),
            critical_height=float(max(LLRbesthist))/float(
                plot_scaling_factor*LLRhistmax),
            LLRhist=None,
            greater=True
        )
        plt.legend(loc='upper left')
        plt.title(plot_title)
        self.save_plot(
            fid=None,
            hypo=None,
            outdir=outdir,
            end='%s_LLRDistribution_median_both_fit_dists_%i_Trials'%(metric_type, num_trials)
        )
        plt.close()
        ## Set up the labels for the histograms
        LLR_labels = [
            r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
                self.tex_axis_label(best_name)) + \
            r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                best_name, alt_name),
        ]
        self.plot_LLR_histograms(
            LLRarrays=[LLRbest],
            LLRhistmax=LLRhistmax,
            binning=binning,
            colors=['r'],
            labels=LLR_labels,
            best_name=best_name,
            alt_name=alt_name,
            critical_value=None,
            critical_label=None,
            critical_height=None,
            LLRhist=None,
            greater=True
        )
        plt.legend(loc='upper left')
        plt.title(plot_title)
        self.save_plot(
            fid=None,
            hypo=None,
            outdir=outdir,
            end='%s_LLRDistribution_best_fit_dist_%i_Trials'%(metric_type, num_trials)
        )
        plt.close()
        ## Set up the labels for the histograms
        LLR_labels = [
            r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
                self.tex_axis_label(best_name)) + \
            r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                best_name, alt_name),
        ]
        self.plot_LLR_histograms(
            LLRarrays=[LLRbest],
            LLRhistmax=LLRhistmax,
            binning=binning,
            colors=['r'],
            labels=LLR_labels,
            best_name=best_name,
            alt_name=alt_name,
            critical_value=best_median,
            critical_label=r"%s Median = $%.4f\pm%.4f$"%(
                self.tex_axis_label(best_name),
                best_median,
                median_error),
            critical_height=float(max(LLRbesthist))/float(
                plot_scaling_factor*LLRhistmax),
            LLRhist=None,
            greater=True
        )
        plt.legend(loc='upper left')
        plt.title(plot_title)
        self.save_plot(
            fid=None,
            hypo=None,
            outdir=outdir,
            end='%s_LLRDistribution_median_best_fit_dist_%i_Trials'%(metric_type, num_trials)
        )
        plt.close()
        ## Set up the labels for the histograms
        LLR_labels = [
            r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
                self.tex_axis_label(alt_name)) + \
            r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                best_name, alt_name)
        ]
        self.plot_LLR_histograms(
            LLRarrays=[LLRalt],
            LLRhistmax=LLRhistmax,
            binning=binning,
            colors=['b'],
            labels=LLR_labels,
            best_name=best_name,
            alt_name=alt_name,
            critical_value=None,
            critical_label=None,
            critical_height=None,
            LLRhist=None,
            greater=True
        )
        plt.legend(loc='upper left')
        plt.title(plot_title)
        self.save_plot(
            fid=None,
            hypo=None,
            outdir=outdir,
            end='%s_LLRDistribution_alt_fit_dist_%i_Trials'%(metric_type, num_trials)
        )
        plt.close()
        ## Set up the labels for the histograms
        LLR_labels = [
            r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
                self.tex_axis_label(alt_name)) + \
            r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                best_name, alt_name)
        ]
        self.plot_LLR_histograms(
            LLRarrays=[LLRalt],
            LLRhistmax=LLRhistmax,
            binning=binning,
            colors=['b'],
            labels=LLR_labels,
            best_name=best_name,
            alt_name=alt_name,
            critical_value=best_median,
            critical_label=r"%s Median = $%.4f\pm%.4f$"%(
                self.tex_axis_label(best_name),
                best_median,
                median_error),
            critical_height=float(max(LLRbesthist))/float(
                plot_scaling_factor*LLRhistmax),
            LLRhist=LLRalthist,
            greater=True
        )
        plt.legend(loc='upper left')
        plt.title(plot_title)
        # Write the p-value on the plot
        plt.figtext(
            0.15,
            0.66,
            r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.4f\pm%.4f$"%(
                best_name, med_p_value, unc_med_p_value),
            color='k',
            size='xx-large'
        )
        self.save_plot(
            fid=None,
            hypo=None,
            outdir=outdir,
            end='%s_LLRDistribution_median_alt_fit_dist_%i_Trials'%(metric_type, num_trials)
        )
        plt.close()

        # In case of critical plot, draw just alt histograms
        ## Set up the label for the histogram
        LLR_labels = [
            r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
                self.tex_axis_label(alt_name)) + \
            r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                best_name, alt_name)
        ]
        self.plot_LLR_histograms(
            LLRarrays=[LLRalt],
            LLRhistmax=LLRhistmax,
            binning=binning,
            colors=['b'],
            labels=LLR_labels,
            best_name=best_name,
            alt_name=alt_name,
            critical_value=critical_value,
            critical_label=r"Critical Value = %.4f"%(critical_value),
            critical_height=float(max(LLRbesthist))/float(
                plot_scaling_factor*LLRhistmax),
            LLRhist=LLRalthist,
            greater=True
        )
        plt.legend(loc='upper left')
        plt.title(plot_title)
        # Write the p-value on the plot
        plt.figtext(
            0.15,
            0.70,
            r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.4f\pm%.4f$"%(
                best_name, crit_p_value, unc_crit_p_value),
            color='k',
            size='xx-large'
        )
        self.save_plot(
            fid=None,
            hypo=None,
            outdir=outdir,
            end='%s_LLRDistribution_critical_%i_Trials'%(metric_type, num_trials)
        )
        plt.close()

        # Make a second critical plot for the alt hypothesis, so we draw the
        # preferred hypothesis
        ## Set up the label for the histogram
        LLR_labels = [
            r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
                self.tex_axis_label(best_name)) + \
            r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                best_name, alt_name)
        ]
        self.plot_LLR_histograms(
            LLRarrays=[LLRbest],
            LLRhistmax=LLRhistmax,
            binning=binning,
            colors=['r'],
            labels=LLR_labels,
            best_name=best_name,
            alt_name=alt_name,
            critical_value=critical_value,
            critical_label=r"Critical Value = %.4f"%(critical_value),
            critical_height=float(max(LLRbesthist))/float(
                plot_scaling_factor*LLRhistmax),
            LLRhist=LLRbesthist,
            greater=False
        )
        plt.legend(loc='upper left')
        plt.title(plot_title)
        # Write the p-value on the plot
        plt.figtext(
            0.15,
            0.70,
            r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.4f\pm%.4f$"%(
                alt_name, alt_crit_p_value, alt_unc_crit_p_value),
            color='k',
            size='xx-large'
        )
        self.save_plot(
            fid=None,
            hypo=None,
            outdir=outdir,
            end='%s_LLRDistribution_critical_alt_%i_Trials'%(
                metric_type, num_trials)
        )
        plt.close()

        # Lastly, show both exclusion regions and then the joined CLs value
        ## Set up the labels for the histograms
        LLR_labels = [
            r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
                self.tex_axis_label(best_name)) + \
            r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                best_name, alt_name),
            r"%s Best Fit - $\log\left[\mathcal{L}\left(\mathcal{H}_"%(
                self.tex_axis_label(alt_name)) + \
            r"{%s}\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                best_name, alt_name)
        ]
        self.plot_LLR_histograms(
            LLRarrays=[LLRbest, LLRalt],
            LLRhistmax=LLRhistmax,
            binning=binning,
            colors=['r', 'b'],
            labels=LLR_labels,
            best_name=best_name,
            alt_name=alt_name,
            critical_value=critical_value,
            critical_label=r"Critical Value = %.4f"%(critical_value),
            critical_height=float(max(LLRbesthist))/float(
                plot_scaling_factor*LLRhistmax),
            LLRhist=[LLRbesthist, LLRalthist],
            CLs=True,
        )
        plt.legend(loc='upper left')
        plt.title(plot_title)
        # Write the p-values on the plot
        plt.figtext(
            0.50,
            0.66,
            r"$\mathrm{CL}_{s}\left(\mathcal{H}_{%s}\right) = %.4f\pm%.4f$"%(
                best_name, cls_value, unc_cls_value),
            horizontalalignment='center',
            color='k',
            size='xx-large'
        )
        plt.figtext(
            0.12,
            0.55,
            r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.2f\pm%.2f$"%(
                alt_name, alt_crit_p_value, alt_unc_crit_p_value),
            bbox=dict(facecolor='none', edgecolor='red', boxstyle='round'),
            horizontalalignment='left',
            color='k',
            size='x-large'
        )
        plt.figtext(
            0.88,
            0.55,
            r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.2f\pm%.2f$"%(
                best_name, crit_p_value, unc_crit_p_value),
            horizontalalignment='right',
            bbox=dict(facecolor='none', edgecolor='blue', boxstyle='round'),
            color='k',
            size='x-large'
        )
        self.save_plot(
            fid=None,
            hypo=None,
            outdir=outdir,
            end='%s_LLRDistribution_CLs_%i_Trials'%(metric_type, num_trials)
        )
        plt.close()

    def check_pickle_files(self, logdir_content):
        """Checks for the expected pickle files in the output directory based
        on the analysis and test type. If they are there, it is made sure that
        they are the most up to date they can be. If not, they are regenerated.
        If they're not even there, then this returns false and the full
        extract_trials happens, at the end of which these pickle files will be
        generated for future use."""
        if self.analysis_type == 'hypo_testing':
            if self.test_type is not None:
                raise ValueError("Not implemented yet.")
            else:
                expected_files = ['data_sets.pckl', 'all_params.pckl',
                                  'minimiser_info.pckl']
        if np.all(np.array([s in logdir_content for s in expected_files])):
            # Processed output files are there so make sure that there
            # have been no more trials run since this last processing.
            ## To do this, get the number of output files
            for basename in nsort(os.listdir(self.logdir)):
                m = self.labels.subdir_re.match(basename)
                if m is None:
                    continue
                # Here is the output directory which contains the files
                subdir = os.path.join(self.logdir, basename)
                # Account for failed jobs. Get the set of file numbers that
                # exist for all h0 and h1 combinations
                self.get_set_file_nums(
                    filedir=os.path.join(self.logdir, basename)
                )
                num_trials = len(self.set_file_nums)
                # Take one of the pickle files to see how many data
                # entries it has.
                data_sets = from_file(os.path.join(self.logdir,
                                                   'data_sets.pckl'))
                # Take the first data key and then the h0 fit to h0 fid
                # which should always exist. The length of this is then
                # the number of trials in the pickle files.
                pckl_trials = len(data_sets[data_sets.keys()[0]][
                    'h0_fit_to_h0_fid'].keys())
                # The number of pickle trials should match the number of
                # trials derived from the output directory.
                if num_trials == pckl_trials:
                    logging.info(
                        'Found files I assume to be from a previous run of'
                        ' this processing script containing %i trials. If '
                        'this seems incorrect please delete the files: '
                        'data_sets.pckl, all_params.pckl and labels.pckl '
                        'from the logdir you have provided.'%pckl_trials
                    )
                    pickle_there = True
                else:
                    logging.info(
                        'Found files I assume to be from a previous run of'
                        ' this processing script containing %i trials. '
                        'However, based on the number of json files in the '
                        'output directory there should be %i trials in '
                        'these pickle files, so they will be regenerated.'%(
                            pckl_trials,num_trials)
                    )
                    pickle_there = False
        else:
            logging.info(
                'Did not find all of the files - %s - expected to indicate '
                'this data has already been extracted.'%expected_files
            )
            pickle_there = False

        return pickle_there

    def get_set_file_nums(self, filedir):
        """This function returns the set of file numbers that exist for all h0
        and h1 combination. This is needed to account for any failed or
        non-transferred jobs. i.e. for trial X you may not have all of the
        necessary fit files so it must be ignored."""
        file_nums = OrderedDict()
        for fnum, fname in enumerate(nsort(os.listdir(filedir))):
            fpath = os.path.join(filedir, fname)
            for x in ['0', '1']:
                for y in ['0', '1']:
                    k = 'h{x}_fit_to_h{y}_fid'.format(x=x, y=y)
                    r = self.labels.dict[k + '_re']
                    m = r.match(fname)
                    if m is None:
                        continue
                    if self.fluctuate_fid:
                        fid_label = int(m.groupdict()['fid_ind'])
                    else:
                        fid_label = labels.fid
                    if k not in file_nums:
                        file_nums[k] = []
                    file_nums[k].append(fid_label)
                    break

        set_file_nums = []
        for hypokey in file_nums.keys():
            if len(set_file_nums) == 0:
                set_file_nums = set(file_nums[hypokey])
            else:
                set_file_nums = set_file_nums.intersection(
                    file_nums[hypokey]
                )
        self.set_file_nums = set_file_nums

    def get_starting_params(self, cfg):
        """Extracts the h0, h1 and data (if possible) params from the config 
        summary file."""
        all_params = {}
        all_params['h0_params'] = {}
        all_params['h1_params'] = {}
        parse_string = ('(.*)=(.*); prior=(.*),'
                        ' range=(.*), is_fixed=(.*),'
                        ' is_discrete=(.*); help="(.*)"')
        if not self.data_is_data:
            all_params['data_params'] = {}
            for param_string in cfg['data_params']:
                bits = re.match(parse_string, param_string, re.M|re.I)
                if bits.group(5) == 'False':
                    all_params['data_params'][bits.group(1)] = {}
                    all_params['data_params'][bits.group(1)]['value'] \
                        = bits.group(2)
                    all_params['data_params'][bits.group(1)]['prior'] \
                        = bits.group(3)
                    all_params['data_params'][bits.group(1)]['range'] \
                        = bits.group(4)
        else:
            all_params['data_params'] = None
        for param_string in cfg['h0_params']:
            bits = re.match(parse_string, param_string, re.M|re.I)
            if bits.group(5) == 'False':
                all_params['h0_params'][bits.group(1)] = {}
                all_params['h0_params'][bits.group(1)]['value'] \
                    = bits.group(2)
                all_params['h0_params'][bits.group(1)]['prior'] \
                    = bits.group(3)
                all_params['h0_params'][bits.group(1)]['range'] \
                    = bits.group(4)
        for param_string in cfg['h1_params']:
            bits = re.match(parse_string, param_string, re.M|re.I)
            if bits.group(5) == 'False':
                all_params['h1_params'][bits.group(1)] = {}
                all_params['h1_params'][bits.group(1)]['value'] \
                    = bits.group(2)
                all_params['h1_params'][bits.group(1)]['prior'] \
                    = bits.group(3)
                all_params['h1_params'][bits.group(1)]['range'] \
                    = bits.group(4)
        self.all_params = all_params

    def get_data(self, cfg):
        """Get all of the data from the logdir"""
        data_sets = OrderedDict()
        minimiser_info = OrderedDict()
        for basename in nsort(os.listdir(self.logdir)):
            m = self.labels.subdir_re.match(basename)
            if m is None:
                continue
        
            if self.fluctuate_data:
                data_ind = int(m.groupdict()['data_ind'])
                dset_label = data_ind
            else:
                dset_label = self.labels.data_prefix
                if not self.labels.data_name in [None, '']:
                    dset_label += '_' + self.labels.data_name
                if not self.labels.data_suffix in [None, '']:
                    dset_label += '_' + self.labels.data_suffix

            lvl2_fits = OrderedDict()
            lvl2_fits['h0_fit_to_data'] = None
            lvl2_fits['h1_fit_to_data'] = None
            minim_info = OrderedDict()
            minim_info['h0_fit_to_data'] = None
            minim_info['h1_fit_to_data'] = None
        
            # Account for failed jobs. Get the set of file numbers that
            # exist for all h0 an h1 combinations
            subdir = os.path.join(self.logdir, basename)
            self.get_set_file_nums(
                filedir=subdir
            )

            fnum = None
            for fnum, fname in enumerate(nsort(os.listdir(subdir))):
                fpath = os.path.join(subdir, fname)
                for x in ['0', '1']:
                    k = 'h{x}_fit_to_data'.format(x=x)
                    if fname == self.labels.dict[k]:
                        lvl2_fits[k] = self.extract_fit(fpath, 'metric_val')
                        break
                    # Also extract fiducial fits if needed
                    if 'toy' in dset_label:
                        ftest = ('hypo_%s_fit_to_%s.json'
                                 %(self.labels.dict['h{x}_name'.format(x=x)],
                                   dset_label))
                    if fname == ftest:
                        k = 'h{x}_fit_to_{y}'.format(x=x, y=dset_label)
                        lvl2_fits[k] = self.extract_fit(
                            fpath,
                            ['metric_val', 'params']
                        )
                        break
                    k = 'h{x}_fit_to_{y}'.format(x=x, y=dset_label)
                    for y in ['0', '1']:
                        k = 'h{x}_fit_to_h{y}_fid'.format(x=x, y=y)
                        r = self.labels.dict[k + '_re']
                        m = r.match(fname)
                        if m is None:
                            continue
                        if self.fluctuate_fid:
                            fid_label = int(m.groupdict()['fid_ind'])
                        else:
                            fid_label = self.labels.fid
                        if k not in lvl2_fits:
                            lvl2_fits[k] = OrderedDict()
                            minim_info[k] = OrderedDict()
                        if fid_label in self.set_file_nums:
                            lvl2_fits[k][fid_label] = self.extract_fit(
                                fpath,
                                ['metric', 'metric_val', 'params']
                            )
                            minim_info[k][fid_label] = self.extract_fit(
                                fpath,
                                ['minimizer_metadata', 'minimizer_time']
                            )
                        break
                        
            if fnum is None:
                raise ValueError('No files?')
        
            data_sets[dset_label] = lvl2_fits
            minimiser_info[dset_label] = minim_info
            data_sets[dset_label]['params'] = self.extract_fit(
                fpath,
                ['params']
            )['params']

        self.data_sets = data_sets
        self.minimiser_info = minimiser_info

    def pickle_data(self):
        """Will pickle the data for easy access later."""
        to_file(
            self.data_sets,
            os.path.join(self.logdir,'data_sets.pckl')
        )
        to_file(
            self.all_params,
            os.path.join(self.logdir,'all_params.pckl')
        )
        to_file(
            self.labels,
            os.path.join(self.logdir,'labels.pckl')
        )
        to_file(
            self.minimiser_info,
            os.path.join(self.logdir,'minimiser_info.pckl')
        )

    def load_from_pickle(self):
        """Load from the pickle files created by the function above in a 
        previous run of this script."""
        self.data_sets = from_file(
            os.path.join(self.logdir, 'data_sets.pckl')
        )
        self.all_params = from_file(
            os.path.join(self.logdir, 'all_params.pckl')
        )
        self.minimiser_info = from_file(
            os.path.join(self.logdir, 'minimiser_info.pckl')
        )

    def make_main_title(self, end, end_center=False):
        """Make main title accounting for detector and selection."""
        main_title = r"\begin{center}"
        if self.detector is not None:
            main_title += "%s "%self.detector
        if self.selection is not None:
            main_title += "%s Event Selection "%self.selection
        main_title += end
        if end_center:
            main_title += r"\end{center}"
        return main_title

    def make_fit_title(self, fid, hypo, trials):
        """Make fit title to go with the main title"""
        FitTitle = ""
        if 'data_name' in self.labels.keys():
            FitTitle += "True %s, "%self.labels['data_name']
        if fid is not None:
            FitTitle += "Fiducial Fit %s, "%self.labels['%s_name'%fid]
        if hypo is not None:
            FitTitle += "Hypothesis %s "%self.labels['%s_name'%hypo]
        if trials is not None:
            FitTitle += "(%i Trials)"%trials
        FitTitle += r"\end{center}"
        return FitTitle

    def get_num_rows(self, data, omit_metric=False):
        """Calculates the number of rows for multiplots based on the number of 
        systematics."""
        if omit_metric:
            num_rows = int((len(data.keys())-1)/4)
        else:
            num_rows = int(len(data.keys())/4)
        if len(data.keys())%4 != 0:
            num_rows += 1
        return num_rows

    def parse_binning_string(self, binning_string):
        """Returns a dictionary that can be used to instantiate a binning
        object from the output of having run str on the original binning
        object."""
        if 'MultiDimBinning' in binning_string:
            raise ValueError(
                'This function is designed to work with OneDimBinning'
                ' objects. You should separate the MultiDimBinning '
                'string in to the separate OneDimBinning strings '
                'before calling this function and then reconnect them'
                ' in to the MultiDimBinning object after.'
            )
        if 'OneDimBinning' not in binning_string:
            raise ValueError(
                'String expected to have OneDimBinning in it.'
                ' Got %s'%binning_string
            )
        binning_dict = {}
        if '1 bin ' in binning_string:
            raise ValueError('Singular bin case not dealt with yet')
        elif 'irregularly' in binning_string:
            parse_string = ('OneDimBinning\((.*), (.*) irregularly-sized ' + \
                            'bins with edges at \[(.*)\] (.*)\)')
            a = re.match(parse_string, binning_string, re.M|re.I)
            # Match should come out None is the bins don't have units
            if a is None:
                parse_string = ('OneDimBinning\((.*), (.*) ' + \
                                'irregularly-sized bins with ' + \
                                'edges at \[(.*)\]\)')
                a = re.match(parse_string, binning_string, re.M|re.I)
            else:
                binning_dict['units'] = a.group(4)
            binning_dict['name'] = a.group(1).strip('\'')
            binning_dict['num_bins'] = int(a.group(2))
            binning_dict['bin_edges'] = [float(i) for i in \
                                         a.group(3).split(', ')]
        elif 'logarithmically' in binning_string:
            parse_string = ('OneDimBinning\((.*), (.*) ' + \
                            'logarithmically-uniform ' + \
                            'bins spanning \[(.*), (.*)\] (.*)\)')
            a = re.match(parse_string, binning_string, re.M|re.I)
            # Match should come out None is the bins don't have units
            if a is None:
                parse_string = ('OneDimBinning\((.*), (.*) logarithmically' + \
                                '-uniform bins spanning \[(.*), (.*)\]\)')
                a = re.match(parse_string, binning_string, re.M|re.I)
            else:
                binning_dict['units'] = a.group(5)
            binning_dict['name'] = a.group(1).strip('\'')
            binning_dict['num_bins'] = int(a.group(2))
            binning_dict['domain'] = [float(a.group(3)), float(a.group(4))]
            binning_dict['is_log'] = True
        elif 'equally-sized' in binning_string:
            parse_string = ('OneDimBinning\((.*), (.*) equally-sized ' + \
                            'bins spanning \[(.*) (.*)\] (.*)\)')
            a = re.match(parse_string, binning_string, re.M|re.I)
            # Match should come out None is the bins don't have units
            if a is None:
                parse_string = ('OneDimBinning\((.*), (.*) equally-sized ' + \
                                'bins spanning \[(.*), (.*)\]\)')
                a = re.match(parse_string, binning_string, re.M|re.I)
            else:
                binning_dict['units'] = a.group(5)
            binning_dict['name'] = a.group(1).strip('\'')
            binning_dict['num_bins'] = int(a.group(2))
            binning_dict['domain'] = [float(a.group(3)), float(a.group(4))]
            binning_dict['is_lin'] = True

        add_tex_to_binning(binning_dict)
        return binning_dict

    def add_tex_to_binning(self, binning_dict):
        """Will add a tex to binning dictionaries parsed with the above
        function."""
        if 'reco' in binning_dict['name']:
            sub_string = 'reco'
        elif 'true' in binning_dict['name']:
            sub_string = 'true'
        else:
            sub_string = None
        if 'energy' in binning_dict['name']:
            binning_dict['tex'] = r'$E_{%s}$'%sub_string
        elif 'coszen' in binning_dict['name']:
            binning_dict['tex'] = r'$\cos\theta_{Z,%s}$'%sub_string

    def parse_pint_string(self, pint_string):
        """Will return the value and units from a string with attached
        pint-style units. i.e. the string "0.97 dimensionless" would return a
        value of 0.97 and units of dimensionless. Both will return as
        strings."""
        val = pint_string.split(' ')[0]
        units = pint_string.split(val+' ')[-1]
        return val, units

    def get_num_rows(self, data, omit_metric=False):
        """Calculates the number of rows for multiplots based on the number of 
        systematics."""
        if omit_metric:
            num_rows = int((len(data.keys())-1)/4)
        else:
            num_rows = int(len(data.keys())/4)
        if len(data.keys())%4 != 0:
            num_rows += 1
        return num_rows

    #### Profile Scan Specific Postprocessing Functions ####

    def get_scan_steps(self):
        """Gets the bin centres, edges, names and units used in the 
        profile scan"""
        all_steps = self.scan_file_dict['steps']
        all_bin_cens = []
        all_bin_units = []
        for step_variable in all_steps.keys():
            bin_cens = []
            if isinstance(all_steps[step_variable][0][1],list):
                all_bin_units.append(all_steps[step_variable][0][1][0][0])
            else:
                all_bin_units.append('dimensionless')
            for val in all_steps[step_variable]:
                if val[0] not in bin_cens:
                    bin_cens.append(val[0])
            all_bin_cens.append(bin_cens)
        all_bin_edges = []
        for bin_cens in all_bin_cens:
            bin_width = bin_cens[1]-bin_cens[0]
            bin_edges = np.linspace(bin_cens[0]-bin_width/2.0,
                                    bin_cens[-1]+bin_width/2.0,
                                    len(bin_cens)+1)
            all_bin_edges.append(bin_edges)
                
        self.all_bin_cens = np.array(all_bin_cens)
        self.all_bin_edges = np.array(all_bin_edges)
        self.all_bin_names = all_steps.keys()
        self.all_bin_units = all_bin_units

    def get_scan_data(self):
        """Gets the data i.e. best fit metric and params over the scan. If a 
        best fit was supplied it will also be extracted. Also stores the 
        metric name to self."""
        self.metric_name = self.scan_file_dict['results'][0]['metric']
        data = {}
        data['metric_vals'] = []
        for result in self.scan_file_dict['results']:
            data['metric_vals'].append(result['metric_val'])
            for param_key in result['params'].keys():
                if not result['params'][param_key]['is_fixed']:
                    if param_key not in data.keys():
                        data[param_key] = {}
                        data[param_key]['vals'] = []
                        data[param_key]['units'] = \
                            result['params'][param_key]['prior']['units']
                    data[param_key]['vals'].append(
                        result['params'][param_key]['value'][0]
                    )
                    
        if self.best_fit_dict is not None:
            best_fit_data = {}
            best_fit_data['metric_val'] = self.best_fit_dict['metric_val']
            for param_key in self.best_fit_dict['params'].keys():
                if not self.best_fit_dict['params'][param_key]['is_fixed']:
                    best_fit_data[param_key] = {}
                    best_fit_data[param_key]['val'] = \
                        self.best_fit_dict['params'][param_key]['value'][0]
                    best_fit_data[param_key]['units'] = \
                        self.best_fit_dict['params'][param_key]['value'][1]
            # Make a list of shifted metrics based on this best fit point
            data['shifted_metric_vals'] = []
            for val in data['metric_vals']:
                data['shifted_metric_vals'].append(
                    val-best_fit_data['metric_val']
                )
        else:
            best_fit_data = None
            
        self.data = data
        self.best_fit_data = best_fit_data

    def get_best_fit(self, xlabel=None, ylabel=None):
        """Extracts the best fit values from the best fit dictionary 
        if it is not None"""
        if self.best_fit_data is not None:
            if xlabel is not None:
                best_fit_x = self.best_fit_data[xlabel]['val']
                if ylabel is not None:
                    best_fit_y = self.best_fit_data[ylabel]['val']
                    self.best_fit_point = [best_fit_x, best_fit_y]
                else:
                    self.best_fit_point = best_fit_x
            elif ylabel is None:
                raise ValueError(
                    "You have not specified a x parameter name but have"
                    " specified a y parameter name - %s."%ylabel
                )
        else:
            self.best_fit_point = None

    def sort_z_labels(self, data_key):
        """Sorts the z labels based on what data_key is passed"""
        if data_key == 'metric_vals':
            zlabel = self.metric_name
            zunits = 'dimensionless'
            zvals = np.array(self.data[data_key])
        elif data_key == 'shifted_metric_vals':
            zlabel = 'contour'
            zunits = 'dimensionless'
            zvals = np.array(self.data[data_key])
        else:
            zlabel = data_key
            zunits = self.data[data_key]['units']
            zvals = np.array(self.data[data_key]['vals'])

        zvals = np.array(np.split(zvals, len(self.all_bin_cens[0])))
        
        return zlabel, zunits, zvals

    def plot_2D_scans(self, xbins=None, xlabel=None, xunits=None, xcens=None,
                      ybins=None, ylabel=None, yunits=None, ycens=None):
        """Makes the 2D scan plots. The x and y bins as well as their
        labels/units can be specified here, or else they will be generated
        from what is stored in self"""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        if xbins is None:
            xbins = self.all_bin_edges[0]
        if xlabel is None:
            xlabel = self.all_bin_names[0]
        if xunits is None:
            xunits = self.all_bin_units[0]
        if xcens is None:
            xcens = self.all_bin_cens[0]
        if ybins is None:
            ybins = self.all_bin_edges[1]
        if ylabel is None:
            ylabel = self.all_bin_names[1]
        if yunits is None:
            yunits = self.all_bin_units[1]
        if ycens is None:
            ycens = self.all_bin_cens[1]

        self.get_best_fit(xlabel=xlabel, ylabel=ylabel)
        title_end = "%s / %s Parameter Scan"%(
            self.tex_axis_label(xlabel),
            self.tex_axis_label(ylabel)
        )
        MainTitle = self.make_main_title(end_center=True, end=title_end)

        for data_key in self.data.keys():
            zlabel, zunits, zvals = self.sort_z_labels(data_key=data_key)
            if zlabel == 'contour':
                # Contour plots need bin centers...
                self.make_2D_hist_plot(
                    zvals=zvals,
                    xbins=xcens,
                    ybins=ycens,
                    xlabel=xlabel,
                    xunits=xunits,
                    ylabel=ylabel,
                    yunits=yunits,
                    zlabel=zlabel,
                    zunits=zunits,
                    levels=[0,4.605]
                )
            else:
                self.make_2D_hist_plot(
                    zvals=zvals,
                    xbins=xbins,
                    ybins=ybins,
                    xlabel=xlabel,
                    xunits=xunits,
                    ylabel=ylabel,
                    yunits=yunits,
                    zlabel=zlabel,
                    zunits=zunits
                )

            if self.best_fit_data is not None:
                plt.plot(
                    self.best_fit_point[0],
                    self.best_fit_point[1],
                    marker='x',
                    linestyle='None',
                    color='k',
                    markersize=10
                )

            plt.title(MainTitle, fontsize=16)
            plt.tight_layout()
            save_end = "%s_%s_2D_%s_scan_%s"%(
                xlabel, ylabel, self.metric_name, zlabel)
            if zlabel != "contour":
                save_end += "_values"
            self.save_plot(outdir=self.outdir, end=save_end)
            plt.close()

    #### Generic Functions Relating to Plotting ####

    def make_2D_hist_plot(self, zvals, xbins, ybins, xlabel, xunits,
                          ylabel, yunits, zlabel, zunits, levels=None,
                          cmap='Blues'):
        """Generic 2D histogram-style plotting function. Set zlabel to contour
        to make a contour plot instead of a histogram."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        if zlabel == 'contour':
            origin = 'lower'
            if levels is None:
                raise ValueError(
                    "Contour plot requested without setting what contours"
                    " to plot."
                )
            X, Y = np.meshgrid(xbins, ybins)
            plt.contour(
                X,
                Y,
                zvals.T,
                levels,
                colors=('k',),
                linewidths=(3,),
                origin=origin
            )
        else:
            plt.pcolormesh(xbins, ybins, zvals.T, cmap=cmap)
            nice_clabel = self.make_label(zlabel, zunits)
            plt.colorbar().set_label(label=nice_clabel,fontsize=24)
            
        plt.xlim(xbins[0],xbins[-1])
        nice_xlabel = self.make_label(xlabel, xunits)
        plt.xlabel(nice_xlabel,fontsize=24)
        plt.ylim(ybins[0],ybins[-1])
        nice_ylabel = self.make_label(ylabel, yunits)
        plt.ylabel(nice_ylabel,fontsize=24)

    def save_plot(self, outdir, end, fid=None, hypo=None):
        """Save plot as each type of file format specified in self.formats"""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        save_name = ""
        if hasattr(self, 'labels'):
            if 'data_name' in self.labels.dict.keys():
                save_name += "true_%s_"%self.labels.dict['data_name']
        if self.detector is not None:
            save_name += "%s_"%self.detector
        if self.selection is not None:
            save_name += "%s_"%self.selection
        if fid is not None:
            save_name += "fid_%s_"%self.labels.dict['%s_name'%fid]
        if hypo is not None:
            save_name += "hypo_%s_"%self.labels.dict['%s_name'%hypo]
        save_name += end
        for fileformat in self.formats:
            full_save_name = save_name + '.%s'%fileformat
            plt.savefig(os.path.join(outdir, full_save_name))

    #### General Style Functions ####

    def make_label(self, label, units):
        """Appends units to a label for plotting."""
        nice_label = self.tex_axis_label(label)
        if not (units == 'dimensionless') and not \
           (units == None) and not (units == []):
            nice_label += ' (%s)'%self.tex_axis_label(units)
        return nice_label

    def tex_axis_label(self, label):
        """Takes the labels used in the objects and turns them in to something
        nice for plotting. This can never truly be exhaustive, but it
        definitely does the trick. If something looks ugly add it to this
        function!"""
        if isinstance(label, list):
            label = label[0]
        if not isinstance(label, basestring):
            raise ValueError("Label must be a string. Got %s of "
                             "type %s"%(label, type(label)))
        label = label.lower()
        pretty_labels = {}
        pretty_labels["atm_muon_scale"] = r"Muon Background Scale"
        pretty_labels["nue_numu_ratio"] = r"$\nu_e/\nu_{\mu}$ Ratio"
        pretty_labels["nu_nubar_ratio"] = r"$\nu/\bar{\nu}$ Ratio"
        pretty_labels["barr_uphor_ratio"] = r"Barr Up/Horizontal Ratio"
        pretty_labels["barr_nu_nubar_ratio"] = r"Barr $\nu/\bar{\nu}$ Ratio"
        pretty_labels["barr_uphor"] = r"Barr Up/Horizontal Ratio"
        pretty_labels["barr_nu_nubar"] = r"Barr $\nu/\bar{\nu}$ Ratio"
        pretty_labels["delta_index"] = r"Atmospheric Index Change"
        pretty_labels["theta13"] = r"$\theta_{13}$"
        pretty_labels["theta23"] = r"$\theta_{23}$"
        pretty_labels["sin2theta23"] = r"$\sin^2\theta_{23}$"
        pretty_labels["deltam31"] = r"$\Delta m^2_{31}$"
        pretty_labels["deltam32"] = r"$\Delta m^2_{32}$"
        pretty_labels["deltam3l"] = r"$\Delta m^2_{3l}$"
        pretty_labels["aeff_scale"] = r"$A_{\mathrm{eff}}$ Scale"
        pretty_labels["energy_scale"] = r"Energy Scale"
        pretty_labels["genie_ma_qe"] = r"GENIE $M_{A}^{QE}$"
        pretty_labels["genie_ma_res"] = r"GENIE $M_{A}^{Res}$"
        pretty_labels["dom_eff"] = r"DOM Efficiency"
        pretty_labels["hole_ice"] = r"Hole Ice"
        pretty_labels["hole_ice_fwd"] = r"Hole Ice Forward"
        pretty_labels["degree"] = r"$^\circ$"
        pretty_labels["radians"] = r"rads"
        pretty_labels["electron_volt ** 2"] = r"$\mathrm{eV}^2$"
        pretty_labels["electron_volt"] = r"$\mathrm{eV}^2$"
        pretty_labels["llh"] = r"Likelihood"
        pretty_labels["conv_llh"] = r"Convoluted Likelihood"
        pretty_labels["chi2"] = r"$\chi^2$"
        pretty_labels["mod_chi2"] = r"Modified $\chi^2$"
        pretty_labels["no"] = r"Normal Ordering"
        pretty_labels["io"] = r"Inverted Ordering"
        pretty_labels["nomsw"] = r"Normal Ordering, Matter Oscillations"
        pretty_labels["iomsw"] = r"Inverted Ordering, Matter Oscillations"
        pretty_labels["novacuum"] = r"Normal Ordering, Vacuum Oscillations"
        pretty_labels["iovacuum"] = r"Inverted Ordering, Vacuum Oscillations"
        pretty_labels["msw"] = r"Matter Oscillations"
        pretty_labels["vacuum"] = r"Vacuum Oscillations"
        pretty_labels["no,llr"] = r"LLR Method"
        pretty_labels["no,llr,nufitpriors"] = r"LLR Method, Nu-Fit Priors"
        pretty_labels["io,llr"] = r"LLR Method"
        pretty_labels["io,llr,nufitpriors"] = r"LLR Method, Nu-Fit Priors"
        pretty_labels["nue"] = r"$\nu_e$"
        pretty_labels["nuebar"] = r"$\bar{\nu}_e$"
        pretty_labels["numu"] = r"$\nu_{\mu}$"
        pretty_labels["numubar"] = r"$\bar{\nu}_{\mu}$"
        pretty_labels["second"] = r"s"
        pretty_labels["seconds"] = r"s"
        pretty_labels["atm_delta_index"] = r"Atmospheric Index Change"
        pretty_labels["pve"] = r"Positive"
        pretty_labels["nve"] = r"Negative"
        pretty_labels["fitwrong"] = r"Sensitivity Stability"
        pretty_labels["fixwrong"] = r"Fitting Relevance"
        pretty_labels["nminusone"] = r"Hidden Potential"
        pretty_labels["minimiser_times"] = r"Minimiser Time (seconds)"
        pretty_labels["minimiser_iterations"] = r"Minimiser Iterations"
        pretty_labels["minimiser_funcevals"] = r"Minimiser Function Evaluations"
        pretty_labels["minimiser_status"] = r"Minimiser Status"
        if label not in pretty_labels.keys():
            logging.warn("I don't know what to do with %s. "
                         "Returning as is."%label)
            return label
        return pretty_labels[label]

    def plot_colour(self, label):
        """Will return a standard colour scheme which can be used for e.g.
        specific truths or specific ice models etc."""
        label = label.lower()
        pretty_colours = {}
        # SPIce HD
        pretty_colours['544'] = 'maroon'
        pretty_colours['545'] = 'goldenrod'
        pretty_colours['548'] = 'blueviolet'
        pretty_colours['549'] = 'forestgreen'
        # H2
        ## DOM Efficiency Sets
        pretty_colours['551'] = 'cornflowerblue'
        pretty_colours['552'] = 'cornflowerblue'
        pretty_colours['553'] = 'cornflowerblue'
        pretty_colours['554'] = 'mediumseagreen'
        pretty_colours['555'] = 'mediumseagreen'
        pretty_colours['556'] = 'mediumseagreen'
        ## Hole Ice Sets
        pretty_colours['560'] = 'olive'
        pretty_colours['561'] = 'olive'
        pretty_colours['564'] = 'darkorange'
        pretty_colours['565'] = 'darkorange'
        pretty_colours['572'] = 'teal'
        pretty_colours['573'] = 'teal'
        ## Dima Hole Ice Set without RDE
        pretty_colours['570'] = 'mediumvioletred'
        ## Baseline
        pretty_colours['585'] = 'slategrey'
        # Systematics
        pretty_colours['aeff_scale'] = 'maroon'
        pretty_colours['atm_muon_scale'] = 'goldenrod'
        pretty_colours['deltam31'] = 'blueviolet'
        pretty_colours['theta23'] = 'forestgreen'
        pretty_colours['hole_ice_fwd'] = 'mediumvioletred'
        pretty_colours['dom_eff'] = 'cornflowerblue'
        pretty_colours['genie_ma_qe'] = 'mediumseagreen'
        pretty_colours['genie_ma_res'] = 'olive'
        pretty_colours['hole_ice'] = 'darkorange'
        pretty_colours['nue_numu_ratio'] = 'teal'
        pretty_colours['theta13'] = 'fuchsia'
        pretty_colours['barr_nu_nubar'] = 'thistle'
        pretty_colours['barr_uphor'] = 'orchid'
        pretty_colours['delta_index'] = 'navy'
        colourlabel = None
        for colourkey in pretty_colours.keys():
            if (colourkey in label) or (colourkey == label):
                colourlabel = pretty_colours[colourkey]
        if colourlabel is None:
            logging.debug("I do not have a colour scheme for your label %s. "
                          "Returning black."%label)
            colourlabel = 'k'
        return colourlabel
    
    def plot_style(self, label):
        """Will return a standard line style for plots similar to above."""
        label = label.lower()
        pretty_styles = {}
        # H2
        ## DOM Efficiency Sets
        pretty_styles['552'] = '--'
        pretty_styles['553'] = '-.'
        pretty_styles['555'] = '--'
        pretty_styles['556'] = '-.'
        ## Hole Ice Sets
        pretty_styles['561'] = '--'
        pretty_styles['565'] = '--'
        pretty_styles['572'] = '--'
        pretty_styles['573'] = '-.'
        colourstyle = None
        for colourkey in pretty_styles.keys():
            if colourkey in label:
                colourstyle = pretty_styles[colourkey]
        if colourstyle is None:
            logging.debug("I do not have a style for your label %s. "
                          "Returning standard."%label)
            colourstyle = '-'
        return colourstyle


def main_hypo_testing():
    Hypotestingpostprocessingargparser()


def main_profile_scan():
    description="""A script for processing the output files of 
    profile_scan.py"""

    # TODO:
    #
    # 1) Processing of 1D scans

    init_args_d = parse_args(description=description,
                             profile_scan=True)

    postprocessor = Postprocessor(
        analysis_type='profile_scan',
        detector=init_args_d['detector'],
        selection=init_args_d['selection'],
        outdir=init_args_d['outdir'],
        formats=init_args_d['formats'],
        scan_file=init_args_d['infile'],
        best_fit_file=init_args_d['best_fit_infile']
    )

    # 1D profile scans
    if len(postprocessor.all_bin_cens) == 1:
        raise NotImplementedError(
            "Postprocessing of profile scans in 1D not implemented in this"
            " script yet.")

    # 2D profile scans
    elif len(postprocessor.all_bin_cens) == 2:
        #postprocessor.plot_2D_scans()
        if (postprocessor.all_bin_names[0] == 'theta23') and \
           (postprocessor.all_bin_names[1] == 'deltam31'):
            deltam21 = postprocessor.scan_file_dict['results'][0][
                'params']['deltam21']['value'][0]
            postprocessor.all_bin_edges[0] = np.power(np.sin(
                postprocessor.all_bin_edges[0]*ureg(
                    postprocessor.all_bin_units[0]
                ).to('radians').magnitude),2)
            postprocessor.all_bin_cens[0] = np.power(np.sin(
                postprocessor.all_bin_cens[0]*ureg(
                    postprocessor.all_bin_units[0]
                ).to('radians').magnitude),2)
            if np.all(np.sign(postprocessor.all_bin_edges[1]) == -1):
                postprocessor.all_bin_edges[1] = \
                    postprocessor.all_bin_edges[1] + deltam21
                postprocessor.all_bin_cens[1] = \
                    postprocessor.all_bin_cens[1] + deltam21
                best_fit_data['deltam32'] = {}
                best_fit_data['deltam32']['units'] = \
                    best_fit_data['deltam31']['units']
                best_fit_data['deltam32']['val'] = \
                    best_fit_data['deltam31']['val'] + deltam21
            elif np.all(np.sign(postprocessor.all_bin_edges[1]) == 1):
                postprocessor.all_bin_edges[1] = \
                    postprocessor.all_bin_edges[1] - deltam21
                postprocessor.all_bin_cens[1] = \
                    postprocessor.all_bin_cens[1] - deltam21
                postprocessor.best_fit_data['deltam32'] = {}
                postprocessor.best_fit_data['deltam32']['units'] = \
                    postprocessor.best_fit_data['deltam31']['units']
                postprocessor.best_fit_data['deltam32']['val'] = \
                    postprocessor.best_fit_data['deltam31']['val'] - deltam21
            else:
                raise ValueError(
                    "All of the deltam31 values did not have the same sign. "
                    "There should only be one hierarchy/ordering tested here."
                )
            postprocessor.best_fit_data['sin2theta23'] = {}
            postprocessor.best_fit_data['sin2theta23']['units'] = \
                postprocessor.best_fit_data['theta23']['units']
            postprocessor.best_fit_data['sin2theta23']['val'] = \
                np.power(np.sin(postprocessor.best_fit_data['theta23'][
                    'val']*ureg(postprocessor.all_bin_units[0]).to(
                        'radians').magnitude),2)
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
    description="""Hypothesis testing: How do two hypotheses compare for
    describing MC or data?

    This computes significances, etc. from the logfiles recorded by the
    `hypo_testing.py` script, for either Asimov or LLR analysis. Plots and
    tables are produced in the case of LLR analysis."""

    # TODO:
    #
    # 1) Some of the "combined" plots currently make it impossible to read the
    #    axis labels. Come up with a better way of doing this. Could involve
    #    making legends and just labelling the axes alphabetically.
    # 2) The important one - Figure out if this script generalises to the case
    #    of analysing data. My gut says it doesn't...

    init_args_d = parse_args(description=description,
                             hypo_testing_analysis=True)

    if init_args_d['asimov']:
        data_sets, all_params, labels, minimiser_info = extract_trials(
            logdir=init_args_d['dir'],
            fluctuate_fid=False,
            fluctuate_data=False
        )
        od = data_sets.values()[0]
        #if od['h1_fit_to_h0_fid']['fid_asimov']['metric_val'] > od['h0_fit_to_h1_fid']['fid_asimov']['metric_val']:
        print np.sqrt(np.abs(
            od['h1_fit_to_h0_fid']['fid_asimov']['metric_val'] -
            od['h0_fit_to_h1_fid']['fid_asimov']['metric_val']
        ))
        return

    # Otherwise: LLR analysis
    if init_args_d['outdir'] is None:
        raise ValueError('Must specify --outdir when processing LLR results.')

    if len(formats) > 0:
        logging.info('Files will be saved in format(s) %s', formats)
    else:
        raise ValueError('Must specify a plot file format, either --png or'
                         ' --pdf, when processing LLR results.')

    postprocessor = Postprocessor(
        analysis_type='hypo_testing',
        test_type=None,
        logdir=init_args_d['dir'],
        detector=init_args_d['detector'],
        selection=init_args_d['selection'],
        outdir=init_args_d['outdir'],
        formats=init_args_d['formats'],
        fluctuate_fid=True,
        fluctuate_data=False
    )

    trial_nums = postprocessor.data_sets[
        'toy_%s_asimov'%postprocessor.labels.dict[
            'data_name']]['h0_fit_to_h1_fid'].keys()

    postprocessor.extract_fid_data()
    postprocessor.extract_data()
    postprocessor.store_extra_points(
        extra_points = init_args_d['extra_point'],
        extra_points_labels = init_args_d['extra_point_label']
    )

    if init_args_d['threshold'] != 0.0:
        logging.info('Outlying trials will be removed with a '
                     'threshold of %.2f', init_args_d['threshold'])
        postprocessor.purge_failed_jobs(
            trial_nums=np.array(trial_nums),
            thresh=init_args_d['threshold']
        )
    else:
        logging.info('All trials will be included in the analysis.')

    if len(trial_nums) != 1:
        postprocessor.make_llr_plots()


def main_injparamscan_postprocessing():
    raise NotImplementedError("Postprocessing of hypo testing injected parameter scans not implemented in this script yet.")


def main_systtests_postprocessing():
    raise NotImplementedError("Postprocessing of hypo testing systematic tests not implemented in this script yet.")


if __name__ == '__main__':
    Postprocessingargparser()
