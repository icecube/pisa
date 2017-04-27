#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This computes significances, etc. from the logfiles recorded by the
`hypo_testing.py` script, for either Asimov or LLR analysis. Plots and tables
are produced in the case of LLR analysis.
"""


# TODO:
#
# 1) Some of the "combined" plots currently make it impossible to read the axis
#    labels. Come up with a better way of doing this. Could involve making
#    legends and just labelling the axes alphabetically.
# 2) The important one - Figure out if this script generalises to the case of
#    analysing data. My gut says it doesn't...


from __future__ import division

import os
import re

import numpy as np
from scipy.stats import spearmanr

from pisa.analysis.hypo_testing import Labels
from pisa.utils.fileio import from_file, mkdir, nsort
from pisa.utils.log import set_verbosity, logging
from pisa.utils.postprocess import Postprocessor, parse_args


__all__ = ['extract_trials', 'extract_fit', 'parse_args', 'main']


def write_latex_preamble(texfile):
    """Write latex preamble needed to make nice-looking tex files."""
    texfile.write("\n")
    texfile.write("\documentclass[a4paper,12pt]{article}\n")
    texfile.write("\usepackage{tabu}\n")
    texfile.write("\usepackage{booktabs}\n")
    texfile.write("\usepackage[font=small,labelsep=space]{caption} %specifies the caption formatting for the document\n")
    texfile.write("\usepackage[margin=2.5cm]{geometry}\n")
    texfile.write("\setlength{\\topmargin}{1.0cm}\n")
    texfile.write("\setlength{\\textheight}{22cm}\n")
    texfile.write("\usepackage{fancyhdr} %allows for headers and footers\n")
    texfile.write("\pagestyle{fancy}\n")
    texfile.write("\\fancyhf{}\n")
    texfile.write("\\fancyhead[R]{\leftmark}\n")
    texfile.write("\usepackage{multirow}\n")
    texfile.write("\n")
    texfile.write("\\begin{document}\n")
    texfile.write("\n")


def setup_latex_table(texfile, tabletype, injected=False):
    """Set up the beginning of the table for the tex output files. Currently
    will make tables for the output fiducial fit params and the chi2 values
    only."""
    texfile.write("\\renewcommand{\\arraystretch}{1.6}\n")
    texfile.write("\n")
    texfile.write("\\begin{table}[t!]\n")
    texfile.write("  \\begin{center}\n")
    if tabletype == 'fiducial_fit_params':
        if injected:
            texfile.write("    \\begin{tabu} to 1.0\\textwidth {| X[2.0,c] | X[1,c] | X[1,c] | X[1,c] | X[1,c] | X[1,c] | X[1,c] | X[1,c] |}\n")
            texfile.write("    \hline\n")
            texfile.write("    \multirow{2}{*}{\\textbf{Parameter}} & \multirow{2}{*}{\\textbf{Inj}} & \multicolumn{3}{c|}{h0} & \multicolumn{3}{c|}{h1} \\\\ \cline{3-8}")
            texfile.write("    & & Prior & Fit & \(\Delta\) & Prior & Fit & \(\Delta\) \\\\ \hline\n")
        else:
            texfile.write("    \\begin{tabu} to 1.0\\textwidth {| X[c] | X[c] | X[c] |}\n")
            texfile.write("    \hline\n")
            texfile.write("    Parameter & h0 & h1 \\\\ \hline\n")
    elif tabletype == 'fiducial_fit_metrics':
        texfile.write("    \\begin{tabu} to 1.0\\textwidth {| X[c] | X[c] | X[c] |}\n")
        texfile.write("    \hline\n")
        texfile.write("    h0 & h1 & $\Delta$ \\\\ \hline\n")
    else:
        raise ValueError("This function is only for making fit metric or fit "
                         "param tables in LaTeX.")


def end_latex_file(texfile, tabletype, detector, selection, h0, h1,
                   truth=None):
    """End the table and the whole document for the tex output files."""
    if tabletype == 'fiducial_fit_params':
        texfile.write("    \end{tabu}\n")
        texfile.write("  \end{center}\n")
        texfile.write("  \\vspace{-10pt}\n")
        if truth is not None:
            texfile.write("  \caption{shows the fiducial fit parameters obtained with the %s %s sample for h0 of %s and h1 of %s. The truth is %s.}\n"%(detector, selection, h0, h1, truth))
        else:
            texfile.write("  \caption{shows the fiducial fit parameters obtained with the %s %s sample for h0 of %s and h1 of %s.}\n"%(detector, selection, h0, h1))
        texfile.write("  \label{tab:%s%s%stable}\n"%(detector, selection, tabletype))
        texfile.write("\end{table}\n")
        texfile.write("\n")
        texfile.write("\end{document}\n")
    elif tabletype == 'fiducial_fit_metrics':
        texfile.write("    \end{tabu}\n")
        texfile.write("  \end{center}\n")
        texfile.write("  \\vspace{-10pt}\n")
        if truth is not None:
            texfile.write("  \caption{shows the fiducial fit metrics obtained with the %s %s sample for h0 of %s and h1 of %s. The truth is %s.}\n"%(detector, selection, h0, h1, truth))
        else:
            texfile.write("  \caption{shows the fiducial fit metrics obtained with the %s %s sample for h0 of %s and h1 of %s.}\n"%(detector, selection, h0, h1))
        texfile.write("  \label{tab:%s%s%stable}\n"%(detector, selection, tabletype))
        texfile.write("\end{table}\n")
        texfile.write("\n")
        texfile.write("\end{document}\n")


def format_table_line(val, dataval, stddev=None, maximum=None, last=False):
    """Formatting the numbers to look nice is awkard so do it in its own
    function"""
    line = ""
    if stddev is not None:
        if (np.abs(stddev) < 1e-2) and (stddev != 0.0):
            line += r'$%.3e\pm%.3e$ &'%(maximum, stddev)
        else:
            line += r'$%.3g\pm%.3g$ &'%(maximum, stddev)
    else:
        if maximum is not None:
            raise ValueError("Both stddev and maximum should be None or "
                             "specified")
        else:
            line += "-- &"
    if (np.abs(val) < 1e-2) and (val != 0.0):
        line += "%.3e"%val
    else:
        line += "%.3g"%val
    if dataval is not None:
        line += " &"
        if isinstance(dataval, basestring):
            line += "%s"%dataval
        else:
            delta = val - dataval
            if (np.abs(delta) < 1e-2) and (delta != 0.0):
                line += "%.3e"%delta
            else:
                line += "%.3g"%delta
    if not last:
        line += " &"
    return line


def make_fiducial_fits(data, fid_data, labels, all_params, detector,
                       selection, outdir):
    """Make tex files which can be then be compiled in to tables showing the
    two fiducial fits and, if applicable, how they compare to what was
    injected."""
    outdir = os.path.join(outdir, 'FiducialFits')
    mkdir(outdir)

    # Make output file to write to
    paramfilename = ("true_%s_%s_%s_fiducial_fit_params.tex"
                     %(labels['data_name'],
                       detector,
                       selection))
    paramfile = os.path.join(outdir, paramfilename)
    pf = open(paramfile, 'w')
    write_latex_preamble(texfile=pf)

    h0_params = fid_data[
        ('h0_fit_to_toy_%s_asimov'%labels['data_name'])
    ]['params']
    h1_params = fid_data[
        ('h1_fit_to_toy_%s_asimov'%labels['data_name'])
    ]['params']

    if 'data_params' in all_params.keys():
        data_params = {}
        for pkey in all_params['data_params'].keys():
            data_params[pkey] = all_params['data_params'][pkey]['value']
        injected = True
    else:
        data_params = None
        injected = False

    setup_latex_table(
        texfile=pf,
        tabletype='fiducial_fit_params',
        injected=injected
    )

    for param in h0_params.keys():
        # Get the units for this parameter
        val, param_units = parse_pint_string(
            pint_string=fid_data[
                'h0_fit_to_toy_%s_asimov'%labels['data_name']
            ]['params'][param]
        )
        # Get priors if they exists
        if 'gaussian' in all_params['h0_params'][param]['prior']:
            h0stddev, h0maximum = extract_gaussian(
                prior_string=all_params['h0_params'][param]['prior'],
                units=param_units
            )
        else:
            h0stddev = None
        if 'gaussian' in all_params['h1_params'][param]['prior']:
            h1stddev, h1maximum = extract_gaussian(
                prior_string=all_params['h1_params'][param]['prior'],
                units=param_units
            )
        else:
            h1stddev = None
        # Include injected parameter, fitted parameters and differences with
        # appropriate formatting.
        if data_params is not None:
            tableline = "      "
            tableline += "%s "%tex_axis_label(param)
            if param == 'deltam31':
                tableline += r" / $10^{-3}$ "
            if param_units != 'dimensionless':
                tableline += "(%s) &"%tex_axis_label(param_units)
            else:
                tableline += "&"
            if param in data_params.keys():
                dataval = extract_paramval(
                    injparams=data_params,
                    systkey=param
                )
                if param == 'deltam31':
                    dataval *= 1000.0
                if (np.abs(dataval) < 1e-2) and (dataval != 0.0):
                    tableline += "%.3e &"%dataval
                else:
                    tableline += "%.3g &"%dataval
            # If no injected parameter, show this and the deltas with a line
            else:
                dataval = '--'
                tableline += "%s &"%dataval
            h0val = extract_paramval(
                injparams=h0_params,
                systkey=param
            )
            if param == 'deltam31':
                h0val *= 1000.0
            if h0stddev is not None:
                tableline += format_table_line(val=h0val, dataval=dataval,
                                               stddev=h0stddev,
                                               maximum=h0maximum)
            else:
                tableline += format_table_line(val=h0val, dataval=dataval)
            h1val = extract_paramval(
                injparams=h1_params,
                systkey=param
            )
            if param == 'deltam31':
                h1val *= 1000.0
            if h1stddev is not None:
                tableline += format_table_line(val=h1val, dataval=dataval,
                                               stddev=h1stddev,
                                               maximum=h1maximum, last=True)
            else:
                tableline += format_table_line(val=h1val, dataval=dataval,
                                               last=True)
            tableline += " \\\\ \hline\n"
            pf.write(tableline)
        # If no injected parameters it's much simpler
        else:
            h0val = extract_paramval(
                injparams=h0_params,
                systkey=param
            )
            h1val = extract_paramval(
                injparams=h1_params,
                systkey=param
            )
            if (np.abs(h0val) < 1e-2) and (h0val != 0.0):
                pf.write("    %s & %.3e & %.3e\n"
                         % (tex_axis_label(param), h0val, h1val))
            else:
                pf.write("    %s & %.3g & %.3g\n"
                         % (tex_axis_label(param), h0val, h1val))

    end_latex_file(
        texfile=pf,
        tabletype='fiducial_fit_params',
        detector=detector,
        selection=selection,
        h0=tex_axis_label(labels['h0_name']),
        h1=tex_axis_label(labels['h1_name']),
        truth=tex_axis_label(labels['data_name'])
    )

    # Make output file to write to
    metricfilename = ("true_%s_%s_%s_fiducial_fit_metrics.tex"
                      %(labels['data_name'], detector, selection))
    metricfile = os.path.join(outdir, metricfilename)
    mf = open(metricfile, 'w')
    write_latex_preamble(texfile=mf)

    setup_latex_table(
        texfile=mf,
        tabletype='fiducial_fit_metrics'
    )

    h0_fid_metric = fid_data[
        'h0_fit_to_toy_%s_asimov'%labels['data_name']
    ][
        'metric_val'
    ]
    h1_fid_metric = fid_data[
        'h1_fit_to_toy_%s_asimov'%labels['data_name']
    ][
        'metric_val'
    ]

    metric_type = data['h0_fit_to_h0_fid']['metric_val']['type']
    metric_type_pretty = tex_axis_label(metric_type)
    # In the case of likelihood, the maximum metric is the better fit.
    # With chi2 metrics the opposite is true, and so we must multiply
    # everything by -1 in order to apply the same treatment.
    if 'chi2' not in metric_type:
        logging.info("Converting likelihood metric to chi2 equivalent.")
        h0_fid_metric *= -1
        h1_fid_metric *= -1

    # If truth is known, report the fits the correct way round
    if labels['data_name'] is not None:
        if labels['data_name'] in labels['h0_name']:
            delta = h1_fid_metric-h0_fid_metric
        elif labels['data_name'] in labels['h1_name']:
            delta = h0_fid_metric-h1_fid_metric
        else:
            logging.warning("Truth is known but could not be identified in "
                            "either of the hypotheses. The difference between"
                            " the best fit metrics will just be reported as "
                            "positive and so will not necessarily reflect if "
                            "the truth was recovered.")
            if h1_fid_metric > h0_fid_metric:
                delta = h0_fid_metric-h1_fid_metric
            else:
                delta = h1_fid_metric-h0_fid_metric
    # Else just report it as delta between best fits
    else:
        if h1_fid_metric > h0_fid_metric:
            delta = h0_fid_metric-h1_fid_metric
        else:
            delta = h1_fid_metric-h0_fid_metric
    # Write this in the file
    mf.write("    %.3g & %.3g & %.3g \\\\ \hline\n"%(h0_fid_metric, h1_fid_metric, delta))
    # Then end the file
    end_latex_file(
        texfile=mf,
        tabletype='fiducial_fit_metrics',
        detector=detector,
        selection=selection,
        h0=tex_axis_label(labels['h0_name']),
        h1=tex_axis_label(labels['h1_name']),
        truth=tex_axis_label(labels['data_name'])
    )


def plot_individual_posterior(data, data_params, h0_params, h1_params,
                              all_params, labels, h0label, h1label, systkey,
                              fhkey, subplotnum=None):
    """Use matplotlib to make a histogram of the vals contained in data. The
    injected value will be plotted along with, where appropriate, the "wrong
    hypothesis" fiducial fit and the prior. The axis labels and the legend are
    taken care of in here. The optional subplotnum argument can be given in the
    combined case so that the y-axis label only get put on when appropriate."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    if systkey == 'metric_val':
        metric_type = data['type']
    systvals = np.array(data['vals'])
    units = data['units']

    hypo = fhkey.split('_')[0]
    fid = fhkey.split('_')[-2]

    plt.grid(axis='y', zorder=0)
    plt.hist(
        systvals,
        bins=10,
        histtype='bar',
        color='darkblue',
        alpha=0.9,
        zorder=3
    )

    # Add injected and hypothesis fit lines
    if not systkey == 'metric_val':
        if data_params is not None:
            if systkey in data_params.keys():
                injval, injlabelproper = extract_paramval(
                    injparams=data_params,
                    systkey=systkey,
                    fid_label=labels['%s_name'%fid],
                    hypo_label=labels['%s_name'%hypo],
                    paramlabel='Injected Value'
                )
                plt.axvline(
                    injval,
                    color='r',
                    linewidth=2,
                    label=injlabelproper,
                    zorder=5
                )
            else:
                injval = None
        else:
            injval = None
        if fid == 'h0':
            fitval, fitlabelproper = extract_paramval(
                injparams=h0_params,
                systkey=systkey,
                fid_label=labels['%s_name'%fid],
                hypo_label=labels['%s_name'%hypo],
                paramlabel=h0label
            )
        elif fid == 'h1':
            fitval, fitlabelproper = extract_paramval(
                injparams=h1_params,
                systkey=systkey,
                fid_label=labels['%s_name'%fid],
                hypo_label=labels['%s_name'%hypo],
                paramlabel=h1label
            )
        else:
            raise ValueError("I got a hypothesis %s. Expected h0 or h1 only."
                             %fid)
        if injval is not None:
            if fitval != injval:
                plt.axvline(
                    fitval,
                    color='g',
                    linewidth=2,
                    label=fitlabelproper,
                    zorder=5
                )
        else:
            plt.axvline(
                fitval,
                color='g',
                linewidth=2,
                label=fitlabelproper,
                zorder=5
            )
    # Add shaded region for prior, if appropriate
    # TODO - Deal with non-gaussian priors
    wanted_params = all_params['%s_params'%hypo]
    for param in wanted_params.keys():
        if param == systkey:
            if 'gaussian' in wanted_params[param]['prior']:
                stddev, maximum = extract_gaussian(
                    prior_string=wanted_params[param]['prior'],
                    units=units
                )
                currentxlim = plt.xlim()
                if (np.abs(stddev) < 1e-2) and (stddev != 0.0):
                    priorlabel = (r'Gaussian Prior '
                                  '($%.3e\pm%.3e$)'%(maximum, stddev))
                else:
                    priorlabel = (r'Gaussian Prior '
                                  '($%.3g\pm%.3g$)'%(maximum, stddev))
                plt.axvspan(
                    maximum-stddev,
                    maximum+stddev,
                    color='k',
                    label=priorlabel,
                    ymax=0.1,
                    alpha=0.5,
                    zorder=5
                )
                # Reset xlimits if prior makes it go far off
                if plt.xlim()[0] < currentxlim[0]:
                    plt.xlim(currentxlim[0], plt.xlim()[1])
                if plt.xlim()[1] > currentxlim[1]:
                    plt.xlim(plt.xlim()[0], currentxlim[1])

    # Make axis labels look nice
    if systkey == 'metric_val':
        systname = tex_axis_label(metric_type)
    else:
        systname = tex_axis_label(systkey)
    if not units == 'dimensionless':
        systname += r' (%s)'%tex_axis_label(units)

    plt.xlabel(systname, size='24')
    if subplotnum is not None:
        if (subplotnum-1)%4 == 0:
            plt.ylabel(r'Number of Trials', size='24')
    else:
        plt.ylabel(r'Number of Trials', size='24')
    plt.ylim(0, 1.35*plt.ylim()[1])
    if not systkey == 'metric_val':
        plt.legend(loc='upper left')

    plt.subplots_adjust(left=0.10, right=0.90, top=0.9, bottom=0.11)


def plot_individual_posteriors(data, fid_data, labels, all_params, detector,
                               selection, outdir, formats):
    """Use `plot_individual_posterior` and save each time."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True


    outdir = os.path.join(outdir, 'IndividualPosteriors')
    mkdir(outdir)

    MainTitle = make_main_title(detector, selection, 'Posterior')

    h0_params = fid_data[
        ('h0_fit_to_toy_%s_asimov'%labels['data_name'])
    ]['params']
    h1_params = fid_data[
        ('h1_fit_to_toy_%s_asimov'%labels['data_name'])
    ]['params']

    if 'data_params' in all_params.keys():
        data_params = {}
        for pkey in all_params['data_params'].keys():
            data_params[pkey] = all_params['data_params'][pkey]['value']
    else:
        data_params = None

    for fhkey in data.keys():
        for systkey in data[fhkey].keys():

            hypo = fhkey.split('_')[0]
            fid = fhkey.split('_')[-2]
            FitTitle = make_fit_title(
                labels=labels,
                fid=fid,
                hypo=hypo,
                trials=len(data[fhkey][systkey]['vals'])
            )
            plot_individual_posterior(
                data=data[fhkey][systkey],
                data_params=data_params,
                h0_params=h0_params,
                h1_params=h1_params,
                all_params=all_params,
                labels=labels,
                h0label='%s Fiducial Fit'%labels['h0_name'],
                h1label='%s Fiducial Fit'%labels['h1_name'],
                systkey=systkey,
                fhkey=fhkey
            )

            plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
            save_plot(
                labels=labels,
                fid=fid,
                hypo=hypo,
                detector=detector,
                selection=selection,
                outdir=outdir,
                formats=formats,
                end='%s_posterior'%(systkey)
            )
            plt.close()


def plot_combined_posteriors(data, fid_data, labels, all_params,
                             detector, selection, outdir, formats):
    """Use `plot_individual_posterior` multiple times but just save once all of
    the posteriors for a given combination of h0 and h1 have been plotted on
    the same canvas."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    outdir = os.path.join(outdir, 'CombinedPosteriors')
    mkdir(outdir)

    MainTitle = make_main_title(detector, selection, 'Posteriors')

    h0_params = fid_data[
        ('h0_fit_to_toy_%s_asimov'%labels['data_name'])
    ]['params']
    h1_params = fid_data[
        ('h1_fit_to_toy_%s_asimov'%labels['data_name'])
    ]['params']

    if 'data_params' in all_params.keys():
        data_params = {}
        for pkey in all_params['data_params'].keys():
            data_params[pkey] = all_params['data_params'][pkey]['value']
    else:
        data_params = None

    labels['MainTitle'] = MainTitle

    for fhkey in data.keys():

        # Set up multi-plot
        num_rows = get_num_rows(data[fhkey], omit_metric=False)
        plt.figure(figsize=(20, 5*num_rows+2))
        subplotnum = 1

        for systkey in data[fhkey].keys():

            hypo = fhkey.split('_')[0]
            fid = fhkey.split('_')[-2]
            FitTitle = make_fit_title(
                labels=labels,
                fid=fid,
                hypo=hypo,
                trials=len(data[fhkey][systkey]['vals'])
            )
            plt.subplot(num_rows, 4, subplotnum)

            plot_individual_posterior(
                data=data[fhkey][systkey],
                data_params=data_params,
                h0_params=h0_params,
                h1_params=h1_params,
                all_params=all_params,
                labels=labels,
                h0label='%s Fiducial Fit'%labels['h0_name'],
                h1label='%s Fiducial Fit'%labels['h1_name'],
                systkey=systkey,
                fhkey=fhkey,
                subplotnum=subplotnum
            )

            subplotnum += 1

        plt.suptitle(MainTitle+r'\\'+FitTitle, fontsize=36)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        save_plot(
            labels=labels,
            fid=fid,
            hypo=hypo,
            detector=detector,
            selection=selection,
            outdir=outdir,
            formats=formats,
            end='posteriors'
        )
        plt.close()


def plot_individual_scatter(xdata, ydata, labels, xsystkey, ysystkey,
                            subplotnum=None, num_rows=None, plot_cor=True):
    """Use matplotlib to make a scatter plot of the vals contained in xdata and
    ydata. The correlation will be calculated and the plot will be annotated
    with this. Axis labels are done in here too. The optional subplotnum
    argument can be given in the combined case so that the y-axis label only
    get put on when appropriate."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    # Extract data and units
    xvals = np.array(xdata['vals'])
    xunits = xdata['units']
    yvals = np.array(ydata['vals'])
    yunits = ydata['units']

    # Make scatter plot
    plt.scatter(xvals, yvals)

    if plot_cor:
        # Calculate correlation and annotate
        if len(set(xvals)) == 1:
            logging.warn(
                'Parameter %s appears to not have been varied. i.e. all of the'
                ' values in the set are the same. This will lead to NaN in the'
                ' correlation calculation and so it will not be done.',
                xsystkey
            )
        if len(set(yvals)) == 1:
            logging.warn(
                'Parameter %s appears to not have been varied. i.e. all of the'
                ' values in the set are the same. This will lead to NaN in the'
                ' correlation calculation and so it will not be done.',
                ysystkey
            )
        if (len(set(xvals)) != 1) and (len(set(yvals)) != 1):
            rho, pval = spearmanr(xvals, yvals)
            if subplotnum is not None:
                row = int((subplotnum-1)/4)
                xtext = 0.25*0.25+((subplotnum-1)%4)*0.25
                ytext = 0.88-(1.0/num_rows)*0.9*row
                plt.figtext(
                    xtext,
                    ytext,
                    'Correlation = %.2f'%rho,
                    fontsize='large'
                )
            else:
                plt.figtext(
                    0.15,
                    0.85,
                    'Correlation = %.2f'%rho,
                    fontsize='large'
                )

    # Make plot range easy to look at
    Xrange = xvals.max() - xvals.min()
    Yrange = yvals.max() - yvals.min()
    if Xrange != 0.0:
        plt.xlim(xvals.min()-0.1*Xrange,
                 xvals.max()+0.1*Xrange)
    if Yrange != 0.0:
        plt.ylim(yvals.min()-0.1*Yrange,
                 yvals.max()+0.3*Yrange)

    # Make axis labels look nice
    xsystname = tex_axis_label(xsystkey)
    if not xunits == 'dimensionless':
        xsystname += r' (%s)'%tex_axis_label(xunits)
    ysystname = tex_axis_label(ysystkey)
    if not yunits == 'dimensionless':
        ysystname += r' (%s)'%tex_axis_label(yunits)

    plt.xlabel(xsystname)
    plt.ylabel(ysystname)


def plot_individual_scatters(data, labels, detector, selection,
                             outdir, formats):
    """Use `plot_individual_scatter` and save every time."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    outdir = os.path.join(outdir, 'IndividualScatterPlots')
    mkdir(outdir)

    MainTitle = make_main_title(detector, selection, 'Correlation Plot')

    for fhkey in data.keys():
        for xsystkey in data[fhkey].keys():
            if not xsystkey == 'metric_val':
                for ysystkey in data[fhkey].keys():
                    if (ysystkey != 'metric_val') and (ysystkey != xsystkey):

                        hypo = fhkey.split('_')[0]
                        fid = fhkey.split('_')[-2]
                        FitTitle = make_fit_title(
                            labels=labels,
                            fid=fid,
                            hypo=hypo,
                            trials=len(data[fhkey][xsystkey]['vals'])
                        )

                        plot_individual_scatter(
                            xdata=data[fhkey][xsystkey],
                            ydata=data[fhkey][ysystkey],
                            labels=labels,
                            xsystkey=xsystkey,
                            ysystkey=ysystkey
                        )

                        plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
                        save_plot(
                            labels=labels,
                            fid=fid,
                            hypo=hypo,
                            detector=detector,
                            selection=selection,
                            outdir=outdir,
                            formats=formats,
                            end='%s_%s_scatter_plot'%(xsystkey, ysystkey)
                        )
                        plt.close()


def plot_combined_individual_scatters(data, labels, detector,
                                      selection, outdir, formats):
    """Use `plot_individual_scatter` and save once all of the scatter plots for
    a single systematic with every other systematic have been plotted on the
    same canvas for each h0 and h1 combination."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    outdir = os.path.join(outdir, 'CombinedScatterPlots')
    mkdir(outdir)

    MainTitle = make_main_title(detector, selection, 'Correlation Plot')

    for fhkey in data.keys():
        for xsystkey in data[fhkey].keys():
            if not xsystkey == 'metric_val':

                # Set up multi-plot
                num_rows = get_num_rows(data[fhkey], omit_metric=True)
                plt.figure(figsize=(20, 5*num_rows+2))
                subplotnum = 1

                for ysystkey in data[fhkey].keys():
                    if (ysystkey != 'metric_val') and (ysystkey != xsystkey):

                        hypo = fhkey.split('_')[0]
                        fid = fhkey.split('_')[-2]
                        FitTitle = make_fit_title(
                            labels=labels,
                            fid=fid,
                            hypo=hypo,
                            trials=len(data[fhkey][xsystkey]['vals'])
                        )

                        plt.subplot(num_rows, 4, subplotnum)

                        plot_individual_scatter(
                            xdata=data[fhkey][xsystkey],
                            ydata=data[fhkey][ysystkey],
                            labels=labels,
                            xsystkey=xsystkey,
                            ysystkey=ysystkey,
                            subplotnum=subplotnum,
                            num_rows=num_rows
                        )

                        subplotnum += 1

                plt.suptitle(MainTitle+r'\\'+FitTitle, fontsize=36)
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)
                save_plot(
                    labels=labels,
                    fid=fid,
                    hypo=hypo,
                    detector=detector,
                    selection=selection,
                    outdir=outdir,
                    formats=formats,
                    end='%s_scatter_plot'%(xsystkey)
                )
                plt.close()


def plot_combined_scatters(data, labels, detector, selection, outdir, formats):
    """Use `plot_individual_scatter` and save once every scatter plot has been
    plotted on a single canvas for each of the h0 and h1 combinations."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    outdir = os.path.join(outdir, 'CombinedScatterPlots')
    mkdir(outdir)

    MainTitle = make_main_title(detector, selection, 'Correlation Plot')

    for fhkey in data.keys():
        # Systematic number is one less than number of keys since this also
        # contains the metric_val entry
        SystNum = len(data[fhkey].keys())-1
        # Set up multi-plot
        plt.figure(figsize=(3.5*(SystNum-1), 3.5*(SystNum-1)))
        subplotnum = (SystNum-1)*(SystNum-1)+1
        # Set up container to know which correlations have already been plotted
        PlottedSysts = []
        for xsystkey in data[fhkey].keys():
            if not xsystkey == 'metric_val':
                PlottedSysts.append(xsystkey)
                for ysystkey in data[fhkey].keys():
                    if (ysystkey != 'metric_val') and (ysystkey != xsystkey):
                        subplotnum -= 1
                        if ysystkey not in PlottedSysts:

                            hypo = fhkey.split('_')[0]
                            fid = fhkey.split('_')[-2]
                            FitTitle = make_fit_title(
                                labels=labels,
                                fid=fid,
                                hypo=hypo,
                                trials=len(data[fhkey][xsystkey]['vals'])
                            )

                            plt.subplot(SystNum-1, SystNum-1, subplotnum)

                            plot_individual_scatter(
                                xdata=data[fhkey][xsystkey],
                                ydata=data[fhkey][ysystkey],
                                labels=labels,
                                xsystkey=xsystkey,
                                ysystkey=ysystkey,
                                plot_cor=False
                            )

        plt.suptitle(MainTitle+r'\\'+FitTitle, fontsize=120)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        save_plot(
            labels=labels,
            fid=fid,
            hypo=hypo,
            detector=detector,
            selection=selection,
            outdir=outdir,
            formats=formats,
            end='all_scatter_plots'
        )
        plt.close()


def plot_correlation_matrices(data, labels, detector, selection,
                              outdir, formats):
    """Plot the correlation matrices since the individual scatter plots are a
    pain to interpret on their own. This will plot them with a colour scale
    and, if the user has the PathEffects module then it will also write the
    values on the bins. If a number is invalid it will come up bright green.
    """
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    try:
        import matplotlib.patheffects as PathEffects
        logging.warn('PathEffects could be imported, so the correlation values'
                     ' will be written on the bins. This is slow.')
        pe = True
    except ImportError:
        logging.warn('PathEffects could not be imported, so the correlation'
                     ' values will not be written on the bins.')
        pe = False

    outdir = os.path.join(outdir, 'CorrelationMatrices')
    mkdir(outdir)

    MainTitle = make_main_title(detector, selection, 'Correlation Coefficients')
    Systs = []

    for fhkey in data.keys():
        # Systematic number is one less than number of keys since this also
        # contains the metric_val entry
        SystNum = len(data[fhkey].keys())-1
        # Set up array to hold lists of correlation values
        all_corr_lists = []
        for xsystkey in data[fhkey].keys():
            all_corr_values = []
            if not xsystkey == 'metric_val':
                if tex_axis_label(xsystkey) not in Systs:
                    Systs.append(tex_axis_label(xsystkey))
                for ysystkey in data[fhkey].keys():
                    if ysystkey != 'metric_val':
                        hypo = fhkey.split('_')[0]
                        fid = fhkey.split('_')[-2]
                        FitTitle = make_fit_title(
                            labels=labels,
                            fid=fid,
                            hypo=hypo,
                            trials=len(data[fhkey][xsystkey]['vals'])
                        )
                        # Calculate correlation
                        xvals = np.array(data[fhkey][xsystkey]['vals'])
                        yvals = np.array(data[fhkey][ysystkey]['vals'])
                        msg = ('Parameter %s appears to not have been varied.'
                               ' i.e. all of the values in the set are the'
                               ' same. This will lead to NaN in the'
                               ' correlation calculation and so it will not be'
                               ' done.')
                        if len(set(xvals)) == 1:
                            logging.warn(msg, xsystkey)
                        if len(set(yvals)) == 1:
                            logging.warn(msg, ysystkey)
                        if len(set(xvals)) != 1 and len(set(yvals)) != 1:
                            rho, pval = spearmanr(xvals, yvals)
                        else:
                            rho = np.nan
                        all_corr_values.append(rho)
                all_corr_lists.append(all_corr_values)

        all_corr_nparray = np.ma.masked_invalid(np.array(all_corr_lists))
        # Plot it!
        palette = plt.cm.RdBu
        palette.set_bad('lime', 1.0)
        plt.imshow(
            all_corr_nparray,
            interpolation='none',
            cmap=plt.cm.RdBu,
            vmin=-1.0,
            vmax=1.0
        )
        plt.colorbar()
        # Add systematic names as x and y axis ticks
        plt.xticks(
            np.arange(len(Systs)),
            Systs,
            rotation=45,
            horizontalalignment='right'
        )
        plt.yticks(
            np.arange(len(Systs)),
            Systs,
            rotation=0
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.30, left=-0.30, right=1.05, top=0.9)
        plt.title(MainTitle+r'\\'+FitTitle, fontsize=16)
        save_plot(
            labels=labels,
            fid=fid,
            hypo=hypo,
            detector=detector,
            selection=selection,
            outdir=outdir,
            formats=formats,
            end='correlation_matrix'
        )
        if pe:
            for i in range(0, len(all_corr_nparray)):
                for j in range(0, len(all_corr_nparray[0])):
                    plt.text(i, j, '%.2f'%all_corr_nparray[i][j],
                             fontsize='7',
                             verticalalignment='center',
                             horizontalalignment='center',
                             color='w',
                             path_effects=[
                                 PathEffects.withStroke(
                                     linewidth=2.5,
                                     foreground='k'
                                 )
                             ])
        save_plot(
            labels=labels,
            fid=fid,
            hypo=hypo,
            detector=detector,
            selection=selection,
            outdir=outdir,
            formats=formats,
            end='correlation_matrix_values'
        )
        plt.close()


def main():
    """Main"""
    init_args_d = parse_args()

    formats = init_args_d['formats']

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
        formats=formats,
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

    print L

    for injkey in postprocessor.values.keys():

        make_fiducial_fits(
            data=values[injkey],
            fid_data=fid_values[injkey],
            labels=labels.dict,
            all_params=all_params,
            detector=init_args_d['detector'],
            selection=init_args_d['selection'],
            outdir=init_args_d['outdir']
        )

        if init_args_d['fit_information']:
            plot_fit_information(
                minimiser_info=minimiser_info[injkey],
                labels=labels.dict,
                detector=init_args_d['detector'],
                selection=init_args_d['selection'],
                outdir=init_args_d['outdir'],
                formats=formats
            )

        if init_args_d['individual_posteriors']:
            plot_individual_posteriors(
                data=values[injkey],
                fid_data=fid_values[injkey],
                labels=labels.dict,
                all_params=all_params,
                detector=init_args_d['detector'],
                selection=init_args_d['selection'],
                outdir=init_args_d['outdir'],
                formats=formats
            )

        if init_args_d['combined_posteriors']:
            plot_combined_posteriors(
                data=values[injkey],
                fid_data=fid_values[injkey],
                labels=labels.dict,
                all_params=all_params,
                detector=init_args_d['detector'],
                selection=init_args_d['selection'],
                outdir=init_args_d['outdir'],
                formats=formats
            )

        if init_args_d['individual_scatter']:
            plot_individual_scatters(
                data=values[injkey],
                labels=labels.dict,
                detector=init_args_d['detector'],
                selection=init_args_d['selection'],
                outdir=init_args_d['outdir'],
                formats=formats
            )

        if init_args_d['combined_individual_scatter']:
            plot_combined_individual_scatters(
                data=values[injkey],
                labels=labels.dict,
                detector=init_args_d['detector'],
                selection=init_args_d['selection'],
                outdir=init_args_d['outdir'],
                formats=formats
            )

        if init_args_d['combined_scatter']:
            plot_combined_scatters(
                data=values[injkey],
                labels=labels.dict,
                detector=init_args_d['detector'],
                selection=init_args_d['selection'],
                outdir=init_args_d['outdir'],
                formats=formats
            )

        if init_args_d['correlation_matrix']:
            plot_correlation_matrices(
                data=values[injkey],
                labels=labels.dict,
                detector=init_args_d['detector'],
                selection=init_args_d['selection'],
                outdir=init_args_d['outdir'],
                formats=formats
            )


if __name__ == '__main__':
    main()
