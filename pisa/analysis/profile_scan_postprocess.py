#! /usr/bin/env python
# authors: S. Wren
# date:   November 22, 2016
"""
A script for processing the output files of profile_scan.py
"""

import os
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.fileio import from_file
from pisa.utils.log import logging, set_verbosity


def make_pretty(label):
    '''
    Takes the labels used in the objects and turns them in to something nice
    for plotting. This can never truly be exhaustive, but it definitely does 
    the trick. If something looks ugly add it to this function!
    '''
    pretty_labels = {}
    pretty_labels["atm_muon_scale"] = r"Muon Background Scale"
    pretty_labels["nue_numu_ratio"] = r"$\nu_e/\nu_{\mu}$ Ratio"
    pretty_labels["Barr_uphor_ratio"] = r"Barr Up/Horizontal Ratio"
    pretty_labels["Barr_nu_nubar_ratio"] = r"Barr $\nu/\bar{\nu}$ Ratio"
    pretty_labels["delta_index"] = r"Atmospheric Index Change"
    pretty_labels["theta13"] = r"$\theta_{13}$"
    pretty_labels["theta23"] = r"$\theta_{23}$"
    pretty_labels["deltam31"] = r"$\Delta m^2_{31}$"
    pretty_labels["aeff_scale"] = r"$A_{\mathrm{eff}}$ Scale"
    pretty_labels["energy_scale"] = r"Energy Scale"
    pretty_labels["Genie_Ma_QE"] = r"GENIE $M_{A}^{QE}$"
    pretty_labels["Genie_Ma_RES"] = r"GENIE $M_{A}^{Res}$"
    pretty_labels["dom_eff"] = r"DOM Efficiency"
    pretty_labels["hole_ice"] = r"Hole Ice"
    pretty_labels["hole_ice_fwd"] = r"Hole Ice Forward"
    pretty_labels["degree"] = r"$^\circ$"
    pretty_labels["radians"] = r"rads"
    pretty_labels["electron_volt ** 2"] = r"$\mathrm{eV}^2$"
    pretty_labels["electron_volt"] = r"$\mathrm{eV}^2$"
    pretty_labels["llh"] = r"Likelihood"
    pretty_labels["chi2"] = r"$\chi^2$"
    pretty_labels["mod_chi2"] = r"Modified $\chi^2$"
    if label not in pretty_labels.keys():
        logging.warn("I don't know what to do with %s. Returning as is."%label)
        return label
    return pretty_labels[label]


def make_label(label, units):
    '''
    Appends units to a label for plotting.
    '''
    nice_label = make_pretty(label)
    if not units == 'dimensionless':
        nice_label += ' (%s)'%make_pretty(units)
    return nice_label


def make_2D_scan_title(detector, selection, param1, param2):
    '''
    Make a title based on choices of detector and selection for 2D parameter 
    scans.
    '''
    if (detector is not None) and (selection is not None):
        MainTitle = ("%s %s Event Selection %s / %s Parameter Scan"
                     %(detector,
                       selection,
                       make_pretty(param1),
                       make_pretty(param2)))
    elif (detector is not None):
        MainTitle = ("%s %s / %s Parameter Scan"
                     %(detector,
                       make_pretty(param1),
                       make_pretty(param2)))
    elif (selection is not None):
        MainTitle = ("%s Event Selection %s / %s Parameter Scan"
                     %(detector,
                       selection,
                       make_pretty(param1),
                       make_pretty(param2)))
    else:
        MainTitle = ("%s / %s Parameter Scan"
                     %(make_pretty(param1),
                       make_pretty(param2)))
    return MainTitle


def make_2D_hist_plot(zvals, xbins, ybins, xlabel,
                      xunits, ylabel, yunits, zlabel, zunits):
    '''
    Generic 2D histogram-style plotting function.
    '''
    plt.pcolormesh(xbins, ybins, zvals.T, cmap='Blues')
    plt.xlim(xbins[0],xbins[-1])
    nice_xlabel = make_label(xlabel, xunits)
    plt.xlabel(nice_xlabel,fontsize=24)
    plt.ylim(ybins[0],ybins[-1])
    nice_ylabel = make_label(ylabel, yunits)
    plt.ylabel(nice_ylabel,fontsize=24)
    nice_clabel = make_label(zlabel, zunits)
    plt.colorbar().set_label(label=nice_clabel,fontsize=24)


def plot_2D_hists_individual(data, xbins, ybins, detector, selection,
                             MainTitle, xlabel, xunits, ylabel, yunits,
                             metric_name, outdir):
    '''
    Plot all of the 2D plots histogram-style individually.
    '''
    for data_key in data.keys():

        if data_key == 'metric_vals':
            zlabel = metric_name
            zunits = 'dimensionless'
            zvals = np.array(data[data_key])
        else:
            zlabel = data_key
            zunits = data[data_key]['units']
            zvals = np.array(data[data_key]['vals'])

        zvals = np.array(np.split(zvals, len(xbins)-1))
        
        make_2D_hist_plot(
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
        
        plt.title(MainTitle, fontsize=16)
        plt.tight_layout()
        SaveName = ("%s_%s_%s_%s_2D_%s_scan_%s_values.png"
                    %(detector,
                      selection,
                      xlabel,
                      ylabel,
                      metric_name,
                      zlabel))
        plt.savefig(os.path.join(outdir,SaveName))
        plt.close()


def plot_2D_hists_together(data, xbins, ybins, detector, selection,
                           MainTitle, xlabel, xunits, ylabel, yunits,
                           metric_name, outdir):
    '''
    Plot all of the 2D plots histogram-style on one plot
    '''
    # Set up multi-plot
    num_rows = int(len(data.keys())/4)
    if len(data.keys())%4 != 0:
        num_rows += 1
    num_plots = len(data.keys())
    if num_plots > 4:
        num_cols = 4
    else:
        num_cols = num_plots
    plt.figure(figsize=(9*num_cols,5*num_rows+2))
    subplotnum=1
    for data_key in data.keys():

        if data_key == 'metric_vals':
            zlabel = metric_name
            zunits = 'dimensionless'
            zvals = np.array(data[data_key])
        else:
            zlabel = data_key
            zunits = data[data_key]['units']
            zvals = np.array(data[data_key]['vals'])

        zvals = np.array(np.split(zvals, len(xbins)-1))

        plt.subplot(num_rows,num_cols,subplotnum)
        
        make_2D_hist_plot(
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

        subplotnum += 1
        
    plt.suptitle(MainTitle, fontsize=36)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    SaveName = ("%s_%s_%s_%s_2D_%s_scan_all_values.png"
                %(detector,
                  selection,
                  xlabel,
                  ylabel,
                  metric_name))
    plt.savefig(os.path.join(outdir,SaveName))
    plt.close()
        

def plot_2D_hists(data, xbins, ybins, detector, selection,
                  xlabel, xunits, ylabel, yunits, metric_name, outdir):
        
    MainTitle = make_2D_scan_title(
        detector=detector,
        selection=selection,
        param1=xlabel,
        param2=ylabel
    )

    plot_2D_hists_individual(
        data=data,
        xbins=xbins,
        ybins=ybins,
        detector=detector,
        selection=selection,
        MainTitle=MainTitle,
        xlabel=xlabel,
        xunits=xunits,
        ylabel=ylabel,
        yunits=yunits,
        metric_name=metric_name,
        outdir=outdir
    )

    plot_2D_hists_together(
        data=data,
        xbins=xbins,
        ybins=ybins,
        detector=detector,
        selection=selection,
        MainTitle=MainTitle,
        xlabel=xlabel,
        xunits=xunits,
        ylabel=ylabel,
        yunits=yunits,
        metric_name=metric_name,
        outdir=outdir
    )


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '--infile', metavar='FILE', type=str, required=True,
        help='''Output file of profile_scan.py to processs.'''
    )
    parser.add_argument(
        '--detector',type=str,default='',
        help="Name of detector to put in histogram titles."
    )
    parser.add_argument(
        '--selection',type=str,default='',
        help="Name of selection to put in histogram titles."
    )
    parser.add_argument(
        '--outdir', metavar='DIR', type=str, required=True,
        help='''Store the output plot to this directory.'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.outdir):
        raise ValueError('Output directory selected does not exist!')

    set_verbosity(args.v)

    infile = from_file(args.infile)

    all_steps = infile['steps']
    all_bin_cens = []
    all_bin_names = all_steps.keys()
    all_bin_units = []
    for step_variable in all_steps.keys():
        bin_cens = []
        for val in all_steps[step_variable]:
            if val[0] not in bin_cens:
                bin_cens.append(val[0])
            if val[1][0][0] not in all_bin_units:
                all_bin_units.append(val[1][0][0])
        all_bin_cens.append(bin_cens)

    results = infile['results']
    metric_name = results[0]['metric']

    # Store the metric vals
    metric_vals = []
    # Store the metric vals and resulting parameter values from those profiled
    data = {}
    data['metric_vals'] = []
    for result in results:
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

    metric_vals = np.array(metric_vals)

    if len(all_bin_cens) == 1:
        # One-dimensional parameter scan
        scan_values = all_bin_cens[0]
        scan_name = all_bin_names[0]
        plt.plot(scan_values, metric_vals)
        plt.xlabel(make_pretty(scan_name))
        plt.ylabel(make_pretty(metric_name))
        plt.savefig('%s_1D_%s_scan.png'%(scan_name, metric_name))

    elif len(all_bin_cens) == 2:
        # Two-dimensional parameter scan
        metric_vals = np.array(np.split(metric_vals, len(all_bin_cens[0])))
        xbins_width = all_bin_cens[0][1]-all_bin_cens[0][0]
        ybins_width = all_bin_cens[1][1]-all_bin_cens[1][0]

        xbin_edges = np.linspace(all_bin_cens[0][0]-xbins_width/2.0,
                                 all_bin_cens[0][-1]+xbins_width/2.0,
                                 len(all_bin_cens[0])+1)
        ybin_edges = np.linspace(all_bin_cens[1][0]-ybins_width/2.0,
                                 all_bin_cens[1][-1]+ybins_width/2.0,
                                 len(all_bin_cens[1])+1)

        '''
        # Now make a contour plot with the levels specified,
        # and with the colormap generated automatically from a list
        # of colors.
        origin = 'lower'
        levels = [0, 10, 30, 100]
        CS3 = plt.contourf(all_bin_cens[0],
                           all_bin_cens[1],
                           metric_vals.T,
                           levels,
                           colors=('r', 'g', 'b'),
                           origin=origin,
                           extend='both')
        CS3.cmap.set_under('yellow')
        CS3.cmap.set_over('cyan')
        CS4 = plt.contour(all_bin_cens[0],
                          all_bin_cens[1],
                          metric_vals.T,
                          levels,
                          colors=('k',),
                          linewidths=(3,),
                          origin=origin)
        plt.clabel(CS4, fmt='%2.1f', colors='w', fontsize=14)
        plt.colorbar(CS3).set_label(label=make_pretty(metric_name),size=36)
        plt.xlim([min(xbin_edges),max(xbin_edges)])
        plt.ylim([min(ybin_edges),max(ybin_edges)])
        plt.xlabel(make_pretty(all_bin_names[0]))
        plt.ylabel(make_pretty(all_bin_names[1]))
        plt.savefig(
            '%s_%s_2D_%s_scan.png'%(all_bin_names[0],
                                    all_bin_names[1],
                                    metric_name)
        )
        '''
  
        plot_2D_hists(
            data=data,
            xbins=xbin_edges,
            ybins=ybin_edges,
            detector=args.detector,
            selection=args.selection,
            xlabel=all_bin_names[0],
            xunits=all_bin_units[0],
            ylabel=all_bin_names[1],
            yunits=all_bin_units[1],
            metric_name=metric_name,
            outdir=args.outdir
        )
                

    else:
        raise NotImplementedError(
            'Only one-dimensional and two-dimensional parameter scan processing'
            ' has been implemented. Datafile contains a %i-dimensional '
            'parameter scan.'%len(all_bin_cens)
        )

if __name__ == '__main__':
    main()
