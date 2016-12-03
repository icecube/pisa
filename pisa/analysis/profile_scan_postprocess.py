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
    pretty_labels["Genie_Ma_QE"] = r"GENIE $M_{A}^{QE}$"
    pretty_labels["Genie_Ma_RES"] = r"GENIE $M_{A}^{Res}$"
    pretty_labels["dom_eff"] = r"DOM Efficiency"
    pretty_labels["hole_ice"] = r"Hole Ice"
    pretty_labels["hole_ice_fwd"] = r"Hole Ice Forward"
    pretty_labels["degree"] = r"$^\circ$"
    pretty_labels["radians"] = r"rads"
    pretty_labels["electron_volt ** 2"] = r"$\mathrm{eV}^2$"
    pretty_labels["llh"] = r"Likelihood"
    pretty_labels["chi2"] = r"$\chi^2$"
    pretty_labels["mod_chi2"] = r"Modified $\chi^2$"
    if label not in pretty_labels.keys():
        logging.warn("I don't know what to do with %s. Returning as is."%label)
        return label
    return pretty_labels[label]


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--infile', metavar='FILE', type=str, required=True,
                        help='''Output file of profile_scan.py to processs.''')
    parser.add_argument('--outdir', metavar='DIR', type=str, required=True,
                        help='''Store the output plot to this directory.''')
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
        raise ValueError('Output directory selected does not exist!')

    set_verbosity(args.v)

    infile = from_file(args.infile)

    all_steps = infile['steps']
    all_bin_cens = []
    all_bin_names = all_steps.keys()
    for step_variable in all_steps.keys():
        bin_cens = []
        for val in all_steps[step_variable]:
            if val[0] not in bin_cens:
                bin_cens.append(val[0])
        all_bin_cens.append(bin_cens)

    results = infile['results']
    metric_name = results[0]['metric']
    metric_vals = []
    for result in results:
        metric_vals.append(result['metric_val'])

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
        plt.colorbar(CS3).set_label(label=metric_tex_name(metric_name),size=36)
        plt.xlim([min(xbin_edges),max(xbin_edges)])
        plt.ylim([min(ybin_edges),max(ybin_edges)])
        plt.xlabel(parameter_tex_name(all_bin_names[0]))
        plt.ylabel(parameter_tex_name(all_bin_names[1]))
        plt.savefig(
            '%s_%s_2D_%s_scan.png'%(all_bin_names[0],
                                    all_bin_names[1],
                                    metric_name)
        )

    else:
        raise NotImplementedError(
            'Only one-dimensional and two-dimensional parameter scan processing'
            ' has been implemented. Datafile contains a %i-dimensional '
            'parameter scan.'%len(all_bin_cens)
        )
