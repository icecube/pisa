#!/usr/bin/env python
"""
From a MapSet json file, create a "standard" CFX plot.
"""

import os
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
from uncertainties import unumpy as unp
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

from pisa.core.map import MapSet
from pisa.core.binning import MultiDimBinning
from pisa.utils import fileio
from pisa.utils.log import logging, set_verbosity

class FullPaths(argparse.Action):
    """
    Append user- and relative-paths
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

def is_dir(dirname):
    """
    Checks if a path is an actual directory, if not found then it creates the
    directory
    """
    if not os.path.isdir(dirname):
        fileio.mkdir(dirname)
    return dirname

def is_valid_file(filepath):
    """
    Checks if a path is an actual file
    """
    if not os.path.exists(filepath):
        msg = 'The file {0} does not exist!'.format(filepath)
        raise argparse.ArgumentError(msg)
    else:
        return filepath

def parse_args():
    """Get command line arguments"""
    parser = ArgumentParser(
        description='''Takes an outputted json file containing a MapSet
        object with a single Map in it and makes a plot of it.''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-i', '--infile', type=is_valid_file, action=FullPaths, metavar='FILE',
        help='location/filename of the json file containing the MapSet'
    )

    parser.add_argument(
        '-o', '--outdir', type=is_dir, action=FullPaths, metavar='DIR',
        default='$PISA/pisa/images/cfx/',
        help='location onto which to store the plot'
    )

    parser.add_argument(
        '-n', '--outname', metavar='FILE', type=str,
        default='untitled.png', help='output filename'
    )

    parser.add_argument(
        '--logy', default=True, action='store_true',
        help='flag to specifiy whether to use a log scale on the y axis'
    )

    parser.add_argument(
        '--ylim', type=float, default=None, nargs=2, metavar=('YMIN', 'YMAX'),
        help='set the limits of the y axis'
    )

    parser.add_argument(
        '--ylabel', type=str, default=None, metavar='STR',
        help='set the label of the y axis'
    )

    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='set verbosity level'
    )

    args = parser.parse_args()
    return args

def plot_CFX_one(map, outfile, logy=False, ylim=None, ylabel=None):
    """Plot all maps in one plot."""
    logging.info('Plotting Map {0}'.format(map.name))
    bins = map.binning
    try:
        coszen_binning = bins.coszen
        energy_binning = bins.energy
    except:
        try:
            coszen_binning = bins.reco_coszen
            energy_binning = bins.reco_energy
        except:
            raise ValueError('energy/coszen bins not found')
    map.reorder_dimensions(
        MultiDimBinning([coszen_binning, energy_binning])
    )

    def add(x, y):
        return str(x) + ' - ' + str(y)
    coszen_str = [add(coszen_binning.bin_edges[i].m,
                      coszen_binning.bin_edges[i+1].m)
                  for i in range(coszen_binning.num_bins)]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    if energy_binning.is_log:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlim(np.min(energy_binning.bin_edges.m),
                np.max(energy_binning.bin_edges.m))

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_xlabel(energy_binning.tex, fontsize=18)
    if ylabel is not None:
        ylabel = r'{0}'.format(ylabel)
        ax.set_ylabel(ylabel, fontsize=18)
    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(y=ymaj, ls=':', color='gray', alpha=0.7, linewidth=1)
    for xmaj in ax.xaxis.get_majorticklocs():
        ax.axvline(x=xmaj, ls=':', color='gray', alpha=0.7, linewidth=1)

    colour_cycle = plt.cm.RdYlBu(np.linspace(0, 1, coszen_binning.num_bins))
    for iZen in xrange(coszen_binning.num_bins):
        hist = map.hist[iZen, :]
        hist_0 = np.concatenate([[hist[0]], hist])
        colour = colour_cycle[iZen]
        ax.step(
            energy_binning.bin_edges.m, unp.nominal_values(hist_0),
            alpha=1, drawstyle='steps-pre', label=coszen_str[iZen],
            linewidth=2, linestyle='-', color=colour
        )
        ax.errorbar(
            energy_binning.weighted_centers.m, unp.nominal_values(hist),
            color=colour, xerr=0, yerr=unp.std_devs(hist), capsize=1.3,
            alpha=1, linestyle='None', markersize=2, linewidth=2
        )
    legend = ax.legend(title=coszen_binning.tex, prop=dict(size=12))
    plt.setp(legend.get_title(), fontsize=18)
    at = AnchoredText(r'$%s$' % map.tex, prop=dict(size=20), frameon=True,
                      loc=2)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.5")
    ax.add_artist(at)
    fig.savefig(outfile, bbox_inches='tight', dpi=150)

if __name__ == "__main__":
    args = parse_args()
    set_verbosity(args.verbose)

    logging.info('Loading Map from file {0}'.format(args.infile))
    input_MapSet = MapSet.from_json(args.infile)
    assert len(input_MapSet) == 1
    input_Map = input_MapSet.pop()

    outfile = args.outdir + '/' + args.outname
    logging.info('outfile {0}'.format(outfile))
    plot_CFX_one(
        map = input_Map,
        outfile = outfile,
        logy = args.logy,
        ylim = args.ylim,
        ylabel = args.ylabel
    )
