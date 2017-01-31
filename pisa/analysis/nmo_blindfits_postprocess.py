#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Will process the output of the blind fits for the NMO analysis

"""

from __future__ import division

from argparse import ArgumentParser
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import numpy as np
import re

from scipy.stats import chi2

from pisa.analysis.hypo_testing import Labels
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.utils.fileio import from_file, to_file, nsort
from pisa.utils.log import set_verbosity, logging
from pisa.utils.postprocess import tex_axis_label, parse_binning_string


__all__ = ['extract_trials', 'extract_fit', 'parse_args', 'main']


def make_binned_chi2_plots(maps_binned, metric_name, total_metric,
                           num_fit_params, num_bound_params,
                           outdir, detector, selection):
    '''
    Takes the binned chi2 maps and plots them for inspection. Tests to see if 
    PathEffects can be imported and, if so, writes the values on the bins. A 
    histogram is also made of the individual chi2 values, on to which the 
    p-value is written for inspection. 
    '''
    outdir += 'BinnedChi2Maps'
    if not os.path.exists(outdir):
        logging.info('Making output directory %s'%outdir)
        os.makedirs(outdir)
    try:
        import matplotlib.patheffects as PathEffects
        logging.warn("PathEffects could be imported, so the chi2 values"
                     " will be written on the bins. This is slow.")
        pe = True
    except:
        logging.warn("PathEffects could not be imported, so the correlation" 
                     " values will not be written on the bins.")
        pe = False
    all_metric_vals = np.array([])
    for map_name in maps_binned.keys():
        if 'MultiDimBinning' in maps_binned[map_name]['binning']:
            binning_strings = re.findall(
                'OneDimBinning\([^)]*\)',
                maps_binned[map_name]['binning'].split('MultiDimBinning')[-1]
            )
            binning = []
            for binning_string in binning_strings:
                binning_dict = parse_binning_string(
                    binning_string = binning_string
                )
                binning.append(OneDimBinning(**binning_dict))
            binning = MultiDimBinning(binning)
        else:
            binning_dict = parse_binning_string(
                binning_string = maps_binned[map_name]['binning']
            )
            binning = OneDimBinning(**binning_dict)
        total_map = Map(
            name = map_name,
            hist = maps_binned[map_name]['hist'],
            binning = binning
        )
        all_metric_vals = np.concatenate(
            [all_metric_vals,
             maps_binned[map_name]['hist'].flatten()]
        )
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,6.5))
        pid_names = total_map.binning['pid'].bin_names
        if pid_names is None:
            logging.warn('There are no names given for the PID bins, thus '
                         'they will just be numbered in both the the plot '
                         'save names and titles.')
            pid_names = [x for x in range(0, total_map.binning['pid'].num_bins)]
        for i,pid_name in enumerate(pid_names):
            map_to_plot = total_map.split(
                dim='pid',
                bin=pid_name
            )
            map_to_plot = map_to_plot.reorder_dimensions(
                order = ['reco_coszen','reco_energy']
            )
            map_to_plot.plot(
                fig=fig,
                ax=axes[i],
                title='PID Bin %i'%i,
                cmap=plt.cm.afmhot,
            )
            if pe:
                for j in range(0,len(map_to_plot.hist)):
                    for k in range(0, len(map_to_plot.hist[0])):
                        axes[i].text(
                            map_to_plot.binning[
                                'reco_coszen'].bin_edges[j].magnitude + \
                            ((map_to_plot.binning[
                                'reco_coszen'].bin_edges[j+1].magnitude - \
                              map_to_plot.binning[
                                  'reco_coszen'].bin_edges[j].magnitude)/2),
                            np.log10(map_to_plot.binning[
                                'reco_energy'].bin_edges[k].magnitude) + \
                            ((np.log10(map_to_plot.binning[
                                'reco_energy'].bin_edges[k+1].magnitude) - \
                              np.log10(map_to_plot.binning[
                                  'reco_energy'].bin_edges[k].magnitude))/2),
                            '%.2f'%map_to_plot.hist[j][k],
                            fontsize='12',
                            verticalalignment='center',
                            horizontalalignment='center',
                            color='w',
                            path_effects=[
                                PathEffects.withStroke(
                                    linewidth=2.5,
                                    foreground='k'
                                )
                            ]
                        )
        MainTitle = '%s %s Blind Fits'%(detector, selection)
        SubTitle = 'Bin-by-Bin Contribution to %s'%tex_axis_label(metric_name)
        plt.suptitle(
            MainTitle + r'\\' + SubTitle,
            fontsize=36
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.75)
        SaveName = '%s_%s_map_%s_binned_%s_maps.png'%(
            detector,
            selection,
            total_map.name,
            metric_name
        )
        fig.savefig(os.path.join(outdir,SaveName))
        plt.close()

    num_free_params = num_fit_params - num_bound_params
    pvalue = 1 - chi2.cdf(
        total_metric,
        len(all_metric_vals)-num_free_params
    )

    plt.hist(all_metric_vals, bins=10)
    plt.xlabel(tex_axis_label(metric_name))
    plt.ylabel('Number of Instances')
    plt.title(MainTitle + ' ' + SubTitle, fontsize=16)
    # Write the total chi2 on the plot
    plt.figtext(
        0.35,
        0.80,
        r"Total %s = %.2f"%(tex_axis_label(metric_name),total_metric),
        color='k',
        size='xx-large'
    )
    plt.figtext(
        0.35,
        0.73,
        r"Number of bins = %i"%len(all_metric_vals),
        color='k',
        size='xx-large'
    )
    plt.figtext(
        0.35,
        0.66,
        r"Number of fit params = %i (%i free)"%(num_fit_params,num_free_params),
        color='k',
        size='xx-large'
    )
    plt.figtext(
        0.35,
        0.59,
        r"p-value = {:.2f}%".format(pvalue*100.0),
        color='k',
        size='xx-large'
    )
    SaveName = '%s_%s_binned_%s_total_hist.png'%(
        detector,
        selection,
        metric_name
    )
    plt.savefig(os.path.join(outdir,SaveName))
    plt.close()

    
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '-d', '--dir', required=True,
        metavar='DIR', type=str,
        help="""Directory into which the output of nmo_blindfits.py was 
        stored."""
    )
    parser.add_argument(
        '--detector',type=str,default='',
        help="""Name of detector to put in histogram titles."""
    )
    parser.add_argument(
        '--selection',type=str,default='',
        help="""Name of selection to put in histogram titles."""
    )
    parser.add_argument(
        '--bound-params', type=int, default=0,
        help="""The number of parameters for which a prior was applied. This
        will be subtracted from the total number of fit parameters when
        calculating an approximation of the degrees of freedom in the problem
        from which the goodness-of-fit will be calculated."""
    )
    parser.add_argument(
        '--outdir', metavar='DIR', type=str, required=True,
        help="""Store all output plots to this directory. This will make
        further subdirectories, if needed, to organise the output plots."""
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help="""set verbosity level"""
    )
    return parser.parse_args()


def main():
    
    args = parse_args()
    init_args_d = vars(args)

    # NOTE: Removing extraneous args that won't get passed to instantiate the
    # HypoTesting object via dictionary's `pop()` method.

    set_verbosity(init_args_d.pop('v'))

    detector = init_args_d.pop('detector')
    selection = init_args_d.pop('selection')
    outdir = init_args_d.pop('outdir')

    logdir_content = os.listdir(args.dir)
    blind_fit_result = from_file(os.path.join(args.dir,logdir_content[0]))
    metric_name = blind_fit_result['metric']
    total_metric = blind_fit_result['metric_val']
    num_fit_params = len(blind_fit_result['params'])
    num_bound_params = init_args_d.pop('bound_params')

    make_binned_chi2_plots(
        maps_binned = blind_fit_result['detailed_metric_info'][metric_name][
            'maps_binned'],
        metric_name=metric_name,
        total_metric=total_metric,
        num_fit_params=num_fit_params,
        num_bound_params=num_bound_params,
        outdir=outdir,
        detector=detector,
        selection=selection
    )
                
        
if __name__ == '__main__':
    main()
