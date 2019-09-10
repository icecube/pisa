#!/usr/bin/env python

"""
Create asymmetry plots (aka Akhmedov**-style plots) showing the significance to
distinguish the two hypotheses specified by h0 and h1. Current output will be a
plot in the style of (h1-h0) / sqrt(h0)

------
**E.K. Akhmedov et al., JHEP 02, 082 (2013), figure 5.

"""


from __future__ import absolute_import, division

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fnmatch import fnmatch
import os

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np

from pisa.analysis.hypo_testing import HypoTesting
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map
from pisa.utils.fileio import mkdir
from pisa.utils.format import text2tex
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource


__all__ = ['plot_asymmetry', 'parse_args', 'normcheckpath', 'main']

__author__ = 'J.L. Lanfranchi, P. Eller, S. Wren'

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


def plot_asymmetry(h0_map, h0_name, h1_map, h1_name, fulltitle, savename,
                   outdir, atype='asymmetry', ftype='pdf'):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['mathtext.fontset'] = 'stixsans'

    gridspec_kw = dict(left=0.04, right=0.966, wspace=0.32)
    fig, axes = plt.subplots(nrows=1, ncols=3, gridspec_kw=gridspec_kw,
                             sharex=False, sharey=False, figsize=(15, 5))

    if atype == 'asymmetry':
        asymmetry_hist = (h1_map.hist-h0_map.hist) / np.sqrt(h0_map.hist)
    elif atype == 'difference':
        asymmetry_hist = (h1_map.hist-h0_map.hist)
    elif atype == 'fraction':
        asymmetry_hist = (h1_map.hist-h0_map.hist) / h0_map.hist
    elif atype == 'percentage':
        asymmetry_hist = (h1_map.hist-h0_map.hist) / h0_map.hist * 100.0
        
    asymmetry_to_plot = Map(
        name=atype,
        hist=asymmetry_hist,
        binning=h0_map.binning
    )

    asymmetrylabel = (r'$\left(N_{%s}-N_{%s}\right)'
                      r'/\sqrt{N_{%s}}$'
                      % (text2tex(h1_name), text2tex(h0_name),
                         text2tex(h0_name)))

    vmax = max(np.nanmax(h0_map.hist), np.nanmax(h1_map.hist))

    h0_map.plot(
        fig=fig,
        ax=axes[0],
        title='Hypothesis 0: $%s$' % text2tex(h0_name),
        cmap=plt.cm.afmhot,
        vmax=vmax
    )

    h1_map.plot(
        fig=fig,
        ax=axes[1],
        title='Hypothesis 1: $%s$' % text2tex(h1_name),
        cmap=plt.cm.afmhot,
        vmax=vmax
    )

    asymmetry_to_plot.plot(
        fig=fig,
        ax=axes[2],
        title='%s'%atype,
        symm=True,
        cmap=plt.cm.seismic
    )

    plt.subplots_adjust(bottom=0.12, top=0.8)
    plt.suptitle(fulltitle, size='xx-large')
    if savename != '' and savename[-1] != '_':
        savename += '_'
    fname = '%s%s_%s_%s.pdf' % (savename, h0_name, h1_name, atype)
    fname = fname.replace(' ', '_')
    mkdir(outdir, warn=False)
    fig.savefig(os.path.join(outdir, fname))
    plt.close(fig.number)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '-d', '--logdir', required=True,
        metavar='DIR', type=str,
        help='Directory into which to store results and metadata.'
    )
    parser.add_argument(
        '--h0-pipeline', required=True,
        type=str, action='append', metavar='PIPELINE_CFG',
        help='''Settings for the generation of hypothesis h0
        distributions; repeat this argument to specify multiple pipelines.'''
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
        help='''Name for hypothesis h0. E.g., "NO" for normal ordering in the
        neutrino mass ordering analysis. Note that the name here has no bearing
        on the actual process, so it's important that you be careful to use a
        name that appropriately identifies the hypothesis.'''
    )
    parser.add_argument(
        '--h1-pipeline',
        type=str, action='append', default=None, metavar='PIPELINE_CFG',
        help='''Settings for the generation of hypothesis h1 distributions;
        repeat this argument to specify multiple pipelines. If omitted, the
        same settings as specified for --h0-pipeline are used to generate
        hypothesis h1 distributions (and so you have to use the
        --h1-param-selections argument to generate a hypotheses distinct from
        hypothesis h0 but still use h0's distribution maker).'''
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
        help='''Name for hypothesis h1. E.g., "IO" for inverted ordering in the
        neutrino mass ordering analysis. Note that the name here has no bearing
        on the actual process, so it's important that you be careful to use a
        name that appropriately identifies the hypothesis.'''
    )
    parser.add_argument(
        '--detector',
        type=str, default='',
        help='Name of detector to put in histogram titles'
    )
    parser.add_argument(
        '--selection',
        type=str, default='',
        help='Name of selection to put in histogram titles'
    )
    parser.add_argument(
        '--atype',
        type=str, default='asymmetry',
        help='''Type of asymmetry to plot. Choices are: asymmetry,
        difference, fraction and percentage. Asymmetry (default) is
        the Akhmedov (h1-h0)/sqrt(h0). Difference is (h1-h0). Fraction
        is (h1-h0)/h0 and percentage is fraction * 100.0.'''
    )
    parser.add_argument(
        '--return-bits',
        action='store_true',
        help='''Flag this if you want the asymmetry plots to be between
        the separated outputs of the config files. If this is not flagged
        then the outputs will be summed together in to a 'total' map.
        If PID is used as a stage rather than a binning dimension then
        this should be flagged.'''
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
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    return parser.parse_args()


# TODO: make this work with Python package resources, not merely absolute
# paths! ... e.g. hash on the file or somesuch?
def normcheckpath(path, checkdir=False):
    normpath = find_resource(path)
    if checkdir:
        kind = 'dir'
        check = os.path.isdir
    else:
        kind = 'file'
        check = os.path.isfile

    if not check(normpath):
        raise IOError('Path "%s" which resolves to "%s" is not a %s.'
                      % (path, normpath, kind))
    return normpath


def main():
    args = parse_args()
    init_args_d = vars(args)

    # NOTE: Removing extraneous args that won't get passed to instantiate the
    # HypoTesting object via dictionary's `pop()` method.

    set_verbosity(init_args_d.pop('v'))

    detector = init_args_d.pop('detector')
    selection = init_args_d.pop('selection')
    atype = init_args_d.pop('atype')
    return_total = not init_args_d.pop('return_bits')

    # Normalize and convert `*_pipeline` filenames; store to `*_maker`
    # (which is argument naming convention that HypoTesting init accepts).
    for maker in ['h0', 'h1']:
        filenames = init_args_d.pop(maker + '_pipeline')
        if filenames is not None:
            filenames = sorted(
                [normcheckpath(fname) for fname in filenames]
            )
        init_args_d[maker + '_maker'] = filenames

        ps_name = maker + '_param_selections'
        ps_str = init_args_d[ps_name]
        if ps_str is None:
            ps_list = None
        else:
            ps_list = [x.strip().lower() for x in ps_str.split(',')]
        init_args_d[ps_name] = ps_list

    # Add dummies to the argument we don't care about for making these plots
    init_args_d['minimizer_settings'] = {}
    init_args_d['data_is_data'] = None
    init_args_d['fluctuate_data'] = None
    init_args_d['fluctuate_fid'] = None
    init_args_d['metric'] = 'chi2'

    if init_args_d['h1_maker'] is None:
        init_args_d['h1_maker'] = init_args_d['h0_maker']

    init_args_d['h0_maker'] = DistributionMaker(init_args_d['h0_maker'])
    init_args_d['h1_maker'] = DistributionMaker(init_args_d['h1_maker'])
    init_args_d['h1_maker'].select_params(init_args_d['h1_param_selections'])

    # Instantiate the analysis object
    hypo_testing = HypoTesting(**init_args_d)

    h0_maker = hypo_testing.h0_maker
    h0_maker.select_params(init_args_d['h0_param_selections'])
    for h0_pipeline in h0_maker.pipelines:
        # Need a special case where PID is a separate stage
        if 'pid' in h0_pipeline.stage_names:
            if return_total:
                raise ValueError(
                    "PID is a separate stage but you have requested"
                    " return_total in the arguments to this script."
                )
            return_h0_sum = False
        else:
            return_h0_sum = return_total
    h0_maps = h0_maker.get_outputs(return_sum=return_h0_sum)

    # Assume just a singular pipeline used here.
    # Not sure how else to deal with PID as a separate stage.
    if not return_h0_sum:
        h0_maps = h0_maps[0]

    h1_maker = hypo_testing.h1_maker
    h1_maker.select_params(init_args_d['h1_param_selections'])
    for h1_pipeline in h1_maker.pipelines:
        # Need a special case where PID is a separate stage
        if 'pid' in h1_pipeline.stage_names:
            if return_total:
                raise ValueError(
                    "PID is a separate stage but you have requested"
                    " return_total in the arguments to this script."
                )
            return_h1_sum = False
        else:
            return_h1_sum = return_total
    h1_maps = h1_maker.get_outputs(return_sum=return_h1_sum)

    # Assume just a singular pipeline used here.
    # Not sure how else to deal with PID as a separate stage.
    if not return_h1_sum:
        h1_maps = h1_maps[0]

    if not sorted(h0_maps.names) == sorted(h1_maps.names):
        raise ValueError(
            "The output names of your h0 and h1 pipelines "
            "do not agree - %s and %s."%(
                sorted(h0_maps.names), sorted(h1_maps.names)
            )
        )

    det_sel = []
    if detector.strip() != '':
        det_sel.append(detector.strip())
    if selection.strip() != '':
        det_sel.append(selection.strip())
    det_sel_label = ' '.join(det_sel)

    det_sel_plot_label = det_sel_label
    if det_sel_plot_label != '':
        det_sel_plot_label += ', '

    det_sel_file_label = det_sel_label
    if det_sel_file_label != '':
        det_sel_file_label += '_'
    det_sel_file_label = det_sel_file_label.replace(' ', '_')

    # Need a special case where PID is a separate stage
    if fnmatch(''.join(h0_maps.names), '*_tr*ck*'):

        h0_trck_map = h0_maps.combine_wildcard('*_tr*ck')
        h1_trck_map = h1_maps.combine_wildcard('*_tr*ck')
        h0_cscd_map = h0_maps.combine_wildcard('*_c*sc*d*')
        h1_cscd_map = h1_maps.combine_wildcard('*_c*sc*d*')

        plot_asymmetry(
            h0_map=h0_trck_map,
            h1_map=h1_trck_map,
            h0_name='%s' % args.h0_name,
            h1_name='%s' % args.h1_name,
            fulltitle='%sevents identified as track' % det_sel_plot_label,
            savename='%strck' % det_sel_file_label,
            outdir=args.logdir,
            atype=atype
        )

        plot_asymmetry(
            h0_map=h0_cscd_map,
            h1_map=h1_cscd_map,
            h0_name='%s' % args.h0_name,
            h1_name='%s' % args.h1_name,
            fulltitle=('%sevents identified as cascade'
                       % det_sel_plot_label),
            savename='%scscd' % det_sel_file_label,
            outdir=args.logdir,
            atype=atype
        )

    # Otherwise, PID is assumed to be a binning dimension
    elif 'pid' in h0_maps[h0_maps.names[0]].binning.names:

        for map_name in h0_maps.names:
            h0_map = h0_maps[map_name]
            h0_map.set_errors(error_hist=None)

            h1_map = h1_maps[map_name]
            h1_map.set_errors(error_hist=None)

            pid_names = h0_map.binning['pid'].bin_names
            if pid_names != h1_map.binning['pid'].bin_names:
                raise ValueError(
                    "h0 and h1 maps must have same PID bin names"
                    " in order to make the asymmetry plots"
                )
            if pid_names is None:
                logging.warning(
                    "There are no names given for the PID bins, thus "
                    "they will just be numbered in both the the plot "
                    "save names and titles."
                )
                pid_names = [
                    x for x in range(0, h0_map.binning['pid'].num_bins)
                ]

            for pid_name in pid_names:
                    
                h0_to_plot = h0_map.split(
                    dim='pid',
                    bin=pid_name
                )
                
                h1_to_plot = h1_map.split(
                    dim='pid',
                    bin=pid_name
                )

                if isinstance(pid_name, int):
                    pid_name = 'PID Bin %i' % (pid_name)

                plot_asymmetry(
                    h0_map=h0_to_plot,
                    h1_map=h1_to_plot,
                    h0_name='%s' % args.h0_name,
                    h1_name='%s' % args.h1_name,
                    fulltitle=('%sevents identified as %s'
                               % (det_sel_plot_label, pid_name)),
                    savename=('%s_%s%s' % (map_name,
                                           det_sel_file_label,
                                           pid_name)),
                    outdir=args.logdir,
                    atype=atype
                )

    else:

        for map_name in h0_maps.names:
            
            h0_map = h0_maps[map_name]
            h0_map.set_errors(error_hist=None)
            
            h1_map = h1_maps[map_name]
            h1_map.set_errors(error_hist=None)

            plot_asymmetry(
                h0_map=h0_map,
                h1_map=h1_map,
                h0_name='%s' % args.h0_name,
                h1_name='%s' % args.h1_name,
                fulltitle=('%sevents'%(det_sel_plot_label)),
                savename=('%s_%s' % (map_name, det_sel_file_label)),
                outdir=args.logdir,
                atype=atype
            )


if __name__ == '__main__':
    main()
