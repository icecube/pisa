#!/usr/bin/env python
# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    October 16, 2016
"""
Create plots showing the effect of shifting a systematic by 10% or +/- 1 sigma
if such a definition is possible. This will be shown as 2D maps per PID bin.
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
from pisa.utils.postprocess import tex_axis_label


__all__ = ['plot_asymmetry', 'parse_args', 'normcheckpath', 'main']

__author__ = 'S. Wren'

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


def plot_variation(baseline_maps, up_maps, down_maps,
                   h1_name, fulltitle, savename,
                   outdir, ftype='pdf'):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['mathtext.fontset'] = 'stixsans'

    gridspec_kw = dict(left=0.04, right=0.966, wspace=0.32)
    fig, axes = plt.subplots(nrows=1, ncols=3, gridspec_kw=gridspec_kw,
                             sharex=False, sharey=False, figsize=(15, 5))

    asymmetry_hist = (h1_map.hist-h0_map.hist) / np.sqrt(h0_map.hist)
    asymmetry_to_plot = Map(
        name='asymmetry',
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
        title='Asymmetry',
        symm=True,
        cmap=plt.cm.seismic
    )

    plt.subplots_adjust(bottom=0.12, top=0.8)
    plt.suptitle(fulltitle, size='xx-large')
    if savename != '' and savename[-1] != '_':
        savename += '_'
    fname = '%s%s_%s_asymmetry.pdf' % (savename, h0_name, h1_name)
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
        '--pipeline', required=True,
        type=str, action='append', metavar='PIPELINE_CFG',
        help='''Settings for the generation of the baseline data distributions;
        repeat this argument to specify multiple pipelines.'''
    )
    parser.add_argument(
        '--param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated (no spaces) list of param selectors to apply to
        the pipelines.'''
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

    # Normalize and convert `*_pipeline` filenames; store to `*_maker`
    # (which is argument naming convention that HypoTesting init accepts).
    filenames = init_args_d.pop('pipeline')
    if filenames is not None:
        filenames = sorted(
            [normcheckpath(fname) for fname in filenames]
        )
    ps_str = init_args_d['param_selections']
    if ps_str is None:
        ps_list = None
    else:
        ps_list = [x.strip().lower() for x in ps_str.split(',')]

    data_maker = DistributionMaker(filenames)
    data_maker.select_params(ps_list)

    for data_pipeline in data_maker.pipelines:
        # Need a special case where PID is a separate stage
        if 'pid' in data_pipeline.stage_names:
            raise ValueError("Special case for separate PID stage currently "
                             "not implemented.")
        else:
            return_sum=True
    baseline_maps = data_maker.get_outputs(return_sum=return_sum)

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

    for data_param in data_maker.params.free:
        # Calculate a shifted value based on the prior if possible
        if hasattr(data_param, 'prior') and (data_param.prior is not None):
            # Gaussian priors are easy - just do 1 sigma
            if data_param.prior.kind == 'gaussian':
                data_param.value = \
                    data_param.value + data_param.prior.stddev
                shift_label = r"$1\sigma$"
            # Else do 10%, or +/- 1 if the baseline is zero
            else:
                if data_param.value != 0.0:
                    data_param.value = 1.1 * data_param.value
                    shift_label = r"10%"
                else:
                    data_param.value = 1.0
                    shift_label = r"1"
        # For no prior also do 10%, or +/- 1 if the baseline is zero
        else:
            if data_param.value != 0.0:
                data_param.value = 1.1 * data_param.value
                shift_label = r"10%"
            else:
                data_param.value = 1.0
                shift_label = r"1"
        up_maps = data_maker.get_outputs(return_sum=return_sum)
        data_maker.params.reset_free()

        if hasattr(data_param, 'prior') and (data_param.prior is not None):
            if data_param.prior.kind == 'gaussian':
                data_param.value = \
                    data_param.value - data_param.prior.stddev
            else:
                if data_param.value != 0.0:
                    data_param.value = 0.9 * data_param.value
                else:
                    data_param.value = -1.0
        else:
            if data_param.value != 0.0:
                data_param.value = 0.9 * data_param.value
            else:
                data_param.value = -1.0
        down_maps = data_maker.get_outputs(return_sum=return_sum)
        data_maker.params.reset_free()

        baseline_map = baseline_maps['total']
        baseline_map.set_errors(error_hist=None)
        up_map = up_maps['total']
        up_map.set_errors(error_hist=None)
        down_map = down_maps['total']
        down_map.set_errors(error_hist=None)

        pid_names = baseline_map.binning['pid'].bin_names
        if pid_names is None:
            logging.warning(
                'There are no names given for the PID bins, thus '
                'they will just be numbered in both the the plot '
                'save names and titles.'
            )
            pid_names = [x for x in range(
                0, baseline_map.binning['pid'].num_bins)]

        gridspec_kw = dict(left=0.04, right=0.966, wspace=0.32)
        fig, axes = plt.subplots(
            nrows=2,
            ncols=len(pid_names),
            gridspec_kw=gridspec_kw,
            sharex=False,
            sharey=False,
            figsize=(7*len(pid_names), 14)
        )

        for i,pid_name in enumerate(pid_names):

            baseline = baseline_map.split(
                dim='pid',
                bin=pid_name
            )
            up_to_plot = up_map.split(
                dim='pid',
                bin=pid_name
            )
            up_to_plot = (up_to_plot - baseline) / baseline * 100.0
            down_to_plot = down_map.split(
                dim='pid',
                bin=pid_name
            )
            down_to_plot = (down_to_plot - baseline) / baseline * 100.0

            if isinstance(pid_name, int):
                pid_name = 'PID Bin %i'%(pid_name)
            else:
                pid_name += ' Channel'

            up_to_plot.plot(
                fig=fig,
                ax=axes[0][i],
                title="%s "%(pid_name)+"\n"+" %s + %s"%(
                    tex_axis_label(data_param.name),
                    shift_label),
                titlesize=30,
                cmap=plt.cm.seismic,
                clabel='% Change from Baseline',
                clabelsize=30,
                xlabelsize=24,
                ylabelsize=24,
                symm=True
            )
            down_to_plot.plot(
                fig=fig,
                ax=axes[1][i],
                title="%s "%(pid_name)+"\n"+" %s - %s"%(
                    tex_axis_label(data_param.name),
                    shift_label),
                titlesize=30,
                cmap=plt.cm.seismic,
                clabel='% Change from Baseline',
                clabelsize=30,
                xlabelsize=24,
                ylabelsize=24,
                symm=True
            )

        fig.subplots_adjust(hspace=0.4)
        savename = det_sel_file_label
        if savename != '' and savename[-1] != '_':
            savename += '_'
        savename += '%s_variation.png'%(data_param.name)
        mkdir(args.logdir, warn=False)
        fig.savefig(os.path.join(args.logdir, savename),bbox_inches='tight')
        plt.close(fig.number)

if __name__ == '__main__':
    main()
