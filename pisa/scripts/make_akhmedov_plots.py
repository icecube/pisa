#!/usr/bin/env python
# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    October 16, 2016
"""
Create Akhmedov-style plots showing the significance to distinguish the two
hypotheses specified by h0 and h1. Current output will be a plot in the style
of (h1-h0) / sqrt(h0)
"""


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    plt.rcParams['text.usetex'] = True
except:
    print "Could not use tex"
import numpy as np

from pisa.analysis.hypo_testing import HypoTesting
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.tests import baseplot


__all__ = ['do_akhmedov', 'parse_args', 'normcheckpath', 'main']


def do_akhmedov(h0_map, h0_name, h1_map, h1_name, fulltitle,
                savename, outdir, ftype='png'):

    gridspec_kw = dict(left=0.04, right=0.966, wspace=0.32)
    fig, axes = plt.subplots(nrows=1, ncols=3, gridspec_kw=gridspec_kw,
                             sharex=False, sharey=False, figsize=(15,5))

    akhmedov_to_plot = {}
    akhmedov_to_plot['ebins'] = h0_map['ebins']
    akhmedov_to_plot['czbins'] = h0_map['czbins']
    akhmedov_to_plot['map'] = ((h1_map['map']-h0_map['map'])
                               / np.sqrt(h0_map['map']))

    akhmedovlabel = r'$\left(N_{\mathrm{%s}}-N_{\mathrm{%s}}\right)/\sqrt{N_{\mathrm{%s}}}$'%(h1_name,h0_name,h0_name)

    baseplot(m=h0_map,
             title='hypothesis 0 = %s'%h0_name,
             evtrate=True,
             ax=axes[0],
             vmax = max(np.nanmax(h0_map['map']),np.nanmax(h1_map['map'])))
    baseplot(m=h1_map,
             title='hypothesis 1 = %s'%h1_name,
             evtrate=True,
             ax=axes[1],
             vmax = max(np.nanmax(h0_map['map']),np.nanmax(h1_map['map'])))
    baseplot(m=akhmedov_to_plot,
             title='Asymmetry Plot',
             symm=True,
             clabel=akhmedovlabel,
             ax=axes[2])
    plt.subplots_adjust(bottom=0.12,top=0.8)
    plt.suptitle(fulltitle,size='xx-large')
    fig.savefig(outdir+'/%s_%s_%s_Akhmedov.png'%(savename,h0_name,h1_name))
    plt.close(fig.number)


def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
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
        type=str,default='',
        help="Name of detector to put in histogram titles"
    )
    parser.add_argument(
        '--selection',
        type=str,default='',
        help="Name of selection to put in histogram titles"
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
                      %(path, normpath, kind))
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

    # Instantiate the analysis object
    hypo_testing = HypoTesting(**init_args_d)

    h0_maker =  hypo_testing.h0_maker
    h0_maker.select_params(init_args_d['h0_param_selections'])
    h0_maps = h0_maker.get_outputs(return_sum=True)

    h1_maker = hypo_testing.h1_maker
    h1_maker.select_params(init_args_d['h1_param_selections'])
    h1_maps = h1_maker.get_outputs(return_sum=True)

    if 'trck' in ''.join(h0_maps.names):

        h0_trck_map = h0_maps.combine_wildcard('*_trck')
        h1_trck_map = h1_maps.combine_wildcard('*_trck')
        h0_cscd_map = h0_maps.combine_wildcard('*_cscd')
        h1_cscd_map = h1_maps.combine_wildcard('*_cscd')

        h0_trck_to_plot = {}
        h0_trck_to_plot['ebins'] = \
            h0_trck_map.binning['reco_energy'].bin_edges.magnitude
        h0_trck_to_plot['czbins'] = \
        h0_trck_map.binning['reco_coszen'].bin_edges.magnitude
        h0_trck_to_plot['map'] = h0_trck_map.hist

        h1_trck_to_plot = {}
        h1_trck_to_plot['ebins'] = \
            h0_trck_map.binning['reco_energy'].bin_edges.magnitude
        h1_trck_to_plot['czbins'] = \
            h0_trck_map.binning['reco_coszen'].bin_edges.magnitude
        h1_trck_to_plot['map'] = h1_trck_map.hist

        h0_cscd_to_plot = {}
        h0_cscd_to_plot['ebins'] = \
            h0_cscd_map.binning['reco_energy'].bin_edges.magnitude
        h0_cscd_to_plot['czbins'] = \
        h0_cscd_map.binning['reco_coszen'].bin_edges.magnitude
        h0_cscd_to_plot['map'] = h0_cscd_map.hist

        h1_cscd_to_plot = {}
        h1_cscd_to_plot['ebins'] = \
            h0_cscd_map.binning['reco_energy'].bin_edges.magnitude
        h1_cscd_to_plot['czbins'] = \
            h0_cscd_map.binning['reco_coszen'].bin_edges.magnitude
        h1_cscd_to_plot['map'] = h1_cscd_map.hist

    if 'evts' in ''.join(h0_maps.names):

        h0_map = h0_maps['evts']
        h0_map.set_errors(error_hist=None)

        h1_map = h1_maps['evts']
        h1_map.set_errors(error_hist=None)
	
	h0_trck_to_plot = {}
        h0_trck_to_plot['ebins'] = \
            h0_map.binning['reco_energy'].bin_edges.magnitude
        h0_trck_to_plot['czbins'] = \
            h0_map.binning['reco_coszen'].bin_edges.magnitude
        h0_trck_to_plot['map'] = h0_map.hist[:,:,1]

        h1_trck_to_plot = {}
        h1_trck_to_plot['ebins'] = \
            h0_map.binning['reco_energy'].bin_edges.magnitude
        h1_trck_to_plot['czbins'] = \
            h0_map.binning['reco_coszen'].bin_edges.magnitude
        h1_trck_to_plot['map'] = h1_map.hist[:,:,1]

        h0_cscd_to_plot = {}
        h0_cscd_to_plot['ebins'] = \
            h0_map.binning['reco_energy'].bin_edges.magnitude
        h0_cscd_to_plot['czbins'] = \
            h0_map.binning['reco_coszen'].bin_edges.magnitude
        h0_cscd_to_plot['map'] = h0_map.hist[:,:,0]

        h1_cscd_to_plot = {}
        h1_cscd_to_plot['ebins'] = \
            h0_map.binning['reco_energy'].bin_edges.magnitude
        h1_cscd_to_plot['czbins'] = \
            h0_map.binning['reco_coszen'].bin_edges.magnitude
        h1_cscd_to_plot['map'] = h1_map.hist[:,:,0]

    do_akhmedov(h0_map=h0_trck_to_plot,
                h1_map=h1_trck_to_plot,
                h0_name='%s'%args.h0_name,
                h1_name='%s'%args.h1_name,
                fulltitle='%s %s Events Identified as Track'%(detector, selection),
                savename='trck',
                outdir=args.logdir)

    do_akhmedov(h0_map=h0_cscd_to_plot,
                h1_map=h1_cscd_to_plot,
                h0_name='%s'%args.h0_name,
                h1_name='%s'%args.h1_name,
                fulltitle='%s %s Events Identified as Cascade'%(detector, selection),
                savename='cscd',
                outdir=args.logdir)


if __name__ == '__main__':
    main()
