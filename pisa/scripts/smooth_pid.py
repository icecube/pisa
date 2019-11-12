#! /usr/bin/env python

"""
One-dimensional PID's are smoothed and sampled, yielding data points meant to
be interpolated between) as functions of energy and cosine-zenith. Due to not
much coszen dependence (this must be verified though by examining produced
plots!), this dependence is not currently used by PISA. Nonetheless, the output
is there if ever it's deemed worthwhile to use.

Note that the default settings seem to work well, at least for PINGU geometries
V36, V38, and V39.
"""

# TODO: make smoothing a function that's called once for E-dep and once for
#       CZ-dep
# TODO: store metadata about how smoothing is done in produced files
# TODO: use CombinedFlavIntData for storage of the results
# TODO: use weights in Blackman-window smoothing algo


import os,sys
from copy import deepcopy
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
from scipy.interpolate import splrep, splev, interp1d

from pisa.core.events import Events
from pisa.utils.log import logging, set_verbosity
from pisa.utils import flavInt
from pisa.utils import jsons
from pisa.utils import utils
from pisa.pid.PIDServiceMC import PIDServiceMC


__all__ = []

__author__ = 'J.L. Lanfranchi'

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


if __name__ == '__main__':
    parser = ArgumentParser(
        '''Smooth PID at energy bin centers. NOTE: at present, uses *ONLY* the
        MC-true-upgoing events, but smoothes in reconstructed energy and
        coszen.''',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--events', metavar='RESOURCE_LOC', type=str, required=True,
        help='''PISA events file resource location. It is expected that, within the
        file, events of flavors/interaction types have been joined together in the
        following groupings: nuecc+nuebarcc, numucc+numubarcc, nutaucc+nutaubarcc,
        and nuall_nc+nuallbar_nc.'''
    )
    parser.add_argument(
        '--pid-ver', metavar='PID_VERSION', type=str, default='1',
        help='''PID version (correspoinding to the events'
        detector/geom/processing) to implement.'''
    )
    parser.add_argument(
        '--outdir', metavar='DIR', type=str, required=True,
        help='Directory into which to place the produced files.'
    )
    parser.add_argument(
        '--emin', metavar='EMIN_GeV', type=float, default=1,
        help='Energy bins\' left-most edge, in GeV'
    )
    parser.add_argument(
        '--emax', metavar='EMAX_GeV', type=float, default=90,
        help='Energy bins\' righ-most edge, in GeV'
    )
    parser.add_argument(
        '--n-ebins', type=int, default=300,
        help='Number of energy bins (logarithmically-spaced)'
    )
    #parser.add_argument(
    #    '--czmin', metavar='COSZEN', type=float,
    #    required=True,
    #    help='Cosine-zenith bins\' lowest edge'
    #)
    #parser.add_argument(
    #    '--czmax', metavar='COSZEN', type=float,
    #    required=True,
    #    help='Cosine-zenithy bins\' highest edge'
    #)
    parser.add_argument(
        '--n-czbins', type=int, default=200,
        help='Number of cosine-zenith bins (linearly-spaced)'
    )
    parser.add_argument(
        '--dependent-sig', metavar='SIGNATURE', type=str, default='trck',
        help='The PID signature that is left as being dependent upon the other(s).'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Do not make debug plots (which are made by default)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count', default=0,
        help='Set verbosity level'
    )
    args = parser.parse_args()

    set_verbosity(args.verbose)

    events_fpath = args.events
    outdir = os.path.expandvars(os.path.expanduser(args.outdir))
    make_plots = not args.no_plots

    # Load the events
    events = Events(events_fpath)

    grouped = sorted([
        flavInt.NuFlavIntGroup(fi)
        for fi in events.metadata['flavints_joined']
    ])
    should_be_grouped = sorted([
        flavInt.NuFlavIntGroup('nue_cc+nuebar_cc'),
        flavInt.NuFlavIntGroup('numu_cc+numubar_cc'),
        flavInt.NuFlavIntGroup('nutau_cc+nutaubar_cc'),
        flavInt.NuFlavIntGroup('nuall_nc+nuallbar_nc'),
    ])
    if grouped != should_be_grouped:
        if len(grouped) == 0:
            grouped = None
        raise ValueError('Only works with groupings (%s) but instead got'
                         ' groupings (%s).' % (should_be_grouped, grouped))

    # Define binning
    emin, emax, n_ebins = args.emin, args.emax, args.n_ebins
    czmin, czmax, n_czbins = -1, +1, args.n_czbins

    ebins = np.logspace(np.log10(emin), np.log10(emax), n_ebins+1)
    czbins = np.linspace(czmin, czmax, n_czbins+1)

    # NOTE: Since energy-dependence smoothing is done in log-space, use
    # logarithmically-centerd points (open for debate whether this is more or less
    # correct than linear midpoints)
    ebin_midpoints = np.sqrt(ebins[:-1]*ebins[1:])
    czbin_midpoints = (czbins[:-1] + czbins[1:])/2.0

    # Bin the PID after removing true-downgoing events, and collapsing the
    # reco-coszen dependence by using a single czbin
    single_czbin = [-1, 0]
    pidmc = PIDServiceMC(ebins=ebins, czbins=single_czbin, events=events,
                         pid_ver=args.pid_ver, remove_true_downgoing=True,
                         compute_error=True, replace_invalid=True)

    # Check dependent_sig arg
    all_sigs = pidmc.pid_spec.get_signatures()
    assert args.dependent_sig in all_sigs
    sigs_to_spline = deepcopy(all_sigs)
    sigs_to_spline.remove(args.dependent_sig)

    if make_plots:
        import matplotlib as mpl
        mpl.use('pdf')
        import matplotlib.pyplot as plt
        from pisa.utils.plot import stepHist

        FIGSIZE = (11,7)
        #plt.close(1)
        plt.figure(1, figsize=FIGSIZE).clf()
        fig, axgrp = plt.subplots(2, 2, sharex=True, num=1, figsize=FIGSIZE)
        axgrp = axgrp.flatten()
        basetitle = (
            'Fraction of events ID\'d, %s geometry %s, MC runs %s with V'
            '%s processing & and PID version %s' %
            (events.metadata['detector'].upper(), events.metadata['geom'].upper(),
             utils.list2hrlist(events.metadata['runs']),
             events.metadata['proc_ver'], args.pid_ver)
        )
        fig.suptitle(basetitle, fontsize=12)

    rel_errors = pidmc.get_rel_error()

    param_pid_maps = {'binning': {'ebins':ebins, 'czbins':czbins}}
    histdata = {}
    smooth_energy_dep = {}
    esmooth_store = {'ebin_midpoints': ebin_midpoints}
    for label, label_data in pidmc.pid_maps.items():
        if label == 'binning':
            continue
        pid_fract_totals = np.zeros(n_ebins)

        histdata[label] = {}
        smooth_energy_dep[label] = {}
        esmooth_store[label] = {}
        for sig in sigs_to_spline:
            histdata[label][sig] = np.squeeze(label_data[sig])

            assert np.sum(np.isinf(np.abs(rel_errors[label]))) == 0
            assert np.sum(np.isnan(rel_errors[label])) == 0
            #abs_err = np.squeeze(rel_errors[label])*histdata[label][sig]
            #rel_errors[label][
            weights = 1./rel_errors[label]
            weights[weights == 0] = 1./np.max(rel_errors[label])
            assert np.sum(np.isinf(np.abs(weights))) == 0
            assert np.sum(np.isnan(weights)) == 0
            #weights = 1./abs_err
            #weights[weights == 0] = 1./np.max(abs_err)

            n_smooth = 60
            smoothing_kernel = np.blackman(n_smooth)
            norm = 1/np.convolve(np.ones(n_ebins), smoothing_kernel, 'same')
            smoothed = np.convolve(histdata[label][sig], smoothing_kernel, 'same') \
                    * norm
            spline = splrep(
                np.log10(ebin_midpoints), smoothed, #histdata[label][sig],
                w=weights,
                k=5, s=0.25,
            )

            # Individually do not allow the fraction to be outside [0, 1]
            #pid_fract_vals = np.clip(splev(np.log10(ebin_midpoints), spline),
            #                         a_min=0, a_max=1)
            pid_fract_vals = np.clip(smoothed, a_min=0, a_max=1)

            pid_fract_totals += pid_fract_vals

            # Store the intermediate results
            smooth_energy_dep[label][sig] = pid_fract_vals

        # If total anywhere is greater than 1, figure out by how much so we can
        # scale each fraction down in the end
        maxtotal = np.max(pid_fract_totals)
        scale_factor = 1
        if maxtotal > 1:
            scale_factor = 1/maxtotal

        # Compute "dependent" signature's PID in terms of the difference between 1
        # and the total of all other PID signatures
        sig = args.dependent_sig
        histdata[label][sig] = np.squeeze(label_data[sig])
        # Apply the scale factor to 1 to avoid negative numbers; in the end we will
        # scale this down again
        pid_fract_vals = 1.0/scale_factor - pid_fract_totals
        smooth_energy_dep[label][sig] = pid_fract_vals

        for sig in all_sigs:
            smooth_energy_dep[label][sig] *= scale_factor
            # Populate datastructure to be written to disk
            esmooth_store[label][sig] = {
                'histo': histdata[label][sig],
                'histo_err': np.squeeze(rel_errors[label])*np.squeeze(histdata[label][sig]),
                'smooth': smooth_energy_dep[label][sig],
            }

        if make_plots:
            if label == 'nue_cc':
                grp = flavInt.NuFlavIntGroup('nuecc+nuebarcc')
                axnum = 0
            elif label == 'numu_cc':
                axnum = 1
                grp = flavInt.NuFlavIntGroup('numucc+numubarcc')
            elif label == 'nutau_cc':
                axnum = 2
                grp = flavInt.NuFlavIntGroup('nutaucc+nutaubarcc')
            else:
                axnum = 3
                grp = flavInt.NuFlavIntGroup('nuallnc+nuallbarnc')
            ax = axgrp[axnum]

            for sig in ['trck']:
                if len(sigs_to_spline) == 1:
                    lab_sfx = ''
                    leg_sfx = r'; id$\Rightarrow$' + sig
                else:
                    lab_sfx = r'; id$\Rightarrow$' + sig
                    leg_sfx = ''

                store = esmooth_store[label][sig]
                stepHist(
                    ebins, y=store['histo'], yerr=store['histo_err'],
                    ax=ax, color=(0.8,0.2,0.6),
                    label='Histogram' + lab_sfx
                )
                # Plot the splined points
                ax.plot(
                    ebin_midpoints, store['smooth'],
                    'k-', lw=1.5, ms=1,
                    label='Smoothed' + lab_sfx
                )

            ax.set_xscale('log')
            ax.set_xlim(1, 100)
            #ylim = np.clip(ax.get_ylim(), 0, 1)
            #ax.set_ylim(ylim)
            #if axnum in [0,2,3]:
            #    ax.set_ylim(0, 0.5)
            ax.set_ylim(0, 1)
            if axnum in [2,3]:
                ax.set_xlabel('Reconstructed energy (GeV)')
            ax.grid(b=True, which='both', ls='-', alpha=0.7, color=[0.5]*3, lw=0.5)

            leg = ax.legend(
                loc='best', frameon=False,
                title=flavInt.tex(grp,d=1) + leg_sfx,
            )
            leg.get_title().set_fontsize(16)


    # Derive output filename
    outfname = (
        'pid_energy_dependence__%s_%s__runs_%s__proc_%s__pid_%s.json' % (
            events.metadata['detector'],
            events.metadata['geom'],
            utils.list2hrlist(events.metadata['runs']),
            events.metadata['proc_ver'],
            args.pid_ver,
        )
    )
    outfpath = os.path.join(outdir, outfname)
    logging.info('Saving PID energy dependence info to file "%s"' % outfpath)
    jsons.to_json(esmooth_store, outfpath)


    if make_plots:
        fig.tight_layout(rect=(0,0,1,0.96))
        basefname = (
            'pid_energy_dependence__%s_%s__runs_%s__proc_%s__pid_%s'
            % (events.metadata['detector'], events.metadata['geom'],
               utils.list2hrlist(events.metadata['runs']),
               events.metadata['proc_ver'], args.pid_ver)
        )
        fig.savefig(os.path.join(outdir, basefname + '.pdf'))
        fig.savefig(os.path.join(outdir, basefname + '.png'))


    #===============================================================================
    # COSZEN
    #===============================================================================
    single_ebin = [1, 80]
    pidmc = PIDServiceMC(ebins=single_ebin, czbins=czbins, events=events,
                         pid_ver=args.pid_ver, remove_true_downgoing=False,
                         compute_error=True, replace_invalid=True)

    if make_plots:
        #plt.close(2)
        plt.figure(2, figsize=FIGSIZE).clf()
        fig, axgrp = plt.subplots(2, 2, sharex=True, num=2, figsize=FIGSIZE)
        axgrp = axgrp.flatten()
        basetitle = (
            'Fraction of events ID\'d, %s geometry %s, MC runs %s with V'
            '%s processing & and PID version %s' %
            (events.metadata['detector'].upper(), events.metadata['geom'].upper(),
             utils.list2hrlist(events.metadata['runs']),
             events.metadata['proc_ver'], args.pid_ver)
        )
        fig.suptitle(basetitle, fontsize=12)

    rel_errors = pidmc.get_rel_error()

    param_pid_maps = {'binning': {'ebins':ebins, 'czbins':czbins}}
    histdata = {}
    smooth_coszen_dep = {}
    czdep_store = {'czbin_midpoints': czbin_midpoints}
    for label, label_data in pidmc.pid_maps.items():
        if label == 'binning':
            continue
        pid_fract_totals = np.zeros(n_czbins)

        histdata[label] = {}
        smooth_coszen_dep[label] = {}
        czdep_store[label] = {}
        for sig in sigs_to_spline:
            histdata[label][sig] = np.squeeze(label_data[sig])

            assert np.sum(np.isinf(np.abs(rel_errors[label]))) == 0
            assert np.sum(np.isnan(rel_errors[label])) == 0
            #abs_err = np.squeeze(rel_errors[label])*histdata[label][sig]
            #rel_errors[label][
            weights = 1./rel_errors[label]
            weights[weights == 0] = 1./np.max(rel_errors[label])
            assert np.sum(np.isinf(np.abs(weights))) == 0
            assert np.sum(np.isnan(weights)) == 0
            #weights = 1./abs_err
            #weights[weights == 0] = 1./np.max(abs_err)

            n_smooth = 25
            smoothing_kernel = np.blackman(n_smooth)
            norm = 1/np.convolve(np.ones(n_czbins), smoothing_kernel, 'same')
            smoothed = np.convolve(histdata[label][sig], smoothing_kernel, 'same') \
                    * norm
            #spline = splrep(
            #    czbin_midpoints, smoothed, #histdata[label][sig],
            #    w=weights,
            #    k=5, s=0.25,
            #)

            # Individually do not allow the fraction to be outside [0, 1]
            #pid_fract_vals = np.clip(splev(czbin_midpoints, spline),
            #                         a_min=0, a_max=1)
            pid_fract_vals = np.clip(smoothed, a_min=0, a_max=1)

            pid_fract_totals += pid_fract_vals

            # Store the intermediate results
            smooth_coszen_dep[label][sig] = pid_fract_vals

        # If total anywhere is greater than 1, figure out by how much so we can
        # scale each fraction down in the end
        maxtotal = np.max(pid_fract_totals)
        scale_factor = 1
        if maxtotal > 1:
            scale_factor = 1/maxtotal

        # Compute "dependent" signature's PID in terms of the difference between 1
        # and the total of all other PID signatures
        sig = args.dependent_sig
        histdata[label][sig] = np.squeeze(label_data[sig])
        # Apply the scale factor to 1 to avoid negative numbers; in the end we will
        # scale this down again
        pid_fract_vals = 1.0/scale_factor - pid_fract_totals
        smooth_coszen_dep[label][sig] = pid_fract_vals

        for sig in all_sigs:
            smooth_coszen_dep[label][sig] *= scale_factor
            # Populate datastructure to be written to disk
            czdep_store[label][sig] = {
                'histo': histdata[label][sig],
                'histo_err': np.squeeze(rel_errors[label])*np.squeeze(histdata[label][sig]),
                'smooth': smooth_coszen_dep[label][sig],
            }

        if make_plots:
            if label == 'nue_cc':
                grp = flavInt.NuFlavIntGroup('nuecc+nuebarcc')
                axnum = 0
            elif label == 'numu_cc':
                axnum = 1
                grp = flavInt.NuFlavIntGroup('numucc+numubarcc')
            elif label == 'nutau_cc':
                axnum = 2
                grp = flavInt.NuFlavIntGroup('nutaucc+nutaubarcc')
            else:
                axnum = 3
                grp = flavInt.NuFlavIntGroup('nuallnc+nuallbarnc')
            ax = axgrp[axnum]

            for sig in ['trck']:
                if len(sigs_to_spline) == 1:
                    lab_sfx = ''
                    leg_sfx = r'; id$\Rightarrow$' + sig
                else:
                    lab_sfx = r'; id$\Rightarrow$' + sig
                    leg_sfx = ''

                store = czdep_store[label][sig]
                stepHist(
                    czbins, y=store['histo'], yerr=store['histo_err'],
                    ax=ax, color=(0.2,0.6,0),
                    label='Histogram' + lab_sfx
                )
                # Plot the splined points
                ax.plot(
                    czbin_midpoints, store['smooth'],
                    'k-', lw=1.5, ms=1,
                    label='Smoothed' + lab_sfx
                )

            ax.set_xscale('linear')
            ax.set_xlim(-1, 1)
            #ylim = np.clip(ax.get_ylim(), 0, 1)
            #ax.set_ylim(ylim)
            if axnum in [0,2,3]:
                ax.set_ylim(0, 0.5)
            if axnum in [1]:
                ax.set_ylim(0, 1)
            if axnum in [2,3]:
                ax.set_xlabel(r'Reconstructed $\cos\,\theta_{\rm zen}$')
            ax.grid(b=True, which='both', ls='-', alpha=0.7, color=[0.5]*3, lw=0.5)

            leg = ax.legend(
                loc='best', frameon=False,
                title=flavInt.tex(grp,d=1) + leg_sfx,
            )
            leg.get_title().set_fontsize(16)

    # Derive output filename
    outfname = (
        'pid_coszen_smooth__%s_%s__runs_%s__proc_%s__pid_%s.json' % (
            events.metadata['detector'],
            events.metadata['geom'],
            utils.list2hrlist(events.metadata['runs']),
            events.metadata['proc_ver'],
            args.pid_ver,
        )
    )
    outfpath = os.path.join(outdir, outfname)
    logging.info('Saving PID coszen smooth info to file "%s"' % outfpath)
    jsons.to_json(czdep_store, outfpath)

    if make_plots:
        fig.tight_layout(rect=(0,0,1,0.96))
        basefname = (
            'pid_coszen_smooth__%s_%s__runs_%s__proc_%s__pid_%s'
            % (events.metadata['detector'], events.metadata['geom'],
               utils.list2hrlist(events.metadata['runs']),
               events.metadata['proc_ver'], args.pid_ver)
        )
        fig.savefig(os.path.join(outdir, basefname + '.pdf'))
        fig.savefig(os.path.join(outdir, basefname + '.png'))
