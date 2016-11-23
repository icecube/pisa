#! /usr/bin/env python
#
# author: J.L. Lanfranchi
# date:   2016-03-01
#
"""
One-dimensional effective areas are smoothed and sampled (meant to be
interpolated between) as functions of energy and cosine-zenith, independently.

Events of a given flavor/interaction type (or all events from grouped
flavor/interaction types) from a PISA have their effective areas computed. This
is smoothed with a spline, and the spline fit is sampled at the specified
energy bins' midpoints (on a linear scale).
"""

# TODO: make energy-dependent and coszen-dependent smoothing a function
# TODO: store metadata about how smoothing was done
# TODO: use CombinedFlavIntData for storage of the results


import os,sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import izip

import numpy as np
from scipy.interpolate import splrep, splev

from pisa.core.events import Events
from pisa.utils.log import logging, set_verbosity
from pisa.utils.flavInt import ALL_NUFLAVINTS, FlavIntData, NuFlavIntGroup, tex
from pisa.utils.format import list2hrlist
from pisa.utils import jsons
# TODO: this no longer exists; fix up for PISA 3
from pisa.aeff.AeffServiceMC import AeffServiceMC


__all__ = []


if __name__ == '__main__':
    parser = ArgumentParser(
        '''Generate smoothed effective areas at energy bin centers. NOTE: at
        present, uses *ONLY* the MC-true-upgoing events.''',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--events-for-esmooth', metavar='RESOURCE_LOC', type=str,
        required=True,
        help='''PISA events file used for computing energy smooth. It is
        expected that nuall_nc and nuallbar_nc are joined in this file, while other
        flavor/interaction types are unjoined.'''
    )
    parser.add_argument(
        '--events-for-czsmooth', metavar='RESOURCE_LOC', type=str,
        required=True,
        help='''PISA events file used for computing coszen smooth. It is
        expected that nue_cc+nuebar_cc, numu_cc+numubar_cc, nutau_cc+nutaubar_cc,
        and nuall_nc+nuallbar_nc are joined in this file.'''
    )
    parser.add_argument(
        '--outdir', metavar='DIR', type=str,
        required=True,
        help='Directory into which to place the produced files.'
    )
    parser.add_argument(
        '--emin', metavar='EMIN_GeV', type=float,
        required=True,
        help='Energy bins\' left-most edge, in GeV'
    )
    parser.add_argument(
        '--emax', metavar='EMAX_GeV', type=float,
        required=True,
        help='Energy bins\' righ-most edge, in GeV'
    )
    parser.add_argument(
        '--n-ebins', type=int,
        required=True,
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
        '--n-czbins', type=int,
        required=True,
        help='Number of cosine-zenith bins (linearly-spaced)'
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

    esmooth_events_fpath = args.events_for_esmooth
    czsmooth_events_fpath = args.events_for_czsmooth
    outdir = os.path.expandvars(os.path.expanduser(args.outdir))
    make_plots = not args.no_plots

    # Load the events
    esmooth_events = Events(esmooth_events_fpath)
    czsmooth_events = Events(czsmooth_events_fpath)

    # Verify user-specified files are compatible with one another
    assert czsmooth_events.metadata['detector'] == esmooth_events.metadata['detector']
    assert czsmooth_events.metadata['geom'] == esmooth_events.metadata['geom']
    assert np.alltrue(czsmooth_events.metadata['runs'] == esmooth_events.metadata['runs'])
    assert czsmooth_events.metadata['proc_ver'] == esmooth_events.metadata['proc_ver']
    assert np.alltrue(czsmooth_events.metadata['cuts'] == esmooth_events.metadata['cuts'])

    # Define binning for 1D A_eff smooth data. Note that a single CZ bin is
    # employed to collapse that dimension of the histogram for characterizing
    # energy smooth, and likewise a single E bin is employed to collapse that
    # dimension for characterizing CZ smooth.
    emin, emax, n_ebins = args.emin, args.emax, args.n_ebins
    czmin, czmax, n_czbins = -1, +1, args.n_czbins

    ebins = np.logspace(np.log10(emin), np.log10(emax), n_ebins+1)
    czbins = np.linspace(czmin, czmax, n_czbins+1)

    ebin_midpoints = (ebins[:-1] + ebins[1:])/2.0
    czbin_midpoints = (czbins[:-1] + czbins[1:])/2.0

    #===============================================================================
    # Energy
    #===============================================================================
    # NOTE forcing only upgoing to be included for computing energy-smooth
    single_czbin = [-1, 0]
    assert len(single_czbin)-1 == 1

    # Verify flavints joined in the file are those expected
    grouped = sorted([NuFlavIntGroup(fi)
                      for fi in esmooth_events.metadata['flavints_joined']])
    should_be_grouped = sorted([NuFlavIntGroup('nuall_nc'),
                                NuFlavIntGroup('nuallbar_nc')])
    if grouped != should_be_grouped:
        if len(grouped) == 0:
            grouped = None
        raise ValueError('Only works with groupings (%s) but instead got'
                         ' groupings (%s).' % (should_be_grouped, grouped))

    # Get *un*joined flavints
    individual_flavints = NuFlavIntGroup(ALL_NUFLAVINTS)
    for group in grouped:
        individual_flavints -= group
    ungrouped = sorted([NuFlavIntGroup(fi) for fi in individual_flavints])

    logging.debug("Groupings: %s" % grouped)
    logging.debug("Ungrouped: %s" % ungrouped)

    if make_plots:
        import matplotlib as mpl
        mpl.use('pdf')
        import matplotlib.pyplot as plt
        from pisa.utils.plot import stepHist

        FIGSIZE = (11,7)
        #plt.close(1)
        #plt.close(2)
        plt.figure(1).clf()
        plt.figure(2).clf()
        fig_part, ax_part = plt.subplots(2, 2, sharex=True, num=1, figsize=FIGSIZE)
        fig_anti, ax_anti = plt.subplots(2, 2, sharex=True, num=2, figsize=FIGSIZE)
        ax_part = ax_part.flatten()
        ax_anti = ax_anti.flatten()
        basetitle = (
            'effective areas [m$^2$], %s geometry %s, MC runs %s with ver'
            ' %s processing' % (esmooth_events.metadata['detector'],
                                esmooth_events.metadata['geom'],
                                list2hrlist(esmooth_events.metadata['runs']),
                                esmooth_events.metadata['proc_ver'])
        )
        fig_part.suptitle('Particle ' + basetitle, fontsize=12)
        fig_anti.suptitle('Antiparticle ' + basetitle, fontsize=12)

    aeff_svc = AeffServiceMC(ebins, single_czbin, esmooth_events_fpath,
                             compute_error=True)
    aeff_data, aeff_err = aeff_svc.get_aeff_with_error()

    spline_fit = FlavIntData()
    smoothed_aeff = FlavIntData()
    esmooth_store = {'ebin_midpoints': ebin_midpoints}
    for group in ungrouped + grouped:
        # Only need to do computations for a single flavint from the group, since
        # all data in other flavints is just duplicated
        rep_flavint = group.flavints()[0]

        s_aeff = np.squeeze(aeff_data[rep_flavint])
        s_aeff_err = np.squeeze(aeff_err[rep_flavint])

        # Eliminate ~0 and nan in errors to avoid infinite weights
        # Smallest non-zero error
        zero_and_nan_indices = np.squeeze(
            (s_aeff == 0) | (s_aeff != s_aeff) |
            (s_aeff_err == 0) | (s_aeff_err != s_aeff_err)
        )
        min_err = np.min(s_aeff_err[s_aeff_err > 0])
        s_aeff_err[zero_and_nan_indices] = min_err

        # Smooth histogrammed A_eff(E) using a spline
        spline = spline_fit[rep_flavint] = splrep(
            ebin_midpoints, s_aeff, w=1./np.array(s_aeff_err), k=3, s=100
        )

        # Sample the spline at the bin midpoints
        smoothed_aeff[rep_flavint] = splev(ebin_midpoints, spline)

        # Force bins that were previously NaN or 0 back to 0
        smoothed_aeff[rep_flavint][zero_and_nan_indices] = 0

        # Populate datastructure to be written to disk
        esmooth_store[repr(group)] = {
            'histo': s_aeff,
            'histo_err': s_aeff_err,
            'smooth': smoothed_aeff[rep_flavint],
        }

        if make_plots:
            # Figure out whether particle or anti-particle axis group
            if rep_flavint.isParticle():
                axgrp = ax_part
            else:
                axgrp = ax_anti

            # Determine axis by flavor
            axnum = 3
            if rep_flavint in NuFlavIntGroup('nuecc+nuebarcc'):
                axnum = 0
            elif rep_flavint in NuFlavIntGroup('numucc+numubarcc'):
                axnum = 1
            elif rep_flavint in NuFlavIntGroup('nutaucc+nutaubarcc'):
                axnum = 2
            ax = axgrp[axnum]

            # Plot the histogram with errors
            stepHist(ebins, y=s_aeff, yerr=s_aeff_err,
                     ax=ax, label='Histogramed', color=(0.8,0.2,0.6))
            # Plot the smoothed curve
            ax.plot(ebin_midpoints, smoothed_aeff[rep_flavint], 'k-o',
                    lw=0.25, ms=2,
                    label='Spline-smoothed',)

            ax.set_yscale('log')
            if axnum in [2,3]:
                ax.set_xlabel('Energy (GeV)')

            if axnum in [0, 1]:
                ax.set_ylim(1e-7, 2e-4)
            else:
                ax.set_ylim(1e-8, 2e-4)

            ax.grid(b=True, which='both', ls='-', alpha=0.5, color=[0.7]*3, lw=0.5)

            leg = ax.legend(loc='lower right', title=tex(group, d=True),
                            frameon=False)
            leg.get_title().set_fontsize(18)

    # Derive output filename
    outfname = (
        'aeff_energy_smooth__%s_%s__runs_%s__proc_%s.json' % (
            esmooth_events.metadata['detector'],
            esmooth_events.metadata['geom'],
            list2hrlist(esmooth_events.metadata['runs']),
            esmooth_events.metadata['proc_ver']
        )
    )
    outfpath = os.path.join(outdir, outfname)
    logging.info('Saving Aeff energy smooth info to file "%s"' % outfpath)
    jsons.to_json(esmooth_store, outfpath)

    if make_plots:
        fig_part.tight_layout(rect=(0,0,1,0.96))
        fig_anti.tight_layout(rect=(0,0,1,0.96))
        basefname = (
            'aeff_energy_smooth__%s_%s__runs_%s__proc_%s__'
            % (esmooth_events.metadata['detector'], esmooth_events.metadata['geom'],
               list2hrlist(esmooth_events.metadata['runs']),
               esmooth_events.metadata['proc_ver'])
        )
        fig_part.savefig(os.path.join(outdir, basefname + 'particles.pdf'))
        fig_part.savefig(os.path.join(outdir, basefname + 'particles.png'))
        fig_anti.savefig(os.path.join(outdir, basefname + 'antiparticles.pdf'))
        fig_anti.savefig(os.path.join(outdir, basefname + 'antiparticles.png'))


    #===============================================================================
    # COSZEN
    #===============================================================================
    # Verify flavints joined in the file are those expected
    grouped = sorted([NuFlavIntGroup(fi)
                      for fi in czsmooth_events.metadata['flavints_joined']])
    should_be_grouped = sorted([
        NuFlavIntGroup('nue_cc+nuebar_cc'),
        NuFlavIntGroup('numu_cc+numubar_cc'),
        NuFlavIntGroup('nutau_cc+nutaubar_cc'),
        NuFlavIntGroup('nuall_nc+nuallbar_nc'),
    ])
    if set(grouped) != set(should_be_grouped):
        if len(grouped) == 0:
            grouped = None
        raise ValueError('Only works with groupings (%s) but instead got'
                         ' groupings (%s).' % (should_be_grouped,
                                               grouped))

    # Get *un*joined flavints
    individual_flavints = NuFlavIntGroup(ALL_NUFLAVINTS)
    for group in grouped:
        individual_flavints -= group
    ungrouped = sorted([NuFlavIntGroup(fi) for fi in individual_flavints])

    logging.debug("Groupings: %s" % grouped)
    logging.debug("Ungrouped: %s" % ungrouped)

    # Look at coszen smooth for all energies included in the specified binning,
    # lumped together into a single bin

    # NOTE: using 1-80 GeV biases low energies with high-energy behavior. We should
    # really just parameterize in energy, using slices of CZ, since marginalizing
    # energy seems like a bad (too unphysical) thing to do.
    #single_ebin = [emin, emax]
    single_ebin = [1, 20]
    assert len(single_ebin)-1 == 1

    if make_plots:
        import matplotlib as mpl
        mpl.use('pdf')
        import matplotlib.pyplot as plt
        from pisa.utils.plot import stepHist

        FIGSIZE = (11,7)
        #plt.close(3)
        plt.figure(3).clf()
        fig, axgrp = plt.subplots(2, 2, sharex=True, num=3, figsize=FIGSIZE)
        axgrp = axgrp.flatten()
        fig.suptitle('Particle+antiparticle ' + basetitle, fontsize=12)

    aeff_svc = AeffServiceMC(single_ebin, czbins, czsmooth_events_fpath,
                             compute_error=True)
    aeff_data, aeff_err = aeff_svc.get_aeff_with_error()

    spline_fit = FlavIntData()
    smoothed_aeff = FlavIntData()
    czsmooth_store = {'czbin_midpoints': czbin_midpoints}
    for group in ungrouped + grouped:
        rep_flavint = group.flavints()[0]
        s_aeff = np.squeeze(aeff_data[rep_flavint])
        s_aeff_err = np.squeeze(aeff_err[rep_flavint])
        zero_and_nan_indices = np.squeeze(
            (s_aeff == 0) | (s_aeff != s_aeff) |
            (s_aeff_err == 0) | (s_aeff_err != s_aeff_err)
        )
        min_err = np.min(s_aeff_err[s_aeff_err > 0])
        s_aeff_err[zero_and_nan_indices] = min_err
        spline_fit[rep_flavint] = splrep(czbin_midpoints, s_aeff,
                                         w=1./np.squeeze(s_aeff_err),
                                         k=5, s=200)

        # Sample the spline at the bin midpoints
        smoothed_aeff[rep_flavint] = splev(czbin_midpoints,
                                           spline_fit[rep_flavint])

        # Force bins that were previously NaN or 0 back to 0
        smoothed_aeff[rep_flavint][zero_and_nan_indices] = 0

        # Populate datastructure to be written to disk
        czsmooth_store[repr(group)] = {
            'histo': s_aeff,
            'histo_err': s_aeff_err,
            'smooth': smoothed_aeff[rep_flavint],
        }

        if make_plots:
            # Determine axis by flavor
            axnum = 3
            if rep_flavint in NuFlavIntGroup('nuecc+nuebarcc'):
                axnum = 0
            elif rep_flavint in NuFlavIntGroup('numucc+numubarcc'):
                axnum = 1
            elif rep_flavint in NuFlavIntGroup('nutaucc+nutaubarcc'):
                axnum = 2
            ax = axgrp[axnum]

            # Plot the histogram with errors
            stepHist(czbins, y=s_aeff, yerr=s_aeff_err,
                     ax=ax, label='Histogramed', color=(0.2,0.6,0))
            # Plot the smoothed curve
            ax.plot(czbin_midpoints, smoothed_aeff[rep_flavint], 'k-o',
                    lw=0.25, ms=2,
                    label='Spline-smoothed',)

            ax.set_yscale('linear')
            if axnum in [2,3]:
                ax.set_xlabel(r'$\cos\,\theta_z$')

            legparams = dict(title=tex(group, d=True), frameon=False)
            if axnum in [0,1]:
                ax.set_ylim(4.0e-5, 8.5e-5)
                leg = ax.legend(loc='lower center', **legparams)
            elif axnum in [2]:
                ax.set_ylim(3.5e-5, 5.5e-5)
                leg = ax.legend(loc='best', **legparams)
            else:
                ax.set_ylim(1.6e-5, 2.6e-5)
                leg = ax.legend(loc='best', **legparams)

            ax.get_yaxis().get_major_formatter().set_powerlimits((-3,4))

            ax.grid(b=True, which='both', ls='-', alpha=0.5, color=[0.7]*3, lw=0.5)

            leg.get_title().set_fontsize(18)

    # Derive output filename
    outfname = (
        'aeff_coszen_smooth__%s_%s__runs_%s__proc_%s.json' % (
            czsmooth_events.metadata['detector'],
            czsmooth_events.metadata['geom'],
            list2hrlist(czsmooth_events.metadata['runs']),
            czsmooth_events.metadata['proc_ver']
        )
    )
    outfpath = os.path.join(outdir, outfname)
    logging.info('Saving Aeff coszen smooth info to file "%s"' % outfpath)
    jsons.to_json(czsmooth_store, outfpath)

    if make_plots:
        fig.tight_layout(rect=(0,0,1,0.96))
        basefname = (
            'aeff_coszen_smooth__%s_%s__runs_%s__proc_%s'
            % (czsmooth_events.metadata['detector'], czsmooth_events.metadata['geom'],
               list2hrlist(czsmooth_events.metadata['runs']),
               czsmooth_events.metadata['proc_ver'])
        )
        fig.savefig(os.path.join(outdir, basefname + '.pdf'))
        fig.savefig(os.path.join(outdir, basefname + '.png'))
