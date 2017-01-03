#! /usr/bin/env python
#
# author: J.L. Lanfranchi
# date:   2016-03-08
#
"""
One-dimensional (in energy) effective areas are smoothed and sampled, one for
each slide in cosine-zenith.

Interpolation can be achieved by...?

Events of a given flavor/interaction type (or all events from grouped
flavor/interaction types) from a PISA events file have their effective areas
computed. This is smoothed with a spline, and the spline fit is sampled at the
specified energy bins' midpoints (on a linear scale).
"""


import os
from copy import deepcopy
from itertools import izip
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
from scipy.interpolate import splrep, splev

from pisa.core.events import Events
from pisa.utils.log import logging, set_verbosity
from pisa.utils import flavInt
from pisa.utils import hdf
from pisa.utils import utils
from pisa.aeff import Aeff


__all__ = ['slice_smooth', 'plot_slice_smooth']


def slice_smooth(events, n_ebins, n_czbins, e_smooth_factor, cz_smooth_factor,
                 emin, emax, czmin, czmax):
    # Load/populate the events
    events = Events(events)

    # TODO: load default energy ranges from mc_sim_run_settings
    # if not specified by user

    # Create metadata dict to record all parameters used in generating the
    # slice-smoothed characterization of effective areas.
    # ...
    # Initialize with info about events used
    metadata = deepcopy(events.metadata)

    # Record info about smoothing parameters
    metadata.update(dict(e_smooth_factor=e_smooth_factor,
                         cz_smooth_factor=cz_smooth_factor))

    # Generate binning: log in energy, linear in coszen
    ebins = np.logspace(np.log10(emin), np.log10(emax), n_ebins+1)
    czbins = np.linspace(czmin, czmax, n_czbins+1)

    ebin_midpoints = (ebins[:-1] + ebins[1:])/2.0
    czbin_midpoints = (czbins[:-1] + czbins[1:])/2.0

    # Record info to metadata
    metadata.update(dict(emin=emin, emax=emax, n_ebins=n_ebins,
                         czmin=czmin, czmax=czmax, n_czbins=n_czbins))

    # Verify flavints joined in the file are those expected
    grouped = sorted([flavInt.NuFlavIntGroup(fi)
                      for fi in events.metadata['flavints_joined']])
    should_be_grouped = sorted([flavInt.NuFlavIntGroup('nuall_nc'),
                                flavInt.NuFlavIntGroup('nuallbar_nc')])
    if grouped != should_be_grouped:
        if len(grouped) == 0:
            grouped = None
        raise ValueError('Only works with groupings (%s) but instead got'
                         ' groupings (%s).' % (should_be_grouped, grouped))

    # Get *un*joined flavints
    individual_flavints = flavInt.NuFlavIntGroup(flavInt.ALL_NUFLAVINTS)
    for group in grouped:
        individual_flavints -= group
    ungrouped = sorted([flavInt.NuFlavIntGroup(fi)
                        for fi in individual_flavints])

    logging.debug("Groupings: %s" % grouped)
    logging.debug("Ungrouped: %s" % ungrouped)

    smooth_store = {'ebins': ebins,
                    'czbins': czbins}
    slice_store = flavInt.FlavIntData()
    for flavint in slice_store.flavints():
        slice_store[flavint] = []

    # Instantiate aeff service (specifying first CZ bin)
    aeff_mc_service = Aeff.aeff_service_factory(
        aeff_mode='mc', ebins=ebins, czbins=czbins[0:2],
        aeff_weight_file=events, compute_error=True
    )

    for czbin in izip(czbins[:-1], czbins[1:]):
        aeff_mc_service.update(ebins, czbin)
        aeff_data, aeff_err = aeff_mc_service.get_aeff_with_error()

        spline_fit = flavInt.FlavIntData()
        for group in ungrouped + grouped:
            # Only need to do computations for a single flavint from the group,
            # since all data in other flavints is just duplicated
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
                #ebin_midpoints, s_aeff, w=1./np.array(s_aeff_err), k=3, s=100
                ebin_midpoints, s_aeff, w=1./np.array(s_aeff_err),
                k=3, s=e_smooth_factor
            )

            # Sample the spline at the bin midpoints
            smoothed_aeff = splev(ebin_midpoints, spline)

            # Force bins that were previously NaN or 0 back to 0
            # NOTE: removed, since we *want* zero bins to be filled in
            #smoothed_aeff[zero_and_nan_indices] = 0

            # Make sure no nan or inf bins remain
            assert not np.any(np.isnan(smoothed_aeff) +
                              np.isinf(np.abs(smoothed_aeff)))

            # Populate datastructure to be written to disk
            if not repr(group) in smooth_store:
                smooth_store[repr(group)] = {
                    'hist': [],
                    'hist_err': [],
                    'smooth_cz_slices': [],
                    'smooth': [],
                }
            smooth_store[repr(group)]['hist'].append(s_aeff)
            smooth_store[repr(group)]['hist_err'].append(s_aeff_err)
            smooth_store[repr(group)]['smooth_cz_slices'].append(smoothed_aeff)

            # Append to the slice list for this flavint/group
            slice_store[rep_flavint].append(smoothed_aeff)

    # Convert lists-of-arrays to 2D arrays
    for group in ungrouped + grouped:
        branch = smooth_store[repr(group)]
        for k, v in branch.iteritems():
            # Transpose so that energy is first index (low->high) and coszen is
            # second index
            branch[k] = np.array(v).T

    # Similar process, but slice in E (using already-existing binning +
    # smoothed samples therefrom) and smooth-spline CZ dependence
    for group in ungrouped + grouped:
        branch = smooth_store[repr(group)]
        hist = branch['hist']
        hist_err = branch['hist_err']
        smooth_cz_slices = branch['smooth_cz_slices']
        smooth = []
        for e_slice_n in xrange(smooth_cz_slices.shape[0]):
            e_slice = smooth_cz_slices[e_slice_n,:]
            # TODO: propagate errors!
            spline = spline_fit[rep_flavint] = splrep(
                czbin_midpoints, e_slice, w=None,
                k=3, s=cz_smooth_factor
            )

            # Sample the spline at the bin midpoints
            smoothed_aeff = splev(czbin_midpoints, spline)

            smooth.append(smoothed_aeff)

        smooth = np.array(smooth)
        branch['smooth'] = smooth

    return smooth_store, metadata


def plot_slice_smooth(smooth_store, metadata=None, save_basename=None):
    import matplotlib as mpl
    mpl.use('pdf')
    mpl.rc('font', **{'family':'serif', 'weight':'normal', 'size': 10})
    import matplotlib.pyplot as plt
    from pisa.utils.plot import stepHist

    if isinstance(smooth_store, basestring):
        assert metadata is None
        smooth_store, metadata = hdf.from_hdf(smooth_store, return_attrs=True)

    grouped = sorted([flavInt.NuFlavIntGroup(fi)
                      for fi in metadata['flavints_joined']])
    individual_flavints = flavInt.NuFlavIntGroup(flavInt.ALL_NUFLAVINTS)
    for group in grouped:
        individual_flavints -= group
    ungrouped = sorted([flavInt.NuFlavIntGroup(fi)
                        for fi in individual_flavints])

    def p2ticks(ax):
        """Powers-of-two tick locations"""
        ymin, ymax = ax.get_ylim()
        yticks = 2**np.arange(np.floor(np.log2(ymin)), np.log2(ymax), 1,
                              dtype=int)
        ax.set_yticks(yticks, minor=False)
        ax.set_yticks([], minor=True)
        ax.set_yticklabels(yticks)

    def plotmap(x, y, z, vmin, vmax, ax, title, cmap=mpl.cm.Paired):
        qm = ax.pcolormesh(x, y, z, vmin=vmin, vmax=vmax, cmap=cmap,
                           linewidth=0, rasterized=True)
        qm.set_edgecolor('face')
        ax.set_yscale('log')
        ax.axis('tight')
        p2ticks(ax)
        ax.grid(b=True, which='major', ls='-', c='k', alpha=0.2, lw=1)
        ax.set_xlabel(r'$\cos\,\theta_{z}$')
        ax.set_ylabel(r'Energy (GeV)')
        ax.set_title(title)
        plt.colorbar(mappable=qm, ax=ax)
        return qm

    ebins, czbins = smooth_store['ebins'], smooth_store['czbins']
    X, Y = np.meshgrid(czbins, ebins)

    # Fine-binned result
    ebins_fine = np.logspace(np.log10(np.min(ebins)),
                             np.log10(np.max(ebins)),
                             1000)
    czbins_fine = np.linspace(np.min(czbins), np.max(czbins), 1000)
    X_fine, Y_fine = np.meshgrid(czbins_fine, ebins_fine)
    aeff_slice_service = Aeff.aeff_service_factory(
        aeff_mode='slice_smooth', ebins=ebins_fine, czbins=czbins_fine,
        aeff_slice_smooth=smooth_store
    )
    aeff_fine = aeff_slice_service.get_aeff()

    for group in ungrouped + grouped:
        branch = smooth_store[repr(group)]
        hist = branch['hist']
        hist_err = branch['hist_err']
        smooth_cz_slices = branch['smooth_cz_slices']
        smooth = branch['smooth']

        rep_flavint = group.flavints()[0]
        fine = aeff_fine[rep_flavint]

        log_h = np.ma.masked_invalid(np.log10(hist))
        log_scz = np.ma.masked_invalid(np.log10(smooth_cz_slices))
        log_s = np.ma.masked_invalid(np.log10(smooth))
        log_fine = np.ma.masked_invalid(np.log10(fine))
        fract_err = np.ma.masked_invalid((smooth-hist)/hist)

        # Derive vmin and vmax from the histogram (separately for each group)
        vmin = np.inf
        vmax = -np.inf

        for p in [log_h, log_scz, log_s]:
            vmin = np.min([vmin, np.min(p)])
            vmax = np.max([vmax, np.max(p)])

        fig, axgrp = plt.subplots(1, 5, figsize=(22,5), dpi=70)
        fig.suptitle('$' + group.tex() + '$', fontsize=16)
        axiter = iter(axgrp.flatten())

        plotmap(X, Y, log_h, vmin=vmin, vmax=vmax, ax=axiter.next(),
                title=r'log$_{10}$(Monte Carlo)')

        plotmap(X, Y, log_scz, vmin=vmin, vmax=vmax, ax=axiter.next(),
                title='CZ slices, spline-smooth in E')

        plotmap(X, Y, log_s, vmin=vmin, vmax=vmax, ax=axiter.next(),
                title='E slices, spline-smooth in CZ')

        plotmap(X_fine, Y_fine, log_fine, vmin=vmin, vmax=vmax,
                ax=axiter.next(),
                title='Fine binning')

        maxdev = np.max(np.abs(fract_err))
        vmin, vmax = -1.0, 1.0
        plotmap(X, Y, fract_err, vmin=vmin, vmax=vmax, ax=axiter.next(),
                cmap=mpl.cm.coolwarm,
                title=r'${\rm (smooth - MC) / MC}$' + '\n' +
                r'$\mu = %.4f, \; \sigma = %.4f$' % (np.mean(fract_err),
                                                     np.std(fract_err)))
        plt.tight_layout(rect=(0, 0, 1, 0.97))

        logging.info(str(group) + ' fractional error compared with histogram')
        logging.info('    mean: %.5e' %  np.mean(fract_err))
        logging.info('  stddev: %.5e' %  np.std(fract_err))
        logging.info('     RMS: %.5e' %  np.sqrt(np.mean(fract_err*fract_err)))
        logging.info('     max: %.5e' %  np.max(fract_err))
        logging.info('     min: %.5e' %  np.min(fract_err))

        if save_basename is not None:
            plotfbase = save_basename + '__' + str(group)
            plt.savefig(plotfbase + '.png')
            plt.savefig(plotfbase + '.pdf')


if __name__ == "__main__":
    parser = ArgumentParser(
        '''Generate smoothed effective areas at energy bin centers. NOTE: at
        present, uses *ONLY* the MC-true-upgoing events.''',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--events', metavar='RESOURCE_LOC', type=str,
        required=True,
        help='''PISA events resource location used for computing smooth. It is
        expected that nuall_nc and nuallbar_nc are joined in this file, while
        other flavor/interaction types are unjoined.'''
    )
    parser.add_argument(
        '--outdir', metavar='dir', type=str,
        required=True,
        help='Directory into which to place the produced files.'
    )
    parser.add_argument(
        '--n-ebins', type=int,
        default=39,
        help='Number of energy bins (logarithmically-spaced)'
    )
    parser.add_argument(
        '--n-czbins', type=int,
        default=40,
        help='Number of cosine-zenith bins (linearly-spaced)'
    )
    parser.add_argument(
        '--e-smooth-factor', type=float,
        default=50,
        help='Spline smoothing factor to apply as function of energy'
    )
    parser.add_argument(
        '--cz-smooth-factor', type=float,
        default=50,
        help='Spline smoothing factor to apply as function of coszen'
    )
    parser.add_argument(
        '--emin', metavar='emin_gev', type=float,
        required=True,
        help='Energy bins\' left-most edge, in gev'
    )
    parser.add_argument(
        '--emax', metavar='emax_gev', type=float,
        required=True,
        help='Energy bins\' righ-most edge, in gev'
    )
    parser.add_argument(
        '--czmin', metavar='COSZEN', type=float,
        required=True,
        help='Cosine-zenith bins\' lowest edge'
    )
    parser.add_argument(
        '--czmax', metavar='COSZEN', type=float,
        required=True,
        help='Cosine-zenith bins\' highest edge'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Do *not* make debug plots (which are made by default)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count', default=0,
        help='Set verbosity level'
    )
    cmdline_args = vars(parser.parse_args())

    set_verbosity(cmdline_args.pop('verbose'))
    outdir = os.path.expandvars(os.path.expanduser(cmdline_args.pop('outdir')))
    make_plots = not cmdline_args.pop('no_plots')

    # Do the smoothing
    smooth_store, metadata = slice_smooth(**cmdline_args)

    # Derive output file path + base file name
    outfbase = os.path.join(outdir,
        'aeff_slice_smooth__%s_%s__runs_%s__proc_%s' % (
            metadata['detector'],
            metadata['geom'],
            utils.list2hrlist(metadata['runs']),
            metadata['proc_ver']
        )
    )

    # Save to HDF5 file
    utils.mkdir(outdir)
    hdf.to_hdf(smooth_store, outfbase + '.hdf5', attrs=metadata)

    if make_plots:
        plot_slice_smooth(smooth_store=smooth_store, metadata=metadata,
                          save_basename=outfbase)
