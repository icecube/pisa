#! /usr/bin/env python
# author: S.Wren
# date:   October 25, 2016
"""
A set of functions for calculating flux weights given an array of energy and
cos(zenith) values based on the Honda atmospheric flux tables. A lot of this
functionality will be copied from honda.py but since I don't want to initialise
this as a stage it makes sense to copy it in to here so somebody can't
accidentally do the wrong thing with that script.
"""


from __future__ import division

import numpy as np
import scipy.interpolate as interpolate

from pisa.utils.log import logging
from pisa.utils.resources import open_resource


__all__ = ['load_2D_table', 'calculate_flux_weights']


PRIMARIES = ['numu', 'numubar', 'nue', 'nuebar']


def load_2D_table(flux_file):
    """Manipulate 2 dimensional flux tables.

    2D is expected to mean energy and cosZenith, where azimuth is averaged
    over (before being stored in the table) and the zenith range should
    include both hemispheres.

    Parameters
    ----------
    flux_file : string
        The location of the flux file you want to spline. Should be a honda
        azimuth-averaged file.

    """

    if not isinstance(flux_file, basestring):
        raise ValueError('Flux file name must be a string')
    logging.debug("Loading atmospheric flux table %s" % flux_file)

    # columns in Honda files are in the same order
    cols = ['energy'] + PRIMARIES

    # Load the data table
    table = np.genfromtxt(open_resource(flux_file),
                          usecols=range(len(cols)))
    mask = np.all(np.isnan(table) | np.equal(table, 0), axis=1)
    table = table[~mask].T

    flux_dict = dict(zip(cols, table))
    for key in flux_dict.iterkeys():
        # There are 20 lines per zenith range
        flux_dict[key] = np.array(np.split(flux_dict[key], 20))

    # Set the zenith and energy range as they are in the tables
    # The energy may change, but the zenith should always be
    # 20 bins, full sky.
    flux_dict['energy'] = flux_dict['energy'][0]
    flux_dict['coszen'] = np.linspace(0.95, -0.95, 20)

    # Now get a spline representation of the flux table.
    logging.debug('Make spline representation of flux')
    logging.debug('Doing this integral-preserving.')

    spline_dict = {}

    # Do integral-preserving method as in IceCube's NuFlux
    # This one will be based purely on SciPy rather than ROOT
    # Stored splines will be 1D in integrated flux over energy
    int_flux_dict = {}
    # Energy and CosZenith bins needed for integral-preserving
    # method must be the edges of those of the normal tables
    int_flux_dict['logenergy'] = np.linspace(-1.025, 4.025, 102)
    int_flux_dict['coszen'] = np.linspace(-1, 1, 21)
    for nutype in PRIMARIES:
        # spline_dict now wants to be a set of splines for
        # every table cosZenith value.
        splines = {}
        CZiter = 1
        for energyfluxlist in flux_dict[nutype]:
            int_flux = []
            tot_flux = 0.0
            int_flux.append(tot_flux)
            for energyfluxval, energyval in zip(energyfluxlist,
                                                flux_dict['energy']):
                # Spline works best if you integrate flux * energy
                tot_flux += energyfluxval*energyval
                int_flux.append(tot_flux)

            spline = interpolate.splrep(int_flux_dict['logenergy'],
                                        int_flux, s=0)
            CZvalue = '%.2f'%(1.05-CZiter*0.1)
            splines[CZvalue] = spline
            CZiter += 1

        spline_dict[nutype] = splines

    return spline_dict


def calculate_flux_weights(true_energies, true_coszens, en_splines):
    """Calculate flux weights for given array of energy and cos(zenith).

    Arrays of true energy and zenith are expected to be for MC events, so
    they are tested to be of the same length.
    En_splines should be the spline for the primary of interest. The entire
    dictionary is calculated in the previous function.

    Parameters
    ----------
    true_energies : list or numpy array
        A list of the true energies of your MC events
    true_coszens : list or numpy array
        A list of the true coszens of your MC events
    en_splines : list of splines
        A list of the initialised energy splines from the previous function
        for your desired primary.

    Example
    -------
    Use the previous function to calculate the spline dict for the South Pole.

        spline_dict = load_2D_table('flux/honda-2015-spl-solmax-aa.d')

    Then you must have some equal length arrays of energy and zenith.

        ens = [3.0, 4.0, 5.0]
        czs = [-0.4, 0.7, 0.3]

    These are used in this function, along with whatever primary you are
    interested in calculating the flux weights for.

        flux_weights = calculate_flux_weights(ens, czs, spline_dict['numu'])

    Done!

    """
    if not isinstance(true_energies, np.ndarray):
        if not isinstance(true_energies, list):
            raise TypeError('true_energies must be a list or numpy array')
        else:
            true_energies = np.array(true_energies)
    if not isinstance(true_coszens, np.ndarray):
        if not isinstance(true_coszens, list):
            raise TypeError('true_coszens must be a list or numpy array')
        else:
            true_coszens = np.array(true_coszens)
    if not ((true_coszens >= -1.0).all() and (true_coszens <= 1.0).all()):
        raise ValueError('Not all coszens found between -1 and 1')
    if not len(true_energies) == len(true_coszens):
        raise ValueError('length of energy and coszen arrays must match')

    czkeys = ['%.2f'%x for x in np.linspace(-0.95, 0.95, 20)]
    cz_spline_points = np.linspace(-1, 1, 21)

    flux_weights = []
    for true_energy, true_coszen in zip(true_energies, true_coszens):
        true_log_energy = np.log10(true_energy)
        spline_vals = [0]
        for czkey in czkeys:
            # Have to multiply by bin widths to get correct derivatives
            # Here the bin width is 0.05 (in log energy)
            spval = interpolate.splev(true_log_energy,
                                      en_splines[czkey],
                                      der=1)*0.05
            spline_vals.append(spval)
        spline_vals = np.array(spline_vals)
        int_spline_vals = np.cumsum(spline_vals)
        spline = interpolate.splrep(cz_spline_points,
                                        int_spline_vals, s=0)
        flux_weights.append(interpolate.splev(true_coszen,
                                              spline,
                                              der=1)*(0.1/true_energy))

    flux_weights = np.array(flux_weights)
    return flux_weights


if __name__ == '__main__':
    """
    This is a slightly longer example than that given in the docstring of the
    calculate_flux_weights function. This will make a quick plot of the flux
    at 5.0 GeV and 20.0 GeV across all of cos(zenith) for NuMu just to make
    sure everything looks sensible.
    """
    import matplotlib
    matplotlib.use('pdf')
    from matplotlib import pyplot as plt

    spline_dict = load_2D_table('flux/honda-2015-spl-solmax-aa.d')
    czs = np.linspace(-1,1,81)
    low_ens = 5.0*np.ones_like(czs)
    high_ens = 20.0*np.ones_like(czs)

    low_en_flux_weights = calculate_flux_weights(low_ens,
                                                 czs,
                                                 spline_dict['numu'])

    high_en_flux_weights = calculate_flux_weights(high_ens,
                                                  czs,
                                                  spline_dict['numu'])

    plt.plot(czs, low_en_flux_weights)
    plt.xlabel('cos(zenith)')
    plt.ylabel('NuMu Flux at 5.0 GeV')
    plt.savefig('/tmp/fluxweightstest5GeV.pdf')
    plt.close()

    plt.plot(czs, high_en_flux_weights)
    plt.xlabel('cos(zenith)')
    plt.ylabel('NuMu Flux at 20.0 GeV')
    plt.savefig('/tmp/fluxweightstest20GeV.pdf')
    plt.close()
