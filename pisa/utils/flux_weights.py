#! /usr/bin/env python

"""
A set of functions for calculating flux weights given an array of energy and
cos(zenith) values based on the Honda atmospheric flux tables. A lot of this
functionality will be copied from honda.py but since I don't want to initialise
this as a stage it makes sense to copy it in to here so somebody can't
accidentally do the wrong thing with that script.
"""


from __future__ import absolute_import, division

import numpy as np
import scipy.interpolate as interpolate

from pisa.utils.log import logging
from pisa.utils.resources import open_resource


__all__ = ['load_2d_honda_table', 'load_2d_bartol_table', 'load_2d_table',
           'calculate_2d_flux_weights', 'load_3d_honda_table', 'load_3d_table',
           'calculate_3d_flux_weights', ]

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


PRIMARIES = ['numu', 'numubar', 'nue', 'nuebar']
TEXPRIMARIES = [r'$\nu_{\mu}$', r'$\bar{\nu}_{\mu}$', r'$\nu_{e}$',
                r'$\bar{\nu}_{e}$']


def load_2d_honda_table(flux_file, enpow=1, return_table=False):

    logging.debug("Loading atmospheric flux table %s", flux_file)

    # columns in Honda files are in the same order
    cols = ['energy'] + PRIMARIES

    # Load the data table
    table = np.genfromtxt(open_resource(flux_file),
                          usecols=list(range(len(cols))))
    mask = np.all(np.isnan(table) | np.equal(table, 0), axis=1)
    table = table[~mask].T

    flux_dict = dict(zip(cols, table))
    for key in flux_dict.keys():
        # There are 20 lines per zenith range
        flux_dict[key] = np.array(np.split(flux_dict[key], 20))

    # Set the zenith and energy range as they are in the tables
    # The energy may change, but the zenith should always be
    # 20 bins, full sky.
    flux_dict['energy'] = flux_dict['energy'][0]
    flux_dict['coszen'] = np.linspace(-0.95, 0.95, 20)

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
        cz_iter = 1
        for energyfluxlist in flux_dict[nutype]:
            int_flux = []
            tot_flux = 0.0
            int_flux.append(tot_flux)
            for energyfluxval, energyval in zip(energyfluxlist,
                                                flux_dict['energy']):
                # Spline works best if you integrate flux * energy
                tot_flux += energyfluxval*np.power(energyval, enpow)*0.05
                int_flux.append(tot_flux)

            spline = interpolate.splrep(int_flux_dict['logenergy'],
                                        int_flux, s=0)
            cz_value = '%.2f'%(1.05-cz_iter*0.1)
            splines[cz_value] = spline
            cz_iter += 1

        spline_dict[nutype] = splines

    for prim in PRIMARIES:
        flux_dict[prim] = flux_dict[prim][::-1]

    if return_table:
        return spline_dict, flux_dict

    return spline_dict


def load_2d_bartol_table(flux_file, enpow=1, return_table=False):

    logging.debug("Loading atmospheric flux table %s", flux_file)

    # Bartol tables have been modified to look like Honda tables
    cols = ['energy'] + PRIMARIES

    # Load the data table
    table = np.genfromtxt(open_resource(flux_file),
                          usecols=list(range(len(cols))))
    mask = np.all(np.isnan(table) | np.equal(table, 0), axis=1)
    table = table[~mask].T

    flux_dict = dict(zip(cols, table))
    for key in flux_dict.keys():
        # There are 20 lines per zenith range
        flux_dict[key] = np.array(np.split(flux_dict[key], 20))

    # Set the zenith and energy range as they are in the tables
    # The energy may change, but the zenith should always be
    # 20 bins, full sky.
    flux_dict['energy'] = flux_dict['energy'][0]
    flux_dict['coszen'] = np.linspace(-0.95, 0.95, 20)

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
    low_log_energy = np.linspace(-1, 1, 41)
    high_log_energy = np.linspace(1.1, 4, 30)
    int_flux_dict['logenergy'] = np.concatenate(
        [low_log_energy, high_log_energy]
    )
    int_flux_dict['coszen'] = np.linspace(-1, 1, 21)
    for nutype in PRIMARIES:
        # spline_dict now wants to be a set of splines for
        # every table cosZenith value.
        splines = {}
        cz_iter = 1
        for energyfluxlist in flux_dict[nutype]:
            int_flux = []
            tot_flux = 0.0
            int_flux.append(tot_flux)
            for energyfluxval, energyval in zip(energyfluxlist,
                                                flux_dict['energy']):
                # Spline works best if you integrate flux * energy
                if energyval < 10.0:
                    tot_flux += energyfluxval*np.power(energyval, enpow)*0.05
                else:
                    tot_flux += energyfluxval*np.power(energyval, enpow)*0.1
                int_flux.append(tot_flux)

            spline = interpolate.splrep(int_flux_dict['logenergy'],
                                        int_flux, s=0)
            cz_value = '%.2f'%(1.05-cz_iter*0.1)
            splines[cz_value] = spline
            cz_iter += 1

        spline_dict[nutype] = splines

    for prim in PRIMARIES:
        flux_dict[prim] = flux_dict[prim][::-1]

    if return_table:
        return spline_dict, flux_dict

    return spline_dict


def load_2d_table(flux_file, enpow=1, return_table=False):
    """Manipulate 2 dimensional flux tables.

    2D is expected to mean energy and cosZenith, where azimuth is averaged
    over (before being stored in the table) and the zenith range should
    include both hemispheres.

    Parameters
    ----------
    flux_file : string
        The location of the flux file you want to spline. Should be a honda
        azimuth-averaged file.
    enpow : integer
        The power to which the energy will be raised in the construction of the
        splines. If you don't know what this means, leave it as 1.
    return_table : boolean
        Flag to true if you want the function to also return a dictionary
        of the underlying values from the tables. Useful for comparisons.

    """
    if not isinstance(enpow, int):
        raise TypeError('Energy power must be an integer')
    if not isinstance(return_table, bool):
        raise TypeError('return_table must be a boolean')
    if not isinstance(flux_file, str):
        raise TypeError('Flux file name must be a string')
    if 'aa' not in flux_file:
        raise ValueError('Azimuth-averaged tables are expected')
    if 'honda' not in flux_file:
        if 'bartol' in flux_file:
            if return_table:
                spline_dict, flux_dict = load_2d_bartol_table(flux_file,
                                                              enpow=enpow,
                                                              return_table=True)
            else:
                spline_dict = load_2d_bartol_table(flux_file,
                                                   enpow=enpow)
            spline_dict['name'] = 'bartol'

        else:
            raise ValueError('Flux file must be from the Honda or '
                             'Bartol groups')
    else:
        if return_table:
            spline_dict, flux_dict = load_2d_honda_table(flux_file,
                                                         enpow=enpow,
                                                         return_table=True)
        else:
            spline_dict = load_2d_honda_table(flux_file, enpow=enpow)
        spline_dict['name'] = 'honda'

    if return_table:
        return spline_dict, flux_dict

    return spline_dict


def calculate_2d_flux_weights(true_energies, true_coszens, en_splines,
                              enpow=1, out=None):
    """Calculate flux weights for given array of energy and cos(zenith).
    Arrays of true energy and zenith are expected to be for MC events, so
    they are tested to be of the same length.
    `en_splines` should be the spline for the primary of interest. The entire
    dictionary is calculated in the previous function.

    Parameters
    ----------
    true_energies : list or numpy array
        A list of the true energies of your MC events. Pass this in GeV!
    true_coszens : list or numpy array
        A list of the true coszens of your MC events
    en_splines : list of splines
        A list of the initialised energy splines from the previous function
        for your desired primary.
    enpow : integer
        The power to which the energy was raised in the construction of the
        splines. If you don't know what this means, leave it as 1.
    out : np.array
        optional array to store results

    Example
    -------
    Use the previous function to calculate the spline dict for the South Pole.

        spline_dict = load_2d_table('flux/honda-2015-spl-solmax-aa.d')

    Then you must have some equal length arrays of energy and zenith.

        ens = [3.0, 4.0, 5.0]
        czs = [-0.4, 0.7, 0.3]

    These are used in this function, along with whatever primary you are
    interested in calculating the flux weights for.

        flux_weights = calculate_2d_flux_weights(ens, czs, spline_dict['numu'])

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
    if not isinstance(enpow, int):
        raise TypeError('Energy power must be an integer')

    num_cz_points = 20
    czkeys = ['%.2f'%x for x in np.linspace(-0.95, 0.95, num_cz_points)]
    cz_spline_points = np.linspace(-1, 1, num_cz_points+1)

    if out is None:
        out = np.empty_like(true_energies)

    spline_vals = np.zeros(num_cz_points+1)
    for i in range(len(true_energies)):
        true_log_energy = np.log10(true_energies[i])
        for j in range(num_cz_points):
            spline_vals[j+1] = interpolate.splev(true_log_energy,
                                                 en_splines[czkeys[j]],
                                                 der=1)

        int_spline_vals = np.cumsum(spline_vals)*0.1
        spline = interpolate.splrep(cz_spline_points,
                                    int_spline_vals, s=0)

        out[i] = interpolate.splev(true_coszens[i], spline, der=1) / np.power(true_energies[i], enpow)

    return out


def load_3d_honda_table(flux_file, enpow=1, return_table=False):

    logging.debug("Loading atmospheric flux table %s", flux_file)

    # columns in Honda files are in the same order
    cols = ['energy'] + PRIMARIES

    # Load the data table
    table = np.genfromtxt(open_resource(flux_file),
                          usecols=list(range(len(cols))))
    mask = np.all(np.isnan(table) | np.equal(table, 0), axis=1)
    table = table[~mask].T

    flux_dict = dict(zip(cols, table))
    for key in flux_dict.keys():
        # There are 20 lines per zenith range
        coszenith_lists = np.array(np.split(flux_dict[key], 20))
        azimuth_lists = []
        for coszenith_list in coszenith_lists:
            azimuth_lists.append(np.array(np.split(coszenith_list, 12)).T)
        flux_dict[key] = np.array(azimuth_lists)
        if not key == 'energy':
            flux_dict[key] = flux_dict[key].T

    # Set the zenith and energy range as they are in the tables
    # The energy may change, but the zenith should always be
    # 20 bins and the azimuth should always be 12 bins, full sky
    flux_dict['energy'] = flux_dict['energy'][0].T[0]
    flux_dict['coszen'] = np.linspace(0.95, -0.95, 20)
    flux_dict['azimuth'] = np.linspace(15, 345, 12)

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
    int_flux_dict['coszen'] = np.linspace(1, -1, 21)
    for nutype in PRIMARIES:
        # spline_dict now wants to be a set of splines for
        # every table cosZenith value.
        # In 3D mode we have a set of these sets for every
        # table azimuth value.
        az_splines = {}
        for az, f in zip(flux_dict['azimuth'], flux_dict[nutype]):
            splines = {}
            cz_iter = 1
            for energyfluxlist in f.T:
                int_flux = []
                tot_flux = 0.0
                int_flux.append(tot_flux)
                for energyfluxval, energyval in zip(energyfluxlist,
                                                    flux_dict['energy']):
                    # Spline works best if you integrate flux * energy
                    tot_flux += energyfluxval*np.power(energyval, enpow)*0.05
                    int_flux.append(tot_flux)

                spline = interpolate.splrep(int_flux_dict['logenergy'],
                                            int_flux, s=0)
                cz_value = '%.2f'%(1.05-cz_iter*0.1)
                splines[cz_value] = spline
                cz_iter += 1

            az_splines[az] = splines

        spline_dict[nutype] = az_splines

    if return_table:
        return spline_dict, flux_dict

    return spline_dict


def load_3d_table(flux_file, enpow=1, return_table=False):
    """Manipulate 3 dimensional flux tables.

    3D is expected to mean energy, cosZenith and azimuth. The angular range
    should be fully sky.

    Parameters
    ----------
    flux_file : string
        The location of the flux file you want to spline. Should be a honda
        azimuth-averaged file.
    enpow : integer
        The power to which the energy will be raised in the construction of the
        splines. If you don't know what this means, leave it as 1.
    return_table : boolean
        Flag to true if you want the function to also return a dictionary
        of the underlying values from the tables. Useful for comparisons.
    """

    if not isinstance(enpow, int):
        raise TypeError('Energy power must be an integer')
    if not isinstance(return_table, bool):
        raise TypeError('return_table must be a boolean')
    if not isinstance(flux_file, str):
        raise ValueError('Flux file name must be a string')
    if 'aa' in flux_file:
        raise ValueError('Azimuth-dependent tables are expected')
    if 'honda' not in flux_file:
        raise ValueError('Flux file must be from the Honda group')
    if return_table:
        spline_dict, flux_dict = load_3d_honda_table(flux_file,
                                                     enpow=enpow,
                                                     return_table=True)
    else:
        spline_dict = load_3d_honda_table(flux_file,
                                          enpow=enpow)
    spline_dict['name'] = 'honda'

    if return_table:
        return spline_dict, flux_dict

    return spline_dict


def calculate_3d_flux_weights(true_energies, true_coszens, true_azimuths,
                              en_splines, enpow=1, az_linear=True):
    """Calculate flux weights for given array of energy, cos(zenith) and
    azimuth.

    Arrays of true energy, zenith and azimuth are expected to be for MC events,
    so they are tested to be of the same length.
    En_splines should be the spline for the primary of interest. The entire
    dictionary is calculated in the previous function.

    Parameters
    ----------
    true_energies : list or numpy array
        A list of the true energies of your MC events. Pass this in GeV!
    true_coszens : list or numpy array
        A list of the true coszens of your MC events
    true_azimuths : list or numpy array
        A list of the true azimuths of your MC events. Pass this in radians!
    en_splines : list of splines
        A list of the initialised energy splines from the previous function
        for your desired primary.
    enpow : integer
        The power to which the energy was raised in the construction of the
        splines. If you don't know what this means, leave it as 1.
    az_linear : boolean
        Whether or not to linearly interpolate in the azimuthal direction. If
        you don't know why this is an option, leave it as true.

    Example
    -------
    Use the previous function to calculate the spline dict for the South Pole.

        spline_dict = load_3d_table('flux/honda-2015-spl-solmax.d')

    Then you must have some equal length arrays of energy, zenith and azimuth.

        ens = [3.0, 4.0, 5.0]
        czs = [-0.4, 0.7, 0.3]
        azs = [0.3, 1.2, 2.1]

    These are used in this function, along with whatever primary you are
    interested in calculating the flux weights for.

        flux_weights = calculate_3d_flux_weights(ens,
                                                 czs,
                                                 azs,
                                                 spline_dict['numu'])

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
    if not isinstance(true_azimuths, np.ndarray):
        if not isinstance(true_azimuths, list):
            raise TypeError('true_azimuths must be a list or numpy array')
        else:
            true_azimuths = np.array(true_azimuths)
    if not ((true_coszens >= -1.0).all() and (true_coszens <= 1.0).all()):
        raise ValueError('Not all coszens found between -1 and 1')
    ensczs_match = len(true_energies) == len(true_coszens)
    ensazs_match = len(true_energies) == len(true_azimuths)
    if not (ensczs_match and ensazs_match):
        raise ValueError('length of energy, coszen and azimuth arrays must '
                         'match')
    if not (true_azimuths >= 0.0).all():
        raise ValueError('Azimuths should be given as the angle, so should '
                         'all be positive')

    azkeys = np.linspace(15.0, 345.0, 12)
    if not az_linear:
        az_spline_points = np.linspace(0.0, 360.0, 13)
    else:
        az_spline_points = np.linspace(15.0, 375.0, 13)
    czkeys = ['%.2f'%x for x in np.linspace(-0.95, 0.95, 20)]
    cz_spline_points = np.linspace(-1, 1, 21)

    flux_weights = []
    for true_energy, true_coszen, true_azimuth in zip(true_energies,
                                                      true_coszens,
                                                      true_azimuths):
        true_azimuth *= 180.0/np.pi
        true_log_energy = np.log10(true_energy)
        az_spline_vals = []
        for azkey in azkeys:
            cz_spline_vals = [0]
            for czkey in czkeys:
                spval = interpolate.splev(true_log_energy,
                                          en_splines[azkey][czkey],
                                          der=1)

                cz_spline_vals.append(spval)
            cz_spline_vals = np.array(cz_spline_vals)
            cz_int_spline_vals = np.cumsum(cz_spline_vals)*0.1
            cz_spline = interpolate.splrep(cz_spline_points,
                                           cz_int_spline_vals, s=0)
            az_spline_vals.append(interpolate.splev(true_coszen,
                                                    cz_spline,
                                                    der=1))
        # Treat the azimuthal dimension in an integral-preserving manner.
        # This is not recommended.
        if not az_linear:
            az_spline_vals = np.array(az_spline_vals)
            az_spline_vals = np.insert(az_spline_vals, 0, 0)
            az_int_spline_vals = np.cumsum(az_spline_vals)*30.0
            az_spline = interpolate.splrep(az_spline_points,
                                           az_int_spline_vals, s=0)
            flux_weights.append(
                interpolate.splev(
                    true_azimuth,
                    az_spline,
                    der=1
                )/np.power(true_energy, enpow)
            )
        # Treat the azimuthal dimension with a linear interpolation.
        # This is the best treatment.
        else:
            # Make the azimuthal spline cyclic
            az_spline_vals.append(az_spline_vals[0])
            # Account for the energy power that was applied in the first splines
            az_spline_vals /= np.power(true_energy, enpow)
            az_spline = interpolate.splrep(az_spline_points,
                                           az_spline_vals, k=1)
            if true_azimuth < 15.0:
                true_azimuth += 360.0
            flux_weights.append(
                interpolate.splev(
                    true_azimuth,
                    az_spline,
                    der=0
                )
            )
    flux_weights = np.array(flux_weights)
    return flux_weights


def main():
    """This is a slightly longer example than that given in the docstring of
    the calculate_flux_weights function. This will make a quick plot of the
    flux at 5.0 GeV and 20.0 GeV across all of cos(zenith) for NuMu just to
    make sure everything looks sensible.

    """
    from matplotlib import pyplot as plt

    spline_dict = load_2d_table('flux/honda-2015-spl-solmax-aa.d')
    czs = np.linspace(-1, 1, 81)
    low_ens = 5.0*np.ones_like(czs)
    high_ens = 20.0*np.ones_like(czs)

    low_en_flux_weights = calculate_2d_flux_weights(low_ens,
                                                    czs,
                                                    spline_dict['numu'])

    high_en_flux_weights = calculate_2d_flux_weights(high_ens,
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


if __name__ == '__main__':
    main()
