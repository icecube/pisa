#! /usr/bin/env python

"""
A set of tests on the flux weights calculated by PISA.
"""


from __future__ import absolute_import, division

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True
import matplotlib.colors as colors

from pisa.utils.log import logging, set_verbosity
from pisa.utils.flux_weights import (load_2d_table, calculate_2d_flux_weights,
                                     PRIMARIES, TEXPRIMARIES, load_3d_table,
                                     calculate_3d_flux_weights)

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


def plot_1d_slices(xintvals, yintvals, xtabvals, ytabvals, xtabbins, xlabel,
                   ylabel, xtext, ytext, text, table_name, save_name, log):
    """Test function to show interpolation and tables overlaid in 1D slices.
    See main function for how to use this function.

    Parameters
    ----------
    xintvals : list
        A list of the x points where the spline was evaluated.
    yintvals : list
        A list of the y points which the spline evaluated to.
    xtabvals : list
        A list of the x points where the table is defined.
    ytabvals : list
        A list of the y points where the table is defined.
    xtabbins : list
        A list of the bin edges. Should have xtabvals as the bin centres.
    xlabel : string
        A label for the x-axis of the plot.
    ylabel : string
        A label for the y-axis of the plot.
    xtext : float
        The position for the text label showing the slice along x.
    ytext : float
        The position for the text label showing the slice along y.
    text : string
        The text label showing the slice.
    table_name : string
        The text label naming the tables used
    save_name : string
        The place and name to save the plot.
    log : bool
        A boolean to whether the axes should be made logarithmic.
        Will do both.
    """

    plt.plot(xintvals,
             yintvals,
             color='r',
             linewidth=2,
             label='IP Interpolation')
    plt.hist(xtabvals,
             weights=ytabvals,
             bins=xtabbins,
             color='k',
             linewidth=2,
             histtype='step',
             label=table_name)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    if log:
        plt.xlim(xtabbins[0], xtabbins[-1])
        plt.xscale("log")
        plt.yscale("log")
        ymin = min(min(np.log10(yintvals)), min(np.log10(ytabvals)))
        ymax = max(max(np.log10(yintvals)), max(np.log10(ytabvals)))
        ydiff = ymax - ymin
        plt.xlim(min(xtabbins), max(xtabbins))
        plt.ylim(np.power(10, ymin-0.1*ydiff), np.power(10, ymax+0.1*ydiff))
        if 'numu' in save_name:
            plt.legend(loc='lower right')
        elif 'nue' in save_name:
            plt.legend(loc='lower left')
    else:
        ymin = min(min(yintvals), min(ytabvals))
        ymax = max(max(yintvals), max(ytabvals))
        ydiff = ymax-ymin
        plt.xlim(min(xtabbins), max(xtabbins))
        if min(xtabbins) == 0.0 and max(xtabbins) == 360.0:
            plt.ylim(ymin-0.1*ydiff, ymax+0.8*ydiff)
        else:
            plt.ylim(ymin-0.1*ydiff, ymax+0.1*ydiff)
        plt.legend(loc='upper right')
    plt.figtext(xtext,
                ytext,
                text,
                verticalalignment='center',
                horizontalalignment='center',
                color='k',
                fontsize=24)
    plt.savefig(save_name)
    plt.close()


def logplot(m, title, ax, clabel, cmap=plt.cm.afmhot, logz=True,
            largelabels=False, medlabels=False):
    """Simple plotting of a 2D histogram (map)"""
    hist = np.ma.masked_invalid(m['map'])
    if 'ebins' in m.keys():
        y = m['ebins']
        ylabel = r'Energy (GeV)'
        logy = True
        if 'czbins' in m.keys():
            x = m['czbins']
            xlabel = r'$\cos\theta_Z$'
        else:
            x = m['azbins']
            xlabel = r'$\phi_{Az}$ (${}^{\circ}$)'
    else:
        x = m['czbins']
        xlabel = r'$\cos\theta_Z$'
        y = m['azbins']
        ylabel = r'$\phi_{Az}$ (${}^{\circ}$)'
        logy = False
    X, Y = np.meshgrid(x, y)
    if logy:
        ax.set_yscale('log')
    vmin = hist.min()
    vmax = hist.max()
    if logz:
        pcmesh = ax.pcolormesh(X, Y, hist,
                               norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                               cmap=cmap)
    else:
        pcmesh = ax.pcolormesh(X, Y, hist,
                               norm=colors.Normalize(vmin=vmin, vmax=vmax),
                               cmap=cmap)
    cbar = plt.colorbar(mappable=pcmesh, ax=ax)
    if clabel is not None:
        if largelabels:
            cbar.set_label(clabel, labelpad=-1, fontsize=36)
            cbar.ax.tick_params(labelsize=36)
        elif medlabels:
            cbar.set_label(clabel, labelpad=-1, fontsize=36)
            cbar.ax.tick_params(labelsize=36)
        else:
            cbar.set_label(clabel, labelpad=-1)
            cbar.ax.tick_params(labelsize='large')
    if largelabels:
        ax.set_xlabel(xlabel, fontsize=36)
        ax.set_ylabel(ylabel, labelpad=-3, fontsize=36)
        ax.set_title(title, y=1.03, fontsize=36)
        plt.tick_params(axis='both', which='major', labelsize=36)
    elif medlabels:
        ax.set_xlabel(xlabel, fontsize=36)
        ax.set_ylabel(ylabel, labelpad=-3, fontsize=36)
        ax.set_title(title, y=1.03, fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=36)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, labelpad=-3)
        ax.set_title(title, y=1.03)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))


def take_average(interp_map, oversampling):
    average_map = interp_map.reshape(
        [len(interp_map)/oversampling,
         oversampling,
         len(interp_map[0])/oversampling,
         oversampling]
    ).mean(3).mean(1)
    return average_map


def do_1d_2d_honda_test(spline_dict, flux_dict, legend_filename,
                        save_name, outdir, enpow=1):

    czs = np.linspace(-1, 1, 81)
    low_ens = 5.0119*np.ones_like(czs)
    high_ens = 50.119*np.ones_like(czs)

    ens = np.logspace(-1.025, 4.025, 1020)
    upgoing = -0.95*np.ones_like(ens)
    downgoing = 0.35*np.ones_like(ens)

    for flav, flavtex in zip(PRIMARIES, TEXPRIMARIES):

        low_en_flux_weights = calculate_2d_flux_weights(low_ens,
                                                        czs,
                                                        spline_dict[flav],
                                                        enpow=enpow)

        high_en_flux_weights = calculate_2d_flux_weights(high_ens,
                                                         czs,
                                                         spline_dict[flav],
                                                         enpow=enpow)

        flux5 = flux_dict[flav].T[np.where(flux_dict['energy'] == 5.0119)][0]
        flux50 = flux_dict[flav].T[np.where(flux_dict['energy'] == 50.119)][0]

        plot_1d_slices(
            xintvals=czs,
            yintvals=low_en_flux_weights,
            xtabvals=flux_dict['coszen'],
            ytabvals=flux5,
            xtabbins=np.linspace(-1, 1, 21),
            xlabel=r'$\cos\theta_Z$',
            ylabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
            xtext=0.75,
            ytext=0.7,
            text='Slice at \n 5.0119 GeV',
            table_name=legend_filename,
            save_name=os.path.join(
                outdir, '%s_%sfluxweightstest5GeV.png'%(save_name, flav)
            ),
            log=False
        )

        plot_1d_slices(
            xintvals=czs,
            yintvals=high_en_flux_weights,
            xtabvals=flux_dict['coszen'],
            ytabvals=flux50,
            xtabbins=np.linspace(-1, 1, 21),
            xlabel=r'$\cos\theta_Z$',
            ylabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
            xtext=0.75,
            ytext=0.7,
            text='Slice at \n 50.119 GeV',
            table_name=legend_filename,
            save_name=os.path.join(
                outdir, '%s_%sfluxweightstest50GeV.png'%(save_name, flav)
            ),
            log=False
        )

        upgoing_flux_weights = calculate_2d_flux_weights(ens,
                                                         upgoing,
                                                         spline_dict[flav],
                                                         enpow=enpow)

        downgoing_flux_weights = calculate_2d_flux_weights(ens,
                                                           downgoing,
                                                           spline_dict[flav],
                                                           enpow=enpow)

        upgoing_flux_weights *= np.power(ens, 3)
        downgoing_flux_weights *= np.power(ens, 3)

        coszen_strs = ['%.2f'%coszen for coszen in flux_dict['coszen']]
        coszen_strs = np.array(coszen_strs)

        fluxupgoing = flux_dict[flav][np.where(coszen_strs == '-0.95')][0]
        fluxdowngoing = flux_dict[flav][np.where(coszen_strs == '0.35')][0]

        fluxupgoing *= np.power(flux_dict['energy'], 3)
        fluxdowngoing *= np.power(flux_dict['energy'], 3)

        if 'numu' in flav:
            xtext = 0.68
            ytext = 0.25
        elif 'nue' in flav:
            xtext = 0.35
            ytext = 0.25

        plot_1d_slices(
            xintvals=ens,
            yintvals=upgoing_flux_weights,
            xtabvals=flux_dict['energy'],
            ytabvals=fluxupgoing,
            xtabbins=np.logspace(-1.025, 4.025, 102),
            xlabel='Neutrino Energy (GeV)',
            ylabel=r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
            xtext=xtext,
            ytext=ytext,
            text=r'Slice at $\cos\theta_Z=-0.95$',
            table_name=legend_filename,
            save_name=os.path.join(
                outdir, '%s_%sfluxweightstest-0.95cz.png'%(save_name, flav)
            ),
            log=True
        )

        plot_1d_slices(
            xintvals=ens,
            yintvals=downgoing_flux_weights,
            xtabvals=flux_dict['energy'],
            ytabvals=fluxdowngoing,
            xtabbins=np.logspace(-1.025, 4.025, 102),
            xlabel='Neutrino Energy (GeV)',
            ylabel=r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
            xtext=xtext,
            ytext=ytext,
            text=r'Slice at $\cos\theta_Z=0.35$',
            table_name=legend_filename,
            save_name=os.path.join(
                outdir, '%s_%sfluxweightstest0.35cz.png'%(save_name, flav)
            ),
            log=True
        )


def do_2d_2d_honda_test(spline_dict, flux_dict, outdir, ip_checks,
                        oversample, save_name, title_filename, enpow=1):

    all_ens_bins = np.logspace(-1.025, 4.025, 101*oversample+1)
    all_log_ens_bins = np.linspace(-1.025, 4.025, 101*oversample+1)
    log_en_bin_width = all_log_ens_bins[1] - all_log_ens_bins[0]
    all_ens = np.logspace(all_log_ens_bins[0] + log_en_bin_width/2.0,
                          all_log_ens_bins[-1] - log_en_bin_width/2.0,
                          101*oversample)
    all_czs_bins = np.linspace(-1.0, 1.0, 20*oversample+1)
    cz_bin_width = all_czs_bins[1] - all_czs_bins[0]
    all_czs = np.linspace(all_czs_bins[0] + cz_bin_width/2.0,
                          all_czs_bins[-1] - cz_bin_width/2.0,
                          20*oversample)

    all_ens_mg, all_czs_mg = np.meshgrid(all_ens, all_czs)

    for flav, flavtex in zip(PRIMARIES, TEXPRIMARIES):

        logging.info('Doing 2D verification of %s 2D Honda tables', flav)

        all_flux_weights = calculate_2d_flux_weights(all_ens_mg.flatten(),
                                                     all_czs_mg.flatten(),
                                                     spline_dict[flav],
                                                     enpow=enpow)

        all_flux_weights = np.array(np.split(all_flux_weights,
                                             len(all_czs)))
        all_flux_weights_map = {}
        all_flux_weights_map['map'] = all_flux_weights.T
        all_flux_weights_map['ebins'] = all_ens_bins
        all_flux_weights_map['czbins'] = all_czs_bins

        gridspec_kw = dict(left=0.15, right=0.90, wspace=0.32)
        fig, axes = plt.subplots(nrows=1, ncols=1, gridspec_kw=gridspec_kw,
                                 sharex=False, sharey=False, figsize=(12, 10))

        logplot(m=all_flux_weights_map,
                title='Finely Interpolated %s Flux'%flavtex,
                ax=axes,
                clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                largelabels=True)

        fig.savefig(
            os.path.join(
                outdir,
                '%s_%s2dinterpolation.png'%(save_name, flav)
            )
        )

        if ip_checks:

            logging.info('Doing ip verification of %s 2D Honda tables', flav)

            downsampled_flux_map = {}
            downsampled_flux_map['map'] = take_average(
                all_flux_weights.T, oversample
            )
            downsampled_flux_map['ebins'] = np.logspace(-1.025, 4.025, 102)
            downsampled_flux_map['czbins'] = np.linspace(-1.0, 1.0, 21)

            honda_tables = {}
            honda_tables['map'] = flux_dict[flav].T
            honda_tables['ebins'] = np.logspace(-1.025, 4.025, 102)
            honda_tables['czbins'] = np.linspace(-1.0, 1.0, 21)

            diff_map = {}
            diff_map['map'] = honda_tables['map']-downsampled_flux_map['map']
            diff_map['ebins'] = np.logspace(-1.025, 4.025, 102)
            diff_map['czbins'] = np.linspace(-1.0, 1.0, 21)

            diff_ratio_map = {}
            diff_ratio_map['map'] = diff_map['map'] / honda_tables['map']
            diff_ratio_map['ebins'] = np.logspace(-1.025, 4.025, 102)
            diff_ratio_map['czbins'] = np.linspace(-1.0, 1.0, 21)

            gridspec_kw = dict(left=0.03, right=0.968, wspace=0.32)
            fig, axes = plt.subplots(nrows=1, ncols=5,
                                     gridspec_kw=gridspec_kw,
                                     sharex=False, sharey=False,
                                     figsize=(20, 5))

            logplot(m=all_flux_weights_map,
                    title='Oversampled by %i'%oversample,
                    ax=axes[0],
                    clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,)
            logplot(m=downsampled_flux_map,
                    title='Downsampled to Honda Binning',
                    ax=axes[1],
                    clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,)
            logplot(m=honda_tables,
                    title='Honda Tables',
                    ax=axes[2],
                    clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,)
            logplot(m=diff_map,
                    title='Difference',
                    ax=axes[3],
                    clabel=None,
                    logz=False)
            logplot(m=diff_ratio_map,
                    title='Percentage Difference',
                    ax=axes[4],
                    clabel=None,
                    logz=False)

            plt.suptitle(
                'Integral Preserving Tests for %s %s Flux Tables'
                %(flavtex, title_filename), fontsize=36
            )
            plt.subplots_adjust(top=0.8)
            fig.savefig(
                os.path.join(
                    outdir,
                    '%s_%siptest_fullrange.png'%(save_name, flav)
                )
            )
            plt.close(fig.number)

            gridspec_kw = dict(left=0.15, right=0.90, wspace=0.32)
            fig, axes = plt.subplots(nrows=1, ncols=1, gridspec_kw=gridspec_kw,
                                     sharex=False, sharey=False,
                                     figsize=(12, 10))

            diff_ratio_map['map'] = diff_ratio_map['map'] * 100.0
            logplot(
                m=diff_ratio_map,
                title='%s Flux Integral Deviation'%flavtex,
                ax=axes,
                clabel=r'Ratio to Honda Table Value (\%)',
                largelabels=True,
                logz=False
            )

            fig.savefig(
                os.path.join(
                    outdir,
                    '%s_%s2dintegraldeviation.png'%(save_name, flav)
                )
            )




def do_1d_2d_bartol_test(spline_dict, flux_dict, outdir, enpow=1):

    czs = np.linspace(-1, 1, 81)
    low_ens = 4.732*np.ones_like(czs)
    high_ens = 44.70*np.ones_like(czs)

    ens = np.logspace(-1, 4, 701)
    upgoing = -0.95*np.ones_like(ens)
    downgoing = 0.35*np.ones_like(ens)

    for flav, flavtex in zip(PRIMARIES, TEXPRIMARIES):

        low_en_flux_weights = calculate_2d_flux_weights(low_ens,
                                                        czs,
                                                        spline_dict[flav],
                                                        enpow=enpow)

        high_en_flux_weights = calculate_2d_flux_weights(high_ens,
                                                         czs,
                                                         spline_dict[flav],
                                                         enpow=enpow)

        flux5 = flux_dict[flav].T[np.where(flux_dict['energy'] == 4.732)][0]
        flux50 = flux_dict[flav].T[np.where(flux_dict['energy'] == 44.70)][0]

        plot_1d_slices(
            xintvals=czs,
            yintvals=low_en_flux_weights,
            xtabvals=flux_dict['coszen'],
            ytabvals=flux5,
            xtabbins=np.linspace(-1, 1, 21),
            xlabel=r'$\cos\theta_Z$',
            ylabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
            xtext=0.75,
            ytext=0.7,
            text='Slice at \n 4.732 GeV',
            table_name='Bartol SNO 2004',
            save_name=os.path.join(
                outdir, 'bartol_%sfluxweightstest5GeV.png'%flav
            ),
            log=False
        )

        plot_1d_slices(
            xintvals=czs,
            yintvals=high_en_flux_weights,
            xtabvals=flux_dict['coszen'],
            ytabvals=flux50,
            xtabbins=np.linspace(-1, 1, 21),
            xlabel=r'$\cos\theta_Z$',
            ylabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
            xtext=0.75,
            ytext=0.7,
            text='Slice at \n 44.70 GeV',
            table_name='Bartol SNO 2004',
            save_name=os.path.join(
                outdir, 'bartol_%sfluxweightstest50GeV.png'%flav
            ),
            log=False
        )

        upgoing_flux_weights = calculate_2d_flux_weights(ens,
                                                         upgoing,
                                                         spline_dict[flav],
                                                         enpow=enpow)

        downgoing_flux_weights = calculate_2d_flux_weights(ens,
                                                           downgoing,
                                                           spline_dict[flav],
                                                           enpow=enpow)

        upgoing_flux_weights *= np.power(ens, 3)
        downgoing_flux_weights *= np.power(ens, 3)

        coszen_strs = ['%.2f'%coszen for coszen in flux_dict['coszen']]
        coszen_strs = np.array(coszen_strs)

        fluxupgoing = flux_dict[flav][np.where(coszen_strs == '-0.95')][0]
        fluxdowngoing = flux_dict[flav][np.where(coszen_strs == '0.35')][0]

        fluxupgoing *= np.power(flux_dict['energy'], 3)
        fluxdowngoing *= np.power(flux_dict['energy'], 3)

        low_log_energy = np.logspace(-1, 1, 41)
        high_log_energy = np.logspace(1.1, 4, 30)
        xtabbins = np.concatenate(
            [low_log_energy, high_log_energy]
        )

        if 'numu' in flav:
            xtext = 0.68
            ytext = 0.25
        elif 'nue' in flav:
            xtext = 0.35
            ytext = 0.25

        plot_1d_slices(
            xintvals=ens,
            yintvals=upgoing_flux_weights,
            xtabvals=flux_dict['energy'],
            ytabvals=fluxupgoing,
            xtabbins=xtabbins,
            xlabel='Neutrino Energy (GeV)',
            ylabel=r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
            xtext=xtext,
            ytext=ytext,
            text=r'Slice at $\cos\theta_Z=-0.95$',
            table_name='Bartol SNO 2004',
            save_name=os.path.join(
                outdir, 'bartol_%sfluxweightstest-0.95cz.png'%flav
            ),
            log=True
        )

        plot_1d_slices(
            xintvals=ens,
            yintvals=downgoing_flux_weights,
            xtabvals=flux_dict['energy'],
            ytabvals=fluxdowngoing,
            xtabbins=xtabbins,
            xlabel='Neutrino Energy (GeV)',
            ylabel=r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
            xtext=xtext,
            ytext=ytext,
            text=r'Slice at $\cos\theta_Z=0.35$',
            table_name='Bartol SNO 2004',
            save_name=os.path.join(
                outdir, 'bartol_%sfluxweightstest0.35cz.png'%flav
            ),
            log=True
        )


def do_2d_2d_bartol_test(spline_dict, flux_dict, outdir, ip_checks,
                         oversample, enpow=1):

    all_en_bins_low = np.logspace(-1.0, 1.0, 40*oversample+1)
    all_log_en_bins_low = np.linspace(-1.0, 1.0, 40*oversample+1)
    log_en_bin_width_low = all_log_en_bins_low[1] - all_log_en_bins_low[0]
    all_ens_low = np.logspace(
        all_log_en_bins_low[0]+log_en_bin_width_low/2.0,
        all_log_en_bins_low[-1]-log_en_bin_width_low/2.0,
        40*oversample
    )

    all_en_bins_high = np.logspace(1.0, 4.0, 30*oversample+1)
    all_log_en_bins_high = np.linspace(1.0, 4.0, 30*oversample+1)
    log_en_bin_width_high = all_log_en_bins_high[1] - all_log_en_bins_high[0]
    all_ens_high = np.logspace(
        all_log_en_bins_high[0]+log_en_bin_width_high/2.0,
        all_log_en_bins_high[-1]-log_en_bin_width_high/2.0,
        30*oversample
    )

    all_en_bins = [all_en_bins_low, all_en_bins_high]
    all_ens = [all_ens_low, all_ens_high]

    all_all_ens_bins = np.concatenate(
        (
            np.logspace(-1.0, 1.0, 40*oversample+1),
            np.logspace(1.0+log_en_bin_width_high, 4.0, 30*oversample)
        )
    )

    all_czs_bins = np.linspace(-1.0, 1.0, 20*oversample+1)
    cz_bin_width = all_czs_bins[1] - all_czs_bins[0]
    all_czs = np.linspace(
        all_czs_bins[0] + cz_bin_width/2.0,
        all_czs_bins[-1] - cz_bin_width/2.0,
        20*oversample
    )

    en_labels = ['3DCalc', '1DCalc']

    all_fluxes = {}
    for flav in PRIMARIES:
        all_fluxes[flav] = []

    for all_en, all_ens_bins, en_label in zip(all_ens,
                                              all_en_bins,
                                              en_labels):
        all_ens_mg, all_czs_mg = np.meshgrid(all_en, all_czs)

        for flav, flavtex in zip(PRIMARIES, TEXPRIMARIES):

            logging.info('Doing 2D verification of %s 2D Bartol tables', flav)
            logging.info('Doing %s segment', en_label)

            all_flux_weights = calculate_2d_flux_weights(all_ens_mg.flatten(),
                                                         all_czs_mg.flatten(),
                                                         spline_dict[flav],
                                                         enpow=enpow)

            all_flux_weights = np.array(np.split(all_flux_weights,
                                                 len(all_czs)))

            if len(all_fluxes[flav]) == 0:
                all_fluxes[flav] = all_flux_weights.T
            else:
                all_fluxes[flav] = np.concatenate((all_fluxes[flav],
                                                   all_flux_weights.T))


            all_flux_weights_map = {}
            all_flux_weights_map['map'] = all_flux_weights.T
            all_flux_weights_map['ebins'] = all_ens_bins
            all_flux_weights_map['czbins'] = all_czs_bins

            gridspec_kw = dict(left=0.15, right=0.90, wspace=0.32)
            fig, axes = plt.subplots(nrows=1, ncols=1, gridspec_kw=gridspec_kw,
                                     sharex=False, sharey=False, figsize=(12, 10))

            logplot(m=all_flux_weights_map,
                    title='Finely Interpolated %s Flux'%flavtex,
                    ax=axes,
                    clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                    largelabels=True)

            fig.savefig(os.path.join(outdir,
                                     'bartol_%s_%s2dinterpolation.png'%(en_label, flav)))

            if ip_checks:

                logging.info('Doing ip verification of %s 2D Bartol '
                             'tables', flav)
                logging.info('Doing %s segment', en_label)

                bartol_tables = {}
                if en_label == '3DCalc':
                    bartol_tables['map'] = flux_dict[flav].T[:40]
                    bartol_tables['ebins'] = np.logspace(-1.0, 1.0, 41)
                elif en_label == '1DCalc':
                    bartol_tables['map'] = flux_dict[flav].T[40:]
                    bartol_tables['ebins'] = np.logspace(1.0, 4.0, 31)
                bartol_tables['czbins'] = np.linspace(-1.0, 1.0, 21)

                downsampled_flux_map = {}
                downsampled_flux_map['map'] = take_average(
                    all_flux_weights.T, oversample
                )
                downsampled_flux_map['ebins'] = bartol_tables['ebins']
                downsampled_flux_map['czbins'] = np.linspace(-1.0, 1.0, 21)

                diff_map = {}
                diff_map['map'] = bartol_tables['map']-downsampled_flux_map['map']
                diff_map['ebins'] = bartol_tables['ebins']
                diff_map['czbins'] = np.linspace(-1.0, 1.0, 21)

                diff_ratio_map = {}
                diff_ratio_map['map'] = diff_map['map'] / bartol_tables['map']
                diff_ratio_map['ebins'] = bartol_tables['ebins']
                diff_ratio_map['czbins'] = np.linspace(-1.0, 1.0, 21)

                gridspec_kw = dict(left=0.03, right=0.968, wspace=0.32)
                fig, axes = plt.subplots(nrows=1, ncols=5,
                                         gridspec_kw=gridspec_kw,
                                         sharex=False, sharey=False,
                                         figsize=(20, 5))

                logplot(m=all_flux_weights_map,
                        title='Oversampled by %i'%oversample,
                        ax=axes[0],
                        clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,)
                logplot(m=downsampled_flux_map,
                        title='Downsampled to Bartol Binning',
                        ax=axes[1],
                        clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,)
                logplot(m=bartol_tables,
                        title='Bartol Tables',
                        ax=axes[2],
                        clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,)
                logplot(m=diff_map,
                        title='Difference',
                        ax=axes[3],
                        clabel=None,
                        logz=False)
                logplot(m=diff_ratio_map,
                        title='Percentage Difference',
                        ax=axes[4],
                        clabel=None,
                        logz=False)

                plt.suptitle('Integral Preserving Tests for %s Bartol Sudbury 2015 Flux Tables'%flavtex, fontsize=36)
                plt.subplots_adjust(top=0.8)
                fig.savefig(os.path.join(outdir, 'bartol_%s_%siptest_fullrange.png'%(en_label, flav)))
                plt.close(fig.number)

                gridspec_kw = dict(left=0.15, right=0.90, wspace=0.32)
                fig, axes = plt.subplots(nrows=1, ncols=1,
                                         gridspec_kw=gridspec_kw,
                                         sharex=False, sharey=False,
                                         figsize=(12, 10))

                diff_ratio_map['map'] = diff_ratio_map['map'] * 100.0
                logplot(
                    m=diff_ratio_map,
                    title='%s Flux Integral Deviation'%flavtex,
                    ax=axes,
                    clabel=r'Ratio to Bartol Table Value (\%)',
                    largelabels=True,
                    logz=False
                )

                fig.savefig(
                    os.path.join(
                        outdir,
                        'bartol_%s_%s2dintegraldeviation_fullrange.png'%(
                            en_label, flav)
                    )
                )

    for flav, flavtex in zip(PRIMARIES, TEXPRIMARIES):

        all_flux_weights_map = {}
        all_flux_weights_map['map'] = all_fluxes[flav]
        all_flux_weights_map['ebins'] = all_all_ens_bins
        all_flux_weights_map['czbins'] = all_czs_bins

        gridspec_kw = dict(left=0.15, right=0.90, wspace=0.32)
        fig, axes = plt.subplots(nrows=1, ncols=1, gridspec_kw=gridspec_kw,
                                 sharex=False, sharey=False, figsize=(12, 10))

        logplot(m=all_flux_weights_map,
                title='Finely Interpolated %s Flux'%flavtex,
                ax=axes,
                clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                largelabels=True)

        fig.savefig(os.path.join(outdir,
                                 'bartol_%s2dinterpolation.png'%flav))


def do_2d_2d_comparisons(honda_spline_dict, bartol_spline_dict,
                         outdir, oversample, enpow=1):

    all_ens_bins = np.logspace(-1.0, 4.0, 100*oversample+1)
    all_czs_bins = np.linspace(-1.0, 1.0, 20*oversample+1)
    # need log energy bin width for defining evaluation points
    log_en_bin_width = np.linspace(-1.0, 4.0, 100*oversample+1)[1]-np.linspace(-1.0, 4.0, 100*oversample+1)[0]
    cz_bin_width = all_czs_bins[1]-all_czs_bins[0]
    all_ens = np.logspace(-1.0+log_en_bin_width/2.0,
                          4.0-log_en_bin_width/2.0,
                          100*oversample)
    all_czs = np.linspace(-1.0+cz_bin_width/2.0,
                          1.0-cz_bin_width/2.0,
                          20*oversample)

    all_ens_mg, all_czs_mg = np.meshgrid(all_ens, all_czs)

    for flav, flavtex in zip(PRIMARIES, TEXPRIMARIES):

        logging.info('Doing 2D comparisons of %s 2D Honda/Bartol tables', flav)

        honda_flux_weights = calculate_2d_flux_weights(
            all_ens_mg.flatten(),
            all_czs_mg.flatten(),
            honda_spline_dict[flav],
            enpow=enpow
        )
        bartol_flux_weights = calculate_2d_flux_weights(
            all_ens_mg.flatten(),
            all_czs_mg.flatten(),
            bartol_spline_dict[flav],
            enpow=enpow
        )

        honda_flux_weights = np.array(np.split(honda_flux_weights,
                                               len(all_czs)))
        bartol_flux_weights = np.array(np.split(bartol_flux_weights,
                                                len(all_czs)))

        honda_flux_weights_map = {}
        honda_flux_weights_map['map'] = honda_flux_weights.T
        honda_flux_weights_map['ebins'] = all_ens_bins
        honda_flux_weights_map['czbins'] = all_czs_bins

        bartol_flux_weights_map = {}
        bartol_flux_weights_map['map'] = bartol_flux_weights.T
        bartol_flux_weights_map['ebins'] = all_ens_bins
        bartol_flux_weights_map['czbins'] = all_czs_bins

        diff_map = {}
        diff_map['map'] = honda_flux_weights_map['map']-bartol_flux_weights_map['map']
        diff_map['ebins'] = all_ens_bins
        diff_map['czbins'] = all_czs_bins

        diff_ratio_map = {}
        diff_ratio_map['map'] = diff_map['map'] / honda_flux_weights_map['map']
        diff_ratio_map['ebins'] = all_ens_bins
        diff_ratio_map['czbins'] = all_czs_bins

        gridspec_kw = dict(left=0.03, right=0.968, wspace=0.32)
        fig, axes = plt.subplots(nrows=1, ncols=4,
                                 gridspec_kw=gridspec_kw,
                                 sharex=False, sharey=False,
                                 figsize=(16, 5))

        logplot(m=honda_flux_weights_map,
                title='Honda SNO 2015 %s Flux'%flavtex,
                ax=axes[0],
                clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex)
        logplot(m=bartol_flux_weights_map,
                title='Bartol SNO 2004 %s Flux'%flavtex,
                ax=axes[1],
                clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex)
        logplot(m=diff_map,
                title='Difference',
                ax=axes[2],
                clabel=None,
                logz=False)
        logplot(m=diff_ratio_map,
                title='Percentage Difference to Honda',
                ax=axes[3],
                clabel=None,
                logz=False)

        plt.suptitle('Comparisons for %s Honda 2015 and Bartol 2004 Sudbury Flux Tables'%flavtex, fontsize=36)
        plt.subplots_adjust(top=0.8)

        fig.savefig(os.path.join(outdir,
                                 'honda_bartol_%s2dcomparisons.png'%flav))


def do_1d_3d_honda_test(spline_dict, flux_dict, legend_filename,
                        save_name, outdir, enpow=1):

    czs = np.linspace(-1, 1, 81)
    low_ens = 5.0119*np.ones_like(czs)
    high_ens = 50.119*np.ones_like(czs)
    low_azs = 75.0*np.ones_like(czs)*np.pi/180.0
    high_azs = 285.0*np.ones_like(czs)*np.pi/180.0

    ens = np.logspace(-1.025, 4.025, 1020)
    upgoing = -0.95*np.ones_like(ens)
    downgoing = 0.35*np.ones_like(ens)
    low_azs_two = 75.0*np.ones_like(ens)*np.pi/180.0
    high_azs_two = 285.0*np.ones_like(ens)*np.pi/180.0

    azs = np.linspace(0.0, 360.0, 121)*np.pi/180.0
    low_ens_two = 5.0119*np.ones_like(azs)
    high_ens_two = 50.119*np.ones_like(azs)
    upgoing_two = -0.95*np.ones_like(azs)
    downgoing_two = 0.35*np.ones_like(azs)

    lin_azs = [True, False]
    name_additions = ['lin_az', 'ip_az']

    for lin_az, name_addition in zip(lin_azs, name_additions):

        new_save_name = save_name + '_%s'%name_addition

        for flav, flavtex in zip(PRIMARIES, TEXPRIMARIES):

            low_en_low_az_flux_weights = calculate_3d_flux_weights(
                low_ens,
                czs,
                low_azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )

            high_en_low_az_flux_weights = calculate_3d_flux_weights(
                high_ens,
                czs,
                low_azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )

            low_en_high_az_flux_weights = calculate_3d_flux_weights(
                low_ens,
                czs,
                high_azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )

            high_en_high_az_flux_weights = calculate_3d_flux_weights(
                high_ens,
                czs,
                high_azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )

            flux5lowaz = flux_dict[flav][
                np.where(flux_dict['azimuth'] == 75.0)][0][
                    np.where(flux_dict['energy'] == 5.0119)][0]
            flux50lowaz = flux_dict[flav][
                np.where(flux_dict['azimuth'] == 75.0)][0][
                    np.where(flux_dict['energy'] == 50.119)][0]

            flux5highaz = flux_dict[flav][
                np.where(flux_dict['azimuth'] == 285.0)][0][
                    np.where(flux_dict['energy'] == 5.0119)][0]
            flux50highaz = flux_dict[flav][
                np.where(flux_dict['azimuth'] == 285.0)][0][
                    np.where(flux_dict['energy'] == 50.119)][0]

            plot_1d_slices(
                xintvals=czs,
                yintvals=low_en_low_az_flux_weights,
                xtabvals=flux_dict['coszen'],
                ytabvals=flux5lowaz,
                xtabbins=np.linspace(-1, 1, 21),
                xlabel=r'$\cos\theta_Z$',
                ylabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext=0.75,
                ytext=0.68,
                text='Slice at\n5.0119 GeV\n' + r'$\phi_{Az}=75^{\circ}$',
                table_name=legend_filename,
                save_name=os.path.join(
                    outdir, '%s_%sfluxweightstest5GeV75Az.png'%(new_save_name, flav)
                ),
                log=False
            )

            plot_1d_slices(
                xintvals=czs,
                yintvals=high_en_low_az_flux_weights,
                xtabvals=flux_dict['coszen'],
                ytabvals=flux50lowaz,
                xtabbins=np.linspace(-1, 1, 21),
                xlabel=r'$\cos\theta_Z$',
                ylabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext=0.75,
                ytext=0.68,
                text='Slice at\n50.119 GeV\n' + r'$\phi_{Az}=75^{\circ}$',
                table_name=legend_filename,
                save_name=os.path.join(
                    outdir, '%s_%sfluxweightstest50GeV75Az.png'%(new_save_name, flav)
                ),
                log=False
            )

            plot_1d_slices(
                xintvals=czs,
                yintvals=low_en_high_az_flux_weights,
                xtabvals=flux_dict['coszen'],
                ytabvals=flux5highaz,
                xtabbins=np.linspace(-1, 1, 21),
                xlabel=r'$\cos\theta_Z$',
                ylabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext=0.75,
                ytext=0.68,
                text='Slice at\n5.0119 GeV\n' + r'$\phi_{Az}=285^{\circ}$',
                table_name=legend_filename,
                save_name=os.path.join(
                    outdir, '%s_%sfluxweightstest5GeV285Az.png'%(new_save_name, flav)
                ),
                log=False
            )

            plot_1d_slices(
                xintvals=czs,
                yintvals=high_en_high_az_flux_weights,
                xtabvals=flux_dict['coszen'],
                ytabvals=flux50highaz,
                xtabbins=np.linspace(-1, 1, 21),
                xlabel=r'$\cos\theta_Z$',
                ylabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext=0.75,
                ytext=0.68,
                text='Slice at\n50.119 GeV\n' + r'$\phi_{Az}=285^{\circ}$',
                table_name=legend_filename,
                save_name=os.path.join(
                    outdir, '%s_%sfluxweightstest50GeV285Az.png'%(new_save_name, flav)
                ),
                log=False
            )

            upgoing_flux_weights_low_azs = calculate_3d_flux_weights(
                ens,
                upgoing,
                low_azs_two,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )

            downgoing_flux_weights_low_azs = calculate_3d_flux_weights(
                ens,
                downgoing,
                low_azs_two,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )

            upgoing_flux_weights_high_azs = calculate_3d_flux_weights(
                ens,
                upgoing,
                high_azs_two,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )

            downgoing_flux_weights_high_azs = calculate_3d_flux_weights(
                ens,
                downgoing,
                high_azs_two,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )

            upgoing_flux_weights_low_azs *= np.power(ens, 3)
            downgoing_flux_weights_low_azs *= np.power(ens, 3)
            upgoing_flux_weights_high_azs *= np.power(ens, 3)
            downgoing_flux_weights_high_azs *= np.power(ens, 3)

            coszen_strs = ['%.2f'%coszen for coszen in flux_dict['coszen']]
            coszen_strs = np.array(coszen_strs)

            flux5lowaz = flux_dict[flav][
                np.where(flux_dict['azimuth'] == 75.0)][0][
                    np.where(flux_dict['energy'] == 5.0119)][0]
            flux50lowaz = flux_dict[flav][
                np.where(flux_dict['azimuth'] == 75.0)][0][
                    np.where(flux_dict['energy'] == 50.119)][0]

            flux5highaz = flux_dict[flav][
                np.where(flux_dict['azimuth'] == 285.0)][0][
                    np.where(flux_dict['energy'] == 5.0119)][0]
            flux50highaz = flux_dict[flav][
                np.where(flux_dict['azimuth'] == 285.0)][0][
                    np.where(flux_dict['energy'] == 50.119)][0]

            fluxupgoinglowaz = flux_dict[flav][
                np.where(flux_dict['azimuth'] == 75.0)][0].T[
                    np.where(coszen_strs == '-0.95')][0]
            fluxdowngoinglowaz = flux_dict[flav][
                np.where(flux_dict['azimuth'] == 75.0)][0].T[
                    np.where(coszen_strs == '0.35')][0]
            fluxupgoinghighaz = flux_dict[flav][
                np.where(flux_dict['azimuth'] == 285.0)][0].T[
                    np.where(coszen_strs == '-0.95')][0]
            fluxdowngoinghighaz = flux_dict[flav][
                np.where(flux_dict['azimuth'] == 285.0)][0].T[
                    np.where(coszen_strs == '0.35')][0]

            fluxupgoinglowaz *= np.power(flux_dict['energy'], 3)
            fluxdowngoinglowaz *= np.power(flux_dict['energy'], 3)
            fluxupgoinghighaz *= np.power(flux_dict['energy'], 3)
            fluxdowngoinghighaz *= np.power(flux_dict['energy'], 3)

            if 'numu' in flav:
                xtext = 0.68
                ytext = 0.28
            elif 'nue' in flav:
                xtext = 0.35
                ytext = 0.28

            plot_1d_slices(
                xintvals=ens,
                yintvals=upgoing_flux_weights_low_azs,
                xtabvals=flux_dict['energy'],
                ytabvals=fluxupgoinglowaz,
                xtabbins=np.logspace(-1.025, 4.025, 102),
                xlabel='Neutrino Energy (GeV)',
                ylabel=r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
                xtext=xtext,
                ytext=ytext,
                text=r'Slice at $\cos\theta_Z=-0.95$'+'\n'+r'$\phi_{Az}=75^{\circ}$',
                table_name=legend_filename,
                save_name=os.path.join(
                    outdir, '%s_%sfluxweightstest-0.95cz75Az.png'%(new_save_name, flav)
                ),
                log=True
            )

            plot_1d_slices(
                xintvals=ens,
                yintvals=downgoing_flux_weights_low_azs,
                xtabvals=flux_dict['energy'],
                ytabvals=fluxdowngoinglowaz,
                xtabbins=np.logspace(-1.025, 4.025, 102),
                xlabel='Neutrino Energy (GeV)',
                ylabel=r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
                xtext=xtext,
                ytext=ytext,
                text=r'Slice at $\cos\theta_Z=0.35$'+'\n'+r'$\phi_{Az}=75^{\circ}$',
                table_name=legend_filename,
                save_name=os.path.join(
                    outdir, '%s_%sfluxweightstest0.35cz75Az.png'%(new_save_name, flav)
                ),
                log=True
            )

            plot_1d_slices(
                xintvals=ens,
                yintvals=upgoing_flux_weights_high_azs,
                xtabvals=flux_dict['energy'],
                ytabvals=fluxupgoinghighaz,
                xtabbins=np.logspace(-1.025, 4.025, 102),
                xlabel='Neutrino Energy (GeV)',
                ylabel=r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
                xtext=xtext,
                ytext=ytext,
                text=r'Slice at $\cos\theta_Z=-0.95$'+'\n'+r'$\phi_{Az}=285^{\circ}$',
                table_name=legend_filename,
                save_name=os.path.join(
                    outdir, '%s_%sfluxweightstest-0.95cz285Az.png'%(new_save_name, flav)
                ),
                log=True
            )

            plot_1d_slices(
                xintvals=ens,
                yintvals=downgoing_flux_weights_high_azs,
                xtabvals=flux_dict['energy'],
                ytabvals=fluxdowngoinghighaz,
                xtabbins=np.logspace(-1.025, 4.025, 102),
                xlabel='Neutrino Energy (GeV)',
                ylabel=r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
                xtext=xtext,
                ytext=ytext,
                text=r'Slice at $\cos\theta_Z=0.35$'+'\n'+r'$\phi_{Az}=285^{\circ}$',
                table_name=legend_filename,
                save_name=os.path.join(
                    outdir, '%s_%sfluxweightstest0.35cz285Az.png'%(new_save_name, flav)
                ),
                log=True
            )

            low_en_upgoing_flux_weights = calculate_3d_flux_weights(
                low_ens_two,
                upgoing_two,
                azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )

            high_en_upgoing_flux_weights = calculate_3d_flux_weights(
                high_ens_two,
                upgoing_two,
                azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )

            low_en_downgoing_flux_weights = calculate_3d_flux_weights(
                low_ens_two,
                downgoing_two,
                azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )

            high_en_downgoing_flux_weights = calculate_3d_flux_weights(
                high_ens_two,
                downgoing_two,
                azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )

            flux5downgoing = flux_dict[flav].T[
                np.where(flux_dict['coszen'] == 0.35)][0][
                    np.where(flux_dict['energy'] == 5.0119)][0]
            flux50downgoing = flux_dict[flav].T[
                np.where(flux_dict['coszen'] == 0.35)][0][
                    np.where(flux_dict['energy'] == 50.119)][0]

            flux5upgoing = flux_dict[flav].T[
                np.where(flux_dict['coszen'] == -0.95)][0][
                    np.where(flux_dict['energy'] == 5.0119)][0]
            flux50upgoing = flux_dict[flav].T[
                np.where(flux_dict['coszen'] == -0.95)][0][
                    np.where(flux_dict['energy'] == 50.119)][0]

            plot_1d_slices(
                xintvals=np.linspace(0.0, 360.0, 121),
                yintvals=low_en_upgoing_flux_weights,
                xtabvals=flux_dict['azimuth'],
                ytabvals=flux5upgoing,
                xtabbins=np.linspace(0.0, 360.0, 13),
                xlabel=r'$\phi_{Az}$',
                ylabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext=0.75,
                ytext=0.68,
                text="Slice at \n 5.0119 GeV \n"+r" $\cos\theta_Z=-0.95$",
                table_name=legend_filename,
                save_name=os.path.join(
                    outdir, '%s_%sfluxweightstest5GeV-0.95cz.png'%(new_save_name, flav)
                ),
                log=False
            )

            plot_1d_slices(
                xintvals=np.linspace(0.0, 360.0, 121),
                yintvals=high_en_upgoing_flux_weights,
                xtabvals=flux_dict['azimuth'],
                ytabvals=flux50upgoing,
                xtabbins=np.linspace(0.0, 360.0, 13),
                xlabel=r'$\phi_{Az}$',
                ylabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext=0.75,
                ytext=0.68,
                text="Slice at\n50.119 GeV\n"+r"$\cos\theta_Z=-0.95$",
                table_name=legend_filename,
                save_name=os.path.join(
                    outdir, '%s_%sfluxweightstest50GeV-0.95cz.png'%(new_save_name, flav)
                ),
                log=False
            )

            plot_1d_slices(
                xintvals=np.linspace(0.0, 360.0, 121),
                yintvals=low_en_downgoing_flux_weights,
                xtabvals=flux_dict['azimuth'],
                ytabvals=flux5downgoing,
                xtabbins=np.linspace(0.0, 360.0, 13),
                xlabel=r'$\phi_{Az}$',
                ylabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext=0.75,
                ytext=0.68,
                text="Slice at\n5.0119 GeV\n" + r"$\cos\theta_Z=0.35$",
                table_name=legend_filename,
                save_name=os.path.join(
                    outdir, '%s_%sfluxweightstest5GeV0.35cz.png'%(new_save_name, flav)
                ),
                log=False
            )

            plot_1d_slices(
                xintvals=np.linspace(0.0, 360.0, 121),
                yintvals=high_en_downgoing_flux_weights,
                xtabvals=flux_dict['azimuth'],
                ytabvals=flux50downgoing,
                xtabbins=np.linspace(0.0, 360.0, 13),
                xlabel=r'$\phi_{Az}$',
                ylabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext=0.75,
                ytext=0.68,
                text="Slice at \n 50.119 GeV \n"+r"$\cos\theta_Z=0.35$",
                table_name=legend_filename,
                save_name=os.path.join(
                    outdir, '%s_%sfluxweightstest50GeV0.35cz.png'%(new_save_name, flav)
                ),
                log=False
            )


def do_2d_3d_honda_test(spline_dict, flux_dict, outdir, oversample, save_name,
                        title_filename, flav, flavtex, enpow=1):

    all_ens_bins = np.logspace(-1.025, 4.025, 101*oversample+1)
    all_log_ens_bins = np.linspace(-1.025, 4.025, 101*oversample+1)
    log_en_bin_width = all_log_ens_bins[1] - all_log_ens_bins[0]
    all_ens = np.logspace(all_log_ens_bins[0] + log_en_bin_width/2.0,
                          all_log_ens_bins[-1] - log_en_bin_width/2.0,
                          101*oversample)
    all_czs_bins = np.linspace(-1.0, 1.0, 20*oversample+1)
    cz_bin_width = all_czs_bins[1] - all_czs_bins[0]
    all_czs = np.linspace(all_czs_bins[0] + cz_bin_width/2.0,
                          all_czs_bins[-1] - cz_bin_width/2.0,
                          20*oversample)
    all_azs_bins = np.linspace(0.0, 360.0, 12*oversample+1)
    az_bin_width = all_azs_bins[1] - all_azs_bins[0]
    all_azs = np.linspace(all_azs_bins[0] + az_bin_width/2.0,
                          all_azs_bins[-1] - az_bin_width/2.0,
                          12*oversample)*np.pi/180.0

    all_ens_czs_mg, all_czs_ens_mg = np.meshgrid(all_ens, all_czs)
    az_slice = (45.0*np.pi/180.0)*np.ones_like(all_ens_czs_mg)
    all_ens_azs_mg, all_azs_ens_mg = np.meshgrid(all_ens, all_azs)
    cz_slice = -0.15*np.ones_like(all_ens_azs_mg)
    all_czs_azs_mg, all_azs_czs_mg = np.meshgrid(all_czs, all_azs)
    en_slice = 5.0119*np.ones_like(all_czs_azs_mg)

    logging.info('Doing 2D verification of %s 3D Honda tables', flav)

    ens_czs_az45_flux_weights = calculate_3d_flux_weights(
        all_ens_czs_mg.flatten(),
        all_czs_ens_mg.flatten(),
        az_slice.flatten(),
        spline_dict[flav],
        enpow=enpow
    )

    ens_czs_az45_flux_weights = np.array(
        np.split(
            ens_czs_az45_flux_weights,
            len(all_czs)
        )
    )
    ens_czs_az45_flux_weights_map = {}
    ens_czs_az45_flux_weights_map['map'] = ens_czs_az45_flux_weights.T
    ens_czs_az45_flux_weights_map['ebins'] = all_ens_bins
    ens_czs_az45_flux_weights_map['czbins'] = all_czs_bins

    gridspec_kw = dict(left=0.15, right=0.90, wspace=0.32)
    fig, axes = plt.subplots(nrows=1, ncols=1, gridspec_kw=gridspec_kw,
                             sharex=False, sharey=False, figsize=(12, 10))

    logplot(m=ens_czs_az45_flux_weights_map,
            title=('Finely Interpolated %s Flux '%flavtex
                   + r'(slice at $\phi_{Az}=45^{\circ}$)'),
            ax=axes,
            clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
            medlabels=True)

    fig.savefig(
        os.path.join(
            outdir,
            '%s_az45_%s2dinterpolation.png'%(save_name, flav)
        )
    )

    ens_czm015_azs_flux_weights = calculate_3d_flux_weights(
        all_ens_azs_mg.flatten(),
        cz_slice.flatten(),
        all_azs_ens_mg.flatten(),
        spline_dict[flav],
        enpow=enpow
    )

    ens_czm015_azs_flux_weights = np.array(
        np.split(
            ens_czm015_azs_flux_weights,
            len(all_azs)
        )
    )
    ens_czm015_azs_flux_weights_map = {}
    ens_czm015_azs_flux_weights_map['map'] = ens_czm015_azs_flux_weights.T
    ens_czm015_azs_flux_weights_map['ebins'] = all_ens_bins
    ens_czm015_azs_flux_weights_map['azbins'] = all_azs_bins

    gridspec_kw = dict(left=0.15, right=0.90, wspace=0.32)
    fig, axes = plt.subplots(nrows=1, ncols=1, gridspec_kw=gridspec_kw,
                             sharex=False, sharey=False, figsize=(12, 10))

    logplot(m=ens_czm015_azs_flux_weights_map,
            title=('Finely Interpolated %s Flux '%flavtex
                   + r'(slice at $\cos\theta_Z=-0.15$)'),
            ax=axes,
            clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
            medlabels=True)

    fig.savefig(
        os.path.join(
            outdir,
            '%s_cz-0.15_%s2dinterpolation.png'%(save_name, flav)
        )
    )

    en5_czs_azs_flux_weights = calculate_3d_flux_weights(
        en_slice.flatten(),
        all_czs_azs_mg.flatten(),
        all_azs_czs_mg.flatten(),
        spline_dict[flav],
        enpow=enpow
    )

    en5_czs_azs_flux_weights = np.array(
        np.split(
            en5_czs_azs_flux_weights,
            len(all_azs)
        )
    )
    en5_czs_azs_flux_weights_map = {}
    en5_czs_azs_flux_weights_map['map'] = en5_czs_azs_flux_weights
    en5_czs_azs_flux_weights_map['czbins'] = all_czs_bins
    en5_czs_azs_flux_weights_map['azbins'] = all_azs_bins

    gridspec_kw = dict(left=0.15, right=0.90, wspace=0.32)
    fig, axes = plt.subplots(nrows=1, ncols=1, gridspec_kw=gridspec_kw,
                             sharex=False, sharey=False, figsize=(12, 10))

    logplot(m=en5_czs_azs_flux_weights_map,
            title=('Finely Interpolated %s Flux '%flavtex
                   + r'(slice at $E_{\nu}=5.0119$ GeV)'),
            ax=axes,
            clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
            medlabels=True,
            logz=False)

    fig.savefig(
        os.path.join(
            outdir,
            '%s_en5_%s2dinterpolation.png'%(save_name, flav)
        )
    )


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flux-file-2d', type=str,
                        default=None,
                        help='''2D flux file you want to run tests on. If one
                        is not specified then no 2D tests will be done.''')
    parser.add_argument('--flux-file-3d', type=str,
                        default=None,
                        help='''3D flux file you want to run tests on. If one
                        is not specified then no 3D tests will be done.''')
    parser.add_argument('--onedim-checks', action='store_true',
                        help='''Run verifications on 1D slices.''')
    parser.add_argument('--twodim-checks', action='store_true',
                        help='''Make finely interpolated 2D plots.
                        WARNING - THESE ARE SLOW.''')
    parser.add_argument('--ip-checks', action='store_true',
                        help='''Run checks on integral-preserving nature.
                        WARNING - THESE ARE VERY SLOW.''')
    parser.add_argument('--comparisons', action='store_true',
                        help='''Run comparisons between a Bartol and Honda
                        flux file. WARNING - ALSO VERY SLOW.''')
    parser.add_argument('--oversample', type=int, default=10,
                        help='''Integer to oversample for integral-preserving
                        checks and comparisons between flux files.''')
    parser.add_argument('--enpow', type=int, default=1,
                        help='''Power of energy to use in making the energy
                        splines i.e. flux * (energy**enpow).''')
    parser.add_argument('--ip-az', action='store_true',
                        help='''Perform the interpolation in the azimuthal
                        dimension with the integral-preserving algorithm.
                        NOTE - THIS IS NOT RECOMMENDED.''')
    parser.add_argument('--flavor', type=str, default=None,
                        help='''Choose a flavor to perform the tests on. This
                        is necessary for the 3D 2D and ip checks since they
                        take so much memory otherwise...''')
    parser.add_argument('--outdir', metavar='DIR', type=str, required=True,
                        help='''Store all output plots to this directory.''')
    parser.add_argument('-v', action='count', default=None,
                        help='set verbosity level')

    args = parser.parse_args()
    set_verbosity(args.v)

    if not os.path.exists(args.outdir):
        logging.info("Making output directory %s", args.outdir)
        os.makedirs(args.outdir)

    if (args.ip_checks) and (not args.twodim_checks):
        logging.info(
            "You have requested to perform the integral-preserving checks and"
            " so the two dimensional checks will be performed too (this adds"
            " nothing extra to the computing time)."
        )
        args.twodim_checks = True

    if args.flux_file_2d is not None:

        if ('honda' not in args.flux_file_2d) and \
           ('bartol' not in args.flux_file_2d):
            raise ValueError('Type of flux file not recognised.')

        spline_dict_2d, flux_dict_2d = load_2d_table(
            args.flux_file_2d,
            enpow=args.enpow,
            return_table=True
        )

        if 'honda' in args.flux_file_2d:

            flux_file_2d_name = args.flux_file_2d.split('/')[-1]
            flux_file_2d_bits = flux_file_2d_name.split('-')
            year = flux_file_2d_bits[1]
            site = flux_file_2d_bits[2]

            title_filename = 'Honda'
            legend_filename = 'Honda'

            if site == 'spl':
                title_filename += ' South Pole'
                legend_filename += ' SPL'
            elif site == 'sno':
                title_filename += ' Sudbury'
                legend_filename += ' SNO'
            else:
                logging.warning(
                    'Don\'t know what to do with site %s.'
                    'Omitting from titles', site
                )

            title_filename += ' %s'%year
            legend_filename += ' %s'%year
            save_name = 'honda_2d_%s_%s'%(site, year)

            if args.onedim_checks:
                do_1d_2d_honda_test(
                    spline_dict=spline_dict_2d,
                    flux_dict=flux_dict_2d,
                    legend_filename=legend_filename,
                    save_name=save_name,
                    outdir=args.outdir,
                    enpow=args.enpow
                )

            if args.twodim_checks:
                do_2d_2d_honda_test(
                    spline_dict=spline_dict_2d,
                    flux_dict=flux_dict_2d,
                    outdir=args.outdir,
                    ip_checks=args.ip_checks,
                    oversample=args.oversample,
                    save_name=save_name,
                    title_filename=title_filename,
                    enpow=args.enpow
                )

        else:

            if args.onedim_checks:
                do_1d_2d_bartol_test(
                    spline_dict=spline_dict_2d,
                    flux_dict=flux_dict_2d,
                    outdir=args.outdir,
                    enpow=args.enpow
                )

            if args.twodim_checks:
                do_2d_2d_bartol_test(
                    spline_dict=spline_dict_2d,
                    flux_dict=flux_dict_2d,
                    outdir=args.outdir,
                    ip_checks=args.ip_checks,
                    oversample=args.oversample,
                    enpow=args.enpow
                )

    if args.flux_file_3d is not None:

        spline_dict_3d, flux_dict_3d = load_3d_table(
            args.flux_file_3d,
            enpow=args.enpow,
            return_table=True
        )

        flux_file_3d_name = args.flux_file_3d.split('/')[-1]
        flux_file_3d_bits = flux_file_3d_name.split('-')
        year = flux_file_3d_bits[1]
        site = flux_file_3d_bits[2]

        title_filename = 'Honda'
        legend_filename = 'Honda'

        if site == 'spl':
            title_filename += ' South Pole'
            legend_filename += ' SPL'
        elif site == 'sno':
            title_filename += ' Sudbury'
            legend_filename += ' SNO'
        else:
            logging.warning(
                'Don\'t know what to do with site %s.'
                'Omitting from titles', site
            )

        title_filename += ' %s'%year
        legend_filename += ' %s'%year
        save_name = 'honda_3d_%s_%s'%(site, year)

        if args.onedim_checks:
            do_1d_3d_honda_test(
                spline_dict=spline_dict_3d,
                flux_dict=flux_dict_3d,
                legend_filename=legend_filename,
                save_name=save_name,
                outdir=args.outdir,
                enpow=args.enpow
            )

        if args.twodim_checks:
            if args.flavor is None:
                raise ValueError('You must specify a flavor for these tests!')
            if args.flavor not in PRIMARIES:
                raise ValueError('Invalid flavor chosen. Please specify one '
                                 'from the following: %s'%PRIMARIES)
            flavortex = TEXPRIMARIES[PRIMARIES.index(args.flavor)]
            do_2d_3d_honda_test(
                spline_dict=spline_dict_3d,
                flux_dict=flux_dict_3d,
                outdir=args.outdir,
                oversample=args.oversample,
                save_name=save_name,
                title_filename=title_filename,
                flav=args.flavor,
                flavtex=flavortex,
                enpow=args.enpow
            )

        if args.ip_checks:
            if args.flavor is None:
                raise ValueError('You must specify a flavor for these tests!')
            if args.flavor not in PRIMARIES:
                raise ValueError('Invalid flavor chosen. Please specify one '
                                 'from the following: %s'%PRIMARIES)
            flavortex = TEXPRIMARIES[PRIMARIES.index(args.flavor)]
            do_ip_3d_honda_test(
                spline_dict=spline_dict_3d,
                flux_dict=flux_dict_3d,
                outdir=args.outdir,
                oversample=args.oversample,
                save_name=save_name,
                title_filename=title_filename,
                flav=args.flavor,
                flavtex=flavortex,
                enpow=args.enpow,
                az_linear=not args.ip_az

            )

    if args.comparisons:

        logging.warning('Comparisons will be of Honda 2015 SNO and '
                        'Bartol 2004 SNO 2D tables regardless of what you set '
                        'in the flux_file argument(s).')

        honda_spline_dict_2d = load_2d_table(
            'flux/honda-2015-sno-solmax-aa.d',
            enpow=args.enpow
        )

        bartol_spline_dict_2d = load_2d_table(
            'flux/bartol-2004-sno-solmax-aa.d',
            enpow=args.enpow
        )

        do_2d_2d_comparisons(
            honda_spline_dict=honda_spline_dict_2d,
            bartol_spline_dict=bartol_spline_dict_2d,
            outdir=args.outdir,
            oversample=args.oversample,
            enpow=args.enpow
        )


if __name__ == '__main__':
    main()
