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

from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
import numpy as np
import scipy.interpolate as interpolate

from pisa.utils.log import logging
from pisa.utils.resources import open_resource

primaries = ['numu', 'numubar', 'nue', 'nuebar']
texprimaries = [r'$\nu_{\mu}$', r'$\bar{\nu}_{\mu}$', r'$\nu_{e}$', r'$\bar{\nu}_{e}$']


def load_Honda_table(flux_file, enpow=1, returnTable=False):

    logging.debug("Loading atmospheric flux table %s" % flux_file)

    # columns in Honda files are in the same order
    cols = ['energy'] + primaries
    
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
    for nutype in primaries:
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
                tot_flux += energyfluxval*np.power(energyval,enpow)
                int_flux.append(tot_flux)

            spline = interpolate.splrep(int_flux_dict['logenergy'],
                                        int_flux, s=0)
            CZvalue = '%.2f'%(1.05-CZiter*0.1)
            splines[CZvalue] = spline
            CZiter += 1

        spline_dict[nutype] = splines

    for prim in primaries:
        flux_dict[prim] = flux_dict[prim][::-1]

    if returnTable:
        return spline_dict, flux_dict
    else:
        return spline_dict


def load_Bartol_table(flux_file, enpow=1, returnTable=False):

    logging.debug("Loading atmospheric flux table %s" % flux_file)

    # Bartol tables have been modified to look like Honda tables
    cols = ['energy'] + primaries
    
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
    low_log_energy = np.linspace(-1,1,41)
    high_log_energy = np.linspace(1.1,4,30)
    int_flux_dict['logenergy'] = np.concatenate(
        [low_log_energy,high_log_energy]
    )
    int_flux_dict['coszen'] = np.linspace(-1, 1, 21)
    for nutype in primaries:
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

                #
                # UNCOMMENT THIS AND COMMENT WHAT'S BELOW IF YOU WANT THE
                # INTERPOLATION TO BREAK AT 10 GEV 
                #tot_flux += energyfluxval*np.power(energyval,enpow)
                #
                
                if energyval < 10.0:
                    tot_flux += energyfluxval*np.power(energyval,enpow)*0.05
                else:
                    tot_flux += energyfluxval*np.power(energyval,enpow)*0.1
                int_flux.append(tot_flux)

            spline = interpolate.splrep(int_flux_dict['logenergy'],
                                        int_flux, s=0)
            CZvalue = '%.2f'%(1.05-CZiter*0.1)
            splines[CZvalue] = spline
            CZiter += 1

        spline_dict[nutype] = splines

    if returnTable:
        return spline_dict, flux_dict
    else:
        return spline_dict


def load_2D_table(flux_file, enpow=1, returnTable=False):
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
    if 'honda' not in flux_file:
        if 'bartol' in flux_file:
            logging.warn('WARNING - Usage of the Bartol files in '
                         'integral-preserving mode will give WRONG results.')
            if returnTable:
                spline_dict, flux_dict = load_Bartol_table(flux_file,
                                                           enpow=enpow,
                                                           returnTable=True)
            else:
                spline_dict = load_Bartol_table(flux_file,
                                                enpow=enpow)
            spline_dict['name'] = 'bartol'

        else:
            raise ValueError('Flux file must be from the Honda group')
    else:
        if returnTable:
            spline_dict, flux_dict = load_Honda_table(flux_file,
                                                      enpow=enpow,
                                                      returnTable=True)
        else:
             spline_dict = load_Honda_table(flux_file,
                                            enpow=enpow)
        spline_dict['name'] = 'honda'

    if returnTable:
        return spline_dict, flux_dict
    else:
        return spline_dict


def calculate_flux_weights(true_energies, true_coszens, en_splines,
                           table_name='honda', enpow=1):
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
    table_name : string
        The name of the table used. Should be bartol or honda.

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
            if table_name == 'honda':
                # Here the bin width is 0.05 (in log energy)
                spval = interpolate.splev(true_log_energy,
                                          en_splines[czkey],
                                          der=1)*0.05
            elif table_name == 'bartol':

                #
                # THIS IS ALSO PART OF BREAKING THE INTERPOLATION AT 10 GEV
                #if true_energy < 10.0:
                #    spval = interpolate.splev(true_log_energy,
                #                              en_splines[czkey],
                #                              der=1)*0.05
                #else:
                #    spval = interpolate.splev(true_log_energy,
                #                              en_splines[czkey],
                #                              der=1)*0.1
                #

                spval = interpolate.splev(true_log_energy,
                                          en_splines[czkey],
                                          der=1)

            spline_vals.append(spval)
        spline_vals = np.array(spline_vals)
        int_spline_vals = np.cumsum(spline_vals)
        spline = interpolate.splrep(cz_spline_points,
                                    int_spline_vals, s=0)
        flux_weights.append(interpolate.splev(true_coszen,
                                              spline,
                                              der=1)*(0.1/np.power(true_energy,enpow)))

    flux_weights = np.array(flux_weights)
    return flux_weights


if __name__ == '__main__':
    '''
    This is a slightly longer test than that given in the docstring of the
    calculate_flux_weights function. This will:

      * Make 1D plots over cos(zenith) for 5 and 20 GeV slices.
      * Make 1D plots over energy for -0.95 and 0.35 cos(zenith) slices.
    
    This will be done for every flavour and the table values will also be
    overlaid on the plot to highlight whether the interpolation is working
    or not.
    '''

    import os

    import matplotlib
    maplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams['text.usetex'] = True
    import matplotlib.colors as colors


    def Plot1DSlices(xintvals, yintvals, xtabvals, ytabvals, xtabbins,
                     xlabel, ylabel, xtext, ytext, text, tablename,
                     savename, log):
        '''
        Test function to show interpolation and tables overlaid in 1D slices.
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
        tablename : string
            The text label naming the tables used
        savename : string 
            The place and name to save the plot.
        log : bool
            A boolean to whether the axes should be made logarithmic. 
            Will do both.
        '''

        plt.plot(xintvals,
                 yintvals,
                 color='r',
                 linewidth=2,
                 label='IP Interpolation')
        plt.hist(xtabvals,
                 weights = ytabvals,
                 bins = xtabbins,
                 color = 'k',
                 linewidth = 2,
                 histtype='step',
                 label=tablename)
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        if log:
            plt.xlim(xtabbins[0],xtabbins[-1])
            plt.xscale("log")
            plt.yscale("log")
            ymin = min(np.log10(yintvals))
            ymax = max(np.log10(yintvals))
            ydiff = ymax - ymin
            plt.ylim(np.power(10,ymin-0.1*ydiff),np.power(10,ymax+0.1*ydiff))
            if 'numu' in savename:
                plt.legend(loc='lower right')
            elif 'nue' in savename:
                plt.legend(loc='lower left')
        else:
            ymin = min(yintvals)
            ymax = max(yintvals)
            ydiff = ymax-ymin
            plt.ylim(ymin-0.1*ydiff,ymax+0.1*ydiff)
            plt.legend(loc='upper right')
        plt.figtext(xtext,
                    ytext,
                    text,
                    verticalalignment='center',
                    horizontalalignment='center',
                    color='k',
                    fontsize=24)
        plt.savefig(savename)
        plt.close()
    

    def logplot(m, title, ax, clabel, cmap=plt.cm.afmhot, largelabels=False):
        """Simple plotting of a 2D histogram (map)"""
        hist = np.ma.masked_invalid(m['map'])
        y = m['ebins']
        x = m['czbins']
        X, Y = np.meshgrid(x, y)
        ax.set_yscale('log')
        vmin = hist.min()
        vmax = hist.max()
        if clabel is not None:
            pcmesh = ax.pcolormesh(X, Y, hist,
                                   norm=colors.LogNorm(vmin=vmin,vmax=vmax),
                                   cmap=cmap)
        else:
            pcmesh = ax.pcolormesh(X, Y, hist,
                                   cmap=cmap)
        cbar = plt.colorbar(mappable=pcmesh, ax=ax)
        if clabel is not None:
            if largelabels:
                cbar.set_label(clabel,labelpad=-1,fontsize=36)
                cbar.ax.tick_params(labelsize=36)
            else:
                cbar.set_label(clabel,labelpad=-1)
                cbar.ax.tick_params(labelsize='large')
        if largelabels:
            ax.set_xlabel(r'$\cos\theta_Z$',fontsize=36)
            ax.set_ylabel(r'Energy (GeV)',labelpad=-3,fontsize=36)
            ax.set_title(title, y=1.03, fontsize=36)
            plt.tick_params(axis='both', which='major', labelsize=36)
        else:
            ax.set_xlabel(r'$\cos\theta_Z$')
            ax.set_ylabel(r'Energy (GeV)',labelpad=-3)
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


    def do_1D_honda_test(spline_dict, flux_dict, outdir, enpow=1):

        czs = np.linspace(-1,1,81)
        low_ens = 5.0119*np.ones_like(czs)
        high_ens = 50.119*np.ones_like(czs)

        ens = np.logspace(-1.025,4.025,1020)
        upgoing = -0.95*np.ones_like(ens)
        downgoing = 0.35*np.ones_like(ens)

        for flav, flavtex in zip(primaries, texprimaries):
    
            low_en_flux_weights = calculate_flux_weights(low_ens,
                                                         czs,
                                                         spline_dict[flav],
                                                         table_name='honda',
                                                         enpow=enpow)
            
            high_en_flux_weights = calculate_flux_weights(high_ens,
                                                          czs,
                                                          spline_dict[flav],
                                                          table_name='honda',
                                                          enpow=enpow)

            flux5 = flux_dict[flav].T[np.where(flux_dict['energy']==5.0119)][0]
            flux50 = flux_dict[flav].T[np.where(flux_dict['energy']==50.119)][0]

            Plot1DSlices(
                xintvals = czs,
                yintvals = low_en_flux_weights,
                xtabvals = flux_dict['coszen'],
                ytabvals = flux5,
                xtabbins = np.linspace(-1,1,21),
                xlabel = r'$\cos\theta_Z$',
                ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext = 0.75,
                ytext = 0.7,
                text = 'Slice at \n 5.0119 GeV',
                tablename = 'Honda SPL 2015',
                savename = os.path.join(
                    outdir,'honda_%sfluxweightstest5GeV.png'%flav
                ),
                log = False
            )
        
            Plot1DSlices(
                xintvals = czs,
                yintvals = high_en_flux_weights,
                xtabvals = flux_dict['coszen'],
                ytabvals = flux50,
                xtabbins = np.linspace(-1,1,21),
                xlabel = r'$\cos\theta_Z$',
                ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext = 0.75,
                ytext = 0.7,
                text = 'Slice at \n 50.119 GeV',
                tablename = 'Honda SPL 2015',
                savename = os.path.join(
                    outdir,'honda_%sfluxweightstest50GeV.png'%flav
                ),
                log = False
            )

            upgoing_flux_weights = calculate_flux_weights(ens,
                                                          upgoing,
                                                          spline_dict[flav],
                                                          table_name='honda',
                                                          enpow=enpow)

            downgoing_flux_weights = calculate_flux_weights(ens,
                                                            downgoing,
                                                            spline_dict[flav],
                                                            table_name='honda',
                                                            enpow=enpow)

            upgoing_flux_weights *= np.power(ens,3)
            downgoing_flux_weights *= np.power(ens,3)

            fluxupgoing = flux_dict[flav][np.where(flux_dict['coszen']==-0.95)][0]
            fluxdowngoing = flux_dict[flav][np.where(flux_dict['coszen']==0.35)][0]

            fluxupgoing *= np.power(flux_dict['energy'],3)
            fluxdowngoing *= np.power(flux_dict['energy'],3)

            if 'numu' in flav:
                xtext = 0.68
                ytext = 0.25
            elif 'nue' in flav:
                xtext = 0.35
                ytext = 0.25

            Plot1DSlices(
                xintvals = ens,
                yintvals = upgoing_flux_weights,
                xtabvals = flux_dict['energy'],
                ytabvals = fluxupgoing,
                xtabbins = np.logspace(-1.025,4.025,102),
                xlabel = 'Neutrino Energy (GeV)',
                ylabel = r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
                xtext = xtext,
                ytext = ytext,
                text = r'Slice at $\cos\theta_Z=-0.95$',
                tablename = 'Honda SPL 2015',
                savename = os.path.join(
                    outdir,'honda_%sfluxweightstest-0.95cz.png'%flav
                ),
                log = True
            )
            
            Plot1DSlices(
                xintvals = ens,
                yintvals = downgoing_flux_weights,
                xtabvals = flux_dict['energy'],
                ytabvals = fluxdowngoing,
                xtabbins = np.logspace(-1.025,4.025,102),
                xlabel = 'Neutrino Energy (GeV)',
                ylabel = r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
                xtext = xtext,
                ytext = ytext,
                text = r'Slice at $\cos\theta_Z=0.35$',
                tablename = 'Honda SPL 2015',
                savename = os.path.join(
                    outdir,'honda_%sfluxweightstest0.35cz.png'%flav
                ),
                log = True
            )


    def do_2D_honda_test(spline_dict, flux_dict, outdir, ip_checks,
                         oversample, enpow=1):

        if oversample == 100:
            all_ens = np.logspace(-1.02475,4.02475,10100)
            all_ens_bins = np.logspace(-1.025,4.025,10101)
            all_czs = np.linspace(-0.9995,0.9995,2000)
            all_czs_bins = np.linspace(-1.0,1.0,2001)
            reduced_ens = np.logspace(-0.02475,4.02475,8100)
            reduced_ens_bins = np.logspace(-0.025,4.025,8101)
            reduced_bin = 2000
        elif oversample == 10:
            all_ens = np.logspace(-1.0225,4.0225,1010)
            all_ens_bins = np.logspace(-1.025,4.025,1011)
            all_czs = np.linspace(-0.995,0.995,200)
            all_czs_bins = np.linspace(-1.0,1.0,201)
            reduced_ens = np.logspace(-0.0225,4.0225,810)
            reduced_ens_bins = np.logspace(-0.025,4.025,811)
            reduced_bin = 200
        elif oversample == 1:
            all_ens = np.logspace(-1.0,4.0,101)
            all_ens_bins = np.logspace(-1.025,4.025,102)
            all_czs = np.linspace(-0.95,0.95,20)
            all_czs_bins = np.linspace(-1.0,1.0,21)
            reduced_ens = np.logspace(-0.0,4.0,81)
            reduced_ens_bins = np.logspace(-0.025,4.025,82)
            reduced_bin = 20

        all_ens_mg, all_czs_mg = np.meshgrid(all_ens, all_czs)

        for flav, flavtex in zip(primaries, texprimaries):

            all_flux_weights = calculate_flux_weights(all_ens_mg.flatten(),
                                                      all_czs_mg.flatten(),
                                                      spline_dict[flav],
                                                      table_name='honda',
                                                      enpow=enpow)

            all_flux_weights = np.array(np.split(all_flux_weights,
                                                 len(all_czs)))
            all_flux_weights_map = {}
            all_flux_weights_map['map'] = all_flux_weights.T
            all_flux_weights_map['ebins'] = all_ens_bins
            all_flux_weights_map['czbins'] = all_czs_bins
        
            gridspec_kw = dict(left=0.15, right=0.90, wspace=0.32)
            fig, axes = plt.subplots(nrows=1, ncols=1, gridspec_kw=gridspec_kw,
                                     sharex=False, sharey=False, figsize=(12,10))

            logplot(m=all_flux_weights_map,
                    title='Finely Interpolated %s Flux'%flavtex,
                    ax=axes,
                    clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                    largelabels=True)

            fig.savefig(os.path.join(outdir,
                                     'honda_%s2dinterpolation.png'%flav))

            if ip_checks:

                downsampled_flux_map = {}
                downsampled_flux_map['map'] = take_average(
                    all_flux_weights.T, oversample
                )
                downsampled_flux_map['ebins'] = np.logspace(-1.025,4.025,102)
                downsampled_flux_map['czbins'] = np.linspace(-1.0,1.0,21)

                honda_tables = {}
                honda_tables['map'] = flux_dict[flav].T
                honda_tables['ebins'] = np.logspace(-1.025,4.025,102)
                honda_tables['czbins'] = np.linspace(-1.0,1.0,21)

                diff_map = {}
                diff_map['map'] = honda_tables['map']-downsampled_flux_map['map']
                diff_map['ebins'] = np.logspace(-1.025,4.025,102)
                diff_map['czbins'] = np.linspace(-1.0,1.0,21)

                diff_ratio_map = {}
                diff_ratio_map['map'] = diff_map['map'] / honda_tables['map']
                diff_ratio_map['ebins'] = np.logspace(-1.025,4.025,102)
                diff_ratio_map['czbins'] = np.linspace(-1.0,1.0,21)
                
                gridspec_kw = dict(left=0.03, right=0.968, wspace=0.32)
                fig, axes = plt.subplots(nrows=1, ncols=5,
                                         gridspec_kw=gridspec_kw,
                                         sharex=False, sharey=False,
                                         figsize=(20,5))

                logplot(m=all_flux_weights_map,
                        title='Finely Interpolated',
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
                        clabel=None)
                logplot(m=diff_ratio_map,
                        title='Percentage Difference',
                        ax=axes[4],
                        clabel=None)

                plt.suptitle('Integral Preserving Tests for %s Honda South Pole 2015 Flux Tables'%flavtex, fontsize=36)
                plt.subplots_adjust(top=0.8)
                fig.savefig(os.path.join(outdir,'honda_%siptest_fullrange.png'%flav))
                plt.close(fig.number)

                reduced_flux_weights = np.delete(
                    all_flux_weights,
                    np.s_[0:reduced_bin],
                    axis=1
                )

                reduced_honda = np.delete(
                    flux_dict[flav],
                    np.s_[0:20],
                    axis=1
                )

                all_flux_weights_map = {}
                all_flux_weights_map['map'] = reduced_flux_weights.T
                all_flux_weights_map['ebins'] = reduced_ens_bins
                all_flux_weights_map['czbins'] = all_czs_bins

                downsampled_flux_map = {}
                downsampled_flux_map['map'] = take_average(
                    reduced_flux_weights.T,
                    oversample
                )
                downsampled_flux_map['ebins'] = np.logspace(-0.025,4.025,82)
                downsampled_flux_map['czbins'] = np.linspace(-1.0,1.0,21)

                honda_tables = {}
                honda_tables['map'] = reduced_honda.T
                honda_tables['ebins'] = np.logspace(-0.025,4.025,82)
                honda_tables['czbins'] = np.linspace(-1.0,1.0,21)

                diff_map = {}
                diff_map['map'] = honda_tables['map']-downsampled_flux_map['map']
                diff_map['ebins'] = np.logspace(-0.025,4.025,82)
                diff_map['czbins'] = np.linspace(-1.0,1.0,21)

                diff_ratio_map = {}
                diff_ratio_map['map'] = diff_map['map'] / honda_tables['map']
                diff_ratio_map['ebins'] = np.logspace(-0.025,4.025,82)
                diff_ratio_map['czbins'] = np.linspace(-1.0,1.0,21)
                
                gridspec_kw = dict(left=0.03, right=0.968, wspace=0.32)
                fig, axes = plt.subplots(nrows=1, ncols=5,
                                         gridspec_kw=gridspec_kw,
                                         sharex=False, sharey=False,
                                         figsize=(20,5))

                logplot(m=all_flux_weights_map,
                        title='Finely Interpolated',
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
                        clabel=None)
                logplot(m=diff_ratio_map,
                        title='Percentage Difference',
                        ax=axes[4],
                        clabel=None)

                plt.suptitle('Integral Preserving Tests for %s Honda South Pole 2015 Flux Tables'%flavtex, fontsize=36)
                plt.subplots_adjust(top=0.8)
                fig.savefig(os.path.join(outdir,'honda_%siptest_reducedenrange.png'%flav))
                plt.close(fig.number)
            

    def do_1D_bartol_test(spline_dict, flux_dict, outdir, enpow=1):

        czs = np.linspace(-1,1,81)
        low_ens = 4.732*np.ones_like(czs)
        high_ens = 44.70*np.ones_like(czs)

        ens = np.logspace(-1,4,701)
        upgoing = -0.95*np.ones_like(ens)
        downgoing = 0.35*np.ones_like(ens)

        for flav, flavtex in zip(primaries, texprimaries):
    
            low_en_flux_weights = calculate_flux_weights(low_ens,
                                                         czs,
                                                         spline_dict[flav],
                                                         table_name='bartol',
                                                         enpow=enpow)
            
            high_en_flux_weights = calculate_flux_weights(high_ens,
                                                          czs,
                                                          spline_dict[flav],
                                                          table_name='bartol',
                                                          enpow=enpow)

            flux5 = flux_dict[flav].T[np.where(flux_dict['energy']==4.732)][0]
            flux50 = flux_dict[flav].T[np.where(flux_dict['energy']==44.70)][0]

            Plot1DSlices(
                xintvals = czs,
                yintvals = low_en_flux_weights,
                xtabvals = flux_dict['coszen'],
                ytabvals = flux5,
                xtabbins = np.linspace(-1,1,21),
                xlabel = r'$\cos\theta_Z$',
                ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext = 0.75,
                ytext = 0.7,
                text = 'Slice at \n 4.732 GeV',
                tablename = 'Bartol SNO 2004',
                savename = os.path.join(
                    outdir,'bartol_%sfluxweightstest5GeV.png'%flav
                ),
                log = False
            )
        
            Plot1DSlices(
                xintvals = czs,
                yintvals = high_en_flux_weights,
                xtabvals = flux_dict['coszen'],
                ytabvals = flux50,
                xtabbins = np.linspace(-1,1,21),
                xlabel = r'$\cos\theta_Z$',
                ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext = 0.75,
                ytext = 0.7,
                text = 'Slice at \n 44.70 GeV',
                tablename = 'Bartol SNO 2004',
                savename = os.path.join(
                    outdir,'bartol_%sfluxweightstest50GeV.png'%flav
                ),
                log = False
            )

            upgoing_flux_weights = calculate_flux_weights(ens,
                                                          upgoing,
                                                          spline_dict[flav],
                                                          table_name='bartol',
                                                          enpow=enpow)

            downgoing_flux_weights = calculate_flux_weights(ens,
                                                            downgoing,
                                                            spline_dict[flav],
                                                            table_name='bartol',
                                                            enpow=enpow)

            upgoing_flux_weights *= np.power(ens,3)
            downgoing_flux_weights *= np.power(ens,3)

            fluxupgoing = flux_dict[flav][np.where(flux_dict['coszen']==-0.95)][0]
            fluxdowngoing = flux_dict[flav][np.where(flux_dict['coszen']==0.35)][0]

            fluxupgoing *= np.power(flux_dict['energy'],3)
            fluxdowngoing *= np.power(flux_dict['energy'],3)

            low_log_energy = np.logspace(-1,1,41)
            high_log_energy = np.logspace(1.1,4,30)
            xtabbins = np.concatenate(
                [low_log_energy,high_log_energy]
            )

            if 'numu' in flav:
                xtext = 0.68
                ytext = 0.25
            elif 'nue' in flav:
                xtext = 0.35
                ytext = 0.25

            Plot1DSlices(
                xintvals = ens,
                yintvals = upgoing_flux_weights,
                xtabvals = flux_dict['energy'],
                ytabvals = fluxupgoing,
                xtabbins = xtabbins,
                xlabel = 'Neutrino Energy (GeV)',
                ylabel = r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
                xtext = xtext,
                ytext = ytext,
                text = r'Slice at $\cos\theta_Z=-0.95$',
                tablename = 'Bartol SNO 2004',
                savename = os.path.join(
                    outdir,'bartol_%sfluxweightstest-0.95cz.png'%flav
                ),
                log = True
            )
            
            Plot1DSlices(
                xintvals = ens,
                yintvals = downgoing_flux_weights,
                xtabvals = flux_dict['energy'],
                ytabvals = fluxdowngoing,
                xtabbins = xtabbins,
                xlabel = 'Neutrino Energy (GeV)',
                ylabel = r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
                xtext = xtext,
                ytext = ytext,
                text = r'Slice at $\cos\theta_Z=0.35$',
                tablename = 'Bartol SNO 2004',
                savename = os.path.join(
                    outdir,'bartol_%sfluxweightstest0.35cz.png'%flav
                ),
                log = True
            )
        

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flux_file',type=str,
                        default='flux/honda-2015-spl-solmax-aa.d',
                        help="Flux file you want to run tests on")
    parser.add_argument('--onedim_checks', action='store_true',
                        help='''Run verifications on 1D slices.''')
    parser.add_argument('--twodim_checks', action='store_true',
                        help='''Make finely interpolated 2D plot.
                        WARNING - THIS IS SLOW.''')
    parser.add_argument('--ip_checks', action='store_true',
                        help='''Run checks on integral-preserving nature.
                        WARNING - THESE ARE VERY SLOW.''')
    parser.add_argument('--oversample', type=int, default=10,
                        help='''Integer to oversample for integral-preserving
                        checks.''')
    parser.add_argument('--enpow', type=int, default=1,
                        help='''Power of energy to use in making the energy
                        splines i.e. flux * (energy**enpow).''')
    parser.add_argument('--outdir', metavar='DIR', type=str,
                        help='''Store all output plots to this directory.''')

    args = parser.parse_args()

    spline_dict, flux_dict = load_2D_table(args.flux_file,
                                           enpow=args.enpow,
                                           returnTable=True)

    if 'honda' in args.flux_file:
        
        if args.onedim_checks:
            do_1D_honda_test(spline_dict, flux_dict, args.outdir,
                             enpow = args.enpow)

        if args.twodim_checks:
            do_2D_honda_test(spline_dict, flux_dict,
                             args.outdir, args.ip_checks,
                             oversample = args.oversample,
                             enpow = args.enpow)

    else:

        if args.onedim_checks:
            do_1D_bartol_test(spline_dict, flux_dict, args.outdir,
                              enpow = args.enpow)
