import numpy as np
import uncertainties as unc
from uncertainties import unumpy as unp
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import math

import matplotlib as mpl
# headless mode
mpl.use('Agg')
# fonts
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from pisa.core.map import Map, MapSet
from pisa.core.binning import MultiDimBinning
from pisa.core.transform import BinnedTensorTransform, TransformSet
import itertools
from pisa.utils.log import logging


__all__ = ['Plotter']


class Plotter(object):
    """

    Plotting library for PISA to plot Maps and MapSets

    Params:
    ------

    outdir : str
        output directory path
    stamp : str
        stamp to be put on every subplot, e.g. 'Preliminary' or 'DeepCore nutau' or ...
    fmt : str or iterable of str
        formats to be plotted, e.g. ['pdf', 'png']
    size : (int, int)
        canvas size
    log : bool
        logarithmic z-axis
    label : str
        z-axis label
    grid : bool
        plot grid
    ratio : bool
        add ratio plots in 1-d histos
    annotate : bool
        annotate counts per bin in 2-d histos
    symmetric : bool
        force symmetric extent of z-axis
    loc : str
        either 'inside' or 'outside', defining where to put axis titles


    Methods:
    -------

    2-d plots:

    plot_2d_single(mapset, **kwargs)
        plot all maps in individual plots
    plot_2d_array(mapset, n_rows=None, n_cols=None, fname=None, **kwrags)
        plot all maps or transforms in a single plot

    1-d plots
    plot_1d_single(mapset, plot_axis, **kwargs)
        plot all maps in individual plots
    plot_1d_array(mapset, plot_axis, n_rows=None, n_cols=None, fname=None, **kwargs)
        plot 1d projections as an array
    plot_1d_slices_array(mapsets, plot_axis, fname=None, **kwargs)
        plot 1d slices as an array
    plot_1d_all(mapset, plot_axis, **kwargs)
        all one a single plot
    plot_1d_stack(mapset, plot_axis, **kwargs)
        all maps stacked on top of each other
    plot_1d_cmp(mapsets, plot_axis, fname=None, **kwargs)
        1d comparisons for two mapsets as projections


    Notes:
    -----

    as **kwargs any matplotlib kwrags can be passed, for example cmap='RdBu' for a 2d plot

    """

    def __init__(self, outdir='.', stamp='PISA cake test', size=(8,8), fmt='pdf', log=True, label='# events', grid=True, ratio=False, annotate=False, symmetric=False,loc='inside'):
        self.outdir = outdir
        self.stamp = stamp
        if isinstance(fmt,basestring):
            fmt = [fmt]
        self.fmt = fmt
        self.size = size
        self.fig = None
        self.log = log
        self.label = label
        self.grid = grid
        self.ratio = ratio
        self.annotate = annotate
        if symmetric: assert(self.log == False), 'cannot do log and symmetric at th same time'
        self.symmetric = symmetric
        self.reset_colors()
        self.color = 'b'
        self.loc = loc
        
    def reset_colors(self):
        self.colors = itertools.cycle(["r", "b", "g"])

    def next_color(self):
        self.color = next(self.colors)

    # --- helper functions ---

    def init_fig(self,figsize=None):
        ''' clear/initialize figure '''
        if figsize is not None:
            size = figsize
        else:
            size = self.size
        if self.fig is not None:
            plt.clf()
            plt.close(self.fig)
        self.fig, self.axes = plt.subplots(1,1,figsize=size)
        self.fig.patch.set_facecolor('none')

    def add_stamp(self, text=None, **kwargs):
        # NOTE add_stamp cannot be used on a subplot that has been
        # de-selected and then re-selected. It will write over existing text.
        ''' ad common stamp with text '''
        if self.loc == 'inside':
            if text is not None:
                a_text = AnchoredText(self.stamp + '\n' + r'$%s$'%text, loc=2, frameon=False, **kwargs)
            else:
                a_text = AnchoredText(self.stamp, loc=2, frameon=False, **kwargs)
            plt.gca().add_artist(a_text)
        elif self.loc == 'outside':
            if text is not None:
                a_text = self.stamp + ' ' + r'$%s$'%text
            else:
                a_text = self.stamp
            plt.gca().set_title(a_text)

    def add_leg(self):
        ''' initialize legend '''
        plt.gca().legend(loc='upper right',ncol=2, frameon=False,numpoints=1)

    def dump(self,fname):
        ''' dump figure to file'''
        for fmt in self.fmt:
            plt.savefig(self.outdir+'/'+fname+'.'+fmt, dpi=150, edgecolor='none',facecolor=self.fig.get_facecolor())

    # --- 2d plots ---

    def plot_2d_single(self, mapset, **kwargs):
        ''' plot all maps in individual plots '''
        if isinstance(mapset, Map):
            mapset = [mapset]
        for map in mapset:
            self.init_fig()
            self.plot_2d_map(map, **kwargs)
            self.add_stamp(map.tex)
            self.dump(map.name)

    def plot_2d_array(self, mapset, n_rows=None, n_cols=None, fname=None,
            **kwargs):
        ''' plot all maps or transforms in a single plot '''
        if fname is None:
            fname = 'test2d'
        self.plot_array(mapset, 'plot_2d_map', n_rows=n_rows, n_cols=n_cols,
                **kwargs)
        self.dump(fname)

    # --- 1d plots ---

    def plot_1d_single(self, mapset, plot_axis, **kwargs):
        ''' plot all maps in individual plots '''
        for map in mapset:
            self.init_fig()
            self.plot_1d_projection(map, plot_axis, **kwargs)
            self.add_stamp(map.tex)
            self.dump(map.name)

    def plot_1d_array(self, mapset, plot_axis, n_rows=None,
            n_cols=None, fname=None, **kwargs):
        ''' plot 1d projections as an array '''
        self.plot_array(mapset, 'plot_1d_projection', plot_axis, n_rows=n_rows,
                n_cols=n_cols, **kwargs)
        self.dump(fname)

    def plot_1d_slices_array(self, mapsets, plot_axis, fname=None, **kwargs):
        ''' plot 1d slices as an array '''
        self.slices_array(mapsets, plot_axis, **kwargs)
        self.dump(fname)

    def plot_1d_all(self, mapset, plot_axis, **kwargs):
        ''' all one a single plot '''
        self.init_fig()
        for map in mapset:
            self.plot_1d_projection(map, plot_axis, **kwargs)
        self.add_stamp()
        self.add_leg()
        self.dump('all')

    def plot_1d_stack(self, mapset, plot_axis, **kwargs):
        ''' all maps stacked on top of each other '''
        self.init_fig()
        for i, map in enumerate(mapset):
            for j in range(i):
                map += mapset[j]
            self.plot_1d_projection(map, plot_axis, **kwargs)
        self.add_stamp()
        self.add_leg()
        self.dump('stack')

    def plot_1d_cmp(self, mapsets, plot_axis, fname=None, **kwargs):
        ''' 1d comparisons for two mapsets as projections'''
        for i in range(len(mapsets[0])):
            maps = [mapset[i] for mapset in mapsets]
            self.init_fig()
            if self.ratio:
                ax1 = plt.subplot2grid((4,1), (0,0), rowspan=3)
                plt.setp(ax1.get_xticklabels(), visible=False)
            self.reset_colors()
            for map in maps:
                self.next_color()
                self.plot_1d_projection(map, plot_axis, **kwargs)
            self.add_stamp()
            self.add_leg()
            if self.ratio:
                plt.subplot2grid((4,1), (3,0),sharex=ax1)
                self.plot_1d_ratio(maps, plot_axis, **kwargs)
            self.dump('%s_%s'%(fname,maps[0].name))


    # --- plotting core functions ---

    def plot_array(self, mapset, fun, *args, **kwargs):
        ''' wrapper funtion to exccute plotting function fun for every map in a set
        distributed over a grid '''
        n_rows = kwargs.pop('n_rows', None)
        n_cols = kwargs.pop('n_cols', None)
        split_axis = kwargs.pop('split_axis', None)
        ''' plot mapset in array using a function fun '''
        if isinstance(mapset, Map):
            mapset = MapSet([mapset])



        # if dimensionality is 3, then still define a spli_axis automatically
        new_maps = []
        for map in mapset:
            if len(map.binning) == 3:
                if split_axis is None:
                    # shortest dimension
                    l = [binning.num_bins for binning in map.binning]
                    idx = l.index(min(l))
                    s_axis = map.binning.names[idx]
                    logging.warning('automatically splitting along %s axis'%s_axis)
                else:
                    s_axis = split_axis
                split_idx = map.binning.names.index(s_axis)
                new_binning = MultiDimBinning([binning for binning in map.binning if binning.name != s_axis])
                for i in range(map.binning[s_axis].num_bins):
                    newmap = Map(name=map.name+'_%s_%i'%(s_axis,i),tex=map.tex+'\ %s\ bin\ %i'%(s_axis,i), hist = np.rollaxis(map.hist, split_idx, 0)[i], binning=new_binning)
                    new_maps.append(newmap)
            elif len(map.binning) == 2:
                new_maps.append(map)
            else:
                raise Exception('Cannot plot %i dimensional map in 2d'%len(map))
        mapset = MapSet(new_maps)

        if isinstance(mapset, MapSet):
            n = len(mapset)
        elif isinstance(mapset, TransformSet):
            n = len([x for x in mapset])
        else:
            raise TypeError('Expecting to plot a MapSet or TransformSet but '
                            'got %s'%type(mapset))
        if n_rows is None and n_cols is None:
            # TODO: auto row/cols
            n_rows = math.floor(math.sqrt(n))
            while n % n_rows != 0:
               n_rows -= 1
            n_cols = n / n_rows
        assert( n <= n_cols * n_rows), 'you are trying to plot %s subplots on a grid with %s x %s cells'%(n, n_cols, n_rows)
        size = (n_cols*self.size[0], n_rows*self.size[1])
        self.init_fig(size)
        plt.tight_layout()
        h_margin = 1. / size[0]
        v_margin = 1. / size[1]
        self.fig.subplots_adjust(hspace=0.3, wspace=0.3, top=1-v_margin, bottom=v_margin, left=h_margin, right=1-h_margin)
        for i, map in enumerate(mapset):
            plt.subplot(n_rows,n_cols,i+1)
            getattr(self, fun)(map, *args, **kwargs)
            self.add_stamp(map.tex)

    def slices_array(self, mapsets, plot_axis, *args, **kwargs):
        ''' plot mapset in array using a function fun '''
        n_cols = len(mapsets[0])
        plt_binning = mapsets[0][0].binning[plot_axis]
        plt_axis_n = mapsets[0][0].binning.names.index(plot_axis)
        # determine how many slices we need accoring to mapset[0]
        n_rows = 0
        assert(len(mapsets[0][0].binning) == 2), 'only supported for 2d maps right now'
        slice_axis_n = int(not plt_axis_n)
        slice_axis = mapsets[0][0].binning.names[slice_axis_n]
        n_rows = mapsets[0][0].binning[slice_axis].num_bins
        size = (n_cols*self.size[0], self.size[1])
        self.init_fig(size)
        plt.tight_layout()
        h_margin = 1. / size[0]
        v_margin = 1. / size[1]
        # big one
        self.fig.subplots_adjust(hspace=0., wspace=0.3, top=1-v_margin, bottom=v_margin, left=h_margin, right=1-h_margin)
        stamp = self.stamp
        for i in range(len(mapsets[0])):
            for j in range(n_rows):
                plt.subplot(n_rows,n_cols,i + (n_rows - j -1)*n_cols + 1)
                self.reset_colors()
                for mapset in mapsets:
                    self.next_color()
                    map = mapset[i]
                    if slice_axis_n == 0:
                        map_slice = map[j,:]  
                    else:
                        map_slice = map[:,j]
                    a_text = map_slice.binning[slice_axis].label + ' [%.2f, %.2f]'%(map_slice.binning[slice_axis].bin_edges[0].m, map_slice.binning[slice_axis].bin_edges[-1].m)
                    self.plot_1d_projection(map_slice,plot_axis, **kwargs)
                    if not j == 0:
                       plt.gca().get_xaxis().set_visible(False) 
                    if j == n_rows - 1:
                        if map.name == 'cscd': title = 'cascades channel'
                        if map.name == 'trck': title = 'tracks channel'
                        plt.gca().set_title(title)
                    plt.gca().set_ylabel('')
                    plt.gca().yaxis.set_tick_params(labelsize=8)
                    self.stamp = a_text
                    self.add_stamp(prop=dict(size=8))
        self.stamp = stamp

    def plot_2d_map(self, map, cmap='rainbow', **kwargs):
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        ''' plot map or transform on current axis in 2d'''
        axis = plt.gca()
        if isinstance(map, BinnedTensorTransform):
            bins = [map.input_binning[name] for name in map.input_binning.names]
            bin_edges = map.input_binning.bin_edges
            bin_centers = map.input_binning.weighted_centers
            linlog = all([(b.is_log or b.is_lin) for b in map.input_binning])
            xform_array = unp.nominal_values(map.xform_array)
            zmap = np.log10(unp.nominal_values(map.xform_array)) if self.log else unp.nominal_values(map.xform_array)
        elif isinstance(map, Map):
            bins = [map.binning[name] for name in map.binning.names]
            bin_edges = map.binning.bin_edges
            bin_centers = map.binning.weighted_centers
            linlog = all([(b.is_log or b.is_lin) for b in map.binning])
            zmap = np.log10(unp.nominal_values(map.hist)) if self.log else unp.nominal_values(map.hist)
        if self.symmetric:
            vmax = max(np.max(np.ma.masked_invalid(zmap)), - np.min(np.ma.masked_invalid(zmap)))
            vmin = -vmax
        else:
            if vmax == None:
                vmax = np.max(zmap[np.isfinite(zmap)])
            if vmin == None:
                vmin = np.min(zmap[np.isfinite(zmap)])
        extent = [np.min(bin_edges[0].m), np.max(bin_edges[0].m), np.min(bin_edges[1].m), np.max(bin_edges[1].m)]
        if linlog:
            # needs to be transposed for imshow
            img = plt.imshow(zmap.T,origin='lower',interpolation='nearest',extent=extent,aspect='auto',
                cmap=cmap, vmin=vmin, vmax=vmax,**kwargs)
        else:
            # only lin or log can be handled by imshow...otherise use colormesh
            x,y = np.meshgrid(bin_edges[0],bin_edges[1])
            img = plt.pcolormesh(x,y,zmap.T,vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        if self.annotate:
            #counts = img.get_array().T
            for i in range(len(bin_centers[0])):
                for j in range(len(bin_centers[1])):
                    bin_x = bin_centers[0][i].m
                    bin_y = bin_centers[1][j].m
                    plt.annotate('%.1f'%(zmap[i,j]), xy=(bin_x, bin_y), xycoords=('data', 'data'), xytext=(bin_x, bin_y), textcoords='data', va='top', ha='center', size=7)

        axis.set_xlabel(bins[0].label)
        axis.set_ylabel(bins[1].label)
        axis.set_xlim(extent[0:2])
        axis.set_ylim(extent[2:4])
        if bins[0].is_log:
            axis.set_xscale('log')
        if bins[1].is_log:
            axis.set_yscale('log')
        col_bar = plt.colorbar(format=r'$10^{%.1f}$') if self.log else plt.colorbar()
        if self.label:
            col_bar.set_label(self.label)
        
    def plot_1d_projection(self, map, plot_axis, **kwargs):
        ''' plot map projected on plot_axis'''
        axis = plt.gca()
        plt_binning = map.binning[plot_axis]
        hist = self.project_1d(map, plot_axis)
        if map.tex == 'data':
            axis.errorbar(plt_binning.weighted_centers.m,
                    unp.nominal_values(hist),yerr=unp.std_devs(hist), fmt='o', markersize='4', label='data', color='k', ecolor='k', mec='k', **kwargs)
        else:
            axis.hist(plt_binning.weighted_centers, weights=unp.nominal_values(hist), bins=plt_binning.bin_edges, histtype='step', lw=1.5, label=r'$%s$'%map.tex, color=self.color, **kwargs)
            axis.bar(plt_binning.bin_edges.m[:-1],2*unp.std_devs(hist),
                    bottom=unp.nominal_values(hist)-unp.std_devs(hist),
                    width=plt_binning.bin_widths, alpha=0.25, linewidth=0, color=self.color,
                    **kwargs)
        axis.set_xlabel(plt_binning.label)
        if self.label:
            axis.set_ylabel(self.label)
        if plt_binning.is_log:
            axis.set_xscale('log')
        if self.log:
            axis.set_yscale('log')
        else:
            axis.set_ylim(0,np.max(unp.nominal_values(hist))*1.4)
        axis.set_xlim(plt_binning.bin_edges.m[0], plt_binning.bin_edges.m[-1])
        if self.grid:
            plt.grid(True, which="both", ls='-', alpha=0.2)

    def project_1d(self, map, plot_axis):
        ''' sum up a map along all axes except plot_axis '''
        hist = map.hist
        plt_axis_n = map.binning.names.index(plot_axis)
        for i in range(len(map.binning)):
            if i == plt_axis_n:
                continue
            hist = np.sum(map.hist, i)
        return hist

    def plot_1d_ratio(self, maps, plot_axis):
        ''' make a ratio plot for a 1d projection '''
        axis = plt.gca()
        map0 = maps[0]
        plt_binning = map0.binning[plot_axis]
        hist = self.project_1d(map0, plot_axis)
        hist0 = unp.nominal_values(hist)
        err0 = unp.std_devs(hist)

        axis.set_xlim(plt_binning.bin_edges.m[0], plt_binning.bin_edges.m[-1])
        maximum = 1.0
        minimum = 1.0
        self.reset_colors()
        for j,map in enumerate(maps):
            self.next_color()
            hist = self.project_1d(map, plot_axis)
            hist1 = unp.nominal_values(hist)
            err1 = unp.std_devs(hist)
            ratio = np.zeros_like(hist0)
            ratio_error = np.zeros_like(hist0)
            for i in range(len(hist0)):
                if hist1[i]==0 and hist0[i]==0:
                    ratio[i] = 1.
                    ratio_error[i] = 1.
                elif hist1[i]!=0 and hist0[i]==0:
                    logging.warning('deviding non 0 by 0 for ratio')
                    ratio[i] = 0.
                    ratio_error[i] = 1.
                else:
                    ratio[i] = hist1[i]/hist0[i]
                    ratio_error[i] = err1[i]/hist0[i]
                    minimum = min(minimum,ratio[i])
                    maximum = max(maximum,ratio[i])

            if map.tex == 'data':
                axis.errorbar(plt_binning.weighted_centers.m,
                    ratio,yerr=ratio_error, fmt='o', markersize='4', label='data', color='k', ecolor='k', mec='k')
            else:
                h,b,p = axis.hist(plt_binning.weighted_centers, weights=ratio,
                    bins=plt_binning.bin_edges, histtype='step', lw=1.5,
                    label=r'$%s$'%map.tex, color=self.color)
                axis.bar(plt_binning.bin_edges.m[:-1],2*ratio_error, bottom=ratio-ratio_error,
                        width=plt_binning.bin_widths, alpha=0.25,
                        linewidth=0, color=self.color)

        if self.grid:
            plt.grid(True, which="both", ls='-', alpha=0.2)
        self.fig.subplots_adjust(hspace=0)
        axis.set_ylabel('ratio')
        axis.set_xlabel(plt_binning.label)
        # calculate nice scale:
        off = max(maximum-1, 1-minimum)
        axis.set_ylim(1 - 1.2 * off, 1 + 1.2 * off )

    def plot_xsec(self, MapSet, ylim=None, logx=True):
        from pisa.utils import fileio

        zero_np_element = np.array([0])
        for Map in MapSet:
            bins = Map.binning
            try:
                energy_binning = bins.true_energy
            except:
                energy_binning = bins.reco_energy

            fig = plt.figure(figsize=self.size)
            fig.suptitle(Map.name, y=0.95)
            ax = fig.add_subplot(111)
            ax.grid(b=True, which='major')
            ax.grid(b=True, which='minor', linestyle=':')
            plt.xlabel(energy_binning.label, size=18)
            plt.ylabel(self.label, size=18)
            if self.log:
                ax.set_yscale('log')
            if logx:
                ax.set_xscale('log')
            if ylim:
                ax.set_ylim(ylim)
            ax.set_xlim(np.min(energy_binning.bin_edges.m),
                        np.max(energy_binning.bin_edges.m))

            hist = Map.hist
            array_element = np.hstack((hist, zero_np_element))
            ax.step(energy_binning.bin_edges.m, array_element, where='post')

            fileio.mkdir(self.outdir)
            fig.savefig(self.outdir+'/'+Map.name+'.png', bbox_inches='tight',
                        dpi=150)

