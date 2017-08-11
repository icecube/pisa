"""
Plotter class for doing plots easily

"""


import itertools

import numpy as np
from uncertainties import unumpy as unp

import matplotlib as mpl
# Headless mode; must set prior to pyplot import
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.ticker
import scipy.stats as STATS
import os

from pisa.core.map import Map, MapSet
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.format import dollars, text2tex, tex_join
from pisa.utils.log import logging
from pisa.utils import fileio
from pisa.utils.fileio import from_file, to_file


__all__ = ['Plotter']


# set fonts
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['xtick.minor.size'] = 0
mpl.rcParams['xtick.minor.width'] = 0


class Plotter(object):
    """
    Plotting library for PISA `Map`s, `MapSet`s, and `BinnedTensorTransform`s

    Parameters
    ----------
    outdir : str
        output directory path

    stamp : str
        stamp to be put on every subplot, e.g. 'Preliminary',
        'DeepCore nutau', etc.

    fmt : str or iterable of str
        formats to be plotted, e.g. ['pdf', 'png']

    size : (int, int)
        canvas size (inches)

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

    """
    def __init__(self, outdir='.', stamp=None, size=(8, 8), fmt='pdf',
                 log=True, label='# events', grid=True, ratio=False,
                 annotate=False, symmetric=False, loc='outside'):
        self.fig = None
        self.axes = None

        self.outdir = outdir
        self.stamp = stamp
        if isinstance(fmt, basestring):
            fmt = [fmt]
        self.fmt = fmt
        self.size = size
        self.fig = None
        self.log = log
        self.label = label
        self.grid = grid
        self.ratio = ratio
        self.annotate = annotate
        if symmetric:
            assert(self.log == False), 'cannot do log and symmetric at the same time'
        self.symmetric = symmetric
        self.reset_colors()
        self.color = 'b'
        self.loc = loc

    def reset_colors(self):
        self.colors = itertools.cycle(["r", "b", "g"])

    def next_color(self):
        self.color = next(self.colors)

    # --- helper functions ---

    def init_fig(self, figsize=None):
        """clear/initialize figure"""
        if figsize is not None:
            size = figsize
        else:
            size = self.size
        if self.fig is not None:
            plt.clf()
            plt.close(self.fig)
        self.fig, self.axes = plt.subplots(1, 1, figsize=size)
        self.fig.patch.set_facecolor('none')

    def add_stamp(self, text=None, **kwargs):
        """Add common stamp with text.

        NOTE add_stamp cannot be used on a subplot that has been de-selected
        and then re-selected. It will write over existing text.

        """
        stamp = tex_join('\n', self.stamp, text)
        if self.loc == 'inside':
            a_text = AnchoredText(dollars(stamp), loc=2, frameon=False,
                                  **kwargs)
            plt.gca().add_artist(a_text)
        elif self.loc == 'outside':
            plt.gca().set_title(dollars(stamp))

    def add_leg(self):
        """initialize legend """
        plt.gca().legend(loc='upper right', ncol=2, frameon=False, numpoints=1)

    def dump(self, fname):
        """dump figure to file"""
        for fmt in self.fmt:
            plt.savefig(self.outdir+'/'+fname+'.'+fmt, dpi=150,
                        edgecolor='none', facecolor=self.fig.get_facecolor())
    
    def add_xticks(self, xticks):
        plt.xticks(xticks)
        plt.gca().set_xticklabels(xticks)
        plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    def add_yticks(self, yticks):
        plt.yticks(yticks)
        plt.gca().set_yticklabels(yticks)
        plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # --- 2d plots ---

    def plot_2d_single(self, map_set, **kwargs):
        """plot all maps in individual plots"""
        if isinstance(map_set, Map):
            map_set = [map_set]
        for map in map_set:
            self.init_fig()
            self.plot_2d_map(map, **kwargs)
            self.add_stamp(map.tex)
            self.dump(map.name)

    def plot_2d_array(self, map_set, n_rows=None, n_cols=None, fname=None,
                      **kwargs):
        """plot all maps or transforms in a single plot"""
        if fname is None:
            fname = 'test2d'
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        self.plot_array(map_set, 'plot_2d_map', n_rows=n_rows, n_cols=n_cols,vmin=vmin, vmax=vmax,
                        **kwargs)
        self.dump(fname)

    # --- 1d plots ---

    def plot_1d_single(self, map_set, plot_axis, **kwargs):
        """plot all maps in individual plots"""
        for map in map_set:
            self.init_fig()
            self.plot_1d_projection(map, plot_axis, **kwargs)
            self.add_stamp(map.tex)
            self.dump(map.name)

    def plot_1d_array(self, map_set, plot_axis, n_rows=None, n_cols=None,
                      fname=None, **kwargs):
        """plot 1d projections as an array"""
        self.plot_array(map_set, 'plot_1d_projection', plot_axis, n_rows=n_rows,
                        n_cols=n_cols, **kwargs)
        self.dump(fname)

    def plot_1d_slices_array(self, map_sets, plot_axis, fname=None, **kwargs):
        """plot 1d projections as an array"""
        self.slices_array(map_sets, plot_axis, **kwargs)
        self.dump(fname)

    def plot_1d_all(self, map_set, plot_axis, **kwargs):
        """all one one canvas"""
        self.init_fig()
        for map in map_set:
            self.plot_1d_projection(map, plot_axis, **kwargs)
        self.add_stamp()
        self.add_leg()
        self.dump('all')

    def plot_1d_stack(self, map_set, plot_axis, **kwargs):
        """all maps stacked on top of each other"""
        self.init_fig()
        for i, map in enumerate(map_set):
            for j in range(i):
                map += map_set[j]
            self.plot_1d_projection(map, plot_axis, **kwargs)
        self.add_stamp()
        self.add_leg()
        self.dump('stack')

    def plot_1d_cmp(self, map_sets, plot_axis, fname, **kwargs):
        """1d comparisons for two map_sets as projections"""
        hist_vals = []
        x_bins = []
        for i in range(len(map_sets[0])):
            maps = [map_set[i] for map_set in map_sets]
            self.init_fig()
            if self.ratio:
                ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
                plt.setp(ax1.get_xticklabels(), visible=False)
            self.reset_colors()
            for map in maps:
                self.next_color()
                hist_val, x_bin = self.plot_1d_projection(map, plot_axis, **kwargs)
                hist_vals.append(hist_val)
                x_bins.append(x_bin)
            self.add_stamp()
            self.add_leg()
            if self.ratio:
                plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
                self.plot_1d_ratio(maps, plot_axis, **kwargs)
            self.dump('%s_%s'%(fname, maps[0].name))
        return hist_vals, x_bins


    # --- plotting core functions ---

    def plot_array(self, map_set, fun, *args, **kwargs):
        """wrapper funtion to exccute plotting function fun for every map in a
        set distributed over a grid"""
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        n_rows = kwargs.pop('n_rows', None)
        n_cols = kwargs.pop('n_cols', None)
        split_axis = kwargs.pop('split_axis', None)
        if isinstance(map_set, Map):
            map_set = MapSet([map_set])

        # if dimensionality is 3, then still define a spli_axis automatically
        new_maps = []
        for map in map_set:
            if map.binning.num_dims == 3:
                if split_axis is None:
                    # Find shortest dimension
                    l = map.binning.num_bins
                    idx = l.index(min(l))
                    split_axis_ = map.binning.names[idx]
                    logging.warning(
                        'Plotter automatically splitting map %s along %s axis'
                        % (map.name, split_axis_)
                    )
                else:
                    split_axis_ = split_axis
                new_maps.extend(map.split(split_axis_))
            elif len(map.binning) == 2:
                new_maps.append(map)
            else:
                raise Exception('Cannot plot %i dimensional map in 2d'
                                %len(map))
        map_set = MapSet(new_maps)

        if isinstance(map_set, MapSet):
            n = len(map_set)
        elif isinstance(map_set, TransformSet):
            n = len([x for x in map_set])
        else:
            raise TypeError('Expecting to plot a MapSet or TransformSet but '
                            'got %s'%type(map_set))
        if n_rows is None and n_cols is None:
            # TODO: auto row/cols
            n_rows = np.floor(np.sqrt(n))
            while n % n_rows != 0:
                n_rows -= 1
            n_cols = n / n_rows
        assert (n <= n_cols * n_rows), 'trying to plot %s subplots on a grid with %s x %s cells'%(n, n_cols, n_rows)
        size = (n_cols*self.size[0], n_rows*self.size[1])
        self.init_fig(size)
        plt.tight_layout()
        h_margin = 1. / size[0]
        v_margin = 1. / size[1]
        self.fig.subplots_adjust(hspace=0.3, wspace=0.3, top=1-v_margin,
                                 bottom=v_margin, left=h_margin,
                                 right=1-h_margin)
        for i, map in enumerate(map_set):
            plt.subplot(n_rows, n_cols, i+1)
            getattr(self, fun)(map, vmin=vmin, vmax=vmax,*args, **kwargs)
            self.add_stamp(map.tex)

    def slices_array(self, map_sets, plot_axis, *args, **kwargs):
        """plot map_set in array using a function fun"""
        n_cols = len(map_sets[0])
        plt_binning = map_sets[0][0].binning[plot_axis]
        plt_axis_n = map_sets[0][0].binning.names.index(plot_axis)
        # determine how many slices we need accoring to map_set[0]
        n_rows = 0
        assert(len(map_sets[0][0].binning) == 2), 'only supported for 2d maps right now'
        slice_axis_n = int(not plt_axis_n)
        slice_axis = map_sets[0][0].binning.names[slice_axis_n]
        n_rows = map_sets[0][0].binning[slice_axis].num_bins
        size = (n_cols*self.size[0], self.size[1])
        self.init_fig(size)
        plt.tight_layout()
        h_margin = 1. / size[0]
        v_margin = 1. / size[1]
        # big one
        self.fig.subplots_adjust(hspace=0., wspace=0.3, top=1-v_margin,
                                 bottom=v_margin, left=h_margin,
                                 right=1-h_margin)
        stamp = self.stamp
        for i in range(len(map_sets[0])):
            for j in range(n_rows):
                plt.subplot(n_rows, n_cols, i + (n_rows - j -1)*n_cols + 1)
                self.reset_colors()
                for map_set in map_sets:
                    self.next_color()
                    map = map_set[i]
                    if slice_axis_n == 0:
                        map_slice = map[j, :]
                    else:
                        map_slice = map[:, j]
                    a_text = (
                        map_slice.binning[slice_axis].label
                        + ' [%.2f, %.2f]'
                        %(map_slice.binning[slice_axis].bin_edges[0].m,
                          map_slice.binning[slice_axis].bin_edges[-1].m)
                    )
                    self.plot_1d_projection(map_slice, plot_axis,
                                            *args, **kwargs)
                    if j != 0:
                        plt.gca().get_xaxis().set_visible(False)
                    if j == n_rows - 1:
                        if map.name == 'cscd':
                            title = 'cascades channel'
                        if map.name == 'trck':
                            title = 'tracks channel'
                        plt.gca().set_title(title)
                    plt.gca().set_ylabel('')
                    plt.gca().yaxis.set_tick_params(labelsize=8)
                    self.stamp = a_text
                    self.add_stamp(prop=dict(size=8))
        self.stamp = stamp

    def plot_2d_map(self, map, cmap='rainbow', **kwargs):
        """plot map or transform on current axis in 2d"""
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        axis = plt.gca()

        if isinstance(map, BinnedTensorTransform):
            binning = map.input_binning
        elif isinstance(map, Map):
            binning = map.binning
        else:
            raise TypeError('Unhandled `map` type %s' % map.__class__.__name__)

        dims = binning.dims
        bin_centers = binning.weighted_centers
        bin_edges = binning.bin_edges
        #print "bin_edges = ", bin_edges
        linlog = all([(d.is_log or d.is_lin) for d in binning])

        zmap = map.nominal_values
        if self.log:
            zmap = np.log10(zmap)

        if self.symmetric:
            vmax = max(np.max(np.ma.masked_invalid(zmap)),
                       - np.min(np.ma.masked_invalid(zmap)))
            vmin = -vmax
        else:
            if vmax == None:
                vmax = np.max(zmap[np.isfinite(zmap)])
            if vmin == None:
                vmin = np.min(zmap[np.isfinite(zmap)])
        extent = [np.min(bin_edges[0].m), np.max(bin_edges[0].m),
                  np.min(bin_edges[1].m), np.max(bin_edges[1].m)]

        # Only lin or log can be handled by imshow...otherise use colormesh
        if linlog:
            # Needs to be transposed for imshow
            #img = plt.imshow(
            #    #zmap.T, origin='lower', interpolation='nearest', extent=extent,
            #    zmap.T, origin='lower', extent=extent,
            #    aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, **kwargs
            #)
            x, y = np.meshgrid(bin_edges[0], bin_edges[1])
            img = plt.pcolormesh(x, y, zmap.T, vmin=vmin, vmax=vmax, cmap=cmap,
                                 **kwargs)
        else:
            x, y = np.meshgrid(bin_edges[0], bin_edges[1])
            img = plt.pcolormesh(x, y, zmap.T, vmin=vmin, vmax=vmax, cmap=cmap,
                                 **kwargs)
        if self.annotate:
            for i in range(len(bin_centers[0])):
                for j in range(len(bin_centers[1])):
                    bin_x = bin_centers[0][i].m
                    bin_y = bin_centers[1][j].m
                    plt.annotate(
                        '%.2f'%(zmap[i, j]),
                        xy=(bin_x, bin_y),
                        xycoords=('data', 'data'),
                        xytext=(bin_x, bin_y),
                        textcoords='data',
                        va='top',
                        ha='center',
                        size=7
                    )

        axis.set_xlabel(dollars(text2tex(dims[0].label)))
        axis.set_ylabel(dollars(text2tex(dims[1].label)))
        axis.set_xlim(extent[0:2])
        axis.set_ylim(extent[2:4])
        # TODO: use log2 scale & integer tick labels if too few major gridlines
        # result from default log10 scale
        if dims[0].is_log:
            axis.set_xscale('log')
        if dims[1].is_log:
            axis.set_yscale('log')

        if self.log:
            col_bar = plt.colorbar(format=r'$10^{%.1f}$')
        else:
            col_bar = plt.colorbar()

        if self.label:
            col_bar.set_label(dollars(text2tex(self.label)))
        self.add_xticks(bin_edges[0].m)
        self.add_yticks(bin_edges[1].m)


    def plot_1d_projection(self, map, plot_axis, **kwargs):
        """plot map projected on plot_axis"""
        axis = plt.gca()
        plt_binning = map.binning[plot_axis]
        hist = self.project_1d(map, plot_axis)
        split_axis = kwargs.pop('split_axis', None)
        if map.tex == 'data':
            axis.errorbar(
                plt_binning.weighted_centers.m, unp.nominal_values(hist),
                yerr=unp.std_devs(hist),
                fmt='o', markersize='4', label=dollars(text2tex('data')),
                color='k', ecolor='k', mec='k', **kwargs
            )
        else:
            #axis.bar(
            #    left=plt_binning.bin_edges.m[:-1], height=2*unp.std_devs(hist),
            axis.bar( plt_binning.weighted_centers.m, height=2*unp.std_devs(hist),
                bottom=unp.nominal_values(hist)-unp.std_devs(hist),
                width=plt_binning.bin_widths, alpha=0.25, linewidth=0,
                color=self.color, **kwargs
            )
            hist_val,x_bins,_ = axis.hist(
                x=plt_binning.weighted_centers.m, weights=unp.nominal_values(hist),
                bins=plt_binning.bin_edges.m, histtype='step', lw=1.5,
                label=dollars(text2tex(map.tex)), color=self.color, **kwargs
            )
        axis.set_xlabel(dollars(text2tex(plt_binning.label)))
        if self.label:
            axis.set_ylabel(dollars(text2tex(self.label)))
        if plt_binning.is_log:
            axis.set_xscale('log')
        if self.log:
            axis.set_yscale('log')
        else:
            #axis.set_ylim(0, np.max(unp.nominal_values(hist))*1.4)
            axis.set_ylim(0, 9000)
        axis.set_xlim(plt_binning.bin_edges.m[0], plt_binning.bin_edges.m[-1])
        if self.grid:
            plt.grid(True, which="major", ls='-', alpha=0.2, xdata=(plt_binning.bin_edges.m))
        self.add_xticks(plt_binning.bin_edges.m)
        return hist_val, x_bins

    def project_1d(self, map, plot_axis):
        """sum up a map along all axes except plot_axis"""
        hist = map.hist
        plt_axis_n = map.binning.names.index(plot_axis)
        idx = range(0,len(map.binning))
        idx.remove(plt_axis_n)
        sum_over_axis = tuple(idx)
        hist = np.sum(map.hist, sum_over_axis)
        return hist

    def plot_1d_ratio(self, maps, plot_axis):
        """make a ratio plot for a 1d projection"""
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
        for j, map in enumerate(maps):
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
                    minimum = min(minimum, ratio[i])
                    maximum = max(maximum, ratio[i])

            if map.tex == 'data':
                axis.errorbar(
                    plt_binning.weighted_centers.m, ratio, yerr=ratio_error,
                    fmt='o', markersize='4', label=dollars(text2tex('data')),
                    color='k', ecolor='k', mec='k'
                )
            else:
                h, b, p = axis.hist(
                    plt_binning.weighted_centers, weights=ratio,
                    bins=plt_binning.bin_edges, histtype='step', lw=1.5,
                    label=dollars(text2tex(map.tex)), color=self.color)
                axis.bar(
                    plt_binning.bin_edges.m[:-1], 2*ratio_error,
                    bottom=ratio-ratio_error, width=plt_binning.bin_widths,
                    alpha=0.25, linewidth=0, color=self.color
                )

        if self.grid:
            plt.grid(True, which="both", ls='-', alpha=0.2)
        self.fig.subplots_adjust(hspace=0)
        axis.set_ylabel(dollars(text2tex('ratio')))
        axis.set_xlabel(dollars(text2tex(plt_binning.label)))
        # Calculate nice scale:
        off = max(maximum-1, 1-minimum)
        axis.set_ylim(1 - 1.2 * off, 1 + 1.2 * off )

    def plot_xsec(self, map_set, ylim=None, logx=True):

        zero_np_element = np.array([0])
        for map in map_set:
            binning = map.binning
            if 'true_energy' in binning.names:
                energy_binning = binning.true_energy
            elif 'reco_energy' in binning.names:
                energy_binning = binning.reco_energy
            else:
                dim_idx = binning.index('energy', use_basenames=True)
                energy_binning = binning.dims[dim_idx]

            fig = plt.figure(figsize=self.size)
            fig.suptitle(map.name, y=0.95)
            ax = fig.add_subplot(111)
            ax.grid(b=True, which='major')
            ax.grid(b=True, which='minor', linestyle=':')
            plt.xlabel(dollars(text2tex(energy_binning.label)), size=18)
            plt.ylabel(dollars(text2tex(self.label)), size=18)
            if self.log:
                ax.set_yscale('log')
            if logx:
                ax.set_xscale('log')
            if ylim:
                ax.set_ylim(ylim)
            ax.set_xlim(np.min(energy_binning.bin_edges.m),
                        np.max(energy_binning.bin_edges.m))

            hist = map.hist
            array_element = np.hstack((hist, zero_np_element))
            ax.step(energy_binning.bin_edges.m, array_element, where='post')

            fileio.mkdir(self.outdir)
            fig.savefig(self.outdir+'/'+map.name+'.png', bbox_inches='tight',
                        dpi=150)

    def plot_low_level_quantities_data_vs_data(self, data1_arrays, data2_arrays, label1, label2, param_to_plot, fig_name, outdir, title, logy=False, **kwargs):
        #print "Plotting ", param_to_plot
        data1_param = data1_arrays[param_to_plot]
        data2_param = data2_arrays[param_to_plot]
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        plt.setp(ax1.get_xticklabels(), visible=False)
        x_range = self.special_range(param_to_plot)
        if x_range=="":
            x_range = (min(np.min(data1_param), np.min(data2_param)), max(np.max(data1_param), np.max(data2_param)))
        nbins= self.special_bins(param_to_plot)
        hist_y1, hist_bin_edges,_ = ax1.hist(data1_param, bins=nbins, range = x_range, histtype='step',lw=2,color='b', linestyle='solid', label=label1)
        hist_y2, hist_bin_edges,_ = ax1.hist(data2_param, bins=hist_bin_edges, range = x_range, histtype='step',lw=2,color='r', linestyle='solid', label=label2)
        ax2=plt.subplot2grid((4,1), (3,0),sharex=ax1)
        ax2.axhline(y=1,linewidth=1, color='k')
        ratio_2_1 = hist_y2/hist_y1
        x_edges=hist_bin_edges
        x = (x_edges[:-1]+x_edges[1:])/2.0
        #print "max ratio_2_1", max(ratio_2_1)
        hist_ratio, hist_bin_edges,_ = ax2.hist(x, weights=ratio_2_1 , bins=x_edges, histtype='step',lw=2,color='g', linestyle='solid')
        #ax2.set_ylim([0.94,1.06])
        ax2.set_ylim([0.8,1.2])
        if param_to_plot=='pid':
            plt.gca().set_xscale('symlog')
        if param_to_plot=='dunkman_L5':
            plt.xlabel('bdt_score')
        else:
            plt.xlabel(param_to_plot)
        plt.legend()
        plt.grid()
        plt.title(title)
        file_name = fig_name    # fig name
        file_name += '_low_level_%s'% (param_to_plot)
        plot_name = outdir+'/'+file_name
        plt.savefig(plot_name+'.pdf')
        plt.savefig(plot_name+'.png')

        #hist_ratio_best_to_data, hist_bin_edges,_ = ax2.hist(x, weights=ratio_best_to_data , bins=x_edges, histtype='step',lw=2,color='g', linestyle='solid')

        #data1_y, x_edges = np.histogram(data1_param, bins=nbins)
        #data2_y, _ = np.histogram(data2_param, bins=x_edges)
        #ratio_2_1 = data2_y/data1_y
        #hist_ratio, hist_bin_edges,_ = plt.hist(x, weights=ratio_2_1 , bins=x_edges, histtype='step',lw=2,color='g', linestyle='solid')

    def special_range(self, param):
        x_ranges = {'santa_direct_doms': (0,45),
                'num_hit_doms': (0, 90),
                'separation': (0, 600),
                'CausalVetoHits': (0, 9),
                'pid': (-3, 400),
                'linefit_speed': (0,0.5),
                'dunkman_L5': (0.2,0.7),
                'cog_q1_z': (-475, -174),
                'DCFiducialPE': (0,200),
                'rt_fid_charge': (0,200),
                'santa_direct_charge':(0,200),
                'STW9000_DTW300PE':(0,200),
                'ICVetoPE':(0,5),
                'direct_C_q_early_pulses': (0,100)
                }
        if param in x_ranges.keys():
            return x_ranges[param]
        else:
            return ""

    def special_bins(self, param):
        x_bins={'first_hlc_rho': 200,
                'pid': np.array([ -3, -2.35354689, -1.76935606, -1.33017144, 0.0, 1.0, 1.33017144, 1.76935606, 2.35354689, 3.13062086, 4.16426245, 5.53918298, 7.368063, 9.80078696, 13.0367269, 17.34108178, 23.06661171, 30.68254809, 40.81304915, 54.28835233, 72.21281575, 96.05542506, 127.77018299, 169.9562482 , 226.07094726, 300.71311728, 400.0])
                }
        if param in x_bins.keys():
            return x_bins[param]
        else:
            return 20

    def plot_quantities(self, mc_arrays, icc_arrays, data_arrays, param_to_plot, fig_name, outdir, title, data_type='pseudo', logy=False, save_ratio=False, **kwargs):

        # get data param (data could be MC simulated "data" or real data)
        if data_arrays is not None:
            if data_type=='pseudo':
                data_param = np.array([])
                data_weight = np.array([])
                if type(data_arrays)==list:
                    for data_array in data_arrays:
                        for flav in data_array.keys():
                            data_param= np.append(data_param,data_array[flav][param_to_plot])
                            data_weight = np.append(data_weight, data_array[flav]['weight'])
                else:
                    for flav in data_arrays.keys():
                        param = data_arrays[flav][param_to_plot]
                        data_param = np.append(data_param, param)
                        data_weight = np.append(data_weight, data_arrays[flav]['weight'])
            if data_type=='real':
                data_param = data_arrays[param_to_plot]
                data_weight = np.ones(len(data_param))

        #print "IN PLOTTER, param ", param_to_plot
        file_name = fig_name    # fig name
        plt.figure()
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        if data_arrays is not None and self.ratio:
            plt.setp(ax1.get_xticklabels(), visible=False)
        else:
            plt.setp(ax1.get_xticklabels(), visible=True)
        file_name += '_%s'% (param_to_plot)
        nbins=30
        if mc_arrays is not None:
            mc_param_all = np.array([])
            mc_weight_all = np.array([])
            mc_sumw2_all = np.array([])
            for flav in mc_arrays.keys():
                param = mc_arrays[flav][param_to_plot]
                mc_param_all = np.append(mc_param_all, param)
                weight = mc_arrays[flav]['weight']
                sumw2 = mc_arrays[flav]['sumw2']
                mc_weight_all = np.append(mc_weight_all, weight)
                mc_sumw2_all = np.append(mc_sumw2_all, sumw2)

            # get rid of nans
            cut=np.logical_not(np.isnan(mc_param_all))
            mc_param_all = mc_param_all[cut]
            mc_weight_all= mc_weight_all[cut]
            mc_sumw2_all = mc_sumw2_all[cut]

            if param_to_plot=='first_hlc_rho':
                mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=200, **kwargs)
            elif param_to_plot=='santa_direct_doms':
                mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=nbins, range=(0, 45),**kwargs)
            elif param_to_plot=='num_hit_doms':
                mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=nbins, range=(0, 90),**kwargs)
            elif param_to_plot=='separation':
                mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=nbins, range=(0, 600),**kwargs)
            elif param_to_plot=='CausalVetoHits':
                mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=nbins, range=(0, 9),**kwargs)
            elif param_to_plot=='pid':
                x_bins = np.array([ -3, -2.35354689, -1.76935606, -1.33017144, 0.0, 1.0, 1.33017144, 1.76935606, 2.35354689, 3.13062086, 4.16426245,
                    5.53918298, 7.368063, 9.80078696, 13.0367269, 17.34108178, 23.06661171, 30.68254809, 40.81304915, 54.28835233, 72.21281575,
                    96.05542506, 127.77018299, 169.9562482 , 226.07094726, 300.71311728, 400.0])
                mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=x_bins, range=(-3, 400),**kwargs)
                plt.gca().set_xscale('symlog')
            elif param_to_plot=='linefit_speed':
                mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=nbins, range=(0,0.5),**kwargs)
            elif param_to_plot=='dunkman_L5':
                #mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=nbins, range=(-0.2,0.65),**kwargs)
                #mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=nbins, range=(0,0.7),**kwargs)
                mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=nbins, range=(0.2,0.7),**kwargs)
            elif param_to_plot=='cog_q1_z':
                mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=nbins, range=(-475, -174),**kwargs)
            elif param_to_plot in ['DCFiducialPE', 'rt_fid_charge', 'santa_direct_charge', 'STW9000_DTW300PE', 'total_charge']:
                mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=nbins, range=(0,200),**kwargs)
            elif param_to_plot=='ICVetoPE':
                mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=nbins, range=(0,5),**kwargs)
            elif param_to_plot=='direct_C_q_early_pulses':
                print "okayyyy"
                mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=nbins, range=(0,100),**kwargs)
            else:
                #print "aram = ", param_to_plot
                #print "max mc_param_all ", max(mc_param_all)
                #print "min mc_param_all ", min(mc_param_all)
                #print "\n"
                mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=nbins, **kwargs)
            mc_sumw2, x_edges = np.histogram(mc_param_all, weights=mc_sumw2_all, bins=x_edges, **kwargs)
        else:
            mc_y = np.zeros(nbins)
            mc_sumw2 = np.zeros(nbins)
            x_edges = None

        # add icc background
        if icc_arrays is not None:
            icc_param = icc_arrays[param_to_plot]
            if x_edges==None:
                icc_y, x_edges = np.histogram(icc_param, bins=nbins, **kwargs)
            else:
                icc_y, x_edges = np.histogram(icc_param, bins=x_edges, **kwargs)
            best_y = mc_y + icc_y * icc_arrays['weight']
            best_sumw2 = mc_sumw2 + icc_y * (icc_arrays['weight']**2)
        else:
            best_y = mc_y
            best_sumw2 = mc_sumw2

        if np.all(best_y==0):
            print "no mc or icc"
        else:
            x = (x_edges[:-1]+x_edges[1:])/2.0
            plt.hist(x, weights=best_y, bins=x_edges, histtype='step', lw=1.5, color='b', **kwargs)

        if data_arrays is not None:
            cut=np.logical_not(np.isnan(data_param))
            data_param = data_param[cut]
            data_weight = data_weight[cut]
            if x_edges==None:
                data_y, x_edges = np.histogram(data_param, weights=data_weight,bins=nbins)
            else:
                data_y, x_edges = np.histogram(data_param, weights=data_weight,bins=x_edges)
            err = np.sqrt(data_y)
            x = (x_edges[:-1]+x_edges[1:])/2.0
            plt.errorbar(x, data_y, yerr=err, fmt='o',color='k')

            if self.ratio:
                assert(mc_arrays!=None or icc_arrays!=None)
                ax2=plt.subplot2grid((4,1), (3,0),sharex=ax1)
                ax2.axhline(y=1,linewidth=1, color='k')
                ratio_best_to_data = np.zeros(len(best_y))
                ratio_error=np.zeros(len(best_y))
                #cut_nonzero = data_y!=0
                #data_y = data_y[cut_nonzero]
                #best_y = best_y[cut_nonzero]
                #best_sumw2 = best_sumw2[cut_nonzero]
                #x_array = x[cut_nonzero]
                for i in range(0, len(data_y)):
                    if data_y[i]==0:
                        if best_y[i]==0:
                            ratio_best_to_data[i]=1
                            ratio_error[i]=1
                        else:
                            ratio_best_to_data[i]=0
                            ratio_error[i]=1
                    else:
                        ratio_best_to_data[i] = best_y[i]/data_y[i]
                        ratio_error[i] = np.sqrt(best_sumw2)[i]/data_y[i]
                #hist_ratio_best_to_data,_,_ = ax2.hist(x, weights=ratio_best_to_data , bins=x_edges, histtype='step',lw=2,color='g', linestyle='solid', label='best fit / data')
                hist_ratio_best_to_data, hist_bin_edges,_ = ax2.hist(x, weights=ratio_best_to_data , bins=x_edges, histtype='step',lw=2,color='g', linestyle='solid')
                ax2.set_ylim([0.8,1.2])
                #bin_width=np.abs(x_array[:-1]-x_array[1:])
                #bin_width=np.abs(x[:-1]-x[1:])
                bin_width=np.abs(x_edges[1:]-x_edges[:-1])
                #print "len ratio_error ", len(ratio_error)
                #ax2.bar(x_edges[0:-1], 2*ratio_error,
                ax2.bar(x, 2*ratio_error,
                    bottom=hist_ratio_best_to_data-ratio_error, width=bin_width,
                    alpha=0.25, linewidth=0, color='g',
                )

                #chi2
                cut_nonzero = best_y!=0
                #cut_nonzero = np.logical_and(best_y!=0, data_y!=0)
                data_y = data_y[cut_nonzero]
                best_y = best_y[cut_nonzero]
                best_sumw2 = best_sumw2[cut_nonzero]
                x_array = x[cut_nonzero]

                #chi2_array = np.square(data_y-best_y)/(best_y)
                chi2_array = np.square(data_y-best_y)/(best_y+best_sumw2)
                chi2 = np.sum(chi2_array)
                dof = len(best_y)
                chi2_p = STATS.chisqprob(chi2, df=len(best_y))
                #print "chi2/dof, chi2_p = %.2f/%i %.2e"%(chi2, dof, chi2_p)
                #print "\n"
                a_text = ('chi2/dof = %.2f / %i, p = %.4f'%(chi2, dof, chi2_p))
                text_x = 0.98
                text_y= 0.98
                horizon_align = 'right'
                vertical_align='top'
                if 'QR3' in param_to_plot or 'QR6' in param_to_plot:
                    text_x = 0.02
                    text_y= 0.98
                    horizon_align = 'left'
                    vertical_align='top'
                ax1.text(text_x, text_y, a_text,
                        horizontalalignment=horizon_align,
                        verticalalignment=vertical_align,
                        transform=ax1.transAxes)

        plt.legend()
        if logy:
            print "in plotter logy ", logy
            plt.gca().set_yscale('log')
        if param_to_plot=='dunkman_L5':
            plt.xlabel('bdt_score')
        else:
            plt.xlabel(param_to_plot)
        plt.grid()
        plt.title(title)
        if not os.path.isdir(outdir):
            fileio.mkdir(outdir)
        plot_name = outdir+'/'+file_name
        plt.savefig(plot_name+'.pdf')
        plt.savefig(plot_name+'.png')
        plt.clf()

        if (data_arrays is not None) and self.ratio:
            plt.figure()
            plt.hist(x_array, weights=chi2_array, bins=x_edges, histtype='step', lw=1.5, color='r', **kwargs)
            if param_to_plot=='pid':
                plt.gca().set_xscale('symlog')
            # labels
            if param_to_plot=='dunkman_L5':
                plt.xlabel('bdt_score')
            else:
                plt.xlabel(param_to_plot)
            plt.ylabel('chi2')
            plt.grid()
            plt.title(title)
            plt.savefig(plot_name+'_chi2_distribution.pdf')
            #plt.savefig(outdir+'/'+'chi2_distribution'+file_name+'.pdf')
            plt.clf()

            if save_ratio:
                ratio_dict = {}
                # remove zero ratios from array
                zero_idx = np.where(hist_ratio_best_to_data == 0)[0]
                ratio_dict['bin_vals'] = np.delete(hist_ratio_best_to_data, zero_idx)
                ratio_dict['bin_edges'] = np.delete(hist_bin_edges, zero_idx)
                return ratio_dict
