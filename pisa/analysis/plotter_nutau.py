from pisa.utils.plotter import Plotter
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
from pisa.utils.log import logging
# Headless mode; must set prior to pyplot import
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.ticker
import scipy.stats as STATS
import os
import math
from copy import deepcopy

from pisa.utils.format import dollars, text2tex, tex_join
from pisa.core.map import Map, MapSet
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils import fileio

psublue='#1E407C'
psulightblue='#5f74e2'

# force matplotlib to use latex, code from Michael Richman 
plt.rc ('text', usetex=True)
plt.rc ('font', family='sans-serif')
plt.rc ('font', **{'sans-serif': 'Computer Modern Sans Serif'})
mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{sansmath}',
    r'\SetSymbolFont{operators}   {sans}{OT1}{cmss} {m}{n}'
    r'\SetSymbolFont{letters}     {sans}{OML}{cmbrm}{m}{it}'
    r'\SetSymbolFont{symbols}     {sans}{OMS}{cmbrs}{m}{n}'
    r'\SetSymbolFont{largesymbols}{sans}{OMX}{iwona}{m}{n}'
    r'\sansmath'
]
mpl.rcParams['axes.unicode_minus'] = False

class PlotterNutau(Plotter):
    def plot_2d_array(self, map_set, n_rows=None, n_cols=None, fname=None,
                      **kwargs):
        """plot all maps or transforms in a single plot"""
        if fname is None:
            fname = 'test2d'
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        suptitle = kwargs.pop('suptitle', '')
        colbar_percent = kwargs.pop('colbar_percent', False)
        self.plot_array(map_set, 'plot_2d_map', n_rows=n_rows, n_cols=n_cols,vmin=vmin, vmax=vmax,
                colbar_percent=colbar_percent, suptitle=suptitle, **kwargs)
        self.dump(fname)

    def plot_array(self, map_set, fun, *args, **kwargs):
        """wrapper funtion to exccute plotting function fun for every map in a
        set distributed over a grid"""
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        n_rows = kwargs.pop('n_rows', None)
        n_cols = kwargs.pop('n_cols', None)
        suptitle = kwargs.pop('suptitle', '')
        split_axis = kwargs.pop('split_axis', None)
        colbar_percent = kwargs.pop('colbar_percent', False)
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
        self.fig.suptitle(r"$\rm{%s}$"%suptitle, fontsize=23, x=0.5, y=0.92)
        #self.fig.suptitle(r"\rm{%s}"%suptitle, fontsize=22, x=0.5, y=0.92)
        plt.tight_layout()
        h_margin = 1. / size[0]
        v_margin = 1. / size[1]
        self.fig.subplots_adjust(hspace=0.3, wspace=0.1, top=1-v_margin, bottom=v_margin, left=h_margin, right=1-h_margin)
        for i, map in enumerate(map_set):
            plt.subplot(n_rows, n_cols, i+1)
            self.add_stamp(map.tex)
            show_colorbar = True if i+1==len(map_set) else False 
            show_y_label = True if i==0 else False
            getattr(self, fun)(map, vmin=vmin, vmax=vmax, show_y_label=show_y_label, show_colorbar=show_colorbar, colbar_percent=colbar_percent, *args, **kwargs)

    def plot_2d_map(self, map, cmap='rainbow', colorbarlabel=None, colbar_percent=False, **kwargs):
        """plot map or transform on current axis in 2d"""
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        show_colorbar = kwargs.pop('show_colorbar', True)
        show_y_label = kwargs.pop('show_y_label', True)
        #colbar_percent = kwargs.pop('colbar_percent', False)
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
            if vmax == None and vmin == None:
                vmax = max(np.max(np.ma.masked_invalid(zmap)),
                        - np.min(np.ma.masked_invalid(zmap)))
                vmin = -vmax
            else:
                if vmax is not None:
                    vmin = -vmax
                if vmin is not None:
                    vmax = np.abs(vmin)
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
                        '%.3f'%(zmap[i, j]),
                        xy=(bin_x, bin_y),
                        xycoords=('data', 'data'),
                        xytext=(bin_x, bin_y),
                        textcoords='data',
                        va='top',
                        ha='center',
                        size=7
                    )

        axis.set_xlabel(dollars(text2tex(dims[0].label)),fontsize=23)
        if show_y_label:
            axis.set_ylabel(dollars(text2tex(dims[1].label)),fontsize=23)
        axis.set_xlim(extent[0:2])
        axis.set_ylim(extent[2:4])
        # TODO: use log2 scale & integer tick labels if too few major gridlines
        # result from default log10 scale
        if dims[0].is_log:
            axis.set_xscale('log')
        if dims[1].is_log:
            axis.set_yscale('log')
        #self.add_xticks(bin_edges[0].m)
        #self.add_xticks(bin_edges[0].m[1::2])
        #axis.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        tick_x=bin_edges[0].m
        tick_x_label=['5.6', '', '10', '', '18', '', '32', '', '56']
        plt.xticks(tick_x)
        plt.gca().set_xticklabels(tick_x_label)
        if show_y_label:
            #self.add_yticks(bin_edges[1].m)
            #self.add_yticks(bin_edges[1].m[1::2])
            #axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axis.set_yticklabels(['-1.0','','-0.5','','0.0','','0.5','','1.0'])
        else:
            axis.set_yticklabels('')
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize(18) 
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(18) 
        if show_colorbar:
            self.add_colorbar(colorbarlabel, colbar_percent)

    def add_colorbar(self, colorbarlabel=None, colbar_percent=False, **kwargs):
        self.fig.subplots_adjust(right=0.84)
        axes=self.axes
        def percent(x, pos):
            #return r'{:1.1f}\%'.format(100.0*x)
            return r'{:2g}\%'.format(100.0*x)
        formatter = FuncFormatter(percent)
        if self.log:
            col_bar = plt.colorbar(format=r'$10^{%.1f}$',ax=self.axes.ravel().tolist())
        else:
            divider = make_axes_locatable(self.axes)
            print "axes location : ", divider.get_position()
            #cax = divider.append_axes("right", size="2%", pad=0.05)
            cax = divider.append_axes("right", size="2.5%", pad=0.95)
            #cax=self.fig.add_axes([0.116279069767, 0.2, 0.3, 0.6])
            #cax = divider.append_axes("right")
            #cax = divider.new_horizontal(size="2%", pad=0.05)
            #cax = divider.new_horizontal(size="2%")
            #cax = divider.new_vertical(size="2%")
            print "cax location: ", cax.get_position()
            cax.set_position((0.116279069767, 0.2, 0.3, 0.6))
            print "after set_positoin, cax location: ", cax.get_position()
            print "axes location : ", divider.get_position()
            if colbar_percent:
                col_bar = plt.colorbar(format=formatter,cax=cax)
            else:
                col_bar = plt.colorbar(cax=cax)
        if colorbarlabel is not None:
            col_bar.set_label(colorbarlabel, fontsize=21)
        elif self.label:
            print "text2tex(self.label)", text2tex(self.label)
            col_bar.set_label(dollars(text2tex(self.label)), fontsize=21)
        col_bar.ax.tick_params(labelsize=16) 

    def add_stamp(self, text=None, **kwargs):
        """Add common stamp with text.

        NOTE add_stamp cannot be used on a subplot that has been de-selected
        and then re-selected. It will write over existing text.

        """
        if text is not None and 'PID' in text and '0' in text:
            #text='Cascade-like'
            text=r'Cascade-like'
        if text is not None and 'PID' in text and '1' in text:
            #text='Track-like'
            text=r'Track-like'
        #stamp = tex_join('\n', self.stamp, text)
        stamp = ''.join((self.stamp, text))
        if self.loc == 'inside':
            a_text = AnchoredText(dollars(stamp), loc=2, frameon=False,
                                  **kwargs)
            plt.gca().add_artist(a_text)
        elif self.loc == 'outside':
            #plt.gca().set_title(dollars(stamp), fontsize=14)
            #plt.title(r'$\rm{%s}$'%stamp, fontsize=14)
            plt.gca().set_title(r'$\operatorname{%s}$'%stamp, fontsize=23)

    def plot_low_level_quantities_data_vs_data(self, data1_arrays, data2_arrays, label1, label2, param_to_plot, fig_name, outdir, title, logy=False, **kwargs):
        #print "Plotting ", param_to_plot
        data1_param = data1_arrays[param_to_plot]
        data2_param = data2_arrays[param_to_plot]
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        plt.setp(ax1.get_xticklabels(), visible=False)
        x_range = self.get_x_range(param_to_plot)
        if x_range=="":
            x_range = (min(np.min(data1_param), np.min(data2_param)), max(np.max(data1_param), np.max(data2_param)))
        nbins= self.get_x_nbins(param_to_plot)
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
            plt.gca().set_xscale('log')
        if param_to_plot=='dunkman_L5':
            plt.xlabel(r'BDT score')
        else:
            plt.xlabel(param_to_plot)
        plt.legend()
        plt.grid()
        if title!='':
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

    def get_x_label(self, param_to_plot):
        xlabels={'dunkman_L5': r'BDT Score',
                'l_over_e': r'$L/E \rm{\ [reco] \ (km/GeV)}$',
                #'l_over_e': r'$\rm{L/E \ [reco] \ (km/GeV)}$',
                'reco_energy': r'${E_\rm{reco} \rm{(GeV)}}$',
                #'reco_energy': r'$\rm{E_{reco} (GeV)}$',
                #'reco_coszen': r'$\mathrm{\cos{\theta}_{reco}}$',
                'reco_coszen': r'$\cos{\theta}_{\rm{reco}}$',
                'num_hit_doms': r'$\rm{No. \ of \ Hit \ DOMs}$',
                'rt_fid_charge': r'$\rm{RT \ Fiducial \ Charge}$',
                'CausalVetoPE': r'$\rm{Causal \ Veto \ PE}$',
                'pid': r'$\Delta \log{\mathcal{L_\mathrm{reco}}}$' 
                }
        #if param_to_plot=='dunkman_L5':
        #    return r'BDT score'
        #elif param_to_plot=='l_over_e':
        #    #return r'$\mathrm{L/E [reco]}$'+' (km/GeV)'
        #    return r'$\rm{L/E \ [reco] \ (km/GeV)}$'
        #elif param_to_plot=='reco_energy':
        #    #return r'$\rm{E [reco] (GeV)}$'
        #    return r'$\rm{E_{reco} (GeV)}$'
        #elif param_to_plot=='reco_coszen':
        #    #return r'$\mathrm{\cos{\theta} [reco]}$'
        #    return r'$\mathrm{\cos{\theta}_{reco}}$'
        if param_to_plot in xlabels.keys():
            return xlabels[param_to_plot]
        else:
            return param_to_plot.replace('_',' ')

    def get_flavor_label(self, group_mc_flavs):
        output_label=[None]*len(group_mc_flavs)
        for i,flav in enumerate(group_mc_flavs):
            if 'nue' in flav:
                output_label[i]=r'$\nu_e'
            if 'numu' in flav:
                output_label[i]=r'$\nu_{\mu}'
            if 'nutau' in flav:
                output_label[i]=r'$\nu_{\tau}'
            if flav=='nu_nc':
                output_label[i]=r'$\mathrm{\nu_{all}}'
            if flav=='muon':
                output_label[i]=r'$\mathrm{Atm.} \  \mu$'
            if flav!='muon':
                interaction=flav.split('_')[1]
                if interaction=='cc':
                    output_label[i]+=r'\ \rm{CC}$'
                if interaction=='nc':
                    output_label[i]+=r'\ \rm{NC}$'
        return output_label

    def get_x_range(self, param_to_plot):
        x_ranges = {'santa_direct_doms': (0,45),
                #'num_hit_doms': (0, 90),
                #'num_hit_doms': (8, 50),
                'separation': (0, 600),
                'CausalVetoHits': (0, 9),
                #'pid': (-3.60329646, 400),
                #'pid': (-3.60329646, 50),
                'linefit_speed': (0,0.5),
                #'dunkman_L5': (0.1,0.7),
                'dunkman_L5': (0.2,0.7),
                #'dunkman_L5': (-0.5,0.7),
                'cog_q1_z': (-475, -174),
                'DCFiducialPE': (0,200),
                #'rt_fid_charge': (0,200),
                'rt_fid_charge': (7,100),
                'santa_direct_charge':(0,200),
                'STW9000_DTW300PE':(0,200),
                'ICVetoPE':(0,5),
                'direct_C_q_early_pulses': (0,100),
                'l_over_e': (0,2500)
                }
        if param_to_plot in x_ranges.keys():
            return x_ranges[param_to_plot]
        else:
            return ""

    def get_x_nbins(self, param_to_plot):
        x_bins={'first_hlc_rho': 200,
                #'pid': np.array([-3.60329646, -2.0, -1.11009462, -0.61615503, -0.34199519, -0.18982351, -0.10536103, -0.05848035, -0.03245936, -0.01801648, -0.01, 0.01, 0.01801648, 0.03245936, 0.05848035, 0.10536103, 0.18982351, 0.34199519, 0.61615503, 1.11009462, 2.0, 3.60329646, 6.49187269, 11.69607095, 21.07220554, 37.96470182, 68.39903787, 123.23100555, 222.01892311, 400.]),
                #'pid': np.array([-3, -1, -0.3, 0, 0.3, 1.0, 3.0, 10, 30, 100]),
                #'pid': np.array([-3.0, -1.11009462, -0.34199519, -0.05848035, -0.01, 0.01, 0.05848035, 0.34199519, 0.61615503, 1.11009462, 2.0, 3.60329646, 6.49187269, 11.69607095, 21.07220554, 37.96470182, 68.39903787, 123.23100555, 222.01892311, 400.]),
                'pid': np.array([-3.0, -1.11009462, -0.34199519, 0.0, 0.34199519, 0.61615503, 1.11009462, 2.0, 3.60329646, 6.49187269, 11.69607095, 21.07220554, 37.96470182, 68.39903787, 123.23100555, 222.01892311, 400.]),
                'l_over_e': np.logspace(np.log10(0.05),np.log10(2500),30),
                #'dunkman_L5': np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) 
                #'dunkman_L5': np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]) 
                #'dunkman_L5': np.linspace(0.1, 0.7, 15),
                'dunkman_L5': np.linspace(0.2, 0.7, 15),
                'num_hit_doms': np.linspace(8, 50, 22),
                'CausalVetoHits': np.linspace(0, 9, 10),
                'CausalVetoPE': np.linspace(0, 9, 10)
                }
        if param_to_plot in x_bins.keys():
            return x_bins[param_to_plot]
        else:
            return 20

    def plot_variables(self, mc_arrays, muon_arrays, data_arrays, param_to_plot, fig_name, outdir, title, data_type='pseudo', logy=False, save_ratio=False, group_mc=True, group_mc_flavs=['nu_nc', 'numu_cc', 'nue_cc', 'nutau_cc'], extra=[], signal='CC+NC',thin=False,**kwargs):
        print "Plotting ", param_to_plot
        print "thin ", thin
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
        if thin:
            fig = plt.figure(figsize=(4.5, 5.0))
        else:
            fig=plt.figure()
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        if data_arrays is not None and self.ratio:
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax1.get_xaxis().set_tick_params(direction='in')
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
            #print "before cut len mc_weight_all ", len(mc_weight_all)
            nan_cut=np.logical_not(np.isnan(mc_param_all))
            mc_param_all = mc_param_all[nan_cut]
            mc_weight_all= mc_weight_all[nan_cut]
            #print "len mc_weight_all ", len(mc_weight_all)
            #print "np.sum(mc_weight_all) = ", np.sum(mc_weight_all)
            mc_sumw2_all = mc_sumw2_all[nan_cut]

            x_range = self.get_x_range(param_to_plot)
            #if param_to_plot=='l_over_e':
            #    print "     x_min ", min(mc_param_all)
            #    print "     x_max ", max(mc_param_all)
            if x_range=="":
                x_range = (min(mc_param_all), np.max(mc_param_all))
            nbins= self.get_x_nbins(param_to_plot)
            mc_y, x_edges = np.histogram(mc_param_all, weights=mc_weight_all, bins=nbins, range=x_range, **kwargs)
            mc_sumw2, x_edges = np.histogram(mc_param_all, weights=mc_sumw2_all, bins=x_edges, **kwargs)
        else:
            mc_y = np.zeros(nbins)
            mc_sumw2 = np.zeros(nbins)
            x_edges = None


        # add icc background
        if muon_arrays is not None:
            muon_param = muon_arrays[param_to_plot]
            #print "len muon_param ", len(muon_param)
            #print "len muon_arrays['weight']", len(muon_arrays['weight'])
            if x_edges is None:
                muon_y, x_edges = np.histogram(muon_param, weights=muon_arrays['weight'], bins=nbins, **kwargs)
                muon_sumw2, x_edges = np.histogram(muon_param, weights=muon_arrays['weight']**2, bins=nbins, **kwargs)
            else:
                muon_y, x_edges = np.histogram(muon_param, weights=muon_arrays['weight'], bins=x_edges, **kwargs)
                muon_sumw2, x_edges = np.histogram(muon_param, weights=muon_arrays['weight']**2, bins=x_edges, **kwargs)
            best_y = mc_y + muon_y
            best_sumw2 = mc_sumw2 + muon_sumw2
            muon_y_2, x_edges = np.histogram(muon_param, bins=x_edges, **kwargs)
        else:
            best_y = mc_y
            best_sumw2 = mc_sumw2
        #print "     in plotter best fit MC (no muon) ", np.sum(mc_y)
        #print "     in plotter best fit muon ", np.sum(muon_y * muon_arrays['weight'])
        #print "     in plotter best fit MC+muon: ", np.sum(best_y)
        if np.all(best_y==0):
            print "no mc or muon"
        else:
            if param_to_plot=='pid':
                special_x_edges = np.linspace(0,16,17)
                x = (special_x_edges[:-1]+special_x_edges[1:])/2.0
                print "use special_x_edges, x bin centers:", x
                ax1.hist(x, weights=best_y, bins=special_x_edges, histtype='step', lw=2.5, color=psublue, label=r'$\rm{Total}$', **kwargs)
            else:
                x = (x_edges[:-1]+x_edges[1:])/2.0
                ax1.hist(x, weights=best_y, bins=x_edges, histtype='step', lw=2.5, color=psublue, label=r'$\rm{Total}$', **kwargs)

        if extra!=[]:
            [mc_param_all_h0, mc_weight_all_h0, mc_sumw2_all_h0, muon_param_h0, icc_weight_h0, x_edges_h0] = extra
            mc_y_h0, _ = np.histogram(mc_param_all_h0, weights=mc_weight_all_h0, bins=x_edges, range=x_range, **kwargs)
            mc_sumw2_h0, _ = np.histogram(mc_param_all_h0, weights=mc_sumw2_all_h0, bins=x_edges, **kwargs)
            muon_y_h0, _ = np.histogram(muon_param_h0, bins=x_edges, **kwargs)
            best_y_h0 = mc_y_h0 + muon_y_h0 * icc_weight_h0
            best_sumw2_h0 = mc_sumw2_h0 + muon_y_h0 * (icc_weight_h0**2)

        if group_mc and mc_arrays is not None:
            group_mc_params = {}
            group_mc_weight = {}
            group_mc_sumw2 = {}
            #mc_group_y={}
            mc_group_y=[None]*len(group_mc_flavs)
            if group_mc_flavs==['nu_nc', 'numu_cc', 'nue_cc', 'nutau_cc']:
                colors = ['lightgreen', 'lightskyblue', 'burlywood', 'gainsboro', 'r']
            if group_mc_flavs==['nue_nc', 'nue_cc', 'numu_nc', 'numu_cc', 'nutau_nc', 'nutau_cc']:
                #colors = ['blueviolet', 'lavender', 'cornflowerblue', 'lightskyblue', 'darkorange', 'r', 'gainsboro']
                #colors = ['lime', 'limegreen', 'cornflowerblue', 'lightskyblue', 'darkorange', 'r', 'gainsboro']
                colors = ['green', 'lightgreen', 'deepskyblue', 'lightskyblue', 'gainsboro', 'darkorange', 'r']
            #if group_mc_flavs==['nue_nc', 'numu_nc', 'nue_cc', 'numu_cc', 'nutau_nc', 'nutau_cc']:
            #    colors = ['blueviolet', 'lightgreen', 'burlywood', 'lightskyblue', 'darkorange', 'r', 'gainsboro']
            if muon_arrays is None:
                colors.remove('gainsboro')
            for i, group in enumerate(group_mc_flavs):
                group_mc_params[group]=[]
                group_mc_weight[group]=[]
                group_mc_sumw2[group]=[]
                group_flav = group.split('_')[0]
                group_int = group.split('_')[1]
                for flav in mc_arrays.keys():
                    single_flav = flav.split('_')[0]
                    if 'bar' in single_flav:
                        single_flav=single_flav.rstrip('_bar')
                        # combine nu and nubar
                    single_int = flav.split('_')[1]
                    if (single_flav in group_flav or group_flav=='nu') and single_int==group_int:
                        group_mc_params[group].extend(mc_arrays[flav][param_to_plot])
                        group_mc_sumw2[group].extend(mc_arrays[flav]['sumw2'])
                        group_mc_weight[group].extend(mc_arrays[flav]['weight'])
            for i, group in enumerate(group_mc_flavs):
                if x_edges is None:
                    mc_group_y[i], x_edges = np.histogram(group_mc_params[group], bins=nbins, weights=group_mc_weight[group],**kwargs)
                else:
                    #mc_group_y[group], x_edges = np.histogram(group_mc_params[group], bins=x_edges, weights=group_mc_weight[group],**kwargs)
                    mc_group_y[i], x_edges = np.histogram(group_mc_params[group], bins=x_edges, weights=group_mc_weight[group],**kwargs)
                if param_to_plot=='pid':
                    special_x_edges = np.linspace(0,16,17)
                    x = (special_x_edges[:-1]+special_x_edges[1:])/2.0
                else:
                    x = (x_edges[:-1]+x_edges[1:])/2.0

                #ax1.hist(x, weights=mc_group_y[group], bins=x_edges, histtype='step', lw=1.5, color=colors[i], label=group, **kwargs)
                #plt.errorbar(x, mc_group_y[group], yerr=np.array(group_mc_sumw2[group]), fmt='.',color=colors[i])
                #plt.legend()
            # get flav labels
            all_flavs = deepcopy(group_mc_flavs)
            if muon_arrays is not None:
                if logy:
                    all_flavs.insert(0,'muon')
                else:
                    if group_mc_flavs==['nu_nc', 'numu_cc', 'nue_cc', 'nutau_cc']:
                        mc_group_y.insert(3,muon_y)
                        all_flavs.insert(3, 'muon')
                    if group_mc_flavs==['nue_nc', 'nue_cc', 'numu_nc', 'numu_cc', 'nutau_nc', 'nutau_cc']:
                        mc_group_y.insert(4,muon_y)
                        all_flavs.insert(4, 'muon')
            flavor_label=self.get_flavor_label(all_flavs)
            if param_to_plot=='pid':
                ax1.hist([x]*len(mc_group_y), weights=mc_group_y, bins=special_x_edges, lw=1.5, color=colors[0:len(mc_group_y)], label=flavor_label, histtype='bar', stacked=True, **kwargs)
            else:
                ax1.hist([x]*len(mc_group_y), weights=mc_group_y, bins=x_edges, lw=1.5, color=colors[0:len(mc_group_y)], label=flavor_label, histtype='bar', stacked=True, **kwargs)
            if extra!=[]:
                ax1.hist(x, weights=best_y_h0, bins=x_edges, histtype='step', lw=1.5, color='cyan', **kwargs)

        if mc_arrays is None and muon_arrays is not None:
            ax1.hist(x, weights=muon_y, bins=x_edges, histtype='step', lw=1.5, color='gainsboro', **kwargs)

        if data_arrays is not None:
            cut=np.logical_not(np.isnan(data_param))
            data_param = data_param[cut]
            data_weight = data_weight[cut]
            if x_edges is None:
                data_y, x_edges = np.histogram(data_param, weights=data_weight,bins=nbins)
            else:
                data_y, x_edges = np.histogram(data_param, weights=data_weight,bins=x_edges)
            if param_to_plot=='pid':
                print "data_arrays minimum value ", np.min(data_param)
                print "x_edges ", x_edges
                print "data_y", data_y
                print "best_y", best_y
                print "len data_y", len(data_y)
                print "len best_y", len(best_y)
                special_x_edges = np.linspace(0,16,17)
                x = (special_x_edges[:-1]+special_x_edges[1:])/2.0
            else:
                x = (x_edges[:-1]+x_edges[1:])/2.0
            # plot data as errorbar
            #plt.errorbar(x, data_y, yerr=np.sqrt(data_y), fmt='.', marker='.', markersize=1, color='k', label=r'$\rm{Data}$',capthick=1, capsize=3,elinewidth=1)
            plt.errorbar(x, data_y, yerr=np.sqrt(data_y), fmt='none', color='k', label=r'$\rm{Data}$',capthick=1, capsize=3,elinewidth=1)
            #print "     in plotter, data total ", np.sum(data_y)
            #print "     in plotter, best_y total ", np.sum(best_y)
            if thin:
                ax1.legend(loc=1,ncol=2,frameon=True, columnspacing=0.8, handlelength=1.5, prop={'size':9})
            else:
                ax1.legend(loc=1,ncol=3,frameon=True, columnspacing=0.8, handlelength=2, prop={'size':10})
            if self.ratio:
                assert(mc_arrays!=None or muon_arrays!=None)
                ax2=plt.subplot2grid((4,1), (3,0),sharex=ax1)
                fig.subplots_adjust(hspace=0.1)
                def tenToThree(x, pos):
                    return r'{:2g}\dot 10^3'.format(x/1000.)
                #formatter = FuncFormatter(tenToThree)
                #ax1.yaxis.set_major_formatter(formatter)
                #y_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
                #ax1.yaxis.set_major_formatter(y_formatter)

                #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                #formatter = mpl.ticker.ScalarFormatter(useMathText=True)
                #formatter.set_scientific(True)
                #formatter.set_powerlimits((0,2))
                #ax1.yaxis.set_major_formatter(formatter)

                ax2.axhline(y=1,linewidth=1, color=psublue)
                ratio_best_to_data = np.zeros(len(best_y))
                ratio_best_to_best = np.zeros(len(best_y))
                ratio_best_to_data_err=np.zeros(len(best_y))
                ratio_best_to_best_err=np.zeros(len(best_y))

                # plot ratio of best fit to best fit errorbar
                for i in range(0, len(data_y)):
                    if best_y[i]==0:
                        ratio_best_to_best_err[i]=1
                    else:
                        ratio_best_to_best_err[i]=math.sqrt(2*best_sumw2[i])/best_y[i]
                if param_to_plot=='pid':
                    bin_width=np.abs(special_x_edges[1:]-special_x_edges[:-1])
                else:
                    bin_width=np.abs(x_edges[1:]-x_edges[:-1])
                    x = (x_edges[:-1]+x_edges[1:])/2.0
                ax2.bar(x, 2*ratio_best_to_best_err,
                    bottom=np.ones(len(ratio_best_to_best_err))-ratio_best_to_best_err, width=bin_width,
                    linewidth=0, color=psulightblue, label=r'$\rm{Best\  Fit\  Uncert.}$', alpha=0.4
                    #linewidth=0, color=psulightblue, label=r'$\rm{\sigma_{best fit}}$', alpha=0.25
                )

                # plot ratio of data to best fit errorbar
                #cut_nonzero = data_y!=0
                #data_y = data_y[cut_nonzero]
                #best_y = best_y[cut_nonzero]
                #best_sumw2 = best_sumw2[cut_nonzero]
                #x_array = x[cut_nonzero]
                for i in range(0, len(data_y)):
                    if data_y[i]==0:
                        if best_y[i]==0:
                            ratio_best_to_data[i]=1
                            ratio_best_to_data_err[i]=1
                        else:
                            ratio_best_to_data[i]=0
                            ratio_best_to_data_err[i]=1
                    else:
                        # ratio best to data 
                        #ratio_best_to_data[i] = best_y[i]/data_y[i]
                        #ratio_best_to_data_err[i] = np.sqrt(best_sumw2[i]+(best_y[i]**2)/data_y[i])/data_y[i]

                        # ratio data to best
                        ratio_best_to_data[i] = data_y[i]/best_y[i]
                        ratio_best_to_data_err[i] = np.sqrt(data_y[i]+((data_y[i]**2)*best_sumw2[i]/(best_y[i]**2)))/best_y[i]

                #ax2.errorbar(x, ratio_best_to_data, yerr=ratio_best_to_data_err, fmt='.',marker='.',markersize=1, color='black',capthick=1,capsize=3, label=r'$\rm{Data}$', elinewidth=1)
                ax2.errorbar(x, ratio_best_to_data, yerr=ratio_best_to_data_err, fmt='none', color='black',capthick=1,capsize=3, label=r'$\rm{Data}$', elinewidth=1)
                #ax2.errorbar(x, ratio_best_to_data, yerr=ratio_best_to_data_err, fmt='.',marker='.',markersize=4, color='black',capthick=1,capsize=3, label=r'$\rm{\sigma_{data}}$')

                #ax2.legend(loc=2,ncol=2,frameon=False,prop={'size':10})

                # specify tick labels
                #ax2.get_xaxis().set_tick_params(direction='in')
                #ax2.xaxis.set_ticks_position('both')
                if param_to_plot=='pid':
                    #tick_x=[-3.0, -1.11009462, -0.34199519, -0.05848035, -0.01, 0.01, 0.05848035, 0.34199519, 0.61615503, 1.11009462, 2.0, 3.60329646, 6.49187269, 11.69607095, 21.07220554, 37.96470182, 68.39903787, 123.23100555, 222.01892311, 400.]
                    tick_x=special_x_edges
                    #tick_x_label = ('-3', '-1.1', '-0.3', '0.01', '0.3', '0.6', '1.1', '2.0', '3.6', '6.5', '11.7', '21', '38', '68', '123', '222', '400')
                    tick_x_label = ('', '-1.1', '', '0.0', '', '0.6', '', '2.0', '', '6.5', '', '21', '', '68', '', '222', '')
                    plt.xticks(tick_x)
                    plt.gca().set_xticklabels(tick_x_label)
                ax2.tick_params(axis='x', which='both', labelsize=14)
                ax1.tick_params(axis='y', which='both', labelsize=14)
                ax2.tick_params(axis='y', which='both', labelsize=14)
                if param_to_plot=='pid':
                    #ax1.plot((2,2),(0,4550),'k--', alpha=0.4)
                    #ax1.arrow(2,4500,1, 0, alpha=0.4)
                    #ax1.arrow(2,4500,-1,0, alpha=0.4)
                    #ax1.annotate('track', xy=(2, 4500), textcoords='data')
                    ax2_legend_pos_x = 0.1
                elif param_to_plot in ['dunkman_L5', 'num_hit_doms', 'CausalVetoPE', 'rt_fid_charge']:
                    ax2_legend_pos_x = 0.1
                else:
                    ax2_legend_pos_x = 0.5
                if thin:
                    ax2_legend_pos_x = 0.3
                ax2.legend(loc=(ax2_legend_pos_x, 0.68),ncol=2,frameon=True,prop={'size':9},columnspacing=0.5, handlelength=2)
                ax2.set_ylim([0.75,1.35])
                #ax2.set_ylim([0.8,1.2])
                #ax2.set_ylabel('best fit / data')
                #ax2.set_ylabel('data / best fit')
                ax2.set_ylabel(r'$\rm{Ratio \ to \ Best\ Fit}$',fontsize=14)
                ax2.set_xlabel(self.get_x_label(param_to_plot),fontsize=14)

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
                #a_text = ('chi2/dof = %.2f / %i, p = %.4f'%(chi2, dof, chi2_p))
                text_x = 0.98
                text_y= 0.98
                horizon_align = 'right'
                vertical_align='top'
                if 'QR3' in param_to_plot or 'QR6' in param_to_plot:
                    text_x = 0.02
                    text_y= 0.98
                    horizon_align = 'left'
                    vertical_align='top'
                #ax1.text(text_x, text_y, a_text,
                #        horizontalalignment=horizon_align,
                #        verticalalignment=vertical_align,
                #        transform=ax1.transAxes)

        if thin:
            fig.subplots_adjust(hspace=0.1, wspace=0.1, top=0.9, bottom=0.16, left=0.15, right=0.95)
        else:
            fig.subplots_adjust(left=0.15, right=0.95)

        ax1.set_ylabel(r'$\rm{Number \ of \ Events}$',fontsize=14)

        if data_arrays is not None:
            if thin:
                ymax = max(np.max(best_y),np.max(data_y))*1.7
            else:
                ymax = max(np.max(best_y),np.max(data_y))*1.5
        else:
            if thin:
                ymax = max(best_y)*1.7
            else:
                ymax = max(best_y)*1.5
        if logy:
            ax1.set_yscale("log")
            ax1.set_ylim([0.3,max(np.max(best_y),100000)])
        else:
            ax1.set_ylim([0, ymax])
        if param_to_plot=='pid':
            #plt.gca().set_xscale('symlog')
            #plt.gca().set_xscale('log')
            #ax1.set_yscale('log')
            ax1.set_ylim([0,max(np.max(best_y),np.max(data_y))*1.4])
            #ax1.set_ylim([0,max(np.max(best_y),np.max(data_y))*100])
        elif param_to_plot=='l_over_e':
            plt.gca().set_xscale('log')
            #ax1.set_ylim([0,4000])
        a_text = AnchoredText(r'$\nu_\tau \ \rm{Appearance \ (%s)}$'%signal, loc=2, frameon=False, prop={'size':12})
        b_text = AnchoredText(r'$\rm{IceCube \ Preliminary}$', loc=3, prop={'color':'r','alpha':0.4,'size':12}, frameon=False,bbox_to_anchor=(0, 0.75), bbox_transform=ax1.transAxes)
        ax1.add_artist(a_text)
        ax1.add_artist(b_text)
        #plt.grid()
        if title!='':
            ax1.set_title(r'$\rm{%s}$'%title, fontsize=14)
        if not os.path.isdir(outdir):
            fileio.mkdir(outdir)
        plot_name = outdir+'/'+file_name
        plt.savefig(plot_name+'.pdf')
        plt.savefig(plot_name+'.png')
        plt.clf()
        if muon_arrays is not None and mc_arrays is not None:
            return [mc_param_all, mc_weight_all, mc_sumw2_all, muon_param, muon_arrays['weight'], x_edges]
        if muon_arrays is None and mc_arrays is not None:
            return [mc_param_all, mc_weight_all, mc_sumw2_all, [], [], x_edges]
        if muon_arrays is not None and mc_arrays is None:
            return [[], [], [], muon_param, muon_arrays['weight'], x_edges]
       # if (data_arrays is not None) and self.ratio:
       #     plt.figure()
       #     plt.hist(x_array, weights=chi2_array, bins=x_edges, histtype='step', lw=1.5, color='r', **kwargs)
       #     if param_to_plot=='pid':
       #         plt.gca().set_xscale('symlog')
       #     if param_to_plot=='l_over_e':
       #         plt.gca().set_xscale('log')
       #     # labels
       #     if param_to_plot=='dunkman_L5':
       #         plt.xlabel('bdt_score')
       #     elif param_to_plot=='l_over_e':
       #         plt.xlabel('L/E')
       #     else:
       #         plt.xlabel(param_to_plot)
       #     plt.ylabel('chi2')
       #     plt.grid()
       #     plt.title(title)
       #     plt.savefig(plot_name+'_chi2_distribution.pdf')
       #     #plt.savefig(outdir+'/'+'chi2_distribution'+file_name+'.pdf')
       #     plt.clf()

       #     if save_ratio:
       #         ratio_dict = {}
       #         # remove zero ratios from array
       #         zero_idx = np.where(hist_ratio_best_to_data == 0)[0]
       #         ratio_dict['bin_vals'] = np.delete(hist_ratio_best_to_data, zero_idx)
       #         ratio_dict['bin_edges'] = np.delete(hist_bin_edges, zero_idx)
       #         return ratio_dict
