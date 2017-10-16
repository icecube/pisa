from pisa.utils.plotter import Plotter
import numpy as np
import matplotlib as mpl
# Headless mode; must set prior to pyplot import
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.ticker
import scipy.stats as STATS
import os
import math
from copy import deepcopy

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

class PlotterNutau(Plotter):
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
        if param_to_plot=='dunkman_L5':
            return r'BDT score'
        elif param_to_plot=='l_over_e':
            #return r'$\mathrm{L/E [reco]}$'+' (km/GeV)'
            return r'$\rm{L/E \ [reco] \ (km/GeV)}$'
        elif param_to_plot=='reco_energy':
            return r'$\rm{E [reco] (GeV)}$'
        elif param_to_plot=='reco_coszen':
            return r'$\mathrm{\cos{\theta} [reco]}$'
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
                'num_hit_doms': (0, 90),
                'separation': (0, 600),
                'CausalVetoHits': (0, 9),
                'pid': (-3.60329646, 400),
                'linefit_speed': (0,0.5),
                'dunkman_L5': (0.2,0.7),
                'cog_q1_z': (-475, -174),
                'DCFiducialPE': (0,200),
                'rt_fid_charge': (0,200),
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
                'pid': np.array([-3.60329646, -2.0, -1.11009462, -0.61615503, -0.34199519, -0.18982351, -0.10536103, -0.05848035, -0.03245936, -0.01801648, -0.01, 0.01, 0.01801648, 0.03245936, 0.05848035, 0.10536103, 0.18982351, 0.34199519, 0.61615503, 1.11009462, 2.0, 3.60329646, 6.49187269, 11.69607095, 21.07220554, 37.96470182, 68.39903787, 123.23100555, 222.01892311, 400.]),
                'l_over_e': np.logspace(np.log10(0.05),np.log10(2500),30)
                }
        if param_to_plot in x_bins.keys():
            return x_bins[param_to_plot]
        else:
            return 20

    def plot_variables(self, mc_arrays, icc_arrays, data_arrays, param_to_plot, fig_name, outdir, title, data_type='pseudo', logy=False, save_ratio=False, group_mc=True, group_mc_flavs=['nu_nc', 'numu_cc', 'nue_cc', 'nutau_cc'], extra=[], signal='CC+NC',thin=False,**kwargs):
        print "thin ", thin
        print "Plotting ", param_to_plot
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
            print "before cut len mc_weight_all ", len(mc_weight_all)
            nan_cut=np.logical_not(np.isnan(mc_param_all))
            mc_param_all = mc_param_all[nan_cut]
            mc_weight_all= mc_weight_all[nan_cut]
            print "len mc_weight_all ", len(mc_weight_all)
            print "np.sum(mc_weight_all) = ", np.sum(mc_weight_all)
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
        if icc_arrays is not None:
            icc_param = icc_arrays[param_to_plot]
            if x_edges is None:
                icc_y, x_edges = np.histogram(icc_param, bins=nbins, **kwargs)
            else:
                icc_y, x_edges = np.histogram(icc_param, bins=x_edges, **kwargs)
            best_y = mc_y + icc_y * icc_arrays['weight']
            best_sumw2 = mc_sumw2 + icc_y * (icc_arrays['weight']**2)
        else:
            best_y = mc_y
            best_sumw2 = mc_sumw2
        print "     in plotter best fit MC (no icc) ", np.sum(mc_y)
        #print "     in plotter best fit icc ", np.sum(icc_y * icc_arrays['weight'])
        print "     in plotter best fit MC+icc: ", np.sum(best_y)
        if np.all(best_y==0):
            print "no mc or icc"
        else:
            x = (x_edges[:-1]+x_edges[1:])/2.0
            ax1.hist(x, weights=best_y, bins=x_edges, histtype='step', lw=2.5, color=psublue, label=r'$\rm{Total}$', **kwargs)

        if extra!=[]:
            [mc_param_all_h0, mc_weight_all_h0, mc_sumw2_all_h0, icc_param_h0, icc_weight_h0, x_edges_h0] = extra
            mc_y_h0, _ = np.histogram(mc_param_all_h0, weights=mc_weight_all_h0, bins=x_edges, range=x_range, **kwargs)
            mc_sumw2_h0, _ = np.histogram(mc_param_all_h0, weights=mc_sumw2_all_h0, bins=x_edges, **kwargs)
            icc_y_h0, _ = np.histogram(icc_param_h0, bins=x_edges, **kwargs)
            best_y_h0 = mc_y_h0 + icc_y_h0 * icc_weight_h0
            best_sumw2_h0 = mc_sumw2_h0 + icc_y_h0 * (icc_weight_h0**2)

        if group_mc:
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
            if icc_arrays is None:
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
                    mc_group_y[group], x_edges = np.histogram(group_mc_params[group], bins=nbins, weights=group_mc_weight[group],**kwargs)
                else:
                    #mc_group_y[group], x_edges = np.histogram(group_mc_params[group], bins=x_edges, weights=group_mc_weight[group],**kwargs)
                    mc_group_y[i], x_edges = np.histogram(group_mc_params[group], bins=x_edges, weights=group_mc_weight[group],**kwargs)
                x = (x_edges[:-1]+x_edges[1:])/2.0
                #ax1.hist(x, weights=mc_group_y[group], bins=x_edges, histtype='step', lw=1.5, color=colors[i], label=group, **kwargs)
                #plt.errorbar(x, mc_group_y[group], yerr=np.array(group_mc_sumw2[group]), fmt='.',color=colors[i])
                #plt.legend()
            # get flav labels
            all_flavs = deepcopy(group_mc_flavs)
            if icc_arrays is not None:
                if logy:
                    all_flavs.insert(0,icc_y*icc_arrays['weight'])
                else:
                    if group_mc_flavs==['nu_nc', 'numu_cc', 'nue_cc', 'nutau_cc']:
                        mc_group_y.insert(3,icc_y*icc_arrays['weight'])
                        all_flavs.insert(3, 'muon')
                    if group_mc_flavs==['nue_nc', 'nue_cc', 'numu_nc', 'numu_cc', 'nutau_nc', 'nutau_cc']:
                        mc_group_y.insert(4,icc_y*icc_arrays['weight'])
                        all_flavs.insert(4, 'muon')
            flavor_label=self.get_flavor_label(all_flavs)
            ax1.hist([x]*len(mc_group_y), weights=mc_group_y, bins=x_edges, lw=1.5, color=colors[0:len(mc_group_y)], label=flavor_label, histtype='bar', stacked=True, **kwargs)
            if extra!=[]:
                ax1.hist(x, weights=best_y_h0, bins=x_edges, histtype='step', lw=1.5, color='cyan', **kwargs)

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
            x = (x_edges[:-1]+x_edges[1:])/2.0
            plt.errorbar(x, data_y, yerr=np.sqrt(data_y), fmt='.', marker='.', markersize=4, color='k', label=r'$\rm{Data}$',capthick=1, capsize=3)
            print "     in plotter, data total ", np.sum(data_y)
            print "     in plotter, best_y total ", np.sum(best_y)
            if thin:
                ax1.legend(loc=1,ncol=2,frameon=True, columnspacing=0.8, handlelength=1.5, prop={'size':9})
            else:
                ax1.legend(loc=1,ncol=3,frameon=True, columnspacing=0.8, handlelength=2, prop={'size':10})
            if self.ratio:
                assert(mc_arrays!=None or icc_arrays!=None)
                ax2=plt.subplot2grid((4,1), (3,0),sharex=ax1)
                fig.subplots_adjust(hspace=0.1)
                ax2.get_xaxis().set_tick_params(direction='in')
                ax2.xaxis.set_ticks_position('both')
                if param_to_plot=='pid':
                    tick_x = [-3.60329646, -1.11009462, -0.34199519, -0.10536103, -0.03245936, -0.01, 0.01801648, 0.05848035, 0.18982351, 0.61615503, 2.0, 6.49187269, 21.07220554, 68.39903787, 222.01892311]
                    tick_x_label = ['-3.6', '-1.1', '-0.3', '-0.1', '-0.0', '-0.0', '0.0', '0.1', '0.2', '0.6', '2.0', '6.5', '21.1', '68.4', '222.0']
                    ax2.set_xticks(tick_x, tick_x_label)
                ax2.tick_params(axis='x', which='both', labelsize=12)
                ax1.tick_params(axis='y', which='both', labelsize=12)
                ax2.tick_params(axis='y', which='both', labelsize=12)
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
                bin_width=np.abs(x_edges[1:]-x_edges[:-1])
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

                ax2.errorbar(x, ratio_best_to_data, yerr=ratio_best_to_data_err, fmt='.',marker='.',markersize=4, color='black',capthick=1,capsize=3, label=r'$\rm{Data}$')
                #ax2.errorbar(x, ratio_best_to_data, yerr=ratio_best_to_data_err, fmt='.',marker='.',markersize=4, color='black',capthick=1,capsize=3, label=r'$\rm{\sigma_{data}}$')

                #ax2.legend(loc=2,ncol=2,frameon=False,prop={'size':10})
                if param_to_plot=='pid':
                    ax1.plot((2,2),(0,4550),'k--', alpha=0.4)
                    ax1.arrow(2,4500,1, 0, alpha=0.4)
                    ax1.arrow(2,4500,-1,0, alpha=0.4)
                    ax1.annotate('track', xy=(2, 4500), textcoords='data')
                    ax2_legend_pos_x = 0.4
                else:
                    ax2_legend_pos_x = 0.5
                if thin:
                    ax2_legend_pos_x = 0.3
                ax2.legend(loc=(ax2_legend_pos_x, 0.68),ncol=2,frameon=True,prop={'size':9},columnspacing=0.5, handlelength=2)
                ax2.set_ylim([0.75,1.35])
                #ax2.set_ylim([0.8,1.2])
                #ax2.set_ylabel('best fit / data')
                #ax2.set_ylabel('data / best fit')
                ax2.set_ylabel(r'$\rm{Ratio \ to \ Best\ Fit}$',fontsize=12)

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

        ax1.set_ylabel(r'$\rm{Number \ of \ Events}$',fontsize=12)

        if thin:
            ymax = max(np.max(best_y),np.max(data_y))*1.7
        else:
            ymax = max(np.max(best_y),np.max(data_y))*1.5
        if logy:
            ax1.set_yscale("log")
            ax1.set_ylim([0.3,max(np.max(best_y),100000)])
        else:
            ax1.set_ylim([0, ymax])
        if param_to_plot=='pid':
            plt.gca().set_xscale('symlog')
            #plt.gca().set_xscale('log')
            ax1.set_ylim([0,max(np.max(best_y),np.max(data_y))*1.6])
        elif param_to_plot=='l_over_e':
            plt.gca().set_xscale('log')
            #ax1.set_ylim([0,4000])
        ax2.set_xlabel(self.get_x_label(param_to_plot),fontsize=12)
        a_text = AnchoredText(r'$\nu_\tau \ \rm{Appearance \ (%s)}$'%signal, loc=2, frameon=False, prop={'size':12})
        b_text = AnchoredText(r'$\rm{IceCube \ Preliminary}$', loc=3, prop={'color':'r','alpha':0.4,'size':12}, frameon=False,bbox_to_anchor=(0, 0.75), bbox_transform=ax1.transAxes)
        ax1.add_artist(a_text)
        ax1.add_artist(b_text)
        #plt.grid()
        if title!='':
            ax1.set_title(title, fontsize=12)
        if not os.path.isdir(outdir):
            fileio.mkdir(outdir)
        plot_name = outdir+'/'+file_name
        plt.savefig(plot_name+'.pdf')
        plt.savefig(plot_name+'.png')
        plt.clf()
        if icc_arrays is not None:
            return [mc_param_all, mc_weight_all, mc_sumw2_all, icc_param, icc_arrays['weight'], x_edges]
        else:
            return [mc_param_all, mc_weight_all, mc_sumw2_all, [], [], x_edges]

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
