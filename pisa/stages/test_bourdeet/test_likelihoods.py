#!/usr/bin/env python

#
# Test script to compare the performances of
# the generalized poisson llh with the other 
# miminization metrics available in pisa
#
#


import numpy as np
import collections
import scipy.optimize as scp
import pickle
import sys,os

#
# pisa tools and objects
#
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.core.container import Container, ContainerSet
from pisa.core.distribution_maker import DistributionMaker

##################################################################################

if __name__=='__main__':

	import argparse

	parser = argparse.ArgumentParser('1D Toy Monte Carlo to test various likelihoods')

	parser.add_argument('-nd','--ndata',help='total number of data points',type=int,default=200)
	parser.add_argument('-sf','--signal-fraction',help='fraction of the data in the signal dataset',type=float,default=1.)
	parser.add_argument('-s','--stats-factor',help='Defines how much MC weights to produce w.r.t data',type=float,default=1.)
	parser.add_argument('-nt','--ntrials',help='number of pseudo experiments in the bias study',type=int,default=100)
	parser.add_argument('--make-llh-scan',help='if chosen, will run the likelihood scan for all llh',action='store_true')
	parser.add_argument('-o','--output',help='output stem files with plots',default = 'ToyMC_LLh.pdf')

	parser.add_argument('--interactive',help='use interactive plots',action='store_true')


	args = parser.parse_args()

	#
	# Plotting tools
	#
	import matplotlib as mpl 
	if not args.interactive:
		mpl.use('agg')
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_pdf import PdfPages
	import seaborn as sns 
	COLORS = sns.color_palette("hls", 8)
	output_pdf = PdfPages(args.output)



	#
	# Parameters of the data
	#
	n_data = 20000                      # Number of data points to bin
	signal_fraction = 0.8   # fraction of those points that will constitute the signal
	true_mu = 20.                             # True mean of the signal
	true_sigma = 3.1                          # True width of the signal
	nbackground_low = 0.					  # lowest value the background can take
	nbackground_high = 40.					  # highest value the background can take
	nbins = 101                                # number of bins in this likelihood fit

	
	binning = OneDimBinning(name='stuff', bin_edges=np.linspace(nbackground_low,nbackground_high,nbins))
	X = binning.midpoints


	#
	# Statistical factor for the MC
	#
	stats_factor = args.stats_factor


	#
	# Minimization options
	#
	Ntrials = args.ntrials



	#=============================================================
	#
	# Generate the Data sample
	#
	nsig = int(n_data*signal_fraction)
	nbkg = n_data-nsig

	signal = np.random.normal(loc=true_mu,scale=true_sigma,size=nsig)
	background = np.random.uniform(high=nbackground_high ,low=nbackground_low ,size=nbkg)
	total_data = np.concatenate([signal,background])
	counts_data,_ = np.histogram(total_data,bins=binning.bin_edges.magnitude)

	# Convert data histogram into a pisa map
	data_map = Map(name='total', binning=MultiDimBinning([binning]), hist=counts_data)
	data_as_mapset = MapSet([data_map])

	#==============================================================
	#
	# Plot the data
	#
	fig,ax = plt.subplots(figsize=(7,7))
	ax.errorbar(X,data_map.hist.flatten(),yerr=np.sqrt(data_map.hist.flatten()),fmt='-o',drawstyle='steps-mid',color='k')
	ax.set_xlabel('Some variable')
	ax.set_ylabel('Some counts')
	ax.set_title('Pseudo data fed into the likelihoods')
	ax.text(0.75,0.9,r'$\mu_{true}$ = '+'{}'.format(true_mu),fontsize=12,transform=ax.transAxes)
	ax.text(0.75,0.85,r'$\sigma_{true}$ = '+'{}'.format(true_sigma),fontsize=12,transform=ax.transAxes)
	ax.text(0.75,0.8,r'$N_{signal}$ = '+'{}'.format(nsig),fontsize=12,transform=ax.transAxes)
	ax.text(0.75,0.75,r'$N_{bkg}$ = '+'{}'.format(nbkg),fontsize=12,transform=ax.transAxes)
	if args.interactive:
		plt.show()
	output_pdf.savefig(fig)

	

	#===============================================================
	#
	# Generate MC sample using a pisa pipeline
	#
	MCtemplate = DistributionMaker(os.environ['PISA']+'/pisa/stages/test_bourdeet/super_simple/super_simple_pipeline.cfg')

	#
	# Change the default parameters if you so desire
	#
	#MCtemplate.params['stats_factor'].value = args.stats_factor
	#MCtemplate.params['n_events_data'].value = args.ndata
	MC_map = MCtemplate.get_outputs(return_sum=True)



	#===============================================================
	#
	# Plot Data + MC 
	#
	fig,ax = plt.subplots(figsize=(7,7))
	ax.errorbar(X,data_map.hist.flatten(),yerr=np.sqrt(data_map.hist.flatten()),fmt='o',color='k',label='data',drawstyle='steps-mid')
	ax.plot(X,MC_map[0].hist.flatten(),'-g',label='MC',drawstyle='steps-mid')
	ax.set_xlabel('Some variable')
	ax.set_ylabel('Some counts')
	ax.set_title('Same pseudo-data vs. MC')
	ax.text(0.65,0.9,r'$\mu_{MC}$ = '+'{}'.format(MCtemplate.params['mu'].value.m),color='g',fontsize=12,transform=ax.transAxes)
	ax.text(0.65,0.85,r'$\sigma_{MC}$ = '+'{}'.format(MCtemplate.params['sigma'].value.m),color='g',fontsize=12,transform=ax.transAxes)
	ax.text(0.65,0.8,'Stats factor = {}'.format(MCtemplate.params['stats_factor'].value.m),color='g',fontsize=12,transform=ax.transAxes)
	if args.interactive:
		plt.show()
	output_pdf.savefig(fig)




	if args.make_llh_scan:
		#================================================================
		#
		# Perform Likelihood scans fover a range of injected mu values
		#
		metrics_to_test = ['llh','mcllh_eff','mcllh_mean','generalized_poisson_llh']

		LLH_results= {}
		for name in metrics_to_test:
			LLH_results[name] = []


		tested_mus = []
		for tested_mu in np.linspace(19,21.,51):

			tested_mus.append(tested_mu)
			
			#
			# Recompute the truth MC
			#
			MCtemplate.params['mu'].value = tested_mu

		

			for metric in metrics_to_test :

				if metric=='generalized_poisson_llh':
					new_MC =  MCtemplate.get_outputs(return_sum=False, force_standard_output=False)[0]
					llhval = data_as_mapset.maps[0].metric_total(new_MC,metric=metric)
				else:
					new_MC = MCtemplate.get_outputs(return_sum=True)
					llhval	= data_as_mapset.metric_total(new_MC, metric=metric)


				LLH_results[metric].append(-llhval)


		#===============================================================
		#
		# Plot Likelihood scans
		#
		fig2,ax2 = plt.subplots(figsize=(9,9))
		n=0
		for llh_name in metrics_to_test:
			llhvals = LLH_results[llh_name]
			if llh_name=='mcllh_eff':
				ax2.plot(tested_mus,llhvals,'o',color=COLORS[n],label=llh_name)
			else:
				ax2.plot(tested_mus,llhvals,linewidth=2,color=COLORS[n],label=llh_name)

			n+=1

		ax2.set_xlabel(r'injected $\mu$')
		ax2.set_ylabel(r'-LLH / $\chi^{2}$')
		ax2.set_title('Likelihood scans over mu')
		ax2.legend()
		#ax2.set_ylim([0,100])
		ax2.set_xlim([18,22])
		if args.interactive:
			plt.show()

		output_pdf.savefig(fig2)

	output_pdf.close()
	print('PDF output produced: {}'.format(args.output))


	###################################################################################################
	####################################################################################################
	'''

	#
	# Produce a Fixed Truth MC sample 
	#===========================================================================
	# Create a MC set given the stats factor desired

	weight_dict_t        = generate_MC(n=n_data,signal_fraction=signal_fraction,mu=mu,sigma=sigma,stats_factor=1000,binning=binning)

	weight_dict_lowstats = generate_MC(n=n_data,signal_fraction=signal_fraction,mu=mu,sigma=sigma,stats_factor=stats_factor,binning=binning)

	#================================================================
	#
	#
	# Perform Toy experiments
	#
	# - Set the MC to a particular stats level
	# - Run Ntrials pseudo-experiments
	# - Minimize negative llh
	# - Compute TS = -2*(llh_minimized-llh_truth)
	#
	#
	# Note on minimizer: Nelder-Mead is super slow. Powell is 1/2 the time.
	# L-BFGS-B is the fastest

	pseudo_experiments = {}
	import time
	import cProfile, pstats, io


	# To compare the likelihood results with each other, we want the
	# generated MC to be the same for a given trial, for each evaluation.
	# 
	# therefore, for each likelihood we use to perform the minimization,
	# we re-initiate the seed to the same value for a given trial

	seed_list = np.arange(Ntrials)

	for llh_name,llh_obj in list(llh_sets.items()):
		print(('minimizing: ',llh_name))
		t0 = time.time()

		pseudo_experiments = []
		pr = cProfile.Profile()
		pr.enable() # Start profiling time usage

		for n,seed in zip(list(range(Ntrials)),seed_list):
			np.random.seed(seed)



			
			experiment_result = {}

			nsignal = int(n_data*signal_fraction)
			nbackground = n_data-nsignal

			signal = np.random.normal(loc=mu,scale=sigma,size=nsignal)
			background = np.random.uniform(high=nbackground_high,low=nbackground_low,size=nbackground)
			total_data = np.concatenate([signal,background])
			counts_data,_ = np.histogram(total_data,bins=binning)


			# Compute the truth llh value of this pseudo experiment
			# truth - if the truth comes from infinite stats MC
			experiment_result['truth_llh'] = llh_obj['fct'](data=counts_data,dataset_weights=weight_dict_t,**llh_obj['kwargs'])

			# truth if the truth comes from low stats MC
			experiment_result['lowstat_llh'] = llh_obj['fct'](data=counts_data,dataset_weights=weight_dict_lowstats,**llh_obj['kwargs'])


			# minimized llh (high stats) 
			#print '\t high statistics case...'
			Return_values = scp.minimize(fct_to_minimize,x0=mu,args=(sigma,counts_data,n_data,signal_fraction,None,1000.,llh_obj),
										 method='L-BFGS-B',
										 jac = False,
										 options={'maxiter':2000,
												  #'maxfun':2000,
												  #'approx_grad':True,
												  'ftol':0.01},
										 bounds = [(0.,None)])


			experiment_result['highstats_opt'] = Return_values


			# minimized llh (low stats)
			#print '\t low statistics case...'
			Return_values = scp.minimize(fct_to_minimize,x0=mu,args=(sigma,counts_data,n_data,signal_fraction,None,stats_factor,llh_obj),
										 method='L-BFGS-B',
										 jac = False,
										 options={'maxiter':2000,
												  #'maxfun':2000,
												  #'approx_grad':True,
												  'ftol':0.01},
										 bounds = [(0.,None)])
										 

			experiment_result['lowstats_opt'] = Return_values	
			pseudo_experiments.append(experiment_result)
			#print pseudo_experiments
		
		t1 = time.time()
		pr.disable()
		s = io.StringIO()
		sortby = 'tottime'
		ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
		ps.print_stats()
		print((s.getvalue()))
		print(("Time for ",Ntrials," minimizations: ",t1-t0," s"))
		print("Saving to file...")
		pickle.dump(pseudo_experiments,open(args.output+'_pseudo_exp_llh_%s.pckl'%llh_name,'wb'))

		print("Saved.")
	'''