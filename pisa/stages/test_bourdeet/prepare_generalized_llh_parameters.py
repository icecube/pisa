#/usr/bin/env python


#
#
# Stuff stuff stuff
#
#


from __future__ import absolute_import, print_function, division

import math

from numba import guvectorize
import numpy as np

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging

from pisa.utils.profiler import profile, line_profile
# Load the modified index lookup function
from pisa.core.bin_indexing import lookup_indices
from pisa.core.binning import MultiDimBinning

from collections import OrderedDict


class prepare_generalized_llh_parameters(PiStage):
    """
    PISA Pi stage to append an array 

    Parameters
    ----------
    data
    params
        foo : Quantity
        bar : Quanitiy with time dimension
    input_names
    output_names
    debug_mode
    input_specs
    calc_specs
    output_specs

    """

    # this is the constructor with default arguments
    
    def __init__(self,
                 data=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                ):

        # here we register our expected parameters foo and bar so that PISA knows what to expect
        expected_params = ()

        # any in-/output names could be specified here, but we won't need that for now
        input_names = ()
        output_names = ()

        # register any variables that are used as inputs or new variables generated
        # (this may seem a bit abstract right now, but hopefully will become more clear later)

        # what are the keys used from the inputs during apply
        input_apply_keys = ('bin_indices',)
        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ('weights',)
        # what keys are added or altered for the outputs during apply
        output_apply_keys = ('weights','llh_alphas','llh_betas','n_mc_events','new_sum')

        # init base class
        super(prepare_generalized_llh_parameters, self).__init__(data=data,
                                       params=params,
                                       expected_params=expected_params,
                                       input_names=input_names,
                                       output_names=output_names,
                                       debug_mode=debug_mode,
                                       input_specs=input_specs,
                                       calc_specs=calc_specs,
                                       output_specs=output_specs,
                                       input_apply_keys=input_apply_keys,
                                       output_apply_keys=output_apply_keys,
                                       output_calc_keys=output_calc_keys,
                                       )

        # make sure the user specified some modes
        #assert self.input_mode is not None
        #assert self.calc_mode is not None
        #assert self.output_mode is not None

    def setup_function(self):
        """Setup the stage"""

        N_bins = self.output_specs.tot_num_bins

        self.data.data_specs = self.output_specs

        for container in self.data:


            #
            # Generate a new container called bin_indices
            #
            container['llh_alphas'] = np.empty((container.size), dtype=FTYPE)
            container['llh_betas']  = np.empty((container.size), dtype=FTYPE)
            container['n_mc_events']= np.empty((container.size), dtype=FTYPE)
            container['new_sum']    = np.empty((container.size), dtype=FTYPE)




            #
            # Step 1: assert the number of MC events in each bin,
            #         for each container
            self.data.data_specs = 'events'
            nevents_sim = np.zeros(N_bins)
            
            for index in range(N_bins):
                index_mask = container['bin_{}_mask'.format(index)].get('host')
                current_weights = container['weights'].get('host')[index_mask]
                n_weights = current_weights.shape[0]

                # Number of MC events in each bin
                nevents_sim[index] = n_weights
            
            self.data.data_specs = self.output_specs
            np.copyto(src=nevents_sim, dst=container["n_mc_events"].get('host'))


    #@line_profile
    def apply_function(self):
        '''
        Computes the main inputs to the generalized likelihood 
        function on every iteration of the minimizer

        '''
        N_bins = self.output_specs.tot_num_bins



        #
        # Step 2: Find the maximum weight accross all events 
        #         of each MC set. The value of that weight defines
        #         the value of the pseudo-weight that will be included
        #         in empty bins
        
        # for this part we are in events mode
        for container in self.data:


            self.data.data_specs = 'events'

            nevents_sim = np.zeros(N_bins)

            # Find the maximum weight of an entire MC set
            max_weight  = np.amax(container['weights'].get('host'))
            container.add_scalar_data(key='pseudo_weight',data=max_weight)
            


        #
        # 3. Apply the empty bin strategy and mean adjustment
        #    Compute the alphas and betas that go into the 
        #    poisson-gamma mixture of the llh
        #
        self.data.data_specs = self.output_specs

        for container in self.data:

            self.data.data_specs = 'events'
            new_weight_sum = np.zeros(N_bins)
            mean_of_weights= np.zeros(N_bins)
            var_of_weights = np.zeros(N_bins)
            nevents_sim = np.zeros(N_bins)
            '''
            if np.sum(container.array_data['weights'].get('host')<0.)!=0:
                print('\nERROR: array weights are negative!\n')
                print(container.array_data['weights'].get('host'))
                print(np.sum(container.array_data['weights'].get('host')<0.),' out of ',container.array_data['weights'].get('host').shape[0])
                print('\n\n')


            if np.sum(container.binned_data['weights'][1].get('host')<0.)!=0:
                print('\nERROR: binned weights are negative!\n')
                print(container.binned_data['weights'][1].get('host'))
                print('\n\n')
            '''

            # hypersurface fit result
            hypersurface = container.binned_data['hs_scales'][1].get('host')

            for index in range(N_bins):

                index_mask = container['bin_{}_mask'.format(index)].get('host')
                current_weights = container['weights'].get('host')[index_mask]*hypersurface[index]



                # Floor weights to zero
                current_weights = np.clip(current_weights,0,None)
                n_weights = current_weights.shape[0]

                # If no weights and other datasets have some, include a pseudo weight
                # Bins with no mc event in all set will be ignore in the likelihood later
                #
                # make the whole bin treatment here
                if n_weights<=0 or np.sum(current_weights)<=0:
                    pseudo_weight = container.scalar_data['pseudo_weight']

                    if pseudo_weight<=0:
                        print('WARNING: pseudo weight is equal to zero, replacing it to 1,.')
                        for p in self.params:
                            print(p.name,p.value)
                        pseudo_weight = 1.
                    current_weights = np.array([pseudo_weight])



                # write the new weight distribution down
                nevents_sim[index] = n_weights
                new_weight_sum[index]+=np.sum(current_weights)

                # Mean of the current weight distribution
                mean_w = np.mean(current_weights)
                mean_of_weights[index] = mean_w

                # variance of the current weight
                var_of_weights[index]=((current_weights-mean_w)**2).sum()/(float(n_weights))#*hypersurface[index]


            #  Calculate mean adjustment (TODO: save as a container scalar?)
            mean_number_of_mc_events = np.mean(nevents_sim)
            mean_adjustment = -(1.0-mean_number_of_mc_events) + 1.e-3 if mean_number_of_mc_events<1.0 else 0.0


            #  Variance of the poisson-gamma distributed variable
            var_z=(var_of_weights+mean_of_weights**2)

            if sum(var_z<=0)!=0:
                print('warning: var_z is equal to zero')
                print(container.name,var_z)
            
            #  alphas and betas
            betas = mean_of_weights/var_z
            trad_alpha=(mean_of_weights**2)/var_z
            if np.sum(trad_alpha<=0)!=0:
                print('ERROR: alpha should be exclusively positive')
                print(trad_alpha)
                raise Exception

            alphas = (nevents_sim+mean_adjustment)*trad_alpha

            if sum((alphas)<0)!=0:
                print('ARRRRGGGGGHHHH')
                print(nevents_sim,mean_adjustment)
                raise Exception

            # Calculate alphas and betas
            self.data.data_specs = self.output_specs

            np.copyto(src=alphas, dst=container['llh_alphas'].get('host'))
            np.copyto(src=betas, dst=container['llh_betas'].get('host'))
            np.copyto(src=new_weight_sum, dst=container['new_sum'].get('host'))


