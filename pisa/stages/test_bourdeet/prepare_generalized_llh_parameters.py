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
        output_apply_keys = ('weights','llh_alphas','llh_betas','n_mc_events','empty_bins','new_sum')

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

        self.data.data_specs = self.output_specs

        for container in self.data:
            # Generate a new container called bin_indices
            container['llh_alphas'] = np.empty((container.size), dtype=FTYPE)
            container['llh_betas']  = np.empty((container.size), dtype=FTYPE)
            container['n_mc_events']= np.empty((container.size), dtype=FTYPE)
            container['new_sum']    = np.empty((container.size), dtype=FTYPE)
            container['empty_bins'] = np.empty((container.size), dtype=np.int64)



    def apply_function(self):
        '''
        stuff

        '''

        N_bins = self.output_specs.tot_num_bins

        # Step 1: assert the number of MC events in each bin,
        #         the maximum weight of the entire container, and whether 
        #         there are emtpy bins 

        max_weight = OrderedDict()
        for container in self.data:

            # for this part we are in events mode
            self.data.data_specs = 'events'

            nevents_sim = np.zeros(N_bins)
            empty_bins  = np.zeros(N_bins,dtype=np.int64)

            max_weight[container.name] = max(container['weights']) # TODO: save as a scalar quantity?

            for index in range(N_bins):
                index_mask = container['bin_indices'].get('host')==index
                current_weights = container['weights'].get('host')[index_mask]
                n_weights = current_weights.shape[0]

                # Number of MC events in each bin
                nevents_sim[index] = n_weights
                empty_bins[index] = 1 if n_weights<=0 else 0
            

            # For this part we are back in bin mode
            self.data.data_specs = self.output_specs
            np.copyto(src=nevents_sim, dst=container["n_mc_events"].get('host'))
            np.copyto(src=empty_bins, dst=container['empty_bins'].get('host'))




        #
        #  2. Check where there are bins where we need to provide a pseudo MC event count
        #     This should indicate the bin numbers where at least
        #     one container is non-empty
        #
        all_empty_bins = [c['empty_bins'].get('host') for c in self.data]
        bins_we_need_to_fix = np.ones(N_bins,dtype=np.int64)
        for v in all_empty_bins:
            bins_we_need_to_fix*=v.astype(np.int64)

        bin_indices_we_need = np.where(bins_we_need_to_fix==0)[0] #TODO: this must be a trans-pipeline object
        bin_indices_we_need = range(N_bins)
            

        #
        # 3. Apply the empty bin strategy and mean adjustment
        #    Compute the alphas and betas that go into the 
        #    poisson-gamma mixture of the llh
        #
        for container in self.data:

            self.data.data_specs = 'events'
            new_weight_sum = np.zeros(N_bins)
            mean_of_weights= np.zeros(N_bins)
            var_of_weights = np.zeros(N_bins)
            nevents_sim = np.zeros(N_bins)


            for index in range(N_bins):

                index_mask = container['bin_indices'].get('host')==index
                current_weights = container['weights'].get('host')[index_mask]

                # If no weights and other datasets have some, include a pseudo weight
                if current_weights.shape[0]<=0:
                    if index in bin_indices_we_need:
                        current_weights = np.array([max_weight[container.name]]) #TODO: make trans-pipeline aware
                    else: 
                        logging.trace('WOOOO! Empty bin common to all sets: {}'.format(index))
                        current_weights = np.array([0.0])

                # New number
                n_weights = current_weights.shape[0]
                nevents_sim[index] = n_weights
                new_weight_sum[index]+=sum(current_weights)

                # Mean of the current weight distribution
                mean_w = np.mean(current_weights)
                mean_of_weights[index] = mean_w

                # variance of the current weight
                var_of_weights[index]=((current_weights-mean_w)**2).sum()/(float(n_weights))


            #  Calculate mean adjustment (TODO: save as a container scalar?)
            mean_number_of_mc_events = np.mean(nevents_sim)
            mean_adjustment = -(1.0-mean_number_of_mc_events) + 1.e-3 if mean_number_of_mc_events<1.0 else 0.0


            #  Variance of the poisson-gamma distributed variable
            var_z=(var_of_weights+mean_of_weights**2)
            
            #  alphas and betas
            betas = mean_of_weights/var_z
            trad_alpha=(mean_of_weights**2)/var_z

            alphas = (nevents_sim+mean_adjustment)*trad_alpha

            # Calculate alphas and betas
            self.data.data_specs = self.output_specs

            np.copyto(src=alphas, dst=container['llh_alphas'].get('host'))
            np.copyto(src=betas, dst=container['llh_betas'].get('host'))
            np.copyto(src=new_weight_sum, dst=container['new_sum'].get('host'))


