'''
Pisa stage that pre-computes some quantities
needed for the generalized likelihood. More 
specifically:

- based on the provided location of empty
  bins accross ALL monte carlo sets, populates
  empty bins of individual set according to the
  empty bin strategy #2 described in (ARXIV PAPER)

- For cases where the average bweight count in a 
  set is less than one, applies a mean correction 
  to prevent too big a bias when this is low MC

- Once this is done, computes the alpha and beta
  parameters that are fed into the likelihood

The stage appends / modifies the following:

    weights: changes the individual weight distribution
               based on the empty bin filling outcome

    llh_alphas: Map (alpha parameters of the generalized likelihood)

    llh_betas: Map (beta parameters of the generalized likelihood)

    n_mc_events: Map (number of MC events in each bin

    new_sum: Map (Sum of the weights in each bin (ie MC expectation),
             corrected for the empty bin filling and the mean 
             adjustment
    
author: Etienne Bourbeau (etienne.bourbeau@icecube.wisc.edu)
'''
from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.pi_stage import PiStage


# uncomment this to debug stuff
from pisa.utils.log import logging
#from pisa.utils.profiler import profile, line_profile


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
        #
        # A bunch of options we don't need
        #
        expected_params = ()
        input_names = ()
        output_names = ()


        # what are the keys used from the inputs during apply
        input_apply_keys = ('bin_indices',)
        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ('weights',)
        # what keys are added or altered for the outputs during apply
        output_apply_keys = ('weights', 'llh_alphas', 'llh_betas', 'n_mc_events', 'new_sum')

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

    def setup_function(self):
        """
        Declare empty containers and determine the number
        of MC events in each bin of each dataset
        """

        N_bins = self.output_specs.tot_num_bins

        self.data.data_specs = self.output_specs

        for container in self.data:


            #
            # Generate a new container called bin_indices
            #
            container['llh_alphas'] = np.empty((container.size), dtype=FTYPE)
            container['llh_betas'] = np.empty((container.size), dtype=FTYPE)
            container['n_mc_events'] = np.empty((container.size), dtype=FTYPE)
            container['new_sum'] = np.empty((container.size), dtype=FTYPE)


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
            # Find the maximum weight of an entire MC set
            max_weight  = np.amax(container['weights'].get('host'))
            container.add_scalar_data(key='pseudo_weight', data=max_weight)
            


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



            # hypersurface fit result, if hypersurfaces have been run
            if 'hs_scales' in container.binned_data:
                hypersurface = container.binned_data['hs_scales'][1].get('host')
            else:
                hypersurface = np.ones(N_bins)


            for index in range(N_bins):

                index_mask = container['bin_{}_mask'.format(index)].get('host')
                current_weights = container['weights'].get('host')[index_mask]*hypersurface[index]

                n_weights = current_weights.shape[0]

                # If no weights and other datasets have some, include a pseudo weight
                # Bins with no mc event in all set will be ignore in the likelihood later
                #
                # make the whole bin treatment here
                if n_weights <= 0:
                    pseudo_weight = container.scalar_data['pseudo_weight']
                    if pseudo_weight <0 :
                        logging.warn('WARNING: pseudo weight is less than zero, replacing it to 0,.')
                        pseudo_weight = 0.
                    current_weights = np.array([pseudo_weight])
                    n_weights = 1



                # write the new weight distribution down
                nevents_sim[index] = n_weights
                new_weight_sum[index] += np.sum(current_weights)

                # Mean of the current weight distribution
                mean_w = np.mean(current_weights)
                mean_of_weights[index] = mean_w

                # variance of the current weight
                var_of_weights[index] = ((current_weights-mean_w)**2).sum()/(float(n_weights))


            #  Calculate mean adjustment (TODO: save as a container scalar?)
            mean_number_of_mc_events = np.mean(nevents_sim)
            if mean_number_of_mc_events < 1.0:
                mean_adjustment = -(1.0-mean_number_of_mc_events) + 1.e-3 
            else:
                mean_adjustment = 0.0


            #  Variance of the poisson-gamma distributed variable
            var_z=(var_of_weights + mean_of_weights**2)

            if sum(var_z<0) != 0:
                logging.warn('warning: var_z is less than zero')
                logging.warn(container.name, var_z)
                raise Exception
            
            #  alphas and betas
            betas = mean_of_weights/var_z
            trad_alpha = (mean_of_weights**2)/var_z
            alphas = (nevents_sim + mean_adjustment)*trad_alpha


            # Calculate alphas and betas
            self.data.data_specs = self.output_specs

            np.copyto(src=alphas, dst=container['llh_alphas'].get('host'))
            np.copyto(src=betas, dst=container['llh_betas'].get('host'))
            np.copyto(src=new_weight_sum, dst=container['new_sum'].get('host'))


