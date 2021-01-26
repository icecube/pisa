'''
Pisa stage that pre-computes some quantities
needed for the generalized likelihood, and applies
small adjustments to the weight distributions in cases
where the number of mc event per bin is low.

The code does the following, in order:

- Calculate the number of MC events per bin once,
  at the setup time

- Calculate, at setup time, a mean adjustment, based
  on the average number of MC events per bin. If the
  latter is less than one, adjustment is applied, else
  that quantity is equal to zero

- Populate ANY empty mc bin with a pseudo-weight with a
  value equal to the maximal weight value of a given 
  dataset. This correspond to the empty bin strategy #2 
  described in (1902.08831). Note that empty bin strategy #1
  can still be applied later on, if one provides the bin
  indices where no datasets have any MC events. This step 
  runs in the apply function because the value of the pseudo
  weight will change during minimization.


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
'''
from __future__ import absolute_import, print_function, division

__author__ = "Etienne Bourbeau (etienne.bourbeau@icecube.wisc.edu)"

import numpy as np
import copy

from pisa import FTYPE
from pisa.core.stage import Stage


# uncomment this to debug stuff
from pisa.utils.log import logging
from pisa.utils.profiler import profile, line_profile
from pisa.utils.log import set_verbosity, Levels
#set_verbosity(Levels.DEBUG)

PSEUDO_WEIGHT = 0.001


class generalized_llh_params(Stage):
    """
    Pisa stage that applies mean adjustment and
    empty bin filling. Also computes alphas and betas
    that are needed by the generalized poisson likelihood
 
    """
 
    # this is the constructor with default arguments
    def __init__(self,
                 **std_kwargs,
                 ):

        # init base class
        super(generalized_llh_params, self).__init__(expected_params=(),
                                                     **std_kwargs,
                                                     )
 
    def setup_function(self):
        """
        Declare empty containers, determine the number
        of MC events in each bin of each dataset and
        compute mean adjustment
        """
 
        N_bins = self.apply_mode.tot_num_bins
 
        self.data.representation = self.apply_mode
 
        for container in self.data:
 
            #
            # Generate a new container called bin_indices
            #
            container['llh_alphas'] = np.empty((container.size), dtype=FTYPE)
            container['llh_betas'] = np.empty((container.size), dtype=FTYPE)
            container['n_mc_events'] = np.empty((container.size), dtype=FTYPE)
            container['old_sum'] = np.empty((container.size), dtype=FTYPE)
 
            #
            # Step 1: assert the number of MC events in each bin,
            #         for each container
            self.data.representation = 'events'
            nevents_sim = np.zeros(N_bins)
 
            for index in range(N_bins):
                index_mask = container['bin_{}_mask'.format(index)]
                if 'kfold_mask' in container:
                    index_mask*=container['kfold_mask']
                # Number of MC events in each bin
                nevents_sim[index] = np.sum(index_mask)
 
            self.data.representation = self.apply_mode
            np.copyto(src=nevents_sim,
                      dst=container["n_mc_events"])
            container.mark_changed('n_mc_events')
 
            #
            # Step 2: Calculate the mean adjustment for each container
            #
            mean_number_of_mc_events = np.mean(nevents_sim)
            if mean_number_of_mc_events < 1.0:
                mean_adjustment = -(1.0-mean_number_of_mc_events) + 1.e-3
            else:
                mean_adjustment = 0.0
            container.set_aux_data(key='mean_adjustment', data=mean_adjustment)
 
 
            #
            # Add hypersurface containers if they don't exist
            # (to avoid errors in get_outputs, if we want )
            # these to be returned when you call get_output
            #
            if 'hs_scales' not in container.keys():
                container['hs_scales'] =  np.empty((container.size), dtype=FTYPE)
                container['errors'] = np.empty((container.size), dtype=FTYPE)
 
 
    def apply_function(self):
        '''
        Computes the main inputs to the generalized likelihood 
        function on every iteration of the minimizer

        '''
        N_bins = self.apply_mode.tot_num_bins

        #
        # Step 4: Apply the empty bin strategy and mean adjustment
        #    Compute the alphas and betas that go into the
        #    poisson-gamma mixture of the llh
        #
        for container in self.data:

            self.data.representation = 'events'

            #
            # Step 3: Find the maximum weight accross all events
            #         of each MC set. The value of that weight defines
            #         the value of the pseudo-weight that will be included
            #         in empty bins

            # for this part we are in events mode
            # Find the minimum weight of an entire MC set
            pseudo_weight = 0.001
            container.set_aux_data(key='pseudo_weight', data=pseudo_weight)

            old_weight_sum = np.zeros(N_bins)
            new_weight_sum = np.zeros(N_bins)
            alphas_vector = np.zeros(N_bins)
            betas_vector = np.zeros(N_bins)

            #
            # Load the pseudo_weight and mean displacement values
            #
            mean_adjustment = container.scalar_data['mean_adjustment']
            pseudo_weight = container.scalar_data['pseudo_weight']

            for index in range(N_bins):

                index_mask = container['bin_{}_mask'.format(index)]
                if 'kfold_mask' in container:
                    index_mask*=container['kfold_mask']
                current_weights = container['weights'][index_mask]

                old_weight_sum[index] += np.sum(current_weights)

                assert np.all(current_weights>=0),'SOME WEIGHTS BELOW ZERO'
                n_weights = current_weights.shape[0]

                # If no weights and other datasets have some, include a pseudo weight
                # Bins with no mc event in all set will be ignore in the likelihood later
                #
                # make the whole bin treatment here
                if n_weights <= 0:
                    current_weights = np.array([pseudo_weight])
                    n_weights = 1

                # write the new weight distribution down
                new_weight_sum[index] += np.sum(current_weights)

                # Mean of the current weight distribution
                mean_w = np.mean(current_weights)

                # variance of the current weight
                var_of_weights = ((current_weights-mean_w)**2).sum()/(float(n_weights))

                #  Variance of the poisson-gamma distributed variable
                var_z = (var_of_weights + mean_w**2)

                if var_z < 0:
                    logging.warn('warning: var_z is less than zero')
                    logging.warn(container.name, var_z)
                    raise Exception

                # if the weights presents have a mean of zero, 
                # default to alphas values of PSEUDO_WEIGHT and
                # of beta = 1.0, which mimicks a narrow PDF
                # close to 0.0 
                beta = np.divide(mean_w, var_z, out=np.ones(1), where=var_z!=0)
                trad_alpha = np.divide(mean_w**2, var_z, out=np.ones(1)*PSEUDO_WEIGHT, where=var_z!=0)
                alpha = (n_weights + mean_adjustment)*trad_alpha

                alphas_vector[index] = alpha
                betas_vector[index] = beta

            # Calculate alphas and betas
            self.data.representation = self.apply_mode
            np.copyto(src=alphas_vector, dst=container['llh_alphas'])
            np.copyto(src=betas_vector, dst=container['llh_betas'])
            np.copyto(src=new_weight_sum, dst=container['weights'])
            np.copyto(src=old_weight_sum, dst=container['old_sum'])
            container.mark_changed('llh_alphas')
            container.mark_changed('llh_betas')
            container.mark_changed('old_sum')
            container.mark_changed('weights')



