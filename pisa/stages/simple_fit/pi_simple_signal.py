"""
Stage to generate simple 1D data consisting 
of a flat background + gaussian peak with a mean and a width

"""
from __future__ import absolute_import, print_function, division

__author__ = "Etienne Bourbeau (etienne.bourbeau@icecube.wisc.edu)"

import numpy as np

from pisa.core.container import Container
from pisa.core.pi_stage import PiStage

# Load the modified index lookup function
from pisa.core.bin_indexing import lookup_indices


class pi_simple_signal(PiStage):
    """
    random toy event generator PISA Pi class

    Parameters
    ----------
    data
    params
        Expected params .. ::

            n_events : int
                Number of events to be generated per output name
            random
            seed : int
                Seed to be used for random

    input_names
    output_names
    debug_mode
    input_specs
    calc_specs
    output_specs

    """

    def __init__(
        self,
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        input_specs=None,
        calc_specs=None,
        output_specs=None,
    ):
        expected_params = (  # parameters fixed during fit
            'n_events_data',
            'stats_factor',
            'signal_fraction',

            # minimum + maximum bkg values
            'bkg_min',
            'bkg_max',

            # fitted parameters
            'mu',
            'sigma')

        # what keys are added or altered for the outputs during apply
        output_apply_keys = ('weights','errors')

        # init base class
        super().__init__(
            data=data,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            input_specs=input_specs,
            calc_specs=calc_specs,
            output_specs=output_specs,
            output_apply_keys=output_apply_keys
        )

        # doesn't calculate anything
        assert self.calc_mode is None

    def setup_function(self):
        '''
        This is where we figure out how many events to generate,
        define their weights relative to the data statistics
        and initialize the container we will need

        This function is run once when we instantiate the pipeline
        '''

        #
        # figure out how many signal and background events to create
        #
        n_data_events = int(self.params.n_events_data.value.m)
        stats_factor = float(self.params.stats_factor.value.m)
        signal_fraction = float(self.params.signal_fraction.value.m)

        # Number of simulated MC events
        self.n_mc = int(n_data_events*stats_factor)
        # Number of signal MC events
        self.nsig = int(self.n_mc*signal_fraction)
        self.nbkg = self.n_mc-self.nsig                     # Number of bkg MC events

        # Go in events mode
        self.data.data_specs = 'events'

        #
        # Create a signal container, with equal weights
        #
        signal_container = Container('signal')
        signal_container.data_specs = 'events'
        # Populate the signal physics quantity
        signal_container.add_array_data('stuff', np.zeros(self.nsig))
        # Populate its MC weight
        signal_container.add_array_data('weights', np.ones(self.nsig)*1./stats_factor)
        # Populate the error on those weights
        signal_container.add_array_data('errors',(np.ones(self.nsig)*1./stats_factor)**2. )
        # Add empty bin_indices array (used in generalized poisson llh)
        signal_container.add_array_data('bin_indices', np.ones(self.nsig)*-1)
        # Add bin indices mask (used in generalized poisson llh)
        for bin_i in range(self.output_specs.tot_num_bins):
            signal_container.add_array_data(key='bin_{}_mask'.format(
                bin_i), data=np.zeros(self.nsig, dtype=bool))
        # Add container to the data
        self.data.add_container(signal_container)

        #
        # Create a background container
        #
        if self.nbkg > 0:

            bkg_container = Container('background')
            bkg_container.data_specs = 'events'
            bkg_container.add_array_data('stuff', np.zeros(self.nbkg))
            bkg_container.add_array_data('weights', np.ones(self.nbkg)*1./stats_factor)
            bkg_container.add_array_data('errors',(np.ones(self.nbkg)*1./stats_factor)**2. )
            bkg_container.add_array_data('bin_indices', np.ones(self.nbkg)*-1)
            # Add bin indices mask (used in generalized poisson llh)
            for bin_i in range(self.output_specs.tot_num_bins):
                bkg_container.add_array_data(key='bin_{}_mask'.format(
                    bin_i), data=np.zeros(self.nbkg, dtype=bool))

            self.data.add_container(bkg_container)

        #
        # Bin the weights according to the output specs binning
        # Provide a binning if non is specified
        # if self.output_specs is None:
        #    self.output_specs = MultiDimBinning([OneDimBinning(name='stuff', bin_edges=np.linspace(0.,40.,21))])

        for container in self.data:
            container.array_to_binned('weights', binning=self.output_specs, averaged=False)
            container.array_to_binned('errors', binning=self.output_specs, averaged=False)


    def apply_function(self):
        '''
        This is where we actually inject a gaussian signal and a
        flat background according to the parameters

        This function will be called at every iteration of the minimizer
        '''

        #
        # Make sure we are in events mode
        #
        self.data.data_specs = 'events'

        for container in self.data:

            if container.name == 'signal':
                #
                # First, generate the signal
                #
                signal = np.random.normal(
                    loc=self.params['mu'].value.m, scale=self.params['sigma'].value.m, size=self.nsig)
                container['stuff'] = signal

            elif container.name == 'background':
                #
                # Then the background
                #
                background = np.random.uniform(low=self.params.bkg_min.value.m,
                                               high=self.params.bkg_max.value.m,
                                               size=self.nbkg)

                container['stuff'] = background

            #
            # Recompute the bin indices associated with each event
            #
            new_array = lookup_indices(
                sample=[container['stuff']], binning=self.output_specs)
            new_array = new_array.get('host')
            container["bin_indices"] = new_array

            for bin_i in range(self.output_specs.tot_num_bins):
                container['bin_{}_mask'.format(bin_i)] = new_array == bin_i

        #
        # Re-bin the data
        #
        for container in self.data:
            container.array_to_binned(
                'weights', binning=self.output_specs, averaged=False)
            container.array_to_binned(
                'errors', binning=self.output_specs, averaged=False)


            #
            #  Recalculate the number of MC events per bin, if the array already exists
            #
            if "n_mc_events" in container.binned_data.keys():

                self.data.data_specs = 'events'
                nevents_sim = np.zeros(self.output_specs.tot_num_bins)

                for index in range(self.output_specs.tot_num_bins):
                    index_mask = container['bin_{}_mask'.format(index)].get('host')
                    current_weights = container['weights'].get('host')[index_mask]
                    n_weights = current_weights.shape[0]

                    # Number of MC events in each bin
                    nevents_sim[index] = n_weights

                self.data.data_specs = self.output_specs
                np.copyto(src=nevents_sim,
                          dst=container["n_mc_events"].get('host'))

                #
                # Step 2: Re-calculate the mean adjustment for each container
                #
                mean_number_of_mc_events = np.mean(nevents_sim)
                if mean_number_of_mc_events < 1.0:
                    mean_adjustment = -(1.0-mean_number_of_mc_events) + 1.e-3
                else:
                    mean_adjustment = 0.0
                container.scalar_data['mean_adjustment']=mean_adjustment