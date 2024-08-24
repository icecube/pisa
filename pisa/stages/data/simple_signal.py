"""
Stage to generate simple 1D data consisting
of a flat background + gaussian peak with a mean and a width

"""
from __future__ import absolute_import, print_function, division

__author__ = "Etienne Bourbeau (etienne.bourbeau@icecube.wisc.edu)"

import numpy as np
from scipy.stats import norm

from pisa import FTYPE
# Load the modified index lookup function
from pisa.core.bin_indexing import lookup_indices
from pisa.core.binning import MultiDimBinning
from pisa.core.container import Container
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage

__all__ = ['simple_signal', 'init_test']


class simple_signal(Stage):  # pylint: disable=invalid-name
    """
    random toy event generator PISA class

    Parameters
    ----------
    params
        Expected params .. ::

            n_events : int
                Number of events to be generated per output name
            random
            seed : int
                Seed to be used for random

    """

    def __init__(
        self,
        **std_kwargs,
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
            'sigma'
        )

        supported_reps = {
            'apply_mode': [MultiDimBinning]
        }

        # init base class
        super().__init__(
            expected_params=expected_params,
            expected_container_keys=(),
            supported_reps=supported_reps,
            **std_kwargs,
        )

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
        self.stats_factor = float(self.params.stats_factor.value.m)
        signal_fraction = float(self.params.signal_fraction.value.m)

        # Number of simulated MC events
        self.n_mc = int(n_data_events*self.stats_factor)
        # Number of signal MC events
        self.nsig = int(self.n_mc*signal_fraction)
        self.nbkg = self.n_mc-self.nsig                     # Number of bkg MC events

        # Go in events mode
        self.data.representation = 'events'

        #
        # Create a signal container, with equal weights
        #
        signal_container = Container('signal')
        signal_container.representation = 'events'
        # Populate the signal physics quantity over a uniform range
        signal_initial  = np.random.uniform(low=self.params.bkg_min.value.m,
                                               high=self.params.bkg_max.value.m,
                                               size=self.nsig)

        # guys, seriouslsy....?! "stuff"??
        signal_container['stuff'] = signal_initial
        # Populate its MC weight by equal constant factors
        signal_container['weights'] = np.ones(self.nsig, dtype=FTYPE)*1./self.stats_factor
        # Populate the error on those weights
        signal_container['errors'] = (np.ones(self.nsig, dtype=FTYPE)*1./self.stats_factor)**2.

        #
        # Compute the bin indices associated with each event
        #
        sig_indices = lookup_indices(sample=[signal_container['stuff']], binning=self.apply_mode)
        signal_container['bin_indices'] = sig_indices

        #
        # Compute an associated bin mask for each output bin
        #
        for bin_i in range(self.apply_mode.tot_num_bins):
            sig_bin_mask = sig_indices == bin_i
            signal_container['bin_{}_mask'.format(bin_i)] = sig_bin_mask

        #
        # Add container to the data
        #
        self.data.add_container(signal_container)

        #
        # Create a background container
        #
        if self.nbkg > 0:

            bkg_container = Container('background')
            bkg_container.representation = 'events'
            # Create a set of background events
            initial_bkg_events = np.random.uniform(low=self.params.bkg_min.value.m, high=self.params.bkg_max.value.m, size=self.nbkg)
            bkg_container['stuff'] = initial_bkg_events
            # create their associated weights
            bkg_container['weights'] = np.ones(self.nbkg)*1./self.stats_factor
            bkg_container['errors'] = (np.ones(self.nbkg)*1./self.stats_factor)**2.
            # compute their bin indices
            bkg_indices = lookup_indices(sample=[bkg_container['stuff']], binning=self.apply_mode)
            bkg_container['bin_indices'] = bkg_indices
            # Add bin indices mask (used in generalized poisson llh)
            for bin_i in range(self.apply_mode.tot_num_bins):
                bkg_bin_mask = bkg_indices==bin_i
                bkg_container['bin_{}_mask'.format(bin_i)] = bkg_bin_mask

            self.data.add_container(bkg_container)


        #
        # Add the binned counterpart of each events container
        #
        for container in self.data:
            container.array_to_binned('weights',
                self.data.representation, self.apply_mode, averaged=False
            )
            container.array_to_binned('errors',
                self.data.representation, self.apply_mode, averaged=False
            )


    def apply_function(self):
        '''
        This is where we re-weight the signal container
        based on a model gaussian with tunable parameters
        mu and sigma.

        The background is left untouched in this step.

        A possible upgrade to this function would be to make a
        small background re-weighting

        This function will be called at every iteration of the minimizer
        '''

        #
        # Make sure we are in events mode
        #
        self.data.representation = 'events'

        for container in self.data:

            if container.name == 'signal':
                #
                # Signal is a gaussian pdf, weighted to account for the MC statistics and the signal fraction
                #
                reweighting = norm.pdf(container['stuff'], loc=self.params['mu'].value.m, scale=self.params['sigma'].value.m)/self.stats_factor
                reweighting/=np.sum(reweighting)
                reweighting*=(self.nsig/self.stats_factor)

                reweighting[np.isnan(reweighting)] = 0.

                #
                # New MC errors = MCweights squared
                #
                new_errors = reweighting**2.

                #
                # Replace the weight information in the signal container
                #
                np.copyto(src=reweighting, dst=container["weights"])
                np.copyto(src=new_errors, dst=container['errors'])
                container.mark_changed('weights')
                container.mark_changed('errors')

                # Re-bin the events weight into new histograms
                container.array_to_binned('weights',
                    self.data.representation, self.apply_mode, averaged=False
                )
                container.array_to_binned('errors',
                    self.data.representation, self.apply_mode, averaged=False
                )


def init_test(**param_kwargs):
    """Initialisation example"""
    param_set = ParamSet([
        Param(name='n_events_data', value=5512, **param_kwargs),
        Param(name='stats_factor', value=1.0, **param_kwargs),
        Param(name='signal_fraction', value=0.05, **param_kwargs),
        Param(name='bkg_min', value=50., **param_kwargs),
        Param(name='bkg_max', value=100., **param_kwargs),
        Param(name='mu', value=75., **param_kwargs),
        Param(name='sigma', value=8.5, **param_kwargs),
    ])

    return simple_signal(params=param_set)
