"""
A Stage to load data from a PISA style hdf5 file into a PISA pi ContainerSet
"""
from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils import vectorizer
from pisa.utils.profiler import profile
from pisa.core.container import Container
from pisa.core.events_pi import EventsPi


class simple_data_loader(PiStage):
    """
    HDF5 file loader PISA Pi class

    Paramaters
    ----------

    events_file : hdf5 file path
        output from make_events, including flux weights and Genie systematics coefficients

    mc_cuts : cut expr
        e.g. '(true_coszen <= 0.5) & (true_energy <= 70)'

    data_dict : str of a dict
        dictionary to specify what keys from the hdf5 files to be loaded under what name
        entries can be strings that point to the right key in the hdf5 file
        or lists of keys, and the data will be stacked into a 2d array

    Notes
    -----

    """
    def __init__(self,
                 events_file,
                 mc_cuts,
                 data_dict,
                 data=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                ):

        self.events_file = events_file
        self.mc_cuts = mc_cuts
        self.data_dict = data_dict

        expected_params = (
                          )
        input_apply_keys = ('initial_weights',
                           )
        output_apply_keys = ('weights',
                            )

        # init base class
        super(simple_data_loader, self).__init__(data=data,
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
                                                )

        # doesn't calculate anything
        assert self.calc_mode is None

        # check output names
        assert len(self.output_names) > 0, "Must specify at least one element in `output_names`"
        assert len(self.output_names) == len(set(self.output_names)), "Found duplicates in `output_names`"


    def setup_function(self):

        # --- Load the events ---

        # open Events file
        evts = EventsPi(name="Events")
        self.data_dict = eval(self.data_dict)
        evts.load_events_file(self.events_file,self.data_dict)

        #Apply any cuts that the user defined
        if self.mc_cuts is not None:
            logging.info('applying the following cuts to events: %s'%self.mc_cuts)
            evts = evts.apply_cut(self.mc_cuts)
                    
        #Create containers from the events
        for name in self.output_names:

            # check container doesn't already exist
            assert name not in self.data.names, "Cannot add `%s` data container, a container with this name already exists" % name 

            # make container
            container = Container(name)
            container.data_specs = 'events'

            if name not in evts :
                raise ValueError("Output name '%s' not found in events : %s" % (name,evts.keys()) )

            #Add the events data to the container
            for key,val in evts[name].items() :
                container.add_array_data(key, val)

            # create weight arrays
            # define an `initial_weights` array that will be the starting point (and never modified), 
            # and a `weights` array that will be initialised from `initial_weights` modified by the stages
            # user can also provide the `initial_weights` values in the input file if desired
            assert 'weights' not in container.array_data, "Found an existing `weights` array in %s, this would be overwritten (this perhaps should be `initial_weights``"%name
            if "initial_weights" not in container.array_data : #TODO Need to fix the bug in checking for something being `in` a container
              container.add_array_data('initial_weights', np.ones(container.size, dtype=FTYPE))
            container.add_array_data('weights', np.ones(container.size, dtype=FTYPE)) # The values can be anything (will be overwritten by `initial_weights`)

            # add neutrino flavor information for neutrino events
            if name.startswith("nu") :
                # this determination of flavour is the worst possible coding, ToDo
                nubar = -1 if 'bar' in name else 1
                if name.startswith('nutau'):
                    flav = 2
                elif name.startswith('numu'):
                    flav = 1
                elif name.startswith('nue'):
                    flav = 0
                else:
                    raise ValueError('Cannot determine flavour of %s'%name)
                container.add_scalar_data('nubar', nubar)
                container.add_scalar_data('flav', flav)

            self.data.add_container(container)

        # check created at least one container
        assert len(self.data.names) > 0, "No containers created during data loading"

        # test
        if self.output_mode == 'binned':
            #self.data.data_specs = self.output_specs
            for container in self.data:
                container.array_to_binned('weights', self.output_specs)


    @profile
    def apply_function(self):

        # reset the weights to the initial weights prior to the downstream stages running
        self.data.data_specs = self.output_specs
        for container in self.data:
            vectorizer.set(container['initial_weights'],out=container['weights'])

