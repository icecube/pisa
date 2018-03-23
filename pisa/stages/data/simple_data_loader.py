"""
A Stage to load data from a PISA style hdf5 file into a PISA pi ContainerSet
"""
from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.pi_stage import PiStage
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
                 data=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                ):

        expected_params = ('events_file',
                           'mc_cuts',
                           'data_dict',
                          )
        input_apply_keys = ('event_weights',
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

        self.load_events()
        self.apply_cuts_to_events()

    def load_events(self):
        # open events file
        self.evts = EventsPi(name="Events")
        self.evts_file = self.params.events_file.value
        self.data_dict = eval(self.params.data_dict.value)
        self.evts.load_events_file(
            events_file=self.evts_file,
            variable_mapping=self.data_dict
        )

    def apply_cuts_to_events(self):
        # apply any cuts that the user defined
        self.cuts = self.params.mc_cuts.value
        if self.cuts:
            self.evts = self.evts.apply_cut(self.cuts)

    def record_event_properties(self):
        # create containers from the events
        for name in self.output_names:
            # make container
            container = Container(name)
            container.data_specs = 'events'
            event_groups = self.evts.keys()
            if name not in event_groups:
                raise ValueError(
                    'Output name "%s" not found in events. Only found %s.'
                    % (name, event_groups)
                )
            # add the events data to the container
            for key, val in self.evts[name].items():
                container.add_array_data(key, val)

            # add some additional keys
            container.add_array_data('weights', np.ones(container.size, dtype=FTYPE))
            container.add_array_data('event_weights', np.ones(container.size, dtype=FTYPE))
            # this determination of flavour is the worst possible coding, ToDo
            nubar = -1 if 'bar' in name else 1
            if 'tau' in name:
                flav = 2
            elif 'mu' in name:
                flav = 1
            elif 'e' in name:
                flav = 0
            else:
                raise ValueError('Cannot determine flavour of %s' % name)
            container.add_scalar_data('nubar', nubar)
            container.add_scalar_data('flav', flav)

            self.data.add_container(container)

        # test
        if self.output_mode == 'binned':
            #self.data.data_specs = self.output_specs
            for container in self.data:
                container.array_to_binned('weights', self.output_specs)

    def setup_function(self):
        # store event properties from events file present at
        # service initialisation - whenever this service is run later on
        # we have to check whether cuts or events file have changed
        self.record_event_properties()

    def prevent_evts_mods(self):
        # TODO: are there more checks?
        original_variable_mapping = self.data_dict
        requested_variable_mapping = eval(self.params.data_dict.value)
        if requested_variable_mapping != original_variable_mapping:
            raise ValueError(
                'Found changed variable mapping while obtaining event weights:'
                ' %s -> %s. Mapping must not change!'
                % (original_variable_mapping, requested_variable_mapping)
            )
        original_evts_file = self.evts_file
        requested_evts_file = self.params.events_file.value
        if requested_evts_file != original_evts_file:
            raise ValueError(
                'Found changed events file while obtaining event weights:'
                ' %s -> %s. File must not change!'
                % (original_evts_file, requested_evts_file)
            )
        original_cuts = self.cuts
        requested_cuts = self.params.mc_cuts.value
        if requested_cuts != original_cuts:
            raise ValueError(
                'Found changed cuts while obtaining event weights:'
                ' %s -> %s. Cuts must not change!'
                % (original_cuts, requested_cuts)
            )

    @profile
    def apply_function(self):
        # seatbelts against changes to the underlying events
        self.prevent_evts_mods()
        # TODO: do we need this line? Isn't this handled universally
        # by the base class (in PiStage's apply)?
        self.data.data_specs = self.output_specs
        # reset weights to event_weights
        for container in self.data:
            vectorizer.set(container['event_weights'],
                           out=container['weights'])
