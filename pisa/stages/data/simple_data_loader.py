"""
A Stage to load data from a PISA style hdf5 file into a PISA pi ContainerSet
"""

#TODO This class is become dcereasingly "simple"! Make it into a more specific stage for our purposes and recreate a much more simple HDF5 file loader that is generic for any PISA task

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.utils import vectorizer
from pisa.utils.profiler import profile
from pisa.core.container import Container
from pisa.core.events_pi import EventsPi
from pisa.utils.format import arg_str_seq_none, split


class simple_data_loader(Stage):
    """
    HDF5 file loader PISA Pi class

    Parameters
    ----------

    events_file : hdf5 file path
        output from make_events, including flux weights
        and Genie systematics coefficients

    mc_cuts : cut expr
        e.g. '(true_coszen <= 0.5) & (true_energy <= 70)'

    data_dict : str of a dict
        Dictionary to specify what keys from the hdf5 files to be loaded
        under what name. Entries can be strings that point to the right
        key in the hdf5 file or lists of keys, and the data will be
        stacked into a 2d array.

    neutrinos : bool
        Flag indicating whether data events represent neutrinos
        In this case, special handling for e.g. nu/nubar, CC vs NC, ...

    fraction_events_to_keep : float
        Fraction of loaded events to use (use to downsample).
        Must be in range [0.,1.], or disable by setting to `None`.
        Default in None.

    Notes
    -----
    Looks for `initial_weights` fields in events file, which will serve
    as nominal weights for all events included.
    No fields named `weights` may already be present.

    """
    def __init__(self,
                 events_file,
                 mc_cuts,
                 data_dict,
                 neutrinos=True,
                 required_metadata=None,
                 fraction_events_to_keep=None,
                 events_subsample_index=0,
                 seed=123456,
                 output_names=None,
                 **std_kwargs,
                ):

        # instantiation args that should not change
        self.events_file = events_file
        self.mc_cuts = mc_cuts
        self.data_dict = data_dict
        self.neutrinos = neutrinos
        self.required_metadata = required_metadata
        self.fraction_events_to_keep = fraction_events_to_keep
        self.events_subsample_index = int(events_subsample_index)
        self.seed = int(seed)
        self.output_names = output_names

        # Handle list inputs
        self.events_file = split(self.events_file)
        if self.required_metadata is not None :
            self.required_metadata = split(self.required_metadata)

        # instead of adding params here, consider making them instantiation
        # args so nothing external will inadvertently try to change
        # their values
        expected_params = ()

        # init base class
        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

        # check output names
        if len(self.output_names) != len(set(self.output_names)):
            raise ValueError(
                'Found duplicates in `output_names`, but each name must be'
                ' unique.'
            )

        self.load_events()
        self.apply_cuts_to_events()

    def load_events(self):
        '''Loads events from events file'''

        # Create the events structure
        self.evts = EventsPi(
            name='Events',
            neutrinos=self.neutrinos,
            fraction_events_to_keep=self.fraction_events_to_keep,
            events_subsample_index=self.events_subsample_index,
        )

        # If user provided a variable mapping dict, parse it from the input string (if not already done)
        if self.data_dict is not None :
            if isinstance(self.data_dict, str):
                self.data_dict = eval(self.data_dict)

        # Load the event file into the events structure
        self.evts.load_events_file(
            events_file=self.events_file,
            variable_mapping=self.data_dict,
            required_metadata=self.required_metadata,
            seed=self.seed,
        )

        if hasattr(self.evts, "metadata"):
            self.metadata = self.evts.metadata

        # TODO Add option to define eventual binning here so that can cut events
        # now that will be cut later anyway (use EventsPi.keep_inbounds)

    def apply_cuts_to_events(self):
        '''Just apply any cuts that the user defined'''
        if self.mc_cuts:
            self.evts = self.evts.apply_cut(self.mc_cuts)

    def record_event_properties(self):
        '''Adds fields present in events file and selected in `self.data_dict`
        into containers for the specified output names. Also ensures the
        presence of a set of nominal weights.
        '''

        # define which  categories to include in the data
        # user can manually specify what they want using `output_names`, or else just use everything
        output_keys = self.output_names if len(self.output_names) > 0 else self.evts.keys()

        # create containers from the events
        for name in output_keys:

            # make container
            container = Container(name)
            container.representation = 'events'
            event_groups = self.evts.keys()
            if name not in event_groups:
                raise ValueError(
                    'Output name "%s" not found in events. Only found %s.'
                    % (name, event_groups)
                )

            # add the events data to the container
            for key, val in self.evts[name].items():
                container[key] = val

            # create weights arrays:
            # * `initial_weights` as starting point (never modified)
            # * `weights` to be initialised from `initial_weights`
            #   and modified by the stages
            # * user can also provide `initial_weights` in input file
            #TODO Maybe add this directly into EventsPi
            if 'weights' in container.keys:
                # raise manually to give user some helpful feedback
                raise KeyError(
                    'Found an existing `weights` array in "%s"'
                    ' which would be overwritten. Consider renaming it'
                    ' to `initial_weights`.' % name
                )
            container['weights'] = np.ones(container.size, dtype=FTYPE)

            if 'initial_weights' not in container.keys:
                if self.fraction_events_to_keep is None:
                    container['initial_weights'] = np.ones(container.size, dtype=FTYPE)
                else :
                    if 'nu' in name or 'mu' in name:
                        # Need to scale weights if using down-sampling
                        container['initial_weights'] = np.full(container.size, 1. / float(self.fraction_events_to_keep), dtype=FTYPE)
                    else:
                        container['initial_weights'] = np.ones(container.size, dtype=FTYPE)

            # add neutrino flavor information for neutrino events
            #TODO Maybe add this directly into EventsPi
            if self.neutrinos:
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
                container.set_aux_data('nubar', nubar)
                container.set_aux_data('flav', flav)

            self.data.add_container(container)

        # check created at least one container
        if len(self.data.names) == 0:
            raise ValueError(
                'No containers created during data loading for some reason.'
            )


    def setup_function(self):
        '''Store event properties from events file at
        service initialisation. Cf. `Stage` docs.
        '''
        self.record_event_properties()


    def apply_function(self):

        # reset data representation to events
        #TODO This should be fixed more generally at the Pipeline level, see XXX
        self.data.representation = "events"

        # reset weights to initial weights prior to downstream stages running
        for container in self.data:
            container['weights'] = np.copy(container['initial_weights'])
