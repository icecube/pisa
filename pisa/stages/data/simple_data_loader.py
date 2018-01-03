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
from pisa.core.events import Events


class simple_data_loader(PiStage):
    """
    random toy event generator PISA Pi class

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

    def setup_function(self):

        # --- Load the events ---

        # open Events file
        evts = Events(self.params.events_file.value)

        #Apply any cuts that the user defined
        if self.params.mc_cuts.value is not None:
            logging.info('applying the following cuts to events: %s'%self.params.mc_cuts.value)
            evts = evts.applyCut(self.params.mc_cuts.value)

        data_dict = eval(self.params.data_dict.value)
        
        for name in self.output_names:
            # make container
            container = Container(name)
            container.data_specs = 'events'

            for key, val in data_dict.items():
                if isinstance(val, basestring):
                    container.add_array_data(key, evts[name][val].astype(FTYPE))
                elif isinstance(val, list):
                    stack = []
                    for item in val:
                        stack.append(evts[name][item].astype(FTYPE))
                    container.add_array_data(key, np.stack(stack, axis=1))
                else:
                    raise ValueError('Unknown format %s'%val)
            
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
                raise ValueError('Cannot determine flavour of %s'%name)
            container.add_scalar_data('nubar', nubar)
            container.add_scalar_data('flav', flav)

            self.data.add_container(container)

        # test
        if self.output_mode == 'binned':
            #self.data.data_specs = self.output_specs
            for container in self.data:
                container.array_to_binned('weights', self.output_specs)


    @profile
    def apply_function(self):
        # reset weights to event_weights
        for container in self.data:
            vectorizer.set(container['event_weights'],
                           out=container['weights'])
