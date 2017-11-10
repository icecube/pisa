import numpy as np
from numba import guvectorize, SmartArray

from pisa import *
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet


class pi_hist(PiStage):
    """
    stage to histogram events

    Paramaters
    ----------

    None

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

        expected_params = ()
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_keys = ('weights',
                      )
        # what are keys added or altered in the calculation used during apply 
        calc_keys = ()
        # what keys are added or altered for the outputs during apply
        output_keys = ('weights',
                       )

        # init base class
        super(pi_hist, self).__init__(data=data,
                                       params=params,
                                       expected_params=expected_params,
                                       input_names=input_names,
                                       output_names=output_names,
                                       debug_mode=debug_mode,
                                       input_specs=input_specs,
                                       calc_specs=calc_specs,
                                       output_specs=output_specs,
                                       input_keys=input_keys,
                                       calc_keys=calc_keys,
                                       output_keys=output_keys,
                                       )

        assert self.input_mode is not None
        #assert self.input_mode is 'events'
        assert self.calc_mode is None
        assert self.output_mode is 'binned'


    def apply(self):
        
        self.data.data_specs = self.input_specs
        # this is special, we want the actual event weights in the histo

        if self.input_mode == 'binned':
            for container in self.data:
                container.array_to_binned('event_weights', self.output_specs, averaged=False)
                weights = container['weights'].get('host')
                weights *= container['event_weights'].get('host')
                container['weights'].mark_changed('host')

        elif self.input_mode == 'events':
            for container in self.data:
                weights = container['weights'].get('host')
                weights *= container['event_weights'].get('host')
                container['weights'].mark_changed('host')
                container.array_to_binned('weights', self.output_specs, averaged=False)
