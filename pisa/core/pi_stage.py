# Authors
"""
Stage class designed to be inherited by PISA Pi services, such that all basic
functionality is built-in.

"""
import numpy as np

from pisa.core.base_stage import BaseStage
from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.utils.log import logging
from pisa.utils.profiler import profile


__all__ = ['PiStage']
__version__ = 'Pi'
__author__ = 'Philipp Eller (pde3@psu.edu)'


class PiStage(BaseStage):
    """
    PISA stage base class. Should encompass all behaviors common to (almost)
    all stages.

    Specialization should be done via subclasses.

    Parameters
    ----------

    events : Events or None
        object to be passed along in any case

    input_names : None or list of strings

    output_names : None or list of strings

    input_specs : binning or 'evts' or None
        Specify the inputs (i.e. what did the last stage output, or None)

    calc_specs : binning or 'evts' or None
        Specify in what to do the calculation

    output_specs : binning or 'evts' or None
        Specify how to generate the outputs

    """
    def __init__(self,
                 events=None,
                 params=None,
                 expected_params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                 ):

        # init base class!
        super(PiStage, self).__init__(params=params,
                                      expected_params=expected_params,
                                      input_names=input_names,
                                      output_names=output_names,
                                      debug_mode=debug_mode,
                                      )

        self.input_specs = input_specs
        self.calc_specs = calc_specs
        self.output_specs = output_specs
        self.events = events

        if isinstance(self.input_specs, MultiDimBinning):
            self.input_mode = 'binned'
        elif self.input_specs == 'events':
            self.input_mode = 'events'
        elif self.input_specs is None:
            self.input_mode = None
        else:
            raise ValueError('Not understood input_specs %s'%input_specs)

        if isinstance(self.calc_specs, MultiDimBinning):
            self.calc_mode = 'binned'
        elif self.calc_specs == 'events':
            self.calc_mode = 'events'
        elif self.calc_specs is None:
            self.calc_mode = None
        else:
            raise ValueError('Not understood calc_specs %s'%calc_specs)

        if isinstance(self.output_specs, MultiDimBinning):
            self.output_mode = 'binned'
        elif self.output_specs == 'events':
            self.output_mode = 'events'
            if self.events == {}:
                raise ValueError('Cannot do apply mode `events` with no events present')
        elif self.output_specs is None:
            self.output_mode = None
        else:
            raise ValueError('Not understood output_specs %s'%output_specs)


    def setup(self):
        pass

    def compute(self):
        pass

