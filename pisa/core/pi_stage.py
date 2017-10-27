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

    input_spec : binning or 'evts' or None
        Specify the inputs (i.e. what did the last stage output, or None)

    calc_spec : binning or 'evts' or None
        Specify in what to do the calculation

    apply_spec : binning or 'evts' or None
        Specify onto what to applt to generate the outputs

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
                 apply_specs=None,
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
        self.apply_specs = apply_specs
        self.events = events

    def setup(self):
        pass

    def compute(self):
        pass

    def apply(self, inputs=None):
        if inputs is None:
            if self.apply_specs is None:
                pass

            elif self.apply_specs == 'events':
                if self.events is None:
                    raise TypeError('Cannot apply to events with no events present')
                # run apply_atomic on events array

            elif isinstance(self.apply_specs, MultiDimBinning):
                if self.events is None:
                    raise TypeError('Cannot return Map with no inputs and no events present')
                else:
                    # run apply_atomic on events array and a private output array
                    # ToDo: private weights array (or should it be normal weights array?)
                    binning = self.apply_specs
                    bin_edges = bin_edges = [edges.magnitude for edges in binning.bin_edges]
                    binning_cols = binning.names

                    maps = []
                    for name, evts in self.events.items():
                        sample = [evts[colname].get('host') for colname in binning_cols]
                        hist, _ = np.histogramdd(sample=sample,
                                                 weights=evts['weights'].get('host'),
                                                 bins=bin_edges,
                                                 )

                        maps.append(Map(name=name, hist=hist, binning=binning))
                    self.outputs = MapSet(name='bla', maps=maps)
                    return self.outputs
                # histogram events


    #def get_outputs(self, inputs=None):
    #   pass
