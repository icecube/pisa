import numpy as np
from numba import SmartArray

from pisa import *
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet


class dummy_event_loader(PiStage):
    """
    Dummy event loader PISA Pi class

    Paramaters
    ----------

    None

    Notes
    -----

    """
    def __init__(self,
                 events=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                 ):

        expected_params = ('n_events',
                           'seed',
                           )
        input_names = ()
        output_names = ()

        # init base class!
        super(dummy_event_loader, self).__init__(
                                                events=events,
                                                params=params,
                                                expected_params=expected_params,
                                                input_names=input_names,
                                                output_names=output_names,
                                                debug_mode=debug_mode,
                                                input_specs=input_specs,
                                                calc_specs=calc_specs,
                                                output_specs=output_specs,
                                                )

        # doesn't calculate anything
        assert self.calc_mode is None

    def setup(self):

        # create some random events

        n_events = int(self.params.n_events.value.m)

        seed = int(self.params.seed.value.m)
        np.random.seed(seed)

        for name in ['nue', 'numu', 'nutau']:
            # input arrays
            # E and CZ
            energy = np.power(10, np.random.rand(n_events).astype(FTYPE) * 3)
            cz = np.random.rand(n_events).astype(FTYPE) * 2 - 1
            # nubar
            nubar = np.ones(n_events, dtype=np.int32)
            # weights
            event_weights = np.random.rand(n_events).astype(FTYPE)
            weights = np.ones(n_events, dtype=FTYPE)
            flux_nue = np.zeros(n_events, dtype=FTYPE)
            flux_numu = np.ones(n_events, dtype=FTYPE)

            
            arrays = {'true_energy' : SmartArray(energy),
                      'true_coszen' : SmartArray(cz),
                      'nubar' : SmartArray(nubar),
                      'event_weights' : SmartArray(event_weights),
                      'weights' : SmartArray(weights),
                      'flux_nue' : SmartArray(flux_nue),
                      'flux_numu' : SmartArray(flux_numu),
                      }

            # add the events
            self.events[name] = arrays


    def apply(self, inputs=None):
        if inputs is None:
            if self.output_mode is None:
                # nothing to be applied
                self.outputs = self.inputs

            elif self.output_mode == 'events':
                self.outputs = self.inputs
                # apply weights
                for name, val in self.events.items():

                    weights = val['weights'].get('host')
                    weights *= val['event_weights'].get('host')
                    val['weights'].mark_changed('host')

            elif self.output_mode == 'binned':
                # run output_kernel on events array and a private output array
                # ToDo: private weights array (or should it be normal weights array?)
                binning = self.output_specs
                bin_edges = [edges.magnitude for edges in binning.bin_edges]
                binning_cols = binning.names

                maps = []
                for name, evts in self.events.items():
                    sample = [evts[colname].get('host') for colname in binning_cols]
                    hist, _ = np.histogramdd(sample=sample,
                                             weights=evts['weights'].get('host'),
                                             bins=bin_edges,
                                             )

                    maps.append(Map(name=name, hist=hist, binning=binning))
                self.outputs = MapSet(maps)
        return self.outputs
