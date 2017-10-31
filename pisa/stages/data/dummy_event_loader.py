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
                 apply_specs=None,
                 ):

        expected_params = ('n_events')
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
                                                apply_specs=apply_specs,
                                                )

        # doesn't calculate anything
        assert self.calc_mode is None

    def setup(self):

        # create some dumb events

        # number of points for E x CZ grid
        n_events = int(self.params.n_events.value.m)

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

        
        numu = {'true_energy' : SmartArray(energy),
                'true_coszen' : SmartArray(cz),
                'nubar' : SmartArray(nubar),
                'event_weights' : SmartArray(event_weights),
                'weights' : SmartArray(weights),
                'flux_nue' : SmartArray(flux_nue),
                'flux_numu' : SmartArray(flux_numu),
                }

        # add the events
        self.events['numu'] = numu


    def apply(self, inputs=None):
        if inputs is None:
            if self.apply_mode is None:
                # nothing to be applied
                self.outputs = self.inputs

            elif self.apply_mode == 'events':
                self.outputs = self.inputs
                # apply weights
                for name, val in self.events.items():
                    val['weights'] = val['weights'].get('host') * val['event_weights'].get('host')

            elif self.apply_mode == 'binned':
                # run apply_kernel on events array and a private output array
                # ToDo: private weights array (or should it be normal weights array?)
                binning = self.apply_specs
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
