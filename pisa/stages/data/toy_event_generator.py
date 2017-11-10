import numpy as np
from numba import SmartArray

from pisa import *
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.numba_tools import equal_and_scale, WHERE
from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.core.container import Container, ContainerSet


class toy_event_generator(PiStage):
    """
    random toy event generator PISA Pi class

    Paramaters
    ----------

    n_events : int
        Number of events to be generated per output name

    seed : int
        Seed to be used for random

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

        expected_params = ('n_events',
                           'seed',
                           )
        # init base class
        super(toy_event_generator, self).__init__(
                                                data=data,
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

    def setup_function(self):

        n_events = int(self.params.n_events.value.m)
        seed = int(self.params.seed.value.m)
        np.random.seed(seed)

        for name in self.output_names:
            # generate
            true_energy = np.power(10, np.random.rand(n_events).astype(FTYPE) * 3)
            true_coszen = np.random.rand(n_events).astype(FTYPE) * 2 - 1
            nubar = -1 if 'bar' in name else 1
            if 'e' in name: flav = 0
            if 'mu' in name: flav = 1
            if 'tau' in name: flav = 2
            event_weights = np.random.rand(n_events).astype(FTYPE)
            weights = np.ones(n_events, dtype=FTYPE)
            flux_nue = np.zeros(n_events, dtype=FTYPE)
            flux_numu = np.ones(n_events, dtype=FTYPE)

            # make container
            container = Container(name)
            container.add_array_data('true_energy', true_energy)
            container.add_array_data('true_coszen', true_coszen)
            container.add_scalar_data('nubar', nubar)
            container.add_scalar_data('flav', flav)
            container.add_array_data('event_weights', event_weights)
            container.add_array_data('weights', weights)
            container.add_array_data('weighted_aeff', weights)
            container.add_array_data('neutrino_nue_flux', flux_nue)
            container.add_array_data('neutrino_numu_flux', flux_numu)
            self.data.add_container(container)


    def apply_function(self):
        # reset weights
        for container in self.data:
            equal_and_scale(1.,
                            container['event_weights'].get(WHERE),
                            out=container['weights'].get(WHERE))
            container['weights'].mark_changed(WHERE)
