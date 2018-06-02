"""
Stage to generate some random data
"""
from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils import vectorizer
from pisa.core.container import Container


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
                           'random',
                           'seed',
                           )
        # init base class
        super(toy_event_generator, self).__init__(data=data,
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
            container = Container(name)
            container.data_specs = self.input_specs
            nubar = -1 if 'bar' in name else 1
            if 'e' in name:
                flav = 0
            if 'mu' in name:
                flav = 1
            if 'tau' in name:
                flav = 2

            if self.input_mode == 'events':
                # generate
                true_energy = np.power(10, np.random.rand(n_events).astype(FTYPE) * 3)
                true_coszen = np.random.rand(n_events).astype(FTYPE) * 2 - 1
                size = n_events
                container['true_energy'] = true_energy
                container['true_coszen'] = true_coszen
            elif self.input_mode == 'binned':
                size = self.input_specs.size

            if self.params.random.value:
                event_weights = np.random.rand(size).astype(FTYPE)
            else:
                event_weights = np.ones(size, dtype=FTYPE)
            weights = np.ones(size, dtype=FTYPE)
            flux_nue = np.zeros(size, dtype=FTYPE)
            flux_numu = np.ones(size, dtype=FTYPE)
            flux = np.stack([flux_nue, flux_numu], axis=1)

            # make container
            container.add_scalar_data('nubar', nubar)
            container.add_scalar_data('flav', flav)
            container['event_weights'] = event_weights
            container['weights'] = weights
            container['weighted_aeff'] = weights
            container['nominal_nu_flux'] = flux
            container['nominal_nubar_flux'] = flux

            self.data.add_container(container)


    def apply_function(self):
        # reset weights
        for container in self.data:
            vectorizer.scale(1.,
                             container['event_weights'],
                             out=container['weights'])
