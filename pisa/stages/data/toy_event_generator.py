"""
Stage to generate some random data
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.binning import MultiDimBinning
from pisa.core.container import Container
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage

__all__ = ['toy_event_generator', 'init_test']


class toy_event_generator(Stage):  # pylint: disable=invalid-name
    """
    random toy event generator PISA Pi class

    Parameters
    ----------

    output_names : str
        list of output names

    params
        Expected params .. ::

            n_events : int
                Number of events to be generated per output name
            random
            seed : int
                Seed to be used for random

    """
    def __init__(
        self,
        output_names,
        **std_kwargs,
    ):

        expected_params = ('n_events', 'random', 'seed')

        self.output_names = output_names

        # init base class
        super().__init__(
            expected_params=expected_params,
            expected_container_keys=(),
            **std_kwargs,
        )

    def setup_function(self):

        n_events = int(self.params.n_events.value.m)
        seed = int(self.params.seed.value.m)
        self.random_state = np.random.RandomState(seed)

        for name in self.output_names:

            container = Container(name, representation=self.calc_mode)

            nubar = -1 if 'bar' in name else 1
            if 'e' in name:
                flav = 0
            if 'mu' in name:
                flav = 1
            if 'tau' in name:
                flav = 2

            if not isinstance(self.calc_mode, MultiDimBinning):
                # Generate some events in the array representation just to have them
                # here we add those explicitly in the array representation
                container['true_energy'] = np.power(10, self.random_state.rand(n_events).astype(FTYPE) * 3)
                container['true_coszen'] = self.random_state.rand(n_events).astype(FTYPE) * 2 - 1

            size = container.size

            # make some initial weights
            if self.params.random.value:
                container['initial_weights'] = self.random_state.rand(size).astype(FTYPE)
            else:
                container['initial_weights'] = np.ones(size, dtype=FTYPE)

            # other necessary info
            container.set_aux_data('nubar', nubar)
            container.set_aux_data('flav', flav)
            container['weights'] = np.ones(size, dtype=FTYPE)
            container['weighted_aeff'] = np.ones(size, dtype=FTYPE)

            flux_nue = np.zeros(size, dtype=FTYPE)
            flux_numu = np.ones(size, dtype=FTYPE)
            flux = np.stack([flux_nue, flux_numu], axis=1)

            container['nu_flux_nominal'] = flux
            container['nubar_flux_nominal'] = flux

            self.data.add_container(container)

    def apply_function(self):
        # reset weights
        for container in self.data:
            container['weights'] = np.copy(container['initial_weights'])


def init_test(**param_kwargs):
    """Initialisation example"""
    param_set = ParamSet([
        Param(name='n_events', value=100, **param_kwargs),
        Param(name='random', value=1, **param_kwargs),
        Param(name='seed', value=666, **param_kwargs),
    ])
    return toy_event_generator(output_names=['nu_me', 'nu_you'], params=param_set)
