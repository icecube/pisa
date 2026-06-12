"""
Stage to create a grid of data
"""
from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.core.container import Container
from pisa_tests.test_services import TEST_BINNING

__all__ = ['grid', 'init_test']


class grid(Stage):  # pylint: disable=invalid-name
    """
    Create a grid of events

    Parameters
    ----------
    grid_binning : :py:class:`~.MultiDimBinning`
        Binning object defining the grid to be generated

    entity : str
        `entity` arg to be passed to :py:meth:`~.MultiDimBinning.meshgrid`

    output_names : array_like
        List of output names (event types)
    """

    def __init__(
        self,
        grid_binning,
        entity="midpoints",
        output_names=None,
        **std_kwargs,
    ):
        expected_params = ()
        expected_container_keys=()

        # store args
        self.grid_binning = grid_binning
        self.entity = entity
        self.output_names = output_names

        # only intended to work with calc_mode="events"
        supported_reps = {
            'calc_mode': "events",
        }
        # init base class
        super(grid, self).__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            supported_reps=supported_reps,
            **std_kwargs,
        )
        assert self.output_names is not None

    def setup_function(self):

        for name in self.output_names:

            # Create the container
            container = Container(name, self.calc_mode)

            # Determine flavor
            nubar = -1 if 'bar' in name else 1
            if 'e' in name:
                flav = 0
            if 'mu' in name:
                flav = 1
            if 'tau' in name:
                flav = 2

            # Create arrays
            mesh = self.grid_binning.meshgrid(entity=self.entity, attach_units=False)
            size = mesh[0].size
            for var_name, var_vals in zip(self.grid_binning.names, mesh):
                container[var_name] = var_vals.flatten().astype(FTYPE)

            # Add useful info
            container.set_aux_data('nubar', nubar)
            container.set_aux_data('flav', flav)

            # Make some initial weights
            container['initial_weights'] = np.ones(size, dtype=FTYPE)
            container['weights'] = np.ones(size, dtype=FTYPE)

            self.data.add_container(container)

    def apply_function(self):
        # reset weights
        for container in self.data:
            container['weights'] = np.copy(container['initial_weights'])


def init_test(**param_kwargs):
    """Instantiation example"""
    return grid(
        grid_binning=TEST_BINNING, calc_mode='events',
        output_names = ['nue_cc', 'numu_cc', 'nutau_cc', 'nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                        'nue_nc', 'numu_nc', 'nutau_nc', 'nuebar_nc', 'numubar_nc', 'nutaubar_nc']
    )
