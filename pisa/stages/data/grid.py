"""
Stage to create a grid of data
"""
from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.utils import vectorizer
from pisa.core.container import Container


class grid(Stage):
    """
    Create a grid of events

    Parameters
    ----------

        Binning object defining the grid to be generated

    entity : str
        `entity` arg to be passed to `MultiDimBinning.meshgrid` (see that
        fucntion docs for details)

    """
    def __init__(
        self,
        entity="midpoints",
        output_names=None,
        **std_kwargs,
    ):
        expected_params = ()


        # store args
        self.entity = entity
        self.output_names = output_names

        # init base class
        super(grid, self).__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

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
            mesh = self.calc_mode.meshgrid(entity=self.entity, attach_units=False)
            size = mesh[0].size
            for var_name, var_vals in zip(self.calc_mode.names, mesh):
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
