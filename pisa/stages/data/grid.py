"""
Stage to create a grid of data
"""
from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils import vectorizer
from pisa.core.container import Container


class grid(PiStage):
    """
    Create a grid of events

    Paramaters
    ----------

    input_specs : MultiDimBinning
        Binning object defining the grid to be generated

    entity : str
        `entity` arg to be passed to `MultiDimBinning.meshgrid` (see that fucntion docs for details)

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
                 entity="midpoints",
                ):

        expected_params = ()

        input_apply_keys = ('initial_weights',
                           'weights',
                           )

        # store args
        self.entity = entity

        # init base class
        super(grid, self).__init__(data=data,
                                                  params=params,
                                                  expected_params=expected_params,
                                                  input_names=input_names,
                                                  output_names=output_names,
                                                  debug_mode=debug_mode,
                                                  input_specs=input_specs,
                                                  calc_specs=calc_specs,
                                                  output_specs=output_specs,
                                                  input_apply_keys=input_apply_keys,
                                                 )

        # definition must be a grid
        assert self.input_mode == 'binned'

        # doesn't calculate anything
        assert self.calc_mode is None

    def setup_function(self):

        for name in self.output_names:

            # Create the container
            container = Container(name)
            container.data_specs = self.input_specs

            # Determine flavor
            nubar = -1 if 'bar' in name else 1
            if 'e' in name:
                flav = 0
            if 'mu' in name:
                flav = 1
            if 'tau' in name:
                flav = 2

            # Create arrays
            mesh = self.input_specs.meshgrid(entity=self.entity, attach_units=False)
            size = mesh[0].size
            for var_name,var_vals in zip(self.input_specs.names,mesh) :
                container.add_array_data( var_name, var_vals.flatten().astype(FTYPE) )

            # Add useful info
            container.add_scalar_data('nubar', nubar)
            container.add_scalar_data('flav', flav)

            # Make some initial weights
            container['initial_weights'] = np.ones(size, dtype=FTYPE)
            container['weights'] =  np.ones(size, dtype=FTYPE)

            self.data.add_container(container)


    def apply_function(self):
        # reset weights
        for container in self.data:
            vectorizer.scale(1.,
                             container['initial_weights'],
                             out=container['weights'])
