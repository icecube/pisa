"""
A Stage to load data from a CSV datarelease format file into a PISA pi ContainerSet
"""

from __future__ import absolute_import, print_function, division

import numpy as np
import pandas as pd

from pisa import FTYPE
from pisa.core.pi_stage import PiStage
from pisa.utils import vectorizer
from pisa.utils.profiler import profile
from pisa.core.container import Container
from pisa.core.events_pi import EventsPi


class csv_icc_hist(PiStage):
    """
    CSV file loader PISA Pi class

    Parameters
    ----------

    events_file : csv file path

    """
    def __init__(self,
                 events_file,
                 data=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 error_method=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                ):

        # instantiation args that should not change
        self.events_file = events_file

        expected_params = ('atm_muon_scale',)

        input_apply_keys = ('weights',
                           )
        # copy of initial weights, to be modified by later stages
        output_apply_keys = (
            'weights',
        )
        # init base class
        super().__init__(
            data=data,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            error_method=error_method,
            input_specs=input_specs,
            calc_specs=calc_specs,
            output_specs=output_specs,
            input_apply_keys=input_apply_keys,
            output_apply_keys=output_apply_keys,
        )

        assert self.output_mode == 'binned'
        assert self.error_method == 'fixed'

    def setup_function(self):

        events = pd.read_csv(self.events_file)

        container = Container('icc')
        container.data_specs = 'events'

        container['count'] = events['count'].values.astype(FTYPE)
        container['weights'] = np.ones(container.array_length, dtype=FTYPE)
        container['errors'] = events['abs_uncert'].values.astype(FTYPE)
        container['reco_energy'] = events['reco_energy'].values.astype(FTYPE)
        container['reco_coszen'] = events['reco_coszen'].values.astype(FTYPE)
        container['pid'] = events['pid'].values.astype(FTYPE)

        self.data.add_container(container)

        # check created at least one container
        if len(self.data.names) == 0:
            raise ValueError(
                'No containers created during data loading for some reason.'
            )

        # let's convert that into the right binning
        container.array_to_binned('weights', self.output_specs)
        container.array_to_binned('count', self.output_specs)
        container.array_to_binned('errors', self.output_specs)


    @profile
    def apply_function(self):
        scale = self.params.atm_muon_scale.m_as('dimensionless')
        for container in self.data:
            vectorizer.scale(scale, container['count'], out=container['weights'])

