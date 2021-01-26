"""
A Stage to load data from a CSV datarelease format file into a PISA pi ContainerSet
"""

from __future__ import absolute_import, print_function, division

import numpy as np
import pandas as pd

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.utils import vectorizer
from pisa.utils.profiler import profile
from pisa.utils.resources import find_resource
from pisa.core.container import Container


class csv_icc_hist(Stage):
    """
    CSV file loader PISA class

    Parameters
    ----------
    events_file : csv file path

    """
    def __init__(
        self,
        events_file,
        **std_kwargs,
    ):
        # instantiation args that should not change
        self.events_file = find_resource(events_file)

        expected_params = ('atm_muon_scale',)

        # init base class
        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

    def setup_function(self):
        events = pd.read_csv(self.events_file)

        container = Container('icc')
        container.data_specs = 'events'

        container['count'] = events['count'].values.astype(FTYPE)
        container['weights'] = np.ones(container.size, dtype=FTYPE)
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


    def apply_function(self):
        scale = self.params.atm_muon_scale.m_as('dimensionless')

        for container in self.data:
            container['weights'] = container['count'] * scale
