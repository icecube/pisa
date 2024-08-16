"""
A Stage to load data from an sqlite database
"""

from __future__ import absolute_import, print_function, division
import sqlite3

import numpy as np
import pandas as pd

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.core.container import Container

# TODO: docstrings
class sqlite_loader(Stage):  # pylint: disable=invalid-name
    """
    SQLite loader PISA Pi class
    Parameters
    ----------
    database : path to sqlite database
    **kwargs
        Passed to Stage
    """
    def __init__(
        self,
        database,
        output_names,
        post_fix = '_pred',
        **std_kwargs,
    ):

        # instantiation args that should not change
        self.database = database
        self.post_fix = post_fix

        # init base class
        super().__init__(
            expected_params=(),
            expected_container_keys=(),
            **std_kwargs,
        )

        self.output_names = output_names

    def get_pid_and_interaction_type(self,name):
        '''Sorry'''
        if 'bar' in name:
            nubar = -1
        else:
            nubar = 1
        if 'e' in name:
            pid = 12
            flavor = 0
        if 'mu' in name:
            pid = 14
            flavor = 1
        if 'tau' in name:
            pid = 16
            flavor = 2
        if 'cc' in name:
            interaction_type = 1
        if 'nc' in name:
            interaction_type = 2
        return nubar*pid, interaction_type, nubar, flavor

    def query_database(self, interaction_type, pid):
        with sqlite3.connect(self.database) as con:
            # Get truth
            query = 'SELECT * FROM truth WHERE interaction_type = %s and pid = %s'%(interaction_type, pid)
            truth = pd.read_sql(query,con).sort_values('event_no').reset_index(drop = True)
            if self.post_fix == '_retro':
                # Get retro reco
                query = 'SELECT * FROM retro WHERE event_no in %s'%(str(tuple(truth['event_no'])))
                reco = pd.read_sql(query,con).sort_values('event_no').reset_index(drop = True)
            else:
                # Get GNN reco
                query = 'SELECT * FROM reconstruction WHERE event_no in %s'%(str(tuple(truth['event_no'])))
                reco = pd.read_sql(query,con).sort_values('event_no').reset_index(drop = True)
            # Get number of i3 files with specified PID by counting unique combinations of RunID and SubrunID
            query = 'SELECT DISTINCT RunID, SubrunID FROM truth WHERE pid = %s'%pid
            n_files = len(pd.read_sql(query,con))
        return truth, reco, n_files

    def add_truth(self, container, truth, nubar, flavor):
        ''' Adds truth to container'''
        container['true_coszen'] = np.cos(truth['zenith']).values.astype(FTYPE)
        container['true_energy'] = truth['energy'].values.astype(FTYPE)
        container.set_aux_data("nubar", nubar) # This sets true nu/nubar
        container.set_aux_data("flav", flavor) # This sets the true flavor
        return container

    def add_reco(self, container, reco):
        ''' Adds reconstructed quantities to container'''

        container['reco_coszen'] = np.cos(reco['zenith' + self.post_fix]).values.astype(FTYPE)
        container['reco_' + 'energy'] = reco['energy' + self.post_fix].values.astype(FTYPE)
        if self.post_fix == '_retro':
            container['pid'] = reco['L7_PIDClassifier_FullSky_ProbTrack'].values.astype(FTYPE)
        else:
            container['pid'] = reco['track' + self.post_fix].values.astype(FTYPE)
        return container

    def add_aeff_weight(self, container, truth, n_files):
        CM2_TO_M2 = 1e-4
        weighted_aeff = (
            CM2_TO_M2
            * truth["OneWeight"]
            / n_files
            / truth["gen_ratio"]
            / truth["NEvents"]
        )
        container['weighted_aeff'] = weighted_aeff.values.astype(FTYPE)
        return container

    def initialize_weights(self, container):
        container['weights'] = np.ones(container.size, dtype=FTYPE)
        container['initial_weights'] = np.ones(container.size, dtype=FTYPE)
        return container

    def setup_function(self):
        # create containers from the events
        for name in self.output_names:
            # make container
            container = Container(name)
            pid, interaction_type, nubar, flavor = self.get_pid_and_interaction_type(name)
            truth, reco,n_i3files_with_flavor =  self.query_database(interaction_type, pid)
            container = self.add_truth(container,truth, nubar, flavor)
            container = self.add_reco(container,reco)
            container = self.initialize_weights(container)
            container = self.add_aeff_weight(container, truth, n_i3files_with_flavor)

            self.data.add_container(container)

        # check created at least one container
        if len(self.data.names) == 0:
            raise ValueError(
                'No containers created during data loading for some reason.'
            )

    def apply_function(self):
        for container in self.data:
            container['weights'] = np.copy(container['initial_weights'])
