"""
A class to load lic files and weight existing events 
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.utils import vectorizer
from pisa.utils.profiler import profile
from pisa.core.container import Container
from pisa.core.events_pi import EventsPi


class lic_loader_weighter(Stage):
    """
    LeptonWeighter LIC file reader and LI event weighter. Sets two weight containers
        weights
        astro_weights 

    plus duplicates holding the initial weights. This way we can reweight 

    Parameters 
    ----------

    lic_files : string, or list of strings 
    """

    def __init__(self,
                lic_files):
        
        if isinstance(lic_files, str):
            self.lic_files = [lic_files,]
        elif isinstance(lic_files, (list, tuple)):
            self.lic_files = lic_files
        else:
            raise TypeError("Unknown lic_file datatype {}".format(type(lic_files)))

        

    def setup_function(self):
        """
        Load in the lic files, build the weighters, and get all the one-weights. To get the true 
        """

        raw_data = None # pd.read_csv(self.events_file)

        # create containers from the events
        for name in self.output_names:

            # make container
            container = Container(name)
            nubar = -1 if 'bar' in name else 1
            if 'e' in name:
                flav = 0
            if 'mu' in name:
                flav = 1
            if 'tau' in name:
                flav = 2

            # cut out right part
            pdg = nubar * (12 + 2 * flav)

            mask = raw_data['pdg'] == pdg
            if 'cc' in name:
                mask = np.logical_and(mask, raw_data['type'] > 0)
            else:
                mask = np.logical_and(mask, raw_data['type'] == 0)

            events = raw_data[mask]

            container['weighted_aeff'] = events['weight'].values.astype(FTYPE)
            container['weights'] = np.ones(container.size, dtype=FTYPE)
            container['initial_weights'] = np.ones(container.size, dtype=FTYPE)
            container['astro_weights'] = np.ones(container.size, dtype=FTYPE)
            container['astro_initial_weights'] = np.ones(container.size, dtype=FTYPE)

            container['true_energy'] = events['true_energy'].values.astype(FTYPE)
            container['true_coszen'] = events['true_coszen'].values.astype(FTYPE)
            container['reco_energy'] = events['reco_energy'].values.astype(FTYPE)
            container['reco_coszen'] = events['reco_coszen'].values.astype(FTYPE)
            container['pid'] = events['pid'].values.astype(FTYPE)
            container.set_aux_data('nubar', nubar)
            container.set_aux_data('flav', flav)

    def apply_function(self):
        """
        Reset all the weights to the initial weights 
        """
        for container in self.data:
            container['weights'] = np.copy(container['initial_weights'])
            container["astro_weights"] = np.copy(container["initial_astro_weights"])