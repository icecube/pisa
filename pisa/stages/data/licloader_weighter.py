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
        pass

    def apply_function(self):
        """
        Reset all the weights to the initial weights 
        """
        for container in self.data:
            container['weights'] = np.copy(container['initial_weights'])