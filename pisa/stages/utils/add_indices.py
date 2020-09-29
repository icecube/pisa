'''
PISA module to prep incoming data into formats that are
compatible with the mc_uncertainty likelihood formulation

This module takes in events containers from the pipeline, and
introduces an additional array giving the indices where each
event falls into.

module structure imported from bootcamp example
'''

from __future__ import absolute_import, print_function, division

__author__ = "Etienne Bourbeau (etienne.bourbeau@icecube.wisc.edu)"

import numpy as np

from pisa import FTYPE
from pisa.core.stage import Stage
#from pisa.utils.log import logging

# Load the modified index lookup function
from pisa.core.bin_indexing import lookup_indices



class add_indices(Stage):
    """
    PISA Pi stage to map out the index of the analysis
    binning where each event falls into.

    Parameters
    ----------
    params
        foo : Quantity
        bar : Quanitiy with time dimension

    Notes:
    ------
    - input and calc specs are predetermined in the module
        (inputs from the config files will be disregarded)

    - stage appends an array quantity called bin_indices
    - stage also appends an array mask to access events by
      bin index later in the pipeline

    """

    # this is the constructor with default arguments
    
    def __init__(self,
                 **std_kwargs,
                 ):


        # init base class
        super(add_indices, self).__init__(
                                       expected_params=(),
                                       **std_kwargs,
                                       )


    def setup_function(self):
        '''
        Calculate the bin index where each event falls into

        Create one mask for each analysis bin.
        '''
        
        assert self.calc_mode == 'events', 'ERROR: calc specs must be set to "events for this module'

        self.data.representation = 'events'

        for container in self.data:
            # Generate a new container called bin_indices
            container['bin_indices'] = np.empty((container.size), dtype=np.int64)
  
            variables_to_bin = []
            for bin_name in self.apply_mode.names:
                variables_to_bin.append(container[bin_name])

            new_array = lookup_indices(sample=variables_to_bin,
                                       binning=self.apply_mode)

            new_array = new_array
            np.copyto(src=new_array, dst=container["bin_indices"])


            for bin_i in range(self.apply_mode.tot_num_bins):
                container.add_array_data(key='bin_{}_mask'.format(bin_i), 
                                         data=(new_array == bin_i))



