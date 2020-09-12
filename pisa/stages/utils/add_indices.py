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
from pisa.core.pi_stage import PiStage
#from pisa.utils.log import logging

# Load the modified index lookup function
from pisa.core.bin_indexing import lookup_indices



class add_indices(PiStage):
    """
    PISA Pi stage to map out the index of the analysis
    binning where each event falls into.

    Parameters
    ----------
    data
    params
        foo : Quantity
        bar : Quanitiy with time dimension
    input_names
    output_names
    debug_mode
    input_specs: 
    calc_specs : must be events
    output_specs: must be a MultiDimBinnig

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
                 data=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                ):

        #
        # No parameters are expected in this stage
        # same goes for a bunch of other stage options
        #
        expected_params = ()
        input_names = ()
        output_names = ()
        input_apply_keys = ()

        # We add the bin_indices key
        # (but not in the apply function so maybe useless...)
        #
        output_calc_keys = ('bin_indices',)
        output_apply_keys = ()

        # init base class
        super(add_indices, self).__init__(data=data,
                                       params=params,
                                       expected_params=expected_params,
                                       input_names=input_names,
                                       output_names=output_names,
                                       debug_mode=debug_mode,
                                       input_specs=input_specs,
                                       calc_specs=calc_specs,
                                       output_specs=output_specs,
                                       input_apply_keys=input_apply_keys,
                                       output_apply_keys=output_apply_keys,
                                       output_calc_keys=output_calc_keys,
                                       )

        # make sure the user specified some modes
        assert self.input_mode is not None
        assert self.calc_mode is not None
        assert self.output_mode is not None

    def setup_function(self):
        '''
        Calculate the bin index where each event falls into

        Create one mask for each analysis bin.
        '''
        
        assert self.calc_specs == 'events', 'ERROR: calc specs must be set to "events for this module'

        self.data.data_specs = 'events'

        for container in self.data:
            # Generate a new container called bin_indices
            container['bin_indices'] = np.empty((container.size), dtype=np.int64)
  
            variables_to_bin = []
            for bin_name in self.output_specs.names:
                variables_to_bin.append(container[bin_name])

            new_array = lookup_indices(sample=variables_to_bin,
                                       binning=self.output_specs)

            new_array = new_array.get('host')
            np.copyto(src=new_array, dst=container["bin_indices"].get('host'))


            for bin_i in range(self.output_specs.tot_num_bins):
                container.add_array_data(key='bin_{}_mask'.format(bin_i), 
                                         data=(new_array == bin_i))



