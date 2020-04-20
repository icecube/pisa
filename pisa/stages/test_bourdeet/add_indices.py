#/usr/bin/env python


#
#  PISA module to prep incoming data into formats that are
#  compatible with the mc_uncertainty likelihood formulation
#  
# This module takes in events containers from the pipeline, and 
# introduces an additional array giving the indices where each 
# event falls into. 
#
# Etienne bourbeau (etienne.bourbeau@icecube.wisc.edu)
# 
# module structure imported form bootcamp example





from __future__ import absolute_import, print_function, division

import math

from numba import guvectorize
import numpy as np

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging


# Load the modified index lookup function
from pisa.core.bin_indexing import lookup_indices



class add_indices(PiStage):
    """
    PISA Pi stage to append an array 

    Parameters
    ----------
    data
    params
        foo : Quantity
        bar : Quanitiy with time dimension
    input_names
    output_names
    debug_mode
    input_specs
    calc_specs
    output_specs

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

        # here we register our expected parameters foo and bar so that PISA knows what to expect
        expected_params = ()

        # any in-/output names could be specified here, but we won't need that for now
        input_names = ()
        output_names = ()

        # register any variables that are used as inputs or new variables generated
        # (this may seem a bit abstract right now, but hopefully will become more clear later)

        # what are the keys used from the inputs during apply
        input_apply_keys = ()
        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ('bin_indices',)
        # what keys are added or altered for the outputs during apply
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
        
        assert self.calc_specs=='events','ERROR: calc specs must be set to "events for this module'


        self.data.data_specs = 'events'
        for container in self.data:
            # Generate a new container called bin_indices
            container['bin_indices'] = np.empty((container.size), dtype=FTYPE)
  
            E = container['reco_energy']
            C = container['reco_coszen']
            P = container['pid']

            new_array = lookup_indices(sample=[E,C,P],binning=self.output_specs)
            new_array = new_array.get('host')
            np.copyto(src=new_array, dst=container["bin_indices"].get('host'))


            for bin_i in range(self.output_specs.tot_num_bins):
                container.add_array_data(key='bin_{}_mask'.format(bin_i), data=new_array==bin_i)
