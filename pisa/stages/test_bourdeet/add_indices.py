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
from pisa.utils import vectorizer
from pisa.utils.log import logging
from pisa.utils.numba_tools import WHERE


# Load the modified index lookup function
from analysis.sandbox.bourdeet.translation_indices import lookup_indices



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
        input_apply_keys = ('weights',)
        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ()
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
        """Setup the stage"""
        # in case we need to initialize sth, like reading in an external file,
        # or add variables to the data object that we can later populate
        
        # do that in the right representation
        self.data.data_specs = 'events'#self.calc_specs

        #for container in self.data:
            # also notice that PISA uses strict typing for arrays
            #container['bin_indices'] = np.empty((container.size), dtype=FTYPE)

    # def compute_function(self):
    #     """Perform computation"""
    #     # this function is called when parameters of this stage are changed (and the first time the
    #     # pipeline is run). Otherwise it is skipped. We will compute our nonsense scale factors here
        
    #     # get our paramater in the desired dimensions
    #     analysis_binning = self.params.analysis_binning
        
    #     for container in self.data:
    #         # the `.get(WHERE)` statements are necessary for numba to know if these arrays should be read from the host (CPU)
    #         # or the device (GPU).
    #         # No worries, this will work without a GPU too
    #         new_array = lookup(sample=[container['reco_energy'],container['reco_coszen']],
    #                          flat_hist= container.binned_data['weights'][-1],
    #                          binning=self.params.binning,
    #                          )

    #         container.add('bin_indices',new_array.get(WHERE).astype(np.int32))

    def apply_function(self):
        # this function is called everytime the pipeline is run, so here we can just apply our factors
        # that we calculated before to the event weights
        #print(self.data.data_specs)

        for container in self.data:
            # to apply we want to multiply the evenet weights by the factors we computed before
            # we can either implement another vectorized function, or just use one that is already available

            #print(container)
            #print(container.array_data.keys())
            #print(container.binned_data.keys())
            #print(dir(container))

            new_array = lookup_indices(sample=[container['reco_energy'],container['reco_coszen'],container['pid']],
                               binning=self.calc_specs,
                           )

            container.add_array_data('bin_indices',new_array.get(WHERE).astype(np.int32))