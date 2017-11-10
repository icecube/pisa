import numpy as np
from numba import guvectorize, SmartArray

from pisa import *
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet


class pi_aeff(PiStage):
    """
    stage to histogram events

    Paramaters
    ----------

    livetime
    aeff_scale
    nutau_cc_norm

    Notes
    -----

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
                 ):

        expected_params = ('livetime',
                           'aeff_scale',
                           'nutau_cc_norm',
                           )
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_keys = ('weights',
                      )
        # what are keys added or altered in the calculation used during apply 
        calc_keys = ()
        # what keys are added or altered for the outputs during apply
        output_keys = ('weights',
                       )

        # init base class
        super(pi_aeff, self).__init__(data=data,
                                       params=params,
                                       expected_params=expected_params,
                                       input_names=input_names,
                                       output_names=output_names,
                                       debug_mode=debug_mode,
                                       input_specs=input_specs,
                                       calc_specs=calc_specs,
                                       output_specs=output_specs,
                                       input_keys=input_keys,
                                       calc_keys=calc_keys,
                                       output_keys=output_keys,
                                       )

        assert self.input_mode is not None
        assert self.calc_mode is None
        assert self.output_mode is not None

        # right now this stage has no calc mode, as it just applies scales
        # but it could if for example some smoothing will be performed!


    def apply_function(self):

        # read out 
        aeff_scale = self.params.aeff_scale.m_as('dimensionless')
        livetime_s = self.params.livetime.m_as('sec')
        nutau_cc_norm = self.params.nutau_cc_norm.m_as('dimensionless')

        for container in self.data:
            w = container['weights'].get('host')
            aeff = container['weighted_aeff'].get('host')
            factor = aeff_scale * livetime_s
            if container.name in ['nutau_cc', 'nutaubar_cc']:
                factor *= nutau_cc_norm
            w *= aeff * factor
            container['weights'].mark_changed('host')
