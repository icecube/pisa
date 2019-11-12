"""
PISA pi stage to apply effective area weights
"""
from __future__ import absolute_import, print_function, division

from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils import vectorizer


class pi_aeff(PiStage):
    """
    PISA Pi stage to apply aeff weights.
    This combines the detector effective area with the flux weights calculated 
    in an earlier stage to compute the weights.
    Various scalings can be applied for particular event classes.
    The weight is then multiplied by the livetime to get an event count.

    Paramaters
    ----------

    livetime : Quantity with time units

    aeff_scale : dimensionless Quantity

    nutau_cc_norm : dimensionless Quantity

    nutau_norm : dimensionless Quantity

    nu_nc_norm : dimensionless Quantity

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
                           'nutau_norm',
                           'nu_nc_norm',
                          )
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_apply_keys = ('weighted_aeff',
                           )
        # what keys are added or altered for the outputs during apply
        output_apply_keys = ('weights',
                            )

        # init base class
        super().__init__(
            data=data,
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
        )

        assert self.input_mode is not None
        assert self.calc_mode is None
        assert self.output_mode is not None

        # right now this stage has no calc mode, as it just applies scales
        # but it could if for example some smoothing will be performed!


    @profile
    def apply_function(self):

        # read out
        aeff_scale = self.params.aeff_scale.m_as('dimensionless')
        livetime_s = self.params.livetime.m_as('sec')
        nutau_cc_norm = self.params.nutau_cc_norm.m_as('dimensionless')
        nutau_norm = self.params.nutau_norm.m_as('dimensionless')
        nu_nc_norm = self.params.nu_nc_norm.m_as('dimensionless')

        for container in self.data:
            scale = aeff_scale * livetime_s
            if container.name in ['nutau_cc', 'nutaubar_cc']:
                scale *= nutau_cc_norm
            if 'nutau' in container.name:
                scale *= nutau_norm
            if 'nc' in container.name:
                scale *= nu_nc_norm
            vectorizer.multiply_and_scale(scale,
                                          container['weighted_aeff'],
                                          out=container['weights'],
                                         )
