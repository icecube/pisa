"""
PISA pi stage to apply effective area weights
"""

from __future__ import absolute_import, print_function, division

from pisa.core.stage import Stage
from pisa.utils import vectorizer
from pisa.utils.profiler import profile


class aeff(Stage):  # pylint: disable=invalid-name
    """
    PISA Pi stage to apply aeff weights.

    This combines the detector effective area with the flux weights calculated
    in an earlier stage to compute the weights.

    Various scalings can be applied for particular event classes. The weight is
    then multiplied by the livetime to get an event count.

    Parameters
    ----------
    params
        Expected params are .. ::

            livetime : Quantity with time units
            aeff_scale : dimensionless Quantity
            nutau_cc_norm : dimensionless Quantity
            nutau_norm : dimensionless Quantity
            nu_nc_norm : dimensionless Quantity

    """
    def __init__(
        self,
        **std_kwargs,
    ):
        expected_params = (
            'livetime',
            'aeff_scale',
            'nutau_cc_norm',
            'nutau_norm',
            'nu_nc_norm',
        )

        # init base class
        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )


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

            container['weights'] *= container['weighted_aeff'] * scale
            container.mark_changed('weights')
