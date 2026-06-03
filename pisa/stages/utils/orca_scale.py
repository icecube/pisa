"""
PISA pi stage to apply effective area weights
"""

from __future__ import absolute_import, print_function, division

from pisa import ureg
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.profiler import profile


class orca_scale(Stage):  # pylint: disable=invalid-name

    def __init__(
        self,
        **std_kwargs,
    ):
        expected_params = (
            'hpt_norm',
            'shower_norm',
        )

        expected_container_keys = (
            'weights',
        )

        # init base class
        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )


    @profile
    def apply_function(self):
        hpt_norm = self.params.hpt_norm.m_as('dimensionless')
        shower_norm = self.params.shower_norm.m_as('dimensionless')

        for container in self.data:
            hist = container.get_hist('weights')[0]
            hist[:,:,0] *= hpt_norm
            hist[:,:,1] *= shower_norm
            container.mark_changed('weights')
