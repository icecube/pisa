"""
PISA pi stage to apply weights
"""

from __future__ import absolute_import, print_function, division

from pisa.core.stage import Stage
from pisa.utils import vectorizer
from pisa.utils.profiler import profile


class weight(Stage):  # pylint: disable=invalid-name
    """
    PISA Pi stage to apply weights.
    This assumes a weight has already been calculated.
    The weight is then multiplied by the livetime to get an event count.

    Parameters
    ----------
    params : ParamSet or sequence with which to instantiate a ParamSet.
        Expected params are: .. ::

            livetime : Quantity [time]
                Detector livetime for scaling template
            weight_scale : Quantity [dimensionless]
                Overall scaling/normalisation of template

    """
    def __init__(
        self,
        **std_kwargs,
    ):
        expected_params = ('livetime', 'weight_scale')

        # init base class
        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )


    @profile
    def apply_function(self):
        weight_scale = self.params.weight_scale.m_as('dimensionless')
        livetime_s = self.params.livetime.m_as('sec')
        scale = weight_scale * livetime_s

        for container in self.data:
            container['weights'] *= scale
