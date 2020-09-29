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
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        calc_mode=None,
        apply_mode=None,
    ):
        expected_params = ('livetime', 'weight_scale')
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply

        # what keys are added or altered for the outputs during apply

        # init base class
        super().__init__(
            data=data,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            calc_mode=calc_mode,
            apply_mode=apply_mode,
        )

        assert self.calc_mode is None

        # right now this stage has no calc mode, as it just applies scales
        # but it could if for example some smoothing will be performed!

    @profile
    def apply_function(self):
        weight_scale = self.params.weight_scale.m_as('dimensionless')
        livetime_s = self.params.livetime.m_as('sec')
        scale = weight_scale * livetime_s

        for container in self.data:
            vectorizer.scale(
                vals=container['weights'],
                scale=scale,
                out=container['weights'],
            )
