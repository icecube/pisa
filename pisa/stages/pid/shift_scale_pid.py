"""
The purpose of this stage is shift and/or scale the pid values.
"""

from __future__ import absolute_import, print_function, division

from numba import guvectorize
import numpy as np

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils import vectorizer
from pisa.utils.numba_tools import WHERE

__all__ = ['shift_scale_pid']

__author__ = 'L. Fischer'


class shift_scale_pid(Stage):
    """
    Shift/scale pid.

    Parameters
    ----------
    params
        bias : float
            shift pid values by given bias
        scale : float
            scale pid values by given scale factor
    """

    def __init__(self,
                 **std_kwargs,
                 ):

        # register expected parameters
        expected_params = ('bias', 'scale',)

        # init base class
        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

        assert self.calc_mode == 'events'

    def setup_function(self):
        """Setup the stage"""

        # set the correct data mode
        self.data.representation = self.calc_mode
        for container in self.data:
            container['calculated_pid'] = np.empty((container.size), dtype=FTYPE)
            container['original_pid'] = np.empty((container.size), dtype=FTYPE)
            vectorizer.assign(vals=container['pid'], out=container['original_pid'])

    def compute_function(self):
        """Perform computation"""

        # bias/scale have no units.
        bias = self.params.bias.m_as('dimensionless')
        scale = self.params.scale.m_as('dimensionless')

        for container in self.data:
            calculate_pid_function(bias,
                                   scale,
                                   container['original_pid'],
                                   out=container['calculated_pid'])
            container.mark_changed('calculated_pid')

    def apply_function(self):
        for container in self.data:
            # set the pid value to the calculated one
            vectorizer.assign(vals=container['calculated_pid'], out=container['pid'])

signatures = [
    '(f4[:], f4[:], f4[:], f4[:])',
    '(f8[:], f8[:], f8[:], f8[:])'
]

layout = '(),(),()->()'


@guvectorize(signatures, layout, target=TARGET)
def calculate_pid_function(bias_value, scale_factor, pid, out):
    """This function selects a pid cut by shifting the pid variable so
    the default cut at 1.0 is at the desired cut position.

    Parameters
    ----------
    bias_value : scalar
        shift pid values by this bias
    scale_factor : scalar
        scale pid values with this factor
    pid : scalar
        pid variable
    out : scalar
        shifted pid values

    """

    out[0] = (scale_factor[0] * pid[0]) + bias_value[0]
