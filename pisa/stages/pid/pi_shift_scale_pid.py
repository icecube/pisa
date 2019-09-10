"""
The purpose of this stage is shift and/or scale the pid values.
"""

from __future__ import absolute_import, print_function, division

from numba import guvectorize
import numpy as np

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils import vectorizer
from pisa.utils.numba_tools import WHERE

__all__ = ['pi_shift_scale_pid']

__author__ = 'L. Fischer'


class pi_shift_scale_pid(PiStage):
    """
    Shift/scale pid.

    Parameters
    ----------
    data
    params
        bias : float
            shift pid values by given bias
        scale : float
            scale pid values by given scale factor
    input_names
    output_names
    debug_mode
    input_specs
    calc_specs
    output_specs

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

        # register expected parameters
        expected_params = ('bias', 'scale',)

        input_names = ()
        output_names = ()

        input_apply_keys = ('pid',)

        output_calc_keys = ('calculated_pid',)

        output_apply_keys = ('pid',)

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
            output_calc_keys=output_calc_keys,
        )

        assert self.input_mode is not None
        assert self.calc_mode == 'events'
        assert self.output_mode is not None

    def setup_function(self):
        """Setup the stage"""

        # set the correct data mode
        self.data.data_specs = self.calc_specs
        for container in self.data:
            container['calculated_pid'] = np.empty((container.size), dtype=FTYPE)
            container['original_pid'] = np.empty((container.size), dtype=FTYPE)
            vectorizer.set(container['pid'], out=container['original_pid'])

    def compute_function(self):
        """Perform computation"""

        # bias/scale have no units.
        bias = self.params.bias.m_as('dimensionless')
        scale = self.params.scale.m_as('dimensionless')

        for container in self.data:
            calculate_pid_function(bias,
                                   scale,
                                   container['original_pid'].get(WHERE),
                                   out=container['calculated_pid'].get(WHERE))
            container['calculated_pid'].mark_changed(WHERE)

    def apply_function(self):
        for container in self.data:
            # set the pid value to the calculated one
            vectorizer.set(container['calculated_pid'], out=container['pid'])

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
