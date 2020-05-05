"""
Stage to take the initial errors of MC and keep them
for all minimization.

Needed to allow mod_chi2 to behave correctly
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.pi_stage import PiStage
from pisa.utils.profiler import profile
from pisa.utils import vectorizer
from pisa.utils.numba_tools import WHERE

class pi_fix_error(PiStage):  # pylint: disable=invalid-name
    """
    stage to fix the error returned by template_maker.
    """
    def __init__(
        self,
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        error_method=None,
        input_specs=None,
        calc_specs=None,
        output_specs=None,
    ):

        expected_params = ()
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_apply_keys = ("weights", "errors")
        output_calc_keys = ('frozen_errors',)
        output_apply_keys = ("weights", 'errors')
        
        # init base class
        super().__init__(
            data=data,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            error_method=error_method,
            input_specs=input_specs,
            calc_specs=calc_specs,
            output_specs=output_specs,
            output_calc_keys=output_calc_keys,
            input_apply_keys=input_apply_keys,
            output_apply_keys=output_apply_keys,
        )

        assert self.input_mode == "binned"
        assert self.calc_mode == "binned"
        assert self.output_mode == 'binned'

    def setup_function(self):
        self.data.data_specs = self.input_specs
        for container in self.data:
            container['frozen_errors'] = np.empty((container.size), dtype=FTYPE)

    def apply_function(self):
        for container in self.data:
            vectorizer.assign(vals=container['frozen_errors'], out=container['errors'])
            container['errors'].mark_changed(WHERE)
 
    def compute_function(self):
        for container in self.data:
            vectorizer.assign(vals=container["errors"], out=container["frozen_errors"])
            container['frozen_errors'].mark_changed(WHERE)
