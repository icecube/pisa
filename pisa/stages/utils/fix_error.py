"""
Stage to take the initial errors of MC and keep them
for all minimization.

Needed for the DRAGON nutau appearance analysis.
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.stage import Stage


class fix_error(Stage):  # pylint: disable=invalid-name
    """
    stage to fix the error returned by template_maker.
    """
    def __init__(
        self,
        **std_kwargs,
    ):

        expected_container_keys = (
            'errors',
        )

        # init base class
        super().__init__(
            expected_params=(),
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )

    def setup_function(self):
        for container in self.data:
            container['frozen_errors'] = np.empty((container.size), dtype=FTYPE)

    def compute_function(self):
        for container in self.data:
            container["frozen_errors"][:] = container["errors"]
            container.mark_changed('frozen_errors')

    def apply_function(self):
        for container in self.data:
            container['errors'][:] = container['frozen_errors']
            container.mark_changed('errors')
