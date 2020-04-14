"""
Stage to fluctuate MC events in a pipeline.

Each event is interpreted as a unique observation and its weight multiplied by a Poisson
random variable that has an expectation value of one. This allows the simulation of MC
uncertainties that is closest to the 'true' distribution as one can get. It also has the
advantage that the rest of the pipeline can run unaffected.
"""

import numpy as np

from pisa import FTYPE
from pisa.core.pi_stage import PiStage
from pisa.utils.numba_tools import WHERE
from pisa.utils import vectorizer

__author__ = "A. Trettin"

__license__ = """Copyright (c) 2020, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


class pi_fluctuation(PiStage):
    """
    MC fluctuation PISA PiStage

    Parameters
    ----------
    seed (int): Seed for the random number generator. Default is ``None``.
    """

    def __init__(
        self,
        seed=None,
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        input_specs=None,
        calc_specs=None,
        output_specs=None,
    ):

        expected_params = ()
        input_names = ()
        output_names = ()

        input_apply_keys = ("weights",)

        # The weights are simply scaled by a Poisson random number
        output_calc_keys = ("poisson_weights",)
        output_apply_keys = ("weights",)

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
            output_calc_keys=output_calc_keys,
            output_apply_keys=output_apply_keys,
        )

        if seed is None:
            self.seed = None
        else:
            self.seed = int(seed)

        assert self.input_mode is not None
        assert self.calc_mode == "events"
        assert self.output_mode is not None

    def setup_function(self):
        rng = np.random.default_rng(seed=self.seed)
        self.data.data_specs = self.calc_specs
        for container in self.data:
            container["poisson_weights"] = rng.poisson(size=container.size)
            container["poisson_weights"].mark_changed(WHERE)

    def apply_function(self):
        for container in self.data:
            vectorizer.imul(vals=container["poisson_weights"], out=container["weights"])
