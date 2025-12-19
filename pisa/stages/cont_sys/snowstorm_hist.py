"""
PISA stage to apply detector systematics
"""

import numpy as np

from pisa.core.binning import MultiDimBinning
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.core.translation import histogram
from pisa.utils.profiler import profile

__all__ = [
    "snowstorm_hist",
    "init_test"
]

__author__ = "J. Weldert"

__license__ = """Copyright (c) 2014-2025, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


class snowstorm_hist(Stage):  # pylint: disable=invalid-name
    """
    Service to apply detector systematics through snowstorm hists.

    Parameters
    ----------
    systematics : list of str
        List of the systematic parameters.
    simulation_dists : list of str
        The distribution of the systematic parameter in the snowstorm simulation.
        Has to be either 'gauss' or 'uniform'.
    simulation_dists_params : list of tuples of floats
        Parameters of the simulation distributions. (mean, std) for 'gauss' and
        (min, max) for 'uniform'.
    additional_params : list of str
        Parameters that are no detector systematics but if changed require a
        re-calculation of the gradients (e.g. osc params).
    params : ParamSet
        Note that the params required to be in `params` are determined from
        those listed in the `systematics`.
    """

    def __init__(
        self,
        systematics,
        simulation_dists,
        simulation_dists_params,
        additional_params=[],
        **std_kwargs,
    ):
        # evaluation only works on event-by-event basis
        supported_reps = {
            'calc_mode': ['events'],
            'apply_mode': [MultiDimBinning],
        }

        # Store args
        if isinstance(systematics, str):
            self.systematics = eval(systematics)
        else:
            self.systematics = systematics
        assert isinstance(self.systematics, list)

        if isinstance(simulation_dists, str):
            self.simulation_dists = eval(simulation_dists)
        else:
            self.simulation_dists = simulation_dists
        assert isinstance(self.simulation_dists, list)
        assert len(self.simulation_dists) == len(self.systematics)
        for sd in self.simulation_dists:
            assert sd in ["gauss", "uniform"]

        if isinstance(simulation_dists_params, str):
            self.simulation_dists_params = eval(simulation_dists_params)
        else:
            self.simulation_dists = simulation_dists_params
        assert isinstance(self.simulation_dists_params, list)
        assert len(self.simulation_dists_params) == len(self.systematics)

        if isinstance(additional_params, str):
            self.additional_params = eval(additional_params)
        else:
            self.additional_params = additional_params
        assert isinstance(self.additional_params, list)

        # -- Initialize base class -- #
        super().__init__(
            expected_params=self.systematics+self.additional_params,
            expected_container_keys=["weights"]+self.systematics,
            supported_reps=supported_reps,
            **std_kwargs,
        )

    def setup_function(self):
        central_values = []
        for i, sd in enumerate(self.simulation_dists):
            if sd == "gauss":
                central_values.append(self.simulation_dists_params[i][0])
            else: # uniform
                central_values.append(sum(self.simulation_dists_params[i])/2)
        self.central_values = central_values

    @profile
    def compute_function(self):
        for container in self.data:
            container.representation = self.calc_mode
            sample = np.array([container[name] for name in self.apply_mode.names])
            syst = [container[sys] for sys in self.systematics]
            weights = container["weights"]

            container.representation = self.apply_mode
            container["syst_scale"] = np.ones(self.apply_mode.shape)
            for i, sys in enumerate(self.systematics):
                h1 = histogram(
                    list(sample[:, syst[i] > self.central_values[i]]),
                    weights[syst[i] > self.central_values[i]],
                    self.apply_mode,
                    averaged=False
                )
                h2 = histogram(
                    list(sample[:, syst[i] < self.central_values[i]]),
                    weights[syst[i] < self.central_values[i]],
                    self.apply_mode,
                    averaged=False
                )
                if self.simulation_dists[i] == "gauss": # TODO verify
                    correction_factor = 1/self.simulation_dists_params[i][1] * np.sqrt(np.pi/2)
                    grad = np.nan_to_num((h1 - h2) / (h1 + h2) * correction_factor * 2)
                else:
                    diff = (self.simulation_dists_params[i][1] - self.simulation_dists_params[i][0]) / 2
                    grad = np.nan_to_num((h1 - h2) / (h1 + h2) / diff * 2)
                container["syst_scale"] *= 1 + (self.params[sys].m-self.central_values[i])*grad

    def apply_function(self):
        self.data.representation = self.apply_mode
        for container in self.data:
            container["weights"] *= container["syst_scale"]
