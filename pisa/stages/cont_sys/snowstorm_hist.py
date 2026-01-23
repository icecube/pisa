"""
PISA stage to apply detector systematics
"""

import numpy as np

from pisa import ureg
from pisa.core.binning import MultiDimBinning
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.core.translation import histogram

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
            self.simulation_dists_params = simulation_dists_params
        assert isinstance(self.simulation_dists_params, list)
        assert len(self.simulation_dists_params) == len(self.systematics)

        if isinstance(additional_params, str):
            self.additional_params = eval(additional_params)
        else:
            self.additional_params = additional_params
        assert isinstance(self.additional_params, list)

        self.n_split = 2
        self.grads = {}

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

        for container in self.data:
            self.grads[container.name] = {}
        self.additional_params_values = None

    def compute_function(self):
        additional_params_values = [self.params[p].m for p in self.additional_params]
        if additional_params_values != self.additional_params_values:
            calc_grads = True
            self.additional_params_values = additional_params_values
        elif self.apply_mode.shape != self.grads[self.data.names[0]][self.systematics[0]].shape:
            calc_grads = True
        else:
            calc_grads = False

        for container in self.data:
            if calc_grads:
                container.representation = self.calc_mode
                sample = np.array([container[name] for name in self.apply_mode.names])
                syst = [container[sys] for sys in self.systematics]
                weights = container["weights"]

            container.representation = self.apply_mode
            container["syst_scale"] = np.ones(self.apply_mode.shape)
            for i, sys in enumerate(self.systematics):
                if calc_grads:
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
                        self.grads[container.name][sys] = np.nan_to_num((h1-h2) / (h1+h2) * correction_factor * self.n_split)
                    else:
                        diff = (self.simulation_dists_params[i][1] - self.simulation_dists_params[i][0]) / 2
                        self.grads[container.name][sys] = np.nan_to_num((h1-h2) / (h1+h2) / diff * self.n_split)

                container["syst_scale"] *= 1 + (self.params[sys].m-self.central_values[i]) * self.grads[container.name][sys]
            container["syst_scale"] = np.clip(container["syst_scale"], a_min=0, a_max=np.inf)

    def apply_function(self):
        self.data.representation = self.apply_mode
        for container in self.data:
            container["weights"] *= container["syst_scale"]


def init_test(**param_kwargs):
    """Instantiation example"""
    param_set = ParamSet([
        Param(name='dom_eff', value=1.0, **param_kwargs),
        Param(name='deltam31', value=3e-3*ureg.eV**2, **param_kwargs),
    ])
    return snowstorm_hist(
        systematics=['dom_eff'],
        simulation_dists=['gauss'],
        simulation_dists_params=[(1.0, 0.1)],
        additional_params=['deltam31'],
        params=param_set,
    )
