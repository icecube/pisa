"""
PISA pi stage to apply hypersurface fits from discrete systematics parameterizations
"""


from __future__ import absolute_import, print_function, division

import ast
from collections.abc import Mapping
import numpy as np
import pandas as pd

from pisa import FTYPE, ureg
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.format import split

__all__ = ["hypersurfaces",]

__author__ = "P. Eller, T. Ehrhardt, T. Stuttard, J.L. Lanfranchi, A. Trettin"

__license__ = """Copyright (c) 2014-2018, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


class csv_hypersurfaces(Stage):
    """
    Service to apply hypersurface parameterisation produced by
    `scripts.fit_discrete_sys_nd`

    Parameters
    ----------
    fit_results_file : str
        Path to hypersurface fit results file(s)

    links : dict
        A dictionary defining how containers should be linked. Keys are the names of
        the merged containers, values are lists of containers being linked together.
        Keys must be a sub-set of the loaded hypersurfaces.

    nominal_systematics : dict
        Systematics and their nominal values
        
    inter_param : str
        Parameter used for interpolation

    """
    def __init__(
        self,
        fit_results_file,
        nominal_systematics,
        inter_param,
        links=None,
        **std_kwargs,
    ):
        # -- Only allowed/implemented modes -- #
        assert isinstance(std_kwargs['calc_mode'], MultiDimBinning)
        # -- Load hypersurfaces -- #

        # Store args
        self.fit_results_file = split(fit_results_file)
        
        if isinstance(nominal_systematics, str):
            self.nominal_systematics = eval(nominal_systematics)
        elif isinstance(nominal_systematics, dict):
            self.nominal_systematics = nominal_systematics
        else:
            raise ValueError(
                f"Unsupported type {type(nominal_systematics)} for nominal_systematics."
            )
        
        self.inter_param = inter_param

        expected_container_keys = [
            'weights',
        ]
        if 'error_method' in std_kwargs:
            if std_kwargs['error_method']:
                expected_container_keys.append('errors')

        # -- Initialize base class -- #
        super().__init__(
            expected_params=list(self.nominal_systematics.keys()) + [self.inter_param],
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )
        
        if links is None:
            self.links = {}
        elif not isinstance(links, Mapping):
            self.links = ast.literal_eval(links)
        else:
            self.links = links

    # pylint: disable=line-too-long
    def setup_function(self):
        """Load the fit results from the file and make some check compatibility"""

        self.data.representation = self.calc_mode
        
        self.hs = {}
        for f in self.fit_results_file:
            k = f.split('/')[-1].split('.')[0]
            if k.startswith('hs_'):
                k = k[3:]
            self.hs[k] = pd.read_csv(f)

        if self.links is not None:
            for key, val in self.links.items():
                self.data.link_containers(key, val)

        # create containers for scale factors
        for container in self.data:
            container["hs_scales"] = np.empty(container.size, dtype=FTYPE)

        # Check map names match between data container and hypersurfaces
        for container in self.data:
            assert container.name in self.hs, f"No match for map {container.name} found in the hypersurfaces"

        self.data.unlink_containers()
        
    def get_corr_factors(self, hs: pd.DataFrame, param_values: dict[str, float]):
        """Get hypersurface correction factors."""
        # Get difference between param_values and nominal.
        diff = {k: v - self.nominal_systematics[k] for k, v in param_values.items()}

        # Return correction factor.
        return hs["intercept"] + sum([hs[k] * v for k, v in diff.items()])

    # the linter thinks that "logging" refers to Python's built-in
    # pylint: disable=line-too-long, logging-not-lazy, deprecated-method
    def compute_function(self):

        self.data.representation = self.calc_mode

        # Link containers
        if self.links is not None:
            for key, val in self.links.items():
                self.data.link_containers(key, val)

        # Format the params dict that will be passed to get_corr_factors
        param_values = {sys_param_name: self.params[sys_param_name].m
                        for sys_param_name in self.nominal_systematics}
        inter_param_value = self.params[self.inter_param].m

        # Loop over types
        for container in self.data:

            # Get the hypersurfaces
            hs = self.hs[container.name]
            if inter_param_value < hs[self.inter_param].min() or hs[self.inter_param].max() < inter_param_value:
                raise ValueError("%s of %f is outside of interpolation range."%(self.inter_param, inter_param_value))
                
            # Get inter_param bin length.
            inter_param_bins = hs[self.inter_param].unique()
            binlen = inter_param_bins[1] - inter_param_bins[0]

            # Get hypersurface below and above dm31.
            hs_lower = hs.loc[(inter_param_value - binlen < hs[self.inter_param]) & (hs[self.inter_param] < inter_param_value)].reset_index()
            hs_upper = hs.loc[(inter_param_value < hs[self.inter_param]) & (hs[self.inter_param] < inter_param_value + binlen)].reset_index()

            # Create the interpolated hypersurface.
            hs_interpolated = hs_lower.copy()
            for p in param_values.keys():
                grad = (np.array(hs_upper[p]) - np.array(hs_lower[p])) / binlen
                hs_interpolated[p] = grad * (inter_param_value - hs_lower[self.inter_param][0]) + hs_lower[p]

            # Compute the correction factors.
            scales = self.get_corr_factors(hs_interpolated, param_values)
            scales = np.array(scales).reshape(container.size)

            # Where there are no scales (e.g. empty bins), set scale factor to 1
            empty_bins_mask = ~np.isfinite(scales)
            num_empty_bins = np.sum(empty_bins_mask)
            if num_empty_bins > 0. and not self.warning_issued:
                logging.warn("%i empty bins found in hypersurface" % num_empty_bins)
                self.warning_issued = True
            scales[empty_bins_mask] = 1.

            # Add to container
            np.copyto(src=scales, dst=container["hs_scales"])
            container.mark_changed("hs_scales")

        # Unlink the containers again
        self.data.unlink_containers()

    def apply_function(self):

        for container in self.data:
            # update uncertainty first, before the weights are changed. This step is skipped in event mode
            if self.error_method == "sumw2":

                # If computing uncertainties in events mode, warn that
                # hs error propagation will be skipped
                if self.data.representation=='events':
                    logging.trace('WARNING: running stage in events mode. Hypersurface error propagation will be IGNORED.')

                container["errors"] *= container["hs_scales"]
                container.mark_changed('errors')

                if "bin_unc2" in container.keys:
                    container["bin_unc2"] = np.clip(container["bin_unc2"] * container["hs_scales"], a_min=0, a_max=np.inf)
                    container.mark_changed("bin_unc2")

            # Update weights according to hypersurfaces
            container["weights"] = np.clip(container["weights"] * container["hs_scales"], a_min=0, a_max=np.inf)

'''
def init_test(**param_kwargs):
    """Instantiation example"""
    param_set = ParamSet([
        Param(name='opt_eff_overall', value=1.0, **param_kwargs),
        Param(name='opt_eff_lateral', value=25, **param_kwargs),
        Param(name='opt_eff_headon', value=0.0, **param_kwargs),
        Param(name='ice_scattering', value=0.0, **param_kwargs),
        Param(name='ice_absorption', value=0.0, **param_kwargs),
    ])
    dd_en = OneDimBinning(
        'reco_energy', num_bins=8, is_log=True,
        bin_edges=[5.62341325, 7.49894209, 10.0, 13.33521432, 17.7827941, 23.71373706, 31.6227766, 42.16965034, 56.23413252] * ureg.GeV,
        tex=r'E_{\rm reco}'
    )
    dd_cz = OneDimBinning(
        'reco_coszen', num_bins=8, is_lin=True, domain=[-1,1], tex=r'\cos{\theta}_{\rm reco}'
    )
    dd_pid = OneDimBinning('pid', bin_edges=[-0.5, 0.5, 1.5], tex=r'{\rm PID}')
    return hypersurfaces(
        params=param_set,
        fit_results_file='events/IceCube_3y_oscillations/hyperplanes_*.csv.bz2',
        error_method='sumw2',
        calc_mode=MultiDimBinning([dd_en, dd_cz, dd_pid], name='dragon_datarelease'),
        links={
            'nue_cc+nuebar_cc':['test1_cc', 'test2_nc'],
            #TODO: not ideal, because I need to know what test_services fills ContainerSet with
        }
    )
'''
