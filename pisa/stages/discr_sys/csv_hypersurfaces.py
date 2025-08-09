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
from pisa.utils.resources import find_resource

__all__ = ["csv_hypersurfaces",]

__author__ = "B. Benkel, J. Weldert"

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

    propagate_uncertainty : bool, optional
        Propagate the uncertainties from the hypersurface to the uncertainty of
        the output
    """
    def __init__(
        self,
        fit_results_file,
        nominal_systematics,
        inter_param,
        links=None,
        propagate_uncertainty=False,
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
        self.propagate_uncertainty = propagate_uncertainty

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
            self.hs[k] = pd.read_csv(find_resource(f))

        if self.links is not None:
            for key, val in self.links.items():
                self.data.link_containers(key, val)

        # create containers for scale factors
        for container in self.data:
            container["hs_scales"] = np.empty(container.size, dtype=FTYPE)
            if self.propagate_uncertainty:
                hs = self.hs[container.name]

                # get uncertainty at nominal value of interpolation param
                inter_param_value = self.params[self.inter_param].m
                start_idx = np.argmin(np.abs(hs[self.inter_param] - inter_param_value))
                _, c = np.unique(hs[self.inter_param], return_counts=True)
                stop_idx = start_idx + c[0]

                hs_scales_uncertainty = hs['intercept_sigma'][start_idx:stop_idx]
                container["hs_scales_uncertainty"] = np.array(hs_scales_uncertainty).reshape(container.size)

        # Check map names match between data container and hypersurfaces
        for container in self.data:
            assert container.name in self.hs, f"No match for map {container.name} found in the hypersurfaces"

        self.data.unlink_containers()
        
    def get_corr_factors(self, hs, param_values):
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
                
            # Get inter_param bins
            inter_param_bins = hs[self.inter_param].unique()

            # Get hypersurface below and above.
            lower_val = inter_param_bins[inter_param_bins <= inter_param_value].max()
            upper_val = inter_param_bins[inter_param_bins > inter_param_value].min()

            hs_lower = hs.loc[hs[self.inter_param] == lower_val].reset_index()
            hs_upper = hs.loc[hs[self.inter_param] == upper_val].reset_index()

            # Create the interpolated hypersurface.
            hs_interpolated = hs_lower.copy()
            for p in ['intercept']+list(param_values.keys()):
                binlen = hs_upper[self.inter_param][0] - hs_lower[self.inter_param][0]
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

                elif self.propagate_uncertainty:
                    container["errors"] = container["weights"] * container["hs_scales_uncertainty"]
                else:
                    container["errors"] *= container["hs_scales"]
                container.mark_changed('errors')

                if "bin_unc2" in container.keys:
                    container["bin_unc2"] = np.clip(container["bin_unc2"] * container["hs_scales"], a_min=0, a_max=np.inf)
                    container.mark_changed("bin_unc2")

            # Update weights according to hypersurfaces
            container["weights"] = np.clip(container["weights"] * container["hs_scales"], a_min=0, a_max=np.inf)


def init_test(**param_kwargs):
    """Instantiation example"""
    param_set = ParamSet([
        Param(name='dom_eff', value=1.0, **param_kwargs),
        Param(name='hole_ice_p0', value=0.1, **param_kwargs),
        Param(name='hole_ice_p1', value=-0.05, **param_kwargs),
        Param(name='bulk_ice_scatter', value=1.05, **param_kwargs),
        Param(name='bulk_ice_abs', value=1.0, **param_kwargs),
        Param(name='dm31', value=3e-3*ureg.eV**2, **param_kwargs),
    ])
    nominal_systematics = {"dom_eff"          :  1.00,
                           "hole_ice_p0"      :  0.10,
                           "hole_ice_p1"      : -0.05,
                           "bulk_ice_abs"     :  1.00,
                           "bulk_ice_scatter" :  1.00}
    dd_en = OneDimBinning(
        'reco_energy', num_bins=10, is_log=True,
        bin_edges=[6.31, 8.46, 11.34, 15.20, 20.38, 27.31, 36.61, 49.08, 65.79, 88.20, 158.49] * ureg.GeV,
        tex=r'E_{\rm reco}'
    )
    dd_cz = OneDimBinning(
        'reco_coszen', num_bins=10, is_lin=True, domain=[-1, 0.1], tex=r'\cos{\theta}_{\rm reco}'
    )
    dd_pid = OneDimBinning('pid', bin_edges=[0.55, 0.75, 1.], tex=r'{\rm PID}')

    return csv_hypersurfaces(
        fit_results_file='events/hs_test.csv',
        nominal_systematics=nominal_systematics,
        inter_param='dm31',
        links={'test':['test1_cc', 'test2_nc']},
        params=param_set,
        calc_mode=MultiDimBinning([dd_en, dd_cz, dd_pid], name='oscNext_verification'),
    )
