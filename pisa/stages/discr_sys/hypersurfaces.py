"""
PISA pi stage to apply hypersurface fits from discrete systematics parameterizations
"""

#TODO Currently have to defined the `links` value in the stage config to match what the `combine_regex` does in
#     the Hypersurface fitting. The `combine_regex` is stored in the hypersurface instance, so should make the
#     scontainer linking use this instead of having to manually specify links. To do this, should make a variant
#     of the container linking function that accepts a regex (using shared code with Map.py).

from __future__ import absolute_import, print_function, division

import ast
from collections.abc import Mapping
import numpy as np

from pisa import FTYPE, ureg
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.log import logging
import pisa.utils.hypersurface as hs

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


class hypersurfaces(Stage): # pylint: disable=invalid-name
    """
    Service to apply hypersurface parameterisation produced by
    `scripts.fit_discrete_sys_nd`

    Parameters
    ----------
    fit_results_file : str
        Path to hypersurface fit results file, i.e. the JSON file produced by the
        `pisa.scripts.fit_discrete_sys_nd.py` script

    propagate_uncertainty : bool, optional
        Propagate the uncertainties from the hypersurface to the uncertainty of
        the output

    params : ParamSet
        Note that the params required to be in `params` are determined from
        those listed in the `fit_results_file`.
    
    interpolated : bool
        If `True`, indicates that the hypersurfaces to be loaded are interpolated.
    
    links : dict
        A dictionary defining how containers should be linked. Keys are the names of
        the merged containers, values are lists of containers being linked together.
        Keys must be a sub-set of the loaded hypersurfaces.

    """
    def __init__(
        self,
        fit_results_file,
        propagate_uncertainty=False,
        interpolated=False,
        links=None,
        fluctuate=False,
        fluctuate_seed=12345,
        **std_kwargs,
    ):
        # -- Only allowed/implemented modes -- #
        assert isinstance(std_kwargs['calc_mode'], MultiDimBinning)
        # -- Load hypersurfaces -- #

        # Store args
        self.fit_results_file = fit_results_file
        self.propagate_uncertainty = propagate_uncertainty
        self.interpolated = interpolated
        # Expected parameter names depend on the hypersurface and, if applicable,
        # on the parameters in which the hypersurfaces are interpolated.
        # For this reason we need to load the hypersurfaces already in the init function
        self.inter_params = []
        if self.interpolated:
            self.hypersurfaces = hs.load_interpolated_hypersurfaces(self.fit_results_file, expected_binning=std_kwargs['calc_mode'])
            self.inter_params = list(self.hypersurfaces.values())[0].interpolation_param_names
        else:
            self.hypersurfaces = hs.load_hypersurfaces(self.fit_results_file, expected_binning=std_kwargs['calc_mode'])
        self.hypersurface_param_names = list(self.hypersurfaces.values())[0].param_names

        expected_container_keys = [
            'weights',
        ]
        if 'error_method' in std_kwargs:
            if std_kwargs['error_method']:
                expected_container_keys.append('errors')
                #TODO: When in a pipeline after utils.hist,
                # this will cause a warning about missing presence of 'errors'
                # because hist adds this key during apply


        # -- Initialize base class -- #
        super().__init__(
            expected_params=self.hypersurface_param_names + self.inter_params,
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )
        if links is None:
            self.links = {}
        elif not isinstance(links, Mapping):
            self.links = ast.literal_eval(links)
        else:
            self.links = links
        self.warning_issued = False # don't warn more than once about empty bins

        # Handle fluctuation option
        self.fluctuate = fluctuate
        if self.fluctuate :
            self.fluctuate_seed = fluctuate_seed
            logging.info(f"User has selected to fluctuate the hypersurface coefficients (seed is {self.fluctuate_seed})")
            assert self.fluctuate_seed is not None

    # pylint: disable=line-too-long
    def setup_function(self):
        """Load the fit results from the file and make some check compatibility"""

        self.data.representation = self.calc_mode

        if self.links is not None:
            for key, val in self.links.items():
                self.data.link_containers(key, val)

        # create containers for scale factors
        for container in self.data:
            container["hs_scales"] = np.empty(container.size, dtype=FTYPE)
            if self.propagate_uncertainty:
                container["hs_scales_uncertainty"] = np.empty(container.size, dtype=FTYPE)


        # Check map names match between data container and hypersurfaces
        for container in self.data:
            assert container.name in self.hypersurfaces, f"No match for map {container.name} found in the hypersurfaces"

        self.data.unlink_containers()

    # the linter thinks that "logging" refers to Python's built-in
    # pylint: disable=line-too-long, logging-not-lazy, deprecated-method
    def compute_function(self):

        self.data.representation = self.calc_mode

        # Link containers
        if self.links is not None:
            for key, val in self.links.items():
                self.data.link_containers(key, val)

        # Format the params dict that will be passed to `Hypersurface.evaluate`
        #TODO checks on param units
        param_values = {sys_param_name: self.params[sys_param_name].m
                        for sys_param_name in self.hypersurface_param_names}
        if self.interpolated:
            osc_params = {name: self.params[name] for name in self.inter_params}

        # If fluctuating, seed the flucutations
        # Needs to be consistent for each call to this functions
        if self.fluctuate :
            fluctuate_random_state = np.random.RandomState(self.fluctuate_seed)

        # Loop over types
        for container in self.data:

            # Get the hypersurfaces
            if self.interpolated:
                # in the case of interpolated hypersurfaces, the actual hypersurface
                # must be generated for the given oscillation parameters first
                container_hs = self.hypersurfaces[container.name].get_hypersurface(**osc_params)
            else:
                container_hs = self.hypersurfaces[container.name]

            # Fluctute the hypersurfaces, if requested
            if self.fluctuate :
                container_hs = container_hs.fluctuate(random_state=fluctuate_random_state) #TODO oncely need to do this once if not interpolating

            # Get the hypersurface scale factors (reshape to 1D array)
            if self.propagate_uncertainty:
                scales, uncertainties = container_hs.evaluate(param_values, return_uncertainty=True)
                scales = scales.reshape(container.size)
                uncertainties = uncertainties.reshape(container.size)
            else:
                scales = container_hs.evaluate(param_values).reshape(container.size)

            # Where there are no scales (e.g. empty bins), set scale factor to 1
            empty_bins_mask = ~np.isfinite(scales)
            num_empty_bins = np.sum(empty_bins_mask)
            if num_empty_bins > 0. and not self.warning_issued:
                logging.warn("%i empty bins found in hypersurface" % num_empty_bins)
                self.warning_issued = True
            scales[empty_bins_mask] = 1.
            if self.propagate_uncertainty:
                uncertainties[empty_bins_mask] = 0.

            # Add to container
            np.copyto(src=scales, dst=container["hs_scales"])
            container.mark_changed("hs_scales")
            if self.propagate_uncertainty:
                np.copyto(src=uncertainties, dst=container["hs_scales_uncertainty"])
                container.mark_changed("hs_scales_uncertainty")

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
