"""
PISA pi stage to apply hypersurface fits from discrete systematics parameterizations
"""

#TODO Currently have to defined the `links` value in the stage config to match what the `combine_regex` does in
#     the Hypersurface fitting. The `combine_regex` is stored in the hypersurface instance, so should make the
#     scontainer linking use this instead of having to manually specify links. To do this, should make a variant
#     of the container linking function that accepts a regex (using shared code with Map.py).

from __future__ import absolute_import, print_function, division

import ast

from numba import guvectorize
import numpy as np

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.numba_tools import WHERE
from pisa.utils import vectorizer
import pisa.utils.hypersurface as hs
from pisa.utils.log import set_verbosity, Levels
from pisa.utils.profiler import line_profile
#set_verbosity(Levels.DEBUG)

__all__ = ["pi_hypersurfaces",]

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


class pi_hypersurfaces(PiStage): # pylint: disable=invalid-name
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
    """
    def __init__(
        self,
        fit_results_file,
        propagate_uncertainty=False,
        interpolated=False,
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        error_method=None,
        input_specs=None,
        calc_specs=None,
        output_specs=None,
        links=None,
    ):
        # pylint: disable=line-too-long
        # -- Expected input / output names -- #
        input_names = ()
        output_names = ()

        # -- Which keys are added or altered for the outputs during `apply` -- #

        input_calc_keys = ()
        if propagate_uncertainty:
            output_calc_keys = ("hs_scales", "hs_scales_uncertainty")
        else:
            output_calc_keys = ("hs_scales",)

        if error_method == "sumw2":
            output_apply_keys = ("weights", "errors")
            input_apply_keys = output_apply_keys
        else:
            output_apply_keys = ("weights",)
            input_apply_keys = output_apply_keys

        # -- Load hypersurfaces -- #

        # Store args
        self.fit_results_file = fit_results_file
        self.propagate_uncertainty = propagate_uncertainty
        self.interpolated = interpolated
        # expected parameter names depend on the hypersurface and, if applicable,
        # on the parameters in which the hypersurfaces are interpolated
        if self.interpolated:
            hs_params, inter_params = hs.extract_interpolated_hypersurface_params(self.fit_results_file)
            self.hypersurface_param_names = hs_params
            self.inter_params = inter_params
            expected_params = hs_params+inter_params
        else:
            hypersurfaces = hs.load_hypersurfaces(self.fit_results_file, calc_specs)
            self.hypersurface_param_names = list(hypersurfaces.values())[0].param_names
            expected_params = self.hypersurface_param_names

        # -- Initialize base class -- #
        super(pi_hypersurfaces, self).__init__(
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
            input_calc_keys=input_calc_keys,
            output_calc_keys=output_calc_keys,
            input_apply_keys=input_apply_keys,
            output_apply_keys=output_apply_keys,
        )

        # -- Only allowed/implemented modes -- #

        assert self.input_mode is not None
        assert self.calc_mode == "binned"
        assert self.output_mode is not None

        self.links = ast.literal_eval(links)
        self.warning_issued = False # don't warn more than once about empty bins
        self.hypersurfaces = None
    # pylint: disable=line-too-long
    def setup_function(self):
        """Load the fit results from the file and make some check compatibility"""
        # load hypersurfaces
        if self.interpolated:
            self.hypersurfaces = hs.load_interpolated_hypersurfaces(self.fit_results_file, self.calc_specs)
        else:
            self.hypersurfaces = hs.load_hypersurfaces(self.fit_results_file, self.calc_specs)
        self.data.data_specs = self.calc_specs

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

        self.data.data_specs = self.calc_specs

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
        # Evaluate the hypersurfaces
        for container in self.data:
            if self.interpolated:
                # in the case of interpolated hypersurfaces, the actual hypersurface
                # must be generated for the given oscillation parameters first
                container_hs = self.hypersurfaces[container.name].get_hypersurface(**osc_params)
            else:
                container_hs = self.hypersurfaces[container.name]
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
            np.copyto(src=scales, dst=container["hs_scales"].get('host'))
            container["hs_scales"].mark_changed()
            if self.propagate_uncertainty:
                np.copyto(src=uncertainties, dst=container["hs_scales_uncertainty"].get('host'))
                container["hs_scales_uncertainty"].mark_changed()

        # Unlink the containers again
        self.data.unlink_containers()

    @line_profile
    def apply_function(self):

        for container in self.data:
            # update uncertainty first, before the weights are changed. This step is skipped in event mode
            if self.error_method == "sumw2":

                # If computing uncertainties in events mode, warn that
                # hs error propagation will be skipped
                if self.data.data_specs=='events':
                    logging.trace('WARNING: running stage in events mode. Hypersurface error propagation will be IGNORED.')
                
                elif self.propagate_uncertainty:
                    calc_uncertainty(container["weights"].get(WHERE),
                                     container["hs_scales_uncertainty"].get(WHERE),
                                     container["errors"].get(WHERE),
                                    )
                    container['errors'].mark_changed()

                else:
                    vectorizer.imul(container["hs_scales"], out=container["errors"])
                    container['errors'].mark_changed()

            # Update weights according to hypersurfaces
            propagate_hs_scales(container["weights"].get(WHERE),
                                container["hs_scales"].get(WHERE),
                                container["weights"].get(WHERE))

            container['weights'].mark_changed()


if FTYPE == np.float32:
    _SIGNATURE = ['(f4[:], f4[:], f4[:])']
else:
    _SIGNATURE = ['(f8[:], f8[:], f8[:])']
@guvectorize(_SIGNATURE, '(),()->()', target=TARGET)
def calc_uncertainty(weight, scale_uncertainty, out):
    '''vectorized error propagation'''
    out[0] = weight[0]*scale_uncertainty[0]


if FTYPE == np.float32:
    _SIGNATURE = ['(f4[:], f4[:], f4[:])']
else:
    _SIGNATURE = ['(f8[:], f8[:], f8[:])']
@guvectorize(_SIGNATURE, '(),()->()', target=TARGET)
def propagate_hs_scales(weight, hs_scales, out):
    '''vectorized error propagation'''
    out[0] = max(0., weight[0]*hs_scales[0])