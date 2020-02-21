"""
PISA pi stage to apply hypersurface fits from discrete systematics parameterizations
"""

#TODO Currently have to defined the `links` value in the stage config to match what the `combine_regex` does in 
#     the Hypersurface fitting. The `combine_regex` is stored in the hypersurface instance, so should make the 
#     scontainer linking use this instead of having to manually specify links. To do this, should make a variant 
#     of the container linking function that accepts a regex (using shared code with Map.py).

#TODO Store hypersurface uncertainty and propagate this


from __future__ import absolute_import, print_function, division

import ast

import math
from numba import guvectorize
import numpy as np

from pisa import FTYPE, TARGET, ureg
from pisa.core.binning import MultiDimBinning
from pisa.core.pi_stage import PiStage
from pisa.utils.fileio import from_file
from pisa.utils.log import logging
from pisa.utils.numba_tools import WHERE
from pisa.utils import vectorizer
from pisa.utils.hypersurface import load_hypersurfaces
from numba import vectorize

__all__ = ["pi_hypersurfaces",]

__author__ = "P. Eller, T. Ehrhardt, T. Stuttard, J.L. Lanfranchi"

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


class pi_hypersurfaces(PiStage):  # pyint: disable=invalid-name
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

        # -- Expected input / output names -- #
        input_names = ()
        output_names = ()

        # -- Which keys are added or altered for the outputs during `apply` -- #

        input_calc_keys = ()
        if propagate_uncertainty:
            output_calc_keys = ("hypersurface_scalefactors", "hypersurface_scalefactors_uncertainty")
        else:
            output_calc_keys = ("hypersurface_scalefactors",)

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
        
        # Load hypersurfaces
        self.hypersurfaces = load_hypersurfaces(self.fit_results_file, calc_specs)

        # Get the expected param names
        # These are used as the expected param names for the stage
        self.hypersurface_param_names = list(self.hypersurfaces.values())[0].param_names

        # -- Initialize base class -- #
        super(pi_hypersurfaces, self).__init__(
            data=data,
            params=params,
            expected_params=self.hypersurface_param_names,
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


    def setup_function(self):
        """Load the fit results from the file and make some check compatibility"""

        self.data.data_specs = self.calc_specs

        if self.links is not None:
            for key, val in self.links.items():
                self.data.link_containers(key, val)

        # create containers for scale factors
        for container in self.data :
            container["hypersurface_scalefactors"] = np.empty(container.size, dtype=FTYPE)
            if self.propagate_uncertainty:
                container["hypersurface_scalefactors_uncertainty"] = np.empty(container.size, dtype=FTYPE)

        # Check map names match between data container and hypersurfaces
        for container in self.data:
            assert container.name in self.hypersurfaces, "No match for map %s found in the hypersurfaces" % (container.name)

        self.data.unlink_containers()


    def compute_function(self):

        self.data.data_specs = self.calc_specs

        # Link containers
        if self.links is not None:
            for key, val in self.links.items():
                self.data.link_containers(key, val)

        # Format the params dict that will be passed to `Hypersurface.evaluate`
        #TODO checks on param units
        param_values = { sys_param_name: self.params[sys_param_name].m for sys_param_name in self.hypersurface_param_names }

        # Evaluate the hypersurfaces
        for container in self.data:

            # Get the hypersurface scale factors (reshape to 1D array)
            if self.propagate_uncertainty:
                scalefactors, uncertainties = self.hypersurfaces[container.name].evaluate(param_values, return_uncertainty=True)
                scalefactors = scalefactors.reshape(container.size)
                uncertainties = uncertainties.reshape(container.size)
            else:
                scalefactors = self.hypersurfaces[container.name].evaluate(param_values).reshape(container.size)

            # Where there are no scalefactors (e.g. empty bins), set scale factor to 1 
            empty_bins_mask = ~np.isfinite(scalefactors)
            num_empty_bins = np.sum(empty_bins_mask)
            if num_empty_bins > 0. :
                logging.warn("%i empty bins found in hypersurface" % num_empty_bins)
            scalefactors[empty_bins_mask] = 1.
            if self.propagate_uncertainty:
                uncertainties[empty_bins_mask] = 0.
            
            # Add to container
            np.copyto(src=scalefactors, dst=container["hypersurface_scalefactors"].get('host'))
            container["hypersurface_scalefactors"].mark_changed()
            if self.propagate_uncertainty:
                np.copyto(src=uncertainties, dst=container["hypersurface_scalefactors_uncertainty"].get('host'))
                container["hypersurface_scalefactors_uncertainty"].mark_changed()

        # Unlink the containers again
        self.data.unlink_containers()


    def apply_function(self):
        for container in self.data:
            # update uncertainty first, before the weights are changed
            if self.error_method == "sumw2":
                if self.propagate_uncertainty:
                    calc_uncertainty(container["weights"].get(WHERE),
                                     container["hypersurface_scalefactors_uncertainty"].get(WHERE),
                                     container["errors"].get(WHERE),
                                    )                                     
                else:
                    vectorizer.multiply(container["hypersurface_scalefactors"], container["errors"])
                container['errors'].mark_changed()
            # Update weights according to hypersurfaces
            vectorizer.multiply(
                container["hypersurface_scalefactors"], container["weights"]
            )
            container['weights'].mark_changed()
            # Correct negative event counts that can be introduced by hypersurfaces (due to intercept)
            weights = container["weights"].get('host')
            neg_mask = weights < 0.
            if neg_mask.sum() > 0 :
                weights[neg_mask] = 0.
                np.copyto( src=weights, dst=container["weights"].get('host') )
                container["weights"].mark_changed()

# vectorized function to apply uncertainty
# must be outside class
if FTYPE == np.float32:
    _SIGNATURE = ['(f4[:], f4[:], f4[:])']
else:
    _SIGNATURE = ['(f8[:], f8[:], f8[:])']
@guvectorize(_SIGNATURE, '(),()->()', target=TARGET)
def calc_uncertainty(weight, scale_uncertainty, out):
    out[0] = weight[0]*scale_uncertainty[0]
