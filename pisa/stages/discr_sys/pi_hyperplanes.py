"""
PISA pi stage to apply hyperplane fits from discrete systematics parameterizations
"""

from __future__ import absolute_import, print_function, division

from numba import guvectorize
import numpy as np

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.fileio import from_file
from pisa.utils.log import logging
from pisa.utils.numba_tools import WHERE
from pisa.utils import vectorizer

__all__ = ["pi_hyperplanes", "eval_hyperplane"]

__author__ = "P. Eller, T. Ehrhardt, T. Stuttard"

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


# TODO: consider taking into account fit parameter covariances
class pi_hyperplanes(PiStage):  # pyint: disable=invalid-name
    """
    Service to apply hyperplane parameterisation produced by
    `scripts.fit_discrete_sys_nd`

    Parameters
    ----------
    fit_results_file : str

    params : ParamSet
        dom_eff : dimensionless quantity
        hole_ice : dimensionless quantity
        hole_ice_fwd : dimensionless quantity
        spiciness : dimensionless quantity
        bulk_scatter : dimensionless quantity
        bulk_abs : dimensionless quantity
        rde : dimensionless quantity

    Notes
    -----
    the fit_results_file must contain the following keys:
        "param_names" : sequence of strings
            Params fitted during the hyerplane fits (order is important)
        "binning_hash" : string (not present in old file format)
            Hash of binning used in fits
        "hyerplanes" : OrderedDict
            container with hyperplane fit information for each data type,
            <map_name> : hyperplane data is subdivded by event types (map names)
                fit_results : num params + 1 fit parameters per map bin
                        order of params is the same as defined in param_names
    """

    def __init__(
        self,
        fit_results_file,
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
        self.fit_results_file = fit_results_file

        expected_params = (
            "dom_eff",
            # 'rde',
            "hole_ice",
            "hole_ice_fwd",
            "spiciness",
            "bulk_scatter",
            "bulk_abs",
        )
        # will be needed at computation time, where
        # we want to make sure that the params from
        # the hyperplane fits constitute a subset
        self.sys_params = set(expected_params)
        input_names = ()
        output_names = ()

        # what keys are added or altered for the outputs during apply

        input_calc_keys =  ("hyperplane_results",)
        output_calc_keys = ("hyperplane_scalefactors",)

        if error_method == "sumw2":
            output_apply_keys = ("weights", "errors")
            input_apply_keys = output_apply_keys
        else:
            output_apply_keys = ("weights",)
            input_apply_keys = output_apply_keys


        # init base class
        super(pi_hyperplanes, self).__init__(
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

        assert self.input_mode is not None
        assert self.calc_mode == "binned"
        assert self.output_mode is not None

        self.fit_results = None
        """Parsed results of the hyperplane fit"""

        self.fit_sys_list = None
        """List of systematic parameters participating in the external fit"""

        self.fit_binning_hash = None
        """Hash of the binning used in the external fit"""

        self.inactive_sys_params = None
        """Inactive systematic parameters"""

        self.using_old_fit_file = None

        self.links = eval(links)  # pylint: disable=eval-used

    def setup_function(self):
        """Load the fit results from the file and make some check
        compatibility"""
        self.fit_results = from_file(self.fit_results_file)
        self.fit_binning_hash = self.fit_results.get("binning_hash", None)
        if not self.fit_binning_hash:
            logging.warn(
                "Cannot determine the hash of the binning employed"
                " for the hyperplane fits. Correct application of"
                " fits is not guaranteed!"
            )

        self.data.data_specs = self.calc_specs

        if self.links is not None:
            for key, val in self.links.items():
                self.data.link_containers(key, val)

        # handle backwards compatibility for old style fit results files
        if "hyperplanes" in self.fit_results:
            self.using_old_fit_file = False
        elif "sys_list" in self.fit_results:
            self.using_old_fit_file = True
        else:
            raise ValueError("Unrecognised format for input fit file")

        # get the hyperplane fits for each container type
        # reshape to a 1D array to match container
        for container in self.data:
            if self.using_old_fit_file:
                fits = self.fit_results[container.name]
            else:
                fits = self.fit_results["hyperplanes"][container.name]["fit_params"]
            container["hyperplane_results"] = fits.reshape(container.size, -1)
            container["hyperplane_scalefactors"] = np.empty(container.size, dtype=FTYPE)

        # get the list of systematicc parameter names fitted
        # need to conserve order here!
        if self.using_old_fit_file:
            self.fit_sys_list = self.fit_results["sys_list"]
        else:
            self.fit_sys_list = self.fit_results["param_names"]

        # do not require all of the expected parameters to be in the fit file
        # as it should be possible to run this stage with a subset of all
        # supported systematics
        excess = set(self.fit_sys_list).difference(self.sys_params)
        if excess:
            raise KeyError(
                "Fit results contain systematic parameters unaccounted for"
                " by this service, i.e. %s." % excess
            )

        # record the "inactive" systematics, i.e. those which we have no handle
        # on and which can thus not be used in the computation
        self.inactive_sys_params = (self.sys_params).difference(set(self.fit_sys_list))

        # check compatibility...
        # let's be extremely strict here for now: require identical binning
        # (full hash), but allow no hash in file for compatibility with legacy
        # hyperplane fit files
        if self.data.data_mode == "binned" and self.fit_binning_hash is not None:
            if self.data.data_specs.hash != self.fit_binning_hash:
                raise ValueError(
                    "Hash values disagree between data binning and hyperplane fits!"
                )
        self.data.unlink_containers()

    def compute_function(self):
        self.data.data_specs = self.calc_specs
        if self.links is not None:
            for key, val in self.links.items():
                self.data.link_containers(key, val)

        # get parameters, in the right order
        param_values = []
        for sys in self.fit_sys_list:
            # TODO: what about units?
            param_values.append(self.params[sys].magnitude)

        param_values = np.array(param_values, dtype=FTYPE)

        # require inactive params to be fixed
        for sys in self.inactive_sys_params:
            if not self.params[sys].is_fixed:
                raise ValueError(
                    'Please explicitly fix the inactive systematic "%s"'
                    " since it is not part of the hyperplane fit to be applied." % sys
                )
            logging.debug(
                'Ignoring discrete systematic parameter "%s" set to a value of %.2f',
                sys,
                self.params[sys].value,
            )

        for container in self.data:
            eval_hyperplane(
                param_values,
                container["hyperplane_results"].get(WHERE),
                out=container["hyperplane_scalefactors"].get(WHERE),
            )
            container["hyperplane_scalefactors"].mark_changed(WHERE)

        self.data.unlink_containers()

    def apply_function(self):
        for container in self.data:
            vectorizer.multiply(
                container["hyperplane_scalefactors"], container["weights"]
            )
            if self.error_method == "sumw2":
                vectorizer.multiply(
                    container["hyperplane_scalefactors"], container["errors"]
                )


if FTYPE == np.float64:
    SIGNATURE = "(f8[:], f8[:], f8[:])"
else:
    SIGNATURE = "(f4[:], f4[:], f4[:])"


@guvectorize([SIGNATURE], "(a),(b)->()", target=TARGET)
def eval_hyperplane(param_values, hyperplane_results, out):
    """vectorized function to apply hyperplanes"""
    result = hyperplane_results[0]
    for i in range(param_values.size):
        result += hyperplane_results[i + 1] * param_values[i]
    out[0] = result
