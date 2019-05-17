"""
PISA pi stage to apply hyperplane fits from discrete systematics parameterizations
"""

from __future__ import absolute_import, print_function, division

import ast

from numba import guvectorize
import numpy as np

from pisa import FTYPE, TARGET, ureg
from pisa.core.binning import MultiDimBinning
from pisa.core.pi_stage import PiStage
from pisa.utils.fileio import from_file
from pisa.utils.log import logging
from pisa.utils.numba_tools import WHERE
from pisa.utils import vectorizer

__all__ = ["pi_hyperplanes", "SIGNATURE", "eval_hyperplane"]

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


# TODO: consider taking into account fit parameter covariances
class pi_hyperplanes(PiStage):  # pyint: disable=invalid-name
    """
    Service to apply hyperplane parameterisation produced by
    `scripts.fit_discrete_sys_nd`

    Parameters
    ----------
    fit_results_file : str
        Path to hyperplane fit results file, i.e. the JSON file produced by the
        `pisa.scripts.fit_discrete_sys_nd.py` script

    params : ParamSet
        Note that the params required to be in `params` are found from
        those listed in the `fit_results_file`

    Notes
    -----
    Loading the `fit_results_file` results in an OrderedDict containing the following
    keys / values :

        "hyperplanes" : OrderedDict of OrderedDicts of OrderedDicts
            Container with hyperplane fit information for each data type. Format is:

            {
                "<map_name>":
                    "fit_params":
                    "chi2s":
                    "cov_matrices":
                    "finite_mask":
                    "fit_results":
                    hyperplane data is subdivded by event types (map names)
            }

        "datasets" : list of OrderedDicts
            Each dict contains

                {
                    "pipeline_cfg_paths": [cfgpath0, cfgpath1, ...],
                    "pipeline_cfg_txts": [cfgtxt0, cfgtxt1, ...],
                    "distribution_maker_param_values": {
                        param0: [mag, units], param1: [mag, units], ...
                    },
                    "param_values": [
                        hyperplane_fit_param0_mag, hyperplane_fit_param1_mag, ...
                    ],
                    "mapset": MapSet dict,
                    "nominal": bool,
                    "norm_mapset": MapSet (what is this, though?),
                }

        "param_names" : sequence of strings
            Params fitted during the hyperplane fits (order is important)

        "param_units" : sequence of strings, optional
            If not specified, "dimensionless" units are assumed for all fit params.
            Specified in same order as "param_names"

        "fit_cfg_path" : string
            Path to fit config file used for the hyperplane fits

        "fit_cfg_txt" : string
            Full text of the parsed fit config file used for the hyperplane fits

        "sys_param_points" : shape-(n_sys_sets, n_sys_params) ndarray of floats

        "binning" : OrderedDict (not present in older file formats)
            Instantiable via ``MultiDimBinning(**binning)``; used to ensure that binning
            used for fits matches (within single precision--regardless of
            PISA_FTYPE) the binning being used for this stage

        "binning_hash" : string (not present in oldest file format)
            Hash of binning used in fits

        "p0" : OrderedDict
            Initial guesses for each hyperplane fit param for each fit (container)

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
        # -- Read fit_results_file and extract necessary info -- #

        if fit_results_file.endswith('.csv'):
            # in this case those are datarelease files
            form = 'datarelease'
            import pandas as pd
            fit_results = {}
            fit_results['nue_cc+nuebar_cc'] = pd.read_csv(fit_results_file.replace('*', 'nue_cc'))
            fit_results['numu_cc+numubar_cc'] = pd.read_csv(fit_results_file.replace('*', 'numu_cc'))
            fit_results['nutau_cc+nutaubar_cc'] = pd.read_csv(fit_results_file.replace('*', 'nutau_cc'))
            fit_results['nu_nc+nubar_nc'] = pd.read_csv(fit_results_file.replace('*', 'all_nc'))

            fit_param_names = [a for a in fit_results['nu_nc+nubar_nc'].columns if a not in ['pid', 'reco_energy', 'reco_coszen', 'offset']]
            fit_binning = calc_specs

        else: 

            fit_results = from_file(fit_results_file)

            # handle backwards compatibility for old style fit results files
            if "hyperplanes" in fit_results:
                form = 'linear'
            elif "sys_list" in fit_results:
                form = 'legacy'
            else:
                raise ValueError("Unrecognised format for input fit file")

            # get list of systematic parameter names fitted; need to conserve order here!
            if form == 'legacy':
                fit_param_names = fit_results["sys_list"]
            else:
                fit_param_names = fit_results["param_names"]

            # Perfer to have the actual binning, so we can compare bin edges to
            # "reasonable" precision to make sure the hyperplane fits are applicable to the
            # current binning.
            #
            # If there is no binning in the hyperplane fit results file, look for a hash
            # value; barring that, just ensure that the dimensionality & number of bins
            # match.
            binning_spec = fit_results.get("binning", None)
            if binning_spec is not None:
                fit_binning = MultiDimBinning(**binning_spec)
            else:
                fit_binning = None


        if "param_units" in fit_results:
            fit_param_units = fit_results["param_units"]
        else:
            fit_param_units = ["dimensionless" for _ in fit_param_names]
        fit_param_units = [ureg.Unit(u) for u in fit_param_units]

        if fit_binning is not None:
            fit_binning_hash = fit_binning.hash
        else:
            fit_binning_hash = fit_results.get("binning_hash", None)

        if fit_binning_hash is None:
            logging.warn(
                "Cannot determine the hash of the binning employed"
                " for the hyperplane fits. Correct application of"
                " fits is not guaranteed!"
            )

        # -- Expected input / output names -- #

        input_names = ()
        output_names = ()

        # -- Which keys are added or altered for the outputs during `apply` -- #

        input_calc_keys = ()
        output_calc_keys = ("hyperplane_scalefactors",)

        if error_method == "sumw2":
            output_apply_keys = ("weights", "errors")
            input_apply_keys = output_apply_keys
        else:
            output_apply_keys = ("weights",)
            input_apply_keys = output_apply_keys

        # -- Initialize base class -- #

        super(pi_hyperplanes, self).__init__(
            data=data,
            params=params,
            expected_params=fit_param_names,
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

        # -- Add attrs to `self` specific to `pi_hyperplanes` -- #

        self.fit_results_file = fit_results_file
        """str : path to hyperplane fit results file"""

        self.fit_results = fit_results
        """OrderedDict : parsed hyperplane fit file"""

        self.fit_param_names = fit_param_names
        """list : param names used in hyperplane fit, in order they appear in file"""

        self.fit_param_units = fit_param_units
        """list : param untis used in hyperplane fit, in order they appear in file"""

        self.fit_binning = fit_binning
        """MultiDimBinning : binning used for hyperplane fits; one hyperplane per bin"""

        self.fit_binning_hash = fit_binning_hash
        """str : hash of the binning used for hyperplane fits"""

        self.form = form
        """str : format of input file"""

    def setup_function(self):
        """Load the fit results from the file and make some check compatibility"""
        self.data.data_specs = self.calc_specs

        if self.links is not None:
            for key, val in self.links.items():
                self.data.link_containers(key, val)

        # get the hyperplane fits for each container type
        # reshape to a 1D array to match container
        for container in self.data:
            container_name = container.name
            if self.form == 'legacy':
                if container_name not in self.fit_results:
                    raise KeyError(
                        "'{}' not in fit results; valid keys are {}".format(
                            container_name, self.fit_results[container_name]
                        )
                    )
                fits = self.fit_results[container_name]

            elif self.form == 'linear':
                if not container_name in self.fit_results["hyperplanes"]:
                    raise KeyError(
                        "'{}' not in fit results; valid keys are {}".format(
                            container_name, self.fit_results["hyperplanes"].keys()
                        )
                    )
                fits = self.fit_results["hyperplanes"][container_name]["fit_params"]

            elif self.form == 'datarelease':
                fits = np.empty(list(self.calc_specs.shape) + [len(self.fit_param_names) + 1], dtype=FTYPE)
                # convert the numbers from the file back into 3d hists
                bin_edges = [edges.magnitude for edges in self.calc_specs.bin_edges]
                sample = [self.fit_results[container_name][s].values for s in self.calc_specs.names]
                for i,p in enumerate(['offset'] + self.fit_param_names):
                    hist, _ = np.histogramdd(sample=sample, weights=self.fit_results[container_name][p].values, bins=bin_edges)
                    fits[...,i] = hist

            container["hyperplane_results"] = fits.reshape(container.size, -1)
            container["hyperplane_scalefactors"] = np.empty(container.size, dtype=FTYPE)

        # TODO: check binning compatibility, but allow for FP32/FP64 differences
        # regardless of the currently-specified PISA_FTYPE and which was used for fit
        # (e.g., simply check to FP32 precision regardless of PISA_FTYPE?)

        # check compatibility...
        # let's be extremely strict here for now: require identical binning
        # (full hash), but allow no hash in file for compatibility with legacy
        # hyperplane fit files
        # if self.data.data_mode == "binned" and self.fit_binning_hash is not None:
        #    if self.data.data_specs.hash != self.fit_binning_hash:
        #        raise ValueError(
        #            "Hash values disagree between data binning and hyperplane fits!"
        #        )
        self.data.unlink_containers()

    def compute_function(self):
        self.data.data_specs = self.calc_specs
        if self.links is not None:
            for key, val in self.links.items():
                self.data.link_containers(key, val)

        # get parameters, in the right order
        param_values = []
        for sys_param_name, units in zip(self.fit_param_names, self.fit_param_units):
            param_values.append(self.params[sys_param_name].m_as(units))

        param_values = np.array(param_values, dtype=FTYPE)

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
