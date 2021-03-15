"""
A Stage to load data from a FreeDOM hdf5 file (generated using the I3HDFWriter)
"""

from __future__ import absolute_import, print_function, division

import numpy as np
import tables

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.utils.resources import find_resource
from pisa.core.container import Container
from pisa.utils.format import split

X_36 = 46.29
Y_36 = -34.88

FREEDOM_PARNAMES = [
    "x",
    "y",
    "z",
    "time",
    "azimuth",
    "zenith",
    "cascade_energy",
    "track_energy",
]


class freedom_hdf5_loader(Stage):
    """
    FreeDOM hdf5 file loader

    Parameters
    ----------
    events_file : hdf5 file path
    files_per_flavor : list
        number of files for each nu flavor
    reco : str
        which reco to use
    keys : list
        which keys to unpack
    cuts : dict
        cuts to apply when loading the data
    track_E_cut : float
        track energy cut for PID.
        If not set, L7_PIDClassifier_ProbTrack will be used
    **kwargs
        Passed to Stage

    """

    def __init__(
        self,
        events_file,
        files_per_flavor,
        output_names,
        reco,
        keys,
        cuts,
        track_E_cut=None,
        **std_kwargs,
    ):

        # instantiation args that should not change
        self.events_file = find_resource(events_file)

        # init base class
        super().__init__(
            expected_params=(), **std_kwargs,
        )

        self.output_names = output_names
        self.reco = reco
        self.files_per_flavor = split(files_per_flavor)
        self.track_E_cut = track_E_cut
        self.keys = split(keys)
        self.cuts = eval(cuts)

    def setup_function(self):

        data = self.load_hdf5_file(self.events_file)
        data = self.calc_rho36(data)
        if any(key.startswith("unc_est") for key in self.cuts):
            data = self.calc_uncertainties(data)
        data = self.apply_cuts(data, self.cuts)

        for name in self.output_names:
            container = Container(name)

            nubar = -1 if "bar" in name else 1
            if "e" in name:
                flav = 0
                n_files = int(self.files_per_flavor[0])
            if "mu" in name:
                flav = 1
                n_files = int(self.files_per_flavor[1])
            if "tau" in name:
                flav = 2
                n_files = int(self.files_per_flavor[2])

            pdg = nubar * (12 + 2 * flav)

            mask = data["pdg_id"] == pdg
            if "cc" in name:
                mask = np.logical_and(mask, data["interaction_type"] == 1)
            else:
                mask = np.logical_and(mask, data["interaction_type"] == 2)

            events = {key: value[mask] for key, value in data.items()}
            weight_dict = events["I3MCWeightDict"]
            primary = events["MCInIcePrimary"]

            container["true_energy"] = primary["energy"].astype(FTYPE)
            container["true_coszen"] = np.cos(primary["zenith"]).astype(FTYPE)
            container["pdg_code"] = primary["pdg_encoding"].astype(FTYPE)
            container["interaction"] = weight_dict["InteractionType"].astype(FTYPE)

            CM2_TO_M2 = 1e-4
            derived_weight = (
                CM2_TO_M2
                * weight_dict["OneWeight"]
                / n_files
                / weight_dict["gen_ratio"]
                / weight_dict["NEvents"]
            )
            container["weighted_aeff"] = derived_weight.astype(FTYPE)

            reco = self.reco
            reco_total_energy = (
                events[f"{reco}_cascade_energy"] + events[f"{reco}_track_energy"]
            )
            container["reco_energy"] = reco_total_energy.astype(FTYPE)
            container["reco_coszen"] = np.cos(events[f"{reco}_zenith"]).astype(FTYPE)
            container["reco_z"] = events[f"{reco}_z"].astype(FTYPE)
            container["reco_rho"] = events["rho_36"].astype(FTYPE)

            if self.track_E_cut is None:
                container["pid"] = events["L7_PIDClassifier_ProbTrack"].astype(FTYPE)
            else:
                pid = events[f"{reco}_track_energy"] > float(self.track_E_cut)
                container["pid"] = pid.astype(FTYPE)

            container["weights"] = np.ones(container.size, dtype=FTYPE)
            container["initial_weights"] = np.ones(container.size, dtype=FTYPE)

            container.set_aux_data("nubar", nubar)
            container.set_aux_data("flav", flav)

            self.data.add_container(container)

        if len(self.data.names) == 0:
            raise ValueError(
                "No containers created during data loading for some reason."
            )

    def apply_function(self):
        for container in self.data:
            container["weights"] = np.copy(container["initial_weights"])

    def calc_rho36(self, data):
        reco_x = data[f"{self.reco}_x"]
        reco_y = data[f"{self.reco}_y"]

        data["rho_36"] = np.sqrt((reco_x - X_36) ** 2 + (reco_y - Y_36) ** 2)

        return data

    def calc_uncertainties(self, data, epsilon=1e-15):
        """add uncertainty estimates to data; return modified data"""
        for par in FREEDOM_PARNAMES:
            p2s = np.where(data[f"env_p2_{par}"] > 0, data[f"env_p2_{par}"], epsilon)
            data[f"unc_est_{par}"] = 1 / np.sqrt(2 * p2s)

        return data

    def apply_cuts(self, data, cuts):
        """apply cuts in place"""
        cut_mask = np.array([True], dtype=np.bool)
        for cut_key, [cut_low, cut_high] in cuts.items():
            if "{reco}" in cut_key:
                cut_key = cut_key.replace("{reco}", self.reco)

            if cut_low is not None:
                cut_mask = cut_mask & (data[cut_key] > cut_low)
            if cut_high is not None:
                cut_mask = cut_mask & (data[cut_key] < cut_high)

        for key, val in data.items():
            data[key] = val[cut_mask]

        return data

    def load_hdf5_file(self, f_name):
        with tables.File(f_name) as file:
            return self.unpack_file_data(file)

    def unpack_file_data(self, file):
        root = file.root

        var_dict = {}
        for node in root:
            try:
                name = node.name.replace("FreeDOM_test_", "").replace("_params", "")
                name = name.replace("best_fit", "freedom")
            except tables.NoSuchNodeError:
                continue

            if not name in self.keys and name != self.reco:
                continue

            freedom_hdf5_loader.fill_variable_dict(name, node, var_dict)

        return var_dict

    @staticmethod
    def fill_variable_dict(name, node, var_dict):
        if "vector_index" in node.colnames:
            for par, parname in zip(
                node.cols.item[:].reshape((-1, len(FREEDOM_PARNAMES))).T,
                FREEDOM_PARNAMES,
            ):
                var_dict[f"{name}_{parname}"] = par
        elif "value" in node.colnames:
            var_dict[name] = node.cols.value[:]
        else:
            var_dict[name] = node.read()
