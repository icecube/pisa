"""
A class to load in the MEOWS hdf5 files
"""

from time import time

import numpy as np
import h5py as h5

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.core.container import Container
from pisa.utils.resources import find_resource


class meows_loader(Stage):  # pylint: disable=invalid-name
    """
    Docstring incoming...
    """

    def __init__(self, events_file: str, n_files: int, output_names, **std_kwargs):
        self.events_file = events_file
        self._n_files = int(n_files)
        self.output_names = output_names
        super().__init__(expected_params=(), **std_kwargs)

    def setup_function(self):
        """
        Go over all those input files and load them in.

        We load the first data in specifically to setup the containers, and afterwards go through appending to the end of those arrays
        """

        print("Loading data...", end="")
        st = time()
        raw_data = h5.File(find_resource(self.events_file), "r")

        for name in self.output_names:
            # make container
            container = Container(name)

            nubar = -1 if "bar" in name else 1
            if "e" in name:
                flav = 0
            if "mu" in name:
                flav = 1
            if "tau" in name:
                flav = 2

            # cut out right part
            pdg = nubar * (12 + 2 * flav)

            mask = raw_data["PrimaryType"][:] == pdg
            # there's no interaction key in this MC, so we put this in so only the CC are used
            if "cc" in name:
                mask = np.logical_and(mask, raw_data["PrimaryType"] != 0)
            else:
                mask = np.logical_and(mask, raw_data["PrimaryType"] == 0)

            events = raw_data

            container["weighted_aeff"] = (
                events["oneweight"][mask][:].astype(FTYPE) * (1e-4) / (98000 / 5.0)
            )
            container["weights"] = np.ones(container.size, dtype=FTYPE)
            container["initial_weights"] = np.ones(container.size, dtype=FTYPE)

            container["total_column_depth"] = events["TotalColumnDepth"][mask][
                :
            ].astype(FTYPE)
            container["true_bjorkenx"] = events["FinalStateX"][mask][:].astype(FTYPE)
            container["true_bjorkeny"] = events["FinalStateY"][mask][:].astype(FTYPE)

            container["true_energy"] = events["NuEnergy"][mask][:].astype(FTYPE)
            container["true_coszen"] = np.cos(events["NuZenith"][mask][:].astype(FTYPE))
            container["reco_energy"] = events["MuExEnergy"][mask][:].astype(FTYPE)
            container["reco_coszen"] = np.cos(
                events["MuExZenith"][mask][:].astype(FTYPE)
            )
            container["pid"] = events["pid"][mask][:].astype(FTYPE)
            container.set_aux_data("nubar", nubar)
            container.set_aux_data("flav", flav)

            self.data.add_container(container)

        ed = time()
        print(" done! Took {} minutes".format((ed - st) / 60))
        raw_data.close()

    def apply_function(self):
        """
        Resets all the weights to the initial weights
        """
        for container in self.data:
            container["weights"] = np.copy(container["initial_weights"])
            container["astro_weights"] = np.copy(container["initial_weights"])
