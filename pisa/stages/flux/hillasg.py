"""
Stage to evaluate the Hillas-Gaisser expectations from precalculated fluxes

"""

import numpy as np

from pisa import FTYPE
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.flux_weights import load_2d_table, calculate_2d_flux_weights

__all__ = ['hillasg', 'init_test']


class hillasg(Stage):  # pylint: disable=invalid-name
    """
    stage to generate nominal flux

    Parameters
    ----------
    params
        Expected params .. ::

            flux_table : str

        Expected container keys are .. ::

            "true_energy"
            "true_coszen"

    """

    def __init__(self, **std_kwargs):

        expected_params = (
            "flux_table",
        )

        expected_container_keys = (
            'true_energy',
            'true_coszen',
        )

        # init base class
        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )

    def setup_function(self):

        self.flux_table = load_2d_table(self.params.flux_table.value)

        self.data.representation = self.calc_mode
        if self.data.is_map:
            # speed up calculation by adding links
            # as nominal flux doesn't depend on the (outgoing) flavour
            self.data.link_containers(
                "nu",
                [
                    "nue_cc",
                    "numu_cc",
                    "nutau_cc",
                    "nue_nc",
                    "numu_nc",
                    "nutau_nc",
                    "nuebar_cc",
                    "numubar_cc",
                    "nutaubar_cc",
                    "nuebar_nc",
                    "numubar_nc",
                    "nutaubar_nc",
                ],
            )
        for container in self.data:
            container["nu_flux_nominal"] = np.empty((container.size, 3), dtype=FTYPE)
            container["nubar_flux_nominal"] = np.empty((container.size, 3), dtype=FTYPE)
            # container['nu_flux'] = np.empty((container.size, 2), dtype=FTYPE)

        # don't forget to un-link everything again
        self.data.unlink_containers()

    @profile
    def compute_function(self):

        self.data.representation = self.calc_mode

        if self.data.is_map:
            # speed up calculation by adding links
            # as nominal flux doesn't depend on the (outgoing) flavour
            self.data.link_containers(
                "nu",
                [
                    "nue_cc",
                    "numu_cc",
                    "nutau_cc",
                    "nue_nc",
                    "numu_nc",
                    "nutau_nc",
                    "nuebar_cc",
                    "numubar_cc",
                    "nutaubar_cc",
                    "nuebar_nc",
                    "numubar_nc",
                    "nutaubar_nc",
                ],
            )

        # create lists for iteration
        out_names = ["nu_flux_nominal"] * 3 + ["nubar_flux_nominal"] * 3
        indices = [0, 1, 2, 0, 1, 2]
        tables = ["nue", "numu", "nutau", "nuebar", "numubar", "nutaubar"]
        for container in self.data:
            for out_name, index, table in zip(out_names, indices, tables):
                logging.info(
                    "Calculating nominal %s flux for %s", table, container.name
                )
                calculate_2d_flux_weights(
                    true_energies=container["true_energy"],
                    true_coszens=container["true_coszen"],
                    en_splines=self.flux_table[table],
                    out=container[out_name][:, index],
                )
            container.mark_changed("nu_flux_nominal")
            container.mark_changed("nubar_flux_nominal")

        # don't forget to un-link everything again
        self.data.unlink_containers()


def init_test(**param_kwargs):
    """Instantiation example"""
    param_set = ParamSet([
        Param(name='flux_table',
              value='meows_sterile/resources/fluxes/pr_hillas_h3a_SIBYLL23C_AIRS_nominalize-aa.d',
              **param_kwargs),
    ])
    return hillasg(params=param_set)
