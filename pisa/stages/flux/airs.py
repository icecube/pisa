"""
Stage to implement the atmospheric density uncertainty.
Uses the neutrino fluxes calculated in the mceq_barr stage, and scales the weights

Ben Smithers
"""

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.utils.profiler import profile
from pisa.utils.resources import find_resource

import photospline

import numpy as np


class airs(Stage):  # pylint: disable=invalid-name
    """
    Parameters
    ----------
    airs_spline : spline containing the 1-sigma shifts from AIRS data

    params : ParamSet
        Must exclusively have parameters: .. ::

            scale : quantity (dimensionless)
                the scale by which the weights are perturbed via the airs 1-sigma shift
                
        Expected container keys are .. ::

            "true_energy"
            "true_coszen"
            "weights"
    """

    def __init__(self, airs_spline, **std_kwargs):
        _airs_spline_loc = find_resource(airs_spline)
        self.airs_spline = photospline.SplineTable(_airs_spline_loc)

        expected_params = [
            "airs_scale",
        ]
        
        expected_container_keys = (
            'true_energy',
            'true_coszen',
            'weights',
        )

        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )

    def setup_function(self):
        """
        Uses the splines to quickly evaluate the 1-sigma perturbtations at each of the events
        """

        # consider 'true_coszen" and 'true_energy' containers
        for container in self.data:
            if len(container["true_energy"]) == 0:
                container["airs_1s_perturb"] = np.zeros(container.size, dtype=FTYPE)
            else:
                container["airs_1s_perturb"] = self.airs_spline.evaluate_simple(
                    (np.log10(container["true_energy"]), container["true_coszen"])
                )
            container.mark_changed("airs_1s_perturb")

    @profile
    def apply_function(self):
        """
        Modify the weights according to the new scale parameter!
        """
        for container in self.data:
            container["weights"] *= 1.0 + container[
                "airs_1s_perturb"
            ] * self.params.airs_scale.value.m_as("dimensionless")
