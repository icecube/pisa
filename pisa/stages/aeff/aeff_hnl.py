"""
PISA pi stage to apply effective area weights and HNL specific reweighting
"""

from __future__ import absolute_import, print_function, division

from pisa.core.stage import Stage
from pisa.utils.profiler import profile

from pisa import ureg

import numpy as np

# is there a place for constants in PISA, or do they already exist?
LIGHTSPEED = 299792458.0 * ureg.m / ureg.s
REDUCEDPLANCK = 6.582119569e-25 * ureg.GeV * ureg.s


def re_weight_hnl(
    U_tau4_sq,
    mass,
    energy,
    tau,
    distance_min,
    distance_max,
    hnl_decay_width,
    c=LIGHTSPEED,
    hbar=REDUCEDPLANCK,
):
    """
    Function to re-weight HNL events (from sampling 1/L to target exponential)

    Parameters
    ----------
    U_tau4_sq : float
        Square of the HNL mixing angle
    mass : float
        HNL mass in GeV
    energy : float
        HNL energy in GeV
    tau : float
        HNL proper lifetime in ns
    distance_min : float
        Minimum sampling distance of HNL decay in m
    distance_max : float
        Maximum sampling distance of HNL decay in m
    hnl_decay_width : float
        HNL decay width in GeV

    Returns
    -------
    weight_lifetime : float
        Weight to re-weight HNL events
    """

    gamma = (energy + mass) / mass  # (Ekin+E0)/E0
    speed = c * np.sqrt(1 - np.power(1.0 / gamma, 2))  # c * sqrt(1-1/gamma^2)

    tau_min = distance_min / (gamma * speed)
    tau_max = distance_max / (gamma * speed)

    # tau = 1e-09 * tau  # convert to seconds
    tau_proper = hbar / (hnl_decay_width * U_tau4_sq)

    pdf_inverse = (1.0 / (np.log(tau_max.magnitude) - np.log(tau_min.magnitude))) * (
        1.0 / tau.m_as('s')
    )  # for 1/L sampling of decay length

    pdf_exp1 = 1.0 / tau_proper
    pdf_exp2 = np.exp(-tau / tau_proper)

    pdf_exp = pdf_exp1 * pdf_exp2

    weight_lifetime = pdf_exp / pdf_inverse

    return U_tau4_sq.magnitude * weight_lifetime.magnitude


class aeff_hnl(Stage):  # pylint: disable=invalid-name
    """
    PISA Pi stage to apply aeff weights.

    This combines the detector effective area with the flux weights calculated
    in an earlier stage to compute the weights.

    Various scalings can be applied for particular event classes. The weight is
    then multiplied by the livetime to get an event count.

    Parameters
    ----------
    params
        Expected params are .. ::

            livetime : Quantity with time units
            aeff_scale : dimensionless Quantity
            nutau_cc_norm : dimensionless Quantity
            nutau_norm : dimensionless Quantity
            nu_nc_norm : dimensionless Quantity
            U_tau4_sq : dimensionless Quantity
    """

    def __init__(
        self,
        **std_kwargs,
    ):
        expected_params = (
            "livetime",
            "aeff_scale",
            "nutau_cc_norm",
            "nutau_norm",
            "nu_nc_norm",
            "U_tau4_sq",
        )

        # init base class
        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

    @profile
    def apply_function(self):
        livetime = self.params.livetime.m_as("sec")
        aeff_scale = self.params.aeff_scale.m_as("dimensionless")
        nutau_cc_norm = self.params.nutau_cc_norm.m_as("dimensionless")
        nutau_norm = self.params.nutau_norm.m_as("dimensionless")
        nu_nc_norm = self.params.nu_nc_norm.m_as("dimensionless")
        U_tau4_sq = self.params.U_tau4_sq.m_as("dimensionless")

        for container in self.data:
            scale = aeff_scale * livetime
            if container.name in ["nutau_cc", "nutaubar_cc"]:
                scale *= nutau_cc_norm
            if "nutau" in container.name:
                scale *= nutau_norm
            if "nc" in container.name:
                scale *= nu_nc_norm

            hnl_weight_scale = re_weight_hnl(
                U_tau4_sq=U_tau4_sq * ureg.dimensionless,
                mass=container['mHNL'] * ureg.GeV,
                energy=container['hnl_true_energy'] * ureg.GeV,
                tau=container['hnl_proper_lifetime'] * ureg.ns,
                distance_min=container['hnl_distance_min'] * ureg.m,
                distance_max=container['hnl_distance_max'] * ureg.m,
                hnl_decay_width=container['hnl_decay_width'] * ureg.GeV,
            )

            scale *= hnl_weight_scale

            container["weights"] *= container["weighted_aeff"] * scale
            container.mark_changed("weights")
