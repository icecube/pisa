"""
stage to implement getting the contribution to fluxes from astrophysical neutrino sources
"""
import numpy as np

from pisa import FTYPE
from pisa.utils.profiler import profile
from pisa.core.stage import Stage


PIVOT = FTYPE(100.0e3)


class astrophysical(Stage):  # pylint: disable=invalid-name
    """
    Stage to apply power law astrophysical fluxes

    Parameters
    ----------
    params
        Expected params are .. ::
            astro_delta : quantity (dimensionless)
            astro_norm : quantity (dimensionless)

    TODO: flavor ratio as a parameter? Save for later.
    """

    def __init__(self, **std_kwargs):
        self._central_gamma = FTYPE(-2.5)
        self._central_norm = FTYPE(0.787e-18)

        self._e_ratio = FTYPE(1.0)
        self._mu_ratio = FTYPE(1.0)
        self._tau_ratio = FTYPE(1.0)

        expected_params = ("astro_delta", "astro_norm")

        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

    def setup_function(self):
        """
        Setup the nominal flux
        """
        self.data.representation = self.calc_mode
        for container in self.data:
            container["astro_weights"] = np.ones(container.size, dtype=FTYPE)
            container["astro_flux"] = np.ones(container.size, dtype=FTYPE)
            container["astro_flux_nominal"] = np.ones(container.size, dtype=FTYPE)

        # Loop over containers
        for container in self.data:

            # Grab containers here once to save time
            true_energy = container["true_energy"]

            container["astro_flux_nominal"] = self._central_norm * np.power(
                (true_energy / PIVOT), self._central_gamma
            )

            # TODO split this up so that we can use flavor ratios
            # nu_flux_nominal[:,0] = _precalc*self._e_ratio
            # nu_flux_nominal[:,1] = _precalc*self._mu_ratio
            # nu_flux_nominal[:,2] = _precalc*self._tau_ratio

            container.mark_changed("astro_flux_nominal")

    @profile
    def compute_function(self):
        """
        Tilt it, scale it, bop it
        """
        self.data.representation = self.calc_mode

        delta = self.params.astro_delta.value.m_as("dimensionless")
        norm = self.params.astro_norm.value

        for container in self.data:
            apply_sys_loop(
                container["true_energy"],
                container["true_coszen"],
                FTYPE(delta),
                FTYPE(norm),
                container["astro_flux_nominal"],
                out=container["astro_flux"],
            )
            container.mark_changed("astro_flux")

    @profile
    def apply_function(self):
        for container in self.data:
            container["astro_weights"] = (
                container["initial_weights"] * container["astro_flux"]
            )


def spectral_index_scale(true_energy, delta_index):
    """
    Calculate spectral index scale.
    Adjusts the weights for events in an energy dependent way according to a
    shift in spectral index, applied about a user-defined energy pivot.
    """
    return np.power(true_energy / PIVOT, delta_index)


def apply_sys_loop(
    true_energy,
    true_coszen,
    delta_index,
    norm,
    astroflux_nominal,
    out,
):
    """
    Calculation:
      1) Start from nominal flux
      2) Apply spectral index shift
      3) Add contributions from MCEq-computed gradients

    Array dimensions :
        true_energy : [A]
        true_coszen : [A]
        delta_index : scalar float
        norm : scalar float
        astroflux_nominal : [A,B]
        out : [A,B] (sys flux)
    where:
        A = num events
        B = num flavors in flux (=3, e.g. e, mu, tau)
    """

    n_evts = astroflux_nominal.shape[0]

    for event in range(n_evts):
        spec_scale = spectral_index_scale(true_energy[event], delta_index)
        out[event] = norm * astroflux_nominal[event] * spec_scale
