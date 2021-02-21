"""
Stage to implement the inclusive neutrino flux as calculated with MCEq,
and the systematic flux variations based on the Barr scheme.

It requires spline tables created by the `pisa/scripts/create_barr_sys_tables_mceq.py`
Pre-generated tables can be found at `fridge/analysis/common/data/flux/

Tom Stuttard, Ida Storehaug, Philipp Eller, Summer Blot
"""

from __future__ import absolute_import, print_function, division

from bz2 import BZ2File
import collections
import pickle

import numpy as np
from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile, line_profile
from pisa.utils.numba_tools import WHERE, myjit
from pisa.utils.resources import find_resource


class mceq_barr_red(Stage):
    """
    Stage that uses gradients calculated with MCEq to handle flux uncertainties.
    This stage calculates flux variations relative to a nominal flux that must
    be calculated ahead of time. Parameters to control pion and kaon production
    are reduced from the full Barr scheme, by forcing many of them to scale in a
    fully correlated way.

    Parameters
    ----------
    table_file : pickle file containing pre-generated tables from MCEq

    params : ParamSet
        Must exclusively have parameters: .. ::

            delta_index : quantity (dimensionless)
                Shift in the spectral index of the neutrino flux. Prior with a mean of 0.

            energy_pivot : quantity (GeV)
                The spectral index is shifting around a pivot point

            barr_*_Pi : quantity (dimensionless)
                * from a to i
                Uncertainty on pi+ production in a region of phase space *,
                further defined in Barr 2006

            pion_ratio : quantity (dimensionless)
                The uncertainty on pi- production is assumed to be correlated
                to the pi+ production uncertainty,
                as the pi+/pi- ratio is measured. Thus the uncertainty on pi-
                production is defined by pion_ratio and barr_*_pi

            barr_*_K : quantity (dimensionless)
                * from w to z
                Uncertainty on K+ production in a region of phase space *,
                further defined in Barr 2006

            barr_*_antiK : quantity (dimensionless)
                * from w to z
                Uncertainty on K- and K+ production is assumed to be
                uncorrelated as the ratio is badly determined.

    Notes
    -----
    The nominal flux is calculated ahead of time using the honda_ip stage,
    then multiplied with a shift in spectral index, and then modifications due
    to meson production (barr variables) are added.

    The MCEq-table has 2 solutions of the cascade equation per Barr variable (12)
    - one solution for meson and one solution for the antimeson production uncertainty.

    Each solution consists of 4 splines: "dnumu", "dnumubar", "dnue", and
    "dnuebar". These are the gradients that govern how the neutrino fluxes vary
    depending on modifications to the Barr params.

    """

    def __init__(
        self,
        table_file,
        **std_kwargs,
    ):
        #
        # Define parameterisation
        #

        # Define the Barr parameters
        # TODO Barr block definition could change - need to make this more flexible
        # Perhaps let user define the blocks, and then table is generated on the
        # fly as a first step? Could take up to 1 hour to produce table though...
        self.barr_param_names = [
            # pions
            "af",
            "g",
            "h",
            "i",
            # kaons
            "w",
            "x",
            "y",
            "z",
        ]

        # Define signs for Barr params
        # +  -> meson production
        # -  -> antimeson production
        self.barr_param_signs = ["+", "-"]

        # Atmopshere model params
        # TODO

        # Get the overall list of params for which we have gradients stored
        # Define a mapping to index values, will be useful later
        self.gradient_param_names = [
            n + s for n in self.barr_param_names for s in self.barr_param_signs
        ]
        self.gradient_param_indices = collections.OrderedDict(
            [(n, i) for i, n in enumerate(self.gradient_param_names)]
        )

        #
        # Call stage base class constructor
        #

        # Define stage parameters
        expected_params = (
            # pion
            "pion_ratio",
            "barr_af_Pi",
            "barr_g_Pi",
            "barr_h_Pi",
            "barr_i_Pi",
            # kaon
            "barr_w_K",
            "barr_x_K",
            "barr_y_K",
            "barr_z_K",
            "barr_w_antiK",
            "barr_x_antiK",
            "barr_y_antiK",
            "barr_z_antiK",
            # CR
            "delta_index",
            "energy_pivot",
        )

        # Using Honda for nominal flux. Keys should already exist
        # what are keys added or altered in the calculation used during apply
        # what keys are added or altered for the outputs during apply

        # store args
        self.table_file = table_file

        # init base class
        super(mceq_barr_red, self).__init__(
            expected_params=expected_params,
            **std_kwargs,
        )


    def setup_function(self):

        self.data.representation = self.calc_mode

        #
        # Init arrays
        #

        # Prepare some array shapes
        gradient_params_shape = (len(self.gradient_param_names),)

        if self.data.is_map:
            # speed up calculation by adding links
            # as nominal flux doesn't depend on the (outgoing) flavour
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc'])

            self.data.link_containers('nubar', ['nuebar_cc', 'numubar_cc',
                                                'nutaubar_cc', 'nuebar_nc',
                                                'numubar_nc', 'nutaubar_nc'])

        # Loop over containers
        for container in self.data:

            # Define shapes for containers

            # TODO maybe include toggles for nutau (only needed if prompt
            # considered) and for nu+nubar (only needed if nu->nubar
            # oscillations included) for better speed/memory performance

            # [ N events, 2 flavors in flux, nu vs nubar ]
            # SDB - reduced flavours to 2 (nue, numu) since nutau flux not
            # stored in MCEq splines
            flux_container_shape = (container.size, 2)
            gradients_shape = tuple(
                list(flux_container_shape) + list(gradient_params_shape)
            )

            container["nu_flux"] = np.full(flux_container_shape, np.NaN, dtype=FTYPE)
            container["gradients"] = np.full(gradients_shape, np.NaN, dtype=FTYPE)

        # Also create an array container to hold the gradient parameter values
        # Only want this once, e.g. not once per container
        self.gradient_params = np.empty(gradient_params_shape, dtype=FTYPE)

        #
        # Load MCEq splines
        #

        # Have splines for each Barr parameter, plus +/- versions of each
        # Barr parameter corresponding to mesons/antimesons.

        # For a given Barr parameter, an underlying dictionary have the following
        # keywords: "dnumu", "dnumubar", "dnue", dnuebar"

        # Units are changed to m^-2 in creates_splines.., rather than cm^2 which
        # is the unit of calculation in MCEq!!!!

        # Note that doing this all on CPUs, since the splines reside on the CPUs
        # The actual `compute_function` computation can be done on GPUs though

        # Load the MCEq splines
        spline_file = find_resource(self.table_file)
        logging.info("Loading MCEq spline tables from : %s", spline_file)
        # Encoding is to support pickle files created with python v2
        self.spline_tables_dict = pickle.load(BZ2File(spline_file), encoding="latin1")

        # Ensure that the user is not loading an incompatible spline
        for bp in self.barr_param_names:
            bp_p = bp+'+' # meson
            bp_m = bp+'-' # antimeson
            assert bp_p in self.spline_tables_dict.keys(), (
                "Gradient parameter '%s' missing from table" % bp_p
            )
            assert bp_m in self.spline_tables_dict.keys(), (
                "Gradient parameter '%s' missing from table" % bp_m
            )


        # Loop over containers
        for container in self.data:

            # Grab containers here once to save time
            # TODO make spline generation script store splines directly in
            # terms of energy, not ln(energy)
            true_log_energy = np.log(container["true_energy"])
            true_abs_coszen = np.abs(container["true_coszen"])
            gradients = container["gradients"]
            nubar = container["nubar"]

            #
            # Flux gradients
            #

            # Evaluate splines to get the flux graidents w.r.t the Barr parameter values
            # Need to correctly map nu/nubar and flavor to the output arrays

            # Loop over parameters
            for (
                gradient_param_name,
                gradient_param_idx,
            ) in self.gradient_param_indices.items():

                # nue(bar)
                self._eval_spline(
                    true_log_energy=true_log_energy,
                    true_abs_coszen=true_abs_coszen,
                    spline=self.spline_tables_dict[gradient_param_name][
                        "dnue" if nubar > 0 else "dnuebar"
                    ],
                    out=gradients[:, 0, gradient_param_idx],
                )

                # numu(bar)
                self._eval_spline(
                    true_log_energy=true_log_energy,
                    true_abs_coszen=true_abs_coszen,
                    spline=self.spline_tables_dict[gradient_param_name][
                        "dnumu" if nubar > 0 else "dnumubar"
                    ],
                    out=gradients[:, 1, gradient_param_idx],
                )

                # nutau(bar)
                # TODO include nutau flux in splines
                # SDB - there is no nutau flux in splines
                ## gradients[:, 2, gradient_param_idx].fill(0.0)

            # Tell the smart arrays we've changed the flux gradient values on the host
            container.mark_changed("gradients")

        # don't forget to un-link everything again
        self.data.unlink_containers()

    def _eval_spline(self, true_log_energy, true_abs_coszen, spline, out):
        """
        Evaluate the spline for the full arrays of [ ln(energy), abs(coszen) ] values
        """

        # Evalate the spine
        result = spline(true_abs_coszen, true_log_energy, grid=False)

        # Copy to output array
        # TODO Can I directly write to the original array, will be faster
        np.copyto(src=result, dst=out)

    def antipion_production(self, barr_var, pion_ratio):
        """
        Combine pi+ param and pi+/pi- ratio to get pi- param
        """
        return ((1 + barr_var) / (1 + pion_ratio)) - 1

    @profile
    def compute_function(self):

        self.data.representation = self.calc_mode

        if self.data.is_map:
            # speed up calculation by adding links
            # as nominal flux doesn't depend on the (outgoing) flavour
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc'])

            self.data.link_containers('nubar', ['nuebar_cc', 'numubar_cc',
                                                'nutaubar_cc', 'nuebar_nc',
                                                'numubar_nc', 'nutaubar_nc'])

        #
        # Get params
        #

        # Spectral index (and required energy pivot)
        delta_index = self.params.delta_index.value.m_as("dimensionless")
        energy_pivot = self.params.energy_pivot.value.m_as("GeV")

        # Grab the pion ratio
        pion_ratio = self.params.pion_ratio.value.m_as("dimensionless")

        # Map the user parameters into the Barr +/- params
        # pi- production rates is restricted by the pi-ratio, just as in arXiv:0611266
        # TODO might want dedicated priors for pi- params (but without corresponding free params)
        gradient_params_mapping = collections.OrderedDict()
        gradient_params_mapping["af+"] = self.params.barr_af_Pi.value.m_as("dimensionless")
        gradient_params_mapping["g+"] = self.params.barr_g_Pi.value.m_as("dimensionless")
        gradient_params_mapping["h+"] = self.params.barr_h_Pi.value.m_as("dimensionless")
        gradient_params_mapping["i+"] = self.params.barr_i_Pi.value.m_as("dimensionless")
        for k in list(gradient_params_mapping.keys()):
            gradient_params_mapping[k.replace("+", "-")] = self.antipion_production(
                gradient_params_mapping[k], pion_ratio
            )

        # kaons
        # as the kaon ratio is unknown, K- production is not restricted
        gradient_params_mapping["w+"] = self.params.barr_w_K.value.m_as("dimensionless")
        gradient_params_mapping["w-"] = self.params.barr_w_antiK.value.m_as("dimensionless")
        gradient_params_mapping["x+"] = self.params.barr_x_K.value.m_as("dimensionless")
        gradient_params_mapping["x-"] = self.params.barr_x_antiK.value.m_as("dimensionless")
        gradient_params_mapping["y+"] = self.params.barr_y_K.value.m_as("dimensionless")
        gradient_params_mapping["y-"] = self.params.barr_y_antiK.value.m_as("dimensionless")
        gradient_params_mapping["z+"] = self.params.barr_z_K.value.m_as("dimensionless")
        gradient_params_mapping["z-"] = self.params.barr_z_antiK.value.m_as("dimensionless")

        # Populate array Barr param array
        for (
            gradient_param_name,
            gradient_param_idx,
        ) in self.gradient_param_indices.items():
            self.gradient_params[gradient_param_idx] = gradient_params_mapping[
                gradient_param_name
            ]

        #
        # Loop over containers
        #

        for container in self.data:

            #
            # Apply the systematics to the flux
            #

            nubar = container["nubar"]
            if nubar > 0: flux_key = "nu_flux_nominal"
            elif nubar < 0: flux_key = "nubar_flux_nominal"
            
            apply_sys_loop(
                container["true_energy"],
                container["true_coszen"],
                FTYPE(delta_index),
                FTYPE(energy_pivot),
                container[flux_key],
                container["gradients"],
                self.gradient_params,
                out=container["nu_flux"],
            )
            container.mark_changed("nu_flux")

            # Check for negative results from spline
            # TODO - add more spline error/misusage handling
            # e.g. if events have energy outside spline range throw ERROR
            negative_mask = container["nu_flux"] < 0
            if np.any(negative_mask):
                container["nu_flux"][negative_mask] = 0.0

            container.mark_changed("nu_flux")

        # don't forget to un-link everything again
        self.data.unlink_containers()

@myjit
def spectral_index_scale(true_energy, energy_pivot, delta_index):
    """
      Calculate spectral index scale.
      Adjusts the weights for events in an energy dependent way according to a
      shift in spectral index, applied about a user-defined energy pivot.
      """
    return np.power((true_energy / energy_pivot), delta_index)

@myjit
def apply_sys_loop(
    true_energy,
    true_coszen,
    delta_index,
    energy_pivot,
    nu_flux_nominal,
    gradients,
    gradient_params,
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
        nubar : scalar integer
        delta_index : scalar float
        energy_pivot : scalar float
        nu_flux_nominal : [A,B]
        gradients : [A,B,C]
        gradient_params : [C]
        out : [A,B] (sys flux)
    where:
        A = num events
        B = num flavors in flux (=3, e.g. e, mu, tau)
        C = num gradients
    Not that first dimension (of length A) is vectorized out
    """

    n_evts, n_flavs = nu_flux_nominal.shape

    for event in range(n_evts):
        spec_scale = spectral_index_scale(true_energy[event], energy_pivot, delta_index)
        for flav in range(n_flavs):
            out[event, flav] = nu_flux_nominal[event, flav] * spec_scale
            for i in range(len(gradient_params)):
                out[event, flav] += gradients[event, flav, i] * gradient_params[i]
