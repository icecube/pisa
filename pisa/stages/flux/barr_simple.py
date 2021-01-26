# pylint: disable=not-callable, wrong-import-position

"""
Stage to implement the old PISA/oscfit flux systematics
"""

from __future__ import absolute_import, print_function, division

import math
import os
import sys

import numpy as np
from numba import guvectorize, cuda

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import WHERE, myjit, ftype
from pisa.utils.resources import find_resource
from pisa.utils.barr_parameterization import modRatioNuBar, modRatioUpHor


class barr_simple(Stage):  # pylint: disable=invalid-name
    """
    stage to apply Barr style flux uncertainties
    uses parameterisations of plots from Barr 2006 paper

    Parameters
    ----------
    params
        Expected params are .. ::

            nue_numu_ratio : quantity (dimensionless)
            nu_nubar_ratio : quantity (dimensionless)
            delta_index : quantity (dimensionless)
            Barr_uphor_ratio : quantity (dimensionless)
            Barr_nu_nubar_ratio : quantity (dimensionless)

    """
    def __init__(
        self,
        **std_kwargs,
    ):
        expected_params = (
            "nue_numu_ratio",
            "nu_nubar_ratio",
            "delta_index",
            "Barr_uphor_ratio",
            "Barr_nu_nubar_ratio",
        )

        # init base class
        super(barr_simple, self).__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

    def setup_function(self):
        self.data.represenatation = self.calc_mode
        for container in self.data:
            container["nu_flux"] = np.empty((container.size, 2), dtype=FTYPE)

    @profile
    def compute_function(self):
        self.data.represenatation = self.calc_mode

        nue_numu_ratio = self.params.nue_numu_ratio.value.m_as("dimensionless")
        nu_nubar_ratio = self.params.nu_nubar_ratio.value.m_as("dimensionless")
        delta_index = self.params.delta_index.value.m_as("dimensionless")
        Barr_uphor_ratio = self.params.Barr_uphor_ratio.value.m_as("dimensionless")
        Barr_nu_nubar_ratio = self.params.Barr_nu_nubar_ratio.value.m_as("dimensionless")

        for container in self.data:
            apply_sys_vectorized(
                container["true_energy"],
                container["true_coszen"],
                container["nu_flux_nominal"],
                container["nubar_flux_nominal"],
                container["nubar"],
                nue_numu_ratio,
                nu_nubar_ratio,
                delta_index,
                Barr_uphor_ratio,
                Barr_nu_nubar_ratio,
                out=container["nu_flux"],
            )
            container.mark_changed('nu_flux')


@myjit
def apply_ratio_scale(ratio_scale, sum_constant, in1, in2, out):
    """ apply ratio scale to flux values

    Parameters
    ----------
    ratio_scale : float

    sum_constant : bool
        if Ture, then the sum of the new flux will be identical to the old flux

    in1 : float

    in2 : float

    out : array

    """
    if in1 == 0. and in2 == 0.:
        out[0] = 0.
        out[1] = 0.
        return

    if sum_constant:
        orig_ratio = in1 / in2
        orig_sum = in1 + in2
        new = orig_sum / (1. + ratio_scale * orig_ratio)
        out[0] = ratio_scale * orig_ratio * new
        out[1] = new
    else:
        out[0] = ratio_scale * in1
        out[1] = in2


@myjit
def spectral_index_scale(true_energy, egy_pivot, delta_index):
    """ calculate spectral index scale """
    return math.pow((true_energy / egy_pivot), delta_index)


@myjit
def apply_sys_kernel(
    true_energy,
    true_coszen,
    nu_flux_nominal,
    nubar_flux_nominal,
    nubar,
    nue_numu_ratio,
    nu_nubar_ratio,
    delta_index,
    Barr_uphor_ratio,
    Barr_nu_nubar_ratio,
    out,
):
    # nue/numu ratio
    new_nu_flux = cuda.local.array(shape=(2), dtype=ftype)
    new_nubar_flux = cuda.local.array(2, dtype=ftype)
    apply_ratio_scale(
        nue_numu_ratio, True, nu_flux_nominal[0], nu_flux_nominal[1], new_nu_flux
    )
    apply_ratio_scale(
        nue_numu_ratio,
        True,
        nubar_flux_nominal[0],
        nubar_flux_nominal[1],
        new_nubar_flux,
    )

    # apply flux systematics
    # spectral idx
    idx_scale = spectral_index_scale(true_energy, 24.0900951261, delta_index)
    new_nu_flux[0] *= idx_scale
    new_nu_flux[1] *= idx_scale
    new_nubar_flux[0] *= idx_scale
    new_nubar_flux[1] *= idx_scale

    # nu/nubar ratio
    new_nue_flux = cuda.local.array(2, dtype=ftype)
    new_numu_flux = cuda.local.array(2, dtype=ftype)
    apply_ratio_scale(
        nu_nubar_ratio, True, new_nu_flux[0], new_nubar_flux[0], new_nue_flux
    )
    apply_ratio_scale(
        nu_nubar_ratio, True, new_nu_flux[1], new_nubar_flux[1], new_numu_flux
    )
    if nubar < 0:
        out[0] = new_nue_flux[1]
        out[1] = new_numu_flux[1]
    else:
        out[0] = new_nue_flux[0]
        out[1] = new_numu_flux[0]

    # Barr flux
    out[0] *= modRatioNuBar(nubar, 0, true_energy, true_coszen, Barr_nu_nubar_ratio)
    out[1] *= modRatioNuBar(nubar, 1, true_energy, true_coszen, Barr_nu_nubar_ratio)

    out[0] *= modRatioUpHor(0, true_energy, true_coszen, Barr_uphor_ratio)
    out[1] *= modRatioUpHor(1, true_energy, true_coszen, Barr_uphor_ratio)


# vectorized function to apply
# must be outside class
if FTYPE == np.float64:
    SIGNATURE = "(f8, f8, f8[:], f8[:], i4, f8, f8, f8, f8, f8, f8[:])"
else:
    SIGNATURE = "(f4, f4, f4[:], f4[:], i4, f4, f4, f4, f4, f4, f4[:])"


@guvectorize([SIGNATURE], "(),(),(d),(d),(),(),(),(),(),()->(d)", target=TARGET)
def apply_sys_vectorized(
    true_energy,
    true_coszen,
    nu_flux_nominal,
    nubar_flux_nominal,
    nubar,
    nue_numu_ratio,
    nu_nubar_ratio,
    delta_index,
    Barr_uphor_ratio,
    Barr_nu_nubar_ratio,
    out,
):
    apply_sys_kernel(
        true_energy,
        true_coszen,
        nu_flux_nominal,
        nubar_flux_nominal,
        nubar,
        nue_numu_ratio,
        nu_nubar_ratio,
        delta_index,
        Barr_uphor_ratio,
        Barr_nu_nubar_ratio,
        out,
    )
