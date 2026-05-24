"""
Stage to implement the old PISA/oscfit flux systematics
"""

from __future__ import absolute_import, print_function, division

import math

import numpy as np
from numba import guvectorize, cuda, jit, njit, void

from pisa import FTYPE, TARGET
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import myjit, myjit_2, ftype, FX, IX
from pisa.utils.barr_parameterization import modRatioNuBar, modRatioUpHor

__all__ = ['barr_simple', 'apply_ratio_scale', 'spectral_index_scale',
            'apply_sys_kernel', 'apply_sys_vectorized', 'init_test']


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

        Expected container keys are .. ::

            "true_energy"
            "true_coszen"
            "nu_flux_nominal"
            "nubar_flux_nominal"
            "nubar"

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

        expected_container_keys = (
            "true_energy",
            "true_coszen",
            "nu_flux_nominal",
            "nubar_flux_nominal",
            "nubar"
        )

        # init base class
        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )

    def setup_function(self):
        self.data.representation = self.calc_mode
        for container in self.data:
            container["nu_flux"] = np.empty((container.size, 2), dtype=FTYPE)

    @profile
    def compute_function(self):
        self.data.representation = self.calc_mode

        # If a parameter below has not been modified by minimizer,
        # its magnitude below will be a native float, otherwise np.float64,
        # so convert to expected FTYPE
        nue_numu_ratio = FTYPE(self.params.nue_numu_ratio.value.m_as("dimensionless"))
        nu_nubar_ratio = FTYPE(self.params.nu_nubar_ratio.value.m_as("dimensionless"))
        delta_index = FTYPE(self.params.delta_index.value.m_as("dimensionless"))
        Barr_uphor_ratio = FTYPE(self.params.Barr_uphor_ratio.value.m_as("dimensionless"))
        Barr_nu_nubar_ratio = FTYPE(self.params.Barr_nu_nubar_ratio.value.m_as("dimensionless"))

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


#@njit(f"void({FX}, b1, {FX}, {FX}, {FX}[:])", inline='always')
@myjit_2(f"void({FX}, b1, {FX}, {FX}, {FX}[:])", inline="always")
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

    flag = 1 if sum_constant else 0

    # when sum_constant = True
    # just a single division operation
    single_denom = in2 + ratio_scale * in1
    inv_denom = 1.0 / single_denom

    orig_sum = in1 + in2
    new = orig_sum * inv_denom
    out0 = ratio_scale * in1 * new
    out1 = new

    # sum_constant = False
    out0_sum_not_const = ratio_scale * in1
    out1_sum_not_const = in2

    # finally assign
    out[0] = (1 - flag) * out0_sum_not_const + flag * out0
    out[1] = (1 - flag) * out1_sum_not_const + flag * out1


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


@guvectorize([f"({FX}, {FX}, {FX}[:], {FX}[:], {IX}, {FX}, {FX}, {FX}, {FX}, {FX}, {FX}[:])"], "(),(),(d),(d),(),(),(),(),(),()->(d)", target=TARGET)
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
    # nue/numu ratio
    new_nu_flux = np.empty(shape=(2), dtype=ftype)
    new_nubar_flux = np.empty(2, dtype=ftype)
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
    #idx_scale = spectral_index_scale(true_energy, 24.0900951261, delta_index)
    energy_pivot = 24.0900951261
    idx_scale = math.exp(delta_index * math.log(true_energy / energy_pivot))
    new_nu_flux[0] *= idx_scale
    new_nu_flux[1] *= idx_scale
    new_nubar_flux[0] *= idx_scale
    new_nubar_flux[1] *= idx_scale

    # nu/nubar ratio
    new_nue_flux = np.empty(2, dtype=ftype)
    new_numu_flux = np.empty(2, dtype=ftype)
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


def init_test(**param_kwargs):
    """Instantiation example"""
    param_set = ParamSet([
        Param(name="nue_numu_ratio", value=1.0, **param_kwargs),
        Param(name="nu_nubar_ratio", value=1.0, **param_kwargs),
        Param(name="delta_index", value=0.0, **param_kwargs),
        Param(name="Barr_uphor_ratio", value=0.0, **param_kwargs),
        Param(name="Barr_nu_nubar_ratio", value=0.0, **param_kwargs)
    ])

    return barr_simple(params=param_set)
