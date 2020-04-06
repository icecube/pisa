# pylint: disable = invalid-name


"""
Host function wrappers for numba_osc_kernels.

These allow calling any kernel defined there (using SmartArray and approrpiate
types as defined in signatures here) from a host, whether TARGET is "host" or
"cuda".
"""


from __future__ import absolute_import, print_function, division

__all__ = ["FX", "CX", "IX", "propagate_array", "fill_probs"]

import numpy as np
from numba import guvectorize, njit

from pisa import FTYPE, ITYPE, TARGET
from pisa.stages.osc.prob3numba.numba_osc_kernels import (
    # osc_probs_vacuum_kernel,
    osc_probs_layers_kernel,
    get_transition_matrix,
    get_transition_matrix_massbasis,
    get_H_vac,
    get_H_mat,
    get_dms,
    get_product,
    convert_from_mass_eigenstate,
)


assert FTYPE in [np.float32, np.float64], str(FTYPE)

FX = "f4" if FTYPE == np.float32 else "f8"
"""Float string code to use, understood by both Numba and Numpy"""

CX = "c8" if FTYPE == np.float32 else "c16"
"""Complex string code to use, understood by both Numba and Numpy"""

IX = "i4" if ITYPE == np.int32 else "i8"
"""Signed integer string code to use, understood by both Numba and Numpy"""


# @guvectorize(
#    [f"({FX}[:,:], {CX}[:,:], {IX}, {FX}, {FX}[:], {FX}[:,:])"],
#    "(a,a), (a,a), (), (), (i) -> (a,a)",
#    target=TARGET,
# )
# def propagate_array_vacuum(dm, mix, nubar, energy, distances, probability):
#    """wrapper to run `osc_probs_vacuum_kernel` from host (whether TARGET is
#    "cuda" or "host")"""
#    osc_probs_vacuum_kernel(dm, mix, nubar, energy, distances, probability)
#
#
# @njit([f"({FX}[:,:], {CX}[:,:], {IX}, {FX}, {FX}[:], {FX}[:,:])"], target=TARGET)
# def propagate_scalar_vacuum(dm, mix, nubar, energy, distances, probability):
#    """wrapper to run `osc_probs_vacuum_kernel` from host (whether TARGET is
#    "cuda" or "host")"""
#    osc_probs_vacuum_kernel(dm, mix, nubar, energy, distances, probability)


@guvectorize(
    [f"({FX}[:,:], {CX}[:,:], {CX}[:,:], {IX}, {FX}, {FX}[:], {FX}[:], {FX}[:,:])"],
    "(a,a), (a,a), (b,c), (), (), (i), (i) -> (a,a)",
    target=TARGET,
)
def propagate_array(dm, mix, mat_pot, nubar, energy, densities, distances, probability):
    """wrapper to run `osc_probs_layers_kernel` from host (whether TARGET
    is "cuda" or "host")"""
    osc_probs_layers_kernel(
        dm, mix, mat_pot, nubar, energy, densities, distances, probability
    )


@njit(
    [f"({FX}[:,:], {CX}[:,:], {CX}[:,:], {IX}, {FX}, {FX}[:], {FX}[:], {FX}[:,:])"],
    target=TARGET,
)
def propagate_scalar(
    dm, mix, mat_pot, nubar, energy, densities, distances, probability
):
    """wrapper to run `osc_probs_layers_kernel` from host (whether TARGET
    is "cuda" or "host")"""
    osc_probs_layers_kernel(
        dm, mix, mat_pot, nubar, energy, densities, distances, probability
    )


@njit(
    [
        "("
        f"{IX}, "  # nubar
        f"{FX}, "  # energy
        f"{FX}, "  # rho
        f"{FX}, "  # baseline
        f"{CX}[:,:], "  # mix_nubar
        f"{CX}[:,:], "  # mix_nubar_conj_transp
        f"{CX}[:,:], "  # mat_pot
        f"{CX}[:,:], "  # H_vac
        f"{FX}[:,:], "  # dm
        f"{CX}[:,:], "  # transition_matrix
        ")"
    ],
    target=TARGET,
)
def get_transition_matrix_hostfunc(
    nubar,
    energy,
    rho,
    baseline,
    mix_nubar,
    mix_nubar_conj_transp,
    mat_pot,
    H_vac,
    dm,
    transition_matrix,
):
    """wrapper to run `get_transition_matrix` from host (whether TARGET is
    "cuda" or "host")"""
    get_transition_matrix(
        nubar,
        energy,
        rho,
        baseline,
        mix_nubar,
        mix_nubar_conj_transp,
        mat_pot,
        H_vac,
        dm,
        transition_matrix,
    )


@njit([f"({FX}, {FX}, {CX}[:,:], {CX}[:,:], {CX}[:,:], {CX}[:,:])"], target=TARGET)
def get_transition_matrix_massbasis_hostfunc(
    baseline,
    energy,
    dm_mat_vac,
    dm_mat_mat,
    H_mat_mass_eigenstate_basis,
    transition_matrix,
):
    """wrapper to run `get_transition_matrix_massbasis` from host (whether
    TARGET is "cuda" or "host")"""
    get_transition_matrix_massbasis(
        baseline,
        energy,
        dm_mat_vac,
        dm_mat_mat,
        H_mat_mass_eigenstate_basis,
        transition_matrix,
    )


@njit([f"({CX}[:,:], {CX}[:,:], {FX}[:,:], {CX}[:,:])"], target=TARGET)
def get_H_vac_hostfunc(mix_nubar, mix_nubar_conj_transp, dm_vac_vac, H_vac):
    """wrapper to run `get_H_vac` from host (whether TARGET is "cuda" or "host")"""
    get_H_vac(mix_nubar, mix_nubar_conj_transp, dm_vac_vac, H_vac)


# @guvectorize(
#     [f"({FX}, {CX}[:,:], {IX}, {CX}[:,:])"], "(), (m, m), () -> (m, m)", target=TARGET
# )
@njit([f"({FX}, {CX}[:,:], {IX}, {CX}[:,:])"], target=TARGET)
def get_H_mat_hostfunc(rho, mat_pot, nubar, H_mat):
    """wrapper to run `get_H_mat` from host (whether TARGET is "cuda" or "host")"""
    get_H_mat(rho, mat_pot, nubar, H_mat)


@njit([f"({FX}, {CX}[:,:], {FX}[:,:], {CX}[:,:], {CX}[:,:])"], target=TARGET)
def get_dms_hostfunc(energy, H_mat, dm_vac_vac, dm_mat_mat, dm_mat_vac):
    """wrapper to run `get_dms` from host (whether TARGET is "cuda" or "host")"""
    get_dms(energy, H_mat, dm_vac_vac, dm_mat_mat, dm_mat_vac)


@njit([f"({FX}, {CX}[:,:], {CX}[:,:], {CX}[:,:], {CX}[:,:,:])"], target=TARGET)
def get_product_hostfunc(
    energy, dm_mat_vac, dm_mat_mat, H_mat_mass_eigenstate_basis, product
):
    """wrapper to run `get_product` from host (whether TARGET is "cuda" or "host")"""
    get_product(energy, dm_mat_vac, dm_mat_mat, H_mat_mass_eigenstate_basis, product)


@njit([f"({IX}, {CX}[:,:], {CX}[:])"], target=TARGET)
def convert_from_mass_eigenstate_hostfunc(state, mix_nubar, psi):
    """wrapper to run `convert_from_mass_eigenstate` from host (whether TARGET
    is "cuda" or "host")"""
    convert_from_mass_eigenstate(state, mix_nubar, psi)


@guvectorize(
    [f"({FX}[:,:], {IX}, {IX}, {FX}[:])"], "(a,b), (), () -> ()", target=TARGET
)
def fill_probs(probability, initial_flav, flav, out):
    """Fill `out` with transition probabilities to go from `initial_flav` to
    `flav`, from values in `probaility` matrix.

    Parameters
    ----------
    probability : real 2d array
    initial_flav : signed(?) int (int4 or int8)
    flav : signed(?) int (int4 or int8)
    out : real 1-d array

    """
    out[0] = probability[initial_flav, flav]
