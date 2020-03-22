# pylint: disable = not-callable, bad-whitespace, invalid-name
"""Neutrino flavour oscillation in matter calculation
Based on the original prob3++ implementation of Roger Wendell
http://www.phy.duke.edu/~raw22/public/Prob3++/ (2012)
"""
from __future__ import absolute_import, print_function, division

__all__ = [
    "PMNS_NUFIT32",
    "get_transition_matrix",
    "osc_probs_vacuum_kernel",
    "propagate_array_vacuum",  # host-callable `osc_probs_vacuum_kernel`
    "osc_probs_layers_kernel",
    "propagate_array",  # host-callable `osc_probs_layers_kernel`
    "fill_probs",
]

__version__ = "0.2"

import cmath
import math

import numpy as np
from numba import guvectorize, njit, SmartArray

from pisa import FTYPE, ITYPE, TARGET
from pisa.utils.comparisons import ALLCLOSE_KW
from pisa.utils.numba_tools import (
    WHERE,
    myjit,
    conjugate_transpose,
    conjugate,
    matrix_dot_matrix,
    matrix_dot_vector,
    clear_matrix,
    copy_matrix,
    cuda,
    ctype,
    ftype,
)


ALLCLOSE_KW = dict(atol=np.finfo(FTYPE).resolution, rtol=ALLCLOSE_KW["rtol"] * 100)

assert FTYPE in [np.float32, np.float64], str(FTYPE)

FX = "f4" if FTYPE == np.float32 else "f8"
"""Float string code to use, understood by both Numba and Numpy"""

CX = "c8" if FTYPE == np.float32 else "c16"
"""Complex string code to use, understood by both Numba and Numpy"""

IX = "i4" if ITYPE == np.int32 else "i8"
"""Signed integer string code to use, understood by both Numba and Numpy"""
# ICODES = ["i8"]
# """string codes of all signed ints, for function sigs which do not care"""


# ---------------------------------------------------------------------------- #


@myjit
def get_H_vac(mix_nubar, mix_nubar_conj_transp, dm_vac_vac, H_vac):
    """ Calculate vacuum Hamiltonian in flavor basis for neutrino or antineutrino

    Parameters:
    -----------
    mix_nubar : complex 2d-array
        Mixing matrix (comjugate for anti-neutrinos)

    mix_nubar_conj_transp : comjugate 2d-array
        conjugate transpose of mixing matrix

    dm_vac_vac: 2d-array
        Matrix of mass splittings

    H_vac: complex 2d-array (empty)
        Hamiltonian in vacuum, modulo a factor 2 * energy

    Notes
    ------
    The Hailtonian does not contain the energy dependent factor of
    1/(2 * E), as it will be added later

    """
    dm_vac_diag = cuda.local.array(shape=(3, 3), dtype=ctype)
    tmp = cuda.local.array(shape=(3, 3), dtype=ctype)

    clear_matrix(dm_vac_diag)

    dm_vac_diag[1, 1] = dm_vac_vac[1, 0] + 0j
    dm_vac_diag[2, 2] = dm_vac_vac[2, 0] + 0j

    matrix_dot_matrix(dm_vac_diag, mix_nubar_conj_transp, tmp)
    matrix_dot_matrix(mix_nubar, tmp, H_vac)


@njit(
    [f"({CX}[:,:], {CX}[:,:], {FX}[:,:], {CX}[:,:])"], target=TARGET,
)
def get_H_vac_hostfunc(mix_nubar, mix_nubar_conj_transp, dm_vac_vac, H_vac):
    """wrapper to run `get_H_vac` from host (whether TARGET is "cuda" or "host")"""
    get_H_vac(mix_nubar, mix_nubar_conj_transp, dm_vac_vac, H_vac)


def test_get_H_vac():
    """unit tests for get_H_vac / get_H_vac_hostfunc"""
    # inputs
    mix = SmartArray(PMNS_NUFIT32)
    mix_nubar_conj_transp = SmartArray(PMNS_NUFIT32.conj().T)
    dm_vac_vac = SmartArray(DM)

    # output
    H_vac = SmartArray(np.ones(shape=(3, 3), dtype=CX))

    get_H_vac_hostfunc(
        mix.get(WHERE),
        mix_nubar_conj_transp.get(WHERE),
        dm_vac_vac.get(WHERE),
        H_vac.get(WHERE),
    )
    H_vac.mark_changed(WHERE)

    # Ref retrieved via PISA_FTYPE=fp64 PISA_TARGET=cpu 2020-03-21 .. ::
    #
    #   print("H_VAC_REF")
    #   print(np.array2string(H_vac.get(), precision=20, separator=", "))
    #
    ref = H_VAC_REF
    test = H_vac.get()
    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"


# ---------------------------------------------------------------------------- #


@myjit
def get_H_mat(rho, nsi_eps, nubar, H_mat):
    """ Calculate matter Hamiltonian in flavor basis

    Parameters:
    -----------
    rho : float
        density

    nsi_eps : complex 2-d array
        Non-standard interaction terms

    nubar : int
        +1 for neutrinos, -1 for antineutrinos

    H_mat : complex 2d-array (empty)

    Notes
    -----
    in the following, `a` is just the standard effective matter potential
    induced by charged-current weak interactions with electrons

    """

    # 2*sqrt(2)*Gfermi in (eV^2-cm^3)/(mole-GeV)
    tworttwoGf = 1.52588e-4
    a = 0.5 * rho * tworttwoGf
    if nubar == -1:
        a = -a

    # standard matter interaction Hamiltonian
    clear_matrix(H_mat)
    H_mat[0, 0] = a

    # Obtain effective non-standard matter interaction Hamiltonian
    nsi_rho_scale = (
        3.0  # // assume 3x electron density for "NSI"-quark (e.g., d) density
    )
    fact = nsi_rho_scale * a
    for i in range(3):
        for j in range(3):
            H_mat[i, j] += fact * nsi_eps[i, j]


@guvectorize(
    [f"({FX}, {CX}[:,:], {IX}, {CX}[:,:])"], "(), (m, m), () -> (m, m)", target=TARGET,
)
def get_H_mat_hostfunc(rho, nsi_eps, nubar, H_mat):
    """wrapper to run `get_H_mat` from host (whether TARGET is "cuda" or "host")"""
    get_H_mat(rho, nsi_eps, nubar, H_mat)


def test_get_H_mat():
    """unit tests for `get_H_mat` and `get_H_mat_hostfunc`"""
    # inputs
    rho = RHO_REF
    nubar = NUBAR_REF
    nsi_eps = SmartArray(NSI_EPS_REF)

    # output
    H_mat = SmartArray(np.ones(shape=(3, 3), dtype=CX))

    get_H_mat_hostfunc(rho, nsi_eps.get(WHERE), nubar, H_mat.get(WHERE))
    H_mat.mark_changed(WHERE)

    # Ref retrieved via PISA_FTYPE=fp64 PISA_TARGET=cpu 2020-03-21 .. ::
    #
    #   print("H_MAT_REF")
    #   print(np.array2string(H_mat.get(), precision=20, separator=", "))
    #
    ref = H_MAT_REF
    test = H_mat.get()
    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"


# ---------------------------------------------------------------------------- #


@myjit
def get_dms(energy, H_mat, dm_vac_vac, dm_mat_mat, dm_mat_vac):
    """Compute the matter-mass vector M, dM = M_i-M_j and dMimj

    Parameters
    ----------
    energy : float
        Neutrino energy

    H_mat : complex 2d-array
        matter hamiltonian

    dm_vac_vac : 2d array

    dm_mat_mat : complex 2d-array (empty)

    dm_mat_vac : complex 2d-array (empty)


    Notes
    -----
    Calculate mass eigenstates in matter
    neutrino or anti-neutrino (type already taken into account in Hamiltonian)
    of energy energy.

    - only god knows what happens in this function, somehow it seems to work

    """

    real_product_a = (H_mat[0, 1] * H_mat[1, 2] * H_mat[2, 0]).real
    real_product_b = (H_mat[0, 0] * H_mat[1, 1] * H_mat[2, 2]).real

    norm_H_e_mu_sq = H_mat[0, 1].real ** 2 + H_mat[0, 1].imag ** 2
    norm_H_e_tau_sq = H_mat[0, 2].real ** 2 + H_mat[0, 2].imag ** 2
    norm_H_mu_tau_sq = H_mat[1, 2].real ** 2 + H_mat[1, 2].imag ** 2

    c1 = (
        (H_mat[0, 0].real * (H_mat[1, 1] + H_mat[2, 2])).real
        - (H_mat[0, 0].imag * (H_mat[1, 1] + H_mat[2, 2])).imag
        + (H_mat[1, 1].real * H_mat[2, 2]).real
        - (H_mat[1, 1].imag * H_mat[2, 2]).imag
        - norm_H_e_mu_sq
        - norm_H_mu_tau_sq
        - norm_H_e_tau_sq
    )

    c0 = (
        H_mat[0, 0].real * norm_H_mu_tau_sq
        + H_mat[1, 1].real * norm_H_e_tau_sq
        + H_mat[2, 2].real * norm_H_e_mu_sq
        - 2.0 * real_product_a
        - real_product_b
    )

    c2 = -H_mat[0, 0].real - H_mat[1, 1].real - H_mat[2, 2].real

    one_over_two_e = 0.5 / energy
    one_third = 1.0 / 3.0
    two_third = 2.0 / 3.0

    x = dm_vac_vac[1, 0]
    y = dm_vac_vac[2, 0]

    c2_v = -one_over_two_e * (x + y)

    p = c2 ** 2 - 3.0 * c1
    p_v = one_over_two_e ** 2 * (x ** 2 + y ** 2 - x * y)
    p = max(0.0, p)

    q = -13.5 * c0 - c2 ** 3 + 4.5 * c1 * c2
    q_v = one_over_two_e ** 3 * (x + y) * ((x + y) ** 2 - 4.5 * x * y)

    tmp = p ** 3 - q ** 2
    tmp_v = p_v ** 3 - q_v ** 2

    tmp = max(0.0, tmp)

    theta = cuda.local.array(shape=(3), dtype=ftype)
    theta_v = cuda.local.array(shape=(3), dtype=ftype)
    m_mat = cuda.local.array(shape=(3), dtype=ftype)
    m_mat_u = cuda.local.array(shape=(3), dtype=ftype)
    m_mat_v = cuda.local.array(shape=(3), dtype=ftype)

    a = two_third * math.pi
    res = math.atan2(math.sqrt(tmp), q) * one_third
    theta[0] = res + a
    theta[1] = res - a
    theta[2] = res
    res_v = math.atan2(math.sqrt(tmp_v), q_v) * one_third
    theta_v[0] = res_v + a
    theta_v[1] = res_v - a
    theta_v[2] = res_v

    b = two_third * math.sqrt(p)
    b_v = two_third * math.sqrt(p_v)

    for i in range(3):
        m_mat_u[i] = (
            2.0 * energy * (b * math.cos(theta[i]) - c2 * one_third + dm_vac_vac[0, 0])
        )
        m_mat_v[i] = (
            2.0
            * energy
            * (b_v * math.cos(theta_v[i]) - c2_v * one_third + dm_vac_vac[0, 0])
        )

    # Sort according to which reproduce the vaccum eigenstates
    for i in range(3):
        tmp_v = abs(dm_vac_vac[i, 0] - m_mat_v[0])
        k = 0
        for j in range(3):
            tmp = abs(dm_vac_vac[i, 0] - m_mat_v[j])
            if tmp < tmp_v:
                k = j
                tmp_v = tmp
        m_mat[i] = m_mat_u[k]

    for i in range(3):
        for j in range(3):
            dm_mat_mat[i, j] = m_mat[i] - m_mat[j]
            dm_mat_vac[i, j] = m_mat[i] - dm_vac_vac[j, 0]


@njit([f"({FX}, {CX}[:,:], {FX}[:,:], {CX}[:,:], {CX}[:,:])"], target=TARGET)
def get_dms_hostfunc(energy, H_mat, dm_vac_vac, dm_mat_mat, dm_mat_vac):
    """wrapper to run `get_dms` from host (whether TARGET is "cuda" or "host")"""
    get_dms(energy, H_mat, dm_vac_vac, dm_mat_mat, dm_mat_vac)


def test_get_dms():
    """unit tests for `get_dms`, `get_dms_hostfunc`"""
    # inputs
    energy = ENERGY_REF
    H_mat = SmartArray(H_MAT_REF)
    dm_vac_vac = SmartArray(DM)

    # outputs
    dm_mat_mat = SmartArray(np.ones(shape=(3, 3), dtype=CX))
    dm_mat_vac = SmartArray(np.ones(shape=(3, 3), dtype=CX))

    get_dms_hostfunc(
        energy,
        H_mat.get(WHERE),
        dm_vac_vac.get(WHERE),
        dm_mat_mat.get(WHERE),
        dm_mat_vac.get(WHERE),
    )
    dm_mat_mat.mark_changed(WHERE)
    dm_mat_vac.mark_changed(WHERE)

    # Ref retrieved via PISA_FTYPE=fp64 PISA_TARGET=cpu 2020-03-21 .. ::
    #
    #   print("DM_MAT_MAT_REF")
    #   print(np.array2string(dm_mat_mat.get(), precision=20, separator=", "))
    #   print("DM_MAT_VAC_REF")
    #   print(np.array2string(dm_mat_vac.get(), precision=20, separator=", "))
    #
    ref = DM_MAT_MAT_REF
    test = dm_mat_mat.get()
    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"

    ref = DM_MAT_VAC_REF
    test = dm_mat_vac.get()
    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"


# ---------------------------------------------------------------------------- #


@myjit
def get_product(energy, dm_mat_vac, dm_mat_mat, H_mat_mass_eigenstate_basis, product):
    """
    Parameters
    ----------

    energy : float
        Neutrino energy

    dm_mat_vac : complex 2d-array

    dm_mat_mat : complex 2d-array

    H_mat_mass_eigenstate_basis : complex 2d-array

    product : complex 3d-array (empty)

    """

    H_minus_M = cuda.local.array(shape=(3, 3, 3), dtype=ctype)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                H_minus_M[i, j, k] = 2.0 * energy * H_mat_mass_eigenstate_basis[i, j]
                if i == j:
                    H_minus_M[i, j, k] -= dm_mat_vac[k, j]
                # also, cler product
                product[i, j, k] = 0.0

    # Calculate the product in eq.(10) of H_minus_M for j!=k
    for i in range(3):
        for j in range(3):
            for k in range(3):
                product[i, j, 0] += H_minus_M[i, k, 1] * H_minus_M[k, j, 2]
                product[i, j, 1] += H_minus_M[i, k, 2] * H_minus_M[k, j, 0]
                product[i, j, 2] += H_minus_M[i, k, 0] * H_minus_M[k, j, 1]
            product[i, j, 0] /= dm_mat_mat[0, 1] * dm_mat_mat[0, 2]
            product[i, j, 1] /= dm_mat_mat[1, 2] * dm_mat_mat[1, 0]
            product[i, j, 2] /= dm_mat_mat[2, 0] * dm_mat_mat[2, 1]


@njit([f"({FX}, {CX}[:,:], {CX}[:,:], {CX}[:,:], {CX}[:,:,:])"], target=TARGET)
def get_product_hostfunc(
    energy, dm_mat_vac, dm_mat_mat, H_mat_mass_eigenstate_basis, product
):
    """wrapper to run `get_product` from host (whether TARGET is "cuda" or "host")"""
    get_product(energy, dm_mat_vac, dm_mat_mat, H_mat_mass_eigenstate_basis, product)


def test_get_product():
    """unit tests for `get_product` and `get_product_hostfunc`"""
    # inputs
    energy = ENERGY_REF
    dm_mat_mat = SmartArray(DM_MAT_MAT_REF)
    dm_mat_vac = SmartArray(DM_MAT_VAC_REF)
    H_mat_mass_eigenstate_basis = SmartArray(H_MAT_REF)

    # output
    product = SmartArray(np.ones(shape=(3, 3, 3), dtype=CX))

    get_product_hostfunc(
        energy,
        dm_mat_vac.get(WHERE),
        dm_mat_mat.get(WHERE),
        H_mat_mass_eigenstate_basis.get(WHERE),
        product.get(WHERE),
    )
    product.mark_changed(WHERE)

    # Ref retrieved via PISA_FTYPE=fp64 PISA_TARGET=cpu 2020-03-21 .. ::
    #
    #   print("PRODUCT_REF")
    #   print(np.array2string(product.get(), precision=20, separator=", "))
    #
    ref = PRODUCT_REF
    test = product.get()
    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"


# ---------------------------------------------------------------------------- #


@myjit
def get_transition_matrix_massbasis(
    baseline,
    energy,
    dm_mat_vac,
    dm_mat_mat,
    H_mat_mass_eigenstate_basis,
    transition_matrix,
):
    """
    Calculate the transition amplitude matrix

    Parameters
    ----------

    baseline : float
        baseline traversed

    energy : float
        neutrino energy

    dm_mat_vac : complex 2d-array

    dm_mat_mat complex : 2d-array

    H_mat_mass_eigenstate_basis : complex 2-d array

    transition_matrix : complex 2d-array (empty)
        in mass eigenstate basis

    Notes
    -----
    - corrsponds to matrix A (equation 10) in original Barger paper
    - take into account generic potential matrix (=Hamiltonian)

    """
    product = cuda.local.array(shape=(3, 3, 3), dtype=ctype)

    clear_matrix(transition_matrix)

    get_product(energy, dm_mat_vac, dm_mat_mat, H_mat_mass_eigenstate_basis, product)

    # (1/2)*(1/(h_bar*c)) in units of GeV/(eV^2-km)
    hbar_c_factor = 2.534

    for k in range(3):
        arg = -dm_mat_vac[k, 0] * (baseline / energy) * hbar_c_factor
        c = cmath.exp(arg * 1.0j)
        for i in range(3):
            for j in range(3):
                transition_matrix[i, j] += c * product[i, j, k]


@njit(
    [f"({FX}, {FX}, {CX}[:,:], {CX}[:,:], {CX}[:,:], {CX}[:,:])"], target=TARGET,
)
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


def test_get_transition_matrix_massbasis():
    """unit tests for `get_transition_matrix_massbasis` and
    `get_transition_matrix_massbasis_hostfunc`"""
    # inputs
    baseline = BASELINE_REF
    energy = ENERGY_REF
    dm_mat_vac = SmartArray(DM_MAT_VAC_REF)
    dm_mat_mat = SmartArray(DM_MAT_MAT_REF)
    H_mat_mass_eigenstate_basis = SmartArray(H_MAT_REF)

    # output
    transition_matrix = SmartArray(np.ones(shape=(3, 3), dtype=CX))

    get_transition_matrix_massbasis_hostfunc(
        baseline,
        energy,
        dm_mat_vac.get(WHERE),
        dm_mat_mat.get(WHERE),
        H_mat_mass_eigenstate_basis.get(WHERE),
        transition_matrix.get(WHERE),
    )
    transition_matrix.mark_changed(WHERE)

    # Ref retrieved via PISA_FTYPE=fp64 PISA_TARGET=cpu 2020-03-21 .. ::
    #
    #   print("TRANSITION_MATRIX_MB_REF")
    #   print(np.array2string(transition_matrix.get(), precision=20, separator=", "))
    #
    ref = TRANSITION_MATRIX_MB_REF
    test = transition_matrix.get()
    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"


# ---------------------------------------------------------------------------- #


@myjit
def convert_from_mass_eigenstate(state, mix_nubar, psi):
    """
    Parameters
    ----------

    state : (un?)signed int

    mix_nubar : complex 2d-array

    psi : complex 1d-array (empty)


    Notes
    -----

    this is untested!
    """
    mass = cuda.local.array(shape=(3), dtype=ctype)

    lstate = state - 1
    for i in range(3):
        mass[i] = 1.0 if lstate == i else 0.0

    # note: mix_nubar is already taking into account whether we're considering
    # nu or anti-nu
    matrix_dot_vector(mix_nubar, mass, psi)


@njit([f"({IX}, {CX}[:,:], {CX}[:])"], target=TARGET)
def convert_from_mass_eigenstate_hostfunc(state, mix_nubar, psi):
    """wrapper to run `convert_from_mass_eigenstate` from host (whether TARGET
    is "cuda" or "host")"""
    convert_from_mass_eigenstate(state, mix_nubar, psi)


def test_convert_from_mass_eigenstate():
    """unit tests for `convert_from_mass_eigenstate`,
    `convert_from_mass_eigenstate_hostfunc"""
    # inputs
    state = STATE_REF
    mix_nubar = SmartArray(PMNS_NUFIT32)

    # output
    psi = SmartArray(np.ones(shape=(3), dtype=CX))

    convert_from_mass_eigenstate_hostfunc(state, mix_nubar.get(WHERE), psi.get(WHERE))
    psi.mark_changed(WHERE)

    # Ref retrieved via PISA_FTYPE=fp64 PISA_TARGET=cpu 2020-03-21 .. ::
    #
    #   print("PSI_REF")
    #   print("")
    #
    ref = PSI_REF
    test = psi.get()
    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"


# ---------------------------------------------------------------------------- #


@myjit
def get_transition_matrix(
    nubar,
    energy,
    rho,
    baseline,
    mix_nubar,
    mix_nubar_conj_transp,
    nsi_eps,
    H_vac,
    dm,
    transition_matrix,
):
    """ Calculate neutrino flavour transition amplitude matrix

    Parameters
    ----------

    nubar : real int4 or int8

    energy : real float

    rho : real float

    baseline : real float

    mix_nubar : complex 2d-array
        Mixing matrix, already conjugated if antineutrino

    mix_nubar_conj_transp : complex conjugate 2d-array
        conjugate transpose of mixing matrix

    nsi_eps : complex 2d-array

    H_vac : complex 2d-array

    dm : real 2d-array

    transition_matrix : complex 2d-array (empty)
        in mass eigenstate basis

    Notes
    -----
    for neutrino (nubar > 0) or antineutrino (nubar < 0)
    with energy energy traversing layer of matter of
    uniform density rho with thickness baseline

    """

    H_mat = cuda.local.array(shape=(3, 3), dtype=ctype)
    dm_mat_vac = cuda.local.array(shape=(3, 3), dtype=ctype)
    dm_mat_mat = cuda.local.array(shape=(3, 3), dtype=ctype)
    H_full = cuda.local.array(shape=(3, 3), dtype=ctype)
    tmp = cuda.local.array(shape=(3, 3), dtype=ctype)
    H_mat_mass_eigenstate_basis = cuda.local.array(shape=(3, 3), dtype=ctype)

    # Compute the matter potential including possible non-standard interactions
    # in the flavor basis
    get_H_mat(rho, nsi_eps, nubar, H_mat)

    # Get the full Hamiltonian by adding together matter and vacuum parts
    one_over_two_e = 0.5 / energy
    for i in range(3):
        for j in range(3):
            H_full[i, j] = H_vac[i, j] * one_over_two_e + H_mat[i, j]

    # Calculate modified mass eigenvalues in matter from the full Hamiltonian and
    # the vacuum mass splittings
    get_dms(energy, H_full, dm, dm_mat_mat, dm_mat_vac)

    # Now we transform the matter (TODO: matter? full?) Hamiltonian back into the
    # mass eigenstate basis so we don't need to compute products of the effective
    # mixing matrix elements explicitly
    matrix_dot_matrix(H_mat, mix_nubar, tmp)
    matrix_dot_matrix(mix_nubar_conj_transp, tmp, H_mat_mass_eigenstate_basis)

    # We can now proceed to calculating the transition amplitude from the Hamiltonian
    # in the mass basis and the effective mass splittings
    get_transition_matrix_massbasis(
        baseline,
        energy,
        dm_mat_vac,
        dm_mat_mat,
        H_mat_mass_eigenstate_basis,
        transition_matrix,
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
        f"{CX}[:,:], "  # nsi_eps
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
    nsi_eps,
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
        nsi_eps,
        H_vac,
        dm,
        transition_matrix,
    )


def test_get_transition_matrix():
    """unit tests of `get_transition_matrix` and `get_transition_matrix_hostfunc`"""
    # inputs
    nubar = NUBAR_REF
    energy = ENERGY_REF
    rho = RHO_REF
    baseline = BASELINE_REF
    mix_nubar = SmartArray(PMNS_NUFIT32)
    mix_nubar_conj_transp = SmartArray(PMNS_NUFIT32.conj().T)
    nsi_eps = SmartArray(NSI_EPS_REF)
    H_vac = SmartArray(H_VAC_REF)
    dm = SmartArray(DM)

    # output
    transition_matrix = SmartArray(np.ones(shape=(3, 3), dtype=CX))

    get_transition_matrix_hostfunc(
        nubar,
        energy,
        rho,
        baseline,
        mix_nubar.get(WHERE),
        mix_nubar_conj_transp.get(WHERE),
        nsi_eps.get(WHERE),
        H_vac.get(WHERE),
        dm.get(WHERE),
        transition_matrix.get(WHERE),
    )
    transition_matrix.mark_changed(WHERE)

    # Ref retrieved via PISA_FTYPE=fp64 PISA_TARGET=cpu 2020-03-21 .. ::
    #
    #   print("TRANSITION_MATRIX_REF")
    #   print(np.array2string(transition_matrix.get(), precision=20, separator=", "))
    #
    ref = TRANSITION_MATRIX_REF
    test = transition_matrix.get()
    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"


# ---------------------------------------------------------------------------- #


@myjit
def osc_probs_vacuum_kernel(dm, mix, nubar, energy, distance_in_layer, osc_probs):
    """ Calculate vacumm mixing probabilities

    Parameters
    ----------

    dm : 2d-array
        Mass splitting matrix

    mix : complex 2d-array
        PMNS mixing matrix

    nubar : int
        +1 for neutrinos, -1 for antineutrinos

    energy : float
        Neutrino energy

    distance_in_layer : 1d-array
        Baselines (will be summed up)

    osc_probs : 2d-array (empty)
        Returned oscillation probabilities in the form:
        osc_prob[i,j] = probability of flavor i to oscillate into flavor j
        with 0 = electron, 1 = muon, 3 = tau


    Notes
    -----

    This is largely unvalidated so far

    """

    # no need to conjugate mix matrix, as we anyway only need real part
    # can this be right?

    clear_matrix(osc_probs)
    osc_probs_local = cuda.local.array(shape=(3, 3), dtype=ftype)

    # sum up length from all layers
    baseline = 0.0
    for i in range(distance_in_layer.shape[0]):
        baseline += distance_in_layer[i]

    # make more precise 20081003 rvw
    l_over_e = 1.26693281 * baseline / energy
    s21 = math.sin(dm[1, 0] * l_over_e)
    s32 = math.sin(dm[2, 0] * l_over_e)
    s31 = math.sin((dm[2, 1] + dm[3, 2]) * l_over_e)

    # does anybody understand this loop?
    # ista = abs(*nutype) - 1
    for ista in range(3):
        for iend in range(2):
            osc_probs_local[ista, iend] = (
                (mix[ista, 0].real * mix[ista, 1].real * s21) ** 2
                + (mix[ista, 1].real * mix[ista, 2].real * s32) ** 2
                + (mix[ista, 2].real * mix[ista, 0].real * s31) ** 2
            )
            if iend == ista:
                osc_probs_local[ista, iend] = 1.0 - 4.0 * osc_probs_local[ista, iend]
            else:
                osc_probs_local[ista, iend] = -4.0 * osc_probs_local[ista, iend]

        osc_probs_local[ista, 2] = (
            1.0 - osc_probs_local[ista, 0] - osc_probs_local[ista, 1]
        )

    # is this necessary?
    if nubar > 0:
        copy_matrix(osc_probs_local, osc_probs)
    else:
        for i in range(3):
            for j in range(3):
                osc_probs[i, j] = osc_probs_local[j, i]


@guvectorize(
    [f"({FX}[:,:], {CX}[:,:], {IX}, {FX}, {FX}[:], {FX}[:,:])"],
    "(a,b), (c,d), (), (), (i) -> (a,b)",
    target=TARGET,
)
def propagate_array_vacuum(dm, mix, nubar, energy, distances, probability):
    """wrapper to run `osc_probs_vacuum_kernel` from host (whether TARGET is
    "cuda" or "host")"""
    osc_probs_vacuum_kernel(dm, mix, nubar, energy, distances, probability)


def test_osc_probs_vacuum_kernel():
    """unit tests for `osc_probs_vacuum_kernel` and its wrapper,
    `propagate_array_vacuum`"""
    # inputs
    dm = SmartArray(DM)
    mix = SmartArray(PMNS_NUFIT32)
    nubar = NUBAR_REF
    energy = ENERGY_REF
    # the vacuum osc function simply sums these up
    distance_in_layer = SmartArray(LAYER_THICKNESSES_REF)

    # output
    osc_probs = SmartArray(np.ones(shape=(3, 3), dtype=FX))

    propagate_array_vacuum(
        dm.get(WHERE),
        mix.get(WHERE),
        nubar,
        energy,
        distance_in_layer.get(WHERE),
        osc_probs.get(WHERE),
    )
    osc_probs.mark_changed(WHERE)

    # Ref retrieved via PISA_FTYPE=fp64 PISA_TARGET=cpu 2020-03-21 .. ::
    #
    #   print("OSC_PROBS_VAC_REF")
    #   print(np.array2string(osc_probs.get(), precision=20, separator=", "))
    #
    ref = OSC_PROBS_VAC_REF
    test = osc_probs.get(WHERE)
    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"


# ---------------------------------------------------------------------------- #


@myjit
def osc_probs_layers_kernel(
    dm, mix, nsi_eps, nubar, energy, density_in_layer, distance_in_layer, osc_probs,
):
    """ Calculate oscillation probabilities

    given layers of length and density

    Parameters
    ----------

    dm : 2d-array
        Mass splitting matrix

    mix : 2d-array
        PMNS mixing matrix

    H_vac : complex 2-d array
        Hamiltonian in vacuum, without the 1/2E term

    nsi_eps : 2d-array
        Non-standard interactions (set to 3x3 zeros for only standard oscillations)

    nubar : int
        1 for neutrinos, -1 for antineutrinos

    energy : float
        Neutrino energy

    density_in_layer : 1d-array
        density per layer

    distance_in_layer : 1d-array
        distance per layer traversed

    osc_probs : 2d-array (empty)
        Returned oscillation probabilities in the form:
        osc_prob[i,j] = probability of flavor i to oscillate into flavor j
        with 0 = electron, 1 = muon, 3 = tau


    Notes
    -----

    !!! Right now, because of CUDA, the maximum number of layers
    is hard coded and set to 120 (59Layer PREM + Atmosphere).
    This is used for cached layer computation, where earth layer, which
    are typically traversed twice (it's symmetric) are not recalculated
    but rather cached..

    """

    # 3x3 complex
    H_vac = cuda.local.array(shape=(3, 3), dtype=ctype)
    mix_nubar = cuda.local.array(shape=(3, 3), dtype=ctype)
    mix_nubar_conj_transp = cuda.local.array(shape=(3, 3), dtype=ctype)
    transition_product = cuda.local.array(shape=(3, 3), dtype=ctype)
    transition_matrix = cuda.local.array(shape=(3, 3), dtype=ctype)
    tmp = cuda.local.array(shape=(3, 3), dtype=ctype)

    clear_matrix(H_vac)
    clear_matrix(osc_probs)

    # 3-vector complex
    raw_input_psi = cuda.local.array(shape=(3), dtype=ctype)
    output_psi = cuda.local.array(shape=(3), dtype=ctype)

    use_mass_eigenstates = False

    cache = True
    # cache = False

    # TODO:
    # * ensure convention below is respected in MC reweighting
    #   (nubar > 0 for nu, < 0 for anti-nu)
    # * nubar is passed in, so could already pass in the correct form
    #   of mixing matrix, i.e., possibly conjugated
    if nubar > 0:
        # in this case the mixing matrix is left untouched
        copy_matrix(mix, mix_nubar)

    else:
        # here we need to complex conjugate all entries
        # (note that this only changes calculations with non-clear_matrix deltacp)
        conjugate(mix, mix_nubar)

    conjugate_transpose(mix_nubar, mix_nubar_conj_transp)

    get_H_vac(mix_nubar, mix_nubar_conj_transp, dm, H_vac)

    if cache:
        # allocate array to store all the transition matrices
        # doesn't work in cuda...needs fixed shape
        transition_matrices = cuda.local.array(shape=(120, 3, 3), dtype=ctype)

        # loop over layers
        for i in range(distance_in_layer.shape[0]):
            density = density_in_layer[i]
            distance = distance_in_layer[i]
            if distance > 0.0:
                layer_matrix_index = -1
                # chaeck if exists
                for j in range(i):
                    # if density_in_layer[j] == density and distance_in_layer[j] == distance:
                    if (abs(density_in_layer[j] - density) < 1e-5) and (
                        abs(distance_in_layer[j] - distance) < 1e-5
                    ):
                        layer_matrix_index = j

                # use from cached
                if layer_matrix_index >= 0:
                    for j in range(3):
                        for k in range(3):
                            transition_matrices[i, j, k] = transition_matrices[
                                layer_matrix_index, j, k
                            ]

                # only calculate if necessary
                else:
                    get_transition_matrix(
                        nubar,
                        energy,
                        density,
                        distance,
                        mix_nubar,
                        mix_nubar_conj_transp,
                        nsi_eps,
                        H_vac,
                        dm,
                        transition_matrix,
                    )
                    # copy
                    for j in range(3):
                        for k in range(3):
                            transition_matrices[i, j, k] = transition_matrix[j, k]
            else:
                # identity matrix
                for j in range(3):
                    for k in range(3):
                        if j == k:
                            transition_matrix[j, k] = 0.0
                        else:
                            transition_matrix[j, k] = 1.0

        # now multiply them all
        first_layer = True
        for i in range(distance_in_layer.shape[0]):
            distance = distance_in_layer[i]
            if distance > 0.0:
                for j in range(3):
                    for k in range(3):
                        transition_matrix[j, k] = transition_matrices[i, j, k]
                if first_layer:
                    copy_matrix(transition_matrix, transition_product)
                    first_layer = False
                else:
                    matrix_dot_matrix(transition_matrix, transition_product, tmp)
                    copy_matrix(tmp, transition_product)

    else:
        # non-cache loop
        first_layer = True
        for i in range(distance_in_layer.shape[0]):
            density = density_in_layer[i]
            distance = distance_in_layer[i]
            # only do something if distance > 0.
            if distance > 0.0:
                get_transition_matrix(
                    nubar,
                    energy,
                    density,
                    distance,
                    mix_nubar,
                    mix_nubar_conj_transp,
                    nsi_eps,
                    H_vac,
                    dm,
                    transition_matrix,
                )
                if first_layer:
                    copy_matrix(transition_matrix, transition_product)
                    first_layer = False
                else:
                    matrix_dot_matrix(transition_matrix, transition_product, tmp)
                    copy_matrix(tmp, transition_product)

    # convrt to flavour eigenstate basis
    matrix_dot_matrix(transition_product, mix_nubar_conj_transp, tmp)
    matrix_dot_matrix(mix_nubar, tmp, transition_product)

    # loop on neutrino types, and compute probability for neutrino i:
    for i in range(3):
        for j in range(3):
            raw_input_psi[j] = 0.0

        if use_mass_eigenstates:
            convert_from_mass_eigenstate(i + 1, mix_nubar, raw_input_psi)
        else:
            raw_input_psi[i] = 1.0

        matrix_dot_vector(transition_product, raw_input_psi, output_psi)
        osc_probs[i][0] += output_psi[0].real ** 2 + output_psi[0].imag ** 2
        osc_probs[i][1] += output_psi[1].real ** 2 + output_psi[1].imag ** 2
        osc_probs[i][2] += output_psi[2].real ** 2 + output_psi[2].imag ** 2


@guvectorize(
    [f"({FX}[:,:], {CX}[:,:], {CX}[:,:], {IX}, {FX}, {FX}[:], {FX}[:], {FX}[:,:])"],
    "(a,b), (c,d), (e,f), (), (), (g), (h) -> (a,b)",
    target=TARGET,
)
def propagate_array(dm, mix, nsi_eps, nubar, energy, densities, distances, probability):
    """wrapper to run `osc_probs_layers_kernel` from host (whether TARGET
    is "cuda" or "host")"""
    osc_probs_layers_kernel(
        dm, mix, nsi_eps, nubar, energy, densities, distances, probability
    )


def test_osc_probs_layers_kernel():
    """unit tests for `osc_probs_layers_kernel` and its wrapper, `propagate_array`"""
    # inputs
    dm = SmartArray(DM)
    mix = SmartArray(PMNS_NUFIT32)
    nsi_eps = SmartArray(NSI_EPS_REF)
    nubar = NUBAR_REF
    energy = ENERGY_REF
    density_in_layer = SmartArray(LAYER_DENSITY_REF)
    distance_in_layer = SmartArray(LAYER_THICKNESSES_REF)

    # output
    osc_probs = SmartArray(np.ones(shape=(3, 3), dtype=FX))

    propagate_array(
        dm.get(WHERE),
        mix.get(WHERE),
        nsi_eps.get(WHERE),
        nubar,
        energy,
        density_in_layer.get(WHERE),
        distance_in_layer.get(WHERE),
        osc_probs.get(WHERE),
    )
    osc_probs.mark_changed(WHERE)

    # Ref retrieved via PISA_FTYPE=fp64 PISA_TARGET=cpu 2020-03-21 .. ::
    #
    #   print("OSC_PROBS_LAYERS_REF")
    #   print(np.array2string(osc_probs.get(), precision=20, separator=", "))
    #
    ref = OSC_PROBS_LAYERS_REF
    test = osc_probs.get(WHERE)
    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"


# ---------------------------------------------------------------------------- #


@guvectorize(
    [f"({FX}[:,:], {IX}, {IX}, {FX}[:])"], "(a,b), (), () -> ()", target=TARGET,
)
def fill_probs(probability, initial_flav, flav, out):
    """Fill `out` with transition probabilities to go from `initial_flav` to
    `flav`, from values in `probaility` matrix.

    Parameters
    ----------
    probability : real 2-d array
    initial_flav : signed(?) int (int4 or int8)
    flav : signed(?) int (int4 or int8)
    out : real 1-d array

    """
    out[0] = probability[initial_flav, flav]


# ---------------------------------------------------------------------------- #


def complex_allclose(a, b, **kwargs):
    """Compare whether magnitude and phase of complex arrays are all close"""
    if a.shape != b.shape:
        return False
    mag_a, mag_b = np.absolute(a), np.absolute(b)
    if not np.allclose(mag_a, mag_b, **kwargs):
        return False
    ang_a, ang_b = np.angle(a), np.angle(b)
    if not np.allclose(ang_a, ang_b, **kwargs):
        return False
    return True


# ---------------------------------------------------------------------------- #
# Define relevant values for testing purposes (from nufit3.2, from intermediate
# calculations performed here, or arbitary values).
#
# NOTE: !!DO NOT CHANGE!! (unless a function is incorrect) tests rely on these
# ---------------------------------------------------------------------------- #

T12 = np.deg2rad(33.62)
"""theta12, nufit 3.2 best-fit nominal value (normal ordering). DO NOT CHANGE"""

T23 = np.deg2rad(47.2)
"""theta23, nufit 3.2 best-fit nominal value (normal ordering). DO NOT CHANGE"""

T13 = np.deg2rad(8.54)
"""theta13, nufit 3.2 best-fit nominal value (normal ordering). DO NOT CHANGE"""

DCP = np.deg2rad(234)
"""deltaCP, nufit 3.2 best-fit nominal value (normal ordering). DO NOT CHANGE"""

DM21 = 7.4e-5
"""Delta m^2_21/eV^2, nufit 3.2 best-fit nom val (normal ordering). DO NOT CHANGE"""

DM31 = 2.494e-3
"""Delta m^2_31/eV^2, nufit 3.2 best-fit nom val (normal ordering). DO NOT CHANGE"""

DM32 = DM31 - DM21
"""Delta m^2_32/eV^2"""

DM = np.array([[0, -DM21, -DM31], [DM21, 0, -DM32], [DM31, DM32, 0],], dtype=FX,)
"""Delta m^2 matrix"""

C12, S12 = np.cos(T12), np.sin(T12)
C23, S23 = np.cos(T23), np.sin(T23)
C13, S13 = np.cos(T13), np.sin(T13)

PMNS_NUFIT32 = (
    np.array([[1, 0, 0], [0, C23, S23], [0, -S23, C23],])
    @ np.array(
        [
            [C13, 0, S13 * np.exp(-1j * DCP)],
            [0, 1, 0],
            [-S13 * np.exp(1j * DCP), 0, C13],
        ]
    )
    @ np.array([[C12, S12, 0], [-S12, C12, 0], [0, 0, 1],])
).astype(CX)
"""PMNS matrix, nufit 3.2 best-fit nominal values (normal ordering).
DO NOT CHANGE, as unit tests rely on these values"""

H_VAC_REF = np.array(
    [
        [
            7.7183660602589364e-05 + 0.0000000000000000e00j,
            -1.3359407349676454e-04 + 2.1542972826327976e-04j,
            -1.6969456972360504e-04 + 1.9949011568048700e-04j,
        ],
        [
            -1.3359407349676451e-04 - 2.1542972826327973e-04j,
            1.3399872228213837e-03 + 0.0000000000000000e00j,
            1.1903461409985979e-03 - 4.0990114580586629e-06j,
        ],
        [
            -1.6969456972360504e-04 - 1.9949011568048697e-04j,
            1.1903461409985979e-03 + 4.0990114580586629e-06j,
            1.1508291165760267e-03 + 0.0000000000000000e00j,
        ],
    ],
    dtype=CX,
)
"""See `test_get_H_vac`"""

H_MAT_REF = np.array(
    [
        [0.000305176 + 0.0j, 0.000228882 + 0.0j, 0.000228882 + 0.0j],
        [0.000228882 + 0.0j, 0.000228882 + 0.0j, 0.000228882 + 0.0j],
        [0.000228882 + 0.0j, 0.000228882 + 0.0j, 0.000228882 + 0.0j],
    ],
    dtype=CX,
)
"""See `test_get_H_mat`"""

ENERGY_REF = FTYPE(1)  # GeV
BASELINE_REF = FTYPE(1)  # m
STATE_REF = ITYPE(1)
NUBAR_REF = ITYPE(1)
RHO_REF = FTYPE(1)
NSI_EPS_REF = np.ones(shape=(3, 3), dtype=CX)
N_LAYERS_REF = 10
LAYER_THICKNESSES_REF = np.logspace(0, 2, N_LAYERS_REF, dtype=FX)
LAYER_DENSITY_REF = np.logspace(-1, 3, N_LAYERS_REF, dtype=FX)

DM_MAT_MAT_REF = np.array(
    [
        [
            0.0000000000000000e00 + 0.0j,
            -9.7824328003015635e-05 + 0.0j,
            -1.4280556719969842e-03 + 0.0j,
        ],
        [
            9.7824328003015635e-05 + 0.0j,
            0.0000000000000000e00 + 0.0j,
            -1.3302313439939686e-03 + 0.0j,
        ],
        [
            1.4280556719969842e-03 + 0.0j,
            1.3302313439939686e-03 + 0.0j,
            0.0000000000000000e00 + 0.0j,
        ],
    ],
    dtype=CX,
)
"""See `test_get_dms`"""

DM_MAT_VAC_REF = np.array(
    [
        [
            1.0842021724855044e-19 + 0.0j,
            -7.3999999999999888e-05 + 0.0j,
            -2.4940000000000001e-03 + 0.0j,
        ],
        [
            9.7824328003015743e-05 + 0.0j,
            2.3824328003015747e-05 + 0.0j,
            -2.3961756719969845e-03 + 0.0j,
        ],
        [
            1.4280556719969842e-03 + 0.0j,
            1.3540556719969842e-03 + 0.0j,
            -1.0659443280030159e-03 + 0.0j,
        ],
    ],
    dtype=CX,
)
"""See `test_get_dms`"""

OSC_PROBS_VAC_REF = np.array(
    [
        [0.9851749216997459, -0.014825078300254105, 0.02965015660050825],
        [-0.48420178468166286, 0.5157982153183371, 0.9684035693633257],
        [-0.47951617325806206, -0.47951617325806206, 1.959032346516124],
    ],
    dtype=FX,
)
"""See `test_osc_probs_vacuum_kernel`"""

OSC_PROBS_LAYERS_REF = np.array(
    [
        [0.18544164187134157, 0.4090972513764466, 0.4054611067522087],
        [0.4096985555162951, 0.08824684348573122, 0.5020546009979748],
        [0.4048598026123579, 0.5026559051378211, 0.0924842922498177],
    ],
    dtype=FX,
)
"""See `test_osc_probs_layers_kernel`"""

PRODUCT_REF = np.array(
    [
        [
            [
                3.7895558246780265e-16 + 0.0j,
                6.1470786693528068e-01 - 0.0j,
                3.8529213306471910e-01 + 0.0j,
            ],
            [
                2.4248302618816731e-01 + 0.0j,
                -6.0443866741020869e-01 - 0.0j,
                3.6195564122204155e-01 + 0.0j,
            ],
            [
                8.1723333420714646e00 + 0.0j,
                -9.1174448996070403e00 - 0.0j,
                9.4511155753557430e-01 + 0.0j,
            ],
        ],
        [
            [
                2.4248302618816711e-01 + 0.0j,
                -6.0443866741020869e-01 - 0.0j,
                3.6195564122204155e-01 + 0.0j,
            ],
            [
                2.1588797707535934e-01 + 0.0j,
                4.4202207719233289e-01 - 0.0j,
                3.4208994573230783e-01 + 0.0j,
            ],
            [
                7.9148163682596335e00 + 0.0j,
                -8.8409902988732032e00 - 0.0j,
                9.2617393061357112e-01 + 0.0j,
            ],
        ],
        [
            [
                8.1723333420714646e00 + 0.0j,
                -9.1174448996070421e00 - 0.0j,
                9.4511155753557430e-01 + 0.0j,
            ],
            [
                7.9148163682596318e00 + 0.0j,
                -8.8409902988732032e00 - 0.0j,
                9.2617393061357112e-01 + 0.0j,
            ],
            [
                3.4128243713193335e01 + 0.0j,
                -3.7783458166771375e01 - 0.0j,
                4.6552144535780409e00 + 0.0j,
            ],
        ],
    ],
    dtype=CX,
)
"""See `test_get_product`"""

PSI_REF = np.array(
    [
        0.8234950912632637 + 0.0j,
        -0.3228630670998579 + 0.07340455327326298j,
        0.4556387475142645 + 0.06797336162473219j,
    ],
    dtype=CX,
)
"""See `test_convert_from_mass_eigenstate`"""

TRANSITION_MATRIX_MB_REF = np.array(
    [
        [
            9.9999745842790577e-01 - 0.0015466289234961912j,
            -2.3513202954128509e-06 - 0.0011599711188971414j,
            -5.9079587375610032e-06 - 0.0011599665348858614j,
        ],
        [
            -2.3513202956348955e-06 - 0.0011599711188971414j,
            9.9999774659820162e-01 - 0.0013474872731349206j,
            -5.7924591063684971e-06 - 0.0011599666837487869j,
        ],
        [
            -5.9079587393373600e-06 - 0.001159966534885861j,
            -5.7924591081448540e-06 - 0.0011599666837487869j,
            9.9997068101418662e-01 - 0.007479733306172477j,
        ],
    ],
    dtype=CX,
)
"""See `test_get_transition_matrix_massbasis`"""

TRANSITION_MATRIX_REF = np.array(
    [
        [
            9.9999771012826755e-01 - 0.0013461327141801491j,
            9.2295560300686974e-06 - 0.0008227497563155284j,
            -5.0701986705203872e-05 - 0.0014450231484626139j,
        ],
        [
            -1.2153178716590995e-05 - 0.0008227105464077983j,
            9.9999905034999947e-01 - 0.0006913979128080287j,
            -4.2393535334653287e-05 - 0.0008616672624234467j,
        ],
        [
            3.5998806432152586e-05 - 0.001445464866498789j,
            3.3424172405496777e-05 - 0.0008620606152535083j,
            9.9996383535656774e-01 - 0.00833624907916226j,
        ],
    ],
    dtype=CX,
)
"""See `test_get_transition_matrix_massbasis`"""


if __name__ == "__main__":
    test_get_H_vac()
    test_get_H_mat()
    test_get_dms()
    test_get_product()
    test_get_transition_matrix()
    test_convert_from_mass_eigenstate()
    test_get_transition_matrix()
    test_osc_probs_vacuum_kernel()
    test_osc_probs_layers_kernel()
