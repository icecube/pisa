# pylint: disable = not-callable, bad-whitespace, invalid-name
'''Neutrino flavour oscillation in matter calculation
Based on the original prob3++ implementation of Roger Wendell
http://www.phy.duke.edu/~raw22/public/Prob3++/ (2012)
'''
from __future__ import absolute_import, print_function, division

__all__ = ['get_transition_matrix',
           'osc_probs_layers_kernel',
           'osc_probs_vacuum_kernel',
           'propagate_array',
           'propagate_array_vacuum',
           'fill_probs',
          ]
__version__ = '0.1'

import cmath
import math
import numpy as np
from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa.utils.numba_tools import (
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


if FTYPE == np.float64:
    FX = "f8"
    CX = "c16"
elif FTYPE == np.float32:
    FX = "f4"
    CX = "c8"
else:
    raise TypeError(str(FTYPE))


@myjit
def get_H_vac(mix_nubar, mix_nubar_conj_transp, dm_vac_vac, H_vac):
    ''' Calculate vacuum Hamiltonian in flavor basis for neutrino or antineutrino

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

    '''
    dm_vac_diag = cuda.local.array(shape=(3,3), dtype=ctype)
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)

    clear_matrix(dm_vac_diag)

    dm_vac_diag[1,1] = dm_vac_vac[1,0] + 0j
    dm_vac_diag[2,2] = dm_vac_vac[2,0] + 0j

    matrix_dot_matrix(dm_vac_diag, mix_nubar_conj_transp, tmp)
    matrix_dot_matrix(mix_nubar, tmp, H_vac)

def test_get_H_vac():
    mix = np.ones(shape=(3,3), dtype=ctype)
    dm_vac_vac = np.ones(shape=(3,3), dtype=ftype)
    H_vac = np.ones(shape=(3,3), dtype=ctype)

    get_H_vac(mix, mix.conj().T, dm_vac_vac, H_vac)
    #print(H_vac)


@myjit
def get_H_mat(rho, mat_pot, nubar, H_mat):
    ''' Calculate matter Hamiltonian in flavor basis

    Parameters:
    -----------
    rho : float
        electron number density (in moles/cm^3)
        (numerically, this is just the product of
        electron fraction and mass density in g/cm^3,
        since the number of grams per cm^3 corresponds to
        the number of moles of nucleons per cm^3)

    mat_pot : complex 2-d array
        Generalised matter potential matrix without a factor
        (will be multiplied with "a" factor)
        set to diag(1, 0, 0) for only standard oscillations

    nubar : int
        +1 for neutrinos, -1 for antineutrinos

    H_mat : complex 2d-array (empty)

    Notes
    -----
    in the following, `a` is just the standard effective matter potential
    induced by charged-current weak interactions with electrons

    '''

    # 2*sqrt(2)*Gfermi in (eV^2-cm^3)/(mole-GeV)
    tworttwoGf = 1.52588e-4
    a = 0.5 * rho * tworttwoGf

    # standard matter interaction Hamiltonian
    clear_matrix(H_mat)
    #formalism of Hamiltonian: not 1+epsilon_ee^f in [0,0] element but just epsilon...
    #EL changed when fitting in Thomas' NSI branch

    # Obtain effective non-standard matter interaction Hamiltonian
    #EL changed when fitting in Thomas' NSI branch
    for i in range(3):
        for j in range(3):
            # matter potential V -> -V* for anti-neutrinos
            if nubar == -1:
                H_mat[i, j] = -a * mat_pot[i, j].conjugate()
            elif nubar == 1:
                H_mat[i, j] = a * mat_pot[i, j]

def test_get_H_mat():
    rho = 1.
    nubar = -1
    mat_pot = np.ones(shape=(3,3), dtype=ctype)
    H_mat = np.ones(shape=(3,3), dtype=ctype)

    get_H_mat(rho, mat_pot, nubar, H_mat)
    #print(H_mat)


@myjit
def get_dms(energy, H_mat, dm_vac_vac, dm_mat_mat, dm_mat_vac):
    '''Compute the matter-mass vector M, dM = M_i-M_j and dMimj

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

    '''
    # the following is for solving the characteristic polynomial of H_mat:
    # P(x) = x**3 + c2*x**2 + c1*x + c0
    real_product_a = (H_mat[0,1] * H_mat[1,2] * H_mat[2,0]).real
    real_product_b = (H_mat[0,0] * H_mat[1,1] * H_mat[2,2]).real

    norm_H_e_mu_sq = H_mat[0,1].real**2 + H_mat[0,1].imag**2
    norm_H_e_tau_sq = H_mat[0,2].real**2 + H_mat[0,2].imag**2
    norm_H_mu_tau_sq = H_mat[1,2].real**2 + H_mat[1,2].imag**2

    # c1 = H_{11} * H_{22} + H_{11} * H_{33} + H_{22} * H_{33}
    #      - |H_{12}|**2 - |H_{13}|**2 - |H_{23}|**2
    # given Hermiticity of Hamiltonian (real diagonal elements),
    # this coefficient must be real
    c1 = ((H_mat[0,0].real * (H_mat[1,1] + H_mat[2,2])).real
          - (H_mat[0,0].imag * (H_mat[1,1] + H_mat[2,2])).imag
          + (H_mat[1,1].real * H_mat[2,2]).real
          - (H_mat[1,1].imag * H_mat[2,2]).imag
          - norm_H_e_mu_sq
          - norm_H_mu_tau_sq
          - norm_H_e_tau_sq
         )

    # c0 = H_{11} * |H_{23}|**2 + H_{22} * |H_{13}|**2 + H_{33} * |H_{12}|**2
    #      - H_{11} * H_{22} * H_{33} - 2*Re(H*_{13} * H_{12} * H_{23})
    # hence, this coefficient is also real
    c0 = (H_mat[0,0].real * norm_H_mu_tau_sq
          + H_mat[1,1].real * norm_H_e_tau_sq
          + H_mat[2,2].real * norm_H_e_mu_sq
          - 2. * real_product_a
          - real_product_b
         )

    # c2 = -H_{11} - H_{22} - H_{33}
    # hence, this coefficient is also real
    c2 = - H_mat[0,0].real - H_mat[1,1].real - H_mat[2,2].real

    one_over_two_e = 0.5 / energy
    one_third = 1./3.
    two_third = 2./3.

    # we also have to perform the corresponding algebra
    # for the vacuum case, where the relevant elements of the
    # hamiltonian are mass differences
    x = dm_vac_vac[1,0]
    y = dm_vac_vac[2,0]

    c2_v = - one_over_two_e * (x + y)

    # p is real due to reality of c1 and c2
    p = c2**2 - 3.*c1
    p_v = one_over_two_e**2 * (x**2 + y**2 - x*y)
    p = max(0., p)

    # q is real
    q = -13.5*c0 - c2**3 + 4.5*c1*c2
    q_v = one_over_two_e**3 * (x + y) * ((x + y)**2 - 4.5*x*y)

    # we need the quantity p**3 - q**2 to obtain the eigenvalues,
    # but let's prevent inaccuracies and instead write
    tmp = 27*(0.25 * c1**2 * (p - c1) + c0 * (q + 6.75 * c0))  
    #EL changed from p**3 - q**2 when fitting in Thomas' NSI branch
    # TODO: can we simplify this quantity to reduce numerical inaccuracies?
    tmp_v = p_v**3 - q_v**2

    tmp = max(0., tmp)

    theta = cuda.local.array(shape=(3), dtype=ftype)
    theta_v = cuda.local.array(shape=(3), dtype=ftype)
    m_mat = cuda.local.array(shape=(3), dtype=ftype)
    m_mat_u = cuda.local.array(shape=(3), dtype=ftype)
    m_mat_v = cuda.local.array(shape=(3), dtype=ftype)

    a = two_third * math.pi
    # intermediate result, needed to calculate the three
    # mass eigenvalues (theta0, theta1, theta2 are the three
    # corresponding arguments of the cosine, see m_mat_u)
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
        m_mat_u[i] = 2. * energy * (b * math.cos(theta[i]) - c2*one_third + dm_vac_vac[0,0])
        m_mat_v[i] = 2. * energy * (b_v * math.cos(theta_v[i]) - c2_v*one_third + dm_vac_vac[0,0])

    # Sort according to which reproduce the vaccum eigenstates
    for i in range(3):
        tmp_v = abs(dm_vac_vac[i,0] - m_mat_v[0])
        k = 0
        for j in range(3):
            tmp = abs(dm_vac_vac[i,0] - m_mat_v[j])
            if tmp < tmp_v:
                k = j
                tmp_v = tmp
        m_mat[i] = m_mat_u[k]

    for i in range(3):
        for j in range(3):
            dm_mat_mat[i,j] = m_mat[i] - m_mat[j]
            dm_mat_vac[i,j] = m_mat[i] - dm_vac_vac[j,0]

def test_get_dms():
    energy = 1.
    dm_vac_vac = np.ones(shape=(3,3), dtype=ftype)
    dm_mat_mat = np.ones(shape=(3,3), dtype=ctype)
    dm_mat_vac = np.ones(shape=(3,3), dtype=ctype)
    H_mat = np.ones(shape=(3,3), dtype=ctype)

    get_dms(energy, H_mat, dm_vac_vac, dm_mat_mat, dm_mat_vac)


@myjit
def get_product(energy,
                dm_mat_vac,
                dm_mat_mat,
                H_mat_mass_eigenstate_basis,
                product):
    '''
    Parameters
    ----------

    energy : float
        Neutrino energy

    dm_mat_vac : complex 2d-array

    dm_mat_mat : complex 2d-array

    H_mat_mass_eigenstate_basis : complex 2d-array

    product : 3d-array (empty)

    '''

    H_minus_M = cuda.local.array(shape=(3,3,3), dtype=ctype)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                H_minus_M[i,j,k] = 2. * energy * H_mat_mass_eigenstate_basis[i,j]
                if i == j:
                    H_minus_M[i,j,k] -= dm_mat_vac[k,j]
                #also, cler product
                product[i,j,k] = 0.

    # Calculate the product in eq.(10) of H_minus_M for j!=k
    for i in range(3):
        for j in range(3):
            for k in range(3):
                product[i,j,0] += (H_minus_M[i,k,1] * H_minus_M[k,j,2])
                product[i,j,1] += (H_minus_M[i,k,2] * H_minus_M[k,j,0])
                product[i,j,2] += (H_minus_M[i,k,0] * H_minus_M[k,j,1])
            product[i,j,0] /= (dm_mat_mat[0,1] * dm_mat_mat[0,2])
            product[i,j,1] /= (dm_mat_mat[1,2] * dm_mat_mat[1,0])
            product[i,j,2] /= (dm_mat_mat[2,0] * dm_mat_mat[2,1])

def test_get_product():
    baseline = 1.
    energy = 1.
    dm_mat_mat = np.ones(shape=(3,3), dtype=ctype)
    dm_mat_vac = np.ones(shape=(3,3), dtype=ctype)
    H_mat_mass_eigenstate_basis = np.ones(shape=(3,3), dtype=ctype)
    product = np.ones(shape=(3,3,3), dtype=ctype)

    get_product(energy,
                dm_mat_vac,
                dm_mat_mat,
                H_mat_mass_eigenstate_basis,
                product)


@myjit
def get_transition_matrix_massbasis(baseline,
                                    energy,
                                    dm_mat_vac,
                                    dm_mat_mat,
                                    H_mat_mass_eigenstate_basis,
                                    transition_matrix):
    '''
    Calculate the transition amplitude matrix

    Parameters
    ----------

    baseline : float
        baseline traversed

    energy : float
        neutrino energy

    dm_mat_vac : 2d-array

    dm_mat_mat : 2d-array

    H_mat_mass_eigenstate_basis : 2-d array

    transition_matrix : 2d-array (empty)
        in mass eigenstate basis

    Notes
    -----
    - corrsponds to matrix A (equation 10) in original Barger paper
    - take into account generic potential matrix (=Hamiltonian)

    '''
    product = cuda.local.array(shape=(3,3,3), dtype=ctype)

    clear_matrix(transition_matrix)

    get_product(energy,
                dm_mat_vac,
                dm_mat_mat,
                H_mat_mass_eigenstate_basis,
                product)

    # (1/2)*(1/(h_bar*c)) in units of GeV/(eV^2-km)
    hbar_c_factor = 2.534

    for k in range(3):
        arg = - dm_mat_vac[k,0] * (baseline / energy) * hbar_c_factor
        c = cmath.exp(arg * 1.j)
        for i in range(3):
            for j in range(3):
                transition_matrix[i,j] += c * product[i,j,k]


def test_get_transition_matrix_massbasis():
    baseline = 1.
    energy = 1.
    mix = np.ones(shape=(3,3), dtype=ctype)
    dm_mat_mat = np.ones(shape=(3,3), dtype=ctype)
    dm_mat_vac = np.ones(shape=(3,3), dtype=ctype)
    H_mat_mass_eigenstate_basis = np.ones(shape=(3,3), dtype=ctype)
    transition_matrix = np.ones(shape=(3,3), dtype=ctype)

    get_transition_matrix_massbasis(baseline,
                                    energy,
                                    dm_mat_vac,
                                    dm_mat_mat,
                                    H_mat_mass_eigenstate_basis,
                                    transition_matrix)


@myjit
def convert_from_mass_eigenstate(state, mix_nubar, psi):
    '''
    Parameters
    ----------

    state : int

    mix_nubar : 2d-array

    psi : 1d-array (empty)


    Notes
    -----

    this is untested!
    '''
    mass = cuda.local.array(shape=(3), dtype=ctype)

    lstate = state - 1
    for i in range(3):
        mass[i] = 1. if lstate == i else 0.

    # note: mix_nubar is already taking into account whether we're considering
    # nu or anti-nu
    matrix_dot_vector(mix_nubar, mass, psi)

def test_convert_from_mass_eigenstate():
    state = 1
    psi = np.ones(shape=(3), dtype=ctype)
    mix_nubar = np.ones(shape=(3,3), dtype=ctype)

    convert_from_mass_eigenstate(state, mix_nubar, psi)
    #print(mix_nubar)


@myjit
def get_transition_matrix(nubar,
                          energy,
                          rho,
                          baseline,
                          mix_nubar,
                          mix_nubar_conj_transp,
                          mat_pot,
                          H_vac,
                          dm,
                          transition_matrix):
    ''' Calculate neutrino flavour transition amplitude matrix

    Parameters
    ----------

    nubar : int

    energy : float

    rho : float

    baseline : float

    mix_nubar : 2d-array
        Mixing matrix, already conjugated if antineutrino

    mix_nubar_conj_transp : comjugate 2d-array
        conjugate transpose of mixing matrix

    mat_pot : complex 2-d array
        Generalised matter potential matrix without a factor
        (will be multiplied with "a" factor)
        set to diag(1, 0, 0) for only standard oscillations

    H_vac : 2d-array

    dm : 2d-array

    transition_matrix : 2d-array (empty)
        in mass eigenstate basis

    Notes
    -----
    for neutrino (nubar > 0) or antineutrino (nubar < 0)
    with energy energy traversing layer of matter of
    uniform density rho with thickness baseline

    '''

    H_mat = cuda.local.array(shape=(3,3), dtype=ctype)
    dm_mat_vac = cuda.local.array(shape=(3,3), dtype=ctype)
    dm_mat_mat = cuda.local.array(shape=(3,3), dtype=ctype)
    H_full = cuda.local.array(shape=(3,3), dtype=ctype)
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)
    H_mat_mass_eigenstate_basis = cuda.local.array(shape=(3,3), dtype=ctype)

    # Compute the matter potential including possible generalised interactions
    # in the flavor basis
    get_H_mat(rho, mat_pot, nubar, H_mat)

    # Get the full Hamiltonian by adding together matter and vacuum parts
    one_over_two_e = 0.5 / energy
    for i in range(3):
        for j in range(3):
            H_full[i,j] = H_vac[i,j] * one_over_two_e + H_mat[i,j]

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
    get_transition_matrix_massbasis(baseline,
                                    energy,
                                    dm_mat_vac,
                                    dm_mat_mat,
                                    H_mat_mass_eigenstate_basis,
                                    transition_matrix)

def test_get_transition_matrix():
    nubar = 1
    energy = 1.
    rho = 1.
    baseline = 1.
    mix_nubar = np.ones(shape=(3,3), dtype=ctype)
    mat_pot = np.ones(shape=(3,3), dtype=ctype)
    H_vac = np.ones(shape=(3,3), dtype=ctype)
    m = np.linspace(0,1,9, dtype=ftype)
    dm = m.reshape(3,3)
    transition_matrix = np.ones(shape=(3,3), dtype=ctype)

    get_transition_matrix(nubar,
                          energy,
                          rho,
                          baseline,
                          mix_nubar,
                          mix_nubar.conj().T,
                          mat_pot,
                          H_vac,
                          dm,
                          transition_matrix)
    #print(transition_matrix)

@myjit
def osc_probs_vacuum_kernel(dm,
                            mix,
                            nubar,
                            energy,
                            distance_in_layer,
                            osc_probs):
    ''' Calculate vacumm mixing probabilities

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

    '''

    # no need to conjugate mix matrix, as we anyway only need real part
    # can this be right?

    clear_matrix(osc_probs)
    osc_probs_local = cuda.local.array(shape=(3,3), dtype=ftype)

    # sum up length from all layers
    baseline = 0.
    for i in range(distance_in_layer.shape[0]):
        baseline += distance_in_layer[i]

    # make more precise 20081003 rvw
    l_over_e = 1.26693281 * baseline / energy
    s21 = math.sin(dm[1,0] * l_over_e)
    s32 = math.sin(dm[2,0] * l_over_e)
    s31 = math.sin((dm[2,1] + dm[3,2]) * l_over_e)

    # does anybody understand this loop?
    # ista = abs(*nutype) - 1
    for ista in range(3):
        for iend in range(2):
            osc_probs_local[ista,iend] = ((mix[ista,0].real * mix[ista,1].real * s21)**2
                                          + (mix[ista,1].real * mix[ista,2].real * s32)**2
                                          + (mix[ista,2].real * mix[ista,0].real * s31)**2
                                         )
            if iend == ista:
                osc_probs_local[ista,iend] = 1. - 4. * osc_probs_local[ista,iend]
            else:
                osc_probs_local[ista,iend] = - 4. * osc_probs_local[ista,iend]

        osc_probs_local[ista,2] = 1. - osc_probs_local[ista,0] - osc_probs_local[ista,1]

    # is this necessary?
    if nubar > 0 :
        copy_matrix(osc_probs_local, osc_probs)
    else:
        for i in range(3):
            for j in range(3):
                osc_probs[i,j] = osc_probs_local[j,i]

@myjit
def osc_probs_layers_kernel(dm,
                            mix,
                            mat_pot,
                            nubar,
                            energy,
                            density_in_layer,
                            distance_in_layer,
                            osc_probs):
    ''' Calculate oscillation probabilities

    given layers of length and density

    Parameters
    ----------

    dm : 2d-array
        Mass splitting matrix

    mix : 2d-array
        PMNS mixing matrix

    H_vac : complex 2-d array
        Hamiltonian in vacuum, without the 1/2E term

    mat_pot : complex 2-d array
        Generalised matter potential matrix without a factor
        (will be multiplied with "a" factor)
        set to diag(1, 0, 0) for only standard oscillations

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

    '''

    # 3x3 complex
    H_vac = cuda.local.array(shape=(3,3), dtype=ctype)
    mix_nubar = cuda.local.array(shape=(3,3), dtype=ctype)
    mix_nubar_conj_transp = cuda.local.array(shape=(3,3), dtype=ctype)
    transition_product = cuda.local.array(shape=(3,3), dtype=ctype)
    transition_matrix = cuda.local.array(shape=(3,3), dtype=ctype)
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)

    clear_matrix(H_vac)
    clear_matrix(osc_probs)

    # 3-vector complex
    raw_input_psi = cuda.local.array(shape=(3), dtype=ctype)
    output_psi = cuda.local.array(shape=(3), dtype=ctype)

    use_mass_eigenstates = False

    cache = True
    #cache = False

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
        transition_matrices = cuda.local.array(shape=(120,3,3), dtype=ctype)

        # loop over layers
        for i in range(distance_in_layer.shape[0]):
            density = density_in_layer[i]
            distance = distance_in_layer[i]
            if distance > 0.:
                layer_matrix_index = -1
                # chaeck if exists
                for j in range(i):
                    #if density_in_layer[j] == density and distance_in_layer[j] == distance:
                    if (abs(density_in_layer[j] - density) < 1e-5) and (abs(distance_in_layer[j] - distance) < 1e-5):
                        layer_matrix_index = j

                # use from cached
                if layer_matrix_index >= 0:
                    for j in range(3):
                        for k in range(3):
                            transition_matrices[i,j,k] = transition_matrices[layer_matrix_index,j,k]

                # only calculate if necessary
                else:
                    get_transition_matrix(nubar,
                                          energy,
                                          density,
                                          distance,
                                          mix_nubar,
                                          mix_nubar_conj_transp,
                                          mat_pot,
                                          H_vac,
                                          dm,
                                          transition_matrix,
                                         )
                    # copy
                    for j in range(3):
                        for k in range(3):
                            transition_matrices[i,j,k] = transition_matrix[j,k]
            else:
                # identity matrix
                for j in range(3):
                    for k in range(3):
                        if j == k:
                            transition_matrix[j,k] = 0.
                        else:
                            transition_matrix[j,k] = 1.

        # now multiply them all
        first_layer = True
        for i in range(distance_in_layer.shape[0]):
            distance = distance_in_layer[i]
            if distance > 0.:
                for j in range(3):
                    for k in range(3):
                        transition_matrix[j,k] = transition_matrices[i,j,k]
                if first_layer:
                    copy_matrix(transition_matrix, transition_product)
                    first_layer = False
                else:
                    matrix_dot_matrix(transition_matrix,transition_product, tmp)
                    copy_matrix(tmp, transition_product)

    else:
        # non-cache loop
        first_layer = True
        for i in range(distance_in_layer.shape[0]):
            density = density_in_layer[i]
            distance = distance_in_layer[i]
            # only do something if distance > 0.
            if distance > 0.:
                get_transition_matrix(nubar,
                                      energy,
                                      density,
                                      distance,
                                      mix_nubar,
                                      mix_nubar_conj_transp,
                                      mat_pot,
                                      H_vac,
                                      dm,
                                      transition_matrix,
                                     )
                if first_layer:
                    copy_matrix(transition_matrix, transition_product)
                    first_layer = False
                else:
                    matrix_dot_matrix(transition_matrix,transition_product, tmp)
                    copy_matrix(tmp, transition_product)

    # convrt to flavour eigenstate basis
    matrix_dot_matrix(transition_product, mix_nubar_conj_transp, tmp)
    matrix_dot_matrix(mix_nubar, tmp, transition_product)

    # loop on neutrino types, and compute probability for neutrino i:
    for i in range(3):
        for j in range(3):
            raw_input_psi[j] = 0.

        if use_mass_eigenstates:
            convert_from_mass_eigenstate(i+1, mix_nubar, raw_input_psi)
        else:
            raw_input_psi[i] = 1.

        matrix_dot_vector(transition_product, raw_input_psi, output_psi)
        osc_probs[i][0] += output_psi[0].real**2 + output_psi[0].imag**2
        osc_probs[i][1] += output_psi[1].real**2 + output_psi[1].imag**2
        osc_probs[i][2] += output_psi[2].real**2 + output_psi[2].imag**2

def test_osc_probs_layers_kernel():
    mix = np.ones(shape=(3,3), dtype=ctype)
    mat_pot = np.ones(shape=(3,3), dtype=ctype)
    M = np.linspace(0,1,9, dtype=ftype)
    dm = M.reshape(3,3)
    nubar = 1
    energy = 1.
    n_layers = 10
    density_in_layer = np.ones(shape=(n_layers), dtype=ftype)
    distance_in_layer = np.ones(shape=(n_layers), dtype=ftype)
    osc_probs = np.ones(shape=(3,3), dtype=ftype)

    osc_probs_layers_kernel(dm,
                            mix,
                            mat_pot,
                            nubar,
                            energy,
                            density_in_layer,
                            distance_in_layer,
                            osc_probs)

# define numba functions

@guvectorize(
    [f'({FX}[:,:], {CX}[:,:], {CX}[:,:], i4, {FX}, {FX}[:], {FX}[:], {FX}[:,:])'],
    '(a,b), (c,d), (e,f), (), (), (g), (h) -> (a,b)',
    target=TARGET,
)
def propagate_array(dm, mix, nsi_eps, nubar, energy, densities, distances, probability):
    osc_probs_layers_kernel(dm, mix, nsi_eps, nubar, energy, densities, distances, probability)


@guvectorize(
    [f'({FX}[:,:], {CX}[:,:], i4, {FX}, {FX}[:], {FX}[:,:])'],
    '(a,b), (c,d), (), (), (i) -> (a,b)',
    target=TARGET,
)
def propagate_array_vacuum(dm, mix, nubar, energy, distances, probability):
    osc_probs_vacuum_kernel(dm, mix, nubar, energy, distances, probability)


@guvectorize([f'({FX}[:,:], i4, i4, {FX}[:])'], '(a,b), (), () -> ()', target=TARGET)
def fill_probs(probability, initial_flav, flav, out):
    out[0] = probability[initial_flav,flav]


if __name__=='__main__':

    assert TARGET == 'cpu', "Cannot test functions on GPU, set PISA_TARGET to 'cpu'"
    test_get_H_vac()
    test_get_H_mat()
    test_get_dms()
    test_get_product()
    test_get_transition_matrix()
    test_convert_from_mass_eigenstate()
    test_get_transition_matrix()
    test_osc_probs_layers_kernel()
