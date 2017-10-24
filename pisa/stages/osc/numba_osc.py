from __future__ import print_function

import numpy as np
from numba import jit, float64, complex64, int32, float32, complex128
import math, cmath

from numba_tools import *


@myjit
def get_H_vac(mix_nubar, delta_M_vac_vac, H_vac):
    ''' Calculate vacuum Hamiltonian in flavor basis for neutrino or antineutrino

    Parameters:
    -----------
    mix_nubar : complex 2d-array
        Mixing matrix (comjugate transpose for anti-neutrinos)

    delta_M_vac_vac: 2d-array
        Matrix of mass splittings

    H_vac: complex 2d-array (empty)
        Hamiltonian in vacuum, modulo a factor 2 * energy

    Notes
    ------
    The Hailtonian does not contain the energy dependent factor of
    1/(2 * E), as it will be added later

    '''
    delta_M_vac_diag = cuda.local.array(shape=(3,3), dtype=ctype)
    mix_nubar_conj_transpose = cuda.local.array(shape=(3,3), dtype=ctype)
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)

    clear_matrix(delta_M_vac_diag)

    delta_M_vac_diag[1,1] = delta_M_vac_vac[1,0] + 0j
    delta_M_vac_diag[2,2] = delta_M_vac_vac[2,0] + 0j

    conjugate_transpose(mix_nubar, mix_nubar_conj_transpose)
    matrix_dot_matrix(delta_M_vac_diag, mix_nubar_conj_transpose, tmp)
    matrix_dot_matrix(mix_nubar, tmp, H_vac)

def test_get_H_vac():
    mix = np.ones(shape=(3,3), dtype=ctype)
    delta_M_vac_vac = np.ones(shape=(3,3), dtype=ftype)

    H_vac = np.ones(shape=(3,3), dtype=ctype)
    get_H_vac(mix, delta_M_vac_vac, H_vac)
    #print(H_vac)


@myjit
def get_H_mat(rho, nsi_eps, nubar, H_mat):
    ''' Calculate full matter Hamiltonian in flavor basis 

    Parameters:
    -----------
    rho : float

    nsi_eps : complex 2-d array

    nubar : int

    H_mat : complex 2d-array (empty)

    Notes
    -----
    in the following, `a` is just the standard effective matter potential
    induced by charged-current weak interactions with electrons

    '''

    # 2*sqrt(2)*Gfermi in (eV^2-cm^3)/(mole-GeV)
    tworttwoGf = 1.52588e-4
    a = rho * tworttwoGf * nubar / 2.

    # standard matter interaction Hamiltonian
    clear_matrix(H_mat)
    H_mat[0,0] = a + 0j

    # Obtain effective non-standard matter interaction Hamiltonian
    nsi_rho_scale = 3. #// assume 3x electron density for "NSI"-quark (e.g., d) density
    fact = nsi_rho_scale * a
    for i in range(3):
        for j in range(3):
            H_mat[i,j] += fact * nsi_eps[i,j]

def test_get_H_mat():
    rho = 1.
    nubar = -1
    nsi_eps = np.ones(shape=(3,3), dtype=ctype)
    H_mat = np.ones(shape=(3,3), dtype=ctype)

    get_H_mat(rho, nsi_eps, nubar, H_mat)
    #print(H_mat)

@myjit
def get_delta_Ms(energy, H_mat, delta_M_vac_vac, delta_M_mat_mat, delta_M_mat_vac):
    '''Compute the matter-mass vector M, dM = M_i-M_j and dMimj

    Parameters
    ----------
    energy : float
        Neutrino energy

    H_mat : complex 2d-array

    delta_M_vac_vac : 2d array

    delta_M_mat_mat : 2d-array (empty)

    delta_M_mat_vac : 2d-array (empty)


    Notes
    -----
    Calculate mass eigenstates in matter 
    neutrino or anti-neutrino (type already taken into account in Hamiltonian)
    of energy energy. 

    '''

    real_product_a = (H_mat[0,1] * H_mat[1,2] * H_mat[2,0]).real
    real_product_b = (H_mat[0,0] * H_mat[1,1] * H_mat[2,2]).real

    norm_H_e_mu_sq = H_mat[0,1].real**2 + H_mat[0,1].imag**2
    norm_H_e_tau_sq = H_mat[0,2].real**2 + H_mat[0,2].imag**2
    norm_H_mu_tau_sq = H_mat[1,2].real**2 + H_mat[1,2].imag**2

    c1 = ((H_mat[0,0].real * (H_mat[1,1] + H_mat[2,2])).real
          - (H_mat[0,0].imag * (H_mat[1,1] + H_mat[2,2])).imag
          + (H_mat[1,1].real * H_mat[2,2]).real
          - (H_mat[1,1].imag * H_mat[2,2]).imag
          - norm_H_e_mu_sq
          - norm_H_mu_tau_sq
          - norm_H_e_tau_sq
         )

    c0 = (H_mat[0,0].real * norm_H_mu_tau_sq
          + H_mat[1,1].real * norm_H_e_tau_sq
          + H_mat[2,2].real * norm_H_e_mu_sq
          - 2. * real_product_a
          - real_product_b
         )

    c2 = - H_mat[0,0].real - H_mat[1,1].real - H_mat[2,2].real

    twoE = 2.*energy
    twoE_sq = twoE * twoE
    twoE_cu = twoE_sq * twoE

    c2V = (-1./twoE) * (delta_M_vac_vac[1,0] + delta_M_vac_vac[2,0])

    p = c2*c2 - 3.*c1
    pV = (1./twoE_sq) * (delta_M_vac_vac[1,0] * delta_M_vac_vac[1,0]
                         + delta_M_vac_vac[2,0] * delta_M_vac_vac[2,0]
                         - delta_M_vac_vac[1,0] * delta_M_vac_vac[2,0]
                        )
    p = max(0., p)

    q = -27. * c0/2. - c2*c2*c2 + 9. * c1*c2 / 2.
    qV = (1./twoE_cu) * ((delta_M_vac_vac[1,0] + delta_M_vac_vac[2,0])
                          * (delta_M_vac_vac[1,0] + delta_M_vac_vac[2,0])
                          * (delta_M_vac_vac[1,0] + delta_M_vac_vac[2,0])
                          - (9./2.) * delta_M_vac_vac[1,0] * delta_M_vac_vac[2,0]
                          * (delta_M_vac_vac[1,0] + delta_M_vac_vac[2,0])
                         )

    tmp = p*p*p - q*q
    tmpV = pV*pV*pV - qV*qV

    tmp = max(0., tmp)

    theta = cuda.local.array(shape=(3), dtype=ftype)
    thetaV = cuda.local.array(shape=(3), dtype=ftype)
    mMat = cuda.local.array(shape=(3), dtype=ftype)
    mMatU = cuda.local.array(shape=(3), dtype=ftype)
    mMatV = cuda.local.array(shape=(3), dtype=ftype)


    a = (2./3.) * math.pi
    res = math.atan2(math.sqrt(tmp), q) / 3.
    theta[0] = res + a
    theta[1] = res - a
    theta[2] = res
    resV = math.atan2(math.sqrt(tmpV), qV) / 3.
    thetaV[0] = resV + a
    thetaV[1] = resV - a
    thetaV[2] = resV
    
    for i in range(3):
        mMatU[i] = 2.*energy * ((2./3.) * math.sqrt(p) * math.cos(theta[i])
                                - c2/3. + delta_M_vac_vac[0,0])
        mMatV[i] = 2.*energy * ((2./3.) * math.sqrt(pV) * math.cos(thetaV[i])
                                - c2V/3. + delta_M_vac_vac[0,0])

    # Sort according to which reproduce the vaccum eigenstates 
    for i in range(3):
        tmpV = abs(delta_M_vac_vac[i,0] - mMatV[0])
        k = 0
        for j in range(3):
            tmp = abs(delta_M_vac_vac[i,0] - mMatV[j])
            if tmp < tmpV:
                k = j
                tmpV = tmp
        mMat[i] = mMatU[k]

    for i in range(3):
        for j in range(3):
              delta_M_mat_mat[i,j] = mMat[i] - mMat[j]
              delta_M_mat_vac[i,j] = mMat[i] - delta_M_vac_vac[j,0]

def test_get_delta_Ms():
    energy = 1.
    delta_M_vac_vac = np.ones(shape=(3,3), dtype=ftype)
    delta_M_mat_mat = np.ones(shape=(3,3), dtype=ctype)
    delta_M_mat_vac = np.ones(shape=(3,3), dtype=ctype)
    H_mat = np.ones(shape=(3,3), dtype=ctype)

    get_delta_Ms(energy, H_mat, delta_M_vac_vac, delta_M_mat_mat, delta_M_mat_vac)

@myjit
def get_product(energy,
                delta_M_mat_vac,
                delta_M_mat_mat,
                H_mat_mass_eigenstate_basis,
                product):
    '''
    Parameters
    ----------

    energy : float

    delta_M_mat_vac : 2d-array

    delta_M_mat_mat : 2d-array

    H_mat_mass_eigenstate_basis : 2d-array

    product : 3d-array (empty)

    '''

    H_minus_M = cuda.local.array(shape=(3,3,3), dtype=ctype)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                H_minus_M[i,j,k] = 2. * energy * H_mat_mass_eigenstate_basis[i,j]
                if i == j:
                    H_minus_M[i,j,k] -= delta_M_mat_vac[k,j]
                #also, cler product
                product[i,j,k] = 0.

    # Calculate the product in eq.(10) of H_minus_M for j!=k 
    for i in range(3):
        for j in range(3):
            for k in range(3):
                product[i,j,0] += (H_minus_M[i,k,1] * H_minus_M[k,j,2])
                product[i,j,1] += (H_minus_M[i,k,2] * H_minus_M[k,j,0])
                product[i,j,2] += (H_minus_M[i,k,0] * H_minus_M[k,j,1])
            product[i,j,0] /= (delta_M_mat_mat[0,1] * delta_M_mat_mat[0,2])
            product[i,j,1] /= (delta_M_mat_mat[1,2] * delta_M_mat_mat[1,0])
            product[i,j,2] /= (delta_M_mat_mat[2,0] * delta_M_mat_mat[2,1])

def test_get_product():
    baseline = 1.
    energy = 1.
    delta_M_mat_mat = np.ones(shape=(3,3), dtype=ctype)
    delta_M_mat_vac = np.ones(shape=(3,3), dtype=ctype)
    H_mat_mass_eigenstate_basis = np.ones(shape=(3,3), dtype=ctype)
    product = np.ones(shape=(3,3,3), dtype=ctype)

    get_product(energy,
                delta_M_mat_vac,
                delta_M_mat_mat,
                H_mat_mass_eigenstate_basis,
                product)

@myjit
def get_transition_amplitude_matrix(baseline,
                                    energy,
                                    mix,
                                    delta_M_mat_vac,
                                    delta_M_mat_mat,
                                    H_mat_mass_eigenstate_basis,
                                    transition_matrix):
    '''
    Calculate the transition amplitude matrix

    Parameters
    ----------

    baseline : float

    energy : float

    mix : 2d-array

    delta_M_mat_vac : 2d-array

    delta_M_mat_mat : 2d-array
    
    H_mat_mass_eigenstate_basis : 2-d array

    transition_matrix : 2d-array
    
    Notes
    -----
    - corrsponds to matrix A (equation 10) in original Barger paper
    - take into account generic potential matrix (=Hamiltonian)

    '''
    X = cuda.local.array(shape=(3,3), dtype=ctype)
    product = cuda.local.array(shape=(3,3,3), dtype=ctype)
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)
    mix_conj_transpose = cuda.local.array(shape=(3,3), dtype=ctype)

    clear_matrix(X)
    clear_matrix(transition_matrix)

    get_product(energy,
                delta_M_mat_vac,
                delta_M_mat_mat,
                H_mat_mass_eigenstate_basis,
                product)

    # (1/2)*(1/(h_bar*c)) in units of GeV/(eV^2-km)
    hbar_c_factor = 2.534

    for k in range(3):
        arg = - delta_M_mat_vac[k,0] * (baseline / energy) * hbar_c_factor
        for i in range(3):
            for j in range(3):
                X[i,j] += cmath.exp(arg * 1.j) * product[i,j,k]

    # Compute the product with the mixing matrices 
    conjugate_transpose(mix, mix_conj_transpose)
    matrix_dot_matrix(X, mix_conj_transpose, tmp)
    matrix_dot_matrix(mix, tmp, transition_matrix)

def test_get_transition_amplitude_matrix():
    baseline = 1.
    energy = 1.
    mix = np.ones(shape=(3,3), dtype=ctype)
    delta_M_mat_mat = np.ones(shape=(3,3), dtype=ctype)
    delta_M_mat_vac = np.ones(shape=(3,3), dtype=ctype)
    H_mat_mass_eigenstate_basis = np.ones(shape=(3,3), dtype=ctype)
    transition_matrix = np.ones(shape=(3,3), dtype=ctype)

    get_transition_amplitude_matrix(baseline,
                                    energy,
                                    mix,
                                    delta_M_mat_vac,
                                    delta_M_mat_mat,
                                    H_mat_mass_eigenstate_basis,
                                    transition_matrix)


@myjit
def convert_from_mass_eigenstate(state, mix_nubar, psi):
    '''
    Parameters
    ----------

    state : int

    mix_nubar : 2d-array

    psi : 1d-array


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
                          nsi_eps,
                          H_vac,
                          delta_M,
                          transition_matrix):
    ''' Calculate neutrino flavour transition amplitude matrixi
    
    Parameters
    ----------
    
    nubar : int
    
    energy : float

    baseline : float

    mix_nubar : 2d-array

    nsi_eps : 2d-array

    H_vac : 2d-array

    delta_M : 2d-array

    transition_matrix : 2d-array
    
    Notes
    -----
    for neutrino (nubar > 0) or antineutrino (nubar < 0)
    with energy energy traversing layer of matter of
    uniform density rho with thickness baseline

    '''

    H_mat = cuda.local.array(shape=(3,3), dtype=ctype)
    delta_M_mat_vac = cuda.local.array(shape=(3,3), dtype=ctype)
    delta_M_mat_mat = cuda.local.array(shape=(3,3), dtype=ctype)
    H_full = cuda.local.array(shape=(3,3), dtype=ctype)
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)
    H_mat_mass_eigenstate_basis = cuda.local.array(shape=(3,3), dtype=ctype)
    mix_nubar_conj_transpose = cuda.local.array(shape=(3,3), dtype=ctype)

    # Compute the matter potential including possible non-standard interactions
    # in the flavor basis 
    get_H_mat(rho, nsi_eps, nubar, H_mat)

    # Get the full Hamiltonian by adding together matter and vacuum parts 
    for i in range(3):
        for j in range(3):
            H_full[i,j] = H_vac[i,j] / (2.*energy) + H_mat[i,j]

    # Calculate modified mass eigenvalues in matter from the full Hamiltonian and
    # the vacuum mass splittings 
    get_delta_Ms(energy, H_full, delta_M, delta_M_mat_mat, delta_M_mat_vac)

    # Now we transform the matter (TODO: matter? full?) Hamiltonian back into the
    # mass eigenstate basis so we don't need to compute products of the effective
    # mixing matrix elements explicitly 
    conjugate_transpose(mix_nubar, mix_nubar_conj_transpose)
    matrix_dot_matrix(H_mat, mix_nubar, tmp)
    matrix_dot_matrix(mix_nubar_conj_transpose, tmp, H_mat_mass_eigenstate_basis)

    # We can now proceed to calculating the transition amplitude from the Hamiltonian
    # in the mass basis and the effective mass splittings 
    get_transition_amplitude_matrix(baseline,
                                    energy,
                                    mix_nubar,
                                    delta_M_mat_vac,
                                    delta_M_mat_mat,
                                    H_mat_mass_eigenstate_basis,
                                    transition_matrix)

def test_get_transition_matrix():
    nubar = 1
    energy = 1.
    rho = 1.
    baseline = 1.
    mix_nubar = np.ones(shape=(3,3), dtype=ctype)
    nsi_eps = np.ones(shape=(3,3), dtype=ctype)
    H_vac = np.ones(shape=(3,3), dtype=ctype)
    m = np.linspace(0,1,9, dtype=ftype)
    delta_M = m.reshape(3,3)
    transition_matrix = np.ones(shape=(3,3), dtype=ctype)

    get_transition_matrix(nubar,
                          energy,
                          rho,
                          baseline,
                          mix_nubar,
                          nsi_eps,
                          H_vac,
                          delta_M,
                          transition_matrix)
    #print(transition_matrix)


@myjit
def propagate_array_kernel(delta_M,
                           mix,
                           nsi_eps,
                           nubar,
                           flav,
                           energy,
                           n_layers,
                           density_in_layer,
                           distance_in_layer,
                           osc_probs):
    '''
    Parameters
    ----------

    delta_M : 2d-array

    mix : 2d-array

    nsi_eps : 2d-array

    nubar : int

    flav : int

    energy : float

    n_layers : int

    density_in_layer : 1d-array

    distance_in_layer : 1d-array

    osc_probs : 2d-array

    '''

    # 3x3 complex
    H_vac = cuda.local.array(shape=(3,3), dtype=ctype)
    mix_nubar = cuda.local.array(shape=(3,3), dtype=ctype)
    transition_product = cuda.local.array(shape=(3,3), dtype=ctype)
    transition_matrix = cuda.local.array(shape=(3,3), dtype=ctype)
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)

    clear_matrix(H_vac)
    clear_matrix(osc_probs)

    # 3-vector complex
    raw_input_psi = cuda.local.array(shape=(3), dtype=ctype)
    output_psi = cuda.local.array(shape=(3), dtype=ctype)

    use_mass_eigenstates = False

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
        conjugate_transpose(mix, mix_nubar)

    get_H_vac(mix_nubar, delta_M, H_vac)


    for i in range(n_layers):
        density = density_in_layer[i]
        distance = distance_in_layer[i]
        get_transition_matrix(nubar,
                              energy,
                              density,
                              distance,
                              mix_nubar,
                              nsi_eps,
                              H_vac,
                              delta_M,
                              transition_matrix,
                              )
        if i == 0:
            copy_matrix(transition_matrix, transition_product)
        else:
            matrix_dot_matrix(transition_matrix,transition_product, tmp)
            copy_matrix(tmp, transition_product)
        
    # loop on neutrino types, and compute probability for neutrino i:
    # We actually don't care about nutau -> anything since the flux there is zero!
    for i in range(2):
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

def test_propagate_array_kernel():
    mix = np.ones(shape=(3,3), dtype=ctype)
    nsi_eps = np.ones(shape=(3,3), dtype=ctype)
    M = np.linspace(0,1,9, dtype=ftype)
    delta_M = M.reshape(3,3)
    nubar = 1
    flav = 1
    energy = 1.
    n_layers = 10
    density_in_layer = np.ones(shape=(n_layers), dtype=ftype)
    distance_in_layer = np.ones(shape=(n_layers), dtype=ftype)
    osc_probs = np.ones(shape=(3,3), dtype=ftype)

    propagate_array_kernel(delta_M,
                           mix,
                           nsi_eps,
                           nubar,
                           flav,
                           energy,
                           n_layers,
                           density_in_layer,
                           distance_in_layer,
                           osc_probs)


if __name__=='__main__':
    
    assert target == 'cpu', "Cannot test functions on GPU, set target='cpu'"
    test_get_H_vac()
    test_get_H_mat()
    test_get_delta_Ms()
    test_get_product()
    test_get_transition_matrix()
    test_convert_from_mass_eigenstate()
    test_get_transition_matrix()
    test_propagate_array_kernel()
