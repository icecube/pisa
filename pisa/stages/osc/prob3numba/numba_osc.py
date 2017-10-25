'''Neutrino flavour oscillation in matter calculation
Based on the original prob3++ implementation of Roger Wendell
http://www.phy.duke.edu/~raw22/public/Prob3++/ (2012)
'''

from __future__ import print_function, division

__all__ = ['get_transition_matrix',
           'osc_probs_layers_kernel',
           ]
__version__ = '0.1'
__author__ = 'Philipp Eller (pder@psu.edu)'

import math, cmath

import numpy as np
from numba import jit, float64, complex64, int32, float32, complex128

from numba_tools import *


@myjit
def get_H_mat(rho, nsi_eps, nubar, H_mat):
    ''' Calculate matter Hamiltonian in flavor basis 

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
    a = 0.5 * rho * tworttwoGf * nubar

    # standard matter interaction Hamiltonian
    clear_matrix(H_mat)
    H_mat[0,0] = a

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

    - only god knows what happens in this function, somehow it seems to work

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

    one_over_two_e = 0.5 / energy
    one_third = 1./3.
    two_third = 2./3.

    x = delta_M_vac_vac[1,0]
    y = delta_M_vac_vac[2,0]

    c2_v = - one_over_two_e * (x + y)

    p = c2**2 - 3.*c1
    p_v = one_over_two_e**2 * (x**2 + y**2 - x*y)
    p = max(0., p)

    q = -13.5*c0 - c2**3 + 4.5*c1*c2
    q_v = one_over_two_e**3 * (x + y) * ((x + y)**2 - 4.5*x*y)

    tmp = p**3 - q**2
    tmp_v = p_v**3 - q_v**2

    tmp = max(0., tmp)

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
        m_mat_u[i] = 2. * energy * (b * math.cos(theta[i]) - c2*one_third + delta_M_vac_vac[0,0])
        m_mat_v[i] = 2. * energy * (b_v * math.cos(theta_v[i]) - c2_v*one_third + delta_M_vac_vac[0,0])

    # Sort according to which reproduce the vaccum eigenstates 
    for i in range(3):
        tmp_v = abs(delta_M_vac_vac[i,0] - m_mat_v[0])
        k = 0
        for j in range(3):
            tmp = abs(delta_M_vac_vac[i,0] - m_mat_v[j])
            if tmp < tmp_v:
                k = j
                tmp_v = tmp
        m_mat[i] = m_mat_u[k]

    for i in range(3):
        for j in range(3):
              delta_M_mat_mat[i,j] = m_mat[i] - m_mat[j]
              delta_M_mat_vac[i,j] = m_mat[i] - delta_M_vac_vac[j,0]

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

    transition_matrix : 2d-array (empty)
    
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
        c = cmath.exp(arg * 1.j)
        for i in range(3):
            for j in range(3):
                X[i,j] += c * product[i,j,k]

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

    transition_matrix : 2d-array (empty)
    
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
    one_over_two_e = 0.5 / energy
    for i in range(3):
        for j in range(3):
            H_full[i,j] = H_vac[i,j] * one_over_two_e + H_mat[i,j]

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
def osc_probs_layers_kernel(delta_M,
                           mix,
                           H_vac,
                           nsi_eps,
                           nubar,
                           energy,
                           density_in_layer,
                           distance_in_layer,
                           osc_probs):
    ''' Calculate oscillation probabilities

    given layers of length and density

    Parameters
    ----------

    delta_M : 2d-array

    mix : 2d-array

    nsi_eps : 2d-array

    nubar : int

    energy : float

    n_layers : int

    density_in_layer : 1d-array

    distance_in_layer : 1d-array

    osc_probs : 2d-array (empty)

    '''

    # 3x3 complex
    #H_vac = cuda.local.array(shape=(3,3), dtype=ctype)
    mix_nubar = cuda.local.array(shape=(3,3), dtype=ctype)
    transition_product = cuda.local.array(shape=(3,3), dtype=ctype)
    transition_matrix = cuda.local.array(shape=(3,3), dtype=ctype)
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)

    #clear_matrix(H_vac)
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
        conjugate_transpose(mix, mix_nubar)


    if cache:
        # allocate array to store all the transition matrices
        #transition_matrices = cuda.local.array(shape=(distance_in_layer.shape[0],3,3), dtype=ctype)
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
                                          nsi_eps,
                                          H_vac,
                                          delta_M,
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
                                      nsi_eps,
                                      H_vac,
                                      delta_M,
                                      transition_matrix,
                                      )
                if first_layer:
                    copy_matrix(transition_matrix, transition_product)
                    first_layer = False
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

def test_osc_probs_layers_kernel():
    mix = np.ones(shape=(3,3), dtype=ctype)
    nsi_eps = np.ones(shape=(3,3), dtype=ctype)
    M = np.linspace(0,1,9, dtype=ftype)
    delta_M = M.reshape(3,3)
    nubar = 1
    energy = 1.
    n_layers = 10
    density_in_layer = np.ones(shape=(n_layers), dtype=ftype)
    distance_in_layer = np.ones(shape=(n_layers), dtype=ftype)
    osc_probs = np.ones(shape=(3,3), dtype=ftype)

    osc_probs_layers_kernel(delta_M,
                           mix,
                           nsi_eps,
                           nubar,
                           energy,
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
    test_osc_probs_layers_kernel()
