from __future__ import print_function
import numpy as np
import time
import inspect
from numba import jit, vectorize, guvectorize, float64, complex64, int32, float32, complex128
import math, cmath

target='cuda'
#target='parallel'
#target='cpu'

if target == 'cuda':
    from numba import cuda
    ctype = complex128
    ftype = float64
else:
    ctype = np.complex128
    ftype = np.float64
    cuda = lambda: None
    cuda.jit = lambda x: x

def myjit(f):
    '''
    Decorator to assign the right jit for different targets
    In case of non-cuda targets, all instances of `cuda.local.array`
    are replaced by `np.empty`. This is a dirty fix, hopefully in the
    near future numba will support numpy array allocation and this will
    not be necessary anymore
    '''
    if target == 'cuda':
        return cuda.jit(f, device=True)
    else:
        source = inspect.getsource(f).splitlines()
        assert '@myjit' in source[0]
        source = '\n'.join(source[1:]) + '\n'
        source = source.replace('cuda.local.array', 'np.empty')
        exec(source)
        fun = eval(f.__name__)
        return jit(fun, nopython=True)

@myjit
def conjugate_transpose(A, B):
    '''
    B is the conjugate transpose of A
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i,j] = A[j,i].conjugate()

@myjit
def matrix_dot_matrix(A, B, C):
    '''
    dot-product of two 2d arrays
    C = A * B
    '''
    for j in range(B.shape[1]):
        for i in range(A.shape[0]):
            for n in range(C.shape[0]):
                C[i,j] += A[i,n] * B[n,j]

def test_matrix_dot_matrix():
    A = np.linspace(1., 8., 9).reshape(3,3)
    B = np.linspace(1., 8., 9).reshape(3,3)
    C = np.clear_matrixs((3,3))
    matrix_dot_matrix(A, B, C)
    assert np.array_equal(C, np.dot(A, B))

@myjit
def matrix_dot_vector(A, v, w):
    '''
    dot-product of a 2d array and a vector
    w = A * v
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            w[i] += A[i,j] * v[j]

def test_matrix_dot_vector():
    A = np.linspace(1., 8., 9).reshape(3,3)
    v = np.linspace(1., 3., 3)
    w = np.clear_matrixs((3))
    matrix_dot_vector(A, v, w)
    assert np.array_equal(w, np.dot(A, v))

@myjit
def clear_matrix(A):
    '''
    clear out 2d array
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i,j] = 0.

def test_clear_matrix():
    A = np.ones((3,3))
    clear_matrix(A)
    assert np.array_equal(A, np.clear_matrixs((3,3)))

@myjit
def copy_matrix(A, B):
    '''
    copy elemnts of 2d array A to array B
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i,j] = A[i,j]

def test_copy_matrix():
    A = np.ones((3,3))
    B = np.clear_matrixs((3,3))
    copy_matrix(A, B)
    assert np.array_equal(A, B)

@myjit
def get_HVac2energy(Mix, delta_M_vac_vac, HVac2energy):
    '''
    Calculate vacuum Hamiltonian in flavor basis for neutrino or 
    antineutrino (need complex conjugate mixing matrix) of energy E.
    '''
    delta_MVacDiag = cuda.local.array(shape=(3,3), dtype=ctype)
    MixConjTranspose = cuda.local.array(shape=(3,3), dtype=ctype)
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)

    clear_matrix(delta_MVacDiag)
    clear_matrix(tmp)

    delta_MVacDiag[1,1] = delta_M_vac_vac[1,0] + 0j
    delta_MVacDiag[2,2] = delta_M_vac_vac[2,0] + 0j

    conjugate_transpose(Mix, MixConjTranspose)
    matrix_dot_matrix(delta_MVacDiag, MixConjTranspose, tmp)
    matrix_dot_matrix(Mix, tmp, HVac2energy)

def test_get_HVac2energy():
    Mix = np.ones(shape=(3,3), dtype=ctype)
    delta_M_vac_vac = np.ones(shape=(3,3), dtype=ftype)

    HVac2energy = np.ones(shape=(3,3), dtype=ctype)
    get_HVac2energy(Mix, delta_M_vac_vac, HVac2energy)
    #print(HVac2energy)

@myjit
def get_H_mat(rho, nsi_eps, antitype, H_mat):
    '''
    Calculate full matter Hamiltonian in flavor basis 

    in the following, `a` is just the standard effective matter potential
    induced by charged-current weak interactions with electrons
    (modulo a factor of 2E)
    Calculate effective non-standard interaction Hamiltonian in flavor basis 
    '''
    # 2*sqrt(2)*Gfermi in (eV^2-cm^3)/(mole-GeV)
    tworttwoGf = 1.52588e-4

    # standard matter interaction Hamiltonian
    a = rho * tworttwoGf * antitype / 2.
    H_mat[0,0] = a + 0j

    # Obtain effective non-standard matter interaction Hamiltonian
    nsi_rho_scale = 3. #// assume 3x electron density for "NSI"-quark (e.g., d) density
    fact = nsi_rho_scale * a
    for i in range(3):
        for j in range(3):
            H_mat[i,j] += fact * nsi_eps[i,j]

def test_get_H_mat():
    rho = 1.
    antitype = -1
    nsi_eps = np.ones(shape=(3,3), dtype=ctype)
    H_mat = np.ones(shape=(3,3), dtype=ctype)

    get_H_mat(rho, nsi_eps, antitype, H_mat)
    #print(H_mat)

@myjit
def get_M(energy, rho, delta_M_vac_vac, delta_M_mat_mat, delta_M_mat_vac, H_mat):
    '''
    Compute the matter-mass vector M, dM = M_i-M_j and dMimj

    Calculate mass eigenstates in matter of uniform density rho for
    neutrino or anti-neutrino (type already taken into account in Hamiltonian)
    of energy energy. 
    '''

    real_product_a = (H_mat[0,1] * H_mat[1,2] * H_mat[2,0]).real
    real_product_b = (H_mat[0,0] * H_mat[1,1] * H_mat[2,2]).real

    norm_H_e_mu_sq =   H_mat[0,1].real**2 + H_mat[0,1].imag**2
    norm_H_e_tau_sq =  H_mat[0,2].real**2 + H_mat[0,2].imag**2
    norm_H_mu_tau_sq = H_mat[1,2].real**2 + H_mat[1,2].imag**2

    c1 = ( (H_mat[0,0].real * (H_mat[1,1] + H_mat[2,2])).real
         - (H_mat[0,0].imag * (H_mat[1,1] + H_mat[2,2])).imag
         + (H_mat[1,1].real * H_mat[2,2]).real
         - (H_mat[1,1].imag * H_mat[2,2]).imag
         - norm_H_e_mu_sq
         - norm_H_mu_tau_sq
         - norm_H_e_tau_sq
         )

    c0 = ( H_mat[0,0].real * norm_H_mu_tau_sq
         + H_mat[1,1].real * norm_H_e_tau_sq
         + H_mat[2,2].real   * norm_H_e_mu_sq
         - 2. * real_product_a
         - real_product_b
         )

    c2 = - H_mat[0,0].real - H_mat[1,1].real - H_mat[2,2].real

    twoE = 2.*energy
    twoE_sq = twoE * twoE
    twoE_cu = twoE_sq * twoE

    c2V = (-1./twoE) * (delta_M_vac_vac[1,0] + delta_M_vac_vac[2,0])

    p = c2 * c2 - 3. * c1
    pV = (1./twoE_sq) * (delta_M_vac_vac[1,0] * delta_M_vac_vac[1,0]
                         + delta_M_vac_vac[2,0] * delta_M_vac_vac[2,0]
                         - delta_M_vac_vac[1,0] * delta_M_vac_vac[2,0]
                        )
    p = max(0., p)

    q = -27. * c0 / 2.0 - c2*c2*c2 + 9. * c1*c2 / 2.
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
        mMatU[i] = 2.*energy * ((2./3.) * math.sqrt(p) * math.cos(theta[i]) - c2/3. + delta_M_vac_vac[0,0])
        mMatV[i] = 2.*energy * ((2./3.) * math.sqrt(pV) * math.cos(thetaV[i]) - c2V/3. + delta_M_vac_vac[0,0])

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

def test_get_M():
    energy = 1.
    # not needed here??
    rho = 1.
    delta_M_vac_vac = np.ones(shape=(3,3), dtype=ftype)
    delta_M_mat_mat = np.ones(shape=(3,3), dtype=ctype)
    delta_M_mat_vac = np.ones(shape=(3,3), dtype=ctype)
    H_mat = np.ones(shape=(3,3), dtype=ctype)

    get_M(energy, rho, delta_M_vac_vac, delta_M_mat_mat, delta_M_mat_vac, H_mat)
    #print(delta_M_mat_mat)
    #print(delta_M_mat_vac)

@myjit
def getProduct(baseline, energy, rho, delta_M_mat_vac, delta_M_mat_mat, H_matmass_eigenstate_basis, product):

    twoEHmM = cuda.local.array(shape=(3,3,3), dtype=ctype)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                twoEHmM[i,j,k] = 2. * energy * H_matmass_eigenstate_basis[i,j]  

    for n in range(3):
        for j in range(3):
            twoEHmM[n,n,j] -= delta_M_mat_vac[j,n]

    # Calculate the product in eq.(10) of twoEHmM for j!=k 
    for i in range(3):
        for j in range(3):
            for k in range(3):
                product[i,j,k] = 0.

    #print(delta_M_mat_mat)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                product[i,j,0] += (twoEHmM[i,k,1] * twoEHmM[k,j,2])
                product[i,j,1] += (twoEHmM[i,k,2] * twoEHmM[k,j,0])
                product[i,j,2] += (twoEHmM[i,k,0] * twoEHmM[k,j,1])
            product[i,j,0] /= (delta_M_mat_mat[0,1] * delta_M_mat_mat[0,2])
            product[i,j,1] /= (delta_M_mat_mat[1,2] * delta_M_mat_mat[1,0])
            product[i,j,2] /= (delta_M_mat_mat[2,0] * delta_M_mat_mat[2,1])

def test_getProduct():
    baseline = 1.
    energy = 1.
    rho = 1.
    delta_M_mat_mat = np.ones(shape=(3,3), dtype=ctype)
    delta_M_mat_vac = np.ones(shape=(3,3), dtype=ctype)
    H_matmass_eigenstate_basis = np.ones(shape=(3,3), dtype=ctype)
    product = np.ones(shape=(3,3,3), dtype=ctype)

    getProduct(baseline, energy, rho, delta_M_mat_vac, delta_M_mat_mat, H_matmass_eigenstate_basis, product)
    #print(product)

@myjit
def get_A(baseline, energy, rho, Mix, delta_M_mat_vac, delta_M_mat_mat, H_matmass_eigenstate_basis, phase_offset, transition_matrix):
    '''
    get_A (take into account generic potential matrix (=Hamiltonian))
    Calculate the transition amplitude matrix A (equation 10)
    '''
    X = cuda.local.array(shape=(3,3), dtype=ctype)
    product = cuda.local.array(shape=(3,3,3), dtype=ctype)
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)
    MixConjTranspose = cuda.local.array(shape=(3,3), dtype=ctype)

    clear_matrix(X)
    clear_matrix(tmp)
    clear_matrix(transition_matrix)

    if phase_offset == 0.:
        getProduct(baseline, energy, rho, delta_M_mat_vac, delta_M_mat_mat, H_matmass_eigenstate_basis, product)
    # what if phase_offset != 0.0??

    # (1/2)*(1/(h_bar*c)) in units of GeV/(eV^2-km)
    hbar_c_factor = 2.534

    for k in range(3):
        arg = - delta_M_mat_vac[k,0] * (baseline / energy) * hbar_c_factor
        if k == 2:
            arg += phase_offset 
        for i in range(3):
            for j in range(3):
                X[i,j] += cmath.exp(arg * 1.j) * product[i,j,k]

    # Compute the product with the mixing matrices 
    conjugate_transpose(Mix, MixConjTranspose)
    matrix_dot_matrix(X, MixConjTranspose, tmp)
    matrix_dot_matrix(Mix, tmp, transition_matrix)

def test_get_A():
    baseline = 1.
    energy = 1.
    rho = 1.
    Mix = np.ones(shape=(3,3), dtype=ctype)
    delta_M_mat_mat = np.ones(shape=(3,3), dtype=ctype)
    delta_M_mat_vac = np.ones(shape=(3,3), dtype=ctype)
    H_matmass_eigenstate_basis = np.ones(shape=(3,3), dtype=ctype)
    phase_offset = 0.
    transition_matrix = np.ones(shape=(3,3), dtype=ctype)

    get_A(baseline, energy, rho, Mix,  delta_M_mat_vac, delta_M_mat_mat, H_matmass_eigenstate_basis, phase_offset, transition_matrix)
    #print(transition_matrix)


@myjit
def convert_from_mass_eigenstate(state, pure, mixNuType):
    '''
    untested!
    '''
    mass = cuda.local.array(shape=(3), dtype=ctype)

    lstate  = state - 1
    for i in range(3):
        mass[i] = 1. if lstate == i else 0.

    # note: mixNuType is already taking into account whether we're considering
    # nu or anti-nu
    matrix_dot_vector(mixNuType, mass, pure)

def test_convert_from_mass_eigenstate():
    state = 1
    pure = np.ones(shape=(3), dtype=ctype)
    mixNuType = np.ones(shape=(3,3), dtype=ctype)

    convert_from_mass_eigenstate(state, pure, mixNuType)
    #print(mixNuType)

@myjit
def get_transition_matrix(antitype, energy, rho, baseline,
                           phase_offset,
                           mixNuType,  nsi_eps,
                           HVac2energy,  delta_M, transition_matrix):
    '''
    Calculate neutrino flavour transition amplitude matrix for neutrino (antitype > 0)
    or antineutrino (antitype < 0) with energy energy travernp.sing layer of matter of
    uniform density rho with thickness Len.
    '''
    H_mat = cuda.local.array(shape=(3,3), dtype=ctype)
    delta_M_mat_vac = cuda.local.array(shape=(3,3), dtype=ctype)
    delta_M_mat_mat = cuda.local.array(shape=(3,3), dtype=ctype)
    HFull = cuda.local.array(shape=(3,3), dtype=ctype)
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)
    H_matmass_eigenstate_basis = cuda.local.array(shape=(3,3), dtype=ctype)
    mixNuTypeConjTranspose = cuda.local.array(shape=(3,3), dtype=ctype)

    clear_matrix(H_mat)
    clear_matrix(delta_M_mat_vac)
    clear_matrix(delta_M_mat_mat)
    clear_matrix(HFull)
    clear_matrix(tmp)
    clear_matrix(H_matmass_eigenstate_basis)

    # Compute the matter potential including possible non-standard interactions
    # in the flavor basis 
    get_H_mat(rho, nsi_eps, antitype, H_mat)

    # Get the full Hamiltonian by adding together matter and vacuum parts 
    for i in range(3):
        for j in range(3):
            HFull[i,j] = HVac2energy[i,j] / (2.*energy) + H_mat[i,j]

    # Calculate modified mass eigenvalues in matter from the full Hamiltonian and
    # the vacuum mass splittings 
    get_M(energy, rho, delta_M, delta_M_mat_mat, delta_M_mat_vac, HFull)

    # Now we transform the matter (TODO: matter? full?) Hamiltonian back into the
    # mass eigenstate basis so we don't need to compute products of the effective
    # mixing matrix elements explicitly 
    conjugate_transpose(mixNuType, mixNuTypeConjTranspose)
    matrix_dot_matrix(H_mat, mixNuType, tmp)
    matrix_dot_matrix(mixNuTypeConjTranspose, tmp, H_matmass_eigenstate_basis)

    # We can now proceed to calculating the transition amplitude from the Hamiltonian
    # in the mass basis and the effective mass splittings 
    get_A(baseline, energy, rho, mixNuType, delta_M_mat_vac, delta_M_mat_mat, H_matmass_eigenstate_basis, phase_offset, transition_matrix)

def test_get_transition_matrix():
    antitype = 1
    energy = 1.
    rho = 1.
    baseline = 1.
    phase_offset = 0.
    mixNuType = np.ones(shape=(3,3), dtype=ctype)
    nsi_eps = np.ones(shape=(3,3), dtype=ctype)
    HVac2energy = np.ones(shape=(3,3), dtype=ctype)
    m = np.linspace(0,1,9, dtype=ftype)
    delta_M = m.reshape(3,3)
    transition_matrix = np.ones(shape=(3,3), dtype=ctype)

    get_transition_matrix(antitype, energy, rho, baseline,
                           phase_offset,
                           mixNuType,  nsi_eps,
                           HVac2energy,  delta_M, transition_matrix)
    #print(transition_matrix)

@myjit
def propagate_array_kernel(delta_M, mix, nsi_eps, antitype, flav, energy, n_layers, density_in_layer, distance_in_layer, osc_probs):

    # 3x3 complex
    HVac2energy = cuda.local.array(shape=(3,3), dtype=ctype)
    mixNuType = cuda.local.array(shape=(3,3), dtype=ctype)
    transition_product = cuda.local.array(shape=(3,3), dtype=ctype)
    transition_matrix = cuda.local.array(shape=(3,3), dtype=ctype)
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)

    clear_matrix(HVac2energy)
    clear_matrix(mixNuType)
    clear_matrix(transition_product)
    clear_matrix(transition_matrix)
    clear_matrix(tmp)
    clear_matrix(osc_probs)

    # 3-vector complex
    raw_input_psi = cuda.local.array(shape=(3), dtype=ctype)
    output_psi = cuda.local.array(shape=(3), dtype=ctype)
    for i in range(3):
        raw_input_psi[i] = 0.
        output_psi[i] = 0.


    use_mass_eigenstates = False

    #TODO: * ensure convention below is respected in MC reweighting
    #          (antitype > 0 for nu, < 0 for anti-nu)
    #        * antitype is passed in, so could already pass in the correct form
    #          of mixing matrix, i.e., possibly conjugated
    if antitype > 0:
        # in this case the mixing matrix is left untouched
        copy_matrix(mix, mixNuType)
    
    else:
        # here we need to complex conjugate all entries
        # (note that this only changes calculations with non-clear_matrix deltacp)
        conjugate_transpose(mix, mixNuType)

    get_HVac2energy(mixNuType, delta_M, HVac2energy)


    for i in range(n_layers):
        density = density_in_layer[i]
        distance = distance_in_layer[i]
        get_transition_matrix(antitype,
                               energy,
                               density,
                               distance,
                               0.0,
                               mixNuType,
                               nsi_eps,
                               HVac2energy,
                               delta_M,
                               transition_matrix)
        if i == 0:
            copy_matrix(transition_matrix, transition_product)
        else:
            clear_matrix(tmp)
            matrix_dot_matrix(transition_matrix,transition_product, tmp)
            copy_matrix(tmp, transition_product)
        
    # loop on neutrino types, and compute probability for neutrino i:
    # We actually don't care about nutau -> anything since the flux there is clear_matrix!
    for i in range(2):
        for j in range(3):
            raw_input_psi[j] = 0.
            # clear_matrix out here?
            output_psi[j] = 0.

        if use_mass_eigenstates:
            convert_from_mass_eigenstate(i+1, raw_input_psi, mixNuType)
        else:
            raw_input_psi[i] = 1. + 0.j

        matrix_dot_vector(transition_product, raw_input_psi, output_psi)
        osc_probs[i][0] += output_psi[0].real**2 + output_psi[0].imag**2
        osc_probs[i][1] += output_psi[1].real**2 + output_psi[1].imag**2
        osc_probs[i][2] += output_psi[2].real**2 + output_psi[2].imag**2

def test_propagate_array_kernel():
    mix = np.ones(shape=(3,3), dtype=ctype)
    nsi_eps = np.ones(shape=(3,3), dtype=ctype)
    M = np.linspace(0,1,9, dtype=ftype)
    delta_M = M.reshape(3,3)
    antitype = 1
    flav = 1
    energy = 1.
    n_layers = 10
    density_in_layer = np.ones(shape=(n_layers), dtype=ftype)
    distance_in_layer = np.ones(shape=(n_layers), dtype=ftype)
    osc_probs = np.ones(shape=(3,3), dtype=ftype)

    propagate_array_kernel(delta_M, mix, nsi_eps, antitype, flav, energy, n_layers, density_in_layer, distance_in_layer, osc_probs)
    #print(osc_probs)


if __name__=='__main__':
    
    test_matrix_dot_matrix()
    test_matrix_dot_vector()
    test_clear_matrix()
    test_copy_matrix()
    test_get_HVac2energy()
    print('test successfull')
    test_get_H_mat()
    print('test successfull')
    test_get_M()
    print('test successfull')
    test_getProduct()
    print('test successfull')
    test_get_A()
    print('test successfull')
    test_convert_from_mass_eigenstate()
    print('test successfull')
    test_get_transition_matrix()
    print('test successfull')
    test_propagate_array_kernel()
    print('test porpagate array kernel successfull')
