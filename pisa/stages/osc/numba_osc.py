from __future__ import print_function
import numpy as np
from numba_tools import *
from numba import jit, guvectorize, float64, complex64, int32, float32, complex128, complex64
import time
import math, cmath

ctype=complex128
ftype=float64

@myjit
def getHVac2Enu(Mix, dmVacVac, HVac2Enu):
    '''
    Calculate vacuum Hamiltonian in flavor basis for neutrino or 
    antineutrino (need complex conjugate mixing matrix) of energy Enu.
    '''
    dmVacDiag = cuda.local.array(shape=(3,3), dtype=ctype)
    zero(dmVacDiag)
    dmVacDiag[1,1] = dmVacVac[1,0] + 0j
    dmVacDiag[2,2] = dmVacVac[2,0] + 0j
    MixConjTranspose = cuda.local.array(shape=(3,3), dtype=ctype)
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)
    conjugate_transpose(Mix, MixConjTranspose)
    MdotM(dmVacDiag, MixConjTranspose, tmp)
    MdotM(Mix, tmp, HVac2Enu)

@myjit
def getHMat(rho, NSIEps, antitype, HMat):
    '''
    Calculate full matter Hamiltonian in flavor basis 

    in the following, `a` is just the standard effective matter potential
    induced by charged-current weak interactions with electrons
    (modulo a factor of 2E)
    Calculate effective non-standard interaction Hamiltonian in flavor basis 
    '''
    tworttwoGf = 1.52588e-4

    # standard matter interaction Hamiltonian
    a = rho * tworttwoGf * antitype / 2.

    HMat[0,0] = a + 0j

    # Obtain effective non-standard matter interaction Hamiltonian
    NSIRhoScale = 3. #// assume 3x electron density for "NSI"-quark (e.g., d) density
    fact = NSIRhoScale * a
    for i in range(HMat.shape[0]):
        for j in range(HMat.shape[1]):
            HMat[i,j] += fact * NSIEps[i,j]

@myjit
def getM(Enu, rho, dmVacVac, dmMatMat, dmMatVac, HMat):
    '''
    Compute the matter-mass vector M, dM = M_i-M_j and dMimj

    Calculate mass eigenstates in matter of uniform density rho for
    neutrino or anti-neutrino (type already taken into account in Hamiltonian)
    of energy Enu. 
    '''

    ReHEMuHMuTauHTauE = (HMat[0,1]*HMat[1,2]*HMat[2,0]).real
    ReHEEHMuMuHTauTau = (HMat[0,0]*HMat[1,1]*HMat[2,2]).real

    HEMuModulusSq =   HMat[0,1].real**2 + HMat[0,1].imag**2
    HETauModulusSq =  HMat[0,2].real**2 + HMat[0,2].imag**2
    HMuTauModulusSq = HMat[1,2].real**2 + HMat[1,2].imag**2

    c1 =   (HMat[0,0].real * (HMat[1,1] + HMat[2,2])).real \
         - (HMat[0,0].imag * (HMat[1,1] + HMat[2,2])).imag \
         + (HMat[1,1].real * HMat[2,2]).real \
         - (HMat[1,1].imag * HMat[2,2]).imag \
         - HEMuModulusSq \
         - HMuTauModulusSq \
         - HETauModulusSq

    c0 =   HMat[0,0].real * HMuTauModulusSq \
         + HMat[1,1].real * HETauModulusSq \
         + HMat[2,2].real   * HEMuModulusSq \
         - 2. * ReHEMuHMuTauHTauE \
         - ReHEEHMuMuHTauTau

    c2 = - HMat[0,0].real - HMat[1,1].real - HMat[2,2].real

    twoE = 2. * Enu
    twoESq = twoE * twoE
    twoECu = twoESq * twoE

    c2V = (-1. / twoE) * (dmVacVac[1,0] + dmVacVac[2,0])

    p = c2 * c2 - 3. * c1
    pV = (1. / twoESq) * (  dmVacVac[1,0] * dmVacVac[1,0]
                          + dmVacVac[2,0] * dmVacVac[2,0]
                          - dmVacVac[1,0] * dmVacVac[2,0])
    p = max(0., p)

    q = -27. * c0 / 2.0 - c2 * c2 * c2 + 9. * c1 * c2 / 2.
    qV = (1. / twoECu) * ((dmVacVac[1,0] + dmVacVac[2,0]) * 
                          (dmVacVac[1,0] + dmVacVac[2,0]) *
                          (dmVacVac[1,0] + dmVacVac[2,0]) -
                          (9. / 2.) * dmVacVac[1,0] * dmVacVac[2,0] *
                          (dmVacVac[1,0] + dmVacVac[2,0]))

    tmp = p * p * p - q * q
    tmpV = pV * pV * pV - qV * qV

    tmp = max(0., tmp)

    theta = cuda.local.array(shape=(3), dtype=ftype)
    thetaV = cuda.local.array(shape=(3), dtype=ftype)
    mMat = cuda.local.array(shape=(3), dtype=ftype)
    mMatU = cuda.local.array(shape=(3), dtype=ftype)
    mMatV = cuda.local.array(shape=(3), dtype=ftype)

    a = (2. / 3.) * math.pi
    res = math.atan2(math.sqrt(tmp), q) / 3.
    theta[0] = res + a
    theta[1] = res - a
    theta[2] = res
    resV = math.atan2(math.sqrt(tmpV), qV) / 3.
    thetaV[0] = resV + a
    thetaV[1] = resV - a
    thetaV[2] = resV
    
    for i in range(theta.shape[0]):
        mMatU[i] = 2. * Enu * ((2. / 3.) * math.sqrt(p) *  math.cos(theta[i])  - c2 / 3.  + dmVacVac[0,0])
        mMatV[i] = 2. * Enu * ((2. / 3.) * math.sqrt(pV) * math.cos(thetaV[i]) - c2V / 3. + dmVacVac[0,0])

    # Sort according to which reproduce the vaccum eigenstates 
    for i in range(3):
        tmpV = abs(dmVacVac[i,0]-mMatV[0])
        k = 0
        for j in range(3):
            tmp = abs(dmVacVac[i,0]-mMatV[j])
            if tmp < tmpV:
                k = j
                tmpV = tmp
        mMat[i] = mMatU[k]
    for i in range(3):
        for j in range(3):
              dmMatMat[i,j] = mMat[i] - mMat[j]
              dmMatVac[i,j] = mMat[i] - dmVacVac[j,0]

@myjit
def getA(L, E, rho, Mix,  dmMatVac, dmMatMat, HMatMassEigenstateBasis, phase_offset, TransitionMatrix):
    '''
    getA (take into account generic potential matrix (=Hamiltonian))
    Calculate the transition amplitude matrix A (equation 10)
    '''
    LoEfac = 2.534

    X = cuda.local.array(shape=(3,3), dtype=ctype)
    zero(X)
    product = cuda.local.array(shape=(3,3,3), dtype=ctype)

    if phase_offset == 0.0:
        get_product(L, E, rho, dmMatVac, dmMatMat, HMatMassEigenstateBasis, product)
        # what if phase_offset != 0.0??

    for k in range(3):
        arg = - LoEfac * dmMatVac[k,0] * L / E
        if k == 2:
            arg += phase_offset 
        for i in range(3):
            for j in range(3):
                X[i,j] += cmath.exp(arg * 1j) * product[i,j,k]

    # Compute the product with the mixing matrices 
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)
    MixConjTranspose = cuda.local.array(shape=(3,3), dtype=ctype)
    conjugate_transpose(Mix, MixConjTranspose)
    MdotM(X, MixConjTranspose, tmp)
    MdotM(Mix, tmp, TransitionMatrix)

@myjit
def get_product(L, E, rho, dmMatVac, dmMatMat, HMatMassEigenstateBasis, product):
    twoEHmM = cuda.local.array(shape=(3,3,3), dtype=ctype)
    for i in range(product.shape[0]):
        for j in range(product.shape[1]):
            for k in range(product.shape[2]):
                twoEHmM[i,j,k] = 2. * E * HMatMassEigenstateBasis[i,j]  

    for n in range(3):
        for j in range(3):
            twoEHmM[n,n,j] -= dmMatVac[j,n]

    # Calculate the product in eq.(10) of twoEHmM for j!=k 
    for i in range(product.shape[0]):
        for j in range(product.shape[1]):
            for k in range(product.shape[2]):
                product[i,j,0] = twoEHmM[i,k,1] * twoEHmM[k,j,2]
                product[i,j,1] = twoEHmM[i,k,2] * twoEHmM[k,j,0]
                product[i,j,2] = twoEHmM[i,k,0] * twoEHmM[k,j,1]
            product[i,j,0] /= (dmMatMat[0,1] * dmMatMat[0,2])
            product[i,j,1] /= (dmMatMat[1,2] * dmMatMat[1,0])
            product[i,j,2] /= (dmMatMat[2,0] * dmMatMat[2,1])

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
    Mdotv(mixNuType, mass, pure)

@myjit
def get_transition_matrix(nutype, Enu, rho, Len,
                           phase_offset,
                           mixNuType,  nsi_eps,
                           HVac2Enu,  dm, TransitionMatrix):
    '''
    Calculate neutrino flavour transition amplitude matrix for neutrino (nutype > 0)
    or antineutrino (nutype < 0) with energy Enu travernp.sing layer of matter of
    uniform density rho with thickness Len.
    '''
    HMat = cuda.local.array(shape=(3,3), dtype=ctype)
    dmMatVac = cuda.local.array(shape=(3,3), dtype=ctype)
    dmMatMat = cuda.local.array(shape=(3,3), dtype=ctype)

    # Compute the matter potential including possible non-standard interactions
    # in the flavor basis 
    getHMat(rho, nsi_eps, nutype, HMat)

    # Get the full Hamiltonian by adding together matter and vacuum parts 
    HFull = cuda.local.array(shape=(3,3), dtype=ctype)
    for i in range(HMat.shape[0]):
        for j in range(HMat.shape[1]):
            HFull[i,j] = HVac2Enu[i,j] / (2. * Enu) + HMat[i,j]

    # Calculate modified mass eigenvalues in matter from the full Hamiltonian and
    # the vacuum mass splittings 
    getM(Enu, rho, dm, dmMatMat, dmMatVac, HFull)

    # Now we transform the matter (TODO: matter? full?) Hamiltonian back into the
    # mass eigenstate basis so we don't need to compute products of the effective
    # mixing matrix elements explicitly 
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)
    zero(tmp)
    HMatMassEigenstateBasis = cuda.local.array(shape=(3,3), dtype=ctype)
    zero(HMatMassEigenstateBasis)
    mixNuTypeConjTranspose = cuda.local.array(shape=(3,3), dtype=ctype)
    conjugate_transpose(mixNuType, mixNuTypeConjTranspose)
    MdotM(HMat, mixNuType, tmp)
    MdotM(mixNuTypeConjTranspose, tmp, HMatMassEigenstateBasis)

    # We can now proceed to calculating the transition amplitude from the Hamiltonian
    # in the mass basis and the effective mass splittings 
    getA(Len, Enu, rho, mixNuType, dmMatVac, dmMatMat, HMatMassEigenstateBasis, phase_offset, TransitionMatrix)

@myjit
def propagateArray_kernel(dm,
                   mix,
                   nsi_eps,
                   kNuBar,
                   kFlav,
                   energy,
                   numberOfLayers,
                   densityInLayer,
                   distanceInLayer,
                   Probability):

    # 3x3 complex
    HVac2Enu = cuda.local.array(shape=(3,3), dtype=ctype)
    zero(HVac2Enu)
    mixNuType = cuda.local.array(shape=(3,3), dtype=ctype)
    zero(mixNuType)
    TransitionProduct = cuda.local.array(shape=(3,3), dtype=ctype)
    zero(TransitionProduct)
    TransitionMatrix = cuda.local.array(shape=(3,3), dtype=ctype)
    zero(TransitionMatrix)
    tmp = cuda.local.array(shape=(3,3), dtype=ctype)
    zero(tmp)

    # 3-vector complex
    RawInputPsi = cuda.local.array(shape=(3), dtype=ctype)
    OutputPsi = cuda.local.array(shape=(3), dtype=ctype)

    kUseMassEstates = False

    #TODO: * ensure convention below is respected in MC reweighting
    #          (kNuBar > 0 for nu, < 0 for anti-nu)
    #        * kNuBar is passed in, so could already pass in the correct form
    #          of mixing matrix, i.e., possibly conjugated
    if (kNuBar > 0):
        # in this case the mixing matrix is left untouched
        copy(mix, mixNuType)
    
    else:
        # here we need to complex conjugate all entries
        # (note that this only changes calculations with non-zero deltacp)
        conjugate_transpose(mix, mixNuType)

    getHVac2Enu(mixNuType, dm, HVac2Enu)


    for i in range(numberOfLayers):
        density = densityInLayer[i]
        distance = distanceInLayer[i]
        get_transition_matrix(kNuBar,
                               energy,
                               density,
                               distance,
                               0.0,
                               mixNuType,
                               nsi_eps,
                               HVac2Enu,
                               dm,
                               TransitionMatrix)
        if (i==0):
            copy(TransitionMatrix, TransitionProduct)
        else:
            MdotM(TransitionMatrix,TransitionProduct, tmp)
            copy(tmp, TransitionProduct)
        
    # loop on neutrino types, and compute probability for neutrino i:
    # We actually don't care about nutau -> anything since the flux there is zero!
    for i in range(2):
        for j in range(3):
            RawInputPsi[j] = 0. + 0.j
            # zero out here?
            OutputPsi[j] = 0. +0.j

        if( kUseMassEstates ):
            convert_from_mass_eigenstate(i+1, RawInputPsi, mixNuType)
        else:
            RawInputPsi[i] = 1. + 0.j

        Mdotv(TransitionProduct, RawInputPsi, OutputPsi)
        Probability[i][0] += OutputPsi[0].real**2 + OutputPsi[0].imag**2
        Probability[i][1] += OutputPsi[1].real**2 + OutputPsi[1].imag**2
        Probability[i][2] += OutputPsi[2].real**2 + OutputPsi[2].imag**2

@guvectorize([(float64[:,:], complex128[:,:], complex128[:,:], int32, int32, float64, int32, float64[:], float64[:], float64[:,:])], '(a,b),(c,d),(e,f),(),(),(),(),(g),(h)->(a,b)', target=target)
def propagateArray(dm, mix, nsi_eps, kNuBar, kFlav, energy, numberOfLayers, densityInLayer, distanceInLayer, Probability):
    propagateArray_kernel(dm, mix, nsi_eps, kNuBar, kFlav, energy, numberOfLayers, densityInLayer, distanceInLayer, Probability)
