from __future__ import print_function
import numpy as np
from numba import jit

elec=0
muon=1
tau=2

tworttwoGf = 1.52588e-4
LoEfac = 2.534

nopython=True
cache=True

@jit(nopython=nopython, cache=cache)
def getHVac2Enu( Mix,  dmVacVac):
    '''
    Calculate vacuum Hamiltonian in flavor basis for neutrino or 
    antineutrino (need complex conjugate mixing matrix) of energy Enu.
    '''
    dmVacDiag = np.zeros((3,3)) + np.zeros((3,3)) * 1j
    dmVacDiag[1,1] = dmVacVac[1,0]
    dmVacDiag[2,2] = dmVacVac[2,0]
    return np.dot(np.dot(Mix,dmVacDiag),np.conjugate(Mix).T)

@jit(nopython=nopython, cache=cache)
def getHNSI(rho, NSIEps, antitype):
    '''
    Calculate effective non-standard interaction Hamiltonian in flavor basis 
    '''
    NSIRhoScale = 3. #// assume 3x electron density for "NSI"-quark (e.g., d) density
    fact = NSIRhoScale * rho * tworttwoGf * antitype / 2.
    HNSI = fact * NSIEps
    # only real NSI for now
    HNSI = HNSI.real + np.zeros((3,3)) * 1j
    return HNSI

@jit(nopython=nopython, cache=cache)
def getHMat(rho, NSIEps, antitype):
    '''
    Calculate full matter Hamiltonian in flavor basis 

    in the following, `a` is just the standard effective matter potential
    induced by charged-current weak interactions with electrons
    (modulo a factor of 2E)
    '''
    a = rho * tworttwoGf * antitype / 2.

    HSI = np.zeros((3,3)) + np.zeros((3,3)) * 1j
    HSI[elec,elec] = a + 0j

    # Obtain effective non-standard matter interaction Hamiltonian
    HNSI = getHNSI(rho, NSIEps, antitype)

    # This is where the full matter Hamiltonian is created
    return HSI + HNSI

@jit(nopython=nopython, cache=cache)
def getM(Enu, rho, dmVacVac, dmMatMat, dmMatVac, HMat):
    '''
    Compute the matter-mass vector M, dM = M_i-M_j and dMimj

    Calculate mass eigenstates in matter of uniform density rho for
    neutrino or anti-neutrino (type already taken into account in Hamiltonian)
    of energy Enu. 
    '''

    ReHEMuHMuTauHTauE = (HMat[elec,muon]*HMat[muon,tau]*HMat[tau,elec]).real

    HEMuModulusSq = HMat[elec,muon].real**2 + HMat[elec,muon].imag**2
    HETauModulusSq = HMat[elec,tau].real**2 + HMat[elec,tau].imag**2
    HMuTauModulusSq = HMat[muon,tau].real**2 + HMat[muon,tau].imag**2

    HEEHMuMuHTauTau = (HMat[elec,elec]*HMat[muon,muon]*HMat[tau,tau]).real

    c1 = (HMat[elec,elec].real * (HMat[muon,muon] + HMat[tau,tau])).real \
         -(HMat[elec,elec].imag * (HMat[muon,muon] + HMat[tau,tau])).imag \
         + (HMat[muon,muon].real * HMat[tau,tau]).real \
         - HEMuModulusSq \
         - HMuTauModulusSq \
         - HETauModulusSq

    c0 = HMat[elec,elec].real * HMuTauModulusSq \
         + HMat[muon,muon].real * HETauModulusSq \
         + HMat[tau,tau].real * HEMuModulusSq \
         - 2.0 * ReHEMuHMuTauHTauE \
         - HEEHMuMuHTauTau

    c2 = - np.trace(HMat.real)

    c2V = (-1.0/(2.0*Enu))*(dmVacVac[1,0] + dmVacVac[2,0])

    p = c2*c2 - 3.0*c1
    pV = (1.0/(2.0*Enu*2.0*Enu))*(dmVacVac[1,0]*dmVacVac[1,0] +
                              dmVacVac[2,0]*dmVacVac[2,0] - 
                              dmVacVac[1,0]*dmVacVac[2,0])
    p = max(0., p)

    q = -27.0*c0/2.0 - c2*c2*c2 + 9.0*c1*c2/2.0
    qV = (1.0/(2.0*Enu*2.0*Enu*2.0*Enu))*(
        (dmVacVac[1,0] + dmVacVac[2,0])*(dmVacVac[1,0] + dmVacVac[2,0])*
        (dmVacVac[1,0] + dmVacVac[2,0]) - (9.0/2.0)*dmVacVac[1,0]*dmVacVac[2,0]*
        (dmVacVac[1,0] + dmVacVac[2,0]))

    tmp = p*p*p - q*q
    tmpV = pV*pV*pV - qV*qV

    tmp = max(0., tmp)

    res = np.arctan2(np.sqrt(tmp), q) / 3.
    theta = np.array([res, res, res])
    resV = np.arctan2(np.sqrt(tmpV), qV) / 3.
    thetaV = np.array([resV, resV, resV])
    a = (2./3.)*np.pi
    theta[0] += a
    thetaV[0] += a
    theta[1] -= a
    thetaV[1] -= a

    mMatU = 2.0*Enu*((2.0/3.0)*np.sqrt(p)*np.cos(theta) - c2/3.0 + dmVacVac[0,0])
    mMatV = 2.0*Enu*((2.0/3.0)*np.sqrt(pV)*np.cos(thetaV) - c2V/3.0 + dmVacVac[0,0])
    mMat = np.zeros((3))

    # Sort according to which reproduce the vaccum eigenstates 
    for i in range(3):
        tmpV = np.fabs(dmVacVac[i,0]-mMatV[0])
        k = 0
        for j in range(3):
            tmp = np.fabs(dmVacVac[i,0]-mMatV[j])
            if (tmp<tmpV):
                k = j
                tmpV = tmp
        mMat[i] = mMatU[k]
    for i in range(3):
        for j in range(3):
              dmMatMat[i,j] = mMat[i] - mMat[j]
              dmMatVac[i,j] = mMat[i] - dmVacVac[j,0]

@jit(nopython=nopython, cache=cache)
def getA(L, E, rho, Mix,  dmMatVac, dmMatMat, HMatMassEigenstateBasis, phase_offset):
    '''
    getA (take into account generic potential matrix (=Hamiltonian))
    Calculate the transition amplitude matrix A (equation 10)
    '''

    X = np.zeros((3,3)) + np.zeros((3,3)) * 1j
    product = np.zeros((3,3,3)) + np.zeros((3,3,3)) * 1j

    if (phase_offset==0.0):
        get_product(L, E, rho, dmMatVac, dmMatMat, HMatMassEigenstateBasis, product)

    for k in range(3):
        arg = -LoEfac*dmMatVac[k,0]*L/E
        if ( k==2 ):
            arg += phase_offset 
        X += (np.cos(arg) - 1j*np.sin(arg)) * product[:,:,k]

    # Compute the product with the mixing matrices 
    # is this correct?
    return np.dot(np.dot(Mix,X),np.conjugate(Mix).T)

@jit(nopython=nopython, cache=cache)
def get_product(L, E, rho, dmMatVac, dmMatMat, HMatMassEigenstateBasis, product):
    twoEHmM = np.zeros((3,3,3)) + np.zeros((3,3,3)) * 1j
    twoEHmM[:,:,0] = 2. * E * HMatMassEigenstateBasis
    twoEHmM[:,:,1] = twoEHmM[:,:,0]
    twoEHmM[:,:,2] = twoEHmM[:,:,0]

    for n in range(3):
        for j in range(3):
            twoEHmM[n,n,j] -= dmMatVac[j,n]

    # Calculate the product in eq.(10) of twoEHmM for j!=k 
    product[:,:,0] = (np.dot(twoEHmM[:,:,1],twoEHmM[:,:,2])) / (dmMatMat[0,1] * dmMatMat[0,2])
    product[:,:,1] = (np.dot(twoEHmM[:,:,2],twoEHmM[:,:,0])) / (dmMatMat[1,2] * dmMatMat[1,0])
    product[:,:,2] = (np.dot(twoEHmM[:,:,0],twoEHmM[:,:,1])) / (dmMatMat[2,0] * dmMatMat[2,1])

@jit(nopython=nopython, cache=cache)
def convert_from_mass_eigenstate(state, pure, mixNuType):
    mass = np.zeros((3)) + np.zeros((3)) * 1j
    lstate  = state - 1

    for i in range(3): 
        mass[i] = 1. if lstate == i else 0.
    # note: mixNuType is already taking into account whether we're considering
    # nu or anti-nu
    pure = np.dot(mixNuType, mass)

@jit(nopython=nopython, cache=cache)
def get_transition_matrix(nutype, Enu, rho, Len,
                           phase_offset,
                           mixNuType,  nsi_eps,
                           HVac2Enu,  dm):
    '''
    Calculate neutrino flavour transition amplitude matrix for neutrino (nutype > 0)
    or antineutrino (nutype < 0) with energy Enu travernp.sing layer of matter of
    uniform density rho with thickness Len.
    '''
    dmMatVac = np.zeros((3,3))
    dmMatMat = np.zeros((3,3))

    # Compute the matter potential including possible non-standard interactions
    # in the flavor basis 
    HMat = getHMat(rho, nsi_eps, nutype)

    # Get the full Hamiltonian by adding together matter and vacuum parts 
    HFull = HVac2Enu / (2. * Enu) + HMat

    # Calculate modified mass eigenvalues in matter from the full Hamiltonian and
    # the vacuum mass splittings 
    getM(Enu, rho, dm, dmMatMat, dmMatVac, HFull)

    # Now we transform the matter (TODO: matter? full?) Hamiltonian back into the
    # mass eigenstate basis so we don't need to compute products of the effective
    # mixing matrix elements explicitly 
    HMatMassEigenstateBasis = np.dot(np.dot(np.conjugate(mixNuType).T,HMat),mixNuType)

    # We can now proceed to calculating the transition amplitude from the Hamiltonian
    # in the mass basis and the effective mass splittings 
    return getA(Len, Enu, rho, mixNuType, dmMatVac, dmMatMat, HMatMassEigenstateBasis, phase_offset)
