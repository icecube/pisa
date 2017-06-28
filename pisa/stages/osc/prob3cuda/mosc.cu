#include "mosc.h"
#include "mosc3.h"
#include <stdio.h>

#define elec (0)
#define muon (1)
#define tau  (2)
#define re (0)
#define im (1)

__device__ fType tworttwoGf = 1.52588e-4;
/* (1/2)*(1/(h_bar*c)) in units of GeV/(eV^2-km) */
__device__ fType LoEfac = 2.534;

/* Calculate vacuum Hamiltonian in flavor basis for neutrino or
   antineutrino (need complex conjugate mixing matrix) of energy Enu. */
__device__ void getHVac2Enu(fType Mix[][3][2], fType dmVacVac[][3],
                            fType HVac2Enu[][3][2])
{
  fType dmVacDiag[3][3][2], MixConjTranspose[3][3][2], tmp[3][3][2];
  clear_complex_matrix(dmVacDiag);
  clear_complex_matrix(MixConjTranspose);
  clear_complex_matrix(tmp);
  dmVacDiag[1][1][re] = dmVacVac[1][0];
  dmVacDiag[2][2][re] = dmVacVac[2][0];
  conjugate_transpose_complex_matrix(Mix, MixConjTranspose);
  multiply_complex_matrix(dmVacDiag, MixConjTranspose, tmp);
  multiply_complex_matrix(Mix, tmp, HVac2Enu);
}

/* Calculate effective non-standard interaction Hamiltonian in flavor basis */
__device__ void getHNSI(fType rho, fType NSIEps[][3], int antitype, fType HNSI[][3][2])
{
  fType NSIRhoScale = 3.0;// assume 3x electron density for "NSI"-quark (e.g., d) density
  fType fact = NSIRhoScale*rho*tworttwoGf/2.0;
  if (antitype<0) fact =  -fact; /* Anti-neutrinos */
  else        fact = fact; /* Neutrinos */
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
        HNSI[i][j][re] = fact*NSIEps[i][j];
        HNSI[i][j][im] = 0.0; // only real NSI for now
    }
  }
}

/* From the full matter Hamiltonian in flavor basis, transform to mass eigenstate
   basis using mixing matrix */
__device__ void getHMatMassEigenstateBasis(fType Mix[][3][2], fType HMat[][3][2],
                                           fType HMatMassEigenstateBasis[][3][2])
{
    fType MixConjTranspose[3][3][2], tmp[3][3][2];
    clear_complex_matrix(MixConjTranspose);
    clear_complex_matrix(tmp);
    conjugate_transpose_complex_matrix(Mix, MixConjTranspose);
    multiply_complex_matrix(HMat, Mix, tmp);
    multiply_complex_matrix(MixConjTranspose, tmp, HMatMassEigenstateBasis);
}

/* Calculate full matter Hamiltonian in flavor basis */
__device__ void getHMat(fType rho, fType NSIEps[][3], int antitype,
                        fType HMat[][3][2])
{
  fType HSI[3][3][2], HNSI[3][3][2];

  /* in the following, `a` is just the standard effective matter potential
  induced by charged-current weak interactions with electrons
  (modulo a factor of 2E) */
  fType a = rho*tworttwoGf/2.0;
  clear_complex_matrix(HSI); clear_complex_matrix(HNSI);
  if (antitype<0) a =  -a; /* Anti-neutrinos */
  else        a = a; /* Neutrinos */
  HSI[elec][elec][re] = a;

  // Obtain effective non-standard matter interaction Hamiltonian
  getHNSI(rho, NSIEps, antitype, HNSI);

  // This is where the full matter Hamiltonian is created
   add_complex_matrix(HSI, HNSI, HMat);
}

// add_complex_matrix adapted to vacuum and matter Hamiltonian
__device__ void add_HVac_HMat(fType Enu, fType HVac2Enu[][3][2], fType HMat[][3][2],
                              fType HFull[][3][2])
{
  for (unsigned i=0; i<3; i++) {
    for (unsigned j=0; j<3; j++) {
        HFull[i][j][re] = HVac2Enu[i][j][re]/(2.0*Enu) + HMat[i][j][re];
        HFull[i][j][im] = HVac2Enu[i][j][im]/(2.0*Enu) + HMat[i][j][im];
    }
  }
}

/***********************************************************************
  getM
  Compute the matter-mass vector M, dM = M_i-M_j and dMimj
***********************************************************************/
/* Calculate mass eigenstates in matter of uniform density rho for
   neutrino or anti-neutrino (type already taken into account in Hamiltonian)
   of energy Enu. */
__device__ void getM(fType Enu, fType rho, fType dmVacVac[][3],
                     fType dmMatMat[][3], fType dmMatVac[][3],
                     fType HMat[][3][2])
{
  int i,j,k;
  fType c0, c1, c2, c2V;
  fType p, q, pV, qV;
  fType theta0, theta1, theta2, tmp, M1Sq, M2Sq, M3Sq;
  fType theta0V, theta1V, theta2V, tmpV, M1SqV, M2SqV, M3SqV;

  fType mMatU[3], mMatV[3], mMat[3];
  fType HEEHMuMuHTauTau;
  fType HMuTauModulusSq, HETauModulusSq, HEMuModulusSq, ReHEMuHMuTauHTauE;

#ifndef ZERO_CP

  ReHEMuHMuTauHTauE = HMat[elec][muon][re]*(HMat[muon][tau][re]*HMat[tau][elec][re] -
                                            HMat[muon][tau][im]*HMat[tau][elec][im]) -
    HMat[elec][muon][im]*(HMat[muon][tau][im]*HMat[tau][elec][re] + HMat[muon][tau][re]*HMat[tau][elec][im]);

  HEMuModulusSq = HMat[elec][muon][re]*HMat[elec][muon][re] + HMat[elec][muon][im]*HMat[elec][muon][im];
  HETauModulusSq = HMat[elec][tau][re]*HMat[elec][tau][re] + HMat[elec][tau][im]*HMat[elec][tau][im];
  HMuTauModulusSq = HMat[muon][tau][re]*HMat[muon][tau][re] + HMat[muon][tau][im]*HMat[muon][tau][im];

  HEEHMuMuHTauTau = HMat[elec][elec][re]*(HMat[muon][muon][re]*HMat[tau][tau][re] -
                                            HMat[muon][muon][im]*HMat[tau][tau][im]) -
    HMat[elec][elec][im]*(HMat[muon][muon][im]*HMat[tau][tau][re] + HMat[muon][muon][re]*HMat[tau][tau][im]);


  //c1 = H_{ee}H_{\mu\mu} + H_{ee}H_{\tau\tau} + H_{\mu\mu}H_{\tau\tau} - |H_{e\mu}|^2
  //     - |H_{\mu\tau}|^2 - |H_{e\tau}|^2


  c1 = HMat[elec][elec][re]*(HMat[muon][muon][re] + HMat[tau][tau][re]) -
       HMat[elec][elec][im]*(HMat[muon][muon][im] + HMat[tau][tau][im]) +
       HMat[muon][muon][re]*HMat[tau][tau][re] - HMat[muon][muon][im]*HMat[tau][tau][im] -
       HEMuModulusSq - HMuTauModulusSq - HETauModulusSq;

#else

  ReHEMuHMuTauHTauE = HMat[elec][muon][re]*(HMat[muon][tau][re]*HMat[tau][elec][re]);

  HEMuModulusSq = HMat[elec][muon][re]*HMat[elec][muon][re];
  HETauModulusSq = HMat[elec][tau][re]*HMat[elec][tau][re];
  HMuTauModulusSq = HMat[muon][tau][re]*HMat[muon][tau][re];

  HEEHMuMuHTauTau = HMat[elec][elec][re]*(HMat[muon][muon][re]*HMat[tau][tau][re]);

  c1 = HMat[elec][elec][re]*(HMat[muon][muon][re] + HMat[tau][tau][re]) +
       HMat[muon][muon][re]*HMat[tau][tau][re] -
       HEMuModulusSq - HMuTauModulusSq - HETauModulusSq;

#endif
  //c0 = H_{ee}|H_{\mu\tau}|^2 + H_{\mu\mu}|H_{e\tau}|^2 + H_{\tau\tau}|H_{e\mu}|^2
  //     - 2Re(H_{e\mu}H_{\mu\tau}H_{\tau e}) - H_{ee}H_{\mu\mu}H_{\tau\tau}

  c0 = HMat[elec][elec][re]*HMuTauModulusSq + HMat[muon][muon][re]*HETauModulusSq +
       HMat[tau][tau][re]*HEMuModulusSq - 2.0*ReHEMuHMuTauHTauE - HEEHMuMuHTauTau;

  //c2 = -H_{ee} - H_{\mu\mu} - H_{\tau\tau}

  c2 = -HMat[elec][elec][re] - HMat[muon][muon][re] - HMat[tau][tau][re];

  c2V = (-1.0/(2.0*Enu))*(dmVacVac[1][0] + dmVacVac[2][0]);

  p = c2*c2 - 3.0*c1;
  pV = (1.0/(2.0*Enu*2.0*Enu))*(dmVacVac[1][0]*dmVacVac[1][0] +
                              dmVacVac[2][0]*dmVacVac[2][0] - 
                              dmVacVac[1][0]*dmVacVac[2][0]);
  if (p<0.0) {
      printf("getM: p < 0 ! \n");
      p = 0.0;
  }
  
  q = -27.0*c0/2.0 - c2*c2*c2 + 9.0*c1*c2/2.0;
  qV = (1.0/(2.0*Enu*2.0*Enu*2.0*Enu))*(
        (dmVacVac[1][0] + dmVacVac[2][0])*(dmVacVac[1][0] + dmVacVac[2][0])*
        (dmVacVac[1][0] + dmVacVac[2][0]) - (9.0/2.0)*dmVacVac[1][0]*dmVacVac[2][0]*
        (dmVacVac[1][0] + dmVacVac[2][0]));

  tmp = p*p*p - q*q;
  tmpV = pV*pV*pV - qV*qV;
  if (tmp<0.0) {
    printf("getM: p^3 - q^2 < 0 !\n");
    tmp = 0.0;
  }

  theta0 = theta1 = theta2 = atan2(sqrt(tmp), q)/3.0;
  theta0V = theta1V = theta2V = atan2(sqrt(tmpV), qV)/3.0;
  theta0 += (2.0/3.0)*M_PI;
  theta0V += (2.0/3.0)*M_PI;
  theta1 -= (2.0/3.0)*M_PI;
  theta1V -= (2.0/3.0)*M_PI;

  M1Sq = 2.0*Enu*((2.0/3.0)*sqrt(p)*cos(theta0) - c2/3.0 + dmVacVac[0][0]);
  M2Sq = 2.0*Enu*((2.0/3.0)*sqrt(p)*cos(theta1) - c2/3.0 + dmVacVac[0][0]);
  M3Sq = 2.0*Enu*((2.0/3.0)*sqrt(p)*cos(theta2) - c2/3.0 + dmVacVac[0][0]);
  M1SqV = 2.0*Enu*((2.0/3.0)*sqrt(pV)*cos(theta0V) - c2V/3.0 + dmVacVac[0][0]);
  M2SqV = 2.0*Enu*((2.0/3.0)*sqrt(pV)*cos(theta1V) - c2V/3.0 + dmVacVac[0][0]);
  M3SqV = 2.0*Enu*((2.0/3.0)*sqrt(pV)*cos(theta2V) - c2V/3.0 + dmVacVac[0][0]);

  mMatU[0] = M1Sq;
  mMatU[1] = M2Sq;
  mMatU[2] = M3Sq;
  mMatV[0] = M1SqV;
  mMatV[1] = M2SqV;
  mMatV[2] = M3SqV;

  /* Sort according to which reproduce the vaccum eigenstates */
  for (i=0; i<3; i++) {
    tmpV = fabs(dmVacVac[i][0]-mMatV[0]);
    k = 0;
    for (j=1; j<3; j++) {
      tmp = fabs(dmVacVac[i][0]-mMatV[j]);
      if (tmp<tmpV) {
        k = j;
        tmpV = tmp;
      }
    }
    mMat[i] = mMatU[k];
  }
  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      dmMatMat[i][j] = mMat[i] - mMat[j];
      dmMatVac[i][j] = mMat[i] - dmVacVac[j][0];
    }
  }
}


/***********************************************************************
 getAGen (take into account generic potential matrix (=Hamiltonian))
 Calculate the transition amplitude matrix A (equation 10)
***********************************************************************/
__device__ void getAGen(fType L, fType E, fType rho,
                        fType Mix[][3][2], fType dmMatVac[][3],
                        fType dmMatMat[][3], fType HMatMassEigenstateBasis[][3][2],
                        fType A[3][3][2],
                        fType phase_offset)
{

  fType arg, c, s;
  fType X[3][3][2] = {0.0};
  fType product[3][3][3][2] = {0.0};

  if ( phase_offset==0.0 )
    {
      get_productGen(L, E, rho, dmMatVac, dmMatMat, HMatMassEigenstateBasis,
                     product);
    }

  for (int k=0; k<3; k++)
    {
      arg = -LoEfac*dmMatVac[k][0]*L/E;
      if ( k==2 ) arg += phase_offset ;
      c = cos(arg);
      s = sin(arg);
      for (int i=0; i<3; i++)
        {
          for (int j=0; j<3; j++)
            {
#ifndef ZERO_CP
              X[i][j][re] += c*product[i][j][k][re] - s*product[i][j][k][im];
              X[i][j][im] += c*product[i][j][k][im] + s*product[i][j][k][re];
#else
              X[i][j][re] += c*product[i][j][k][re];
              X[i][j][im] += s*product[i][j][k][re];
#endif
            }
        }
    }


  /* Compute the product with the mixing matrices */
  for(int i=0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 2; k++)
        A[i][j][k] = 0;

  for (int n=0; n<3; n++) {
    for (int m=0; m<3; m++) {
      for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
#ifndef ZERO_CP
          A[n][m][re] +=
            Mix[n][i][re]*X[i][j][re]*Mix[m][j][re] +
            Mix[n][i][re]*X[i][j][im]*Mix[m][j][im] +
            Mix[n][i][im]*X[i][j][re]*Mix[m][j][im] -
            Mix[n][i][im]*X[i][j][im]*Mix[m][j][re];
          A[n][m][im] +=
            Mix[n][i][im]*X[i][j][im]*Mix[m][j][im] +
            Mix[n][i][im]*X[i][j][re]*Mix[m][j][re] +
            Mix[n][i][re]*X[i][j][im]*Mix[m][j][re] -
            Mix[n][i][re]*X[i][j][re]*Mix[m][j][im];
#else
          A[n][m][re] +=
            Mix[n][i][re]*X[i][j][re]*Mix[m][j][re];
          A[n][m][im] +=
            Mix[n][i][re]*X[i][j][im]*Mix[m][j][re];
#endif
        }
      }
    }
  }

}


__device__ void get_productGen(fType L, fType E, fType rho,
                               fType dmMatVac[][3], fType dmMatMat[][3],
                               fType HMatMassEigenstateBasis[][3][2],
                               fType product[][3][3][2])
{

  fType twoEHmM[3][3][3][2];

  /* Calculate the matrix 2EH-M_j */
  for (int n=0; n<3; n++) {
    for (int m=0; m<3; m++) {

      twoEHmM[n][m][0][re] = 2.0*E*HMatMassEigenstateBasis[n][m][re];
#ifndef ZERO_CP

      twoEHmM[n][m][0][im] = 2.0*E*HMatMassEigenstateBasis[n][m][im];

#else

      twoEHmM[n][m][0][im] = 0.0 ;

#endif

      twoEHmM[n][m][1][re] = twoEHmM[n][m][2][re] = twoEHmM[n][m][0][re];
      twoEHmM[n][m][1][im] = twoEHmM[n][m][2][im] = twoEHmM[n][m][0][im];

      if (n==m) for (int j=0; j<3; j++)
                  twoEHmM[n][m][j][re] -= dmMatVac[j][n];
    }
  }

  /* Calculate the product in eq.(10) of twoEHmM for j!=k */
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      for (int k=0; k<3; k++) {

#ifndef ZERO_CP

        product[i][j][0][re] +=
          twoEHmM[i][k][1][re]*twoEHmM[k][j][2][re] -
          twoEHmM[i][k][1][im]*twoEHmM[k][j][2][im];
        product[i][j][0][im] +=
          twoEHmM[i][k][1][re]*twoEHmM[k][j][2][im] +
          twoEHmM[i][k][1][im]*twoEHmM[k][j][2][re];
        product[i][j][1][re] +=
          twoEHmM[i][k][2][re]*twoEHmM[k][j][0][re] -
          twoEHmM[i][k][2][im]*twoEHmM[k][j][0][im];
        product[i][j][1][im] +=
          twoEHmM[i][k][2][re]*twoEHmM[k][j][0][im] +
          twoEHmM[i][k][2][im]*twoEHmM[k][j][0][re];
        product[i][j][2][re] +=
          twoEHmM[i][k][0][re]*twoEHmM[k][j][1][re] -
          twoEHmM[i][k][0][im]*twoEHmM[k][j][1][im];
        product[i][j][2][im] +=
          twoEHmM[i][k][0][re]*twoEHmM[k][j][1][im] +
          twoEHmM[i][k][0][im]*twoEHmM[k][j][1][re];

#else
        product[i][j][0][re] +=
          twoEHmM[i][k][1][re]*twoEHmM[k][j][2][re];
        product[i][j][1][re] +=
          twoEHmM[i][k][2][re]*twoEHmM[k][j][0][re];
        product[i][j][2][re] +=
          twoEHmM[i][k][0][re]*twoEHmM[k][j][1][re];

#endif
      }
#ifndef ZERO_CP

      product[i][j][0][re] /= (dmMatMat[0][1]*dmMatMat[0][2]);
      product[i][j][0][im] /= (dmMatMat[0][1]*dmMatMat[0][2]);
      product[i][j][1][re] /= (dmMatMat[1][2]*dmMatMat[1][0]);
      product[i][j][1][im] /= (dmMatMat[1][2]*dmMatMat[1][0]);
      product[i][j][2][re] /= (dmMatMat[2][0]*dmMatMat[2][1]);
      product[i][j][2][im] /= (dmMatMat[2][0]*dmMatMat[2][1]);

#else
      product[i][j][0][re] /= (dmMatMat[0][1]*dmMatMat[0][2]);
      product[i][j][1][re] /= (dmMatMat[1][2]*dmMatMat[1][0]);
      product[i][j][2][re] /= (dmMatMat[2][0]*dmMatMat[2][1]);

#endif
    }
  }
}
