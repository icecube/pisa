#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mosc.h"
//#include "mosc3.h"

#define re (0)
#define im (1)

static double dm[3][3];
static double mix[3][3][2];
static double Ain[3][3][2];
static double dm21,dm32,s12,s23,s31,dcp;


/*
 *     Enu  : neutrino energy (GeV)
 *     rho  : electron density in matter (ne/Na/(g/cm^3))
 *     dm21 : m2^2 - m1^2 (eV2)
 *     dm31 : m3^2 - m1^2 (eV2)
 *     s12  : sin(th12)   
 *     s23  : sin(th23)   
 *     s31  : sin(th31)   
 *     dcp  : delta cp
 *     cd   : cos(delta)  
 *     sd   : sin(delta)  
 */


/*
 *   Initialize mixing matrix, etc.
 */
void init_mixing_matrix(dm21f,dm32f,s12f,s23f,s31f,dcpf)
     double dm21f,dm32f,s12f,s23f,s31f,dcpf;
{
  dm21=dm21f ;  dm32=dm32f ;  
  s12=s12f   ;  s23=s23f   ; s31=s31f ;
  dcp=dcpf     ;
  setMatterFlavor(nue_type);
  setmix_sin(s12,s23,s31,dcp,mix);
  setmass(dm21,dm32,dm);
  memset(Ain,0,3*3*2*sizeof(double));
  Ain[0][0][re] = Ain[1][1][re] = Ain[2][2][re] = 1.0;

  /***********
  printf("dm21,dm32   : %f %f \n",dm21,dm32);
  printf("s12,s23,s31 : %f %f %f \n",s12,s23,s31);
  printf("dm  : %f %f %f \n",dm[0][0],dm[0][1],dm[0][2]);
  printf("dm  : %f %f %f \n",dm[1][0],dm[1][1],dm[1][2]);
  printf("dm  : %f %f %f \n",dm[2][0],dm[2][1],dm[2][2]);
  ***********/
  /***********
  printf("mix : %f %f %f \n",mix[0][0][0],mix[0][1][0],mix[0][2][0]);
  printf("mix : %f %f %f \n",mix[1][0][0],mix[1][1][0],mix[1][2][0]);
  printf("mix : %f %f %f \n",mix[2][0][0],mix[2][1][0],mix[2][2][0]);
  ***********/


}



void get_oscillation_parameters(dm21f,dm32f,s12f,s23f,s31f,dcpf)
     double dm21f,dm32f,s12f,s23f,s31f,dcpf;
{
  dm21f = dm21;
  dm32f = dm32;
  s12f = s12;
  s23f = s23;
  s31f = s31;
  dcpf = dcp ;
}


/*
 *   return oscillation length of delta M^23 sector
 */
void get_wavelength_23(energy, lambda23)
     double energy; double* lambda23;
{
  *lambda23 = 2.480*(energy)/fabs(dm32);
}


/*
 *   Return real part of mixing matrix
 *   (for calculating oscillation probability in vacuum)
 */
void get_mixing_matrix_real(mixtmp)
     double mixtmp[3][3];
{
  int i,j;
  for(j=0;j<3;j++) {
    for (i=0;i<3;i++) {
      mixtmp[j][i] = mix[j][i][re];
    }
  }
}


/*
 *   Obtain transition matrix
 */
void get_transition_matrix(nutypei,Enuf,rhof,Lenf,Aout,phase_offsetf)
     int nutypei;
     double Enuf,rhof,Lenf;
     double Aout[][3][2];
     double phase_offsetf ;
{
  int nutype;
  double Enu, rho, Len ;
  double dmMatVac[3][3], dmMatMat[3][3];
  double phase_offset;
  nutype=nutypei;
  Enu=Enuf ;
  rho=rhof ;
  Len=Lenf ;
  phase_offset = phase_offsetf ;
  /*   propagate_mat(Ain,rho,Len,Enu,mix,dm,nutype,Aout);    */
  getM(Enu, rho, mix, dm, nutype, dmMatMat, dmMatVac);
  getA(Len, Enu, rho, mix, dmMatVac, dmMatMat, nutype, Aout,phase_offset);
}



/*
 *   multiply complex 3x3 matrix 
 *        C = A X B
 */
void multiply_complex_matrix(A,B,C)
     double A[][3][2];
     double B[][3][2];
     double C[][3][2];
{
  int i,j,k;
  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      for (k=0; k<3; k++) {
        C[i][j][re] += A[i][k][re]*B[k][j][re]-A[i][k][im]*B[k][j][im];
        C[i][j][im] += A[i][k][im]*B[k][j][re]+A[i][k][re]*B[k][j][im];
      }
    }
  }
}



/*
 *   multiply complex 3x3 matrix and 3 vector
 *        W = A X V
 */
void multiply_complex_matvec(A,V,W)
     double A[][3][2];
     double V[][2];
     double W[][2];
{
  int i;
  for(i=0;i<3;i++) {       
    W[i][re] = A[i][0][re]*V[0][re]-A[i][0][im]*V[0][im]+
               A[i][1][re]*V[1][re]-A[i][1][im]*V[1][im]+
               A[i][2][re]*V[2][re]-A[i][2][im]*V[2][im] ;
    W[i][im] = A[i][0][re]*V[0][im]+A[i][0][im]*V[0][re]+
               A[i][1][re]*V[1][im]+A[i][1][im]*V[1][re]+
               A[i][2][re]*V[2][im]+A[i][2][im]*V[2][re] ;
  }
}


/*
 *   copy complex 3x3 matrix 
 *        A --> B
 */
void copy_complex_matrix(A,B)
     double A[][3][2];
     double B[][3][2];
{
  memcpy(B,A,sizeof(double)*18);
}

/*
 *   clear complex 3x3 matrix 
 *        
 */
void clear_complex_matrix(A)
     double A[][3][2];
{
  memset(A,0,sizeof(double)*18);
}



/*
 *   oscillation probability in vacuum (w/o CP effect)
 *        
 *   nutype  : nue=1, numu=2, nutau=3 
 *   energy  : neutrino energy (GeV)
 *   path    : path length (km)
 *   prob[3] : oscillation prob
 *
 */
void get_vacuum_probability(nutype,energy,path,prob)
     int nutype ; 
     double energy, path;
     double prob[][3];
{
  double lovere ;
  double s21, s32, s31, ss21, ss32, ss31 ;
  int ista, iend ;

  // make more precise 20081003 rvw
  lovere= 1.26693281*(path)/(energy);
  s21 = sin(dm21*lovere);
  s32 = sin(dm32*lovere);       
  s31 = sin((dm21+dm32)*lovere) ;
  ss21 = s21*s21 ;
  ss32 = s32*s32 ;
  ss31 = s31*s31 ;

  /* ista = abs(*nutype) - 1 ; */
  for ( ista=0 ; ista<3 ; ista++ ) {
  for ( iend=0 ; iend<2 ; iend++ ) {
    prob[ista][iend]  = mix[ista][0][re]*mix[iend][0][re]*
                  mix[ista][1][re]*mix[iend][1][re]*ss21;
    prob[ista][iend] += mix[ista][1][re]*mix[iend][1][re]*
                  mix[ista][2][re]*mix[iend][2][re]*ss32;
    prob[ista][iend] += mix[ista][2][re]*mix[iend][2][re]*
                  mix[ista][0][re]*mix[iend][0][re]*ss31;
    if ( iend == ista ) {
      prob[ista][iend]  = 1.0-4.0*prob[ista][iend];
    } else {
      prob[ista][iend]  =    -4.0*prob[ista][iend];
    }
  }
  prob[ista][2]=1.0-prob[ista][0]-prob[ista][1];
  }
}


// want to output flavor composition of
// pure mass eigenstate, state
void convert_from_mass_eigenstate( state, flavor, pure )
                         int    state;
                         int    flavor;
                         double pure [][2];
{
  int    i,j;
  double mass    [3][2];
  double conj    [3][3][2];
  int    lstate  = state - 1;
  int    factor  = ( flavor > 0 ? -1. : 1. ); 
  // need the conjugate for neutrinos but not for
  // anti-neutrinos

  for (i=0; i<3; i++) {
        mass[i][0] = ( lstate == i ? 1.0 : 0. );
        mass[i][1] = (                     0. );
  }

  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
        conj[i][j][re] =        mix[i][j][re];
        conj[i][j][im] = factor*mix[i][j][im];
   }
 }
  multiply_complex_matvec(conj, mass, pure);

}


void conjugate_mixing_matrix()
{
  int i, j;
  double a[3][3][2];
 
  copy_complex_matrix(mix,a);
  
  for (i=0; i<3; i++){ 
    for (j=0; j<3; j++) {
        mix[i][j][re] =  a[i][j][re];
        mix[i][j][im] = -a[i][j][im];
    }
  }
}
