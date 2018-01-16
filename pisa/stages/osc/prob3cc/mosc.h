/***********************************************************************
  $Id: mosc.h,v 1.1.1.1 2008-07-03 23:38:06 skrep Exp $
***********************************************************************/

#ifndef __MOSCHINCLUDED__
#define __MOSCHINCLUDED__

typedef enum nu_type {
  data_type, 
  nue_type, 
  numu_type, 
  nutau_type,
  sterile_type,
  unknown_type} NuType;

void setmass(double dms21, double dms23, double dmVacVac[][3]);
 void setmix(double th12, double th13, double th23, double d, 
            double Mix[][3][2]);
 void putmix(double Mix[][3][2]);
 void setmix_barger(double th1, double th2, double th3, double d,
                   double Mix[][3][2]);
 void getM(double Enu, double rho, 
          double Mix[][3][2], double dmVacVac[][3], int antitype,
          double dmMatMat[][3], double dmMatVac[][3]);
 void getA(double L, double E, double rho, 
	  double Mix[][3][2], double dmMatVac[][3], double dmMatMat[][3],
	  int antitype, double A[3][3][2], double phase_offset) ;
 void setMatterFlavor(int flavor);
 void trans2p(double A[][3][2], double P[][3]);
 void propagate_vac(double Ain[][3][2], double L, double E, 
                   double Mix[][3][2], double dmVacVac[][3],
                   double Aout[][3][2]);
 void propagate_mat(double Ain[][3][2], double nelec, double L, double E, 
                   double Mix[][3][2], double dmVacVac[][3],
                   int antitype, double Aend[][3][2]);

 void setmix_sin(double,double,double,double,double[][3][2]);

 typedef enum matrix_type {
  standard_type,
  barger_type} MatrixType;

#endif /* __MOSCHINCLUDED__ */
