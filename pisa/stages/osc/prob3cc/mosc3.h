#ifndef __mosc3__
#define __mosc3__
	

extern "C" {
 void   init_mixing_matrix( double, double, double, double, double, double );
 void   get_oscillation_parameters( double, double, double, double, double, double );

 void   get_wavelength_23(double, double* );

 void   get_mixing_matrix_real( double [][3]);

 void   convert_from_mass_eigenstate( int, int , double [][2]);

 void   get_transition_matrix( int, double, double , double, double [][3][2], double );

 void   multiply_complex_matrix(double [][3][2], double [][3][2] , double [][3][2] );

 void   multiply_complex_matvec(double [][3][2], double [][2], double [][2] );


 void   copy_complex_matrix(double [][3][2], double [][3][2]);
 void   conjugate_mixing_matrix();

 void   clear_complex_matrix( double [][3][2]);

 void   get_vacuum_probability(int ,double ,double ,double[][3] );
}

#endif
