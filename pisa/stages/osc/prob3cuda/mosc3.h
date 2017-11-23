#ifndef __MOSC3_H__
#define __MOSC3_H__

// Sets all entries to zero
__device__ void clear_complex_matrix( fType A[][3][2] );

__device__ void copy_complex_matrix( fType A[][3][2], fType B[][3][2] );

__device__ void multiply_complex_matrix( fType A[][3][2], fType B[][3][2], fType C[][3][2] );

__device__ void clear_probabilities( fType A[3][3] );

__device__ void multiply_complex_matvec( fType A[][3][2], fType V[][2], fType W[][2] );

__device__ void convert_from_mass_eigenstate( int state, int flavor, fType pure[][2],
                                              fType mix[][3][2] );

__device__ void get_transition_matrix( int nutype, fType Enu, fType rho, fType Len,
                                       fType Aout[][3][2], fType phase_offset,
                                       fType mix[][3][2], fType dm[][3]);


#endif
