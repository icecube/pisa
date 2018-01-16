/***************************************************
 * Swig module description file for wrapping the
 * the BargerPropagator class.
 ***************************************************/

%module BargerPropagator

%{
#define SWIG_FILE_WITH_INIT
#include "BargerPropagator.h"
#include "EarthDensity.h"
#include "NeutrinoPropagator.h"

#include "mosc3.h"
#include "mosc.h"
%}
%include "numpy.i"

%init %{
  import_array();
%}

%apply (double* IN_ARRAY1, int DIM1) {(double* ecen, int ecen_length), (double* czcen, int czcen_length)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* probList, int prop_length), (double* evals, int evals_length), (double* czvals, int czvals_length)};

%include "BargerPropagator.h"
