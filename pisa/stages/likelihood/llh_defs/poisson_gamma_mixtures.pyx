import numpy as np
cimport numpy as np
import sys


np.import_array()

cdef extern from "poisson_gamma.h":
    
    void generalized_pg_mixture_marginalized_combined(int k,double *new_alphas,  double *betas,  double *gammas, double *alphas_2, double *betas_2, size_t w_size,size_t w_size_2, double *result);
    
    void generalized_pg_mixture_marginalized(int k, double *gammas, double *deltas,  double *epsilons , size_t w_size, double *result);

    void generalized_pg_mixture(int k, double *alphas, double *betas, size_t w_size, double *result);

    void single_pgg(int k, double A, double B, double Q, double kmc, double gamma, double *log_sterlings, int sterling_size, double *res);
    void multi_pgg(int k, double *A, double *B, double *Q, double *kmc, double *gamma, int nsources, double *log_sterlings, int sterling_size, double *res);
  
def c_generalized_pg_mixture(int k, np.ndarray[np.float64_t, ndim=1] alphas, np.ndarray[np.float64_t, ndim=1] betas):
        
    cdef double res=0.0
    generalized_pg_mixture(k, <double*> alphas.data, <double*> betas.data, alphas.size, &res)

    return res

## eq. 97 which only looks at the marginalized expressions ,but drops the alphas/betas
## used for generalization (1)
def c_generalized_pg_mixture_marginalized(int k, np.ndarray[np.float64_t, ndim=1] gammas, np.ndarray[np.float64_t, ndim=1] deltas, np.ndarray[np.float64_t, ndim=1] epsilons):
        
    cdef double res=0.0
    generalized_pg_mixture_marginalized(k, <double*> gammas.data, <double*> deltas.data,  <double*> epsilons.data, gammas.size, &res)

    return res

def c_generalized_pg_mixture_marginalized_combined(int k, np.ndarray[np.float64_t, ndim=1] alphas, np.ndarray[np.float64_t, ndim=1] betas, np.ndarray[np.float64_t, ndim=1] gammas, np.ndarray[np.float64_t, ndim=1] alphas_2, np.ndarray[np.float64_t, ndim=1] betas_2):
        
    cdef double res=0.0
    generalized_pg_mixture_marginalized_combined(k, <double*> alphas.data, <double*> betas.data, <double*> gammas.data, <double*> alphas_2.data, <double*> betas_2.data, alphas.size, alphas_2.size, &res)

    return res

def c_single_pgg(int k, double A,double B, double Q, double kmc, double gamma, np.ndarray[np.float64_t, ndim=2] log_sterlings):

    cdef double res=0.0
    if((log_sterlings.shape[0]-1)<k):
        print "sterling matrix too small .. requires at least k+1, k is ..", k
        sys.exit(-1)
   
    single_pgg(k, A, B, Q, kmc, gamma, <double*> log_sterlings.data, int(log_sterlings.shape[0]), &res)

    return res


def c_multi_pgg(int k, np.ndarray[np.float64_t, ndim=1] A,np.ndarray[np.float64_t, ndim=1] B, np.ndarray[np.float64_t, ndim=1] Q, np.ndarray[np.float64_t, ndim=1] kmc, np.ndarray[np.float64_t, ndim=1] gamma, np.ndarray[np.float64_t, ndim=2] log_sterlings):

    cdef double res=0.0
    if((log_sterlings.shape[0]-1)<k):
        print "sterling matrix too small .. requires at least k+1, k is ..", k
        sys.exit(-1)
   
    multi_pgg(k, <double*> A.data, <double*> B.data, <double*> Q.data, <double*> kmc.data, <double*> gamma.data, int(A.shape[0]), <double*> log_sterlings.data, int(log_sterlings.shape[0]),  &res)

    return res

