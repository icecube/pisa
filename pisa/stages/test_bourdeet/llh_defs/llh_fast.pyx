import numpy as np
cimport numpy as np


np.import_array()

cdef extern from "llh_fast_defs.h":
    
    void R_n(int n, double *bs,  double *z,size_t w_size, double *result)
    
    void pg_general_iterative_mult(int k, double *weights, size_t w_size, double alpha, double *result)
    void pg_general_jac(int k, double *weights, size_t w_size, double alpha, double *result)

    void pg_say(int k, double *weights, size_t w_size,double *result)
    void standard_poisson(int k, double *weights, size_t w_size,double *result)


    void pg_general(int k, double *weights, size_t w_size, double alpha,double extra_prior_counter, double extra_prior_weight,  double *result)
    void pg_general_equivalent(int k, double *weights, size_t w_size, double alpha,double extra_prior_counter, double extra_prior_weight,  double *result)
    void pgpg_general(int k, double *weights, size_t w_size, double alpha, double beta,double extra_prior_counter, double extra_prior_weight,  double *result)

    void pgpg_absolute_general(int k, double *weights, size_t w_size, double alpha, double beta, double v,double extra_prior_counter, double extra_prior_weight,  double *result)
    void pgpg_absolute_general_equivalent(int k, double *weights, size_t w_size, double alpha, double beta, double v,double extra_prior_counter, double extra_prior_weight,  double *result)

cdef extern from "poisson_gamma.h":
    
    void generalized_pg_mixture_marginalized_combined(int k,double *new_alphas,  double *betas,  double *gammas, double *alphas_2, double *betas_2, size_t w_size,size_t w_size_2, double *result);

  


def c_Rn(int n, np.ndarray[np.float64_t, ndim=1] bs,np.ndarray[np.float64_t, ndim=1] z):
        
    cdef double res=0.0
    R_n(n,  <double*> bs.data,<double*> z.data, z.size, &res)

    return res



def c_pg_general_iterative_mult(int k, np.ndarray[np.float64_t, ndim=1] weights, double alpha=0):
        
    cdef double res=0.0
    pg_general_iterative_mult(k, <double*> weights.data, weights.size, alpha, &res)

    return res

def c_pg_general_jac(int k, np.ndarray[np.float64_t, ndim=1] weights, double alpha=0):
        
    cdef np.ndarray result_array = np.zeros(len(weights), dtype=float)
    pg_general_jac(k, <double*> weights.data, weights.size, alpha, <double*> result_array.data)

    return result_array

def c_pg_say(int k, np.ndarray[np.float64_t, ndim=1] weights):
        
    cdef double res
    pg_say(k, <double*> weights.data, weights.size, &res)

    return res
def c_standard_poisson(int k, np.ndarray[np.float64_t, ndim=1] weights):
        
    cdef double res
    standard_poisson(k, <double*> weights.data, weights.size, &res)

    return res


def c_pg_general(int k, np.ndarray[np.float64_t, ndim=1] weights, double alpha=0,double extra_prior_counter=0.0, double extra_prior_weight=0.0):
        
    cdef double res=0.0
    pg_general(k, <double*> weights.data, weights.size, alpha, extra_prior_counter, extra_prior_weight, &res)

    return res

def c_pg_general_equivalent(int k, np.ndarray[np.float64_t, ndim=1] weights, double alpha=0,double extra_prior_counter=0.0, double extra_prior_weight=0.0):
        
    cdef double res=0.0
    pg_general_equivalent(k, <double*> weights.data, weights.size, alpha, extra_prior_counter, extra_prior_weight, &res)

    return res

def c_pgpg_general(int k, np.ndarray[np.float64_t, ndim=1] weights, double alpha=0, double beta=0,double extra_prior_counter=0.0, double extra_prior_weight=0.0):
        
    cdef double res=0.0
    pgpg_general(k, <double*> weights.data, weights.size, alpha, beta, extra_prior_counter, extra_prior_weight, &res)

    return res

def c_pgpg_absolute_general(int k, np.ndarray[np.float64_t, ndim=1] weights, double alpha=0.0, double beta=0.0, double v=1.0,double extra_prior_counter=0.0, double extra_prior_weight=0.0):
        
    cdef double res=0.0
    pgpg_absolute_general(k, <double*> weights.data, weights.size, alpha, beta,v, extra_prior_counter, extra_prior_weight, &res)

    return res

def c_pgpg_absolute_general_equivalent(int k, np.ndarray[np.float64_t, ndim=1] weights, double alpha=0.0, double beta=0.0, double v=1.0,double extra_prior_counter=0.0, double extra_prior_weight=0.0):
        
    cdef double res=0.0
    pgpg_absolute_general_equivalent(k, <double*> weights.data, weights.size, alpha, beta,v, extra_prior_counter, extra_prior_weight, &res)

    return res















