import numpy as np
import scipy
from . import poisson_gamma_mixtures
import copy


########################################################################################
####### Relevant Poisson generalizations from the paper https://arxiv.org/abs/1712.01293
####### and the newer one https://arxiv.org/abs/1902.08831
####### All formulas return the log-likelihood or log-probability
####### Formulas are not optimized for speed, but for clarity (except c implementations to some extent).
####### They can definately be sped up by smart indexing etc., and everyone has to adjust them to their use case anyway.
####### Formulas are not necessarily vetted, please try them out yourself first.
####### Any questions: thorsten.gluesenkamp@fau.de
########################################################################################

np.seterr(divide="warn")

######################################
#### standard Poisson likelihood
######################################
def poisson(k, lambd):
 
    return (-lambd+k*np.log(lambd)-scipy.special.gammaln(k+1)).sum()

#############################################################


##############################################################
### generalization (2) - allbin expression, https://arxiv.org/abs/1902.08831 eq. 47
##############################################################

## the next 3 functions are used to calculate the convolution of N poisson-gamma mixtures in a safe way
### PG conv PG conv PG ... etc
import itertools
def bars_and_stars_iterator(tot_k, num_bins):

    for c in itertools.combinations(range(tot_k+num_bins-1), num_bins-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(tot_k+num_bins-1,))]

### calculate single poisson gamma mixture in calc_pg vectorized over alpha/beta
def calc_pg(k, alpha, beta):
    return (scipy.special.gammaln(k+alpha) -scipy.special.gammaln(k+1.0)-scipy.special.gammaln(alpha) + (alpha)* np.log(beta) - (alpha+k)*np.log(1.0+beta))


## second way to calculate generalized pg mixture, based on iterative sum
def generalized_pg_mixture_2nd(k, alphas, betas):
    
    iters=[np.array(i) for i in bars_and_stars_iterator(int(k), len(betas))]
    
    log_res=[]
    for it in iters:
        
        log_res.append(calc_pg(it, alphas, betas).sum())
    
    return scipy.special.logsumexp(log_res)


## calculate c-based version .. if it doesnt suffice in precision, go to direct convolution
def fast_pgmix(k, alphas=None, betas=None):
    '''
    Core function that computes the generalized likelihood 2

    '''
    assert isinstance(k,np.int64),'ERROR: k must be an int'
    assert isinstance(alphas,np.ndarray),'ERROR: alphas must be numpy arrays'
    assert isinstance(betas,np.ndarray),'ERROR: betas must be numpy arrays'

    assert np.sum(alphas<=0)==0, 'ERROR: detected alpha values <=0'
    assert np.sum(betas<=0)==0,'ERROR: detected beta values <=0'

    ret=poisson_gamma_mixtures.c_generalized_pg_mixture(k, alphas, betas)
    #print('k: {0},nweights: {1}, return value : {2}\t alphas {3} \t betas: {4}'.format(k,len(betas),ret,alphas,betas))

    if np.isnan(ret):
        return 1.

    if not np.isfinite(ret):
        for a,b in zip(alphas,betas):
            print(a,b,poisson_gamma_mixtures.c_generalized_pg_mixture(k,np.array([a]),np.array([b])))


    if(ret>1e-300):
        return np.log(ret)

    elif ret>=0. and ret<=1e-300:
        # Replace a probability of exatcly  to a small number
        # to avoid errors in logarithm
        return np.log(1e-300)
    else:
        print('ERROR: running the c-based method failed.')
        print(k,ret,alphas,betas)
        raise Exception

def normal_log_probability(k,weight_sum=None):
    '''
    return a simple normal probability of
    mu = weight_sum and sigma = sqrt(weight_sum)

    '''
    import scipy as scp 
    from scipy.stats import norm

    P = norm.pdf(k, loc=weight_sum, scale=np.sqrt(weight_sum))

    logP = np.log(max([1.e-10,P]))

    return logP