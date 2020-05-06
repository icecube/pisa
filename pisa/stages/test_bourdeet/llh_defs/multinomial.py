import numpy
import scipy
import scipy.special
import itertools
import time
from . import lauricella_fd
from . import poisson

## 0) The standard approach: assume we have inifnite statistics and calculate standard multinomial probability
def multinomial_standard(k, lambd):

    lsum=lambd.sum()

    return (k*numpy.log(lambd/lsum)).sum() - scipy.special.gammaln(k+1).sum() + scipy.special.gammaln(sum(k)+1)


#1) Easiest case
## Mnomial extension for equal weighs in ALL bins... simple Dirichlet-Multinomial distribution DM(k;alpha) (eq. 28) 
## factorial is expressed Gamma(x+1)

def log_DM(k_s, alphas):

    k_mcs=alphas
    tot_mc=sum(k_mcs)
    tot_k=sum(k_s)

    return scipy.special.gammaln(k_mcs+k_s).sum()-scipy.special.gammaln(k_mcs).sum()-scipy.special.gammaln(numpy.array([tot_mc+tot_k])) + scipy.special.gammaln(numpy.array([tot_mc])) + scipy.special.gammaln(numpy.array([tot_k+1]))-scipy.special.gammaln(k_s+1).sum()

#/*******************************************************************/


#2) Slightly harder case
## MNomial extension with equal weights per bin, but different weights between bins (eq. 41) -> Integral over multinomial factor with scaled Dirichlet density.
## Approximating the laruicella function F_D(a,b,c,z) for c>a 

def log_multinomial_equal_weights(k_s, k_mcs, weights, nthrows=100000,integral_type="standard_lauricella", prior_factor=0.0):

    kmcs_w_prior=k_mcs+prior_factor
    tot_mc=sum(kmcs_w_prior)
    tot_k=sum(k_s)

    DM_prefac=log_DM(k_s, kmcs_w_prior)

    ## there is only one possibility
    smallest=min(weights)
    sort_mask=numpy.argsort(weights)
    sorted_ws=numpy.array(weights)[sort_mask]

    zs=(1.0-smallest/sorted_ws)[1:]
    
    sorted_k_mcs=kmcs_w_prior[sort_mask][1:]
    sorted_k_s=k_s[sort_mask][1:]

    bs=sorted_k_mcs+sorted_k_s
    #simple_bs=sorted_k_mcs
    c=tot_mc+tot_k

    a=tot_mc
    
    log_W=numpy.sum( (sorted_k_mcs)*(numpy.log(1.0-zs))   ) 
    

    ## single integral (one-dimensional) applicable since c > a
    lauric=lauricella_fd.log_lauricella_dumb_integral_single(a, bs, c, zs, nthrows=nthrows) 

    extended_zs=numpy.append(zs, [1])
    bs=numpy.append(bs,[c-sum(bs)])

        
    return DM_prefac+log_W+lauric
 
## Calculate all combinations of tot_k distributed over num_bins, such that sum k_i = tot_k
def bars_and_stars_iterator(tot_k, num_bins):

    for c in itertools.combinations(range(tot_k+num_bins-1), num_bins-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(tot_k+num_bins-1,))]

#3) General case
## MNomial extension with general weights per bin (eq. 43) -> Combinatorial ratio of many Lauricella functions
## Calculates all combinations at once, therefore only one calculation is necessary for p(k), and then all other p(k) 
## are automatically calculated. Only works for low-dimensional problems because of combinatorial burden.
######### CAUTION: this is a generator that returns a function, not a log-probability. Also for too high k_tot this will take too long. ################
def log_multinomial_general_weights_generator(tot_k, weight_list, nthrows=100000, prior_factor=0.0):

    total_weights=[] # a list of all weights in one single array - used for denominator
    bin_mask=[] # keeps track of which weights are in which bin

    for bin_index, i in enumerate(weight_list):
        for j in i:
            total_weights.append(j)
            bin_mask.append(bin_index)

    denom_log_list=[]
        
    ## distribute total k over all individual weights
    bin_distributions=[i for i in bars_and_stars_iterator(tot_k, len(total_weights))]
    total_weights=numpy.array(total_weights)
    bin_mask=numpy.array(bin_mask)

    for bin_dist in bin_distributions:
        denom_log_list.append(log_multinomial_equal_weights(numpy.array(bin_dist), numpy.ones(len(total_weights)), total_weights, nthrows=nthrows, prior_factor=prior_factor))
    
    numerator_log_list=[]
    probs=dict()

    tbef=time.time()
    ## now find the numerator for the particular k_s given to the function
    for dist_index, bin_dist in enumerate(bin_distributions):
        #if(dist_index%100==0):
        #    print "Done .. ", float(dist_index)/float(len(bin_distributions)), time.time()-tbef
        new_k_tuple=()
        for cur_bin_index, _ in enumerate(weight_list):
            
            new_k_tuple+=(sum(numpy.array(bin_dist)[bin_mask==cur_bin_index]),)
            

        if not new_k_tuple in probs.keys():
            probs[new_k_tuple]=[]
        probs[new_k_tuple].append(denom_log_list[dist_index])

    def return_fn(new_k_s):
        return scipy.special.logsumexp(probs[tuple(new_k_s)])-scipy.special.logsumexp(denom_log_list)

    return return_fn

### Now the two ratio constructions
### 1) Equal weights (eq. 52)
### k_s - numpy array of observed events per bin
### k_mcs numpy array of numbber of mc events per bin
### avg_weights - numpy array of avg weight per bin

## based on some older formula with sampled lauricella funtions that was very slow 
"""
def log_multinomial_poisson_ratio_equal_weights(k_s, k_mcs, avg_weights, lauricella_calc="exact"):

    numerator=poisson.pg_equal_weights(k_s,k_mcs,avg_weights, prior_factor=0.0)

    index_list=[]
    total_weights=[]
    for ind in range(len(k_s)):
        total_weights.extend(k_mcs[ind]*[avg_weights[ind]])
        index_list.append(numpy.ones(k_mcs[ind])*ind)

    total_weights=numpy.array(total_weights)
    denominator=poisson.fast_pg_single_bin(sum(k_s), total_weights, lauricella_calc=lauricella_calc)
   
    return numerator-denominator
"""

### 2) General weights ratio construction (eq. 53)
### k_s - number of observed events in a bin - numpy array
### all_weights - numpy array of all weights in all bins
### index_list - a list of numpy arrays with indices indexing the weights from 'all_weights', one index_arraay per bin

def log_multinomial_poisson_ratio_general_weights(k_s, all_weights, index_list):

    numerator=0.0

    for ind in range(len(k_s)):
       
        numerator+=poisson.fast_pg_single_bin(k_s[ind], all_weights[index_list[ind]])
    

    denominator=poisson.fast_pg_single_bin(sum(k_s), all_weights)
    
    return numerator-denominator

