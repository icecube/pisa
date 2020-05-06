import numpy
import scipy.special
import scipy
import itertools
from . import llh_fast
import sys

# Numerical evaluation of integral representation 1, multi-dimensional integral over simplex
def log_lauricella_dumb_integral_multi(a, bs, c, zs, nthrows=1000):
    """
    First known Lauricella representation as an integral over the k-1 simplex. Only valid if c>sum(b_i).
    Everything done logarithmically to avoid numerical problems.
    """
    
    ### sampling from a simplex in n+1 dimensional space ... exponential random variables, normalized.... 
    ### this gives a uniform sample on the n simplex in n+1 dimension space (sum x=1).. but we want open (sum x <= 1) on n simplex in n dimensiona space ... easy, just drop last coordinate
    if(type(bs)==list):
        bs=numpy.array(bs)
    if(type(zs)==list):
        zs=numpy.array(zs)
    
    ## normalizing factor of gamma functions
    log_normalizing_factor=scipy.special.gammaln(numpy.array([c]))-sum(scipy.special.gammaln(bs))-scipy.special.gammaln(numpy.array([c])-sum(bs))-scipy.special.gammaln(len(zs)+1)

    ## random numbers for numerical integration
    random_nos=-numpy.log(numpy.random.uniform(size=(nthrows,len(zs)+1)))
    rsum=numpy.reshape(random_nos.sum(axis=1), (nthrows,1 ))
    random_nos=random_nos/rsum
      
    ## use average integration volume per throw
    log_integration_volume_per_throw=-numpy.log(float(nthrows))
    
    ## subtract last random no to land in k-1 simplex
    final_randoms=random_nos[:,:-1]
    
    
    ## the dirichlet part
    basic_dirichlet_logintegrand=( numpy.resize((bs-1), final_randoms.shape)*numpy.log(final_randoms) ).sum(axis=1) +(c-numpy.sum(bs)-1)*numpy.log(1.0-numpy.sum(final_randoms, axis=1))
    
   
    ## the extra integrand part
    log_integrand_vals2=-a*numpy.log((1.0-numpy.exp( scipy.special.logsumexp(numpy.log(final_randoms)+numpy.resize(numpy.log(zs), final_randoms.shape), axis=1))) ) 
    
    ## the log of the total integrated integrand
    log_result= scipy.special.logsumexp(log_integration_volume_per_throw+basic_dirichlet_logintegrand+log_integrand_vals2)
    
    return log_result+log_normalizing_factor
    
# Numerical evaluation of Integral representation 2 - 1-d integral over the line
def log_lauricella_dumb_integral_single(a,bs, c, zs, nthrows=1000):
    """
    Second known integral lauricella representation in form of a single integral over [0,1]. Only valid if c>a.
    """
    if(a==c):
        ## get out since this is a limiting case and not defined in the integral rep
        return (-bs*numpy.log(1.0-zs)).sum()

    random_vals=numpy.random.uniform(size=nthrows)
    random_vals2=numpy.resize(random_vals, (len(bs),nthrows)).T
    
    log_res=(a-1.0)*numpy.log(random_vals) + (c-a-1.0)* numpy.log(1.0-random_vals) + (  -bs*numpy.log(1.0-random_vals2*zs)).sum( axis=1)- numpy.log(float(nthrows))

    log_normfactor=scipy.special.gammaln(c)-scipy.special.gammaln(a)-scipy.special.gammaln(c-a)

    return scipy.special.logsumexp(log_normfactor+log_res)

def log_lauricella_from_carlson_r(a,bs,c,zs):
    
    new_bs=numpy.array(numpy.array(bs).tolist()+[c-sum(bs)])

    new_zs=numpy.array(numpy.array(1.0/(1.0-zs)).tolist()+[1.0])
    
    prefac=(bs*numpy.log(1.0/(1.0-zs))).sum()

    return prefac+numpy.log(llh_fast.c_Rn(int(a-c),new_bs, new_zs))

#### exact Lauricella 2F1(a,b,c,z) with c>a

def log_bin_fac(up,down):
    return scipy.special.gammaln(up+1.0)-scipy.special.gammaln(up-down+1.0)-scipy.special.gammaln(down+1.0)

def log_beta(a,b):
    return scipy.special.gammaln(a)+scipy.special.gammaln(b)-scipy.special.gammaln(a+b)

def lauricella_2f1_exact(a,b,c,z):
    if(a<2):
        sys.exit("a needs to be >=2")
        return None
    log_prefac=log_bin_fac(c-1.0, a-1.0)-(c-1)*numpy.log(z)
    
    #print "log prefac ", log_prefac
    
    summands=[]
    signs=[]
    k=c-a
    ## all the first summands are negative
    for m in range(a-2+1):
        summands.append(-log_bin_fac(m+c-a, m)+(m+c-a)*numpy.log(z))
        signs.append(-1.0)
        
        #print "first ", summands[-1]
    
    #print "summands after rfirst ", summands
    if(k>=2):
        sum_to_i=(1.0/(numpy.arange(k-1)+1.)).sum()
        #print sum_to_i
        
        summands.append(numpy.log(k)+numpy.log(sum_to_i))
        signs.append((-1)**(k))
        
        
        #print "sumto i", sum_to_i
        summands.append(numpy.log(k)+numpy.log(sum_to_i)+(k-1)*numpy.log(1-z))
        signs.append((-1)**(k-1))
        #print summands[-2:]
        
        
        #print "two summands ", summands[-2:]
        #print signs[-2:]
        #print scipy.special.logsumexp(summands[-2:], b=signs[-2:])
        
        
    if(k>=3):
        for i in numpy.arange(k-2)+1:
            #print "i ", i
            signs.append((-1)**(k-i))
            #print signs
            sum_to_j=(1.0/(numpy.arange(k-1-i)+1.)).sum()
            #print "sumto j", sum_to_j
            #print log_bin_fac(k-1,i)
            
            summands.append(numpy.log(k)+log_bin_fac(k-1,i)+numpy.log(sum_to_j)+i*numpy.log(z))
            #print "last one ", summands[-1]
    ## second
    
    #non_log_summand= k*((-1)**(k))*(1-z)**(k-1)*numpy.log(1-z)
    #print non_log_summand
    non_log_summand2= numpy.log(k)+(k-1)*numpy.log(1-z)+numpy.log(-1.0*numpy.log(1-z))
    summands.append(non_log_summand2)
    signs.append((-1)**(k-1)) ## k-1 because +1 is pulled into logarithm to make it positive for another log
    #print "log summand ", numpy.exp(summands[-1])
    ret, si=scipy.special.logsumexp(summands, b=signs, return_sign=True)
    #return numpy.exp(log_prefac+ret)*si+numpy.exp(log_prefac)*non_log_summand
    
    return log_prefac+ret
def lauricella_2f1_exact_second_way(a,b,c,z):
    
    prefac=-log_beta(a,c-a)
    summands=[]
    signs=[]
    for i in range(c-a):
        signs.append((-1)**i)
        summands.append(lauricella_2f1_exact(a+i,b,a+i+1,z)+log_bin_fac(c-a-1,i)-numpy.log(a+i))
    
    #print "summands ", summands
    #print "signs", signs
    return prefac+scipy.special.logsumexp(summands, b=signs)
