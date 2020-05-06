import scipy.special
import scipy.misc


## carlson-R function - a scalar, b,z equal-sized vectors .. can be calculated faster in c-implementation (see llh_fast_defs.c)
def log_carlson_r_positive_a(a,b,z):

    log_prefac=scipy.special.gammaln(a+1)+scipy.special.gammaln(sum(b))-scipy.special.gammaln(a+sum(b))
    new_zs=z
        
    new_zs_log=numpy.log(new_zs)
    new_bs=b
   
    cs=[0.0]
    
    if(a>0):
        
        lambdas=[]
        
        new_bs_log=numpy.log(new_bs)
        running_lambda_vec=new_bs_log
        
        for cur_ind in range(a):
            running_lambda_vec+=new_zs_log
            lambdas.append(scipy.misc.logsumexp(running_lambda_vec).sum())
            
            new_cs=scipy.misc.logsumexp( numpy.array(lambdas[::-1])+numpy.array(cs))-numpy.log(cur_ind+1)
            cs.append(new_cs)

    return log_prefac+cs[-1]