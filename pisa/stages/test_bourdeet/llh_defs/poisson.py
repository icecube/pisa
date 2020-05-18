import numpy as np
import scipy
from . import lauricella_fd
from . import llh_fast
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

def poisson_gen2(data, individual_weights_dict, mean_adjustments, larger_weight_variance=False):
    '''
    Main function that gets called when we select the
    second generalization option.
    '''

    tot_llh=0.0
    all_alphas = []
    all_betas  = []
    llh_per_bin = []

    for cur_bin_index, _ in enumerate(list(individual_weights_dict.values())[0]):

        # 
        # If the data count is a above a certain number, return a normal 
        # probability

        if data[cur_bin_index] >100:
            weight_sum = 0.0
            a = {}
            b = {}
            for src in individual_weights_dict.keys():
                this_weights=individual_weights_dict[src][cur_bin_index]
                weight_sum+=sum(this_weights)
                a[src] = np.NaN
                b[src] = np.NaN

            all_alphas.append(a)
            all_betas.append(b)

            if weight_sum<0:
                print('\nERROR: negative weight sum should not happen...')
                raise Exception

            logP = normal_log_probability(k=data[cur_bin_index],weight_sum=weight_sum)
            print('thorsten llh, bin :',cur_bin_index, weight_sum,data[cur_bin_index])
            llh_per_bin.append(logP)
            tot_llh+=logP

        else:


            alphas={}
            betas={}

            #
            # Computing the approximate values of betas and alphas
            # based on the moments of the weigt distribution
            for src in individual_weights_dict.keys():

                this_weights=individual_weights_dict[src][cur_bin_index]

                if(len(this_weights)>0):
                    kmc=float(len(this_weights))
                    mu=float(len(this_weights))
                    
                    exp_w=0.0
                    
                    
                    exp_w=np.mean(this_weights)
                    var_w=0.0

                    if(larger_weight_variance):
                        var_w=(this_weights**2).sum()/float(len(this_weights))
                    else:
                        var_w=((this_weights-exp_w)**2).sum()/(float(len(this_weights)))

                    var_z=(var_w+exp_w**2)

                    beta=exp_w/var_z
                    trad_alpha=(exp_w**2)/var_z

                    #sumw=this_weights.sum()
                    #sqrw=(this_weights**2).sum()
                    

                    extra_fac=mean_adjustments[src]

                    alphas[src] = (mu+extra_fac)*trad_alpha
                    betas[src]  = beta

                else:
                    print(src,this_weights)
                    raise Exception
                    alphas[src] = np.NaN
                    betas[src]  = np.NaN

            all_alphas.append(alphas)
            all_betas.append(betas)

            # Compute the likelihood only if there are non-NaNs values in at least one of the sets
            array_of_alphas  = np.array(list(alphas.values()))

            if sum(np.isfinite(array_of_alphas))>0:
                A = data[cur_bin_index]
                #print('running the "fast" gamma mix...')

                AAA = np.array(list(alphas.values()))

                BBB = np.array(list(betas.values()))

                new_llh=fast_pgmix(A, AAA[np.isfinite(AAA)], BBB[np.isfinite(BBB)])
                llh_per_bin.append(new_llh)
                tot_llh+=new_llh

                if not np.isfinite(tot_llh):
                    raise Exception('WOW! llh is now infinite!')
            else:
                print('EMPTY BIIIIIN', cur_bin_index)
                raise Exception

    return tot_llh, llh_per_bin

########################################
# end generalization (2)
########################################


####################################################
### effective generalization (2) - allbin expression, https://arxiv.org/abs/1902.08831 , eq 48
####################################################

def poisson_gen2_effective(data, individual_weights_dict, mean_adjustments):

    tot_llh=0.0

    for cur_bin_index, _ in enumerate(list(individual_weights_dict.values())[0]):

        mus=[]
        all_weights=[]

        for src in individual_weights_dict.keys():

            this_weights=individual_weights_dict[src][cur_bin_index]

            if(len(this_weights)>0):
                
                all_weights.extend(this_weights.tolist())
                
                mus.append(float(len(this_weights))+mean_adjustments[src])


        all_weights=np.array(all_weights)

        if(len(all_weights)>0):
            #print all_weights
            kmc=float(len(all_weights))
           

            exp_w=np.mean(all_weights)
            var_w=((exp_w-all_weights)**2).sum()/kmc

            var_z=(var_w+exp_w**2)

            #print "expw,varw,varz", exp_w,var_w,var_z
            beta=exp_w/var_z
            trad_alpha=(exp_w**2)/var_z
            
            alpha=sum(mus)*trad_alpha
            k=data[cur_bin_index]

            #print "alpha,beta,k", alpha,beta,k

            this_llh= (scipy.special.gammaln((k+alpha)) -scipy.special.gammaln(k+1.0)-scipy.special.gammaln(alpha) + (alpha)* np.log(beta) - (alpha+k)*np.log(1.0+beta)).sum()

            tot_llh+=this_llh

    return tot_llh

##################################
### generalization (3) functions, https://arxiv.org/abs/1902.08831, eq. 51
##################################

### Genertes stirling numbers (logarithmic) up to maximum no of max_val
### returns a lower-triangular matrix
def generate_log_stirling(max_val=1000):

    arr=np.zeros(shape=(max_val+1, max_val+1))*(-np.inf)
    arr[0][0]=0.0
    for i in range(max_val+1):
        for j in range(i+1):
            if(i==0 and j==0):
                arr[i][j]=0.0
                continue
            
            if(j==0):
                arr[i][j]=-np.inf
                continue
            if(i==j):
                arr[i][j]=0.0
                continue
          
            arr[i][j]=scipy.special.logsumexp([np.log(i-1.0)+arr[i-1][j],arr[i-1][j-1]])
    
    return arr

def poisson_gen3(k, individual_weights_dict, mean_adjustments, log_stirlings,s_factor=1.0, larger_weight_variance=False):

    tot_llh=0.0

    num_sources=len(individual_weights_dict.keys())

    for cur_bin_index, _ in enumerate(list(individual_weights_dict.values())[0]):

        As=[]
        Bs=[]
        Qs=[]
        kmcs=[]
        gammas=[]

        for src in individual_weights_dict.keys():

            this_weights=individual_weights_dict[src][cur_bin_index]

            if(len(this_weights)>0):
                kmc=float(len(this_weights))
                mu=float(len(this_weights))

                Q=0.0
                if(larger_weight_variance>0):
                    exp_w=np.mean(this_weights)
                    var_w=0.0

                    ## pdf is a mixture of gammas
                    var_w=(this_weights**2).sum()/float(len(this_weights))
                    
                    var_z=(var_w+exp_w**2)
                    beta=exp_w/var_z
                    trad_alpha=(exp_w**2)/var_z
                    Q=trad_alpha
                else:

                    sumw=this_weights.sum()
                    sumw2=(this_weights**2).sum()
                    
                    beta=sumw/sumw2

                    trad_alpha=sumw**2/sumw2
                    Q=(1.0/kmc)*(trad_alpha)
                
                A=beta/(1.0+beta)
                B=1.0/(1.0+beta)

                extra_fac=mean_adjustments[src]


                As.append(A)
                Bs.append(B)
                Qs.append(Q)

                kmcs.append( (mu+extra_fac)/s_factor)
                gammas.append(1.0/s_factor)

        As=np.array(As)
        Bs=np.array(Bs)
        Qs=np.array(Qs)
        kmcs=np.array(kmcs)
        gammas=np.array(gammas)

        tot_llh+=poisson_gamma_mixtures.c_multi_pgg(int(k[cur_bin_index]), As,Bs,Qs,kmcs,gammas, log_stirlings)

    return tot_llh          

######### ##################
#### end generalization (3)
############################

############################################
### generic preprocessing to include prior information and empty bins as described 
### in paper (1) https://arxiv.org/abs/1712.01293 and  paper (2) https://arxiv.org/abs/1902.08831
############################################

def generic_pdf(data, dataset_weights, type="gen2", empty_bin_strategy=1, empty_bin_weight="max", mean_adjustment=True, s_factor=1.0, larger_weight_variance=False, log_stirling=None):
    """
    data - a np array of counts for each bin
    dataset_weights_list - a dictionary of lists of np arrays. Each list corresponds to a dataset and contains np arrays with weights for a given bin. empty bins here mean an empty array
    type - basic_pg/gen1/gen2/gen2_effective/gen3 - handles the various formulas from the two papers - (basic_pg (paper 1), all others (paper 2))
    empty_bin_strategy - 0 (no filling), 1 (fill up bins which have at least one event from other sources), 2 (fill up all bins)
    empty_bin_weight - what weight to use for pseudo counts in empty  bins? "max" , maximum of all weights of dataset (used in paper) .. could be mean etc
    mead_adjustment - apply mean adjustment as implemented in the paper? yes/no
    weight_moments - change to more "unbiased" way of determining weight distribution moments as implemented in the paper

    default settings (as stated towards the end of the paper): gen2 , empty_bin_strategy=1, mead_adjustment=True
    """

    ## calculate number of mc events / bin / dataset
    kmc_dict=dict()
    max_weights=dict()
    for dsname in dataset_weights.keys():
        mw=max([max(w) if len(w)>0 else 1.0 for w in dataset_weights[dsname]])
        kmc_dict[dsname]=np.array([len(w) for w in dataset_weights[dsname]])
        max_weights[dsname]=mw


    ## calculate mean adjustment per dataset
    mean_adjustments=dict()
    for dsname in kmc_dict.keys():

        avg_kmc=np.mean(kmc_dict[dsname])
                            
        delta_alpha=0.0
        if(avg_kmc<1.0):
            delta_alpha=-(1.0-avg_kmc)+1e-3
        mean_adjustments[dsname]=delta_alpha


    ## fill in empty bins - update the weights
    new_weights=copy.deepcopy(dataset_weights)

    ## strategy 1 - fill up only bins that have at least 1 mc event from any dataset
    if(empty_bin_strategy==1):

        for bin_index in range(len(data)):
            
            weight_found=False
            for dsname in kmc_dict.keys():
                if(kmc_dict[dsname][bin_index] > 0):
                    weight_found=True
                    

            if(weight_found):

                for dsname in kmc_dict.keys():
                    if(kmc_dict[dsname][bin_index]==0):
                        
                        new_weights[dsname][bin_index]=np.array([max_weights[dsname]])
                        

    # strategy 2 - fill up all bins
    elif(empty_bin_strategy==2):
        for bin_index in range(len(data)):
            
            for dsname in kmc_dict.keys():
                if(kmc_dict[dsname][bin_index]==0):
                    new_weights[dsname][bin_index]=np.array([max_weights[dsname]])


    ## now loop through all bins and call respective likelihood
    ## manifest mean adjustment possible only in gen2 and gen3 (see table in paper)

    llh_res=0.0

    if(type=="gen3"):
        if(log_stirling is None):
            log_stirling=generate_log_stirling(max_val=max([max(data),1]))

        llh_res=poisson_gen3(data, new_weights, mean_adjustments, log_stirling,s_factor=s_factor, larger_weight_variance=larger_weight_variance)
        
    elif(type=="gen2"):
        llh_res, llh_per_bin =poisson_gen2(data, new_weights, mean_adjustments, larger_weight_variance=larger_weight_variance)
    elif(type=="gen2_effective"):
        llh_res=poisson_gen2_effective(data, new_weights, mean_adjustments)
    else:

        ## calculate an effective mean adjustment per weight (no manifest mean adjustment possible)

        for bin_index in range(len(data)):
            total_weights=[]
            individual_mean_adjustments=[]

            for dsname in new_weights.keys():
                this_weights=new_weights[dsname][bin_index].tolist()
                this_len=len(this_weights)
                total_weights.extend(this_weights)

                ## alpha* = alpha/N
                if(this_len>0):
                    individual_mean_adjustments.extend(this_len*[mean_adjustments[dsname]/float(this_len)])

            if(len(total_weights)>0):

                total_weights=np.array(total_weights)
                individual_mean_adjustments=np.array(individual_mean_adjustments)

                if(type=="basic_pg"):
                    llh_res+=fast_pg_single_bin(data[bin_index], total_weights, mean_adjustment=individual_mean_adjustments)
                elif(type=="gen1"):
                    llh_res+=fast_pgpg_single_bin(data[bin_index], total_weights, mean_adjustment=individual_mean_adjustments)
                


    return llh_res, llh_per_bin, mean_adjustments, data, new_weights

#################################
### arXiv:1901.04645, Arg+elles et al.
##################################
## a=1 is effective version, recommended by the authors
##########################################
def asy_llh(data,dataset_weights , a_prior=1,use_original_code=False):
    """
    data - array of data counts of size J
    all_weights - array of all weights of size N
    weight_indices - list of np arrays of size J (one index array per bin, picks out the wieghts from all_weights)
    """
    n_bins = len(dataset_weights[list(dataset_weights.keys())[0]])

    #
    # Compute the sum of all weights and the sum of the weights squared
    #


    if use_original_code:
        '''
        Calling the python version of the effective log-likelihood
        code, as written by the original authors
        '''
        from .SAY_original_llh import LEff



        # This is a list  containing nset lists of length n_bins
        bincontent_from_alldatasets = list(dataset_weights.values())

        tot_llh = 0.
        for bin_i in range(n_bins):
            allset_weights = [a[bin_i] for a in bincontent_from_alldatasets]

            binwise_weight_sum = np.concatenate(allset_weights).sum()
            binwise_weight_sq_sum = np.square(np.concatenate(allset_weights)).sum()

            if binwise_weight_sum!=0:
                tot_llh += LEff(data[bin_i],binwise_weight_sum,binwise_weight_sq_sum)


    else:
        '''
        Call the python implementation written by Thorsten
        '''


        #Re-arrange the dataset_weights object into the SAY-compatible format
        # Reminder about dataset_weight: a dictionary of lists of np arrays. 
        #                                Each list corresponds to a dataset and 
        #                                contains np arrays with weights for 
        #                                a given bin. empty bins here mean an empty array
        
        all_weights = np.array([])
        weight_indices = [np.array([],dtype=int) for k in range(n_bins)]
        
        n_weights=0
        for _,dataset in list(dataset_weights.items()):
            
            for i,bin_content in zip(list(range(n_bins)),dataset):

                n_elements_in_bin = bin_content.shape[0]
                all_weights = np.append(all_weights,bin_content)
                weight_indices[i] = np.append(weight_indices[i],(n_weights+np.arange(n_elements_in_bin)))
                n_weights+=n_elements_in_bin



        tot_llh=0.0
        for ind, cur_weight_mask in enumerate(weight_indices):

            cur_weights=all_weights[cur_weight_mask]
            weight_sum=cur_weights.sum()
            sqr_weight_sum=(cur_weights**2).sum()

            alpha=(weight_sum**2)/sqr_weight_sum+a_prior
            beta=weight_sum/sqr_weight_sum
            tot_llh+= (scipy.special.gammaln((data[ind]+alpha)) -scipy.special.gammaln(data[ind]+1.0)-scipy.special.gammaln(alpha) + (alpha)* np.log(beta) - (alpha+data[ind])*np.log(1.0+beta)).sum()

    return tot_llh    


##########################
###### Barlow/Beeston (https://www.sciencedirect.com/science/article/pii/001046559390005W) (without const terms)
#########################

def barlow_beeston_llh(data, dataset_weights):
    """
    data - array of data counts of size J
    all_weights - array of all weights of size N
    weight_indices - list of np arrays of size J (one index array per bin, picks out the wieghts from all_weights
    """


    # Re-arrange the dataset_weights object into the barlow-compatible format
    # Reminder about dataset_weight: a dictionary of lists of numpy arrays. 
    #                                Each list corresponds to a dataset and 
    #                                contains numpy arrays with weights for 
    #                                a given bin. empty bins here mean an empty array

    
    n_bins = len(dataset_weights[list(dataset_weights.keys())[0]])

    weights_dict = {}
    indices_dict = {}
    
    for setname,dataset in list(dataset_weights.items()):
        n_weights =0

        set_weights = np.array([])
        set_weight_indices = [np.array([],dtype=int) for k in range(n_bins)]

        for i,bin_content in zip(list(range(n_bins)),dataset):

            n_elements_in_bin = bin_content.shape[0]
            set_weights = np.append(set_weights,bin_content)
            set_weight_indices[i] = np.append(set_weight_indices[i],(n_weights+np.arange(n_elements_in_bin)))
            n_weights+=n_elements_in_bin

        weights_dict[setname] = set_weights
        indices_dict[setname] = set_weight_indices


    def func(x, w, d):
        #print "x ", x
        #print "w ", w
        #print "d ", d
        """
        Reweighting function: 1/sum(w/(1+xw))
        The function should be equal to (1-x)/N_exp
        for reweighting variable x. Note, that w is an array
        since it is the (i,j) entry of w_hist.
        """
        return 1./np.sum(w/(1. + x*w)) - (1. - x)/d


    if data.ndim == 1:
        # array of reweighting factors

        new_avg_weights=[]
        new_kmcs=[]

        ## ind goes over all bins
        
        for ind in np.arange(len(list(indices_dict.values())[0])):
            these_weights=[]

            for src_key in indices_dict.keys():
                #print all_weights_src_list[ind_src][weight_indices_src_list[ind_src][ind]]
                
                if(len(indices_dict[src_key][ind])>0):
                    these_src_weights=weights_dict[src_key][indices_dict[src_key][ind]]

                    these_weights.extend([np.mean(these_src_weights)]*len(these_src_weights))
            
            new_avg_weights.append(np.array(these_weights))
            

        lagrange = np.array([(scipy.optimize.brentq(func, -0.999999/max(w), 1., args=(w,d), full_output=False)\
                    if d else 1.) if (len(w)>0) else 0. for (d, w) in zip(data, new_avg_weights)])


        # llh with new weights
        llh = np.array([np.sum((np.log(1. + lagrange[i]*w))) if(len(w)>0) else 0 for (i,w) in enumerate(new_avg_weights)])\
              + data * np.log(1.-(lagrange-(lagrange == 1.)))


    else:
        raise NotImplementedError("`data` has more than 1 dimensions.")


    return -llh.sum()
        


##############################################################################
### Chirkin (https://arxiv.org/abs/1304.0735)
###########################################################################################


def chirkin_llh(data, dataset_weights):
    """
    data - array of data counts of size J
    all_weights - array of all weights of size N
    weight_indices - list of np arrays of size J (one index array per bin, picks out the wieghts from all_weights)
    """
    #def chirkin_llh(data, all_weights, weight_indices):


    # Re-arrange the dataset_weights object into the chirkin-compatible format
    # Reminder about dataset_weight: a dictionary of lists of numpy arrays. 
    #                                Each list corresponds to a dataset and 
    #                                contains numpy arrays with weights for 
    #                                a given bin. empty bins here mean an empty array

    n_bins = len(dataset_weights[list(dataset_weights.keys())[0]])
    all_weights = np.array([])
    weight_indices = [np.array([],dtype=int) for k in range(n_bins)]
    
    n_weights=0
    for setname,dataset in list(dataset_weights.items()):
        
        for i,bin_content in zip(list(range(n_bins)),dataset):


            n_elements_in_bin = bin_content.shape[0]
            all_weights = np.append(all_weights,bin_content)
            weight_indices[i] = np.append(weight_indices[i],(n_weights+np.arange(n_elements_in_bin)))
            n_weights+=n_elements_in_bin


    
    
    def func(x, w, d):
        """
        Reweighting function: 1/sum(w/(1+xw))
        The function should be equal to (1-x)/N_exp
        for reweighting variable x. Note, that w is an array
        since it is the (i,j) entry of w_hist.
        """
        return 1./np.sum(w/(1. + x*w)) - (1. - x)/d

    if data.ndim == 1:

        # array of reweighting factors
        lagrange = np.array([(scipy.optimize.brentq(func, -0.999999/max(all_weights[w]), 1., args=(all_weights[w],d), full_output=False)\
                    if d else 1.) if (len(w)>0) else 0. for (d, w) in zip(data, weight_indices)])
        # llh with new weights
        #print "dima lagrange ", lagrange
        llh = np.array([np.sum(np.log(1. + lagrange[i]*all_weights[w])) if(len(w)>0) else 0 for (i,w) in enumerate(weight_indices)])\
              + data * np.log(1.-(lagrange-(lagrange == 1.)))

    else:
        raise NotImplementedError("`data` has more than 1 dimensions.")

    return -llh.sum()



