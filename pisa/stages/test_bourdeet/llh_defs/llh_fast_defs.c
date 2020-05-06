#include <string.h>
#include <stdio.h>
#include <math.h>
#include "poisson_gamma.h"

void R_n(int n, double *bs, double *z, size_t w_size, double *result)
{

    int i=0,j=0;

    double deltas[n+2], sum_terms[n+1];
    deltas[0]=1.0;
    double bsum=0.0;
    for(i = 0; i < w_size; i++)
    {
        bsum+=bs[i];
    }

    if(n>0)
    {
        for(i=1; i<n+1; i++)
        {
            sum_terms[i]=0.0;

            for(j=0; j<w_size;j++)
            {
                sum_terms[i]+=bs[j]*pow(z[j], i);
            }

            deltas[i]=0.0;
            for(j=1;j<=i; j++)
            {
                deltas[i]+=sum_terms[j]*deltas[i-j];
            }
            deltas[i]/=(double)(i);
        }      
    }

    double log_gamma=lgamma((double)(n+1))+lgamma(bsum)-lgamma(bsum+(double)n);
    *result=deltas[n]*exp(log_gamma);
}



// general generalization of the poisson_gamma mixture
void pg_general_iterative_mult(int k, double *weights, size_t w_size, double alpha, double *result)
{

    int i=0,j=0;

    
    double alpha_individual=1.0+alpha/(double)w_size;
    
    double first_var_vec[w_size], powers_of_z[w_size];

    double deltas[k+2], sum_terms[k+1];
    deltas[0]=1.0;

    double prefac=1.0;

    for (i=0; i < w_size;i++)
    {   
        first_var_vec[i]=1.0/(1.0+1.0/weights[i]);
        powers_of_z[i]=first_var_vec[i];
        prefac*=pow((1.0/(1.0+weights[i])), alpha_individual);

    }
    
    if(k>0)
    {
        for(i=1; i<k+1; i++)
        {
            sum_terms[i]=0.0;

            for(j=0; j<w_size;j++)
            {
                sum_terms[i]+=(alpha_individual)* powers_of_z[j];
                powers_of_z[j]*=first_var_vec[j];
            }


            deltas[i]=0.0;
            for(j=1;j<=i; j++)
            {
                deltas[i]+=sum_terms[j]*deltas[i-j];
            }
            deltas[i]/=(double)(i);

        }
        
    }
    *result=prefac*deltas[k];
    
}

// calculates jacobian of pg_general w.r.t individual weights

void pg_general_jac(int k, double *weights, size_t w_size, double alpha, double *result)
{

    int i,j,l;
    
    double alpha_individual=1.0+alpha/(double)w_size;
    
    double first_var_vec[w_size], powers_of_z[w_size];

    // we have w_size terms for the derivatives w.r.t. weights
    double deriv_sum_terms[(int)w_size][k];
    double deriv_deltas[(int)w_size][k+1];
    // TODO: ( the first entry of the following arrays is never used, as the loop
    // starts at i=1 later ..  could probably be changed to save some temporary bytes), but this form emulates the paper formula
    double deltas[k+2], sum_terms[k+1];// delats = 1 .. k+1 entries .. sum_terms = 1 ... k entries 

    
    memset(deriv_sum_terms, 0, sizeof(deriv_sum_terms));
    memset(deriv_deltas, 0, sizeof(deriv_deltas));
    memset(deltas, 0, sizeof(deltas));
    memset(sum_terms, 0, sizeof(sum_terms));

    deltas[0]=1.0;

    double prefac=1.0;

    for (i=0; i < w_size;i++)
    {   
        first_var_vec[i]=1.0/(1.0+1.0/weights[i]);
        powers_of_z[i]=first_var_vec[i];
        prefac*=pow((1.0/(1.0+weights[i])), alpha_individual);
        deriv_deltas[i][0]=1.0;
    }

    prefac*=alpha_individual;


    if(k>0)
    {
        for(i=1; i<k+1; i++)
        {   

            // also handle derivs
            if(i<k)
            {
                for(j=0; j<w_size;j++)
                {
                    sum_terms[i]+=alpha_individual*powers_of_z[j];
                }

                for(l=0;l<w_size;l++)
                {
                    deriv_sum_terms[l][i]=sum_terms[i]+powers_of_z[l];
                    
                    powers_of_z[l]*=first_var_vec[l];
                }
                
                for(j=1;j<=i; j++)
                {
                    deltas[i]+=sum_terms[j]*deltas[i-j];
                    for(l=0;l<w_size;l++)
                    {
                        deriv_deltas[l][i]+=deriv_sum_terms[l][j]*deriv_deltas[l][i-j];
                    } 
                }
                deltas[i]/=(double)(i);
                for(l=0;l<w_size;l++)
                {
                    deriv_deltas[l][i]/=(double)(i);
                }
            }
            else
            {
                // only std 
                for(j=0; j<w_size;j++)
                {
                    sum_terms[i]+=(alpha_individual)*powers_of_z[j];
                }
                for(j=1;j<=i; j++)
                {
                    deltas[i]+=sum_terms[j]*deltas[i-j];
                }
                deltas[i]/=(double)(i);

            }

        }
        
    }


    double one_over_w;
    
    for(l=0; l < w_size;l++)
    {
        one_over_w=1.0/(1.0+weights[l]);
        result[l]=prefac*one_over_w*(one_over_w*deriv_deltas[l][k-1]-deltas[k]);
    }
    
}
/*
cur_weights=all_weights[cur_weight_mask]
        weight_sum=cur_weights.sum()
        sqr_weight_sum=(cur_weights**2).sum()

        alpha=(weight_sum**2)/sqr_weight_sum
        beta=weight_sum/sqr_weight_sum
        tot_llh+= (scipy.special.gammaln((data[ind]+alpha)) -scipy.special.gammaln(data[ind]+1.0)-scipy.special.gammaln(alpha) + (alpha)* numpy.log(beta) - (alpha+data[ind])*numpy.log(1.0+beta)).sum()
*/
void pg_say(int k, double *weights, size_t w_size, double *result)
{
    double  sumweights=0.0;
    double wsum2=0.0;
    double this_weight;
    int i;

    for(i =0; i < w_size;i++)
    {
        this_weight=weights[i];
        sumweights+=this_weight;
        wsum2+=this_weight*this_weight;
    }

    double alpha=(sumweights*sumweights)/wsum2;
    double beta=sumweights/wsum2;

    *result=lgamma(k+alpha)-lgamma(k+1.0)-lgamma(alpha)+alpha*log(beta)-(alpha+k)*log(1.0+beta);

}

void standard_poisson(int k, double *weights, size_t w_size, double *result)
{
    double  sumweights=0.0;
    int i;

    for(i =0; i < w_size;i++)
    {
        sumweights+=weights[i];
    }

    *result=k*log(sumweights)-sumweights*log(M_E)-(lgamma((double)(k+1)));
}

void pg_general(int k, double *weights, size_t w_size, double alpha, double extra_prior_counter, double extra_prior_weight, double *result)
{

    int i=0,j=0;

    
    double alpha_individual=1.0+alpha/(double)w_size;
    
    double first_var_vec[w_size];

    double deltas[k+2], sum_terms[k+1];
    deltas[0]=1.0;

    double prefac=1.0;

    for (i=0; i < w_size;i++)
    {   
        first_var_vec[i]=1.0/(1.0+1.0/weights[i]);
        prefac*=pow((1.0/(1.0+weights[i])), alpha_individual);
    }
    prefac*=pow( (1.0/(1.0+extra_prior_weight)), extra_prior_counter);

    // predefine to speed up a little bit
    double extra_prior_weight_factor=1.0/(1.0+1.0/extra_prior_weight);

    double running_vec[w_size];
    double running_prior=1.0;

    for(j=0; j<w_size;j++)
    {
        running_vec[j]=1.0;
    }

    if(k>0)
    {
        for(i=1; i<k+1; i++)
        {
            sum_terms[i]=0.0;

            for(j=0; j<w_size;j++)
            {
                running_vec[j]*=first_var_vec[j];
                sum_terms[i]+=(alpha_individual)*running_vec[j];
            }

            running_prior*=extra_prior_weight_factor;
            sum_terms[i]+= extra_prior_counter*running_prior; 

            deltas[i]=0.0;
            for(j=1;j<=i; j++)
            {
                deltas[i]+=sum_terms[j]*deltas[i-j];
                //printf("SUMMING IN DELTAS ... , %f, %f\n", sum_terms[j], deltas[i-j]);
            }
            deltas[i]/=(double)(i);

            //printf("i - %d, %f", i, deltas[i]);

            
        }
        
    }
    *result=prefac*deltas[k];
    
}


void pg_general_equivalent(int k, double *weights, size_t w_size, double alpha, double extra_prior_counter, double extra_prior_weight, double *result)
{

    size_t new_size=w_size+1;
    double effective_alpha[new_size];
    double effective_beta[new_size];

    double alpha_individual=1.0+alpha/(double)(w_size);
    int i;

    for(i =0; i < w_size; i++)
    {
        effective_alpha[i]=alpha_individual;
        effective_beta[i]=1.0/weights[i];
    }
    effective_alpha[w_size]=extra_prior_counter;
    effective_beta[w_size]=1.0/extra_prior_weight;

    generalized_pg_mixture(k, effective_alpha,  effective_beta, new_size, result);
}


void pgpg_general(int k, double *weights, size_t w_size, double alpha, double beta, double extra_prior_counter,double extra_prior_weight, double *result)
{

    int i=0,j=0;

    double E=0, o_o_m_c=0;

    double alpha_individual=alpha/(double)w_size;
    double beta_individual=beta/(double)w_size;
    double o_p_b=1.0+beta_individual;
    double o_p_b_minus_a=o_p_b-alpha_individual;
    double first_var_vec[w_size];
    double sec_var_vec[w_size];

    double deltas[k+2], sum_terms[k+1];
    deltas[0]=1.0;

    double prefac=1.0;

    for (i=0; i < w_size;i++)
    {   
        E=1.0/(1.0+1.0/weights[i]);
        o_o_m_c=(2.0+2.0*weights[i])/(1.0+2.0*weights[i]);

        prefac*=pow((1.0/(1.0+weights[i])), alpha_individual);
        prefac*=pow( 0.5*o_o_m_c, o_p_b);

        first_var_vec[i]=E*o_o_m_c;
        sec_var_vec[i]=E;

    }
    prefac*=pow( (1.0/(1.0+extra_prior_weight)), extra_prior_counter);

    // predefine to speed up a little bit
    double extra_prior_weight_factor=1.0/(1.0+1.0/extra_prior_weight);

    // optimization for powers .. 

    double running_first_vec[w_size];
    double running_second_vec[w_size];
    double running_prior=1.0;

    for(j=0; j<w_size;j++)
    {
        running_first_vec[j]=1.0;
        running_second_vec[j]=1.0;
    }

    if(k>0)
    {
        for(i=1; i<k+1; i++)
        {
            sum_terms[i]=0.0;

            for(j=0; j<w_size;j++)
            {
                running_first_vec[j]*=first_var_vec[j];
                running_second_vec[j]*=sec_var_vec[j];
                sum_terms[i]+=(o_p_b)* running_first_vec[j] - (o_p_b_minus_a)*running_second_vec[j];
                //printf("obp, %f , obp ma , %f, firstvarvec: %f, secvarvec: %f\n", o_p_b, o_p_b_minus_a, first_var_vec[j], sec_var_vec[j]);
            }
            running_prior*=extra_prior_weight_factor;

            sum_terms[i]+= extra_prior_counter*running_prior; 
            //printf("sum terms %d - %f\n", i, sum_terms[i]);

            deltas[i]=0.0;
            for(j=1;j<=i; j++)
            {
                deltas[i]+=sum_terms[j]*deltas[i-j];
                //printf("SUMMING IN DELTAS ... , %f, %f\n", sum_terms[j], deltas[i-j]);
            }
            deltas[i]/=(double)(i);

            
        }
        
    }
    *result=prefac*deltas[k];
    
}

void pgpg_absolute_general(int k, double *weights, size_t w_size, double alpha, double beta, double v, double extra_prior_counter,double extra_prior_weight, double *result)
{

    int i=0,j=0;

    double E=0, o_o_m_c=0;

    double alpha_individual=alpha/(double)w_size;
    double beta_individual=beta/(double)w_size;
    double o_p_b=1.0+beta_individual;
    double o_p_b_minus_a=o_p_b-alpha_individual;
    double first_var_vec[w_size];
    double sec_var_vec[w_size];

    double deltas[k+2], sum_terms[k+1];
    deltas[0]=1.0;

    double prefac=1.0;

    double gamma_prefac=1.0/(1.0+v);

    for (i=0; i < w_size;i++)
    {   
        E=1.0/(1.0+1.0/weights[i]);
        o_o_m_c=(1.0+v/(1.0+weights[i]*(1.0+v)));

        prefac*=pow((1.0/(1.0+weights[i])), alpha_individual);
        prefac*=pow( gamma_prefac*o_o_m_c, o_p_b);

        first_var_vec[i]=E*o_o_m_c;
        sec_var_vec[i]=E;

    }
    prefac*=pow( (1.0/(1.0+extra_prior_weight)), extra_prior_counter);

    // predefine to speed up a little bit
    double extra_prior_weight_factor=1.0/(1.0+1.0/extra_prior_weight);

    // optimization for powers .. 

    double running_first_vec[w_size];
    double running_second_vec[w_size];
    double running_prior=1.0;

    for(j=0; j<w_size;j++)
    {
        running_first_vec[j]=1.0;
        running_second_vec[j]=1.0;
    }

    if(k>0)
    {
        for(i=1; i<k+1; i++)
        {
            sum_terms[i]=0.0;

            for(j=0; j<w_size;j++)
            {
                running_first_vec[j]*=first_var_vec[j];
                running_second_vec[j]*=sec_var_vec[j];
                sum_terms[i]+=(o_p_b)* running_first_vec[j] - (o_p_b_minus_a)*running_second_vec[j];
                //printf("obp, %f , obp ma , %f, firstvarvec: %f, secvarvec: %f\n", o_p_b, o_p_b_minus_a, first_var_vec[j], sec_var_vec[j]);
            }
            running_prior*=extra_prior_weight_factor;

            sum_terms[i]+= extra_prior_counter*running_prior; 
            //printf("sum terms %d - %f\n", i, sum_terms[i]);

            deltas[i]=0.0;
            for(j=1;j<=i; j++)
            {
                deltas[i]+=sum_terms[j]*deltas[i-j];
                //printf("SUMMING IN DELTAS ... , %f, %f\n", sum_terms[j], deltas[i-j]);
            }
            deltas[i]/=(double)(i);

            
        }
        
    }
    *result=prefac*deltas[k];
    
}

void pgpg_absolute_general_equivalent(int k, double *weights, size_t w_size, double alpha, double beta, double v, double extra_prior_counter,double extra_prior_weight, double *result)
{

    size_t new_size=w_size+1;

    // these a the new inputs for the poisson-gamma mixture
    double effective_alpha[w_size];
    double effective_beta[w_size];
    double effective_gamma[w_size];

    double effective_alpha_2[new_size];
    double effective_beta_2[new_size];

    double extra_alpha_individual=alpha/(double)(w_size);
    double extra_beta_individual=beta/(double)(w_size);
    int i;
    
    for(i =0; i < w_size; i++)
    {
        effective_alpha[i]=1.0+extra_beta_individual;
        effective_beta[i]=1.0/weights[i];
        effective_gamma[i]=1.0/v;

        effective_alpha_2[i]=extra_alpha_individual;
        effective_beta_2[i]=1.0/weights[i];
    }

    effective_alpha_2[w_size]=extra_prior_counter;
    effective_beta_2[w_size]=1.0/extra_prior_weight;

    printf("\nhmm %.8f\n", 1.0/extra_prior_weight);

    generalized_pg_mixture_marginalized_combined(k, effective_alpha,  effective_beta, effective_gamma,effective_alpha_2, effective_beta_2, w_size, new_size, result);


}