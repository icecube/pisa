#include <string.h>
#include <stdio.h>
#include <math.h>
#include "poisson_gamma.h"


/* eq. 91 - generalized mixture from standard Poisson-Gamma mixtures */
void generalized_pg_mixture(int k, double *alphas, double *betas, size_t w_size, double *result)
{
    int i=0,j=0;

    double first_var_vec[w_size];

    double deltas[k+2], sum_terms[k+1];
    deltas[0]=1.0;

    double prefac=1.0;
    double running_vec[w_size];

    for (i=0; i < w_size;i++)
    {   
        first_var_vec[i]=1.0/(1.0+betas[i]);
        prefac*=pow((1.0/(1.0+1.0/betas[i])), alphas[i]);
        running_vec[i]=1.0;
    }

    if(k>0)
    {
        for(i=1; i<k+1; i++)
        {
            sum_terms[i]=0.0;

            for(j=0; j<w_size;j++)
            {
                running_vec[j]*=first_var_vec[j];
                sum_terms[i]+=alphas[j]*running_vec[j];
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

/* eq. 96 - generalized mixture without the standard Poisson-Gamma part */

void generalized_pg_mixture_marginalized(int k,double *gammas,  double *deltas,  double *epsilons, size_t w_size, double *result)
{

    int i=0,j=0;

    double E=0, o_o_m_c=0;

    double first_var_vec[w_size];
    double sec_var_vec[w_size];

    double Ds[k+2], sum_terms[k+1];
    Ds[0]=1.0;

    double prefac=1.0;

    double running_first_vec[w_size];
    double running_second_vec[w_size];


    for (i=0; i < w_size;i++)
    {   
        E=1.0/(1.0+gammas[i]);
        o_o_m_c=(1.0+gammas[i]/(1.0+epsilons[i]*(1.0+gammas[i])));

        prefac*=pow( (1.0/(1.0+1.0/epsilons[i]))*o_o_m_c, deltas[i]);

        first_var_vec[i]=E*o_o_m_c;
        sec_var_vec[i]=E;
        running_first_vec[i]=1.0;
        running_second_vec[i]=1.0;
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
                sum_terms[i]+=deltas[j]*running_first_vec[j] - deltas[j]*running_second_vec[j];
                //printf("obp, %f , obp ma , %f, firstvarvec: %f, secvarvec: %f\n", o_p_b, o_p_b_minus_a, first_var_vec[j], sec_var_vec[j]);
            }
          
            Ds[i]=0.0;
            for(j=1;j<=i; j++)
            {
                Ds[i]+=sum_terms[j]*Ds[i-j];
                //printf("SUMMING IN DELTAS ... , %f, %f\n", sum_terms[j], deltas[i-j]);
            }
            Ds[i]/=(double)(i);

            
        }
        
    }
    *result=prefac*Ds[k];
    
}


/* eq. 96 - generalized mixture */
void generalized_pg_mixture_marginalized_combined(int k,double *new_alphas,  double *betas,  double *gammas, double *alphas_2, double *betas_2, size_t w_size,size_t w_size_2, double *result)
{

    int i=0,j=0;

    double E=0, o_o_m_c=0;

    double first_var_vec[w_size];
    double sec_var_vec[w_size];
    double old_var_vec[w_size_2];

    double deltas[k+2], sum_terms[k+1];
    deltas[0]=1.0;

    double prefac=1.0;

    double running_first_vec[w_size];
    double running_second_vec[w_size];

    double running_old_vec[w_size_2];

    for (i=0; i < w_size;i++)
    {   
        E=1.0/(1.0+betas[i]);
        o_o_m_c=(1.0+betas[i]/(1.0+gammas[i]*(1.0+betas[i])));

        prefac*=pow( (1.0/(1.0+1.0/gammas[i]))*o_o_m_c, new_alphas[i]);

        first_var_vec[i]=E*o_o_m_c;
        sec_var_vec[i]=E;
    
        running_first_vec[i]=1.0;
        running_second_vec[i]=1.0;
        
    }

    for(i=0; i < w_size_2;i++)
    {
        prefac*=pow( 1.0/(1.0+1.0/betas_2[i]), alphas_2[i]);
        old_var_vec[i]=1.0/(1.0+betas_2[i]);
        running_old_vec[i]=1.0;
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
                
                sum_terms[i]+=new_alphas[j]*running_first_vec[j] - new_alphas[j]*running_second_vec[j];
                //printf("obp, %f , obp ma , %f, firstvarvec: %f, secvarvec: %f\n", o_p_b, o_p_b_minus_a, first_var_vec[j], sec_var_vec[j]);
            }
            for(j=0; j<w_size_2;j++)
            {
                running_old_vec[j]*=old_var_vec[j];
                sum_terms[i]+=alphas_2[j]*running_old_vec[j];
            }
          
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

double log_sum_exp(double arr[], int count) 
{
   int i;
   if(count > 0 ){
      double maxVal = arr[0];
      double sum = 0;

      for (i = 1 ; i < count ; i++){
         if (arr[i] > maxVal){
            maxVal = arr[i];
         }
      }

      for (i = 0; i < count ; i++){
         sum += exp(arr[i] - maxVal);
      }
      return log(sum) + maxVal;

   }
   else
   {
      return 0.0;
   }
}

/* eq 85 (generalization (3)) for single source dataset */
void single_pgg(int k, double A, double B, double Q, double kmc, double gamma, double *log_sterlings, int sterling_size, double *res)
{   
    double log_log_factor=log(1.0/(gamma-Q*log(A)));
    double log_Q=log(Q);
    double prefac=kmc*log_log_factor-lgamma((double)k+1)-lgamma(kmc)+kmc*log(gamma)+k*log(B);
    int i;
    double sumvec[k+1];

    for(i=0; i<=k; i++)
    {
        sumvec[i]=lgamma(kmc+i)+i*(log_Q+log_log_factor)+log_sterlings[k*sterling_size+i];
    }

    *res=log_sum_exp(sumvec, k+1)+prefac;
}



double log_sum_two_vecs_inverse(double *a, double *b, int len)
{

    double maxVal = a[0]+b[len-1];
    double sum = 0;
    double temp[len];
    int i;
    temp[0]=maxVal;

    for (i = 1 ; i < len ; i++){
        temp[i]=a[i]+b[len-1-i];
        if ( temp[i] > maxVal)
        {
            maxVal = temp[i];
        }
    }

    for (i = 0; i < len ; i++)
    {
        sum += exp(temp[i] - maxVal);
    }
    return log(sum) + maxVal;
}

void convolve_two(double *a,double *b, double *res, int len)
{
    // len should be k+1

    int i;
    double temp[len];

    // i goes from 0 to k
    for(i=0; i < len; i++)
    {  
        temp[i]=log_sum_two_vecs_inverse(a,b,i+1);
    }
    for(i=0; i<len;i++)
    {
        res[i]=temp[i];
    }
}


/* calculate generalization (3) based on convolution of individual PGG distriutions, involves logsumexp for numerical stability 
    eq. (51)
*/
void multi_pgg(int k, double *A, double *B, double *Q, double *kmc, double *gamma, int nsources, double *log_sterlings, int sterling_size, double *res)
{   

    double log_Q, log_log_factor,prefac,running_log_B,running,log_B,running_log_factorial;
   
    int i,j,z,cur_k,tmp;

    double sumvec[k+1];
    
    double intermediate_results[nsources][k+1];
    double precalculated_results[nsources][k+1];
    double temp_conv_vec[k+1];

    double log_ks[k+1];


    for(i=0; i<nsources; i++)
    {
        log_log_factor=log(1.0/(gamma[i]-Q[i]*log(A[i])));
        log_Q=log(Q[i]);
        log_B=log(B[i]);
        prefac=kmc[i]*log_log_factor+kmc[i]*log(gamma[i]);

        running_log_B=0.0;
        running_log_factorial=0.0;
       
        
        precalculated_results[i][0]=prefac;
        intermediate_results[i][0]=precalculated_results[i][0];

        if(i==0)
        {
            temp_conv_vec[0]=intermediate_results[i][0];
        }

        for(j=1; j<=k;j++)
        {
            precalculated_results[i][j]=precalculated_results[i][j-1]+log(kmc[i]+(float)j-1.0)+ log_Q+log_log_factor; 
            if(i==0)
            {
                log_ks[j]=log(j);
            }  

            running_log_B+=log_B;
            running_log_factorial+=log_ks[j];
            tmp=j*sterling_size;

            for(z=0;z<=j;z++)
            {
                sumvec[z]=precalculated_results[i][z]+log_sterlings[tmp+z];
            }

            intermediate_results[i][j]=log_sum_exp(sumvec, j+1)+running_log_B-running_log_factorial;
            if(i==0)
            {
                temp_conv_vec[j]=intermediate_results[i][j];
            }


        }
    } 

    if(nsources==1)
    {
        *res=temp_conv_vec[k];
        return;
    }

    for(i=1; i<nsources-1;i++)
    {
        convolve_two(temp_conv_vec,intermediate_results[i],temp_conv_vec, k+1);
    }


    *res=log_sum_two_vecs_inverse(temp_conv_vec, intermediate_results[nsources-1], k+1);

}








