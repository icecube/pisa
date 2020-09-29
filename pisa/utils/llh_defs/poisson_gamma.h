
void generalized_pg_mixture(int k, double *alphas, double *betas, size_t w_size, double *result);

//void generalized_pg_mixture_marginalized(int k,double *new_alphas,  double *betas,  double *gammas, size_t w_size, double *result);
void generalized_pg_mixture_marginalized(int k, double *gammas, double *deltas,  double *epsilons , size_t w_size, double *result);
void generalized_pg_mixture_marginalized_combined(int k,double *new_alphas,  double *betas,  double *gammas, double *alphas_2, double *betas_2, size_t w_size,size_t w_size_2, double *result);

double log_sum_exp(double arr[], int count);

void single_pgg(int k, double A, double B, double Q, double kmc, double gamma, double *log_sterlings, int sterling_size, double *res);  
void multi_pgg(int k, double *A, double *B, double *Q, double *kmc, double *gamma, int nsources, double *log_sterlings, int sterling_size, double *res);
