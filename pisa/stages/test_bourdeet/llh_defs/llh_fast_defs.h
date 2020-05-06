
void R_n(int n, double *bs, double *z, size_t w_size,  double *result);


void pg_general_iterative_mult(int k, double *weights, size_t w_size, double alpha, double *result);
void pg_general_jac(int k, double *weights, size_t w_size, double alpha, double *result);

void pg_say(int k, double *weights, size_t w_size, double *result);
void standard_poisson(int k, double *weights, size_t w_size, double *result);


void pg_general(int k, double *weights, size_t w_size, double alpha, double extra_prior_counter, double extra_prior_weight, double *result);
void pg_general_equivalent(int k, double *weights, size_t w_size, double alpha, double extra_prior_counter, double extra_prior_weight, double *result);

void pgpg_general(int k, double *weights, size_t w_size, double alpha, double beta, double extra_prior_counter, double extra_prior_weight, double *result);
void pgpg_absolute_general(int k, double *weights, size_t w_size, double alpha, double beta, double v, double extra_prior_counter, double extra_prior_weight,  double *result);
void pgpg_absolute_general_equivalent(int k, double *weights, size_t w_size, double alpha, double beta, double v, double extra_prior_counter,double extra_prior_weight, double *result);