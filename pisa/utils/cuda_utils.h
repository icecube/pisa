#ifndef __UTILS_H__
#define __UTILS_H__

/*
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}
*/

#ifdef DOUBLE_PRECISION
__device__ double atomicAdd_custom(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hanging in case of nan
    // (nan != nan)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

#ifdef SINGLE_PRECISION
__device__ float atomicAdd_custom(float* address, float val)
{
  return atomicAdd(address, val);
}
#endif


#endif
