import numpy as np
import time

from numba_tools import *
dtype = float64
Ndim=3

@myjit
def sum_row_kernel(mix, bla, inp, out):
    C = cuda.local.array(shape=(Ndim,Ndim), dtype=dtype)
    #C = cuda.local.array(shape=(3,3), dtype=mix.dtype)
    zero(C)
    #C = mix + mix
    dot(mix, mix, C)
    bla *= 0.1
    out[0] = C[1,1] * bla.real

@guvectorize(['void(float64[:,:], complex128, int32[:], int32[:])'], '(a,b),(),(f)->()', target=target)
def sum_row(mix, bla, inp, out):
    sum_row_kernel(mix, bla, inp, out)

mix = np.ones((3,3), dtype=np.float64)
n = 10000000
inp = np.arange(3*n, dtype=np.int32).reshape(n, 3)
out = np.empty((n), dtype=np.int32)
start_t = time.time()
sum_row(mix, 42.+2j, inp, out=out)
end_t = time.time()
print 'took %.5f'%(end_t - start_t)
print out
