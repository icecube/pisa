import numpy as np
import time

from numba_tools import *
from numba import guvectorize, SmartArray

@myjit
def sum_row_kernel(mix, bla, inp, out):
    C = cuda.local.array(shape=(3,3), dtype=ftype)
    D = cuda.local.array(shape=(3), dtype=ctype)
    E = cuda.local.array(shape=(3), dtype=ctype)
    matrix_dot_matrix(mix, mix, C)
    D[0] = 0.+2.j
    D[1] = 1.+2.j
    D[2] = 1.+2.j
    matrix_dot_vector(C,D,E) 
    bla *= 0.1
    out[0] += E[1].real * bla.real + inp[0]

@guvectorize(['void(float64[:,:], complex128, int32[:], int32[:])'], '(a,b),(),(f)->()', target=target)
def sum_row(mix, bla, inp, out):
    sum_row_kernel(mix, bla, inp, out)

print 'ftype=',ftype

# hist arrays
mix = np.ones((3,3), dtype=np.float64)
n = 1000000
inp = np.arange(3*n, dtype=np.int32).reshape(n, 3)
out = np.ones((n), dtype=np.int32)

inp = SmartArray(inp)
out = SmartArray(out)

if target == 'cuda':
    where='gpu'
else:
    where='host'

start_t = time.time()
sum_row(mix, 42.+2j, inp.get(where), out=out.get(where))
end_t = time.time()
print 'took %.5f'%(end_t - start_t)
start_t = time.time()
sum_row(mix, 42.+2j, inp.get(where), out=out.get(where))
end_t = time.time()
print 'took %.5f'%(end_t - start_t)
out.mark_changed(where)

print out.get('host')
