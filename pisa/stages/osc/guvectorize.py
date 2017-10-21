from __future__ import print_function
from numba import jit, vectorize, guvectorize, float64, complex64, int32, float32
import numpy as np
import time
import inspect

target='cuda'
#target='parallel'
#target='cpu'

if target == 'cuda':
    from numba import cuda
else:
    cuda = lambda: None
    cuda.jit = lambda x: x

def magic(f):
    '''
    Decorator to assign the right jit for different targets
    In case of non-cuda targets, all instances of `cuda.local.array`
    are replaced by `np.empty`. This is a dirty fix, hopefully in the
    near future numba will support numpy array allocation and this will
    not be necessary anymore
    '''
    if target == 'cuda':
        return cuda.jit(f, device=True)
    else:
        source = inspect.getsource(f).splitlines()
        assert source[0] == '@magic'
        source = '\n'.join(source[1:])
        source += '\n'
        source = source.replace('cuda.local.array', 'np.empty')
        exec(source)
        fun = eval(f.__name__)
        return jit(fun, nopython=True)

@magic
def dot(A, B, C):
    for n in range(C.shape[0]):
        for m in range(C.shape[1]):
            for i in range(A.shape[1]):
                for j in range(B.shape[0]):
                    C[n,m] = A[n,i] * B[j,m]

@magic
def sum_row_kernel(mix, bla, inp, out):
    C = cuda.local.array(shape=(3,3), dtype=float64)
    dot(mix, mix, C)
    tmp = 0.
    for i in range(inp.shape[0]):
        tmp += inp[i]
    out[0] = tmp - C[1,2] + 3

@guvectorize(['void(float64[:,:], float64, int32[:], int32[:])'], '(a,b),(),(f)->()', target=target)
def sum_row(mix, bla, inp, out):
    sum_row_kernel(mix, bla, inp, out)

mix = np.ones((3,3), dtype=np.float64)
n = 100000000
inp = np.arange(3*n, dtype=np.int32).reshape(n, 3)
out = np.empty((n), dtype=np.int32)
start_t = time.time()
sum_row(mix, 42., inp, out=out)
end_t = time.time()
print ('took %.5f'%(end_t - start_t))
print(out)

