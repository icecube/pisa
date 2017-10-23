import numpy as np
import inspect
from numba import jit, vectorize, guvectorize, float64, complex64, int32, float32, complex128

target='cuda'
#target='parallel'
#target='cpu'

if target == 'cuda':
    from numba import cuda
else:
    cuda = lambda: None
    cuda.jit = lambda x: x

def myjit(f):
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
        assert source[0] == '@myjit'
        source = '\n'.join(source[1:])
        source = source.replace('cuda.local.array', 'np.empty')
        exec(source)
        fun = eval(f.__name__)
        return jit(fun, nopython=False)
        return jit(fun, nopython=True)

@myjit
def conjugate_transpose(A, B):
    '''
    B is the conjugate transpose of A
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i,j] = A[j,i].conjugate()

@myjit
def MdotM(A, B, C):
    '''
    dot-product of two 2d arrays
    C = A * B
    '''
    for j in range(B.shape[1]):
        for i in range(A.shape[0]):
            for n in range(C.shape[0]):
                C[i,j] += A[i,n] * B[n,j]

@myjit
def Mdotv(A, v, w):
    '''
    dot-product of a 2d array and a vector
    w = A * v
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            w[j] += A[i,j] * v[j]

@myjit
def zero(A):
    '''
    zero out 2d array
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i,j] = 0.

@myjit
def copy(A, B):
    '''
    copy elemnts of 2d array A to array B
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i,j] = A[i,j]



if __name__ == '__main__':
    import numpy as np
    import time

    @myjit
    def sum_row_kernel(mix, bla, inp, out):
        C = cuda.local.array(shape=(3,3), dtype=float64)
        zero(C)
        dot(mix, mix, C)
        out[0] = C[1,1]

    @guvectorize(['void(float64[:,:], float64, int32[:], int32[:])'], '(a,b),(),(f)->()', target=target, nopython=True)
    def sum_row(mix, bla, inp, out):
        sum_row_kernel(mix, bla, inp, out)

    mix = np.ones((3,3), dtype=np.float64)
    n = 10000000
    inp = np.arange(3*n, dtype=np.int32).reshape(n, 3)
    out = np.empty((n), dtype=np.int32)
    start_t = time.time()
    sum_row(mix, 42., inp, out=out)
    end_t = time.time()
    print 'took %.5f'%(end_t - start_t)

