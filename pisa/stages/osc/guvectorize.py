from __future__ import print_function
from numba import jit, vectorize, guvectorize, float64, complex64, int32, float32
import numpy as np

@jit(int32(complex64, int32))
def mandelbrot(c,maxiter):
    nreal = 0
    real = 0
    imag = 0
    for n in range(maxiter):
        nreal = real*real - imag*imag + c.real
        imag = 2* real*imag + c.imag
        real = nreal;
        if real * real + imag * imag > 4.0:
            return n
    return 0

@guvectorize([(complex64[:], int32[:], int32[:])], '(n),()->(n)',target='parallel')
def mandelbrot_numpy(c, maxit, output):
    maxiter = maxit[0]
    for i in range(c.shape[0]):
        output[i] = mandelbrot(c[i],maxiter)
        
def mandelbrot_set2(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    n3 = mandelbrot_numpy(c,maxiter)
    return (r1,r2,n3.T) 


@guvectorize(['void(float64[:,:], float64, int32[:], int32[:])'], '(a,b),(),(f)->()', nopython=False)
def sum_row(mix, bla, inp, out):
    tmp = 0.
    for i in range(inp.shape[0]):
        tmp += inp[i]
    print(mix, bla, inp, out)
    #out = tmp * bla
    out[0] = tmp * bla
    #out = tmp

mix = np.ones((3,3))
inp = np.arange(30, dtype=np.int32).reshape(10, 3)
out = np.empty(10, dtype=np.int32)
sum_row(mix, 42., inp, out)
print(out)
