'''Numba tools

This is a colection of functions used for numba functions
that work for targets cpu as well as cuda
'''
from __future__ import print_function

__all__ = ['myjit',
           'conjugate_transpose',
           'conjugate',
           'matrix_dot_matrix',
           'matrix_dot_vector',
           'clear_matrix',
           'copy_matrix',
           'target',
           'cuda',
           'ctype',
           'ftype'
           ]
__version__ = '0.1'
__author__ = 'Philipp Eller (pde3@psu.edu)'


import numpy as np
import inspect
from numba import jit, float64, complex64, int32, float32, complex128
import math, cmath

from pisa import FTYPE

target='cuda'
#target='parallel'
#target='cpu'

if target == 'cuda':
    from numba import cuda
    if FTYPE == np.float64:
        ctype = complex128
        ftype = float64
    elif FTYPE == np.float32:
        ctype = complex64
        ftype = float32
else:
    if FTYPE == np.float64:
        ctype = np.complex128
        ftype = np.float64
    elif FTYPE == np.float32:
        ctype = np.complex64
        ftype = np.float32
    cuda = lambda: None
    cuda.jit = lambda x: x

def myjit(f):
    '''
    f : function

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
        assert '@myjit' in source[0]
        source = '\n'.join(source[1:]) + '\n'
        source = source.replace('cuda.local.array', 'np.empty')
        exec(source)
        fun = eval(f.__name__)
        newfun = jit(fun, nopython=True)
        # needs to be exported to globals
        globals()[f.__name__] = newfun
        return newfun

@myjit
def conjugate_transpose(A, B):
    '''
    A : 2d array
    B : 2d array

    B is the conjugate transpose of A
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i,j] = A[j,i].conjugate()

@myjit
def conjugate(A, B):
    '''
    A : 2d array
    B : 2d array

    B is the conjugate of A
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i,j] = A[i,j].conjugate()

@myjit
def matrix_dot_matrix(A, B, C):
    '''
    dot-product of two 2d arrays
    C = A * B
    '''
    for j in range(B.shape[1]):
        for i in range(A.shape[0]):
            C[i,j] = 0.
            for n in range(C.shape[0]):
                C[i,j] += A[i,n] * B[n,j]

def test_matrix_dot_matrix():
    A = np.linspace(1., 8., 9).reshape(3,3)
    B = np.linspace(1., 8., 9).reshape(3,3)
    C = np.zeros((3,3))
    matrix_dot_matrix(A, B, C)
    assert np.array_equal(C, np.dot(A, B))

@myjit
def matrix_dot_vector(A, v, w):
    '''
    dot-product of a 2d array and a vector
    w = A * v
    '''
    for i in range(A.shape[0]):
        w[i] = 0.
        for j in range(A.shape[1]):
            w[i] += A[i,j] * v[j]

def test_matrix_dot_vector():
    A = np.linspace(1., 8., 9).reshape(3,3)
    v = np.linspace(1., 3., 3)
    w = np.zeros((3))
    matrix_dot_vector(A, v, w)
    assert np.array_equal(w, np.dot(A, v))

@myjit
def clear_matrix(A):
    '''
    clear out 2d array
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i,j] = 0.

def test_clear_matrix():
    A = np.ones((3,3))
    clear_matrix(A)
    assert np.array_equal(A, np.zeros((3,3)))

@myjit
def copy_matrix(A, B):
    '''
    copy elemnts of 2d array A to array B
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i,j] = A[i,j]

def test_copy_matrix():
    A = np.ones((3,3))
    B = np.zeros((3,3))
    copy_matrix(A, B)
    assert np.array_equal(A, B)



if __name__=='__main__':
    
    assert target == 'cpu', "Cannot test functions on GPU, set target='cpu'"
    test_matrix_dot_matrix()
    test_matrix_dot_vector()
    test_clear_matrix()
    test_copy_matrix()
