# pylint: disable=invalid-name, ungrouped-imports

"""
Numba tools

This is a colection of functions used for numba functions
that work for targets cpu as well as cuda
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    "cuda",
    "ctype",
    "ftype",
    "WHERE",
    "myjit",
    "conjugate_transpose",
    "conjugate_transpose_guf",
    "test_conjugate_transpose",
    "conjugate",
    "conjugate_guf",
    "test_conjugate",
    "matrix_dot_matrix",
    "matrix_dot_matrix_guf",
    "test_matrix_dot_matrix",
    "matrix_dot_vector",
    "matrix_dot_vector_guf",
    "test_matrix_dot_vector",
    "clear_matrix",
    "clear_matrix_guf",
    "test_clear_matrix",
    "copy_matrix",
    "copy_matrix_guf",
    "test_copy_matrix",
    "cuda_copy",
]
__version__ = "0.2"
__author__ = "Philipp Eller (pde3@psu.edu)"

from argparse import ArgumentParser
import inspect

# NOTE: Following must be imported to be in the namespace for use by `myjit`
# when re-compiling modified (external) function code
import cmath  # pylint: disable=unused-import
import math  # pylint: disable=unused-import

import numpy as np

import numba

from numba import (  # pylint: disable=unused-import
    complex64,
    complex128,
    float32,
    float64,
    int32,
    int64,
    uint32,
    uint64,
    guvectorize,
    jit,
)

from pisa import FTYPE, TARGET
from pisa.utils.comparisons import ALLCLOSE_KW
from pisa.utils.log import Levels, logging, set_verbosity


if TARGET is None:
    raise NotImplementedError("Numba not supported.")

# the `WHERE` variable is for usage with smart arrays
if TARGET == "cuda":
    from numba import cuda

    if FTYPE == np.float64:
        ctype = complex128
        ftype = float64
    elif FTYPE == np.float32:
        ctype = complex64
        ftype = float32
    WHERE = "gpu"
else:
    if FTYPE == np.float64:
        ctype = np.complex128
        ftype = np.float64
    elif FTYPE == np.float32:
        ctype = np.complex64
        ftype = np.float32
    cuda = lambda: None
    cuda.jit = lambda x: x
    WHERE = "host"


if FTYPE == np.float32:
    FX = "f4"
    CX = "c8"
elif FTYPE == np.float64:
    FX = "f8"
    CX = "c16"


def myjit(func):
    """
    Decorator to assign the right jit for different targets
    In case of non-cuda targets, all instances of `cuda.local.array`
    are replaced by `np.empty`. This is a dirty fix, hopefully in the
    near future numba will support numpy array allocation and this will
    not be necessary anymore

    Parameters
    ----------
    func : callable

    Returns
    -------
    new_nb_func: numba callable
        Refactored version of `func` but with `cuda.local.array` replaced by
        `np.empty` if `TARGET == "cpu"`. For either TARGET, the returned
        function will be callable within numba code for that target.

    """
    # pylint: disable=exec-used, eval-used

    if TARGET == "cuda":
        new_nb_func = cuda.jit(func, device=True)

    else:
        source = inspect.getsource(func).splitlines()
        assert source[0].strip().startswith("@myjit")
        source = "\n".join(source[1:]) + "\n"
        source = source.replace("cuda.local.array", "np.empty")
        exec(source)
        new_py_func = eval(func.__name__)
        new_nb_func = jit(new_py_func, nopython=True)
        # needs to be exported to globals
        globals()[func.__name__] = new_nb_func

    return new_nb_func


# --------------------------------------------------------------------------- #


def cuda_copy(func):
    ''' Handle copying back device array'''
    def wrapper(*args, **kwargs):
        out = kwargs.pop("out")
        if TARGET == "cuda" and not isinstance(out, numba.cuda.devicearray.DeviceNDArray):
            d_out = numba.cuda.to_device(out)
            func(*args, **kwargs, out=d_out)
            d_out.copy_to_host(out)
        else:
            func(*args, **kwargs, out=out)
    return wrapper

@myjit
def conjugate_transpose(A, B):
    """B is the conjugate (Hermitian) transpose of A .. ::

        B[j, i] = A[i, j]*

    A : 2d array of shape (M, N)
    B : 2d array of shape (N, M)

    """
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[j, i] = A[i, j].conjugate()


@guvectorize(
    [f"({XX}[:, :], {XX}[:, :])" for XX in [FX, CX]], "(i, j) -> (j, i)", target=TARGET,
)
def conjugate_transpose_guf(A, out):
    """gufunc that calls conjugate_transpose"""
    conjugate_transpose(A, out)


def test_conjugate_transpose():
    """Unit tests of `conjugate_transpose` and `conjugate_transpose_guf`"""
    A = (np.linspace(1, 12, 12) + 1j * np.linspace(21, 32, 12)).reshape(4, 3).astype(CX)
    d_A = numba.cuda.to_device(A)
    B = np.ones((3, 4), dtype=CX)
    d_B = numba.cuda.to_device(B)

    conjugate_transpose_guf(d_A, d_B)

    test = d_B.copy_to_host()
    ref = A.conj().T
    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"

    A = np.linspace(1, 12, 12, dtype=FX).reshape(3, 4)
    B = np.ones((4, 3), dtype=FX)
    d_A = numba.cuda.to_device(A)
    d_B = numba.cuda.to_device(B)

    conjugate_transpose_guf(d_A, d_B)

    test = d_B.copy_to_host()
    ref = A.conj().T
    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"

    logging.info("<< PASS : test_conjugate_transpose >>")


# --------------------------------------------------------------------------- #


@myjit
def conjugate(A, B):
    """B is the element-by-element conjugate of A .. ::

        B[i, j] = A[i, j]*

    Parameters
    ----------
    A : 2d array
    B : 2d array

    """
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i, j] = A[i, j].conjugate()


@guvectorize(
    [f"({XX}[:, :], {XX}[:, :])" for XX in [FX, CX]], "(i, j) -> (i, j)", target=TARGET,
)
def conjugate_guf(A, out):
    """gufunc that calls `conjugate`"""
    conjugate(A, out)


def test_conjugate():
    """Unit tests of `conjugate` and `conjugate_guf`"""
    A = numba.cuda.to_device(
        (np.linspace(1, 12, 12) + 1j * np.linspace(21, 32, 12)).reshape(4, 3).astype(CX)
    )
    B = numba.cuda.to_device(np.ones((4, 3), dtype=CX))

    conjugate_guf(A, B)

    test = B.copy_to_host()
    ref = A.copy_to_host().conj()

    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"

    A = numba.cuda.to_device(np.linspace(1, 12, 12, dtype=FX).reshape(3, 4))
    B = numba.cuda.to_device(np.ones((3, 4), dtype=FX))

    conjugate_guf(A, B)

    test = B.copy_to_host()
    ref = A.copy_to_host().conj()
    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"

    logging.info("<< PASS : test_conjugate >>")


# --------------------------------------------------------------------------- #


@myjit
def matrix_dot_matrix(A, B, C):
    """Dot-product of two 2d arrays .. ::

        C = A * B

    """
    for j in range(B.shape[1]):
        for i in range(A.shape[0]):
            C[i, j] = 0.0
            for n in range(B.shape[0]):
                C[i, j] += A[i, n] * B[n, j]


@guvectorize(
    [f"({XX}[:, :], {XX}[:, :], {XX}[:, :])" for XX in [FX, CX]],
    "(i, n), (n, j) -> (i, j)",
    target=TARGET,
)
def matrix_dot_matrix_guf(A, B, out):
    """gufunc that calls matrix_dot_matrix"""
    matrix_dot_matrix(A, B, out)


def test_matrix_dot_matrix():
    """Unit tests of `matrix_dot_matrix` and `matrix_dot_matrix_guf`"""
    A = numba.cuda.to_device(np.linspace(1, 12, 12, dtype=FTYPE).reshape(3, 4))
    B = numba.cuda.to_device(np.linspace(1, 12, 12, dtype=FTYPE).reshape(4, 3))
    C = numba.cuda.to_device(np.ones((3, 3), dtype=FTYPE))

    matrix_dot_matrix_guf(A, B, C)

    test = C.copy_to_host()
    ref = np.dot(A.copy_to_host(), B.copy_to_host()).astype(FTYPE)
    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"

    logging.info("<< PASS : test_matrix_dot_matrix >>")


# --------------------------------------------------------------------------- #


@myjit
def matrix_dot_vector(A, v, w):
    """Dot-product of a 2d array and a vector .. ::

        w = A * v

    """
    for i in range(A.shape[0]):
        w[i] = 0.0
        for j in range(A.shape[1]):
            w[i] += A[i, j] * v[j]


@guvectorize(
    [f"({XX}[:, :], {XX}[:], {XX}[:])" for XX in [FX, CX]],
    "(i, j), (j) -> (i)",
    target=TARGET,
)
def matrix_dot_vector_guf(A, B, out):
    """gufunc that calls matrix_dot_vector"""
    matrix_dot_vector(A, B, out)


def test_matrix_dot_vector():
    """Unit tests of `matrix_dot_vector` and `matrix_dot_vector_guf`"""
    A = numba.cuda.to_device(np.linspace(1, 12, 12, dtype=FTYPE).reshape(4, 3))
    v = numba.cuda.to_device(np.linspace(1, 3, 3, dtype=FTYPE))
    w = numba.cuda.to_device(np.ones(4, dtype=FTYPE))

    matrix_dot_vector_guf(A, v, w)

    test = w.copy_to_host()
    ref = np.dot(A.copy_to_host(), v.copy_to_host()).astype(FTYPE)
    assert np.allclose(test, ref, **ALLCLOSE_KW), f"test:\n{test}\n!= ref:\n{ref}"

    logging.info("<< PASS : test_matrix_dot_vector >>")


# --------------------------------------------------------------------------- #


@myjit
def clear_matrix(A):
    """Zero out 2D matrix .. ::

        A[i, j] = 0

    """
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = 0


@guvectorize(
    [f"({XX}[:, :], {XX}[:, :])" for XX in [FX, CX]], "(i, j) -> (i, j)", target=TARGET,
)
def clear_matrix_guf(dummy, out):  # pylint: disable=unused-argument
    """gufunc that calls `clear_matrix`"""
    clear_matrix(out)


def test_clear_matrix():
    """Unit tests of `clear_matrix` and `clear_matrix_guf`"""
    A = numba.cuda.to_device(np.ones((4, 3), dtype=FTYPE))

    clear_matrix_guf(A, A)

    test = A.copy_to_host()
    ref = np.zeros((4, 3), dtype=FTYPE)
    assert np.array_equal(test, ref), f"test:\n{test}\n!= ref:\n{ref}"

    logging.info("<< PASS : test_clear_matrix >>")


# --------------------------------------------------------------------------- #


@myjit
def copy_matrix(A, B):
    """Copy elemnts of 2d array A to array B .. ::

        B[i, j] = A[i, j]

    """
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i, j] = A[i, j]


@guvectorize(
    [f"({XX}[:, :], {XX}[:, :])" for XX in [FX, CX]], "(i, j) -> (i, j)", target=TARGET,
)
def copy_matrix_guf(A, out):
    """gufunc that calls `copy_matrix`"""
    copy_matrix(A, out)


def test_copy_matrix():
    """Unit tests of `copy_matrix` and `copy_matrix_guf`"""
    A = numba.cuda.to_device(np.ones((3, 3), dtype=FTYPE))
    B = numba.cuda.to_device(np.zeros((3, 3), dtype=FTYPE))

    copy_matrix_guf(A, B)

    test = B.copy_to_host()
    ref = A.copy_to_host()
    assert np.array_equal(test, ref), f"test:\n{test}\n!= ref:\n{ref}"

    logging.info("<< PASS : test_copy_matrix >>")


# --------------------------------------------------------------------------- #


def parse_args():
    """parse command line args"""
    parser = ArgumentParser()
    parser.add_argument("-v", action="count", default=Levels.WARN, help="Verbosity")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    set_verbosity(parse_args()["v"])
    test_conjugate_transpose()
    test_conjugate()
    test_matrix_dot_matrix()
    test_matrix_dot_vector()
    test_clear_matrix()
    test_copy_matrix()
