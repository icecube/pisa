"""
Test functions used during developpement of the osc. code
Please ignore unless you are the author
"""
from __future__ import print_function

import time

import numpy as np
from numba import guvectorize

from pisa import TARGET
from pisa.utils.numba_tools import (
    WHERE,
    cuda,
    myjit,
    ftype,
    ctype,
    matrix_dot_matrix,
    matrix_dot_vector,
)


@myjit
def sum_row_kernel(mix, bla, inp, out):
    C = cuda.local.array(shape=(3, 3), dtype=ftype)
    D = cuda.local.array(shape=(3), dtype=ctype)
    E = cuda.local.array(shape=(3), dtype=ctype)
    matrix_dot_matrix(mix, mix, C)
    D[0] = 0.0 + 2.0j
    D[1] = 1.0 + 2.0j
    D[2] = 1.0 + 2.0j
    matrix_dot_vector(C, D, E)
    bla *= 0.1
    out[0] += E[1].real * bla.real + inp[0]


@guvectorize(
    ["void(float64[:,:], complex128, int32[:], int32[:])"],
    "(a,b),(),(f)->()",
    target=TARGET,
)
def sum_row(mix, bla, inp, out):
    sum_row_kernel(mix, bla, inp, out)


def main():
    print("ftype=", ftype)

    # hist arrays
    mix = np.ones((3, 3), dtype=np.float64)
    n = 1000000
    inp = np.arange(3 * n, dtype=np.int32).reshape(n, 3)
    out = np.ones((n), dtype=np.int32)

    start_t = time.time()
    sum_row(mix, 42.0 + 2j, inp, out=out)
    end_t = time.time()
    print("took %.5f" % (end_t - start_t))
    start_t = time.time()
    sum_row(mix, 42.0 + 2j, inp, out=out)
    end_t = time.time()
    print("took %.5f" % (end_t - start_t))
    out

    print(out.get("host"))


if __name__ == "__main__":
    main()
