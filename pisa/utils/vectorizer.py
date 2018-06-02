"""
Collection of useful vectorized functions
"""

from __future__ import absolute_import, print_function

__version__ = '0.1'
__author__ = 'Philipp Eller (pde3@psu.edu)'

import math

import numpy as np
from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa.utils.numba_tools import WHERE


def multiply_and_scale(scale, value, out):
    multiply_and_scale_gufunc(scale,
                              value.get(WHERE),
                              out=out.get(WHERE))
    out.mark_changed(WHERE)

def scale(scale, value, out):
    scale_gufunc(scale,
                 value.get(WHERE),
                 out=out.get(WHERE))
    out.mark_changed(WHERE)

def multiply(val, out):
    multiply_gufunc(val.get(WHERE),
                    out=out.get(WHERE))
    out.mark_changed(WHERE)

def divide(val, out):
    """Divide one aray by another.

    Division by zero results in 0 for that element.
    """
    divide_gufunc(val.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)

def set(val, out):
    set_gufunc(val.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)

def square(val, out):
    square_gufunc(val.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)

def sqrt(val, out):
    sqrt_gufunc(val.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)

def replace(counts, min_count, vals, out):
    """Replace `out` with `vals` when `count` > `min_count`"""
    replace_gufunc(counts.get(WHERE),
                   min_count,
                   vals.get(WHERE),
                   out=out.get(WHERE))

if FTYPE == np.float32:
    _SIGNATURE = ['(f4[:], f4[:], f4[:])']
else:
    _SIGNATURE = ['(f8[:], f8[:], f8[:])']

@guvectorize(_SIGNATURE, '(),()->()', target=TARGET)
def multiply_and_scale_gufunc(scale, value, out):
    out[0] *= scale[0] * value[0]

@guvectorize(_SIGNATURE, '(),()->()', target=TARGET)
def scale_gufunc(scale, value, out):
    out[0] = scale[0] * value[0]


if FTYPE == np.float32:
    _SIGNATURE = ['(f4[:], f4[:])']
else:
    _SIGNATURE = ['(f8[:], f8[:])']

@guvectorize(_SIGNATURE, '()->()', target=TARGET)
def multiply_gufunc(val, out):
    out[0] *= val[0]

@guvectorize(_SIGNATURE, '()->()', target=TARGET)
def divide_gufunc(val, out):
    if val[0] == 0.:
        out[0] = 0.
    else:
        out[0] /= val[0]

@guvectorize(_SIGNATURE, '()->()', target=TARGET)
def set_gufunc(val, out):
    out[0] = val[0]

@guvectorize(_SIGNATURE, '()->()', target=TARGET)
def square_gufunc(val, out):
    out[0] = val[0]**2

@guvectorize(_SIGNATURE, '()->()', target=TARGET)
def sqrt_gufunc(val, out):
    out[0] = math.sqrt(val[0])


if FTYPE == np.float32:
    _SIGNATURE = ['(f4[:], i4[:], f4[:], f4[:])']
else:
    _SIGNATURE = ['(f8[:], i4[:], f8[:], f8[:])']

@guvectorize(_SIGNATURE, '(),(),()->()', target=TARGET)
def replace_gufunc(counts, min_count, vals, out):
    if counts[0] > min_count[0]:
        out[0] = vals[0]


def test():
    from numba import SmartArray
    a = np.linspace(0, 1, 1000, dtype=FTYPE)
    a = SmartArray(a)

    out = np.ones_like(a)
    out = SmartArray(out)

    multiply_and_scale(10., a, out)

    assert np.allclose(out.get('host'), np.linspace(0, 10, 1000, dtype=FTYPE))

if __name__ == '__main__':
    test()
