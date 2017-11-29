'''
Collection of useful vectorized functions
'''
from __future__ import print_function

__version__ = '0.1'
__author__ = 'Philipp Eller (pde3@psu.edu)'


import numpy as np
from numba import guvectorize
import math, cmath

from pisa.utils.numba_tools import WHERE, myjit
from pisa import FTYPE, TARGET

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

def set(val, out):
    set_gufunc(val.get(WHERE),
               out=out.get(WHERE))
    out.mark_changed(WHERE)

def square(val, out):
    square_gufunc(val.get(WHERE),
                  out=out.get(WHERE))
    out.mark_changed(WHERE)

def sqrt(val, out):
    sqrt_gufunc(val.get(WHERE),
                out=out.get(WHERE))
    out.mark_changed(WHERE)

# vectorized function to apply
# must be outside class
if FTYPE == np.float64:
    signature = '(f8, f8, f8[:])'
else:
    signature = '(f4, f4, f4[:])'

@guvectorize([signature], '(),()->()', target=TARGET)
def multiply_and_scale_gufunc(scale, value, out):
    out[0] *= scale * value

@guvectorize([signature], '(),()->()', target=TARGET)
def scale_gufunc(scale, value, out):
    out[0] = scale * value

if FTYPE == np.float64:
    signature = '(f8, f8[:])'
else:
    signature = '(f4, f4[:])'

@guvectorize([signature], '()->()', target=TARGET)
def multiply_gufunc(val, out):
    out[0] *= val

@guvectorize([signature], '()->()', target=TARGET)
def set_gufunc(val, out):
    out[0] = val

@guvectorize([signature], '()->()', target=TARGET)
def square_gufunc(val, out):
    out[0] = val**2

@guvectorize([signature], '()->()', target=TARGET)
def sqrt_gufunc(val, out):
    out[0] = math.sqrt(val)

