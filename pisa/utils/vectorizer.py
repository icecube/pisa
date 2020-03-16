# pylint: disable=redefined-outer-name


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


FX = 'f4' if FTYPE == np.float32 else 'f8'


def multiply_and_scale(scale, value, out):
    """Multiply and scale .. ::

        out *= scale * value

    """
    multiply_and_scale_gufunc(scale, value.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


def scale(scale, value, out):
    """Scale .. ::

        out = scale * value

    """
    scale_gufunc(scale, value.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


def multiply(val, out):
    """Multipy one array by another .. ::

        out *= val

    """
    multiply_gufunc(val.get(WHERE),
                    out=out.get(WHERE))
    out.mark_changed(WHERE)


def divide(val, out):
    """Divide one array by another .. ::

        out /= val

    Division by zero results in 0 for that element.
    """
    divide_gufunc(val.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


def set(val, out):  # pylint: disable=redefined-builtin
    """Set array values from another array .. ::

        out[:] = val[:]

    """
    set_gufunc(val.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


def square(val, out):
    """Square values .. ::

        out = val**2

    """
    square_gufunc(val.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


def sqrt(val, out):
    """Square root of values .. ::

        out = sqrt(val)

    """
    sqrt_gufunc(val.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


def replace(counts, min_count, vals, out):
    """Replace `out[i]` with `vals[i]` when `counts[i]` > `min_count`"""
    replace_gufunc(counts.get(WHERE),
                   min_count,
                   vals.get(WHERE),
                   out=out.get(WHERE))


@guvectorize([f'({FX}[:], {FX}[:], {FX}[:])'], '(),()->()', target=TARGET)
def multiply_and_scale_gufunc(scale, value, out):
    """Multiply and scale .. ::

        out *= scale * value

    """
    out[0] *= scale[0] * value[0]


@guvectorize([f'({FX}[:], {FX}[:], {FX}[:])'], '(),()->()', target=TARGET)
def scale_gufunc(scale, value, out):
    """Scale .. ::

        out = scale * value

    """
    out[0] = scale[0] * value[0]


@guvectorize([f'({FX}[:], {FX}[:])'], '()->()', target=TARGET)
def multiply_gufunc(val, out):
    """Multipy one array by another .. ::

        out *= val

    """
    out[0] *= val[0]


@guvectorize([f'({FX}[:], {FX}[:])'], '()->()', target=TARGET)
def divide_gufunc(val, out):
    """Divide one array by another .. ::

        out /= val

    Division by zero results in 0 for that element.
    """
    if val[0] == 0.:
        out[0] = 0.
    else:
        out[0] /= val[0]


@guvectorize([f'({FX}[:], {FX}[:])'], '()->()', target=TARGET)
def set_gufunc(val, out):
    """Set array values from another array .. ::

        out[:] = val[:]

    """
    out[0] = val[0]


@guvectorize([f'({FX}[:], {FX}[:])'], '()->()', target=TARGET)
def square_gufunc(val, out):
    """Square values .. ::

        out = val**2

    """
    out[0] = val[0]**2


@guvectorize([f'({FX}[:], {FX}[:])'], '()->()', target=TARGET)
def sqrt_gufunc(val, out):
    """Square root of values .. ::

        out = sqrt(val)

    """
    out[0] = math.sqrt(val[0])


@guvectorize([f'({FX}[:], i4[:], {FX}[:], {FX}[:])'], '(),(),()->()', target=TARGET)
def replace_gufunc(counts, min_count, vals, out):
    """Replace `out[i]` with `vals[i]` when `counts[i]` > `min_count`"""
    if counts[0] > min_count[0]:
        out[0] = vals[0]


def test_multiply_and_scale():
    """Unit tests for function ``multiply_and_scale``"""
    from numba import SmartArray
    a = np.linspace(0, 1, 1000, dtype=FTYPE)
    a = SmartArray(a)

    out = np.ones_like(a)
    out = SmartArray(out)

    multiply_and_scale(10., a, out)

    assert np.allclose(out.get('host'), np.linspace(0, 10, 1000, dtype=FTYPE))


if __name__ == '__main__':
    test_multiply_and_scale()
