"""
Utilities for working wtih coordinates (e.g. coord-system transformations).

"""


from __future__ import division

import numpy as np

from pisa.utils.log import logging, set_verbosity


__all__ = ['abs2rel', 'rel2abs',
           'test_abs2rel', 'test_rel2abs']


# NOTE: If we want to scale the resolutions about some reference point
# (x_r) and then shift them by `shift`:
#
#   x1 = x_r + (x - x_r)*scale + shift
#
# but we should take the sample points as fixed, and instead sample
# from locations we desire but transformed to the correct locatons on
# the original curve
#
#    x = (x1 - x_r - shift)/scale + x_r
#


def abs2rel(abs_coords, abs_bin_midpoint, rel_scale_ref, scale, abs_obj_shift):
    """Viewing an object that is defined in a relative coordinate space, """
    return (
        (abs_coords - abs_bin_midpoint - abs_obj_shift + rel_scale_ref)/scale
    )


def rel2abs(rel_coords, abs_bin_midpoint, rel_scale_ref, scale, abs_obj_shift):
    """Convert coordinates defined relative to a bin to absolute
    coordinates.

    """
    return (
        rel_coords*scale - rel_scale_ref + abs_bin_midpoint + abs_obj_shift
    )
    #return (
    #    (rel_coords - rel_scale_ref)*scale + rel_scale_ref
    #    + abs_bin_midpoint + abs_obj_shift
    #)


def test_abs2rel():
    """Unit tests for abs2rel function"""
    xabs = np.array([-2, -1, 0, 1, 2])

    # The identity transform
    xrel = abs2rel(abs_coords=xabs, abs_bin_midpoint=0, rel_scale_ref=0,
                   scale=1, abs_obj_shift=0)
    assert np.all(xrel == xabs)

    # Absolute bin midpoint: if the bin is centered at 2, then the relative
    # coordinates should range from -4 to 0
    xrel = abs2rel(abs_coords=xabs, abs_bin_midpoint=2, rel_scale_ref=0,
                   scale=1, abs_obj_shift=0)
    assert xrel[0] == -4 and xrel[-1] == 0

    # Scale: an object that is 4 units wide absolute space scaled by 2
    # should appear to be 2 units wide in relative space... or in other words,
    # the relative coordinates should be spaced more narrowly to one another by
    # a factor of 2 than coordinates in the absolute space
    xrel = abs2rel(abs_coords=xabs, abs_bin_midpoint=0, rel_scale_ref=0,
                   scale=2, abs_obj_shift=0)
    assert (xabs[1]-xabs[0]) == 2*(xrel[1]-xrel[0])

    # Relative scale reference point: If an object living in relative space is
    # centered at 1 and scaled by 2 with rel_scale_ref=1, then it should still
    # be centered at 1 but be wider by a factor of 2. This means that all
    # coordinates must scale relative to 1: 2 gets 2x closer to 1, 0 gets 2x
    # closer to 1, etc
    # if stuff stuff
    xrel = abs2rel(abs_coords=xabs, abs_bin_midpoint=0, rel_scale_ref=1,
                   scale=2, abs_obj_shift=0)
    assert (xabs[1]-xabs[0]) == 2*(xrel[1]-xrel[0])
    assert np.all(xrel == np.array([-0.5, 0, 0.5, 1, 1.5]))
    xrel = abs2rel(abs_coords=xabs, abs_bin_midpoint=0, rel_scale_ref=-2,
                   scale=2, abs_obj_shift=0)
    assert np.all(xrel == np.array([-2, -1.5, -1, -0.5, 0]))

    # Shift: an object that lives in the relative space centered at 0 should
    # now be centered at 1 in absolute space. Relative coordinates should be
    # shifted to the left such that object appears to be shifted to the right.
    xrel = abs2rel(abs_coords=xabs, abs_bin_midpoint=0, rel_scale_ref=0,
                   scale=1, abs_obj_shift=1)
    assert xrel[0] == -3 and xrel[-1] == 1

    logging.info('<< PASS : test_abs2rel >>')


def test_rel2abs():
    """Unit tests for rel2abs function"""
    xabs = np.array([-2, -1, 0, 1, 2])
    kwargs = dict(abs_bin_midpoint=12, rel_scale_ref=-3.3, scale=5.4,
                  abs_obj_shift=19)
    xrel = abs2rel(xabs, **kwargs)
    assert np.allclose(rel2abs(abs2rel(xabs, **kwargs), **kwargs), xabs)
    logging.info('<< PASS : test_rel2abs >>')



if __name__ == '__main__':
    set_verbosity(3)
    test_abs2rel()
    test_rel2abs()
