"""
Functions to retrieve the bin location of each elements
of an array, inside a Container, based on its specified
output binning.

functions were adapted from translation.py


NOTE:
----

binning convention in pisa is that both lower and upper
bounds of an inner bin edges is included in the lower bin

first bin is bin 0

last bin is bin (NBins -1)

bounds falling on the lowest bin edge is included in bin 0

bounds falling in the highest bin edge is included in bin Nbins

values falling below the lowest edge have index -1

values falling above the highest edge have index Nbins

if a value falls on an inner edge, it is included in the lowest
bin bound by that edge


"""

from __future__ import absolute_import, print_function, division

import numpy as np
from numba import guvectorize, SmartArray

from pisa import FTYPE, TARGET
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.log import logging, set_verbosity
from pisa.utils.numba_tools import WHERE

from translation import find_index

__all__ = ["lookup_indices", "test_lookup_indices"]


# ---------- Lookup methods ---------------


def lookup_indices(sample, binning):
    """The inverse of histograming

    Paramters
    ---------
    sample : list of SmartArrays

    binning : pisa.core.binning.OneDimBinning or pisa.core.binning.MultiDimBinning

    Returns: for each event the index of the histogram in which it falls into

    Notes
    -----
    this method works for 1d, 2d and 3d histogram only

    """
    if isinstance(binning, OneDimBinning):
        binning = MultiDimBinning(binning)
    if not isinstance(binning, MultiDimBinning):
        raise TypeError(
            "binning must be OneDimBinning or MultiDimBinning; got {}".format(
                type(binning)
            )
        )
    assert 1 <= binning.num_dims <= 3, "can only do 1d, 2d and 3d at the moment"

    bin_edges = [edges.magnitude for edges in binning.bin_edges]

    array = SmartArray(np.zeros_like(sample[0], dtype=np.int64))

    if binning.num_dims == 1:

        assert (
            len(sample) == 1
        ), "ERROR: binning provided has 1 dimension, but sample provided has not"

        lookup_index_vectorized_1d(
            sample[0].get(WHERE), bin_edges[0], out=array.get(WHERE)
        )

    elif binning.num_dims == 2:

        assert (
            len(sample) == 2
        ), "ERROR: binning provided has 2 dimensions, but sample provided has not."

        lookup_index_vectorized_2d(
            sample[0].get(WHERE),
            sample[1].get(WHERE),
            bin_edges[0],
            bin_edges[1],
            out=array.get(WHERE),
        )

    elif binning.num_dims == 3:

        assert (
            len(sample) == 3
        ), "ERROR: binning provided has 3 dimensions, but sample provided has not."

        lookup_index_vectorized_3d(
            sample[0].get(WHERE),
            sample[1].get(WHERE),
            sample[2].get(WHERE),
            bin_edges[0],
            bin_edges[1],
            bin_edges[2],
            out=array.get(WHERE),
        )

    else:
        raise NotImplementedError()

    array.mark_changed(WHERE)
    return array


# -----------------------------------------------------------------------
# Numba vectorized functions

if FTYPE == np.float32:
    _SIGNATURE = ["(f4[:], f4[:], i8[:])"]
else:
    _SIGNATURE = ["(f8[:], f8[:], i8[:])"]


@guvectorize(_SIGNATURE, "(), (j) -> ()", target=TARGET)
def lookup_index_vectorized_1d(sample_x, bin_edges_x, out):
    sample_x_ = sample_x[0]
    idx = find_index(sample_x_, bin_edges_x)
    out[0] = idx


if FTYPE == np.float32:
    _SIGNATURE = ["(f4[:], f4[:], f4[:], f4[:], i8[:])"]
else:
    _SIGNATURE = ["(f8[:], f8[:], f8[:], f8[:], i8[:])"]


@guvectorize(_SIGNATURE, "(), (), (j), (k) -> ()", target=TARGET)
def lookup_index_vectorized_2d(sample_x, sample_y, bin_edges_x, bin_edges_y, out):
    """Same as above, except we get back the index"""
    sample_x_ = sample_x[0]
    sample_y_ = sample_y[0]

    idx_x = find_index(sample_x_, bin_edges_x)
    idx_y = find_index(sample_y_, bin_edges_y)

    n_x_bins = len(bin_edges_x) - 1
    n_y_bins = len(bin_edges_y) - 1
    n_bins = n_x_bins * n_y_bins

    if idx_x == -1 or idx_y == -1:
        # any dim underflowed
        out[0] = -1
    elif idx_x == n_x_bins or idx_y == n_y_bins:
        # any dim overflowed
        out[0] = n_bins
    else:
        idx = idx_x * n_y_bins + idx_y
        out[0] = idx


if FTYPE == np.float32:
    _SIGNATURE = ["(f4[:], f4[:], f4[:], f4[:], f4[:], f4[:], i8[:])"]
else:
    _SIGNATURE = ["(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], i8[:])"]


@guvectorize(_SIGNATURE, "(), (), (), (j), (k), (l) -> ()", target=TARGET)
def lookup_index_vectorized_3d(
    sample_x, sample_y, sample_z, bin_edges_x, bin_edges_y, bin_edges_z, out
):
    """Vectorized gufunc to perform the lookup"""
    sample_x_ = sample_x[0]
    sample_y_ = sample_y[0]
    sample_z_ = sample_z[0]

    idx_x = find_index(sample_x_, bin_edges_x)
    idx_y = find_index(sample_y_, bin_edges_y)
    idx_z = find_index(sample_z_, bin_edges_z)

    n_x_bins = len(bin_edges_x) - 1
    n_y_bins = len(bin_edges_y) - 1
    n_z_bins = len(bin_edges_z) - 1
    n_bins = n_x_bins * n_y_bins * n_z_bins

    if idx_x == -1 or idx_y == -1 or idx_z == -1:
        # any dim underflowed
        out[0] = -1
    elif idx_x == n_x_bins or idx_y == n_y_bins or idx_z == n_z_bins:
        # any dim overflowed
        out[0] = n_bins
    else:
        idx = (idx_x * n_y_bins + idx_y) * n_z_bins + idx_z
        out[0] = idx


def test_lookup_indices():
    """Unit tests for `lookup_indices` function"""

    #
    # Test a variety of points.
    # Points falling exactly on the bound are included in the
    #
    n_evts = 100

    x = np.array([-5, 0.5, 1.5, 7.0, 6.5, 8.0, 6.5], dtype=FTYPE)
    y = np.array([-5, 0.5, 1.5, 1.5, 3.0, 1.5, 2.5], dtype=FTYPE)
    z = np.array([-5, 0.5, 1.5, 1.5, 0.5, 6.0, 0.5], dtype=FTYPE)

    w = np.ones(n_evts, dtype=FTYPE)

    x = SmartArray(x)
    y = SmartArray(y)
    z = SmartArray(z)

    w = SmartArray(w)

    binning_x = OneDimBinning(name="x", num_bins=7, is_lin=True, domain=[0, 7])
    binning_y = OneDimBinning(name="y", num_bins=4, is_lin=True, domain=[0, 4])
    binning_z = OneDimBinning(name="z", num_bins=2, is_lin=True, domain=[0, 2])

    binning_1d = binning_x
    binning_2d = binning_x * binning_y
    binning_3d = binning_x * binning_y * binning_z

    # 1D case: check that each event falls into its predicted bin
    #
    # All values higher or equal to the last bin edges are assigned an index of zero
    #
    logging.trace("TEST 1D:")
    logging.trace("Total number of bins: {}".format(7))
    logging.trace("array in 1D: {}".format(x.get(WHERE)))
    logging.trace("Binning: {}".format(binning_1d.bin_edges[0]))
    indices = lookup_indices([x], binning_1d)
    logging.trace("indices of each array element: {}".format(indices.get(WHERE)))
    logging.trace("*********************************")
    assert np.array_equal(indices.get(WHERE), np.array([-1, 0, 1, 6, 6, 7, 6]))

    # 2D case:
    #
    # The binning edges are flattened as follows:
    #   [(x=0, y=0), (x=0, y=1), (x=1, y=0), ...]
    #
    logging.trace("TEST 2D:")
    logging.trace("Total number of bins: {}".format(7 * 4))
    logging.trace(
        "array in 2D: {}".format([(i, j) for i, j in zip(x.get(WHERE), y.get(WHERE))])
    )
    logging.trace("Binning: {}".format(binning_2d.bin_edges))
    indices = lookup_indices([x, y], binning_2d)
    logging.trace("indices of each array element: {}".format(indices.get(WHERE)))
    logging.trace("*********************************")
    assert np.array_equal(indices.get(WHERE), np.array([-1, 0, 5, 25, 26, 28, 26]))

    # 3D case:
    #
    # the binning edges are flattened as follows:
    #   [(x=0, y=0, z=0), (x=0, y=0, z=1), (x=0, y=1, z=0)...]
    #
    logging.trace("TEST 3D:")
    logging.trace("Total number of bins: {}".format(7 * 4 * 2))
    logging.trace(
        "array in 3D: {}".format(
            [(i, j, k) for i, j, k in zip(x.get(WHERE), y.get(WHERE), z.get(WHERE))]
        )
    )
    logging.trace("Binning: {}".format(binning_3d.bin_edges))
    indices = lookup_indices([x, y, z], binning_3d)
    logging.trace("indices of each array element: {}".format(indices.get(WHERE)))
    logging.trace("*********************************")
    assert np.array_equal(indices.get(WHERE), np.array([-1, 0, 11, 51, 52, 56, 52]))

    logging.info("<< PASS : test_lookup_indices >>")


if __name__ == "__main__":
    set_verbosity(1)
    test_lookup_indices()
