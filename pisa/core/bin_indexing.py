"""
Functions to retrieve the bin index for a 1- to 3-dimensional sample.

Functions were adapted from translation.py


Notes
-----
The binning convention in PISA (from numpy.histogramdd) is that the lower edge
is inclusive and upper edge is exclusive for a given bin, except for the
upper-most bin whose upper edge is also inclusive. Visually, for 1D:

    [ bin 0 ) [ bin 1 ) ... [ bin num_bins - 1 ]

First bin is index = 0 and last bin is index = (num_bins - 1)

* Values below the lowermost-edge of any dimension's binning return index = -1
* NaN values return index = -1
* Otherwise, values above the uppermost-edge of any dimension's binning return
  index = num_bins

"""

from __future__ import absolute_import, print_function, division

import numpy as np
from numba import guvectorize, SmartArray

from pisa import FTYPE, TARGET
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.translation import find_index
from pisa.utils.log import logging, set_verbosity
from pisa.utils.numba_tools import WHERE


__all__ = ["lookup_indices", "test_lookup_indices"]


FX = "f4" if FTYPE == np.float32 else "f8"


@guvectorize(
    [f"({FX}[:], {FX}[:], i8[:])"],
    "(), (j) -> ()",
    target=TARGET,
)
def lookup_indices_vectorized_1d(sample_x, bin_edges_x, out):
    """Lookup bin indices for sample_x values, where binning is defined by
    `bin_edges_x`."""
    out[0] = find_index(sample_x[0], bin_edges_x)


@guvectorize(
    [f"({FX}[:], {FX}[:], {FX}[:], {FX}[:], i8[:])"],
    "(), (), (a), (b) -> ()",
    target=TARGET,
)
def lookup_indices_vectorized_2d(sample_x, sample_y, bin_edges_x, bin_edges_y, out):
    """Same as above, except we get back the index"""
    idx_x = find_index(sample_x[0], bin_edges_x)
    idx_y = find_index(sample_y[0], bin_edges_y)

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
        out[0] = idx_x * n_y_bins + idx_y


@guvectorize(
    [f"({FX}[:], {FX}[:], {FX}[:], {FX}[:], {FX}[:], {FX}[:], i8[:])"],
    "(), (), (), (a), (b), (c) -> ()",
    target=TARGET,
)
def lookup_indices_vectorized_3d(
    sample_x, sample_y, sample_z, bin_edges_x, bin_edges_y, bin_edges_z, out
):
    """Vectorized gufunc to perform the lookup"""
    idx_x = find_index(sample_x[0], bin_edges_x)
    idx_y = find_index(sample_y[0], bin_edges_y)
    idx_z = find_index(sample_z[0], bin_edges_z)

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
        out[0] = (idx_x * n_y_bins + idx_y) * n_z_bins + idx_z


def lookup_indices(sample, binning):
    """Lookup (flattened) bin index for sample points.

    Parameters
    ----------
    sample : length-M_dimensions sequence of length-N_events SmartArrays
        All smart arrays must have the same lengths; corresponding elements of
        the arrays are the coordinates of an event in the dimensions each array
        represents.

    binning : pisa.core.binning.MultiDimBinning or convertible thereto
        `binning` is passed to instantiate ``MultiDimBinning``, so e.g., a
        pisa.core.binning.OneDimBinning is valid to pass as `binning`

    Returns
    -------
    indices : length-N_events SmartArray
        One for each event the index of the histogram in which it falls into

    Notes
    -----
    this method works for 1d, 2d and 3d histogram only

    """
    # Convert non-MultiDimBinning objects into MultiDimBinning if possible;
    # if this fails, an error will result, as it should
    binning = MultiDimBinning(binning)

    if len(sample) != binning.num_dims:
        raise ValueError(
            f"`binning` has {binning.num_dims} dimension(s), but `sample`"
            f"contains {len(sample)} arrays (so represents {len(sample)}"
            f" dimensions)"
        )

    lookup_funcs = {
        1: lookup_indices_vectorized_1d,
        2: lookup_indices_vectorized_2d,
        3: lookup_indices_vectorized_3d,
    }

    if binning.num_dims not in lookup_funcs:
        raise NotImplementedError(
            "binning must have num_dims in {}; got {}".format(
                sorted(lookup_funcs.keys()), binning.num_dims
            )
        )

    lookup_func = lookup_funcs[binning.num_dims]

    lookup_func_args = (
        [a.get(WHERE) for a in sample]
        + [SmartArray(dim.edge_magnitudes.astype(FTYPE)).get(WHERE) for dim in binning]
    )
    logging.trace("lookup_func_args = {}".format(lookup_func_args))

    # Create an array to store the results
    indices = SmartArray(np.empty_like(sample[0], dtype=np.int64))

    # Perform the lookup
    lookup_func(*lookup_func_args, out=indices.get(WHERE))

    indices.mark_changed(WHERE)

    return indices


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
    logging.trace("array in 1D: {}".format(x.get()))
    logging.trace("Binning: {}".format(binning_1d.bin_edges[0]))
    indices = lookup_indices([x], binning_1d)
    logging.trace("indices of each array element: {}".format(indices.get()))
    logging.trace("*********************************")
    test = indices.get()
    ref = np.array([-1, 0, 1, 6, 6, 7, 6])
    assert np.array_equal(test, ref), "test={} != ref={}".format(test, ref)

    # 2D case:
    #
    # The binning edges are flattened as follows:
    #   [(x=0, y=0), (x=0, y=1), (x=1, y=0), ...]
    #
    logging.trace("TEST 2D:")
    logging.trace("Total number of bins: {}".format(7 * 4))
    logging.trace("array in 2D: {}".format(list(zip(x.get(), y.get()))))
    logging.trace("Binning: {}".format(binning_2d.bin_edges))
    indices = lookup_indices([x, y], binning_2d)
    logging.trace("indices of each array element: {}".format(indices.get()))
    logging.trace("*********************************")
    test = indices.get()
    ref = np.array([-1, 0, 5, 25, 27, 28, 26])
    assert np.array_equal(test, ref), "test={} != ref={}".format(test, ref)

    # 3D case:
    #
    # the binning edges are flattened as follows:
    #   [(x=0, y=0, z=0), (x=0, y=0, z=1), (x=0, y=1, z=0)...]
    #
    logging.trace("TEST 3D:")
    logging.trace("Total number of bins: {}".format(7 * 4 * 2))
    logging.trace("array in 3D: {}".format(list(zip(x.get(), y.get(), z.get()))))
    logging.trace("Binning: {}".format(binning_3d.bin_edges))
    indices = lookup_indices([x, y, z], binning_3d)
    logging.trace("indices of each array element: {}".format(indices.get()))
    logging.trace("*********************************")
    test = indices.get()
    ref = np.array([-1, 0, 11, 51, 54, 56, 52])
    assert np.array_equal(test, ref), "test={} != ref={}".format(test, ref)

    logging.info("<< PASS : test_lookup_indices >>")


if __name__ == "__main__":
    set_verbosity(1)
    test_lookup_indices()
