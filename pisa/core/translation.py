# pylint: disable=unsubscriptable-object, too-many-function-args, not-callable, unexpected-keyword-arg, no-value-for-parameter, too-many-boolean-expressions

"""
Module for data representation translation methods
"""

# TODO:
#    - right now we distinguish on histogramming/lookup for scalars (normal) or array, which means that instead
#    of just a single value per e.g. histogram bin, there can be an array of values
#    This should be made more general that one function can handle everything...since now we have several
#    functions doing similar things. not very pretty

from __future__ import absolute_import, print_function, division

from copy import deepcopy

import numpy as np
from numba import guvectorize

import numba

from pisa import FTYPE
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.comparisons import recursiveEquality
from pisa.utils.log import logging, set_verbosity
from pisa.utils.numba_tools import myjit
from pisa.utils import vectorizer

__all__ = [
    'resample',
    'histogram',
    'lookup',
    'find_index',
    'find_index_unsafe',
    'test_histogram',
    'test_find_index',
]


FX = 'f4' if FTYPE == np.float32 else 'f8'


# --------- resampling ------------

def resample(weights, old_sample, old_binning, new_sample, new_binning):
    """Resample binned data with a given binning into any arbitrary
    `new_binning`

    Parameters
    ----------
    weights : np.ndarray
    old_sample : list of np.ndarray
    old_binning : PISA MultiDimBinning
    new_sample : list of np.ndarray
    new_binning : PISA MultiDimBinning

    Returns
    -------
    new_hist_vals

    """
    if old_binning.names != new_binning.names:
        raise ValueError(f'cannot translate betwen {old_binning} and {new_binning}')

    # This is a two step process: first histogram the weights into the new binning
    # and keep the flat_hist_counts
    flat_hist = histogram_np(old_sample, weights, new_binning, apply_weights=True)
    flat_hist_counts = histogram_np(old_sample, weights, new_binning, apply_weights=False)

    with np.errstate(divide='ignore', invalid='ignore'):
        flat_hist /= flat_hist_counts
        flat_hist = np.nan_to_num(flat_hist)

    # now do the inverse, a lookup of hist vals at `new_sample` points
    new_hist_vals = lookup(new_sample, weights, old_binning)

    # Now, for bin we have 1 or less counts, take the lookedup value instead:
    mask = flat_hist_counts > 1
    new_hist_vals[mask] = flat_hist[mask]

    return new_hist_vals


# --------- histogramming methods ---------------

def histogram(sample, weights, binning, averaged, apply_weights=True):
    """Histogram `sample` points, weighting by `weights`, according to `binning`.

    Parameters
    ----------
    sample : list of np.ndarray

    weights : np.ndarray

    binning : PISA MultiDimBinning

    averaged : bool
        If True, the histogram entries are averages of the numbers that end up
        in a given bin. This for example must be used when oscillation
        probabilities are translated, otherwise we end up with
        probability*count per bin

    apply_weights : bool
        wether to use weights or not

    """
    flat_hist = histogram_np(sample, weights, binning, apply_weights=True)

    if averaged:
        flat_hist_counts = histogram_np(sample, weights, binning, apply_weights=False)

        with np.errstate(divide='ignore', invalid='ignore'):
            flat_hist /= flat_hist_counts
            flat_hist = np.nan_to_num(flat_hist)

    return flat_hist


def histogram_np(sample, weights, binning, apply_weights=True):  # pylint: disable=missing-docstring
    """helper function for numpy historams"""
    binning = MultiDimBinning(binning)

    bin_edges = [edges.magnitude for edges in binning.bin_edges]
    if weights is not None and weights.ndim == 2:
        # that means it's 1-dim data instead of scalars
        hists = []
        for i in range(weights.shape[1]):
            w = weights[:, i] if apply_weights else None
            hist, _ = np.histogramdd(sample=sample, weights=w, bins=bin_edges)
            hists.append(hist.ravel())
        flat_hist = np.stack(hists, axis=1)
    else:
        w = weights if apply_weights else None
        hist, _ = np.histogramdd(sample=sample, weights=w, bins=bin_edges)
        flat_hist = hist.ravel()
    return flat_hist.astype(FTYPE)


# ---------- Lookup methods ---------------

def lookup(sample, flat_hist, binning):
    """The inverse of histograming: Extract the histogram values at `sample`
    points.

    Parameters
    ----------
    sample : num_dims list of length-num_samples np.ndarray
        Points at which to find histogram's values
    flat_hist : np.ndarray
        Histogram values
    binning : num_dims MultiDimBinning
        Histogram's binning

    Returns
    -------
    hist_vals : len-num_samples np.ndarray

    Notes
    -----
    Handles up to 3D.

    """
    assert binning.num_dims <= 3, 'can only do up to 3D at the moment'
    bin_edges = [edges.magnitude for edges in binning.bin_edges]

    if flat_hist.ndim == 1:
        #print 'looking up 1D'

        hist_vals = np.zeros_like(sample[0])

        if binning.num_dims == 1:
            lookup_vectorized_1d(
                sample[0],
                flat_hist,
                bin_edges[0],
                out=hist_vals,
            )
        elif binning.num_dims == 2:
            lookup_vectorized_2d(
                sample[0],
                sample[1],
                flat_hist,
                bin_edges[0],
                bin_edges[1],
                out=hist_vals,
            )
        elif binning.num_dims == 3:
            lookup_vectorized_3d(
                sample[0],
                sample[1],
                sample[2],
                flat_hist,
                bin_edges[0],
                bin_edges[1],
                bin_edges[2],
                out=hist_vals,
            )
    elif flat_hist.ndim == 2:
        #print 'looking up ND'
        hist_vals = np.zeros((sample[0].size, flat_hist.shape[1]), dtype=FTYPE)

        if binning.num_dims == 1:
            lookup_vectorized_1d_arrays(
                sample[0],
                flat_hist,
                bin_edges[0],
                out=hist_vals,
            )
        elif binning.num_dims == 2:
            lookup_vectorized_2d_arrays(
                sample[0],
                sample[1],
                flat_hist,
                bin_edges[0],
                bin_edges[1],
                out=hist_vals,
            )
        elif binning.num_dims == 3:
            lookup_vectorized_3d_arrays(
                sample[0],
                sample[1],
                sample[2],
                flat_hist,
                bin_edges[0],
                bin_edges[1],
                bin_edges[2],
                out=hist_vals,
            )
    else:
        raise NotImplementedError()

    return hist_vals


@myjit
def find_index(val, bin_edges):
    """Find index in binning for `val`. If `val` is below binning range or is
    nan, return -1; if `val` is above binning range, return num_bins. Edge
    inclusivity/exclusivity is defined as .. ::

        [ bin 0 ) [ bin 1 ) ... [ bin num_bins-1 ]

    Using these indices to produce histograms should yield identical results
    (ignoring underflow and overflow, which `find_index` has) that are
    equivalent to those produced by ``numpy.histogramdd``.

    Parameters
    ----------
    val : scalar
        Value for which to find bin index

    bin_edges : 1d numpy ndarray of 2 or more scalars
        Must be monotonically increasing, and all bins are assumed to be
        adjacent

    Returns
    -------
    bin_idx : int in [-1, num_bins]
        -1 is returned for underflow or if `val` is nan. `num_bins` is returned
        for overflow. Otherwise, for bin_edges[0] <= `val` <= bin_edges[-1],
        0 <= `bin_idx` <= num_bins - 1

    """
    # TODO: support fast computation for lin and log binnings?

    num_edges = len(bin_edges)
    num_bins = num_edges - 1
    assert num_bins >= 1, 'bin_edges must define at least one bin'

    underflow_idx = -1
    overflow_idx = num_bins

    if val >= bin_edges[0]:
        if val <= bin_edges[-1]:
            bin_idx = find_index_unsafe(val, bin_edges)
            # Paranoia: In case of unforseen numerical issues, force clipping of
            # returned bin index to [0, num_bins - 1] (any `val` outside of binning
            # is already handled, so this should be valid)
            bin_idx = min(max(0, bin_idx), num_bins - 1)
        else:
            bin_idx = overflow_idx
    else:  # either value is below first bin or is NaN
        bin_idx = underflow_idx

    return bin_idx


@myjit
def find_index_unsafe(val, bin_edges):
    """Find bin index of `val` within binning defined by `bin_edges`.

    Validity of `val` and `bin_edges` is not checked.

    Parameters
    ----------
    val : scalar
        Assumed to be within range of `bin_edges` (including lower and upper
        bin edges)
    bin_edges : array

    Returns
    -------
    index

    See also
    --------
    find_index : includes bounds checking and handling of special cases

    """
    # Initialize to point to left-most edge
    left_edge_idx = 0

    # Initialize to point to right-most edge
    right_edge_idx = len(bin_edges) - 1

    while left_edge_idx < right_edge_idx:
        # See where value falls w.r.t. an edge ~midway between left and right edges
        # ``>> 1``: integer division by 2 (i.e., divide w/ truncation)
        test_edge_idx = (left_edge_idx + right_edge_idx) >> 1

        # ``>=``: bin left edges are inclusive
        if val >= bin_edges[test_edge_idx]:
            left_edge_idx = test_edge_idx + 1
        else:
            right_edge_idx = test_edge_idx

    # break condition of while loop is that left_edge_idx points to the
    # right edge of the bin that `val` is inside of; that is one more than
    # that _bin's_ index
    return left_edge_idx - 1


@guvectorize(
    [f'({FX}[:], {FX}[:], {FX}[:], {FX}[:])'],
    '(), (j), (k) -> ()',
    target='cpu',
)
def lookup_vectorized_1d(
    sample,
    flat_hist,
    bin_edges,
    weights,
):
    """Vectorized gufunc to perform the lookup"""
    x = sample[0]
    if (bin_edges[0] <= x <= bin_edges[-1]):
        idx = find_index_unsafe(x, bin_edges)
        weights[0] = flat_hist[idx]
    else:  # outside of binning or nan
        weights[0] = 0.

@guvectorize(
    [f'({FX}[:], {FX}[:, :], {FX}[:], {FX}[:])'],
    '(), (j, d), (k) -> (d)',
    target='cpu',
)
def lookup_vectorized_1d_arrays(
    sample,
    flat_hist,
    bin_edges,
    weights,
):
    """Vectorized gufunc to perform the lookup"""
    x = sample[0]
    if (bin_edges[0] <= x <= bin_edges[-1]):
        idx = find_index_unsafe(x, bin_edges)
        for i in range(weights.size):
            weights[i] = flat_hist[idx, i]
    else:  # outside of binning or nan
        for i in range(weights.size):
            weights[i] = 0.

@guvectorize(
    [f'({FX}[:], {FX}[:], {FX}[:], {FX}[:], {FX}[:], {FX}[:])'],
    '(), (), (j), (k), (l) -> ()',
    target='cpu',
)
def lookup_vectorized_2d(
    sample_x,
    sample_y,
    flat_hist,
    bin_edges_x,
    bin_edges_y,
    weights,
):
    """Vectorized gufunc to perform the lookup"""
    x = sample_x[0]
    y = sample_y[0]
    if (
        x >= bin_edges_x[0]
        and x <= bin_edges_x[-1]
        and y >= bin_edges_y[0]
        and y <= bin_edges_y[-1]
    ):
        idx_x = find_index_unsafe(x, bin_edges_x)
        idx_y = find_index_unsafe(y, bin_edges_y)
        idx = idx_x * (len(bin_edges_y) - 1) + idx_y
        weights[0] = flat_hist[idx]
    else:  # outside of binning or nan
        weights[0] = 0.


@guvectorize(
    [f'({FX}[:], {FX}[:], {FX}[:, :], {FX}[:], {FX}[:], {FX}[:])'],
    '(), (), (j, d), (k), (l) -> (d)',
    target='cpu',
)
def lookup_vectorized_2d_arrays(
    sample_x,
    sample_y,
    flat_hist,
    bin_edges_x,
    bin_edges_y,
    weights,
):
    """Vectorized gufunc to perform the lookup while flat hist and weights have
    both a second dimension
    """
    x = sample_x[0]
    y = sample_y[0]
    if (
        x >= bin_edges_x[0]
        and x <= bin_edges_x[-1]
        and y >= bin_edges_y[0]
        and y <= bin_edges_y[-1]
    ):
        idx_x = find_index_unsafe(x, bin_edges_x)
        idx_y = find_index_unsafe(y, bin_edges_y)
        idx = idx_x * (len(bin_edges_y) - 1) + idx_y
        for i in range(weights.size):
            weights[i] = flat_hist[idx, i]
    else:  # outside of binning or nan
        for i in range(weights.size):
            weights[i] = 0.


@guvectorize(
    [f'({FX}[:], {FX}[:], {FX}[:], {FX}[:], {FX}[:], {FX}[:], {FX}[:], {FX}[:])'],
    '(), (), (), (j), (k), (l), (m) -> ()',
    target='cpu',
)
def lookup_vectorized_3d(
    sample_x,
    sample_y,
    sample_z,
    flat_hist,
    bin_edges_x,
    bin_edges_y,
    bin_edges_z,
    weights,
):
    """Vectorized gufunc to perform the lookup"""
    x = sample_x[0]
    y = sample_y[0]
    z = sample_z[0]
    if (
        x >= bin_edges_x[0]
        and x <= bin_edges_x[-1]
        and y >= bin_edges_y[0]
        and y <= bin_edges_y[-1]
        and z >= bin_edges_z[0]
        and z <= bin_edges_z[-1]
    ):
        idx_x = find_index_unsafe(x, bin_edges_x)
        idx_y = find_index_unsafe(y, bin_edges_y)
        idx_z = find_index_unsafe(z, bin_edges_z)
        idx = (idx_x * (len(bin_edges_y) - 1) + idx_y) * (len(bin_edges_z) - 1) + idx_z
        weights[0] = flat_hist[idx]
    else:  # outside of binning or nan
        weights[0] = 0.


@guvectorize(
    [f'({FX}[:], {FX}[:], {FX}[:], {FX}[:, :], {FX}[:], {FX}[:], {FX}[:], {FX}[:])'],
    '(), (), (), (j, d), (k), (l), (m) -> (d)',
    target='cpu',
)
def lookup_vectorized_3d_arrays(
    sample_x,
    sample_y,
    sample_z,
    flat_hist,
    bin_edges_x,
    bin_edges_y,
    bin_edges_z,
    weights,
):
    """Vectorized gufunc to perform the lookup while flat hist and weights have
    both a second dimension"""
    x = sample_x[0]
    y = sample_y[0]
    z = sample_z[0]
    if (
        x >= bin_edges_x[0]
        and x <= bin_edges_x[-1]
        and y >= bin_edges_y[0]
        and y <= bin_edges_y[-1]
        and z >= bin_edges_z[0]
        and z <= bin_edges_z[-1]
    ):
        idx_x = find_index_unsafe(x, bin_edges_x)
        idx_y = find_index_unsafe(y, bin_edges_y)
        idx_z = find_index_unsafe(z, bin_edges_z)
        idx = (idx_x * (len(bin_edges_y) - 1) + idx_y) * (len(bin_edges_z) - 1) + idx_z
        for i in range(weights.size):
            weights[i] = flat_hist[idx, i]
    else:  # outside of binning or nan
        for i in range(weights.size):
            weights[i] = 0.


def test_histogram():
    """Unit tests for `histogram` function.

    Correctness is defined as matching the histogram produced by
    numpy.histogramdd.
    """
    all_num_bins = [2, 3, 4]
    n_evts = 10000
    rand = np.random.RandomState(seed=0)

    weights = rand.rand(n_evts).astype(FTYPE)
    binning = []
    sample = []
    for num_dims, num_bins in enumerate(all_num_bins, start=1):
        binning.append(
            OneDimBinning(
                name=f'dim{num_dims - 1}',
                num_bins=num_bins,
                is_lin=True,
                domain=[0, num_bins],
            )
        )

        s = rand.rand(n_evts).astype(FTYPE) * num_bins
        sample.append(s)

        bin_edges = [b.edge_magnitudes for b in binning]
        test = histogram(sample, weights, binning, averaged=False)
        ref, _ = np.histogramdd(sample=sample, bins=bin_edges, weights=weights)
        ref = ref.astype(FTYPE).ravel()
        assert recursiveEquality(test, ref), f'\ntest:\n{test}\n\nref:\n{ref}'

        test_avg = histogram(sample, weights, binning, averaged=True)
        ref_counts, _ = np.histogramdd(sample=sample, bins=bin_edges, weights=None)
        ref_counts = ref_counts.astype(FTYPE).ravel()
        ref_avg = (ref / ref_counts).astype(FTYPE)
        assert recursiveEquality(test_avg, ref_avg), \
                f'\ntest_avg:\n{test_avg}\n\nref_avg:\n{ref_avg}'

    logging.info('<< PASS : test_histogram >>')


def test_find_index():
    """Unit tests for `find_index` function.

    Correctness is defined as producing the same histogram as numpy.histogramdd
    by using the output of `find_index` (ignoring underflow and overflow values).
    Additionally, -1 should be returned if a value is below the range
    (underflow) or is nan, and num_bins should be returned for a value above
    the range (overflow).
    """
    # Negative, positive, integer, non-integer, binary-unrepresentable (0.1) edges
    basic_bin_edges = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 3, 4]

    failures = 0
    for basic_bin_edges in [
        # Negative, positive, integer, non-integer, binary-unrepresentable (0.1) edges
        [-1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 3, 4],

        # A single infinite bin: [-np.inf, np.inf]
        [],

        # Half-infinite bins (lower or upper edge) & [-inf, .1, +inf]
        [0.1],

        # Single bin with finite edges & +/-inf-edge(s)-added variants
        [-0.1, 0.1],
    ]:
        # Bin edges from above, w/ and w/o +/-inf as left and/or right edges
        for le, re in [
            (None, None),
            (-np.inf, None),
            (None, np.inf),
            (-np.inf, np.inf)
        ]:
            bin_edges = deepcopy(basic_bin_edges)
            if le is not None:
                bin_edges = [le] + bin_edges
            if re is not None:
                bin_edges = bin_edges + [re]
            if len(bin_edges) < 2:
                continue
            logging.debug('bin_edges being tested: %s', bin_edges)
            bin_edges = np.array(bin_edges, dtype=FTYPE)

            num_bins = len(bin_edges) - 1
            underflow_idx = -1
            overflow_idx = num_bins

            #
            # Construct test values to try out
            #

            non_finite_vals = [-np.inf, +np.inf, np.nan]

            # Values within bins (i.e., not on edges)
            inbin_vals = []
            for idx in range(len(bin_edges) - 1):
                lower_be = bin_edges[idx]
                upper_be = bin_edges[idx + 1]
                if np.isfinite(lower_be):
                    if np.isfinite(upper_be):
                        inbin_val = (lower_be + upper_be) / 2
                    else:
                        inbin_val = lower_be + 10.5
                else:
                    if np.isfinite(upper_be):
                        inbin_val = upper_be - 10.5
                    else:
                        inbin_val = 10.5
                inbin_vals.append(inbin_val)

            # Values above/below bin edges by one unit of floating point
            # accuracy
            eps = np.finfo(FTYPE).eps  # pylint: disable=no-member
            below_edges_vals = [FTYPE((1 - eps)*be) for be in bin_edges]
            above_edges_vals = [FTYPE((1 + eps)*be) for be in bin_edges]

            test_vals = np.concatenate(
                [
                    non_finite_vals,
                    bin_edges,
                    inbin_vals,
                    below_edges_vals,
                    above_edges_vals,
                ]
            )
            logging.trace('test_vals = %s', test_vals)

            #
            # Run tests
            #
            for val in test_vals:
                val = FTYPE(val)

                np_histvals, _ = np.histogramdd([val], np.atleast_2d(bin_edges))
                nonzero_indices = np.nonzero(np_histvals)[0]  # select first & only dim
                if np.isnan(val):
                    assert len(nonzero_indices) == 0, str(len(nonzero_indices))
                    expected_idx = underflow_idx
                elif val < bin_edges[0]:
                    assert len(nonzero_indices) == 0, str(len(nonzero_indices))
                    expected_idx = underflow_idx
                elif val > bin_edges[-1]:
                    assert len(nonzero_indices) == 0, str(len(nonzero_indices))
                    expected_idx = overflow_idx
                else:
                    assert len(nonzero_indices) == 1, str(len(nonzero_indices))
                    expected_idx = nonzero_indices[0]

                found_idx = find_index(val, bin_edges)

                if found_idx != expected_idx:
                    failures += 1
                    msg = 'val={}, edges={}: Expected idx={}, found idx={}'.format(
                        val, bin_edges, expected_idx, found_idx
                    )
                    logging.error(msg)

    assert failures == 0, f"{failures} failures, inspect ERROR messages above for info"

    logging.info('<< PASS : test_find_index >>')


if __name__ == '__main__':
    set_verbosity(1)
    test_find_index()
    test_histogram()
