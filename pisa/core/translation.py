# pylint: disable = unsubscriptable-object, too-many-function-args, not-callable, unexpected-keyword-arg, no-value-for-parameter
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
from numba import guvectorize, SmartArray, cuda

from pisa import FTYPE, TARGET
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.log import logging, set_verbosity
from pisa.utils.numba_tools import myjit, WHERE
from pisa.utils import vectorizer

__all__ = [
    'resample',
    'get_hist',
    'histogram',
    'lookup',
    'find_index',
    'resample',
    'test_histogram',
    'test_find_index',
]


# --------- resampling ------------

def resample(weights, old_sample, old_binning, new_sample, new_binning):
    """Resample binned data with a given binning into any arbitrary
    `new_binning`

    Paramters
    ---------
    weights : SmartArray

    old_sample : list of SmartArrays

    old_binning : PISA MultiDimBinning

    inew_sample : list of SmartArrays

    new_binning : PISA MultiDimBinning

    """
    # make sure thw two binning have the same dimensions!
    assert old_binning.names == new_binning.names, 'cannot translate betwen %s and %s'%(old_binning, new_binning)

    # this is a two step process, first histogram the weights into the new binning:
    # and keep the flat_hist_counts
    if TARGET == 'cuda':
        flat_hist = get_hist_gpu(old_sample, weights, new_binning, apply_weights=True)
        flat_hist_counts = get_hist_gpu(old_sample, weights, new_binning, apply_weights=False)
    else:
        #print(old_sample[0].get('host'))
        #print(weights.get('host'))
        flat_hist = get_hist_np(old_sample, weights, new_binning, apply_weights=True)
        flat_hist_counts = get_hist_np(old_sample, weights, new_binning, apply_weights=False)
    vectorizer.divide(flat_hist_counts, flat_hist)

    # now do the inverse, a lookup
    lookup_flat_hist = lookup(new_sample, weights, old_binning)

    # Now, for bin we have 1 or less counts, take the lookedup value instead:
    vectorizer.replace(flat_hist_counts, 1, flat_hist, out=lookup_flat_hist)

    return lookup_flat_hist


# --------- histogramming methods ---------------

def get_hist(sample, weights, binning, averaged):
    """Histogram `sample` points, weighting by `weights`, according to `binning`.

    Paramters
    ---------
    sample : list of SmartArrays

    weights : SmartArray

    binning : PISA MultiDimBinning

    averaged : bool
        If True, the histogram entries are averages of the numbers that end up
        in a given bin. This for example must be used when oscillation
        probabilities are translated, otherwise we end up with
        probability*count per bin

    """
    if TARGET == 'cuda':
        flat_hist = get_hist_gpu(sample, weights, binning, apply_weights=True)
        if averaged:
            flat_hist_counts = get_hist_gpu(sample, weights, binning, apply_weights=False)
    else:
        flat_hist = get_hist_np(sample, weights, binning, apply_weights=True)
        if averaged:
            flat_hist_counts = get_hist_np(sample, weights, binning, apply_weights=False)
    if averaged:
        #print(flat_hist_counts.get('host').shape)
        #print(flat_hist.get('host').shape)
        vectorizer.divide(flat_hist_counts, flat_hist)
    return flat_hist


def histogram(sample, weights, binning, averaged):
    return get_hist(sample, weights, binning, averaged)

histogram.__doc__ = get_hist.__doc__


def get_hist_gpu(sample, weights, binning, apply_weights=True):
    # ToDo:
    # * make for d > 3
    if binning.num_dims in [2, 3]:
        bin_edges = [edges.magnitude for edges in binning.bin_edges]
        if len(weights.shape) > 1:
            # so we have arrays
            flat_hist = SmartArray(np.zeros((binning.size, weights.shape[1]), dtype=FTYPE))
            arrays = True
        else:
            flat_hist = SmartArray(np.zeros(binning.size, dtype=FTYPE))
            arrays = False
        size = weights.shape[0]
        d_bin_edges_x = cuda.to_device(bin_edges[0])
        d_bin_edges_y = cuda.to_device(bin_edges[1])
        if binning.num_dims == 2:
            if arrays:
                histogram_2d_kernel_arrays[(size + 511) // 512, 512](
                    sample[0].get('gpu'),
                    sample[1].get('gpu'),
                    flat_hist,
                    d_bin_edges_x,
                    d_bin_edges_y,
                    weights.get('gpu'),
                    apply_weights,
                )
            else:
                histogram_2d_kernel[(size + 511) // 512, 512](
                    sample[0].get('gpu'),
                    sample[1].get('gpu'),
                    flat_hist,
                    d_bin_edges_x,
                    d_bin_edges_y,
                    weights.get('gpu'),
                    apply_weights,
                )
        elif binning.num_dims == 3:
            d_bin_edges_z = cuda.to_device(bin_edges[2])
            if arrays:
                histogram_3d_kernel_arrays[(size + 511) // 512, 512](
                    sample[0].get('gpu'),
                    sample[1].get('gpu'),
                    sample[2].get('gpu'),
                    flat_hist,
                    d_bin_edges_x,
                    d_bin_edges_y,
                    d_bin_edges_z,
                    weights.get('gpu'),
                    apply_weights,
                )
            else:
                histogram_3d_kernel[(size + 511) // 512, 512](
                    sample[0].get('gpu'),
                    sample[1].get('gpu'),
                    sample[2].get('gpu'),
                    flat_hist,
                    d_bin_edges_x,
                    d_bin_edges_y,
                    d_bin_edges_z,
                    weights.get('gpu'),
                    apply_weights,
                )
        return flat_hist
    else:
        raise NotImplementedError(
            'Other dimesnions that 2 and 3 on the GPU not supported right now'
        )

get_hist_gpu.__doc__ = get_hist.__doc__


def get_hist_np(sample, weights, binning, apply_weights=True):
    """helper function for numoy historams"""
    bin_edges = [edges.magnitude for edges in binning.bin_edges]
    sample = [s.get('host') for s in sample]
    weights = weights.get('host')
    if weights.ndim == 2:
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
    return SmartArray(flat_hist.astype(FTYPE))


# TODO: can we do just n-dimensional? And scalars or arbitrary array shapes? This is so ugly :/
# Furthermore: optimize using shared memory
@cuda.jit
def histogram_2d_kernel(sample_x, sample_y, flat_hist, bin_edges_x, bin_edges_y, weights, apply_weights):
    i = cuda.grid(1)
    if i < sample_x.size:
        if (sample_x[i] >= bin_edges_x[0]
                and sample_x[i] <= bin_edges_x[-1]
                and sample_y[i] >= bin_edges_y[0]
                and sample_y[i] <= bin_edges_y[-1]):
            idx_x = find_index(sample_x[i], bin_edges_x)
            idx_y = find_index(sample_y[i], bin_edges_y)
            idx = idx_x * (bin_edges_y.size - 1) + idx_y
            if apply_weights:
                cuda.atomic.add(flat_hist, idx, weights[i])
            else:
                cuda.atomic.add(flat_hist, idx, 1.)


@cuda.jit
def histogram_2d_kernel_arrays(sample_x, sample_y, flat_hist, bin_edges_x, bin_edges_y, weights, apply_weights):
    i = cuda.grid(1)
    if i < sample_x.size:
        if (sample_x[i] >= bin_edges_x[0]
                and sample_x[i] <= bin_edges_x[-1]
                and sample_y[i] >= bin_edges_y[0]
                and sample_y[i] <= bin_edges_y[-1]):
            idx_x = find_index(sample_x[i], bin_edges_x)
            idx_y = find_index(sample_y[i], bin_edges_y)
            idx = idx_x * (bin_edges_y.size - 1) + idx_y
            for j in range(flat_hist.shape[1]):
                if apply_weights:
                    cuda.atomic.add(flat_hist, (idx, j), weights[i, j])
                else:
                    cuda.atomic.add(flat_hist, (idx, j), 1.)


@cuda.jit
def histogram_3d_kernel(sample_x, sample_y, sample_z, flat_hist, bin_edges_x, bin_edges_y, bin_edges_z, weights, apply_weights):
    i = cuda.grid(1)
    if i < sample_x.size:
        if (sample_x[i] >= bin_edges_x[0]
                and sample_x[i] <= bin_edges_x[-1]
                and sample_y[i] >= bin_edges_y[0]
                and sample_y[i] <= bin_edges_y[-1]
                and sample_z[i] >= bin_edges_z[0]
                and sample_z[i] <= bin_edges_z[-1]):
            idx_x = find_index(sample_x[i], bin_edges_x)
            idx_y = find_index(sample_y[i], bin_edges_y)
            idx_z = find_index(sample_z[i], bin_edges_z)
            idx = idx_x * (bin_edges_y.size - 1) * (bin_edges_z.size - 1) + idx_y * (bin_edges_z.size - 1) + idx_z
            if apply_weights:
                cuda.atomic.add(flat_hist, idx, weights[i])
            else:
                cuda.atomic.add(flat_hist, idx, 1.)


@cuda.jit
def histogram_3d_kernel_arrays(sample_x, sample_y, sample_z, flat_hist, bin_edges_x, bin_edges_y, bin_edges_z, weights, apply_weights):
    i = cuda.grid(1)
    if i < sample_x.size:
        if (sample_x[i] >= bin_edges_x[0]
                and sample_x[i] <= bin_edges_x[-1]
                and sample_y[i] >= bin_edges_y[0]
                and sample_y[i] <= bin_edges_y[-1]
                and sample_z[i] >= bin_edges_z[0]
                and sample_z[i] <= bin_edges_z[-1]):
            idx_x = find_index(sample_x[i], bin_edges_x)
            idx_y = find_index(sample_y[i], bin_edges_y)
            idx_z = find_index(sample_z[i], bin_edges_z)
            idx = idx_x * (bin_edges_y.size - 1) * (bin_edges_z.size - 1) + idx_y * (bin_edges_z.size - 1) + idx_z
            for j in range(flat_hist.shape[1]):
                if apply_weights:
                    cuda.atomic.add(flat_hist, (idx, j), weights[i, j])
                else:
                    cuda.atomic.add(flat_hist, (idx, j), 1.)


# ---------- Lookup methods ---------------

def lookup(sample, flat_hist, binning):
    """The inverse of histograming

    Paramters
    --------
    sample : list of SmartArrays

    flat_hist : SmartArray

    binning : PISA MultiDimBinning

    Notes
    -----
    this is only a 2d method right now

    """
    #print(binning)
    assert binning.num_dims in [2, 3], 'can only do 2d and 3d at the moment'
    bin_edges = [edges.magnitude for edges in binning.bin_edges]
    # todo: directly return smart array
    if flat_hist.ndim == 1:
        #print 'looking up 1D'
        array = SmartArray(np.zeros_like(sample[0]))
        if binning.num_dims == 2:
            lookup_vectorized_2d(
                sample[0].get(WHERE),
                sample[1].get(WHERE),
                flat_hist.get(WHERE),
                bin_edges[0],
                bin_edges[1],
                out=array.get(WHERE),
            )
        elif binning.num_dims == 3:
            lookup_vectorized_3d(
                sample[0].get(WHERE),
                sample[1].get(WHERE),
                sample[2].get(WHERE),
                flat_hist.get(WHERE),
                bin_edges[0],
                bin_edges[1],
                bin_edges[2],
                out=array.get(WHERE),
            )
    elif flat_hist.ndim == 2:
        #print 'looking up ND'
        array = SmartArray(np.zeros((sample[0].size, flat_hist.shape[1]), dtype=FTYPE))
        if binning.num_dims == 2:
            lookup_vectorized_2d_arrays(
                sample[0].get(WHERE),
                sample[1].get(WHERE),
                flat_hist.get(WHERE),
                bin_edges[0],
                bin_edges[1],
                out=array.get(WHERE),
            )
        elif binning.num_dims == 3:
            lookup_vectorized_3d_arrays(
                sample[0].get(WHERE),
                sample[1].get(WHERE),
                sample[2].get(WHERE),
                flat_hist.get(WHERE),
                bin_edges[0],
                bin_edges[1],
                bin_edges[2],
                out=array.get(WHERE),
            )
    else:
        raise NotImplementedError()
    array.mark_changed(WHERE)
    return array


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
    assert num_bins >= 1

    underflow_idx = -1
    overflow_idx = num_bins

    if val >= bin_edges[0]:
        if val <= bin_edges[-1]:
            #
            # Binary search within binning (inclusive of left and right edges)
            #

            # Initialize to point to left-most edge
            left_edge_idx = 0

            # Initialize to point to right-most edge
            right_edge_idx = num_edges - 1

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
            bin_idx = left_edge_idx - 1

            # Paranoia: In case of unforseen numerical issues, force clipping of
            # returned bin index to [0, num_bins - 1] (any `val` outside of binning
            # is already handled, so this should be valid)
            bin_idx = min(max(0, bin_idx), num_bins - 1)
        else:
            bin_idx = overflow_idx
    else:  # either value is below first bin or is NaN
        bin_idx = underflow_idx

    return bin_idx


if FTYPE == np.float32:
    _SIGNATURE = ['(f4[:], f4[:], f4[:], f4[:], f4[:], f4[:])']
else:
    _SIGNATURE = ['(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])']

@guvectorize(_SIGNATURE, '(), (), (j), (k), (l)->()', target=TARGET)
def lookup_vectorized_2d(sample_x, sample_y, flat_hist, bin_edges_x, bin_edges_y, weights):
    """Vectorized gufunc to perform the lookup"""
    sample_x_ = sample_x[0]
    sample_y_ = sample_y[0]

    if (sample_x_ >= bin_edges_x[0]
            and sample_x_ <= bin_edges_x[-1]
            and sample_y_ >= bin_edges_y[0]
            and sample_y_ <= bin_edges_y[-1]):
        idx_x = find_index(sample_x_, bin_edges_x)
        idx_y = find_index(sample_y_, bin_edges_y)
        idx = idx_x*(len(bin_edges_y)-1) + idx_y
        weights[0] = flat_hist[idx]
    else:
        weights[0] = 0.


if FTYPE == np.float32:
    _SIGNATURE = ['(f4[:], f4[:], f4[:, :], f4[:], f4[:], f4[:])']
else:
    _SIGNATURE = ['(f8[:], f8[:], f8[:, :], f8[:], f8[:], f8[:])']

@guvectorize(_SIGNATURE, '(), (), (j, d), (k), (l)->(d)', target=TARGET)
def lookup_vectorized_2d_arrays(sample_x, sample_y, flat_hist, bin_edges_x, bin_edges_y, weights):
    """Vectorized gufunc to perform the lookup while flat hist and weights have
    both a second dimension
    """
    sample_x_ = sample_x[0]
    sample_y_ = sample_y[0]
    if (sample_x_ >= bin_edges_x[0]
            and sample_x_ <= bin_edges_x[-1]
            and sample_y_ >= bin_edges_y[0]
            and sample_y_ <= bin_edges_y[-1]):
        idx_x = find_index(sample_x_, bin_edges_x)
        idx_y = find_index(sample_y_, bin_edges_y)
        idx = idx_x*(len(bin_edges_y)-1) + idx_y
        for i in range(weights.size):
            weights[i] = flat_hist[idx, i]
    else:
        for i in range(weights.size):
            weights[i] = 0.


if FTYPE == np.float32:
    _SIGNATURE = ['(f4[:], f4[:], f4[:], f4[:], f4[:], f4[:], f4[:], f4[:])']
else:
    _SIGNATURE = ['(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])']

@guvectorize(_SIGNATURE, '(), (), (), (j), (k), (l), (m)->()', target=TARGET)
def lookup_vectorized_3d(sample_x, sample_y, sample_z, flat_hist, bin_edges_x, bin_edges_y, bin_edges_z, weights):
    """Vectorized gufunc to perform the lookup"""
    sample_x_ = sample_x[0]
    sample_y_ = sample_y[0]
    sample_z_ = sample_z[0]
    if (sample_x_ >= bin_edges_x[0]
            and sample_x_ <= bin_edges_x[-1]
            and sample_y_ >= bin_edges_y[0]
            and sample_y_ <= bin_edges_y[-1]
            and sample_z_ >= bin_edges_z[0]
            and sample_z_ <= bin_edges_z[-1]):
        idx_x = find_index(sample_x_, bin_edges_x)
        idx_y = find_index(sample_y_, bin_edges_y)
        idx_z = find_index(sample_z_, bin_edges_z)
        idx = (idx_x*(len(bin_edges_y)-1) + idx_y)*(len(bin_edges_z)-1) + idx_z
        weights[0] = flat_hist[idx]
    else:
        weights[0] = 0.


if FTYPE == np.float32:
    _SIGNATURE = ['(f4[:], f4[:], f4[:], f4[:, :], f4[:], f4[:], f4[:], f4[:])']
else:
    _SIGNATURE = ['(f8[:], f8[:], f8[:], f8[:, :], f8[:], f8[:], f8[:], f8[:])']

@guvectorize(_SIGNATURE, '(), (), (), (j, d), (k), (l), (m)->(d)', target=TARGET)
def lookup_vectorized_3d_arrays(sample_x, sample_y, sample_z, flat_hist, bin_edges_x, bin_edges_y, bin_edges_z, weights):
    """Vectorized gufunc to perform the lookup while flat hist and weights have
    both a second dimension"""
    sample_x_ = sample_x[0]
    sample_y_ = sample_y[0]
    sample_z_ = sample_z[0]
    if (sample_x_ >= bin_edges_x[0]
            and sample_x_ <= bin_edges_x[-1]
            and sample_y_ >= bin_edges_y[0]
            and sample_y_ <= bin_edges_y[-1]
            and sample_z_ >= bin_edges_z[0]
            and sample_z_ <= bin_edges_z[-1]):
        idx_x = find_index(sample_x_, bin_edges_x)
        idx_y = find_index(sample_y_, bin_edges_y)
        idx_z = find_index(sample_z_, bin_edges_z)
        idx = (idx_x*(len(bin_edges_y)-1) + idx_y)*(len(bin_edges_z)-1) + idx_z
        for i in range(weights.size):
            weights[i] = flat_hist[idx, i]
    else:
        for i in range(weights.size):
            weights[i] = 0.


def test_histogram():
    """Unit tests for `histogram` function"""
    n_evts = 100
    x = np.arange(n_evts, dtype=FTYPE)
    y = np.arange(n_evts, dtype=FTYPE)
    w = np.ones(n_evts, dtype=FTYPE)

    x = SmartArray(x)
    y = SmartArray(y)
    w = SmartArray(w)

    binning_x = OneDimBinning(name='x', num_bins=10, is_lin=True, domain=[0, 100])
    binning_y = OneDimBinning(name='y', num_bins=10, is_lin=True, domain=[0, 100])
    binning = MultiDimBinning([binning_x, binning_y])

    histo = histogram(sample=[x, y], weights=w, binning=binning, averaged=False).get()
    assert np.array_equal(
        histo.reshape(10, 10),
        np.diag(np.full(shape=10, fill_value=10))
    ), str(histo.reshape(10, 10))

    histo = histogram(sample=[x, y], weights=w, binning=binning, averaged=True).get()
    assert np.array_equal(histo.reshape(10, 10), np.diag(np.ones(10))), str(histo.reshape(10, 10))

    logging.info('<< PASS : test_histogram >>')


def test_find_index():
    """Unit tests for `find_index` function.

    Correctness is defined as, using the result and producing a histogram,
    giving the same histogram as numpy.histogramdd. Additionally, -1 should be
    returned if a value is below the range and num_bins should be returned for
    a value above the range.
    """
    # Negative, positive, integer, non-integer, binary-unrepresentable (0.1) edges
    basic_bin_edges = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 3, 4]

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
            logging.debug("bin_edges being tested: %s", bin_edges)
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
            logging.trace("test_vals = %s", test_vals)

            #
            # Run tests
            #
            for val in test_vals:
                val = FTYPE(val)

                np_histvals, _ = np.histogramdd([val], np.atleast_2d(bin_edges))
                nonzero_indices = np.nonzero(np_histvals)[0]  # select first & only dim
                if np.isnan(val):
                    assert len(nonzero_indices) == 0
                    expected_idx = underflow_idx
                elif val < bin_edges[0]:
                    assert len(nonzero_indices) == 0
                    expected_idx = underflow_idx
                elif val > bin_edges[-1]:
                    assert len(nonzero_indices) == 0
                    expected_idx = overflow_idx
                else:
                    assert len(nonzero_indices) == 1
                    expected_idx = nonzero_indices[0]

                found_idx = find_index(val, bin_edges)

                if found_idx != expected_idx:
                    msg = "val={}, edges={}: Expected idx={}, found idx={}".format(
                        val, bin_edges, expected_idx, found_idx
                    )
                    assert False, msg

    logging.info('<< PASS : test_find_index >>')


if __name__ == '__main__':
    set_verbosity(1)
    test_find_index()
    test_histogram()
