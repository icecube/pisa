"""
Produce a set of transforms mapping true events values (energy and coszen) onto
reconstructed values.

For each bin in true energy and true coszen, a corresponding distribution of
pid, reconstructed energy, and reconstructed coszen is estimated using
variable-bandwidth KDE to characterize this transformation using Monte Carlo
events.
"""


# TODO/BUG: setting any tgt_max_binwidth_factors to 0 can result in no events!
# TODO/BUG: Not regenerating KDEs with new cut applied to events...
# TODO: nutau needn't be treated below 3.5 GeV! ...
# TODO: write "closest bin" as a function
# TODO: muons
# TODO: Grab events from all dimensions but with weighting of which dimension
#       is preferred
# TODO: Figure out a dynamic similarity metric such that the parameters can be
#       figured out by the software, rather than set by the user. E.g., use
#       some statistical clustering technique? See e.g. Scargle's Bayesian
#       Blocks, change point identification routines (RelULSIF, Bayesian, etc.)
# TODO: Move (most/all) functions defined here into module(s) in utils dir
# TODO: Add n_dct as a class instantiation argument?
# TODO: Separate VBWKDE parameters for each dimension, specified either in
#       char_deps_downsampling or as a separate arg?


from __future__ import absolute_import, division

from ast import literal_eval
from collections.abc import Mapping, Sequence
from collections import OrderedDict, namedtuple
from copy import deepcopy
from os import path
from math import exp, log
import threading
import traceback

import numpy as np
from numpy import inf # pylint: disable=unused-import

from pisa import EPSILON, FTYPE, NUMBA_AVAIL, OMP_NUM_THREADS, numba_jit
from pisa.core.binning import MultiDimBinning
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet

from pisa.utils.comparisons import EQUALITY_SIGFIGS, isscalar
from pisa.utils.fileio import mkdir, to_file
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup
from pisa.utils.gaussians import gaussians
from pisa.utils.hash import hash_obj
from pisa.utils.parallel import parallel_run
from pisa.utils.vbwkde import vbwkde as vbwkde_func
from pisa.utils.log import logging, set_verbosity


__all__ = ['KDEProfile', 'collect_enough_events', 'weight_coszen_tails',
           'coszen_error_edges', 'vbwkde']

__author__ = 'J.L. Lanfranchi'

__license__ = '''Copyright (c) 2014-2017, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


KDEProfile = namedtuple('KDEProfile', ['x', 'counts'])
"""namedtuple type for storing the normalized KDE profile: (x, counts)"""


@numba_jit(nopython=True, nogil=True, fastmath=True, cache=True)
def collect_enough_events(values, bin_edges, is_log, min_num_events,
                          tgt_num_events, tgt_max_binwidth_factor):
    """Heuristic to collect enough events close to the provided bin such
    that KDEs can be applied and achieve robust results.

    Events are selected via the logic:
    * If there are `tgt_num_events` or more in the energy bin, simply return
      all events in the bin.
    * Otherwise, go up to `tgt_max_binwidth_factor` times the width of the
      `bin` to find up to `tgt_num_events`. (Note that the distance is
      measured from the bin's weighted center.)
    * However, keep going as far as necessary outside the energy bin (i.e.,
      ignore `tgt_max_binwidth_factor`) to find `min_num_events` events.

    Note that in the above, _all_ events in the bin will _always_ be returned.
    The rest of the algorithm decides how far outside it has to reach to get a
    "desired" number of events (`tgt_num_events`) without going too far, but
    will reach as far as necessary to get the "minimum" number of events
    (`min_num_events').

    Distances are measured on a logarithmic scale, so events that are one-half
    and twice the energy bin's weighted center (respectively) are considered to
    be the same distance to the weighted center.

    In the case of "exact ties" as described above, _both_ events will be kept.
    Therefore, one may get _more_ than `tgt_num_events` when that threshold is
    used or more than `min_num_events` when that threshold is used.

    Finally, note that both edges of the `bin` are _inclusive_, i.e., it
    defines a closed interval.


    Parameters
    ----------
    values : array of floats

    bin_edges : length-2 sequence of floats

    min_num_events : int
        At least this many events will collected, regardless how far outside
        the boundaries of `bin` is necessary to find them.

    tgt_num_events : int >= min_num_events
        Ideally each bin will have this many events. Allow the bin to expand a
        fraction (`tgt_max_binwidth_factor`) to try to hit this number.

    tgt_max_binwidth_factor : float >= 0
        Expand by up to this factor of the width of the bin in order to
        collect tgt_num_events.

    Returns
    -------
    mask : array
        A mask that selects which `values` fulfill the criteria outlined above.

    """
    if is_log:
        bin_width = log(bin_edges[1] / bin_edges[0])
        bin_center = np.log(np.sqrt(bin_edges[0] * bin_edges[1]))
    else:
        bin_width = bin_edges[1] - bin_edges[0]
        bin_center = (bin_edges[0] + bin_edges[1]) / 2

    bin_half_width = bin_width / 2

    n_events = len(values)

    if n_events == 0:
        raise ValueError(
            'No events were found! Do you apply a cut that removes all'
            ' relevant events? E.g. this can occur if you cut away'
            ' downgoing events but then define true-coszen bins in the'
            ' downgoing region.'
        )

    # If either min_num_events or tgt_num_events is greater than the total
    # number of events we have to work with, force to be equal to the number
    # of events we have.
    min_num_events = min_num_events if min_num_events < n_events else n_events
    tgt_num_events = tgt_num_events if tgt_num_events < n_events else n_events

    # Absolute distance from these events to the center of the bin, sorted in
    # ascending order (so events closest to bin center come first)
    if is_log:
        sorted_abs_dist = np.sort(np.abs(np.log(values) - bin_center))
    else:
        sorted_abs_dist = np.sort(np.abs(values - bin_center))

    # Distance from the bin center you have to go to obtain `tgt_num_events`
    tgt_num_events_dist = sorted_abs_dist[tgt_num_events - 1]

    # Maximum distance the  tgt_max_binwidth_factor` allows us to go in order
    # to obtain `tgt_num_events` events
    tgt_max_dist = bin_half_width + bin_width*tgt_max_binwidth_factor

    # Define a single "target" distance taking into consideration that we
    # should neither exceed `tgt_max_dist` nor `tgt_num_events`
    if tgt_num_events_dist < tgt_max_dist:
        tgt_dist = tgt_num_events_dist
    else:
        tgt_dist = tgt_max_dist

    # If the target distance is within the bin, simply take all events from
    # the bin and we're done, as we will get >= tgt_num_events within the bin.
    if tgt_dist <= bin_half_width:
        thresh_dist = bin_half_width

    else:
        # Figure out how far out we have to go to get `min_num_events`
        min_num_events_dist = sorted_abs_dist[min_num_events - 1]

        # If this is _further_ than `tgt_dist`, then we have to suck it up
        # and go `min_num_events_dist` away to ensure we collect enough events
        if min_num_events_dist > tgt_dist:
            thresh_dist = min_num_events_dist

        # But if we can stop at tgt_dist away and get more events than
        # min_num_events, we should do that to maximize our statistics
        else:
            thresh_dist = tgt_dist

    lower_edge = bin_center - thresh_dist
    upper_edge = bin_center + thresh_dist

    if is_log:
        lower_edge = exp(lower_edge)
        upper_edge = exp(upper_edge)

    mask = (values >= lower_edge) & (values <= upper_edge)

    return mask


def inf2finite(x):
    """Convert +/- infinities to largest/smallest representable numbers
    according to the current pisa.FTYPE"""
    return np.clip(x, a_min=np.finfo(FTYPE).min, a_max=np.finfo(FTYPE).max)


@numba_jit(nopython=True, nogil=True, fastmath=True, cache=True)
def weight_coszen_tails(cz_diff, cz_bin_edges, input_weights):
    """Calculate weights that compensate for fewer points in the inherent tails
    of the coszen-difference (usually coszen-error) distribution.

    Parameters
    ----------
    cz_diff : array
        Cosine-zenith differences. E.g., `coszen_reco - coszen_true`

    cz_bin_edges : sequnce of two scalars
        Edges from the true-coszen bin in which the coszen differences were
        computed.

    input_weights : None or array of same size as `cz_diff`
        Existing weights that are to be multiplied by the tail weights to
        arrive at an overall weight for each event. If provided, must have same
        shape as `cz_diff`.

    Returns
    -------
    weights : array
    diff_limits : tuple of two scalars
        (diff_lower_lim, diff_upper_lim)

    """
    new_weights = np.empty_like(cz_diff)
    num_elements = FTYPE(len(cz_diff))

    # Shortcuts for accessing bin edges
    bin_lower_edge = np.min(cz_bin_edges)
    bin_upper_edge = np.max(cz_bin_edges)

    # Identify limits of possible diff distribution
    diff_lower_lim = -1 - bin_upper_edge
    diff_upper_lim = +1 - bin_lower_edge
    diff_limits = (diff_lower_lim, diff_upper_lim)

    # Identify inner limits of the tails
    lower_tail_upper_lim = -1 - bin_lower_edge
    upper_tail_lower_lim = +1 - bin_upper_edge

    # Identify tail widths
    lower_tail_width = lower_tail_upper_lim - diff_lower_lim
    upper_tail_width = diff_upper_lim - upper_tail_lower_lim

    total = 0.0
    if len(input_weights) > 0:
        for n, orig_weight in enumerate(input_weights):
            cz_d = cz_diff[n]
            if cz_d > upper_tail_lower_lim:
                new_weight = (
                    orig_weight * upper_tail_width / (diff_upper_lim - cz_d)
                )
            elif cz_d < lower_tail_upper_lim:
                new_weight = (
                    orig_weight * lower_tail_width / (cz_d - diff_lower_lim)
                )
            else:
                new_weight = orig_weight
            total += new_weight
            new_weights[n] = new_weight
    else:
        for n, cz_d in enumerate(cz_diff):
            if cz_d > upper_tail_lower_lim:
                new_weight = upper_tail_width / (diff_upper_lim - cz_d)
            elif cz_d < lower_tail_upper_lim:
                new_weight = lower_tail_width / (cz_d - diff_lower_lim)
            else:
                new_weight = 1.0
            total += new_weight
            new_weights[n] = new_weight

    norm_factor = num_elements / total

    for n, wt in enumerate(new_weights):
        new_weights[n] = wt * norm_factor

    return new_weights, diff_limits


#@numba_jit(nogil=True, nopython=True, fastmath=True, cache=True)
def coszen_error_edges(true_edges, reco_edges):
    """Return a list of edges in coszen-error space given 2 true-coszen
    edges and reco-coszen edges. Systematics are not implemented at this time.

    Parameters
    ----------
    true_edges : sequence of 2 scalars
    reco_edges : sequence of scalars
    bias : scalar // NOT IMPLEMENTED YET!
    scale : scalar > 0 // NOT IMPLEMENTED YET!

    Returns
    -------
    all_dcz_binedges : list of scalars
        The interleaved coszen-error (delta-cz) bin edges found both from the
        full reco range possible (given by `true_edges` and assuming reco can
        be from -1 to +1), and from the spans due to `reco_edges`.

    reco_indices : tuple with 2 lists of scalars
        Each list contains `len(reco_edges - 1)` scalars. These are the
        indices for locating the edge in `all_binedges`, corresponding to
        `(reco_edges[:-1], reco_edges[1:])`.

    """
    n_reco_edges = len(reco_edges)
    reco_lower_binedges = reco_edges[:-1]
    reco_upper_binedges = reco_edges[1:]
    true_lower_binedge = true_edges[0]
    true_upper_binedge = true_edges[1]
    true_bin_width = abs(true_upper_binedge - true_lower_binedge)
    true_bin_midpoint = (true_lower_binedge + true_upper_binedge) / 2

    full_reco_range_lower_binedge = np.round(
        FTYPE(-1) - true_upper_binedge, np.int64(EQUALITY_SIGFIGS)
    )
    full_reco_range_upper_binedge = np.round(
        FTYPE(+1) - true_lower_binedge, np.int64(EQUALITY_SIGFIGS)
    )

    dcz_lower_binedges = np.round(
        reco_lower_binedges - true_upper_binedge, np.int64(EQUALITY_SIGFIGS)
    )
    dcz_upper_binedges = np.round(
        reco_upper_binedges - true_lower_binedge, np.int64(EQUALITY_SIGFIGS)
    )

    all_dcz_binedges, indices = np.unique(
        np.concatenate([
            [full_reco_range_lower_binedge],
            dcz_lower_binedges,
            dcz_upper_binedges,
            [full_reco_range_upper_binedge]
        ]),
        return_inverse=True
    )

    # Note: first index is for `full_reco_range_lower_binedge`,
    # next `n_reco_edges - 1` edges correspond to `dcz_lower_binedges`,
    # next `n_reco_edges - 1` edges correspond to `dcz_upper_binedges`,
    # and last index is for `full_reco_range_upper_binedge`.
    # We drop the first and last as it is assumed the full range requires all
    # the bins, so indices for indicating the "full range" are trivial (i.e. 0
    # and len(bins) - 1)
    reco_indices = (indices[1:n_reco_edges], indices[n_reco_edges:-1])

    # Note that the final (uppermost) edge doesn't matter as it does
    # not define the *start* of a range.
    num_bins = len(all_dcz_binedges) - 1
    bin_reps = np.empty(shape=num_bins)
    for bin_lower_edge_idx in range(num_bins):
        reps = 0
        for lower_idx, upper_idx in zip(*reco_indices):
            if lower_idx <= bin_lower_edge_idx < upper_idx:
                reps += 1
        bin_reps[bin_lower_edge_idx] = reps

    # Simplistic version of the above, assuming input bin has 0-width
    augmented_reco_binedges, indices = np.unique(
        np.concatenate([
            [FTYPE(-1)],
            reco_edges,
            [FTYPE(+1)]
        ]),
        return_inverse=True
    )
    simple_reco_indices = indices[1:-1]
    simple_dcz_binedges = augmented_reco_binedges - true_bin_midpoint

    results = dict(
        true_bin_width=true_bin_width,
        all_dcz_binedges=all_dcz_binedges,
        reco_indices=reco_indices,
        bin_reps=bin_reps,
        simple_dcz_binedges=simple_dcz_binedges,
        simple_reco_indices=simple_reco_indices
    )

    return results


@numba_jit(nogil=True, nopython=True, fastmath=True, cache=True)
def sorted_fast_histogram(a, bins, weights):
    """Fast but less precise histogramming of a sorted (in ascending order)
    array with weights.

    Note that due to the assumption that `a` is sorted, this histogram function
    is slightly faster and more precise than `fast_histogram` operating on an
    unsorted array.

    Parameters
    ----------
    a : sorted array
    bins : sequence
    weights : array

    Returns
    -------
    hist, bin_edges

    """
    nbins = len(bins) - 1
    ndata = len(a)
    bin_min = bins[0]
    bin_max = bins[nbins]
    hist = np.zeros(nbins, np.float64)

    # Note that initialization of first bin is here, since set is monotonic
    lo = 0
    #for view_a, view_weights in zip(np.nditer(a), np.nditer(weights)):
    for idx in range(ndata):
        v_a = a[idx]
        if not bin_min <= v_a <= bin_max:
            # Value is out of bounds, ignore (this also catches NaNs)
            continue
        # Bisect in bins[:-1]
        hi = nbins - 1
        while lo < hi:
            # Note the `+ 1` is necessary to avoid an infinite
            # loop where mid = lo => lo = mid
            mid = (lo + hi + 1) >> 1
            if v_a < bins[mid]:
                hi = mid - 1
            else:
                lo = mid
        hist[lo] += weights[idx]
    return hist, bins


@numba_jit(nogil=True, nopython=True, fastmath=True, cache=True)
def fast_histogram(a, bins, weights):
    """Fast but less precise histogramming of an array with weights.

    It is recommended that `a` and `weights` be sorted according to ascending
    `weights` to achieve the best numerical precision. This is due to the
    finite-precision effect whereby a small number added to a large number can
    have no effect. Therefore adding a sequence of small numbers, one at a
    time, to a large number also has no effect. However, adding the sequence of
    small numbers together before being added to the large number is more
    likely to account for all of the small numbers.

    Parameters
    ----------
    a : array
    bins : sequence
    weights : array

    Returns
    -------
    hist, bin_edges

    See Also
    --------
    sorted_fast_histogram
        Small speedup and more precision if `a` is sorted

    """
    nbins = len(bins) - 1
    ndata = len(a)
    bin_min = bins[0]
    bin_max = bins[nbins]
    hist = np.zeros(nbins, np.float64)

    # Note that initialization of first bin is here, since set is monotonic
    #for view_a, view_weights in zip(np.nditer(a), np.nditer(weights)):
    for idx in range(ndata):
        v_a = a[idx]
        if not bin_min <= v_a <= bin_max:
            # Value is out of bounds, ignore (this also catches NaNs)
            continue
        # Bisect in bins[:-1]
        lo = 0
        hi = nbins - 1
        while lo < hi:
            # Note the `+ 1` is necessary to avoid an infinite
            # loop where mid = lo => lo = mid
            mid = (lo + hi + 1) >> 1
            if v_a < bins[mid]:
                hi = mid - 1
            else:
                lo = mid
        hist[lo] += weights[idx]
    return hist, bins


if NUMBA_AVAIL:
    HIST_FUNC = fast_histogram
else:
    HIST_FUNC = np.histogram


class vbwkde(Stage): # pylint: disable=invalid-name
    r"""
    From simulated events, a set of transforms are created which map
    bins of true events onto distributions of reconstructed events using
    variable-bandwidth kernel density estimation. These transforms can be
    accessed by [true_energy][true_coszen][reco_energy][reco_coszen].
    These distributions represent the probability that a true event
    (true_energy, true_coszen) with be reconstructed as (pid, reco_energy,
    reco_coszen).

    From these transforms and "true" event rate maps as inputs, calculates
    the reconstructed even rate maps.

    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameters:

        reco_events : string or Events
            PISA Events object or filename/path to use to derive transforms, or
            a string specifying the resource location of the same.

        reco_weights_name : None or string
            Field to use in MC events to apply MC weighting for the computation

        transform_events_keep_criteria : None or string
            Additional cuts that are applied to events prior to computing
            transforms with them. E.g., "true_coszen <= 0" removes all MC-true
            downgoing events. See `pisa.core.events.Events` class for details
            on cut specifications.

        res_scale_ref : string
            One of "mean", "mode", or "zero". This is the reference point about
            which resolutions are scaled. "zero" scales about the zero-error
            point (i.e., the bin midpoint), "mean" scales about the mean of the
            KDE, and "mode" scales about the peak of the KDE.

        e_res_scale : float
            A scaling factor for energy resolutions.

        cz_res_scale : float
            A scaling factor for coszen resolutions.

        e_reco_bias : float

        cz_reco_bias : float

    particles : string
        Must be one of 'neutrinos' or 'muons' (though only neutrinos are
        supported at this time).

    input_names : string or list of strings
        Names of inputs expected. These should follow the standard PISA
        naming conventions for flavor/interaction types OR groupings
        thereof. Note that this service's outputs are named the same as its
        inputs. See Conventions section in the documentation for more info.

    transform_groups : string
        Specifies which particles/interaction types to combine together in
        computing the transforms. See Notes section for more details on how
        to specify this string

    sum_grouped_flavints : bool

    input_binning : MultiDimBinning or convertible thereto
        Input binning is in true variables, with names prefixed by "true_".
        Each must match a corresponding dimension in `output_binning`.

    output_binning : MultiDimBinning or convertible thereto
        Output binning is in reconstructed variables, which can include pid.

    min_num_events : int or mapping
        For KDEs, each bin is expanded up and down in true-energy as much as
        necessary to collect this many events. See `collect_enough_events` for
        more details.

    tgt_num_events : int or mapping
        Ideally each bin will have `tgt_num_events`. Allow the bin to expand up
        to a fraction times the bin's width (`tgt_max_binwidth_factors`) to try
        to hit this number. See `collect_enough_events` for more details.

    tgt_max_binwidth_factors : float
        Expand bin by up to this fraction in order to collect
        `tgt_num_events`. See `collect_enough_events` for more details.

    char_deps_downsampling : mapping
        For each dimension to be characterized, define the dimensions it
        depends upon. I.e., events will be collected in bins with dimensions of
        the defined dependencies. Note that the final dimension upon which
        there is a dependency is _not_ a strict binning. It's a starting event
        set, but in order to hit min_num_events or tgt_num_events, events from
        outside the boundaries of this dimension may be exceeded. Therefore,
        this should be the _least_ correlated with the dimension being
        characterized.

        Dimension names in either `char_binning` or `output_binning` may be
        included as dependencies, but if a binning is defined in both places,
        the binning in `char_binning` will be used.

        E.g.:
        ```
            {
                'pid': 'true_energy',
                'energy': ['pid', 'true_energy'],
                'coszen': ['pid', 'true_coszen', 'true_energy']
            }
        ```

    energy_inbin_smoothing : bool
        Smear energy-error KDE profiles by boxcar funciton of same width as
        input (true-energy) bin it applies to. This is intended to account for
        the ambiguity in where the KDE profile is located when the input bin is
        of finite width, and so the effect of this parameter should decrease
        with smaller `input_binning`.

    coszen_inbin_smoothing : bool
        Smear coszen-error KDE profiles by boxcar function of same width as
        input (true-coszen) bin it applies to. See `energy_inbin_smoothing` for
        more explanation.

    disk_cache : bool, string, etc.
        Simplest to set to True or False for enabling/disabling disk caching,
        respectively, but other inputs are possible (see docs for
        `pisa.core.stage.Stage` class for more info). The KDE profiles are
        cached to disk by this service, _not_ the full transform (since the
        latter can be multiple GB, depending on input/output binning).

    transforms_cache_depth : int >= 0
        Default is 1 since transforms for this service can be huge (gigabytes)

    outputs_cache_depth : int >= 0
        Default is 20 since the outputs from this stage are generally not too
        large.

    memcache_deepcopy : bool

    debug_mode : None, bool, or string
        Whether to run extra checks and store extra debug info:
            debug_mode = None or False
                No debug
            debug_mode = True or any string
                Peform extra value checks (ranges, sums, etc.)
            debug_mode = 'plot'
                Save info that is necessary to debug the VBW KDE to directory
                /tmp/pisa/reco.vbwkde

    Notes
    -----
    The `transform_groups` string is interpreted (and therefore defined) by
    pisa.utils.flavInt.flavint_groups_string. E.g. commonly one might use:

    'nue_cc+nuebar_cc, numu_cc+numubar_cc, nutau_cc+nutaubar_cc, nuall_nc+nuallbar_nc'

    Any particle type not explicitly mentioned in `transform_groups` is taken
    as a singleton group.

    Plus signs add types to a group, while groups are separated by commas.
    Whitespace is ignored.

    Input "true" event rate maps should be binned finely enough that the
    smearing that is removed when we lump all events in the char_binning
    together and KDE the reco error is not an issue.

    """
    def __init__(self, params, particles, input_names, transform_groups,
                 sum_grouped_flavints, input_binning, output_binning,
                 char_deps_downsampling, min_num_events, tgt_num_events,
                 tgt_max_binwidth_factors, energy_inbin_smoothing,
                 coszen_inbin_smoothing,
                 error_method=None,
                 disk_cache=False,
                 transforms_cache_depth=1,
                 outputs_cache_depth=20,
                 memcache_deepcopy=False,
                 debug_mode=None):
        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'reco_events', 'reco_weights_name',
            'transform_events_keep_criteria',
            'res_scale_ref', 'e_res_scale', 'cz_res_scale',
            'e_reco_bias', 'cz_reco_bias'
        )

        #============================================================
        # Parse, translate, normalize, and/or validate init args...
        #============================================================

        # `particles` ...

        assert isinstance(particles, str)
        particles = particles.strip().lower()
        assert particles in ['neutrinos']
        self.particles = particles

        # `transform_groups` ...

        if isinstance(transform_groups, str):
            transform_groups = flavintGroupsFromString(transform_groups)
        elif transform_groups is None:
            transform_groups = flavintGroupsFromString('')
        self.transform_groups = transform_groups

        # `sum_grouped_flavints` ...

        assert isinstance(sum_grouped_flavints, bool)
        if not sum_grouped_flavints:
            raise NotImplementedError(
                'Grouped flavints must be summed at this time, as logic for'
                ' not doing so is not yet implemented.'
            )
        self.sum_grouped_flavints = sum_grouped_flavints

        # `char_deps_downsampling` ...

        if isinstance(char_deps_downsampling, str):
            char_deps_downsampling = literal_eval(char_deps_downsampling)

        new_dict = dict()
        for char_dim_name, deps in char_deps_downsampling.items():
            assert isinstance(char_dim_name, str)
            new_dict[char_dim_name] = OrderedDict()

            if isinstance(deps, str):
                char_deps_downsampling[char_dim_name] = [deps]
                new_dict[char_dim_name][deps] = 1
                continue

            if isinstance(deps, Sequence):
                if (len(deps) == 2 and isinstance(deps[0], str)
                        and isscalar(deps[1])):
                    new_dict[char_dim_name][deps[0]] = deps[1]
                    continue

                for subseq in deps:
                    if isinstance(subseq, str):
                        new_dict[char_dim_name][subseq] = 1
                    elif isinstance(subseq, Sequence):
                        assert isinstance(subseq[0], str)
                        if len(subseq) == 2:
                            assert isscalar(subseq[1])
                            new_dict[char_dim_name][subseq[0]] = subseq[1]
                        else:
                            new_dict[char_dim_name][subseq[0]] = 1
        char_deps_downsampling = new_dict

        # Until `collect_enough_events` is fixed to work with dimensions other
        # than `true_energy` as the final dimension, have to enforce this
        for dim_name, dependencies in char_deps_downsampling.items():
            if dependencies.keys()[-1] != 'true_energy':
                raise NotImplementedError(
                    "Only 'true_energy' can come last in the list of"
                    " dependencies at this time."
                    " (Dimension: %s, dependencies supplied = %s)"
                    % (dim_name, dependencies.keys())
                )

        # `min_num_events` ...

        if isinstance(min_num_events, str):
            min_num_events = literal_eval(min_num_events)

        if isscalar(min_num_events):
            min_num_events = {d: int(min_num_events)
                              for d in char_deps_downsampling.keys()}

        assert isinstance(min_num_events, Mapping), str(min_num_events)

        # `tgt_num_events` ...

        if isinstance(tgt_num_events, str):
            tgt_num_events = literal_eval(tgt_num_events)

        if isscalar(tgt_num_events):
            tgt_num_events = {d: int(tgt_num_events)
                              for d in char_deps_downsampling.keys()}

        assert isinstance(tgt_num_events, Mapping), str(tgt_num_events)

        # `tgt_max_binwidth_factors` ...

        if isinstance(tgt_max_binwidth_factors, str):
            tgt_max_binwidth_factors = literal_eval(tgt_max_binwidth_factors)

        if isscalar(tgt_max_binwidth_factors):
            tgt_max_binwidth_factors = {
                d: float(tgt_max_binwidth_factors)
                for d in char_deps_downsampling.keys()
            }

        assert isinstance(tgt_max_binwidth_factors, Mapping), \
                str(tgt_max_binwidth_factors)

        # `energy_inbin_smoothing` ...

        assert isinstance(energy_inbin_smoothing, bool)
        self.energy_inbin_smoothing = energy_inbin_smoothing

        # `coszen_inbin_smoothing` ...

        assert isinstance(coszen_inbin_smoothing, bool)
        self.coszen_inbin_smoothing = coszen_inbin_smoothing

        # `input_names` ...

        if isinstance(input_names, str):
            input_names = (''.join(input_names.split(' '))).split(',')

        # Define the names of objects expected in inputs and produced as
        # outputs
        if self.particles == 'neutrinos':
            if self.sum_grouped_flavints:
                output_names = [str(g) for g in self.transform_groups]
            else:
                output_names = input_names
        elif self.particles == 'muons':
            raise NotImplementedError('`particles` = %s not implemented.'
                                      % self.particles)
        else:
            raise ValueError('Invalid value for `particles`: "%s"'
                             % self.particles)

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
        super().__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            memcache_deepcopy=memcache_deepcopy,
            input_binning=input_binning,
            output_binning=output_binning,
            debug_mode=debug_mode,
        )

        # We have some number of dimensions to characterize (KDE). Each one of
        # these has "dimension dependencies," whereby we need to isolate events
        # in _those_ dimensions before we can characterize the dimension. By
        # isolating events as such, we reduce or remove correlations among the
        # involved dimensions. Ego we reduce the dimensionality that needs to
        # be characterized and so can get away with fewer events for the
        # characterization. (This also means that we can simply use
        # one-dimensional KDE for the characterization.)

        self.char_binning = dict()
        """dict formatted as `{out_dim.basename: MultiDimBinning}`, used
        for binning events for characterizing resolutions (namely via VBWKDE)
        in the output dimensions"""

        self.min_num_events = dict()
        """min_num_events for each dimension dependency for each dimension
        being characterized"""

        self.tgt_num_events = dict()
        """tgt_num_events for each dimension dependency for each dimension
        being characterized"""

        self.tgt_max_binwidth_factors = dict()
        """tgt_max_binwidth_factors for each dimension dependency for each
        dimension being characterized"""

        # Only characterize dimensions that are to be output
        for out_dim in output_binning.dims:
            char_dim_name = out_dim.basename

            # Populate class attrs with info for this char dimension
            self.min_num_events[char_dim_name] = min_num_events[char_dim_name]
            self.tgt_num_events[char_dim_name] = tgt_num_events[char_dim_name]
            self.tgt_max_binwidth_factors[char_dim_name] = (
                tgt_max_binwidth_factors[char_dim_name]
            )

            # Construct the binning for this char dimension
            dep_binnings = []
            for dep_dim_name, downsamp_factor \
                    in char_deps_downsampling[char_dim_name].items():
                if dep_dim_name in input_binning:
                    dep_binnings.append(
                        input_binning[dep_dim_name].downsample(downsamp_factor)
                    )
                elif dep_dim_name in output_binning:
                    dep_binnings.append(
                        output_binning[dep_dim_name].downsample(downsamp_factor)
                    )
                else:
                    raise ValueError('Dimension "%s" was neither found in'
                                     ' `input_binning` nor in `output_binning`'
                                     % dep_dim_name)

            self.char_binning[char_dim_name] = MultiDimBinning(dep_binnings)

        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('transform_groups')
        self.include_attrs_for_hashes('sum_grouped_flavints')
        self.include_attrs_for_hashes('char_binning')
        self.include_attrs_for_hashes('min_num_events')
        self.include_attrs_for_hashes('tgt_num_events')
        self.include_attrs_for_hashes('tgt_max_binwidth_factors')
        self.include_attrs_for_hashes('energy_inbin_smoothing')
        self.include_attrs_for_hashes('coszen_inbin_smoothing')

        self.kde_profiles = dict()
        """dict containing `KDEProfile`s. Structure is:
            {dim_basename: {flavintgroup: {(coord): (KDEProfile)}}}

        For example:
            {'pid': {
                numu_cc: {
                    (true_energy=0): (x=[...], counts=[...]),
                    (true_energy=1): (x=[...], counts=[...])
                },
                numubar_cc: {
                    (true_energy=0): (x=[...], counts=[...]),
                    (true_energy=1): (x=[...], counts=[...])
                }},
             'energy': {
                numu_cc: {
                    (pid=0, true_energy=0): (x=[...], counts=[...]),
                    (pid=0, true_energy=1): (x=[...], counts=[...]),
                    (pid=1, true_energy=0): (x=[...], counts=[...]),
                    (pid=1, true_energy=1): (x=[...], counts=[...])
                },
                numubar_cc: {
                    (pid=0, true_energy=0): (x=[...], counts=[...]),
                    (pid=0, true_energy=1): (x=[...], counts=[...]),
                    (pid=1, true_energy=0): (x=[...], counts=[...]),
                    (pid=1, true_energy=1): (x=[...], counts=[...])
                }}
            }
        """

        self._kde_profiles_lock = threading.Lock()

        self._kde_hashes = dict()

        self.xform_kernels = dict()
        """Storage of the N-dim smearing kernels, one per flavintgroup"""

        self._xform_kernels_lock = threading.Lock()

        if debug_mode == 'plot':
            self.debug_dir = '/tmp/pisa/reco.vbwkde'
            mkdir(self.debug_dir)


    def validate_binning(self):
        """Require input dimensions of "true_energy" and "true_coszen" (in any
        order).

        Require output dimensions of "reco_energy" and "reco_coszen", and
        optionally allow output dimension of "pid"; can be in any order.

        """
        input_names = set(self.input_binning.names)
        assert input_names == set(['true_energy', 'true_coszen']), \
                str(input_names)

        output_names = set(self.output_binning.names)
        outs1 = set(['reco_energy', 'reco_coszen'])
        outs2 = set(['reco_energy', 'reco_coszen', 'pid'])
        assert output_names == outs1 or output_names == outs2

        input_basenames = set(self.input_binning.basenames)
        output_basenames = set(self.output_binning.basenames)
        for base_d in input_basenames:
            assert base_d in output_basenames

    def _compute_transforms(self):
        """Generate reconstruction smearing kernels by estimating the
        distribution of reconstructed events corresponding to each bin of true
        events using VBW-KDE.

        The resulting transform is an MxN-dimensional histogram, where M is the
        dimensionality of the input binning and N is the dimensionality of the
        output binning. The transform maps the truth bin counts to the
        reconstructed bin counts.

        I.e., for the case of 1D input binning, the i-th element of the
        reconstruction kernel will be a map showing the distribution of events
        over all the reco space from truth bin i. This will be normalised to
        the total number of events in truth bin i.

        """
        self.load_events(self.params.reco_events)
        self.cut_events(self.params.transform_events_keep_criteria)
        self.characterize_resolutions()
        self.generate_all_kernels()

        xforms = []
        for xform_flavints in self.transform_groups:
            xform_input_names = []
            for input_name in self.input_names:
                input_flavs = NuFlavIntGroup(input_name)
                if set(xform_flavints).intersection(input_flavs):
                    xform_input_names.append(input_name)

            for output_name in self.output_names:
                if output_name not in xform_flavints:
                    continue

                logging.trace('  inputs: %s, output: %s, xform: %s',
                              xform_input_names, output_name, xform_flavints)

                xform = BinnedTensorTransform(
                    input_names=xform_input_names,
                    output_name=output_name,
                    input_binning=self.input_binning,
                    output_binning=self.output_binning,
                    xform_array=self.xform_kernels[xform_flavints].hist,
                    sum_inputs=True
                )
                xforms.append(xform)

        return TransformSet(transforms=xforms)

    def characterize_resolutions(self):
        """Compute the KDEs for each (pid, E) bin. If PID is not present, this
        is just (E). The results are propagated to each (pid, E, cz) bin, as
        the transforms are assumed to not be cz-dependent.

        The results are cached to disk and simply loaded from that cache to
        avoid re-computation.

        """
        weights_name = self.params.reco_weights_name.value

        # TODO: add sourcecode hash for pisa.utils.vbwkde module (entire module
        #       is probably safest, due to all the functions there)

        hash_items = [FTYPE, self.source_code_hash, self.events.hash,
                      self.transform_groups, self.particles,
                      self.sum_grouped_flavints]

        # Create a copy of the events sorted according to ascending true_energy
        sorted_events = dict()
        for flavintgroup in self.transform_groups:
            repr_flavint = flavintgroup[0]
            data_node = self.events[repr_flavint]
            sortind = np.argsort(data_node['true_energy'])

            sorted_events[flavintgroup] = {}
            for key, value in data_node.items():
                sorted_events[flavintgroup][key] = value[sortind]

        for char_dim, dep_dims_binning in self.char_binning.items():
            logging.debug('Working on KDE dimension "%s"', char_dim)
            new_hash = hash_obj(deepcopy(hash_items) + [dep_dims_binning.hash])

            # See if we already have correct kde_profiles for this dim
            if (char_dim in self._kde_hashes
                    and new_hash == self._kde_hashes[char_dim]):
                logging.debug('  > Already have KDEs for "%s"', char_dim)
                continue

            # Try to load from disk cache
            if self.disk_cache is not None:
                try:
                    if new_hash in self.disk_cache:
                        logging.debug(
                            '  > Loading KDEs for "%s" from disk cache',
                            char_dim
                        )
                        self.kde_profiles[char_dim] = self.disk_cache[new_hash]
                        self._kde_hashes[char_dim] = new_hash
                        continue
                except Exception:
                    logging.error('Loading from disk cache failed.')

            # Reset the hash for this dim so if anything fails below, the wrong
            # info won't be loaded
            self._kde_hashes[char_dim] = None

            # Clear out all previous kde info
            self.kde_profiles[char_dim] = OrderedDict()
            for flavintgroup in self.transform_groups:
                self.kde_profiles[char_dim][flavintgroup] = OrderedDict()

            if FTYPE == np.float64:
                ftype_bytes = 8
            elif FTYPE == np.float32:
                ftype_bytes = 4

            sizeof_kde_profiles = 0

            for bin_num, bin_binning in enumerate(dep_dims_binning.iterbins()):
                bin_dims = bin_binning.dims
                bin_coord = dep_dims_binning.index2coord(bin_num)
                logging.debug('  > characterizing bin %s (%d of %d)',
                              bin_coord, bin_num+1, dep_dims_binning.size)

                # Formulate a single cut string that can be evaluated for
                # each flavintgroup
                criteria = []
                for dim in bin_dims[:-1]:
                    dim_name = dim.name
                    criteria.append(dim.inbounds_criteria.replace(
                        dim_name, 'flav_events["%s"]' % dim_name
                    ))
                crit_str = (' & '.join(criteria)).strip()

                last_dim = bin_dims[-1]
                last_dim_bin_edges = last_dim.bin_edges.m
                last_dim_name = last_dim.name
                last_dim_is_log = last_dim.is_log

                for flavintgroup in self.transform_groups:
                    logging.trace('    > flavintgroup = %s', flavintgroup)

                    flav_events = sorted_events[flavintgroup]
                    if crit_str:
                        try:
                            mask1 = eval(crit_str)
                        except:
                            logging.error(
                                'Failed during eval of the string "%s"',
                                crit_str
                            )
                            raise
                    else:
                        mask1 = slice(None)

                    values = flav_events[last_dim_name][mask1]
                    mask2 = collect_enough_events(
                        values=values,
                        bin_edges=last_dim_bin_edges,
                        is_log=last_dim_is_log,
                        min_num_events=self.min_num_events[char_dim],
                        tgt_num_events=self.tgt_num_events[char_dim],
                        tgt_max_binwidth_factor=self.tgt_max_binwidth_factors[char_dim]
                    )
                    logging.trace(
                        '  ... total %d values strictly in dim(s) excluding'
                        ' last dim; selected %d to characterize in-bin res.',
                        len(values), np.sum(mask2)
                    )

                    weights = None
                    if weights_name in flav_events.keys():
                        weights = flav_events[weights_name][mask1][mask2]
                        assert len(weights) > 0
                        weights_total = np.sum(weights)
                        if weights_total != 0:
                            weights = weights * (len(weights)/weights_total)
                        else:
                            weights = None

                    if char_dim == 'pid':
                        feature = flav_events['pid'][mask1][mask2]
                        fmin, fmax = min(feature), max(feature)
                        half_width = (fmax - fmin)/2
                        lowerlim = fmin - half_width
                        upperlim = fmax + half_width
                        vbwkde_kwargs = dict(
                            n_dct=int(2**6),
                            min=lowerlim, max=upperlim,
                            evaluate_at=np.linspace(lowerlim, upperlim,
                                                    int(1e4))
                        )

                    elif char_dim == 'energy':
                        feature = np.log(
                            flav_events['reco_energy'][mask1][mask2]
                            / flav_events['true_energy'][mask1][mask2]
                        )

                        fmin, fmax = min(feature), max(feature)
                        lowerlim = fmin
                        upperlim = fmax
                        # Note that this only evaluates the KDE profile within
                        # the range of datapoints, so as to not extrapolate
                        vbwkde_kwargs = dict(
                            n_dct=int(2**6),
                            min=lowerlim, max=upperlim,
                            evaluate_at=np.linspace(lowerlim, upperlim,
                                                    int(1e4))
                        )

                    elif char_dim == 'coszen':
                        feature = (flav_events['reco_coszen'][mask1][mask2]
                                   - flav_events['true_coszen'][mask1][mask2])
                        if weights is not None:
                            w = weights
                        else:
                            w = np.array([], dtype=FTYPE)
                        weights, error_limits = weight_coszen_tails(
                            cz_diff=feature,
                            cz_bin_edges=bin_binning.true_coszen.bin_edges.m,
                            input_weights=w
                        )

                        # TODO: try the following to fix the tails falling off
                        # too abruptly:
                        # 1. Simply mirror half the points about the error
                        #    limits, KDE, then take the central portion
                        # 2. Mirror about mode (but only place "new" datapoints
                        #    *outside* the current error limits)
                        # 3. Evaluate KDE as now, but evaluate a range outside
                        #    the allowed limits; fold the shapes in by
                        #    reflecting at the limits and adding this in.

                        # Trying combination of methods 1+3 now: compute
                        # bandwidths with half of dataset mirrored about upper
                        # limit, and half mirrored about lower limit. Then only
                        # evaluate gaussians attached datapoints within the
                        # limits, but fold their tails in at the limits & sum

                        error_width = error_limits[1] - error_limits[0]
                        lower_mask = feature <= error_limits[0] + error_width/2
                        upper_mask = feature > error_limits[0] + error_width/2

                        orig_feature = feature
                        orig_weights = weights

                        feature_to_cat = [orig_feature]
                        weights_to_cat = [orig_weights]
                        if np.sum(lower_mask) > 0:
                            feature_to_cat.append(
                                2*error_limits[0] - feature[lower_mask]
                            )
                            weights_to_cat.append(weights[lower_mask])
                        if np.sum(upper_mask) > 0:
                            feature_to_cat.append(
                                2*error_limits[1] - feature[upper_mask]
                            )
                            weights_to_cat.append(weights[upper_mask])
                        feature = np.concatenate(feature_to_cat)
                        weights = np.concatenate(weights_to_cat)

                        extended_lower_lim = error_limits[0] - 0.5*error_width
                        extended_upper_lim = error_limits[1] + 0.5*error_width

                        vbwkde_kwargs = dict(
                            n_dct=int(2**6),
                            min=extended_lower_lim,
                            max=extended_upper_lim,
                            evaluate_dens=False,
                            evaluate_at=None
                        )

                    else:
                        raise NotImplementedError(
                            'Applying KDEs to dimension "%s" is not'
                            ' implemented.' % char_dim
                        )

                    bw, x, counts = vbwkde_func(
                        feature, weights=weights, **vbwkde_kwargs
                    )

                    if char_dim == 'coszen':
                        x = np.linspace(extended_lower_lim, extended_upper_lim,
                                        2e4)
                        counts = gaussians(x=x, mu=orig_feature,
                                           sigma=bw[:len(orig_feature)],
                                           weights=orig_weights)

                        mirrored_length = len(x) // 4
                        below_range_mask = x < error_limits[0]
                        above_range_mask = x > error_limits[1]
                        in_range_mask = ~(below_range_mask | above_range_mask)

                        x_in_range = x[in_range_mask]
                        counts_in_range = counts[in_range_mask]

                        counts_in_range[:mirrored_length] += (
                            counts[below_range_mask][-mirrored_length:][::-1]
                        )
                        counts_in_range[-mirrored_length:] += (
                            counts[above_range_mask][:mirrored_length][::-1]
                        )

                        x = x_in_range
                        counts = counts_in_range

                    # NOTE: removed this sort such that convolution version of
                    # the code works (which assumes profile counts are sorted
                    # by x).

                    ## Sort according to ascending weight to improve numerical
                    ## precision of "poor-man's" histogram
                    #sortind = counts.argsort()
                    #x = x[sortind]
                    #counts = counts[sortind]

                    self.kde_profiles[char_dim][flavintgroup][bin_coord] = (
                        KDEProfile(x=x, counts=counts)
                    )

                    sizeof_kde_profiles += 2*len(x) * ftype_bytes

                    if self.debug_mode == 'plot':
                        info = dict(
                            bin_binning=bin_binning,
                            x=x,
                            counts=counts,
                            weights=weights,
                            feature=feature
                        )
                        debug_info_basename = path.join(
                            self.debug_dir,
                            'profile_%s_%s_%s' % (char_dim, flavintgroup,
                                                  bin_coord)
                        )
                        to_file(obj=info, fname=debug_info_basename + '.pkl')

            self._kde_hashes[char_dim] = new_hash

            if self.disk_cache is not None:
                try:
                    self.disk_cache[new_hash] = self.kde_profiles[char_dim]
                except Exception as exc:
                    logging.error(
                        'Failed to write KDE profiles for dimension %s'
                        ' (%d bytes) to disk cache. To debug issue, see'
                        ' exception message below.',
                        char_dim, sizeof_kde_profiles
                    )
                    traceback.format_exc()
                    logging.exception(exc)
                    logging.warning('Proceeding without disk caching.')

    def generate_all_kernels(self):
        """Dispatches `generate_single_kernel` for all specified transform
        flavintgroups, in parallel if that is possible."""
        if not NUMBA_AVAIL or OMP_NUM_THREADS == 1:
            for flavintgroup in self.transform_groups:
                self.generate_single_kernel(flavintgroup)
            return

        parallel_run(
            func=self.generate_single_kernel,
            kind='threads',
            num_parallel=OMP_NUM_THREADS,
            scalar_func=True,
            divided_args_mask=None,
            divided_kwargs_names=['flavintgroup'],
            flavintgroup=self.transform_groups
        )

    def generate_single_kernel(self, flavintgroup):
        """Construct a smearing kernel for the flavintgroup specified.

        The resulting array can be indexed for clarity using two indexes,
        one for input and one for output dimensions. E.g.:
           kernel[true_energy_i, true_coszen_j][reco_energy_k, reco_coszen_l]
        or if PID is included,
           kernel[true_energy_i, true_coszen_j][reco_energy_k, reco_coszen_l, pid_m]
        where ordering within the two sets of brackets is arbitrary.

        In other words, the indices point from a single MC-true histogram bin
        indexed by (i,j) to a single reco histogram bin indexed by (k,l{,m}).

        Parameters
        ----------
        flavintgroup

        Returns
        -------
        kernel : MxN-dimensional array of float
            Mapping from the number of events in each bin of the 2D
            MC-true-events histogram to the number of events reconstructed in
            each bin of the 2D reconstructed-events histogram. Dimensions are
                input_binning.shape x output_binning.shape
            Note that PID binning can optionally be included, and behaves as
            another output (reco) dimension.

        """
        logging.debug('Generating smearing kernel for %s', flavintgroup)

        pid_kde_profiles = self.kde_profiles['pid'][flavintgroup]
        e_kde_profiles = self.kde_profiles['energy'][flavintgroup]
        dcz_kde_profiles = self.kde_profiles['coszen'][flavintgroup]

        char_binning = self.char_binning

        # Events in each input bin will be smeared out into all output bins.
        # To characterize the smearing, get the KDE profile from each input
        # dimension that was created from events closest to this input bin.

        kernel_binning = self.input_binning * self.output_binning
        kernel = kernel_binning.empty(name='kernel')

        # Shortcut names
        true_energy = self.input_binning.true_energy
        true_coszen = self.input_binning.true_coszen
        reco_energy = self.output_binning.reco_energy
        reco_coszen = self.output_binning.reco_coszen
        pid = self.output_binning.pid

        # Get the following once so we don't have to repeat within the loops

        pid_kde_e_centers = inf2finite(
            char_binning['pid'].true_energy.weighted_centers.m
        )
        e_kde_e_centers = inf2finite(
            char_binning['energy'].true_energy.weighted_centers.m
        )
        cz_kde_e_centers = char_binning['coszen'].true_energy.weighted_centers.m
        cz_kde_cz_centers = (
            char_binning['coszen'].true_coszen.weighted_centers.m
        )

        num_pid_bins = len(pid)

        e_res_scale = self.params.e_res_scale.value.m

        # TODO: implement these systematics
        cz_res_scale = self.params.cz_res_scale.value.m
        cz_reco_bias = self.params.cz_reco_bias.value.m
        if cz_res_scale != 1 or cz_reco_bias != 0:
            raise NotImplementedError(
                'cz_res_scale and cz_reco_bias systematics are not implemented'
                ' yet, so must be fixed at 1 and 0, respectively.'
            )

        pid_edges = inf2finite(pid.bin_edges.m)

        # NOTE: when we get scaling-about-the-mode working, will have to change
        # this.
        reco_e_edges = (
            np.log(inf2finite(reco_energy.bin_edges.m)
                   - self.params.e_reco_bias.value.m)
            / e_res_scale
        )

        reco_cz_edges = np.asarray(reco_coszen.bin_edges.m, dtype=FTYPE)
        if self.debug_mode:
            assert np.all(np.isfinite(reco_cz_edges)), str(reco_cz_edges)

        # Compute info for the delta-coszen bin edges, for each true-coszen
        # input bin and each reco-coszen output bin; also get the bin edges
        # across the entire _possible_ reco-coszen range, for purposes of
        # normalization.

        # NOTE: reco-coszen systematics will be applied to the reco-coszen
        # bin edges, but (for proper treatment) must know the mode location of
        # the coszen KDE profile (which is only known in the innermost loop).
        # The bias isn't costly to do, but scaling will require shifting the
        # mode to 0, applying the scale factor, and then shifting back (or
        # shifting by the bias). So we need to compute as much here as possible
        # (since it's relatively cheap) but we may still need to do some of the
        # work within the innermost loop (yuck).

        true_e_centers = true_energy.weighted_centers.m
        true_cz_centers = true_coszen.weighted_centers.m

        allbins_dcz_edge_info = []
        cz_closest_cz_indices = []
        for center, true_edgetuple in zip(true_cz_centers,
                                          true_coszen.iteredgetuples()):
            allbins_dcz_edge_info.append(
                coszen_error_edges(
                    true_edges=np.asarray(true_edgetuple, dtype=FTYPE),
                    reco_edges=reco_cz_edges
                )
            )

            cz_closest_cz_indices.append(
                np.argmin(np.abs(center - cz_kde_cz_centers))
            )

        for true_e_bin_num, (true_e_center, true_e_edges) \
                in enumerate(zip(true_e_centers, true_energy.iteredgetuples())):
            logging.debug('  > Working on true_e_bin_num %d of %d',
                          true_e_bin_num+1, true_energy.size)

            true_e_bin_width = np.log(true_e_edges[1] / true_e_edges[0])

            idx = np.argmin(np.abs(np.log(true_e_center/pid_kde_e_centers)))
            pid_closest_kde_coord = char_binning['pid'].coord(true_energy=idx)

            # Figure out PID fractions
            pid_kde_profile = pid_kde_profiles[pid_closest_kde_coord]

            pid_total = np.sum(pid_kde_profile.counts)
            if pid_total == 0:
                pid_fractions = np.zeros(size=len(pid_edges) - 1, dtype=FTYPE)
                logging.warning('Zero events in PID bin!')
            else:
                pid_norm = 1 / pid_total
                pid_counts, _ = HIST_FUNC(
                    pid_kde_profile.x, weights=pid_kde_profile.counts,
                    bins=pid_edges
                )
                pid_fractions = pid_norm * pid_counts

            if self.debug_mode:
                assert np.all(pid_fractions >= 0), str(pid_fractions)
                assert np.all(pid_fractions <= 1), str(pid_fractions)
                assert np.sum(pid_fractions) < 1 + 10*EPSILON, \
                        str(pid_fractions)

            for pid_bin_num in range(num_pid_bins):
                pid_fraction = pid_fractions[pid_bin_num]

                energy_indexer = kernel_binning.indexer(
                    true_energy=true_e_bin_num,
                    pid=pid_bin_num
                )

                # If PID is zero, no need to figure out anything further for
                # (..., true_energy=this, ..., pid=this, ...) bins
                if not np.any(np.abs(pid_fraction) > EPSILON):
                    kernel[energy_indexer] = 0
                    continue

                # Get the energy smearing for this (PID, true-energy) bin
                closest_e_idx = np.argmin(
                    np.abs(np.log(true_e_center / e_kde_e_centers))
                )
                e_closest_kde_coord = char_binning['energy'].coord(
                    pid=pid_bin_num,
                    true_energy=closest_e_idx
                )
                e_kde_profile = e_kde_profiles[e_closest_kde_coord]

                energy_total = np.sum(e_kde_profile.counts)

                if energy_total == 0:
                    reco_energy_fractions = np.zeros(
                        shape=len(reco_e_edges) - 1,
                        dtype=FTYPE
                    )
                    logging.warning('Zero events in energy bin!')
                else:
                    # TODO: scale about the mode of the KDE! i.e., implement
                    #       `res_scale_ref`

                    # Tranform reco-energy bin edges into the log-ratio space
                    # where we characterized the energy resolutions
                    e_err_edges = (
                        reco_e_edges - np.log(true_e_center)/e_res_scale
                    )

                    if self.energy_inbin_smoothing:
                        # Formulate a "boxcar" function for convolution...
                        x_range = (
                            np.max(e_kde_profile.x) - np.min(e_kde_profile.x)
                        )
                        num_x = len(e_kde_profile.x)
                        dx = x_range / (num_x - 1)
                        boxcar_width = int(np.ceil(true_e_bin_width / dx))
                        boxcar = np.full(shape=boxcar_width,
                                         fill_value=1/boxcar_width)

                        # Convolve profile with input bin (logarithmic) width
                        # to account for uncertainty in where event is located
                        # within the input bin
                        y = np.convolve(e_kde_profile.counts, boxcar,
                                        mode='same')
                    else:
                        y = e_kde_profile.counts

                    energy_norm = 1 / energy_total

                    reco_energy_counts, _ = HIST_FUNC(
                        e_kde_profile.x, weights=y,
                        bins=e_err_edges
                    )
                    reco_energy_fractions = energy_norm * reco_energy_counts

                    if self.debug_mode:
                        assert np.all(reco_energy_fractions >= 0), \
                                str(reco_energy_fractions)
                        assert np.all(reco_energy_fractions <= 1), \
                                str(reco_energy_fractions)
                        assert np.sum(reco_energy_fractions < 1 + EPSILON), \
                                str(reco_energy_fractions)

                # pid and true_energy are covered by the `energy_indexer`;
                # then we broadcast reco_energy to
                # (true_coszen, reco_coszen, reco_energy)
                kernel[energy_indexer] = kernel_binning.broadcast(
                    pid_fraction * reco_energy_fractions,
                    from_dim='reco_energy',
                    to_dims=['true_coszen', 'reco_coszen']
                )

                # Do this just once for the energy bin, prior to looping over
                # coszen
                cz_closest_e_idx = np.argmin(np.abs(
                    np.log(true_e_center / cz_kde_e_centers)
                ))

                # Get closest coszen smearing for (PID, true-cz, true-E) bin

                # TODO: implement `res_scale_ref` and `cz_reco_bias`! Note that
                # this is why `true_cz_lower_edge` and `true_cz_upper_edge` are
                # enumerated over in the below loop, since these will be
                # necessary to implement the systamtic(s).
                for true_cz_bin_num, cz_closest_cz_idx \
                        in enumerate(cz_closest_cz_indices):
                    cz_closest_kde_coord = char_binning['coszen'].coord(
                        pid=pid_bin_num,
                        true_coszen=cz_closest_cz_idx,
                        true_energy=cz_closest_e_idx
                    )

                    # Get KDE profile (in "delta-cz" space)
                    dcz_kde_profile = dcz_kde_profiles[cz_closest_kde_coord]
                    dcz_edge_info = allbins_dcz_edge_info[true_cz_bin_num]

                    if self.coszen_inbin_smoothing:
                        x_range = (np.max(dcz_kde_profile.x)
                                   - np.min(dcz_kde_profile.x))
                        num_x = len(dcz_kde_profile.x)
                        dx = x_range / (num_x - 1)

                        boxcar_width = int(
                            np.ceil(dcz_edge_info['true_bin_width'] / dx)
                        )
                        boxcar = np.full(shape=boxcar_width,
                                         fill_value=1/boxcar_width)
                        y = np.convolve(dcz_kde_profile.counts, boxcar,
                                        mode='same')
                    else:
                        y = dcz_kde_profile.counts

                    reco_coszen_counts, _ = HIST_FUNC(
                        dcz_kde_profile.x, weights=y,
                        #bins=dcz_edge_info['all_dcz_binedges']
                        bins=dcz_edge_info['simple_dcz_binedges']
                    )
                    #reco_coszen_counts /= dcz_edge_info['bin_reps']

                    if self.debug_mode:
                        logging.debug(
                            'true_cz_bin_num = %d, cz_closest_kde_coord = %s',
                            true_cz_bin_num, cz_closest_kde_coord,
                        )

                    ## Collect the relevant hist sections to describe each
                    ## quantity of interest, starting with normalization
                    #reco_indices = dcz_edge_info['cz_reco_indices']

                    ##coszen_norm = 1 / np.sum(
                    ##    reco_coszen_counts #* dcz_edge_info['bin_reps']
                    ##)

                    #reco_coszen_fractions = []
                    #for reco_lower, reco_upper in zip(*reco_indices):
                    #    reco_coszen_fractions.append(
                    #        np.sum(reco_coszen_counts[reco_lower:reco_upper])
                    #    )
                    #reco_coszen_fractions = np.asarray(reco_coszen_fractions,
                    #                                   dtype=FTYPE)
                    #reco_coszen_fractions /= np.sum(reco_coszen_fractions)

                    reco_coszen_total = np.sum(reco_coszen_counts)
                    coszen_norm = 1 / reco_coszen_total
                    # Note that the indices are into bin _edges_, so to convert
                    # to bin number, leave off the right-most index
                    use_indices = dcz_edge_info['simple_reco_indices'][:-1]
                    reco_coszen_fractions = (
                        reco_coszen_counts[use_indices] * coszen_norm
                    )

                    if self.debug_mode:
                        assert np.all(reco_coszen_fractions <= 1), \
                                str(reco_coszen_fractions)
                        assert np.all(reco_coszen_fractions >= 0), \
                                str(reco_coszen_fractions)
                        assert np.sum(reco_coszen_fractions) <= 1 + EPSILON, \
                                str(reco_coszen_fractions)

                    # Here we index directly into (i.e. the smearing profile
                    # applies direclty to) a single
                    # `(true_energy, true_coszen, pid)` coordinate.
                    coszen_indexer = kernel_binning.indexer(
                        true_energy=true_e_bin_num,
                        true_coszen=true_cz_bin_num,
                        pid=pid_bin_num
                    )

                    # At that coordinate, we broadcast the information from
                    # the `reco_coszen` dimension into the entire `reco_energy`
                    # dimension.
                    kernel.hist[coszen_indexer] *= kernel_binning.broadcast(
                        reco_coszen_fractions,
                        from_dim='reco_coszen',
                        to_dims=['reco_energy']
                    )

        with self._xform_kernels_lock:
            self.xform_kernels[flavintgroup] = kernel


def test_coszen_error_edges():
    """Unit tests for function coszen_error_edges"""
    true_edges = np.array([-0.55, -0.50], dtype=FTYPE)
    true_lower_edge, true_upper_edge = true_edges
    reco_edges = np.linspace(-0.1, 0.0, 5, dtype=FTYPE)

    edge_info = coszen_error_edges(true_edges, reco_edges)
    all_dcz_binedges = edge_info['all_dcz_binedges']
    reco_indices = edge_info['reco_indices']
    bin_reps = edge_info['bin_reps']
    simple_dcz_binedges = edge_info['simple_dcz_binedges']
    simple_reco_indices = edge_info['simple_reco_indices']

    logging.debug('true_edges:\n%s', true_edges)
    logging.debug('reco_edges:\n%s', reco_edges)
    logging.debug('all_dcz_binedges:\n%s', all_dcz_binedges)
    logging.debug('simple_dcz_binedges:\n%s', simple_dcz_binedges)
    logging.debug('simple_reco_indices:\n%s', simple_reco_indices)
    logging.debug('zip(*reco_indices):\n%s', list(zip(*reco_indices)))
    logging.debug('bin_reps:\n%s', bin_reps)
    for reco_lower_edge, reco_upper_edge, lower_idx, upper_idx \
            in zip(reco_edges[:-1], reco_edges[1:], *reco_indices):
        dcz_lower_edge = all_dcz_binedges[lower_idx]
        dcz_upper_edge = all_dcz_binedges[upper_idx]
        dcz_lower_edge_ref = reco_lower_edge - true_upper_edge
        dcz_upper_edge_ref = reco_upper_edge - true_lower_edge
        assert np.isclose(dcz_lower_edge, dcz_lower_edge_ref,
                          rtol=EQUALITY_SIGFIGS)
        assert np.isclose(dcz_upper_edge, dcz_upper_edge_ref,
                          rtol=EQUALITY_SIGFIGS)
    logging.info('<< PASS : test_coszen_error_edges >>')


if __name__ == '__main__':
    set_verbosity(2)
    test_coszen_error_edges()
