"""
Map class to contain 2D histogram, error, and metadata about the contents.
MapSet class to contain a set of maps.

Also provide basic mathematical operations that user applies directly to the
containers but that get passed down to operate on the contained data.
"""


from __future__ import absolute_import, division

from collections.abc import Iterable, Mapping, Sequence
from collections import OrderedDict 
from copy import deepcopy, copy
from fnmatch import fnmatch
from functools import reduce
from itertools import permutations
from operator import add, getitem, setitem
import os
import re
import shutil
import tempfile

from decorator import decorate
import numpy as np
from scipy.stats import poisson, norm
import uncertainties
from uncertainties import ufloat
from uncertainties import unumpy as unp

from pisa import ureg, HASH_SIGFIGS
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.comparisons import normQuant, recursiveEquality
from pisa.utils.flavInt import NuFlavIntGroup
from pisa.utils.hash import hash_obj
from pisa.utils import jsons
from pisa.utils.fileio import get_valid_filename, mkdir
from pisa.utils.format import (make_valid_python_name, strip_outer_dollars,
                               text2tex)
from pisa.utils.log import logging, set_verbosity
from pisa.utils.random_numbers import get_random_state
from pisa.utils import stats


__all__ = ['type_error', 'reduceToHist', 'rebin', 'valid_nominal_values',
           'Map', 'MapSet', 'test_Map', 'test_MapSet']

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


# TODO: inconsistent treatment of metrics in *chi2*, *llh*, and metric* methods
# (Issue #264: https://github.com/jllanfranchi/pisa/issues/264)

# TODO: make sure logic works for PID-separated-maps as well as
# PID-as-binning-dimension maps

# TODO: CUDA and numba implementations of rebin if these libs are available

# TODO: move these utilities functions to a generic utils module?

def type_error(value):
    """Generic formulation of a TypeError that can be called throughout the
    code"""
    raise TypeError('Type of argument not supported: "%s"'
                    % value.__class__.__name__)


def reduceToHist(obj):
    """Recursively sum to reduce an object to a single histogram.

    Parameters
    ----------
    obj : numpy.ndarray, Map, MapSet, or iterable of MapSets

    Returns
    -------
    hist : numpy.ndarray
        Single histogram version of `obj`

    Raises
    ------
    TypeError if `obj` is an unhandled type

    """
    if isinstance(obj, np.ndarray):
        hist = obj
    elif isinstance(obj, Map):
        hist = obj.hist
    elif isinstance(obj, MapSet):
        hist = sum(obj).hist
    elif isinstance(obj, Iterable):
        hist = sum([reduceToHist(x) for x in obj])
    else:
        raise TypeError('Unhandled type for `obj`: %s' % type(obj))
    return hist


def rebin(hist, orig_binning, new_binning, normalize_values=True):
    """Rebin a histogram.

    Note that the new binning's edges must be a subset of the original
    binning's edges (i.e. no sub-division or extrapolation of bins is
    implemented).

    Parameters
    ----------
    hist : numpy.ndarray
        Array containing the (original) histogram's entries

    orig_binning : MultiDimBinning
        Original binning

    new_binning : MultiDimBinning
        Desired binning, where `new_binning.bin_edges` must be a subset of
        `orig_binning.bin_edges`.

    normalize_values : bool
        Whether to apply `pisa.utils.comparisons.normQuant` to the bin edges
        prior to comparing `new_binning` to `orig_binning`. This is
        computationally expensive but ensures similar binnings and eqivalent
        units do not cause erroneous results. It is recommended to set
        `normalize_values=True` unless you know the two binning specs are
        consistently defined.

    Returns
    -------
    new_hist : numpy.ndarray
        New histogram rebinned from `hist`

    """
    if set(new_binning.basenames) != set(orig_binning.basenames):
        raise ValueError(
            "`new_binning` dimensions' basenames %s do not have 1:1"
            " correspondence (modulo pre/suffixes) to original binning"
            " dimensions' basenames %s"
            % (new_binning.basenames, orig_binning.basenames)
        )

    if orig_binning.edges_hash == new_binning.edges_hash:
        return hist

    orig_dim_indices = []
    new_dim_indices = []
    for new_dim_idx, new_dim in enumerate(new_binning):
        orig_dim_idx = orig_binning.index(new_dim.name, use_basenames=False)

        new_dim_indices.append(new_dim_idx)
        orig_dim_indices.append(orig_dim_idx)

        orig_dim = orig_binning.dimensions[orig_dim_idx]

        if normalize_values:
            orig_edges = normQuant(orig_dim.bin_edges, sigfigs=HASH_SIGFIGS)
            new_edges = normQuant(new_dim.bin_edges, sigfigs=HASH_SIGFIGS)
        else:
            orig_edges = orig_dim.bin_edges
            new_edges = new_dim.bin_edges
        if not np.all(new_edges == orig_edges):
            orig_edge_idx = np.array([np.where(orig_edges == n)
                                      for n in new_edges]).ravel()
            hist = np.add.reduceat(hist, orig_edge_idx[:-1],
                                   axis=orig_dim_idx)

    new_hist = np.moveaxis(hist, source=orig_dim_indices,
                           destination=new_dim_indices)

    return new_hist


def _new_obj(original_function):
    """Decorator to deepcopy unaltered states into new Map object."""
    def new_function(*args, **kwargs):
        """Augmented function to replace `original_function`. Note that this
        docstring and the function signature will be overwritten by those from
        `original_function` upon the call to `decorate` below."""
        # pylint: disable=protected-access
        func = args[0]
        self = args[1]
        args = args[2:]
        new_state = OrderedDict()
        state_updates = func(self, *args, **kwargs)
        for slot in self._state_attrs:
            if state_updates is not None and slot in state_updates:
                new_state[slot] = state_updates[slot]
            else:
                new_state[slot] = deepcopy(getattr(self, slot))
        if len(new_state['binning']) == 0:
            return np.asscalar(new_state['hist'])
        return Map(**new_state)
    return decorate(original_function, new_function)



def valid_nominal_values(data_array):
    """Get the the nominal values that are valid for an array"""
    return np.ma.masked_invalid(unp.nominal_values(data_array))


# TODO: implement strategies for decreasing dimensionality (i.e.
# projecting map onto subset of dimensions in the original map)

# TODO: Should all calls to np.<...> be replaced with unp.<...> as is done for
# unp.sqrt below?

class Map(object):
    """Class to contain a multi-dimensional histogram, error, and metadata
    about the histogram. Also provides basic mathematical operations for the
    contained data. See Examples below for how to use a Map object.


    Parameters
    ----------
    name : string
        Name for the map. Used to identify the map.

    hist : numpy.ndarray (incl. obj array from uncertainties.unumpy.uarray)
        The "data" (counts, etc.) in the map.  The shape of `hist` must be
        compatible with the `binning` specified.

    binning : MultiDimBinning
        Describes the binning of the Map.

    error_hist : numpy ndarray
        Must be same shape as `hist`. If specified, sets the error standard
        deviations for the contained `hist`, replacing any stddev information
        that might be contained in the passed `hist` arg.

    hash : None, or immutable object (typically an integer)
        Hash value to attach to the map.

    tex : None or string
        TeX string that can be used for e.g. plotting.

    full_comparison : bool
        Whether to perform full (recursive) comparisons when testing the
        equality of this map with another. See `__eq__` method.


    Examples
    --------
    >>> from pisa.core.binning import MultiDimBinning
    >>> binning = MultiDimBinning([dict(name='energy', is_log=True, num_bins=4,
    ...                                 domain=[1, 80], units='GeV'),
    ...                            dict(name='coszen', is_lin=True, num_bins=5,
    ...                                 domain=[-1, 0])])
    >>> m0 = Map(name='x', binning=binning, hist=np.zeros(binning.shape))
    >>> m0
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> m0.binning
    energy: 4 logarithmically-uniform bins spanning [1.0, 80.0] GeV
    coszen: 5 equally-sized bins spanning [-1.0, 0.0]
    >>> m0.hist[0:4, 0] = 1
    >>> m0
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.]])
    >>> m1 = m0[0:3, 0:2]
    >>> m1.binning
    energy: 3 logarithmically-uniform bins spanning [1.0, 26.7496121991]
    coszen: 2 equally-sized bins spanning [-1.0, -0.6]
    >>> m1
    array([[ 1.,  0.],
           [ 1.,  0.],
           [ 1.,  0.]])
    >>> for bin in m1.iterbins():
    ...     print '({0:~.2f}, {1:~.2f}): {2:0.1f}'.format(
    ...             bin.binning.energy.midpoints[0],
    ...             bin.binning.coszen.midpoints[0],
    ...             bin.hist[0, 0])
    (2.00 GeV, -0.90 ): 1.0
    (2.00 GeV, -0.70 ): 0.0
    (5.97 GeV, -0.90 ): 1.0
    (5.97 GeV, -0.70 ): 0.0
    (17.85 GeV, -0.90 ): 1.0
    (17.85 GeV, -0.70 ): 0.0

    """
    _slots = ('name', 'hist', 'binning', 'hash', '_hash', 'tex',
              'full_comparison', 'parent_indexer', '_normalize_values')
    _state_attrs = ('name', 'hist', 'binning', 'hash', 'tex',
                    'full_comparison')

    def __init__(self, name, hist, binning, error_hist=None, hash=None,
                 tex=None, full_comparison=False):
        # Set Read/write attributes via their defined setters
        super().__setattr__('_name', name)
        super().__setattr__('_tex', tex)
        super().__setattr__('_hash', hash)
        super().__setattr__('_full_comparison', full_comparison)

        if not isinstance(binning, MultiDimBinning):
            if isinstance(binning, Sequence):
                binning = MultiDimBinning(dimensions=binning)
            elif isinstance(binning, Mapping):
                binning = MultiDimBinning(**binning)
            else:
                raise ValueError('Do not know what to do with `binning`=%s of'
                                 ' type %s' %(binning, type(binning)))
        self.parent_indexer = None

        # Do the work here to set read-only attributes
        super().__setattr__('_binning', binning)
        binning.assert_array_fits(hist)
        super().__setattr__(
            '_hist', np.ascontiguousarray(hist)
        )
        if error_hist is not None:
            self.set_errors(error_hist)
        self._normalize_values = True

    def __repr__(self):
        previous_precision = np.get_printoptions()['precision']
        np.set_printoptions(precision=18)
        try:
            state = self.serializable_state
            state['hist'] = np.array_repr(state['hist'])
            if state['error_hist'] is not None:
                state['error_hist'] = np.array_repr(state['error_hist'])
            argstrs = [('%s=%r' % item) for item in
                       self.serializable_state.items()]
            r = '%s(%s)' % (self.__class__.__name__, ',\n    '.join(argstrs))
        finally:
            np.set_printoptions(precision=previous_precision)
        return r

    def __str__(self):
        attrs = ['name', 'tex', 'full_comparison', 'hash', 'parent_indexer',
                 'binning', 'hist']
        state = {a: getattr(self, a) for a in attrs}
        state['name'] = repr(state['name'])
        state['tex'] = repr(state['tex'])
        state['hist'] = np.array_repr(state['hist'])
        argstrs = [('%s=%s' % (a, state[a])) for a in attrs]
        s = '%s(%s)' % (self.__class__.__name__, ',\n    '.join(argstrs))
        return s

    def __pretty__(self, p, cycle):
        """Method used by the `pretty` library for formatting"""
        myname = self.__class__.__name__
        if cycle:
            p.text('%s(...)' % myname)
        else:
            p.begin_group(4, '%s(' % myname)
            attrs = ['name', 'tex', 'full_comparison', 'hash',
                     'parent_indexer', 'binning', 'hist']
            for n, attr in enumerate(attrs):
                p.breakable()
                p.text(attr + '=')
                p.pretty(getattr(self, attr))
                if n < len(attrs)-1:
                    p.text(',')
            p.end_group(4, ')')

    def _repr_pretty_(self, p, cycle):
        """Method used by e.g. ipython/Jupyter for formatting"""
        return self.__pretty__(p, cycle)

    def item(self, *args):
        """Call ``item(*args)`` method on the contained `hist`, returning a
        single Python scalar corresponding to `*args`. See help for
        :method:`numpy.ndarray.item` for more info.

        Note that this method is called by :method:`numpy.asscalar`.

        Parameters
        ----------
        *args
            Passed to :method:`numpy.ndarray.item`

        Returns
        -------
        z : Standard Python scalar object

        """
        return self.hist.item(*args)

    def slice(self, **kwargs):
        """Slice the map, where each argument is the name of a dimension.
        Dimensions not named are included in full (i.e., via `np.slice(None)`).

        Note that the resulting map maintains the same number of dimensions as
        its parent, including the ordering of the dimensions. The size of each
        dimension, however, is reduced by slicing.

        Note also that modifications to the returned object's `hist` will
        modify the parent's `hist`.


        Examples
        --------
        Indexing can be done as in the following examples:

        >>> mdb = MultiDimBinning([
        ...     dict(name='x', domain=[0,1], is_lin=True, num_bins=5),
        ...     dict(name='y', domain=[1,2], is_lin=True, num_bins=10)
        ... ])
        >>> ones = mdb.ones(name='ones')
        >>> print ones.slice(x=0,)
        Map(name='ones',
                tex='{\\rm ones}',
                full_comparison=False,
                hash=None,
                parent_indexer=(0, slice(None, None, None)),
                binning=MultiDimBinning([
                            OneDimBinning(name=OneDimBinning('x', 1 bin with edges at [0.0, 0.2] (behavior is linear))),
                            OneDimBinning(name=OneDimBinning('y', 10 equally-sized bins spanning [1.0, 2.0]))]),
                hist=array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]]))
        >>> print ones.slice(x=0, y=slice(None)).hist
        [[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]
        >>> print ones.slice(x=0, y=0).hist
        [[ 1.]]

        Modifications to the slice modifies the original:

        >>> mdb = MultiDimBinning([
        ...     dict(name='x', domain=[0,1], is_lin=True, num_bins=5),
        ...     dict(name='y', domain=[1,2], is_lin=True, num_bins=10)
        ... ])
        >>> ones = mdb.ones(name='ones')
        >>> sl = ones.slice(x=2)
        >>> sl.hist[...] = 0
        >>> print sl.hist
        >>> print ones.hist
        [[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
         [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
         [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]


        See Also
        --------
        pisa.core.binning.MultiDimBinning.indexer
            Method used to generate a raw indexer (that can be used to
            index into a map or a Numpy array of same dimensionality).
            This method is accessible from a Map `map_x` object via its
            `binning` attribute: `map_x.binning.indexer(...)`

        pisa.core.binning.MultiDimBinning.broadcast
            Broadcast a 1D Numpy array to dimensionality with reference to this
            object's dimensionality.

        """
        return self[self.binning.indexer(**kwargs)]

    def set_poisson_errors(self):
        """Approximate poisson errors using sqrt(n)."""
        nom_values = self.nominal_values
        super().__setattr__(
            '_hist',
            unp.uarray(nom_values, np.sqrt(nom_values))
        )

    def set_errors(self, error_hist):
        """Manually define the error with an array the same shape as the
        contained histogram. Can also remove errors by passing None.

        Parameters
        ----------
        error_hist : None or ndarray (same shape as hist)
            Standard deviations to apply to `self.hist`.
            If None is passed, any errors present are removed, making
            `self.hist` a bare numpy array.

        """
        if error_hist is None:
            super().__setattr__(
                '_hist', self.nominal_values
            )
            return
        self.assert_compat(error_hist)
        super().__setattr__(
            '_hist',
            unp.uarray(self._hist, np.ascontiguousarray(error_hist))
        )

    # TODO: make this return an OrderedDict to organize all of the returned
    # objects
    def compare(self, ref):
        """Compare this map with another, where the other map is taken to be
        the "reference" against which this is compared.

        Parameters
        ----------
        ref : Map
            Map against with to compare this one. `ref is taken as reference.
            Each dimension in `ref.binning` must have the same name and
            bin edges as this map, but the order of the dimensions does not
            matter.

        Returns
        -------
        comparisons : OrderedDict containing the following key/value pairs:
          * 'diff' : Map, `self - ref`
          * 'fract' : Map, `self / ref`
          * 'fractdiff' : Map, `(self - ref) / ref`
          * 'max_abs_diff' : float, `max(abs(diff))`
          * 'max_abs_fractdiff' : float, `max(abs(fractdiff))`
          * 'nanmatch' : bool, whether nan elements match
          * 'infmatch' : bool, whether +inf (and separately -inf) entries match

        """
        assert isinstance(ref, Map)
        assert ref.binning == self.binning
        diff = self - ref
        with np.errstate(divide='ignore', invalid='ignore'):
            fract = self / ref
            fractdiff = diff / ref

        max_abs_fractdiff = np.nanmax(np.abs(fractdiff.nominal_values))

        # Handle cases where ratio returns infinite
        # This isn't necessarily a fail, since all it means is the referene was
        # zero; if the new value is sufficiently close to zero then it's still
        # fine.
        if np.isinf(max_abs_fractdiff):
            # First find all the finite elements
            finite_mask = np.isfinite(fractdiff.nominal_values)
            # Then find the nanmax of this, will be our new test value
            max_abs_fractdiff = np.nanmax(np.abs(
                fractdiff.nominal_values[finite_mask]
            ))

            # TODO(bug): Why is ~finite_mask used to select elements here?
            # Shouldn't all elements be considered, regardless if fractdiff is
            # inf somewhere?

            # Also find all the infinite elements; compute a second test value
            max_abs_diff = np.nanmax(np.abs(diff.nominal_values[~finite_mask]))
        else:
            # Without any infinite elements we can ignore this second test
            max_abs_diff = np.nanmax(np.abs(diff.nominal_values))

        nanmatch = bool(np.all(np.isnan(self.nominal_values)
                               == np.isnan(ref.nominal_values)))
        infmatch = bool(np.all(
            self.nominal_values[np.isinf(self.nominal_values)]
            == ref.nominal_values[np.isinf(ref.nominal_values)]
        ))

        comparisons = OrderedDict([
            ('diff', diff),
            ('fract', fract),
            ('fractdiff', fractdiff),
            ('max_abs_fractdiff', max_abs_fractdiff),
            ('max_abs_diff', max_abs_diff),
            ('nanmatch', nanmatch),
            ('infmatch', infmatch)
        ])

        return comparisons

    def plot(self, symm=False, logz=False, vmin=None, vmax=None, backend=None,
             ax=None, title=None, cmap=None, clabel=None, clabelsize=None,
             xlabelsize=None, ylabelsize=None, titlesize=None, fig_kw=None,
             pcolormesh_kw=None, colorbar_kw=None, outdir=None, fname=None,
             fmt=None, binlabel_format=None):
        """Plot a 2D map.

        Parameters
        ----------
        symm : bool, optional
            Plot with symmetric (about 0) value-range limits.

        logz : bool, optional
            Plot logarithmic value-range

        vmin, vmax : float, optional
            Minimum and maximum values for the value-range of the plot. If None
            specified, these are set according to `symm` and/or the values of
            the `hist` in this Map.

        backend : string, optional
            Matplotlib backend to use (only takes effect if matplotlib is first
            imported by this function).

        ax : matplotlib.axis.Axis, optional
            Provide an axis onto which the plot is drawn; if None is specified,
            a new figure and axis are created.

        title : string, optional
            Set the title to this value; if None is specified, the title is
            taken from the name of this Map.

        cmap : string or matplotlib.colors.Colormap, optional

        clabel : string, optional
            Label to place on the colorbar

        clabelsize, xlabelsize, ylabelsize, titlesize : float, optional
            Size of the colorbar, x-axis label, y-axis label, and title text

        fig_kw : mapping, optional
            Keyword arguments passed to call to `matplotlib.pyplot.figure`;
            this is only done, however, if `ax` is None and so a new figure
            needs to be created.

        pcolormesh_kw : mapping, optional
            Keyword arguments to pass to call to `matplotlib.pyplot.pcolormesh`
            (if Map is two or more dimensions).

        colorbar_kw : mapping, optional
            Keyword arguments to pass to call to `matplotlib.colorbar`.

        fmt : string in ('pdf', 'png') or iterable thereof, optional
            File format(s) in which to save the file. If None, then the plot
            will not be saved.

        outdir : string, optional
            Directory into which to save the plot. If None is provided, the the
            default is the current directory. Note that if `fmt` is None, then
            this argument is irrelevant.

        fname : string, optional
             Custom filename to set for saved figure. If not provided, a name
             is derived from the `name` attribute of the Map. Note that if
             `fmt` is None, then this argument is irrelevant.
        
        binlabel_format : string, optional
            Format string to label the content in each bin. If None (default), the bins will not
            be labeled.

        Returns
        -------
        fig : :class:`matplotlib.figure.Figure` object
        ax : :class:`matplotlib.axes.Axes` object
        pcmesh : :class:`matplotlib.collections.QuadMesh`
        colorbar : :class:`matplotlib.colorbar.Colorbar`

        """
        import matplotlib as mpl
        if (backend is not None
                and mpl.get_backend().lower() != backend.lower()):
            mpl.use(backend)
        import matplotlib.pyplot as plt

        cmap_seq = plt.cm.inferno
        cmap_seq.set_bad(color=(0.0, 0.2, 0.0), alpha=1)

        cmap_div = plt.cm.RdBu_r
        cmap_div.set_bad(color=(0.5, 0.9, 0.5), alpha=1)

        # TODO: use https://matplotlib.org/users/colormapnorms.html
        # to allow for both symm and logz (and to implement logz in the first
        # place!)
        assert not(symm and logz)

        if title is None:
            title = '$%s$' % (self.name if self.tex is None else self.tex)

        if fname is None:
            fname = get_valid_filename(self.name)

        fig_kw = {} if fig_kw is None else fig_kw
        pcolormesh_kw = {} if pcolormesh_kw is None else pcolormesh_kw
        colorbar_kw = {} if colorbar_kw is None else colorbar_kw
        if fmt is not None:
            if isinstance(fmt, str):
                fmt = [fmt]
            fmt = set(f.strip().lower().lstrip('.') for f in fmt)
            if outdir is None:
                outdir = './'
            else:
                mkdir(outdir, warn=False)

        if ax is None:
            fig = plt.figure(**fig_kw)
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        # TODO: allow plotting of N-dimensional arrays: 1D should be simple; >
        # 2D by arraying them as 2D slices in the smallest dimension(s)
        if len(self.binning) == 2:
            to_plot = self
        else:
            to_plot = self.squeeze()
        assert len(to_plot.binning) == 2

        if fmt is not None and fname is None:
            fname = get_valid_filename(to_plot.name)

        hist = valid_nominal_values(to_plot.hist)
        if symm:
            cmap = cmap_div if cmap is None else cmap
            if vmin is None and vmax is None:
                vmax_ = np.nanmax(np.abs(hist))
            elif vmin is None and vmax is not None:
                vmax_ = np.abs(vmax)
            elif vmin is not None and vmax is None:
                vmax_ = np.abs(vmin)
            else:
                assert vmax > vmin and vmax == -vmin
                vmax_ = vmax
            vmin_ = -vmax_
        else:
            cmap = cmap_seq if cmap is None else cmap
            vmin_ = vmin if vmin is not None else np.nanmin(hist)
            vmax_ = vmax if vmax is not None else np.nanmax(hist)

        x = to_plot.binning.dims[0].bin_edges.magnitude
        y = to_plot.binning.dims[1].bin_edges.magnitude

        if to_plot.binning.dims[0].is_log:
            xticks = 2**(np.arange(np.ceil(np.log2(min(x))),
                                   np.floor(np.log2(max(x)))+1))
            x = np.log10(x)
        if to_plot.binning.dims[1].is_log:
            yticks = 2**(np.arange(np.ceil(np.log2(min(y))),
                                   np.floor(np.log2(max(y)))+1))
            y = np.log10(y)

        defaults = dict(
            vmin=vmin_, vmax=vmax_, cmap=cmap,
            shading='flat', edgecolors='face'
        )
        if logz:
            defaults['norm'] = mpl.colors.LogNorm(
                hist[hist > 0].min(), hist.max(), clip=True
            )
        for key, dflt_val in defaults.items():
            if key not in pcolormesh_kw:
                pcolormesh_kw[key] = dflt_val

        X, Y = np.meshgrid(x, y)
        pcmesh = ax.pcolormesh(X, Y, hist.T, **pcolormesh_kw)
        if binlabel_format is not None:
            X_mid = np.true_divide(X[1:, 1:] + X[1:, :-1], 2)
            Y_mid = np.true_divide(Y[1:, 1:] + Y[:-1, 1:], 2)
            for xi, yi, zi in zip(np.ravel(X_mid), np.ravel(Y_mid), np.ravel(hist.T)):
                ax.text(xi, yi, binlabel_format.format(zi),
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=10)
        colorbar = plt.colorbar(mappable=pcmesh, ax=ax, **colorbar_kw)
        colorbar.ax.tick_params(labelsize='large')
        if clabel is not None:
            colorbar.set_label(label=clabel, size=clabelsize)

        xlabel = '$%s$' % to_plot.binning.dims[0].label
        ylabel = '$%s$' % to_plot.binning.dims[1].label

        ax.set_xlabel(xlabel, size=xlabelsize)
        ax.set_ylabel(ylabel, size=ylabelsize)
        ax.set_title(title, y=1.03, size=titlesize)
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))

        if to_plot.binning.dims[0].is_log:
            ax.set_xticks(np.log10(xticks))
            ax.set_xticklabels([str(int(xt)) for xt in xticks])
        if to_plot.binning.dims[1].is_log:
            ax.set_yticks(np.log10(yticks))
            ax.set_yticklabels([str(int(yt)) for yt in yticks])

        if fmt is not None:
            for fmt_ in fmt:
                path = os.path.join(outdir, fname + '.' + fmt_)
                fig.savefig(os.path.join(*path))
                logging.debug('>>>> Plot for inspection saved at %s', path)

        return fig, ax, pcmesh, colorbar

    @_new_obj
    def reorder_dimensions(self, order):
        """Rearrange the dimensions in the map. This affects both the binning
        and the contained histogram.

        Parameters
        ----------
        order : MultiDimBinning or sequence of str, int, or OneDimBinning
            Ordering desired for the dimensions of this map. See
            `binning.reorder_dimensions` for details on how to specify `order`.

        Returns
        -------
        Map : copy of this map but with dimensions reordered

        See Also
        --------
        rebin
            Modify Map (and its binning) by splitting or combining adjacent
            bins

        downsample
            Modify Map (and its binning) by combining adjacent bins

        """
        new_binning = self.binning.reorder_dimensions(order)
        orig_order = list(range(len(self.binning)))
        new_order = [self.binning.index(b, use_basenames=False)
                     for b in new_binning]
        # TODO: should this be a deepcopy rather than a simple veiw of the
        # original hist (the result of np.moveaxis)?
        new_hist = np.moveaxis(self.hist, source=new_order,
                               destination=orig_order)
        return {'hist': new_hist, 'binning': new_binning}

    @_new_obj
    def squeeze(self):
        """Remove any singleton dimensions (i.e. that have only a single bin).
        Analagous to `numpy.squeeze`.

        Returns
        -------
        Map with equivalent values but singleton dimensions removed

        """
        new_binning = self.binning.squeeze()
        new_hist = self.hist.squeeze()
        return {'hist': new_hist, 'binning': new_binning}

    @_new_obj
    def sum(self, axis=None, keepdims=False):
        """Sum over dimensions corresponding to `axis` specification. Similar
        in behavior to `numpy.sum` method.

        Parameters
        ----------
        axis : None; or str, int, or sequence thereof
            Dimension(s) to be summed over. If None, sum over _all_ dimensions.

        keepdims : bool
            If True, marginalizes out (removes) the specified dimensions. If
            False, the binning in the summed dimension(s) is expanded to the
            full range of the binning for each dimension over which the sum is
            performed.

        Returns
        -------
        s : Map or scalar
            If all contained dimensiosn are summed over and `keepdims` is
            False, a scalar is returned. Otherwise, a Map is returned with
            dimensions marginalized out in the sum removed if `keepdims` is
            False.

        """
        if axis is None:
            axis = self.binning.names
        if isinstance(axis, (str, int)):
            axis = [axis]
        # Note that the tuple is necessary here (I think...)
        sum_indices = tuple([self.binning.index(dim) for dim in axis])
        new_hist = self.hist.sum(axis=sum_indices, keepdims=keepdims)

        new_binning = []
        for idx, dim in enumerate(self.binning.dims):
            if idx in sum_indices:
                if keepdims:
                    new_binning.append(dim.downsample(len(dim)))
            else:
                new_binning.append(dim)
        return {'hist': new_hist, 'binning': new_binning}

    def project(self, axis, keepdims=False):
        """Project all dimensions onto a single `axis`.

        Parameters
        ----------
        axis : string or int
            Dimensions to be projected onto.
        keepdims : bool
            If True, marginalizes out (removes) the _un_specified dimensions.
            If False, the binning in the summed dimension(s) includes
            the full range of the binning for each dimension in the original
            Map. Note that if you want to remove all _singleton_ dimensions
            (which could include the `axis` specified here), call the
            `squeeze` method on the result of `project`.

        Returns
        -------
        projection : Map

        """
        keep_index = self.binning.index(axis)
        sum_indices = list(range(len(self.binning.dims)))
        sum_indices.remove(keep_index)

        return self.sum(axis=sum_indices, keepdims=keepdims)

    @_new_obj
    def rebin(self, new_binning):
        """Rebin the map with bin edge locations and names according to those
        specified in `new_binning`.

        Calls the `rebin` function in the pisa.core.map.rebin module to do the
        actual work.

        Parameters
        ----------
        new_binning : MultiDimBinning
            Dimensions specified in `new_binning` must match (modulo
            pre/suffixes) the current dimensions.

        Returns
        -------
        Map binned according to `new_binning`.

        See Also
        ---------
        `pisa.core.map.rebin` : function called to do the work

        """
        # TODO: put uncertainties in
        new_hist = rebin(hist=self.hist, orig_binning=self.binning,
                         new_binning=new_binning)
        return {'hist': new_hist, 'binning': new_binning}

    def downsample(self, *args, **kwargs):
        """Downsample by integer factor(s), summing together merged bins'
        values.

        See pisa.utils.binning.MultiDimBinning.downsample for args/kwargs
        details.

        """
        new_binning = self.binning.downsample(*args, **kwargs)
        return self.rebin(new_binning)

    @_new_obj
    def fluctuate(self, method, random_state=None, jumpahead=0):
        """Apply fluctuations to the map's values.

        Parameters
        ----------
        method : None or string
            Valid strings are '', 'none', 'poisson', 'gauss', or
            'gauss+poisson'. Strings are case-insensitive and whitespace is
            removed.

        random_state : None or type accepted by utils.random_numbers.get_random_state

        jumpahead : int >= 0
            After instantiating the random_state object, move `jumpahead`
            positions forward in the Mersenne twister's finite state machine

        Returns
        -------
        fluctuated_map : Map
            New map with entries fluctuated as compared to this map

        """
        orig = method
        method = str(method).strip().lower().replace(' ', '')
        if method == 'poisson':
            random_state = get_random_state(random_state, jumpahead=jumpahead)
            with np.errstate(invalid='ignore'):
                orig_hist = self.nominal_values
                nan_at = np.isnan(orig_hist)
                valid_mask = ~nan_at

                hist_vals = np.empty_like(orig_hist, dtype=np.float64)
                hist_vals[valid_mask] = poisson.rvs(
                    orig_hist[valid_mask],
                    random_state=random_state
                )
                hist_vals[nan_at] = np.nan

                error_vals = np.empty_like(orig_hist, dtype=np.float64)
                error_vals[valid_mask] = np.sqrt(orig_hist[valid_mask])
                error_vals[nan_at] = np.nan
            return {'hist': unp.uarray(hist_vals, error_vals)}

        elif method == 'gauss+poisson':
            random_state = get_random_state(random_state, jumpahead=jumpahead)
            with np.errstate(invalid='ignore'):
                orig_hist = self.nominal_values
                sigma = self.std_devs
                nan_at = np.isnan(orig_hist)
                valid_mask = ~nan_at
                gauss = np.empty_like(orig_hist, dtype=np.float64)
                gauss[valid_mask] = norm.rvs(
                    loc=orig_hist[valid_mask], scale=sigma[valid_mask]
                )

                hist_vals = np.empty_like(orig_hist, dtype=np.float64)
                hist_vals[valid_mask] = poisson.rvs(
                    gauss[valid_mask],
                    random_state=random_state
                )
                hist_vals[nan_at] = np.nan

                error_vals = np.empty_like(orig_hist, dtype=np.float64)
                error_vals[valid_mask] = np.sqrt(orig_hist[valid_mask])
                error_vals[nan_at] = np.nan
            return {'hist': unp.uarray(hist_vals, error_vals)}

        elif method == 'gauss':
            random_state = get_random_state(random_state, jumpahead=jumpahead)
            with np.errstate(invalid='ignore'):
                orig_hist = self.nominal_values
                sigma = self.std_devs
                nan_at = np.isnan(orig_hist)
                valid_mask = ~nan_at
                hist_vals = np.empty_like(orig_hist, dtype=np.float64)
                hist_vals[valid_mask] = norm.rvs(loc=orig_hist[valid_mask],
                                                 scale=sigma[valid_mask])
                hist_vals[nan_at] = np.nan
                error_vals = np.empty_like(orig_hist, dtype=np.float64)
                error_vals[valid_mask] = np.sqrt(orig_hist[valid_mask])
                error_vals[nan_at] = np.nan
            return {'hist': unp.uarray(hist_vals, error_vals)}

        elif method in ['', 'none']:
            return {}

        else:
            raise ValueError('unhandled `method` = %s' % orig)

    @property
    def shape(self):
        """tuple : shape of the map, akin to `nump.ndarray.shape`"""
        return self.hist.shape

    @property
    def size(self):
        """int : total number of elements"""
        return self.hist.size

    @property
    def num_entries(self):
        """int : total number of weighted entries in all bins"""
        return np.sum(valid_nominal_values(self.hist))

    @property
    def serializable_state(self):
        state = OrderedDict()
        state['name'] = self.name
        state['hist'] = self.nominal_values
        state['binning'] = self.binning.serializable_state
        stddevs = self.std_devs
        stddevs = None if np.all(stddevs == 0) else stddevs
        state['error_hist'] = stddevs
        state['hash'] = self.hash
        state['tex'] = self._tex
        state['full_comparison'] = self.full_comparison
        return state

    @property
    def hashable_state(self):
        state = OrderedDict()
        state['name'] = self.name
        if self.normalize_values:
            state['hist'] = normQuant(self.nominal_values,
                                      sigfigs=HASH_SIGFIGS)
            stddevs = normQuant(self.std_devs, sigfigs=HASH_SIGFIGS)
        else:
            state['hist'] = self.nominal_values
            stddevs = self.std_devs
        state['binning'] = self.binning.hashable_state
        # TODO: better check here to see if the contained datatype is unp, as
        # opposed to 0 stddev (which could be the case but the user wants for
        # uncertainties to propagate)
        if np.all(stddevs == 0):
            stddevs = None
        elif self.normalize_values:
            stddevs = normQuant(stddevs, sigfigs=HASH_SIGFIGS)
        state['error_hist'] = stddevs
        state['full_comparison'] = self.full_comparison
        return state

    @property
    def normalize_values(self):
        return self._normalize_values

    @normalize_values.setter
    def normalize_values(self, b):
        assert isinstance(b, bool)
        self._normalize_values = b

    def __getstate__(self):
        return self.serializable_state

    def __setstate__(self, state):
        self.__init__(**state)

    def to_json(self, filename, **kwargs):
        """Serialize the state to a JSON file that can be instantiated as a new
        object later.

        Parameters
        ----------
        filename : str
            Filename; must be either a relative or absolute path (*not
            interpreted as a PISA resource specification*)
        **kwargs
            Further keyword args are sent to `pisa.utils.jsons.to_json()`

        See Also
        --------
        from_json : Intantiate new object from the file written by this method
        pisa.utils.jsons.to_json

        """
        jsons.to_json(self.serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, resource):
        """Instantiate a new Map object from a JSON file.

        The format of the JSON is generated by the `Map.to_json` method, which
        converts a Map object to basic types and then numpy arrays are
        converted in a call to `pisa.utils.jsons.to_json`.

        Parameters
        ----------
        resource : str
            A PISA resource specification (see pisa.utils.resources)

        See Also
        --------
        to_json
        pisa.utils.jsons.to_json

        """
        state = jsons.from_json(resource)
        # State is a dict with kwargs, so instantiate with double-asterisk
        # syntax
        return cls(**state)

    def assert_compat(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            return
        elif isinstance(other, np.ndarray):
            self.binning.assert_array_fits(other)
        elif isinstance(other, Map):
            self.binning.assert_compat(other.binning)
        else:
            raise TypeError('Unhandled type %s' % type(other))

    def iterbins(self):
        """Returns a bin iterator which yields a map containing a single bin
        each time. Note that modifications to that single-bin map will be
        reflected in this (the parent) map.

        Note that the returned map has the attribute `parent_indexer` for
        indexing directly into to the parent map (or to a similar map).

        Yields
        ------
        Map object containing one of each bin of this Map

        """
        for i in range(self.size):
            idx_coord = self.binning.index2coord(i)
            idx_view = [slice(x, x+1) for x in idx_coord]
            single_bin_map = Map(
                name=self.name, hist=self.hist[idx_view],
                binning=self.binning[idx_coord], hash=None, tex=self.tex,
                full_comparison=self.full_comparison
            )
            single_bin_map.parent_indexer = idx_coord
            yield single_bin_map

    # TODO : example!
    def itercoords(self):
        """Iterator that yields the coordinate of each bin in the map."""
        return self.binning.itercoords()

    def __hash__(self):
        if self.hash is not None:
            return self.hash
        raise ValueError('No hash defined.')

    def __setattr__(self, attr, value):
        """Only allow setting attributes defined in slots"""
        if attr not in self._slots:
            raise ValueError('Attribute "%s" not allowed to be set.' % attr)
        super().__setattr__(attr, value)

    def __getattr__(self, attr):
        return super().__getattribute__(attr)

    def _slice_or_index(self, idx):
        """Slice or index into the map. Indexing single element in self.hist
        e.g. hist[1,3] returns a 0D array while hist[1,3:8] returns a 1D array,
        but we need 2D (in his example)... so reshape after indexing (the
        indexed binning obj implements this logic and so knows the shape the
        hist should be).

        """
        new_binning = self.binning[idx]

        new_map = Map(name=self.name,
                      hist=np.reshape(self.hist[idx], new_binning.shape),
                      binning=self.binning[idx],
                      hash=self.hash,
                      tex=self.tex,
                      full_comparison=self.full_comparison)
        new_map.parent_indexer = idx
        return new_map

    def __getitem__(self, idx):
        return self._slice_or_index(idx)

    # TODO: if no bin name is given (i.e., 1D indexing), then split into maps
    # and return a MapSet with a map per bin; append '__%s__%s' %(dim_name,
    # bin_name) to this map's name to name each new map, and if no bin names
    # are given, use str(int(ind)) instead for bin_name.
    def split(self, dim, bin=None, use_basenames=False):
        """Split this map into one or more maps by selecting the `dim`
        dimension and optionally the specific bin(s) within that dimension
        specified by `bin`.

        If both `dim` and `bin` are specified and this identifies a single bin,
        a single Map is returned, while if this locates multiple bins, a MapSet
        is returned where each map corresponds to a bin (in the order dictated
        by the `bin` specification).

        If only `dim` is specified, _regardless_ if multiple bins meet the
        (dim, bin) criteria, the maps corresponding to each `bin` are collected
        into a MapSet and returned.

        Resulting maps are ordered according to the binning and are renamed as:

            new_map[j].name = orig_map.name__dim.binning.bin_names[i]

        if the current map has a name, or

            new_map[j].name = dim.binning.bin_names[i]

        if the current map has a zero-length name.

        In the above, j is the index into the new MapSet and i is the index to
        the bin in the original binning spec. `map.name` is the current
        (pre-split) map's name, and if the bins do not have names, then the
        stringified integer index to the bin, str(i), is used instead.

        Parameters
        ----------
        dim : string, int
            Name or index of a dimension in the map
        bin : None or bin indexing object (str, int, slice, ellipsis)
            Optionally specify specific bin(s) to split out from the chosen
            dimension.

        Returns
        -------
        split_maps : Map or MapSet
            If only `dim` is passed, returns MapSet regardless of how many maps
            are found. If both `dim` and `bin` are specified and this results
            in selecting more than one bin, also returns a MapSet. However if
            both `dim` and `bin` are specified and this selects a single bin,
            just the indexed Map is returned. Naming of the maps and MapSet is
            updated to reflect what the map represents, while the hash value is
            copied into the new map(s).

        """
        dim_index = self.binning.index(dim, use_basenames=use_basenames)
        spliton_dim = self.binning.dims[dim_index]

        # Move the dimension we're going to split on to be the first dim
        new_order = list(range(len(self.binning)))
        new_order.pop(dim_index)
        new_order = [dim_index] + new_order
        rearranged_map = self.reorder_dimensions(new_order)
        rearranged_hist = rearranged_map.hist
        rearranged_dims = rearranged_map.binning.dims

        # Take all dims except the one being split on
        new_binning = rearranged_dims[1:]

        singleton = False
        if bin is not None:
            if isinstance(bin, (int, str)):
                bin_indices = [spliton_dim.index(bin)]
            elif isinstance(bin, slice):
                bin_indices = list(range(len(spliton_dim)))[bin]
            elif bin is Ellipsis:
                bin_indices = list(range(len(spliton_dim)))

            if len(bin_indices) == 1:
                singleton = True
        else:
            bin_indices = list(range(len(spliton_dim)))

        maps = []
        for bin_index in bin_indices:
            bin = spliton_dim[bin_index]
            new_hist = rearranged_hist[bin_index, ...]
            if bin.bin_names is not None:
                bin_name = bin.bin_names[0]
                bin_tex = '=' + text2tex(bin_name)
            else:
                bin_name = 'bin_%d' % bin_index
                bin_tex = r'{\;}bin{\;}%d' % bin_index

            name_elements = []
            for s in [self.name, spliton_dim.name, bin_name]:
                if s is not None and len(s) > 0:
                    name_elements.append(s)
            new_name = '_'.join(name_elements)

            new_tex = self.tex + ',' + r'{\;}' + spliton_dim.tex + bin_tex

            maps.append(
                Map(name=new_name, hist=new_hist, binning=new_binning,
                    hash=self.hash, tex=new_tex,
                    full_comparison=self.full_comparison)
            )

        if singleton:
            assert len(maps) == 1
            return maps[0]

        if len(self.name) > 0:
            mapset_name = '%s__split_on__%s' % (self.name, spliton_dim.name)
        else:
            mapset_name = 'split_on__%s' % spliton_dim.name

        if self.tex is not None and len(self.tex) > 0:
            mapset_tex = r'%s, \; %s' % (self.tex, spliton_dim.tex)
        else:
            mapset_tex = r'%s' % spliton_dim.tex

        return MapSet(maps=maps, name=mapset_name, tex=mapset_tex)

    def llh(self, expected_values, binned=False):
        """Calculate the total log-likelihood value between this map and the
        map described by `expected_values`; self is taken to be the "actual
        values" (or (pseudo)data), and `expected_values` are the expectation
        values for each bin.

        Parameters
        ----------
        expected_values : numpy.ndarray or Map of same dimension as this

        binned : bool

        Returns
        -------
        total_llh : float or binned_llh if binned=True

        """
        expected_values = reduceToHist(expected_values)

        if binned:
            return stats.llh(actual_values=self.hist,
                             expected_values=expected_values)

        return np.sum(stats.llh(actual_values=self.hist,
                                expected_values=expected_values))
    
    def mcllh_mean(self, expected_values, binned=False):
        """Calculate the total LMean log-likelihood value between this map and the
        map described by `expected_values`; self is taken to be the "actual
        values" (or (pseudo)data), and `expected_values` are the expectation
        values for each bin.

        Parameters
        ----------
        expected_values : numpy.ndarray or Map of same dimension as this

        binned : bool

        Returns
        -------
        total_llh : float or binned_llh if binned=True

        """
        expected_values = reduceToHist(expected_values)

        if binned:
            return stats.mcllh_mean(actual_values=self.hist,
                             expected_values=expected_values)

        return np.sum(stats.mcllh_mean(actual_values=self.hist,
                                expected_values=expected_values))


    def mcllh_eff(self, expected_values, binned=False):
        """Calculate the total LEff log-likelihood value between this map and the
        map described by `expected_values`; self is taken to be the "actual
        values" (or (pseudo)data), and `expected_values` are the expectation
        values for each bin.

        Parameters
        ----------
        expected_values : numpy.ndarray or Map of same dimension as this

        binned : bool

        Returns
        -------
        total_llh : float or binned_llh if binned=True

        """
        expected_values = reduceToHist(expected_values)

        if binned:
            return stats.mcllh_eff(actual_values=self.hist,
                             expected_values=expected_values)

        return np.sum(stats.mcllh_eff(actual_values=self.hist,
                                expected_values=expected_values))

    def conv_llh(self, expected_values, binned=False):
        """Calculate the total convoluted log-likelihood value between this map
        and the map described by `expected_values`; self is taken to be the
        "actual values" (or (pseudo)data), and `expected_values` are the
        expectation values for each bin.

        Parameters
        ----------
        expected_values : numpy.ndarray or Map of same dimension as this

        binned : bool

        Returns
        -------
        total_conv_llh : float or binned_conv_llh if binned=True

        """
        expected_values = reduceToHist(expected_values)

        if binned:
            return stats.conv_llh(actual_values=self.hist,
                                  expected_values=expected_values)

        return np.sum(stats.conv_llh(actual_values=self.hist,
                                     expected_values=expected_values))

    def barlow_llh(self, expected_values, binned=False):
        """Calculate the total barlow log-likelihood value between this map and
        the map described by `expected_values`; self is taken to be the "actual
        values" (or (pseudo)data), and `expected_values` are the expectation
        values for each bin. I assumes at the moment some things that are not
        true, namely that the weights are uniform

        Parameters
        ----------
        expected_values : numpy.ndarray or Map of same dimension as this

        binned : bool

        Returns
        -------
        total_barlow_llh : float or binned_barlow_llh if binned=True

        """
        # TODO: should this handle reduceToHist / expected_values as other
        # methods do, or should they handle these the way this method does?
        if isinstance(expected_values, (np.ndarray, Map, MapSet)):
            expected_values = reduceToHist(expected_values)
        elif isinstance(expected_values, Iterable):
            expected_values = [reduceToHist(x) for x in expected_values]

        if binned:
            return stats.barlow_llh(actual_values=self.hist,
                                    expected_values=expected_values)

        return np.sum(stats.barlow_llh(actual_values=self.hist,
                                       expected_values=expected_values))

    def mod_chi2(self, expected_values, binned=False):
        """Calculate the total modified chi2 value between this map and the map
        described by `expected_values`; self is taken to be the "actual values"
        (or (pseudo)data), and `expected_values` are the expectation values for
        each bin.

        Parameters
        ----------
        expected_values : numpy.ndarray or Map of same dimension as this.

        binned : bool

        Returns
        -------
        total_mod_chi2 : float or binned_mod_chi2 if binned=True

        """
        expected_values = reduceToHist(expected_values)

        if binned:
            return stats.mod_chi2(actual_values=self.hist,
                                  expected_values=expected_values)

        return np.sum(stats.mod_chi2(actual_values=self.hist,
                                     expected_values=expected_values))

    def chi2(self, expected_values, binned=False):
        """Calculate the total chi-squared value between this map and the map
        described by `expected_values`; self is taken to be the "actual values"
        (or (pseudo)data), and `expected_values` are the expectation values for
        each bin.

        Parameters
        ----------
        expected_values : numpy.ndarray or Map of same dimension as this

        binned : bool

        Returns
        -------
        total_chi2 : float or binned_chi2 if binned=True

        """
        expected_values = reduceToHist(expected_values)

        if binned:
            return stats.chi2(actual_values=self.hist,
                              expected_values=expected_values)

        return np.sum(stats.chi2(actual_values=self.hist,
                                 expected_values=expected_values))

    def metric_total(self, expected_values, metric):
        # TODO: should this use reduceToHist as in chi2 and llh above?
        if metric in stats.ALL_METRICS:
            return getattr(self, metric)(expected_values)
        else:
            raise ValueError('`metric` "%s" not recognized; use one of %s.'
                             % (metric, stats.ALL_METRICS))

    def __setitem__(self, idx, val):
        return setitem(self.hist, idx, val)

    @property
    def name(self):
        """string : Map's name"""
        return self._name

    @name.setter
    def name(self, value):
        """map name"""
        assert isinstance(value, str)
        return super().__setattr__('_name', value)

    @property
    def tex(self):
        """string : TeX label"""
        if self._tex is None:
            return text2tex(self.name)
        return self._tex

    @tex.setter
    def tex(self, value):
        assert value is None or isinstance(value, str)
        if value is not None:
            value = strip_outer_dollars(value)
        return super().__setattr__('_tex', value)

    @property
    def hash(self):
        """int or None : Hash value"""
        return self._hash

    @hash.setter
    def hash(self, value):
        """Hash must be an immutable type (i.e., have a __hash__ method)"""
        assert hasattr(value, '__hash__')
        super().__setattr__('_hash', value)

    @property
    def hist(self):
        """numpy.ndarray : Histogram array underlying the Map"""
        return self._hist

    @property
    def nominal_values(self):
        """numpy.ndarray : Bin values stripped of uncertainties"""
        return unp.nominal_values(self._hist)

    @property
    def std_devs(self):
        """numpy.ndarray : Uncertainties (standard deviations) per bin"""
        return unp.std_devs(self._hist)

    @property
    def binning(self):
        """pisa.core.binning.MultiDimBinning : Map's binning"""
        return self._binning

    @property
    def full_comparison(self):
        """Compare element-by-element instead of just comparing hashes."""
        return self._full_comparison

    @full_comparison.setter
    def full_comparison(self, value):
        assert isinstance(value, bool)
        super().__setattr__('_full_comparison', value)

    # Common mathematical operators

    @_new_obj
    def __abs__(self):
        state_updates = {
            #'name': "|%s|" % (self.name,),
            #'tex': r"{\left| %s \right|}" % strip_outer_parens(self.tex),
            'hist': np.abs(self.hist)
        }
        return state_updates

    @_new_obj
    def __add__(self, other):
        """Add `other` to self"""
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            state_updates = {
                #'name': "(%s + %s)" % (self.name, other),
                #'tex': r"{(%s + %s)}" % (self.tex, other),
                'hist': self.hist + other
            }
        elif isinstance(other, np.ndarray):
            state_updates = {
                #'name': "(%s + array)" % self.name,
                #'tex': r"{(%s + X)}" % self.tex,
                'hist': self.hist + other
            }
        elif isinstance(other, Map):
            state_updates = {
                #'name': "(%s + %s)" % (self.name, other.name),
                #'tex': r"{(%s + %s)}" % (self.tex, other.tex),
                'hist': self.hist + other.hist,
                'full_comparison': (self.full_comparison or
                                    other.full_comparison),
            }
        else:
            type_error(other)
        return state_updates

    #def __cmp__(self, other):

    @_new_obj
    def __div__(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            state_updates = {
                #'name': "(%s / %s)" % (self.name, other),
                #'tex': r"{(%s / %s)}" % (self.tex, other),
                'hist': self.hist / other
            }
        elif isinstance(other, np.ndarray):
            state_updates = {
                #'name': "(%s / array)" % self.name,
                #'tex': r"{(%s / X)}" % self.tex,
                'hist': self.hist / other
            }
        elif isinstance(other, Map):
            state_updates = {
                #'name': "(%s / %s)" % (self.name, other.name),
                #'tex': r"{(%s / %s)}" % (self.tex, other.tex),
                'hist': self.hist / other.hist,
                'full_comparison': (self.full_comparison or
                                    other.full_comparison),
            }
        else:
            type_error(other)
        return state_updates

    def __truediv__(self, other):
        return self.__div__(other)

    def __floordiv__(self, other):
        raise NotImplementedError('floordiv not implemented for type Map')

    # TODO: figure out what we actually want to overload "==" with, and how
    # to implement all the possible kinds of "==" that might be useful for the
    # user, possibly with different methods altogether
    def __eq__(self, other):
        """Check if full state of maps are equal. *Not* element-by-element
        equality as for a numpy array. Call this.hist == other.hist for the
        element-by-element nominal value and the error.

        If `full_comparison` is true for either map, or if either map lacks a
        hash, performs a full comparison of the contents of each map.

        Otherwise, simply checks that the hashes are equal.

        """
        if np.isscalar(other):
            return np.all(self.nominal_values == other)

        if type(other) is uncertainties.core.Variable \
                or isinstance(other, np.ndarray):
            return (np.all(self.nominal_values
                           == unp.nominal_values(other))
                    and np.all(self.std_devs
                               == unp.std_devs(other)))

        if isinstance(other, Map):
            if (self.full_comparison or other.full_comparison
                    or self.hash is None or other.hash is None):
                return recursiveEquality(self.hashable_state,
                                         other.hashable_state)
            return self.hash == other.hash

        type_error(other)

    @_new_obj
    def log(self):
        """Take natural logarithm of map's values, returning a new map.

        Returns
        -------
        log_map : Map

        """
        state_updates = {
            #'name': "log(%s)" % self.name,
            #'tex': r"\ln\left( %s \right)" % self.tex,
            'hist': np.log(self.hist)
        }
        return state_updates

    @_new_obj
    def log10(self):
        """Take base-10 logarithm of map's values, returning a new map.

        Returns
        -------
        log10_map : Map

        """
        state_updates = {
            #'name': "log10(%s)" % self.name,
            #'tex': r"\log_{10}\left( %s \right)" % self.tex,
            'hist': np.log10(self.hist)
        }
        return state_updates

    @_new_obj
    def __mul__(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            state_updates = {
                #'name': "%s * %s" % (other, self.name),
                #'tex': r"%s \cdot %s" % (other, self.tex),
                'hist': self.hist * other
            }
        elif isinstance(other, np.ndarray):
            state_updates = {
                #'name': "array * %s" % self.name,
                #'tex': r"X \cdot %s" % self.tex,
                'hist': self.hist * other,
            }
        elif isinstance(other, Map):
            state_updates = {
                #'name': "%s * %s" % (self.name, other.name),
                #'tex': r"%s \cdot %s" % (self.tex, other.tex),
                'hist': self.hist * other.hist,
                'full_comparison': (self.full_comparison or
                                    other.full_comparison),
            }
        else:
            type_error(other)
        return state_updates

    def __ne__(self, other):
        return not self.__eq__(other)

    @_new_obj
    def __neg__(self):
        state_updates = {
            #'name': "-%s" % self.name,
            #'tex': r"-%s" % self.tex,
            'hist': -self.hist,
        }
        return state_updates

    @_new_obj
    def __pow__(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            state_updates = {
                #'name': "%s**%s" % (self.name, other),
                #'tex': "%s^{%s}" % (self.tex, other),
                'hist': np.power(self.hist, other)
            }
        elif isinstance(other, np.ndarray):
            state_updates = {
                #'name': "%s**(array)" % self.name,
                #'tex': r"%s^{X}" % self.tex,
                'hist': np.power(self.hist, other),
            }
        elif isinstance(other, Map):
            state_updates = {
                #'name': "%s**(%s)" % (self.name,
                #                      strip_outer_parens(other.name)),
                #'tex': r"%s^{%s}" % (self.tex, strip_outer_parens(other.tex)),
                'hist': np.power(self.hist, other.hist),
                'full_comparison': (self.full_comparison or
                                    other.full_comparison),
            }
        else:
            type_error(other)
        return state_updates

    def __radd__(self, other):
        return self + other

    def __rdiv__(self, other):
        if isinstance(other, Map):
            return other / self
        return self.__rdiv(other)

    @_new_obj
    def __rdiv(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            state_updates = {
                #'name': "(%s / %s)" % (other, self.name),
                #'tex': "{(%s / %s)}" % (other, self.tex),
                'hist': other / self.hist,
            }
        elif isinstance(other, np.ndarray):
            state_updates = {
                #'name': "array / %s" % self.name,
                #'tex': "{(X / %s)}" % self.tex,
                'hist': other / self.hist,
            }
        else:
            type_error(other)
        return state_updates

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        if isinstance(other, Map):
            return other - self
        return self.__rsub(other)

    @_new_obj
    def __rsub(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            state_updates = {
                #'name': "(%s - %s)" % (other, self.name),
                #'tex': "{(%s - %s)}" % (other, self.tex),
                'hist': other - self.hist,
            }
        elif isinstance(other, np.ndarray):
            state_updates = {
                #'name': "(array - %s)" % self.name,
                #'tex': "{(X - %s)}" % self.tex,
                'hist': other - self.hist,
            }
        else:
            type_error(other)
        return state_updates

    @_new_obj
    def sqrt(self):
        """Take square root of map's values, returning a new map.

        Returns
        -------
        sqrt_map : Map

        """
        state_updates = {
            #'name': "sqrt(%s)" % self.name,
            #'tex': r"\sqrt{%s}" % self.tex,
            #'hist': np.asarray(unp.sqrt(self.hist), dtype='float'),
            'hist': unp.sqrt(self.hist),
        }
        return state_updates

    @_new_obj
    def __sub__(self, other):
        if np.isscalar(other) or type(other) is uncertainties.core.Variable:
            state_updates = {
                #'name': "(%s - %s)" % (self.name, other),
                #'tex': "{(%s - %s)}" % (self.tex, other),
                'hist': self.hist - other,
            }
        elif isinstance(other, np.ndarray):
            state_updates = {
                #'name': "(%s - array)" % self.name,
                #'tex': "{(%s - X)}" % self.tex,
                'hist': self.hist - other,
            }
        elif isinstance(other, Map):
            state_updates = {
                #'name': "%s - %s" % (self.name, other.name),
                #'tex': "{(%s - %s)}" % (self.tex, other.tex),
                'hist': self.hist - other.hist,
                'full_comparison': (self.full_comparison or
                                    other.full_comparison),
            }
        else:
            type_error(other)
        return state_updates

# TODO: instantiate individual maps from dicts if passed as such, so user
# doesn't have to instantiate each map. Also, check for name collisions with
# one another and with attrs (so that __getattr__ can retrieve the map by name)

# TODO: add docstrings

class MapSet(object):
    """
    Ordered set of event rate maps (aka histograms) defined over an arbitrary
    regluar hyper-rectangular binning.


    Parameters
    ----------
    maps : Map or sequence of Map

    name : string

    tex : string

    hash : immutable

    collate_by_name : bool
        If True, when this MapSet is passed alongside another MapSet to a
        function that operates on the maps, contained maps in each will be
        accessed by name. Hence, only maps with the same names will be operated
        on simultaneously.

        If false, the contained maps in each MapSet will be accessed by their
        order in each MapSet. This behavior is useful if maps are renamed
        through some operation but their order is maintained, and then
        comparisons are sought with their progenitors with the original
        (different) name.

    """
    __slots = ('_name', '_hash', 'hash')
    __state_attrs = ('name', 'maps', 'tex', 'hash', 'collate_by_name')
    def __init__(self, maps, name=None, tex=None, hash=None,
                 collate_by_name=True):
        if isinstance(maps, Map):
            maps = [maps]

        maps_ = []
        for m in maps:
            if isinstance(m, Map):
                maps_.append(m)
            elif isinstance(m, MapSet):
                maps_.extend(m)
            else:
                maps_.append(Map(**m))

        super().__setattr__('maps', maps_)
        super().__setattr__('name', name)
        super().__setattr__('tex', tex)
        super().__setattr__(
            'collate_by_name', collate_by_name
        )
        super().__setattr__('collate_by_num', not collate_by_name)
        self.hash = hash

    def __repr__(self):
        previous_precision = np.get_printoptions()['precision']
        np.set_printoptions(precision=18)
        try:
            argstrs = [('%s=%r' % item) for item in
                       self.serializable_state.items()]
            r = '%s(%s)' % (self.__class__.__name__, ',\n    '.join(argstrs))
        finally:
            np.set_printoptions(precision=previous_precision)
        return r

    def __str__(self):
        state = OrderedDict()
        attrs = ['name', 'tex', 'hash', 'maps']
        state['name'] = repr(self.name)
        state['tex'] = repr(self.tex)
        state['hash'] = repr(self.hash)
        state['maps'] = ('[\n' + ' '*8 + '%s    \n]'
                         % ',\n        '.join([str(m) for m in self]))
        argstrs = [('%s=%s' % (a, state[a])) for a in attrs]
        s = '%s(%s)' % (self.__class__.__name__, ',\n    '.join(argstrs))
        return s

    def __pretty__(self, p, cycle):
        """Method used by the `pretty` library for formatting"""
        myname = self.__class__.__name__
        if cycle:
            p.text('%s(...)' % myname)
        else:
            p.begin_group(4, '%s(' % myname)
            attrs = ['name', 'tex', 'hash', 'maps']
            for n, attr in enumerate(attrs):
                p.breakable()
                p.text(attr + '=')
                p.pretty(getattr(self, attr))
                if n < len(attrs)-1:
                    p.text(',')
            p.end_group(4, ')')

    def _repr_pretty_(self, p, cycle):
        """Method used by e.g. ipython/Jupyter for formatting"""
        return self.__pretty__(p, cycle)

    @property
    def serializable_state(self):
        """OrderedDict : all state needed to reconstruct object"""
        state = OrderedDict()
        state['maps'] = [m.serializable_state for m in self]
        state['name'] = self.name
        state['tex'] = self._tex
        state['collate_by_name'] = self.collate_by_name
        return state

    def __getstate__(self):
        return self.serializable_state

    def __setstate__(self, state):
        self.__init__(**state)

    def to_json(self, filename, **kwargs):
        """Serialize the state to a JSON file that can be instantiated as a new
        object later.

        Parameters
        ----------
        filename : str
            Filename; must be either a relative or absolute path (*not
            interpreted as a PISA resource specification*)
        **kwargs
            Further keyword args are sent to `pisa.utils.jsons.to_json()`

        See Also
        --------
        from_json : Intantiate new object from the file written by this method
        pisa.utils.jsons.to_json

        """
        jsons.to_json(self.serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, resource):
        """Instantiate a new MapSet object from a JSON file.

        The format of the JSON is generated by the `MapSet.to_json` method,
        which converts a MapSet object to basic types and then numpy arrays are
        converted in a call to `pisa.utils.jsons.to_json`.

        Parameters
        ----------
        resource : str
            A PISA resource specification (see pisa.utils.resources)

        See Also
        --------
        to_json
        pisa.utils.jsons.to_json

        """
        state = jsons.from_json(resource)
        # State is a dict for Map, so instantiate with double-asterisk syntax
        return cls(**state)

    def index(self, x):
        """Find map corresponding to `x` and return its index. Accepts either
        an integer index or a map name to make interface consistent.

        Parameters
        ----------
        x : int, string, or Map
            Map, map name, or integer index of map in this MapSet. If a Map is
            passed, only its name is matched to the maps in this set.

        Returns
        -------
        integer index to the map

        """
        try:
            if isinstance(x, int):
                l = len(self)
                assert x >= -l and x < l
            elif isinstance(x, Map):
                x = self.names.index(x.name)
            elif isinstance(x, str):
                x = self.names.index(x)
            else:
                raise TypeError('Unhandled type "%s" for `x`' % type(x))
        except (AssertionError, ValueError):
            raise ValueError(
                "A map corresponding to '%s' cannot be found in the set."
                " Valid maps are %s" % (x, self.names)
            )
        return x

    def pop(self, *args): #x=None):
        """Remove a map and return it. If a value is passed, the map
        corresponding to `index(value)` is removed and returned instead.

        Parameters
        ----------
        x (optional) : int, string, or Map
            Map, map name, or integer index of map in this MapSet. If a Map is
            passed, only its name is matched to the maps in this set.

        Returns
        -------
        Map removed from this MapSet

        See Also
        --------
        list.pop

        """
        if len(args) == 0:
            m = self.maps.pop()
        elif len(args) == 1:
            idx = self.index(args[0])
            m = self.maps.pop(idx)
        else:
            raise ValueError('`pop` takes 0 or 1 argument; %d passed instead.'
                             % len(args))
        return m

    # TODO: add different aggregation options OR rename to sum_{wildcard|re}
    def combine_re(self, regexes):
        r"""For each regex passed, add together contained maps whose names
        match.

        If a string or regex is passed, the corresponding maps are combined and
        returned as a Map object. If an iterable of one or more regexes is
        passed, each grouping found is combined into a Map separately and the
        resulting Maps are populated into a new MapSet to be returned.

        Parameters
        ----------
        regexes : compiled regex, str representing a regex, or iterable thereof
            See Python module `re` for formatting.

        Returns
        -------
        combined
            Map if `regexes` is a string or regex; MapSet if `regexes` is an
            iterable of one or more strings or regexes

        Raises
        ------
        ValueError
            If any `regexes` fail to match any map names.

        Notes
        -----
        If special characters are used in the regex, like a backslash, be sure
        to use a Python raw string (which does not interpret such special
        characters) by prefixing the string with an "r". E.g., the regex to
        match a period requires passing
            `regex=r'\.'`

        Examples
        --------
        Get total of trck and cscd maps, which are named with suffixes "trck"
        and "cscd", respectively.

        >>> total_trck_map = outputs.combine_re('.*trck')
        >>> total_cscd_map = outputs.combine_re('.*cscd')

        Get a MapSet with both of the above maps in it (and a single command)

        >>> total_pid_maps = outputs.combine_re(['.*trck', '.*cscd'])

        Strict name-checking, combine  nue_cc + nuebar_cc, including both
        cascades and tracks.

        >>> nue_cc_nuebar_cc_map = outputs.combine_re(
        ...     '^nue(bar){0,1}_cc_(cscd|trck)$')

        Lenient nue_cc + nuebar_cc including both cascades and tracks.

        >>> nue_cc_nuebar_cc_map = outputs.combine_re('nue.*_cc_.*')

        Total of all maps

        >>> total = outputs.combine_re('.*')

        See Also
        --------
        combine_wildcard
            Similar method but using wildcards (i.e., globbing, like filename
            matching in the Unix shell)

        References
        ----------
        re : Python module used for parsing regular expressions
            https://docs.python.org/2/library/re.html

        """
        is_scalar = False
        if isinstance(regexes, (str, re.Pattern)):
            is_scalar = True
            regexes = [regexes]

        resulting_maps = []
        for regex in regexes:
            if hasattr(regex, 'pattern'):
                pattern = regex.pattern
            else:
                pattern = regex
            maps_to_combine = []
            names_to_combine = []
            for m in self:
                name = m.name
                if re.match(regex, name) is not None:
                    logging.debug('Map "%s" will be added...', name)
                    maps_to_combine.append(m)
                    names_to_combine.append(name)
            if len(maps_to_combine) == 0:
                raise ValueError('No map names match `regex` "%s"' % pattern)
            if len(maps_to_combine) > 1:
                m = reduce(add, maps_to_combine)
                try:
                    nufig = NuFlavIntGroup(names_to_combine)
                    new_name = make_valid_python_name(str(nufig))
                    new_tex = nufig.tex
                except:
                    # Reasonable name for giving user an idea of what the map
                    # represents
                    new_name = make_valid_python_name(regex)
                    new_tex = None
                if new_name == '':
                    new_name = 'combined'
                m.name = new_name
                m.tex = new_tex
            else:
                m = copy(maps_to_combine[0])
            resulting_maps.append(m)

        if is_scalar:
            combined = resulting_maps[0]
        else:
            combined = MapSet(maps=resulting_maps,
                              name=self.name,
                              tex=self.tex,
                              collate_by_name=self.collate_by_name)
        return combined

    def combine_wildcard(self, expressions):
        """For each expression passed, add together contained maps whose names
        match.

        Expressions can contain wildcards like those used in the Unix shell.

        Valid wildcards (from fnmatch docs, link below):
            "*" : matches everything
            "?" : mateches any single character
            "[`seq`]" : matches any character in `seq`
            "[!`seq`]" : matches any character not in `seq`

        Note that if a string is passed, the matching maps are combined and
        returned as a Map object. If an iterable of strings is passed, each
        grouping found is combined into a Map separately and the resulting Maps
        are populated into a new MapSet to be returned.

        Parameters
        ----------
        expressions : string or sequence thereof
            See Python module `fnmatch` for more info.

        Returns
        -------
        combined
            Map if `expressions` is a string; MapSet if `expressions` is an
            iterable of one or more strings

        Raises
        ------
        ValueError
            If any `expressions` fail to match any map names.

        Examples
        --------
        >>> total_trck_map = outputs.combine_wildcard('*trck')
        >>> total_cscd_map = outputs.combine_wildcard('*cscd')
        >>> total_pid_maps = outpubs.combine_wildcard(['*trck', '*cscd'])
        >>> nue_cc_nuebar_cc_map = outputs.combine_wildcard('nue*_cc_*')
        >>> total = outputs.combine_wildcard('*')

        See Also
        --------
        combine_re : similar method but using regular expressions

        References
        ----------
        fnmatch : Python module used for parsing the expression with wildcards
            https://docs.python.org/2/library/fnmatch.html

        """
        is_scalar = False
        if isinstance(expressions, str):
            is_scalar = True
            expressions = [expressions]

        resulting_maps = []
        for expr in expressions:
            maps_to_combine = []
            names_to_combine = []
            for mapnum, m in enumerate(self):
                name = m.name
                if fnmatch(name, expr):
                    logging.debug('Map %d, "%s", will be added...',
                                  mapnum, name)
                    maps_to_combine.append(m)
                    names_to_combine.append(name)
            if len(maps_to_combine) == 0:
                raise ValueError('No map names match `expr` "%s"' % expr)
            if len(maps_to_combine) > 1:
                m = reduce(add, maps_to_combine)
                try:
                    nufig = NuFlavIntGroup(names_to_combine)
                    new_name = make_valid_python_name(str(nufig))
                    new_tex = nufig.tex
                except:
                    # Reasonable name for giving user an idea of what the map
                    # represents
                    new_name = make_valid_python_name(expr)
                    new_tex = None
                if new_name == '':
                    new_name = 'combined'
                m.name = new_name
                m.tex = new_tex
            else:
                m = copy(maps_to_combine[0])
            resulting_maps.append(m)

        if is_scalar:
            combined = resulting_maps[0]
        else:
            combined = MapSet(maps=resulting_maps,
                              name=self.name,
                              tex=self.tex,
                              collate_by_name=self.collate_by_name)
        return combined

    def compare(self, ref):
        """Compare maps in this MapSet against a reference MapSet.

        Parameters
        ----------
        ref : MapSet
            Maps taken as the reference against which to compare maps
            contained within this MapSet.

        Returns
        -------
        stats : OrderedDict
            Each key is the name of a map, and each value is istelf an
            OrderedDict as returned by the `Map.compare` method

        Examples
        --------
        >>> stats = map_set_test.compare(map_set_ref)

        """
        assert isinstance(ref, MapSet) and len(self) == len(ref)
        rslt = OrderedDict()
        for m, r in zip(self, ref):
            out = m.compare(r)
            rslt[m.name] = out
        return rslt

    def __eq__(self, other):
        return recursiveEquality(self.hashable_state, other.hashable_state)

    @property
    def name(self):
        """string : name of the map (legal Python name)"""
        return super().__getattribute__('_name')

    @name.setter
    def name(self, name):
        """string : name of the map (legal Python name)"""
        return super().__setattr__('_name', name)

    @property
    def hash(self):
        """Hash value of the map set is based upon the contained maps.
            * If all maps have the same hash value, this value is returned as
              the map set's hash
            * If one or more maps have different hash values, a list of the
              contained maps' hash values is hashed
            * If any contained map has None for hash value, the hash value of
              the map set is also None (i.e., invalid)

        """
        hashes = self.hashes
        if all([(h is not None and h == hashes[0]) for h in hashes]):
            return hashes[0]
        if all([(h is not None) for h in hashes]):
            return hash_obj(hashes)
        return None

    @hash.setter
    def hash(self, val):
        """Setting a hash to `val` for the map set sets the hash values of all
        contained maps to `val`."""
        if val is not None:
            for m in self:
                setattr(m, 'hash', val)

    @property
    def names(self):
        """list of strings : name of each map"""
        return [mp.name for mp in self]

    @property
    def hashes(self):
        """list of int : hash of each map"""
        return [mp.hash for mp in self]

    def hash_maps(self, map_names=None):
        """Generate a hash on the contained maps (i.e. exclude state pertaining
        only to the MapSet itself, but include all state pertaining to the
        contained Maps).

        Parameters
        ----------
        map_names : None or sequence of strings
            If sequence of strings, use these as the map names instead of any
            names contained.

        Returns
        -------
        hash : None or int
            If any contained map hashes to None, the resulting hash will also
            be None.

        """
        if map_names is None:
            map_names = [m.name for m in self]
        hashes = [m.hash for m in self if m.name in map_names]
        if all([(h != None) for h in hashes]):
            return hash_obj(hashes)
        return None

    def collate_with_names(self, vals):
        ret_dict = OrderedDict()
        for name, val in zip(self.names, vals):
            setitem(ret_dict, name, val)
        return ret_dict

    def find_map(self, value):
        idx = None
        if isinstance(value, Map):
            pass
        elif isinstance(value, str):
            try:
                idx = self.names.index(value)
            except ValueError:
                pass
        if idx is None:
            raise ValueError('Could not find map name "%s" among maps %s'
                             % (value, self.names))
        return self[idx]

    def apply_to_maps(self, attr, *args, **kwargs):
        if len(kwargs) != 0:
            raise NotImplementedError('Keyword arguments are not handled')
        if hasattr(attr, '__name__'):
            attrname = attr.__name__
        else:
            attrname = attr
        do_not_have_attr = np.array([(not hasattr(mp, attrname))
                                     for mp in self.maps])
        if np.any(do_not_have_attr):
            missing_in_names = ', '.join(
                np.array(self.names)[do_not_have_attr]
            )
            num_missing = np.sum(do_not_have_attr)
            num_total = len(do_not_have_attr)
            raise AttributeError(
                'Maps %s (%d of %d maps in set) do not have attribute "%s"'
                % (missing_in_names, num_missing, num_total, attrname)
            )

        # Retrieve the corresponding callables from contained maps
        val_per_map = [getattr(mp, attr) for mp in self]

        if not all([hasattr(meth, '__call__') for meth in val_per_map]):
            # If all results are maps, populate a new map set & return that
            if all([isinstance(r, Map) for r in val_per_map]):
                return MapSet(maps=val_per_map, name=self.name, tex=self.tex,
                              collate_by_name=self.collate_by_name)

            # Otherwise put in an ordered dict with <name>: <val> pairs ordered
            # according to the map ordering in this map set
            return self.collate_with_names(val_per_map)

        # Rename for clarity
        method_per_map = val_per_map

        # Create a set of args for *each* map in this map set: If an arg is a
        # MapSet, convert that arg into the map in that set corresponding to
        # the same map in this set.
        args_per_map = []
        for map_num, mp in enumerate(self):
            map_name = mp.name
            this_map_args = []
            for arg in args:
                if (np.isscalar(arg) or
                        type(arg) is uncertainties.core.Variable or
                        isinstance(arg, (str, np.ndarray))):
                    this_map_args.append(arg)
                elif isinstance(arg, MapSet):
                    if self.collate_by_name:
                        this_map_args.append(arg[map_name])
                    elif self.collate_by_num:
                        this_map_args.append(arg[map_num])

                # TODO: test to make sure this works for e.g. metric_per_map
                elif isinstance(arg, Iterable):
                    list_arg = []
                    for item in arg:
                        if isinstance(item, MapSet):
                            if self.collate_by_name:
                                list_arg.append(item[map_name])
                            elif self.collate_by_num:
                                list_arg.append(item[map_num])
                    this_map_args.append(list_arg)
                else:
                    raise TypeError('Unhandled arg %s / type %s'
                                    % (arg, type(arg)))
            args_per_map.append(this_map_args)

        # Make the method calls and collect returned values
        returned_vals = [meth(*args)
                         for meth, args in zip(method_per_map, args_per_map)]

        # If all results are maps, put them into a new map set & return
        if all([isinstance(r, Map) for r in returned_vals]):
            return MapSet(maps=returned_vals, name=self.name, tex=self.tex,
                          collate_by_name=self.collate_by_name)

        # If None returned by all, return a single None
        if all([(r is None) for r in returned_vals]):
            return None

        # Otherwise put into an ordered dict with name: val pairs
        return self.collate_with_names(returned_vals)

    def __contains__(self, name):
        return name in [m.name for m in self]

    #def __setattr__(self, attr, val):
    #    print '__setattr__ being accessed, attr = %s, val = %s' %(attr, val)
    #    if attr in MapSet.__slots:
    #        print 'attr "%s" found in MapSet.slots.' %attr
    #        object.__setattr__(self, attr, val)
    #    else:
    #        returned_vals = [setattr(mp, attr, val) for mp in self]
    #        if all([(r is None) for r in returned_vals]):
    #            return
    #        return self.collate_with_names(returned_vals)

    def __getattr__(self, attr):
        if attr in [m.name for m in self]:
            return self[attr]
        return self.apply_to_maps(attr)

    def __iter__(self):
        return iter(self.maps)

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, item):
        """Retrieve a map by name or retrieve maps' histogram values by index
        or slice.

        If `item` is a string, retrieve map by name.
        If `item is an integer or one-dim slice, retrieve maps by index/slice
        If `item` is length-2 tuple or two-dim slice, retrieve value(s) of all
            contained maps, each indexed by map[`item`]. The output is returned
            in an ordered dict with format {<map name>: <values>, ...}

        """
        if isinstance(item, str):
            return self.find_map(item)

        if isinstance(item, (int, slice)):
            rslt = self.maps[item]
            if hasattr(rslt, '__len__') and len(rslt) > 1:
                return MapSet(maps=rslt, name=self.name, tex=self.tex,
                              collate_by_name=self.collate_by_name)
            return rslt

        if isinstance(item, Iterable):
            if not isinstance(item, Sequence):
                item = list(item)

            if len(item) == 1:
                return self.maps[item]

            if len(item) == 2:
                return MapSet(maps=[getitem(m, item) for m in self],
                              name=self.name, tex=self.tex,
                              collate_by_name=self.collate_by_name)

            raise IndexError('too many indices for 2D hist')

        raise TypeError('getitem does not support `item` of type %s'
                        % type(item))

    def __abs__(self):
        return self.apply_to_maps('__abs__')

    def __add__(self, val):
        return self.apply_to_maps('__add__', val)

    def __truediv__(self, val):
        return self.apply_to_maps('__truediv__', val)

    def __div__(self, val):
        return self.apply_to_maps('__div__', val)

    def log(self):
        return self.apply_to_maps('log')

    def log10(self):
        return self.apply_to_maps('log10')

    def __mul__(self, val):
        return self.apply_to_maps('__mul__', val)

    def __neg__(self):
        return self.apply_to_maps('__neg__')

    def __pow__(self, val):
        return self.apply_to_maps('__pow__', val)

    def __radd__(self, val):
        return self.apply_to_maps('__radd__', val)

    def __rdiv__(self, val):
        return self.apply_to_maps('__rdiv__', val)

    def __rmul__(self, val):
        return self.apply_to_maps('__rmul__', val)

    def __rsub__(self, val):
        return self.apply_to_maps('__rsub__', val)

    def sqrt(self):
        return self.apply_to_maps('sqrt')

    def __sub__(self, val):
        return self.apply_to_maps('__sub__', val)

    def sum(self, *args, **kwargs):
        return MapSet(maps=[m.sum(*args, **kwargs) for m in self],
                      name=self.name, tex=self.tex,
                      collate_by_name=self.collate_by_name)

    def project(self, axis, keepdims=False):
        """Per-map projections onto single axis. See Map.project for more
        detailed help.

        Parameters
        ----------
        axis : string or int
        keepdims : bool

        Returns
        -------
        projection : MapSet
            Each map in this MapSet projected onto `axis`.

        See Also
        --------
        sum
            Sum over specified dimension(s)

        Map.project
            Method called for each map in this MapSet to perform the actual
            projection.

        """
        return MapSet(maps=[m.project(axis=axis, keepdims=keepdims)
                            for m in self],
                      name=self.name, tex=self.tex,
                      collate_by_name=self.collate_by_name)

    def reorder_dimensions(self, order):
        """Return a new MapSet object with dimensions ordered
        according to `order`.

        Parameters
        ----------
        order : MultiDimBinning or sequence of string, int, or OneDimBinning
            Order of dimensions to use. Strings are interpreted as dimension
            basenames, integers are interpreted as dimension indices, and
            OneDimBinning objects are interpreted by their `basename`
            attributes (so e.g. the exact binnings in `order` do not have to
            match this object's exact binnings; only their basenames). Note
            that a MultiDimBinning object is a valid sequence type to use for
            `order`.

        Notes
        -----
        Dimensions specified in `order` that are not in this object are
        ignored, but dimensions in this object that are missing in `order`
        result in an error.

        Returns
        -------
        MapSet object with reordred dimensions.

        Raises
        ------
        ValueError if dimensions present in this object are missing from
        `order`.

        """
        return MapSet(maps=[m.reorder_dimensions(order=order) for m in self],
                      name=self.name, tex=self.tex,
                      collate_by_name=self.collate_by_name)

    def squeeze(self):
        """Remove any singleton dimensions (i.e. that have only a single bin)
        from all contained maps. Analagous to `numpy.squeeze`.

        Returns
        -------
        MapSet with equivalent values but singleton Map dimensions removed

        """
        return MapSet(maps=[m.squeeze() for m in self], name=self.name,
                      tex=self.tex, collate_by_name=self.collate_by_name)

    def rebin(self, *args, **kwargs):
        return MapSet(maps=[m.rebin(*args, **kwargs) for m in self],
                      name=self.name, tex=self.tex,
                      collate_by_name=self.collate_by_name)

    def downsample(self, *args, **kwargs):
        return MapSet(maps=[m.downsample(*args, **kwargs) for m in self],
                      name=self.name, tex=self.tex,
                      collate_by_name=self.collate_by_name)

    def metric_per_map(self, expected_values, metric):
        if isinstance(metric, str):
            metric = metric.lower()
            if 'binned_' in metric:
                metric = metric.replace('binned_', '')
                binned = True
            else:
                binned = False
        if metric in stats.ALL_METRICS:
            return self.apply_to_maps(metric, expected_values, binned)
        else:
            raise ValueError('`metric` "%s" not recognized; use one of %s.'
                             % (metric, stats.ALL_METRICS))

    def metric_total(self, expected_values, metric):
        return np.sum(self.metric_per_map(expected_values, metric).values())

    def chi2_per_map(self, expected_values):
        return self.apply_to_maps('chi2', expected_values)

    def chi2_total(self, expected_values):
        return np.sum(self.chi2_per_map(expected_values))

    def fluctuate(self, method, random_state=None, jumpahead=0):
        """Add fluctuations to the maps in the set and return as a new MapSet.

        Parameters
        ----------
        method : None or string
        random_stae : None, numpy.random.RandomState, or seed spec

        """
        random_state = get_random_state(random_state=random_state,
                                        jumpahead=jumpahead)
        new_maps = [m.fluctuate(method=method, random_state=random_state)
                    for m in self]
        return MapSet(maps=new_maps, name=self.name, tex=self.tex, hash=None,
                      collate_by_name=self.collate_by_name)

    def llh_per_map(self, expected_values):
        return self.apply_to_maps('llh', expected_values)

    def llh_total(self, expected_values):
        return np.sum(self.llh(expected_values))

    def set_poisson_errors(self):
        return self.apply_to_maps('set_poisson_errors')

## Now dynamically add all methods from Map to MapSet that don't already exist
## in MapSet (and make these methods distribute to contained maps)
##for method_name, method in sorted(Map.__dict__.items()):
#add_methods = '''__abs__ __add__ __div__ __mul__ __neg__ __pow__ __radd__
#__rdiv__ __rmul__ __rsub__ __sub__'''.split()
#
#for method_name in add_methods:
#    #if not hasattr(method, '__call__') or method_name in MapSet.__dict__:
#    #    continue
#    disallowed = ('__getattr__', '__setattr__', '__getattribute__',
#                  '__getitem__', '__eq__', '__ne__', '__str__', '__repr__')
#    if method_name in disallowed:
#        continue
#    print 'adding method "%s" to MapSet as an apply func' % method_name
#    arg_str = ', *args' # if len(args) > 0 else ''
#    eval(('def {method_name}(self{arg_str}):\n'
#          '    return self.apply_to_maps({method_name}{arg_str})')
#          .format(method_name=method_name, arg_str=arg_str))
#    #f.__doc__ = 'Apply method %s to all contained maps' % method_name
#    #method = getattr(Map, method_name)
#    #if method.__doc__:
#    #    f.__doc__ += '... ' + method.__doc__
#    setattr(MapSet, method_name, MethodType(eval(method_name), None, MapSet))


# TODO: add tests for llh, chi2 methods
def test_Map():
    """Unit tests for Map class"""
    import pickle
    n_ebins = 10
    n_czbins = 5
    n_azbins = 2
    e_binning = OneDimBinning(name='energy', tex=r'E_\nu', num_bins=n_ebins,
                              domain=(1, 80)*ureg.GeV, is_log=True)
    cz_binning = OneDimBinning(name='coszen', tex=r'\cos\,\theta',
                               num_bins=n_czbins, domain=(-1, 0), is_lin=True)
    az_binning = OneDimBinning(name='azimuth', tex=r'\phi',
                               num_bins=n_azbins, domain=(0, 2*np.pi),
                               is_lin=True)
    # set directly unumpy array with errors
    shape = (e_binning * cz_binning).shape
    m1 = Map(name='x',
             hist=unp.uarray(np.ones(shape), np.sqrt(np.ones(shape))),
             binning=(e_binning, cz_binning))
    # or call init poisson error afterwards
    m1 = Map(name='x', hist=np.ones((n_ebins, n_czbins)), hash=23,
             binning=(e_binning, cz_binning))

    # Test rebin
    _ = m1.rebin(m1.binning.downsample(2, 5))
    m_rebinned = m1.rebin(m1.binning.downsample(n_ebins, n_czbins))
    assert m_rebinned.hist[0, 0] == np.sum(m1.hist)


    # Test sum()
    m1 = Map(
        name='x',
        hist=np.arange(0, n_ebins*n_czbins).reshape((n_ebins, n_czbins)),
        binning=(e_binning, cz_binning)
    )
    s1 = m1.sum('energy', keepdims=True)
    assert np.all(s1.hist == np.array([[225, 235, 245, 255, 265]]))
    assert s1.shape == (1, 5)
    assert 'energy' in s1.binning
    assert 'coszen' in s1.binning
    s2 = m1.sum('energy', keepdims=False)
    assert np.all(s2.hist == np.array([225, 235, 245, 255, 265]))
    assert s2.shape == (5,)
    assert 'energy' not in s2.binning
    assert 'coszen' in s2.binning

    m1 = Map(name='x', hist=np.ones((n_ebins, n_czbins)), hash=23,
             binning=(e_binning, cz_binning))
    logging.debug(str(("downsampling =====================")))
    m2 = m1 * 2
    logging.debug(str((m2.downsample(1))))
    logging.debug(str((m2.downsample(5))))
    logging.debug(str((m2.downsample(1, 1))))
    logging.debug(str((m2.downsample(1, 5))))
    logging.debug(str((m2.downsample(5, 5))))
    logging.debug(str((m2.downsample(10, 5))))
    logging.debug(str((m2.downsample(10, 5).binning)))
    logging.debug(str(("===================== downsampling")))

    assert m1.hash == 23
    m1.hash = 42
    assert m1.hash == 42
    m1.set_poisson_errors()
    # or no errors at all
    m2 = Map(name='y', hist=2*np.ones((n_ebins, n_czbins)),
             binning=(e_binning, cz_binning))
    m3 = Map(name='z', hist=4*np.ones((n_ebins, n_czbins, n_azbins)),
             binning=(e_binning, cz_binning, az_binning))

    assert m3[0, 0, 0] == 4, 'm3[0, 0, 0] = %s' % m3[0, 0, 0]
    testdir = tempfile.mkdtemp()
    try:
        for m in [m1, m2, m1+m2, m1-m2, m1/m2, m1*m2]:
            m_file = os.path.join(testdir, m.name + '.json')
            m.to_json(m_file, warn=False)
            m_ = Map.from_json(m_file)
            assert m_ == m, 'm=\n%s\nm_=\n%s' % (m, m_)
            jsons.to_json(m, m_file, warn=False)
            m_ = Map.from_json(m_file)
            assert m_ == m, 'm=\n%s\nm_=\n%s' % (m, m_)
            # Had bug where datastruct containing MapSet failed to be saved.
            # Test tuple containing list containing OrderedDict containing
            # Map here.
            struct = ([OrderedDict(map=m)],)
            jsons.to_json(struct, m_file, warn=False)
            loaded = jsons.from_json(m_file)
            m_ = Map(**loaded[0][0]['map'])
            assert m_ == m
            # Now try with pickle
            m_file = os.path.join(testdir, m.name + '.pkl')
            pickle.dump(struct, open(m_file, 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
            loaded = pickle.load(open(m_file, 'rb'))
            m_ = loaded[0][0]['map']
            assert m_ == m
    finally:
        shutil.rmtree(testdir, ignore_errors=True)

    logging.debug(str((m1, m1.binning)))
    logging.debug(str((m2, m2.binning)))
    logging.debug(str((m1.nominal_values)))
    logging.debug(str((m1.std_devs)))
    r = m1 + m2
    # compare only nominal val
    assert r == 3
    logging.debug(str((r)))
    logging.debug(str(('m1+m2=3:', r, r[0, 0])))
    r = m2 + m1
    # or compare including errors
    assert r == ufloat(3, 1)
    logging.debug(str(('m2+m1=3:', r, r[0, 0])))
    r = 2*m1
    assert r == ufloat(2, 2), str(r.hist)
    logging.debug(str(('2*m1=2:', r, r[0, 0])))
    r = (2*m1 + 8) / m2
    logging.debug(str(('(2*(1+/-1) + 8) / 2:', r, r[0, 0])))
    assert r == ufloat(5, 1), str(r.hist)
    logging.debug(str(('(2*m1 + 8) / m2=5:', r, r.hist[0, 0])))
    #r[:, 1] = 1
    #r[2, :] = 2
    logging.debug(str(('r[0:2, 0:5].hist:', r[0:2, 0:5].hist)))
    logging.debug(str(('r[0:2, 0:5].binning:', r[0:2, 0:5].binning)))
    r = m1 / m2
    assert r == ufloat(0.5, 0.5)
    logging.debug(str((r, '=', r[0, 0])))
    logging.debug(str(([b.binning.energy.midpoints[0]
                        for b in m1.iterbins()][0:2])))

    # Test reorder_dimensions
    e_binning = OneDimBinning(
        name='energy', num_bins=2, is_log=True, domain=[1, 80]*ureg.GeV
    )
    cz_binning = OneDimBinning(
        name='coszen', num_bins=3, is_lin=True, domain=[-1, 1]
    )
    az_binning = OneDimBinning(
        name='azimuth', num_bins=4, is_lin=True,
        domain=[0, 2*np.pi]*ureg.rad
    )
    a = []
    for i in range(len(e_binning)):
        b = []
        for j in range(len(cz_binning)):
            c = []
            for k in range(len(az_binning)):
                c.append(i*100 + j*10 + k)
            b.append(c)
        a.append(b)
    a = np.array(a)
    m_orig = Map(name='orig', hist=deepcopy(a),
                 binning=[e_binning, cz_binning, az_binning])
    m_new = m_orig.reorder_dimensions(['azimuth', 'energy', 'coszen'])

    assert np.alltrue(m_orig[:, 0, 0].hist.flatten() ==
                      m_new[0, :, 0].hist.flatten())
    assert np.alltrue(m_orig[0, :, 0].hist.flatten() ==
                      m_new[0, 0, :].hist.flatten())
    assert np.alltrue(m_orig[0, 0, :].hist.flatten() ==
                      m_new[:, 0, 0].hist.flatten())

    for dim in m_orig.binning.names:
        assert m_orig[:, 0, 0].binning[dim] == m_new[0, :, 0].binning[dim]
        assert m_orig[0, :, 0].binning[dim] == m_new[0, 0, :].binning[dim]
        assert m_orig[0, 0, :].binning[dim] == m_new[:, 0, 0].binning[dim]

    deepcopy(m_orig)

    logging.info(str(('<< PASS : test_Map >>')))


# TODO: add tests for llh, chi2 methods
# TODO: make tests use assert rather than rely on logging.debug(str((!)))
def test_MapSet():
    """Unit tests for MapSet class"""
    import pickle
    n_ebins = 6
    n_czbins = 3
    e_binning = OneDimBinning(name='energy', tex=r'E_\nu', num_bins=n_ebins,
                              domain=(1, 80)*ureg.GeV, is_log=True)
    cz_binning = OneDimBinning(name='coszen', tex=r'\cos\,\theta',
                               num_bins=n_czbins, domain=(-1, 0), is_lin=True)
    binning = MultiDimBinning([e_binning, cz_binning])
    m1 = Map(name='ones', hist=np.ones(binning.shape), binning=binning,
             hash='xyz')
    m1.set_poisson_errors()
    m2 = Map(name='twos', hist=2*np.ones(binning.shape), binning=binning,
             hash='xyz')
    ms01 = MapSet([m1, m2])

    # Test rebin
    _ = ms01.rebin(m1.binning.downsample(3))
    ms01_rebinned = ms01.rebin(m1.binning.downsample(6, 3))
    for m_orig, m_rebinned in zip(ms01, ms01_rebinned):
        assert m_rebinned.hist[0, 0] == np.sum(m_orig.hist)

    logging.debug(str(("downsampling =====================")))
    logging.debug(str((ms01.downsample(3))))
    logging.debug(str(("===================== downsampling")))
    ms01 = MapSet((m1, m2), name='ms01')
    ms02 = MapSet((m1, m2), name='map set 1')
    ms1 = MapSet(maps=(m1, m2), name='map set 1', collate_by_name=True,
                 hash=None)
    assert np.all(ms1.combine_re(r'.*').hist == ms1.combine_wildcard('*').hist)
    assert np.all(ms1.combine_re(r'.*').hist == (ms1.ones + ms1.twos).hist)
    assert np.all(ms1.combine_re(r'^(one|two)s.*$').hist ==
                  ms1.combine_wildcard('*s').hist)
    assert np.all(ms1.combine_re(r'^(one|two)s.*$').hist == (ms1.ones +
                                                             ms1.twos).hist)
    logging.debug(str((ms1.combine_re(r'^o').hist)))
    logging.debug(str((ms1.combine_wildcard(r'o*').hist)))
    logging.debug(str((ms1.combine_re(r'^o').hist
                       - ms1.combine_wildcard(r'o*').hist)))
    logging.debug(str(('map sets equal after combining?',
                       ms1.combine_re(r'^o') == ms1.combine_wildcard(r'o*'))))
    logging.debug(str(('hist equal after combining?',
                       np.all(ms1.combine_re(r'^o').hist ==
                              ms1.combine_wildcard(r'o*').hist))))
    assert np.all(ms1.combine_re(r'^o.*').nominal_values
                  == ms1.combine_wildcard('o*').nominal_values), \
            '%s\n%s' % (ms1.combine_re(r'^o.*'), ms1.combine_wildcard('o*'))
    logging.debug(str(('5', ms1.names)))
    assert np.all(ms1.combine_re(r'^o').nominal_values == ms1.ones.nominal_values)
    assert np.all(ms1.combine_wildcard(r'o*').nominal_values == ms1.ones.nominal_values)
    logging.debug(str(('6', ms1.names)))
    try:
        ms1.combine_re('three')
    except ValueError:
        pass
    else:
        assert False
    try:
        ms1.combine_wildcard('three')
    except ValueError:
        pass
    else:
        assert False

    assert ms1.hash == 'xyz'
    assert ms1.name == 'map set 1'
    ms1.hash = 10
    assert ms1.hash == 10
    assert m1.hash == 10
    assert m2.hash == 10
    ms1.hash = -10
    assert ms1.hash == -10
    # "Remove" the hash from the MapSet...
    ms1.hash = None
    # ... but this should not remove hashes from the contained maps
    assert m1.hash == -10
    # ... and hashing on the MapSet should see that all contained maps have the
    # SAME hash val, and so should just return the hash value shared among them
    # all
    assert ms1.hash == -10
    # However changing a single map's hash means not all hashes are the same
    # for all maps...
    m1.hash = 40
    # ... so a hash should be computed from all contained hashes
    assert ms1.hash != 40 and ms1.hash != -10

    assert ms1.maps == [m1, m2]
    assert ms1.names == ['ones', 'twos']
    assert ms1.tex is None
    # Check the Poisson errors
    assert np.all(ms1[0].nominal_values == np.ones(binning.shape))
    assert np.all(ms1[0].std_devs == np.ones(binning.shape))
    assert np.all(ms1[1].hist == 2*np.ones(binning.shape))
    logging.debug(str(('ms1[0:2].hist:', ms1[0:2].hist)))
    logging.debug(str(('ms1[0:2, 0:2].hist:', ms1[0:2, 0:2].hist)))
    assert np.all(ms1.apply_to_maps('__add__', 1).ones == 2)

    m1 = Map(name='threes', hist=3*np.ones((n_ebins, n_czbins)),
             binning=binning)
    m2 = Map(name='fours', hist=4*np.ones((n_ebins, n_czbins)), binning=binning)
    ms2 = MapSet(maps=(m1, m2), name='map set 2', collate_by_name=False)

    try:
        logging.debug(str((ms1.__add__(ms2))))
    except ValueError:
        pass
    else:
        raise Exception('Should have errored out!')

    m1 = Map(name='fives', hist=5*np.ones((n_ebins, n_czbins)),
             binning=binning)
    m2 = Map(name='sixes', hist=6*np.ones((n_ebins, n_czbins)),
             binning=binning)
    ms3 = MapSet(maps=(m1, m2), name='map set 3', collate_by_name=False)
    ms4 = MapSet(maps=(m1, m2), name='map set 3', collate_by_name=False)
    assert ms3 == ms4

    logging.debug(str(('ms2.maps:', ms2.maps)))
    logging.debug(str(("(ms2 + ms3).names", (ms2 + ms3).names)))
    logging.debug(str(("(ms2 + ms3)[0, 0].hist", (ms2 + ms3)[0, 0].hist)))
    logging.debug(str(("ms1['ones'][0, 0]:", ms1['ones'][0, 0])))
    logging.debug(str(('ms1.__mul__(2)[0, 0]:', ms1.__mul__(2)[0, 0])))
    logging.debug(str(('(ms1 * 2)[0, 0]:', (ms1 * 2)[0, 0])))
    logging.debug(str(('ms1.__add__(ms1)[0, 0]:', ms1.__add__(ms1)[0, 0])))
    logging.debug(str(('(ms1 + ms1)[0, 0]:', (ms1 + ms1)[0, 0])))
    logging.debug(str((ms1.names)))
    logging.debug(str(('(ms1/ ms1)[0, 0]:', (ms1 / ms1)[0, 0])))
    logging.debug(str(('(ms1/ms1 - 1)[0, 0]:', (ms1/ms1 - 1)[0, 0])))
    #logging.debug(str(("ms1.log10()['ones']:", ms1.log10()['ones'])))
    #logging.debug(str(("ms1.log10()[0, 0]['ones']:",
    #                   ms1.log10()[0, 0]['ones'])))
    #logging.debug(str(('np.log10(ms1):', np.log10(ms1))))
    logging.debug(str(('(ms1 * np.e).binning:', (ms1 * np.e).binning)))
    #logging.debug(str(('np.log(ms1 * np.e)[0][0, 0]:',
    #                   (np.log(ms1 * np.e))[0][0, 0])))
    #logging.debug(str(('np.sqrt(ms1)[0][0:4, 0:2].hist:',
    #                   np.sqrt(ms1)[0][0:4, 0:2].hist)))
    logging.debug(str(('str(ms1)', str(ms1))))
    logging.debug(str(('str(ms4)', str(ms4))))
    logging.debug(str(('ms3', ms3)))
    logging.debug(str(('ms4', ms4)))

    testdir = tempfile.mkdtemp()
    try:
        for ms in [ms01, ms02, ms1, ms2, ms3, ms4]:
            ms_file = os.path.join(testdir, ms.name + '.json')
            ms.to_json(ms_file, warn=False)
            ms_ = MapSet.from_json(ms_file)
            assert ms_ == ms, 'ms=\n%s\nms_=\n%s' % (ms, ms_)
            jsons.to_json(ms, ms_file, warn=False)
            ms_ = MapSet.from_json(ms_file)
            assert ms_ == ms, 'ms=\n%s\nms_=\n%s' % (ms, ms_)

            # Had bug where datastruct containing MapSet failed to be saved.
            # Test tuple containing list containing OrderedDict containing
            # MapSet.
            struct = ([OrderedDict(mapset=ms)],)
            jsons.to_json(struct, ms_file, warn=False)
            loaded = jsons.from_json(ms_file)
            ms_ = MapSet(**loaded[0][0]['mapset'])
            assert ms_ == ms

            # Now try with pickle
            ms_file = os.path.join(testdir, ms.name + '.pkl')
            pickle.dump(struct, open(ms_file, 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
            loaded = pickle.load(open(ms_file, 'rb'))
            ms_ = loaded[0][0]['mapset']
            assert ms_ == ms

    finally:
        shutil.rmtree(testdir, ignore_errors=True)

    deepcopy(ms01)

    # Test reorder_dimensions (this just tests that it succeeds on the map set;
    # correctness of the reordering is tested in the unit test for Map)
    for ms in [ms01, ms02, ms1, ms2, ms3, ms4]:
        for p in permutations(ms[0].binning.dimensions):
            ms.reorder_dimensions(p)

    logging.info('<< PASS : test_MapSet >>')


if __name__ == "__main__":
    set_verbosity(1)
    test_Map()
    test_MapSet()
