# author : J.L. Lanfranchi
#          jll1062+pisa@phys.psu.edu
#
# date   : March 25, 2016

"""
Class to carry information about 2D binning in energy and cosine-zenity, and to
provide basic operations with the binning.
"""

# TODO: include Iterables where only Sequence is allowed now?
# TODO: make indexing accessible by name?

from __future__ import division

from collections import Iterable, Mapping, OrderedDict, Sequence
from copy import copy, deepcopy
from functools import wraps
from itertools import izip
import re

import numpy as np
import pint

from pisa import ureg, HASH_SIGFIGS
from pisa.utils.comparisons import isbarenumeric, normQuant, recursiveEquality
from pisa.utils.hash import hash_obj
from pisa.utils import jsons
from pisa.utils.log import logging, set_verbosity


__all__ = ['NAME_FIXES', 'NAME_SEPCHARS', 'NAME_FIXES_REGEXES',
           'basename', '_new_obj',
           'OneDimBinning', 'MultiDimBinning',
           'test_OneDimBinning', 'test_MultiDimBinning']


NAME_FIXES = ['true', 'truth', 'reco', 'reconstructed']
NAME_SEPCHARS = r'([_\s-])*'
NAME_FIXES_REGEXES = [re.compile(p + NAME_SEPCHARS, re.IGNORECASE)
                      for p in NAME_FIXES]


# TODO: move this to a centralized utils location
def basename(n):
    """Remove "true" or "reco" prefix(es) and/or suffix(es) from binning
    name `n` along with any number of possible separator characters.

    * Valid (pre/suf)fix(es): "true", "reco"
    * Valid separator characters: "<whitespace>", "_", "-" (any number)

    Parameters
    ----------
    n : string or OneDimBinning
        Name from which to have pre/suffixes stripped.

    Returns
    -------
    basename : string

    Examples
    --------
    >>> print basename('true_energy')
    'energy'
    >>> print basename('Reconstructed coszen')
    'coszen'
    >>> print basename('energy___truth')
    'energy'

    """
    # Type checkingn and conversion
    orig_type = type(n)
    if isinstance(n, OneDimBinning):
        n = n.name
    if not isinstance(n, basestring):
        raise ValueError('Unhandled type %s' %orig_type)
    # Remove all (pre/suf)fixes and any separator chars
    for regex in NAME_FIXES_REGEXES:
        n = regex.sub('', n)
    return n


def _new_obj(original_function):
    """Decorator to deepcopy unaltered states into new OneDimBinning object."""
    @wraps(original_function)
    def new_function(self, *args, **kwargs):
        new_state = OrderedDict()
        state_updates = original_function(self, *args, **kwargs)
        for attr in self._hash_attrs:
            if attr in state_updates:
                new_state[attr] = state_updates[attr]
            else:
                new_state[attr] = deepcopy(getattr(self, attr))
        return OneDimBinning(**new_state)
    return new_function


class OneDimBinning(object):
    """Histogram-oriented binning specialized to a single dimension.

    Either `domain` or `bin_edges` must be specified, but not both. `is_lin`
    and `is_log` are mutually exclusive and *must* be specified if `domain` is
    provided (along with `num_bins`), but these are optional if `bin_edges` is
    specified.

    In the case that `bin_edges` is provided and defines just a single bin, if
    this bin should be treated logarithmically (e.g. for oversampling),
    `is_log=True` must be specified (otherwise, `is_lin` will be assumed to be
    true).

    Parameters
    ----------
    name : str, of length > 0
        Name for this dimension. Must be valid Python name (since it will be
        accessed with the dot operator).

    tex : str or None
        TeX label for this dimension.

    bin_edges : sequence
        Numerical values (including Pint units, if there are units) that
        represent the *edges* of the bins. `bin_edges` needn't be specified if
        `domain`, `num_bins`, and some combination of `is_lin` and `is_log` are
        specified. Pint units can be attached to `bin_edges`, but will be
        converted to `units` if these are specified.

    units : None, Pint unit or object convertible to Pint unit
        If None, units will be read from either `bin_edges` or `domain`, and if
        none of these have units, the binning has unit 'dimensionless'
        attached.

    is_lin : bool
        If `num_bins` and `domain` are specified,

    is_log : bool
        Whether bin spacing is to be equal on a log-scale. Specify along with
        `domain` to generate `bin_edges` on the fly. On the other hand, if
        `bin_edges` is passed, the nature of the binning will try to be
        detected. This fails to detect log binning in some cases (e.g. a single
        bin, which defaults to linear binning), so pass `is_log=True` in such
        cases to explicitly set the nature of the binning.

    domain : length-2 sequence of numerical
        Units may be specified.

    num_bins : int
        Number of bins; specify if `domain` and either `is_lin` or `is_log` are
        specified, but redundant if `bin_edges` is specified.

    bin_names : None or sequence of strings
        Strings by which each bin can be identified. This is expected to be
        useful when one needs to easily identify bins by name where the actual
        numerical values can be non-obvious e.g. the PID dimension.
        None is also acceptable if there is no reason to name the bins.


    Notes
    -----
    Consistency is enforced for all redundant parameters passed to the
    constructor.


    Examples
    --------
    >>> from pisa import ureg
    >>> ebins = OneDimBinning(name='energy', is_log=True,
    ...                       num_bins=40, domain=[1, 80]*ureg.GeV)
    >>> print ebins
    energy: 40 logarithmically-uniform bins spanning [1.0, 80.0] GeV
    >>> ebins2 = ebins.to('joule')
    >>> print ebins2

    >>> czbins = OneDimBinning(name='coszen',
    ...                        is_lin=True, num_bins=4, domain=[-1, 0])
    >>> print czbins
    coszen: 4 equally-sized bins spanning [-1.0, 0.0]
    >>> czbins2 = OneDimBinning(name='coszen',
    ...                         bin_edges=[-1, -0.75, -0.5, -0.25, 0])
    >>> czbins == czbins2
    True

    """

    # `is_log` and `is_lin` are required for state alongsize bin_edges so that
    # a sub-sampling down to a single bin that is then resampled to > 1 bin
    # will retain the log/linear property of the original OneDimBinning.
    _hash_attrs = ('name', 'tex', 'bin_edges', 'is_log', 'is_lin', 'bin_names')

    def __init__(self, name, tex=None, bin_edges=None, units=None, domain=None,
                 num_bins=None, is_lin=None, is_log=None, bin_names=None):
        if not isinstance(name, basestring):
            raise TypeError('`name` must be basestring; got "%s".' %type(name))
        if domain is not None:
            assert isinstance(domain, Iterable)
            assert len(domain) == 2
        self._normalize_values = True
        self._name = name
        self._basename = basename(name)
        if tex is None:
            tex = r'{\rm ' + name + '}'
        self._tex = tex

        # If None, leave this and try to get units from bin_edges or domain
        # (and if nothing has units in the end, *then* make quantity have the
        # units 'dimensionless')
        if units is not None and not isinstance(units, pint.unit._Unit):
            units = ureg(units)

        # Temporarily strip units (if any were provided) to make constructing
        # bins consistent (in particular, log(x) isn't valid if x has units),
        # then reattach units after bin_edges has been defined. Default units
        # are "dimensionless".
        if isinstance(bin_edges, pint.quantity._Quantity):
            if units is not None:
                if bin_edges.dimensionality != units.dimensionality:
                    raise ValueError('All units specified must be compatible.')
                # Explicitly-passed units have precedence, so convert to those
                bin_edges.ito(units)
            units = bin_edges.units
            bin_edges = bin_edges.magnitude

        if domain is not None and \
                (isinstance(domain[0], pint.quantity._Quantity) or \
                 isinstance(domain[1], pint.quantity._Quantity)):
            if domain[0].dimensionality != domain[1].dimensionality:
                raise ValueError(
                    'Incompatible units: '
                    ' `domain` limits have units of (%s) and (%s).'
                    %(domain[0].dimensionality, domain[1].dimensionality)
                )
            # TODO: hack to test simple unit equality by converting to string
            # (probably an issue with unit registries?)
            if str(domain[0].units) != str(domain[1].units):
                logging.warn(
                    'Different (but compatible) units used to specify `domain`'
                    ' limits: (%s) and (%s).'
                    %(domain[0].units, domain[1].units)
                )
            if units is not None:
                if domain[0].dimensionality != units.dimensionality:
                    raise ValueError(
                        'Incompatible units: units passed/deduced are (%s) but'
                        ' `domain` has units of (%s).'
                        %(units.dimensionality, domain[0].dimensionality)
                    )
                if str(units) != str(domain[0].units) \
                        or str(units) != str(domain[1].units):
                    logging.warn(
                        'Different (but compatible) units deduced/passed vs.'
                        ' units used to specify `domain` limits:'
                        ' (%s) vs. (%s and %s).'
                        %(units, domain[0].units, domain[1].units)
                    )
                # Explicitly-passed AND bin_edges' units have precedence, so
                # convert to wihichever of those has been populated to `units`
                domain = [d.ito(units) for d in domain]
            else:
                units = domain[0].units
            # Strip units off of domain
            domain = np.array([d.magnitude for d in domain])

        # Now if no units have been discovered from the input args, default to
        # units of 'dimensionless'
        if units is None:
            units = ureg.dimensionless

        # If both `is_log` and `is_lin` are specified, both cannot be true
        # (but both can be False, in case of irregularly-spaced bins)
        if is_log and is_lin:
            raise ValueError('`is_log=%s` contradicts `is_lin=%s`'
                             %(is_log, is_lin))

        # If no bin edges specified, the number of bins, domain, and either
        # log or linear spacing are all required to generate bins
        if bin_edges is None:
            assert num_bins is not None and domain is not None \
                    and (is_log or is_lin), '%s, %s' %(num_bins, domain)
            if is_log:
                is_lin = False
                bin_edges = np.logspace(np.log10(domain[0]),
                                        np.log10(domain[1]),
                                        num_bins + 1)
            elif is_lin:
                is_log = False
                bin_edges = np.linspace(domain[0], domain[1], num_bins + 1)

        elif domain is not None:
            assert domain[0] == bin_edges[0] and domain[1] == bin_edges[-1]

        bin_edges = np.array(bin_edges)
        if is_lin:
            assert self.is_bin_spacing_lin(bin_edges), str(bin_edges)
            is_log = False
        elif is_log:
            assert self.is_binning_ok(bin_edges, is_log=True), str(bin_edges)
            is_lin = False
        else:
            is_lin = self.is_bin_spacing_lin(bin_edges)
            try:
                is_log = self.is_bin_spacing_log(bin_edges)
            except ValueError:
                is_log = False

        #if not is_lin and not is_log:
        #    is_log = self.is_bin_spacing_log(bin_edges)

        # (Re)attach units to bin edges
        self._bin_edges = bin_edges * units

        # (Re)define domain and attach units
        self._domain = np.array([np.min(bin_edges), np.max(bin_edges)]) * units

        # Store units for convenience
        if isinstance(units, pint.quantity._Quantity):
            units = units.units
        self._units = units

        # Derive rest of unspecified parameters from bin_edges or enforce
        # them if they were specified as arguments to init
        if num_bins is None:
            num_bins = len(self.bin_edges) - 1
        else:
            assert num_bins == len(self.bin_edges) - 1, \
                    '%s, %s' %(num_bins, self.bin_edges)
        self._num_bins = num_bins

        self._is_lin = is_lin
        self._is_log = is_log
        self._is_irregular = not (self.is_lin or self.is_log)
        self._midpoints = (self.bin_edges[:-1] + self.bin_edges[1:])/2.0
        if self.is_log:
            self._weighted_centers = np.sqrt(self.bin_edges[:-1] *
                                             self.bin_edges[1:])
        else:
            self._weighted_centers = self.midpoints

        if bin_names is not None:
            # Ok to pass a string if only one bin to be named
            if isinstance(bin_names, basestring) and num_bins == 1:
                bin_names = [bin_names]

            if not isinstance(bin_names, Iterable):
                raise ValueError('bin_names object needs to be iterable.')

            # Convert bin_names to list so it can be indexed and has len().
            if not isinstance(bin_names, Sequence):
                bin_names = list(bin_names)

            if len(set(bin_names)) != len(bin_names):
                raise ValueError(
                    'All bin names must be unique; `bin_names` passed: %s'
                    %bin_names
                )

            if len(bin_names) != num_bins:
                raise ValueError(
                    'Must pass as many names as there are bins. Got %d names'
                    ' but there are %d bins. `bin_names` passed: %s'
                    %(len(bin_names), num_bins, bin_names)
                )

            for bin_name in bin_names:
                if not isinstance(bin_name, basestring):
                    raise TypeError('Bin name %s is of type %s, but must be'
                                    ' string.' %(bin_name, type(bin_name)))

        self._bin_names = bin_names

        # TODO: define hash based upon conversion of things to base units (such
        # that a valid comparison can be made between indentical binnings but
        # that use different units). Be careful to round to just less than
        # double-precision limits after conversion so that hashes will work out
        # to be the same after conversion to the base units.

        self._hash = None
        self._edges_hash = None
        self.rehash()

    def __repr__(self):
        previous_precision = np.get_printoptions()['precision']
        np.set_printoptions(precision=18)
        try:
            argstrs = [('%s=%r' %item) for item in
                       self._serializable_state.items()]
            r = '%s(%s)' %(self.__class__.__name__, ',\n    '.join(argstrs))
        finally:
            np.set_printoptions(precision=previous_precision)
        return r

    def __str__(self):
        domain_str = (
            'spanning '
            + '[%s, %s] %s' %(self.bin_edges[0].magnitude,
                              self.bin_edges[-1].magnitude,
                              format(self.units, '~'))
        ).strip()

        edge_str = (
            'with edges at ['
            + ', '.join([str(e) for e in self.bin_edges.m])
            + '] '
            + format(self.bin_edges.u, '~')
        ).strip()

        if self.num_bins == 1:
            descr = 'one bin %s' %edge_str
            if self.is_lin:
                descr += ' (behavior is linear)'
            elif self.is_log:
                descr += ' (behavior is logarithmic)'
        elif self.is_lin:
            descr = '%d equally-sized bins %s' %(self.num_bins, domain_str)
        elif self.is_log:
            descr = '%d logarithmically-uniform bins %s' %(self.num_bins,
                                                           domain_str)
        else:
            descr = '%d irregularly-sized bins %s' %(self.num_bins, edge_str)

        if self.bin_names is not None:
            descr += (', bin_names=['
                      + ', '.join([("'%s'"%n) for n in self.bin_names])
                      + ']')

        return (self.__class__.__name__
                + "('{name:s}', {descr:s}".format(name=self.name, descr=descr)
                + ")")

    def __pretty__(self, p, cycle):
        """Method used by the `pretty` library for formatting"""
        if cycle:
            p.text('%s(...)' %self.__class__.__name__)
        else:
            p.begin_group(4, '%s(name=' %self.__class__.__name__)
            p.text(str(self))
            p.end_group(4, ')')

    def _repr_pretty_(self, p, cycle):
        """Method used by e.g. ipython/Jupyter for formatting"""
        return self.__pretty__(p, cycle)

    def __contains__(self, x):
        try:
            self.index(x)
        except ValueError:
            return False
        return True

    def index(self, x):
        """Return integer index of bin identified by `x`.

        Parameters
        ----------
        x : int, string
            If int, ensure it is a valid index and return; if string, look for
            bin with corresponding name.

        Returns
        -------
        idx: int
            index of bin corresponding to `x`

        Raises
        ------
        ValueError if `x` cannot identify a valid bin

        """
        try:
            if isinstance(x, basestring):
                assert self.bin_names is not None
                return self.bin_names.index(x)
            if isinstance(x, int):
                assert x >= 0 and x < len(self)
                return x
            raise TypeError('`x` must be either int or string; got %s instead.'
                            %type(x))
        except (AssertionError, ValueError):
            valid_range = [0, len(self)-1]
            if self.bin_names is None:
                valid_names = ''
            else:
                valid_names = ' or a valid bin name in %s' %self.bin_names
            raise ValueError("Bin corresponding to '%s' could not be located."
                             " Specify an int in %s%s."
                             %(x, valid_range, valid_names))

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
        from_json : Instantiate new OneDimBinning object from the file written
            by this method pisa.utils.jsons.to_json

        """
        jsons.to_json(self._serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, resource):
        """Instantiate a new object from the contents of a JSON file as
        formatted by the `to_json` method.

        Parameters
        ----------
        resource : str
            A PISA resource specification (see pisa.utils.resources)

        See Also
        --------
        to_json

        """
        state = jsons.from_json(resource)
        return cls(**state)

    @property
    def _serializable_state(self):
        state = OrderedDict()
        state['name'] = self.name
        state['tex'] = self.tex
        state['bin_edges'] = self.bin_edges.magnitude
        state['units'] = str(self.units)
        state['is_log'] = self.is_log
        state['is_lin'] = self.is_lin
        state['bin_names'] = self.bin_names
        return state

    @property
    def _hashable_state(self):
        state = OrderedDict()
        state['name'] = self.name
        if self.normalize_values:
            bin_edges = normQuant(self.bin_edges, sigfigs=HASH_SIGFIGS)
        else:
            bin_edges = self.bin_edges
        state['bin_edges'] = bin_edges
        state['is_log'] = self.is_log
        state['is_lin'] = self.is_lin
        state['bin_names'] = self.bin_names
        return state

    @property
    def name(self):
        return self._name

    @property
    def basename(self):
        return self._basename

    @property
    def tex(self):
        return self._tex

    @property
    def bin_edges(self):
        return self._bin_edges

    @property
    def bin_names(self):
        return self._bin_names

    @property
    def domain(self):
        return self._domain

    @property
    def units(self):
        return self._units

    @property
    def num_bins(self):
        return self._num_bins

    @property
    def is_lin(self):
        return self._is_lin

    @property
    def is_log(self):
        return self._is_log

    @property
    def is_irregular(self):
        return self._is_irregular

    @property
    def midpoints(self):
        return self._midpoints

    @property
    def weighted_centers(self):
        return self._weighted_centers

    @property
    def hash(self):
        """Hash value based upon less-than-double-precision-rounded
        numerical values and any other state (includes name, tex, is_log, and
        is_lin attributes). Rounding is done to `HASH_SIGFIGS` significant
        figures.

        Set this class attribute to None to keep full numerical precision in
        the values hashed (but be aware that this can cause equal things
        defined using different unit orders-of-magnitude to hash differently).

        """
        if self._hash is None:
            s = self._hashable_state
            self._hash = hash_obj(s)
        return self._hash

    @property
    def normalize_values(self):
        """Normalize quantities' units prior to hashing"""
        return self._normalize_values

    @normalize_values.setter
    def normalize_values(self, b):
        assert isinstance(b, bool)
        if b == self._normalize_values:
            return
        self._normalize_values = b
        # Invalidate the hash, since the hasing behavior has changed
        self._hash = None
        self._edges_hash = None

    @property
    def edges_hash(self):
        """Hash value based *solely* upon bin edges' values.

        The hash value is obtained on the edges after "normalizing" their
        values if `self.normalize_values` is True; see
        `pisa.utils.comparsions.normQuant` for details of the normalization
        process.

        """
        if self._edges_hash is None:
            if self.normalize_values:
                bin_edges = normQuant(self.bin_edges, sigfigs=HASH_SIGFIGS)
            else:
                bin_edges = self.bin_edges
            self._edges_hash = hash_obj(bin_edges)
        return self._edges_hash

    def rehash(self):
        self._hash = None
        self._edges_hash = None
        _ = self.hash
        _ = self.edges_hash

    def __hash__(self):
        return self.hash

    @property
    def label(self):
        units = format(self.units, '~').strip()
        if units == '':
            return self.tex
        return self.tex + r' \; (%s)'%units

    @property
    def bin_widths(self):
        return np.abs(np.diff(self.bin_edges))

    @property
    def inbounds_criteria(self):
        """Return string boolean criteria indicating e.g. an event falls within
        the limits of the defined binning.

        This can be used for e.g. applying cuts to events.

        See Also
        --------
        pisa.core.events.keepEventsInBins

        """
        be = self.bin_edges
        crit = '((%s >= %.14e) & (%s <= %.14e))' %(self.name,
                                                   min(be.magnitude),
                                                   self.name,
                                                   max(be.magnitude))
        return crit

    def __len__(self):
        """Number of bins (*not* number of bin edges)."""
        return self.num_bins

    def __mul__(self, other):
        if isinstance(other, OneDimBinning):
            return MultiDimBinning([self, other])
        elif isinstance(other, MultiDimBinning):
            return MultiDimBinning([self] + [d for d in other])
        return OneDimBinning(name=self.name, tex=self.tex,
                             bin_edges=self.bin_edges * other)

    # TODO: if same or contained dimension, modify the current binning OR
    # create a smarter MultiDimBinning object that allows for multiple
    # disconnected binning regions with arbitrary binning within each
    # region
    def __add__(self, other):
        if isinstance(other, OneDimBinning):
            return MultiDimBinning([self, other])
        elif isinstance(other, MultiDimBinning):
            return MultiDimBinning([self] + [d for d in other])

        if isbarenumeric(other):
            other = other * ureg.dimensionless
        if isinstance(other, pint.quantity._Quantity):
            new_bin_edges = self.bin_edges + other
            return OneDimBinning(name=self.name, tex=self.tex,
                                 bin_edges=new_bin_edges)
        else:
            raise TypeError('Unhandled type %s for __add__' %type(other))

    @_new_obj
    def __deepcopy__(self, memo):
        """Explicit deepcopy constructor"""
        return {}

    @staticmethod
    def is_bin_spacing_log(bin_edges):
        """Check if `bin_edges` define a logarithmically-uniform bin spacing.

        Parameters
        ----------
        bin_edges : sequence
            Fewer than 2 `bin_edges` - raises ValueError
            Two `bin_edges` - returns False as a reasonable guess (spacing is
                assumed to be linear)
            More than two `bin_edges` - whether spacing is linear is computed

        Returns
        -------
        bool

        """
        bin_edges = np.array(bin_edges)
        if len(bin_edges) < 3:
            raise ValueError('%d bin edge(s) passed; require at least 3 to'
                             ' determine nature of bin spacing.'
                             %len(bin_edges))
        with np.errstate(divide='ignore', invalid='ignore'):
            log_spacing = bin_edges[1:] / bin_edges[:-1]
        if np.allclose(log_spacing, log_spacing[0]):
            return True
        return False

    @staticmethod
    def is_bin_spacing_lin(bin_edges):
        """Check if `bin_edges` define a linearly-uniform bin spacing.

        Parameters
        ----------
        bin_edges : sequence
            Fewer than 2 `bin_edges` - raises ValueError
            Two `bin_edges` - returns True as a reasonable guess
            More than two `bin_edges` - whether spacing is linear is computed

        Returns
        -------
        bool

        Raises
        ------
        ValueError if fewer than 2 `bin_edges` are specified.

        """
        bin_edges = np.array(bin_edges)
        if len(bin_edges) == 1:
            raise ValueError('Single bin edge passed; require at least 2 to'
                             ' determine nature of bin spacing.')
        if not np.all(np.isfinite(bin_edges)):
            return False
        # Default is to assume linear behavior if only two bin edges are given
        if len(bin_edges) == 2:
            return True
        lin_spacing = np.diff(bin_edges)
        if np.allclose(lin_spacing, lin_spacing[0]):
            return True
        return False

    @staticmethod
    def is_binning_ok(bin_edges, is_log):
        """Check monotonicity and that bin spacing is logarithmically uniform
        (if `is_log=True`)

        Parameters
        ----------
        bin_edges : sequence
            Bin edges to check the validity of

        is_log : bool
            Whether binning is expected to be logarithmically uniform.

        Returns
        -------
        bool, True if binning is OK, False if not

        """
        # Must be at least two edges to define a single bin
        if len(bin_edges) < 2:
            return False
        # Bin edges must be monotonic and strictly increasing
        lin_spacing = np.diff(bin_edges)
        if not np.all(lin_spacing > 0):
            return False
        # Log binning must have equal widths in log-space (but a single bin
        # has no "spacing" or stride, so no need to check)
        if is_log and len(bin_edges) > 2:
            return OneDimBinning.is_bin_spacing_log(bin_edges)
        return True

    # TODO: refine compatibility test to handle compatible units; as of now,
    # both upsampling and downsampling are allowed. Is this reasonable
    # behavior?
    def is_compat(self, other):
        """Compatibility -- for now -- is defined by all of self's bin
        edges form a subset of other's bin edges, or vice versa, and the units
        match. This might bear revisiting, or redefining just for special
        circumstances.

        Parameters
        ----------
        other : OneDimBinning

        Returns
        -------
        bool

        """
        if self.units.dimensionality != other.units.dimensionality:
            return False

        if self.normalize_values:
            my_normed_bin_edges = set(normQuant(self.bin_edges))
            other_normed_bin_edges = set(normQuant(other.bin_edges))
        else:
            my_normed_bin_edges = set(self.bin_edges)
            other_normed_bin_edges = set(other.bin_edges)

        if len(my_normed_bin_edges.difference(other_normed_bin_edges)) == 0:
            return True
        if len(other_normed_bin_edges.difference(my_normed_bin_edges)) == 0:
            return True
        return False

    @_new_obj
    def oversample(self, factor):
        """Return a OneDimBinning object oversampled relative to this object's
        binning.

        Parameters
        ----------
        factor : integer
            Factor by which to oversample the binning, with `factor`-times
            as many bins (*not* bin edges) as this object has.

        Returns
        -------
        OneDimBinning object
        """
        if factor < 1 or factor != int(factor):
            raise ValueError('`factor` must be integer >= 0; got %s' %factor)

        factor = int(factor)

        if factor == 1:
            return self

        if self.is_log:
            bin_edges = np.logspace(np.log10(self.domain[0].m),
                                    np.log10(self.domain[-1].m),
                                    self.num_bins * factor + 1)
        elif self.is_lin:
            bin_edges = np.linspace(self.domain[0].m, self.domain[-1].m,
                                    self.num_bins * factor + 1)
        else: # irregularly-spaced
            bin_edges = []
            for lower, upper in izip(self.bin_edges[:-1].m,
                                     self.bin_edges[1:].m):
                this_bin_new_edges = np.linspace(lower, upper, factor+1)
                # Exclude the last edge, as this will be first edge for the
                # next divided bin
                bin_edges.extend(this_bin_new_edges[:-1])
            # Final bin needs final edge
            bin_edges.append(this_bin_new_edges[-1])

        return {'bin_edges': np.array(bin_edges)*self.units,
                'bin_names': None}

    @_new_obj
    def downsample(self, factor):
        assert int(factor) == float(factor)
        factor = int(factor)
        if factor == 1:
            return self
        assert factor > 0 and factor <= self.num_bins
        assert self.num_bins % factor == 0
        return {'bin_edges': self.bin_edges[::factor],
                'bin_names': None}

    def ito(self, units):
        """Convert units in-place. Cf. Pint's `ito` method."""
        if units is None:
            units = ''
        for attr in ['bin_edges', 'domain', 'midpoints', 'weighted_centers']:
            getattr(self, attr).ito(units)

    @_new_obj
    def to(self, units):
        if units is None:
            units = 'dimensionless'
        return {'bin_edges': self.bin_edges.to(ureg(str(units)))}

    def __getattr__(self, attr):
        return super(self.__class__, self).__getattribute__(attr)

    # TODO: make this actually grab the bins specified (and be able to grab
    # disparate bins, whether or not they are adjacent)... i.e., fill in all
    # upper bin edges, and handle the case that it goes from linear or log
    # to uneven (or if it stays lin or log, keep that attribute for the
    # subselection). Granted, a OneDimBinning object right now requires
    # monotonically-increasing and adjacent bins.

    # TODO: make indexing allow for sequence containing a single ellipsis
    @_new_obj
    def __getitem__(self, index):
        """Return a new OneDimBinning, sub-selected by `index`.

        Parameters
        ----------
        index : int, slice, ellipsis, str, or length-one Sequence
            The *bin indices* (not bin-edge indices) to return. Generated
            OneDimBinning object must obey the usual rules (monotonic, etc.).
            If a str is supplied it must match a name in bin_names

        Returns
        -------
        A new OneDimBinning but only with bins selected by `index`.

        """
        # Ellipsis: binninng[...] returns everything
        if index is Ellipsis:
            return {}

        magnitude = self.bin_edges.magnitude
        units = self.bin_edges.units
        orig_index = index

        # Deal with indexing by name first so as to not break anything else
        if isinstance(index, basestring):
            assert self.bin_names is not None
            index = self.bin_names.index(index)

        # Simple to get all but final bin edge
        bin_edges = magnitude[index].tolist()

        if np.isscalar(bin_edges):
            bin_edges = [bin_edges]
        else:
            bin_edges = list(bin_edges)

        # Convert index/indices to positive-number sequence
        if isinstance(index, slice):
            index = range(*index.indices(len(self)))
        if isinstance(index, int):
            index = [index]

        if isinstance(index, Iterable):
            if not isinstance(index, Sequence):
                index = list(index)
            for bin_index in index:
                if isinstance(bin_index, str):
                    raise ValueError('Slicing by seq of names currently not'
                                     ' supported')
            if len(index) == 0:
                raise ValueError('`index` "%s" results in no bins being'
                                 ' specified.' %orig_index)
            if len(index) > 1 and not np.all(np.diff(index) == 1):
                raise ValueError('Bin indices must be monotonically'
                                 ' increasing and adjacent.')
            new_edges = set()
            new_names = []
            for bin_index in index:
                if bin_index < -len(self) or bin_index >= len(self):
                    raise ValueError(
                        "Dimension '%s': bin index %s is invalid. Bin index"
                        " must be >= %+d and <= %+d"
                        %(self.name, bin_index, -len(self), len(self)-1)
                    )
                edge_ind0 = bin_index % len(self)
                edge_ind1 = edge_ind0 + 1
                if self.bin_names is not None:
                    new_names.append(self.bin_names[edge_ind0])
                new_edges = new_edges.union((self.bin_edges[edge_ind0].m,
                                             self.bin_edges[edge_ind1].m))
        else:
            raise TypeError('Unhandled index type %s' %type(orig_index))

        if new_names == []:
            new_names = None
        # Retrieve current state; only bin_edges and bin_names need to be
        # updated
        return {'bin_edges': np.array(sorted(new_edges))*units,
                'bin_names': new_names}

    def __iter__(self):
        indices = range(len(self))
        for idx in indices:
            yield self[idx]

    def __eq__(self, other):
        if not isinstance(other, OneDimBinning):
            return False
        return recursiveEquality(self._hashable_state, other._hashable_state)

    def __ne__(self, other):
        return not self.__eq__(other)

# TODO: make this able to be loaded from a pickle!!!
class MultiDimBinning(object):
    """
    Multi-dimensional binning object. This can contain one or more
    OneDimBinning objects, and all subsequent operations (e.g. slicing) will
    act on these in the order they are supplied.

    Parameters
    ----------
    dimensions : OneDimBinning or sequence convertible thereto
        Dimensions for the binning object. Indexing into the MultiDimBinning
        object follows the order in which dimensions are provided.

    See Also
    --------
    OneDimBinning : each item that is not a OneDimBinning object is passed to
        this class to be instantiated as such.

    """
    def __init__(self, dimensions):
        if not isinstance(dimensions, (Sequence, Iterable)):
            if isinstance(dimensions, Mapping):
                assert len(dimensions) == 1 and 'dimensions' in dimensions
                dimensions = dimensions['dimensions']
            dimensions = [dimensions]
        tmp_dimensions = []
        for obj_num, obj in enumerate(dimensions):
            if isinstance(obj, OneDimBinning):
                one_dim_binning = obj
            elif isinstance(obj, Mapping):
                one_dim_binning = OneDimBinning(**obj)
            else:
                raise TypeError('Argument/object #%d unhandled type: %s'
                                %(obj_num, type(obj)))
            tmp_dimensions.append(one_dim_binning)
        self._dimensions = tmp_dimensions
        self._compute_metadata()

    def _compute_metadata(self):
        self._shape = tuple([b.num_bins for b in self._dimensions])
        self._num_dims = len(self._dimensions)

    def __repr__(self):
        previous_precision = np.get_printoptions()['precision']
        np.set_printoptions(precision=18)
        try:
            argstrs = [('%s=%r' %item) for item in
                       self._serializable_state.items()]
            r = '%s(%s)' %(self.__class__.__name__, ',\n    '.join(argstrs))
        finally:
            np.set_printoptions(precision=previous_precision)
        return r

    def __str__(self):
        return (self.__class__.__name__ + '(\n    '
                + ',\n    '.join([str(dim) for dim in self._dimensions])
                + '\n)')

    def __pretty__(self, p, cycle):
        """Method used by the `pretty` library for formatting"""
        if cycle:
            p.text('%s(...)' %self.__class__.__name__)
        else:
            p.begin_group(4, '%s([' %self.__class__.__name__)
            for n, dim in enumerate(self):
                p.breakable()
                p.pretty(dim)
                if n < len(self)-1:
                    p.text(',')
            p.end_group(4, '])')

    def _repr_pretty_(self, p, cycle):
        """Method used by e.g. ipython/Jupyter for formatting"""
        return self.__pretty__(p, cycle)

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
        from_json
            Instantiate new object from the file written by this method
        pisa.utils.jsons.to_json

        """
        jsons.to_json(self._serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, resource):
        """Instantiate a new MultiDimBinning object from a JSON file.

        The format of the JSON is generated by the `MultiDimBinning.to_json`
        method, which converts a MultiDimBinning object to basic types and
        numpy arrays are converted in a call to `pisa.utils.jsons.to_json`.

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
        return cls(**state)

    @property
    def names(self):
        return [dim.name for dim in self]

    @property
    def basenames(self):
        """List of binning names with prefixes and/or suffixes along with any
        number of possible separator characters removed. See function
        `basename` for detailed specifications."""
        return [b.basename for b in self]

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def dims(self):
        return self._dimensions

    @property
    def num_dims(self):
        return self._num_dims

    @property
    def shape(self):
        return self._shape

    @property
    def _serializable_state(self):
        """Everything necessary to fully describe this object's state. Note
        that objects may be returned by reference, so to prevent external
        modification, the user must call deepcopy() separately on the returned
        dict.

        Returns
        -------
        state dict; can be passed to instantiate a new MultiDimBinning via
            MultiDimBinning(**state)

        """
        return OrderedDict({'dimensions': [d._serializable_state
                                           for d in self]})

    @property
    def _hashable_state(self):
        """Everything necessary to fully describe this object's state. Note
        that objects may be returned by reference, so to prevent external
        modification, the user must call deepcopy() separately on the returned
        OrderedDict.

        Returns
        -------
        state : OrderedDict that can be passed to instantiate a new
            MultiDimBinning via MultiDimBinning(**state)

        """
        state = OrderedDict()
        state['dimensions'] = [self[name]._hashable_state
                               for name in sorted(self.names)]
        return state

    @property
    def hash(self):
        """Unique hash value for this object"""
        return hash_obj(self._hashable_state)

    def __hash__(self):
        return self.hash

    @property
    def edges_hash(self):
        return hash_obj([d.edges_hash for d in self])

    @property
    def bin_edges(self):
        """Return a list of the contained dimensions' bin_edges that is
        compatible with the numpy.histogramdd `hist` argument.

        """
        return [d.bin_edges for d in self]

    @property
    def domains(self):
        """Return a list of the contained dimensions' domains"""
        return [d.domain for d in self]

    @property
    def midpoints(self):
        """Return a list of the contained dimensions' midpoints"""
        return [d.midpoints for d in self]

    @property
    def weighted_centers(self):
        """Return a list of the contained dimensions' weighted_centers (e.g.
        equidistant from bin edges on logarithmic scale, if the binning is
        logarithmic; otherwise linear). Access `midpoints` attribute for
        always-linear alternative."""
        return [d.weighted_centers for d in self]

    @property
    def num_bins(self):
        """Return a list of the contained dimensions' num_bins."""
        return [d.num_bins for d in self]

    @property
    def tot_num_bins(self):
        """Return total number of bins."""
        return np.product(self.num_bins)

    @property
    def units(self):
        """Return a list of the contained dimensions' units"""
        return [d.units for d in self]

    @property
    def inbounds_criteria(self):
        """Return string boolean criteria indicating e.g. an event falls within
        the limits of the defined binning.

        This can be used for e.g. applying cuts to events.

        See Also
        --------
        pisa.core.events.keepEventsInBins

        """
        crit = '(%s)' %(' & '.join([dim.inbounds_criteria for dim in self]))
        return crit

    def index(self, dim, use_basenames=False):
        """Find dimension implied by `dim` and return its integer index.

        Parameters
        ----------
        dim : int, string, OneDimBinning
            An integer index, dimesion name, or identical OneDimBinning object
            to locate within the contained dimensions

        use_basenames : bool
            Dimension names are only compared after pre/suffixes are stripped,
            allowing for e.g. `dim`='true_energy' to find 'reco_energy'.

        Returns
        -------
        idx : integer
            index of the dimension corresponding to `dim`

        Raises
        ------
        ValueError if `dim` cannot be found

        """
        names = self.basenames if use_basenames else self.names
        if isinstance(dim, OneDimBinning):
            d = dim.basename if use_basenames else dim.name
            try:
                idx = names.index(d)
            except ValueError:
                what = 'index'
                raise ValueError(
                    'Dimension %s not present. Valid dimensions are in range %s'
                    %(d, [0, len(self)-1])
                )
        elif isinstance(dim, basestring):
            d = basename(dim) if use_basenames else dim
            try:
                idx = names.index(d)
            except ValueError:
                what = 'basename' if use_basenames else 'name'
                raise ValueError(
                    "Dimension %s '%s' not present. Valid dimension %ss are %s"
                    %(what, d, what, names)
                )
        elif isinstance(dim, int):
            if dim < 0 or dim >= len(self):
                raise ValueError(
                    'Dimension %d does not exist. Valid dimensions indices'
                    ' are in the range %s.' %(dim, [0, len(self)-1])
                )
            idx = dim
        else:
            raise TypeError('Unhandled type for `dim`: "%s"' %type(dim))
        return idx

    def remove(self, dims):
        """Remove dimensions.

        Parameters
        ----------
        dims : str, int, or sequence thereof
            Dimensions to be removed

        Returns
        -------
        binning : MultiDimBinning
            Identical binning as this but with `dims` removed.

        """
        if isinstance(dims, (basestring, int)):
            dims = [dims]

        keep_idx = range(len(self))
        for dim in dims:
            idx = self.index(dim)
            keep_idx.remove(idx)

        keep_dims = [deepcopy(self.dimensions[idx]) for idx in keep_idx]
        return MultiDimBinning(keep_dims)

    # TODO: examples!
    def reorder_dimensions(self, order, use_deepcopy=False):
        """Return a new MultiDimBinning object with dimensions ordered
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
        MultiDimBinning object with reordred dimensions.

        Raises
        ------
        ValueError if dimensions present in this object are missing from
        `order`.

        Examples
        --------
        >>> b0 = MultiDimBinning(...)
        >>> b1 = MultiDimBinning(...)
        >>> b2 = b0.reorder_dimensions(b1)
        >>> print b2.binning.names

        """
        if hasattr(order, 'binning') and isinstance(order.binning,
                                                    MultiDimBinning):
            order = order.binning.dims
        elif isinstance(order, MultiDimBinning):
            order = order.dims

        indices = []
        for dim in order:
            try:
                idx = self.index(dim, use_basenames=True)
            except ValueError:
                continue
            indices.append(idx)
        if set(indices) != set(range(len(self))):
            raise ValueError(
                'Invalid `order`: Only a subset of the dimensions present'
                ' were specified. `order`=%s; dimensions=%s' %(order, self)
            )
        if use_deepcopy:
            new_dimensions = [deepcopy(self._dimensions[n]) for n in indices]
        else:
            new_dimensions = [self._dimensions[n] for n in indices]
        new_binning = MultiDimBinning(new_dimensions)
        return new_binning

    def oversample(self, *args, **kwargs):
        """Return a Binning object oversampled relative to this binning.

        Parameters
        ----------
        *args : each factor an int
            Factors by which to oversample the binnings. There must either be
            one factor (one arg)--which will be broadcast to all dimensions--or
            there must be as many factors (args) as there are dimensions.
            If positional args are specified (i.e., non-kwargs), then kwargs
            are forbidden.

        **kwargs : name=factor pairs

        Notes
        -----
        Can either specify oversmapling by passing in args (ordered values, no
        keywords) or kwargs (order doesn't matter, but uses keywords), but not
        both.

        """
        factors = self._args_kwargs_to_list(*args, **kwargs)
        new_binning = [dim.oversample(f)
                       for dim, f in izip(self._dimensions, factors)]
        return MultiDimBinning(new_binning)

    def downsample(self, *args, **kwargs):
        """Return a Binning object downsampled relative to this binning.

        Parameters
        ----------
        *args : each factor an int
            Factors by which to downsample the binnings. There must either be
            one factor (one arg)--which will be broadcast to all dimensions--or
            there must be as many factors (args) as there are dimensions.
            If positional args are specified (i.e., non-kwargs), then kwargs
            are forbidden.

        **kwargs : name=factor pairs

        Notes
        -----
        Can either specify downsampling by passing in args (ordered values, no
        keywords) or kwargs (order doesn't matter, but uses keywords), but not
        both.

        """
        factors = self._args_kwargs_to_list(*args, **kwargs)
        new_binning = [dim.downsample(f)
                       for dim, f in izip(self._dimensions, factors)]
        return MultiDimBinning(new_binning)

    def assert_array_fits(self, array):
        """Check if a 2D array of values fits into the defined bins (i.e., has
        the exact shape defined by this binning).

        Parameters
        ----------
        array : 2D array (or sequence-of-sequences)

        Returns
        -------
        fits : bool, True if array fits or False otherwise

        """
        assert array.shape == self.shape, \
                '%s does not match self shape %s' %(array.shape, self.shape)

    def assert_compat(self, other):
        """Check if a (possibly different) binning can map onto the defined
        binning. Allows for simple re-binning schemes (but no interpolation).

        Parameters
        ----------
        other : Binning or container with attribute "binning"

        Returns
        -------
        compat : bool

        """
        if not isinstance(other, MultiDimBinning):
            for val in other.__dict__.values():
                if isinstance(val, MultiDimBinning):
                    other = val
                    break
        assert isinstance(other, MultiDimBinning), str(type(other))
        if other == self:
            return True
        for my_dim, other_dim in zip(self, other):
            if not my_dim.assert_compat(other_dim):
                return False
        return True

    def squeeze(self):
        """Remove any singleton dimensions (i.e. that have only a single bin).
        Analagous to `numpy.squeeze`.

        Returns
        -------
        MultiDimBinning with only non-singleton dimensions

        """
        return MultiDimBinning([d for d in self if len(d) > 1])

    def _args_kwargs_to_list(self, *args, **kwargs):
        """Take either args or kwargs (but not both) and convert into a simple
        sequence of values. Broadcasts a single arg to all dimensions."""
        if not np.logical_xor(len(args), len(kwargs)):
            raise ValueError('Either args (values specified by order and not'
                             ' specified by name) or kwargs (values specified'
                             ' by name=value pairs) can be used, but not'
                             ' both.')

        if len(args) == 1:
            return [args[0]]*self.num_dims

        if len(args) > 1:
            if len(args) != self.num_dims:
                raise ValueError('Specified %s args, but binning is'
                                 ' %s-dim.' %(len(args), self.num_dims))
            return args

        if set(kwargs.keys()) != set(self.names):
            raise ValueError('Specified dimensions "%s" but this has'
                             ' dimensions "%s"' %(sorted(kwargs.keys()),
                                                  self.names))
        return [kwargs[name] for name in self.names]

    def ito(self, *args, **kwargs):
        """Convert units in-place. Cf. Pint's `ito` method."""
        units_list = self._args_kwargs_to_list(*args, **kwargs)
        [dim.ito(units) for dim, units in izip(self._dimensions, units_list)]

    def to(self, *args, **kwargs):
        """Convert the contained dimensions to the passed units. Unspecified
        dimensions will be omitted.

        """
        units_list = self._args_kwargs_to_list(*args, **kwargs)
        new_binnings = [dim.to(units)
                        for dim, units in izip(self._dimensions, units_list)]
        return MultiDimBinning(new_binnings)

    # TODO: magnitude method that replicates Pint version's behavior (???)
    #def magnitude(self):

    def meshgrid(self, entity, attach_units=True):
        """Apply NumPy's meshgrid method on various entities of interest.

        Parameters
        ----------
        entity : string
            One of 'midpoints', 'weighted_centers', 'bin_edges', or
            'bin_widths'.

        attach_units : bool
            Whether to attach units to the result (can save computation time by
            not doing so).

        Returns
        -------
        numpy ndarray or Pint quantity of the same

        See Also
        --------
        numpy.meshgrid

        """
        entity = entity.lower().strip()
        if entity == 'midpoints':
            mg = np.meshgrid(*[d.midpoints for d in self._dimensions],
                             indexing='ij')
        elif entity == 'weighted_centers':
            mg = np.meshgrid(*[d.weighted_centers for d in self._dimensions],
                             indexing='ij')
        elif entity == 'bin_edges':
            mg = np.meshgrid(*[d.bin_edges for d in self._dimensions],
                             indexing='ij')
        elif entity == 'bin_widths':
            mg = np.meshgrid(*[d.bin_widths for d in self._dimensions],
                             indexing='ij')
        else:
            raise ValueError('Unrecognized `entity`: "%s"' %entity)

        if attach_units:
            return [m*dim.units for m, dim in izip(mg, self._dimensions)]
        return mg

    # TODO: modify technique depending upon grid size for memory concerns, or
    # even take a `method` argument to force method manually.
    def bin_volumes(self, attach_units=True):
        meshgrid = self.meshgrid(entity='bin_widths', attach_units=False)
        volumes = reduce(lambda x, y: x*y, meshgrid)
        if attach_units:
            return (
                volumes
                * reduce(lambda x, y: x*y, [ureg(str(d.units)) for d in
                                            self._dimensions])
            )

        return volumes

    def empty(self, **kwargs):
        """Return an "empty" numpy ndarray with same dimensions as this
        binning.

        The contents are not _actually_ empty, just undefined. Therefore be
        careful to populate the array prior to using its contents.

        Parameters
        ----------
        **kwargs : passed to numpy.empty()

        """
        np.empty(self.shape, **kwargs)

    def __contains__(self, x):
        if isinstance(x, OneDimBinning):
            return x in self.dims
        if isinstance(x, basestring):
            return x in self.names
        return False

    def __eq__(self, other):
        if not isinstance(other, MultiDimBinning):
            return False
        return recursiveEquality(self._hashable_state, other._hashable_state)

    def __add__(self, other):
        other = MultiDimBinning(other)
        return MultiDimBinning([d for d in self] + [d for d in other])

    def __mul__(self, other):
        other = MultiDimBinning(other)
        return MultiDimBinning([d for d in self] + [d for d in other])

    # TODO: should __getattr__ raise its own exception if the attr is not found
    # as a dimension rather than call parent's __getattribute__ method, since
    # presumably that already failed?
    def __getattr__(self, attr):
        # If youve gotten here, __getattribute__ has failed. Try to get the
        # attr as a contained dimension:
        try:
            return self.__getitem__(attr)
        except (KeyError, ValueError):
            # If that failed, re-run parent's __getattribute__ which will raise
            # an appropriate exception
            return super(self.__class__, self).__getattribute__(attr)

    # TODO: refine handling of ellipsis such that the following work as in
    # Numpy:
    #       * ['dim0', 'dim3', ...]
    #       * ['dim0', 3, ...]
    #       * [...]
    #       * [0, ...]
    #       * [..., 2]
    #       * [..., 2, 1, 4]
    def __getitem__(self, index):
        """Interpret indices as indexing bins and *not* bin edges.
        Indices refer to dimensions in same order they were specified at
        instantiation, and all dimensions must be present.

        Parameters
        ----------
        index : str, int, len-N-sequence of ints, or len-N-sequence of slices
            If str is passed: Return the binning corresponding to the name
            If an integer is passed:
              * If num_dims is 1, `index` indexes into the bins of the sole
                OneDimBinning. The bin is returned.
              * If num_dims > 1, `index` indexes which contained OneDimBinning
                object to return.
            If a len-N-sequence of integers or slices is passed, dimensions are
            indexed by these in the order in which dimensions are stored
            internally.

        Returns
        -------
        A MultiDimBinning object new Binning object but with the bins specified
        by `index`. Whether or not behavior is logarithmic is unchanged.

        """
        if index is Ellipsis:
            return self

        if isinstance(index, basestring):
            for d in self._dimensions:
                if d.name == index:
                    return d

        # TODO: implement a "linearization" like np.flatten() to iterate
        # through each bin individually without hassle for the user...
        #if self.num_dims == 1 and np.isscalar(index):
        #    return self._dimensions[0]

        if isinstance(index, Iterable) and not isinstance(index, Sequence):
            index = list(index)

        if not isinstance(index, Sequence):
            index = [index]

        input_dim = len(index)
        if input_dim != self.num_dims:
            raise ValueError('Binning is %dD, but %dD indexing was passed'
                             %(self.num_dims, input_dim))

        new_binning = {'dimensions': [dim[idx] for dim, idx in
                                      zip(self._dimensions, index)]}

        return MultiDimBinning(**new_binning)

    def __iter__(self):
        return iter(self._dimensions)

    def __len__(self):
        return self.num_dims

    def __ne__(self, other):
        return not self.__eq__(other)


def test_OneDimBinning():
    import os
    import pickle
    import shutil
    import tempfile

    b1 = OneDimBinning(name='energy', num_bins=40, is_log=True,
                       domain=[1, 80]*ureg.GeV,
                       bin_names=[str(i) for i in range(40)])
    b2 = OneDimBinning(name='coszen', num_bins=40, is_lin=True,
                       domain=[-1, 1], bin_names=None)
    logging.debug('len(b1): %s' %len(b1))
    logging.debug('b1: %s' %b1)
    logging.debug('b2: %s' %b2)
    logging.debug('b1.oversample(10): %s' %b1.oversample(10))
    logging.debug('b1.oversample(1): %s' %b1.oversample(1))
    # Slicing
    logging.debug('b1[1:5]: %s' %b1[1:5])
    logging.debug('b1[:]: %s' %b1[:])
    logging.debug('b1[-1]: %s' %b1[-1])
    logging.debug('b1[:-1]: %s' %b1[:-1])
    logging.debug('copy(b1): %s' %copy(b1))
    logging.debug('deepcopy(b1): %s' %deepcopy(b1))
    # Indexing by Ellipsis
    assert b1[...] == b1
    # TODO: make pickle great again
    #pickle.dumps(b1, pickle.HIGHEST_PROTOCOL)
    try:
        b1[-1:-3]
    except ValueError:
        pass
    else:
        assert False

    b3 = OneDimBinning(name='distance', num_bins=10, is_log=True,
                       domain=[0.1, 10]*ureg.m)
    b4 = OneDimBinning(name='distance', num_bins=10, is_log=True,
                       domain=[1e5, 1e7]*ureg.um)
    b3.normalize_values = True
    b4.normalize_values = True

    # Without rounding, converting bin edges to base units yields different
    # results due to finite precision effects
    assert np.any(normQuant(b3.bin_edges, sigfigs=None)
                  != normQuant(b4.bin_edges, sigfigs=None))

    # Normalize function should take care of this
    assert np.all(normQuant(b3.bin_edges, sigfigs=HASH_SIGFIGS)
                  == normQuant(b4.bin_edges, sigfigs=HASH_SIGFIGS)), \
            'normQuant(b3.bin_edges)=\n%s\nnormQuant(b4.bin_edges)=\n%s' \
            %(normQuant(b3.bin_edges, HASH_SIGFIGS),
              normQuant(b4.bin_edges, HASH_SIGFIGS))

    # And the hashes should be equal, reflecting the latter result
    assert b3.hash == b4.hash, \
            '\nb3=%s\nb4=%s' %(b3._hashable_state, b4._hashable_state)
    assert b3.hash == b4.hash, 'b3.hash=%s; b4.hash=%s' %(b3.hash, b4.hash)

    # TODO: make pickle great again
    #s = pickle.dumps(b3, pickle.HIGHEST_PROTOCOL)
    #b3_loaded = pickle.loads(s)
    #assert b3_loaded == b3

    testdir = tempfile.mkdtemp()
    try:
        for b in [b1, b2, b3, b4]:
            b_file = os.path.join(testdir, 'one_dim_binning.json')
            b.to_json(b_file, warn=False)
            b_ = OneDimBinning.from_json(b_file)
            assert b_ == b, 'b=\n%s\nb_=\n%s' %(b, b_)
            jsons.to_json(b, b_file, warn=False)
            b_ = OneDimBinning.from_json(b_file)
            assert b_ == b, 'b=\n%s\nb_=\n%s' %(b, b_)
    except:
        print 'b that failed:', b
        raise
    finally:
        shutil.rmtree(testdir, ignore_errors=True)

    logging.info('<< PASSED >> test_OneDimBinning')


def test_MultiDimBinning():
    import os
    import pickle
    import shutil
    import tempfile

    b1 = OneDimBinning(name='energy', num_bins=40, is_log=True,
                       domain=[1, 80]*ureg.GeV)
    b2 = OneDimBinning(name='coszen', num_bins=40, is_lin=True,
                       domain=[-1, 1])
    mdb = MultiDimBinning([b1, b2])
    b00 = mdb[0, 0]
    x0 = mdb[0:, 0]
    x1 = mdb[0:, 0:]
    x2 = mdb[0, 0:]
    x3 = mdb[-1, -1]
    logging.debug(str(mdb.energy))
    logging.debug('copy(mdb): %s' %copy(mdb))
    logging.debug('deepcopy(mdb): %s' %deepcopy(mdb))
    assert deepcopy(mdb) == mdb
    #s = pickle.dumps(mdb, pickle.HIGHEST_PROTOCOL)
    # TODO: add these back in when we get pickle loading working!
    #mdb2 = pickle.loads(s)
    #assert mdb2 == mdb1

    binning = MultiDimBinning([
        dict(name='energy', is_log=True, domain=[1, 80]*ureg.GeV, num_bins=40),
        dict(name='coszen', is_lin=True, domain=[-1, 0], num_bins=20)
    ])

    assert binning.num_bins == [40, 20]
    assert binning.tot_num_bins == 40 * 20

    assert binning.oversample(10).shape == (400, 200)

    assert binning.oversample(10, 1).shape == (400, 20)
    assert binning.oversample(1, 3).shape == (40, 60)

    assert binning.oversample(coszen=10, energy=2).shape == (80, 200)
    assert binning.oversample(1, 1) == binning

    assert binning.to('MeV', '') == binning, 'converted=%s\norig=%s' \
            %(binning.to('MeV', ''), binning)
    assert binning.to('MeV', '').hash == binning.hash

    mg = binning.meshgrid(entity='bin_edges')
    mg = binning.meshgrid(entity='weighted_centers')
    mg = binning.meshgrid(entity='midpoints')
    bv = binning.bin_volumes(attach_units=False)
    bv = binning.bin_volumes(attach_units=True)
    binning.to('MeV', None)
    binning.to('MeV', '')
    binning.to(ureg.joule, '')

    testdir = tempfile.mkdtemp()
    try:
        b_file = os.path.join(testdir, 'multi_dim_binning.json')
        binning.to_json(b_file, warn=False)
        b_ = MultiDimBinning.from_json(b_file)
        assert b_ == binning, 'binning=\n%s\nb_=\n%s' %(binning, b_)
        jsons.to_json(binning, b_file, warn=False)
        b_ = MultiDimBinning.from_json(b_file)
        assert b_ == binning, 'binning=\n%s\nb_=\n%s' %(binning, b_)
    finally:
        shutil.rmtree(testdir, ignore_errors=True)

    # Test that reordering dimensions works correctly
    e_binning = OneDimBinning(
        name='energy', num_bins=80, is_log=True, domain=[1, 80]*ureg.GeV
    )
    cz_binning = OneDimBinning(
        name='coszen', num_bins=40, is_lin=True, domain=[-1, 1]
    )
    az_binning = OneDimBinning(
        name='azimuth', num_bins=10, is_lin=True,
        domain=[0*ureg.rad, 2*np.pi*ureg.rad]
    )
    mdb_2d_orig = MultiDimBinning([e_binning, cz_binning])
    orig_order = mdb_2d_orig.names

    # Reverse ordering; reorder by dimension names
    new_order = orig_order[::-1]
    mdb_2d_new = MultiDimBinning(mdb_2d_orig)
    mdb_2d_new = mdb_2d_new.reorder_dimensions(new_order)

    assert mdb_2d_new.names == new_order
    new_order = ['azimuth', 'energy', 'coszen']
    mdb_2d_new = mdb_2d_new.reorder_dimensions(new_order)
    assert mdb_2d_new == mdb_2d_orig
    mdb_2d_new2 = MultiDimBinning([e_binning, cz_binning])

    mdb_3d_orig = MultiDimBinning([e_binning, cz_binning, az_binning])
    orig_order = mdb_3d_orig.names
    new_order = [orig_order[2], orig_order[0], orig_order[1]]

    mdb_3d_new = MultiDimBinning(mdb_3d_orig)
    mdb_3d_new = mdb_3d_new.reorder_dimensions(new_order)
    assert mdb_3d_new.names == new_order
    # Reorder by MultiDimBinning object
    mdb_3d_new = mdb_3d_new.reorder_dimensions(mdb_3d_orig)
    assert mdb_3d_new.names == orig_order

    # Reorder by indices
    mdb_3d_new = MultiDimBinning(mdb_3d_orig)
    mdb_3d_new = mdb_3d_new.reorder_dimensions([2, 0, 1])
    assert mdb_3d_new.names == new_order

    # Reorder by combination of index, OneDimBinning, and name
    mdb_3d_new = MultiDimBinning(mdb_3d_orig)
    mdb_3d_new = mdb_3d_new.reorder_dimensions(
        [2, 'energy', mdb_2d_orig.dimensions[1]]
    )
    assert mdb_3d_new.names == new_order

    # Reorder by superset
    mdb_2d_new = MultiDimBinning(mdb_3d_orig.dimensions[0:2])
    mdb_2d_new = mdb_2d_new = mdb_2d_new.reorder_dimensions(new_order)
    assert mdb_2d_new.names == [o for o in new_order if o in mdb_2d_new]

    # Reorder by subset
    mdb_3d_new = MultiDimBinning(mdb_3d_orig)
    try:
        mdb_3d_new = mdb_3d_new.reorder_dimensions(new_order[0:2])
    except Exception:
        pass
    else:
        raise Exception('Should not be able to reorder by subset.')

    logging.info('<< PASSED >> test_MultiDimBinning')


if __name__ == "__main__":
    set_verbosity(1)
    test_OneDimBinning()
    test_MultiDimBinning()
