"""
Class to define binning in one dimension (OneDimBinning) and then a container
class (MultiDimBinning) for arbitrarily many of dimensions (one or more). These
classes have many useful methods for working with binning.
"""

# TODO: include Iterables where only Sequence is allowed now?
# TODO: iterbins, itercoords are _*slow*_. Figure out how to speed these up, if
#       that is possible in pure-Python loops... E.g.
#           `indices = [i for i in range(mdb.size)]`
#       takes 70 ms while
#           `coords = [c for c in mdb.itercoords()]`
#       takes 10 seconds.
# TODO: Create non-validated version of OneDimBinning.__init__ to make
#       iterbins() fast
# TODO: explicitly set is_bin_spacing_log_uniform and
#       is_bin_spacing_lin_uniform to FP32 precision (since binning can be
#       defined/saved in FP32 but want code able to run in FP64


from __future__ import absolute_import, division

from collections.abc import Iterable, Mapping, Sequence
from collections import OrderedDict, namedtuple
from copy import copy, deepcopy
from functools import reduce, wraps
from itertools import chain, product
from operator import mul
import re

import numpy as np

from pisa import FTYPE, HASH_SIGFIGS, ureg
from pisa.utils.comparisons import interpret_quantity, normQuant, recursiveEquality
from pisa.utils.comparisons import ALLCLOSE_KW
from pisa.utils.format import (make_valid_python_name, text2tex,
                               strip_outer_dollars)
from pisa.utils.hash import hash_obj
from pisa.utils import jsons
from pisa.utils.log import logging, set_verbosity, tprofile


__all__ = ['NAME_FIXES', 'NAME_SEPCHARS', 'NAME_FIXES_REGEXES',
           'basename', '_new_obj', 'is_binning',
           'OneDimBinning', 'MultiDimBinning',
           'test_OneDimBinning', 'test_MultiDimBinning']

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


NAME_FIXES = ('tru(e|th)', 'reco(nstruct(ed)?)?')
NAME_SEPCHARS = r'[_\s-]+'
NAME_FIXES_REGEXES = tuple(
    [re.compile(NAME_SEPCHARS + p, re.IGNORECASE) for p in NAME_FIXES]
    + [re.compile(p + NAME_SEPCHARS, re.IGNORECASE) for p in NAME_FIXES]
    + [re.compile(p, re.IGNORECASE) for p in NAME_FIXES]
)


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
    >>> print(basename('true_energy'))
    energy
    >>> print(basename('Reconstructed coszen'))
    coszen
    >>> print(basename('coszen  reco'))
    coszen
    >>> print(basename('energy___truth'))
    energy
    >>> print(basename('trueenergy'))
    energy
    >>> print(basename('energytruth'))
    energy

    """
    # Type checkingn and conversion
    orig_type = type(n)
    if isinstance(n, OneDimBinning):
        n = n.name
    if not isinstance(n, str):
        raise ValueError('Unhandled type %s' %orig_type)
    # Remove all (pre/suf)fixes and any separator chars
    for regex in NAME_FIXES_REGEXES:
        n = regex.sub('', n)
    return n.strip()


def is_binning(something):
    """Return True if argument is a PISA binning (of any dimension), False
    otherwise"""
    return isinstance(something, (OneDimBinning, MultiDimBinning))


# TODO: generalize to any object and move this to a centralized utils location
def _new_obj(original_function):
    """Decorator to deepcopy unaltered states into new OneDimBinning object."""
    @wraps(original_function)
    def new_function(cls, *args, **kwargs):
        """<< docstring will be inherited from wrapped function >>"""
        new_state = OrderedDict()
        state_updates = original_function(cls, *args, **kwargs)
        for attr in cls._attrs_to_create_new:  # pylint: disable=protected-access
            if attr in state_updates:
                new_state[attr] = state_updates[attr]
            else:
                new_state[attr] = deepcopy(getattr(cls, attr))
        return OneDimBinning(**new_state)
    return new_function


class OneDimBinning(object):
    # pylint: disable=line-too-long
    """Histogram-oriented binning specialized to a single dimension.

    If neither `is_lin` nor `is_log` is specified, linear behavior
    is assumed (i.e., `is_lin` is set to True).

    Parameters
    ----------
    name : str, of length > 0
        Name for this dimension. Must be valid Python name (since it will be
        accessed with the dot operator). If not, name will be converted to a
        valid Python name.

    tex : str or None
        TeX label for this dimension.

    bin_edges : sequence of scalars, or None
        Numerical values (optionally including Pint units) that represent the
        *edges* of the bins. `bin_edges` needn't be specified if `domain`,
        `num_bins`, and optionally `is_log` is specified. Pint units can be
        attached to `bin_edges`, but will be converted to `units` if this
        argument is specified.

    units : Pint unit or object convertible to Pint unit, or None
        If None, units will be read from either `bin_edges` or `domain`, and if
        none of these have units, the binning has unit 'dimensionless'
        attached.

    is_lin : bool or None
        Binning behavior is linear for purposes of resampling, plotting, etc.
        Mutually exclusive with `is_log`. If neither `is_lin` or `is_log` is
        True (i.e., both are None), default behavior is linear (`is_lin` is set
        to True internally).

    is_log : bool or None
        Binning behavior is logarithmic for purposes of resampling, plotting,
        etc. Mutually exclusive with `is_lin`. If neither `is_lin` or `is_log`
        is True (i.e., both are None), default behavior is linear (`is_lin` is
        set to True internally).

    domain : length-2 sequence of scalars, or None
        Units may be specified. Required along with `num_bins` if `bin_edges`
        is not specified (optionally specify `is_log=True` to define the
        `bin_edges` to be log-uniform).

    num_bins : int or None
        Number of bins. Required along with `domain` if `bin_edges` is not
        specified (optionally specify `is_log=True` to define the `bin_edges`
        to be log-uniform).

    bin_names : sequence of nonzero-length strings, or None
        Strings by which each bin can be identified. This is expected to be
        useful when one needs to easily identify bins by name where the actual
        numerical values can be non-obvious e.g. the PID dimension.
        None is also acceptable if there is no reason to name the bins.


    Notes
    -----
    Consistency is enforced for all redundant parameters passed to the
    constructor.

    You can avoid passing `bin_edges` if `num_bins` and `domain` are specified.
    Specify `is_lin=True` or `is_log=True` to define the binning to be linear
    or logarithmic (but note that if neither is specified as True, linear
    behavior is the default).

    Be careful, though, since bin edges will be defined slightly differently
    depending on the ``pisa.FTYPE`` defined (PISA_FTYPE environment variable).


    Examples
    --------
    >>> from pisa import ureg
    >>> from pisa.core.binning import OneDimBinning
    >>> ebins = OneDimBinning(name='energy', is_log=True,
    ...                       num_bins=40, domain=[1, 100]*ureg.GeV)
    >>> print(ebins)
    OneDimBinning('energy', 40 logarithmically-regular bins spanning [1.0, 100.0] GeV (behavior is logarithmic))
    >>> ebins2 = ebins.to('joule')
    >>> print(ebins2)
    OneDimBinning('energy', 40 logarithmically-regular bins spanning [1.6021766339999998e-10, 1.602176634e-08] J (behavior is logarithmic))
    >>> czbins = OneDimBinning(name='coszen',
    ...                        is_lin=True, num_bins=4, domain=[-1, 0])
    >>> print(czbins)
    OneDimBinning('coszen', 4 linearly-regular bins spanning [-1.0, 0.0] (behavior is linear))
    >>> czbins2 = OneDimBinning(name='coszen',
    ...                         bin_edges=[-1, -0.75, -0.5, -0.25, 0])
    >>> czbins == czbins2
    True

    """
    # pylint: enable=line-too-long

    # NOTE: Only one of `is_log` or `is_lin` is technically required for state,
    #   since it's either one or the other, but these used to imply
    #   log-_uniform_ and linear-_uniform_ behavior. As nothing fundamnetally
    #   in the behavior is changed by assuming uniformity, and to keep
    #   backwards compatibility (including for state / hashes), both are kept
    #   (for now) as "state" variables. -JLL, April, 2020

    _attrs_to_create_new = ('name', 'tex', 'bin_edges', 'is_log', 'is_lin', 'bin_names')

    def __init__(self, name, tex=None, bin_edges=None, units=None, domain=None,
                 num_bins=None, is_lin=None, is_log=None, bin_names=None):
        # Basic validation and translation of args; note that iterables are
        # converted to sequences later on
        if not isinstance(name, str):
            raise TypeError('`name` must be a string; got "%s".' %type(name))
        if domain is not None:
            assert (
                isinstance(domain, Iterable)
                or (isinstance(domain, ureg.Quantity) and domain.size > 1)
            )
        if bin_names is not None:
            if isinstance(bin_names, str):
                bin_names = (bin_names,)
            if (isinstance(bin_names, Iterable)
                    and all(isinstance(n, str) and n
                            for n in bin_names)):
                bin_names = tuple(bin_names)
            else:
                raise ValueError(
                    '`bin_names` must either be None or an iterable of'
                    ' nonzero-length strings.'
                )
        if bin_edges is not None:
            assert (
                isinstance(bin_edges, Iterable)
                or (isinstance(bin_edges, ureg.Quantity) and bin_edges.size > 1)
            )
            if domain is not None:
                raise ValueError(
                    'Both `domain` and `bin_edges` are specified.'
                    )
            
        # Type checking
        assert is_lin is None or isinstance(is_lin, bool), str(type(is_lin))
        assert is_log is None or isinstance(is_log, bool), str(type(is_log))

        if is_lin is None and is_log is None:  # neither specified: default to linear
            is_lin = True
            is_log = not is_lin

        elif is_lin is not None:  # is_lin is specified but not is_log
            is_log = not is_lin

        elif is_log is not None:  # is_log is specified but not is_lin
            is_lin = not is_log

        else:  # both specified: check consistency
            if is_log == is_lin:
                raise ValueError(
                    '`is_log=%s` contradicts `is_lin=%s`' % (is_log, is_lin)
                )

        self._normalize_values = True
        self._name = make_valid_python_name(name)
        if self._name != name:
            logging.warning('Converted `name` "%s" to valid Python: "%s"',
                            name, self._name)
        self._tex = tex
        self._basename = None
        self._bin_names = bin_names
        self._hashable_state = None
        self._serializable_state = None
        self._normalized_state = None
        self._midpoints = None
        self._weighted_centers = None
        self._edge_magnitudes = None
        self._bin_widths = None
        self._weighted_bin_widths = None
        self._inbounds_criteria = None

        # TODO: define hash based upon conversion of things to base units (such
        # that a valid comparison can be made between indentical binnings but
        # that use different units). Be careful to round to just less than
        # double-precision limits after conversion so that hashes will work out
        # to be the same after conversion to the base units.

        self._hash = None
        self._edges_hash = None

        # Figure out the units (if any) for each quantity passed in. Precedence
        # for units is:
        #   1. `units`
        #   2. `bin_edges`
        #   3. `domain`
        #   4. default to units of `ureg.dimensionless`

        if units is not None:
            if isinstance(units, ureg.Quantity):
                units = units.units
            elif not isinstance(units, ureg.Unit):
                units = ureg.Unit(units)
            units_dimensionality = units.dimensionality

        dimensionless_bin_edges = None

        if bin_edges is not None:
            if isinstance(bin_edges, ureg.Quantity):
                be_units = bin_edges.units
                if units is None:
                    units = be_units
                else:
                    if be_units.dimensionality != units_dimensionality:
                        raise ValueError(
                            '`bin_edges` units %s are incompatible with units'
                            ' %s.' % (be_units, units)
                        )
                    if be_units != units:
                        logging.warning(
                            '`bin_edges` are specified in units of %s'
                            ' but `units` is specified as %s.'
                            ' Converting `bin_edges` to the latter.',
                            be_units, units
                        )
                        bin_edges.ito(units)
                dimensionless_bin_edges = bin_edges.magnitude

            elif bin_edges is not None:
                dimensionless_bin_edges = np.asarray(bin_edges, dtype=FTYPE)
                bin_edges = None

        dimensionless_domain = None

        if domain is not None:
            if isinstance(domain, ureg.Quantity):
                domain_units = domain.units
                if units is None:
                    units = domain_units
                else:
                    if domain_units.dimensionality != units_dimensionality:
                        raise ValueError(
                            '`domain` units %s are incmompatible with units'
                            ' %s.' % (domain_units, units)
                        )
                    if domain_units != units:
                        logging.warning(
                            '`domain` units %s will be converted to' ' %s.',
                            domain_units,
                            units,
                        )
                        domain.ito(units)
                dimensionless_domain = domain.magnitude

            else:
                domain_lower_is_quant = isinstance(domain[0], ureg.Quantity)
                domain_upper_is_quant = isinstance(domain[1], ureg.Quantity)
                assert domain_lower_is_quant == domain_upper_is_quant
                if domain_lower_is_quant:
                    assert domain[0].dimensionality == domain[1].dimensionality
                    if domain[1].units != domain[0].units:
                        domain[1] = domain[1].to(domain[0].units)
                    dimensionless_domain = (domain[0].magnitude,
                                            domain[1].magnitude)
                else:
                    dimensionless_domain = tuple(domain)
                    domain = None

        # If no units have been discovered from the input args, assign default
        # units
        if units is None:
            units = ureg.dimensionless

        if dimensionless_bin_edges is None:
            if num_bins is None or dimensionless_domain is None:
                raise ValueError(
                    'If not specifying bin edges explicitly, `domain` and'
                    ' `num_bins` must be specified (and optionally set'
                    ' `is_log=True`).'
                )
            if is_log:
                dimensionless_bin_edges = np.logspace(
                    np.log10(dimensionless_domain[0]),
                    np.log10(dimensionless_domain[1]),
                    num_bins + 1,
                    dtype=FTYPE,
                )
            else:  # is_lin
                dimensionless_bin_edges = np.linspace(
                    dimensionless_domain[0],
                    dimensionless_domain[1],
                    num_bins + 1,
                    dtype=FTYPE,
                )
        elif dimensionless_domain is not None:
            assert dimensionless_domain[0] == dimensionless_bin_edges[0]
            assert dimensionless_domain[1] == dimensionless_bin_edges[-1]

        # TODO: should we warn a user if logarithmically- or linearly-uniform
        # bin_edges are passed while is_log, is_lin, or the default (is_lin)
        # "contradict" this?

        # if not (is_lin or is_log):  # infer is_log/is_lin from spacing if not set
        #     is_lin = self.is_bin_spacing_lin_uniform(dimensionless_bin_edges)
        #     try:
        #         is_log = self.is_bin_spacing_log_uniform(dimensionless_bin_edges)
        #     except ValueError:
        #         is_log = False

        if dimensionless_domain is None:
            dimensionless_domain = (dimensionless_bin_edges[0],
                                    dimensionless_bin_edges[-1])

        if bin_edges is None:
            self._bin_edges = dimensionless_bin_edges * units
        else:
            self._bin_edges = bin_edges

        if domain is None:
            self._domain = dimensionless_domain * units
        else:
            self._domain = domain

        self._units = units

        # Derive rest of unspecified parameters from bin_edges or enforce
        # them if they were specified as arguments to init
        if num_bins is None:
            num_bins = len(self.bin_edges) - 1
        else:
            assert num_bins == len(self.bin_edges) - 1, \
                    '%s, %s' %(num_bins, self.bin_edges)
        self._num_bins = num_bins

        if (self._bin_names is not None
                and len(self._bin_names) != self._num_bins):
            raise ValueError(
                'There are %d bins, so there must be %d `bin_names` (or None)'
                ' provided; got %d bin name instead: %s.'
                % (self._num_bins, self._num_bins, len(self._bin_names),
                   self._bin_names)
            )

        self._is_log = is_log
        self._is_lin = not self._is_log
        self._is_irregular = None

    def __repr__(self):
        previous_precision = np.get_printoptions()['precision']
        np.set_printoptions(precision=18)
        try:
            argstrs = [('%s=%r' %item) for item in
                       self.serializable_state.items()]
            r = '%s(%s)' %(self.__class__.__name__, ',\n    '.join(argstrs))
        finally:
            np.set_printoptions(precision=previous_precision)
        return r

    def __str__(self):
        lin_reg = "linearly-regular "
        log_reg = "logarithmically-regular "

        regularity = None
        if self.is_irregular:
            # Test for regularity in the other domain
            if self.is_lin and self.is_bin_spacing_log_uniform(self.bin_edges):
                regularity = log_reg
            elif self.is_log and self.is_bin_spacing_lin_uniform(self.bin_edges):
                regularity = lin_reg
        else:
            regularity = log_reg if self.is_log else lin_reg

        if regularity is None or self.num_bins == 1:
            edges = 'with edges at [{}]{}'.format(
                ', '.join(str(e) for e in self.bin_edges.m),
                format(self.bin_edges.u, '~'),
            ).strip()
            regularity = ""
        else:
            edges = 'spanning [{}, {}] {:s}'.format(
                self.bin_edges[0].magnitude,
                self.bin_edges[-1].magnitude,
                format(self.units, '~'),
            ).strip()

        plural = '' if self.num_bins == 1 else 's'
        linlog = 'logarithmic' if self.is_log else 'linear'

        if self.bin_names is None:
            bnames = ""
        else:
            bnames = ', bin_names=[{}]'.format(
                ', '.join(f"'{n:s}'" for n in self.bin_names)
            )

        descr = (
            f'{self.num_bins:d} {regularity:s}bin{plural:s} {edges:s}'
            f' (behavior is {linlog:s}){bnames:s}'
        )

        return f"{self.__class__.__name__:s}('{self.name:s}', {descr:s})"

    def __pretty__(self, p, cycle):
        """Method used by the `pretty` library for formatting"""
        if cycle:
            p.text('%s(...)' % self.__class__.__name__)
        else:
            p.begin_group(4, '%s' % self)
            p.end_group(4, ')')

    def _repr_pretty_(self, p, cycle):
        """Method used by e.g. ipython/Jupyter for formatting"""
        return self.__pretty__(p, cycle)

    def __getstate__(self):
        """Method invoked during pickling"""
        return self.serializable_state

    def __setstate__(self, state):
        """Method invoked during unpickling"""
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
        from_json : Instantiate new OneDimBinning object from the file written
            by this method

        """
        jsons.to_json(self.serializable_state, filename=filename, **kwargs)

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
            if isinstance(x, str):
                assert self.bin_names is not None
                return self.bin_names.index(x)
            if isinstance(x, int):
                assert 0 <= x < len(self)
                return x
            raise TypeError('`x` must be either int or string; got %s instead.'
                            % type(x))
        except (AssertionError, ValueError):
            valid_range = [0, len(self)-1]
            if self.bin_names is None:
                valid_names = ''
            else:
                valid_names = ' or a valid bin name in %s' % (self.bin_names,)
            raise ValueError('Bin corresponding to "%s" could not be located.'
                             ' Specify an int in %s%s.'
                             % (x, valid_range, valid_names))

    def iterbins(self):
        """Return an iterator over each bin. The elments returned by the
        iterator are each a OneDimBinning object, just containing a single bin.

        Note that for one test, `iterbins` is about 500x slower than
        `iteredgetuples`.

        Returns
        -------
        bin_iterator

        See Also
        --------
        iteredgetuples
            Faster but only returns edges of bins, not OneDimBinning objects.

        """
        return (self[i] for i in range(len(self)))

    def iteredgetuples(self):
        """Return an iterator over each bin's edges. The elments returned by
        the iterator are each a tuple, containing the edges of the bin. Units
        are stripped prior to iteration for purposes of speed.

        Returns
        -------
        edges_iterator

        See Also
        --------
        iterbins
            Similar, but returns a OneDimBinning object for each bin; slower
            than this method (by as much as 500x in one test) but easier to
            work with.

        """
        mags = self.edge_magnitudes
        return ((e0, e1) for e0, e1 in zip(mags[:-1], mags[1:]))

    @property
    def serializable_state(self):
        """OrderedDict containing savable state attributes"""
        if self._serializable_state is None:
            state = OrderedDict()
            state['name'] = self.name
            state['bin_edges'] = self.edge_magnitudes
            state['units'] = str(self.units)
            state['is_log'] = self.is_log
            state['is_lin'] = self.is_lin
            state['bin_names'] = self.bin_names
            self._serializable_state = state
        # Since the tex property can be modified, must set every time this
        # property is called
        self._serializable_state['tex'] = self.tex
        return self._serializable_state

    @property
    def hashable_state(self):
        """OrderedDict containing simplified state attributes (i.e. some state
        attributes are represented by their hashes) used for testing equality
        between two objects.

        Use `hashable_state` for faster equality checks and `normalized_state`
        for inspecting the contents of each state attribute pre-hashing
        """
        if self._hashable_state is None:
            state = OrderedDict()
            state['name'] = self.name
            state['edges_hash'] = self.edges_hash
            state['is_log'] = self.is_log
            state['is_lin'] = self.is_lin
            state['bin_names'] = self.bin_names
            self._hashable_state = state
        return self._hashable_state

    @property
    def normalized_state(self):
        """OrderedDict containing normalized (base units, and rounded to
        appropriate precision) state attributes used for testing equality
        between two objects.

        Use `hashable_state` for faster equality checks and `normalized_state`
        for inspecting the contents of each state attribute pre-hashing
        """
        if self._normalized_state is None:
            state = OrderedDict()
            state['name'] = self.name
            bin_edges = normQuant(self.bin_edges, sigfigs=HASH_SIGFIGS)
            state['bin_edges'] = bin_edges
            state['is_log'] = self.is_log
            state['is_lin'] = self.is_lin
            state['bin_names'] = self.bin_names
            self._normalized_state = state
        return self._normalized_state

    @property
    def edge_magnitudes(self):
        """Bin edges' magnitudes"""
        if self._edge_magnitudes is None:
            self._edge_magnitudes = self.bin_edges.magnitude
        return self._edge_magnitudes

    @property
    def name(self):
        """Name of the dimension"""
        return self._name

    @property
    def basename(self):
        """Basename of the dimension, stripping "true", "reco", underscores,
        whitespace, etc. from the `name` attribute."""
        if self._basename is None:
            self._basename = basename(self.name)
        return self._basename

    # TODO: reimplement just the translate-on-input (or not?), but is this a
    # performance hit for e.g. iterbins()? That could argue for
    # translate-on-output...

    @property
    def tex(self):
        """string : TeX label"""
        if self._tex is None:
            return text2tex(self.name)
        return self._tex

    @tex.setter
    def tex(self, val):
        """None or TeX string for dimension; surrounding dollars-signs ($) are
        stripped off (and must be added prior to e.g. plotting)"""
        assert val is None or isinstance(val, str)
        if val is not None:
            val = strip_outer_dollars(val)
        self._tex = val

    @property
    def label(self):
        """TeX-formatted axis label, including units (if not dimensionless)"""
        if self.tex is None:
            name_tex = r'{\rm %s}' % text2tex(self.name)
        else:
            name_tex = self.tex

        if self.units == ureg.dimensionless:
            units_tex = ''
        else:
            units_tex = r' \; \left( {:~L} \right)'.format(self.units)

        return name_tex + units_tex

    @property
    def shape(self):
        """tuple : shape of binning, akin to `nump.ndarray.shape`"""
        return (self.num_bins,)

    @property
    def size(self):
        """int : total number of bins"""
        return self.num_bins

    @property
    def bin_edges(self):
        """array : Edges of the bins."""
        return self._bin_edges

    @property
    def bin_names(self):
        """list of strings or None : Bin names"""
        return self._bin_names

    @property
    def domain(self):
        """array : domain of the binning, (min, max) bin edges"""
        if self._domain is None:
            bin_edges = self.edge_magnitudes
            self._domain = np.array([np.min(bin_edges),
                                     np.max(bin_edges)]) * self.units
        return self._domain

    @property
    def range(self):
        """float : range of the binning, (max-min) bin edges"""
        domain = self.domain
        return domain[1] - domain[0]

    @property
    def units(self):
        """pint.Unit : units of the bins' edges"""
        return self._units

    @units.setter
    def units(self, u):
        """str or pint.Unit : units of the bins' edges"""
        self.ito(u)

    @property
    def num_bins(self):
        """int : Number of bins"""
        return self._num_bins

    @property
    def is_lin(self):
        """bool : Whether binning is to be treated in a linear space"""
        return self._is_lin

    @is_lin.setter
    def is_lin(self, b):
        """bool"""
        assert isinstance(b, bool)
        if b != self._is_lin:
            # NOTE: use tuple unpacking to help ensure state is consistent
            (
                self._is_lin,
                self._is_log,
                self._is_irregular,
                self._weighted_centers,
                self._weighted_bin_widths,
            ) = (b, not b, None, None, None)

    @property
    def is_log(self):
        """bool : Whether binning is to be treated in a log space"""
        return self._is_log

    @is_log.setter
    def is_log(self, b):
        """bool"""
        assert isinstance(b, bool)
        if b != self._is_log:
            # NOTE: use tuple unpacking to help ensure state is consistent
            (
                self._is_log,
                self._is_lin,
                self._is_irregular,
                self._weighted_centers,
                self._weighted_bin_widths,
            ) = (b, not b, None, None, None)

    @property
    def is_irregular(self):
        """bool : True if bin spacing is not unform in the space defined (i.e.,
        NOT linearly-uniform if `is_lin` or NOT logarithmically-uniform if
        `is_log`)."""
        if self._is_irregular is None:
            if self.num_bins == 1:
                self._is_irregular = False
            elif self.is_log:
                self._is_irregular = not self.is_bin_spacing_log_uniform(self.bin_edges)
            else:  # self.is_lin
                self._is_irregular = not self.is_bin_spacing_lin_uniform(self.bin_edges)
        return self._is_irregular

    @property
    def midpoints(self):
        """array : Midpoints of the bins: linear average of each bin's
        edges."""
        if self._midpoints is None:
            self._midpoints = (self.bin_edges[:-1] + self.bin_edges[1:])/2.0
        return self._midpoints

    @property
    def weighted_centers(self):
        """array : Centers of the bins taking e.g. logarithmic behavior
        into account. I.e., if binning is logarithmic, this is **not**
        the same `midpoints`, whereas in all other cases, it is identical."""
        if self._weighted_centers is None:
            if self.is_log:
                self._weighted_centers = np.sqrt(self.bin_edges[:-1] *
                                                 self.bin_edges[1:])
            else:
                self._weighted_centers = self.midpoints
        return self._weighted_centers

    @property
    def hash(self):
        """int : Hash value based upon less-than-double-precision-rounded
        numerical values and any other state (includes name, tex, is_log, and
        is_lin attributes). Rounding is done to `HASH_SIGFIGS` significant
        figures.

        Set this class attribute to None to keep full numerical precision in
        the values hashed (but be aware that this can cause equal things
        defined using different unit orders-of-magnitude to hash differently).

        """
        if self._hash is None:
            s = self.hashable_state
            self._hash = hash_obj(s)
        return self._hash

    def rehash(self):
        """Force `hash` and `edges_hash` attributes to be recomputed"""
        self._hash = None
        self._edges_hash = None
        _ = self.hash
        _ = self.edges_hash

    def __hash__(self):
        return self.hash

    @property
    def normalize_values(self):
        """bool : Whether to normalize quantities' units prior to hashing"""
        return self._normalize_values

    @normalize_values.setter
    def normalize_values(self, b):
        assert isinstance(b, bool)
        if b == self._normalize_values:
            return
        # Invalidate the hash, since the hashing behavior has changed
        self._hash = None
        self._edges_hash = None
        self._normalize_values = b

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

    @property
    def bin_widths(self):
        """Absolute widths of bins."""
        if self._bin_widths is None:
            self._bin_widths = np.abs(np.diff(self.bin_edges.m)) * self.units
        return self._bin_widths

    @property
    def weighted_bin_widths(self):
        """Absolute widths of bins."""
        if self._weighted_bin_widths is None:
            if self.is_log:
                self._weighted_bin_widths = (
                    np.log(self.edge_magnitudes[1:] / self.edge_magnitudes[:-1])
                ) * ureg.dimensionless
            else:
                self._weighted_bin_widths = self.bin_widths
        return self._weighted_bin_widths

    @property
    def inbounds_criteria(self):
        """Return string boolean criteria indicating e.g. an event falls within
        the limits of the defined binning.

        This can be used for e.g. applying cuts to events.

        See Also
        --------
        pisa.core.events.keepInbounds

        """
        if self._inbounds_criteria is None:
            be = self.edge_magnitudes
            crit = '(%s >= %.15e) & (%s <= %.15e)' % (self.name, min(be),
                                                      self.name, max(be))
            self._inbounds_criteria = crit
        return self._inbounds_criteria

    def __len__(self):
        """Number of bins (*not* number of bin edges)."""
        return self.num_bins

    def __mul__(self, other):
        if isinstance(other, OneDimBinning):
            return MultiDimBinning([self, other])
        if isinstance(other, MultiDimBinning):
            return MultiDimBinning(chain([self], other))
        return OneDimBinning(name=self.name, tex=self.tex,
                             bin_edges=self.bin_edges * other)

    # TODO: if same or contained dimension, modify the current binning OR
    # create a smarter MultiDimBinning object that allows for multiple
    # disconnected binning regions with arbitrary binning within each
    # region
    def __add__(self, other):
        if isinstance(other, OneDimBinning):
            return MultiDimBinning([self, other])

        if isinstance(other, MultiDimBinning):
            return MultiDimBinning(chain([self], other))

        other = interpret_quantity(other, expect_sequence=True)

        new_bin_edges = self.bin_edges + other

        return OneDimBinning(name=self.name, tex=self.tex, bin_edges=new_bin_edges)

    @_new_obj
    def __deepcopy__(self, memo):
        """Explicit deepcopy constructor"""
        return {}

    @staticmethod
    def is_bin_spacing_log_uniform(bin_edges):
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
        if hasattr(bin_edges, 'magnitude'):
            bin_edges = bin_edges.magnitude
        bin_edges = np.asarray(bin_edges)
        if len(bin_edges) < 3:
            raise ValueError('%d bin edge(s) passed; require at least 3 to'
                             ' determine nature of bin spacing.'
                             % len(bin_edges))
        with np.errstate(divide='raise', over='raise', under='raise',
                         invalid='raise'):
            try:
                log_spacing = bin_edges[1:] / bin_edges[:-1]
            except (AssertionError, FloatingPointError, ZeroDivisionError):
                return False
        if np.allclose(log_spacing, log_spacing[0], **ALLCLOSE_KW):
            return True
        return False

    @staticmethod
    def is_bin_spacing_lin_uniform(bin_edges):
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
        if hasattr(bin_edges, 'magnitude'):
            bin_edges = bin_edges.magnitude
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
        if np.allclose(lin_spacing, lin_spacing[0], **ALLCLOSE_KW):
            return True
        return False

    @staticmethod
    def is_binning_ok(bin_edges):
        """Check that there are 2 or more bin edges, and that they are
        monotonically increasing.

        Parameters
        ----------
        bin_edges : sequence
            Bin edges to check the validity of

        Returns
        -------
        bool, True if binning is OK, False if not

        """
        # Must be at least two edges to define a single bin
        if len(bin_edges) < 2:
            return False
        # Bin edges must be monotonic and strictly increasing
        if np.any(np.diff(bin_edges) <= 0):
            return False
        return True

    # TODO: as of now, only downsampling is allowed. Is this reasonable?
    def is_compat(self, other):
        """Compatibility -- for now -- is defined by all of self's bin
        edges form a subset of other's bin edges (i.e. you can downsample to
        get from the other binning to this binning), and the units must be
        compatible.

        Note that this might bear revisiting, or redefining just for special
        circumstances.

        Parameters
        ----------
        other : OneDimBinning

        Returns
        -------
        bool

        """
        if self.name != other.name:
            logging.trace('Dimension names do not match')
            return False

        if self.units.dimensionality != other.units.dimensionality:
            logging.trace('Incompatible units')
            return False

        # TODO: should we force normalization?
        # TODO: Should we use FTYPE_SIGFIGS or # HASH_SIGFIGS?
        if self.normalize_values:
            my_normed_bin_edges = set(
                normQuant(self.bin_edges, sigfigs=HASH_SIGFIGS).magnitude
            )
            other_normed_bin_edges = set(
                normQuant(other.bin_edges, sigfigs=HASH_SIGFIGS).magnitude
            )
        else:
            my_normed_bin_edges = set(self.bin_edges.magnitude)
            other_normed_bin_edges = set(other.bin_edges.magnitude)

        if my_normed_bin_edges.issubset(other_normed_bin_edges):
            return True

        logging.trace('self.bin_edges not a subset of other.bin_edges')
        logging.trace('Bins in this map not found in other = %s',
                      my_normed_bin_edges.difference(other_normed_bin_edges))

        return False

    @property
    @_new_obj
    def basename_binning(self):
        """Identical binning but named as the basename of this binning. Note
        that the `tex` property is not carried over into the new binning."""
        return {'name': self.basename, 'tex': None}

    @property
    @_new_obj
    def finite_binning(self):
        """Identical binning but with infinities in bin edges replaced by
        largest/smallest floating-point numbers representable with the current
        pisa.FTYPE."""
        float_info = np.finfo(FTYPE)
        finite_edges = np.clip(self.edge_magnitudes, a_min=float_info.min,
                               a_max=float_info.max)
        return {'bin_edges': finite_edges}

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
        new_binning : OneDimBinning
            New binning, oversampled from the current binning.

        Raises
        ------
        ValueError if illegal value is specified for `factor`

        Notes
        -----
        Bin names are _not_ preserved for any `factor` except 1 since it is
        ambiguous how names should be propagated. If you wish to have bin
        names after oversampling, assign them afterwards.

        """
        if factor < 1 or factor != int(factor):
            raise ValueError('`factor` must be integer >= 1; got %s' %factor)

        factor = int(factor)

        if factor == 1:
            return self

        if self.is_log:
            spacing_func = np.geomspace
        else:  # is_lin
            spacing_func = np.linspace

        old_bin_edges = self.edge_magnitudes
        new_bin_edges = []
        for old_lower, old_upper in zip(old_bin_edges[:-1], old_bin_edges[1:]):
            thisbin_new_edges = spacing_func(old_lower, old_upper, factor + 1)

            # Use the original lower bin edge to avoid precision issues with
            # its version created by `spacing_func`
            new_bin_edges.append(old_lower)

            # Add the new bin edges we created in between lower and upper; we
            # omit the upper bin edge because it is the first bin edge of the
            # next bin
            new_bin_edges.extend(thisbin_new_edges[1:-1])

        # Include the uppermost bin edge from original binning
        new_bin_edges.append(old_upper)

        return {'bin_edges': new_bin_edges * self.units, 'bin_names': None}

    # TODO: do something cute with bin names, if they exist?
    @_new_obj
    def downsample(self, factor):
        """Downsample the binning by an integer factor that evenly divides the
        current number of bins.

        Parameters
        ----------
        factor : int >= 1
            Downsampling factor that evenly divides the current number of
            bins. E.g., if the current number of bins is 4, `factor` can be
            one of 1, 2, or 4. Note that floats are converted into integers
            if `float(factor) == int(factor)`.

        Returns
        -------
        new_binning : OneDimBinning
            New binning, downsampled from the current binning.

        Raises
        ------
        ValueError if illegal value is specified for `factor`

        Notes
        -----
        Bin names are _not_ preserved for any `factor` except 1 since it is
        ambiguous how names should be propagated. If you wish to have bin
        names after downsampling, assign them afterwards.

        """
        if int(factor) != float(factor):
            raise ValueError('Floating point `factor` is non-integral.')
        factor = int(factor)

        if factor == 1:
            return self

        if factor < 1 or factor > self.num_bins:
            raise ValueError(
                '`factor` %d is out of range; must be >= 1 and <= number of'
                ' bins (%d).' % (factor, self.num_bins)
            )

        if self.num_bins % factor != 0:
            raise ValueError(
                '`factor` %d does not evenly divide number of bins (%d).'
                % (factor, self.num_bins)
            )

        return {'bin_edges': self.bin_edges[::factor],
                'bin_names': None}

    def ito(self, units):
        """Convert units in-place. Cf. Pint's `ito` method."""
        if units is None:
            units = ''

        units = ureg.Unit(units)
        if units == self._units:
            return
        self._units = units

        # Invalidate (expensive) derived properties that rely on units
        for attr in ['_inbounds_criteria']:
            setattr(self, attr, None)

        # Convert already-defined quantities
        attrs = [
            '_bin_edges',
            '_domain',
            '_midpoints',
            '_weighted_centers',
            '_bin_widths',
            '_edge_magnitudes',
        ]
        for attr in attrs:
            val = getattr(self, attr)
            if val is None:
                continue
            val.ito(units)

    @_new_obj
    def to(self, units): # pylint: disable=invalid-name
        """Convert bin edges' units to `units`.

        Parameters
        ----------
        units : None, string, or pint.Unit

        Returns
        -------
        new_binning : OneDimBinning
            New binning object whose edges have units `units`

        """
        if units is None:
            units = 'dimensionless'
        return {'bin_edges': self.bin_edges.to(ureg(str(units)))}

    def __getattr__(self, attr):
        return super().__getattribute__(attr)

    # TODO: make this actually grab the bins specified (and be able to grab
    # disparate bins, whether or not they are adjacent)... i.e., fill in all
    # upper bin edges, and handle the case that it goes from linear or log
    # to uneven (or if it stays lin or log, keep that attribute for the
    # subselection). Granted, a OneDimBinning object right now requires
    # monotonically-increasing and adjacent bins.

    # TODO: make indexing allow for sequence containing a single ellipsis
    # TODO: for some reason, this is crazy, crazy slow when indexing with
    #       ellipsis... why?
    # NOTE: mabye we don't care, since using ellipsis (or even an isolated,
    #       single colon) in a one-dimensional object is a "violation of the
    #       contract": http://stackoverflow.com/a/118508
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

        magnitude = self.edge_magnitudes
        units = self.units
        orig_index = index
        mylen = len(magnitude) - 1
        bin_names = self.bin_names

        # Deal with indexing by name first so as to not break anything else
        if isinstance(index, str):
            assert bin_names is not None
            index = bin_names.index(index)

        # Simple to get all but final bin edge
        bin_edges = magnitude[index].tolist()

        if np.isscalar(bin_edges):
            bin_edges = [bin_edges]
        else:
            bin_edges = list(bin_edges)

        # Convert index/indices to positive-number sequence
        if isinstance(index, slice):
            index = list(range(*index.indices(mylen)))
        if isinstance(index, int):
            index = [index]

        if isinstance(index, Iterable):
            if not isinstance(index, Sequence):
                index = list(index)
            for bin_index in index:
                if isinstance(bin_index, str):
                    raise ValueError('Slicing by seq of names currently not'
                                     ' supported')
            if not index:
                raise ValueError('`index` "%s" results in no bins being'
                                 ' specified.' %orig_index)
            if len(index) > 1 and not np.all(np.diff(index) == 1):
                raise ValueError('Bin indices must be monotonically'
                                 ' increasing and adjacent.')
            new_edges = set()
            new_names = []
            for bin_index in index:
                if bin_index < -mylen or bin_index >= mylen:
                    raise ValueError(
                        "Dimension '%s': bin index %s is invalid. Bin index"
                        " must be >= %+d and <= %+d"
                        %(self.name, bin_index, -mylen, mylen-1)
                    )
                edge_ind0 = bin_index % mylen
                edge_ind1 = edge_ind0 + 1
                if bin_names is not None:
                    new_names.append(bin_names[edge_ind0])
                mag0 = magnitude[edge_ind0]
                mag1 = magnitude[edge_ind1]
                new_edges = new_edges.union((mag0, mag1))
        else:
            raise TypeError('Unhandled index type %s' %type(orig_index))

        if new_names == []:
            new_names = None
        # Retrieve current state; only bin_edges and bin_names need to be
        # updated
        new_edges = sorted(new_edges)
        new_edges = np.array(new_edges)
        new_edges = new_edges * units
        return {'bin_edges': new_edges, 'bin_names': new_names}

    def __iter__(self):
        return self.iterbins()

    def __eq__(self, other):
        if not isinstance(other, OneDimBinning):
            return False
        return recursiveEquality(self.hashable_state, other.hashable_state)

    def __ne__(self, other):
        return not self.__eq__(other)


class MultiDimBinning(object):
    # pylint: disable=line-too-long
    r"""
    Multi-dimensional binning object. This can contain one or more
    OneDimBinning objects, and all subsequent operations (e.g. slicing) will
    act on these in the order they are supplied.

    Note that it is convenient to construct MultiDimBinning objects via the *
    operator (which implementes the outer product) from multiple OneDimBinning
    objects. See Examples below for details.


    Parameters
    ----------
    dimensions : OneDimBinning or sequence convertible thereto
        Dimensions for the binning object. Indexing into the MultiDimBinning
        object follows the order in which dimensions are provided.


    See Also
    --------
    OneDimBinning : each item that is not a OneDimBinning object is passed to
        this class to be instantiated as such.


    Examples
    --------
    >>> from pisa import ureg
    >>> from pisa.core.binning import MultiDimBinning, OneDimBinning
    >>> ebins = OneDimBinning(name='energy', is_log=True,
    ...                       num_bins=40, domain=[1, 100]*ureg.GeV)
    >>> czbins = OneDimBinning(name='coszen',
    ...                        is_lin=True, num_bins=4, domain=[-1, 0])
    >>> mdb = ebins * czbins
    >>> print(mdb)
    MultiDimBinning(
        OneDimBinning('energy', 40 logarithmically-regular bins spanning [1.0, 100.0] GeV (behavior is logarithmic)),
        OneDimBinning('coszen', 4 linearly-regular bins spanning [-1.0, 0.0] (behavior is linear))
    )

    >>> print(mdb.energy)
    OneDimBinning('energy', 40 logarithmically-regular bins spanning [1.0, 100.0] GeV (behavior is logarithmic))
    >>> print(mdb[0, 0])
    MultiDimBinning(
        OneDimBinning('energy', 1 logarithmically-regular bin with edges at [1.0, 1.1220184543019633]GeV (behavior is logarithmic)),
        OneDimBinning('coszen', 1 linearly-regular bin with edges at [-1.0, -0.75] (behavior is linear))
    )
    >>> print(mdb.slice(energy=2))
    MultiDimBinning(
        OneDimBinning('energy', 1 logarithmically-regular bin with edges at [1.2589254117941673, 1.4125375446227544]GeV (behavior is logarithmic)),
        OneDimBinning('coszen', 4 linearly-regular bins spanning [-1.0, 0.0] (behavior is linear))
    )
    >>> smaller_binning = mdb[0:2, 0:3]
    >>> map = smaller_binning.ones(name='my_map')
    >>> print(map)
    Map(name='my_map',
        tex='{\\rm my\\_map}',
        full_comparison=False,
        hash=None,
        parent_indexer=None,
        binning=MultiDimBinning(
        OneDimBinning('energy', 2 logarithmically-regular bins spanning [1.0, 1.2589254117941673] GeV (behavior is logarithmic)),
        OneDimBinning('coszen', 3 linearly-regular bins spanning [-1.0, -0.25] (behavior is linear))
    ),
        hist=array([[1., 1., 1.],
           [1., 1., 1.]]))

    """
    # pylint: enable=line-too-long
    def __init__(self, dimensions, name=None):
        self.__map_class = None

        if isinstance(dimensions, OneDimBinning):
            dimensions = [dimensions]
        if not isinstance(dimensions, Sequence):
            if isinstance(dimensions, Mapping):
                if len(dimensions) == 1 and hasattr(dimensions, 'dimensions'):
                    dimensions = dimensions['dimensions']
                dimensions = [dimensions]
            elif isinstance(dimensions, Iterable):
                pass
            else:
                raise TypeError('`dimensions` unhandled type: %s'
                                % type(dimensions))
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
        self._dimensions = tuple(tmp_dimensions)
        self._names = None
        self._basenames = None
        self._hash = None
        self._num_dims = None
        self._size = None
        self._shape = None
        self._hashable_state = None
        self._coord = None
        self._name = name

    @property
    def name(self):
        """Name of the dimension"""
        return self._name


    def __repr__(self):
        previous_precision = np.get_printoptions()['precision']
        np.set_printoptions(precision=18)
        try:
            argstrs = [('%s=%r' %item) for item in
                       self.serializable_state.items()]
            r = '%s(%s)' %(self.__class__.__name__, ',\n    '.join(argstrs))
        finally:
            np.set_printoptions(precision=previous_precision)
        return r

    def __str__(self):
        b = ' x '.join(['%i (%s)'%(dim.num_bins, dim.name) for dim in self._dimensions])
        return '"%s":\n%s'%(self.name, b)

    def __pretty__(self, p, cycle):
        """Method used by the `pretty` library for formatting"""
        if cycle:
            p.text('%s(...)' % self.__class__.__name__)
        else:
            p.begin_group(4, '%s([' % self.__class__.__name__)
            for n, dim in enumerate(self):
                p.breakable()
                p.pretty(dim)
                if n < len(self)-1:
                    p.text(',')
            p.end_group(4, '])')

    def _repr_pretty_(self, p, cycle):
        """Method used by e.g. ipython/Jupyter for formatting"""
        return self.__pretty__(p, cycle)

    def __getstate__(self):
        """Method invoked during pickling"""
        return self.serializable_state

    def __setstate__(self, state):
        """Method invoked during unpickling"""
        self.__init__(**state)

    @property
    def _map_class(self):
        if self.__map_class is None:
            from pisa.core.map import Map  # pylint: disable=wrong-import-position
            self.__map_class = Map
        return self.__map_class

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
        jsons.to_json(self.serializable_state, filename=filename, **kwargs)

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
        """list of strings : names of each dimension contained"""
        if self._names is None:
            self._names = [dim.name for dim in self]
        return self._names

    @property
    def basenames(self):
        """List of binning names with prefixes and/or suffixes along with any
        number of possible separator characters removed. See function
        `basename` for detailed specifications."""
        if self._basenames is None:
            self._basenames = [b.basename for b in self]
        return self._basenames

    @property
    def basename_binning(self):
        """Identical binning but with dimensions named by their basenames.
        Note that the `tex` properties for the dimensions are not carried over
        into the new binning."""
        return MultiDimBinning(d.basename_binning for d in self)

    @property
    def finite_binning(self):
        """Identical binning but with infinities in bin edges replaced by
        largest/smallest floating-point numbers representable with the current
        pisa.FTYPE."""
        return MultiDimBinning(d.finite_binning for d in self)

    @property
    def dimensions(self):
        """tuple of OneDimBinning : each dimension's binning in a list"""
        return self._dimensions

    @property
    def dims(self):
        """tuple of OneDimBinning : shortcut for `dimensions`"""
        return self._dimensions

    def iterdims(self):
        """Iterator over contained `dimensions`, each a OneDimBinning"""
        return iter(self._dimensions)

    @property
    def num_dims(self):
        """int : number of dimensions"""
        if self._num_dims is None:
            self._num_dims = len(self._dimensions)
        return self._num_dims

    @property
    def shape(self):
        """tuple : shape of binning, akin to `nump.ndarray.shape`"""
        if self._shape is None:
            self._shape = tuple(b.num_bins for b in self._dimensions)
        return self._shape

    @property
    def size(self):
        """int : total number of bins"""
        if self._size is None:
            self._size = reduce(mul, self.shape)
        return self._size

    @property
    def coord(self):
        """namedtuple : coordinate for indexing into binning by dim names"""
        if self._coord is None:
            self._coord = namedtuple('coord', self.names)
        return self._coord

    @property
    def normalize_values(self):
        """bool : Normalize quantities' units prior to hashing"""
        nv = [dim.normalize_values for dim in self]
        if not all(x == nv[0] for x in nv):
            raise ValueError(
                'Contained dimensions have `normalize_values` both True and'
                ' False. Set `normalize_values` to either True or False on'
                ' this MultiDimBinning object to force consistency among'
                ' contained OneDimBinning objects.'
            )

    @normalize_values.setter
    def normalize_values(self, b):
        for dim in self:
            dim.normalize_values = b

    @property
    def serializable_state(self):
        """Attributes of the object that are stored to disk. Note that
        attributes may be returned as references to other objects, so to
        prevent external modification of those objects, the user must call
        deepcopy() separately on the returned OrderedDict.

        Returns
        -------
        state dict : OrderedDict
            can be passed to instantiate a new MultiDimBinning via
            `MultiDimBinning(**state)`

        """
        d = OrderedDict({'dimensions': [d.serializable_state for d in self]})
        d['name'] = self.name
        return d

    @property
    def hashable_state(self):
        """Everything necessary to fully describe this object's state. Note
        that objects may be returned by reference, so to prevent external
        modification, the user must call deepcopy() separately on the returned
        OrderedDict.

        Returns
        -------
        state : OrderedDict that can be passed to instantiate a new
            MultiDimBinning via MultiDimBinning(**state)

        """
        if self._hashable_state is None:
            state = OrderedDict()
            # TODO: Shouldn't order matter?
            #state['dimensions'] = [self[name]._hashable_state
            #                       for name in sorted(self.names)]
            state['dimensions'] = [d.hashable_state for d in self]
            state['name'] = self.name
            self._hashable_state = state
        return self._hashable_state

    @property
    def normalized_state(self):
        """OrderedDict containing normalized (base units, and rounded to
        appropriate precision) state attributes used for testing equality
        between two objects.

        Use `hashable_state` for faster equality checks and `normalized_state`
        for inspecting the contents of each state attribute pre-hashing
        """
        state = OrderedDict()
        state['dimensions'] = [d.normalized_state for d in self]
        return state

    @property
    def hash(self):
        """Unique hash value for this object"""
        if self._hash is None:
            self._hash = hash_obj(self.hashable_state)
        return self._hash

    def __hash__(self):
        return self.hash

    @property
    def edges_hash(self):
        """int : hash on the list of hashes for each dimension's edge values"""
        return hash_obj([d.edges_hash for d in self])

    @property
    def bin_edges(self):
        """Return a list of the contained dimensions' bin_edges that is
        compatible with the numpy.histogramdd `hist` argument."""
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
        """list : Return a list of the contained dimensions' units"""
        return [d.units for d in self]

    @units.setter
    def units(self, *args):
        """sequence or *args containing units for each contained dim"""
        self.ito(*args[0])

    @property
    def inbounds_criteria(self):
        """Return string boolean criteria indicating e.g. an event falls within
        the limits of the defined binning.

        This can be used for e.g. applying cuts to events.

        See Also
        --------
        pisa.core.events.keepEventsInBins

        """
        crit = '(%s)' %(' & '.join(dim.inbounds_criteria for dim in self))
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
        elif isinstance(dim, str):
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
        if isinstance(dims, (str, int)):
            dims = [dims]

        keep_idx = list(range(len(self)))
        for dim in dims:
            idx = self.index(dim)
            keep_idx.remove(idx)

        keep_dims = [deepcopy(self.dimensions[idx]) for idx in keep_idx]
        return MultiDimBinning(keep_dims)

    # TODO: add *args to handle positional indexing (?) (also would need to
    # add this to `slice` method if implemented.
    def indexer(self, **kwargs):
        """Any dimension index/slice not specified by name in kwargs will
        default to ":" (all elements).

        Parameters
        ---------
        **kwargs
            kwargs are names of dimension(s) and assigned to these are either
            an integer index into that dimension or a Python `slice` object for
            that dimension. See examples below for details.

        Returns
        -------
        indexer : tuple

        See Also
        --------
        broadcast
            Assignment of a one-dimensional array to a higher-dimensional array
            is simplified greatly by using `broadcast` in conjunction with
            `indexer` or `pisa.core.map.Map.slice`. See examples in
            docs for `broadcast`.
        broadcaster
            Similar to `broadcast`, but returns a tuple that can be applied to
            broadcast any one-dimensional array.
        slice
            Apply the `indexer` returned by this method to this MultiDimBinning
            object, returning a new MultiDimBinning object.
        pisa.core.map.Map.slice
            Same operation, but slices a Map object by dimension-name
            (internally, calls `indexer`).

        Examples
        --------
        >>> from pisa import ureg
        >>> from pisa.core.binning import MultiDimBinning, OneDimBinning
        >>> ebins = OneDimBinning(name='energy', is_log=True,
        ...                       num_bins=40, domain=[1, 80]*ureg.GeV)
        >>> czbins = OneDimBinning(name='coszen',
        ...                        is_lin=True, num_bins=4, domain=[-1, 0])
        >>> mdb = ebins * czbins
        >>> print(mdb.indexer(energy=0))
        (0, slice(None, None, None))

        Omitting a dimension (coszen in the above) is equivalent to slicing
        with a colon (i.e., `(0, slice(None))`):

        >>> print(mdb.indexer(energy=0, coszen=slice(None)))
        (0, slice(None, None, None))

        >>> print(mdb.indexer(energy=slice(None), coszen=1))
        (slice(None, None, None), 1)

        Now create an indexer to use on a Numpy array:

        >>> x = np.random.RandomState(0).uniform(size=mdb.shape)
        >>> indexer = mdb.indexer(energy=slice(0, 5), coszen=1)
        >>> print(x[indexer])
        [0.71518937 0.64589411 0.38344152 0.92559664 0.83261985]

        """
        indexer = []
        for dim in self.dims:
            if dim.name in kwargs:
                val = kwargs[dim.name]
                if isinstance(val, str):
                    val = dim.index(val)
                indexer.append(val)
            else:
                indexer.append(slice(None))
        return tuple(indexer)

    def slice(self, **kwargs):
        """Slice the binning by dimension name. Any dimension/index not
        specified by name in kwargs will default to ":" (all bins).

        Uses `indexer` internally to define the indexing tuple.

        Returns
        -------
        sliced_binning : MultiDimBinning

        """
        return self[self.indexer(**kwargs)]

    def broadcast(self, a, from_dim, to_dims):
        """Take a one-dimensional array representing one input dimension and
        broadcast it across some number of output dimensions.

        Parameters
        ----------
        a : 1D array
            Data from the `from_dim` dimension. `a` must have same length as
            the dimension it comes from (or Numpy must be able to automatically
            cast it into this dimension).

        from_dim : string
            Name of dimension that the data in `a` comes from.

        to_dims : string or iterable of strings
            Dimension(s) to cast `a` into.

        Returns
        -------
        a_broadcast : array
            Broadcast version of `a`

        See Also
        --------
        broadcaster
            The method used internally to derive the tuple used to broadcast
            the array. This can be used directly to return the broadcaster for
            use on other Maps or Numpy arrays.

        """
        assert isinstance(a, np.ndarray)
        a_shape = a.shape
        assert len(a_shape) == 1
        return a[self.broadcaster(from_dim=from_dim, to_dims=to_dims)]

    def broadcaster(self, from_dim, to_dims):
        """Generate an indexder that, if applied to a one-dimensional array
        representing data from one dimension, broadcasts that array into some
        number of other dimensions.

        Parameters
        ----------
        from_dim : string
            Name of dimension that the data in comes from.

        to_dims : string or iterable of strings
            Dimension(s) to cast into.

        Returns
        -------
        bcast : tuple
            Tuple that can be applied to a Numpy array for purposes of
            broadcasting it. E.g. use as `np.array([0,1,2])[bcast]`.

        """
        if isinstance(to_dims, str):
            to_dims = [to_dims]

        bcast = []
        for name in self.names:
            if name == from_dim:
                bcast.append(slice(None))
            elif name in to_dims:
                bcast.append(np.newaxis)

        return tuple(bcast)

    def iterbins(self):
        """Return an iterator over each N-dimensional bin. The elments returned
        by the iterator are each a MultiDimBinning, just containing a single
        bin.

        Returns
        -------
        bin_iterator

        See Also
        --------
        index2coord
            convert the (flat) index to multi-dimensional coordinate, which is
            useful when using e.g. `enumerate(iterbins)`

        """
        return (MultiDimBinning(dims) for dims in product(*self.dims))

    def iteredgetuples(self):
        """Return an iterator over each bin's edges. The elments returned by
        the iterator are a tuple of tuples, where the innermost tuples
        correspond to each dimension (in the order they're defined here).

        Units are stripped prior to iteration for purposes of speed.

        Note that this method is, according to one simple test, about 5000x
        faster than `iterbins`.

        Returns
        -------
        edges_iterator

        See Also
        --------
        iterbins
            Similar, but returns a OneDimBinning object for each bin. This is
            slower that `iteredgetuples` but easier to work with.

        """
        return product(*(dim.iteredgetuples() for dim in self.dims))

    def itercoords(self):
        """Return an iterator over each N-dimensional coordinate into the
        binning. The elments returned by the iterator are each a namedtuple,
        which can be used to directly index into the binning.

        Returns
        -------
        coord_iterator

        See Also
        --------
        iterbins
            Iterator over each bin
        index2coord
            convert the (flat) index to multi-dimensional coordinate, which is
            useful when using e.g. `enumerate(iterbins)`

        """
        return (self.index2coord(i) for i in range(self.size))

    def index2coord(self, index):
        """Convert a flat index into an N-dimensional bin coordinate.

        Useful in conjunction with `enumerate(iterbins)`

        Parameters
        ----------
        index : integer
            The flat index

        Returns
        -------
        coord : self.coord namedtuple
            Coordinates are in the same order as the binning is here defined
            and each coordinate is named by its corresponding dimension.
            Therefore integer indexing into `coord` as well as named indexing
            are possible.

        """
        coord = []
        quot = index
        for dim_length in self.shape[::-1]:
            quot, rem = divmod(quot, dim_length)
            coord.append(rem)
        return self.coord(*coord[::-1]) # pylint: disable=not-callable

    # TODO: examples!
    def reorder_dimensions(self, order, use_deepcopy=False,
                           use_basenames=False):
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
        >>> print(b2.binning.names)

        """
        if hasattr(order, 'binning') and isinstance(order.binning,
                                                    MultiDimBinning):
            order = order.binning.dims
        elif isinstance(order, MultiDimBinning):
            order = order.dims

        indices = []
        for dim in order:
            try:
                idx = self.index(dim, use_basenames=use_basenames)
            except ValueError:
                continue
            indices.append(idx)
        if set(indices) != set(range(len(self))):
            raise ValueError(
                'Invalid `order`: Only a subset of the dimensions present'
                ' were specified. `order`=%s, but dimensions=%s'
                %(order, self.names)
            )
        if use_deepcopy:
            new_dimensions = [deepcopy(self._dimensions[n]) for n in indices]
        else:
            new_dimensions = [self._dimensions[n] for n in indices]
        new_binning = MultiDimBinning(new_dimensions)
        return new_binning

    def is_compat(self, other):
        """Check if another binning is compatible with this binning.

        Note that for now, only downsampling is allowed from other to this, and
        not vice versa.

        Parameters
        ----------
        other : MultiDimBinning

        Returns
        -------
        is_compat : bool

        """
        if not set(self.names) == set(other.names):
            logging.trace('dimension names do not match')
            return False

        for name in self.names:
            if not self[name].is_compat(other[name]):
                return False

        return True

    def oversample(self, *args, **kwargs):
        """Return a MultiDimBinning object oversampled relative to this one.

        Parameters
        ----------
        *args : each factor an int
            Factors by which to oversample the binnings. There must either be
            one factor (one arg)--which will be broadcast to all dimensions--or
            there must be as many factors (args) as there are dimensions.
            If positional args are specified (i.e., non-kwargs), then kwargs
            are forbidden. For more detailed control, use keyword arguments to
            specify the dimension(s) to be oversampled and their factors.

        **kwargs : name=factor pairs
            Dimensions not specified default to oversample factor of 1 (i.e.,
            no oversampling)

        Returns
        -------
        new_binning : MultiDimBinning
            New binning, oversampled from the current binning.

        Notes
        -----
        You can either specify oversmapling by passing in args (ordered values,
        no keywords) or kwargs (order doesn't matter, but uses keywords), but
        not both.

        Specifying simple args (no keywords) requires either a single scalar
        (in which case all dimensions will be oversampled by the same factor)
        or one scalar per dimension (which oversamples the dimensions in the
        order specified).

        Specifying keyword args is far more explicit (and general), where each
        dimension's oversampling can be specified by name=factor pairs, but not
        every dimension must be specified (where no oversampling is applied to
        unspecified dimensions).

        See Also
        --------
        downsample
            Similar to this, but downsample the MultiDimBinning

        OneDimBinning.oversample
            Oversample a OneDimBinning object; this method is called to
            actually perform the oversampling for each dimension within this
            MultiDimBinning object

        OneDimBinning.downsample
            Same but downsample for OneDimBinning

        Examples
        --------
        >>> x = OneDimBinning('x', bin_edges=[0, 1, 2])
        >>> y = OneDimBinning('y', bin_edges=[0, 20])
        >>> mdb = x * y

        The following are all equivalent:

        >>> print(mdb.oversample(2))
        MultiDimBinning(
            OneDimBinning('x', 4 linearly-regular bins spanning [0.0, 2.0] (behavior is linear)),
            OneDimBinning('y', 2 linearly-regular bins spanning [0.0, 20.0] (behavior is linear))
        )
        >>> print(mdb.oversample(2, 2))
        MultiDimBinning(
            OneDimBinning('x', 4 linearly-regular bins spanning [0.0, 2.0] (behavior is linear)),
            OneDimBinning('y', 2 linearly-regular bins spanning [0.0, 20.0] (behavior is linear))
        )
        >>> print(mdb.oversample(x=2, y=2))
        MultiDimBinning(
            OneDimBinning('x', 4 linearly-regular bins spanning [0.0, 2.0] (behavior is linear)),
            OneDimBinning('y', 2 linearly-regular bins spanning [0.0, 20.0] (behavior is linear))
        )

        But with kwargs, you can specify only the dimensions you want to
        oversample, and the other dimension(s) remain unchanged:

        >>> print(mdb.oversample(y=5))
        MultiDimBinning(
            OneDimBinning('x', 2 linearly-regular bins spanning [0.0, 2.0] (behavior is linear)),
            OneDimBinning('y', 5 linearly-regular bins spanning [0.0, 20.0] (behavior is linear))
        )

        """
        if args:
            assert len(args) in [1, len(self)]
        elif kwargs:
            for name in self.names:
                if name not in kwargs:
                    kwargs[name] = 1
        factors = self._args_kwargs_to_list(*args, **kwargs)
        new_binning = [dim.oversample(f)
                       for dim, f in zip(self._dimensions, factors)]
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

        Returns
        -------
        new_binning : MultiDimBinning
            New binning, downsampled from the current binning.

        Notes
        -----
        Can either specify downsampling by passing in args (ordered values, no
        keywords) or kwargs (order doesn't matter, but uses keywords), but not
        both.

        See Also
        --------
        oversample
            Oversample (upsample) a the MultiDimBinning

        OneDimBinning.downsample
            The method actually called to perform the downsampling for each
            OneDimBinning within this MultiDimBinning object.

        OneDimBinning.oversample
            Same, but oversample (upsample) a OneDimBinning object

        """
        if args:
            assert len(args) in [1, len(self)]
        elif kwargs:
            for name in self.names:
                if name not in kwargs:
                    kwargs[name] = 1
        factors = self._args_kwargs_to_list(*args, **kwargs)
        new_binning = [dim.downsample(f)
                       for dim, f in zip(self._dimensions, factors)]
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

        Raises
        ------
        ValueError if array shape does not match the binning shape

        """
        if array.shape != self.shape:
            raise ValueError(
                'Array shape %s does not match binning shape %s'
                % (array.shape, self.shape)
            )

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
        return MultiDimBinning(d for d in self if len(d) > 1)

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
        for dim, units in zip(self.iterdims(), units_list):
            dim.ito(units)

    def to(self, *args, **kwargs): # pylint: disable=invalid-name
        """Convert the contained dimensions to the passed units. Unspecified
        dimensions will be omitted.

        """
        units_list = self._args_kwargs_to_list(*args, **kwargs)
        new_binnings = [dim.to(units)
                        for dim, units in zip(self.iterdims(), units_list)]
        return MultiDimBinning(new_binnings)

    def meshgrid(self, entity, attach_units=True):
        """Apply NumPy's meshgrid method on various entities of interest.

        Parameters
        ----------
        entity : string
            Can be any attribute of OneDimBinning that returns a 1D array with
            units. E.g., one of 'midpoints', 'weighted_centers', 'bin_edges',
            'bin_widths', or 'weighted_bin_widths'

        attach_units : bool
            Whether to attach units to the result (can save computation time by
            not doing so).

        Returns
        -------
        [X1, X2,..., XN] : list of numpy ndarray or Pint quantities of the same
            One ndarray or quantity is returned per dimension; see docs for
            `numpy.meshgrid` for details

        See Also
        --------
        numpy.meshgrid

        """
        entity = entity.lower().strip()

        arrays = []
        units = []
        for dim in self.iterdims():
            try:
                quantity_array = getattr(dim, entity)
            except AttributeError:
                logging.error(
                    "Dimension %s does not contain entity '%s'", dim.name, entity
                )
                raise
            if attach_units:
                units.append(quantity_array.units)
            arrays.append(quantity_array.magnitude)

        # NOTE: numpy versions prior to 1.13.0, meshgrid returned float64 even
        # if inputs are float32 to mesghrid. Use `astype` as a fix. Note that
        # since `astype` already creates a copy of the array even if dtype of
        # input is the same, setting `copy` to False is ok in the argument to
        # meshgrid; i.e., if a user modifies an element of the returned array,
        # it should not affect the original `entity` from which the meshgrid
        # was generated.
        mg = [a.astype(FTYPE) for a in np.meshgrid(*arrays, indexing='ij', copy=False)]

        if attach_units:
            return [m*u for m, u in zip(mg, units)]

        return mg

    # TODO: modify technique depending upon grid size for memory concerns, or
    # even take a `method` argument to force method manually.
    def bin_volumes(self, attach_units=True):
        """Bin "volumes" defined in `num_dims`-dimensions

        Parameters
        ----------
        attach_units : bool
            Whether to attach pint units to the resulting array

        Returns
        -------
        volumes : array
            Bin volumes

        """
        meshgrid = self.meshgrid(entity='bin_widths', attach_units=False)
        volumes = reduce(mul, meshgrid)
        if attach_units:
            volumes *= reduce(mul, (ureg(str(d.units)) for d in self.iterdims()))
        return volumes

    def weighted_bin_volumes(self, attach_units=True):
        """Bin "volumes" defined in `num_dims`-dimensions, but unlike
        `bin_volumes`, the volume is evaluated in the space of the binning.
        E.g., logarithmic bins have `weighted_bin_volumes` of equal size in
        log-space.

        Parameters
        ----------
        attach_units : bool
            Whether to attach pint units to the resulting array

        Returns
        -------
        volumes : array
            Bin volumes

        """
        meshgrid = self.meshgrid(entity='weighted_bin_widths', attach_units=False)
        volumes = reduce(mul, meshgrid)
        if attach_units:
            # NOTE we use the units from `weighted_bin_widths` because these
            # can be different from those of the dimension
            volumes *= reduce(
                mul,
                (ureg(str(d.weighted_bin_widths.units)) for d in self.iterdims()),
            )
        return volumes

    def empty(self, name, map_kw=None, **kwargs):
        """Return a Map whose hist is an "empty" numpy ndarray with same
        dimensions as this binning.

        The contents are not _actually_ empty, just undefined. Therefore be
        careful to populate the array prior to using its contents.

        Parameters
        ----------
        name : string
            Name of the Map

        map_kw : None or dict
            keyword arguments sent to instantiate the new Map (except `name`
            which is specified above)

        **kwargs
            keyword arguments passed on to numpy.empty() (except `shape` which
            must be omitted)

        Returns
        -------
        map : Map

        """
        assert 'shape' not in kwargs
        if map_kw is None:
            map_kw = {}
        if 'dtype' not in kwargs:
            kwargs['dtype'] = FTYPE
        hist = np.empty(self.shape, **kwargs)
        return self._map_class(name=name, hist=hist, binning=self, **map_kw) # pylint: disable=not-callable

    def zeros(self, name, map_kw=None, **kwargs):
        """Return a numpy ndarray filled with 0's with same dimensions as this
        binning.

        Parameters
        ----------
        name : string
            Name of the map

        map_kw : None or dict
            keyword arguments sent to instantiate the new Map (except `name`
            which is specified above)

        **kwargs
            keyword arguments passed on to numpy.zeros() (except `shape` which
            must be omitted)

        Returns
        -------
        map : Map

        """
        assert 'shape' not in kwargs
        if map_kw is None:
            map_kw = {}
        if 'dtype' not in kwargs:
            kwargs['dtype'] = FTYPE
        hist = np.zeros(self.shape, **kwargs)
        return self._map_class(name=name, hist=hist, binning=self, **map_kw) # pylint: disable=not-callable

    def ones(self, name, map_kw=None, **kwargs):
        """Return a numpy ndarray filled with 1's with same dimensions as this
        binning.

        Parameters
        ----------
        name : string
            Name of the map

        map_kw : None or dict
            keyword arguments sent to instantiate the new Map (except `name`
            which is specified above)

        **kwargs
            keyword arguments passed on to numpy.ones() (except `shape` which
            must be omitted)

        Returns
        -------
        map : Map

        """
        assert 'shape' not in kwargs
        if map_kw is None:
            map_kw = {}
        if 'dtype' not in kwargs:
            kwargs['dtype'] = FTYPE
        hist = np.ones(self.shape, **kwargs)
        return self._map_class(name=name, hist=hist, binning=self, **map_kw) # pylint: disable=not-callable

    def full(self, fill_value, name, map_kw=None, **kwargs):
        """Return a map whose `hist` is filled with `fill_value` of same
        dimensions as this binning.

        Parameters
        ----------
        fill_value
            Value with which to fill the map

        name : string
            Name of the map

        map_kw : None or dict
            keyword arguments sent to instantiate the new Map (except `name`
            which is specified above)

        **kwargs
            keyword arguments passed on to numpy.fill_value() (except `shape`,
            which must be omitted)

        Returns
        -------
        map : Map

        """
        assert 'shape' not in kwargs
        if map_kw is None:
            map_kw = {}
        if 'dtype' not in kwargs:
            kwargs['dtype'] = FTYPE
        hist = np.full(self.shape, fill_value, **kwargs)
        return self._map_class(name=name, hist=hist, binning=self, **map_kw) # pylint: disable=not-callable

    def __contains__(self, x):
        if isinstance(x, OneDimBinning):
            return x in self.dims
        if isinstance(x, str):
            return x in self.names
        return False

    def __eq__(self, other):
        if not isinstance(other, MultiDimBinning):
            return False
        return recursiveEquality(self.hashable_state, other.hashable_state)

    # TODO: remove this method, as it should just be considered an outer
    # product to increase dimensionality (i.e. the "*" operator, or __mul__
    # makes more sense than "+" or __add__)?
    def __add__(self, other):
        other = MultiDimBinning(other)
        return MultiDimBinning(chain(self, other))

    def __mul__(self, other):
        if isinstance(other, (Mapping, OneDimBinning)):
            other = [other]
        other = MultiDimBinning(other)
        return MultiDimBinning(chain(self, other))

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
            return super().__getattribute__(attr)

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
            If str is passed: Return the binning corresponding to the named
            dimension

            If an integer is passed:
              * If num_dims is 4, `index` indexes into the bins of the sole
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

        if isinstance(index, str):
            for d in self.iterdims():
                if d.name == index:
                    return d
            raise ValueError(f"index '{index}' not in {self.names}")

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
                                      zip(self.iterdims(), index)]}

        return MultiDimBinning(**new_binning)

    def __iter__(self):
        """Iterate over dimensions. Use `iterbins` to iterate over bins."""
        return iter(self._dimensions)

    def __len__(self):
        return self.num_dims

    def __ne__(self, other):
        return not self.__eq__(other)


def test_OneDimBinning():
    """Unit tests for OneDimBinning class"""
    # pylint: disable=line-too-long, wrong-import-position
    import pickle
    import os
    import shutil
    import tempfile
    # needed so that eval(repr(b)) works
    from numpy import array, float32, float64 # pylint: disable=unused-variable

    b1 = OneDimBinning(name='true_energy', num_bins=40, is_log=True,
                       domain=[1, 80]*ureg.GeV, tex=r'E_{\rm true}',
                       bin_names=[str(i) for i in range(40)])
    b2 = OneDimBinning(name='coszen', num_bins=40, is_lin=True,
                       domain=[-1, 1], bin_names=None,
                       tex=r'\cos\theta')
    b3 = OneDimBinning(name='reco_energy', num_bins=40, is_log=True,
                       domain=[1, 80]*ureg.GeV, tex=r'E_{\rm reco}',
                       bin_names=[str(i) for i in range(40)])

    # Test label
    _ = b1.label
    _ = b1.label

    assert b1.basename_binning == b1.basename_binning
    assert b1.basename_binning == b3.basename_binning
    assert b1.basename_binning != b2.basename_binning
    
    # Oversampling/downsampling
    b1_over = b1.oversample(2)
    assert b1_over.is_bin_spacing_log_uniform(b1_over.bin_edges)
    b1_down = b1.downsample(2)
    assert b1_down.is_bin_spacing_log_uniform(b1_down.bin_edges)
    assert b1_down.is_compat(b1)
    assert b1.is_compat(b1_over)
    assert b1_down.is_compat(b1_over)
    
    # Bin width consistency
    assert np.isclose(
        np.sum(b1_over.bin_widths.m),
        np.sum(b1.bin_widths.m),
        **ALLCLOSE_KW,
    )
    assert np.isclose(
        np.sum(b1_down.bin_widths.m),
        np.sum(b1.bin_widths.m),
        **ALLCLOSE_KW,
    )
    assert np.isclose(
        np.sum(b1_over.bin_widths.m),
        np.sum(b1_down.bin_widths.m),
        **ALLCLOSE_KW,
    )
    # Weighted bin widths must also sum up to the same total width
    assert np.isclose(
        np.sum(b1_over.weighted_bin_widths.m),
        np.sum(b1.weighted_bin_widths.m),
        **ALLCLOSE_KW,
    )
    assert np.isclose(
        np.sum(b1_down.weighted_bin_widths.m),
        np.sum(b1.weighted_bin_widths.m),
        **ALLCLOSE_KW,
    )
    assert np.isclose(
        np.sum(b1_over.weighted_bin_widths.m),
        np.sum(b1_down.weighted_bin_widths.m),
        **ALLCLOSE_KW,
    )
    
    logging.debug('len(b1): %s', len(b1))
    logging.debug('b1: %s', b1)
    logging.debug('b2: %s', b2)
    logging.debug('b1.oversample(10): %s', b1.oversample(10))
    logging.debug('b1.oversample(1): %s', b1.oversample(1))
    # Slicing
    logging.debug('b1[1:5]: %s', b1[1:5])
    logging.debug('b1[:]: %s', b1[:])
    logging.debug('b1[-1]: %s', b1[-1])
    logging.debug('b1[:-1]: %s', b1[:-1])
    logging.debug('copy(b1): %s', copy(b1))
    logging.debug('deepcopy(b1): %s', deepcopy(b1))
    # Indexing by Ellipsis
    assert b1[...] == b1
    # Pickling
    s = pickle.dumps(b1, pickle.HIGHEST_PROTOCOL)
    b1_loaded = pickle.loads(s)
    s = pickle.dumps(b1[0], pickle.HIGHEST_PROTOCOL)
    b1_loaded = pickle.loads(s)
    assert b1_loaded == b1[0]

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
    _ = hash_obj(b3)
    _ = b3.hash
    _ = hash(b3)
    _ = hash_obj(b3[0])
    _ = b3[0].hash # pylint: disable=no-member
    _ = hash(b3[0])

    b3.normalize_values = True
    b4.normalize_values = True

    _ = hash_obj(b3)
    _ = b3.hash
    _ = hash(b3)
    _ = hash_obj(b3[0])
    _ = b3[0].hash # pylint: disable=no-member
    _ = hash(b3[0])

    # Without rounding, converting bin edges to base units yields different
    # results due to finite precision effects
    assert np.any(normQuant(b3.bin_edges, sigfigs=None)
                  != normQuant(b4.bin_edges, sigfigs=None))

    # Normalize function should take care of this
    assert np.all(normQuant(b3.bin_edges, sigfigs=HASH_SIGFIGS, full_norm=True)
                  == normQuant(b4.bin_edges, sigfigs=HASH_SIGFIGS, full_norm=True)), \
            'normQuant(b3.bin_edges)=\n%s\nnormQuant(b4.bin_edges)=\n%s' \
            %(normQuant(b3.bin_edges, sigfigs=HASH_SIGFIGS, full_norm=True),
              normQuant(b4.bin_edges, sigfigs=HASH_SIGFIGS, full_norm=True))

    # And the hashes should be equal, reflecting the latter result
    assert b3.hash == b4.hash, \
            '\nb3=%s\nb4=%s' % (b3.hashable_state, b4.hashable_state)
    assert b3.hash == b4.hash, 'b3.hash=%s; b4.hash=%s' %(b3.hash, b4.hash)

    s = pickle.dumps(b3, pickle.HIGHEST_PROTOCOL)
    b3_loaded = pickle.loads(s)
    assert b3_loaded == b3

    testdir = tempfile.mkdtemp()
    try:
        for b in [b1, b2, b3, b4]:
            assert eval(repr(b)) == b, repr(b) # pylint: disable=eval-used
            b_file = os.path.join(testdir, 'one_dim_binning.json')
            b.to_json(b_file, warn=False)
            b_ = OneDimBinning.from_json(b_file)
            assert b_ == b, 'b=\n%s\nb_=\n%s' %(b, b_)
            jsons.to_json(b, b_file, warn=False)
            b_ = OneDimBinning.from_json(b_file)
            assert b_ == b, 'b=\n%s\nb_=\n%s' %(b, b_)

            # Had bug where datastruct containing MultiDimBinning failed to be
            # saved. # Test tuple containing list containing OrderedDict
            # containing OneDimBinning here.
            struct = ([OrderedDict(odb=b)],)
            jsons.to_json(struct, b_file, warn=False)
            loaded = jsons.from_json(b_file)
            b_ = OneDimBinning(**loaded[0][0]['odb'])
            assert b_ == b

            # Now try with pickle
            b_file = os.path.join(testdir, 'one_dim_binning.pkl')
            with open(b_file, 'wb') as fobj:
                pickle.dump(struct, fobj, protocol=pickle.HIGHEST_PROTOCOL)
            with open(b_file, 'rb') as fobj:
                loaded = pickle.load(fobj)
            b_ = loaded[0][0]['odb']
            assert b_ == b

    except:
        logging.error('b that failed: %s', b)
        raise
    finally:
        shutil.rmtree(testdir, ignore_errors=True)

    logging.info('<< PASS : test_OneDimBinning >>')


def test_MultiDimBinning():
    """Unit tests for MultiDimBinning class"""
    # pylint: disable=wrong-import-position
    import pickle
    import os
    import shutil
    import tempfile
    import time
    # needed so that eval(repr(mdb)) works
    from numpy import array, float32, float64 # pylint: disable=unused-variable

    b1 = OneDimBinning(name='energy', num_bins=40, is_log=True,
                       domain=[1, 80]*ureg.GeV)
    b2 = OneDimBinning(name='coszen', num_bins=40, is_lin=True,
                       domain=[-1, 1])
    mdb = MultiDimBinning([b1, b2])

    assert eval(repr(mdb)) == mdb # pylint: disable=eval-used

    _ = hash_obj(mdb)
    _ = mdb.hash
    _ = hash(mdb)
    _ = hash_obj(mdb[0, 0])
    _ = mdb[0, 0].hash
    _ = hash(mdb[0, 0])
    _ = mdb[0, 0]
    _ = mdb[0:, 0]
    _ = mdb[0:, 0:]
    _ = mdb[0, 0:]
    _ = mdb[-1, -1]
    # TODO: following should work as in Numpy:
    # assert mdb[:] == mdb
    # assert mdb[0] == b1
    # assert mdb[1] == b2
    assert mdb[:, :] == mdb

    # Index by dim names
    assert mdb["energy"] == b1
    assert mdb["coszen"] == b2
    try:
        mdb["nonexistent"]
    except Exception:
        pass
    else:
        raise Exception('non-existent name should raise exception')

    # Index by dim number
    assert MultiDimBinning([b1])[0] == MultiDimBinning([b1[0]])

    logging.debug('%s', mdb.energy)
    logging.debug('copy(mdb): %s', copy(mdb))
    logging.debug('deepcopy(mdb): %s', deepcopy(mdb))
    assert deepcopy(mdb) == mdb

    s = pickle.dumps(mdb, pickle.HIGHEST_PROTOCOL)
    mdb2 = pickle.loads(s)
    assert mdb2 == mdb

    s = pickle.dumps(mdb[0, 0], pickle.HIGHEST_PROTOCOL)
    mdb2 = pickle.loads(s)
    assert mdb2 == mdb[0, 0]

    binning = MultiDimBinning([
        dict(name='energy', is_log=True, domain=[1, 80]*ureg.GeV, num_bins=40),
        dict(name='coszen', is_lin=True, domain=[-1, 0], num_bins=20)
    ])

    ord_dict_of_binnings = OrderedDict([('x', mdb), ('y', binning)])
    _ = hash_obj(ord_dict_of_binnings)
    _ = normQuant(ord_dict_of_binnings)

    for flatindex, this_bin in enumerate(binning.iterbins()):
        coord = binning.index2coord(flatindex)
        assert this_bin == binning[coord]

    assert binning.num_bins == [40, 20]
    assert binning.tot_num_bins == 40 * 20

    assert binning.oversample(10).shape == (400, 200)

    assert binning.oversample(10, 1).shape == (400, 20)
    assert binning.oversample(1, 3).shape == (40, 60)
    assert binning.downsample(4, 2).shape == (10, 10)
    
    assert binning.oversample(coszen=10, energy=2).shape == (80, 200)
    assert binning.oversample(1, 1) == binning

    assert binning.to('MeV', '')['energy'].units == ureg.MeV
    assert binning.to('MeV', '') == binning, 'converted=%s\norig=%s' \
            %(binning.to('MeV', ''), binning)
    assert binning.to('MeV', '').hash == binning.hash

    _ = binning.meshgrid(entity='bin_edges')
    _ = binning.meshgrid(entity='weighted_centers')
    _ = binning.meshgrid(entity='midpoints')
    _ = binning.meshgrid(entity='bin_widths')
    _ = binning.meshgrid(entity='weighted_bin_widths')
    _ = binning.bin_volumes(attach_units=False)
    _ = binning.bin_volumes(attach_units=True)
    _ = binning.weighted_bin_volumes(attach_units=False)
    _ = binning.weighted_bin_volumes(attach_units=True)
    binning.to('MeV', None)
    binning.to('MeV', '')
    binning.to(ureg.joule, '')
    
    oversampled = binning.oversample(10, 3)
    assert oversampled.shape == (400, 60)
    downsampled = binning.downsample(4, 2)
    assert downsampled.shape == (10, 10)
    
    over_vols = oversampled.bin_volumes(attach_units=False)
    down_vols = downsampled.bin_volumes(attach_units=False)
    norm_vols = binning.bin_volumes(attach_units=False)
    assert np.isclose(np.sum(over_vols), np.sum(norm_vols), **ALLCLOSE_KW)
    assert np.isclose(np.sum(down_vols), np.sum(norm_vols), **ALLCLOSE_KW)
    assert np.isclose(np.sum(down_vols), np.sum(over_vols), **ALLCLOSE_KW)
    
    over_vols = oversampled.weighted_bin_volumes(attach_units=False)
    down_vols = downsampled.weighted_bin_volumes(attach_units=False)
    norm_vols = binning.weighted_bin_volumes(attach_units=False)
    assert np.isclose(np.sum(over_vols), np.sum(norm_vols), **ALLCLOSE_KW)
    assert np.isclose(np.sum(down_vols), np.sum(norm_vols), **ALLCLOSE_KW)
    assert np.isclose(np.sum(down_vols), np.sum(over_vols), **ALLCLOSE_KW)
    
    testdir = tempfile.mkdtemp()
    try:
        b_file = os.path.join(testdir, 'multi_dim_binning.json')
        binning.to_json(b_file, warn=False)
        b_ = MultiDimBinning.from_json(b_file)
        assert b_ == binning, 'binning=\n%s\nb_=\n%s' %(binning, b_)
        jsons.to_json(binning, b_file, warn=False)
        b_ = MultiDimBinning.from_json(b_file)
        assert b_ == binning, 'binning=\n%s\nb_=\n%s' %(binning, b_)

        # Had bug where datastruct containing MultiDimBinning failed to be
        # saved. Test tuple containing list containing OrderedDict
        # containing MultiDimBinning here, just to make sure MultiDimBinning
        # can be written inside a nested structure.
        b = binning
        struct = ([OrderedDict(mdb=b)],)
        jsons.to_json(struct, b_file, warn=False)
        loaded = jsons.from_json(b_file)
        b_ = MultiDimBinning(**loaded[0][0]['mdb'])
        assert b_ == b

        # Now try with pickle
        b_file = os.path.join(testdir, 'multi_dim_binning.pkl')
        with open(b_file, 'wb') as fobj:
            pickle.dump(struct, fobj, protocol=pickle.HIGHEST_PROTOCOL)
        with open(b_file, 'rb') as fobj:
            loaded = pickle.load(fobj)
        b_ = loaded[0][0]['mdb']
        assert b_ == b

    finally:
        shutil.rmtree(testdir, ignore_errors=True)

    # Test that reordering dimensions works correctly
    e_binning = OneDimBinning(
        name='true_energy', num_bins=40, is_log=True, domain=[1, 80]*ureg.GeV
    )
    reco_e_binning = OneDimBinning(
        name='reco_energy', num_bins=40, is_log=True, domain=[1, 80]*ureg.GeV
    )
    cz_binning = OneDimBinning(
        name='true_coszen', num_bins=40, is_lin=True, domain=[-1, 1]
    )
    reco_cz_binning = OneDimBinning(
        name='reco_coszen', num_bins=40, is_lin=True, domain=[-1, 1]
    )
    az_binning = OneDimBinning(
        name='true_azimuth', num_bins=10, is_lin=True,
        domain=[0*ureg.rad, 2*np.pi*ureg.rad]
    )
    reco_az_binning = OneDimBinning(
        name='true_azimuth', num_bins=10, is_lin=True,
        domain=[0*ureg.rad, 2*np.pi*ureg.rad]
    )

    mdb_2d_orig = MultiDimBinning([e_binning, cz_binning])
    orig_order = mdb_2d_orig.names

    # Reverse ordering; reorder by dimension names
    new_order = orig_order[::-1]
    mdb_2d_new = MultiDimBinning(mdb_2d_orig)
    mdb_2d_new = mdb_2d_new.reorder_dimensions(new_order)

    assert mdb_2d_new.names == new_order
    new_order = ['true_azimuth', 'true_energy', 'true_coszen']
    mdb_2d_new = mdb_2d_new.reorder_dimensions(new_order)
    assert mdb_2d_new == mdb_2d_orig
    _ = MultiDimBinning([e_binning, cz_binning])

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
        [2, 'true_energy', mdb_2d_orig.dimensions[1]]
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

    # Create a basename-equivalent binning
    mdb_3d_reco = MultiDimBinning([reco_e_binning, reco_cz_binning,
                                   reco_az_binning])
    assert mdb_3d_reco.basename_binning == mdb_3d_orig.basename_binning

    t0 = time.time()
    _ = [tup for tup in mdb_3d_reco.iteredgetuples()]
    tprofile.info('Time to iterate over %d edge tuples: %.6f sec',
                  mdb_3d_reco.size, time.time() - t0)

    t0 = time.time()
    _ = [b for b in mdb_3d_reco.iterbins()]
    tprofile.info('Time to iterate over %d bins: %.6f sec',
                  mdb_3d_reco.size, time.time() - t0)

    logging.info('<< PASS : test_MultiDimBinning >>')


if __name__ == "__main__":
    set_verbosity(1)
    test_OneDimBinning()
    test_MultiDimBinning()
