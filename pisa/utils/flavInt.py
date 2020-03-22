#! /usr/bin/env python
# pylint: disable=global-statement

"""
Classes for working with neutrino flavors (NuFlav), interactions types
(IntType), "flavints" (a flavor and an interaction type) (NuFlavInt), and
flavint groups (NuFlavIntGroup) in a consistent and convenient manner.

FlavIntData class for working with data stored by flavint (flavor &
interaction type). This should replace the PISA convention of using raw
doubly-nested dictionaries indexed as [<flavor>][<interaction type>]. For
now, FlavIntData objects can be drop-in replacements for such dictionaries
(they can be accessed and written to in the same way since FlavIntData
subclasses dict) but this should be deprecated; eventually, all direct access
of the data structure should be eliminated and disallowed by the FlavIntData
object.

Define convenience tuples ALL_{x} for easy iteration
"""


# TODO: Make strings convertible to various types less liberal. E.g., I already
# converted NuFlav to NOT accept 'numucc' such that things like 'numu nue' or
# 'nu xyz mutation' would also be rejected; this should be true also for
# interaction type and possibly others I haven't thought about yet. Note that I
# achieved this using the IGNORE regex that ignores all non-alpha characters
# but asserts one and only one match to the regex (consult NuFlav for details).

# TODO: make simple_str() method convertible back to NuFlavIntGroup, either by
# increasing the intelligence of interpret(), by modifying what simple_str()
# produces, or by adding another function to interpret simple strings. (I'm
# leaning towards the second option at the moment, since I don't see how to
# make the first interpret both a simple_str AND nue as nuecc+nuenc, and I
# don't think there's a way to know "this is a simple str" vs not easily.)


from __future__ import absolute_import, division

from collections.abc import MutableSequence, MutableMapping, Mapping, Sequence
from copy import deepcopy
from functools import reduce, total_ordering
from itertools import product, combinations
from operator import add
import re

import numpy as np

from pisa import ureg
from pisa.utils import fileio
from pisa.utils.log import logging, set_verbosity
from pisa.utils.comparisons import recursiveAllclose, recursiveEquality


__all__ = [
    'NuFlav', 'NuFlavInt', 'NuFlavIntGroup', 'FlavIntData',
    'FlavIntDataGroup', 'xlateGroupsStr',
    'flavintGroupsFromString', 'IntType', 'BarSep', 'set_bar_ssep',
    'get_bar_ssep', 'ALL_NUPARTICLES', 'ALL_NUANTIPARTICLES',
    'ALL_NUFLAVS', 'ALL_NUINT_TYPES', 'CC', 'NC',
    'NUE', 'NUEBAR', 'NUMU', 'NUMUBAR', 'NUTAU', 'NUTAUBAR',
    'ALL_NUFLAVINTS', 'ALL_NUCC', 'ALL_NUNC',
    'NUECC', 'NUEBARCC', 'NUMUCC', 'NUMUBARCC', 'NUTAUCC', 'NUTAUBARCC',
    'NUENC', 'NUEBARNC', 'NUMUNC', 'NUMUBARNC', 'NUTAUNC', 'NUTAUBARNC',
]

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

__BAR_SSEP__ = ''


class BarSep(object):
    """
    Context manager to make global __BAR_SSEP__ modification slightly less
    error-prone.

    __BAR_SSEP__ defines the separator between a flavor and the string "bar"
    (to define the flavor as the antiparticle version; e.g. nuebar). Some
    datastructures in PISA 2 used '_' between the two ("nue_bar") while others
    did not use this. To make dealing with this (slightly) less painful, this
    context manager was introduced to switch between the PISA 3 default (no
    '_') and the sometimes-use '_' separator. See Examples for usage.

    Parameters
    ----------
    val : string
        Separator to use between flavor ("nue", "numu", "nutau") and "bar".

    Examples
    --------
    >>> nuebar = NuFlav('nuebar')
    >>> print(str(nuebar))
    nuebar
    >>> with BarSep('_'):
    ...     print(nuebar)
    nue_bar
    >>> print(str(nuebar))
    nuebar

    """
    def __init__(self, val):
        global __BAR_SSEP__
        self.old_val = __BAR_SSEP__
        self.new_val = val

    def __enter__(self):
        global __BAR_SSEP__
        __BAR_SSEP__ = self.new_val

    def __exit__(self, type, value, traceback):
        global __BAR_SSEP__
        __BAR_SSEP__ = self.old_val


def set_bar_ssep(val):
    """Set the separator between "base" flavor ("nue", "numu", or "nutau") and
    "bar" when converting antineutrino `NuFlav`s or `NuFlavInt`s to strings.

    Parameters
    ----------
    val : string
        Separator

    """
    global __BAR_SSEP__
    assert isinstance(val, str)
    __BAR_SSEP__ = val


def get_bar_ssep():
    """Get the separator that is set to be used between "base" flavor ("nue",
    "numu", or "nutau") and "bar" when converting antineutrino `NuFlav`s or
    `NuFlavInt`s to strings.

    Returns
    -------
    sep : string
        Separator

    """
    global __BAR_SSEP__
    return __BAR_SSEP__


@total_ordering
class NuFlav(object):
    """Class for handling neutrino flavors (and anti-flavors)"""
    PART_CODE = 1
    ANTIPART_CODE = -1
    NUE_CODE = 12
    NUMU_CODE = 14
    NUTAU_CODE = 16
    NUEBAR_CODE = -12
    NUMUBAR_CODE = -14
    NUTAUBAR_CODE = -16
    IGNORE = re.compile(r'[^a-zA-Z]')
    FLAV_RE = re.compile(
        r'^(?P<fullflav>(?:nue|numu|nutau)(?P<barnobar>bar){0,1})$'
    )
    def __init__(self, val):
        self.fstr2code = {
            'nue': self.NUE_CODE,
            'numu': self.NUMU_CODE,
            'nutau': self.NUTAU_CODE,
            'nuebar': self.NUEBAR_CODE,
            'numubar': self.NUMUBAR_CODE,
            'nutaubar': self.NUTAUBAR_CODE
        }
        self.barnobar2code = {
            None: self.PART_CODE,
            '': self.PART_CODE,
            'bar': self.ANTIPART_CODE,
        }
        self.f2tex = {
            self.NUE_CODE: r'{\nu_e}',
            self.NUMU_CODE: r'{\nu_\mu}',
            self.NUTAU_CODE: r'{\nu_\tau}',
            self.NUEBAR_CODE: r'{\bar\nu_e}',
            self.NUMUBAR_CODE: r'{\bar\nu_\mu}',
            self.NUTAUBAR_CODE: r'{\bar\nu_\tau}',
        }
        # Instantiate this neutrino flavor object by interpreting val
        orig_val = val
        try:
            if isinstance(val, str):
                # Sanitize the string
                sanitized_val = self.IGNORE.sub('', val.lower())
                matches = self.FLAV_RE.findall(sanitized_val)
                if len(matches) != 1:
                    raise ValueError('Invalid NuFlav spec: "%s"' % val)
                self.__flav = self.fstr2code[matches[0][0]]
                self.__barnobar = self.barnobar2code[matches[0][1]]
            elif isinstance(val, self.__class__):
                self.__flav = val.code
                self.__barnobar = np.sign(self.__flav)
            elif hasattr(val, 'flav'):
                self.__flav = val.flav.code
                self.__barnobar = np.sign(self.__flav)
            else:
                if val in self.fstr2code.values():
                    self.__flav = int(val)
                    self.__barnobar = np.sign(self.__flav)
                else:
                    raise ValueError('Invalid neutrino flavor/code: "%s"' %
                                     str(val))
            # Double check than flav and barnobar codes are valid
            assert self.__flav in self.fstr2code.values()
            assert self.__barnobar in self.barnobar2code.values()
        except (AssertionError, ValueError, AttributeError):
            raise ValueError('Could not interpret `val` = "%s" as %s.'
                             ' type(val) = %s'
                             % (orig_val, self.__class__.__name__,
                                type(orig_val)))

    def __str__(self):
        global __BAR_SSEP__
        fstr = [s for s, code in self.fstr2code.items() if code == self.__flav]
        fstr = fstr[0]
        fstr = fstr.replace('bar', __BAR_SSEP__+'bar')
        return fstr

    # TODO: copy, deepcopy, and JSON serialization
    #def __copy__(self):
    #    return self.__str__()

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__flav)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            try:
                other = self.__class__(other)
            except Exception:
                return False
        return other.code == self.code

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            try:
                other = self.__class__(other)
            except:
                raise ValueError('Cannot compare %s to %s.'
                                 % (other.__class__.__name__,
                                    self.__class__.__name__))

        my_abs_code = np.abs(self.code)
        other_abs_code = np.abs(other.code)

        # A particle comes before its own antiparticle
        if other_abs_code == my_abs_code:
            return self.code > other.code

        # nue or nuebar < numu or numubar < nutau or nutaubar
        return my_abs_code < other_abs_code

    def __neg__(self):
        return NuFlav(self.__flav*-1)

    def __add__(self, other):
        return NuFlavIntGroup(self, other)

    @property
    def tex(self):
        """TeX string"""
        return self.f2tex[self.__flav]

    @property
    def code(self):
        """int : PDG code"""
        return self.__flav

    @property
    def bar_code(self):
        """Return +/-1 for particle/antiparticle"""
        return self.__barnobar

    @property
    def particle(self):
        """Is this a particle (vs. antiparticle) flavor?"""
        return self.__barnobar == self.PART_CODE

    @property
    def antiparticle(self):
        """Is this an antiparticle flavor?"""
        return self.__barnobar == self.ANTIPART_CODE

    def pidx(self, d, *args):
        """Extract data from a nested dictionary `d` whose format is commonly
        found in PISA

        The dictionary must have the format
            d = {"<flavor>": <data object>}
            <flavor> is one of "nue", "nue_bar", "numu", "numu_bar", "nutau",
                "nutau_bar"
        """
        with BarSep('_'):
            field = d[str(self)]
        for idx in args:
            field = field[idx]
        return field

    @property
    def prob3_codes(self):
        """(int,int) : flavor and particle/antiparticle codes, as used by prob3"""
        if np.abs(self.code) == self.NUE_CODE : prob3flav = 0
        elif np.abs(self.code) == self.NUMU_CODE : prob3flav = 1
        elif np.abs(self.code) == self.NUTAU_CODE : prob3flav = 2
        prob3bar = self.bar_code
        return (prob3flav,prob3bar)


NUE = NuFlav('nue')
NUEBAR = NuFlav('nuebar')
NUMU = NuFlav('numu')
NUMUBAR = NuFlav('numubar')
NUTAU = NuFlav('nutau')
NUTAUBAR = NuFlav('nutaubar')

ALL_NUPARTICLES = (NUE, NUMU, NUTAU)
ALL_NUANTIPARTICLES = (NUEBAR, NUMUBAR, NUTAUBAR)
ALL_NUFLAVS = tuple(sorted(ALL_NUPARTICLES + ALL_NUANTIPARTICLES))


# TODO: are the following two classes redundant now?
class AllNu(object):
    def __init__(self):
        self.__flav = [p for p in ALL_NUPARTICLES]

    @property
    def flav(self):
        return self.__flav

    def __str__(self):
        return 'nuall'

    @property
    def tex(self):
        return r'{\nu_{\rm all}}'


class AllNuBar(object):
    def __init__(self):
        self.__flav = [p for p in ALL_NUANTIPARTICLES]

    @property
    def flav(self):
        return self.__flav

    def __str__(self):
        return 'nuallbar'

    @property
    def tex(self):
        return r'{\bar\nu_{\rm all}}'


@total_ordering
class IntType(object):
    """
    Interaction type object.

    Parameters
    ----------
    val
        See Notes.

    Notes
    -----
    Instantiate via a `val` of:
      * Numerical code: 1=CC, 2=NC
      * String (case-insensitive; all characters besides valid tokens are
        ignored)
      * Instantiated IntType object (or any method implementing int_type.code
        which returns a valid interaction type code)
      * Instantiated NuFlavInt object (or any object implementing int_type
        which returns a valid IntType object)

    Examples
    --------
    The following, e.g., are all interpreted as charged-current IntTypes:

    >>> IntType('cc')
    >>> IntType('\n\t _cc \n')
    >>> IntType('numubarcc')
    >>> IntType(1)
    >>> IntType(1.0)
    >>> IntType(IntType('cc'))
    >>> IntType(NuFlavInt('numubarcc'))

    """
    CC_CODE = 1
    NC_CODE = 2
    IGNORE = re.compile(r'[^a-zA-Z]')
    IT_RE = re.compile(r'^(cc|nc)$')
    def __init__(self, val):
        self.istr2code = {
            'cc': self.CC_CODE,
            'nc': self.NC_CODE,
        }
        self.i2tex = {
            self.CC_CODE: r'{\rm CC}',
            self.NC_CODE: r'{\rm NC}'
        }

        # Interpret `val`
        try:
            orig_val = val
            if isinstance(val, str):
                sanitized_val = self.IGNORE.sub('', val.lower())
                int_type = self.IT_RE.findall(sanitized_val)
                if len(int_type) != 1:
                    raise ValueError('Invalid IntType spec: "%s"' % val)
                self.__int_type = self.istr2code[int_type[0]]
            elif isinstance(val, self.__class__):
                self.__int_type = val.code
            elif hasattr(val, 'int_type') and hasattr(val.int_type, 'code'):
                self.__int_type = val.int_type.code
            else:
                if val in self.istr2code.values():
                    self.__int_type = int(val)
                else:
                    raise TypeError(
                        '`val` = "%s" could not be converted to an %s;'
                        ' type(val) = %s'
                        % (orig_val, self.__class__.__name__, type(orig_val))
                    )
            # Double check that the interaction type code set is valid
            assert self.__int_type in self.istr2code.values()
        except (AssertionError, TypeError, ValueError, AttributeError):
            raise ValueError('Could not interpret `val` = "%s" as %s;'
                             ' type(val) = %s'
                             % (orig_val, self.__class__.__name__,
                                type(orig_val)))

    def __str__(self):
        return [s for s, code in self.istr2code.items()
                if code == self.__int_type][0]

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__int_type)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            try:
                other = self.__class__(other)
            except Exception:
                return False
        return other.cc == self.cc

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            try:
                other = self.__class__(other)
            except:
                raise ValueError('Cannot compare %s to %s.'
                                 % (other.__class__.__name__,
                                    self.__class__.__name__))
        return self.code < other.code

    @property
    def cc(self): # pylint: disable=invalid-name
        """Is this interaction type charged current (CC)?"""
        return self.__int_type == self.CC_CODE

    @property
    def nc(self): # pylint: disable=invalid-name
        """Is this interaction type neutral current (NC)?"""
        return self.__int_type == self.NC_CODE

    @property
    def code(self):
        """Integer code for this interaction type"""
        return self.__int_type

    @property
    def tex(self):
        """TeX representation of this interaction type"""
        return self.i2tex[self.__int_type]


CC = IntType('cc')
NC = IntType('nc')
ALL_NUINT_TYPES = (CC, NC)


@total_ordering
class NuFlavInt(object):
    """A neutrino "flavint" encompasses both the neutrino flavor and its
    interaction type.

    Instantiate via
      * String containing a single flavor and a single interaction type
        e.g.: 'numucc', 'nu_mu_cc', 'nu mu CC', 'numu_bar CC', etc.
      * Another instantiated NuFlavInt object
      * Two separate objects that can be converted to a valid NuFlav
        and a valid IntType (in that order)
      * An iterable of length two which contains such objects
      * kwargs `flav` and `int_type` specifying such objects

    String specifications simply ignore all characters not recognized as a
    valid token.

    """
    TOKENS = re.compile('(nu|e|mu|tau|bar|nc|cc)')
    FINT_RE = re.compile(
        r'(?P<fullflav>(?:nue|numu|nutau)'
        r'(?P<barnobar>bar){0,1})'
        r'(?P<int_type>cc|nc){0,1}'
    )
    FINT_SSEP = '_'
    FINT_TEXSEP = r' \, '
    # TODO: use multiple inheritance to clean up the below?
    def __init__(self, *args, **kwargs):
        if kwargs:
            if args:
                raise TypeError('Either positional or keyword args may be'
                                ' provided, but not both')
            keys = kwargs.keys()
            if set(keys).difference(set(('flav', 'int_type'))):
                raise TypeError('Invalid kwarg(s) specified: %s' %
                                kwargs.keys())
            flavint = (kwargs['flav'], kwargs['int_type'])
        elif args:
            if not args:
                raise TypeError('No flavint specification provided')
            elif len(args) == 1:
                flavint = args[0]
            elif len(args) == 2:
                flavint = args
            elif len(args) > 2:
                raise TypeError('More than two args')

        if not isinstance(flavint, str) \
                and hasattr(flavint, '__len__') and len(flavint) == 1:
            flavint = flavint[0]

        if isinstance(flavint, str):
            orig_flavint = flavint
            try:
                flavint = ''.join(self.TOKENS.findall(flavint.lower()))
                flavint_dict = self.FINT_RE.match(flavint).groupdict()
                self.__flav = NuFlav(flavint_dict['fullflav'])
                self.__int_type = IntType(flavint_dict['int_type'])
            except (TypeError, UnboundLocalError, ValueError, AttributeError):
                raise ValueError('Could not interpret `val` = "%s" as %s;'
                                 ' type(val) = %s'
                                 % (orig_flavint, self.__class__.__name__,
                                    type(orig_flavint)))
        elif isinstance(flavint, NuFlavInt):
            self.__flav = NuFlav(flavint.flav)
            self.__int_type = IntType(flavint.int_type.code)
        elif hasattr(flavint, '__len__'):
            assert len(flavint) == 2, \
                    'Need 2 components to define flavor and interaction type'
            self.__flav = NuFlav(flavint[0])
            self.__int_type = IntType(flavint[1])
        else:
            raise TypeError('Unhandled type: "' + str(type(flavint)) +
                            '"; class: "' + str(flavint.__class__) +
                            '; value: "' + str(flavint) + '"')

    def __str__(self):
        return '%s%s%s' % (self.flav, self.FINT_SSEP, self.int_type)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.flav.code, self.int_type.code))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            try:
                other = self.__class__(other)
            except Exception:
                return False
        return (other.flav, other.int_type) == (self.flav, self.int_type)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            try:
                other = self.__class__(other)
            except:
                raise ValueError('Cannot compare %s to %s.'
                                 % (other.__class__.__name__,
                                    self.__class__.__name__))
        if self.int_type.code < other.int_type.code:
            return True
        if self.int_type.code == other.int_type.code:
            if np.abs(self.flav.code) == np.abs(other.flav.code):
                return self.flav.code > other.flav.code
            if np.abs(self.flav.code) < np.abs(other.flav.code):
                return True
        return False

    def __neg__(self):
        return NuFlavInt(-self.__flav, self.__int_type)

    def __add__(self, other):
        return NuFlavIntGroup(self, other)

    def pidx(self, d, *args):
        """Extract data from a nested dictionary `d` whose format is commonly
        found in PISA

        The dictionary must have the format::

            d = {"<flavor>": {"<interaction type>": <data object>}}
            <flavor> is one of "nue", "nue_bar", "numu", "numu_bar", "nutau",
                "nutau_bar"
            <interaction type> is one of "cc", "nc"

        """

        with BarSep('_'):
            field = d[str(self.flav)][str(self.int_type)]
        for idx in args:
            field = field[idx]
        return field

    @property
    def flav(self):
        """Return just the NuFlav part of this NuFlavInt"""
        return self.__flav

    @property
    def particle(self):
        """Is this a particle (vs. antiparticle) flavor?"""
        return self.__flav.particle

    @property
    def antiparticle(self):
        """Is this an antiparticle flavor?"""
        return self.__flav.antiparticle

    @property
    def cc(self): # pylint: disable=invalid-name
        """Is this interaction type charged current (CC)?"""
        return self.__int_type.cc

    @property
    def nc(self): # pylint: disable=invalid-name
        """Is this interaction type neutral current (NC)?"""
        return self.__int_type.nc

    @property
    def int_type(self):
        """Return IntType object that composes this NuFlavInt"""
        return self.__int_type

    @property
    def tex(self):
        """TeX string representation of this NuFlavInt"""
        return '{%s%s%s}' % (self.flav.tex,
                             self.FINT_TEXSEP,
                             self.int_type.tex)


NUECC = NuFlavInt('nuecc')
NUEBARCC = NuFlavInt('nuebarcc')
NUMUCC = NuFlavInt('numucc')
NUMUBARCC = NuFlavInt('numubarcc')
NUTAUCC = NuFlavInt('nutaucc')
NUTAUBARCC = NuFlavInt('nutaubarcc')

NUENC = NuFlavInt('nuenc')
NUEBARNC = NuFlavInt('nuebarnc')
NUMUNC = NuFlavInt('numunc')
NUMUBARNC = NuFlavInt('numubarnc')
NUTAUNC = NuFlavInt('nutaunc')
NUTAUBARNC = NuFlavInt('nutaubarnc')


@total_ordering
class NuFlavIntGroup(MutableSequence):
    """Grouping of neutrino flavors+interaction types (flavints)

    Grouping of neutrino flavints. Specification can be via
      * A single `NuFlav` object; this gets promoted to include both
        interaction types
      * A single `NuFlavInt` object
      * String:
        * Ignores anything besides valid tokens
        * A flavor with no interaction type specified will include both CC
          and NC interaction types
        * Multiple flavor/interaction-type specifications can be made;
          use of delimiters is optional
        * Interprets "nuall" as nue+numu+nutau and "nuallbar" as
          nuebar+numubar+nutaubar
      * Iterable containing any of the above (i.e., objects convertible to
        `NuFlavInt` objects). Note that a valid iterable is another
        `NuFlavIntGroup` object.
    """
    TOKENS = re.compile('(nu|e|mu|tau|all|bar|nc|cc)')
    IGNORE = re.compile(r'[^a-zA-Z]')
    FLAVINT_RE = re.compile(
        r'((?:nue|numu|nutau|nuall)(?:bar){0,1}(?:cc|nc){0,2})'
    )
    FLAV_RE = re.compile(r'(?P<fullflav>(?:nue|numu|nutau|nuall)(?:bar){0,1})')
    IT_RE = re.compile(r'(cc|nc)')
    def __init__(self, *args):
        self.flavint_ssep = '+'
        self.__flavints = []
        # Possibly a special case if len(args) == 2, so send as a single entity
        # if this is the case
        if len(args) == 2:
            args = [args]
        for a in args:
            self += a

    def __add__(self, val):
        flavint_list = sorted(set(self.__flavints + self.interpret(val)))
        return NuFlavIntGroup(flavint_list)

    def __iadd__(self, val):
        self.__flavints = sorted(set(self.__flavints + self.interpret(val)))
        return self

    def __delitem__(self, idx):
        self.__flavints.__delitem__(idx)

    def remove(self, value):
        """Remove a flavint from this group.

        Parameters
        ----------
        value : anything accepted by `interpret` method

        """
        flavint_list = sorted(set(self.interpret(value)))
        for k in flavint_list:
            try:
                idx = self.__flavints.index(k)
            except ValueError:
                pass
            else:
                del self.__flavints[idx]

    def __sub__(self, val):
        cp = deepcopy(self)
        cp.remove(val)
        return cp

    def __isub__(self, val):
        self.remove(val)
        return self

    def __setitem__(self, idx, val):
        self.__flavints[idx] = val

    def insert(self, index, value):
        """Insert flavint `value` before `index`"""
        self.__flavints.insert(index, value)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            try:
                self.__class__(other)
            except Exception:
                return False
        return set(self) == set(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            try:
                self.__class__(other)
            except:
                raise ValueError('Cannot compare %s with %s.'
                                 % (other.__class__.__name__,
                                    self.__class__.__name__))

        if len(other) != len(self):
            return len(self) < len(other)

        return sorted(self.flavints)[0] < sorted(other.flavints)[0]

    def __contains__(self, val):
        return all([(k in self.__flavints) for k in self.interpret(val)])

    def __len__(self):
        return len(self.__flavints)

    def __getitem__(self, idx):
        return self.__flavints[idx]

    def __str__(self):
        allkg = set(self.flavints)

        # Check if nuall or nuallbar CC, NC, or both
        nuallcc, nuallbarcc, nuallnc, nuallbarnc = False, False, False, False
        cc_flavints = NuFlavIntGroup(self.cc_flavints)
        nc_flavints = NuFlavIntGroup(self.nc_flavints)
        if len(cc_flavints.particles) == 3:
            nuallcc = True
        if len(cc_flavints.antiparticles) == 3:
            nuallbarcc = True
        if len(nc_flavints.particles) == 3:
            nuallnc = True
        if len(nc_flavints.antiparticles) == 3:
            nuallbarnc = True

        # Construct nuall(bar) part(s) of string
        strs = []
        if nuallcc and nuallnc:
            strs.append('nuall')
            for k in ALL_NUPARTICLES:
                allkg.remove(NuFlavInt(k, 'cc'))
            for k in ALL_NUPARTICLES:
                allkg.remove(NuFlavInt(k, 'nc'))
        elif nuallcc:
            strs.append('nuall' + NuFlavInt.FINT_SSEP + str(CC))
            for k in ALL_NUPARTICLES:
                allkg.remove(NuFlavInt(k, 'cc'))
        elif nuallnc:
            strs.append('nuall' + NuFlavInt.FINT_SSEP + str(NC))
            for k in ALL_NUPARTICLES:
                allkg.remove(NuFlavInt(k, 'nc'))

        if nuallbarcc and nuallbarnc:
            strs.append('nuallbar')
            for k in ALL_NUANTIPARTICLES:
                allkg.remove(NuFlavInt(k, 'cc'))
            for k in ALL_NUANTIPARTICLES:
                allkg.remove(NuFlavInt(k, 'nc'))
        elif nuallbarcc:
            strs.append('nuallbar' + NuFlavInt.FINT_SSEP + str(CC))
            for k in ALL_NUANTIPARTICLES:
                allkg.remove(NuFlavInt(k, 'cc'))
        elif nuallbarnc:
            strs.append('nuallbar' + NuFlavInt.FINT_SSEP + str(NC))
            for k in ALL_NUANTIPARTICLES:
                allkg.remove(NuFlavInt(k, 'nc'))

        # Among remaining flavints, group by flavor and combine if both CC and
        # NC are present for individual flavors (i.e., eliminate the int_type
        # string altogether)
        for flav in ALL_NUPARTICLES + ALL_NUANTIPARTICLES:
            if flav in [k.flav for k in allkg]:
                cc, nc = False, False
                if NuFlavInt(flav, 'cc') in allkg:
                    cc = True
                if NuFlavInt(flav, 'nc') in allkg:
                    nc = True
                if cc and nc:
                    strs.append(str(flav))
                    allkg.remove(NuFlavInt(flav, 'cc'))
                    allkg.remove(NuFlavInt(flav, 'nc'))
                elif cc:
                    strs.append(str(NuFlavInt(flav, 'cc')))
                    allkg.remove(NuFlavInt(flav, 'cc'))
                elif nc:
                    strs.append(str(NuFlavInt(flav, 'nc')))
                    allkg.remove(NuFlavInt(flav, 'nc'))
        return self.flavint_ssep.join(strs)

    def __repr__(self):
        return self.__str__()

    # TODO:
    # Technically, since this is a mutable type, the __hash__ method shouldn't
    # be implemented as this will allow for "illegal" behavior, like using
    # a NuFlavIntGroup as a key in a dict. So this should be fixed, maybe.
    #__hash__ = None
    def __hash__(self):
        return hash(tuple(self.__flavints))

    @staticmethod
    def interpret(val):
        """Interpret a NuFlavIntGroup arg"""
        if isinstance(val, str):
            orig_val = val
            try:
                flavints = []
                orig_val = val

                # Eliminate anything besides valid tokens
                val = NuFlavIntGroup.IGNORE.sub('', val.lower())
                #val = ''.join(NuFlavIntGroup.TOKENS.findall(val))

                # Find all flavints specified
                allflavints_str = NuFlavIntGroup.FLAVINT_RE.findall(val)
                # Remove flavints
                val = NuFlavIntGroup.FLAVINT_RE.sub('', val)

                for flavint_str in allflavints_str:
                    match = NuFlavIntGroup.FLAV_RE.match(flavint_str)
                    flav = match.groupdict()['fullflav']

                    # A flavint found above can include 'all' which is actually
                    # three different flavors
                    if 'all' in flav:
                        flavs = [flav.replace('all', x)
                                 for x in ('e', 'mu', 'tau')]
                    else:
                        flavs = [flav]

                    ints = sorted(set(
                        NuFlavIntGroup.IT_RE.findall(flavint_str)
                    ))

                    # If flavint_str does not include 'cc' or 'nc', include both
                    if not ints:
                        ints = ['cc', 'nc']

                    # Add all combinations of (flav, int) found in this
                    # flavint_str
                    flavints.extend([''.join(fi)
                                     for fi in product(flavs, ints)])

            except (ValueError, AttributeError):
                raise ValueError('Could not interpret `val` = "%s" as %s;'
                                 ' type(val) = %s'
                                 % (orig_val, NuFlavIntGroup, type(orig_val)))

        elif isinstance(val, NuFlav):
            flavints = [NuFlavInt((val, 'cc')), NuFlavInt((val, 'nc'))]
        elif isinstance(val, NuFlavInt):
            flavints = [val]
        elif isinstance(val, NuFlavIntGroup):
            flavints = list(val.flavints)
        elif np.isscalar(val):
            flavints = [val]
        elif val is None:
            flavints = []
        elif hasattr(val, '__len__'):
            flavints = []
            # Treat length-2 iterables as special case, in case the two
            # elements can form a single NuFlavInt.
            if len(val) == 2:
                try_again = True
                try:
                    # Start with counter-hypothesis: that the two elements of
                    # `val` can form two valid, independent NuFlavInts...
                    k1 = NuFlavIntGroup.interpret(val[0])
                    k2 = NuFlavIntGroup.interpret(val[1])
                    if k1 and k2:
                        # Success: Two independent NuFlavInts were created
                        try_again = False
                        flavints.extend(k1)
                        flavints.extend(k2)
                except (UnboundLocalError, ValueError, AssertionError,
                        TypeError):
                    pass
                if try_again:
                    # If the two elements of the iterable did not form two
                    # NuFlavInts, try forming a single NuFlavInt with `val`
                    flavints = [NuFlavInt(val)]
            else:
                # If 1 or >2 elements in `val`, make a flavint out of each
                for x in val:
                    flavints.extend(NuFlavIntGroup.interpret(x))
        else:
            raise Exception('Unhandled val: ' + str(val) + ', class '
                            + str(val.__class__) + ' type ' + str(val))

        flavint_list = []
        for k in flavints:
            try:
                nk = NuFlavInt(k)
                flavint_list.append(nk)
            except TypeError:
                # If NuFlavInt failed, try NuFlav; if this fails, give up.
                flav = NuFlav(k)
                flavint_list.append(NuFlavInt((flav, 'cc')))
                flavint_list.append(NuFlavInt((flav, 'nc')))
        return flavint_list

    @property
    def flavints(self):
        """Return tuple of all NuFlavInts that make up this group"""
        return tuple(self.__flavints)

    @property
    def flavs(self):
        """Return tuple of unique flavors that make up this group"""
        return tuple(sorted(set([k.flav for k in self.__flavints])))

    @property
    def cc_flavints(self):
        """Return tuple of unique charged-current-interaction NuFlavInts that
        make up this group"""
        return tuple([k for k in self.__flavints
                      if k.int_type == CC])

    @property
    def nc_flavints(self):
        """Return tuple of unique neutral-current-interaction NuFlavInts that
        make up this group"""
        return tuple([k for k in self.__flavints
                      if k.int_type == NC])

    @property
    def particles(self):
        """Return tuple of unique particle (vs antiparticle) NuFlavInts that
        make up this group"""
        return tuple([k for k in self.__flavints if k.particle])

    @property
    def antiparticles(self):
        """Return tuple of unique antiparticle NuFlavInts that make up this
        group"""
        return tuple([k for k in self.__flavints if k.antiparticle])

    @property
    def cc_flavs(self):
        """Return tuple of unique charged-current-interaction flavors that
        make up this group. Note that only the flavors, and not NuFlavInts, are
        returned (cf. method `cc_flavints`"""
        return tuple(sorted(set([k.flav for k in self.__flavints
                                 if k.int_type == CC])))

    @property
    def nc_flavs(self):
        """Return tuple of unique neutral-current-interaction flavors that
        make up this group. Note that only the flavors, and not NuFlavInts, are
        returned (cf. method `nc_flavints`"""
        return tuple(sorted(set([k.flav for k in self.__flavints
                                 if k.int_type == NC])))

    #def unique_flavs(self):
    #    """Return tuple of unique flavors that make up this group"""
    #    return tuple(sorted(set([k.flav for k in self.__flavints])))

    def group_flavs_by_int_type(self):
        """Return a dictionary with flavors grouped by the interaction types
        represented in this group.

        The returned dictionary has format::

            {
                'all_int_type_flavs': [<NuFlav object>, <NuFlav object>, ...],
                'cc_only_flavs':      [<NuFlav object>, <NuFlav object>, ...],
                'nc_only_flavs':      [<NuFlav object>, <NuFlav object>, ...],
            }

        where the lists of NuFlav objects are mutually exclusive
        """
        uniqueF = self.flavs
        fint_d = {f: set() for f in uniqueF}
        for k in self.flavints:
            fint_d[k.flav].add(k.int_type)
        grouped = {
            'all_int_type_flavs': [],
            'cc_only_flavs': [],
            'nc_only_flavs': []
        }
        for f in uniqueF:
            if len(fint_d[f]) == 2:
                grouped['all_int_type_flavs'].append(f)
            elif list(fint_d[f])[0] == CC:
                grouped['cc_only_flavs'].append(f)
            else:
                grouped['nc_only_flavs'].append(f)
        return grouped

    def __simple_str(self, flavsep, intsep, flavintsep, addsep, func):
        grouped = self.group_flavs_by_int_type()
        all_nu = AllNu()
        all_nubar = AllNuBar()
        for k, v in grouped.items():
            if all([f in v for f in all_nubar.flav]):
                for f in all_nubar.flav:
                    grouped[k].remove(f)
                grouped[k].insert(0, all_nubar)
            if all([f in v for f in all_nu.flav]):
                for f in all_nu.flav:
                    grouped[k].remove(f)
                grouped[k].insert(0, all_nu)
        all_s = flavsep.join([func(f) for f in grouped['all_int_type_flavs']])
        cc_only_s = flavsep.join([func(f) for f in grouped['cc_only_flavs']])
        nc_only_s = flavsep.join([func(f) for f in grouped['nc_only_flavs']])
        strs = []
        if all_s:
            if not cc_only_s and not nc_only_s:
                strs.append(all_s)
            else:
                strs.append(all_s + intsep + func(CC) + addsep + func(NC))
        if cc_only_s:
            strs.append(cc_only_s + intsep + func(CC))
        if nc_only_s:
            strs.append(nc_only_s + intsep + func(NC))
        return flavintsep.join(strs)

    def simple_str(self, flavsep='+', intsep=' ', flavintsep=', ',
                   addsep='+'):
        """Simple string representation of this group"""
        return self.__simple_str(flavsep=flavsep, intsep=intsep,
                                 flavintsep=flavintsep, addsep=addsep, func=str)

    def file_str(self, flavsep='_', intsep='_', flavintsep='__', addsep=''):
        """String representation for this group useful for file names"""
        return self.__simple_str(flavsep=flavsep, intsep=intsep,
                                 flavintsep=flavintsep, addsep=addsep, func=str)

    def simple_tex(self, flavsep=r' + ', intsep=r' \, ',
                   flavintsep=r'; \; ', addsep=r'+'):
        """Simplified TeX string reperesentation of this group"""
        return self.__simple_str(
            flavsep=flavsep, intsep=intsep,
            flavintsep=flavintsep, addsep=addsep, func=lambda x: x.tex
        )

    @property
    def tex(self):
        """TeX string representation for this group"""
        return self.simple_tex()

    @property
    def unique_flavs_tex(self):
        """TeX string representation of the unique flavors present in this
        group"""
        return ' + '.join([f.tex for f in self.flavs])


ALL_NUFLAVINTS = NuFlavIntGroup('nuall,nuallbar')
ALL_NUCC = NuFlavIntGroup('nuall_cc,nuallbar_cc')
ALL_NUNC = NuFlavIntGroup('nuall_nc,nuallbar_nc')


class FlavIntData(dict):
    """Container class for storing data for each NuFlavInt.

    Parameters
    ----------
    val : string, dict, or None
        Data with which to populate the hierarchy.

        If string, interpret as PISA resource and load data from it
        If dict, populate data from the dictionary
        If None, instantiate with None for all data

        The interpreted version of `val` must be a valid data structure: A
        dict with keys 'nue', 'numu', 'nutau', 'nue_bar', 'numu_bar', and
        'nutau_bar'; and each item corresponding to these keys must itself be a
        dict with keys 'cc' and 'nc'.

    Notes
    -----
    Accessing data (both for getting and setting) is fairly flexible. It uses
    dict-like square-brackets syntax, but can accept any object (or two
    objects) that are convertible to a NuFlav or NuFlavInt object. In the
    former case, the entire flavor dictionary (which includes both 'cc' and
    'nc') is returned, while in the latter case whatever lives at the node is
    returned.

    Initializing, setting and getting data in various ways:

    >>> fi_dat = FlavIntData()
    >>> fi_dat['nue', 'cc'] = 1
    >>> fi_dat['nuenc'] = 2
    >>> fi_dat['numu'] = {'cc': 'cc data...', 'nc': 'nc data...'}
    >>> fi_dat[NuFlav(16), IntType(1)] == 4

    >>> fi_dat['nuecc'] == 1
    True
    >>> fi_dat['NUE_NC'] == 2
    True
    >>> fi_dat['nu_e'] == {'cc': 1, 'nc': 2}
    True
    >>> fi_dat['nu mu cc'] == 'cc data...'
    True
    >>> fi_dat['nu mu'] == {'cc': 'cc data...', 'nc': 'nc data...'}
    True
    >>> fi_dat['nutau cc'] == 4
    True

    """
    def __init__(self, val=None):
        super().__init__()
        if isinstance(val, str):
            d = self.__load(val)
        elif isinstance(val, dict):
            d = val
        elif val is None:
            # Instantiate empty FlavIntData
            with BarSep('_'):
                d = {str(f): {str(it): None for it in ALL_NUINT_TYPES}
                     for f in ALL_NUFLAVS}
        else:
            raise TypeError('Unrecognized `val` type %s' % type(val))
        self.validate(d)
        self.update(d)

    @staticmethod
    def _interpret_index(idx):
        if not isinstance(idx, str) and hasattr(idx, '__len__') \
                and len(idx) == 1:
            idx = idx[0]
        with BarSep('_'):
            try:
                nfi = NuFlavInt(idx)
                return [str(nfi.flav), str(nfi.int_type)]
            except (AssertionError, ValueError, TypeError):
                try:
                    return [str(NuFlav(idx))]
                except:
                    raise ValueError('Invalid index: %s' %str(idx))

    def __getitem__(self, *args):
        assert len(args) <= 2
        key_list = self._interpret_index(args)
        tgt_obj = super().__getitem__(key_list[0])
        if len(key_list) == 2:
            tgt_obj = tgt_obj[key_list[1]]
        return tgt_obj

    def __setitem__(self, *args):
        assert len(args) > 1
        item, value = args[:-1], args[-1]
        key_list = self._interpret_index(item)
        if len(key_list) == 1:
            self.__validate_inttype_dict(value)
            value = self.__translate_inttype_dict(value)
        tgt_obj = self
        for key in key_list[:-1]:
            tgt_obj = dict.__getitem__(tgt_obj, key)
        dict.__setitem__(tgt_obj, key_list[-1], value)

    def __eq__(self, other):
        """Recursive, exact equality"""
        return recursiveEquality(self, other)

    @staticmethod
    def __basic_validate(fi_container):
        for flavint in ALL_NUFLAVINTS:
            with BarSep('_'):
                f = str(flavint.flav)
                it = str(flavint.int_type)
            assert isinstance(fi_container, dict), "container must be of" \
                    " type 'dict'; instead got %s" % type(fi_container)
            assert f in fi_container, "container missing flavor '%s'" % f
            assert isinstance(fi_container[f], dict), \
                    "Child of flavor '%s': must be type 'dict' but" \
                    " got %s instead" % (f, type(fi_container[f]))
            assert it in fi_container[f], \
                    "Flavor '%s' sub-dict must contain a both interaction" \
                    " types, but missing (at least) int_type '%s'" % (f, it)

    @staticmethod
    def __validate_inttype_dict(d):
        assert isinstance(d, MutableMapping), \
                "Value must be an inttype (sub-) dict if you only specify a" \
                " flavor (and not an int type) as key"
        keys = d.keys()
        assert (len(keys) == 2) and \
                ([str(k).lower() for k in sorted(keys)] == ['cc', 'nc']), \
                "inttype (sub-) dict must contain exactly 'cc' and 'nc' keys"

    @staticmethod
    def __translate_inttype_dict(d):
        for key in d.keys():
            if not isinstance(key, str) or key.lower() != key:
                val = d.pop(key)
                d[str(key).lower()] = val
        return d

    def __load(self, fname, **kwargs):
        d = fileio.from_file(fname, **kwargs)
        self.validate(d)
        return d

    @property
    def flavs(self):
        """tuple of NuFlav : all flavors present"""
        return tuple(sorted([NuFlav(k) for k in self.keys()]))

    @property
    def flavints(self):
        """tuple of NuFlavInt : all flavints present"""
        fis = []
        for flav in self.keys():
            for int_type in self[flav].keys():
                fis.append(NuFlavInt(flav, int_type))
        return tuple(sorted(fis))

    def allclose(self, other, rtol=1e-05, atol=1e-08):
        """Returns True if all data structures are equal and all numerical
        values contained are within relative (rtol) and/or absolute (atol)
        tolerance of one another.
        """
        return recursiveAllclose(self, other, rtol=rtol, atol=atol)

    def validate(self, fi_container):
        """Perform basic validation on the data structure"""
        self.__basic_validate(fi_container)

    def save(self, fname, **kwargs):
        """Save data structure to a file; see fileio.to_file for details"""
        fileio.to_file(self, fname, **kwargs)

    def id_dupes(self, rtol=None, atol=None):
        """Identify flavints with duplicated data (exactly or within a
        specified tolerance), convert these NuFlavInt's into NuFlavIntGroup's
        and returning these along with the data associated with each.

        Parameters
        ----------
        rtol
            Set to positive value to use as rtol argument for numpy allclose
        atol
            Set to positive value to use as atol argument for numpy allclose

        If either `rtol` or `atol` is 0, exact equality is enforced.

        Returns
        -------
        dupe_flavintgroups : list of NuFlavIntGroup
            A NuFlavIntGroup object is returned for each group of NuFlavInt's
            found with duplicate data
        dupe_flavintgroups_data : list of objects
            Data associated with each NuFlavIntGroup in dupe_flavintgroups.
            Each object in `dupe_flavintgroups_data` corresponds to, and in the
            same order as, the objects in `dupe_flavintgroups`.
        """
        exact_equality = True
        kwargs = {}
        if rtol is not None and rtol > 0 and atol != 0:
            exact_equality = False
            kwargs['rtol'] = rtol
        if atol is not None and atol > 0 and rtol != 0:
            exact_equality = False
            kwargs['atol'] = atol
        if exact_equality:
            cmpfunc = recursiveEquality
        else:
            cmpfunc = lambda x, y: recursiveAllclose(x, y, **kwargs)

        dupe_flavintgroups = []
        dupe_flavintgroups_data = []
        for flavint in self.flavints:
            this_datum = self[flavint]
            match = False
            for n, group_datum in enumerate(dupe_flavintgroups_data):
                if len(this_datum) != len(group_datum):
                    continue
                if cmpfunc(this_datum, group_datum):
                    dupe_flavintgroups[n] += flavint
                    match = True
                    break
            if not match:
                dupe_flavintgroups.append(NuFlavIntGroup(flavint))
                dupe_flavintgroups_data.append(this_datum)

        sort_inds = np.argsort(dupe_flavintgroups)
        dupe_flavintgroups = [dupe_flavintgroups[i] for i in sort_inds]
        dupe_flavintgroups_data = [dupe_flavintgroups_data[i]
                                   for i in sort_inds]

        return dupe_flavintgroups, dupe_flavintgroups_data


class FlavIntDataGroup(dict):
    """Container class for storing data for some set(s) of NuFlavIntGroups
    (cf. FlavIntData, which stores one datum for each NuFlavInt separately)

    Parameters
    ----------
    val: None, str, or dict
        Data with which to populate the hierarchy
    flavint_groups: None, str, or iterable
        User-defined groupings of NuFlavIntGroups. These can be specified
        in several ways.

        None
            If val == None, flavint_groups must be specified
            If val != None, flavitn_groups are deduced from the data
        string
            If val is a string, it is expected to be a comma-separated
            list, each field of which describes a NuFlavIntGroup. The
            returned list of groups encompasses all possible flavor/int
            types, but the groups are mutually exclusive.
        iterable of strings or NuFlavIntGroup
            If val is an iterable, each member of the iterable is
            interpreted as a NuFlavIntGroup.
    """
    def __init__(self, val=None, flavint_groups=None):
        super().__init__()
        self._flavint_groups = None
        if flavint_groups is None:
            if val is None:
                raise ValueError('Error - must input at least one of '
                                 '`flavint_groups` or `val`.')
        else:
            self.flavint_groups = flavint_groups

        if val is None:
            # Instantiate empty FlavIntDataGroup
            d = {str(group): None for group in self.flavint_groups}
        else:
            if isinstance(val, str):
                d = self.__load(val)
            elif isinstance(val, dict):
                d = val
            else:
                raise TypeError('Unrecognized `val` type %s' % type(val))
            d = {str(NuFlavIntGroup(key)): d[key] for key in d.keys()}
            if d.keys() == ['']:
                raise ValueError('NuFlavIntGroups not found in data keys')

            fig = [NuFlavIntGroup(fig) for fig in d.keys()]
            if flavint_groups is None:
                self.flavint_groups = fig
            else:
                if set(fig) != set(self.flavint_groups):
                    raise ValueError(
                        'Specified `flavint_groups` does not match `val` '
                        'signature.\n`flavint_groups` - {0}\n`val groups` '
                        '- {1}'.format(self.flavint_groups, fig)
                    )

        self.validate(d)
        self.update(d)

    @property
    def flavint_groups(self):
        return self._flavint_groups

    @flavint_groups.setter
    def flavint_groups(self, value):
        assert 'muons' not in value
        fig = self._parse_flavint_groups(value)
        all_flavints = reduce(add, [f.flavints for f in fig])
        for fi in set(all_flavints):
            if all_flavints.count(fi) > 1:
                raise ValueError(
                    'FlavInt {0} referred to multiple times in flavint_group '
                    '{1}'.format(fi, fig)
                )
        self._flavint_groups = fig

    def transform_groups(self, flavint_groups):
        """Transform FlavIntDataGroup into a structure given by the input
        flavint_groups.

        Parameters
        ----------
        flavint_groups : string, or sequence of strings or sequence of
                         NuFlavIntGroups

        Returns
        -------
        transformed_fidg : FlavIntDataGroup

        """
        flavint_groups = self._parse_flavint_groups(flavint_groups)

        original_flavints = reduce(add, [list(f.flavints) for f in
                                         self.flavint_groups])
        inputted_flavints = reduce(add, [list(f.flavints) for f in
                                         flavint_groups])
        if not set(inputted_flavints).issubset(set(original_flavints)):
            raise ValueError(
                'Mismatch between underlying group of flavints given as input '
                'and original flavint_group.\nOriginal {0}\nInputted '
                '{1}'.format(set(original_flavints), set(inputted_flavints))
            )

        transformed_fidg = FlavIntDataGroup(flavint_groups=flavint_groups)
        for in_fig in flavint_groups:
            for or_fig in self.flavint_groups:
                if or_fig in in_fig:
                    if transformed_fidg[in_fig] is None:
                        transformed_fidg[in_fig] = deepcopy(self[or_fig])
                    else:
                        transformed_fidg[in_fig] = \
                                self._merge(transformed_fidg[in_fig],
                                            self[or_fig])
                elif in_fig in or_fig:
                    raise ValueError(
                        'Cannot decouple original flavint_group {0} into input'
                        'flavint_group {1}'.format(or_fig, in_fig)
                    )
        logging.trace('Transformed from\n{0}\nto '
                      '{1}'.format(self.flavint_groups, flavint_groups))
        return transformed_fidg

    def allclose(self, other, rtol=1e-05, atol=1e-08):
        """Returns True if all data structures are equal and all numerical
        values contained are within relative (rtol) and/or absolute (atol)
        tolerance of one another.
        """
        return recursiveAllclose(self, other, rtol=rtol, atol=atol)

    def validate(self, fi_container):
        """Perform basic validation on the data structure"""
        self.__basic_validate(fi_container)

    def save(self, fname, **kwargs):
        """Save data structure to a file; see fileio.to_file for details"""
        fileio.to_file(self, fname, **kwargs)

    @staticmethod
    def _parse_flavint_groups(flavint_groups):
        if isinstance(flavint_groups, str):
            return flavintGroupsFromString(flavint_groups)
        elif isinstance(flavint_groups, NuFlavIntGroup):
            return [flavint_groups]
        elif isinstance(flavint_groups, Sequence):
            if all(isinstance(f, NuFlavIntGroup) for f in flavint_groups):
                return flavint_groups
            elif all(isinstance(f, NuFlavInt) for f in flavint_groups):
                return [NuFlavIntGroup(f) for f in flavint_groups]
            elif all(isinstance(f, str) for f in flavint_groups):
                return [NuFlavIntGroup(f) for f in flavint_groups]
            else:
                raise ValueError(
                    'Elements in `flavint_groups` not all type '
                    'NuFlavIntGroup or string: %s' % flavint_groups
                )
        else:
            raise TypeError('Unrecognized `flavint_groups` type %s' %
                            type(flavint_groups))

    @staticmethod
    def _merge(a, b, path=None):
        """Merge dictionaries `a` and `b` by recursively iterating down
        to the lowest level of the dictionary until coincident numpy
        arrays are found, after which the appropriate sub-element is
        made equal to the concatenation of the two arrays.
        """
        if path is None:
            path = []
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    FlavIntDataGroup._merge(a[key], b[key], path + [str(key)])
                elif isinstance(a[key], np.ndarray) and \
                        isinstance(b[key], np.ndarray):
                    a[key] = np.concatenate((a[key], b[key]))
                elif isinstance(a[key], ureg.Quantity) and \
                        isinstance(b[key], ureg.Quantity):
                    if isinstance(a[key].m, np.ndarray) and \
                       isinstance(b[key].m, np.ndarray):
                        units = a[key].units
                        a[key] = np.concatenate((a[key].m, b[key].m_as(units)))
                        a[key] = a[key] * units
                    else:
                        raise Exception(
                            'Conflict at %s' % '.'.join(path + [str(key)])
                        )
                else:
                    raise Exception(
                        'Conflict at %s' % '.'.join(path + [str(key)])
                    )
            else:
                a[key] = b[key]
        return a

    @staticmethod
    def _interpret_index(idx):
        try:
            nfi = NuFlavIntGroup(idx)
            return str(nfi)
        except:
            raise ValueError('Invalid index: %s' % str(idx))

    def __basic_validate(self, fi_container):
        for group in self.flavint_groups:
            f = str(group)
            assert isinstance(fi_container, dict), "container must be of" \
                    " type 'dict'; instead got %s" % type(fi_container)
            assert f in fi_container, \
                    "container missing flavint group '%s'" % f

    @staticmethod
    def __load(fname, **kwargs):
        d = fileio.from_file(fname, **kwargs)
        return d

    def __add__(self, other):
        d = deepcopy(self)
        d = self._merge(d, other)
        combined_flavint_groups = list(
            set(self.flavint_groups + other.flavint_groups)
        )
        return FlavIntDataGroup(val=d, flavint_groups=combined_flavint_groups)

    def __getitem__(self, arg):
        key = self._interpret_index(arg)
        tgt_obj = super().__getitem__(key)
        return tgt_obj

    def __setitem__(self, arg, value):
        key = self._interpret_index(arg)
        if NuFlavIntGroup(key) not in self.flavint_groups:
            self.flavint_groups += [NuFlavIntGroup(key)]
        super().__setitem__(key, value)

    def __eq__(self, other):
        """Recursive, exact equality"""
        return recursiveEquality(self, other)


def flavintGroupsFromString(groups):
    """Interpret `groups` to break into neutrino flavor/interaction type(s)
    that are to be grouped together; also form singleton groups as specified
    explicitly in `groups` or for any unspecified flavor/interaction type(s).

    The returned list of groups encompasses all possible flavor/int types, but
    the groups are mutually exclusive.

    Parameters
    ----------
    groups : None, string, or sequence of strings

    Returns
    -------
    flavint_groups : list of NuFlavIntGroup

    """
    if groups is None or groups == '':
        # None are to be grouped together
        grouped = []
        # All will be singleton groups
        ungrouped = [NuFlavIntGroup(k) for k in ALL_NUFLAVINTS]
    else:
        grouped, ungrouped = xlateGroupsStr(groups)

    # Find any flavints not included in the above groupings
    flavint_groups = grouped + ungrouped
    logging.trace('flav/int in the following group(s) will be joined together:'
                  + ', '.join([str(k) for k in grouped]))
    logging.trace('flav/ints treated individually:'
                  + ', '.join([str(k) for k in ungrouped]))

    # Enforce that flavints composing groups are mutually exclusive
    for grp0, grp1 in combinations(flavint_groups, 2):
        if not set(grp0).isdisjoint(grp1):
            overlapping = sorted(set(grp0).intersection(grp1))
            raise ValueError(
                'All flavint groups must be disjoint with one another, but'
                ' groups %s and %s have overlapping flavint(s): %s'
                % (grp0, grp1, ', '.join(str(g) for g in overlapping))
            )

    return sorted(flavint_groups)


def xlateGroupsStr(val):
    """Translate a ","-separated string into separate `NuFlavIntGroup`s.

    val
        ","-delimited list of valid NuFlavIntGroup strings, e.g.:
            "nuall_nc,nue,numu_cc+numubar_cc"
        Note that specifying NO interaction type results in both interaction
        types being selected, e.g. "nue" implies "nue_cc+nue_nc". For other
        details of how the substrings are interpreted, see docs for
        NuFlavIntGroup.

    returns:
        grouped, ungrouped

    grouped, ungrouped
        lists of NuFlavIntGroups; the first will have more than one flavint
        in each NuFlavIntGroup whereas the second will have just one
        flavint in each NuFlavIntGroup. Either list can be of 0-length.

    This function does not enforce mutual-exclusion on flavints in the
    various flavint groupings, but does list any flavints not grouped
    together in the `ungrouped` return arg. Mutual exclusion can be
    enforced through set operations upon return.
    """
    # What flavints to group together
    grouped = [NuFlavIntGroup(s) for s in re.split('[,;]', val)]

    # Find any flavints not included in the above groupings
    all_flavints = set(ALL_NUFLAVINTS)
    all_grouped_flavints = set(NuFlavIntGroup(grouped))
    ungrouped = [NuFlavIntGroup(k) for k in
                 sorted(all_flavints.difference(all_grouped_flavints))]

    return grouped, ungrouped


# pylint: disable=line-too-long
def test_IntType():
    """IntType unit tests"""
    #==========================================================================
    # Test IntType
    #==========================================================================
    ref = CC
    assert IntType('\n\t _cc \n') == ref
    try:
        IntType('numubarcc')
    except ValueError:
        pass
    else:
        raise Exception()
    assert IntType(1) == ref
    assert IntType(1.0) == ref
    assert IntType(CC) == ref
    assert IntType(NuFlavInt('numubarcc')) == ref
    for int_code in [1, 2]:
        IntType(int_code)
        IntType(float(int_code))
    logging.info('<< PASS : test_IntType >>')


# pylint: disable=line-too-long
def test_NuFlav():
    """NuFlav unit tests"""
    all_f_codes = [12, -12, 14, -14, 16, -16]

    #==========================================================================
    # Test NuFlav
    #==========================================================================
    ref = NuFlav('numu')
    assert ref.code == 14
    assert (-ref).code == -14
    assert ref.bar_code == 1
    assert (-ref).bar_code == -1
    assert ref.particle
    assert not (-ref).particle
    assert not ref.antiparticle
    assert (-ref).antiparticle

    #assert NuFlav('\n\t _ nu_ mu_ cc\n\t\r') == ref
    #assert NuFlav('numucc') == ref
    assert NuFlav(14) == ref
    assert NuFlav(14.0) == ref
    assert NuFlav(NuFlav('numu')) == ref
    assert NuFlav(NuFlavInt('numucc')) == ref
    assert NuFlav(NuFlavInt('numunc')) == ref

    for f in all_f_codes:
        NuFlav(f)
        NuFlav(float(f))
    for (f, bnb) in product(['e', 'mu', 'tau'], ['', 'bar']):
        NuFlav('nu_' + f + '_' + bnb)

    logging.info('<< PASS : test_NuFlav >>')


# pylint: disable=line-too-long
def test_NuFlavInt():
    """NuFlavInt unit tests"""
    all_f_codes = [12, -12, 14, -14, 16, -16]
    all_i_codes = [1, 2]

    #==========================================================================
    # Test NuFlavInt
    #==========================================================================
    try:
        NuFlavInt('numu')
    except ValueError:
        pass

    # Equality
    fi_comb = [fic for fic in product(all_f_codes, all_i_codes)]
    for (fi0, fi1) in product(fi_comb, fi_comb):
        if fi0 == fi1:
            assert NuFlavInt(fi0) == NuFlavInt(fi1)
        else:
            assert NuFlavInt(fi0) != NuFlavInt(fi1)
    assert NuFlavInt((12, 1)) != 'xyz'
    # Sorting: this is my desired sort order
    nfl0 = [NUECC, NUEBARCC, NUMUCC, NUMUBARCC, NUTAUCC, NUTAUBARCC, NUENC,
            NUEBARNC, NUMUNC, NUMUBARNC, NUTAUNC, NUTAUBARNC]
    nfl1 = deepcopy(nfl0)
    np.random.shuffle(nfl1)
    nfl_sorted = sorted(nfl1)
    assert all([v0 == nfl_sorted[n] for n, v0 in enumerate(nfl0)]), str(nfl_sorted)
    assert len(nfl0) == len(nfl_sorted)

    # Test NuFlavInt instantiation
    _ = NuFlav('nue')
    _ = IntType('cc')
    _ = IntType('nc')
    _ = NuFlav('nuebar')
    flavs = list(ALL_NUFLAVS)
    flavs.extend(['nue', 'numu', 'nutau', 'nu_e', 'nu e', 'Nu E', 'nuebar',
                  'nu e bar'])
    flavs.extend(all_f_codes)
    _ = NuFlavInt('nuebarnc')

    # Instantiate with combinations of flavs and int types
    for f, i in product(flavs, [1, 2, 'cc', 'nc', CC, NC]):
        ref = NuFlavInt(f, i)
        assert NuFlavInt((f, i)) == ref
        assert NuFlavInt(flav=f, int_type=i) == ref
        if isinstance(f, str) and isinstance(i, str):
            assert NuFlavInt(f+i) == ref
            assert NuFlavInt(f + '_' + i) == ref
            assert NuFlavInt(f + ' ' + i) == ref

    # Instantiate with already-instantiated `NuFlavInt`s
    assert NuFlavInt(NUECC) == NuFlavInt('nuecc')
    assert NuFlavInt(NUEBARNC) == NuFlavInt('nuebarnc')

    # test negating flavint
    nk = NuFlavInt('numucc')
    assert -nk == NuFlavInt('numubarcc')

    logging.info('<< PASS : test_NuFlavInt >>')


# pylint: disable=line-too-long
def test_NuFlavIntGroup():
    """NuFlavIntGroup unit tests"""
    all_f_codes = [12, -12, 14, -14, 16, -16]
    all_i_codes = [1, 2]

    #==========================================================================
    # Test NuFlavIntGroup
    #==========================================================================
    fi_comb = [fic for fic in product(all_f_codes, all_i_codes)]
    nfl0 = [NuFlavInt(fic) for fic in fi_comb]
    nfl1 = [NuFlavInt(fic) for fic in fi_comb]
    nfl_sorted = sorted(nfl1)
    nkg0 = NuFlavIntGroup(nfl0)
    nkg1 = NuFlavIntGroup(nfl_sorted)
    assert nkg0 == nkg1
    assert nkg0 != 'xyz'
    assert nkg0 != 'xyz'

    # Test inputs
    assert NuFlavIntGroup('nuall,nuallbar').flavs == tuple([NuFlav(c) for c in all_f_codes]), str(NuFlavIntGroup('nuall,nuallbar').flavs)

    #
    # Test NuFlavIntGroup instantiation
    #
    nue = NuFlav('nue')
    numu = NuFlav('numu')
    nue_cc = NuFlavInt('nue_cc')
    nue_nc = NuFlavInt('nue_nc')

    # Empty args
    NuFlavIntGroup()
    NuFlavIntGroup([])

    # String flavor promoted to CC+NC
    assert set(NuFlavIntGroup('nue').flavints) == set((nue_cc, nue_nc))
    # NuFlav promoted to CC+NC
    assert set(NuFlavIntGroup(nue).flavints) == set((nue_cc, nue_nc))
    # List of single flav str same as above
    assert set(NuFlavIntGroup(['nue']).flavints) == set((nue_cc, nue_nc))
    # List of single flav same as above
    assert set(NuFlavIntGroup([nue]).flavints) == set((nue_cc, nue_nc))

    # Single flavint spec
    assert set(NuFlavIntGroup(nue_cc).flavints) == set((nue_cc,))
    # Str with single flavint spec
    assert set(NuFlavIntGroup('nue_cc').flavints) == set((nue_cc,))
    # List of single str containing single flavint spec
    assert set(NuFlavIntGroup(['nue_cc']).flavints) == set((nue_cc,))

    # Multiple flavints as *args
    assert set(NuFlavIntGroup(nue_cc, nue_nc).flavints) == set((nue_cc, nue_nc))
    # List of flavints
    assert set(NuFlavIntGroup([nue_cc, nue_nc]).flavints) == set((nue_cc, nue_nc))
    # List of single str containing multiple flavints spec
    assert set(NuFlavIntGroup(['nue_cc,nue_nc']).flavints) == set((nue_cc, nue_nc))
    # List of str containing flavints spec
    assert set(NuFlavIntGroup(['nue_cc', 'nue_nc']).flavints) == set((nue_cc, nue_nc))

    # Another NuFlavIntGroup
    assert set(NuFlavIntGroup(NuFlavIntGroup(nue_cc, nue_nc)).flavints) == set((nue_cc, nue_nc))

    # Addition of flavints promoted to NuFlavIntGroup
    assert nue_cc + nue_nc == NuFlavIntGroup(nue)
    # Addition of flavs promoted to NuFlavIntGroup including both CC & NC
    assert nue + numu == NuFlavIntGroup(nue, numu)

    # Test remove
    nkg = NuFlavIntGroup('nue_cc+numucc')
    nkg.remove(NuFlavInt((12, 1)))
    assert nkg == NuFlavIntGroup('numucc')

    # Test del
    nkg = NuFlavIntGroup('nue_cc+numucc')
    del nkg[0]
    assert nkg == NuFlavIntGroup('numucc')

    # Equivalent object when converting to string and back to NuFlavIntGroup from
    # that string
    for n in range(1, len(ALL_NUFLAVINTS)+1):
        logging.debug('NuFlavIntGroup --> str --> NuFlavIntGroup, n = %d', n)
        for comb in combinations(ALL_NUFLAVINTS, n):
            ref = NuFlavIntGroup(comb)
            assert ref == NuFlavIntGroup(str(ref))

    # Ordering
    desired_order = [
        NuFlavIntGroup(s) for s in [
            'nuecc', 'nuebarcc', 'numucc', 'numubarcc', 'nutaucc',
            'nutaubarcc', 'nuallnc', 'nuallbarnc'
        ]
    ]
    groups = flavintGroupsFromString('nuallnc, nuallbarnc')
    assert groups == desired_order, str(groups)
    assert sorted(groups) == desired_order, str(sorted(groups))

    desired_order = [
        NuFlavIntGroup(s) for s in [
            'nuecc', 'nuebarcc', 'numucc', 'numubarcc', 'nutaucc',
            'nutaubarcc', 'nuallnc', 'nuallbarnc'
        ]
    ]
    groups = flavintGroupsFromString('nuallnc, nuallbarnc')
    assert groups == desired_order, str(groups)
    assert sorted(groups) == desired_order, str(sorted(groups))

    # test TeX strings
    nkg = NuFlavIntGroup('nuall,nuallbar')
    logging.info(str(nkg))
    logging.info(nkg.tex)
    logging.info(nkg.simple_str())
    logging.info(nkg.simple_tex())
    logging.info(nkg.unique_flavs_tex)

    logging.info('<< ???? : test_NuFlavIntGroup >> checks pass upon inspection'
                 ' of above outputs and generated file(s).')


# pylint: disable=line-too-long
def test_FlavIntData():
    """FlavIntData unit tests"""
    #==========================================================================
    # Test FlavIntData
    #==========================================================================
    # Excercise the "standard" PISA nested-python-dict features, where this
    # dict uses an '_' to separate 'bar' in key names, and the nested dict
    # levels are [flavor][interaction type].

    # Force separator to something weird before starting, to ensure everything
    # still works and this separator is still set when we're done
    oddball_sep = 'xyz'
    set_bar_ssep(oddball_sep)
    ref_pisa_dict = {f: {it: None for it in ['cc', 'nc']} for f in
                     ['nue', 'nue_bar', 'numu', 'numu_bar', 'nutau',
                      'nutau_bar']}
    fi_cont = FlavIntData()
    for f in ['nue', 'nue_bar', 'numu', 'numu_bar', 'nutau', 'nutau_bar']:
        for it in ['cc', 'nc']:
            assert fi_cont[f][it] == ref_pisa_dict[f][it]
            flavint = NuFlavInt(f, it)
            assert flavint.pidx(ref_pisa_dict) == ref_pisa_dict[f][it]
            logging.trace('%s: %s' %('flavint', flavint))
            logging.trace('%s: %s' %('f', f))
            logging.trace('%s: %s' %('it', it))
            logging.trace('%s: %s' %('fi_cont', fi_cont))
            logging.trace('%s: %s' %('fi_cont[f]', fi_cont[f]))
            logging.trace('%s: %s' %('fi_cont[f][it]', fi_cont[f][it]))
            logging.trace('%s: %s' %('fi_cont[flavint]', fi_cont[flavint]))
            logging.trace('%s: %s' %('fi_cont[flavint]', fi_cont[flavint]))
            assert fi_cont[flavint] == fi_cont[f][it]
            assert fi_cont[flavint] == fi_cont[flavint]
    assert get_bar_ssep() == oddball_sep
    set_bar_ssep('')

    # These should fail because you're only allowed to access the flav or
    # flavint part of the data structure, no longer any sub-items (use
    # subsequent [k1][k2]... to do this instead)
    fi_cont['numu', 'cc'] = {'sub-key': {'sub-sub-key': None}}
    try:
        fi_cont['numu', 'cc', 'sub-key']
    except ValueError:
        pass
    else:
        raise Exception('Test failed, exception should have been raised')

    try:
        fi_cont['numu', 'cc', 'sub-key'] = 'new sub-val'
    except ValueError:
        pass
    else:
        raise Exception('Test failed, exception should have been raised')

    # These should fail because setting flavor-only as a string would
    # invalidate the data structure (not a nested dict)
    try:
        fi_cont[NuFlav('numu')] = 'xyz'
    except AssertionError:
        pass
    else:
        raise Exception('Test failed, exception should have been raised')

    try:
        fi_cont[NuFlav('numu')] = {'cc': 'cc_xyz'}
    except AssertionError:
        pass
    else:
        raise Exception('Test failed, exception should have been raised')

    # The previously-valid fi_cont should *still* be valid, as `set` should
    # revert to the original (valid) values rather than keep the invalid values
    # that were attempted to be set above
    fi_cont.validate(fi_cont)

    # This should be okay because datastructure is still valid if the item
    # being set on the flavor (only) is a valid int-type dict
    fi_cont[NuFlav('numu')] = {'cc': 'cc_xyz', 'nc': 'nc_xyz'}

    # Test setting, getting, and JSON serialization of FlavIntData
    fi_cont['nue', 'cc'] = 'this is a string blah blah blah'
    _ = fi_cont[NuFlavInt('nue_cc')]
    fi_cont[NuFlavInt('nue_nc')] = np.pi
    _ = fi_cont[NuFlavInt('nue_nc')]
    fi_cont[NuFlavInt('numu_cc')] = [0, 1, 2, 3]
    _ = fi_cont[NuFlavInt('numu_cc')]
    fi_cont[NuFlavInt('numu_nc')] = {'new': {'nested': {'dict': 'xyz'}}}
    _ = fi_cont[NuFlavInt('numu_nc')]
    fi_cont[NuFlavInt('nutau_cc')] = 1
    _ = fi_cont[NuFlavInt('nutau_cc')]
    fi_cont[NuFlavInt('nutaubar_cc')] = np.array([0, 1, 2, 3])
    _ = fi_cont[NuFlavInt('nutaubar_cc')]
    fname = '/tmp/test_FlavIntData.json'
    logging.info('Writing FlavIntData to file %s; inspect.', fname)
    fileio.to_file(fi_cont, fname, warn=False)
    fi_cont2 = fileio.from_file(fname)
    assert recursiveEquality(fi_cont2, fi_cont), \
            'fi_cont=%s\nfi_cont2=%s' %(fi_cont, fi_cont2)

    logging.info('<< ???? : test_FlavIntData >> checks pass upon inspection of'
                 ' above outputs and generated file(s).')


# pylint: disable=line-too-long
def test_FlavIntDataGroup():
    """FlavIntDataGroup unit tests"""
    flavint_group = 'nue, numu_cc+numubar_cc, nutau_cc'
    FlavIntDataGroup(flavint_groups=flavint_group)
    fidg1 = FlavIntDataGroup(
        flavint_groups='nuall, nu all bar CC, nuallbarnc',
        val={'nuall': np.arange(0, 100),
             'nu all bar CC': np.arange(100, 200),
             'nuallbarnc': np.arange(200, 300)}
    )
    fidg2 = FlavIntDataGroup(
        val={'nuall': np.arange(0, 100),
             'nu all bar CC': np.arange(100, 200),
             'nuallbarnc': np.arange(200, 300)}
    )
    assert fidg1 == fidg2

    try:
        fidg1 = FlavIntDataGroup(
            flavint_groups='nuall, nu all bar, nuallbar',
            val={'nuall': np.arange(0, 100),
                 'nu all bar CC': np.arange(100, 200),
                 'nuallbarnc': np.arange(200, 300)}
        )
    except ValueError:
        pass
    else:
        raise Exception

    try:
        fidg1 = FlavIntDataGroup(flavint_groups=['nuall', 'nue'])
    except (ValueError, AssertionError):
        pass
    else:
        raise Exception

    assert set(fidg1.keys()) == set(('nuall', 'nuallbar_cc', 'nuallbar_nc'))
    fidg1.save('/tmp/test_FlavIntDataGroup.json', warn=False)
    fidg1.save('/tmp/test_FlavIntDataGroup.hdf5', warn=False)
    fidg3 = FlavIntDataGroup(val='/tmp/test_FlavIntDataGroup.json')
    fidg4 = FlavIntDataGroup(val='/tmp/test_FlavIntDataGroup.hdf5')
    assert fidg3 == fidg1
    assert fidg4 == fidg1

    figroups = ('nuecc+nuebarcc,numucc+numubarcc,nutaucc+nutaubarcc,'
                'nuallnc,nuallbarnc')
    cfidat = FlavIntDataGroup(flavint_groups=figroups)

    for k in cfidat.flavint_groups:
        cfidat[k] = np.arange(10)

    cfidat[NuFlavIntGroup('nuecc+nuebarcc')] = np.arange(10)

    logging.debug(str((fidg1 + fidg2)))
    assert fidg1 == fidg2
    try:
        logging.debug(str((fidg1 + cfidat)))
    except (ValueError, AssertionError):
        pass
    else:
        raise Exception

    d1 = {
        'numu+numubar': {
            'energy': np.arange(0, 10)
        },
        'nutau+nutaubar': {
            'energy': np.arange(0, 10)
        }
    }
    d2 = {
        'nue+nuebar': {
            'weights': np.arange(0, 10)
        },
        'nutau+nutaubar': {
            'weights': np.arange(0, 10)
        }
    }
    d1 = FlavIntDataGroup(val=d1)
    d2 = FlavIntDataGroup(val=d2)
    d3 = d1 + d2
    logging.debug(str((d3)))

    tr_d1 = d1.transform_groups(['numu+numubar+nutau+nutaubar'])
    logging.debug(str((tr_d1)))
    tr_d3 = d3.transform_groups('nue+nuebar+numu+numubar, nutau+nutaubar')
    tr_d3_1 = d3.transform_groups(['nue+nuebar+numu+numubar', 'nutau+nutaubar'])
    tr_d3_2 = d3.transform_groups([NuFlavIntGroup('nue+nuebar+numu+numubar'),
                                   NuFlavIntGroup('nutau+nutaubar')])
    logging.debug(str((tr_d3)))
    assert tr_d3 == tr_d3_1 and tr_d3 == tr_d3_2

    try:
        tr_d3.transform_groups(['nue+nuebar'])
    except (ValueError, AssertionError):
        pass
    else:
        raise Exception

    try:
        tr_d3.transform_groups('nue+nuebar, numu+numubar, nutau+nutaubar')
    except (ValueError, AssertionError):
        pass
    else:
        raise Exception

    logging.info('<< PASS : test_FlavIntDataGroup >>')


if __name__ == "__main__":
    set_verbosity(1)
    test_IntType()
    test_NuFlav()
    test_NuFlavInt()
    test_NuFlavIntGroup()
    test_FlavIntData()
    test_FlavIntDataGroup()
