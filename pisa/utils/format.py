# -*- coding: utf-8 -*-

"""
Utilities for interpreting and returning formatted strings.
"""


from __future__ import absolute_import, division, print_function

from collections.abc import Iterable, Sequence
from collections import OrderedDict
import decimal
from numbers import Integral, Number
import re
import time

import numpy as np
import uncertainties

from pisa import FTYPE, ureg
from pisa.utils.flavInt import NuFlavIntGroup
from pisa.utils.log import logging, set_verbosity


__all__ = [
    'WHITESPACE_RE', 'NUMBER_RESTR', 'NUMBER_RE', 'HRGROUP_RESTR',
    'HRGROUP_RE', 'IGNORE_CHARS_RE', 'TEX_BACKSLASH_CHARS',
    'TEX_SPECIAL_CHARS_MAPPING', 'SI_PREFIX_TO_ORDER_OF_MAG',
    'ORDER_OF_MAG_TO_SI_PREFIX', 'BIN_PREFIX_TO_POWER_OF_1024',
    'POWER_OF_1024_TO_BIN_PREFIX', 'split', 'hr_range_formatter',
    'test_hr_range_formatter', 'list2hrlist', 'test_list2hrlist',
    'hrlist2list', 'hrlol2lol', 'hrbool2bool', 'engfmt', 'text2tex',
    'tex_join', 'tex_dollars', 'default_map_tex', 'is_tex', 'int2hex',
    'hash2hex', 'strip_outer_dollars', 'strip_outer_parens',
    'make_valid_python_name', 'sep_three_tens', 'format_num',
    'test_format_num', 'timediff', 'test_timediff', 'timestamp',
    'test_timestamp', 'arg_str_seq_none'
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


WHITESPACE_RE = re.compile(r'\s')
NUMBER_RESTR = r'((?:-|\+){0,1}[0-9.]+(?:e(?:-|\+)[0-9.]+){0,1})'
"""RE str for matching signed, unsigned, and sci.-not. ("1e10") numbers."""

NUMBER_RE = re.compile(NUMBER_RESTR, re.IGNORECASE)
"""Regex for matching signed, unsigned, and sci.-not. ("1e10") numbers."""

# Optional range, e.g., --10 (which means "to negative 10"); in my
# interpretation, the "to" number should be *INCLUDED* in the list
# If there's a range, optional stepsize, e.g., --10 (which means
# "to negative 10")
HRGROUP_RESTR = (
    NUMBER_RESTR
    + r'(?:-' + NUMBER_RESTR
    + r'(?:\:' + NUMBER_RESTR + r'){0,1}'
    + r'){0,1}'
)
HRGROUP_RE = re.compile(HRGROUP_RESTR, re.IGNORECASE)

# Characters to ignore are anything EXCEPT the characters we use
# (the caret ^ inverts the set in the character class)
IGNORE_CHARS_RE = re.compile(r'[^0-9e:.,;+-]', re.IGNORECASE)

TEX_BACKSLASH_CHARS = '%$#_{}'
TEX_SPECIAL_CHARS_MAPPING = {
    '~': r'\textasciitilde',
    '^': r'\textasciicircum',
    ' ': r'\;',
    'sin': r'\sin',
    'cos': r'\cos',
    'tan': r'\tan',
    #'sqrt': r'\sqrt{\,}'
    #'sqrt': r'\surd'
}

ORDER_OF_MAG_TO_SI_PREFIX = OrderedDict([
    (-24, 'y'),
    (-21, 'z'),
    (-18, 'a'),
    (-15, 'f'),
    (-6, 'μ'),
    (-3, 'm'),
    (-9, 'n'),
    (-12, 'p'),
    (0, ''),
    (3, 'k'),
    (6, 'M'),
    (9, 'G'),
    (12, 'T'),
    (15, 'P'),
    (18, 'E'),
    (21, 'Z'),
    (24, 'Y')
])
"""Mapping of powers-of-10 to SI prefixes (orders-of-magnitude)"""

SI_PREFIX_TO_ORDER_OF_MAG = OrderedDict()
"""Mapping of SI prefixes to powers-of-10"""
for K, V in ORDER_OF_MAG_TO_SI_PREFIX.items():
    SI_PREFIX_TO_ORDER_OF_MAG[V] = K
# Allow "u" to map to -6 (micro) as well
SI_PREFIX_TO_ORDER_OF_MAG['u'] = -6

POWER_OF_1024_TO_BIN_PREFIX = OrderedDict([
    (0, ''),
    (1, 'Ki'),
    (2, 'Mi'),
    (3, 'Gi'),
    (4, 'Ti'),
    (5, 'Pi'),
    (6, 'Ei'),
    (7, 'Zi'),
    (8, 'Yi')
])
"""Mapping from powers-of-1024 to binary prefixes"""

BIN_PREFIX_TO_POWER_OF_1024 = OrderedDict()
"""Mapping from binary prefixes to powerorders-of-1024"""
for K, V in POWER_OF_1024_TO_BIN_PREFIX.items():
    BIN_PREFIX_TO_POWER_OF_1024[V] = K


def split(string, sep=',', force_case=None, parse_func=None):
    """Parse a string containing a separated list.

    * Before splitting the list, the string has extraneous whitespace removed
      from either end.
    * The strings that result after the split can have their case forced or be
      left alone.
    * Whitespace surrounding (but not falling between non-whitespace) in each
      resulting string is removed.
    * After all of the above, the value can be parsed further by a
      user-supplied `parse_func`.

    Note that repeating a separator without intervening values yields
    empty-string values.


    Parameters
    ----------
    string : string
        The string to be split

    sep : string
        Separator to look for

    force_case : None, 'lower', or 'upper'
        Whether to force the case of the resulting items: None does not change
        the case, while 'lower' or 'upper' change the case.

    parse_func : None or callable
        If a callable is supplied, each item in the list, after the basic
        parsing, is processed by `parse_func`.

    Returns
    -------
    lst : list of objects
        The types of the items in the list depend upon `parse_func` if it is
        supplied; otherwise, all items are strings.

    Examples
    --------
    >>> print(split(' One, TWO, three ', sep=',', force_case='lower'))
    ['one', 'two', 'three']

    >>> print(split('One:TWO:three', sep=':'))
    ['One', 'TWO', 'three']

    >>> print(split('one  two  three', sep=' '))
    ['one', '', 'two', '' , 'three']

    >>> print(split('1 2 3', sep=' ', parse_func=int))
    [1, 2, 3]

    >>> from ast import literal_eval
    >>> print(split('True; False; None; (1, 2, 3)', sep=',',
    >>>             parse_func=literal_eval))
    [True, False, None, (1, 2, 3)]

    """
    funcs = []
    if force_case == 'lower':
        funcs.append(str.lower)
    elif force_case == 'upper':
        funcs.append(str.upper)

    if parse_func is not None:
        if not callable(parse_func):
            raise TypeError('`parse_func` must be callable; got %s instead.'
                            % type(parse_func))
        funcs.append(parse_func)

    if not funcs:
        aggfunc = lambda x: x
    elif len(funcs) == 1:
        aggfunc = funcs[0]
    elif len(funcs) == 2:
        aggfunc = lambda x: funcs[1](funcs[0](x))

    return [aggfunc(x.strip()) for x in str.split(str(string).strip(), sep)]

def arg_str_seq_none(inputs, name):
    """Simple input handler.
    Parameters
    ----------
    inputs : None, string, or iterable of strings
        Input value(s) provided by caller
    name : string
        Name of input, used for producing a meaningful error message
    Returns
    -------
    inputs : None, or list of strings
    Raises
    ------
    TypeError if unrecognized type
    """
    if isinstance(inputs, str):
        inputs = [inputs]
    elif isinstance(inputs, (Iterable, Sequence)):
        inputs = list(inputs)
    elif inputs is None:
        pass
    else:
        raise TypeError('Input %s: Unhandled type %s' % (name, type(inputs)))
    return inputs

# TODO: allow for scientific notation input to hr*2list, etc.

def hr_range_formatter(start, end, step):
    """Format a range (sequence) in a simple and human-readable format by
    specifying the range's starting number, ending number (inclusive), and step
    size.

    Parameters
    ----------
    start, end, step : numeric

    Notes
    -----
    If `start` and `end` are integers and `step` is 1, step size is omitted.

    The format does NOT follow Python's slicing syntax, in part because the
    interpretation is meant to differ; e.g.,
        '0-10:2' includes both 0 and 10 with step size of 2
    whereas
        0:10:2 (slicing syntax) excludes 10

    Numbers are converted to integers if they are equivalent for more compact
    display.

    Examples
    --------
    >>> hr_range_formatter(start=0, end=10, step=1)
    '0-10'
    >>> hr_range_formatter(start=0, end=10, step=2)
    '0-10:2'
    >>> hr_range_formatter(start=0, end=3, step=8)
    '0-3:8'
    >>> hr_range_formatter(start=0.1, end=3.1, step=1.0)
    '0.1-3.1:1'

    """
    if int(start) == start:
        start = int(start)
    if int(end) == end:
        end = int(end)
    if int(step) == step:
        step = int(step)
    if int(start) == start and int(end) == end and step == 1:
        return '{}-{}'.format(start, end)
    return '{}-{}:{}'.format(start, end, step)


def test_hr_range_formatter():
    """Unit tests for hr_range_formatter"""
    logging.debug(str((hr_range_formatter(start=0, end=10, step=1))))
    logging.debug(str((hr_range_formatter(start=0, end=10, step=2))))
    logging.debug(str((hr_range_formatter(start=0, end=3, step=8))))
    logging.debug(str((hr_range_formatter(start=0.1, end=3.1, step=1.0))))
    logging.info('<< PASS : test_hr_range_formatter >>')


def list2hrlist(lst):
    """Convert a list of numbers to a compact and human-readable string.

    Parameters
    ----------
    lst : sequence

    Notes
    -----
    Adapted to make scientific notation work correctly from [1].

    References
    ----------
    [1] http://stackoverflow.com/questions/9847601 user Scott B's adaptation to
        Python 2 of Rik Poggi's answer to his question

    Examples
    --------
    >>> list2hrlist([0, 1])
    '0,1'
    >>> list2hrlist([0, 3])
    '0,3'
    >>> list2hrlist([0, 1, 2])
    '0-2'
    >>> utils.list2hrlist([0.1, 1.1, 2.1, 3.1])
    '0.1-3.1:1'
    >>> list2hrlist([0, 1, 2, 4, 5, 6, 20])
    '0-2,4-6,20'

    """
    if isinstance(lst, Number):
        lst = [lst]
    lst = sorted(lst)
    rtol = np.finfo(FTYPE).resolution
    n = len(lst)
    result = []
    scan = 0
    while n - scan > 2:
        step = lst[scan + 1] - lst[scan]
        if not np.isclose(lst[scan + 2] - lst[scan + 1], step, rtol=rtol):
            result.append(str(lst[scan]))
            scan += 1
            continue
        for j in range(scan+2, n-1):
            if not np.isclose(lst[j+1] - lst[j], step, rtol=rtol):
                result.append(hr_range_formatter(lst[scan], lst[j], step))
                scan = j+1
                break
        else:
            result.append(hr_range_formatter(lst[scan], lst[-1], step))
            return ','.join(result)
    if n - scan == 1:
        result.append(str(lst[scan]))
    elif n - scan == 2:
        result.append(','.join(map(str, lst[scan:])))

    return ','.join(result)


def test_list2hrlist():
    """Unit tests for list2hrlist"""
    logging.debug(str((list2hrlist([0, 1]))))
    logging.debug(str((list2hrlist([0, 1, 2]))))
    logging.debug(str((list2hrlist([0.1, 1.1, 2.1, 3.1]))))
    logging.info('<< PASS : test_list2hrlist >>')


def _hrgroup2list(hrgroup):
    def isint(num):
        """Test whether a number is *functionally* an integer"""
        try:
            return int(num) == FTYPE(num)
        except ValueError:
            return False

    def num_to_float_or_int(num):
        """Return int if number is effectively int, otherwise return float"""
        try:
            if isint(num):
                return int(num)
        except (ValueError, TypeError):
            pass
        return FTYPE(num)

    # Strip all whitespace, brackets, parens, and other ignored characters from
    # the group string
    hrgroup = IGNORE_CHARS_RE.sub('', hrgroup)
    if (hrgroup is None) or (hrgroup == ''):
        return []
    num_str = HRGROUP_RE.match(hrgroup).groups()
    range_start = num_to_float_or_int(num_str[0])

    # If no range is specified, just return the number
    if num_str[1] is None:
        return [range_start]

    range_stop = num_to_float_or_int(num_str[1])
    if num_str[2] is None:
        step_size = 1 if range_stop >= range_start else -1
    else:
        step_size = num_to_float_or_int(num_str[2])
    all_ints = isint(range_start) and isint(step_size)

    # Make an *INCLUSIVE* list (as best we can considering floating point mumbo
    # jumbo)
    n_steps = np.clip(
        np.floor(np.around(
            (range_stop - range_start)/step_size,
            decimals=12,
        )),
        a_min=0, a_max=np.inf
    )
    lst = np.linspace(range_start, range_start + n_steps*step_size, n_steps+1)
    if all_ints:
        lst = lst.astype(np.int)

    return lst.tolist()


def hrlist2list(hrlst):
    """Convert human-readable string specifying a list of numbers to a Python
    list of numbers.

    Parameters
    ----------
    hrlist : string

    Returns
    -------
    lst : list of numbers

    """
    groups = re.split(r'[,; _]+', WHITESPACE_RE.sub('', hrlst))
    lst = []
    if not groups:
        return lst
    for group in groups:
        lst.extend(_hrgroup2list(group))
    return lst


def hrlol2lol(hrlol):
    """Convert a human-readable string specifying a list-of-lists of numbers to
    a Python list-of-lists of numbers.

    Parameters
    ----------
    hrlol : string
        Human-readable list-of-lists-of-numbers string. Each list specification
        is separated by a semicolon, and whitespace is ignored. Refer to
        `hrlist2list` for list specification.

    Returns
    -------
    lol : list-of-lists of numbers

    Examples
    --------
    A single number evaluates to a list with a list with a single number.

    >>>  hrlol2lol("1")
    [[1]]

    A sequence of numbers or ranges can be specified separated by commas.

    >>>  hrlol2lol("1, 3.2, 19.8")
    [[1, 3.2, 19.8]]

    A range can be specified with a dash; default is a step size of 1 (or -1 if
    the end of the range is less than the start of the range); note that the
    endpoint is included, unlike slicing in Python.

    >>>  hrlol2lol("1-3")
    [[1, 2, 3]]

    The range can go from or to a negative number, and can go in a negative
    direction.

    >>>  hrlol2lol("-1 - -5")
    [[-1, -3, -5]]

    Multiple lists are separated by semicolons, and parentheses and brackets
    can be used to make it easier to understand the string.

    >>>  hrlol2lol("1 ; 8 ; [(-10 - -8:2), 1]")
    [[1], [8], [-10, -8, 1]]

    Finally, all of the above can be combined.

    >>>  hrlol2lol("1.-3.; 9.5-10.6:0.5,3--1:-1; 12.5-13:0.8")
    [[1, 2, 3], [9.5, 10.0, 10.5, 3, 2, 1, 0, -1], [12.5]]

    """
    supergroups = re.split(r'[;]+', hrlol)
    return [hrlist2list(group) for group in supergroups]


def hrbool2bool(s):
    """Convert a string that a user might input to indicate a boolean value of
    either True or False and convert to the appropriate Python bool.

    * Note first that the case used in the string is ignored
    * 't', 'true', '1', 'yes', and 'one' all map to True
    * 'f', 'false', '0', 'no', and 'zero' all map to False

    Parameters
    ----------
    s : string

    Returns
    -------
    b : bool

    """
    s = str(s).strip()
    if s.lower() in ['t', 'true', '1', 'yes', 'one']:
        return True
    if s.lower() in ['f', 'false', '0', 'no', 'zero']:
        return False
    raise ValueError('Could not parse input "%s" to bool.' % s)


def engfmt(n, sigfigs=3, decimals=None, sign_always=False):
    """Format number as string in engineering format (10^(multiples-of-three)),
    including the most common metric prefixes (from atto to Exa).

    Parameters
    ----------
    n : scalar
        Number to be formatted
    sigfigs : int >= 0
        Number of significant figures to limit the result to; default=3.
    decimals : int or None
        Number of decimals to display (zeros filled out as necessary). If None,
        `decimals` is automatically determined by the magnitude of the
        significand and the specified `sigfigs`.
    sign_always : bool
        Prefix the number with "+" sign if number is positive; otherwise,
        only negative numbers are prefixed with a sign ("-")

    """
    if isinstance(n, ureg.Quantity):
        units = n.units
        n = n.magnitude
    else:
        units = ureg.dimensionless

    # Logs don't like negative numbers...
    sign = np.sign(n)
    n *= sign

    mag = int(np.floor(np.log10(n)))
    pfx_mag = int(np.floor(np.log10(n)/3.0)*3)

    if decimals is None:
        decimals = sigfigs-1 - (mag-pfx_mag)
    decimals = int(np.clip(decimals, a_min=0, a_max=np.inf))

    round_to = decimals
    if sigfigs is not None:
        round_to = sigfigs-1 - (mag-pfx_mag)

    scaled_rounded = np.round(n/10.0**pfx_mag, round_to)

    sign_str = ''
    if sign_always and sign > 0:
        sign_str = '+'
    num_str = sign_str + format(sign*scaled_rounded, '.'+str(decimals)+'f')

    # Very large or small quantities have their order of magnitude displayed
    # by printing the exponent rather than showing a prefix; due to my
    # inability to strip off prefix in Pint quantities (and attach my own
    # prefix), just use the "e" notation.
    if pfx_mag not in ORDER_OF_MAG_TO_SI_PREFIX or not units.dimensionless:
        if pfx_mag == 0:
            return str.strip('{0:s} {1:~} '.format(num_str, units))
        return str.strip('{0:s}e{1:d} {2:~} '.format(num_str, pfx_mag, units))

    # Dimensionless quantities are treated separately since Pint apparently
    # can't handle prefixed-dimensionless (e.g., simply "1 k", "2.2 M", etc.,
    # with no units attached).
    #if units.dimensionless:
    return  '{0:s} {1:s}'.format(num_str, ORDER_OF_MAG_TO_SI_PREFIX[pfx_mag])


def append_results(results_dict, result_dict):
    for key, val in result_dict.items():
        if key in results_dict:
            results_dict[key].append(val)
        else:
            results_dict[key] = [val]


def ravel_results(results):
    for key, val in results.items():
        if hasattr(val[0], 'm'):
            results[key] = np.array([v.m for v in val]) * val[0].u

# TODO: mathrm vs. rm?
def text2tex(txt):
    """Convert common characters so they show up the same as TeX"""
    if txt is None:
        return ''

    if is_tex(txt):
        return strip_outer_dollars(txt)

    nfig = NuFlavIntGroup(txt)
    if nfig:
        return nfig.tex

    for c in TEX_BACKSLASH_CHARS:
        txt = txt.replace(c, r'\%s'%c)

    for c, v in TEX_SPECIAL_CHARS_MAPPING.items():
        txt = txt.replace(c, '{%s}'%v)

    # A single character is taken to be a variable name, and so do not make
    # roman, just wrap in braces (to avoid interference with other characters)
    # and return
    if len(txt) == 1:
        return '%s' % txt

    return r'{\rm %s}' % txt


def tex_join(sep, *args):
    """Join TeX-formatted strings together into one, each separated by `sep`.
    Also, this strips surrounding '$' from each string before joining."""
    strs = [strip_outer_dollars(text2tex(a))
            for a in args if a is not None and a != '']
    if not strs:
        return ''
    return str.join(sep, strs)


def tex_dollars(s):
    stripped = strip_outer_dollars(s)
    out_lines = []
    for line in stripped.split('\n'):
        stripped_line = strip_outer_dollars(line)
        if stripped_line == '':
            out_lines.append('')
        else:
            out_lines.append('$%s$' % stripped_line)
    return '\n'.join(out_lines)


def is_tex(s):
    if s is None:
        return False
    for c in TEX_BACKSLASH_CHARS:
        if '\\'+c in s:
            return True
    for seq in TEX_SPECIAL_CHARS_MAPPING.values():
        if seq in s:
            return True
    for seq in [r'\rm', r'\mathrm', r'\theta', r'\phi']:
        if seq in s:
            return True
    if strip_outer_dollars(s) != s:
        return True
    return False


def default_map_tex(map):
    if map.tex is None or map.tex == '':
        return r'{\rm %s}' % text2tex(map.name)
    return strip_outer_dollars(map.tex)


def int2hex(i, bits, signed):
    """Convert a signed or unsigned integer `bits` long to hexadecimal
    representation. As many hex characters are returned to fully specify any
    number `bits` in length regardless of the value of `i`.

    Parameters
    ----------
    i : int
        The integer to be converted. Signed integers have a range of
        -2**(bits-1) to 2**(bits-1)-1), while unsigned integers have a range of
        0 to 2**(bits-1).

    bits : int
        Number of bits long the representation is

    signed : bool
        Whether the number is a signed integer; this is dependent upon the
        representation used for numbers, and _not_ whether the value `i` is
        positive or negative.

    Returns
    -------
    h : string of length ceil(bits/4.0) since it takes this many hex characters
    to represent a number `bits` long.

    """
    if signed:
        i = 2**63 + i
    assert i >= 0
    h = hex(i)[2:].replace('L', '')
    return h.rjust(int(np.ceil(bits/4.0)), '0')


def hash2hex(hash, bits=64):
    """Convert a hash value to its string hexadecimal representation.

    Parameters
    ----------
    hash : integer or string
    bits : integer > 0

    Returns
    -------
    hash : string

    """
    if isinstance(hash, str):
        assert len(hash) == int(np.ceil(bits/4.0))
        hex_hash = hash
    elif isinstance(hash, int):
        hex_hash = int2hex(hash, bits=bits, signed=True)
    else:
        raise TypeError('Unhandled `hash` type %s' %type(hash))
    return hex_hash


def strip_outer_dollars(value):
    """Strip surrounding dollars signs from TeX string, ignoring leading and
    trailing whitespace"""
    if value is None:
        return '{}'
    value = value.strip()
    m = re.match(r'^\$(.*)\$$', value)
    if m is not None:
        value = m.groups()[0]
    return value


def strip_outer_parens(value):
    """Strip parentheses surrounding a string, ignoring leading and trailing
    whitespace"""
    if value is None:
        return ''
    value = value.strip()
    m = re.match(r'^\{\((.*)\)\}$', value)
    if m is not None:
        value = m.groups()[0]
    m = re.match(r'^\((.*)\)$', value)
    if m is not None:
        value = m.groups()[0]
    return value


# TODO: this is relatively slow (and is called in constructors that are used
# frequently, e.g. OneDimBinning, MultiDimBinning); can we speed it up any?
RE_INVALID_CHARS = re.compile('[^0-9a-zA-Z_]')
RE_LEADING_INVALID = re.compile('^[^a-zA-Z_]+')
def make_valid_python_name(name):
    """Make a name a valid Python identifier.

    From user Triptych at http://stackoverflow.com/questions/3303312

    """
    # Remove invalid characters
    name = RE_INVALID_CHARS.sub('', name)
    # Remove leading characters until we find a letter or underscore
    name = RE_LEADING_INVALID.sub('', name)
    return name


def sep_three_tens(strval, direction, sep=None):
    """Insert `sep` char into sequence of chars `strval`.

    Parameters
    ----------
    strval : sequence of chars or string
        Sequence of chars into which to insert the separator

    direction : string, one of {'left', 'right'}
        Use 'left' for left of the decimal, and 'right' for right of the
        decimal

    sep : None or char
        Separator to insert

    Returns
    -------
    formatted : list of chars

    """
    if not sep:
        return strval

    direction = direction.strip().lower()
    assert direction in ('left', 'right'), direction

    formatted = []
    if direction == 'left':
        indices = tuple(range(len(strval)-1, -1, -1))
        edge_indices = (indices[0], indices[-1])
        delta = len(strval)-1
        for c_num in indices:
            formatted = [strval[c_num]] + formatted
            if (((delta-c_num)+1) % 3 == 0) and c_num not in edge_indices:
                formatted = [sep] + formatted
        return formatted

    indices = tuple(range(len(strval)))
    edge_indices = (indices[0], indices[-1])
    for c_num in indices:
        formatted = formatted + [strval[c_num]]
        if ((c_num+1) % 3 == 0) and (c_num not in edge_indices):
            formatted = formatted + [sep]

    return formatted


def format_num(
    value,
    sigfigs=None,
    precision=None,
    fmt=None,
    sci_thresh=(6, -4),
    exponent=None,
    inf_thresh=np.infty,
    trailing_zeros=False,
    always_show_sign=False,
    decstr='.',
    thousands_sep=None,
    thousandths_sep=None,
    left_delimiter=None,
    right_delimiter=None,
    expprefix=None,
    exppostfix=None,
    nanstr='nan',
    infstr='inf',
):
    r"""Fine-grained control over formatting a number as a string.


    Parameters
    ----------
    value : numeric
        The number to be formatted.

    sigfigs : int > 0, optional
        Use up to this many significant figures for displaying a number. You
        can use either `sigfigs` or `precision`, but not both. If neither are
        specified, default is to set `sigfigs` to 8. See also `trailing_zeros`.

    precision : float, optional
        Round `value` to a precision the same as the order of magnitude of
        `precision`. You can use either `precision` or `sigfigs`, but not both.
        If neither is specified, default is to set `sigfigs` to 8. See also
        `trailing_zeros`.

    fmt : None or one of {'sci', 'eng', 'sipre', 'binpre', 'full'}, optional
        Force a particular format to be used::
            * None allows the `value` and what is passed for `sci_thresh` and
              `exponent` to decide whether or not to use scientific notation
            * 'sci' forces scientific notation
            * 'eng' uses the engineering convention of powers divisible by 3
              (10e6, 100e-9, etc.)
            * 'sipre' uses powers divisible by 3 but uses SI prefixes (e.g. k,
              M, G, etc.) instead of displaying the exponent
            * 'binpre' uses powers of 1024 and uses IEC prefixes (e.g. Ki, Mi,
              Gi, etc.) instead displaying the exponent
            * 'full' forces printing all digits left and/or right of the
              decimal to display the number (no exponent notation or SI/binary
              prefix will be used)
        Note that the display of NaN and +/-inf are unaffected by
        `fmt`.

    exponent : None, integer, or string, optional
        Force the number to be scaled with respect to this exponent. If a
        string prefix is passed and `fmt` is None, then the SI prefix
        or binary prefix will be used for the number. E.g., ``exponent=3``
        would cause the number 1 to be expressed as ``'0.001e3'`, while
        ``exponent='k'`` would cause it to be displayed as ``'1 m'``. Both 'μ'
        and 'u' are accepted to mean "micro". A non-``None`` value for
        `exponent` forces some form of scientific/engineering notation, so
        `fmt` cannot be ``'full'`` in this case. Finally, if
        `fmt` is ``'binpre'`` then `exponent` is applied to 1024.
        I.e., 1 maps to kibi (Ki), 2 maps to mebi (Mi), etc.

    sci_thresh : sequence of 2 integers
        When to switch to scientific notation. The first integer is the order
        of magnitude of `value` at or above which scientific notation will be
        used. The second integer indicates the order of magnitude at or below
        which the most significant digit falls for scientific notation to be
        used. E.g., ``sci_thresh=(3, -3)`` means that numbers in the
        ones-of-thousands or greater OR numbers in the ones-of-thousandths or
        less will be displayed using scientific notation. Note that
        `fmt`, if not None, overrides this behavior. Default is
        (10,-5).

    inf_thresh : numeric, optional
        Numbers whose magnitude is equal to or greater than this threhshold are
        considered infinite and therefore displayed using `infstr` (possibly
        including a sign, as appropriate). Default is np.inf.

    trailing_zeros : bool, optional
        Whether to display all significant figures specified by `sigfigs`, even
        if this results in trailing zeros. Default is False.

    always_show_sign : bool, optional
        Always show a sign, whether number is positive or negative, and whether
        exponent (if present) is positive or negative. Default is False.

    decstr : string, optional
        Separator to use for the decimal point. E.g. ``decstr='.'`` or
        ``decstr=','`` for mthe most common cases, but this can also be used in
        TeX tables for alignment on decimal points via ``decstr='&.&'``.
        Default is '.'.

    thousands_sep : None or string, optional
        Separator to use between thousands, e.g. ``thousands_sep=','`` to give
        results like ``'1,000,000'``, or ```thousands_sep=r'\,'`` for TeX
        formatting with small spaces between thousands. Default is None.

    thousandths_sep : None or string, optional
        Separator to use between thousandthss. Default is None.

    left_delimiter, right_delimiter : None or string, optional
        Strings to delimit the left and right sides of the resulting string.
        E.g. ``left_delimiter='${'`` and ``right_delimiter='}$'`` could be used
        to delimit TeX-formatted strings, such that a number is displayed,
        e.g., as ``r'${1\times10^3}$'``. Defaults are None for both.

    expprefix, exppostfix : None or string, optional
        E.g. use `expprefix='e'` for simple "e" scientific notation ("1e3"),
        or use `expprefix=r'\times10^{'` and `exppostfix=r'}' for
        TeX-formatted scientific notation. Use a space (or tex equivalent) for
        binary and SI prefixes. If scientific notation is to be used,
        `expprefix` defaults to 'e'. If either SI or binary prefixes are to be
        used, `expprefix` defaults to ' ' (space). In any case, `exppostfix`
        defaults to None.

    nanstr : string, optional
        Not-a-number (NaN) values will be displayed using this string. Default
        is 'nan' (following the Numpy convention)

    infstr : string, optional
        Infinite values will be displayed using this string (note that the sign
        is prepended, as appropriate). Default is 'inf' (following the Numpy
        convention).


    Returns
    -------
    formatted : string

    """
    with decimal.localcontext() as context:
        # Ensure rounding behavior is same as that of Numpy
        context.rounding = decimal.ROUND_HALF_EVEN
        # Lots of comp precision to avoid floating point <--> decimal issues
        context.prec = 72
        d_10 = decimal.Decimal('10')
        d_1024 = decimal.Decimal('1024')

        if sigfigs is None:
            if precision is None:
                sigfigs = 8
            else:
                precision = decimal.Decimal(precision)
                order_of_precision = precision.adjusted()
        else:
            if precision is not None:
                raise ValueError('You cannot specify both `sigfigs` and'
                                 ' `precision`')
            if not isinstance(sigfigs, Integral):
                assert float(sigfigs) == int(sigfigs), \
                        '`sigfigs`=%s not an int' % sigfigs
                sigfigs = int(sigfigs)
            assert sigfigs > 0, '`sigfigs`=%s is not > 0' % sigfigs

        if sci_thresh[0] < sci_thresh[1]:
            raise ValueError(
                '(`sci_thresh[0]`=%s) must be >= (`sci_thresh[1]`=%s)'
                % sci_thresh
            )
        assert all(isinstance(s, Integral) for s in sci_thresh), str(sci_thresh)

        if isinstance(fmt, str):
            fmt = fmt.strip().lower()
        assert fmt is None or fmt in ('sci', 'eng', 'sipre', 'binpre', 'full')
        if fmt == 'full':
            assert exponent is None

        if exponent is not None:
            if fmt in ('eng', 'sipre'):
                if (exponent not in SI_PREFIX_TO_ORDER_OF_MAG
                        and exponent not in ORDER_OF_MAG_TO_SI_PREFIX):
                    raise ValueError(
                        'For `fmt`="{}", `exponent` is {}, but must either be'
                        ' an SI prefix {} or a power of 10 corresponding to'
                        ' these {}.'.format(fmt,
                                            exponent,
                                            SI_PREFIX_TO_ORDER_OF_MAG.keys(),
                                            ORDER_OF_MAG_TO_SI_PREFIX.keys())
                    )
            elif fmt == 'binpre':
                if (exponent not in BIN_PREFIX_TO_POWER_OF_1024
                        and exponent not in POWER_OF_1024_TO_BIN_PREFIX):
                    raise ValueError(
                        'For `fmt`="{}", `exponent` is {}, but must either be'
                        ' an IEC binary prefix {} or a power of 1024'
                        ' corresponding to these {}.'.format(
                            fmt,
                            exponent,
                            BIN_PREFIX_TO_POWER_OF_1024.keys(),
                            POWER_OF_1024_TO_BIN_PREFIX.keys()
                        )
                    )
            if (not isinstance(exponent, str) and not
                    isinstance(exponent, Integral)):
                assert float(exponent) == int(exponent)
                exponent = int(exponent)

        # TODO: include uncertainties and/or units in final formatted string
        # TODO: scale out SI prefix if `value` is a Pint Quantity

        # Strip off units, if present
        units = None
        quantity_info = None
        if isinstance(value, ureg.Quantity):
            units = value.units if value.units != ureg.dimensionless else None
            quantity_info = value.as_tuple()
            value = value.magnitude

        # Strip off uncertainty, if present
        stddev = None
        if isinstance(value, uncertainties.UFloat):
            stddev = value.std_dev
            value = value.nominal_value

        # In case `value` is a singleton array
        if isinstance(value, np.ndarray):
            value = np.asscalar(value)

        # Fill in empty strings where None might be passed in to mean the same
        thousands_sep = '' if thousands_sep is None else thousands_sep
        thousandths_sep = '' if thousandths_sep is None else thousandths_sep
        left_delimiter = '' if left_delimiter is None else left_delimiter
        right_delimiter = '' if right_delimiter is None else right_delimiter
        exppostfix = '' if exppostfix is None else exppostfix
        # NOTE: expprefix defaults depend on the display mode, so are set later

        if np.isnan(value):
            return left_delimiter + nanstr + right_delimiter

        if np.isneginf(value) or value <= -inf_thresh:
            return left_delimiter + '-' + infstr + right_delimiter

        # NOTE: ``isinf`` check must come _after_ ``neginf`` check since
        # ``isinf`` returns ``True`` for both -inf and +inf
        if np.isinf(value) or value >= inf_thresh:
            if always_show_sign:
                sign = '+'
            else:
                sign = ''
            return left_delimiter + sign + infstr + right_delimiter

        if isinstance(value, Integral):
            value = decimal.Decimal(value)
        else:
            value = decimal.Decimal.from_float(float(value))

        order_of_mag = value.adjusted()
        # Get the sign from the full precision `value`, before rounding
        sign = ''
        if value < 0:
            sign = '-'
        elif value > 0:
            sign = '+'

        # If no value passed for `fmt`, infer the format from the
        # exponent (if it's a binary or SI prefix) OR the order of magnitude of
        # the number w.r.t. `sci_thresh`.
        if fmt is None:
            if isinstance(exponent, str):
                if exponent in BIN_PREFIX_TO_POWER_OF_1024:
                    fmt = 'binpre'
                elif exponent in SI_PREFIX_TO_ORDER_OF_MAG:
                    fmt = 'sipre'
                else:
                    raise ValueError('`exponent`="%s" is not a valid SI or'
                                     ' binary prefix' % exponent)
            elif exponent is None:
                if (order_of_mag >= sci_thresh[0]
                        or order_of_mag <= sci_thresh[1]):
                    fmt = 'sci'
                else:
                    fmt = 'full'
            else:
                fmt = 'sci'

        # Define `exponent` where appropriate, and calculate `scaled_value` to
        # account for the exponent, if there is one.
        scale = 1
        if exponent is None:
            if fmt == 'sci':
                exponent = order_of_mag
                scale = 1 / d_10**exponent
            elif fmt in ('eng', 'sipre'):
                exponent = (order_of_mag // 3) * 3
                scale = 1 / d_10**exponent
                if fmt == 'sipre':
                    exponent = ORDER_OF_MAG_TO_SI_PREFIX[exponent]
            elif fmt == 'binpre':
                if value < 0:
                    raise ValueError('Binary prefix valid only for value >= 0')
                elif value == 0:
                    exponent = 0
                    scale = 1
                else:
                    exponent = value.ln() // d_1024.ln()
                    scale = 1 / d_1024**exponent
                    exponent = POWER_OF_1024_TO_BIN_PREFIX[exponent]
        elif exponent in BIN_PREFIX_TO_POWER_OF_1024:
            scale = 1 / d_1024**BIN_PREFIX_TO_POWER_OF_1024[exponent]
        elif exponent in SI_PREFIX_TO_ORDER_OF_MAG:
            scale = 1 / d_10**SI_PREFIX_TO_ORDER_OF_MAG[exponent]
        else:
            scale = 1 / d_10**exponent

        scaled_value = scale * value

        if sigfigs is not None:
            leastsig_dig = scaled_value.adjusted() - (sigfigs - 1)
            quantize_at = decimal.Decimal('1e%d' % leastsig_dig).normalize()
        else: # only other case is that precision is specified
            quantize_at = (d_10**order_of_precision * scale).normalize()
            leastsig_dig = quantize_at.adjusted()

        rounded = scaled_value.quantize(quantize_at)

        # Eliminate trailing zeros in the Decimal representation
        if not trailing_zeros:
            rounded = rounded.normalize()

        dec_tup = rounded.as_tuple()
        mantissa_digits = dec_tup.digits
        decimal_position = dec_tup.exponent

        # Does the number underflow, making it effectively 0?
        underflow = False
        if sigfigs is not None:
            if len(mantissa_digits) + decimal_position < -sigfigs:
                underflow = True
                decimal_position = -(sigfigs - 1)
                mantissa_digits = (0,)
        else: # `precision` is specified
            if order_of_mag < order_of_precision:
                underflow = True
                mantissa_digits = (0,)
                decimal_position = leastsig_dig

        n_digits = len(mantissa_digits)
        chars = [str(d) for d in mantissa_digits]
        if decimal_position > 0:
            chars += ['0']*decimal_position
            chars = sep_three_tens(chars, direction='left', sep=thousands_sep)
        elif decimal_position < 0:
            if abs(decimal_position) >= n_digits:
                chars = (
                    ['0', decstr]
                    + sep_three_tens(
                        ['0']*(-decimal_position - n_digits) + chars,
                        direction='right', sep=thousandths_sep
                    )
                )
            else:
                chars = (
                    sep_three_tens(chars[:decimal_position], direction='left',
                                   sep=thousands_sep)
                    + [decstr]
                    + sep_three_tens(chars[decimal_position:],
                                     direction='right', sep=thousandths_sep)
                )

        num_str = ''.join(chars)

        if always_show_sign or sign == '-' or underflow:
            num_str = sign + num_str

        if exponent is not None:
            if expprefix is None:
                if fmt in ('sci', 'eng'):
                    expprefix = 'e'
                elif fmt in ('sipre', 'binpre'):
                    expprefix = ' '
                else:
                    expprefix = ''

            if not isinstance(exponent, str):
                if fmt == 'sipre':
                    exponent = ORDER_OF_MAG_TO_SI_PREFIX[exponent]
                elif fmt == 'binpre':
                    exponent = POWER_OF_1024_TO_BIN_PREFIX[exponent]

            if isinstance(exponent, str):
                num_str += expprefix + exponent + exppostfix
            else:
                if exponent < 0:
                    exp_sign = ''
                elif always_show_sign:
                    exp_sign = '+'
                else:
                    exp_sign = ''
                num_str += expprefix + exp_sign + str(exponent) + exppostfix

    return left_delimiter + num_str + right_delimiter


def test_format_num():
    """Unit tests for the `format_num` function"""
    # sci_thresh
    v = format_num(100, sci_thresh=(3, -3))
    assert v == '100'
    v = format_num(1000, sci_thresh=(3, -3))
    assert v == '1e3'
    v = format_num(0.01, sci_thresh=(3, -3))
    assert v == '0.01'
    v = format_num(0.001, sci_thresh=(3, -3))
    assert v == '1e-3'

    # trailing_zeros
    v = format_num(0.00010001, sigfigs=6, exponent=None, trailing_zeros=True)
    assert v == '1.00010e-4', v
    v = format_num(0.00010001, sigfigs=6, exponent=None, trailing_zeros=False)
    assert v == '1.0001e-4', v
    v = format_num(0.00010001, sigfigs=6, exponent=None, trailing_zeros=False,
                   sci_thresh=(7, -8))
    assert v == '0.00010001', v
    v = format_num(0.00010001, sigfigs=6, exponent=None, trailing_zeros=True,
                   sci_thresh=(7, -20))
    assert v == '0.000100010', v

    # sigfigs and trailing_zeros
    v = format_num(1, sigfigs=5, exponent=None, trailing_zeros=True)
    assert v == '1.0000', v
    v = format_num(1, sigfigs=5, exponent=None, trailing_zeros=False)
    assert v == '1', v
    v = format_num(16, sigfigs=5, exponent=None, trailing_zeros=False)
    assert v == '16', v
    v = format_num(160000, sigfigs=5, exponent=None, trailing_zeros=False)
    assert v == '160000', v
    v = format_num(123456789, sigfigs=15, exponent=None, trailing_zeros=False,
                   sci_thresh=(20, -20))
    assert v == '123456789', v
    v = format_num(1.6e6, sigfigs=5, exponent=None, trailing_zeros=False)
    assert v == '1.6e6', v

    # precision
    v = format_num(1.2345, precision=1e0, trailing_zeros=True)
    assert v == '1', v
    v = format_num(1.2345, precision=1e-1, trailing_zeros=True)
    assert v == '1.2', v

    # exponent
    v = format_num(1e6, sigfigs=5, exponent='k', trailing_zeros=False)
    assert v == '1000 k', v
    v = format_num(0.00134, sigfigs=5, exponent='m', trailing_zeros=False)
    assert v == '1.34 m', v
    v = format_num(1024, exponent='Ki')
    assert v == '1 Ki', v
    v = format_num(1024*1000, exponent='Ki')
    assert v == '1000 Ki', v
    v = format_num(1024**2, exponent='Mi')
    assert v == '1 Mi', v

    # displaying zero
    v = format_num(0, sigfigs=5, exponent=4, trailing_zeros=True)
    assert v == '0.0000e4', v
    v = format_num(0, sigfigs=5, exponent=4, trailing_zeros=False)
    assert v == '0e4', v
    v = format_num(0, sigfigs=5, exponent=None, trailing_zeros=True)
    assert v == '0.0000', v
    v = format_num(0, sigfigs=5, exponent=None, trailing_zeros=False)
    assert v == '0'
    v = format_num(0, sigfigs=5, fmt='sci')
    assert v == '0e0'
    v = format_num(0, sigfigs=5, fmt='eng')
    assert v == '0e0'
    v = format_num(0, sigfigs=5, fmt='sipre')
    assert v == '0 '
    v = format_num(0, sigfigs=5, fmt='binpre')
    assert v == '0 '
    v = format_num(0, sigfigs=5, fmt='full')
    assert v == '0'

    # exponent + sigfigs or precision causes underflow
    v = format_num(-0.00010001, sigfigs=6, exponent=4, trailing_zeros=True)
    assert v == '-0.00000e4', v
    v = format_num(0.00010001, sigfigs=6, exponent=4, trailing_zeros=True)
    assert v == '+0.00000e4', v
    v = format_num(-0.00010001, precision=1e-3, exponent=4, trailing_zeros=True)
    assert v == '-0.0000000e4', v
    v = format_num(0.00010001, precision=1e-3, exponent=4, trailing_zeros=True)
    assert v == '+0.0000000e4', v

    # exponent + precision, check sigfigs and trailing zeros...
    # zeros...
    v = format_num(-0.00010001, precision=1e-3, exponent=4,
                   trailing_zeros=True)
    assert v == '-0.0000000e4', v
    v = format_num(0.00010001, precision=1e-3, exponent=4, trailing_zeros=True)
    assert v == '+0.0000000e4', v
    # rounding at least sig digit
    v = format_num(-0.00015001, precision=1e-4, exponent=4, trailing_zeros=True)
    assert v == '-0.00000002e4', v
    v = format_num(0.00015001, precision=1e-4, exponent=4, trailing_zeros=True)
    assert v == '0.00000002e4', v
    # trailing zeros
    v = format_num(-0.015001, precision=1e-4, exponent=4, trailing_zeros=True)
    assert v == '-0.00000150e4', v
    v = format_num(0.015001, precision=1e-4, exponent=4, trailing_zeros=True)
    assert v == '0.00000150e4', v

    # Test thousands_sep and thousandths_sep
    v = format_num(1000.0001, sigfigs=10, trailing_zeros=True,
                   thousands_sep=',', thousandths_sep=' ')
    assert v == '1,000.000 100', v

    # Test specials: +/-inf, nan
    v = format_num(np.nan, sigfigs=10, trailing_zeros=True,
                   thousands_sep=',', thousandths_sep=' ')
    assert v == 'nan', v
    v = format_num(np.inf, infstr='INFINITY', always_show_sign=True)
    assert v == '+INFINITY', v
    v = format_num(-np.inf, infstr='INFINITY')
    assert v == '-INFINITY', v
    v = format_num(1000, inf_thresh=100)
    assert v == 'inf', v
    v = format_num(-1000, inf_thresh=100)
    assert v == '-inf', v

    # eng and sipre with exponent
    v = format_num(1000, exponent=6, precision=1e3)
    assert v == '0.001e6', v
    v = format_num(1000, exponent=6, precision=1e3, fmt='sipre')
    assert v == '0.001 M', v
    v = format_num(115e3, exponent=6, precision=1e5, fmt='sipre',
                   trailing_zeros=True)
    assert v == '0.1 M', v
    v = format_num(115e3, exponent=6, precision=1e4, fmt='sipre',
                   trailing_zeros=True)
    assert v == '0.12 M', v
    v = format_num(115e3, exponent=6, precision=1e3, fmt='sipre',
                   trailing_zeros=True)
    assert v == '0.115 M', v
    v = format_num(115e3, exponent=6, precision=1e2, fmt='sipre',
                   trailing_zeros=True)
    assert v == '0.1150 M', v
    v = format_num(115e3, exponent=6, precision=1e1, fmt='sipre',
                   trailing_zeros=True)
    assert v == '0.11500 M', v

    # TeX formatting (use exp{pre,post}fix, {left,right}_delimiter;
    # also fmt='eng', 'sipre', and 'binpre'
    v = format_num(
        value=12.5e3, sigfigs=4, trailing_zeros=True, fmt='eng',
        expprefix=r' \, \times 10^{',
        exppostfix='}',
        left_delimiter='${',
        right_delimiter='}$'
    )
    assert v == r'${12.50 \, \times 10^{3}}$', v
    v = format_num(
        value=12.5e3, sigfigs=4, trailing_zeros=False, fmt='sipre',
        expprefix=r' \, {\rm ',
        exppostfix='}',
        left_delimiter='${',
        right_delimiter='}$'
    )
    assert v == r'${12.5 \, {\rm k}}$', v
    v = format_num(
        value=12.5e3, sigfigs=4, trailing_zeros=False, fmt='binpre',
        expprefix=r' \, {\rm ',
        exppostfix='}',
        left_delimiter='${',
        right_delimiter='}$'
    )
    assert v == r'${12.21 \, {\rm Ki}}$', v

    # fmt='full'
    v = format_num(12.5e10, sigfigs=4, trailing_zeros=False,
                   fmt='full',)
    assert v == '125000000000', v

    # specify both fmt='full' AND exponent (should raise exception)
    try:
        v = format_num(
            12.5e3, sigfigs=4, trailing_zeros=False, fmt='full',
            exponent=0
        )
    except AssertionError:
        pass
    else:
        assert False, '`fmt`="full" and `exponent` is defined'

    logging.info('<< PASS : test_format_num >>')


def timediff(dt_sec, hms_always=False, sec_decimals=3):
    """Smart string formatting for a time difference (in seconds)

    Parameters
    ----------
    dt_sec : numeric
        Time difference, in seconds
    hms_always : bool
        * True
            Always display hours, minuts, and seconds regardless of the order-
            of-magnitude of dt_sec
        * False
            Display a minimal-length string that is meaningful, by omitting
            units that are more significant than those necessary to display
            dt_sec; if...
            * dt_sec < 1 s
                Use engineering formatting for the number.
            * dt_sec is an integer in the range 0-59 (inclusive)
                `sec_decimals` is ignored and the number is formatted as an
                integer
            See Notes below for handling of units.
        (Default: False)
    sec_decimals : int
        Round seconds to this number of digits

    Notes
    -----
    If colon notation (e.g. HH:MM:SS.xxx, MM:SS.xxx, etc.) is not used, the
    number is only seconds, and is appended by a space ' ' followed by units
    of 's' (possibly with a metric prefix).

    """
    sign_str = ''
    sgn = 1
    if dt_sec < 0:
        sgn = -1
        sign_str = '-'
    dt_sec = sgn*dt_sec

    h, r = divmod(dt_sec, 3600)
    m, s = divmod(r, 60)
    h = int(h)
    m = int(m)

    strdt = ''
    if hms_always or h != 0:
        strdt += format(h, '02d') + ':'
    if hms_always or h != 0 or m != 0:
        strdt += format(m, '02d') + ':'

    if float(s) == int(s):
        s = int(s)
        s_fmt = 'd' if len(strdt) == 0 else '02d'
    else:
        # If no hours or minutes, use SI-prefixed fmt for seconds with 3
        # decimal places
        if (h == 0) and (m == 0) and not hms_always:
            nearest_si_order_of_mag = (
                (decimal.Decimal.from_float(dt_sec).adjusted() // 3) * 3
            )
            sec_str = format_num(dt_sec,
                                 precision=10**(nearest_si_order_of_mag-3),
                                 exponent=nearest_si_order_of_mag,
                                 fmt='sipre')
            return sec_str + 's'
        # Otherwise, round seconds to sec_decimals decimal digits
        s = np.round(s, sec_decimals)
        if len(strdt) == 0:
            s_fmt = '.%df' %sec_decimals
        else:
            if sec_decimals == 0:
                s_fmt = '02.0f'
            else:
                s_fmt = '0%d.%df' %(3+sec_decimals, sec_decimals)
    if len(strdt) > 0:
        strdt += format(s, s_fmt)
    else:
        strdt += format(s, s_fmt) + ' s'

    return sign_str + strdt


def test_timediff():
    """Unit tests for timediff function"""
    v = timediff(1234)
    assert v == '20:34', v
    v = timediff(1234.5678)
    assert v == '20:34.568', v
    v = timediff(1, hms_always=True)
    assert v == '00:00:01', v
    v = timediff(1.1, hms_always=True, sec_decimals=3)
    assert v == '00:00:01.100', v
    v = timediff(1e6)
    assert v == '277:46:40', v
    v = timediff(1e6 + 1.5)
    assert v == '277:46:41.500', v
    logging.info('<< PASS : test_timediff >>')


def timestamp(d=True, t=True, tz=True, utc=False, winsafe=False):
    """Simple utility to print out a time, date, or time+date stamp for the
    time at which the function is called.

    Parameters
    ----------:
    d : bool
        Include date (default: True)
    t : bool
        Include time (default: True)
    tz : bool
        Include timezone offset from UTC (default: True)
    utc : bool
        Include UTC time/date (as opposed to local time/date) (default: False)
    winsafe : bool
        Omit colons between hours/minutes (default: False)

    """
    if utc:
        time_tuple = time.gmtime()
    else:
        time_tuple = time.localtime()

    dts = ''
    if d:
        dts += time.strftime('%Y-%m-%d', time_tuple)
        if t:
            dts += 'T'
    if t:
        if winsafe:
            dts += time.strftime('%H%M%S', time_tuple)
        else:
            dts += time.strftime('%H:%M:%S', time_tuple)

        if tz:
            if utc:
                if winsafe:
                    dts += time.strftime('+0000')
                else:
                    dts += time.strftime('+0000')
            else:
                offset = time.strftime('%z')
                if not winsafe:
                    offset = offset[:-2:] + '' + offset[-2::]
                dts += offset
    return dts


def test_timestamp():
    """Unit tests for timestamp function"""
    print(timestamp())
    logging.info('<< PASS : test_timestamp >>')


if __name__ == '__main__':
    set_verbosity(1)
    test_hr_range_formatter()
    test_list2hrlist()
    test_format_num()
    test_timediff()
    test_timestamp()
