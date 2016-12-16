# author : J.L. Lanfranchi
#          jll1062+pisa@phys.psu.edu
#
# date   : April 8, 2016
"""
Utilities for interpreting and returning formatted strings.
"""


from itertools import imap
import numbers
import re

import numpy as np
import pint

from pisa import ureg
from pisa.utils.log import logging, set_verbosity


__all__ = ['WHITESPACE_RE', 'NUMBER_RESTR', 'NUMBER_RE',
           'HRGROUP_RESTR', 'HRGROUP_RE', 'IGNORE_CHARS_RE',
           'list2hrlist', 'hrlist2list', 'hrlol2lol', 'hrbool2bool', 'engfmt',
           'text2tex', 'int2hex', 'hash2hex']


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


# TODO: allow for scientific notation input to hr*2list, etc.

def hrlist_formatter(start, end, step):
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
    >>> hrlist_formatter(start=0, end=10, step=1)
    '0-10'
    >>> hrlist_formatter(start=0, end=10, step=2)
    '0-10:2'
    >>> hrlist_formatter(start=0, end=3, step=8)
    '0-3:8'
    >>> hrlist_formatter(start=0.1, end=3.1, step=1.0)
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
    if isinstance(lst, numbers.Number):
        lst = [lst]
    lst = sorted(lst)
    rtol = np.finfo(float).resolution
    n = len(lst)
    result = []
    scan = 0
    while n - scan > 2:
        step = lst[scan + 1] - lst[scan]
        if not np.isclose(lst[scan + 2] - lst[scan + 1], step, rtol=rtol):
            result.append(str(lst[scan]))
            scan += 1
            continue
        for j in xrange(scan+2, n-1):
            if not np.isclose(lst[j+1] - lst[j], step, rtol=rtol):
                result.append(hrlist_formatter(lst[scan], lst[j], step))
                scan = j+1
                break
        else:
            result.append(hrlist_formatter(lst[scan], lst[-1], step))
            return ','.join(result)
    if n - scan == 1:
        result.append(str(lst[scan]))
    elif n - scan == 2:
        result.append(','.join(imap(str, lst[scan:])))

    return ','.join(result)


def hrgroup2list(hrgroup):
    def isint(num):
        """Test whether a number is *functionally* an integer"""
        try:
            return int(num) == float(num)
        except ValueError:
            return False

    def num2floatOrInt(num):
        try:
            if isint(num):
                return int(num)
        except (ValueError, TypeError):
            pass
        return float(num)

    # Strip all whitespace, brackets, parens, and other ignored characters from
    # the group string
    hrgroup = IGNORE_CHARS_RE.sub('', hrgroup)
    if (hrgroup is None) or (hrgroup == ''):
        return []
    numstrs = HRGROUP_RE.match(hrgroup).groups()
    range_start = num2floatOrInt(numstrs[0])

    # If no range is specified, just return the number
    if numstrs[1] is None:
        return [range_start]

    range_stop = num2floatOrInt(numstrs[1])
    if numstrs[2] is None:
        step_size = 1 if range_stop >= range_start else -1
    else:
        step_size = num2floatOrInt(numstrs[2])
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
    print n_steps, range_start, range_stop, step_size
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
    if len(groups) == 0:
        return lst
    [lst.extend(hrgroup2list(g)) for g in groups]
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
    prefixes = {-18:'a', -15:'f', -12:'p', -9:'n', -6:'u', -3:'m', 0:'',
                3:'k', 6:'M', 9:'G', 12:'T', 15:'P', 18:'E'}
    if isinstance(n, pint.quantity._Quantity):
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
    if pfx_mag not in prefixes or not units.dimensionless:
        if pfx_mag == 0:
            return str.strip('{0:s} {1:~} '.format(num_str, units))
        else:
            return str.strip('{0:s}e{1:d} {2:~} '.format(num_str, pfx_mag,
                                                         units))

    # Dimensionless quantities are treated separately since Pint apparently
    # can't handle prefixed-dimensionless (e.g., simply "1 k", "2.2 M", etc.,
    # with no units attached).
    #if units.dimensionless:
    return  '{0:s} {1:s}'.format(num_str, prefixes[pfx_mag])


def append_results(results_dict, result_dict):
    for key, val in result_dict.items():
        if results_dict.has_key(key):
            results_dict[key].append(val)
        else:
            results_dict[key] = [val]


def ravel_results(results):
    for key, val in results.items():
        if hasattr(val[0], 'm'):
            results[key] = np.array([v.m for v in val]) * val[0].u


def text2tex(txt):
    return txt.replace('_', r'\_')


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
    if isinstance(hash, str):
        assert len(hash) == int(np.ceil(bits/4.0))
        hex_hash = hash
    elif isinstance(hash, int):
        hex_hash = int2hex(hash, bits=bits, signed=True)
    else:
        raise ValueError('Unhandled `hash` type %s' %type(hash))
    return hex_hash


def test_hrlist_formatter():
    logging.debug(str((hrlist_formatter(start=0, end=10, step=1))))
    logging.debug(str((hrlist_formatter(start=0, end=10, step=2))))
    logging.debug(str((hrlist_formatter(start=0, end=3, step=8))))
    logging.debug(str((hrlist_formatter(start=0.1, end=3.1, step=1.0))))
    logging.info('<< PASSED : test_hrlist_formatter >>')


def test_list2hrlist():
    logging.debug(str((list2hrlist([0, 1]))))
    logging.debug(str((list2hrlist([0, 1, 2]))))
    logging.debug(str((list2hrlist([0.1, 1.1, 2.1, 3.1]))))
    logging.info('<< PASSED : test_list2hrlist >>')


if __name__ == '__main__':
    set_verbosity(1)
    test_hrlist_formatter()
    test_list2hrlist()
