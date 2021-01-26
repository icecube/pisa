"""
Generic file I/O, dispatching specific file readers/writers as necessary
"""


from __future__ import absolute_import

import errno
from functools import reduce
import operator
import os
import pickle
import re

import numpy as np

from pisa.utils import hdf
from pisa.utils import jsons
from pisa.utils import log
from pisa.utils import resources


__all__ = [
    'PKL_EXTS',
    'CFG_EXTS',
    'ZIP_EXTS',
    'TXT_EXTS',
    'XOR_EXTS',
    'NSORT_RE',
    'UNSIGNED_FSORT_RE',
    'SIGNED_FSORT_RE',
    'expand',
    'mkdir',
    'get_valid_filename',
    'nsort',
    'nsort_key_func',
    'fsort',
    'find_files',
    'from_cfg',
    'from_pickle',
    'to_pickle',
    'from_file',
    'to_file',
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


PKL_EXTS = ['pickle', 'pckl', 'pkl', 'p']
CFG_EXTS = ['ini', 'cfg']
ZIP_EXTS = ['bz2']
TXT_EXTS = ['txt', 'dat']
XOR_EXTS = ['xor']

NSORT_RE = re.compile(r'(\d+)')
UNSIGNED_FSORT_RE = re.compile(
    r'''
    (
        (?:\d+(?:\.\d*){0,1}) # Digit(s) followed by opt. "." and opt. digits
        |(?:\.\d+)            # Or starts with "." and must have digits after
        (?:e[+-]?\d+){0,1}    # Opt.: followed by exponent: e12, e-12, e+0, etc.
    )
    ''',
    re.IGNORECASE | re.VERBOSE
)
SIGNED_FSORT_RE = re.compile(
    r'''
    (
        [+-]{0,1}             # Optional sign
        (?:\d+(?:\.\d*){0,1}) # Digit(s) followed by opt. "." and opt. digits
        |(?:\.\d+)            # Or starts with "." but must have digits after
        (?:e[+-]?\d+){0,1}    # Opt.: exponent: e12, e-12, e+0, etc.
    )
    ''',
    re.IGNORECASE | re.VERBOSE
)


def expand(path, exp_user=True, exp_vars=True, absolute=False, resolve_symlinks=False):
    """Convenience function for expanding a path

    Parameters
    ----------
    path : string
        Path to be expanded.

    exp_vars : bool
        Expand the string using environment variables. E.g.
        "$HOME/${vardir}/xyz" will have "$HOME" and "${vardir}$" replaced by
        the values stored in "HOME" and "vardir".

    exp_user : bool
        Expand special home dir spec character, tilde: "~".

    absolute : bool
        Make a relative path (e.g. "../xyz") absolute, referenced from system
        root directory, "/dir/sbudir/xyz".

    resolve_symlinks : bool
        Resolve symlinks to the paths they refer to

    Returns
    -------
    exp_path : string
        Expanded path

    """
    if exp_vars:
        path = os.path.expandvars(path)
    if exp_user:
        path = os.path.expanduser(path)
    if absolute:
        path = os.path.abspath(path)
    if resolve_symlinks:
        path = os.path.realpath(path)
    return path


def check_file_exists(fname, overwrite=True, warn=True):
    """See if a file exists, warning, raising an exception, or doing neither if
    it already exists.

    Note that while this function can warn or raise an exception indicating the
    file will be overwritten, this function does not actually overwrite any
    files.

    Parameters
    ----------
    fname : string
        File name or path to try to find.

    overwrite : bool
        Whether it's okay for the file to be overwritten if it exists. Note
        that this function does not actually overwrite the file.

    warn : bool
        Whether to warn the user that the file will be overwritten if it
        exists. Note that this function does not actually overwrite the file.

    Returns
    -------
    fpath : string
        Expanded path of the `fname` passed in.

    """
    fpath = expand(fname)
    if os.path.exists(fpath):
        if overwrite:
            if warn:
                log.logging.warning("Overwriting file at '%s'", fpath)
        else:
            raise Exception("Refusing to overwrite path '%s'" % fpath)
    return fpath


def mkdir(d, mode=0o0750, warn=True):
    """Simple wrapper around os.makedirs to create a directory but not raise an
    exception if the dir already exists

    Parameters
    ----------
    d : string
        Directory path
    mode : integer
        Permissions on created directory; see os.makedirs for details.
    warn : bool
        Whether to warn if directory already exists.

    """
    try:
        os.makedirs(d, mode=mode)
    except OSError as err:
        if err.errno == errno.EEXIST:
            if warn:
                log.logging.warning('Directory "%s" already exists', d)
        else:
            raise err
    else:
        log.logging.info('Created directory "%s"', d)


def get_valid_filename(s):
    """Sanitize string to make it reasonable to use as a filename.

    From https://github.com/django/django/blob/master/django/utils/text.py

    Parameters
    ----------
    s : string

    Examples
    --------
    >>> print(get_valid_filename(r'A,bCd $%#^#*!()"\' .ext '))
    'a_bcd__.ext'

    """
    s = re.sub(r'[ ,;\t]', '_', s.strip().lower())
    return re.sub(r'(?u)[^-\w.]', '', s)


def nsort(l, reverse=False):
    """Sort a sequence of strings containing integer number fields by the
    value of those numbers, rather than by simple alpha order. Useful
    for sorting e.g. version strings, etc..

    Code adapted from nedbatchelder.com/blog/200712/human_sorting.html#comments

    Parameters
    ----------
    l : sequence of strings
        Sequence of strings to be sorted.

    reverse : bool, optional
        Whether to reverse the sort order (True => descending order)

    Returns
    -------
    sorted_l : list of strings
        Sorted strings

    Examples
    --------
    >>> l = ['f1.10.0.txt', 'f1.01.2.txt', 'f1.1.1.txt', 'f9.txt', 'f10.txt']
    >>> nsort(l)
    ['f1.1.1.txt', 'f1.01.2.txt', 'f1.10.0.txt', 'f9.txt', 'f10.txt']

    See Also
    --------
    fsort
        Sort sequence of strings with floating-point numbers in the strings.

    """
    def _field_splitter(s):
        spl = NSORT_RE.split(s)
        non_numbers = spl[0::2]
        numbers = [int(i) for i in spl[1::2]]
        return reduce(operator.concat, zip(non_numbers, numbers))

    return sorted(l, key=_field_splitter, reverse=reverse)


def nsort_key_func(s):
    """Use as the `key` argument to the `sorted` function or `sort` method.

    Code adapted from nedbatchelder.com/blog/200712/human_sorting.html#comments

    Examples
    --------
    >>> l = ['f1.10.0.txt', 'f1.01.2.txt', 'f1.1.1.txt', 'f9.txt', 'f10.txt']
    >>> sorted(l, key=nsort_key_func)
    ['f1.1.1.txt', 'f1.01.2.txt', 'f1.10.0.txt', 'f9.txt', 'f10.txt']

    """
    spl = NSORT_RE.split(s)
    key = []
    for non_number, number in zip(spl[::2], spl[1::2]):
        key.append(non_number)
        key.append(int(number))
    return key


def fsort(l, signed=True, reverse=False):
    """Sort a sequence of strings with one or more floating point number fields
    in using the floating point value(s) (and intervening strings are treated
    as normally done). Note that + and - preceding a number are included in the
    floating point value unless `signed=False`.

    Code adapted from nedbatchelder.com/blog/200712/human_sorting.html#comments

    Parameters
    ----------
    l : sequence of strings
        Sequence of strings to be sorted.

    signed : bool, optional
        Whether to include a "+" or "-" preceeding a number in its value to be
        sorted. One might specify False if "-" is used exclusively as a
        separator in the string.

    reverse : bool, optional
        Whether to reverse the sort order (True => descending order)

    Returns
    -------
    sorted_l : list of strings
        Sorted strings

    Examples
    --------
    >>> l = ['a-0.1.txt', 'a-0.01.txt', 'a-0.05.txt']
    >>> fsort(l, signed=True)
    ['a-0.1.txt', 'a-0.05.txt', 'a-0.01.txt']

    >>> fsort(l, signed=False)
    ['a-0.01.txt', 'a-0.05.txt', 'a-0.1.txt']

    See Also
    --------
    nsort
        Sort using integer-only values of numbers; good for e.g. version
        numbers, where periods are separators rather than decimal points.

    """
    if signed:
        fsort_re = SIGNED_FSORT_RE
    else:
        fsort_re = UNSIGNED_FSORT_RE

    def _field_splitter(s):
        spl = fsort_re.split(s)
        non_numbers = spl[0::2]
        numbers = [float(i) for i in spl[1::2]]
        return reduce(operator.concat, zip(non_numbers, numbers))

    return sorted(l, key=_field_splitter, reverse=reverse)



def find_files(root, regex=None, fname=None, recurse=True, dir_sorter=nsort,
               file_sorter=nsort):
    """Find files by re or name recursively w/ ordering.

    Code adapted from
    stackoverflow.com/questions/18282370/python-os-walk-what-order

    Parameters
    ----------
    root : str
        Root directory at which to start searching for files

    regex : str or re.SRE_Pattern
        Only yield files matching `regex`.

    fname : str
        Only yield files matching `fname`

    recurse : bool
        Whether to search recursively down from the root directory

    dir_sorter
        Function that takes a list and returns a sorted version of it, for
        purposes of sorting directories

    file_sorter
        Function as specified for `dir_sorter` but used for sorting file names


    Yields
    ------
    fullfilepath : str
    basename : str
    match : re.SRE_Match or None

    """
    root = expand(root)
    if isinstance(regex, str):
        regex = re.compile(regex)

    # Define a function for accepting a filename as a match
    if regex is None:
        if fname is None:
            def _validfilefunc(fn): # pylint: disable=unused-argument
                return True, None
        else:
            def _validfilefunc(fn):
                if fn == fname:
                    return True, None
                return False, None
    else:
        def _validfilefunc(fn):
            match = regex.match(fn)
            if match and (len(match.groups()) == regex.groups):
                return True, match
            return False, None

    if recurse:
        for rootdir, dirs, files in os.walk(root, followlinks=True):
            for basename in file_sorter(files):
                fullfilepath = os.path.join(rootdir, basename)
                is_valid, match = _validfilefunc(basename)
                if is_valid:
                    yield fullfilepath, basename, match
            for dirname in dir_sorter(dirs):
                fulldirpath = os.path.join(rootdir, dirname)
                for basename in file_sorter(os.listdir(fulldirpath)):
                    fullfilepath = os.path.join(fulldirpath, basename)
                    if os.path.isfile(fullfilepath):
                        is_valid, match = _validfilefunc(basename)
                        if is_valid:
                            yield fullfilepath, basename, match
    else:
        for basename in file_sorter(os.listdir(root)):
            fullfilepath = os.path.join(root, basename)
            #if os.path.isfile(fullfilepath):
            is_valid, match = _validfilefunc(basename)
            if is_valid:
                yield fullfilepath, basename, match


def from_cfg(fname):
    """Load a PISA config file"""
    from pisa.utils.config_parser import PISAConfigParser
    config = PISAConfigParser()
    try:
        config.read(fname)
    except:
        log.logging.error(
            'Failed to read PISA config file, `fname`="%s"', fname
        )
        raise
    return config


def from_pickle(fname):
    """Load from a Python pickle file"""
    try:

        # Open the file (binary)
        f = open(fname, 'rb')

        # Try standard pickle load
        try:
            return pickle.load(f)

        # Can get encoding errors when using python3 to open pickle files
        # created with python2 Handle this case
        except UnicodeDecodeError:
            return pickle.load(f, encoding="latin1")

    except:
        log.logging.error('Failed to load pickle file, `fname`="%s"', fname)
        raise

    finally:
        f.close()


def to_pickle(obj, fname, overwrite=True, warn=True):
    """Save object to a pickle file"""
    check_file_exists(fname=fname, overwrite=overwrite, warn=warn)
    return pickle.dump(obj, open(fname, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def from_txt(fname, as_array=False):
    """Load from a text (txt) file"""
    try:
        if as_array:
            with open(fname, 'r') as f:
                a = f.readlines()
            a = [[float(m) for m in l.strip('\n\r').split()] for l in a]
            a = np.array(a)
        else:
            with open(fname, 'r') as f:
                a = f.read()
    except:
        log.logging.error('Failed to load txt file, `fname`="%s"', fname)
        raise
    return a


def to_txt(obj, fname):
    """Save object to a text (txt) file"""
    with open(fname, 'w') as f:
        f.write(obj)


def from_file(fname, fmt=None, **kwargs):
    """Dispatch correct file reader based on `fmt` (if specified) or guess
    based on file name's extension.

    Parameters
    ----------
    fname : string
        File path / name from which to load data.

    fmt : None or string
        If string, for interpretation of the file according to this format. If
        None, file format is deduced by an extension found in `fname`.

    **kwargs
        All other arguments are passed to the function dispatched to read the
        file.

    Returns
    -------
    Object instantiated from the file (string, dictionary, ...). Each format
    is interpreted differently.

    Raises
    ------
    ValueError
        If extension is not recognized

    """
    if fmt is None:
        rootname, ext = os.path.splitext(fname)
        ext = ext.replace('.', '').lower()
    else:
        rootname = fname
        ext = fmt.lower()

    if ext in ZIP_EXTS or ext in XOR_EXTS:
        rootname, inner_ext = os.path.splitext(rootname)
        inner_ext = inner_ext.replace('.', '').lower()
        ext = inner_ext

    fname = resources.find_resource(fname)
    if ext in jsons.JSON_EXTS:
        return jsons.from_json(fname, **kwargs)
    if ext in hdf.HDF5_EXTS:
        return hdf.from_hdf(fname, **kwargs)
    if ext in PKL_EXTS:
        return from_pickle(fname, **kwargs)
    if ext in CFG_EXTS:
        return from_cfg(fname, **kwargs)
    if ext in TXT_EXTS:
        return from_txt(fname, **kwargs)
    errmsg = 'File "%s": unrecognized extension "%s"' % (fname, ext)
    log.logging.error(errmsg)
    raise ValueError(errmsg)


def to_file(obj, fname, fmt=None, overwrite=True, warn=True, **kwargs):
    """Dispatch correct file writer based on fmt (if specified) or guess
    based on file name's extension"""
    if fmt is None:
        rootname, ext = os.path.splitext(fname)
        ext = ext.replace('.', '').lower()
    else:
        rootname = fname
        ext = fmt.lower()

    dirname = os.path.dirname(fname)

    os.makedirs(dirname, exist_ok=True)

    if ext in ZIP_EXTS or ext in XOR_EXTS:
        rootname, inner_ext = os.path.splitext(rootname)
        inner_ext = inner_ext.replace('.', '').lower()
        ext = inner_ext

    if ext in jsons.JSON_EXTS:
        return jsons.to_json(obj, fname, overwrite=overwrite, warn=warn,
                             **kwargs)
    elif ext in hdf.HDF5_EXTS:
        return hdf.to_hdf(obj, fname, overwrite=overwrite, warn=warn, **kwargs)
    elif ext in PKL_EXTS:
        return to_pickle(obj, fname, overwrite=overwrite, warn=warn, **kwargs)
    elif ext in TXT_EXTS:
        if kwargs:
            raise ValueError('Following additional keyword arguments not'
                             ' accepted when writing to text file: %s' %
                             kwargs.keys())
        return to_txt(obj, fname)
    else:
        errmsg = 'Unrecognized file type/extension: ' + ext
        log.logging.error(errmsg)
        raise TypeError(errmsg)
