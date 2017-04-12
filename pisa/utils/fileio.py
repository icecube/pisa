# author: Justin Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   2015-06-13
"""
Generic file I/O, dispatching specific file readers/writers as necessary

"""


import cPickle
import os
import re

import dill

from pisa.utils.betterConfigParser import BetterConfigParser
from pisa.utils import hdf
from pisa.utils import jsons
from pisa.utils import log
from pisa.utils import resources

import numpy as np


__all__ = ['PKL_EXTS', 'DILL_EXTS', 'CFG_EXTS', 'ZIP_EXTS', 'TXT_EXTS',
           'NSORT_RE',
           'expand', 'mkdir', 'get_valid_filename', 'nsort', 'find_files',
           'from_cfg', 'from_pickle', 'to_pickle', 'from_dill', 'to_dill',
           'from_file', 'to_file']


PKL_EXTS = ['pickle', 'pckl', 'pkl', 'p']
DILL_EXTS = ['dill']
CFG_EXTS = ['ini', 'cfg']
ZIP_EXTS = ['bz2']
TXT_EXTS = ['txt', 'dat']

NSORT_RE = re.compile(r'(\d+)')


def expand(path, exp_user=True, exp_vars=True, absolute=False):
    """Convenience function for expanding a path

    Parameters
    ----------
    path : string
        Path to be expanded.

    exp_user : bool
        Expand special home dir spec character, tilde: "~".

    exp_vars : bool
        Expand the string using environment variables. E.g.
        "$HOME/${vardir}/xyz" will have "$HOME" and "${vardir}$" replaced by
        the values stored in "HOME" and "vardir".

    absolute : bool
        Make a relative path (e.g. "../xyz") absolute, referenced from system
        root directory, "/dir/sbudir/xyz".

    Returns
    -------
    exp_path : string
        Expanded path

    """
    if exp_user:
        path = os.path.expanduser(path)
    if exp_vars:
        path = os.path.expandvars(path)
    if absolute:
        path = os.path.abspath(path)
    return path


def check_file_exists(fname, overwrite=True, warn=True):
    fpath = expand(fname)
    if os.path.exists(fpath):
        if overwrite:
            if warn:
                log.logging.warn("Overwriting file at '%s'", fpath)
        else:
            raise Exception("Refusing to overwrite path '%s'", fpath)
    return fpath


def mkdir(d, mode=0750, warn=True):
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
        if err[0] == 17:
            if warn:
                log.logging.warn('Directory "%s" already exists', d)
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
    >>> print get_valid_filename(r'A,bCd $%#^#*!()"\' .ext ')
    'a_bcd__.ext'

    """
    s = re.sub(r'[ ,;\t]', '_', s.strip().lower())
    return re.sub(r'(?u)[^-\w.]', '', s)


def nsort(l):
    """Numbers sorted by value, not by alpha order.

    Code from
    nedbatchelder.com/blog/200712/human_sorting.html#comments
    """
    return sorted(
        l,
        key=lambda a: zip(NSORT_RE.split(a)[0::2],
                          [int(i) for i in NSORT_RE.split(a)[1::2]])
    )


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
    if isinstance(regex, basestring):
        regex = re.compile(regex)

    # Define a function for accepting a filename as a match
    if regex is None:
        if fname is None:
            def validfilefunc(fn):
                return True, None
        else:
            def validfilefunc(fn):
                if fn == fname:
                    return True, None
                return False, None
    else:
        def validfilefunc(fn):
            match = regex.match(fn)
            if match and (len(match.groups()) == regex.groups):
                return True, match
            return False, None

    if recurse:
        for rootdir, dirs, files in os.walk(root, followlinks=True):
            for basename in file_sorter(files):
                fullfilepath = os.path.join(rootdir, basename)
                is_valid, match = validfilefunc(basename)
                if is_valid:
                    yield fullfilepath, basename, match
            for dirname in dir_sorter(dirs):
                fulldirpath = os.path.join(rootdir, dirname)
                for basename in file_sorter(os.listdir(fulldirpath)):
                    fullfilepath = os.path.join(fulldirpath, basename)
                    if os.path.isfile(fullfilepath):
                        is_valid, match = validfilefunc(basename)
                        if is_valid:
                            yield fullfilepath, basename, match
    else:
        for basename in file_sorter(os.listdir(root)):
            fullfilepath = os.path.join(root, basename)
            #if os.path.isfile(fullfilepath):
            is_valid, match = validfilefunc(basename)
            if is_valid:
                yield fullfilepath, basename, match


def from_cfg(fname):
    config = BetterConfigParser()
    config.read(fname)
    return config


def from_pickle(fname):
    return cPickle.load(file(fname, 'rb'))


def to_pickle(obj, fname, overwrite=True, warn=True):
    check_file_exists(fname=fname, overwrite=overwrite, warn=warn)
    return cPickle.dump(obj, file(fname, 'wb'),
                        protocol=cPickle.HIGHEST_PROTOCOL)


def from_txt(fname, as_array=False):
    if as_array:
        with open(fname, 'r') as f:
            a = f.readlines()
        a = [[float(m) for m in l.strip('\n\r').split()] for l in a]
        a = np.array(a)
    else:
        with open(fname, 'r') as f:
            a = f.read()
    return a


def to_txt(obj, fname):
    with open(fname, 'w') as f:
        f.write(obj)


def from_dill(fname):
    return dill.load(file(fname, 'rb'))


def to_dill(obj, fname, overwrite=True, warn=True):
    check_file_exists(fname=fname, overwrite=overwrite, warn=warn)
    return dill.dump(obj, file(fname, 'wb'), protocol=dill.HIGHEST_PROTOCOL)


def from_file(fname, fmt=None, **kwargs):
    """Dispatch correct file reader based on fmt (if specified) or guess
    based on file name's extension.

    Parameters
    ----------
    fname : string
        File path / name from which to load data.
    fmt : None or string
        If string, for interpretation of the file according to this format. If
        None, file format is deduced by an extension found in `fname`.
    **kwargs
        All other arguments are passed to the function called to read the file.

    Returns
    -------
    Object instantiated from the file (string, dictionariy, ...). Each format
    is interpreted differently.

    """
    if fmt is None:
        rootname, ext = os.path.splitext(fname)
        ext = ext.replace('.', '').lower()
    else:
        rootname = fname
        ext = fmt.lower()

    zip_ext = None
    if ext in ZIP_EXTS:
        rootname, inner_ext = os.path.splitext(rootname)
        inner_ext = inner_ext.replace('.', '').lower()
        zip_ext = ext
        ext = inner_ext

    fname = resources.find_resource(fname)
    if ext in jsons.JSON_EXTS:
        return jsons.from_json(fname, **kwargs)
    if ext in hdf.HDF5_EXTS:
        return hdf.from_hdf(fname, **kwargs)
    if ext in PKL_EXTS:
        return from_pickle(fname, **kwargs)
    if ext in DILL_EXTS:
        return from_dill(fname, **kwargs)
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

    zip_ext = None
    if ext in ZIP_EXTS:
        rootname, inner_ext = os.path.splitext(rootname)
        inner_ext = inner_ext.replace('.', '').lower()
        zip_ext = ext
        ext = inner_ext

    if ext in jsons.JSON_EXTS:
        return jsons.to_json(obj, fname, overwrite=overwrite, warn=warn,
                             **kwargs)
    elif ext in hdf.HDF5_EXTS:
        return hdf.to_hdf(obj, fname, overwrite=overwrite, warn=warn, **kwargs)
    elif ext in PKL_EXTS:
        return to_pickle(obj, fname, overwrite=overwrite, warn=warn, **kwargs)
    elif ext in DILL_EXTS:
        return to_dill(obj, fname, overwrite=overwrite, warn=warn, **kwargs)
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


def is_dir(dirname):
    """
    Checks if a path is an actual directory, if not found then it creates the
    directory
    """
    if not os.path.isdir(dirname):
        mkdir(dirname)
    return dirname


def is_valid_file(filepath):
    """
    Checks if a path is an actual file
    """
    if not os.path.exists(filepath):
        msg = 'The file {0} does not exist!'.format(filepath)
        raise ValueError(msg)
    else:
        return filepath
