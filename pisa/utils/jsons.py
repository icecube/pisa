# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
# author: J.L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   2014-01-27
"""
A set of utilities for reading (and instantiating) objects from and writing
objects to JSON files.

Import json from this module everywhere (if you need to at all, and can not
just use from_json, to_json) for... faster JSON serdes?
"""
# TODO: why the second line above?


from __future__ import absolute_import, division

import bz2
from collections import OrderedDict
import os

import numpy as np
import simplejson as json

from pisa import ureg
from pisa.utils.comparisons import isbarenumeric


__all__ = ['JSON_EXTS', 'ZIP_EXTS',
           'json_string', 'from_json', 'to_json']


JSON_EXTS = ['json']
ZIP_EXTS = ['bz2']


def json_string(string):
    """Decode a json string"""
    return json.loads(string)


def dumps(content, indent=2):
    """Dump object to JSON-encoded string"""
    return json.dumps(content, cls=NumpyEncoder, indent=indent,
                      sort_keys=False)


def loads(s):
    """Load (create) object from JSON-encoded string"""
    return json.loads(s, cls=NumpyDecoder)


def from_json(filename):
    """Open a file in JSON format (optionally compressed with bz2) and parse
    the content into Python objects.

    Note that this currently only recognizes bz2-compressed file by its
    extension (i.e., the file must be <root>.json.bz2 if it is compressed).

    Parameters
    ----------
    filename : str

    Returns
    -------
    content: OrderedDict with contents of JSON file

    """
    # Import here to avoid circular imports
    from pisa.utils.resources import open_resource

    _, ext = os.path.splitext(filename)
    ext = ext.replace('.', '').lower()
    assert ext in JSON_EXTS or ext in ZIP_EXTS
    if ext == 'bz2':
        content = json.loads(
            bz2.decompress(open_resource(filename).read()),
            cls=NumpyDecoder,
            object_pairs_hook=OrderedDict
        )
    else:
        content = json.load(open_resource(filename),
                            cls=NumpyDecoder,
                            object_pairs_hook=OrderedDict)
    return content


def to_json(content, filename, indent=2, overwrite=True, warn=True,
            sort_keys=False):
    """Write content to a JSON file using a custom parser that automatically
    converts numpy arrays to lists. If the filename has a ".bz2" extension
    appended, the contents will be compressed (using bz2 and highest-level of
    compression, i.e., -9)

    Parameters
    ----------
    content : obj
        Object to be written to file. Tries making use of the object's own
        `to_json` method if it exists.
    filename : str
        Name of the file to be written to. Extension has to be 'json' or 'bz2'.
    indent : int
        Pretty-printing. Cf. documentation of json.dump() or json.dumps()
    overwrite : bool
        Set to `True` (default) to allow overwriting existing file. Raise
        exception and quit otherwise.
    warn : bool
        Issue a warning message if a file is being overwritten (`True`,
        default). Suppress warning by setting to `False` (e.g. when overwriting
        is the desired behaviour).
    sort_keys : bool
        Output of dictionaries will be sorted by key if set to `True`.
        Default is `False`. Cf. json.dump() or json.dumps().

    """
    if hasattr(content, 'to_json'):
        return content.to_json(filename, indent=indent, overwrite=overwrite,
                               warn=warn, sort_keys=sort_keys)
    # Import here to avoid circular imports
    from pisa.utils.fileio import check_file_exists
    from pisa.utils.log import logging

    check_file_exists(fname=filename, overwrite=overwrite, warn=warn)

    _, ext = os.path.splitext(filename)
    ext = ext.replace('.', '').lower()
    assert ext == 'json' or ext in ZIP_EXTS

    with open(filename, 'w') as outfile:
        if ext == 'bz2':
            outfile.write(
                bz2.compress(
                    json.dumps(
                        content, outfile, indent=indent, cls=NumpyEncoder,
                        sort_keys=sort_keys, allow_nan=True, ignore_nan=False
                    )
                )
            )
        else:
            json.dump(
                content, outfile, indent=indent, cls=NumpyEncoder,
                sort_keys=sort_keys, allow_nan=True, ignore_nan=False
            )
        logging.debug('Wrote %.2f kB to %s', outfile.tell()/1024., filename)


class NumpyEncoder(json.JSONEncoder):
    """Encode special objects to be representable as JSON."""
    def default(self, obj):
        # Import here to avoid circular imports
        from pisa.utils.log import logging

        if isinstance(obj, np.ndarray):
            return obj.astype(np.float64).tolist()

        # TODO: poor form to have a way to get this into a JSON file but no way
        # to get it out of a JSON file... so either write a deserializer, or
        # remove this and leave it to other objects to do the following.
        if isinstance(obj, ureg.Quantity):
            return obj.to_tuple()

        # NOTE: np.bool_ is the *Numpy* bool type, while np.bool is alias for
        # Python bool type, hence this conversion
        if isinstance(obj, np.bool_):
            return bool(obj)

        if hasattr(obj, 'serializable_state'):
            return obj.serializable_state

        if isinstance(obj, np.float32):
            return float(obj)

        try:
            return json.JSONEncoder.default(self, obj)
        except:
            logging.error('JSON serialization for %s, type %s not implemented',
                          obj, type(obj))
            raise


class NumpyDecoder(json.JSONDecoder):
    """Decode JSON array(s) as numpy.ndarray, also returns python strings
    instead of unicode."""
    def __init__(self, encoding=None, object_hook=None, parse_float=None,
                 parse_int=None, parse_constant=None, strict=True,
                 object_pairs_hook=None):
        super(NumpyDecoder, self).__init__(
            encoding=encoding, object_hook=object_hook,
            parse_float=parse_float, parse_int=parse_int,
            parse_constant=parse_constant, strict=strict,
            object_pairs_hook=object_pairs_hook
        )
        # Only need to override the default array handler
        self.parse_array = self.json_array_numpy
        self.parse_string = self.json_python_string
        self.scan_once = json.scanner.py_make_scanner(self)

    def json_array_numpy(self, s_and_end, scan_once, **kwargs):
        values, end = json.decoder.JSONArray(s_and_end, scan_once, **kwargs)
        if not values:
            return values, end

        # TODO: is it faster to convert to numpy array and check if the
        # resulting dtype is pure numeric?
        if len(values) <= 1000:
            check_values = values
        else:
            check_values = values[::max([len(values)//1000, 1])]

        if not all([isbarenumeric(v) for v in check_values]):
            return values, end

        values = np.array(values)
        return values, end

    def json_python_string(self, s, end, encoding, strict):
        values, end = json.decoder.scanstring(s, end, encoding, strict)
        return values.encode('utf-8'), end


# TODO: finish this little bit
def test_NumpyEncoderDecoder():
    from shutil import rmtree
    import sys
    from tempfile import mkdtemp
    from pisa.utils.comparisons import recursiveEquality

    nda1 = np.array([-np.inf, np.nan, np.inf, -1, 0, 1, ])
    temp_dir = mkdtemp()
    try:
        fname = os.path.join(temp_dir, 'nda1.json')
        to_json(nda1, fname)
        fname2 = os.path.join(temp_dir, 'nda1.json.bz2')
        to_json(nda1, fname2)
        for fn in [fname, fname2]:
            nda2 = from_json(fn)
            assert np.allclose(nda2, nda1, rtol=1e-12, atol=0, equal_nan=True), \
                    'nda1=\n%s\nnda2=\n%s\nsee file: %s' %(nda1, nda2, fn)
        d1 = {'nda1': nda1}
        fname = os.path.join(temp_dir, 'd1.json')
        fname2 = os.path.join(temp_dir, 'd1.json.bz2')
        to_json(d1, fname)
        to_json(d1, fname2)
        for fn in [fname, fname2]:
            d2 = from_json(fn)
            assert recursiveEquality(d2, d1), \
                    'd1=\n%s\nd2=\n%s\nsee file: %s' %(d1, d2, fn)
    finally:
        rmtree(temp_dir)

    sys.stdout.write('<< PASSED : test_NumpyEncoderDecoder >>\n')


if __name__ == '__main__':
    test_NumpyEncoderDecoder()
