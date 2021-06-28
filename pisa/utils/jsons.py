"""
A set of utilities for reading (and instantiating) objects from and writing
objects to JSON files.
"""


from __future__ import absolute_import, division

import bz2
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Number, Real
import os
import tempfile

import numpy as np
import simplejson as json
from six import string_types

from pisa import ureg
from pisa.utils.log import logging, set_verbosity

__all__ = [
    'JSON_EXTS',
    'ZIP_EXTS',
    'XOR_EXTS',
    'json_string',
    'dumps',
    'loads',
    'from_json',
    'to_json',
    'NumpyEncoder',
    'NumpyDecoder',
    'test_to_json_from_json',
]

__author__ = 'S. Boeser, J.L. Lanfranchi'

__license__ = '''Copyright (c) 2014-2019, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


JSON_EXTS = ['json']
ZIP_EXTS = ['bz2']
XOR_EXTS = ['xor']


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


def from_json(filename, cls=None):
    """Open a file in JSON format (optionally compressed with bz2 or
    xor-scrambled) and parse the content into Python objects.

    Parameters
    ----------
    filename : str
    cls : class (type) object, optional
        If provided, the class is attempted to be instantiated as described in
        Notes section.

    Returns
    -------
    contents_or_obj : simple Python objects or `cls` instantiated therewith

    Notes
    -----
    If `cls` is provided as a class (type) object, this function attempts to
    instantiate the class with the data loaded from the JSON file, as follows:

        * if `cls` has a `from_json` method, that is called directly: .. ::

                cls.from_json(filename)

        * if the data loaded from the JSON file is a non-string sequence: .. ::

                cls(*data)

        * if the data loaded is a Mapping (dict, OrderedDict, etc.): .. ::

                cls(**data)

        * for all other types loaded from the JSON: .. ::

                cls(data)

    Note that this currently only recognizes files by their extensions. I.e.,
    the file must be named .. ::

        myfile.json
        myfile.json.bz2
        myfile.json.xor

    represent a bsic JSON file, a bzip-compressed JSON, and an xor-scrambled
    JSON, respectively.

    """
    # Import here to avoid circular imports
    from pisa.utils.log import logging
    from pisa.utils.resources import open_resource

    if cls is not None:
        if not isinstance(cls, type):
            raise TypeError(
                "`cls` should be a class object (type); got {} instead".format(
                    type(cls)
                )
            )
        if hasattr(cls, "from_json"):
            return cls.from_json(filename)

        # Otherwise, handle instantiating the class generically (which WILL
        # surely fail for many types) based on the type of the object loaded
        # from JSON file: Mapping is passed via cls(**data), non-string
        # Sequence is passed via cls(*data), and anything else is passed via
        # cls(data)

    _, ext = os.path.splitext(filename)
    ext = ext.replace('.', '').lower()
    assert ext in JSON_EXTS or ext in ZIP_EXTS + XOR_EXTS
    try:
        if ext == 'bz2':
            fobj = open_resource(filename, 'rb')
            try:
                bz2_content = fobj.read()
            finally:
                fobj.close()
            decompressed = bz2.decompress(bz2_content).decode()
            del bz2_content
            content = json.loads(
                decompressed,
                cls=NumpyDecoder,
                object_pairs_hook=OrderedDict
            )
            del decompressed
        elif ext == 'xor':

            with open(filename, 'rb') as infile:
                encrypted_bytes = infile.read()

            # decrypt with key 42
            decypted_bytes = bytearray()
            for byte in encrypted_bytes:
                decypted_bytes.append(byte ^ 42)

            content = json.loads(decypted_bytes.decode(),
                                cls=NumpyDecoder,
                                object_pairs_hook=OrderedDict)
        else:
            fobj = open_resource(filename)
            try:
                content = json.load(
                    fobj,
                    cls=NumpyDecoder,
                    object_pairs_hook=OrderedDict,
                )
            finally:
                fobj.close()
    except:
        logging.error('Failed to load JSON, `filename`="%s"', filename)
        raise

    if cls is None:
        return content

    if isinstance(content, Mapping):
        return cls(**content)
    if not isinstance(string_types) and isinstance(content, Sequence):
        return cls(*content)
    return cls(content)


def to_json(content, filename, indent=2, overwrite=True, warn=True,
            sort_keys=False):
    """Write `content` to a JSON file at `filename`.

    Uses a custom parser that automatically converts numpy arrays to lists.

    If `filename` has a ".bz2" extension, the contents will be compressed
    (using bz2 and highest-level of compression, i.e., -9).

    If `filename` has a ".xor" extension, the contents will be xor-scrambled to
    make them human-unreadable (this is useful for, e.g., blind fits).


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
    # Import here to avoid circular imports
    from pisa.utils.fileio import check_file_exists
    from pisa.utils.log import logging

    if hasattr(content, 'to_json'):
        return content.to_json(filename, indent=indent, overwrite=overwrite,
                               warn=warn, sort_keys=sort_keys)

    check_file_exists(fname=filename, overwrite=overwrite, warn=warn)

    _, ext = os.path.splitext(filename)
    ext = ext.replace('.', '').lower()
    assert ext == 'json' or ext in ZIP_EXTS + XOR_EXTS

    with open(filename, 'wb') as outfile:
        if ext == 'bz2':
            outfile.write(
                bz2.compress(
                    json.dumps(
                        content, outfile, indent=indent, cls=NumpyEncoder,
                        sort_keys=sort_keys, allow_nan=True, ignore_nan=False
                    ).encode()
                )
            )
        elif ext == 'xor':
            json_bytes = json.dumps(
                content, indent=indent, cls=NumpyEncoder,
                sort_keys=sort_keys, allow_nan=True, ignore_nan=False
                ).encode()

            # encrypt with key 42
            encrypted_bytes = bytearray()
            for byte in json_bytes:
                encrypted_bytes.append(byte ^ 42)

            outfile.write(encrypted_bytes)
        else:
            outfile.write(
                json.dumps(
                    content, indent=indent, cls=NumpyEncoder,
                    sort_keys=sort_keys, allow_nan=True, ignore_nan=False
                ).encode()
            )
        logging.debug('Wrote %.2f kiB to %s', outfile.tell()/1024., filename)


# TODO: figure out how to serialize / deserialize scalars and arrays with
# uncertainties

class NumpyEncoder(json.JSONEncoder):
    """
    Subclass of ::class::`json.JSONEncoder` that overrides `default` method to
    allow writing numpy arrays and other special objects PISA uses to JSON
    files.
    """
    def default(self, obj):  # pylint: disable=method-hidden
        """Encode special objects to be representable as JSON."""
        if hasattr(obj, 'serializable_state'):
            return obj.serializable_state

        if isinstance(obj, string_types):
            return obj

        if isinstance(obj, ureg.Quantity):
            converted = [self.default(x) for x in obj.to_tuple()]
            return converted

        # must have checked for & handled strings prior to this or infinite
        # recursion will result
        if isinstance(obj, Iterable):
            return [self.default(x) for x in obj]

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        # NOTE: np.bool_ is *Numpy* bool type, while np.bool is alias for
        # Python bool type, hence this conversion
        if isinstance(obj, np.bool_):
            return bool(obj)

        # NOTE: we check for these more generic types _after_ checking for
        # np.bool_ since np.bool_ is considered to be both Integral and Real,
        # but we want a boolean values (True or False) written out as such
        if isinstance(obj, Integral):
            return int(obj)

        if isinstance(obj, Real):
            return float(obj)

        if isinstance(obj, string_types):
            return obj

        # If we get here, we have a type that cannot be serialized. This call
        # should simply raise an exception.
        return super().default(obj)


class NumpyDecoder(json.JSONDecoder):
    """Decode JSON array(s) as numpy.ndarray; also returns python strings
    instead of unicode."""
    def __init__(
        self,
        encoding=None,
        object_hook=None,
        parse_float=None,
        parse_int=None,
        parse_constant=None,
        strict=True,
        object_pairs_hook=None,
    ):
        super().__init__(
            encoding=encoding,
            object_hook=object_hook,
            parse_float=parse_float,
            parse_int=parse_int,
            parse_constant=parse_constant,
            strict=strict,
            object_pairs_hook=object_pairs_hook,
        )
        # Only need to override the default array handler
        self.parse_array = self.json_array_numpy
        self.scan_once = json.scanner.py_make_scanner(self)

    def json_array_numpy(self, s_and_end, scan_once, **kwargs):
        """Interpret arrays (lists by default) as numpy arrays where this does
        not yield a string or object array; also handle conversion of
        particularly-formatted input to pint Quantities."""
        # Use the default array parser to get list-ified version of the data
        values, end = json.decoder.JSONArray(s_and_end, scan_once, **kwargs)

        # Assumption for all below logic is the result is a Sequence (i.e., has
        # attribute `__len__`)
        assert isinstance(values, Sequence), str(type(values)) + "\n" + str(values)

        if len(values) == 0:
            return values, end

        try:
            # -- Check for pint quantity -- #

            if (
                isinstance(values, ureg.Quantity)
                or any(isinstance(val, ureg.Quantity) for val in values)
            ):
                return values, end

            # Quantity tuple (`quantity.to_tuple()`) with a scalar produces from
            # the raw JSON, e.g.,
            #
            #       [9.8, [['meter', 1.0], ['second', -2.0]]]
            #
            # or an ndarray (here of shape (2, 3)) produces from the raw JSON,
            # e.g.,
            #
            #       [[[0, 1, 2], [2, 3, 4]], [['meter', 1.0], ['second', -2.0]]]
            #
            if (
                len(values) == 2
                and isinstance(values[1], Sequence)
                and all(
                    isinstance(subval, Sequence)
                    and len(subval) == 2
                    and isinstance(subval[0], string_types)
                    and isinstance(subval[1], Number)
                    for subval in values[1]
                )
            ):
                values = ureg.Quantity.from_tuple(values)
                return values, end

            # Units part of quantity tuple (`quantity.to_tuple()[1]`)
            # e.g. m / s**2 is represented as .. ::
            #
            #       [['meter', 1.0], ['second', -2.0]]
            #
            # --> Simply return, don't perform further conversion
            if (
                isinstance(values[0], Sequence)
                and all(
                    len(subval) == 2
                    and isinstance(subval[0], string_types)
                    and isinstance(subval[1], Number)
                    for subval in values
                )
            ):
                return values, end

            # Individual unit (`quantity.to_tuple()[1][0]`)
            # e.g. s^-2 is represented as .. ::
            #
            #     ['second', -2.0]
            #
            # --> Simply return, don't perform further conversion
            if (
                len(values) == 2
                and isinstance(values[0], string_types)
                and isinstance(values[1], Number)
            ):
                return values, end

            try:
                ndarray_values = np.asarray(values)
            except ValueError:
                return values, end

            # Things like lists of dicts, or mixed types, will result in an
            # object array; these are handled in PISA as lists, not numpy
            # arrays, so return the pre-converted (list) version of `values`.
            #
            # Similarly, sequences of strings should stay lists of strings, not
            # become numpy arrays.
            if issubclass(ndarray_values.dtype.type, (np.object0, np.str0, str)):
                return values, end

            return ndarray_values, end

        except TypeError:
            return values, end

# TODO: include more basic types in testing (strings, etc.)
def test_to_json_from_json():
    """Unit tests for writing various types of objects to and reading from JSON
    files (including bz2-compressed and xor-scrambled files)"""
    # pylint: disable=unused-variable
    from shutil import rmtree
    import sys
    from pisa.utils.comparisons import recursiveEquality

    proto_float_array = np.array(
        [-np.inf, np.nan, np.inf, -1.1, 0.0, 1.1], dtype=np.float64
    )
    proto_int_array = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    proto_str_array = np.array(['a', 'ab', 'abc', '', ' '], dtype=str)

    floating_types = [float] + sorted(
        set(t for _, t in np.typeDict.items() if issubclass(t, np.floating)), key=str,
    )
    integer_types = [int] + sorted(
        set(t for _, t in np.typeDict.items() if issubclass(t, np.integer)), key=str,
    )

    test_info = [
        dict(
            proto_array=proto_float_array,
            dtypes=floating_types,
        ),
        dict(
            proto_array=proto_int_array,
            dtypes=integer_types,
        ),
        # TODO: strings currently do not work
        #dict(
        #    proto_array=proto_str_array,
        #    dtypes=[str, np.str0, np.str_, np.string_],
        #),
    ]

    test_data = OrderedDict()
    for info in test_info:
        proto_array = info['proto_array']
        for dtype in info['dtypes']:
            typed_array = proto_array.astype(dtype)
            s_dtype = str(np.dtype(dtype))
            test_data["array_" + s_dtype] = typed_array
            test_data["scalar_" + s_dtype] = dtype(typed_array[0])

    temp_dir = tempfile.mkdtemp()
    try:
        for name, obj in test_data.items():
            # Test that the object can be written / read directly
            base_fname = os.path.join(temp_dir, name + '.json')
            for ext in ['', '.bz2', '.xor']:
                fname = base_fname + ext
                to_json(obj, fname)
                loaded_data = from_json(fname)
                if obj.dtype in floating_types:
                    assert np.allclose(
                        loaded_data, obj, rtol=1e-12, atol=0, equal_nan=True
                    ), '{}=\n{}\nloaded=\n{}\nsee file: {}'.format(
                        name, obj, loaded_data, fname
                    )
                else:
                    assert np.all(loaded_data == obj), \
                        '{}=\n{}\nloaded_nda=\n{}\nsee file: {}'.format(
                            name, obj, loaded_data, fname
                        )

            # Test that the same object can be written / read as a value in a
            # dictionary
            orig = OrderedDict([(name, obj), (name + "x", obj)])
            base_fname = os.path.join(temp_dir, 'd.{}.json'.format(name))
            for ext in ['', '.bz2', '.xor']:
                fname = base_fname + ext
                to_json(orig, fname)
                loaded = from_json(fname)
                assert recursiveEquality(loaded, orig), \
                    'orig=\n{}\nloaded=\n{}\nsee file: {}'.format(
                        orig, loaded, fname
                    )
    finally:
        rmtree(temp_dir)

    logging.info('<< PASS : test_to_json_from_json >>')


if __name__ == '__main__':
    set_verbosity(1)
    test_to_json_from_json()
