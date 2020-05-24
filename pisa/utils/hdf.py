"""Set of utilities for handling HDF5 file I/O"""


from __future__ import absolute_import

from collections.abc import Mapping
from collections import OrderedDict
import os

import numpy as np
import h5py
from six import string_types

from pisa.utils.log import logging, set_verbosity
from pisa.utils.hash import hash_obj
from pisa.utils.resources import find_resource
from pisa.utils.comparisons import recursiveEquality


__all__ = ['HDF5_EXTS', 'from_hdf', 'to_hdf', 'test_hdf']

__author__ = 'S. Boeser, J.L. Lanfranchi'

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


HDF5_EXTS = ['hdf', 'h5', 'hdf5']


# TODO: convert to allow reading of icetray-produced HDF5 files


def from_hdf(val, return_node=None, choose=None):
    """Return the contents of an HDF5 file or node as a nested dict; optionally
    return a second dict containing any HDF5 attributes attached to the
    entry-level HDF5 entity.

    Parameters
    ----------
    val : string or h5py.Group
        Specifies entry-level entity
        * If val is a string, it is interpreted as a filename; file is opened
          as an h5py.File
        * Otherwise, val must be an h5py.Group in an instantiated object

    return_node : None or string
        Not yet implemented

    choose : None or list
        Optionally can provide a list of variables names to parse (items not in 
        this list will be skipped, saving time & memory)

    Returns
    -------
    data : OrderedDict with additional attr of type OrderedDict named `attrs`
        Nested dictionary; keys are HDF5 node names and values contain the
        contents of that node. If the entry-level entity of `val` has "attrs",
        these are extracted and attached as an OrderedDict at `data.attrs`;
        otherwise, this entity is an empty OrderedDict.

    """
    if return_node is not None:
        raise NotImplementedError('`return_node` is not yet implemented.')

    def visit_group(obj, sdict, choose=None):
        """Iteratively parse `obj` to create the dictionary `sdict`"""
        name = obj.name.split('/')[-1]

        if isinstance(obj, h5py.Dataset):
            if (choose is None) or (name in choose) :
                sdict[name] = obj[()]
        if isinstance(obj, (h5py.Group, h5py.File)):
            sdict[name] = OrderedDict()
            for sobj in obj.values():
                visit_group(sobj, sdict[name], choose)

    myfile = False
    if isinstance(val, str):
        try:
            root = h5py.File(find_resource(val), 'r')
        except Exception:
            logging.error('Failed to load HDF5 file, `val`="%s"', val)
            raise
        myfile = True
    else:
        root = val
        logging.trace('root = %s, root.values() = %s', root, root.values())

    data = OrderedDict()
    attrs = OrderedDict()
    try:
        # Retrieve attrs if present
        if hasattr(root, 'attrs'):
            attrs = OrderedDict(root.attrs)
        # Run over the whole dataset
        for obj in root.values():
            visit_group(obj, data, choose)
    finally:
        if myfile:
            root.close()

    data.attrs = attrs

    return data


def to_hdf(data_dict, tgt, attrs=None, overwrite=True, warn=True):
    """Store a (possibly nested) dictionary to an HDF5 file or branch node
    within an HDF5 file (an h5py Group).

    This creates hardlinks for duplicate non-trivial leaf nodes (h5py Datasets)
    to minimize storage space required for redundant datasets. Duplication is
    detected via object hashing.

    NOTE: Branch nodes are sorted before storing (by name) for consistency in
    the generated file despite Python dictionaries having no defined ordering
    among keys.

    Parameters
    ----------
    data_dict : Mapping
        Dictionary, OrderedDict, or other Mapping to be stored

    tgt : str or h5py.Group
        Target for storing data. If `tgt` is a str, it is interpreted as a
        filename; a file is created with that name (overwriting an existing
        file, if present). After writing, the file is closed. If `tgt` is an
        h5py.Group, the data is simply written to that Group and it is left
        open at function return.

    attrs : Mapping
        Attributes to apply to the top-level entity being written. See
        http://docs.h5py.org/en/latest/high/attr.html

    overwrite : bool
        Set to `True` (default) to allow overwriting existing file. Raise
        exception and quit otherwise.

    warn : bool
        Issue a warning message if a file is being overwritten. Suppress
        warning by setting to `False` (e.g. when overwriting is the desired
        behaviour).

    """
    if not isinstance(data_dict, Mapping):
        raise TypeError('`data_dict` only accepts top-level'
                        ' dict/OrderedDict/etc.')

    def store_recursively(fhandle, node, path=None, attrs=None,
                          node_hashes=None):
        """Function for iteratively doing the work"""
        path = [] if path is None else path
        full_path = '/' + '/'.join(path)
        node_hashes = OrderedDict() if node_hashes is None else node_hashes

        if attrs is None:
            sorted_attr_keys = []
        else:
            if isinstance(attrs, OrderedDict):
                sorted_attr_keys = attrs.keys()
            else:
                sorted_attr_keys = sorted(attrs.keys())

        if isinstance(node, Mapping):
            logging.trace('  creating Group "%s"', full_path)
            try:
                dset = fhandle.create_group(full_path)
                for key in sorted_attr_keys:
                    dset.attrs[key] = attrs[key]
            except ValueError:
                pass

            for key in sorted(node.keys()):
                if isinstance(key, str):
                    key_str = key
                else:
                    key_str = str(key)
                    logging.warning(
                        'Making string from key "%s", %s for use as'
                        ' name in HDF5 file', key_str, type(key)
                    )
                val = node[key]
                new_path = path + [key_str]
                store_recursively(fhandle=fhandle, node=val, path=new_path,
                                  node_hashes=node_hashes)
        else:
            # Check for existing node
            node_hash = hash_obj(node)
            if node_hash in node_hashes:
                logging.trace('  creating hardlink for Dataset: "%s" -> "%s"',
                              full_path, node_hashes[node_hash])
                # Hardlink the matching existing dataset
                fhandle[full_path] = fhandle[node_hashes[node_hash]]
                return

            # For now, convert None to np.nan since h5py appears to not handle
            # None
            if node is None:
                node = np.nan
                logging.warning(
                    '  encountered `None` at node "%s"; converting to'
                    ' np.nan', full_path
                )

            # "Scalar datasets don't support chunk/filter options". Shuffling
            # is a good idea otherwise since subsequent compression will
            # generally benefit; shuffling requires chunking. Compression is
            # not done here since it is slow, but can be done by
            # post-processing the generated file(s).
            if np.isscalar(node):
                shuffle = False
                chunks = None
            else:
                shuffle = True
                chunks = True
                # Store the node_hash for linking to later if this is more than
                # a scalar datatype. Assumed that "None" has
                node_hashes[node_hash] = full_path

            # -- Handle special types -- #

            # See h5py docs at
            #
            #   https://docs.h5py.org/en/stable/strings.html#how-to-store-text-strings
            #
            # where using `bytes` objects (i.e., in numpy, np.string_) is
            # deemed the most compatible way to encode objects, but apparently
            # we don't have pytables compatibility right now.
            #
            # For boolean support, see
            #
            #   https://docs.h5py.org/en/stable/faq.html#faq

            # TODO: make written hdf5 files compatible with pytables
            # see docs at https://www.pytables.org/usersguide/datatypes.html

            if isinstance(node, string_types):
                node = np.string_(node)
            elif isinstance(node, bool):  # includes np.bool
                node = np.bool_(node)  # same as np.bool8
            elif isinstance(node, np.ndarray):
                if issubclass(node.dtype.type, string_types):
                    node = node.astype(np.string_)
                elif node.dtype.type in (bool, np.bool):
                    node = node.astype(np.bool_)

            logging.trace('  creating dataset at path "%s", hash %s',
                          full_path, node_hash)
            try:
                dset = fhandle.create_dataset(
                    name=full_path, data=node, chunks=chunks, compression=None,
                    shuffle=shuffle, fletcher32=False
                )
            except TypeError:
                try:
                    shuffle = False
                    chunks = None
                    dset = fhandle.create_dataset(
                        name=full_path, data=node, chunks=chunks,
                        compression=None, shuffle=shuffle, fletcher32=False
                    )
                except Exception:
                    logging.error('  full_path: "%s"', full_path)
                    logging.error('  chunks   : %s', str(chunks))
                    logging.error('  shuffle  : %s', str(shuffle))
                    logging.error('  node     : "%s"', str(node))
                    raise

            for key in sorted_attr_keys:
                dset.attrs[key] = attrs[key]

    # Perform the actual operation using the dict passed in by user
    if isinstance(tgt, str):
        from pisa.utils.fileio import check_file_exists
        fpath = check_file_exists(fname=tgt, overwrite=overwrite, warn=warn)
        h5file = h5py.File(fpath, 'w')
        try:
            if attrs is not None:
                h5file.attrs.update(attrs)
            store_recursively(fhandle=h5file, node=data_dict)
        finally:
            h5file.close()

    elif isinstance(tgt, h5py.Group):
        store_recursively(fhandle=tgt, node=data_dict, attrs=attrs)

    else:
        raise TypeError('to_hdf: Invalid `tgt` type: %s' % type(tgt))


def test_hdf():
    """Unit tests for hdf module"""
    from shutil import rmtree
    from tempfile import mkdtemp

    data = OrderedDict([
        ('top', OrderedDict([
            ('secondlvl1', OrderedDict([
                ('thirdlvl11', np.linspace(1, 100, 10000).astype(np.float64)),
                ('thirdlvl12', b"this is a string"),
                ('thirdlvl13', b"this is another string"),
                ('thirdlvl14', 1),
                ('thirdlvl15', 1.1),
                ('thirdlvl16', np.float32(1.1)),
                ('thirdlvl17', np.float64(1.1)),
                ('thirdlvl18', np.int8(1)),
                ('thirdlvl19', np.int16(1)),
                ('thirdlvl110', np.int32(1)),
                ('thirdlvl111', np.int64(1)),
                ('thirdlvl112', np.uint8(1)),
                ('thirdlvl113', np.uint16(1)),
                ('thirdlvl114', np.uint32(1)),
                ('thirdlvl115', np.uint64(1)),
            ])),
            ('secondlvl2', OrderedDict([
                ('thirdlvl21', np.linspace(1, 100, 10000).astype(np.float32)),
                ('thirdlvl22', b"this is a string"),
                ('thirdlvl23', b"this is another string"),
            ])),
            ('secondlvl3', OrderedDict([
                ('thirdlvl31', np.array(range(1000)).astype(np.int)),
                ('thirdlvl32', b"this is a string"),
            ])),
            ('secondlvl4', OrderedDict([
                ('thirdlvl41', np.linspace(1, 100, 10000)),
                ('thirdlvl42', b"this is a string"),
            ])),
            ('secondlvl5', OrderedDict([
                ('thirdlvl51', np.linspace(1, 100, 10000)),
                ('thirdlvl52', b"this is a string"),
            ])),
            ('secondlvl6', OrderedDict([
                ('thirdlvl61', np.linspace(100, 1000, 10000)),
                ('thirdlvl62', b"this is a string"),
            ])),
        ]))
    ])

    temp_dir = mkdtemp()
    try:
        fpath = os.path.join(temp_dir, 'to_hdf_noattrs.hdf5')
        to_hdf(data, fpath, overwrite=True, warn=False)
        loaded_data1 = from_hdf(fpath)
        assert data.keys() == loaded_data1.keys()
        assert recursiveEquality(data, loaded_data1), \
                str(data) + "\n" + str(loaded_data1)

        attrs = OrderedDict([
            ('float', 9.98237),
            ('float32', np.float32(1.)),
            ('float64', np.float64(1.)),
            ('pi', np.float64(np.pi)),

            ('string', "string attribute!"),

            ('int', 1),
            ('int8', np.int8(1)),
            ('int16', np.int16(1)),
            ('int32', np.int32(1)),
            ('int64', np.int64(1)),

            ('uint8', np.uint8(1)),
            ('uint16', np.uint16(1)),
            ('uint32', np.uint32(1)),
            ('uint64', np.uint64(1)),

            ('bool', True),
            ('bool8', np.bool8(True)),
            ('bool_', np.bool_(True)),
        ])

        attr_type_checkers = {
            "float": lambda x: isinstance(x, float),
            "float32": lambda x: x.dtype == np.float32,
            "float64": lambda x: x.dtype == np.float64,
            "pi": lambda x: x.dtype == np.float64,

            "string": lambda x: isinstance(x, string_types),

            "int": lambda x: isinstance(x, int),
            "int8": lambda x: x.dtype == np.int8,
            "int16": lambda x: x.dtype == np.int16,
            "int32": lambda x: x.dtype == np.int32,
            "int64": lambda x: x.dtype == np.int64,

            "uint8": lambda x: x.dtype == np.uint8,
            "uint16": lambda x: x.dtype == np.uint16,
            "uint32": lambda x: x.dtype == np.uint32,
            "uint64": lambda x: x.dtype == np.uint64,

            "bool": lambda x: isinstance(x, bool),
            "bool8": lambda x: x.dtype == np.bool8,
            "bool_": lambda x: x.dtype == np.bool_,
        }

        fpath = os.path.join(temp_dir, 'to_hdf_withattrs.hdf5')
        to_hdf(data, fpath, attrs=attrs, overwrite=True, warn=False)
        loaded_data2 = from_hdf(fpath)
        loaded_attrs = loaded_data2.attrs
        assert data.keys() == loaded_data2.keys()
        assert attrs.keys() == loaded_attrs.keys(), \
                '\n' + str(attrs.keys()) + '\n' + str(loaded_attrs.keys())
        assert recursiveEquality(data, loaded_data2)
        assert recursiveEquality(attrs, loaded_attrs)

        for key, val in attrs.items():
            tgt_type_checker = attr_type_checkers[key]
            assert tgt_type_checker(val), \
                    "key '%s': val '%s' is type '%s'" % \
                    (key, val, type(loaded_attrs[key]))
    finally:
        rmtree(temp_dir)

    logging.info('<< PASS : test_hdf >>')


if __name__ == "__main__":
    set_verbosity(1)
    test_hdf()
