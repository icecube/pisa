"""Set of utilities for handling HDF5 file I/O"""


from __future__ import absolute_import

from collections.abc import Mapping
from collections import OrderedDict
import os

import numpy as np
import h5py

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


def from_hdf(val, return_node=None, return_attrs=False):
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

    return_attrs : bool
        Whether to return attrs attached to entry-level entity

    Returns
    -------
    data : OrderedDict
        Nested dictionary; keys are HDF5 node names and values contain the
        contents of that node.

    (attrs : OrderedDict)
        Attributes of entry-level entity; only returned if return_attrs=True

    """
    if return_node is not None:
        raise NotImplementedError('`return_node` is not yet implemented.')

    # NOTE: It's generally sub-optimal to have different return type signatures
    # (1 or 2 return values in this case), but defaulting to a single return
    # value (just returning `data`) preserves compatibility with
    # previously-written routines that just assume a single return value; only
    # when the caller explicitly specifies for the function to do so is the
    # second return value returned, which seems the safest compromise for now.

    def visit_group(obj, sdict):
        """Iteratively parse `obj` to create the dictionary `sdict`"""
        name = obj.name.split('/')[-1]
        if isinstance(obj, h5py.Dataset):
            sdict[name] = obj[()]
        if isinstance(obj, (h5py.Group, h5py.File)):
            sdict[name] = OrderedDict()
            for sobj in obj.values():
                visit_group(sobj, sdict[name])

    data = OrderedDict()
    attrs = OrderedDict()
    myfile = False
    if isinstance(val, str):
        try:
            root = h5py.File(find_resource(val), 'r')
        except:
            logging.error('Failed to load HDF5 file, `val`="%s"', val)
            raise
        myfile = True
    else:
        root = val
        logging.trace('root = %s, root.values() = %s', root, root.values())
    try:
        # Retrieve attrs if told to return attrs
        if return_attrs and hasattr(root, 'attrs'):
            attrs = OrderedDict(root.attrs)
        # Run over the whole dataset
        for obj in root.values():
            visit_group(obj, data)
    finally:
        if myfile:
            root.close()

    if return_attrs:
        return data, attrs

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
        """Function for interatively doing the work"""
        path = [] if path is None else path
        node_hashes = OrderedDict() if node_hashes is None else node_hashes
        full_path = '/' + '/'.join(path)
        if attrs is not None:
            if isinstance(attrs, OrderedDict):
                sorted_attr_keys = attrs.keys()
            else:
                sorted_attr_keys = sorted(attrs.keys())
        if isinstance(node, Mapping):
            logging.trace('  creating Group "%s"', full_path)
            try:
                dset = fhandle.create_group(full_path)
                if attrs is not None:
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
            if isinstance(node, str):
                # TODO: Treat strings as follows? Would this break
                # compatibility with pytables/Pandas? What are benefits?
                # Leaving the following two lines out for now...

                #dtype = h5py.special_dtype(vlen=str)
                #fh.create_dataset(k,data=v,dtype=dtype)

                # ... Instead: creating length-1 array out of string; this
                # seems to be compatible with both h5py and pytables
                node = np.array(node)

            logging.trace('  creating dataset at node "%s", hash %s',
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
                except:
                    logging.error('  full_path: %s', full_path)
                    logging.error('  chunks   : %s', str(chunks))
                    logging.error('  shuffle  : %s', str(shuffle))
                    logging.error('  node     : %s', str(node))
                    raise

            if attrs is not None:
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
                ('thirdlvl11', np.linspace(1, 100, 10000)),
                ('thirdlvl12', "this is a string")
            ])),
            ('secondlvl2', OrderedDict([
                ('thirdlvl21', np.linspace(1, 100, 10000)),
                ('thirdlvl22', "this is a string")
            ])),
            ('secondlvl3', OrderedDict([
                ('thirdlvl31', np.linspace(1, 100, 10000)),
                ('thirdlvl32', "this is a string")
            ])),
            ('secondlvl4', OrderedDict([
                ('thirdlvl41', np.linspace(1, 100, 10000)),
                ('thirdlvl42', "this is a string")
            ])),
            ('secondlvl5', OrderedDict([
                ('thirdlvl51', np.linspace(1, 100, 10000)),
                ('thirdlvl52', "this is a string")
            ])),
            ('secondlvl6', OrderedDict([
                ('thirdlvl61', np.linspace(100, 1000, 10000)),
                ('thirdlvl62', "this is a string")
            ])),
        ]))
    ]) # yapf: disable

    temp_dir = mkdtemp()
    try:
        fpath = os.path.join(temp_dir, 'to_hdf_noattrs.hdf5')
        to_hdf(data, fpath, overwrite=True, warn=False)
        loaded_data1 = from_hdf(fpath)
        assert data.keys() == loaded_data1.keys()
        assert recursiveEquality(data, loaded_data1)

        attrs = OrderedDict([
            ('float1', 9.98237),
            ('float2', 1.),
            ('pi', np.pi),
            ('string', "string attribute!"),
            ('int', 1)
        ]) # yapf: disable
        fpath = os.path.join(temp_dir, 'to_hdf_withattrs.hdf5')
        to_hdf(data, fpath, attrs=attrs, overwrite=True, warn=False)
        loaded_data2, loaded_attrs = from_hdf(fpath, return_attrs=True)
        assert data.keys() == loaded_data2.keys()
        assert attrs.keys() == loaded_attrs.keys(), \
                '\n' + str(attrs.keys()) + '\n' + str(loaded_attrs.keys())
        assert recursiveEquality(data, loaded_data2)
        assert recursiveEquality(attrs, loaded_attrs)

        for k, v in attrs.items():
            tgt_type = type(attrs[k])
            assert isinstance(loaded_attrs[k], tgt_type), \
                    "key %s: val '%s' is type '%s' but should be '%s'" % \
                    (k, v, type(loaded_attrs[k]), tgt_type)
    finally:
        rmtree(temp_dir)

    logging.info('<< PASS : test_hdf >>')


if __name__ == "__main__":
    set_verbosity(1)
    test_hdf()
