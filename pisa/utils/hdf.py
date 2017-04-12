# author: Sebastian Boeser
#         sboeser@uni-mainz.de
#
# date:   2015-03-05
"""Set of utilities for handling HDF5 file I/O"""


import os

import numpy as np
import h5py

from pisa.utils.log import logging, set_verbosity
from pisa.utils.hash import hash_obj
from pisa.utils.resources import find_resource
from pisa.utils.comparisons import recursiveEquality


__all__ = ['HDF5_EXTS',
           'from_hdf', 'to_hdf',
           'test_hdf']


HDF5_EXTS = ['hdf', 'h5', 'hdf5']


# TODO: convert to use OrderedDict to preserve ordering
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
    data : dict
        Nested dictionary; keys are HDF5 node names and values contain the
        contents of that node.

    (attrs : dict)
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

    # Function for iteratively parsing the file to create the dictionary
    def visit_group(obj, sdict):
        name = obj.name.split('/')[-1]
        if type(obj) in [h5py.Dataset]:
            sdict[name] = obj.value
        if type(obj) in [h5py.Group, h5py.File]:
            sdict[name] = {}
            for sobj in obj.values():
                visit_group(sobj, sdict[name])

    data = {}
    attrs = {}
    myfile = False
    if isinstance(val, basestring):
        root = h5py.File(find_resource(val), 'r')
        myfile = True
    else:
        root = val
        logging.trace('root = %s, root.values() = %s' % (root, root.values()))
    try:
        # Retrieve attrs if told to return attrs
        if return_attrs and hasattr(root, 'attrs'):
            attrs = dict(root.attrs)
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
    data_dict : dict
        Dictionary to be stored
    tgt : str or h5py.Group
        Target for storing data. If `tgt` is a str, it is interpreted as a
        filename; a file is created with that name (overwriting an existing
        file, if present). After writing, the file is closed. If `tgt` is an
        h5py.Group, the data is simply written to that Group and it is left
        open at function return.
    attrs : dict
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
    if not isinstance(data_dict, dict):
        errmsg = 'to_hdf: `data_dict` only accepts top-level dict.'
        logging.error(errmsg)
        raise TypeError(errmsg)

    # Define a function for interatively doing the work
    def store_recursively(fhandle, node, path=None, attrs=None,
                          node_hashes=None):
        path = [] if path is None else path
        node_hashes = {} if node_hashes is None else node_hashes
        full_path = '/' + '/'.join(path)
        if isinstance(node, dict):
            logging.trace("  creating Group '%s'" % full_path)
            try:
                dset = fhandle.create_group(full_path)
                if attrs is not None:
                    for key in sorted(attrs.keys()):
                        dset.attrs[key] = attrs[key]
            except ValueError:
                pass
            for key in sorted(node.keys()):
                key_str = str(key)
                if not isinstance(key, str):
                    logging.warn("Stringifying key '" + key_str +
                                 "'for use as name in HDF5 file")
                val = node[key]
                new_path = path + [key_str]
                store_recursively(fhandle=fhandle, node=val, path=new_path,
                                  node_hashes=node_hashes)
        else:
            # Check for existing node
            node_hash = hash_obj(node)
            if node_hash in node_hashes:
                logging.trace("  creating hardlink for Dataset: '%s' -> '%s'" %
                              (full_path, node_hashes[node_hash]))
                # Hardlink the matching existing dataset
                fhandle[full_path] = fhandle[node_hashes[node_hash]]
                return
            # For now, convert None to np.nan since h5py appears to not handle
            # None
            if node is None:
                node = np.nan
                logging.warn("  encountered `None` at node '%s'; converting to"
                             " np.nan" % full_path)
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
            if isinstance(node, basestring):
                # TODO: Treat strings as follows? Would this break
                # compatibility with pytables/Pandas? What are benefits?
                # Leaving the following two lines out for now...

                #dtype = h5py.special_dtype(vlen=str)
                #fh.create_dataset(k,data=v,dtype=dtype)

                # ... Instead: creating length-1 array out of string; this
                # seems to be compatible with both h5py and pytables
                node = np.array(node)

            logging.trace("  creating dataset at node '%s', hash %s" %
                          (full_path, node_hash))
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
                    logging.error('  full_path: ' + full_path)
                    logging.error('  chunks   : ' + str(chunks))
                    logging.error('  shuffle  : ' + str(shuffle))
                    logging.error('  node     : ' + str(node))
                    raise

            if attrs is not None:
                for key in sorted(attrs.keys()):
                    dset.attrs[key] = attrs[key]

    # Perform the actual operation using the dict passed in by user
    if isinstance(tgt, basestring):
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
        errmsg = "to_hdf: Invalid `tgt` type: " + type(tgt)
        logging.error(errmsg)
        raise TypeError(errmsg)


def test_hdf():
    from shutil import rmtree
    from tempfile import mkdtemp

    data = {
        'top': {
            'secondlvl1':{
                'thirdlvl11': np.linspace(1, 100, 10000),
                'thirdlvl12': "this is a string"
            },
            'secondlvl2':{
                'thirdlvl21': np.linspace(1, 100, 10000),
                'thirdlvl22': "this is a string"
            },
            'secondlvl3':{
                'thirdlvl31': np.linspace(1, 100, 10000),
                'thirdlvl32': "this is a string"
            },
            'secondlvl4':{
                'thirdlvl41': np.linspace(1, 100, 10000),
                'thirdlvl42': "this is a string"
            },
            'secondlvl5':{
                'thirdlvl51': np.linspace(1, 100, 10000),
                'thirdlvl52': "this is a string"
            },
            'secondlvl6':{
                'thirdlvl61': np.linspace(100, 1000, 10000),
                'thirdlvl62': "this is a string"
            },
        }
    }

    temp_dir = mkdtemp()
    try:
        fpath = os.path.join(temp_dir, 'to_hdf_noattrs.hdf5')
        to_hdf(data, fpath,
               overwrite=True, warn=False)
        loaded_data1 = from_hdf(fpath)
        assert recursiveEquality(data, loaded_data1)

        attrs = {
            'float1': 9.98237,
            'float2': 1.,
            'pi': np.pi,
            'string': "string attribute!",
            'int': 1
        }
        fpath = os.path.join(temp_dir, 'to_hdf_withattrs.hdf5')
        to_hdf(data, fpath, attrs=attrs, overwrite=True, warn=False)
        loaded_data2, loaded_attrs = from_hdf(fpath, return_attrs=True)
        assert recursiveEquality(data, loaded_data2)
        assert recursiveEquality(attrs, loaded_attrs)

        for k, v in attrs.iteritems():
            tgt_type = type(attrs[k])
            assert isinstance(loaded_attrs[k], tgt_type), \
                    "key %s: val '%s' is type '%s' but should be '%s'" % \
                    (k, v, type(loaded_attrs[k]), tgt_type)
    finally:
        rmtree(temp_dir)

    logging.info('<< PASSED : test_hdf >>')


if __name__ == "__main__":
    set_verbosity(1)
    test_hdf()
