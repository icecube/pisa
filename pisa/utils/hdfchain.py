"""
class to access hdf5 files chained together.
"""
from __future__ import print_function


import numpy as n
import tables
from glob import glob
from collections import defaultdict


__all__ = ['HDFTableProxy', 'TableAccessor', 'HDFChain']


class HDFTableProxy(object):
    def __init__(self, table, files):
        self.path = str(table._v_pathname)
        self._v_dtype = table.description._v_dtype
        self.files = files

    def read(self):
        # first loop to calculate number of rows
        lengths = n.zeros(len(self.files), dtype=int)
        for i, file in enumerate(self.files):
            try:
                lengths[i] = len(file.getNode(self.path))
            except tables.NoSuchNodeError:
                print("WARN: node %s does not exist in file %s" % (self.path, file.filename))
                lengths[i] = 0

        # create result array ...
        result = n.zeros(lengths.sum(), dtype=self._v_dtype)
        
        # .. and fill it
        for i, file in enumerate(self.files):
            if lengths[i] == 0:
                continue
            result[lengths[:i].sum():lengths[:i].sum()+lengths[i]] = file.getNode(self.path).read()

        return result

    def read_iter(self):
        for i, file in enumerate(self.files):
            yield file.getNode(self.path).read()
    
    def col_iter(self, colname):
        for i, file in enumerate(self.files):
            yield file.getNode(self.path).col(colname)

    def col(self, colname):
        dtype = self._v_dtype[colname]
        # first loop to calculate number of rows
        lengths = n.zeros(len(self.files), dtype=int)
        #print "INFO: counting rows"
        for i, file in enumerate(self.files):
            try:
                lengths[i] = len(file.getNode(self.path))
            except tables.NoSuchNodeError:
                print("WARN: node %s does not exist in file %s" % (self.path, file.filename))
                lengths[i] = 0

        # create result array ...
        result = n.zeros(lengths.sum(), dtype=dtype)
        
        # .. and fill it
        for i, file in enumerate(self.files):
            #print "INFO: read %d/%d" % (i+1, len(self.files))
            if lengths[i] == 0:
                continue
            result[lengths[:i].sum():lengths[:i].sum()+lengths[i]] = file.getNode(self.path).col(colname)

        return result

    def __len__(self):
        length = 0
        for i, file in enumerate(self.files):
            length += len(file.getNode(self.path))
        return length

    def __repr__(self):
        return ("chained table with %d files:\n" % len(self.files))+self.files[0].getNode(self.path).__repr__()

class TableAccessor(object):
    def __init__(self, tabledict):
        for tabname, proxy in tabledict.items():
            self.__dict__[tabname] = proxy

    def __repr__(self):
        return ", ".join([key for (key,value) in self.__dict__.items() if type(value) is HDFTableProxy])

class HDFChain(object):
    def __init__(self, files, maxdepth=1, verbose=False, **kwargs):
        """ 
            setup a chain of hdf files. 
            files is either a list of filenames or a glob string
            kwargs are passed to tables.openFile (e.g. NODE_CACHE_SLOTS)
        """

        self.files = list()
        self._tables = defaultdict(HDFTableProxy)
        self.verbose = verbose
        self.pathes = dict()

        if self.verbose:
            print("opening files in chain...")
        if type(files) is list:
            if len(files) == 0:
                raise ValueError("provided file list is empty!")
            self.files = [tables.openFile(fname, **kwargs) for fname in files ]
        elif type(files) is str:
            self.files = [tables.openFile(fname, **kwargs) for fname in sorted(glob(files)) ]
            if len(self.files) == 0:
                raise ValueError("glob string matches no file!")
        else:
            raise ValueError("parameter files must be either a list of filenames or a globstring")


        file = self.files[0]
        if self.verbose:
            print("walking through first file %s" % file.filename)
        for table in file.walkNodes(classname="Table"):
            if table._v_depth > maxdepth:
                continue
            if table.name in self._tables:
                print("WARN: skipping additional occurence of table %s at %s (using %s)!" % (table.name,
                      table._v_pathname, self._tables[table.name].path))
                continue
            else:
                proxy = HDFTableProxy(table, self.files)
                self._tables[table.name] = proxy
                self.pathes[table._v_pathname] = proxy

        self.root = TableAccessor(self._tables)

    def __del__(self):
        for tabname, tabproxy in self._tables.items():
            tabproxy.file = None

        for file in self.files:
            file.close()


    def getNode(self, path):
        return self.pathes[path]

