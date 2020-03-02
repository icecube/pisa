"""
MemoryCache and DiskCache classes to store long-to-compute results.
"""


from __future__ import absolute_import

from collections import OrderedDict
import copy
import os
import pickle
import re
import sqlite3
import shutil
import tempfile
import time

from pisa.utils.log import logging, set_verbosity


__all__ = ['MemoryCache', 'DiskCache',
           'test_MemoryCache', 'test_DiskCache']

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


class MemoryCache(object):
    """Simple implementation of a first-in-first-out (FIFO) or least-recently-
    used (LRU) in-memory cache, with a subset of the dict interface.

    Parameters
    ----------
    max_depth : int >= 0
        Maximum number of entries in the cache, pruned either by FIFO or LRU
        logic. A `max_depth` of 0 effectively disables caching altogether
        (e.g., useful for testing without removing the caches).

    is_lru : bool
        Whether to implement LRU logic (True) or FIFO logic (False). LRU logic
        adds a small computational cost.

    deepcopy : bool
        Whether to make deep copies of objects as they are stored to and
        returned from the cache. This can guard aganst an object in the cache
        being modifed after it has been stored to the cache.

    Attributes
    ----------
    GLOBAL_MEMCACHE_DEPTH_OVERRIDE : None or int >= 0
        Set to an integer to override the cache depth for *all* memory caches.
        E.g., set this to 0 to disable caching everywhere.

    Notes
    -----
    Based off of code at www.kunxi.org/blog/2014/05/lru-cache-in-python

    """
    GLOBAL_MEMCACHE_DEPTH_OVERRIDE = None
    def __init__(self, max_depth, is_lru=True, deepcopy=False):
        self.__cache = OrderedDict()
        self.__max_depth = max_depth
        self.__is_lru = is_lru
        self.__deepcopy = deepcopy
        if self.GLOBAL_MEMCACHE_DEPTH_OVERRIDE is not None:
            self.__max_depth = self.GLOBAL_MEMCACHE_DEPTH_OVERRIDE
        assert isinstance(self.__max_depth, int), \
                '`max_depth` must be int; got %s' % type(self.__max_depth)
        assert self.__max_depth >= 0, \
                '`max_depth` must be >= 0; got %s' % self.__max_depth

    def __str__(self):
        return 'MemoryCache(max_depth=%d, is_lru=%s)' % (self.__max_depth,
                                                         self.__is_lru)

    def __repr__(self):
        return str(self) + '; %d keys:\n%s' % (len(self.__cache),
                                               self.__cache.keys())

    def __getitem__(self, key):
        if key is None:
            raise KeyError(
                '`None` is not a valid cache key, so nothing can live there.'
            )
        value = self.__cache[key]
        if self.__is_lru:
            del self.__cache[key]
            self.__cache[key] = value
        if self.__deepcopy:
            value = copy.deepcopy(value)
        return value

    def __setitem__(self, key, value):
        if key is None:
            raise KeyError(
                '`None` is not a valid cache key, so nothing can live there.'
            )
        if self.__max_depth == 0:
            return
        # Same logic here for LRU and FIFO
        try:
            del self.__cache[key]
        except KeyError:
            if len(self) >= self.__max_depth:
                self.__cache.popitem(last=False)
        if self.__deepcopy:
            value = copy.deepcopy(value)
        self.__cache[key] = value

    def __contains__(self, key):
        return key in self.__cache

    def __delitem__(self, key):
        return self.__cache.__delitem__(self, key)

    def __iter__(self):
        return iter(self.__cache)

    def __len__(self):
        return len(self.__cache)

    def __reversed__(self):
        return reversed(self.__cache)

    def clear(self):
        return self.__cache.clear()

    def get(self, key, dflt=None):
        if key in self.__cache:
            return self[key]
        return dflt

    def keys(self):
        return self.__cache.keys()

    def pop(self, k):
        value = self.__cache.pop(k)
        if self.__deepcopy:
            value = copy.deepcopy(value)
        return value

    def popitem(self, last=True):
        return self.pop(last)

    def setdefault(self, key, default=None):
        if not key in self:
            self[key] = default
        return self[key]

    def values(self):
        vals = self.__cache.values()
        if self.__deepcopy:
            vals = [copy.deepcopy(v) for v in vals]
        return vals


class DiskCache(object):
    """
    Implements a subset of dict methods but with persistent storage to an on-
    disk sqlite database.

    Parameters
    ----------
    db_fpath : str
        Path to database file; if existing file is specified, schema must match
        that specified by DiskCache.TABLE_SCHEMA.

    max_depth : int
        Limit on the number of rows in the database's table. Pruning is either
        via first-in-first-out (FIFO) or least-recently-used (LRU) logic.

    is_lru : bool
        If True, implement least-recently-used (LRU) logic for removing items
        beyond `max_depth`. This adds an additional ~400 ms to item retrieval
        time. Otherwise, behaves as a first-in-first-out (FIFO) cache.

    Notes
    -----
    This is not (as of now) thread-safe, but it is multi-process safe. The
    Python sqlite3 api requires a single Python process to have just a single
    connection to a database at a time. Several processes can be safely
    connected to the database simultaneously, however, due to sqlite's locking
    mechanisms that resolve resource contention.

    Large databases are slower to work with than small. Therefore it is
    recommended to use separate databases for each stage's cache rather than
    one centralized database acting as the cache for all stages.

    Examples
    --------
    Access to the database via dict-like syntax:

    >>> x = {'xyz': [0,1,2,3], 'abc': {'first': (4,5,6)}}
    >>> disk_cache = DiskCache('/tmp/diskcache.db', max_depth=5, is_lru=False)
    >>> disk_cache[12] = x
    >>> disk_cache[13] = x
    >>> disk_cache[14] = x
    >>> y = disk_cache[12]
    >>> print(y == x)
    True
    >>> len(disk_cache)
    3
    >>> disk_cache.keys()
    [12, 13, 14]
    >>> del disk_cache[12]
    >>> len(disk_cache)
    2
    >>> disk_cache.keys()
    [13, 14]
    >>> disk_cache.clear()
    >>> len(disk_cache)
    0
    >>> disk_cache.keys()
    []

    Demonstrate max_depth (limit on number of entries / cache depth)

    >>> x = [disk_cache.__setitem__(i, 'foo') for i in range(10)]
    >>> len(disk_cache)
    5

    """
    TABLE_SCHEMA = \
        '''CREATE TABLE cache (hash INTEGER PRIMARY KEY,
                               accesstime INTEGER,
                               data BLOB)'''
    def __init__(self, db_fpath, max_depth=100, is_lru=False):
        self.__db_fpath = os.path.expandvars(os.path.expanduser(db_fpath))
        self.__instantiate_db()
        assert 0 < max_depth < 1e6, 'Invalid `max_depth`:' + str(max_depth)
        self.__max_depth = max_depth
        self.__is_lru = is_lru

    @property
    def path(self):
        return self.__db_fpath

    def __instantiate_db(self):
        exists = True if os.path.isfile(self.__db_fpath) else False

        # Create the directory in with the database file will be created
        if not exists:
            dirpath, _ = os.path.split(self.__db_fpath)
            if not os.path.isdir(dirpath) and dirpath != '':
                os.makedirs(dirpath)

        conn = self.__connect()
        try:
            if exists:
                # Check that the table format is valid
                sql = ("SELECT sql FROM sqlite_master WHERE type='table' AND"
                       " NAME='cache'")
                cursor = conn.execute(sql)
                schema, = cursor.fetchone()
                # Ignore formatting
                schema = re.sub(r'\s', '', schema).lower()
                ref_schema = re.sub(r'\s', '', self.TABLE_SCHEMA).lower()
                if schema != ref_schema:
                    raise ValueError('Existing database at "%s" has'
                                     'non-matching schema:\n"""%s"""'
                                     %(self.__db_fpath, schema))
            else:
                # Create the table for storing (hash, data, timestamp) tuples
                conn.execute(self.TABLE_SCHEMA)
                sql = "CREATE INDEX idx0 ON cache(hash)"
                conn.execute(sql)
                sql = "CREATE INDEX idx1 ON cache(accesstime)"
                conn.execute(sql)
                conn.commit()
        except:
            conn.rollback()
            raise
        finally:
            conn.close()

    def __str__(self):
        s = 'DiskCache(db_fpath=%s, max_depth=%d, is_lru=%s)' % \
                (self.__db_fpath, self.__max_depth, self.__is_lru)
        return s

    def __repr__(self):
        return str(self) + '; %d keys:\n%s' % (len(self), self.keys())

    def __getitem__(self, key):
        if key is None:
            raise KeyError(
                '`None` is not a valid cache key, so nothing can live there.'
            )
        t0 = time.time()
        if not isinstance(key, int):
            raise KeyError('`key` must be int, got "%s"' % type(key))
        conn = self.__connect()
        t1 = time.time()
        logging.trace('conn: %0.4f' % (t1 - t0))
        try:
            if self.__is_lru:
                # Update accesstime
                sql = "UPDATE cache SET accesstime = ? WHERE hash = ?"
                conn.execute(sql, (self.now, key))
                t2 = time.time()
                logging.trace('update: % 0.4f' % (t2 - t1))
            t2 = time.time()

            # Retrieve contents
            sql = "SELECT data FROM cache WHERE hash = ?"
            cursor = conn.execute(sql, (key,))
            t3 = time.time()
            logging.trace('select: % 0.4f' % (t3 - t2))
            tmp = cursor.fetchone()
            if tmp is None:
                raise KeyError(str(key))
            data = tmp[0]
            t4 = time.time()
            logging.trace('fetch: % 0.4f' % (t4 - t3))
            data = pickle.loads(bytes(data))
            t5 = time.time()
            logging.trace('loads: % 0.4f' % (t5 - t4))
        finally:
            conn.commit()
            conn.close()
            logging.trace('')
        return data

    def __setitem__(self, key, obj):
        if key is None:
            raise KeyError(
                '`None` is not a valid cache key, so nothing can live there.'
            )
        t0 = time.time()
        assert isinstance(key, int)
        data = sqlite3.Binary(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
        t1 = time.time()
        logging.trace('dumps: % 0.4f' % (t1 - t0))

        conn = self.__connect()
        t2 = time.time()
        logging.trace('conn: % 0.4f' % (t2 - t1))
        try:
            t = time.time()
            sql = "INSERT INTO cache (hash, accesstime, data) VALUES (?, ?, ?)"
            conn.execute(sql, (key, self.now, data))
            t1 = time.time()
            logging.trace('insert: % 0.4f' % (t1 - t))
            conn.commit()
            t2 = time.time()
            logging.trace('commit: % 0.4f' % (t2 - t1))

            # Remove oldest-accessed rows in excess of limit
            n_to_remove = len(self) - self.__max_depth
            t3 = time.time()
            logging.trace('len: % 0.4f' % (t3 - t2))
            if n_to_remove <= 0:
                return

            # Find the access time of the n-th element (sorted ascending by
            # access time)
            sql = ("SELECT accesstime FROM cache ORDER BY accesstime ASC LIMIT"
                   " 1 OFFSET ?")
            cursor = conn.execute(sql, (n_to_remove,))
            t4 = time.time()
            logging.trace('select old: % 0.4f' % (t4 - t3))
            nth_access_time, = cursor.fetchone()
            t5 = time.time()
            logging.trace('fetch: % 0.4f' % (t5 - t4))

            # Remove all elements that old or older
            sql = "DELETE FROM cache WHERE accesstime < ?"
            conn.execute(sql, (nth_access_time,))
            t6 = time.time()
            logging.trace('delete: % 0.4f' % (t6 - t5))
        except:
            t = time.time()
            conn.rollback()
            logging.trace('rollback: % 0.4f' % (time.time() - t))
            raise
        else:
            t = time.time()
            conn.commit()
            logging.trace('commit: % 0.4f' % (time.time() - t))
        finally:
            t = time.time()
            conn.close()
            logging.trace('close: % 0.4f' % (time.time() - t))
            logging.trace('')

    def __delitem__(self, key):
        conn = self.__connect()
        try:
            sql = "DELETE FROM cache WHERE hash = ?"
            conn.execute(sql, (key,))
        except:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()

    def __len__(self):
        conn = self.__connect()
        try:
            cursor = conn.execute('SELECT COUNT (*) FROM cache')
            count, = cursor.fetchone()
        finally:
            conn.close()
        return count

    def get(self, key, dflt=None):
        rslt = dflt
        try:
            rslt = self.__getitem__(key)
        except KeyError:
            pass
        return rslt

    def clear(self):
        conn = self.__connect()
        try:
            conn.execute('DELETE FROM cache')
        except:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()

    def keys(self):
        conn = self.__connect()
        try:
            sql = "SELECT hash FROM cache ORDER BY accesstime ASC"
            cursor = conn.execute(sql)
            k = [k[0] for k in cursor.fetchall()]
        finally:
            conn.close()
        return k

    def __connect(self):
        conn = sqlite3.connect(
            self.__db_fpath,
            isolation_level=None, check_same_thread=False, timeout=10,
        )

        # Trust journaling to memory
        sql = "PRAGMA journal_mode=MEMORY"
        conn.execute(sql)

        # Trust OS to complete transaction
        sql = "PRAGMA synchronous=0"
        conn.execute(sql)

        # Make file size shrink when items are removed
        sql = "PRAGMA auto_vacuum=FULL"
        conn.execute(sql)

        return conn

    def __contains__(self, key):
        try:
            _ = self[key]
            return True 
        except KeyError:
            return False

    @property
    def now(self):
        """Microseconds since the epoch"""
        return int(time.time() * 1e6)


# TODO: augment test
def test_MemoryCache():
    """Unit tests for MemoryCache class"""
    mc = MemoryCache(max_depth=3, is_lru=True)
    assert 0 not in mc
    mc[0] = 'zero'
    assert mc[0] == 'zero'
    mc[1] = 'one'
    mc[2] = 'two'
    mc[3] = 'three'
    assert 0 not in mc
    assert mc[3] == 'three'

    # Test if objects modified outside of cache are modified inside cache
    for deepcopy in [True, False]:
        mc = MemoryCache(max_depth=3, is_lru=True, deepcopy=deepcopy)
        x_ref = {'k0': 'abc'}
        x = copy.deepcopy(x_ref)
        mc[4] = x
        x['k1'] = 'xyz'
        y = mc[4]
        assert (y == x_ref) == deepcopy

    logging.info('<< PASS : test_MemoryCache >>')


# TODO: augment test
def test_DiskCache():
    """Unit tests for DiskCache class"""
    testdir = tempfile.mkdtemp()
    try:
        # Testing that subfolders get created, hence the long pathname
        tmp_fname = os.path.join(
            testdir, 'subfolder1/subfolder2/DiskCache.sqlite'
        )
        dc = DiskCache(db_fpath=tmp_fname, max_depth=3, is_lru=False)
        assert 0 not in dc
        dc[0] = 'zero'
        assert dc[0] == 'zero'
        dc[1] = 'one'
        dc[2] = 'two'
        dc[3] = 'three'
        assert 0 not in dc
        assert dc[3] == 'three'
    finally:
        shutil.rmtree(testdir, ignore_errors=True)

    logging.info('<< PASS : test_DiskCache >>')


if __name__ == "__main__":
    set_verbosity(1)
    test_MemoryCache()
    test_DiskCache()
