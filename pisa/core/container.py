"""
Class to hold generic data in container.
he data can be unbinned or binned or scalar, while
translation methods between such different representations
are provided.

The data lives in SmartArrays on both CPU and GPU
"""
from __future__ import absolute_import, print_function

from collections.abc import Sequence
from collections import OrderedDict
from itertools import chain

import numpy as np
from numba import SmartArray

from pisa import FTYPE
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.core.translation import histogram, lookup, resample
from pisa.utils.log import logging


class ContainerSet(object):
    '''
    Class to hold a set of container objects


    Parameters
    ----------
    
    name : str

    containers : list or None

    data_specs : MultiDimBinning, 'events' or None


    '''
    def __init__(self, name, containers=None, data_specs=None):
        self.name = name
        self.linked_containers = []
        self.containers = []
        if containers is None:
            containers = []
        for container in containers:
            self.add_container(container)
        self._data_specs = None
        self.data_specs = data_specs

    def add_container(self, container):
        if container.name in self.names:
            raise ValueError('container with name %s already exists'%container.name)
        self.containers.append(container)

    @property
    def data_mode(self):
        '''
        The data mode can be 'events', 'binned' or None,
        depending on the set data_specs
        '''
        if self.data_specs == 'events':
            return 'events'
        elif isinstance(self.data_specs, MultiDimBinning):
            return 'binned'
        elif self.data_specs is None:
            return None

    @property
    def data_specs(self):
        return self._data_specs

    @data_specs.setter
    def data_specs(self, data_specs):
        '''

        Parameters
        ----------

        data_specs : str, MultiDimBinning or None

        Data specs should be set to retreive the right representation
        i.e. the representation one is working in at the moment

        This property is meant to be changed while working with a ContainerSet

        '''
        if not (data_specs == 'events' or isinstance(data_specs, MultiDimBinning) or data_specs is None):
            raise ValueError('cannot understand data_specs %s'%data_specs)
        self._data_specs = data_specs
        for container in self:
            container.data_specs = self._data_specs

    @property
    def names(self):
        return [c.name for c in self.containers]

    def link_containers(self, key, names):
        '''
        Parameters
        ----------

        key : str
            name of linked object

        names : list
            name of containers to be linked under the given key

        when containers are linked, they are treated as a single (virtual) container for binned data
        '''
        containers = [self.__getitem__(name) for name in names]
        logging.debug('Linking containers %s into %s'%(names, key))
        new_container = VirtualContainer(key, containers)
        self.linked_containers.append(new_container)


    def unlink_containers(self):
        '''
        unlink all container
        '''
        logging.debug('Unlinking all containers')
        for c in self.linked_containers:
            c.unlink()
        self.linked_containers = []

    def __getitem__(self, key):
        if key in self.names:
            return self.containers[self.names.index(key)]

    def __iter__(self):
        '''
        iterate over individual non-linked containers and virtual containers for the ones that are linked together
        '''

        containers_to_be_iterated = [c for c in self.containers if not c.linked] + self.linked_containers
        return iter(containers_to_be_iterated)

    def get_mapset(self, key, error=None):
        '''
        Parameters
        ----------

        key : str

        error : None or str
            specify a key that errors are read from

        For a given key, get a PISA MapSet
        '''
        maps = []
        for container in self:
            maps.append(container.get_map(key, error=error))
        return MapSet(name=self.name, maps=maps)

class VirtualContainer(object):
    '''
    Class providing a virtual container for linked individual containers

    It should just behave like a normal container

    For reading, it just uses one container as a representative (no checkng at the mment
    if the others actually contain the same data)

    For writting, it creates one object that is added to all containers

    Parameters
    ----------

    name : str

    containers : list

    '''

    def __init__(self, name, containers):
        self.name = name
        # check and set link flag
        for container in containers:
            if container.linked:
                raise ValueError('Cannot link container %s since it is already linked'%container.name)
            container.linked = True
        self.containers = containers

    def unlink(self):
        '''
        reset link flag
        '''
        for container in self:
            container.linked = False

    def __iter__(self):
        return iter(self.containers)

    def __getitem__(self, key):
        # should we check they're all the same?
        return self.containers[0][key]

    def __setitem__(self, key, value):
        self.containers[0][key] = value
        for container in self.containers[1:]:
            if np.isscalar(value):
                container.scalar_data[key] = self.containers[0].scalar_data[key]
            else:
                container.binned_data[key] = self.containers[0].binned_data[key]

    @property
    def size(self):
        '''
        size of container
        '''
        return self.containers[0].size



class Container(object):
    '''
    Class to hold data in the form of event arrays and/or maps

    for maps, a binning must be provided set

    Parameters
    ----------
    name : string
        identifier

    binning : PISA MultiDimBinning
        binning, if binned data is used

    code : int
        could hold for example a PDG code

    data_specs : str, MultiDimBinning or None
        the representation one is working in at the moment
    '''

    def __init__(self, name, code=None, data_specs=None):
        self.name = name
        self.code = code
        self.array_length = None
        self.scalar_data = OrderedDict()
        self.array_data = OrderedDict()
        self.binned_data = OrderedDict()
        self.data_specs = data_specs
        self.linked = False

    @property
    def data_mode(self):
        if self.data_specs == 'events':
            return 'events'
        elif isinstance(self.data_specs, MultiDimBinning):
            return 'binned'
        elif self.data_specs is None:
            return None

    def keys(self):
        '''
        return list of available keys
        '''
        if self.data_mode == 'events':
            return chain(self.array_data.keys(), self.scalar_data.keys())
        elif self.data_mode == 'binned':
            return chain(self.array_data.keys(), self.scalar_data.keys(), self.data_specs.names)
        else:
            raise ValueError('Need to set data specs first')

    @ property
    def size(self):
        '''
        length of event arrays or number of bins for binned data
        '''
        if self.data_mode is None:
            raise ValueError('data_mode needs to be set first')
        if self.data_mode == 'events':
            return self.array_length
        return self.data_specs.size

    def add_scalar_data(self, key, data):
        '''
        Parameters
        ----------

        key : string
            identifier

        data : number

        '''
        self.scalar_data[key] = data

    def add_array_data(self, key, data):
        '''
        Parameters
        ----------

        key : string
            identifier

        data : ndarray

        '''

        if isinstance(data, np.ndarray):
            data = SmartArray(data)
        if self.array_length is None:
            self.array_length = data.get('host').shape[0]
        assert data.get('host').shape[0] == self.array_length
        self.array_data[key] = data

    def add_binned_data(self, key, data, flat=True):
        ''' add data to binned_data

        key : string

        data : PISA Map or (array, binning)-tuple

        flat : bool
            is the data already flattened (i.e. the binning dimesnions unrolled)
        '''
        # TODO: logic to not copy back and forth

        if isinstance(data, Map):
            flat_array = data.hist.ravel()
            self.binned_data[key] = (SmartArray(flat_array), data.binning)

        elif isinstance(data, Sequence) and len(data) == 2:
            binning, array = data
            assert isinstance(binning, MultiDimBinning)
            if isinstance(array, SmartArray):
                array = array.get('host')
            if flat:
                flat_array = array
            else:
                # first dimesnions must match
                assert array.shape[:binning.num_dims] == binning.shape
                #flat_shape = [-1] + [d for d in array.shape[binning.num_dims-1:-1]]
                flat_shape = [binning.size, -1]
                #print(flat_shape)
                flat_array = array.reshape(flat_shape)
            if not isinstance(flat_array, SmartArray):
                flat_array = SmartArray(flat_array.astype(FTYPE))
            self.binned_data[key] = (binning, flat_array)
        else:
            raise TypeError('unknown dataformat')


    def __getitem__(self, key):
        '''
        retriev data in the set data_specs
        '''
        assert self.data_specs is not None, 'Need to set data_specs to use simple getitem method'

        try:
            if self.data_specs == 'events':
                return self.get_array_data(key)
            elif isinstance(self.data_specs, MultiDimBinning):
                return self.get_binned_data(key, self.data_specs)
        except KeyError:
            try :
                return self.get_scalar_data(key)
            except KeyError:
                raise KeyError('"%s" not found in container "%s"'%(key,self.name))

    def __setitem__(self, key, value):
        '''
        set data in the set data_specs
        '''
        if not hasattr(value, '__len__'):
            self.add_scalar_data(key, value)
        else:
            assert self.data_mode is not None, 'Need to set data_specs to use simple getitem method'
            if self.data_mode == 'events':
                self.add_array_data(key, value)
            elif self.data_mode == 'binned':
                self.add_binned_data(key, (self.data_specs, value))

    def __iter__(self):
        '''
        iterate over all keys in container
        '''
        return self.keys()

    def array_to_binned(self, key, binning, averaged=True):
        '''
        histogram data array into binned data

        Parameters
        ----------

        key : str

        binning : MultiDimBinning

        averaged : bool
            if True, the histogram entries are averages of the numbers that
            end up in a given bin. This for example must be used when oscillation
            probabilities are translated.....otherwise we end up with probability*count
            per bin


        right now CPU only

        ToDo: make work for n-dim

        '''
        logging.debug('Transforming %s array to binned data'%(key))
        weights = self.array_data[key]
        sample = [self.array_data[n] for n in binning.names]

        hist = histogram(sample, weights, binning, averaged)

        self.add_binned_data(key, (binning, hist))

    def binned_to_array(self, key):
        '''
        augmented binned data to array data

        '''
        try:
            binning, hist = self.binned_data[key]
        except KeyError:
            if key in self.array_data:
                logging.debug('No transformation for `%s` array data in container `%s`'%(key,self.name))
                return
            else:
                raise ValueError('Key `%s` does not exist in container `%s`'%(key,self.name))
        logging.debug('Transforming %s binned to array data'%(key))
        sample = [self.array_data[n] for n in binning.names]
        self.add_array_data(key, lookup(sample, hist, binning))

    def binned_to_binned(self, key, new_binning):
        '''
        resample a binned key into a different binning

        Parameters
        ----------

        key : str

        new_binning : MultiDimBinning
            the new binning

        '''
        logging.debug('Resampling %s'%(key))
        old_binning, hist = self.binned_data[key]
        sample = [self.get_binned_data(name, old_binning) for name in old_binning.names]
        new_sample = [SmartArray(self.unroll_binning(name, new_binning)) for name in new_binning.names]
        hist = resample(hist, sample, old_binning, new_sample, new_binning)

        self.add_binned_data(key, (new_binning, hist))

    def scalar_to_array(self, key):
        raise NotImplementedError()

    def scalar_to_binned(self, key):
        raise NotImplementedError()

    def array_to_scalar(self, key):
        raise NotImplementedError()

    def binned_to_scalar(self, key):
        raise NotImplementedError()

    def get_scalar_data(self, key):
        return self.scalar_data[key]

    def get_array_data(self, key):
        return self.array_data[key]

    def get_binned_data(self, key, out_binning=None):
        '''
        get data array from binned data:
        if the key is a binning dimensions, then unroll te binning
        otherwise rtuen the corresponding flattened array
        '''
        if out_binning is not None:
            # check if key is binning dimension
            if key in out_binning.names:
                return self.unroll_binning(key, out_binning)
        binning, data = self.binned_data[key]
        if out_binning is not None:
            if not binning == out_binning:
                logging.warning('Automatically re-beinning data %s'%key)
                sample = [SmartArray(self.unroll_binning(name, binning)) for name in binning.names]
                new_sample = [SmartArray(self.unroll_binning(name, out_binning)) for name in out_binning.names]
                return resample(data, sample, binning, new_sample, out_binning)
        return data

    @staticmethod
    def unroll_binning(key, binning):
        grid = binning.meshgrid(entity='weighted_centers', attach_units=False)
        return SmartArray(grid[binning.index(key)].ravel())


    def get_hist(self, key):
        '''
        return reshaped data as normal n-dimensional histogram
        '''
        if self.data_mode == 'binned':
            binning = self.data_specs
            data = self.get_binned_data(key, binning)
        else:
            binning, data = self.binned_data[key]
        data = data.get('host')
        if data.ndim > 1:#binning.num_dims:
            full_shape = list(binning.shape) + [-1] #list(data.shape)[1:-1]
        else:
            full_shape = list(binning.shape)
        return data.reshape(full_shape), binning

    def get_binning(self, key):
        '''
        return binning of an entry
        '''
        return self.binned_data[key][0]

    def get_map(self, key, error=None):
        '''
        return binned data in the form of a PISA map
        '''
        hist, binning = self.get_hist(key)
        if error is not None:
            error_hist = np.abs(self.get_hist(error)[0])
        else:
            error_hist = None
        #binning = self.get_binning(key)
        assert hist.ndim == binning.num_dims
        return Map(name=self.name, hist=hist, error_hist=error_hist, binning=binning)



def test_container():
    n_evts = 10000
    x = np.arange(n_evts, dtype=FTYPE)
    y = np.arange(n_evts, dtype=FTYPE)
    w = np.ones(n_evts, dtype=FTYPE)
    w *= np.random.rand(n_evts)

    container = Container('test')
    container.add_array_data('x', x)
    container.add_array_data('y', y)
    container.add_array_data('w', w)


    binning_x = OneDimBinning(name='x', num_bins=10, is_lin=True, domain=[0, 100])
    binning_y = OneDimBinning(name='y', num_bins=10, is_lin=True, domain=[0, 100])
    binning = MultiDimBinning([binning_x, binning_y])
    #print(binning.names)
    print(container.get_binned_data('x', binning).get('host'))
    print(Container.unroll_binning('x', binning).get('host'))

    # array
    print('original array')
    print(container.get_array_data('w').get('host'))
    container.array_to_binned('w', binning)
    # binned
    print('binned')
    print(container.get_binned_data('w').get('host'))
    print(container.get_hist('w'))

    print('augmented again')
    # augment
    container.binned_to_array('w')
    print(container.get_array_data('w').get('host'))


def test_container_set():
    container1 = Container('test1')
    container2 = Container('test2')
    
    data = ContainerSet('data', [container1, container2])

    try:
        data.add_container(container1)
    except ValueError:
        pass
    else:
        raise Exception('identical containers added to a containerset, this should not be possible')

if __name__ == '__main__':
    test_container()
    test_container_set()
