'''
Class to hold generic data in container.
The data can be unbinned or binned or scalar, while 
translation methods between such different representations
are provided.

The data lives in SmartArrays on both CPU and GPU
'''
from collections import OrderedDict

import numpy as np
from numba import guvectorize, SmartArray

from pisa import FTYPE, TARGET
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.utils.numba_tools import myjit, WHERE
from pisa.utils.log import logging


class ContainerSet(object):
    '''
    Class to hold a set of container objects
    '''

    def __init__(self, name, containers=None, data_specs=None):
        self.name = name
        self.linked_containers = []
        if containers is None:
            self.containers = []
        else:
            self.containers = containers
        self.data_specs = data_specs

    def add_container(self, container):
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
        logging.info('Linking containers %s into %s'%(names, key))
        new_container = VirtualContainer(key, containers)
        self.linked_containers.append(new_container)


    def unlink_containers(self):
        '''
        Parameters
        ----------

        unlink all container
        '''
        logging.info('Unlinking all containers')
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

    def get_mapset(self, key):
        '''
        Parameters
        ----------

        key : str

        For a given key, get a PISA MapSet
        '''
        maps = []
        for container in self:
            maps.append(container.get_map(key))
        return MapSet(name=self.name, maps=maps)

class VirtualContainer(object):
    '''
    Class providing a virtual container for linked individual containers

    It should just behave like a normal container

    For reading, it just uses one container as a representative (no checkng at the mment
    if the others actually contain the same data)

    For writting, it creates one object that is added to all containers

    '''

    def __init__(self, name, containers):
        self.name = name
        # check and set link flag
        for container in containers:
            assert container.linked is False, 'Cannot link container %s since it is already linked'%container.name
            container.linked = True
        self.containers = containers

    def unlink(self):
        # reset link flag
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
            if not hasattr(value, '__len__'):
                container.scalar_data[key] = self.containers[0].scalar_data[key] 
            else:
                container.binned_data[key] = self.containers[0].binned_data[key] 

    @property
    def size(self):
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

    @ property
    def size(self):
        '''
        length of event arrays or number of bins for binned data
        '''
        assert self.data_mode is not None
        if self.data_mode == 'events':
            return self.array_length
        else:
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
        if isinstance(data, Map):
            flat_array = data.hist.ravel()
            self.binned_data[key] = (SmartArray(flat_array), data.binning)

        elif isinstance(data, tuple):
            binning, array = data
            assert isinstance(binning , MultiDimBinning)
            if isinstance(array, SmartArray):
                array = array.get('host')
            if flat:
                flat_array = array
            else:
                # first dimesnions must match
                assert array.shape[:binning.num_dims] == binning.shape
                flat_shape = [-1] + [d for d in array.shape[binning.num_dims:-1]]
                flat_array = array.reshape(flat_shape)
            if not isinstance(flat_array, SmartArray):
                flat_array = SmartArray(flat_array)
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
            return self.get_scalar_data(key)

    def __setitem__(self, key, value):
        '''
        set data in the set data_specs
        '''
        if not hasattr(value, '__len__'):
            self.add_scalar_data((key, value))
        else:
            assert self.data_mode is not None, 'Need to set data_specs to use simple getitem method'
            if self.data_mode == 'events':
                self.add_array_data(key, value)
            elif self.data_mode == 'binned':
                self.add_binned_data(key, (self.data_specs, value))

    def array_to_binned(self, key, binning, normed=True):
        '''
        histogramm data array into binned data

        right now CPU only

        ToDo: make work for n-dim

        '''
        weights = self.array_data[key].get('host')
        sample = [self.array_data[n].get('host') for n in binning.names]
        bin_edges = binning.bin_edges
        hist, edges = np.histogramdd(sample=sample,
                                     weights=weights,
                                     bins=bin_edges,
                                     )
        if normed:
            norm_hist, edges = np.histogramdd(sample=sample,
                                     bins=bin_edges,
                                     )
            with np.errstate(divide='ignore', invalid='ignore'):
                hist /= norm_hist
                hist[~np.isfinite(hist)] = 0.  # -inf inf NaN
        self.add_binned_data(key, (binning, hist))

    def binned_to_array(self, key):
        '''
        augmented binned data to array data

        ToDo: make work for n-dim
        '''
        binning, hist = self.binned_data[key]
        sample = [self.array_data[n] for n in binning.names]
        self.add_array_data(key, lookup(sample, hist, binning))

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
            assert binning == out_binning, 'no rebinning methods availabkle yet'
        return data

    @staticmethod
    def unroll_binning(key, binning):
        grid = binning.meshgrid(entity='weighted_centers', attach_units=False)
        return SmartArray(grid[binning.index(key)].ravel())


    def get_hist(self, key):
        '''
        return reshaped data as normal n-dimensional histogram
        '''
        binning, data = self.binned_data[key]
        data = data.get('host')
        full_shape = list(binning.shape) + list(data.shape)[1:-1]
        return data.reshape(full_shape)

    def get_binning(self, key):
        '''
        return binning of an entry
        '''
        return self.binned_data[key][0]

    def get_map(self, key):
        '''
        return binned data in the form of a PISA map
        '''
        hist = self.get_hist(key)
        binning = self.get_binning(key)
        assert hist.ndim == binning.num_dims
        return Map(name=self.name, hist=hist, binning=binning)


def histogram(sample, weights, binning):
    '''
    histograming
    2d method right now
    and of course this is super inefficient
    '''
    assert binning.num_dims == 2, 'can only do 2d at the moment'
    bin_edges = [edges.magnitude for edges in binning.bin_edges]
    flat_hist = np.zeros(binning.size, dtype=FTYPE)
    histogram_vectorized_2d(sample[0], sample[1], flat_hist, bin_edges[0], bin_edges[1], out=weights)
    return flat_hist

def lookup(sample, flat_hist, binning):
    '''
    the inverse of histograming
    2d method right now
    and of course this is super inefficient
    '''
    assert binning.num_dims == 2, 'can only do 2d at the moment'
    bin_edges = [edges.magnitude for edges in binning.bin_edges]
    # todo: directly return smart array
    array = SmartArray(np.empty_like(sample[0]))
    lookup_vectorized_2d(sample[0].get(WHERE), sample[1].get(WHERE), flat_hist.get(WHERE), bin_edges[0], bin_edges[1], out=array.get(WHERE))
    array.mark_changed(WHERE)
    return array

@myjit
def find_index(x, bin_edges):
    ''' simple binary search
    
    ToDo: support lin and log binnings
    
    '''
    first = 0
    last = len(bin_edges) - 2
    while (first <= last):
        i = int((first + last)/2)
        if x >= bin_edges[i]:
            if (x < bin_edges[i+1]) or (x < bin_edges[-1] and i == len(bin_edges) - 2):
                break
            else:
                first = i + 1
        else:
            last = i - 1
    return i

if FTYPE == np.float64:
    signature = '(f8, f8, f8[:], f8[:], f8[:], f8[:])'
else:
    signature = '(f4, f4, f4[:], f4[:], f4[:], f4[:])'

@guvectorize([signature], '(),(),(j),(k),(l)->()',target=TARGET)
def lookup_vectorized_2d(sample_x, sample_y, flat_hist, bin_edges_x, bin_edges_y, weights):
    idx_x = find_index(sample_x, bin_edges_x)
    idx_y = find_index(sample_y, bin_edges_y)
    idx = idx_x*(len(bin_edges_y)-1) + idx_y
    weights[0] = flat_hist[idx]


if __name__ == '__main__':


    n_evts = 10000
    x = np.arange(n_evts, dtype=FTYPE)
    y = np.arange(n_evts, dtype=FTYPE)
    w = np.ones(n_evts, dtype=FTYPE)
    w *= np.random.rand(n_evts)
    
    container = Container('test')
    container.add_array_data('x', x)
    container.add_array_data('y', y)
    container.add_array_data('w', w)


    binning_x = OneDimBinning(name='x', num_bins=10, is_lin=True, domain=[0,100])
    binning_y = OneDimBinning(name='y', num_bins=10, is_lin=True, domain=[0,100])
    binning = MultiDimBinning([binning_x, binning_y])
    #print binning.names
    print container.get_binned_data('x', binning).get('host')
    print Container.unroll_binning('x', binning).get('host')

    # array
    print 'original array'
    print container.get_array_data('w').get('host')
    container.array_to_binned('w', binning)
    # binned
    print 'binned'
    print container.get_binned_data('w').get('host')
    print container.get_hist('w')

    print 'augmented again'
    # augment
    container.binned_to_array('w')
    print container.get_array_data('w').get('host')

