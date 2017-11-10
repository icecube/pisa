'''
Class to hold generic data in container.
The data can be unbinned or binned or scalar, while 
translation methods between such different representations
are provided.

The data lives in SmartArrays on both CPU and GPU
'''
from collections import OrderedDict

import numpy as np
from numba import guvectorize, SmartArray, cuda, float32

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
        logging.debug('Linking containers %s into %s'%(names, key))
        new_container = VirtualContainer(key, containers)
        self.linked_containers.append(new_container)


    def unlink_containers(self):
        '''
        Parameters
        ----------

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

        ToDo: logic to not copy back and forth

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

    def array_to_binned(self, key, binning, averaged=True):
        '''
        histogramm data array into binned data

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

        hist =  histogram(sample, weights, binning, averaged)

        self.add_binned_data(key, (binning, hist))

    def binned_to_array(self, key):
        '''
        augmented binned data to array data

        ToDo: make work for n-dim
        '''
        logging.debug('Transforming %s binned to array data'%(key))
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


def histogram(sample, weights, binning, averaged):
    '''
    histograming
    2d method right now
    and of course this is super inefficient
    '''
    bin_edges = [edges.magnitude for edges in binning.bin_edges]
    if not TARGET == 'cuda':
        #bin_edges = binning.bin_edges

        sample = [s.get('host') for s in sample]

        hist, edges = np.histogramdd(sample=sample,
                                     weights=weights.get('host'),
                                     bins=bin_edges,
                                     )
        if averaged:
            #weights = self.array_data['event_weights'].get('host')
            hist_counts, edges = np.histogramdd(sample=sample,
                                     #weights=weights.get('host'),
                                     bins=bin_edges,
                                     )
            with np.errstate(divide='ignore', invalid='ignore'):
                hist /= hist_counts
                hist[~np.isfinite(hist)] = 0.  # -inf inf NaN
        return hist.ravel()

    else:
        # ToDo:
        # * make for d > 3
        # * do division for normed already on GPU
        # * just return SmartArray instead of copying
        if binning.num_dims in [2,3]:
            flat_hist = np.zeros(binning.size, dtype=FTYPE)
            size = len(weights)
            d_flat_hist = cuda.to_device(flat_hist)
            d_bin_edges_x = cuda.to_device(bin_edges[0])
            d_bin_edges_y = cuda.to_device(bin_edges[1])
            if binning.num_dims == 2:
                histogram_2d_kernel[(size+511)/512, 512](sample[0].get('gpu'), sample[1].get('gpu'), d_flat_hist, d_bin_edges_x, d_bin_edges_y, weights.get('gpu'))
            if binning.num_dims == 3:
                d_bin_edges_z = cuda.to_device(bin_edges[2])
                histogram_3d_kernel[(size+511)/512, 512](sample[0].get('gpu'), sample[1].get('gpu'), sample[2].get('gpu'), d_flat_hist, d_bin_edges_x, d_bin_edges_y, d_bin_edges_z, weights.get('gpu'))
            d_flat_hist.to_host()
            if averaged:
                flat_hist_counts = np.zeros(binning.size, dtype=FTYPE)
                d_flat_hist_counts = cuda.to_device(flat_hist_counts)
                if binning.num_dims == 2:
                    histogram_2d_kernel[(size+511)/512, 512](sample[0].get('gpu'), sample[1].get('gpu'), d_flat_hist_counts, d_bin_edges_x, d_bin_edges_y, None)
                if binning.num_dims == 3:
                    histogram_3d_kernel[(size+511)/512, 512](sample[0].get('gpu'), sample[1].get('gpu'), sample[2].get('gpu'), d_flat_hist, d_bin_edges_x, d_bin_edges_y, d_bin_edges_z, None)
                d_flat_hist_counts.to_host()
                with np.errstate(divide='ignore', invalid='ignore'):
                    flat_hist /= flat_hist_counts
                    flat_hist[~np.isfinite(flat_hist)] = 0.  # -inf inf NaN
            return flat_hist

        else:
            raise NotImplementedError('Other dimesnions that 2 and 3 on the GPU not supported right now')

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
    last = len(bin_edges) - 1
    while (first <= last):
        i = int((first + last)/2)
        if x >= bin_edges[i]:
            if (x < bin_edges[i+1]) or (x <= bin_edges[-1] and i == len(bin_edges) - 1):
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

# ToDo: can we do just n-dimensional?
# Furthermore: optimize using shared memory
@cuda.jit
def histogram_2d_kernel(sample_x, sample_y, flat_hist, bin_edges_x, bin_edges_y, weights):
    i = cuda.grid(1)
    if i < sample_x.size:
        #if sample_x[i] >= bin_edges_x[0] and sample_x[i] <= bin_edges_x[-1] and sample_y[i] >= bin_edges_y[0] and sample_y[i] <= bin_edges_y[-1]:
        if (sample_x[i] >= bin_edges_x[0]
                and sample_x[i] <= bin_edges_x[-1]
                and sample_y[i] >= bin_edges_y[0]
                and sample_y[i] <= bin_edges_y[-1]):
            idx_x = find_index(sample_x[i], bin_edges_x)
            idx_y = find_index(sample_y[i], bin_edges_y)
            idx = idx_x * (bin_edges_y.size - 1) + idx_y
            if weights is None:
                cuda.atomic.add(flat_hist, idx, 1.)
            else:
                cuda.atomic.add(flat_hist, idx, weights[i])
@cuda.jit
def histogram_3d_kernel(sample_x, sample_y, sample_z, flat_hist, bin_edges_x, bin_edges_y, bin_edges_z, weights):
    i = cuda.grid(1)
    if i < sample_x.size:
        if (sample_x[i] >= bin_edges_x[0]
                and sample_x[i] <= bin_edges_x[-1]
                and sample_y[i] >= bin_edges_y[0]
                and sample_y[i] <= bin_edges_y[-1]
                and sample_z[i] >= bin_edges_z[0]
                and sample_z[i] <= bin_edges_z[-1]):
            idx_x = find_index(sample_x[i], bin_edges_x)
            idx_y = find_index(sample_y[i], bin_edges_y)
            idx_z = find_index(sample_z[i], bin_edges_z)
            idx = idx_x * (bin_edges_y.size - 1) * (bin_edges_z.size - 1) + idx_y * (bin_edges_z.size - 1) + idx_z
            if weights is None:
                cuda.atomic.add(flat_hist, idx, 1.)
            else:
                cuda.atomic.add(flat_hist, idx, weights[i])

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

