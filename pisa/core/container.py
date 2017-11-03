from collections import OrderedDict

import numpy as np
from numba import guvectorize, SmartArray

from pisa import FTYPE, TARGET
from pisa.map import Map


class ContainerSet(object):
    '''
    Class to hold a set of container objects
    '''

    def __init__(self, name, containers=None):
        self.name = name
        if containers is None:
            self.containers = []
        else:
            self.containers = containers

    def add_container(self, container):
        self.containers.append(container)
    

    def get_containers(self, name):
        if name == 'nu':
            names_we_want = ['nu', 'nue', 'numu', 'nutau']
        elif name == 'nubar':
            names_we_want = ['nubar', 'nue_bar', 'nmu_bar', 'nutau_bar']
        elif name == 'e':
            names_we_want = ['nue', 'nue_bar']
        elif name == 'mu':
            names_we_want = ['numu', 'numu_bar']
        elif name == 'tau':
            names_we_want = ['nutau', 'nutau_bar']
        else:
            names_we_want = [name]
        return [c for c in self.containers if c.name in names_we_want]


class Container(object):
    '''
    Class to hold data in the form of event arrays and/or maps

    if maps are needed, a binning must be set

    contained maps must have the same binning (is this a good idea?)

    Parameters
    ----------
    name : string
        identifier

    binning : PISA MultiDimBinning
        binning, if binned data is used

    code : int
        could hold for example a PDG code 

    '''

    def __init__(self, name, code=None):
        self.name = name
        self.code = code
        self.scalar_data = OrderedDict()
        self.array_data = OrderedDict()
        self.binned_data = OrderedDict()

    def add_scalar_data(self, key, data):
        self.scalar_data[key] = data

    def add_array_data(self, key, data):
        '''
        Parameters
        ----------

        key : string
            identifier

        data : ndarray

        '''
        elif isinstance(data, np.ndarray):
            self.array_data[key] = SmartArray(data)
        elif isinstance(data, SmartArray):
            self.array_data[key] = data
        else:
            raise TypeError('type %s of data not supported'%type(data))

    def add_binned_data(self, key, data):
        ''' add data to binned_data

        key : string

        data : PISA Map or (array, binning)-tuple

        '''
        if isinstance(data, Map):
            flat_array = data.hist.ravel()
            self.binned_data[key] = (SmartArray(flat_array), data.binning)

        elif isinstance(data, tuple):
            binning, array = data
            assert isinstance(binning , MultiDimBinning)
            if isinstance(array, SmartArray):
                array = array.get('host')
            if not isinstance(array, np.ndarray):
                raise TypeError('given array is not an np.ndarray or numba Smart array')
            # first dimesnions must match
            assert array.shape[:binning.num_dims] == binning.shape
            flat_shape = [-1] + [d for d in array.shape[binning.num_dims:-1]]
            flat_array = array.reshape(flat_shape)
            flat_array = SmartArray(flat_array)
            self.binned_data[key] = (binning, flat_array)


    def array_to_binned(self, key, binning):
        '''
        histogramm data array into binned data
        '''
        weights = self.array_data[key]
        sample = [self.array_data[n] for n in binning.bin_names]


        # ToDo: add_flat_binned_data
        self.add_binned_data(key, (binning, histogramm(sample, weights, binning)))

    def binned_to_array(self, key):
        '''
        augmented binned data to array data
        '''
        binning, hist = self.binned_data[key]
        sample = [self.array_data[n] for n in binning.bin_names]
        self.add_array_data(key, lookup(sample, hist, binning))

    def scalar_to_array(self, key):
        raise NotImplementedError()

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
            if key in out_binning.bin_names:
                # unroll
                grid = out_binning.meshgrid(entity='weighted_centers', attach_units=False)
                return SmartArray(grid[binning.index(key)])
        binning, data = self.binned_data[key]
        if out_binning is not None:
            assert binning = out_binning, 'no rebinning methods availabkle yet'
        return data


    def get_hist(self, key):
        '''
        return reshaped data as normal hist
        '''
        binning, data = self.binned_data[key]
        data = data.get('host')
        full_shape = list(binning.shape) + list(data.shape)[1:-1]
        return data.reshape(full_shape)

    def get_binning(self, key):
        '''
        return binning
        '''
        return self.binned_data[key][0]

    def get_map(self, key):
        '''
        return binned data in the form of a PISA map
        '''
        hist = self.get_hist[key].get('host')
        binning = self.get_binning[key]
        assert hist.ndim == binning.num_dims
        return Map(name=key, hist=hist, binning=binning)


def histogram(sample, weights, binning)
    '''
    histograming
    2d method right now
    and of course this is super inefficient
    '''
    assert binning.num_dims == 2, 'can only do 2d at the moment'
    bin_edges = [edges.magnitude for edges in binning.bin_edges]
    hist = np.empty(binning.shape, dtype=FTYPE)
    histogram_vectorized_2d(sample[0], sample[1], weights, bin_edges[0], bin_edges[1], out=hist)
    return hist

def lookup(sample, flat_hist, binning):
    '''
    the inverse of histograming
    2d method right now
    and of course this is super inefficient
    '''
    assert binning.num_dims == 2, 'can only do 2d at the moment'
    bin_edges = [edges.magnitude for edges in binning.bin_edges]
    # todo: directly return smart array
    array = np.empty_like(sample[0])
    lookup_vectorized_2d(sample[0], sample[1], flat_hist, bin_edges[0], bin_edges[1], out=array)
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
    signature = '(f8, f8, f8[:,:], f8[:], f8[:], f8[:])'
else:
    signature = '(f4, f4, f4[:,:], f4[:], f4[:], f4[:])'

@guvectorize([signature], '(),(),(j),(k),(l)->()',target=TARGET)
def lookup_vectorized_2d(sample_x, sample_y, flat_hist, bin_edges_x, bin_edges_y, out):
    idx_x = find_index(sample_x, bin_edges_x)
    idx_y = find_index(sample_y, bin_edges_y)
    idx = idx_x*(len(bin_edges_y)-1) + idx_y
    out[0] = flat_hist[idx]

if FTYPE == np.float64:
    signature = '(f8, f8, f8, f8[:], f8[:], f8[:,:])'
else:
    signature = '(f4, f4, f4, f4[:], f4[:], f4[:,:])'

@guvectorize([signature], '(),(),(),(k),(l)->(i,j)',target=TARGET)
def histogram_vectorized_2d(sample_x, sample_y, weights, bin_edges_x, bin_edges_y, out):
    idx_x = find_index(sample_x, bin_edges_x)
    idx_y = find_index(sample_y, bin_edges_y)
    out[idx_x,idx_y] = weights[0]



