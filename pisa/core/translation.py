'''
module for data representation translation methods
'''

import numpy as np
from numba import guvectorize, SmartArray, cuda, float32

from pisa import FTYPE, TARGET
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.utils.numba_tools import myjit, WHERE
from pisa.utils.log import logging

__all__ = ['histogram',
           'lookup',
           ]


def histogram(sample, weights, binning, averaged):
    '''
    histograming

    Paramters
    ---------

    sample : list of SmartArrays

    weights : SmartArray

    binning : PISA MultiDimBinning

    averaged : bool
            if True, the histogram entries are averages of the numbers that
            end up in a given bin. This for example must be used when oscillation
            probabilities are translated.....otherwise we end up with probability*count
            per bin
        
    Notes
    -----

    There are a lot of ToDos here, this method is far from being optimal

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

    Paramters
    --------

    sample : list of SmartArrays

    flat_hist : SmartArrays

    binning : PISA MultiDimBinning

    Notes
    -----
    this is only a 2d method right now
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
    
    ToDo: support lin and log binnings with
    direct trnasformations instead of search
    
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
    '''
    Vectorized gufunc to perform the lookup
    '''
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

