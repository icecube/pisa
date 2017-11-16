'''
module for data representation translation methods


ToDo:

    - right now we distinguish on histogramming/lookup for scalars (normal) or array, which means that instead
    of just a single value per e.g. histogram bin, there can be an array of values
    This should be made more general that one function can handle everything...since now we have several
    functions doing similar things. not very pretty

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

# --------- histogramming methods ---------------

def get_hist_np(sample, weights, bins, averaged=True):
    '''helper function for numoy historams'''
    hist, edges = np.histogramdd(sample=sample,
                                 weights=weights,
                                 bins=bins,
                                 )
    if averaged:
        #weights = self.array_data['event_weights'].get('host')
        hist_counts, edges = np.histogramdd(sample=sample,
                                 #weights=weights.get('host'),
                                 bins=bins,
                                 )
        with np.errstate(divide='ignore', invalid='ignore'):
            hist /= hist_counts
            hist[~np.isfinite(hist)] = 0.  # -inf inf NaN
    return hist.ravel()

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
        weights = weights.get('host')
        if weights.ndim == 2:
            # that means it's 1-dim data instead of scalars
            hists = []
            for i in range(weights.shape[1]):
                hist = get_hist_np(sample, weights[:,i], bin_edges, averaged)
                hists.append(hist)
            return np.stack(hists, axis=1)
        else:
            return get_hist_np(sample, weights, bin_edges, averaged)

    else:
        # ToDo:
        # * make for d > 3
        # * do division for normed already on GPU
        # * just return SmartArray instead of copying
        if binning.num_dims in [2,3]:
            if len(weights.shape) > 1:
                # so we have arrays
                flat_hist = np.zeros((binning.size, weights.shape[1]), dtype=FTYPE)
                arrays = True
                print 'doing ND'
            else:
                flat_hist = np.zeros(binning.size, dtype=FTYPE)
                arrays = False
                print 'doing 1D'
            size = weights.shape[0]
            d_flat_hist = cuda.to_device(flat_hist)
            d_bin_edges_x = cuda.to_device(bin_edges[0])
            d_bin_edges_y = cuda.to_device(bin_edges[1])
            if binning.num_dims == 2:
                if arrays:
                    histogram_2d_kernel_arrays[(size+511)/512, 512](sample[0].get('gpu'), sample[1].get('gpu'), d_flat_hist, d_bin_edges_x, d_bin_edges_y, weights.get('gpu'))
                else:
                    print '2d kernel'
                    histogram_2d_kernel[(size+511)/512, 512](sample[0].get('gpu'), sample[1].get('gpu'), d_flat_hist, d_bin_edges_x, d_bin_edges_y, weights.get('gpu'))
            elif binning.num_dims == 3:
                d_bin_edges_z = cuda.to_device(bin_edges[2])
                if arrays:
                    histogram_3d_kernel_arrays[(size+511)/512, 512](sample[0].get('gpu'), sample[1].get('gpu'), sample[2].get('gpu'), d_flat_hist, d_bin_edges_x, d_bin_edges_y, d_bin_edges_z, weights.get('gpu'))
                else:
                    histogram_3d_kernel[(size+511)/512, 512](sample[0].get('gpu'), sample[1].get('gpu'), sample[2].get('gpu'), d_flat_hist, d_bin_edges_x, d_bin_edges_y, d_bin_edges_z, weights.get('gpu'))
            d_flat_hist.to_host()
            print flat_hist
            if averaged:
                flat_hist_counts = np.zeros_like(flat_hist)
                d_flat_hist_counts = cuda.to_device(flat_hist_counts)
                if binning.num_dims == 2:
                    if arrays:
                        histogram_2d_kernel_arrays[(size+511)/512, 512](sample[0].get('gpu'), sample[1].get('gpu'), d_flat_hist_counts, d_bin_edges_x, d_bin_edges_y, None)
                    else:
                        print '2d kernel'
                        histogram_2d_kernel[(size+511)/512, 512](sample[0].get('gpu'), sample[1].get('gpu'), d_flat_hist_counts, d_bin_edges_x, d_bin_edges_y, None)
                elif binning.num_dims == 3:
                    if arrays:
                        histogram_3d_kernel_arrays[(size+511)/512, 512](sample[0].get('gpu'), sample[1].get('gpu'), sample[2].get('gpu'), d_flat_hist, d_bin_edges_x, d_bin_edges_y, d_bin_edges_z, None)
                    else:
                        histogram_3d_kernel[(size+511)/512, 512](sample[0].get('gpu'), sample[1].get('gpu'), sample[2].get('gpu'), d_flat_hist, d_bin_edges_x, d_bin_edges_y, d_bin_edges_z, None)
                d_flat_hist_counts.to_host()
                with np.errstate(divide='ignore', invalid='ignore'):
                    flat_hist /= flat_hist_counts
                    flat_hist[~np.isfinite(flat_hist)] = 0.  # -inf inf NaN
            return flat_hist

        else:
            raise NotImplementedError('Other dimesnions that 2 and 3 on the GPU not supported right now')
# ToDo: can we do just n-dimensional? And scalars or arbitrary array shapes? This is so ugly :/
# Furthermore: optimize using shared memory
@cuda.jit
def histogram_2d_kernel(sample_x, sample_y, flat_hist, bin_edges_x, bin_edges_y, weights):
    i = cuda.grid(1)
    if i < sample_x.size:
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
def histogram_2d_kernel_arrays(sample_x, sample_y, flat_hist, bin_edges_x, bin_edges_y, weights):
    i = cuda.grid(1)
    if i < sample_x.size:
        if (sample_x[i] >= bin_edges_x[0]
                and sample_x[i] <= bin_edges_x[-1]
                and sample_y[i] >= bin_edges_y[0]
                and sample_y[i] <= bin_edges_y[-1]):
            idx_x = find_index(sample_x[i], bin_edges_x)
            idx_y = find_index(sample_y[i], bin_edges_y)
            idx = idx_x * (bin_edges_y.size - 1) + idx_y
            for j in range(flat_hist.shape[1]):
                if weights is None:
                    cuda.atomic.add(flat_hist, (idx,j), 1.)
                else:
                    cuda.atomic.add(flat_hist, (idx,j), weights[i,j])

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

@cuda.jit
def histogram_3d_kernel_arrays(sample_x, sample_y, sample_z, flat_hist, bin_edges_x, bin_edges_y, bin_edges_z, weights):
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
            for j in range(flat_hist.shape[1]):
                if weights is None:
                    cuda.atomic.add(flat_hist, (idx,j), 1.)
                else:
                    cuda.atomic.add(flat_hist, (idx,j), weights[i,j])

## doesn't work :(
#@cuda.jit
#def histogram_nd_kernel(sample, flat_hist, bin_edges, weights):
#    i = cuda.grid(1)
#    if i < sample[0].size:
#        #check inside:
#        inside = True
#        n_dim = len(sample)
#        for j in range(n_dim):
#            inside = inside and sample[j][i] >= bin_edges[j][0]
#            inside = inside and sample[j][i] <= bin_edges[j][-1]
#        if inside:
#            idx = 0
#            for j in range(n_dim):
#                offset = 0
#                for k in range(j, n_dim):
#                    offset *= (bin_edges[k].size - 1)
#                pos = find_index(sample[j][i], bin_edges[j])
#                idx += offset * pos
#            if weights is None:
#                cuda.atomic.add(flat_hist, idx, 1.)
#            else:
#                cuda.atomic.add(flat_hist, idx, weights[i])

# ---------- Lookup methods ---------------

def lookup(sample, flat_hist, binning):
    '''
    the inverse of histograming

    Paramters
    --------

    sample : list of SmartArrays

    flat_hist : SmartArray

    binning : PISA MultiDimBinning

    Notes
    -----
    this is only a 2d method right now
    '''
    assert binning.num_dims == 2, 'can only do 2d at the moment'
    bin_edges = [edges.magnitude for edges in binning.bin_edges]
    # todo: directly return smart array
    if flat_hist.ndim == 1:
        print 'looking up 1D'
        array = SmartArray(np.empty_like(sample[0]))
        lookup_vectorized_2d(sample[0].get(WHERE), sample[1].get(WHERE), flat_hist.get(WHERE), bin_edges[0], bin_edges[1], out=array.get(WHERE))
    elif flat_hist.ndim == 2:
        print 'looking up ND'
        array = SmartArray(np.empty((sample[0].size, flat_hist.shape[1]), dtype=FTYPE))
        lookup_vectorized_2d_arrays(sample[0].get(WHERE), sample[1].get(WHERE), flat_hist.get(WHERE), bin_edges[0], bin_edges[1], out=array.get(WHERE))
    else:
        raise NotImplementedError()
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


if FTYPE == np.float64:
    signature = '(f8, f8, f8[:,:], f8[:], f8[:], f8[:])'
else:
    signature = '(f4, f4, f4[:,:], f4[:], f4[:], f4[:])'

@guvectorize([signature], '(),(),(j,d),(k),(l)->(d)',target=TARGET)
def lookup_vectorized_2d_arrays(sample_x, sample_y, flat_hist, bin_edges_x, bin_edges_y, weights):
    '''
    Vectorized gufunc to perform the lookup
    while flat hist and weights have both a second dimension
    '''
    idx_x = find_index(sample_x, bin_edges_x)
    idx_y = find_index(sample_y, bin_edges_y)
    idx = idx_x*(len(bin_edges_y)-1) + idx_y
    for i in range(weights.size):
        weights[i] = flat_hist[idx,i]

