# authors: P.Eller (pde3@psu.edu)
# date:   September 2016
"""
Histogramming on the GPU

"""


import os

import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda

from pisa import FTYPE, C_FTYPE, C_PRECISION_DEF
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource


__all__ = ['GPUHist', 'test_GPUHist']


ITYPE = np.int64
C_ITYPE = 'long'


# TODO: useful comments on what's going on with magical numbers and computations
class GPUHist(object):
    """
    Histogramming class for GPUs

    Parameters
    ---------
    bin_edges_x : array
    bin_edges_y : array
    bin_edges_z : array (optional)

    """

    KERNEL_TEMPLATE = '''//CUDA//
    #define %(c_precision_def)s
    #define fType %(c_ftype)s
    #define iType %(c_itype)s

    #include "cuda_utils.h"

    // Total number of bins (must be known at compile time)
    #define N_FLAT_BINS %(n_flat_bins)i

    #define EVENTS_PER_THREAD %(events_per_thread)i


    __device__ iType GetBin(fType x, const iType n_bins, fType *bin_edges)
    {
        // Search what bin an event belongs in, given the event values x, the
        // number of bins n_bins and the bin_edges array
        iType first = 0;
        iType last = n_bins - 1;
        iType bin;

        // Binary search to speed things up and allow for arbitrary binning
        while (first <= last) {
            bin = (first + last)/2;
            if (x >= bin_edges[bin]) {
                if ((x < bin_edges[bin+1]) || ((x <= bin_edges[n_bins])) && (bin == n_bins - 1))
                    break;
                else
                    first = bin + 1;
            }
            else {
                last = bin - 1;
            }
        }
        return bin;
    }


    __global__ void Hist2D(fType *X, fType *Y, fType *W, const iType n_events,
                           fType *hist,
                           const iType n_bins_x, const iType n_bins_y,
                           fType *bin_edges_x, fType *bin_edges_y)
    {
        __shared__ fType temp_hist[N_FLAT_BINS];

        // Zero out (reset) shared histogram buffer
        iType iterations = (N_FLAT_BINS / blockDim.x) + 1;
        iType bin;
        for (iType i = 0; i < iterations; i++) {
            bin = (i * blockDim.x) + threadIdx.x;
            if (bin < N_FLAT_BINS)
                temp_hist[bin] = 0.0;
        }
        __syncthreads();

        iType idx = EVENTS_PER_THREAD * (threadIdx.x + blockDim.x * blockIdx.x);
        for (iType i = 0; i < EVENTS_PER_THREAD; i++) {
            if (idx < n_events) {
                fType x = X[idx];
                fType y = Y[idx];
                // Check if event is even in range
                if ((x >= bin_edges_x[0]) && (x <= bin_edges_x[n_bins_x]) && (y >= bin_edges_y[0]) && (y <= bin_edges_y[n_bins_y])) {
                    iType bin_x = GetBin(x, n_bins_x, bin_edges_x);
                    iType bin_y = GetBin(y, n_bins_y, bin_edges_y);
                    atomicAdd_custom(&temp_hist[bin_y + bin_x * n_bins_y],
                                     W[idx]);
                }
            }
            idx++;
        }
        __syncthreads();

        // Write shared buffer into global memory
        for (iType i = 0; i < iterations; i++) {
            bin = (i * blockDim.x) + threadIdx.x;
            if (bin < N_FLAT_BINS)
                atomicAdd_custom(&(hist[bin]), temp_hist[bin]);
        }
    }


    __global__ void Hist3D(fType *X, fType *Y, fType *Z, fType *W,
                           const iType n_events,
                           fType *hist,
                           const iType n_bins_x, const iType n_bins_y, const iType n_bins_z,
                           fType *bin_edges_x, fType *bin_edges_y, fType *bin_edges_z)
    {
        __shared__ fType temp_hist[N_FLAT_BINS];

        // Zero out (reset) shared histogram buffer
        iType iterations = (N_FLAT_BINS / blockDim.x) + 1;
        iType bin;
        for (iType i = 0; i < iterations; i++) {
            bin = (i * blockDim.x) + threadIdx.x;
            if (bin < N_FLAT_BINS)
                temp_hist[bin] = 0.0;
        }
        __syncthreads();

        iType idx = EVENTS_PER_THREAD * (threadIdx.x + blockDim.x * blockIdx.x);
        for (iType i = 0; i < EVENTS_PER_THREAD; i++) {
            if (idx < n_events) {
                fType x = X[idx];
                fType y = Y[idx];
                fType z = Z[idx];

                // Check if event is even in range
                if ((x >= bin_edges_x[0]) && (x <= bin_edges_x[n_bins_x])
                      && (y >= bin_edges_y[0]) && (y <= bin_edges_y[n_bins_y])
                      && (z >= bin_edges_z[0]) && (z <= bin_edges_z[n_bins_z])) {
                    iType bin_x = GetBin(x, n_bins_x, bin_edges_x);
                    iType bin_y = GetBin(y, n_bins_y, bin_edges_y);
                    iType bin_z = GetBin(z, n_bins_z, bin_edges_z);
                    atomicAdd_custom(&temp_hist[bin_z + (bin_y * n_bins_z) + (bin_x * n_bins_y * n_bins_z)], W[idx]);
                }
            }
            idx++;
        }
        __syncthreads();

        // Write shared buffer into global memory
        for (iType i = 0; i < iterations; i++) {
            bin = (i * blockDim.x) + threadIdx.x;
            if (bin < N_FLAT_BINS)
                atomicAdd_custom(&(hist[bin]), temp_hist[bin]);
        }
    }
    '''

    def __init__(self, bin_edges_x, bin_edges_y, bin_edges_z=None):
        if bin_edges_z is None:
            self.h3d = False
        else:
            self.h3d = True

        self.bdim = (256, 1, 1)

        # Events to be histogrammed per thread
        self.events_per_thread = 20
        self.n_bins_x = ITYPE(len(bin_edges_x)-1)
        self.n_bins_y = ITYPE(len(bin_edges_y)-1)
        self.n_flat_bins = self.n_bins_x * self.n_bins_y
        if self.h3d:
            self.n_bins_z = ITYPE(len(bin_edges_z)-1)
            self.hist = np.ravel(np.zeros((self.n_bins_x, self.n_bins_y,
                                           self.n_bins_z), dtype=FTYPE))
        else:
            self.n_bins_z = ITYPE(1)
            self.hist = np.ravel(np.zeros((self.n_bins_x, self.n_bins_y),
                                          dtype=FTYPE))
        self.n_flat_bins *= self.n_bins_z

        # Allocate
        self.d_hist = cuda.mem_alloc(self.hist.nbytes)
        self.d_bin_edges_x = cuda.mem_alloc(bin_edges_x.nbytes)
        self.d_bin_edges_y = cuda.mem_alloc(bin_edges_y.nbytes)
        if self.h3d:
            self.d_bin_edges_z = cuda.mem_alloc(bin_edges_z.nbytes)

        # Copy
        cuda.memcpy_htod(self.d_bin_edges_x, bin_edges_x)
        cuda.memcpy_htod(self.d_bin_edges_y, bin_edges_y)
        if self.h3d:
            cuda.memcpy_htod(self.d_bin_edges_z, bin_edges_z)

        kernel_code = self.KERNEL_TEMPLATE %dict(
            c_precision_def=C_PRECISION_DEF,
            c_ftype=C_FTYPE,
            c_itype=C_ITYPE,
            n_flat_bins=self.n_flat_bins,
            events_per_thread=self.events_per_thread
        )

        include_dirs = [
            os.path.abspath(find_resource('../utils'))
        ]

        module = SourceModule(kernel_code, include_dirs=include_dirs,
                              keep=True)
        self.hist2d_fun = module.get_function("Hist2D")
        self.hist3d_fun = module.get_function("Hist3D")

    def clear(self):
        """Clear the histogram bins on the GPU"""
        # Very dumb way to reset to zero...
        self.hist = np.zeros(self.n_flat_bins, dtype=FTYPE)
        cuda.memcpy_htod(self.d_hist, self.hist)

    def update_bin_edges(self, bin_edges_x, bin_edges_y, bin_edges_z=None):
        """Update the bin edge locations to use. Note that the number of bins
        cannot change.

        Parameters
        ----------
        bin_edges_x, bin_edges_y : arrays
            X and Y bin edges

        bin_edges_z : array or None
            Optionally specify Z bin edges

        """
        # Ensure compatibility
        assert self.h3d == bool(bin_edges_z is not None)
        assert self.n_bins_x == ITYPE(len(bin_edges_x)-1)
        assert self.n_bins_y == ITYPE(len(bin_edges_y)-1)
        if self.h3d:
            assert self.n_bins_z == ITYPE(len(bin_edges_z)-1)
        # Copy arrays to GPU
        cuda.memcpy_htod(self.d_bin_edges_x, bin_edges_x)
        cuda.memcpy_htod(self.d_bin_edges_y, bin_edges_y)
        if self.h3d:
            cuda.memcpy_htod(self.d_bin_edges_z, bin_edges_z)

    def get_hist(self, n_events, d_x, d_y, d_w, d_z=None):
        """Retrieve histogram, given device arrays for x&y values as well as
        weights w"""
        # TODO: useful comments on what's going on with magical numbers and
        # computations

        # Block and grid dimensions
        dx, mx = divmod(n_events/self.events_per_thread+1, self.bdim[0])
        gdim = ((dx + (mx > 0)) * self.bdim[0], 1)
        self.clear()

        # Calculate hist
        if self.h3d:
            self.hist3d_fun(
                d_x, d_y, d_z, d_w,
                n_events,
                self.d_hist,
                self.n_bins_x, self.n_bins_y, self.n_bins_z,
                self.d_bin_edges_x, self.d_bin_edges_y, self.d_bin_edges_z,
                block=self.bdim, grid=gdim
            )
        else:
            self.hist2d_fun(
                d_x, d_y, d_w,
                n_events,
                self.d_hist,
                self.n_bins_x, self.n_bins_y,
                self.d_bin_edges_x, self.d_bin_edges_y,
                block=self.bdim, grid=gdim
            )

        # Copy back
        cuda.memcpy_dtoh(self.hist, self.d_hist)
        if self.h3d:
            hist = self.hist.reshape(self.n_bins_x, self.n_bins_y,
                                     self.n_bins_z)
        else:
            hist = self.hist.reshape(self.n_bins_x, self.n_bins_y)

        return hist


def test_GPUHist():
    """Unit tests (and timings) for GPUHist class"""
    from time import time
    from itertools import product
    import pycuda.autoinit

    ftypes = [FTYPE]
    nexps = [0, 1, 2, 3, 4, 5, 6, 7]
    nbinses = [1, 10, 50]
    ft = [False, True]
    for ftype, weight, nexp, n_bins in product(ftypes, ft, nexps, nbinses):
        n_events = ITYPE(10**nexp)
        logging.debug('ftype=%s, weight=%-5s, bins=%2sx2, n_events=10^%s'
                      %(ftype.__name__, weight, n_bins, nexp))

        if ftype == np.float32:
            rtol = 1e-5
        elif ftype == np.float64:
            rtol = 1e-12

        # Draw random samples from the Pareto distribution for energy values
        rs = np.random.RandomState(seed=0)
        a, m = 1., 1
        e = ((rs.pareto(a, n_events) + 1) * m).astype(ftype)
        # Ensure endpoints are in data
        e[0] = 1
        e[-1] = 80

        # Draw random samples from a uniform distribution for coszen values
        rs = np.random.RandomState(seed=1)
        cz = rs.uniform(low=-1, high=+1, size=n_events).astype(ftype)
        # Ensure endpoints are in data
        cz[0] = -1
        cz[-1] = +1

        # Draw random samples from a uniform distribution for pid values
        rs = np.random.RandomState(seed=2)
        pid = rs.uniform(low=-1, high=+2, size=n_events).astype(ftype)
        # Ensure endpoints are in data
        pid[0] = -1
        pid[-1] = +2

        if weight:
            # Draw random samples from a uniform distribution for weights
            rs = np.random.RandomState(seed=3)
            w = rs.uniform(low=0, high=1000, size=n_events).astype(ftype)
            # Ensure a weight of 0 is represented
            w[0] = 0
        else:
            w = np.ones_like(e, dtype=ftype)

        t0 = time()
        d_e = cuda.mem_alloc(e.nbytes)
        d_cz = cuda.mem_alloc(cz.nbytes)
        d_pid = cuda.mem_alloc(pid.nbytes)
        d_w = cuda.mem_alloc(w.nbytes)
        cuda.memcpy_htod(d_e, e)
        cuda.memcpy_htod(d_cz, cz)
        cuda.memcpy_htod(d_pid, pid)
        cuda.memcpy_htod(d_w, w)
        logging.debug('time to copy data: %s' % (time()-t0))

        bin_edges_e = np.logspace(0, 2, n_bins+1, dtype=ftype)
        bin_edges_cz = np.linspace(-1, 1, n_bins+1, dtype=ftype)
        bin_edges_pid = np.array([-1, 0, 2], dtype=ftype)

        histogrammer = GPUHist(
            bin_edges_x=bin_edges_e,
            bin_edges_y=bin_edges_cz,
        )
        for _ in range(3):
            t0 = time()
            hist2d = histogrammer.get_hist(
                n_events=n_events, d_x=d_e, d_y=d_cz, d_w=d_w
            )
            logging.debug('gpu 2d hist took %s' % (time()-t0))

        t0 = time()
        np_hist2d, _, _ = np.histogram2d(
            e, cz,
            bins=(bin_edges_e, bin_edges_cz),
            weights=w
        )
        logging.debug('np  2d hist took %s' % (time()-t0))

        with np.errstate(divide='ignore', invalid='ignore'):
            fract_err = (hist2d/np_hist2d) - 1
            logging.debug(
                '2D hist ftype=%s, weighted=%s, n_events=%s: max abs fract'
                ' err=%s, mean fract err=%s, mean abs fract err=%s'
                %(ftype, weight, n_events,
                  np.nanmax(np.abs(fract_err)),
                  np.nanmean(fract_err),
                  np.nanmean(np.abs(fract_err)))
            )
            if not np.allclose(hist2d, np_hist2d, atol=0, rtol=rtol):
                logging.error('Numpy hist:\n%s' % repr(np_hist2d))
                logging.error('GPUHist hist:\n%s' % repr(hist2d))
                raise ValueError(
                    '2D histogram ftype=%s, weighted=%s, n_events=%s worst'
                    ' fractional error is %s'
                    %(ftype, weight, n_events,
                      np.max(np.abs((hist2d-np_hist2d)/np_hist2d)))
                )

        del histogrammer

        logging.debug('ftype=%s, weight=%-5s, bins=%2sx%2sx2, n_events=10^%s'
                      %(ftype.__name__, weight, n_bins, n_bins, nexp))

        histogrammer = GPUHist(
            bin_edges_x=bin_edges_e,
            bin_edges_y=bin_edges_cz,
            bin_edges_z=bin_edges_pid,
        )
        for _ in range(3):
            t0 = time()
            hist3d = histogrammer.get_hist(
                n_events=n_events, d_x=d_e, d_y=d_cz, d_w=d_w, d_z=d_pid
            )
            logging.debug('gpu 3d hist took %s' % (time()-t0))

        t0 = time()
        np_hist3d, _ = np.histogramdd(
            sample=[e, cz, pid],
            bins=(bin_edges_e, bin_edges_cz, bin_edges_pid),
            weights=w
        )
        logging.debug('np  3d hist took %s' % (time()-t0))

        with np.errstate(divide='ignore', invalid='ignore'):
            fract_err = (hist3d/np_hist3d) - 1
            logging.debug(
                '3D hist ftype=%s, weighted=%s, n_events=%s: max abs fract'
                ' err=%s, mean fract err=%s, mean abs fract err=%s'
                %(ftype, weight, n_events,
                  np.nanmax(np.abs(fract_err)),
                  np.nanmean(fract_err),
                  np.nanmean(np.abs(fract_err)))
            )
            if not np.allclose(hist3d, np_hist3d, atol=0, rtol=rtol):
                logging.error('Numpy hist:\n%s' % repr(np_hist3d))
                logging.error('GPUHist hist:\n%s' % repr(hist3d))
                raise ValueError(
                    '3D histogram ftype=%s, weighted=%s, n_events=%s worst'
                    ' fractional error is %s'
                    %(ftype, weight, n_events,
                      np.max(np.abs((hist3d-np_hist3d)/np_hist3d)))
                )

        del histogrammer

    logging.info('<< PASS : test_GPUHist >>')


if __name__ == '__main__':
    set_verbosity(3)
    test_GPUHist()
