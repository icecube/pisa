# -*- coding: utf-8 -*-

"""
Computation of a single Guassian (function "gaussian") or the sum of multiple Guassians (function "gaussians"). Note that each function requires an
output buffer be provided as the first argument, to which the result is added
(so the user must handle initialization of the buffer).

Use of threads requires compilation with OpenMP support.
"""


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


cimport cython
from cython.parallel cimport prange
from libc.math cimport exp, fabs, sqrt, M_PI


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian_d(double[::1] outbuf,
               double[::1] x,
               double mu,
               double sigma,
               int threads=1):
    """Computation of a single normalized Gaussian function at points `x`,
    given a mean `mu` and standard deviation `sigma`.

    The result is added and stored to the first argument, `outbuf`.

    Parameters
    ----------
    outbuf : array of double
        Output buffer, will be populated with values of the Gaussian function.

    x : array of double
        Points at which to evaluate the Gaussian

    mu : double
        Gaussian mean

    sigma : non-zero double
        Gaussian standard deviation

    threads : int
        Number of OpenMP threads (if compiled with support for OpenMP) to use
        for parallelizing the computation; defaults to one thread for "safe"
        operation in single-CPU cluster jobs

    Returns
    -------
    None

    """
    cdef double norm = 1.0/(fabs(sigma) * sqrt(2*M_PI))
    cdef double twosigma2 = 2*(sigma*sigma)
    cdef double xlessmu
    cdef Py_ssize_t i
    for i in prange(outbuf.shape[0],
                    nogil=True,
                    num_threads=threads,
                    schedule='dynamic'):
        xlessmu = x[i] - mu
        outbuf[i] = exp(-(xlessmu * xlessmu) / twosigma2) * norm


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian_s(float[::1] outbuf,
               float[::1] x,
               float mu,
               float sigma,
               int threads=1):
    """Computation of a single normalized Gaussian function at points `x`,
    given a mean `mu` and standard deviation `sigma`.

    The result is added and stored to the first argument, `outbuf`.

    Parameters
    ----------
    outbuf : array of float
        Output buffer, will be populated with values of the Gaussian function.

    x : array of float
        Points at which to evaluate the Gaussian

    mu : float
        Gaussian mean

    sigma : non-zero float
        Gaussian standard deviation

    threads : int
        Number of OpenMP threads (if compiled with support for OpenMP) to use
        for parallelizing the computation; defaults to one thread for "safe"
        operation in single-CPU cluster jobs

    Returns
    -------
    None

    """
    cdef float norm = 1.0/(fabs(sigma) * sqrt(2*M_PI))
    cdef float twosigma2 = 2*(sigma*sigma)
    cdef float xlessmu
    cdef Py_ssize_t i
    for i in prange(outbuf.shape[0],
                    nogil=True,
                    num_threads=threads,
                    schedule='dynamic'):
        xlessmu = x[i] - mu
        outbuf[i] = exp(-(xlessmu * xlessmu) / twosigma2) * norm


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussians_d(double[::1] outbuf,
                double[::1] x,
                double[::1] mu,
                double[::1] inv_sigma,
                double[::1] inv_sigma_sq,
                double[::1] weights,
                int n_gaussians,
                int threads=1):
    """Sum of multiple, normalized Gaussian function at points `x`, given
    a mean `mu` and standard deviation `sigma`.

    The result overwrites the contents of `outbuf`.

    Parameters
    ----------
    outbuf : initialized array of double
        Populated with the sum-of-Gaussians.

    x : array of double
        Points at which to evaluate the Gaussians

    mu : array of double
        Means of the Gaussians

    inv_sigma : array of non-zero double
        Inverse of standard deviations of the Gaussians

    inv_sigma_sq : array of non-zero double
        -0.5 * inverse of standard deviations, squared

    weights : array of double
        Weights of the Gaussians

    n_gaussians : int
        Number of gaussians

    threads : int
        Number of OpenMP threads (if compiled with support for OpenMP) to use
        for parallelizing the computation; defaults to one thread for "safe"
        operation in single-CPU cluster jobs

    """
    cdef double xlessmu
    cdef double tmp
    cdef Py_ssize_t i, gaus_n
    if weights[0] != -1:
        assert len(weights) == len(mu)
        # NOTE that the order of the loops is important, as
        # updating the outbuf is NOT thread safe!
        for i in prange(x.shape[0],
                        nogil=True,
                        num_threads=threads,
                        schedule='dynamic'):
            tmp = 0
            for gaus_n in range(mu.shape[0]):
                xlessmu = x[i] - mu[gaus_n]
                # NOTE: must use the syntax "tmp = tmp + ...", or else Cython
                # compiler assumes tmp is a reduction variable and compilation
                # fails
                tmp = (tmp + exp((xlessmu * xlessmu) * inv_sigma_sq[gaus_n])
                       * weights[gaus_n] * inv_sigma[gaus_n])
            outbuf[i] = tmp
    else:
        assert len(weights) == 1
        # NOTE that the order of the loops is important, as
        # updating the outbuf is NOT thread safe!
        for i in prange(x.shape[0],
                        nogil=True,
                        num_threads=threads,
                        schedule='dynamic'):
            tmp = 0
            for gaus_n in range(mu.shape[0]):
                xlessmu = x[i] - mu[gaus_n]
                # NOTE: must use the syntax "tmp = tmp + ...", or else Cython
                # compiler assumes tmp is a reduction variable and compilation
                # fails
                tmp = (tmp + exp((xlessmu * xlessmu) * inv_sigma_sq[gaus_n])
                       * inv_sigma[gaus_n])
            outbuf[i] = tmp


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussians_s(float[::1] outbuf,
                float[::1] x,
                float[::1] mu,
                float[::1] inv_sigma,
                float[::1] inv_sigma_sq,
                float[::1] weights,
                int n_gaussians,
                int threads=1):
    """Sum of multiple, normalized Gaussian function at points `x`, given
    a mean `mu` and standard deviation `sigma`.

    The result overwrites the contents of `outbuf`.

    Parameters
    ----------
    outbuf : initialized array of float
        Populated with the sum-of-Gaussians.

    x : array of float
        Points at which to evaluate the Gaussians

    mu : array of float
        Means of the Gaussians

    inv_sigma : array of non-zero float
        Inverse of standard deviations of the Gaussians

    inv_sigma_sq : array of non-zero float
        -0.5 * inverse of standard deviations, squared

    weights : array of float
        Weights of the Gaussians

    n_gaussians : int
        Number of gaussians

    threads : int
        Number of OpenMP threads (if compiled with support for OpenMP) to use
        for parallelizing the computation; defaults to one thread for "safe"
        operation in single-CPU cluster jobs

    """
    cdef float xlessmu
    cdef float tmp
    cdef Py_ssize_t i, gaus_n
    if weights[0] != -1:
        assert len(weights) == len(mu)
        # NOTE that the order of the loops is important, as
        # updating the outbuf is NOT thread safe!
        for i in prange(x.shape[0],
                        nogil=True,
                        num_threads=threads,
                        schedule='dynamic'):
            tmp = 0
            for gaus_n in range(mu.shape[0]):
                xlessmu = x[i] - mu[gaus_n]
                # NOTE: must use the syntax "tmp = tmp + ...", or else Cython
                # compiler assumes tmp is a reduction variable and compilation
                # fails
                tmp = (tmp + exp((xlessmu * xlessmu) * inv_sigma_sq[gaus_n])
                       * weights[gaus_n] * inv_sigma[gaus_n])
            outbuf[i] = tmp
    else:
        assert len(weights) == 1
        # NOTE that the order of the loops is important, as
        # updating the outbuf is NOT thread safe!
        for i in prange(x.shape[0],
                        nogil=True,
                        num_threads=threads,
                        schedule='dynamic'):
            tmp = 0
            for gaus_n in range(mu.shape[0]):
                xlessmu = x[i] - mu[gaus_n]
                # NOTE: must use the syntax "tmp = tmp + ...", or else Cython
                # compiler assumes tmp is a reduction variable and compilation
                # fails
                tmp = (tmp + exp((xlessmu * xlessmu) * inv_sigma_sq[gaus_n])
                       * inv_sigma[gaus_n])
            outbuf[i] = tmp
