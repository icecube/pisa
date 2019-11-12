"""
Implementation of the Improved Sheather Jones (ISJ) KDE bandwidth selection
method outlined in:

  Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
    estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

and extension to use this in varaible bandwidth KDE via

  I.S. Abramson, On bandwidth variation in kernel estimates - A
    square root law, Annals of Stat. Vol. 10, No. 4, 1217-1223 1982

See also

  P. Hall, T. C. Hu, J. S. Marron, Improved Variable Window Kernel
    Estimates of Probability Densities, Annals of Statistics
    Vol. 23, No. 1, 1-10, 1995

Some code in this module is Copyright (c) 2007 Zdravko Botev. See __license__
for details.
"""


# TODO: Make normalization use a relative density value normalized after ISJ
# bandwidth to take into consideration really-high densities (which can have
# narrower bandwidths than ISJ dictates) or really-low densities (which should
# have very wide bandwidths, wider than ISJ dictates)

# TODO : accuracy tests for fbwkde and vbwkde


from __future__ import absolute_import, division

from math import exp, sqrt
from time import time

import numpy as np
from scipy import fftpack, interpolate, optimize

from pisa import FTYPE, numba_jit
from pisa.utils.gaussians import gaussians
from pisa.utils.log import logging, set_verbosity, tprofile


__all__ = ['OPT_TYPE', 'FIXED_POINT_IMPL',
           'fbwkde', 'vbwkde', 'isj_bandwidth', 'test_fbwkde', 'test_vbwkde']

__author__ = 'Z. Botev, J.L. Lanfranchi'

__license__ = '''Original BSD license, applicable *ONLY* to functions
"isj_bandwidth" and "fixed_point" since these were derived from Botev's
original work (this license applies to any future code derived from those
functions as well):

  Copyright (c) 2007, Zdravko Botev
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
  OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

============================================================================

All other code in this module is under the following copyright/license:

Copyright (c) 2014-2017, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


# NOTE: 'minimum' is giving very rough overfit results. Switching to 'root' for
# now.
OPT_TYPE = 'root'
FIXED_POINT_IMPL = 'numba_loops'

_PI = np.pi
_TWOPI = 2*np.pi
_SQRTPI = np.sqrt(np.pi)
_SQRT2PI = np.sqrt(2*np.pi)
_PISQ = np.pi*np.pi


def fbwkde(data, weights=None, n_dct=None, min=None, max=None,
           evaluate_dens=True, evaluate_at=None):
    """Fixed-bandwidth (standard) Gaussian KDE using the Improved
    Sheather-Jones bandwidth.

    Code adapted for Python from the implementation in Matlab by Zdravko Botev.

    Ref: Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
    estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

    Parameters
    ----------
    data : array
    weights : array or None
    n_dct : None or int
        Number of points with which to form a regular grid, from `min` to
        `max`; histogram values at these points are sent through a discrete
        Cosine Transform (DCT), so `n_dct` should be an integer power of 2 for
        speed purposes. If None, uses next-highest-power-of-2 above
        len(data)*10.
    min : float or None
    max : float or None
    evaluate_dens : bool
    evaluate_at : None or array

    Returns
    -------
    bandwidth : float
    evaluate_at : array of float
    density : None or array of float

    """
    if n_dct is None:
        n_dct = int(2**np.ceil(np.log2(len(data)*10)))
    assert int(n_dct) == n_dct
    n_dct = int(n_dct)
    n_datapoints = len(data)

    # Parameters to set up the points on which to evaluate the density
    if min is None or max is None:
        minimum = data.min()
        maximum = data.max()
        data_range = maximum - minimum
        min = minimum - data_range/2 if min is None else min
        max = maximum + data_range/2 if max is None else max

    hist_range = max - min

    # Histogram the data to get a crude first approximation of the density
    data_hist, bins = np.histogram(
        data, bins=n_dct, range=(min, max), density=False, weights=weights
    )

    # Make into a probability mass function
    if weights is None:
        data_hist = data_hist / n_datapoints
    else:
        data_hist = data_hist / np.sum(weights)

    # Define a minimum bandwidth relative to mean of distances between points
    distances = np.diff(np.sort(data))
    min_bandwidth = 2*np.pi*np.mean(distances)
    logging.trace('min_bandwidth, 2pi*mean: %.5e', min_bandwidth)

    # Solve for the ISJ fixed point to obtain the "optimal" bandwidth
    isj_bw, t_star, dct_data = isj_bandwidth(
        y=data_hist, n_datapoints=n_datapoints, x_range=hist_range,
        min_bandwidth=min_bandwidth
    )

    # TODO: Detect numerical instability issues here, prior to returning!

    if not evaluate_dens:
        return isj_bw, evaluate_at, None

    if evaluate_at is None:
        # Smooth the discrete-cosine-transformed data using t_star
        sm_dct_data = dct_data*np.exp(-np.arange(n_dct)**2 * _PISQ*t_star/2)

        # Inverse DCT to get density
        density = fftpack.idct(sm_dct_data, norm=None)*n_dct/hist_range
        if np.any(density < 0):
            logging.trace(
                'ISJ encountered numerical instability in IDCT (resulting in a'
                ' negative density). Result will be computed without the IDCT.'
            )
            evaluate_at = (bins[0:-1] + bins[1:]) / 2
        else:
            evaluate_at = (bins[0:-1] + bins[1:]) / 2
            density = density/np.trapz(density, evaluate_at)
            return isj_bw, evaluate_at, density
    else:
        evaluate_at = np.asarray(evaluate_at, dtype=FTYPE)

    density = gaussians(
        x=evaluate_at,
        mu=data.astype(FTYPE),
        sigma=np.full(shape=n_datapoints, fill_value=isj_bw, dtype=FTYPE),
        weights=weights
    )

    return isj_bw, evaluate_at, density


def vbwkde(data, weights=None, n_dct=None, min=None, max=None, n_addl_iter=0,
           evaluate_dens=True, evaluate_at=None):
    """Variable-bandwidth (standard) Gaussian KDE that uses the function
    `fbwkde` for a pilot density estimate.

    Parameters
    ----------
    data : array
        The data points for which the density estimate is sought

    weights : array or None
        Weight for each point in `data`

    n_dct : None or int
        Number of points with which to form a regular grid, from `min` to
        `max`; histogram values at these points are sent through a discrete
        Cosine Transform (DCT), so `n_dct` should be an integer power of 2 for
        speed purposes. If None, uses next-highest-power-of-2 above
        len(data)*10.

    min : None or float
        Minimum of range over which to compute density.
        If None, defaults to min(data) - range(data)/2

    max : None or float
        Maximum of range over which to compute density.
        If None: max(data) + range(data)/2

    n_addl_iter : int >= 0
        Number of additional iterations on the Abramson VBW density *after*
        the initial VBW estimate. This can help smooth the tails of the
        distribution, at the expense of additional computational cost.

    evaluate_dens : bool
        Whether to evaluate and return the density estimate on the points
        defined by `evaluate_at`

    evaluate_at : None, float, or array of float
        Point(s) at which to evaluate the density estimate. If None,
          evaluate_at = np.linspace(min + dx/2, max - dx/2, n_dct)
        where
          dx = (max - min) / n_dct

    Returns
    -------
    kernel_bandwidths : array

    evaluate_at : array
        Locations at which the density estimates would be evaluated

    density : array
        Density estimates

    Notes
    -----
    Specifying the range:

        The specification of min and max are critical for obtaining a
        reasonable density estimate. If the true underlying density slowly
        decays to zero on one side or the other, like a gaussian, specifying
        too-small a range will distort the edge the VBW-KDE finds. On the
        other hand, an abrupt cut-off in the distribution should be
        accompanied by a similar cutoff in the computational range (min and/or
        max). The algorithm here will approximate such a sharp cut-off with
        roughly the same performance to the reflection method for standard
        KDE's (as the fixed-BW portion uses a DCT of the data), but note that
        this will not perform as well as polynomial-edges or other
        modifications that have been proposed in the literature.

    """
    if n_dct is None:
        n_dct = int(2**np.ceil(np.log2(len(data)*10)))
    assert n_addl_iter >= 0 and int(n_addl_iter) == n_addl_iter
    n_addl_iter = int(n_addl_iter)

    # Pilot density estimate for the VBW KDE comes from fixed bandwidth KDE
    # using the Improved Sheather-Jones algorithm. By specifying
    # `evaluate_at` to be None, `fbwkde` derives a regular grid at which to
    # evaluate the points and does so without needing to do a sum of Gaussians
    # (only a freq.-domain multiply and inverse DCT)
    isj_bw, grid, pilot_dens_on_grid = fbwkde(
        data=data, weights=weights, n_dct=n_dct, min=min, max=max,
        evaluate_dens=True, evaluate_at=None
    )
    if np.any(pilot_dens_on_grid < 0):
        raise ValueError('ISJ')

    # Include edges (min, max) in the grid and extend the densities to the
    # left/right, respectively. (Make the profile constant out to the edge).
    all_gridpoints = []
    all_dens = []
    if grid[0] != min:
        all_gridpoints.append([min])
        all_dens.append([pilot_dens_on_grid[0]])

    all_gridpoints.append(grid)
    all_dens.append(pilot_dens_on_grid)

    if grid[-1] != max:
        all_gridpoints.append([max])
        all_dens.append([pilot_dens_on_grid[-1]])

    grid = np.concatenate(all_gridpoints)
    pilot_dens_on_grid = np.concatenate(all_dens)

    # Use linear interpolation to get density at the data points
    interp = interpolate.interp1d(
        x=grid,
        y=pilot_dens_on_grid,
        kind='linear',
        copy=False,
        bounds_error=True,
        fill_value=np.nan
    )
    # TODO: For some reason this gets translated into an object array, so the
    # `astype` call is necessary
    pilot_dens_at_datapoints = interp(data).astype(FTYPE)

    n_iter = 1 + n_addl_iter
    for n in range(n_iter):
        # Note below diverges from the published Abramson method, by forcing
        # the bandwidth at the max of the density distribution to be exactly
        # the bandwidth found above with the improved Sheather-Jones BW
        # selection technique. Refs:
        #   I.S. Abramson, On bandwidth variation in kernel estimates - A
        #       square root law, Annals of Stat. Vol. 10, No. 4, 1217-1223 1982
        #   P. Hall, T. C. Hu, J. S. Marron, Improved Variable Window Kernel
        #       Estimates of Probability Densities, Annals of Statistics
        #       Vol. 23, No. 1, 1-10, 1995
        kernel_bandwidths = (
            isj_bw * np.sqrt(np.max(pilot_dens_at_datapoints))
            / np.sqrt(pilot_dens_at_datapoints)
        )

        if n < n_addl_iter:
            pilot_dens_at_datapoints = gaussians(
                x=data,
                mu=data,
                sigma=kernel_bandwidths,
                weights=weights
            )

        else: # final iteration
            if evaluate_at is None:
                evaluate_at = grid
            if evaluate_dens:
                density = gaussians(
                    x=evaluate_at,
                    mu=data,
                    sigma=kernel_bandwidths,
                    weights=weights
                )
            else:
                density = None

    return kernel_bandwidths, evaluate_at, density


def isj_bandwidth(y, n_datapoints, x_range, min_bandwidth):
    """
    Parameters
    ----------
    y : array of float
    n_datapoints : int
    x_range : float

    Returns
    -------
    bandwidth : float
    t_star : float
    dct_data : array of float

    """
    # Ensure double-precision datatypes are used
    y = np.asarray(y, dtype=np.float64)
    x_range = np.float64(x_range)

    n_dct = len(y)

    min_t_star = (min_bandwidth/x_range)**2

    i_range = np.arange(1, n_dct, dtype=np.float64)**2
    log_i_range = np.log(i_range)

    dct_data = fftpack.dct(y, norm=None)
    dct_data_sq = 0.25 * (dct_data * dct_data)[1:]

    logging.trace('Fixed point implementation = %s', FIXED_POINT_IMPL)
    logging.trace('Optimization type = %s', OPT_TYPE)
    if FIXED_POINT_IMPL == 'numba_np':
        func = fixed_point_numba_np
        args = n_datapoints, i_range, log_i_range, dct_data_sq
    elif FIXED_POINT_IMPL == 'numba_loops':
        func = fixed_point_numba_loops
        args = n_datapoints, i_range, log_i_range, dct_data_sq
    elif FIXED_POINT_IMPL == 'numba_orig':
        func = fixed_point_numba_orig
        args = n_datapoints, i_range, dct_data_sq
    else:
        raise ValueError('Unknown FIXED_POINT_IMPL "%s"' % FIXED_POINT_IMPL)

    try:
        if OPT_TYPE == 'root':
            t_star = optimize.brentq(
                f=func,
                a=min_t_star/1000,
                b=0.5,
                rtol=np.finfo(np.float64).eps*1e2,
                args=args,
                full_output=True,
                disp=True
            )[0]
        elif OPT_TYPE == 'minimum':
            t_star = optimize.minimize_scalar(
                fun=func,
                bounds=(min_t_star/1000, 0.5),
                method='Bounded',
                args=args,
                options=dict(maxiter=1e6, xatol=1e-22),
            ).x
        else:
            raise ValueError('Unknown OPT_TYPE "%s"' % OPT_TYPE)

        if t_star < min_t_star:
            logging.trace('t_star = %.4e < min_t_star = %.4e;'
                          ' setting to min_t_star', t_star, min_t_star)
            t_star = min_t_star

        # NOTE: Use `math.sqrt` _not_ `numpy.sqrt` to ensure failures raise
        # exceptions (numpy might be coaxed to do so, but it'd probably be
        # slow)
        bandwidth = sqrt(t_star)*x_range

    except ValueError:
        logging.error('Improved Sheather-Jones bandwidth %s-finding failed.',
                      OPT_TYPE)
        if min_bandwidth is not None:
            logging.error('Using supplied `min_bandwidth` instead.')
            bandwidth = min_bandwidth
            t_star = min_t_star
        else:
            raise

    return bandwidth, t_star, dct_data


if OPT_TYPE == 'root':
    def optfunc(f):
        """No-op decorator"""
        return f

elif OPT_TYPE == 'minimum':
    def optfunc(f):
        """Decorator for returning abs of function output"""
        def func(*args, **kw):
            """Return absolute value of function"""
            return np.abs(f(*args, **kw))
        return func


_ELL = 7
_TWOPI2ELL = 2 * _PI**(2*_ELL)
_K0 = np.array([
    (1 + 0.5**(_S + 0.5)) * np.prod(np.arange(1, 2*_S, 2)) * 2/(3*_SQRT2PI)
    for _S in range(_ELL-1, 1, -1)
])

@optfunc
@numba_jit(nopython=True, nogil=True, cache=True, fastmath=False)
def fixed_point_numba_np(t, n_datapoints, i_range, log_i_range, a2):
    """ISJ fixed-point calculation as per Botev et al..

    In this implementation, Numba makes calls to Numpy routines.

    Parameters
    ----------
    t
    n_datapoints
    i_range
    log_i_range
    a2

    Returns
    -------
    t_star

    """
    exparg = _ELL*log_i_range - i_range*(_PISQ*t)
    ex = np.exp(exparg)
    eff = _TWOPI2ELL * np.dot(a2, ex)

    for s in range(_ELL-1, 1, -1):
        t_elap = (_K0[s] / (n_datapoints * eff))**(2 / (3 + 2*s))
        exparg = s*log_i_range - i_range*(_PISQ*t_elap)
        ex = np.exp(exparg)
        dp = np.dot(a2, ex)
        eff = 2*_PI**(2*s) * dp

    return t - (2.0 * n_datapoints * _SQRTPI * eff)**-0.4


@optfunc
@numba_jit('{f}({f}, int64, {f}[:], {f}[:], {f}[:])'.format(f='float64'),
           nopython=True, nogil=True, cache=True, fastmath=False)
def fixed_point_numba_loops(t, n_datapoints, i_range, log_i_range, a2):
    """ISJ fixed-point calculation as per Botev et al..

    In this implementation, Numba explicitly loops over arrays.

    Parameters
    ----------
    t
    n_datapoints
    i_range
    log_i_range
    a2

    Returns
    -------
    t_star

    """
    len_i = len(i_range)

    tot = 0.0
    for idx in range(len_i):
        a2_el = a2[idx]
        log_i_range_el = log_i_range[idx]
        i_range_el = i_range[idx]

        exparg = _ELL*log_i_range_el - i_range_el * _PISQ * t
        tot += a2_el * exp(exparg)
    eff = _TWOPI2ELL * tot

    for s in range(_ELL-1, 1, -1):
        k0 = _K0[s]
        t_elap = (k0 / (n_datapoints * eff))**(2 / (3 + 2*s))

        tot = 0.0
        for idx in range(len_i):
            a2_el = a2[idx]
            log_i_range_el = log_i_range[idx]
            i_range_el = i_range[idx]

            exparg = s*log_i_range_el - i_range_el*_PISQ*t_elap
            tot += a2_el * exp(exparg)
        eff = 2 * _PI**(2*s) * tot

    return t - (2*n_datapoints*_SQRTPI*eff)**-0.4


@optfunc
@numba_jit('{f}({f}, int64, {f}[:], {f}[:])'.format(f='float64'),
           nopython=True, nogil=True, cache=True, fastmath=False)
def fixed_point_numba_orig(t, n_datapoints, i_range, a2):
    """Fixed point algorithm for Improved Sheather Jones bandwidth
    selection.

    Implements the fixed-point finding for the function
    ``t - zeta * gamma^{[l]}(t)``

    See The Annals of Statistics, 38(5):2916-2957, 2010.

    Parameters
    ----------
    t : float64
    n_datapoints : int64
    i_range : array of float64
    a2 : array of float64

    Returns
    -------
    t_star : float64

    NOTES
    -----
    Original implementation by Botev et al. is quad-precision, whereas this is
    double precision only. This might cause discrepancies from the reference
    implementation.

    """
    len_i = len(i_range)

    tot = 0.0
    for idx in range(len_i):
        tot += i_range[idx]**_ELL * a2[idx] * exp(-i_range[idx] * _PISQ * t)
    eff = 2 * _PI**(2*_ELL) * tot

    for s in range(_ELL-1, 1, -1):
        k0 = np.prod(np.arange(1, 2*s, 2)) / _SQRT2PI
        const = (1 + (0.5)**(s+0.5)) / 3.0
        t_elap = (2 * const * k0 / (n_datapoints * eff))**(2.0 / (3.0 + 2.0*s))

        tot = 0.0
        for idx in range(len_i):
            tot += (
                i_range[idx]**s * a2[idx] * exp(-i_range[idx] * _PISQ * t_elap)
            )
        eff = 2 * _PI**(2*s) * tot

    return t - (2.0 * n_datapoints * _SQRTPI * eff)**-0.4


def test_fbwkde():
    """Test speed of fbwkde implementation"""
    n_samp = int(1e4)
    n_dct = int(2**12)
    n_eval = int(1e4)
    x = np.linspace(0, 20, n_eval)
    np.random.seed(0)
    times = []
    for _ in range(3):
        enuerr = np.random.noncentral_chisquare(df=3, nonc=1, size=n_samp)
        t0 = time()
        fbwkde(data=enuerr, n_dct=n_dct, evaluate_at=x)
        times.append(time() - t0)
    tprofile.debug(
        'median time to run fbwkde, %d samples %d dct, eval. at %d points: %f'
        ' ms', n_samp, n_dct, n_eval, np.median(times)*1000
    )
    logging.info('<< PASS : test_fbwkde >>')


def test_vbwkde():
    """Test speed of unweighted vbwkde implementation"""
    n_samp = int(1e4)
    n_dct = int(2**12)
    n_eval = int(5e3)
    n_addl = 0
    x = np.linspace(0, 20, n_samp)
    np.random.seed(0)
    times = []
    for _ in range(3):
        enuerr = np.random.noncentral_chisquare(df=3, nonc=1, size=n_eval)
        t0 = time()
        vbwkde(data=enuerr, n_dct=n_dct, evaluate_at=x, n_addl_iter=n_addl)
        times.append(time() - t0)
    tprofile.debug(
        'median time to run vbwkde, %d samples %d dct %d addl iter, eval. at'
        ' %d points: %f ms', n_samp, n_dct, n_addl, n_eval, np.median(times)*1e3
    )
    logging.info('<< PASS : test_vbwkde >>')


def test_weighted_vbwkde():
    """Test speed of vbwkde implementation using weights"""
    n_samp = int(1e4)
    n_dct = int(2**12)
    n_eval = int(5e3)
    n_addl = 0
    x = np.linspace(0, 20, n_samp)
    np.random.seed(0)
    times = []
    for _ in range(3):
        enuerr = np.random.noncentral_chisquare(df=3, nonc=1, size=n_eval)
        weights = np.random.rand(n_eval)
        t0 = time()
        vbwkde(data=enuerr, weights=weights,
               n_dct=n_dct, evaluate_at=x, n_addl_iter=n_addl)
        times.append(time() - t0)
    tprofile.debug(
        'median time to run weighted vbwkde, %d samples %d dct %d addl iter,'
        ' eval. at %d points: %f ms',
        n_samp, n_dct, n_addl, n_eval, np.median(times)*1e3
    )
    logging.info('<< PASS : test_weighted_vbwkde >>')


if __name__ == "__main__":
    set_verbosity(2)
    test_fbwkde()
    test_vbwkde()
    test_weighted_vbwkde()
