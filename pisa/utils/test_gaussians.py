#!/usr/bin/env python
"""
Unittests for functions that live in the gaussians.pyx Cython module.
"""

from __future__ import division

from itertools import product, izip

import numpy as np
from scipy.stats import norm

from pisa import FTYPE, OMP_NUM_THREADS
from pisa.utils.comparisons import EQUALITY_PREC, FTYPE_PREC
from pisa.utils.gaussians import gaussian, gaussians
from pisa.utils.log import logging, set_verbosity


__all__ = ['test_gaussian', 'test_gaussians', 'speed_test_gaussians']


def test_gaussian():
    x = np.linspace(-10, 10, 1e3, dtype=FTYPE)

    # Test negative and positive ints and floats, and test 0
    means = -2.0, -1, 0, 1, 2.0

    # TODO: scipy populates nan if given negative scale parameter (stddev);
    # should we replicate this behavior or do (as now) and take absolute value
    # of the scale parameter? Or raise ValueError?

    # Test several values for stddev (zero should yield nan results)
    stddevs = 1, 2.0, 1e10, 1e-10, 0

    # Try out the threads functionality for each result; reset the accumulation
    # buffers if any contents are NaN, such that subsequent calculations can
    # actually be tested.
    threads = 1, 2
    for mu, sigma, threads in product(means, stddevs, threads):
        # Place to store result of `gaussian()`
        outbuf = np.full_like(x, np.nan, dtype=FTYPE)

        gaussian(outbuf, x, mu, sigma, threads)
        refbuf = norm.pdf(x, loc=mu, scale=sigma)
        if not np.allclose(outbuf, refbuf, rtol=EQUALITY_PREC*10, atol=0,
                           equal_nan=True):
            raise ValueError(
                'Max abs fractional diff = %e (EQUALITY_PREC=%e); mu=%e,'
                ' sigma=%e'
                % (np.max(np.abs(outbuf/refbuf-1)), EQUALITY_PREC*10, mu,
                   sigma)
            )
    logging.info('<< PASS : test_gaussian >>')


def test_gaussians():
    np.random.seed(0)
    mu = np.array(np.random.randn(int(1e3)), dtype=FTYPE)
    sigma = np.array(np.abs(np.random.randn(len(mu))), dtype=FTYPE)
    np.clip(sigma, a_min=1e-20, a_max=np.inf, out=sigma)

    x = np.linspace(-10, 10, 1e4, dtype=FTYPE)

    # Place to store result of `scipy.stats.norm`
    refbuf = np.zeros_like(x, dtype=FTYPE)

    # Compute the reference result
    for m, s in izip(mu, sigma):
        refbuf += norm.pdf(x, loc=m, scale=s)
    refbuf /= len(mu)

    outbuf = gaussians(x, mu, sigma)
    if not np.allclose(outbuf, refbuf, rtol=EQUALITY_PREC*10, atol=0,
                       equal_nan=True):
        maxfractdiff = np.max(np.abs(outbuf/refbuf - 1))
        logging.error(
            'outbuf=\n%s\nrefbuf=\n%s\nmu=\n%s\nsigma=\n%s\nthreads=%d',
            outbuf, refbuf, mu, sigma, OMP_NUM_THREADS
        )
        raise ValueError(
            '%s failed: max fractional disagreement is %s, which exceeds'
            ' allowed tolerance of %s.'
            % (__name__, maxfractdiff, EQUALITY_PREC*10)
        )
    logging.info('<< PASS : test_gaussians >>')

# TODO: looping over number of threads doesn't work since it is no longer an
# argument to `gaussians`!
def speed_test_gaussians(num_gaussians, num_points):
    import multiprocessing
    import time
    import sys
    raise NotImplementedError()
    assert int(num_gaussians) == float(num_gaussians), \
            'must pass integral value or equivalent for `num_gaussians`'
    assert int(num_points) == float(num_points), \
            'must pass integral value or equivalent for `num_points`'
    num_gaussians = int(num_gaussians)
    num_points = int(num_points)

    num_cpu = multiprocessing.cpu_count()
    logging.info('Reported #CPUs: %d (includes any hyperthreading)', num_cpu)
    logging.info('Summing %d Gaussians evaluated at %d points...',
                 num_gaussians, num_points)

    np.random.seed(0)
    mu = np.array(np.random.randn(num_gaussians), dtype=FTYPE)
    sigma = np.array(np.abs(np.random.randn(len(mu))), dtype=FTYPE)
    np.clip(sigma, a_min=1e-20, a_max=np.inf, out=sigma)

    x = np.linspace(-10, 10, num_points, dtype=FTYPE)

    # Place to store result of `gaussians()`; zero-stuffed in the below lopp
    outbuf = np.empty_like(x, dtype=FTYPE)

    # Place to store result of `scipy.stats.norm`
    refbuf = np.zeros_like(outbuf, dtype=FTYPE)

    # Check default beahvior (possibly controlled by environment var
    # OMP_NUM_THREADS, if this is set)
    t0 = time.time()
    gaussians(outbuf, x, mu, sigma)
    timing = time.time() - t0

    # Try out the threads functionality for each result; reset the accumulation
    # buffer each time.
    timings = []
    logging.info('%7s %10s %7s', 'Threads', 'Time (s)', 'Speedup')
    for threads in range(1, num_cpu+1):
        outbuf.fill(0)
        t0 = time.time()
        outbuf = gaussians(x, mu, sigma)
        timing = time.time() - t0
        timings.append({'threads': threads, 'timing': timing})

        logging.info(
            '%7d %10.3e %7s', threads, timing,
            format(timings[0]['timing']/timing, '5.3f')
        )

    return timings


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        'Run tests on functions in gaussians.pyx; by default, runs unit tests.'
    )
    parser.add_argument(
        '-s', '--speed', action='store_true',
        help='''Run speed test rather than unit tests'''
    )
    parser.add_argument(
        '--num-gaussians', type=float, default=1e4,
        help='Number of Gaussians to sum if running speed test'
    )
    parser.add_argument(
        '--num-points', type=float, default=1e4,
        help='Number of points to evaluate if running speed test'
    )
    parser.add_argument(
        'v', action='count',
        help='Set logging verbosity level; repeat for more verbose output'
    )
    args = parser.parse_args()
    set_verbosity(args.v)
    if args.speed:
        speed_test_gaussians(num_gaussians=args.num_gaussians,
                             num_points=args.num_points)
    else:
        test_gaussian()
        test_gaussians()
