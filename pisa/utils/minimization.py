"""
Common minimization tools and constants.
"""


from __future__ import absolute_import, division

from collections.abc import Sequence

import re
import sys

import numpy as np
import scipy.optimize as optimize
from scipy.optimize import OptimizeResult

from pisa import EPSILON, FTYPE, ureg
from pisa.utils.comparisons import recursiveEquality
from pisa.utils.log import logging
from pisa.utils.random_numbers import get_random_state

__all__ = ['MINIMIZERS_USING_SYMM_GRAD', 'LOCAL_MINIMIZERS_WITH_DEFAULTS',
           'GLOBAL_MINIMIZERS_WITH_DEFAULTS', 'Counter',
           'set_minimizer_defaults', 'validate_minimizer_settings',
           'override_min_opt', '_run_minimizer', 'minimizer_x0_bounds',
           'display_minimizer_header', '_run_local_minimizer',
           '_run_global_minimizer', 'Bounds', 'RandomDisplacementWithBounds']

__author__ = 'J.L. Lanfranchi, P. Eller, S. Wren, T. Ehrhardt'

__license__ = '''Copyright (c) 2014-2020, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


MINIMIZERS_USING_SYMM_GRAD = ('l-bfgs-b', 'slsqp')
"""Minimizers that use symmetrical steps on either side of a point to compute
gradients. See https://github.com/scipy/scipy/issues/4916"""

LOCAL_MINIMIZERS_WITH_DEFAULTS = ('l-bfgs-b', 'slsqp')
"""Local minimizers which can be selected without specifying any configuration
as defaults will be set automatically."""

GLOBAL_MINIMIZERS_WITH_DEFAULTS = ('basinhopping', )
"""Local minimizers which can be selected without specifying any configuration
as defaults will be set automatically."""

# TODO: add Nelder-Mead, as it was used previously...
def set_minimizer_defaults(minimizer_settings):
    """Fill in default values for minimizer settings.

    Parameters
    ----------
    minimizer_settings : dict

    Returns
    -------
    new_minimizer_settings : dict

    """
    new_minimizer_settings = dict(
        method='',
        options=dict()
    )
    new_minimizer_settings.update(minimizer_settings)

    sqrt_ftype_eps = np.sqrt(np.finfo(FTYPE).eps)
    opt_defaults = {}
    method = minimizer_settings['method'].lower()

    if method == 'l-bfgs-b' and FTYPE == np.float64:
        # From `scipy.optimize.lbfgsb._minimize_lbfgsb`
        opt_defaults.update(dict(
            maxcor=10, ftol=2.2204460492503131e-09, gtol=1e-5, eps=1e-8,
            maxfun=15000, maxiter=15000, iprint=-1, maxls=20
        ))
    elif method == 'l-bfgs-b' and FTYPE == np.float32:
        # Adapted to lower precision
        opt_defaults.update(dict(
            maxcor=10, ftol=sqrt_ftype_eps, gtol=1e-3, eps=1e-5,
            maxfun=15000, maxiter=15000, iprint=-1, maxls=20
        ))
    elif method == 'slsqp' and FTYPE == np.float64:
        opt_defaults.update(dict(
            maxiter=100, ftol=1e-6, iprint=0, eps=sqrt_ftype_eps,
        ))
    elif method == 'slsqp' and FTYPE == np.float32:
        opt_defaults.update(dict(
            maxiter=100, ftol=1e-4, iprint=0, eps=sqrt_ftype_eps
        ))
    elif method == 'basinhopping': # TODO: ftype check?
        # cf. `scipy.optimize.basinhopping`
        opt_defaults.update(dict(
            niter=100, T=1.0, stepsize=0.5, interval=50
        ))
    else:
        raise ValueError('Unhandled minimizer "%s" / FTYPE=%s'
                         % (method, FTYPE))

    opt_defaults.update(new_minimizer_settings['options'])

    new_minimizer_settings['options'] = opt_defaults

    return new_minimizer_settings


# TODO: add Nelder-Mead, as it was used previously...
def validate_minimizer_settings(minimizer_settings):
    """Validate minimizer settings.

    See source for specific thresholds set.

    Parameters
    ----------
    minimizer_settings : dict

    Raises
    ------
    ValueError
        If any minimizer settings are deemed to be invalid.

    """
    ftype_eps = np.finfo(FTYPE).eps
    method = minimizer_settings['method'].lower()
    options = minimizer_settings['options']
    if method == 'l-bfgs-b':
        must_have = ('maxcor', 'ftol', 'gtol', 'eps', 'maxfun', 'maxiter',
                     'maxls')
        may_have = must_have + ('args', 'jac', 'bounds', 'disp', 'iprint',
                                'callback')
    elif method == 'slsqp':
        must_have = ('maxiter', 'ftol', 'eps')
        may_have = must_have + ('args', 'jac', 'bounds', 'constraints',
                                'iprint', 'disp', 'callback')

    elif method == 'basinhopping':
        must_have = ('niter', 'T', 'stepsize')
        may_have = must_have + ('take_step', 'callback', 'interval',
                                'disp', 'niter_success', 'seed')
    else:
        raise ValueError('Cannot validate unhandled minimizer "%s".' % method)

    missing = set(must_have).difference(set(options))
    excess = set(options).difference(set(may_have))
    if missing:
        raise ValueError('Missing the following options for %s minimizer: %s'
                         % (method, missing))
    if excess:
        raise ValueError('Excess options for %s minimizer: %s'
                         % (method, excess))

    eps_msg = '%s minimizer option %s(=%e) is < %d * %s_EPS(=%e)'
    eps_gt_msg = '%s minimizer option %s(=%e) is > %e'
    fp64_eps = np.finfo(np.float64).eps

    if method == 'l-bfgs-b':
        err_lim, warn_lim = 2, 10
        for s in ['ftol', 'gtol']:
            val = options[s]
            if val < err_lim * ftype_eps:
                raise ValueError(eps_msg % (method, s, val, err_lim, 'FTYPE',
                                            ftype_eps))
            if val < warn_lim * ftype_eps:
                logging.warn(eps_msg, method, s, val, warn_lim, 'FTYPE',
                             ftype_eps)

        val = options['eps']
        err_lim, warn_lim = 1, 10
        if val < err_lim * fp64_eps:
            raise ValueError(eps_msg % (method, 'eps', val, err_lim, 'FP64',
                                        fp64_eps))
        if val < warn_lim * ftype_eps:
            logging.warn(eps_msg, method, 'eps', val, warn_lim, 'FTYPE',
                         ftype_eps)

        err_lim, warn_lim = 0.25, 0.1
        if val > err_lim:
            raise ValueError(eps_gt_msg % (method, 'eps', val, err_lim))
        if val > warn_lim:
            logging.warn(eps_gt_msg, method, 'eps', val, warn_lim)

        # make sure we only have integers where we can only have integers
        for s in ('maxcor', 'maxfun', 'maxiter', 'maxls', 'iprint'):
            try:
                options[s] = int(options[s])
            except:
                # if the setting doesn't exist in the first place we don't care
                pass

    if method == 'slsqp':
        err_lim, warn_lim = 2, 10
        val = options['ftol']
        if val < err_lim * ftype_eps:
            raise ValueError(eps_msg % (method, 'ftol', val, err_lim, 'FTYPE',
                                        ftype_eps))
        if val < warn_lim * ftype_eps:
            logging.warn(eps_msg, method, 'ftol', val, warn_lim, 'FTYPE',
                         ftype_eps)

        val = options['eps']
        err_lim, warn_lim = 1, 10
        if val < err_lim * fp64_eps:
            raise ValueError(eps_msg % (method, 'eps', val, 1, 'FP64',
                                        fp64_eps))
        if val < warn_lim * ftype_eps:
            logging.warn(eps_msg, method, 'eps', val, warn_lim, 'FP64',
                         fp64_eps)

        err_lim, warn_lim = 0.25, 0.1
        if val > err_lim:
            raise ValueError(eps_gt_msg % (method, 'eps', val, err_lim))
        if val > warn_lim:
            logging.warn(eps_gt_msg, method, 'eps', val, warn_lim)

        # make sure we only have integers where we can only have integers
        for s in ('maxiter', 'iprint'):
            try:
                options[s] = int(options[s])
            except:
                # if the setting doesn't exist in the first place we don't care
                pass

    if method == 'basinhopping':
        for s in ('niter', 'interval', 'niter_success'):
            try:
                options[s] = int(options[s])
            except:
                # if the setting doesn't exist in the first place we don't care
                pass


def override_min_opt(minimizer_settings, min_opt):
    """Override minimizer option:value pair(s) in a minimizer settings dict
    """
    for opt_val_str in min_opt:
        opt, val_str = [s.strip() for s in opt_val_str.split(':')]
        try:
            val = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
        minimizer_settings['options'][opt] = val

# --------------------------------------------------------------------------- #

def minimizer_x0_bounds(free_params, minimizer_settings, padding_factor=1.,
                        randomize_params=None, random_state=None):
    """Ensure values of free parameters are within their bounds
    (given floating point precision) and adapt minimizer bounds
    if necessary to prevent it from stepping outside of
    user-specified bounds.

    Parameters
    ----------
    free_params : ParamSet
        Obtain starting values and user-specified bounds
    minimizer_settings : dict
        Parsed minimizer cfg (method and stepsize relevant)
    padding_factor : int or float
        Bounds for gradient based minimization padded by this factor*stepsize
    randomize_params : sequence of str or bool
        list of param names or `True`/`False`
    random_state : random state or instantiable thereto
        initial random state

    Returns
    -------
    x0 : Sequence
        Normalised and clipped parameter values
    bounds: Sequence (of 2-tuples)
        Normalised and possibly shrunk parameter bounds

    """
    if isinstance(randomize_params, Sequence):
        # just randomise specified parameters
        for pname in randomize_params:
            free_params[pname].randomize(
                random_state=get_random_state(random_state)
            )
    elif isinstance(randomize_params, bool):
        if randomize_params:
            # randomise all free
            free_params.randomize_free(
                random_state=get_random_state(random_state)
            )
    elif randomize_params is not None:
        raise TypeError('Unhandled type "%s" of `randomize_params`!'
                        % type(randomize_params))

    # Get starting free parameter values
    x0 = free_params._rescaled_values # pylint: disable=protected-access
    bounds = [(0, 1)]*len(x0)
    if minimizer_settings is None:
        return x0, bounds
    minimizer_method = minimizer_settings['method'].lower()
    if minimizer_method in MINIMIZERS_USING_SYMM_GRAD:
        logging.info(
            'Local minimizer %s requires artificial boundaries SMALLER than'
            ' the user-specified boundaries (so that numerical gradients do'
            ' not exceed the user-specified boundaries).',
            minimizer_method
        )
        step_size = minimizer_settings['options']['eps']
        bounds = [(0 + padding_factor*step_size, 1 - padding_factor*step_size)]*len(x0)

    clipped_x0 = []
    for param, x0_val, bds in zip(free_params, x0, bounds):
        if x0_val < bds[0] - EPSILON:
            logging.warn(
                'Param %s, initial scaled value %.17e is below lower bound'
                ' %.17e. Fixing this.' % (param.name, x0_val, bds[0])
            )
            x0_val = bds[0] + EPSILON
        if x0_val > bds[1] + EPSILON:
            logging.warn(
                'Param %s, initial scaled value %.17e exceeds upper bound'
                ' %.17e. Fixing this.' % (param.name, x0_val, bds[1])
            )
            x0_val = bds[1] - EPSILON

        clipped_x0_val = np.clip(x0_val, a_min=bds[0], a_max=bds[1])
        clipped_x0.append(clipped_x0_val)

        if recursiveEquality(clipped_x0_val, bds[0]):
            logging.warn(
                'Param %s, initial scaled value %e is at the lower bound;'
                ' minimization may fail as a result.',
                param.name, clipped_x0_val
            )
        if recursiveEquality(clipped_x0_val, bds[1]):
            logging.warn(
                'Param %s, initial scaled value %e is at the upper bound;'
                ' minimization may fail as a result.',
                param.name, clipped_x0_val
            )

    x0 = tuple(clipped_x0)
    return x0, bounds

def test_minimizer_x0_bounds(
    cfg='settings/minimizer/slsqp_ftol1e-6_eps1e-4_maxiter1000.cfg'
):
    """Unit tests of `minimizer_x0_bounds`."""
    from pisa.core.param import Param, ParamSet
    from pisa.utils.comparisons import recursiveEquality
    from pisa.utils.config_parser import parse_minimizer_config
    p0 = Param(name='physics-y parameter', value=0. * ureg.eV**2, prior=None,
               range=[-0.7, 0.5] * ureg.eV**2, is_fixed=False, is_discrete=False)
    p1 = Param(name='very important nuisance', value=0.64, prior=None,
               range=[0., 2.0], is_fixed=False, is_discrete=False)
    p2 = Param(name='not-a-nuisance', value=0.0, prior=None,
               range=[-3., 3.], is_fixed=True, is_discrete=False)
    param_set = ParamSet(p0, p1, p2)
    # first test the simplest setting: no minimizer or any other
    # fancy settings
    x0, bounds = minimizer_x0_bounds(
        free_params=param_set.free,
        minimizer_settings=None
    )
    for x0_val, rescaled_val in zip(x0, param_set._rescaled_values):
        assert recursiveEquality(x0_val, rescaled_val)
    assert bounds == [(0, 1)] * len(x0)

    randomize_params = [p2.name]
    try:
        x0, bounds = minimizer_x0_bounds(
            free_params=param_set.free,
            minimizer_settings=None,
            randomize_params=randomize_params
        )
    except KeyError:
        # the above should fail with a KeyError, since p2 is fixed
        pass
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    else:
        raise RuntimeError("Should have raised a `KeyError`!")

    for randomize_params in [[p0.name], [p1.name], param_set.free.names]:
        x0, bounds = minimizer_x0_bounds(
            free_params=param_set.free,
            minimizer_settings=None,
            randomize_params=randomize_params
        )
        for param, x0_val in zip(param_set.free, x0):
            if param.name in randomize_params:
                assert not recursiveEquality(param.value, param.nominal_value)
                assert recursiveEquality(param._rescaled_value, x0_val)

    # should be handled and warned about
    p0.value = p0.range[0]
    p1.value = p1.range[1]
    x0, bounds = minimizer_x0_bounds(
        free_params=param_set.free,
        minimizer_settings=parse_minimizer_config(cfg),
        randomize_params=False
    )

# --------------------------------------------------------------------------- #


class Bounds(object):
    """Just some parameter bounds.
    """
    def __init__(self, xmax, xmin):
        """Acceptance test to make global minimizer respect bounds
        (this does not mean it won't try to evaluate the objective function
        outside the bounds though, which might lead to an exception).
        (source: `scipy.optimize.basinhopping` docs)

        Parameters
        ----------
        xmax : Sequence
            Upper bounds
        xmin : Sequence
            Lower bounds
        """
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


class RandomDisplacementWithBounds(object):
    """
    Add a random displacement of maximum size `stepsize` to each coordinate,
    respecting each parameter's bounds (modified from basinhopping's internal
    random displacement method).
    Calling this updates `x` in-place.

    Parameters
    ----------
    stepsize : float, optional
        Maximum stepsize in any dimension
    random_state : None or `np.random.RandomState` instance, optional
        The random number generator that generates the displacements
    bounds : list, optional
        Bounds in each dimension

    """
    def __init__(self, stepsize=0.5, random_state=None, bounds=None):
        self.stepsize = stepsize
        self.random_state = get_random_state(random_state)
        self.bounds = bounds

    def __call__(self, x):
        if self.bounds is not None:
            perturbation = self.random_state.uniform(
                np.maximum(self.bounds[:, 0] - x, -self.stepsize),
                np.minimum(self.bounds[:, 1] - x, self.stepsize),
                np.shape(x)
            )
        else:
            perturbation = self.random_state.uniform(
                -self.stepsize, self.stepsize, np.shape(x)
            )
        x += perturbation
        logging.info('New position after random perturbation: %s.' % x)
        return x


class Counter(object):
    """Simple counter object for use e.g. as a minimizer callback."""
    def __init__(self, i=0):
        self._count = i

    def __str__(self):
        return str(self._count)

    def __repr__(self):
        return str(self)

    def __iadd__(self, inc):
        self._count += inc
        return self

    def reset(self):
        """Reset counter"""
        self._count = 0

    @property
    def count(self):
        """int : Current count"""
        return self._count


def dummy(fun, x0, args, **kwargs):
    """Just a dummy method that can be passed to the global optimization
    routine to capture the latter's behaviour without a local minimizer.
    """
    return OptimizeResult({'x': x0,
                           'success': True,
                           'message' : 'Did nothing',
                           'fun': fun(x0, *args),
                           'nfev': 1,
                           'nit': 0})


def _run_global_minimizer(fun, x0, bounds, random_state,
                          minimizer_settings, minimizer_callback,
                          hypo_maker, data_dist, metric, sign,
                          counter, fit_history, pprint, blind,
                          external_priors_penalty):
    """Run global (+local) minimization routine via
    `scipy.optimize` interface:
    `basinhopping`, `brute`, `differential_evolution`

    Parameters
    ----------
    cf. `_run_minimizer`

    Returns
    -------
    optimize_result : OptimizeResult

    """

    method = minimizer_settings['global']['method']
    if method != 'basinhopping':
        logging.warn('Global minimization only tested with basin hopping'
                     ' algorithm. This might fail spectacularly as a result!')
    options = minimizer_settings['global']['options']
    if minimizer_callback is not None:
        options.update({'callback': minimizer_callback})
    logging.debug('Running the global "%s" minimizer...' % method)

    minimizer_kwargs = {
        'args': (hypo_maker, data_dist, metric, sign, counter, fit_history,
                 pprint, blind, external_priors_penalty)
    }
    if minimizer_settings['local'] is not None:
        minimizer_kwargs.update(minimizer_settings['local'])
        # bounds for local minimizer
        minimizer_kwargs['bounds'] = bounds
    else:
        # make sure we don't use the default local minimizer when the user
        # doesn't explicitly request a method
        minimizer_kwargs.update({'method': dummy})

    # custom random perturbation routine that respects all bounds
    step = RandomDisplacementWithBounds(
        stepsize=options['stepsize'], bounds=np.array(bounds, dtype=FTYPE),
        random_state=random_state
    )
    bounds = Bounds(xmax=np.array(bounds)[:, 1], xmin=np.array(bounds)[:, 0])
    global_min = getattr(optimize, method)
    optimize_result = global_min(
        func=fun,
        x0=x0,
        minimizer_kwargs=minimizer_kwargs,
        accept_test=bounds,
        take_step=step,
        **options
    )

    return optimize_result

# TODO: random_state?
def _run_local_minimizer(fun, x0, bounds, random_state,
                         minimizer_settings, minimizer_callback,
                         hypo_maker, data_dist, metric, sign,
                         counter, fit_history, pprint, blind,
                         external_priors_penalty):
    """Run arbitrary local minimization routine
    via `scipy.optimize.minimize` interface.

    Parameters
    ----------
    cf. `_run_minimizer`

    Returns
    -------
    optimize_result : OptimizeResult

    """

    method = minimizer_settings['method']
    options = minimizer_settings['options']
    logging.debug('Running the local "%s" minimizer...' % method)

    optimize_result = optimize.minimize(
        fun=fun,
        x0=x0,
        args=(hypo_maker, data_dist, metric, sign, counter, fit_history, pprint,
              blind, external_priors_penalty),
        bounds=bounds,
        method=method,
        options=options,
        callback=minimizer_callback
    )

    return optimize_result


def _run_minimizer(fun, x0, bounds, random_state,
                   minimizer_settings, minimizer_callback,
                   hypo_maker, data_dist, metric, sign, counter, fit_history, pprint,
                   blind, external_priors_penalty):
    """A wrapper that dispatches a global or a local minimization
    routine according to minimizer_settings.

    Parameters
    ----------
    fun : callable
        function that is minimized
    x0 : Sequence
        minimizer initial guess (normalized to [0,1])
    bounds : Sequence of 2-tuples
        minimizer bounds (one pair per value in x0)
    random_state : random state or instantiable thereto
        for reproducibility of (hopefully all) random processes
    minimizer_settings : dict
        dictionary containing parsed 'global' and/or 'local'
        minimizer configs
    minimizer_callback : callable
        callback function called after each iteration/
        for each minimum found
    hypo_maker : DistributionMaker
    data_dist : MapSet
        (pseudo-)data distribution
    metric : str
        metric to minimize
    sign : +1 or -1
        sign with which to multipy overall metric value
    counter : Counter
        counter passed to minimizer callable that keeps track
        of the number of function calls
    fit_history : Sequence
        passed to minimizer callable to record progress of minimizer
        (metric and parameter values)
    pprint : bool
    blind : bool
    external_priors_penalty : func
        User defined prior penalty function

    Returns
    -------
    optimize_result: OptimizeResult

    """
    if minimizer_settings['global'] is not None:
        # can make use of both global and local minimizers, so pass in
        # whole minimizer_settings
        optimize_result = _run_global_minimizer(
            fun=fun, x0=x0, bounds=bounds, random_state=random_state,
            minimizer_settings=minimizer_settings,
            minimizer_callback=minimizer_callback,
            hypo_maker=hypo_maker, data_dist=data_dist, metric=metric,
            sign=sign, counter=counter, fit_history=fit_history,
            pprint=pprint, blind=blind,
            external_priors_penalty=external_priors_penalty
        )

    elif minimizer_settings['local'] is not None:
        optimize_result = _run_local_minimizer(
            fun=fun, x0=x0, bounds=bounds, random_state=random_state,
            minimizer_settings=minimizer_settings['local'],
            minimizer_callback=minimizer_callback,
            hypo_maker=hypo_maker, data_dist=data_dist, metric=metric,
            sign=sign, counter=counter, fit_history=fit_history,
            pprint=pprint, blind=blind,
            external_priors_penalty=external_priors_penalty
        )
    else:
        raise ValueError("No minimizer routine selected!")
    return optimize_result


def display_minimizer_header(free_params, metric):
    """Display nicely formatted header for use with minimizer.

    Parameters
    ----------
    free_params : ParamSet
    metric : str

    """
    # Display any units on top
    r = re.compile(r'(^[+0-9.eE-]* )|(^[+0-9.eE-]*$)')
    hdr = ' '*(6+1+10+1+12+3)
    unt = []
    for p in free_params:
        u = r.sub('', format(p.value, '~')).replace(' ', '')[0:10]
        if u:
            u = '(' + u + ')'
        unt.append(u.center(12))
    hdr += ' '.join(unt)
    hdr += '\n'

    # Header names
    hdr += ('iter'.center(6) + ' ' + 'funcalls'.center(10) + ' ' +
            metric[0:12].center(12) + ' | ')
    hdr += ' '.join([p.name[0:12].center(12) for p in free_params])
    hdr += '\n'

    # Underscores
    hdr += ' '.join(['-'*6, '-'*10, '-'*12, '+'] + ['-'*12]*len(free_params))
    hdr += '\n'

    sys.stdout.write(hdr)


if __name__ == '__main__':
    test_minimizer_x0_bounds()
