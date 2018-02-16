"""
Common minimization tools and constants.
"""


from __future__ import absolute_import, division

import numpy as np
import scipy.optimize as optimize

from pisa import EPSILON, FTYPE, ureg

__all__ = ['MINIMIZERS_USING_SYMM_GRAD', 'LOCAL_MINIMIZERS_WITH_DEFAULTS',
           'GLOBAL_MINIMIZERS_WITH_DEFAULTS', 'Counter',
           'set_minimizer_defaults', 'validate_minimizer_settings',
           'override_min_opt']

__author__ = 'J.L. Lanfranchi, P. Eller, S. Wren, T. Ehrhardt'

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
        method=dict(value='', desc=''),
        options=dict(value=dict(), desc=dict())
    )
    new_minimizer_settings.update(minimizer_settings)

    sqrt_ftype_eps = np.sqrt(np.finfo(FTYPE).eps)
    opt_defaults = {}
    method = minimizer_settings['method']['value'].lower()

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

    opt_defaults.update(new_minimizer_settings['options']['value'])

    new_minimizer_settings['options']['value'] = opt_defaults

    # Populate the descriptions with something
    for opt_name in new_minimizer_settings['options']['value']:
        if opt_name not in new_minimizer_settings['options']['desc']:
            new_minimizer_settings['options']['desc'] = 'no desc'

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
    method = minimizer_settings['method']['value'].lower()
    options = minimizer_settings['options']['value']
    if method == 'l-bfgs-b':
        must_have = ('maxcor', 'ftol', 'gtol', 'eps', 'maxfun', 'maxiter',
                     'maxls')
        may_have = must_have + ('args', 'jac', 'bounds', 'disp', 'iprint',
                                'callback')
    elif method == 'slsqp':
        must_have = ('maxiter', 'ftol', 'eps')
        may_have = must_have + ('args', 'jac', 'bounds', 'constraints',
                                'iprint', 'disp', 'callback')

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
        minimizer_settings['options']['value'][opt] = val
        minimizer_settings['options']['desc'][opt] = 'no desc'


class Counter(object):
    """Simple counter object for use as a minimizer callback."""
    def __init__(self, i=0):
        self._count = i

    def __str__(self):
        return str(self._count)

    def __repr__(self):
        return str(self)

    def __iadd__(self, inc):
        self._count += inc

    def reset(self):
        """Reset counter"""
        self._count = 0

    @property
    def count(self):
        """int : Current count"""
        return self._count

