"""
Utilities for validating and configuring scipy minimization settings.
"""

from collections.abc import Sequence, Mapping
from copy import deepcopy
from functools import partial

import numpy as np
from pkg_resources import parse_version
import scipy

from pisa import FTYPE
from pisa.analysis.manipulate_params import update_param_values_detector
from pisa.utils.fileio import EVAL_MSG
from pisa.utils.log import logging

__all__ = ['make_scipy_local_minimizer_kwargs', 'set_minimizer_defaults',
           'validate_minimizer_settings', 'make_scipy_constraint_dict',
           'scipy_constraints_to_callables']


def make_scipy_local_minimizer_kwargs(minimizer_settings, constrs=None, bounds=None):
    """Small helper function containing common logic for
    creating minimizer keyword args in calls to global
    routines via their scipy interface."""
    minimizer_kwargs = deepcopy(minimizer_settings)
    minimizer_kwargs["method"] = minimizer_settings["method"]["value"]
    minimizer_kwargs["options"] = minimizer_settings["options"]["value"]
    if constrs is not None:
        minimizer_kwargs["constraints"] = constrs
    if bounds is not None:
        minimizer_kwargs["bounds"] = bounds
    return minimizer_kwargs


def set_minimizer_defaults(minimizer_settings):
    """Fill in default values for minimizer settings.

    Parameters
    ----------
    minimizer_settings : dict

    Returns
    -------
    new_minimizer_settings : dict

    """
    new_minimizer_settings = {
        'method': {'value': '', 'desc': ''},
        'options': {'value': {}, 'desc': {}}
    }
    new_minimizer_settings.update(minimizer_settings)

    sqrt_ftype_eps = np.sqrt(np.finfo(FTYPE).eps)
    opt_defaults = {}
    method = minimizer_settings['method']['value'].lower()

    if method == 'l-bfgs-b' and FTYPE == np.float64:
        # From `scipy.optimize.lbfgsb._minimize_lbfgsb`
        opt_defaults = {
            'maxcor': 10,
            'ftol': 2.2204460492503131e-09,
            'gtol': 1e-5,
            'eps': 1e-8,
            'maxfun': 15000,
            'maxiter': 15000,
            'maxls': 20
        }
    elif method == 'l-bfgs-b' and FTYPE == np.float32:
        # Adapted to lower precision
        opt_defaults = {
            'maxcor': 10,
            'ftol': sqrt_ftype_eps,
            'gtol': 1e-3,
            'eps': 1e-5,
            'maxfun': 15000,
            'maxiter': 15000,
            'maxls': 20
        }
    elif method == 'slsqp' and FTYPE == np.float64:
        opt_defaults = {
            'maxiter': 100,
            'ftol': 1e-6,
            'iprint': 0,
            'eps': sqrt_ftype_eps,
        }
    elif method == 'slsqp' and FTYPE == np.float32:
        opt_defaults = {
            'maxiter': 100,
            'ftol': 1e-4,
            'iprint': 0,
            'eps': sqrt_ftype_eps
        }
    elif method == 'cobyla':
        opt_defaults = {
            'rhobeg': 0.1,
            'maxiter': 1000,
            'tol': 1e-4,
        }
    elif method == 'cobyqa':
        # just make this solver available for now
        pass
    elif method == 'trust-constr':
        opt_defaults = {
            'maxiter': 200,
            'gtol': 1e-4,
            'xtol': 1e-4,
            'barrier_tol': 1e-4
        }
    elif method == 'nelder-mead':
        opt_defaults = {
            'maxfev': 1000,
            'xatol': 1e-4,
            'fatol': 1e-4
        }
    else:
        raise ValueError(f'Unhandled minimizer "{method}" / FTYPE={FTYPE}')

    opt_defaults.update(new_minimizer_settings['options']['value'])

    new_minimizer_settings['options']['value'] = opt_defaults

    # Populate the descriptions with something
    for opt_name in new_minimizer_settings['options']['value']:
        if opt_name not in new_minimizer_settings['options']['desc']:
            new_minimizer_settings['options']['desc'] = 'no desc'

    return new_minimizer_settings


def validate_minimizer_settings(minimizer_settings):
    """Validate minimizer settings.

    Supported minimizers are the same as in `set_minimizer_defaults`.

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
        # TODO: remove disp and iprint eventually (>= scipy 1.18.0)
        may_have = must_have + ('args', 'jac', 'bounds', 'disp', 'iprint',
                                'callback')
    elif method == 'slsqp':
        must_have = ('maxiter', 'ftol', 'eps')
        may_have = must_have + ('args', 'jac', 'bounds', 'constraints',
                                'iprint', 'disp', 'callback')
    elif method == 'cobyla':
        must_have = ('maxiter', 'rhobeg', 'tol')
        may_have = must_have + ('disp', 'catol', 'constraints')
    elif method == 'cobyqa':
        assert parse_version(scipy.__version__) >= parse_version('1.14.0')
        must_have = ()
        may_have = must_have + ('disp', 'maxiter', 'maxfev', 'f_target',
                                'feasibility_tol', 'initial_tr_radius',
                                'final_tr_radius', 'scale', 'constraints')
    elif method == 'trust-constr':
        must_have = ('maxiter', 'gtol', 'xtol', 'barrier_tol')
        may_have = must_have + ('sparse_jacobian', 'initial_tr_radius',
                                'initial_constr_penalty', 'constraints',
                                'initial_barrier_parameter',
                                'initial_barrier_tolerance',
                                'factorization_method',
                                'finite_diff_rel_step',
                                'verbose', 'disp')
    elif method == 'nelder-mead':
        must_have = ('maxfev', 'xatol', 'fatol')
        may_have = must_have + ('disp', 'maxiter', 'return_all',
                                'initial_simplex', 'adaptive',
                                'bounds')
    else:
        raise ValueError(f'Unhandled minimizer "{method}" / FTYPE={FTYPE}')

    missing = set(must_have).difference(set(options))
    excess = set(options).difference(set(may_have))
    if missing:
        raise ValueError(f'Missing the following options for {method} '
                         f'minimizer: {missing}')
    if excess:
        raise ValueError(f'Excess options for {method} minimizer: {excess}')

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
                logging.warning(eps_msg, method, s, val, warn_lim, 'FTYPE', ftype_eps)

        val = options['eps']
        err_lim, warn_lim = 1, 10
        if val < err_lim * fp64_eps:
            raise ValueError(eps_msg % (method, 'eps', val, err_lim, 'FP64',
                                        fp64_eps))
        if val < warn_lim * ftype_eps:
            logging.warning(eps_msg, method, 'eps', val, warn_lim, 'FTYPE', ftype_eps)

        err_lim, warn_lim = 0.25, 0.1
        if val > err_lim:
            raise ValueError(eps_gt_msg % (method, 'eps', val, err_lim))
        if val > warn_lim:
            logging.warning(eps_gt_msg, method, 'eps', val, warn_lim)

    elif method == 'slsqp':
        err_lim, warn_lim = 2, 10
        val = options['ftol']
        if val < err_lim * ftype_eps:
            raise ValueError(eps_msg % (method, 'ftol', val, err_lim, 'FTYPE',
                                        ftype_eps))
        if val < warn_lim * ftype_eps:
            logging.warning(eps_msg, method, 'ftol', val, warn_lim, 'FTYPE', ftype_eps)

        val = options['eps']
        err_lim, warn_lim = 1, 10
        if val < err_lim * fp64_eps:
            raise ValueError(eps_msg % (method, 'eps', val, 1, 'FP64',
                                        fp64_eps))
        if val < warn_lim * ftype_eps:
            logging.warning(eps_msg, method, 'eps', val, warn_lim, 'FP64', fp64_eps)

        err_lim, warn_lim = 0.25, 0.1
        if val > err_lim:
            raise ValueError(eps_gt_msg % (method, 'eps', val, err_lim))
        if val > warn_lim:
            logging.warning(eps_gt_msg, method, 'eps', val, warn_lim)

    elif method == 'cobyla':
        if options['rhobeg'] > 0.5:
            raise ValueError('starting step-size > 0.5 will overstep boundary')
        if options['rhobeg'] < 1e-2:
            logging.warning('starting step-size is very low, convergence will be slow')


def make_scipy_constraint_dict(constr_type, fun, jac=None, args=None):
    """Makes a constraint dictionary in the form accepted by scipy,
    see e.g. https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.optimize.minimize.html"""
    assert constr_type in ["eq", "ineq"]
    t = type(fun)
    if not callable(fun):
        raise TypeError(f"Constraint function has to be callable, not {t}.")
    constr_dict = {'type': constr_type, 'fun': fun}
    if jac is not None:
        t = type(jac)
        if not callable(jac):
            raise TypeError(f"Jacobian has to be callable, not {t}.")
        constr_dict['jac'] = jac
    if args is not None:
        assert isinstance(args, Sequence)
        constr_dict['args'] = args
    return constr_dict


def scipy_constraints_to_callables(constr_dicts, hypo_maker):
    """Convert constraints expressions in terms of ParamSets
    into callables for scipy. Overwrites "fun" entries
    in `constr_dicts`.
    """
    def constr_func(x, constr_func_params):
        hypo_maker._set_rescaled_free_params(x) # pylint: disable=protected-access
        if hypo_maker.__class__.__name__ == "Detectors":
            # updates values for ALL detectors
            update_param_values_detector(hypo_maker, hypo_maker.params.free)
        return constr_func_params(hypo_maker.params)
    logging.warning(EVAL_MSG)
    assert isinstance(constr_dicts, Sequence)
    for cd in constr_dicts:
        assert isinstance(cd, Mapping)
        # the equality constraint is specified as a function that takes a
        # ParamSet as its input
        assert "fun" in cd
        constr = cd["fun"]
        logging.debug("adding scipy constraint: %s", constr)
        if callable(constr):
            constr_func_params = constr
        else:
            constr_func_params = eval(constr)
            t = type(constr_func_params)
            if not callable(constr_func_params):
                raise TypeError(f"Evaluated object not a callable, but {t}.")
        # overwrite
        cd["fun"] = partial(constr_func, constr_func_params=constr_func_params)
