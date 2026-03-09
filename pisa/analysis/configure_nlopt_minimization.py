"""
Utilities for validating and configuring nlopt minimization.
"""

from functools import partial

from pisa.analysis.manipulate_params import update_param_values_detector
from pisa.utils.fileio import EVAL_MSG
from pisa.utils.log import logging

__all__ = ['get_nlopt_inequality_constraint_funcs']


def get_nlopt_inequality_constraint_funcs(method_kwargs, hypo_maker):
    """Convert constraints expressions in terms of ParamSets
    into callables for nlopt. Evals expression(s) from `method_kwargs`
    and returns list of callables.
    """
    def ineq_func(x, grad, ineq_func_params):
        if grad.size > 0:
            raise RuntimeError("gradients not supported")
        hypo_maker._set_rescaled_free_params(x) # pylint: disable=protected-access
        if hypo_maker.__class__.__name__ == "Detectors":
            # updates values for ALL detectors
            update_param_values_detector(hypo_maker, hypo_maker.params.free)
        # In NLOPT, the inequality function must stay negative, while in
        # scipy, the inequality function must stay positive. We keep with
        # the scipy convention by flipping the sign.
        return -ineq_func_params(hypo_maker.params)
    assert "ineq_constraints" in method_kwargs
    logging.warning(EVAL_MSG)
    constr_list = method_kwargs["ineq_constraints"]
    if isinstance(constr_list, str):
        constr_list = [constr_list]
    ineq_funcs = []
    for constr in constr_list:
        # the inequality function is specified as a function that takes a
        # ParamSet as its input
        logging.debug("adding nlopt constraint (must stay positive): %s", constr)
        if callable(constr):
            ineq_func_params = constr
        else:
            ineq_func_params = eval(constr)
            t = type(ineq_func_params)
            if not callable(ineq_func_params):
                raise TypeError(f"Evaluated object not a callable, but {t}.")
        ineq_funcs.append(partial(ineq_func, ineq_func_params=ineq_func_params))
    return ineq_funcs
