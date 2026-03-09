"""
Utilities for manipulating parameters when fitting.
"""

from copy import deepcopy

import numpy as np
from scipy._lib._util import check_random_state

from pisa import ureg
from pisa.core.param import Param, ParamSet
from pisa.core.pipeline import Pipeline

__all__ = ['get_separate_octant_params', 'update_param_values',
           'update_param_values_detector']


class BoundedRandomDisplacement():
    """
    Add a bounded random displacement of maximum size `stepsize` to each coordinate
    Calling this updates `x` in-place.

    Parameters
    ----------
    stepsize : float, optional
        Maximum stepsize in any dimension
    bounds : pair of float or sequence of pairs of float
        Bounds on x
    random_gen : {None, `np.random.RandomState`, `np.random.Generator`}
        The random number generator that generates the displacements
    """
    def __init__(self, stepsize=0.5, bounds=(0, 1), random_gen=None):
        self.stepsize = stepsize
        self.random_gen = check_random_state(random_gen)
        self.bounds = np.array(bounds).T

    def __call__(self, x):
        x += self.random_gen.uniform(-self.stepsize, self.stepsize,
                                     np.shape(x))
        x = np.clip(x, *self.bounds)  # bounds are automatically broadcast
        return x


def get_separate_octant_params(
    hypo_maker, angle_name, inflection_point, tolerance=None
):
    '''
    This function creates versions of the angle param that are confined to
    a single octant. It does this for both octant cases. This is used to allow
    fits to be done where only one of the octants is allowed. The fit can then
    be done for the two octant cases and compared to find the best fit.

    Parameters
    ----------
    hypo_maker : DistributionMaker or Detector
        The hypothesis maker being used by the fitter
    angle_name : string
        Name of the angle for which to create separate octant params.
    inflection_point : quantity
        Point distinguishing between the two octants, e.g. 45 degrees
    tolerance : quantity
        If the starting value is closer to the inflection point than the value of the
        tolerance, it is offset away from the inflection point by this amount.

    Returns
    -------
    angle_orig : Param
        angle param as it was before applying the octant separation
    angle_case1 : Param
        angle param confined to first octant
    angle_case2 : Param
        angle param confined to second octant
    '''

    # Reset angle before starting
    angle = hypo_maker.params[angle_name]
    angle.reset()

    # Store the original theta23 param before we mess with it
    # WARNING: Do not copy here, you want the original object (since this relates
    # to the underlying ParamSelector from which theta23 is extracted). Otherwise
    # end up with an incosistent state later (e.g. after a new call to
    # ParamSelector.select_params, this copied, and potentially modified param
    # will be overwtiten by the original).
    angle_orig = angle

    # Get the octant definition
    octants = (
        (angle.range[0], inflection_point) ,
        (inflection_point, angle.range[1])
        )

    # If angle is maximal (e.g. the transition between octants) or very close
    # to it, offset it slightly to be clearly in one octant (note that fit can
    # still move the value back to maximal). The reason for this is that
    # otherwise checks on the parameter bounds (which include a margin for
    # minimizer tolerance) can an throw exception.
    if tolerance is None:
        tolerance = 0.1 * ureg.degree
    dist_from_inflection = angle.value - inflection_point
    if np.abs(dist_from_inflection) < tolerance :
        # Note this creates +ve shift also for theta == 45 (arbitrary)
        sign = -1. if dist_from_inflection < 0. else +1.
        angle.value = inflection_point + (sign * tolerance)

    # Store the cases
    angle_case1 = deepcopy(angle)
    angle_case2 = deepcopy(angle)

    # Get case 1, e.g. the current octant
    case1_octant_index = 0 if angle_case1.value < inflection_point else 1
    angle_case1.range = octants[case1_octant_index]
    angle_case1.nominal_value = angle_case1.value

    # Also get case 2, e.g. the other octant
    case2_octant_index = 0 if case1_octant_index == 1 else 1
    angle_case2.value = 2*inflection_point - angle_case2.value
    # Set nominal value so that `reset_free` won't try to set it out of bounds
    angle_case2.nominal_value = angle_case2.value
    angle_case2.range = octants[case2_octant_index]

    return angle_orig, angle_case1, angle_case2


def update_param_values(
    hypo_maker,
    params,
    update_nominal_values=False,
    update_range=False,
    update_is_fixed=False
):
    """
    Update just the values of parameters of a DistributionMaker *without* replacing
    the memory references inside.

    This should be used in place of `hypo_maker.update_params(params)` unless one
    explicitly wants to replace the memory references to which the parameters in
    the DistributionMaker are pointing.
    """

    # it is possible that only a single param is given
    if isinstance(params, Param):
        params = [params]

    if isinstance(hypo_maker, Pipeline):
        hypo_maker = [hypo_maker]

    for p in params:
        for pipeline in hypo_maker:
            if p.name not in pipeline.params.names:
                continue
            # it is crucial that we update the range first because the value
            # of the parameter in params might lie outside the range of those in
            # hypo_maker.
            if update_range:
                pipeline.params[p.name].range = p.range
            pipeline.params[p.name].value = p.value
            if update_nominal_values:
                pipeline.params[p.name].nominal_value = p.nominal_value
            if update_is_fixed:
                pipeline.params[p.name].is_fixed = p.is_fixed


def update_param_values_detector(
    hypo_maker,
    params,
    update_nominal_values=False,
    update_range=False,
    update_is_fixed=False
):
    """
    Modification of the update_param_values function to use with the Detectors class.
    """
    assert hypo_maker.__class__.__name__ == "Detectors", \
    "hypo_maker is not Detectors class"

    if isinstance(params, Param):
        params = ParamSet(params)

    for distribution_maker in hypo_maker:
        ps = deepcopy(params)
        for p in ps.names:
            if distribution_maker.detector_name in p:
                p_name = p.replace('_' +distribution_maker.detector_name, "")
                if p_name in ps.names:
                    ps.remove(p_name)
                ps[p].name = p_name
        update_param_values(distribution_maker, ps,
                            update_nominal_values, update_range, update_is_fixed)
    hypo_maker.init_params()
