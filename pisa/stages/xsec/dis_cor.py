"""
Stage to apply simple parameterized DIS corrections. 
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.profiler import profile

__all__ = ['dis_correction', 'BY_correction', 'dis_cor', 'init_test']

def dis_correction(x, flav):
    if 'bar' in flav:
        return np.nan_to_num(0.73*x**(-0.082))
    else:
        return np.nan_to_num(0.82*x**(-0.049))

def BY_correction(x, y, flav):
    if 'bar' in flav:
        return np.nan_to_num(0.6249*x**(-0.1231)*y**(-0.1389))
    else:
        return np.nan_to_num(0.8507*x**(-0.0559)*y**(52.1e-3))


class dis_cor(Stage): # pylint: disable=invalid-name
    """
    Stage to apply simple DIS cross-section corrections based on NuTeV data.
        https://doi.org/10.1103/PhysRevD.74.012008
    A simple polynomial using Bjorken-x is used to adjust the event weights.
    This is based on the work of the LEGS group.
        https://wiki.icecube.wisc.edu/index.php/Legs

    The stage can also apply changes to the Bodek-Yang (BY) correction.
        https://doi.org/10.1063/1.1594324
    Similarly, a polynomial using Bjorken-x and -y is used to adjust the event 
    weights. Note that this polynomial cannot fully account for the difference 
    between BY being on and off.

    Parameters
    ----------

    params : ParamSet or sequence with which to instantiate a ParamSet.
        Expected params are: .. ::

            DIS_cor_factor : quantity (dimensionless)
                Scales between default (GENIE) weight and the correction.
                Sould be between 0 and 1.
            BY_cor_factor : quantity (dimensionless)
                Scales between default (GENIE) weight and the correction.
                Sould be between 0 and 1.
                
        Expected container keys are .. ::

            "weights"
            "bjorken_x"
            "bjorken_y"
            "dis"

    """
    def __init__(
        self,
        **std_kwargs,
    ):
        expected_params = (
            'DIS_cor_factor',
            'BY_cor_factor',
        )

        expected_container_keys = (
            'bjorken_x',
            'bjorken_y',
            'weights',
            'dis',
        )

        # init base class
        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )

    @profile
    def apply_function(self):
        DIS_cor_factor = self.params.DIS_cor_factor.value.m
        BY_cor_factor = self.params.BY_cor_factor.value.m

        assert 0 <= DIS_cor_factor and DIS_cor_factor <= 1, "Don't extrapolate DIS correction"
        assert 0 <= BY_cor_factor and BY_cor_factor <= 1, "Don't extrapolate BY correction"

        for container in self.data:
            dis = container['dis'].astype(int)
            if DIS_cor_factor > 0:
                corr1 = dis_correction(container['bjorken_x'], container.name)
                corr1 = 1.0 + DIS_cor_factor * (corr1 - 1.0)
            else:
                corr1 = 1.0
            if BY_cor_factor > 0:
                corr2 = BY_correction(container['bjorken_x'], container['bjorken_y'], container.name)
                corr2 = 1.0 + BY_cor_factor * (corr2 - 1.0)
            else:
                corr2 = 1.0
            corr = np.where(dis == 1, corr1 * corr2, 1.0)

            container['weights'] *= corr
            container.mark_changed('weights')


def init_test(**param_kwargs):
    """Instantiation example"""
    param_set = ParamSet([
        Param(name="DIS_cor_factor", value=1, **param_kwargs),
        Param(name="BY_cor_factor", value=1, **param_kwargs),
    ])

    return dis_cor(params=param_set)
