"""
Stage for resolution improvement studies

"""

from __future__ import absolute_import, print_function, division
import numpy as np

from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.log import logging

__all__ = ['resolutions', 'init_test']


class resolutions(Stage):  # pylint: disable=invalid-name
    """
    Stage to change the reconstructed information by a given amount.
    This can be used to estimate the impact of improved reconstruction
    resolutions for instance.

    Parameters
    ----------
    params : ParamSet
        Must have parameters::

            energy_improvement : quantity (dimensionless)
               scale the reco error down by this fraction
            coszen_improvement : quantity (dimensionless)
                scale the reco error down by this fraction
            pid_improvement : quantity (dimensionless)
                applies a shift to the classification parameter [if relative_pid=False]
                scales the pid error down by this fraction [if relative_pid=True]

    Notes
    -----

    Expected container keys are::

        "true_energy", "true_coszen", "reco_energy", "reco_coszen", "pid"
    """

    def __init__(
        self,
        relative_pid=False,
        **std_kwargs
    ):
        # FIXME: not meant to be expected_params, as only setup_function implemented
        expected_params = (
            'energy_improvement',
            'coszen_improvement',
            'pid_improvement',
        )
        expected_container_keys = (
            'true_energy',
            'true_coszen',
            'reco_energy',
            'reco_coszen',
            'pid',
        )
        # Implements no apply_function
        supported_reps = {
            "calc_mode": "events",
            "apply_mode": None
        }
        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            supported_reps=supported_reps,
            **std_kwargs,
        )

        self.relative_pid = relative_pid

    def setup_function(self):

        for container in self.data:
            logging.info('Changing energy resolutions')
            container['reco_energy'] += (container['true_energy'] - container['reco_energy']) * self.params.energy_improvement.m_as('dimensionless')
            container.mark_changed('reco_energy')

            logging.info('Changing coszen resolutions')
            container['reco_coszen'] += (container['true_coszen'] - container['reco_coszen']) * self.params.coszen_improvement.m_as('dimensionless')
            container['reco_coszen'] = np.clip(container['reco_coszen'], -1, 1)
            container.mark_changed('reco_coszen')

            logging.info('Changing PID resolutions')
            if container.name in ['numu_cc', 'numubar_cc']:
                if self.relative_pid:
                    container['pid'] += (1 - container['pid']) * self.params.pid_improvement.m_as('dimensionless')
                else:
                    container['pid'] += self.params.pid_improvement.m_as('dimensionless')
            else:
                if self.relative_pid:
                    container['pid'] += (0 - container['pid']) * self.params.pid_improvement.m_as('dimensionless')
                else:
                    container['pid'] -= self.params.pid_improvement.m_as('dimensionless')
            container.mark_changed('pid')


def init_test(**param_kwargs):
    """Instantiation example"""
    param_set = ParamSet([
        Param(name='energy_improvement', value=0.9, **param_kwargs),
        Param(name='coszen_improvement', value=0.5, **param_kwargs),
        Param(name='pid_improvement', value=0.02, **param_kwargs)
    ])
    return resolutions(params=param_set)
