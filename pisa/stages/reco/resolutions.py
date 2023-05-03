# pylint: disable=not-callable

"""
Stage for resolution improvement studies

"""

from __future__ import absolute_import, print_function, division

from pisa.core.stage import Stage
from pisa.utils.log import logging


class resolutions(Stage):
    """
    stage to change the reconstructed information by a given amount
    This can be used to esimate the impact of improved recosntruction
    resolutions for instance

    Parameters
    ----------
    params
        Expected params .. ::

            energy_improvement : quantity (dimensionless)
               scale the reco error down by this fraction
            coszen_improvement : quantity (dimensionless)
                scale the reco error down by this fraction
            pid_improvement : quantity (dimensionless)
                applies a shift to the classification parameter

    """
    def __init__(
        self,
        **std_kwargs
    ):
        expected_params = (
            'energy_improvement',
            'coszen_improvement',
            'pid_improvement',
        )
        # init base class
        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )


    def setup_function(self):

        self.data.representation = self.apply_mode

        #TODO Need to enforce boundaries, e.g. coszen within [-1, +1], E>0, PID within range (e.g. [0,1] for classifier), etc

        for container in self.data:
            logging.info('Changing energy resolutions')
            container['reco_energy'] += (container['true_energy'] - container['reco_energy']) * self.params.energy_improvement.m_as('dimensionless')
            container.mark_changed('reco_energy')

            logging.info('Changing coszen resolutions')
            container['reco_coszen'] += (container['true_coszen'] - container['reco_coszen']) * self.params.coszen_improvement.m_as('dimensionless')
            container.mark_changed('reco_coszen')

            logging.info('Changing PID resolutions')
            if container.name in ['numu_cc', 'numubar_cc']:
                container['pid'] += self.params.pid_improvement.m_as('dimensionless')
            else:
                container['pid'] -= self.params.pid_improvement.m_as('dimensionless')
            container.mark_changed('pid')
