# pylint: disable=not-callable

"""
Stage for resolution improvement studies

"""

from __future__ import absolute_import, print_function, division

from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging


class resolutions(PiStage):
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
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        input_specs=None,
        calc_specs=None,
        output_specs=None,
    ):
        expected_params = (
            'energy_improvement',
            'coszen_improvement',
            'pid_improvement',
        )
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_calc_keys = ()

        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ()

        # what keys are added or altered for the outputs during apply
        output_apply_keys = ()

        # init base class
        super().__init__(
            data=data,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            input_specs=input_specs,
            calc_specs=calc_specs,
            output_specs=output_specs,
            input_calc_keys=input_calc_keys,
            output_calc_keys=output_calc_keys,
            output_apply_keys=output_apply_keys,
        )

        assert self.input_mode is not None
        assert self.calc_mode is None
        assert self.output_mode is not None
        assert self.input_mode == self.output_mode

    def setup_function(self):

        self.data.data_specs = self.output_specs

        for container in self.data:
            logging.info('Changing energy resolutions')
            tmp = container['reco_energy'].get('host')
            tmp += (container['true_energy'].get('host') - container['reco_energy'].get('host')) * self.params.energy_improvement.m_as('dimensionless')
            container['reco_energy'].mark_changed('host')

            logging.info('Changing coszen resolutions')
            tmp = container['reco_coszen'].get('host')
            tmp += (container['true_coszen'].get('host') - container['reco_coszen'].get('host')) * self.params.coszen_improvement.m_as('dimensionless')
            container['reco_coszen'].mark_changed('host')
            # make sure coszen is within -1/1 ?

            logging.info('Changing PID resolutions')
            tmp = container['pid'].get('host')
            if container.name in ['numu_cc', 'numubar_cc']:
                tmp += self.params.pid_improvement.m_as('dimensionless')
            else:
                tmp -= self.params.pid_improvement.m_as('dimensionless')
            container['pid'].mark_changed('host')
