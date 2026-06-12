"""
PISA stage to apply effective area weights and the energy scale systematic as defined by ORCA
"""

from __future__ import absolute_import, print_function, division
import numpy as np

from pisa import ureg
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.profiler import profile


class orca_scale(Stage):  # pylint: disable=invalid-name
    """
    PISA stage to apply scaling systematics as defined by ORCA.

    Parameters
    ----------
    params
        Expected params are .. ::

            hpt_norm : dimensionless Quantity
            shower_norm : dimensionless Quantity
            energy_scale : dimensionless Quantity

        Expected container keys are .. ::

            "weights"
            "reco_energy"


    """
    def __init__(
        self,
        **std_kwargs,
    ):
        expected_params = (
            'hpt_norm',
            'shower_norm',
            'energy_scale',
        )

        expected_container_keys = (
            'weights',
            'reco_energy',
        )

        # init base class
        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )


    @profile
    def apply_function(self):
        hpt_norm = self.params.hpt_norm.m_as('dimensionless')
        shower_norm = self.params.shower_norm.m_as('dimensionless')
        energy_scale = self.params.energy_scale.m_as('dimensionless')

        ob = self.data['output_binning']
        e_bins = ob.bin_edges[ob.names.index('reco_energy')].m
        e_width = np.diff(np.log10(e_bins))
        rel_loss = abs(np.log10(energy_scale)) / e_width
        assert np.all(rel_loss < 1)

        for container in self.data:
            hist = container.get_hist('weights')[0]
            hist[:,:,0] *= hpt_norm
            hist[:,:,1] *= shower_norm

            keep = (1-rel_loss)[:, np.newaxis, np.newaxis] * hist 
            lose = rel_loss[:, np.newaxis, np.newaxis] * hist
            if energy_scale > 1:
                get = np.roll(lose, 1, axis=0)
                get[0,:,:] = 0
            else:
                get = np.roll(lose, -1, axis=0)
                get[-1,:,:] = 0
            container['weights'] = keep + get
