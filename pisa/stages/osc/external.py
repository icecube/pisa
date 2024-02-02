"""
PISA pi stage for the calculation of earth layers and osc. probabilities

Maybe it would amke sense to split this up into a separate earth layer stage
and an osc. stage....todo

"""

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE, TARGET, ureg
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.stages.osc.layers import Layers
from pisa.utils.resources import find_resource

class external(Stage):
    """
    Use an external function to calculate oscillation probabilities

    Parameters
    ----------
    params

        osc_prob : callable
            the external function

        Expected params .. ::

            detector_depth : float
            earth_model : PREM file path
            prop_height : quantity (dimensionless)
            YeI : quantity (dimensionless)
            YeO : quantity (dimensionless)
            YeM : quantity (dimensionless)
    **kwargs
        Other kwargs are handled by Stage
    -----

    """
  
    def __init__(
      self,
      **std_kwargs,
    ):

        expected_params = (
          'detector_depth',
          'earth_model',
          'prop_height',
          'YeI',
          'YeO',
          'YeM',
        )
      

        # init base class
        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

        self.osc_prob = None
        self.external_params = None
        self.layers = None
        self.YeI = None
        self.YeO = None
        self.YeM = None


    def setup_function(self):

        # setup the layers
        #if self.params.earth_model.value is not None:
        earth_model = find_resource(self.params.earth_model.value)
        self.YeI = self.params.YeI.value.m_as('dimensionless')
        self.YeO = self.params.YeO.value.m_as('dimensionless')
        self.YeM = self.params.YeM.value.m_as('dimensionless')
        prop_height = self.params.prop_height.value.m_as('km')
        detector_depth = self.params.detector_depth.value.m_as('km')
        self.layers = Layers(earth_model, detector_depth, prop_height)
        self.layers.setElecFrac(self.YeI, self.YeO, self.YeM)
     

        # --- calculate the layers ---
        if self.is_map:
            # speed up calculation by adding links
            # as layers don't care about flavour
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc',
                                             'nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                             'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])

        for container in self.data:
            self.layers.calcLayers(container['true_coszen'])
            container['densities'] = self.layers.density.reshape((container.size, self.layers.max_layers))
            container['densities_neutron_weighted'] = self.layers.density_neutron_weighted.reshape((container.size, self.layers.max_layers))
            container['distances'] = self.layers.distance.reshape((container.size, self.layers.max_layers))

        # don't forget to un-link everything again
        self.data.unlink_containers()

        # --- setup empty arrays ---
        if self.is_map:
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc'])
            self.data.link_containers('nubar', ['nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                                'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])
        for container in self.data:
            container['probability'] = np.empty((container.size, 3, 3), dtype=FTYPE)
        self.data.unlink_containers()

        # setup more empty arrays
        for container in self.data:
            container['prob_e'] = np.empty((container.size), dtype=FTYPE)
            container['prob_mu'] = np.empty((container.size), dtype=FTYPE)

    def compute_function(self):


        assert self.is_map

        if self.is_map:
            # speed up calculation by adding links
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc'])
            self.data.link_containers('nubar', ['nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                                'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])

        # this can be done in a more clever way (don't have to recalculate all paths)
        YeI = self.params.YeI.value.m_as('dimensionless')
        YeO = self.params.YeO.value.m_as('dimensionless')
        YeM = self.params.YeM.value.m_as('dimensionless')
        if YeI != self.YeI or YeO != self.YeO or YeM != self.YeM:
            self.YeI = YeI; self.YeO = YeO; self.YeM = YeM
            self.layers.setElecFrac(self.YeI, self.YeO, self.YeM)
            for container in self.data:
                self.layers.calcLayers(container['true_coszen'])
                container['densities'] = self.layers.density.reshape((container.size, self.layers.max_layers))
                container['densities_neutron_weighted'] = self.layers.density_neutron_weighted.reshape((container.size, self.layers.max_layers))
                container['distances'] = self.layers.distance.reshape((container.size, self.layers.max_layers))

        for container in self.data:
            energy_idx = self.data.representation.names.index('true_energy')

            energies = self.data.representation.dims[energy_idx].weighted_centers.m
            distances = container['distances'].reshape(*self.data.representation.shape, -1)
            densities = container['densities'].reshape(*self.data.representation.shape, -1)
            densities_neutron_weighted = container['densities_neutron_weighted'].reshape(*self.data.representation.shape, -1)
            if energy_idx == 0:
                distances = distances[0, :]
                densities = densities[0, :]
                densities_neutron_weighted = densities_neutron_weighted[0, :]
            else:
                distances = distances[:, 0]
                densities = densities[:, 0]
                densities_neutron_weighted = densities_neutron_weighted[:, 0]

            if container['nubar'] == 1:
                is_anti = False
            elif container['nubar'] == -1:
                is_anti = True

            p = self.osc_prob(energies, distances, self.external_params, is_anti, densities, densities_neutron_weighted)

            if energy_idx == 0:
                container['probability'] = p[:, :, :3, :3].reshape(-1, 3, 3)
            else:
                container['probability'] = np.swapaxes(p[:, :, :3, :3], 0, 1).reshape(-1, 3, 3)

            container.mark_changed('probability')

        # the following is flavour specific, hence unlink
        self.data.unlink_containers()

        for container in self.data:
            container['prob_e'] = container['probability'][:, 0, container['flav']]
            container['prob_mu'] = container['probability'][:, 1, container['flav']]
            container.mark_changed('prob_e')
            container.mark_changed('prob_mu')

    def apply_function(self):

        # update the outputted weights
        for container in self.data:
            container['weights'] *= (container['nu_flux'][:,0] * container['prob_e']) + (container['nu_flux'][:,1] * container['prob_mu'])

