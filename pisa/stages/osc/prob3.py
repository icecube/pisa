"""
PISA pi stage for the calculation of earth layers and osc. probabilities

Maybe it would amke sense to split this up into a separate earth layer stage
and an osc. stage....todo

"""

from __future__ import absolute_import, print_function, division

import numpy as np
from numba import guvectorize

from pisa import FTYPE, TARGET, ureg
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.stages.osc.nsi_params import StdNSIParams, VacuumLikeNSIParams
from pisa.stages.osc.osc_params import OscParams
from pisa.stages.osc.layers import Layers
from pisa.stages.osc.prob3numba.numba_osc_hostfuncs import propagate_array, fill_probs
from pisa.utils.numba_tools import WHERE
from pisa.utils.resources import find_resource


class prob3(Stage):
    """
    Prob3-like oscillation PISA Pi class

    Parameters
    ----------
    params
        Expected params .. ::

            detector_depth : float
            earth_model : PREM file path
            prop_height : quantity (dimensionless)
            YeI : quantity (dimensionless)
            YeO : quantity (dimensionless)
            YeM : quantity (dimensionless)
            theta12 : quantity (angle)
            theta13 : quantity (angle)
            theta23 : quantity (angle)
            deltam21 : quantity (mass^2)
            deltam31 : quantity (mass^2)
            deltacp : quantity (angle)
            eps_scale : quantity(dimensionless)
            eps_prime : quantity(dimensionless)
            phi12 : quantity(angle)
            phi13 : quantity(angle)
            phi23 : quantity(angle)
            alpha1 : quantity(angle)
            alpha2 : quantity(angle)
            deltansi : quantity(angle)
            eps_ee : quantity (dimensionless)
            eps_emu_magn : quantity (dimensionless)
            eps_emu_phase : quantity (angle)
            eps_etau_magn : quantity (dimensionless)
            eps_etau_phase : quantity (angle)
            eps_mumu : quantity(dimensionless)
            eps_mutau_magn : quantity (dimensionless)
            eps_mutau_phase : quantity (angle)
            eps_tautau : quantity (dimensionless)

    **kwargs
        Other kwargs are handled by Stage
    -----

    """
  
    def __init__(
      self,
      nsi_type=None,
      reparam_mix_matrix=False,
      **std_kwargs,
    ):

        expected_params = (
          'detector_depth',
          'earth_model',
          'prop_height',
          'YeI',
          'YeO',
          'YeM',
          'theta12',
          'theta13',
          'theta23',
          'deltam21',
          'deltam31',
          'deltacp'
        )
      
        # Check whether and if so with which NSI parameters we are to work.
        if nsi_type is not None:
            choices = ['standard', 'vacuum-like']
            nsi_type = nsi_type.strip().lower()
            if not nsi_type in choices:
                raise ValueError(
                    'Chosen NSI type "%s" not available! Choose one of %s.'
                    % (nsi_type, choices)
                )
        self.nsi_type = nsi_type
        """Type of NSI to assume."""

        self.reparam_mix_matrix = reparam_mix_matrix
        """Use a PMNS mixing matrix parameterisation that differs from
           the standard one by an overall phase matrix
           diag(e^(i*delta_CP), 1, 1). This has no impact on
           oscillation probabilities in the *absence* of NSI."""

        if self.nsi_type is None:
            nsi_params = ()
        elif self.nsi_type == 'vacuum-like':
            nsi_params = ('eps_scale',
                          'eps_prime',
                          'phi12',
                          'phi13',
                          'phi23',
                          'alpha1',
                          'alpha2',
                          'deltansi'
            )
        elif self.nsi_type == 'standard':
            nsi_params = ('eps_ee',
                          'eps_emu_magn',
                          'eps_emu_phase',
                          'eps_etau_magn',
                          'eps_etau_phase',
                          'eps_mumu',
                          'eps_mutau_magn',
                          'eps_mutau_phase',
                          'eps_tautau'
            )
        expected_params = expected_params + nsi_params

        # init base class
        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )


        self.layers = None
        self.osc_params = None
        self.nsi_params = None
        # Note that the interaction potential (Hamiltonian) just scales with the
        # electron density N_e for propagation through the Earth,
        # even(to very good approx.) in the presence of generalised interactions
        # (NSI), which is why we can simply treat it as a constant here.
        self.gen_mat_pot_matrix_complex = None
        """Interaction Hamiltonian without the factor sqrt(2)*G_F*N_e."""
        self.YeI = None
        self.YeO = None
        self.YeM = None

    def setup_function(self):

        # object for oscillation parameters
        self.osc_params = OscParams()
        if self.reparam_mix_matrix:
            logging.debug(
                'Working with reparameterizated version of mixing matrix.'
            )
        else:
            logging.debug(
                'Working with standard parameterization of mixing matrix.'
            )
        if self.nsi_type == 'vacuum-like':
            logging.debug('Working in vacuum-like NSI parameterization.')
            self.nsi_params = VacuumLikeNSIParams()
        elif self.nsi_type == 'standard':
            logging.debug('Working in standard NSI parameterization.')
            self.nsi_params = StdNSIParams()

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

    def calc_probs(self, nubar, e_array, rho_array, len_array, out):
        ''' wrapper to execute osc. calc '''
        if self.reparam_mix_matrix:
            mix_matrix = self.osc_params.mix_matrix_reparam_complex
        else:
            mix_matrix = self.osc_params.mix_matrix_complex
        propagate_array(self.osc_params.dm_matrix, # pylint: disable = unexpected-keyword-arg, no-value-for-parameter
                        mix_matrix,
                        self.gen_mat_pot_matrix_complex,
                        nubar,
                        e_array,
                        rho_array,
                        len_array,
                        out=out
                       )

    def compute_function(self):

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
                container['distances'] = self.layers.distance.reshape((container.size, self.layers.max_layers))

        # some safety checks on units
        # trying to avoid issue of angles with no dimension being assumed to be radians
        # here we enforce the user must speficy a valid angle unit
        for angle_param in [self.params.theta12, self.params.theta13, self.params.theta23, self.params.deltacp] :
            assert angle_param.value.units != ureg.dimensionless, "Param %s is dimensionless, but should have angle units [rad, degree]" % angle_param.name

        # --- update mixing params ---
        self.osc_params.theta12 = self.params.theta12.value.m_as('rad')
        self.osc_params.theta13 = self.params.theta13.value.m_as('rad')
        self.osc_params.theta23 = self.params.theta23.value.m_as('rad')
        self.osc_params.dm21 = self.params.deltam21.value.m_as('eV**2')
        self.osc_params.dm31 = self.params.deltam31.value.m_as('eV**2')
        self.osc_params.deltacp = self.params.deltacp.value.m_as('rad')
        if self.nsi_type == 'vacuum-like':
            self.nsi_params.eps_scale = self.params.eps_scale.value.m_as('dimensionless')
            self.nsi_params.eps_prime = self.params.eps_prime.value.m_as('dimensionless')
            self.nsi_params.phi12 = self.params.phi12.value.m_as('rad')
            self.nsi_params.phi13 = self.params.phi13.value.m_as('rad')
            self.nsi_params.phi23 = self.params.phi23.value.m_as('rad')
            self.nsi_params.alpha1 = self.params.alpha1.value.m_as('rad')
            self.nsi_params.alpha2 = self.params.alpha2.value.m_as('rad')
            self.nsi_params.deltansi = self.params.deltansi.value.m_as('rad')
        elif self.nsi_type == 'standard':
            self.nsi_params.eps_ee = self.params.eps_ee.value.m_as('dimensionless')
            self.nsi_params.eps_emu = (
                (self.params.eps_emu_magn.value.m_as('dimensionless'),
                self.params.eps_emu_phase.value.m_as('rad'))
            )
            self.nsi_params.eps_etau = (
                (self.params.eps_etau_magn.value.m_as('dimensionless'),
                self.params.eps_etau_phase.value.m_as('rad'))
            )
            self.nsi_params.eps_mumu = self.params.eps_mumu.value.m_as('dimensionless')
            self.nsi_params.eps_mutau = (
                (self.params.eps_mutau_magn.value.m_as('dimensionless'),
                self.params.eps_mutau_phase.value.m_as('rad'))
            )
            self.nsi_params.eps_tautau = self.params.eps_tautau.value.m_as('dimensionless')

        # now we can proceed to calculate the generalised matter potential matrix
        std_mat_pot_matrix = np.zeros((3, 3), dtype=FTYPE) + 1.j * np.zeros((3, 3), dtype=FTYPE)
        std_mat_pot_matrix[0, 0] += 1.0

        # add effective nsi coupling matrix
        if self.nsi_type is not None:
            logging.debug('NSI matrix:\n%s' % self.nsi_params.eps_matrix)
            self.gen_mat_pot_matrix_complex = (
                std_mat_pot_matrix + self.nsi_params.eps_matrix
            )
            logging.debug('Using generalised matter potential:\n%s'
                          % self.gen_mat_pot_matrix_complex)
        else:
            self.gen_mat_pot_matrix_complex = std_mat_pot_matrix
            logging.debug('Using standard matter potential:\n%s'
                          % self.gen_mat_pot_matrix_complex)

        for container in self.data:
            self.calc_probs(container['nubar'],
                            container['true_energy'],
                            container['densities'],
                            container['distances'],
                            out=container['probability'],
                           )
            container.mark_changed('probability')

        # the following is flavour specific, hence unlink
        self.data.unlink_containers()

        for container in self.data:
            # initial electrons (0)
            fill_probs(container['probability'],
                       0,
                       container['flav'],
                       out=container['prob_e'],
                      )
            # initial muons (1)
            fill_probs(container['probability'],
                       1,
                       container['flav'],
                       out=container['prob_mu'],
                      )

            container.mark_changed('prob_e')
            container.mark_changed('prob_mu')


    def apply_function(self):

        # maybe speed up like this?
        #self.data.representation = self.calc_mode
        #for container in self.data:
        #    container['oscillated_flux'] = (container['nu_flux'][:,0] * container['prob_e']) + (container['nu_flux'][:,1] * container['prob_mu'])

        #self.data.representation = self.apply_mode

        # update the outputted weights
        for container in self.data:
            container['weights'] *= (container['nu_flux'][:,0] * container['prob_e']) + (container['nu_flux'][:,1] * container['prob_mu'])

