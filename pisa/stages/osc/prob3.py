"""
Service for the calculation of three-flavour oscillation probabilities,
allowing for various non-standard effects.

"""

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE, ureg
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.stages.osc.nsi_params import StdNSIParams, VacuumLikeNSIParams
from pisa.stages.osc.osc_params import OscParams
from pisa.stages.osc.decay_params import DecayParams
from pisa.stages.osc.lri_params import LRIParams
from pisa.stages.osc.scaling_params import (
    Mass_scaling, Core_scaling_w_constrain, Core_scaling_wo_constrain,
    FIVE_LAYER_RADII, FIVE_LAYER_RHOS, TOMOGRAPHY_ERROR_MSG
)
from pisa.stages.osc.layers import Layers
from pisa.stages.osc.prob3numba.numba_osc_hostfuncs import propagate_array, fill_probs
from pisa.utils.resources import find_resource

__all__ = ['prob3', 'init_test',
           'LRI_TYPES', 'NSI_TYPES', 'TOMOGRAPHY_TYPES']

LRI_TYPES = ['emu-symmetry', 'etau-symmetry', 'mutau-symmetry']

NSI_TYPES = ['standard', 'vacuum-like']

TOMOGRAPHY_TYPES = ['mass_of_earth', 'mass_of_core_w_constrain', 'mass_of_core_wo_constrain']


class prob3(Stage):  # pylint: disable=invalid-name
    r"""
    Extended Prob3-like oscillations class.

    Expected container keys are:
            "true_energy",
            "true_coszen",
            "nubar",
            "flav",
            "nu_flux",
            "weights"

    Parameters
    ----------

    include_nlo : bool (default: `False`)
        Whether to include a +2.0% NLO correction to the SM CC matter potential,
        as per https://inspirehep.net/literature/2914951 (PRD111(2025)11).

    nsi_type : str or `None` (default: `None`)
        Choice of propagation/NC NSI parameterization.
        If string, either 'standard' or 'vacuum-like'
        (see e.g. https://inspirehep.net/literature/1672932 (JHEP08(2018)180).
        Parameters of the 'standard' parameterization are:
        `eps_ee`, `eps_mumu`, `eps_tautau`, `eps_emu_magn`, `eps_emu_phase`,
        `eps_etau_magn`, `eps_etau_phase`, `eps_mutau_magn`, `eps_mutau_phase`.
        Parameters of the 'vacuum-like' one are:
        `eps_scale`, `eps_prime`, `phi12`, `phi13`, `phi23`, `alpha1`, `alpha2`,
        `deltansi`.

    reparam_mix_matrix : bool (default: `False`)
        Whether to rephase the parameterization of the leptonic mixing matrix
        from its PDG default by :math:`\mathrm{diag}(e^{i\delta_\mathrm{CP}}, 1, 1)`,
        as motivated in https://inspirehep.net/literature/1672932 (JHEP08(2018)180).
        In the *absence* of NSI, this has no observable impact on oscillation
        probabilities, but results in the CPT transformation being realised
        by the transformations :math:`\Delta m^2_{31} \to -\Delta m^2_{32},
        \theta_{12} \to \pi/2 - \theta_{12},
        \delta_\mathrm{CP} \to \pi - \delta_\mathrm{CP}`,
        which is accommodated by the parameters' usual ranges. These
        transformations are then part of the generalized mass ordering
        degeneracy in the presence of NSI (see same reference above).

    neutrino_decay : bool (default: `False`)
        Whether to invoke neutrino decay with oscillations. The model
        implemented assumes the (invisible) decay of the third mass eigenstate
        into a singlet that is lighter than all three active mass eigenstates.
        The decay is parameterized in a model-independent way via the parameter
        :math:`\alpha_3 = m_3/\tau_3`, where :math:`m_3` is the mass and
        :math:`\tau_3` the lifetime in the rest frame. As a result, the vacuum
        Hamiltonian in the flavor basis then reads
        :math:`U \mathrm{diag}(0, \Delta m^2_{21}, \Delta m^2_{31} - i \alpha_3) U^\dagger/(2E)`,
        which deviates from the standard expression by the imaginary subtrahend in the
        last entry. See e.g. https://inspirehep.net/literature/2870979
        (JHEP04(2025)105) for such an analysis in the literature.
        The parameter :math:`\alpha_3` is implemented by `decay_alpha3`.

    tomography_type : str or `None` (default: `None`)
        Whether to allow for certain Earth matter density variations.
        If string, either 'mass_of_earth', 'mass_of_core_w_constrain',
        or 'mass_of_core_wo_constrain'.
        In case of 'mass_of_earth', expects the single parameter
        `density_scale`, which acts as an overall density scaling factor of
        the assumed density profile.
        In case of 'mass_of_core_w_constrain', expects the single parameter
        `core_density_scale`, which acts as a density scaling factor for the
        inner and outer core of the Earth in a pre-determined 5-layer density model,
        conserving the total mass and moment of inertia of the Earth by
        accordingly rescaling the densities of the "inner" and "middle" (but
        not the "outer") mantle.
        In case of 'mass_of_core_wo_constrain', expects the three parameters
        `core_density_scale`, `innermantle_density_scale`,
        `middlemantle_density_scale`. In this case, the corresponding densities
        in the 5-layer model are independently scaled, in contrast to the choice
        above.
        See e.g. https://inspirehep.net/literature/3072379
        (Eur.Phys.J.ST234(2025)16,5055-5064) for example analyses.

    lri_type : str or `None` (default: `None`)
        Choice of model/parameterization of long-range interactions (LRI).
        If string, either 'emu-symmetry', 'etau-symmetry', or 'mutau-symmetry'.
        Implemented is the single parameter `v_lri`.
        In the case of :math:`e`-:math:`\mu` symmetry, it is added to the standard
        :math:`ee` matter potential entry and subtracted from the :math:`\mu\mu` one.
        In the case of :math:`e`-:math:`\tau` symmetry, it is added to the standard
        :math:`ee` matter potential entry and subtracted from the :math:`\tau\tau` one.
        In the case of :math:`\mu`-:math:`\tau` symmetry, it is added to the standard
        :math:`\mu\mu` matter potential entry and subtracted from the :math:`\tau\tau`
        one. See e.g. https://inspirehep.net/literature/2658147 (JHEP 08(2023)101)
        for such an analysis in the literature.

    params
        expected params are .. ::

            detector_depth : float
            earth_model : PREM file path
            prop_height : quantity (dimensionless)
            YeI : quantity (dimensionless)
            YeO : quantity (dimensionless)
            YeM : quantity (dimensionless)
            density_scale : quantity (dimensionless)
            core_density_scale : quantity (dimensionless)
            innermantle_density_scale : quantity (dimensionless)
            middlemantle_density_scale : quantity (dimensionless)
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
            decay_alpha3 : quantity (energy^2)
            v_lri : quantity (eV)

    **std_kwargs
        Other kwargs are handled by Stage

    """

    def __init__(
      self,
      include_nlo=False,
      nsi_type=None,
      reparam_mix_matrix=False,
      neutrino_decay=False,
      tomography_type=None,
      lri_type=None,
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

        expected_container_keys = (
            'true_energy',
            'true_coszen',
            'nubar',
            'flav',
            'nu_flux',
            'weights'
        )

        self.include_nlo = include_nlo
        """Whether to include a +2.0% NLO correction to the
        SM CC matter potential."""

        # Check whether and if so with which NSI parameters we are to work.
        if nsi_type is not None:
            choices = NSI_TYPES
            nsi_type = nsi_type.strip().lower()
            if not nsi_type in choices:
                raise ValueError(
                    f'Chosen NSI type "{nsi_type}" not available!'
                    f' Choose one of {choices}.'
                )
        self.nsi_type = nsi_type
        """Type of NSI to assume."""

        self.reparam_mix_matrix = reparam_mix_matrix
        r"""Use a PMNS mixing matrix parameterisation that differs from
        the standard one by an overall phase matrix
        :math:`\mathrm{diag}(e^{i\delta_\mathrm{CP}}, 1, 1)`. This has no
        impact on oscillation probabilities in the *absence* of NSI."""

        self.neutrino_decay = neutrino_decay
        """Invoke neutrino decay with neutrino oscillation."""

        if neutrino_decay:
            self.decay_flag = 1
        else:
            self.decay_flag = -1

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

        if self.neutrino_decay:
            decay_params = ('decay_alpha3',)
        else:
            decay_params = ()

        if lri_type is not None:
            choices = LRI_TYPES
            lri_type = lri_type.strip().lower()
            if not lri_type in choices:
                raise ValueError(
                    f'Chosen LRI symmetry type "{lri_type}" not available!'
                    f' Choose one of {choices}.'
                )
        self.lri_type = lri_type
        """Type of LRI to assume."""

        if self.lri_type is None:
            lri_params = ()
        else:
            lri_params = ('v_lri',)


        if tomography_type is None:
            tomography_params = ()
        else:
            tomography_type = tomography_type.strip().lower()
            choices = TOMOGRAPHY_TYPES
            if not tomography_type in choices:
                raise ValueError(
                    f'Chosen tomography type "{tomography_type}" not available!'
                    f' Choose one of {choices}.'
                )
            if tomography_type == 'mass_of_earth':
                tomography_params = ('density_scale',)
            elif tomography_type == 'mass_of_core_w_constrain':
                tomography_params = ('core_density_scale',)
            elif tomography_type == 'mass_of_core_wo_constrain':
                tomography_params = ('core_density_scale',
                                     'innermantle_density_scale',
                                     'middlemantle_density_scale'
                )
        self.tomography_type = tomography_type
        """Type of Earth tomography to assume."""


        expected_params = (expected_params + nsi_params + decay_params
                           + lri_params + tomography_params)

        # init base class
        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )

        self.layers = None
        self.osc_params = None
        self.nsi_params = None
        self.tomography_params = None
        self.decay_params = None
        self.decay_matrix = None
        self.lri_params = None
        self.lri_pot = None
        # The interaction potential (Hamiltonian) just scales with the
        # electron density N_e for propagation through the Earth,
        # even(to very good approx.) in the presence of generalised interactions
        # (NSI), which is why we can simply treat it as a constant here.
        self.gen_mat_pot_matrix_complex = None
        """Interaction Hamiltonian without the factor sqrt(2)*G_F*N_e."""
        self.YeI = None # pylint: disable=invalid-name
        self.YeO = None # pylint: disable=invalid-name
        self.YeM = None # pylint: disable=invalid-name

    def setup_function(self):

        # object for oscillation parameters
        self.osc_params = OscParams()
        if self.reparam_mix_matrix:
            logging.debug(
                'Working with reparameterized version of mixing matrix.'
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


        if self.neutrino_decay:
            logging.debug('Working with neutrino decay')
            self.decay_params = DecayParams()

        if self.lri_type is not None:
            logging.debug('Working with LRI')
            self.lri_params = LRIParams()

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

        if self.tomography_type == "mass_of_earth":
            if not self.layers.using_earth_model:
                # already fail here instead of in compute upon attempt to scale
                # (to be on the safe side; not sure if this case can even happen)
                raise ValueError(
                    "You need to provide some Earth model, whose densities can"
                    " be rescaled!"
                )
            logging.debug('Working with tomography with a single density scaling factor.')
            self.tomography_params = Mass_scaling()
        elif self.tomography_type is not None:
            # not elegant but safe: external Earth model must correspond to internal hard-coded one
            if not self.layers.using_earth_model:
                raise ValueError(TOMOGRAPHY_ERROR_MSG)
            radii_ext = self.layers.radii[::-1][:-1]
            rhos_ext = self.layers.rhos_unweighted[::-1][:-1]
            if not (len(radii_ext) == len(FIVE_LAYER_RADII) and len(rhos_ext) == len(FIVE_LAYER_RHOS)):
                raise ValueError(TOMOGRAPHY_ERROR_MSG)
            radii_equal = np.allclose(np.add(radii_ext, 1), np.add(FIVE_LAYER_RADII.m_as("km"), 1))
            rhos_equal = np.allclose(np.add(rhos_ext, 1), np.add(FIVE_LAYER_RHOS.m_as("g/cm**3"), 1))
            if not radii_equal or not rhos_equal:
                raise ValueError(TOMOGRAPHY_ERROR_MSG)
            if self.tomography_type == "mass_of_core_w_constrain":
                logging.debug('Working with tomography with different scaling for different layers.')
                self.tomography_params = Core_scaling_w_constrain()
            elif self.tomography_type == "mass_of_core_wo_constrain":
                logging.debug('Working with tomography without any external constraints.')
                self.tomography_params = Core_scaling_wo_constrain()

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

        logging.debug('matter potential:\n%s', self.gen_mat_pot_matrix_complex)
        logging.debug('decay matrix:\n%s', self.decay_matrix)

        propagate_array(self.osc_params.dm_matrix, # pylint: disable = unexpected-keyword-arg, no-value-for-parameter
                        mix_matrix,
                        self.gen_mat_pot_matrix_complex,
                        self.decay_flag,
                        self.decay_matrix,
                        self.lri_pot,
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
            self.YeI = YeI
            self.YeO = YeO
            self.YeM = YeM
            self.layers.setElecFrac(self.YeI, self.YeO, self.YeM)
            for container in self.data:
                self.layers.calcLayers(container['true_coszen'])
                container['densities'] = self.layers.density.reshape((container.size, self.layers.max_layers))
                container['distances'] = self.layers.distance.reshape((container.size, self.layers.max_layers))


        # some safety checks on units
        # trying to avoid issue of angles with no dimension being assumed to be radians
        # here we enforce the user must speficy a valid angle unit
        for angle_param in [self.params.theta12, self.params.theta13, self.params.theta23, self.params.deltacp]:
            if angle_param.value.units == ureg.dimensionless:
                raise ValueError(f"{angle_param.name} is dimensionless, but needs units rad or deg!")

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
        if self.neutrino_decay:
            self.decay_params.decay_alpha3 = self.params.decay_alpha3.value.m_as('eV**2')

        if self.lri_type is not None:
            self.lri_params.v_lri = self.params.v_lri.value.m_as('eV')
        if self.tomography_type is not None:
            if self.tomography_type == "mass_of_earth":
                self.tomography_params.density_scale = self.params.density_scale.value.m_as('dimensionless')
                self.layers.scaling(scaling_array=self.tomography_params.density_scale)
            elif self.tomography_type == "mass_of_core_w_constrain":
                self.tomography_params.core_density_scale = self.params.core_density_scale.value.m_as('dimensionless')
                self.layers.scaling(scaling_array=self.tomography_params.scaling_array)
            elif self.tomography_type == "mass_of_core_wo_constrain":
                self.tomography_params.core_density_scale = self.params.core_density_scale.value.m_as('dimensionless')
                self.tomography_params.innermantle_density_scale = self.params.innermantle_density_scale.value.m_as('dimensionless')
                self.tomography_params.middlemantle_density_scale = self.params.middlemantle_density_scale.value.m_as('dimensionless')
                self.layers.scaling(scaling_array=self.tomography_params.scaling_factor_array)
            self.layers.setElecFrac(self.YeI, self.YeO, self.YeM)
            for container in self.data:
                self.layers.calcLayers(container['true_coszen'])
                container['densities'] = self.layers.density.reshape((container.size, self.layers.max_layers))


        # now we can proceed to calculate the generalised matter potential matrix
        std_mat_pot_matrix = np.zeros((3, 3), dtype=FTYPE) + 1.j * np.zeros((3, 3), dtype=FTYPE)
        if not self.include_nlo:
            logging.debug("Proceeding with *tree-level* standard CC matter potential....")
            std_mat_pot_matrix[0, 0] += 1.0
        else:
            logging.debug("Proceeding with *NLO* standard CC matter potential....")
            std_mat_pot_matrix[0, 0] += 1.020

        # add effective nsi coupling matrix
        if self.nsi_type is not None:
            logging.debug('NSI matrix:\n%s', self.nsi_params.eps_matrix)
            self.gen_mat_pot_matrix_complex = (
                std_mat_pot_matrix + self.nsi_params.eps_matrix
            )
            logging.debug('Using generalised matter potential:\n%s',
                          self.gen_mat_pot_matrix_complex)
        else:
            self.gen_mat_pot_matrix_complex = std_mat_pot_matrix
            logging.debug('Using standard matter potential:\n%s',
                          self.gen_mat_pot_matrix_complex)

        if self.neutrino_decay:
            self.decay_matrix = self.decay_params.decay_matrix
            logging.debug('Decay matrix:\n%s', self.decay_params.decay_matrix)
        else:
            self.decay_matrix = np.zeros((3, 3), dtype=FTYPE) + 1.j * np.zeros((3, 3), dtype=FTYPE)

        self.lri_pot = np.zeros((3, 3), dtype=FTYPE)
        types_lri = ['emu-symmetry', 'etau-symmetry', 'etau-symmetry']
        if self.lri_type is not None:
            if self.lri_type == 'emu-symmetry':
                self.lri_pot = self.lri_params.potential_matrix_emu
            elif self.lri_type == 'etau-symmetry':
                self.lri_pot = self.lri_params.potential_matrix_etau
            elif self.lri_type == 'mutau-symmetry':
                self.lri_pot = self.lri_params.potential_matrix_mutau
            else:
                # TODO: this just repeats the logic from init with slightly different code!
                raise ValueError("Implemented symmetries are %s" % types_lri)


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


def init_test(**param_kwargs):
    """Initialisation example"""
    param_set = ParamSet([
        Param(name='detector_depth', value=10*ureg.km, **param_kwargs),
        Param(name='prop_height', value=18*ureg.km, **param_kwargs),
        Param(name='earth_model', value='osc/PREM_4layer.dat', **param_kwargs),
        Param(name='YeI', value=0.5, **param_kwargs),
        Param(name='YeO', value=0.5, **param_kwargs),
        Param(name='YeM', value=0.5, **param_kwargs),
        Param(name='theta12', value=33*ureg.degree, **param_kwargs),
        Param(name='theta13', value=8*ureg.degree, **param_kwargs),
        Param(name='theta23', value=50*ureg.degree, **param_kwargs),
        Param(name='deltam21', value=8e-5*ureg.eV**2, **param_kwargs),
        Param(name='deltam31', value=3e-3*ureg.eV**2, **param_kwargs),
        Param(name='deltacp', value=180*ureg.degree, **param_kwargs),
    ])
    return prob3(include_nlo=True, params=param_set)
