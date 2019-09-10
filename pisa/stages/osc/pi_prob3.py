"""
PISA pi stage for the calculation of earth layers and osc. probabilities

Maybe it would amke sense to split this up into a seperate earth layer stage
and an osc. stage....todo

"""
from __future__ import absolute_import, print_function, division

import numpy as np
from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.stages.osc.pi_osc_params import OscParams
from pisa.stages.osc.layers import Layers
from pisa.stages.osc.prob3numba.numba_osc import propagate_array, fill_probs
from pisa.utils.numba_tools import WHERE
from pisa.utils.resources import find_resource


class pi_prob3(PiStage):
    """
    prob3 osc PISA Pi class

    Paramaters
    ----------
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

    None

    Notes
    -----

    """
    def __init__(self,
                 data=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                ):

        expected_params = ('detector_depth',
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
                           'deltacp',
                          )

        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_apply_keys = ('weights',
                            'sys_flux',
                           )
        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ('prob_e',
                            'prob_mu',
                           )
        # what keys are added or altered for the outputs during apply
        output_apply_keys = ('weights',
                      )

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
            input_apply_keys=input_apply_keys,
            output_calc_keys=output_calc_keys,
            output_apply_keys=output_apply_keys,
        )

        assert self.input_mode is not None
        assert self.calc_mode is not None
        assert self.output_mode is not None

        self.layers = None
        self.osc_params = None

    def setup_function(self):

        # object for oscillation parameters
        self.osc_params = OscParams()

        # setup the layers
        #if self.params.earth_model.value is not None:
        earth_model = find_resource(self.params.earth_model.value)
        YeI = self.params.YeI.value.m_as('dimensionless')
        YeO = self.params.YeO.value.m_as('dimensionless')
        YeM = self.params.YeM.value.m_as('dimensionless')
        prop_height = self.params.prop_height.value.m_as('km')
        detector_depth = self.params.detector_depth.value.m_as('km')
        self.layers = Layers(earth_model, detector_depth, prop_height)
        self.layers.setElecFrac(YeI, YeO, YeM)

        # set the correct data mode
        self.data.data_specs = self.calc_specs

        # --- calculate the layers ---
        if self.calc_mode == 'binned':
            # speed up calculation by adding links
            # as layers don't care about flavour
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc',
                                             'nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                             'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])

        for container in self.data:
            self.layers.calcLayers(container['true_coszen'].get('host'))
            container['densities'] = self.layers.density.reshape((container.size, self.layers.max_layers))
            container['distances'] = self.layers.distance.reshape((container.size, self.layers.max_layers))

        # don't forget to un-link everything again
        self.data.unlink_containers()

        # --- setup empty arrays ---
        if self.calc_mode == 'binned':
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
        propagate_array(self.osc_params.dm_matrix, # pylint: disable = unexpected-keyword-arg, no-value-for-parameter
                        self.osc_params.mix_matrix_complex,
                        self.osc_params.nsi_eps,
                        nubar,
                        e_array.get(WHERE),
                        rho_array.get(WHERE),
                        len_array.get(WHERE),
                        out=out.get(WHERE)
                       )
        out.mark_changed(WHERE)

    @profile
    def compute_function(self):

        # set the correct data mode
        self.data.data_specs = self.calc_specs

        if self.calc_mode == 'binned':
            # speed up calculation by adding links
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc'])
            self.data.link_containers('nubar', ['nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                                'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])

        # --- update mixing params ---
        self.osc_params.theta12 = self.params.theta12.value.m_as('rad')
        self.osc_params.theta13 = self.params.theta13.value.m_as('rad')
        self.osc_params.theta23 = self.params.theta23.value.m_as('rad')
        self.osc_params.dm21 = self.params.deltam21.value.m_as('eV**2')
        self.osc_params.dm31 = self.params.deltam31.value.m_as('eV**2')
        self.osc_params.deltacp = self.params.deltacp.value.m_as('rad')


        for container in self.data:
            self.calc_probs(container['nubar'],
                            container['true_energy'],
                            container['densities'],
                            container['distances'],
                            out=container['probability'],
                           )

        # the following is flavour specific, hence unlink
        self.data.unlink_containers()

        for container in self.data:
            # initial electrons (0)
            fill_probs(container['probability'].get(WHERE),
                       0,
                       container['flav'],
                       out=container['prob_e'].get(WHERE),
                      )
            # initial muons (1)
            fill_probs(container['probability'].get(WHERE),
                       1,
                       container['flav'],
                       out=container['prob_mu'].get(WHERE),
                      )

            container['prob_e'].mark_changed(WHERE)
            container['prob_mu'].mark_changed(WHERE)

    @profile
    def apply_function(self):

        # update the outputted weights
        for container in self.data:
            apply_probs(container['sys_flux'].get(WHERE),
                        container['prob_e'].get(WHERE),
                        container['prob_mu'].get(WHERE),
                        out=container['weights'].get(WHERE))
            container['weights'].mark_changed(WHERE)


# vectorized function to apply (flux * prob)
# must be outside class
if FTYPE == np.float64:
    signature = '(f8[:], f8, f8, f8[:])'
else:
    signature = '(f4[:], f4, f4, f4[:])'
@guvectorize([signature], '(d),(),()->()', target=TARGET)
def apply_probs(flux, prob_e, prob_mu, out):
    out[0] *= (flux[0] * prob_e) + (flux[1] * prob_mu)
