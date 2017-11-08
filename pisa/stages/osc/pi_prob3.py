import numpy as np
from numba import guvectorize, SmartArray

from pisa import *
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.core.binning import MultiDimBinning
from pisa.core.container import Container
from pisa.core.map import Map, MapSet
from pisa.stages.osc.osc_params import OscParams
from pisa.stages.osc.layers import Layers
from pisa.stages.osc.prob3numba.numba_osc import *
from pisa.utils.numba_tools import *


class pi_prob3(PiStage):
    """
    prob3 osc PISA Pi class

    Paramaters
    ----------

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

        expected_params = ()

        input_names = ()
        output_names = ()

        assert input_specs is not None
        assert calc_specs is not None
        assert output_specs is not None

        # what are the keys used from the inputs during apply
        input_keys = ('weights',
                      'flux_e',
                      'flux_mu',
                      )
        # what are keys added or altered in the calculation used during apply 
        calc_keys = ('prob_e',
                     'prob_mu',
                     )
        # what keys are added or altered for the outputs during apply
        output_keys = ('weights',
                       )

        # init base class
        super(pi_prob3, self).__init__(data=data,
                                       params=params,
                                       expected_params=expected_params,
                                       input_names=input_names,
                                       output_names=output_names,
                                       debug_mode=debug_mode,
                                       input_specs=input_specs,
                                       calc_specs=calc_specs,
                                       output_specs=output_specs,
                                       input_keys=input_keys,
                                       calc_keys=calc_keys,
                                       output_keys=output_keys,
                                       )

    def setup(self):

        # Set up some dumb mixing parameters
        OP = OscParams(7.5e-5, 2.524e-3, np.sqrt(0.306), np.sqrt(0.02166), np.sqrt(0.441), 261/180.*np.pi)
        self.mix = OP.mix_matrix_complex
        self.dm = OP.dm_matrix
        self.nsi_eps = np.zeros_like(self.mix)

        # setup the layers
        earth_model = '/home/peller/cake/pisa/resources/osc/PREM_59layer.dat'
        det_depth = 2
        atm_height = 20
        myLayers = Layers(earth_model, det_depth, atm_height)
        myLayers.setElecFrac(0.4656, 0.4656, 0.4957)

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
            myLayers.calcLayers(container['true_coszen'].get('host'))
            container['densities'] = myLayers.density.reshape((container.size, myLayers.max_layers))
            container['distances'] = myLayers.distance.reshape((container.size, myLayers.max_layers))

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

        # setup more arrays
        for container in self.data:
            container['prob_e'] = np.empty((container.size), dtype=FTYPE)
            container['prob_mu'] = np.empty((container.size), dtype=FTYPE) 

    @profile
    def calc_probs(self, nubar, e_array, rho_array, len_array, out):
        ''' wrapper to execute osc. calc '''
        propagate_array(self.dm,
                        self.mix,
                        self.nsi_eps,
                        nubar,
                        e_array.get(WHERE),
                        rho_array.get(WHERE),
                        len_array.get(WHERE),
                        out=out.get(WHERE)
                        )
        out.mark_changed(WHERE)

    def compute(self):

        # set the correct data mode 
        self.data.data_specs = self.calc_specs

        if self.calc_mode == 'binned':
            # speed up calculation by adding links
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc'])
            self.data.link_containers('nubar', ['nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                                'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])

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

    def apply_function(self):

        # update the outputted weights
        for container in self.data:
            apply_probs(container['flux_e'].get(WHERE),
                        container['flux_mu'].get(WHERE),
                        container['prob_e'].get(WHERE),
                        container['prob_mu'].get(WHERE),
                        out=container['weights'].get(WHERE))
            container['weights'].mark_changed(WHERE)


# vectorized function to apply flux x osc_probs
if FTYPE == np.float64:
    signature = '(f8, f8, f8, f8, f8[:])'
else:
    signature = '(f4, f4, f4, f4, f4[:])'
@guvectorize([signature], '(),(),(),()->()', target=TARGET)
def apply_probs(flux_e, flux_mu, prob_e, prob_mu, out):
    out[0] *= (flux_e * prob_e) + (flux_mu * prob_mu)
