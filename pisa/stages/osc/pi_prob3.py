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

        # init base class!
        super(pi_prob3, self).__init__(
                                       data=data,
                                       params=params,
                                       expected_params=expected_params,
                                       input_names=input_names,
                                       output_names=output_names,
                                       debug_mode=debug_mode,
                                       input_specs=input_specs,
                                       calc_specs=calc_specs,
                                       output_specs=output_specs,
                                       )

        #assert input_specs is not None
        assert calc_specs is not None
        assert output_specs is not None


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

        

        if self.calc_mode == 'events':
            self.data.data_specs = self.calc_specs
            for container in self.data:
                true_coszen = container['true_coszen'].get('host')
                myLayers.calcLayers(true_coszen)
                # calc layers
                densities = myLayers.density.reshape((container.array_length, myLayers.max_layers))
                distances = myLayers.distance.reshape((container.array_length, myLayers.max_layers))
                # empty array to be filled
                probability = np.zeros((container.array_length, 3, 3), dtype=FTYPE)
                prob_e = np.zeros((container.array_length), dtype=FTYPE)
                prob_mu = np.zeros((container.array_length), dtype=FTYPE)
                container.add_array_data('densities', densities)
                container.add_array_data('distances', distances)
                container.add_array_data('prob_e', prob_e)
                container.add_array_data('prob_mu', prob_mu)
                container.add_array_data('probability', probability)

        elif self.calc_mode == 'binned':
            self.data.data_specs = self.calc_specs
            true_coszen = Container.unroll_binning('true_coszen', self.calc_specs).get('host')
            myLayers.calcLayers(true_coszen)
            size = self.calc_specs.size
            self.grid_densities = SmartArray(myLayers.density.reshape((size, myLayers.max_layers)))
            self.grid_distances = SmartArray(myLayers.distance.reshape((size, myLayers.max_layers)))
            self.grid_probabilities_nu = SmartArray(np.empty((size, 3, 3), dtype=FTYPE))
            self.grid_probabilities_nubar = SmartArray(np.empty((size, 3, 3), dtype=FTYPE))

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
        if self.calc_mode == 'events':
            self.data.data_specs = self.calc_specs
            for container in self.data:
                self.calc_probs(container['nubar'],
                                container['true_energy'],
                                container['densities'],
                                container['distances'],
                                out=container['probability'],
                                )

                fill_probs(container['probability'].get(WHERE),
                           0,
                           container['flav'],
                           out=container['prob_e'].get(WHERE),
                           )
                fill_probs(container['probability'].get(WHERE),
                           1,
                           container['flav'],
                           out=container['prob_mu'].get(WHERE),
                           )

                container['prob_e'].mark_changed(WHERE)
                container['prob_mu'].mark_changed(WHERE)

        elif self.calc_mode == 'binned':
            self.data.data_specs = self.calc_specs
            true_energy = Container.unroll_binning('true_energy', self.calc_specs)
            for probs, nubar in zip([self.grid_probabilities_nu, self.grid_probabilities_nubar],[1,-1]):
                self.calc_probs(nubar,
                                true_energy,
                                self.grid_densities,
                                self.grid_distances,
                                out=probs,
                                )

            # probabilities into container
            for container in self.data:
                nubar = container['nubar']
                flav = container['flav']
                if nubar > 0:
                    probs = self.grid_probabilities_nu.get('host')
                else:
                    probs = self.grid_probabilities_nubar.get('host')
                prob_e = probs[...,0,flav]
                prob_mu = probs[...,1,flav]
                container.add_binned_data('prob_e', (self.calc_specs, prob_e), flat=True)
                container.add_binned_data('prob_mu', (self.calc_specs, prob_mu), flat=True)

    def apply(self, inputs=None):
        self.compute()

        #apply_binned = False
        self.data.data_specs = 'events'

        if self.input_mode == 'binned' and self.calc_mode == 'binned':
            #apply_binned = True
            self.data.data_specs = self.calc_specs

        if self.input_mode == 'binned' and self.calc_mode == 'events' and self.output_mode == 'events':
            for container in self.data:
                container.binned_to_array('flux_e')
                container.binned_to_array('flux_mu')
                container.binned_to_array('weights')
        
        if self.input_mode == 'binned' and self.calc_mode == 'events' and self.output_mode == 'binned':
            for container in self.data:
                container.array_to_binned('prob_e', self.input_specs)
                container.array_to_binned('prob_mu', self.input_specs)
            self.data.data_specs = self.calc_specs
            #apply_binned = True
        
        if self.input_mode == 'events' and self.calc_mode == 'binned' and self.output_mode == 'binned':
            for container in self.data:
                container.array_to_binned('flux_e', self.input_specs)
                container.array_to_binned('flux_mu', self.input_specs)
                container.array_to_binned('weights', self.input_specs)
            self.data.data_specs = self.calc_specs
            #apply_binned = True

        if self.input_mode == 'events' and self.calc_mode == 'binned' and self.output_mode == 'events':
            for container in self.data:
                container.binned_to_array('prob_e')
                container.binned_to_array('prob_mu')
        
        # apply
        for container in self.data:
            #if apply_binned:
            w = container['weights'].get('host')
            w *= (container['flux_e'].get('host') * container['prob_e'].get('host') 
                  + container['flux_mu'].get('host') * container['prob_mu'].get('host'))
            container['weights'].mark_changed('host')
            #else:
            #    w = container.get_array_data('weights').get('host')
            #    w *= (container.get_array_data('flux_e').get('host') * container.get_array_data('prob_e').get('host') 
            #          + container.get_array_data('flux_mu').get('host') * container.get_array_data('prob_mu').get('host'))
            #    container.get_array_data('weights').mark_changed('host')

        if self.data.data_mode == 'binned' and self.output_mode == 'events':
            for container in self.data:
                container.binned_to_array('weights')

        if self.data.data_mode == 'events' and self.output_mode == 'binned':
            for container in self.data:
                container.array_to_binned('weight', self.output_specs)


