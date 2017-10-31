import numpy as np
from numba import guvectorize, SmartArray

from pisa import *
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.stages.osc.osc_params import OscParams
from pisa.stages.osc.layers import Layers
from prob3numba.numba_osc import *
from prob3numba.numba_tools import *


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
                 events=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 apply_specs=None,
                 ):

        expected_params = ()
        input_names = ()
        output_names = ()

        # init base class!
        super(pi_prob3, self).__init__(
                                                events=events,
                                                params=params,
                                                expected_params=expected_params,
                                                input_names=input_names,
                                                output_names=output_names,
                                                debug_mode=debug_mode,
                                                input_specs=input_specs,
                                                calc_specs=calc_specs,
                                                apply_specs=apply_specs,
                                                )

        # that stage doesn't act on anything, it rather just loads events -> this inot base class
        #assert input_specs is not None
        assert calc_specs is not None
        #assert apply_specs is not None


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
        if self.calc_specs == 'events':
            for name, val in self.events.items():
                # calc layers
                myLayers.calcLayers(val['true_coszen'])
                nevts = val['true_coszen'].shape[0]
                numberOfLayers = myLayers.n_layers
                densities = myLayers.density.reshape((nevts,myLayers.max_layers))
                distances = myLayers.distance.reshape((nevts,myLayers.max_layers))
                val['densities'] = SmartArray(densities)
                val['distances'] = SmartArray(distances)
                # empty array to be filled
                probability = np.zeros((nevts,3,3), dtype=FTYPE)
                val['probability'] = SmartArray(probability)
                #probability_vacuum = np.zeros((nevts,3,3), dtype=FTYPE)
        elif isinstance(self.calc_specs, MultiDimBinning):
            # set up the map grid
            self.grid_values = {}
            e = self.calc_specs['true_energy'].weighted_centers.m.astype(FTYPE)
            cz = self.calc_specs['true_coszen'].weighted_centers.m.astype(FTYPE)
            nevts = len(e) * len(cz)
            e_vals, cz_vals = np.meshgrid(e, cz)
            self.grid_values['true_energy'] = SmartArray(e_vals.ravel())
            self.grid_values['true_coszen'] = SmartArray(cz_vals.ravel())
            myLayers.calcLayers(self.grid_values['true_coszen'].get('host'))
            numberOfLayers = myLayers.n_layers
            densities = myLayers.density.reshape((nevts,myLayers.max_layers))
            distances = myLayers.distance.reshape((nevts,myLayers.max_layers))
            self.grid_values['densities'] = SmartArray(densities)
            self.grid_values['distances'] = SmartArray(distances)
            # empty array to be filled
            probability_nu = np.zeros((nevts,3,3), dtype=FTYPE)
            self.grid_values['probability_nu'] = SmartArray(probability_nu)
            probability_nubar = np.zeros((nevts,3,3), dtype=FTYPE)
            self.grid_values['probability_nubar'] = SmartArray(probability_nubar)
            #probability_vacuum = np.zeros((nevts,3,3), dtype=FTYPE)


    def compute(self):

        if TARGET == 'cuda':
            where='gpu'
        else:
            where='host'

        if self.calc_specs == 'events':
            for name, val in self.events.items():
                propagate_array(self.dm,
                                self.mix,
                                self.nsi_eps,
                                val['nubar'],
                                val['true_energy'].get(where),
                                val['densities'].get(where),
                                val['distances'].get(where),
                                out=val['probability'].get(where))
                val['probability'].mark_changed(where)

        elif isinstance(self.calc_specs, MultiDimBinning):
            for nubar, probs in zip([1, -1], ['probability_nu', 'probability_nubar']):
                propagate_array(self.dm,
                                self.mix,
                                self.nsi_eps,
                                nubar,
                                self.grid_values['true_energy'].get(where),
                                self.grid_values['densities'].get(where),
                                self.grid_values['distances'].get(where),
                                out=self.grid_values[probs].get(where))
                self.grid_values[probs].mark_changed(where)


    def apply(self, inputs=None):

        self.compute()

        if isinstance(self.apply_specs, MultiDimBinning):
            maps = []
            assert self.calc_specs == self.apply_specs, 'cannot do different binnings yet'
            flavs = ['e', 'mu', 'tau']
            hists = self.grid_values['probability_nu'].get('host')
            print hists
            n_e = self.apply_specs['true_energy'].num_bins
            n_cz = self.apply_specs['true_coszen'].num_bins
            for i in range(3):
                for j in range(3):
                    hist = hists[:,i,j]
                    hist = hist.reshape(n_e, n_cz)
                    maps.append(Map(name='prob_%s_to_%s'%(flavs[i],flavs[j]), hist=hist, binning=self.apply_specs))
            self.outputs = MapSet(maps)
            return self.outputs


