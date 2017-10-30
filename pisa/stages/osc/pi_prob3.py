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


# define numba functions

if FTYPE == np.float64:
    signature = '(f8[:,:], c16[:,:], c16[:,:], i4, f8, f8[:], f8[:], f8[:,:])'
    #signature_vac = '(f8[:,:], c16[:,:], i4, f8, f8[:], f8[:,:])'
else:
    signature = '(f4[:,:], c8[:,:], c8[:,:], i4, f4, f4[:], f4[:], f4[:,:])'
    #signature_vac = '(f4[:,:], c8[:,:], i4, f4, f4[:], f4[:,:])'

@guvectorize([signature], '(a,b),(c,d),(e,f),(),(),(g),(h)->(a,b)', target=target)
def propagate_array(dm, mix, nsi_eps, nubar, energy, densities, distances, probability):
    osc_probs_layers_kernel(dm, mix, nsi_eps, nubar, energy, densities, distances, probability)

#@guvectorize([signature_vac], '(a,b),(c,d),(),(),(i)->(a,b)', target=target)

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
        earth_model = '/home/peller/cake/pisa/resources/osc/PREM_12layer.dat'
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
            self.grid_values['nubar'] = 1
            myLayers.calcLayers(self.grid_values['true_coszen'].get('host'))
            numberOfLayers = myLayers.n_layers
            densities = myLayers.density.reshape((nevts,myLayers.max_layers))
            distances = myLayers.distance.reshape((nevts,myLayers.max_layers))
            self.grid_values['densities'] = SmartArray(densities)
            self.grid_values['distances'] = SmartArray(distances)
            # empty array to be filled
            probability = np.zeros((nevts,3,3), dtype=FTYPE)
            self.grid_values['probability'] = SmartArray(probability)
            #probability_vacuum = np.zeros((nevts,3,3), dtype=FTYPE)


    def compute(self):

        if target == 'cuda':
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
            propagate_array(self.dm,
                            self.mix,
                            self.nsi_eps,
                            self.grid_values['nubar'],
                            self.grid_values['true_energy'].get(where),
                            self.grid_values['densities'].get(where),
                            self.grid_values['distances'].get(where),
                            out=self.grid_values['probability'].get(where))
            self.grid_values['probability'].mark_changed(where)

    #@staticmethod
    #def apply_kernel(event_weight, weight):
    #    weight[0] = event_weight

    #def apply_vectorizer(self):
    #    if self.apply_specs == 'events':
    #        for name, val in self.events.items():
    #            self.apply_to_arrays(val['true_energy'], val['true_coszen'], val['weights'])
    #    elif isinstance(self.apply_specs, MultiDimBinning):
    #        if self.events is None:
    #            raise TypeError('Cannot return Map with no inputs and no events present')
    #        else:
    #            e = self.apply_specs['true_energy'].bin_centers.m
    #            cz = self.apply_specs['ture_coszen'].bin_centers.m
    #            apply_e_vals, apply_cz_vas = 

    #def get_apply_array(self, name, key):
    #    if self.apply_specs is None:
    #        return None
    #    elif self.apply_specs == 'events':
    #        return self.events[name][key]
    #    elif isinstance(self.apply_specs, MultiDimBinning):




    def apply(self, inputs=None):

        self.compute()

        if isinstance(self.apply_specs, MultiDimBinning):
            maps = []
            assert self.calc_specs == self.apply_specs, 'cannot do different binnings yet'
            flavs = ['e', 'mu', 'tau']
            hists = self.grid_values['probability'].get('host')
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



        #if inputs is None:
        #    if self.apply_specs is None:
        #        pass

        #    elif self.apply_specs == 'events':
        #        if self.events is None:
        #            raise TypeError('Cannot apply to events with no events present')
        #        # nothing else to do
        #    elif isinstance(self.apply_specs, MultiDimBinning):
        #        if self.events is None:
        #            raise TypeError('Cannot return Map with no inputs and no events present')
        #        else:
        #            # run apply_kernel on events array and a private output array
        #            # ToDo: private weights array (or should it be normal weights array?)
        #            binning = self.apply_specs
        #            bin_edges = [edges.magnitude for edges in binning.bin_edges]
        #            binning_cols = binning.names

        #            maps = []
        #            for name, evts in self.events.items():
        #                sample = [evts[colname].get('host') for colname in binning_cols]
        #                hist, _ = np.histogramdd(sample=sample,
        #                                         weights=evts['weights'].get('host'),
        #                                         bins=bin_edges,
        #                                         )

        #                maps.append(Map(name=name, hist=hist, binning=binning))
        #self.outputs = MapSet(maps)
        #return self.outputs



