import numpy as np
from numba import SmartArray

from pisa import *
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet


class dummy_event_loader(PiStage):
    """
    Dummy event loader PISA Pi class

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

        expected_params = ('n_points')
        input_names = ()
        output_names = ()

        # init base class!
        super(dummy_event_loader, self).__init__(
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
        assert input_specs is None
        assert calc_specs is None
        #assert apply_specs is None

        self.compute_args = None
        # args to load into apply function
        self.apply_args = ['event_weights']
        # can be python, numba and numba-cuda
        self.kernel_support = ['python']
    

    def setup(self):

        # create some dumb events

        # number of points for E x CZ grid
        points = int(self.params.n_points.value.m)
        nevts = points**2

        # input arrays
        # E and CZ
        energy_points = np.logspace(0,3,points, dtype=FTYPE)
        cz_points = np.linspace(-1,1,points, dtype=FTYPE)
        energy, cz = np.meshgrid(energy_points, cz_points)
        energy = energy.ravel()
        energy = SmartArray(energy)
        cz = cz.ravel()
        cz = SmartArray(cz)

        # nu /nu-bar
        nubar = np.ones(nevts, dtype=np.int32)
        nubar = SmartArray(nubar)

        # event_weights
        #event_weights = np.ones(nevts, dtype=FTYPE)
        event_weights = np.random.rand(nevts).astype(FTYPE)
        event_weights = SmartArray(event_weights)
        
        weights = SmartArray(np.empty(nevts, dtype=FTYPE))

        numu = {'true_energy' : energy,
                'true_coszen' : cz,
                'nubar' : nubar,
                'event_weights' : event_weights,
                'weights' : weights,
                }

        assert self.events is None
        self.events = {'numu': numu}

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
        if inputs is None:
            if self.apply_specs is None:
                pass

            elif self.apply_specs == 'events':
                if self.events is None:
                    raise TypeError('Cannot apply to events with no events present')
                # nothing else to do
            elif isinstance(self.apply_specs, MultiDimBinning):
                if self.events is None:
                    raise TypeError('Cannot return Map with no inputs and no events present')
                else:
                    # run apply_kernel on events array and a private output array
                    # ToDo: private weights array (or should it be normal weights array?)
                    binning = self.apply_specs
                    bin_edges = [edges.magnitude for edges in binning.bin_edges]
                    binning_cols = binning.names

                    maps = []
                    for name, evts in self.events.items():
                        sample = [evts[colname].get('host') for colname in binning_cols]
                        hist, _ = np.histogramdd(sample=sample,
                                                 weights=evts['weights'].get('host'),
                                                 bins=bin_edges,
                                                 )

                        maps.append(Map(name=name, hist=hist, binning=binning))
        self.outputs = MapSet(maps)
        return self.outputs
