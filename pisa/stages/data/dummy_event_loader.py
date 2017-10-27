import numpy as np
from numba import SmartArray

from pisa import *
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging


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

        # that stage doesn't act on anything, it rather just loads events
        assert input_specs is None
        assert calc_specs is None
        #assert apply_specs is None

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
        event_weights = np.ones(nevts, dtype=FTYPE)
        event_weights = SmartArray(event_weights)
        weights = SmartArray(np.copy(event_weights))
        
        numu = {'true_energy' : energy,
                'true_coszen' : cz,
                'nubar' : nubar,
                'event_weights' : event_weights,
                'weights' : weights,
                }

        assert self.events is None
        self.events = {'numu': numu}

    #def compute(self):
    #    pass

    #def apply(self):
    #    pass

