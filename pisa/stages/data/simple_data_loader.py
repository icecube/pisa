import numpy as np
from numba import SmartArray

from pisa import *
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.core.container import Container, ContainerSet
from pisa.core.events import Events


class simple_data_loader(PiStage):
    """
    random toy event generator PISA Pi class

    Paramaters
    ----------

    events_file : hdf5 file path (output from make_events), including flux weights and Genie systematics coefficients

    mc_cuts : cut expr
        e.g. '(true_coszen <= 0.5) & (true_energy <= 70)'


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

        expected_params = ('events_file',
                           'mc_cuts',
                           )

        # init base class
        super(simple_data_loader, self).__init__(
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

        # doesn't calculate anything
        assert self.calc_mode is None

    def setup(self):

        # --- Load the events ---

        # open Events file
        evts = Events(self.params.events_file.value)

        #Apply any cuts that the user defined
        if self.params.mc_cuts.value is not None:
            logging.info('applying the following cuts to events: %s'%self.params.mc_cuts.value)
            evts = evts.applyCut(self.params.mc_cuts.value)

        for name in self.output_names:
            
            # ToDo:
            # this procedure here is solely for testing, this willa ll need to 
            # be much more dynamic
            # variables to load should be specified in cfg file etc...

            # load
            true_energy = evts[name]['true_energy'].astype(FTYPE)
            true_coszen = evts[name]['true_coszen'].astype(FTYPE)
            # this determination of flavour is the worst possible coding, ToDo
            nubar = -1 if 'bar' in name else 1
            if 'e' in name: flav = 0
            if 'mu' in name: flav = 1
            if 'tau' in name: flav = 2
            weighted_aeff = evts[name]['weighted_aeff'].astype(FTYPE)
            event_weights = np.ones_like(true_energy)
            weights = np.ones_like(true_energy)
            flux_nue = evts[name]['neutrino_nue_flux'].astype(FTYPE)
            flux_numu = evts[name]['neutrino_numu_flux'].astype(FTYPE)

            # make container
            container = Container(name)
            container.add_array_data('true_energy', true_energy)
            container.add_array_data('true_coszen', true_coszen)
            container.add_scalar_data('nubar', nubar)
            container.add_scalar_data('flav', flav)
            container.add_array_data('event_weights', event_weights)
            container.add_array_data('weights', weights)
            container.add_array_data('weighted_aeff', weighted_aeff)
            container.add_array_data('flux_e', flux_nue)
            container.add_array_data('flux_mu', flux_numu)
            self.data.add_container(container)



    def apply_function(self):
        # reset weights
        for container in self.data:
            weights = container['weights'].get('host')
            #new_weights = container['event_weights'].get('host')
            new_weights = container['weighted_aeff'].get('host')
            # we need to re-assign the array!
            weights[:] = new_weights[:]
            container['weights'].mark_changed('host')

