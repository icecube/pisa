import numpy as np
from numba import SmartArray

from pisa import *
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.numba_tools import equal_and_scale, WHERE
from pisa.utils.profiler import profile
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

    def setup_function(self):

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
            reco_energy = evts[name]['reco_energy'].astype(FTYPE)
            reco_coszen = evts[name]['reco_coszen'].astype(FTYPE)
            pid = evts[name]['pid'].astype(FTYPE)
            # this determination of flavour is the worst possible coding, ToDo
            nubar = -1 if 'bar' in name else 1
            if 'e' in name: flav = 0
            if 'mu' in name: flav = 1
            if 'tau' in name: flav = 2
            weighted_aeff = evts[name]['weighted_aeff'].astype(FTYPE)
            event_weights = np.ones_like(true_energy)
            weights = np.ones_like(true_energy)
            neutrino_nue_flux = evts[name]['neutrino_nue_flux'].astype(FTYPE)
            neutrino_numu_flux = evts[name]['neutrino_numu_flux'].astype(FTYPE)
            flux = np.stack([neutrino_nue_flux, neutrino_numu_flux], axis=1)
            neutrino_oppo_nue_flux = evts[name]['neutrino_oppo_nue_flux'].astype(FTYPE)
            neutrino_oppo_numu_flux = evts[name]['neutrino_oppo_numu_flux'].astype(FTYPE)
            oppo_flux = np.stack([neutrino_oppo_nue_flux, neutrino_oppo_numu_flux], axis=1)

            linear_fit_maccqe =  evts[name]['linear_fit_MaCCQE'].astype(FTYPE)
            quad_fit_maccqe =    evts[name]['quad_fit_MaCCQE'].astype(FTYPE)
            linear_fit_maccres = evts[name]['linear_fit_MaCCRES'].astype(FTYPE)
            quad_fit_maccres =   evts[name]['quad_fit_MaCCRES'].astype(FTYPE)

            # make container
            container = Container(name)
            container.add_array_data('true_energy', true_energy)
            container.add_array_data('true_coszen', true_coszen)
            container.add_array_data('reco_energy', reco_energy)
            container.add_array_data('reco_coszen', reco_coszen)
            container.add_array_data('pid', pid)
            container.add_scalar_data('nubar', nubar)
            container.add_scalar_data('flav', flav)
            container.add_array_data('event_weights', event_weights)
            container.add_array_data('weights', weights)
            container.add_array_data('weighted_aeff', weighted_aeff)
            container.add_array_data('nominal_flux', flux)
            container.add_array_data('nominal_opposite_flux', oppo_flux)

            container.add_array_data('linear_fit_maccqe', linear_fit_maccqe)
            container.add_array_data('quad_fit_maccqe', quad_fit_maccqe)
            container.add_array_data('linear_fit_maccres', linear_fit_maccres)
            container.add_array_data('quad_fit_maccres', quad_fit_maccres)

            self.data.add_container(container)

    @profile
    def apply_function(self):
        # reset weights
        for container in self.data:
            equal_and_scale(1.,
                            container['event_weights'].get(WHERE),
                            out=container['weights'].get(WHERE))
            container['weights'].mark_changed(WHERE)

