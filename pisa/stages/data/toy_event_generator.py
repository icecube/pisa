"""
Stage to generate some random toy events.

If desired, parameterisations can be used for fluxes, effective areas,
resolution functions, and event classification efficiencies.
"""

from collections.abc import Sequence
import os
from os.path import basename, splitext

import numpy as np

from pisa import FTYPE, CACHE_DIR
from pisa.core.binning import MultiDimBinning
from pisa.core.container import Container
from pisa.core.events_pi import EventsPi
from pisa.core.stage import Stage
from pisa.utils.flux_weights import load_2d_table
from pisa.utils.random_numbers import get_random_state
from pisa.scripts.add_flux_to_events_file import add_fluxes_to_file
from pisa.scripts.make_toy_events import make_toy_events

__all__ = ['toy_event_generator', 'init_test']


class toy_event_generator(Stage):  # pylint: disable=invalid-name
    """
    Random toy event generator class

    Parameters
    ----------
    output_names : list of str
        list of output names
    n_events : int
        number of events to be generated per output name
    seed : int
        seed to be used for random number generator
    energy_range
    spectral_index
    coszen_range
    aeff_energy_param
    aeff_coszen_param
    reco_param
    pid_param
    pid_dist

    """
    def __init__(
        self,
        n_events,
        seed,
        data_dict=None,
        output_names=None,
        energy_range=None,
        spectral_index=None,
        coszen_range=None,
        add_fluxes=False,
        aeff_energy_param=None,
        aeff_coszen_param=None,
        reco_param=None,
        pid_param=None,
        pid_dist=None,
        **std_kwargs,
    ):
        if isinstance(add_fluxes, str):
            add_fluxes = add_fluxes.lower()
        assert add_fluxes in (None, "honda", "beam")
        self.add_fluxes = add_fluxes
        # TODO: output names currently have no effect
        self.output_names = output_names
        self.n_events = int(n_events)
        self.seed = int(seed)
        self.data_dict = data_dict
        self.random_state = get_random_state(random_state=self.seed)
        if isinstance(energy_range, str):
            energy_range = eval(energy_range)
        energy_range = (int(er) for er in energy_range)
        self.energy_range = energy_range
        if not isinstance(spectral_index, float):
            spectral_index = float(spectral_index)
        self.spectral_index = spectral_index
        if isinstance(coszen_range, str):
            coszen_range = eval(coszen_range)
        coszen_range = (float(cr) for cr in coszen_range)
        self.coszen_range = coszen_range
        self.aeff_energy_param = aeff_energy_param
        self.aeff_coszen_param = aeff_coszen_param
        self.reco_param = reco_param
        self.pid_param = pid_param
        self.pid_dist = pid_dist

        # init base class
        super().__init__(
            expected_params=(),
            expected_container_keys=(),
            **std_kwargs,
        )

        self.events_file = make_toy_events(
            outdir=CACHE_DIR,
            num_events=self.n_events,
            energy_range=self.energy_range,
            spectral_index=self.spectral_index,
            coszen_range=self.coszen_range,
            num_sets=1,
            first_set=0,
            aeff_energy_param=self.aeff_energy_param,
            aeff_coszen_param=self.aeff_coszen_param,
            reco_param=self.reco_param,
            pid_param=self.pid_param,
            pid_dist=self.pid_dist
        )[0]
        if self.add_fluxes == "honda":
            flux_file = "flux/honda-2015-spl-solmin-aa.d"
            flux_file_bname, ext = splitext(basename(flux_file))
            flux_table = load_2d_table(flux_file)
            add_fluxes_to_file(
                data_file_path=self.events_file,
                flux_table=flux_table,
                flux_name='nominal',
                outdir=CACHE_DIR,
                label=flux_file_bname
            )
            bname, ext = splitext(basename(self.events_file))
            self.events_file = os.path.join(CACHE_DIR, '{}__with_fluxes_{}{}'.format(bname, flux_file_bname, ext))

        self.load_events()

    def load_events(self):
        '''Loads events from events file'''

        # Create the events structure
        self.evts = EventsPi(
            name='Events',
            neutrinos=True
        )

        # If user provided a variable mapping dict, parse it from the input string (if not already done)
        if self.data_dict is not None:
            if isinstance(self.data_dict, str):
                self.data_dict = eval(self.data_dict)

        # Load the event file into the events structure
        self.evts.load_events_file(
            events_file=self.events_file,
            variable_mapping=self.data_dict
        )

        if hasattr(self.evts, "metadata"):
            self.metadata = self.evts.metadata

    def record_event_properties(self):
        '''Adds fields present in events file and selected in `self.data_dict`
        into containers for the specified output names. Also ensures the
        presence of a set of nominal weights.
        '''

        # define which  categories to include in the data
        # user can manually specify what they want using `output_names`, or else just use everything
        output_keys = self.output_names if len(self.output_names) > 0 else self.evts.keys()

        # create containers from the events
        for name in output_keys:

            # make container
            container = Container(name)
            container.representation = 'events'
            event_groups = self.evts.keys()
            if name not in event_groups:
                raise ValueError(
                    'Output name "%s" not found in events. Only found %s.'
                    % (name, event_groups)
                )

            # add the events data to the container
            for key, val in self.evts[name].items():
                container[key] = val

            # create weights arrays:
            # * `initial_weights` as starting point (never modified)
            # * `weights` to be initialised from `initial_weights`
            #   and modified by the stages
            # * user can also provide `initial_weights` in input file
            #TODO Maybe add this directly into EventsPi
            if 'weights' in container.keys:
                # raise manually to give user some helpful feedback
                raise KeyError(
                    'Found an existing `weights` array in "%s"'
                    ' which would be overwritten. Consider renaming it'
                    ' to `initial_weights`.' % name
                )
            container['weights'] = np.ones(container.size, dtype=FTYPE)

            if 'initial_weights' not in container.keys:
                container['initial_weights'] = np.ones(container.size, dtype=FTYPE)

            # add neutrino flavor information for neutrino events
            #TODO Maybe add this directly into EventsPi
            nubar = -1 if 'bar' in name else 1
            if name.startswith('nutau'):
                flav = 2
            elif name.startswith('numu'):
                flav = 1
            elif name.startswith('nue'):
                flav = 0
            else:
                raise ValueError('Cannot determine flavour of %s'%name)
            container.set_aux_data('nubar', nubar)
            container.set_aux_data('flav', flav)

            self.data.add_container(container)

        # check created at least one container
        if len(self.data.names) == 0:
            raise ValueError(
                'No containers created during data loading for some reason.'
            )

    def setup_function(self):
        '''Store event properties from events file at
        service initialisation. Cf. `Stage` docs.
        '''
        self.record_event_properties()

    """
    def setup_function(self):

        for name in self.output_names:

            container = Container(name, representation=self.calc_mode)

            nubar = -1 if 'bar' in name else 1
            if 'e' in name:
                flav = 0
            if 'mu' in name:
                flav = 1
            if 'tau' in name:
                flav = 2

            if not isinstance(self.calc_mode, MultiDimBinning):
                # add events explicitly in the array representation
                container['true_energy'] = np.power(10, self.random_state.rand(self.n_events).astype(FTYPE) * 3)
                container['true_coszen'] = self.random_state.rand(self.n_events).astype(FTYPE) * 2 - 1

            size = container.size

            # make some initial weights
            if self.random:
                container['initial_weights'] = self.random_state.rand(size).astype(FTYPE)
            else:
                container['initial_weights'] = np.ones(size, dtype=FTYPE)

            # other necessary info
            container.set_aux_data('nubar', nubar)
            container.set_aux_data('flav', flav)
            container['weights'] = np.ones(size, dtype=FTYPE)
            container['weighted_aeff'] = np.ones(size, dtype=FTYPE)

            flux_nue = np.zeros(size, dtype=FTYPE)
            flux_numu = np.ones(size, dtype=FTYPE)
            flux = np.stack([flux_nue, flux_numu], axis=1)

            container['nu_flux_nominal'] = flux
            container['nubar_flux_nominal'] = flux

            self.data.add_container(container)
    """

    def apply_function(self):

        # reset data representation to events
        #TODO This should be fixed more generally at the Pipeline level, see XXX
        self.data.representation = "events"

        # reset weights to initial weights prior to downstream stages running
        for container in self.data:
            container['weights'] = np.copy(container['initial_weights'])


def init_test(**param_kwargs):
    """Initialisation example"""
    return toy_event_generator(
        output_names=[],
        n_events=100,
        energy_range=[1,10],
        coszen_range=[-1,0],
        spectral_index=0,
        seed=666,
        aeff_energy_param="aeff/vlvnt_aeff_energy_param.json",
        aeff_coszen_param="aeff/vlvnt_aeff_coszen_param.json",
        reco_param="reco/vlvnt_reco_param.json",
        pid_param="pid/vlvnt_pid_energy_param.json",
        pid_dist="discrete"
    )
