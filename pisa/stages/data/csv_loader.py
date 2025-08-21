"""
A Stage to load data from a CSV datarelease format file into a PISA pi ContainerSet
"""

from __future__ import absolute_import, print_function, division

import numpy as np
import pandas as pd

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.utils.resources import find_resource
from pisa.core.container import Container
from pisa.utils.format import split


class csv_loader(Stage):  # pylint: disable=invalid-name
    """
    CSV file loader PISA Pi class

    Parameters
    ----------

    events_file : str or sequence of str
        csv file path(s)

    data_dict : (str of a) dict
        Dictionary to specify what keys from the csv files to be loaded
        under what name. Entries can be strings that point to the right
        key in the csv file or lists of keys, and the data will be
        stacked into a 2d array.
        
    output_names : sequence of str
        Event categories to be recorded, needs to be a subset of names 
        in `events_file`.

    neutrinos : bool, default: True
        Flag indicating whether data events represent neutrinos
        In this case, special handling for e.g. nu/nubar, CC vs NC, ...

    dis_idx : int, default: None
        The deep inelastic scattering (DIS) systematic stage in PISA expects a
        key identifiying if an event is a dis event. This key should be 1 for all
        dis events and 0 otherwise. However, your csv file might only contain an
        `interaction` key that assings integers to different interaction types (e.g.
        dis=1, qel=2, res=3, ...). In that case you need to specify which integer
        corresponds to dis. It is preferred to specify a dedicated `dis` key in the
        data_dict.
        
    scale_aeff : bool, default: False
        Convert effective area from cm^2 to m^2 (PISA flux tables are stored in m^2)
        if given in cm^2.

    """
    def __init__(
        self,
        events_file,
        data_dict,
        output_names,
        neutrinos=True,
        dis_idx=None,
        scale_aeff=False,
        **std_kwargs,
    ):

        # instantiation args that should not change
        self.events_file = split(events_file)
        for i, f in enumerate(self.events_file):
            self.events_file[i] = find_resource(f)

        if isinstance(data_dict, str):
            self.data_dict = eval(data_dict)
        elif isinstance(data_dict, dict):
            self.data_dict = data_dict
        else:
            raise ValueError(
                f"Unsupported type {type(data_dict)} for data_dict."
            )

        self.output_names = output_names
        if len(self.output_names) != len(set(self.output_names)):
            raise ValueError(
                'Found duplicates in `output_names`, but each name must be'
                ' unique.'
            )

        self.neutrinos = neutrinos
        if dis_idx is not None:
            self.dis_idx = int(dis_idx)
        else:
            self.dis_idx = None
        self.scale_aeff = scale_aeff

        # init base class
        super().__init__(
            expected_params=(),
            expected_container_keys=(),
            **std_kwargs,
        )


    def setup_function(self):

        raw_data = pd.concat([pd.read_csv(f) for f in self.events_file])

        # create containers from the events
        for name in self.output_names:

            # make container
            container = Container(name)
            
            if self.neutrinos:
                nubar = -1 if 'bar' in name else 1
                if 'e' in name:
                    flav = 0
                if 'mu' in name:
                    flav = 1
                if 'tau' in name:
                    flav = 2
                container.set_aux_data('nubar', nubar)
                container.set_aux_data('flav', flav)

                # cut out right part
                pdg = nubar * (12 + 2 * flav)
                if 'pdg_code' in raw_data:
                    mask = raw_data['pdg_code'] == pdg
                elif 'pdg' in raw_data:
                    mask = raw_data['pdg'] == pdg
                else:
                    raise ValueError("Either 'pdg' or 'pdg_code' must be in file.")

                if 'cc' in name:
                    mask = np.logical_and(mask, raw_data['type'] == 1)
                else:
                    mask = np.logical_and(mask, raw_data['type'] == 0)

                events = raw_data[mask]
            else:
                events = raw_data

            # fill container
            container['initial_weights'] = np.ones(len(events))
            container['weights'] = np.ones(len(events))
            for key, val in self.data_dict.items():
                container[key] = events[val].values.astype(FTYPE)
                
            if self.scale_aeff and 'weighted_aeff' in container.keys:
                container['weighted_aeff'] *= 1.e-4
            
            # Convert interaction key to dis boolen (if needed)
            if 'dis' not in container.keys and 'interaction' in container.keys and self.dis_idx is not None:
                container['dis'] = (container['interaction'] == self.dis_idx).astype(int)

            self.data.add_container(container)

        # check created at least one container
        if len(self.data.names) == 0:
            raise ValueError(
                'No containers created during data loading for some reason.'
            )

    def apply_function(self):
        # reset data representation to events
        self.data.representation = "events"

        # reset weights to initial weights prior to downstream stages running
        for container in self.data:
            container['weights'] = np.copy(container['initial_weights'])


def init_test(**param_kwargs):
    """Initialisation example"""
    data_dict = {'true_energy':'true_energy',
                 'true_coszen':'true_coszen',
                 'weighted_aeff':'weight',
                 'reco_energy':'reco_energy',
                 'reco_coszen':'reco_coszen',
                 'pid':'pid'
                }
    return csv_loader(events_file='events/IceCube_3y_oscillations/neutrino_mc.csv.bz2',
                      data_dict=data_dict,
                      output_names=['nue_cc', 'numu_cc'],
                     )
