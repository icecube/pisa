from asyncio import events
import h5py
import numpy as np
from pisa import FTYPE
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.core.stage import Stage
from pisa.utils.resources import find_resource
from pisa.core.container import Container

def read_group(h5group):
    data_dict = {key: value[:] for key, value in h5group.items()}
    return data_dict

class h5_loader(Stage):
    """
    Stage to load event data from an HDF5 file.
    """

    def __init__(self, events_file, type="neutrinos", **std_kwargs):
        self.events_file = events_file
        self.type = type

        if type == "muons":
            expected_params = ("orca_muon_scale",)
        else:
            expected_params = ()

        super().__init__(
            expected_params=expected_params,
            expected_container_keys=(),
            **std_kwargs,
        )

    def setup_function(self): 
        with h5py.File(find_resource(self.events_file), 'r') as h5file:
            if self.type == "neutrinos":
                data = read_group(h5file["binned_nu_response"])
            elif self.type == "muons":
                data = read_group(h5file["binned_muon"])
            elif self.type == "data":
                data = read_group(h5file["binned_data"])
            Ct_reco_axis = read_group(h5file["Ct_reco_axis"])
            Ct_true_axis = read_group(h5file["Ct_true_axis"])
            E_reco_axis = read_group(h5file["E_reco_axis"])
            E_true_axis = read_group(h5file["E_true_axis"])
            
            if self.type == "muons" or self.type == "data":
                container = Container('total')
                container.representation = self.calc_mode
                w = data['W'].astype(FTYPE).reshape((3,15,20))
                w_cut = w[:,:,0:10]
                container['weights'] = w_cut.transpose((1,2,0))
                container['initial_weights'] = container['weights'].copy()
                self.data.add_container(container)
                #container['weights'] = data['counts'].astype(FTYPE)

            elif self.type == "neutrinos":
                output_names = ['nue_cc', 'numu_cc', 'nutau_cc', 'nuall_nc', 'nuebar_cc', 'numubar_cc', 'nutaubar_cc', 'nuallbar_nc']
                for name in output_names:
                    container = Container(name)
                    container.representation = "events"
                    if 'cc' in name:
                        mask = data['IsCC'] == True
                    else:
                        mask = data['IsCC'] == False
                    if 'e' in name:
                        flav = 0
                    elif 'mu' in name:
                        flav = 1
                    elif 'tau' in name:
                        flav = 2
                    else:
                        flav = 1  # for nc
                    nubar = -1 if 'bar' in name else 1


                    container.set_aux_data('nubar', nubar)
                    container.set_aux_data('flav', flav)

                    pdg = nubar * (12 + 2 * flav)
                    mask = np.logical_and(mask, data['Pdg'] == pdg)
                    container['weighted_aeff'] = data['W'][mask].astype(FTYPE)
                    container['true_energy'] = data['E_true_bin_center'][mask].astype(FTYPE)
                    container['true_coszen'] = data['Ct_true_bin_center'][mask].astype(FTYPE)
                    container["reco_energy"] = data['E_reco_bin_center'][mask].astype(FTYPE)
                    container["reco_coszen"] = data['Ct_reco_bin_center'][mask].astype(FTYPE)
                    container["pid"] = data['AnaClass'][mask].astype(FTYPE)
                    container['initial_weights'] = np.ones_like(container['weighted_aeff'])
                    container['weights'] = np.ones_like(container['weighted_aeff'])
                    self.data.add_container(container)


    def apply_function(self):
        if self.type == "neutrinos":
            # reset data representation to events
            self.data.representation = "events"

            # reset weights to initial weights prior to downstream stages running
            for container in self.data:
                container['weights'] = np.copy(container['initial_weights'])
                container.mark_changed('weights')

        elif self.type == "muons":
            muon_scale = self.params.orca_muon_scale.m_as('dimensionless')
            for container in self.data:
                container['weights'] = container['initial_weights'] * muon_scale
                container.mark_changed('weights')