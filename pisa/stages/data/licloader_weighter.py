"""
A class to load lic files and weight existing events 
"""

from __future__ import absolute_import, print_function, division

import numpy as np
import h5py as h5

import LeptonWeighter as LW

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.utils import vectorizer
from pisa.utils.profiler import profile
from pisa.core.container import Container
from pisa.core.events_pi import EventsPi
from pisa.utils.resources import find_resource


class lic_loader_weighter(Stage):
    """
    LeptonWeighter LIC file reader and LI event weighter. Sets two weight containers
        weights
        astro_weights 

    plus duplicates holding the initial weights. This way we can reweight 

    Parameters 
    ----------
    in_files : string, or list of strings. Data files
    lic_files : string, or list of strings. LeptonInjector configuration files
    n_files : int. The number of files produced in each run, share a single LIC file. 
    """

    def __init__(self,
                in_files,
                lic_files,
                output_names,
                n_files:int,
                diff_nu_cc_xs="dsdxdy_nu_CC_iso.fits",
                diff_nubar_cc_xs="dsdxdy_nubar_CC_iso.fits",
                diff_nu_nc_xs="dsdxdy_nu_NC_iso.fits",
                diff_nubar_nc_xs="dsdxdy_nubar_NC_iso.fits",
                **std_kwargs):
        
        if isinstance(lic_files, str):
            self._lic_files_paths = [find_resource(lic_files),]
        elif isinstance(lic_files, (list, tuple)):
            self._lic_files_paths = [find_resource(lic_file) for lic_file in lic_files]
        else:
            raise TypeError("Unknown lic_file datatype {}".format(type(lic_files)))

        if isinstance(in_files, str):
            self.in_files = [find_resource(in_files),]
        elif isinstance(in_files, (tuple, list)):
            self.in_files = [find_resource(in_file) for in_file in in_files]
        else:
            raise TypeError("Unknown in_files datatype {}".format(type(in_files)))
        
        # load the lic files! 
        self.lic_files = [LW.MakeGeneratorsFromLICFile(name) for name in self._lic_files_paths]
        self.xs_obj = LW.CrossSectionFromSpline(find_resource(diff_nu_cc_xs), find_resource(diff_nubar_cc_xs),
                                                find_resource(diff_nu_nc_xs), find_resource(diff_nubar_nc_xs))
        
        # the target containers! 
        self.output_names = output_names
        # with this, we just need to multiply the weight by the actual flux. Then it'll work! 
        self._one_weighter = LW.Weighter(LW.ConstantFlux(1.0/n_files), self.xs_obj, self.lic_files)

    def setup_function(self):
        """
        Load in the lic files, build the weighters, and get all the one-weights. To get the true 
        """

        raw_data = h5.File( self.in_files[0] )


        # create containers from the events
        for name in self.output_names:

            # make container
            container = Container(name)
            nubar = -1 if 'bar' in name else 1
            if 'e' in name:
                flav = 0
            if 'mu' in name:
                flav = 1
            if 'tau' in name:
                flav = 2

            # cut out right part
            pdg = nubar * (12 + 2 * flav)

            mask = raw_data['true_pid'] == pdg
            if 'cc' in name:
                mask = np.logical_and(mask, raw_data['type'] > 0)
            else:
                mask = np.logical_and(mask, raw_data['type'] == 0)

            events = raw_data[mask]

            # aaahhhh no this format will only work 
            container['weighted_aeff'] = events['weight'][:].astype(FTYPE)
            container['weights'] = np.ones(container.size, dtype=FTYPE)
            container['initial_weights'] = np.ones(container.size, dtype=FTYPE)
            container['astro_weights'] = np.ones(container.size, dtype=FTYPE)
            container['astro_initial_weights'] = np.ones(container.size, dtype=FTYPE)

            container['total_column_depth']  = events['total_column_depth'][:].astype(FTYPE)
            container['true_bjorkenx']  = events['true_bjorkenx'][:].astype(FTYPE)
            container['true_bjorkeny']  = events['true_bjorkeny'][:].astype(FTYPE)

            container['true_energy'] = events['true_energy'][:].astype(FTYPE)
            container['true_coszen'] = events['true_zenith'][:].astype(FTYPE)
            container['reco_energy'] = events['reco_energy'][:].astype(FTYPE)
            container['reco_coszen'] = events['reco_zenith'][:].astype(FTYPE)
            container['pid'] = events['pid'][:].astype(FTYPE)
            container.set_aux_data('nubar', nubar)
            container.set_aux_data('flav', flav)

            self.data.add_container(container)

    def apply_function(self):
        """
        Reset all the weights to the initial weights 
        """
        for container in self.data:
            container['weights'] = np.copy(container['initial_weights'])
            container["astro_weights"] = np.copy(container["initial_astro_weights"])