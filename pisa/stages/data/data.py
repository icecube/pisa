import os
import sys

import h5py
import numpy as np

from pisa import ureg, Q_
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.comparisons import normQuant
from pisa.utils.resources import find_resource


__author__ = 'P. Eller'

__license__ = '''Copyright (c) 2014-2017, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


class data(Stage):
    """Data loader stage

    Paramaters
    ----------

    params : ParamSet
        data_file : string
            path pointing to the hdf5 file containing the events
        sim_ver: string
            indicateing the sim version, wither 4digit, 5digit or dima
        bdt_cut : float
            futher cut apllied to events for the atm. muon rejections BDT

    Notes
    -----

    The curent versio of this code is a port from pisa v2 nutau branch.
    It clearly needs to be cleand up properly at some point.

    """

    def __init__(self, params, output_binning, disk_cache=None,
                memcache_deepcopy=True, error_method=None,
                outputs_cache_depth=20, debug_mode=None):

        expected_params = (
            'data_file',
            'sim_ver',
            'bdt_cut'
        )

        output_names = ('total')

        super().__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

    def _compute_nominal_outputs(self):
        """load the evnts from file, perform sanity checks and histogram them
        (into final MapSet)

        """
        # get params
        data_file_name = self.params.data_file.value
        sim_version = self.params.sim_ver.value
        bdt_cut = self.params.bdt_cut.value.m_as('dimensionless')

        self.bin_names = self.output_binning.names

        # TODO: convert units using e.g. `comp_units` in stages/reco/hist.py
        self.bin_edges = []
        for name in self.bin_names:
            if 'energy' in  name:
                bin_edges = self.output_binning[name].bin_edges.to('GeV').magnitude
            else:
                bin_edges = self.output_binning[name].bin_edges.magnitude
            self.bin_edges.append(bin_edges)

        # the rest of this function is PISA v2 legacy code...
        # right now only use burn sample with sim_version = '4digit'
        #print "sim_version == ", sim_version
        if sim_version == "4digit":
            Reco_Neutrino_Name = 'IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino'
            Reco_Track_Name = 'IC86_Dunkman_L6_MultiNest8D_PDG_Track'
        elif sim_version == "5digit" or sim_version=="dima":
            Reco_Neutrino_Name = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC'
            Reco_Track_Name = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_Track'
        else:
            raise ValueError(
                'only allow 4digit, 5digit(H2 model for hole ice) or'
                ' dima (dima p1 and p2 for hole ice)!'
            )

        data_file = h5py.File(find_resource(data_file_name), 'r')
        L6_result = np.array(data_file['IC86_Dunkman_L6']['result'])
        dLLH = np.array(data_file['IC86_Dunkman_L6']['delta_LLH'])
        reco_energy_all = np.array(data_file[Reco_Neutrino_Name]['energy'])
        reco_coszen_all = np.array(np.cos(
            data_file[Reco_Neutrino_Name]['zenith']
        ))
        reco_trck_len_all = np.array(data_file[Reco_Track_Name]['length'])
        #print "before L6 cut, no. of burn sample = ", len(reco_coszen_all)

        # sanity check
        santa_doms = data_file['IC86_Dunkman_L6_SANTA_DirectDOMs']['value']
        l3 = data_file['IC86_Dunkman_L3']['value']
        l4 = data_file['IC86_Dunkman_L4']['result']
        l5 = data_file['IC86_Dunkman_L5']['bdt_score']
        assert(np.all(santa_doms>=3) and np.all(l3 == 1) and np.all(l5 >= 0.1))

        # l4==1 was not applied when i3 files were written to hdf5 files, so do
        # it here
        dLLH = dLLH[l4==1]
        reco_energy_all = reco_energy_all[l4==1]
        reco_coszen_all = reco_coszen_all[l4==1]
        l5 = l5[l4==1]
        L6_result = L6_result[l4==1]
        data_file.close()

        dLLH_L6 = dLLH[L6_result==1]
        l5 = l5[L6_result==1]
        reco_energy_L6 = reco_energy_all[L6_result==1]
        reco_coszen_L6 = reco_coszen_all[L6_result==1]
        #print "after L6 cut, no. of burn sample = ", len(reco_coszen_L6)

        # Cut: Only keep bdt score >= 0.2 (from MSU latest result, make data/MC
        # agree much better); if use no such further cut, use bdt_cut = 0.1
        logging.info(
            "Cut2, removing events with bdt_score < %s i.e. only keep bdt > %s"
            %(bdt_cut, bdt_cut)
        )
        cut_events = {}
        cut = l5>=bdt_cut
        cut_events['reco_energy'] = reco_energy_L6[cut]
        cut_events['reco_coszen'] = reco_coszen_L6[cut]
        cut_events['pid'] = dLLH_L6[cut]

        hist, _ = np.histogramdd(sample = np.array(
            [cut_events[bin_name] for bin_name in self.bin_names]
        ).T, bins=self.bin_edges)

        maps = [Map(name=self.output_names[0], hist=hist,
                    binning=self.output_binning)]
        self.template = MapSet(maps, name='data')

    def _compute_outputs(self, inputs=None):
        """return the precomputed MpSets, since this is data, the distributions
        don't change

        """
        return self.template
