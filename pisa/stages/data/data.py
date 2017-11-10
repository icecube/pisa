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
import copy
import pisa.utils.mcSimRunSettings as MCSRS
import pisa.utils.dataProcParams as DPP
from pisa.stages.osc.calc_layers import Layers


class data(Stage):
    """Data loader stage

    Paramaters
    ----------

    params : ParamSet
        data_file : string
            path pointing to the hdf5 file containing the events
        proc_ver: string
            indicateing the proc version, for example msu_5digit
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
            'proc_ver',
            'bdt_cut',
            'data_proc_file',
            'earth_model'
        )

        output_names = ('evts')

        super(self.__class__, self).__init__(
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

        # TODO: convert units using e.g. `comp_units` in stages/reco/hist.py
        self.bin_names = self.output_binning.names
        self.bin_edges = []
        for name in self.bin_names:
            if 'energy' in name:
                bin_edges = self.output_binning[name].bin_edges.to('GeV').magnitude
            else:
                bin_edges = self.output_binning[name].bin_edges.magnitude
            self.bin_edges.append(bin_edges)
        self.data_proc_file = self.params.data_proc_file.value

    def _compute_nominal_outputs(self):
        """load the evnts from file, perform sanity checks and histogram them
        (into final MapSet)

        """
        # get params
        data_file_name = self.params.data_file.value
        proc_version = self.params.proc_ver.value
        bdt_cut = self.params.bdt_cut.value.m_as('dimensionless')

        self.bin_names = self.output_binning.names

        # get data with cuts defined as 'analysis' in data_proc_params.json
        fields = ['reco_energy', 'pid', 'reco_coszen']
        cut_events = self.get_fields(fields,
                        cuts=['analysis'],
                        run_setting_file='events/mc_sim_run_settings.json')
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

    def get_fields(self, fields, cuts='analysis', run_setting_file='events/mc_sim_run_settings.json', no_reco=False):
        """ Return data events' fields

        Paramaters
        ----------
        fields: list of strings
            the quantities to return, for example: ['reco_energy', 'pid', 'reco_coszen']

        """
        # get param
        data_file_name = self.params.data_file.value
        proc_version = self.params.proc_ver.value
        bdt_cut = self.params.bdt_cut.value.m_as('dimensionless')
        data_proc_params = DPP.DataProcParams(
                detector='deepcore',
                proc_ver=proc_version,
                data_proc_params=find_resource(self.data_proc_file))
        run_settings = MCSRS.DetMCSimRunsSettings(find_resource(run_setting_file), detector='deepcore')
        data = data_proc_params.getData(find_resource(data_file_name), run_settings=run_settings, file_type='data')
        fields_for_cuts = copy.deepcopy(fields)
        for param in ['reco_energy', 'reco_coszen', 'pid']:
            if param not in fields:
                fields_for_cuts.append(param)
        # get fields not in data.keys() and will be added after applying cuts, e.g. 'l_over_e' and 'path_length'
        fields_add_later = []
        for param in fields:
            if param not in data.keys():
                fields_for_cuts.remove(param)
                fields_add_later.append(param)

        # bdt_score
        if 'dunkman_L5' in data.keys():
            fields_for_cuts.append('dunkman_L5')
        # get data after cuts
        cut_data = data_proc_params.applyCuts(data, cuts=cuts, return_fields=fields_for_cuts)
        # apply bdt_score cut if needed
        if cut_data.has_key('dunkman_L5') and bdt_cut is not None:
            all_cuts = cut_data['dunkman_L5']>=bdt_cut
            print "bdt_cut = ", bdt_cut
        else:
            for field in fields:
                len_cut_data = len(cut_data[field])
            all_cuts = np.ones(len_cut_data, dtype=bool)
        if no_reco==False:
            for bin_name, bin_edge in zip(self.bin_names, self.bin_edges):
                bin_cut = np.logical_and(cut_data[bin_name]<= bin_edge[-1], cut_data[bin_name]>= bin_edge[0])
                all_cuts = np.logical_and(all_cuts, bin_cut)

        # get fields_add_later
        if fields_add_later!=[]:
            for param in fields_add_later:
                # right now only works with the below 2 params
                assert(param in ['l_over_e', 'path_length'])
            layer = Layers(self.params.earth_model.value)
            cut_data['path_length'] = np.array([layer.DefinePath(reco_cz) for reco_cz in cut_data['reco_coszen']])
            if 'l_over_e' in fields_add_later:
                cut_data['l_over_e'] = cut_data['path_length']/cut_data['reco_energy']

        output_data = {}
        for key in fields:
            output_data[key] = cut_data[key][all_cuts]
        return output_data
