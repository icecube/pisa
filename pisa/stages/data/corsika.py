import os
import sys

import h5py
import numpy as np

from pisa import ureg, Q_, FTYPE
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.core.stage import Stage
from pisa.utils.comparisons import normQuant
from pisa.utils.log import logging
from pisa.utils.resources import find_resource
import copy
import pisa.utils.mcSimRunSettings as MCSRS
import pisa.utils.dataProcParams as DPP
from pisa.stages.osc.calc_layers import Layers


class corsika(Stage):
    """
    Data loader stage

    Paramaters
    ----------
    params : ParamSet
        corsika_file : string
            path pointing to the hdf5 file containing the events
        proc_ver: string
            indicating the proc version, for example msu_4digit, msu_5digit
        livetime : time quantity
            livetime scale factor
        fixed_scale_factor : float
            scale fixed errors

    Notes
    -----
    The current version of this code is a port from pisa v2 nutau branch.
    It clearly needs to be cleaned up properly at some point.

    """
    def __init__(self, params, output_binning, disk_cache=None,
                memcache_deepcopy=True, error_method=None,
                outputs_cache_depth=20, debug_mode=None):

        expected_params = (
            'corsika_file',
            'corsika_scale',
            'proc_ver',
            'num_files',
            'livetime',
            'kde_hist',
            'fixed_scale_factor',
        )

        output_names = ('total')

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

        if self.params.kde_hist.value:
            from pisa.utils.kde_hist import kde_histogramdd
            self.kde_histogramdd = kde_histogramdd

        self.bin_names = self.output_binning.names
        self.bin_edges = []
        for name in self.bin_names:
            if 'energy' in  name:
                bin_edges = self.output_binning[name].bin_edges.to('GeV').magnitude
            else:
                bin_edges = self.output_binning[name].bin_edges.magnitude
            self.bin_edges.append(bin_edges)


    def _compute_nominal_outputs(self, no_reco=False):
        '''
        load events, perform sanity check and put them into histograms,
        if alt_bg file is specified, also put these events into separate histograms,
        that are normalized to the nominal ones (we are only interested in the shape difference)
        '''
        # get params
        corsika_file = self.params.corsika_file.value

        # get data with cuts defined as 'analysis' in data_proc_params.json
        fields = ['reco_energy', 'pid', 'reco_coszen', 'corsika_weight']
        cut_events = self.get_fields(fields, event_file = corsika_file,
                no_reco=no_reco,
                #cuts='analysis',
                cuts='level4_and_icc_def2',
                run_setting_file='events/mc_sim_run_settings.json',
                data_proc_file='events/data_proc_params.json')

        logging.info("Creating a CORSIKA background hists...")
        # make histo
        if self.params.kde_hist.value:
            self.corsika_bg_hist = self.kde_histogramdd(
                        np.array([cut_events[bin_name] for bin_name in self.bin_names]).T,
                        binning=self.output_binning,
                        weights=cut_events['weight'],
                        coszen_name='reco_coszen',
                        use_cuda=True,
                        bw_method='silverman',
                        alpha=0.3,
                        oversample=10,
                        coszen_reflection=0.5,
                        adaptive=True
                    )
        else:
            sample = np.array([cut_events[bin_name] for bin_name in self.bin_names]).T
            self.corsika_bg_hist,_ = np.histogramdd(sample = np.array([cut_events[bin_name] for bin_name in self.bin_names]).T, bins=self.bin_edges, weights=cut_events['weight'])

        conversion = self.params.corsika_scale.value.m_as('dimensionless') / ureg('common_year').to('seconds').m
        logging.info('nominal corsika rate at %.6E Hz'%(self.corsika_bg_hist.sum()*conversion))
        #print "self.corsika_bg_hist.sum()", self.corsika_bg_hist.sum()
        #print 'nominal corsika rate at %.6E Hz'%(self.corsika_bg_hist.sum()*conversion)


    def _compute_outputs(self, inputs=None):
        """Apply scales to histograms, put them into PISA MapSets
        Also asign errors given a method:
            * sumw2 : just sum of weights quared as error (the usual weighte histo error)
            * sumw2+shae : including the shape difference
            * fixed_sumw2+shape : errors estimated from nominal paramter values, i.e. scale-invariant

        """

        scale = self.params.corsika_scale.value.m_as('dimensionless')
        fixed_scale = self.params.corsika_scale.nominal_value.m_as('dimensionless')
        scale *= self.params.livetime.value.m_as('common_year')
        fixed_scale *= self.params.livetime.value.m_as('common_year')
        fixed_scale *= self.params.fixed_scale_factor.value.m_as('dimensionless')

        if self.error_method == 'sumw2':
            maps = [Map(name=self.output_names[0], hist=(self.corsika_bg_hist * scale), error_hist=(np.sqrt(self.corsika_bg_hist) * scale) ,binning=self.output_binning)]
        else:
            maps = [Map(name=self.output_names[0], hist=(self.corsika_bg_hist * scale), binning=self.output_binning)]

        return MapSet(maps, name='corsika')

    def get_fields(self, fields, event_file, no_reco=False, cuts='analysis', run_setting_file='events/mc_sim_run_settings.json',
                        data_proc_file='events/data_proc_params.json'):
        """ Return corsika events' fields with the chosen corsika background definition.

        Paramaters
        ----------
        fields: list of strings
            the quantities to return, for example: ['reco_energy', 'pid', 'reco_coszen']
        event_file: string
            the corsika hdf5 file name
        cuts: string
            cuts applied for corsika, for example: 'analysis'
        """
        # get data
        proc_version = self.params.proc_ver.value
        num_files = self.params.num_files.value
        data_proc_params = DPP.DataProcParams(
                detector='deepcore',
                proc_ver=proc_version,
                data_proc_params=find_resource(data_proc_file))
        run_settings = MCSRS.DetMCSimRunsSettings(find_resource(run_setting_file), detector='deepcore')
        data = data_proc_params.getData(find_resource(event_file), run_settings=run_settings, file_type='data')

        # get fields that'll be used for applying cuts or fields that'll have cuts applied
        fields_for_cuts = copy.deepcopy(fields)
        if no_reco==False:
            for param in ['reco_energy', 'reco_coszen', 'pid', 'corsika_weight']:
                if param not in fields:
                    fields_for_cuts.append(param)
        # apply cuts, defined in 'cuts', plus cuts on bins
        cut_data = data_proc_params.applyCuts(data, cuts=cuts, return_fields=fields_for_cuts)
        all_cuts = np.ones(len(cut_data['reco_energy']), dtype=bool)
        if no_reco==False:
            for bin_name, bin_edge in zip(self.bin_names, self.bin_edges):
                bin_cut = np.logical_and(cut_data[bin_name]<= bin_edge[-1], cut_data[bin_name]>= bin_edge[0])
                all_cuts = np.logical_and(all_cuts, bin_cut)

        output_data = {}
        output_fields = copy.deepcopy(fields)
        if 'weight' not in output_fields:
            output_fields.append('weight')
        for key in output_fields:
            if key=='weight':
                corsika_weight = cut_data['corsika_weight'][all_cuts]/num_files
                scale = self.params.corsika_scale.value.m_as('dimensionless') * self.params.livetime.value.m_as('common_year') * corsika_weight
                #scale = corsika_weight
                output_data['weight'] = scale
            else:
                output_data[key] = cut_data[key][all_cuts]
        return output_data
