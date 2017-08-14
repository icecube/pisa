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


class icc(Stage):
    """
    Data loader stage

    Paramaters
    ----------
    params : ParamSet
        icc_bg_file : string
            path pointing to the hdf5 file containing the events
        proc_ver: string
            indicating the proc version, for example msu_4digit, msu_5digit
        bdt_cut : float
            further cut applied to events for the atm. muon rejections BDT
        livetime : time quantity
            livetime scale factor
        alt_icc_bg_file : string
            path pointing to an hdf5 file containing the events for an
            alternate selection/model, used to generate shape uncertainty terms
        atm_muon_scale: float
            scale factor to be apllied to outputs
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
            'atm_muon_scale',
            'icc_bg_file',
            'proc_ver',
            'livetime',
            'bdt_cut',
            'alt_icc_bg_file',
            'kde_hist',
            'fixed_scale_factor',
            'earth_model'
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


    def _compute_nominal_outputs(self):
        '''
        load events, perform sanity check and put them into histograms,
        if alt_bg file is specified, also put these events into separate histograms,
        that are normalized to the nominal ones (we are only interested in the shape difference)
        '''
        # get params
        icc_bg_file = self.params.icc_bg_file.value
        if 'shape' in self.error_method:
            alt_icc_bg_file = self.params.alt_icc_bg_file.value
        else:
            alt_icc_bg_file = None
        bdt_cut = self.params.bdt_cut.m_as('dimensionless')

        # get data with cuts defined as 'icc_def2' in data_proc_params.json
        fields = ['reco_energy', 'pid', 'reco_coszen']
        cut_events = self.get_fields(fields, icc_file_name = icc_bg_file,
                cuts='icc_def2',
                run_setting_file='events/mc_sim_run_settings.json',
                data_proc_file='events/data_proc_params.json')
        if alt_icc_bg_file is not None:
            alt_cut_events = self.get_fields(fields, icc_file_name = alt_icc_bg_file,
                    cuts='icc_def3',
                    run_setting_file='events/mc_sim_run_settings.json',
                    data_proc_file='events/data_proc_params.json')

        logging.info("Creating a ICC background hists...")
        # make histo
        if self.params.kde_hist.value:
            self.icc_bg_hist = self.kde_histogramdd(
                        np.array([cut_events[bin_name] for bin_name in self.bin_names]).T,
                        binning=self.output_binning,
                        coszen_name='reco_coszen',
                        use_cuda=True,
                        bw_method='silverman',
                        alpha=0.3,
                        oversample=10,
                        coszen_reflection=0.5,
                        adaptive=True
                    )
        else:
            self.icc_bg_hist,_ = np.histogramdd(sample = np.array([cut_events[bin_name] for bin_name in self.bin_names]).T, bins=self.bin_edges)


        conversion = self.params.atm_muon_scale.value.m_as('dimensionless') / ureg('common_year').to('seconds').m
        logging.info('nominal ICC rate at %.6E Hz'%(self.icc_bg_hist.sum()*conversion))

        if alt_icc_bg_file is not None:
            if self.params.kde_hist.value:
                self.alt_icc_bg_hist = self.kde_histogramdd(
                    np.array([alt_cut_events[bin_name] for bin_name in self.bin_names]).T,
                    binning=self.output_binning,
                    coszen_name='reco_coszen',
                    use_cuda=True,
                    bw_method='silverman',
                    alpha=0.3,
                    oversample=10,
                    coszen_reflection=0.5,
                    adaptive=True
                )
            else:
                self.alt_icc_bg_hist,_ = np.histogramdd(sample = np.array([alt_cut_events[bin_name] for bin_name in self.bin_names]).T, bins=self.bin_edges)
            # only interested in shape difference, not rate
            scale = 1
            if alt_icc_bg_file is not None:
                scale = self.icc_bg_hist.sum()/self.alt_icc_bg_hist.sum()
            self.alt_icc_bg_hist *= scale

    def _compute_outputs(self, inputs=None):
        """Apply scales to histograms, put them into PISA MapSets
        Also asign errors given a method:
            * sumw2 : just sum of weights quared as error (the usual weighte histo error)
            * sumw2+shae : including the shape difference
            * fixed_sumw2+shape : errors estimated from nominal paramter values, i.e. scale-invariant

        """

        scale = self.params.atm_muon_scale.value.m_as('dimensionless')
        fixed_scale = self.params.atm_muon_scale.nominal_value.m_as('dimensionless')
        scale *= self.params.livetime.value.m_as('common_year')
        fixed_scale *= self.params.livetime.value.m_as('common_year')
        fixed_scale *= self.params.fixed_scale_factor.value.m_as('dimensionless')

        if self.error_method == 'sumw2':
            maps = [Map(name=self.output_names[0], hist=(self.icc_bg_hist * scale), error_hist=(np.sqrt(self.icc_bg_hist) * scale) ,binning=self.output_binning)]
        elif self.error_method == 'sumw2+shape':
            error = scale * np.sqrt(self.icc_bg_hist + (self.icc_bg_hist - self.alt_icc_bg_hist)**2 )
            maps = [Map(name=self.output_names[0], hist=(self.icc_bg_hist * scale), error_hist=error ,binning=self.output_binning)]
        elif self.error_method == 'shape':
            error = scale * np.abs(self.icc_bg_hist - self.alt_icc_bg_hist)
        elif self.error_method == 'fixed_shape':
            error = fixed_scale * np.abs(self.icc_bg_hist - self.alt_icc_bg_hist)
            maps = [Map(name=self.output_names[0], hist=(self.icc_bg_hist * scale), error_hist=error ,binning=self.output_binning)]
        elif self.error_method == 'fixed_sumw2+shape':
            error = fixed_scale * np.sqrt(self.icc_bg_hist + (self.icc_bg_hist - self.alt_icc_bg_hist)**2 )
            maps = [Map(name=self.output_names[0], hist=(self.icc_bg_hist * scale), error_hist=error ,binning=self.output_binning)]
        elif self.error_method == 'fixed_doublesumw2+shape':
            error = fixed_scale * np.sqrt(2*self.icc_bg_hist + (self.icc_bg_hist - self.alt_icc_bg_hist)**2 )
            maps = [Map(name=self.output_names[0], hist=(self.icc_bg_hist * scale), error_hist=error ,binning=self.output_binning)]
        else:
            maps = [Map(name=self.output_names[0], hist=(self.icc_bg_hist * scale), binning=self.output_binning)]

        return MapSet(maps, name='icc')

    def get_fields(self, fields, icc_file_name, cuts='icc_def2', run_setting_file='events/mc_sim_run_settings.json',
                        data_proc_file='events/data_proc_params.json'):
        """ Return icc events' fields with the chosen icc background definition.

        Paramaters
        ----------
        fields: list of strings
            the quantities to return, for example: ['reco_energy', 'pid', 'reco_coszen']
        icc_file_name: string
            the icc hdf5 file name
        cuts: string
            definition for icc, for example: 'icc_def1', 'icc_def2', 'icc_def3', see their defs in data_proc_params.json

        """
        # get data
        proc_version = self.params.proc_ver.value
        bdt_cut = self.params.bdt_cut.value.m_as('dimensionless')
        data_proc_params = DPP.DataProcParams(
                detector='deepcore',
                proc_ver=proc_version,
                data_proc_params=find_resource(data_proc_file))
        run_settings = MCSRS.DetMCSimRunsSettings(find_resource(run_setting_file), detector='deepcore')
        data = data_proc_params.getData(find_resource(icc_file_name), run_settings=run_settings, file_type='data')

        # get fields that'll be used for applying cuts or fields that'll have cuts applied
        fields_for_cuts = copy.deepcopy(fields)
        for param in ['reco_energy', 'reco_coszen', 'pid']:
            if param not in fields:
                fields_for_cuts.append(param)
        if 'dunkman_L5' in data.keys():
            fields_for_cuts.append('dunkman_L5')

        # get fields not in data.keys() and will be added after applying cuts, e.g. 'l_over_e' and 'path_length'
        fields_add_later = []
        for param in fields:
            if param not in data.keys():
                fields_for_cuts.remove(param)
                fields_add_later.append(param)

        # apply cuts, defined in 'cuts', plus cuts on bins
        cut_data = data_proc_params.applyCuts(data, cuts=cuts, return_fields=fields_for_cuts)
        # apply bdt_score cut if needed
        if cut_data.has_key('dunkman_L5'):
            if bdt_cut is not None:
                bdt_score = cut_data['dunkman_L5']
                all_cuts = bdt_score>=bdt_cut
        else:
            all_cuts = np.ones(len(cut_data['reco_energy']), dtype=bool)
        for bin_name, bin_edge in zip(self.bin_names, self.bin_edges):
            bin_cut = np.logical_and(cut_data[bin_name]<= bin_edge[-1], cut_data[bin_name]>= bin_edge[0])
            all_cuts = np.logical_and(all_cuts, bin_cut)

        # get fields_add_later
        if len(fields_add_later)!=0:
            for param in fields_add_later:
                assert(param in ['l_over_e', 'path_length'])
            layer = Layers(self.params.earth_model.value)
            cut_data['path_length'] = np.array([layer.DefinePath(reco_cz) for reco_cz in cut_data['reco_coszen']])
            if 'l_over_e' in fields_add_later:
                cut_data['l_over_e'] = cut_data['path_length']/cut_data['reco_energy']

        return_data = {}
        for key in fields:
            return_data[key] = cut_data[key][all_cuts]
        # weight is just atm_muon_scale*livetime, will be needed for plotting
        scale = self.params.atm_muon_scale.value.m_as('dimensionless') * self.params.livetime.value.m_as('common_year')
        return_data['weight'] = scale
        return return_data
