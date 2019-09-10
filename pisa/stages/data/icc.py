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


class icc(Stage):
    """
    Data loader stage

    Paramaters
    ----------
    params : ParamSet
        atm_muon_scale: float
            scale factor to be apllied to outputs
        icc_bg_file : string
            path pointing to the hdf5 file containing the events
        use_def1 : bool
            whether ICC definition 1 is used
        sim_ver: string
            indicating the sim version, wither 4digit, 5digit or dima
        livetime : time quantity
            livetime scale factor
        bdt_cut : float
            further cut applied to events for the atm. muon rejections BDT
        alt_icc_bg_file : string
            path pointing to an hdf5 file containing the events for an
        kde_hist : bool
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
            'use_def1',
            'sim_ver',
            'livetime',
            'bdt_cut',
            'alt_icc_bg_file',
            'kde_hist',
            'fixed_scale_factor'
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

        if self.params.kde_hist.value:
            from pisa.utils.kde_hist import kde_histogramdd
            self.kde_histogramdd = kde_histogramdd

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
        sim_ver = self.params.sim_ver.value
        use_def1 = self.params.use_def1.value
        bdt_cut = self.params.bdt_cut.m_as('dimensionless')

        self.bin_names = self.output_binning.names
        self.bin_edges = []
        for name in self.bin_names:
            if 'energy' in  name:
                bin_edges = self.output_binning[name].bin_edges.to('GeV').magnitude
            else:
                bin_edges = self.output_binning[name].bin_edges.magnitude
            self.bin_edges.append(bin_edges)

        # the rest of this function is PISA v2 legacy code...
        logging.info('Initializing BackgroundServiceICC...')
        logging.info('Opening file: %s'%(icc_bg_file))

        try:
            bg_file = h5py.File(find_resource(icc_bg_file),'r')
            if alt_icc_bg_file is not None:
                alt_bg_file = h5py.File(find_resource(alt_icc_bg_file),'r')
        except IOError as e:
            logging.error("Unable to open icc_bg_file %s"%icc_bg_file)
            logging.error(e)
            sys.exit(1)

        # sanity check
        santa_doms = bg_file['IC86_Dunkman_L6_SANTA_DirectDOMs']['value']
        l3 = bg_file['IC86_Dunkman_L3']['value']
        l4 = bg_file['IC86_Dunkman_L4']['result']
        l5 = bg_file['IC86_Dunkman_L5']['bdt_score']
        l6 = bg_file['IC86_Dunkman_L6']
        if use_def1:
            l4_pass = np.all(l4==1)
        else:
            if sim_ver in ['5digit', 'dima']:
                l4_invVICH = bg_file['IC86_Dunkman_L4']['result_invertedVICH']
                l4_pass = np.all(np.logical_or(l4==1, l4_invVICH==1))
            else:
                logging.info(
                    'For the old simulation, def.2 background not done yet,'
                    ' so still use def1 for it.'
                )
                l4_pass = np.all(l4==1)
        assert (np.all(santa_doms>=3) and np.all(l3 == 1) and l4_pass and
                np.all(l5 >= 0.1))
        corridor_doms_over_threshold = l6['corridor_doms_over_threshold']

        inverted_corridor_cut = corridor_doms_over_threshold > 1
        assert (np.all(inverted_corridor_cut) and
                np.all(l6['santa_direct_doms'] >= 3) and
                np.all(l6['mn_start_contained'] == 1.) and
                np.all(l6['mn_stop_contained'] == 1.))

        #load events
        if sim_ver == '4digit':
            variable ='IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino'
        elif sim_ver in ['5digit', 'dima']:
            variable = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC'
        else:
            raise ValueError('Only allow sim_ver  4digit, 5 digit or dima!')
        reco_energy_all = np.array(bg_file[variable]['energy'])
        reco_coszen_all = np.array(np.cos(bg_file[variable]['zenith']))
        pid_all = np.array(bg_file['IC86_Dunkman_L6']['delta_LLH'])
        if alt_icc_bg_file is not None:
            alt_reco_energy_all = np.array(alt_bg_file[variable]['energy'])
            alt_reco_coszen_all = np.array(np.cos(alt_bg_file[variable]['zenith']))
            alt_pid_all = np.array(alt_bg_file['IC86_Dunkman_L6']['delta_LLH'])
            alt_l5 = alt_bg_file['IC86_Dunkman_L5']['bdt_score']

        # Cut: Only keep bdt score >= 0.2 (from MSU latest result, make data/MC
        # agree much better)
        cut_events = {}
        cut = l5>=bdt_cut
        cut_events['reco_energy'] = reco_energy_all[cut]
        cut_events['reco_coszen'] = reco_coszen_all[cut]
        cut_events['pid'] = pid_all[cut]

        if alt_icc_bg_file is not None:
            # Cut: Only keep bdt score >= 0.2 (from MSU latest result, make
            # data/MC agree much better)
            alt_cut_events = {}
            alt_cut = alt_l5>=bdt_cut
            alt_cut_events['reco_energy'] = alt_reco_energy_all[alt_cut]
            alt_cut_events['reco_coszen'] = alt_reco_coszen_all[alt_cut]
            alt_cut_events['pid'] = alt_pid_all[alt_cut]

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
