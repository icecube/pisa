"""
The purpose of this stage is to apply background subtraction and
unfolding to the reconstructed variables.

This service in particular uses the RooUnfold implementation of Bayesian
unfolding.
"""

from __future__ import absolute_import

from copy import deepcopy
from functools import reduce
from operator import add

import numpy as np
from uncertainties import unumpy as unp
import pint

from ROOT import TH1
from ROOT import RooUnfoldResponse, RooUnfoldBayes
from root_numpy import array2hist, hist2array

from pisa import ureg
from pisa.core.stage import Stage
from pisa.core.events import Data
from pisa.core.pipeline import Pipeline
from pisa.core.map import Map, MapSet
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.rooutils import convert_to_th1d, convert_to_th2d
from pisa.utils.rooutils import unflatten_thist
from pisa.utils.flavInt import ALL_NUFLAVINTS, NuFlavIntGroup
from pisa.utils.fileio import from_file
from pisa.utils.random_numbers import get_random_state
from pisa.utils.comparisons import normQuant
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile


__author__ = 'S. Mandalia'

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


TH1.SetDefaultSumw2(False)


class roounfold(Stage):
    """TODO(shivesh): docstring"""
    def __init__(self, params, input_names, output_names, reco_binning,
                 true_binning, error_method=None, disk_cache=None,
                 outputs_cache_depth=20, memcache_deepcopy=True,
                 debug_mode=None):
        self.fit_hash = None
        """Hash of reweighted event sample."""
        self.random_state = None
        """Hash of random state."""
        self.response_hash = None
        """Hash of response object."""
        self.split_data_hash = None
        """Hash of data after it has been separated."""
        self.hist_hash = None
        """Hash of histogrammed objects."""
        self.gen_data_hash = None
        """Hash of generator level events."""
        self.inv_eff_hash = None
        """Hash of inverse efficiency histogram."""
        self.bg_hist_hash = None
        """Hash of background histogram for subtraction."""

        expected_params = (
            'real_data', 'unfold_pipeline_cfg', 'return_eff',
            'stat_fluctuations', 'regularisation', 'optimize_reg',
            'unfold_eff', 'unfold_bg', 'unfold_unweighted'
        )

        self.reco_binning = reco_binning
        self.true_binning = true_binning

        input_names = input_names.replace(' ', '').split(',')
        clean_innames = []
        for name in input_names:
            if 'muons' in name:
                clean_innames.append(name)
            elif 'noise' in name:
                clean_innames.append(name)
            elif 'all_nu' in name:
                clean_innames = [str(NuFlavIntGroup(f))
                                 for f in ALL_NUFLAVINTS]
            else:
                clean_innames.append(str(NuFlavIntGroup(name)))

        if len(reco_binning.names) != len(true_binning.names):
            raise AssertionError('Number of dimensions in reco binning '
                                 'doesn'+"'"+'t match number of dimensions in '
                                 'true binning')
        if len(reco_binning.names) != 2:
            raise NotImplementedError('Bin dimensions != 2 not implemented')

        signal = output_names.replace(' ', '').split(',')
        output_str = []
        for name in signal:
            if 'muons' in name or 'noise' in name:
                raise AssertionError('Are you trying to unfold muons/noise?')
            elif 'all_nu' in name:
                output_str = [str(NuFlavIntGroup(f)) for f in ALL_NUFLAVINTS]
            else:
                output_str.append(NuFlavIntGroup(name))
        output_str = [str(f) for f in output_str]

        super().__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            input_names=clean_innames,
            output_names=output_str,
            error_method=error_method,
            disk_cache=disk_cache,
            outputs_cache_depth=outputs_cache_depth,
            memcache_deepcopy=memcache_deepcopy,
            output_binning=true_binning,
            debug_mode=debug_mode
        )

        if disk_cache is not None:
            self.instantiate_disk_cache()

        self.include_attrs_for_hashes('random_state')

    @profile
    def _compute_outputs(self, inputs=None):
        """Compute histograms for output channels."""
        logging.debug('Entering roounfold._compute_outputs')
        self.fit_hash = deepcopy(inputs.metadata['fit_hash'])
        logging.trace('{0} roounfold fit_hash = '
                      '{1}'.format(inputs.metadata['name'], self.fit_hash))
        if self.random_state is not None:
            logging.trace(
                '{0} roounfold random_state = '
                '{1}'.format(inputs.metadata['name'],
                             hash_obj(self.random_state.get_state()))
            )
        if not isinstance(inputs, Data):
            raise AssertionError('inputs is not a Data object, instead is '
                                 'type {0}'.format(type(inputs)))
        self._data = inputs

        if not self.params['return_eff'].value:
            if len(self.output_names) > 1:
                raise AssertionError('Specified more than one NuFlavIntGroup as '
                                     'signal, {0}'.format(self.output_names))
            self.output_str = str(self.output_names[0])

        real_data = self.params['real_data'].value
        if real_data:
            logging.debug('Using real data')
            if 'nuall' not in self._data:
                raise AssertionError(
                    'When using real data, input Data object must contain '
                    'only one element "nuall" containing the data, instead it '
                    'contains elements {0}'.format(self._data.keys())
                )
            if self.disk_cache is None:
                raise AssertionError(
                    'No disk_cache specified from which to load - using real '
                    'data requires object such as the response object to be '
                    'cached to disk.'
                )

        if self.params['optimize_reg'].value and real_data:
            raise AssertionError(
                'Cannot optimize the regularation if using real data.'
            )
        if int(self.params['stat_fluctuations'].m) != 0 and real_data:
            raise AssertionError(
                'Cannot do poisson fluctuations if using real data.'
            )
        if self.params['return_eff'].value and real_data:
            raise AssertionError(
                'Not implemented return of efficiency maps if using real data.'
            )

        if self.params['return_eff'].value:
            fin_data = self._data
            # Load generator level data for signal
            unfold_pipeline_cfg = self.params['unfold_pipeline_cfg'].value
            pipeline_cfg = from_file(unfold_pipeline_cfg)
            template_maker = Pipeline(pipeline_cfg)
            gen_data = template_maker.get_outputs()

            fin_data = fin_data.transform_groups(self.output_names)
            gen_data = gen_data.transform_groups(self.output_names)

            efficiencies = []
            assert set(fin_data.keys()) == set(gen_data.keys())
            for fig in fin_data.keys():
                figd_f = fin_data[fig]
                figd_g = gen_data[fig]
                inv_eff = self._get_inv_eff(
                    figd_f, figd_g, self.true_binning, fig
                )

                i_mask = ~(inv_eff == 0.)
                eff = unp.uarray(np.zeros(self.true_binning.shape),
                                 np.zeros(self.true_binning.shape))
                eff[i_mask] = 1. / inv_eff[i_mask]
                efficiencies.append(
                    Map(name=fig, hist=eff, binning=self.true_binning)
                )
            return MapSet(efficiencies)

        # TODO(shivesh): [   TRACE] None of the selections ['iron', 'nh'] found in this pipeline.
        # TODO(shivesh): Fix "smearing_matrix" memory leak
        # TODO(shivesh): Fix unweighted unfolding
        # TODO(shivesh): different algorithms
        # TODO(shivesh): implement handling of 0 division inside Map objects
        if real_data:
            unfold_map = self.unfold_real_data()
        else:
            unfold_map = self.unfold_mc()

        return MapSet([unfold_map])

    def unfold_mc(self):
        logging.debug('Unfolding monte carlo sample')
        regularisation = int(self.params['regularisation'].m)
        unfold_bg = self.params['unfold_bg'].value
        unfold_eff = self.params['unfold_eff'].value
        unfold_unweighted = self.params['unfold_unweighted'].value

        # Split data into signal, bg and all (signal+bg)
        signal_data, bg_data, all_data = self.split_data()

        # Load generator level data for signal
        gen_data = self.load_gen_data()

        # Return true map is regularisation is set to 0
        if regularisation == 0:
            logging.info('Returning baseline MapSet')
            true = roounfold._histogram(
                events=gen_data,
                binning=self.true_binning,
                weights=gen_data['pisa_weight'],
                errors=True,
                name=self.output_str
            )
            return MapSet([true])

        # Get the inversed efficiency histogram
        if not unfold_eff:
            inv_eff = self.get_inv_eff(signal_data, gen_data)

        # Set the reco and true data based on cfg file settings
        reco_norm_data = None
        true_norm_data = None
        data = signal_data
        if unfold_bg:
            reco_norm_data = all_data
        if unfold_eff:
            true_norm_data = gen_data
        if reco_norm_data is None:
            reco_norm_data = signal_data
        if true_norm_data is None:
            true_norm_data = signal_data
        if unfold_unweighted:
            ones = np.ones(reco_norm_data['pisa_weight'].shape)
            reco_norm_data['pisa_weight'] = ones
            true_norm_data['pisa_weight'] = ones
            data['pisa_weight'] = ones

        # Create response object
        response = self.create_response(reco_norm_data, true_norm_data, data)

        # Make pseduodata
        all_hist = self._histogram(
            events=all_data,
            binning=self.reco_binning,
            weights=all_data['pisa_weight'],
            errors=False,
            name='all',
            tex=r'\rm{all}'
        )
        seed = int(self.params['stat_fluctuations'].m)
        if seed != 0:
            if self.random_state is None or seed != self.seed:
                self.seed = seed
                self.random_state = get_random_state(seed)
            all_hist = all_hist.fluctuate('poisson', self.random_state)
        else:
            self.seed = None
            self.random_state = None
        all_hist.set_poisson_errors()

        # Background Subtraction
        if unfold_bg:
            reco = all_hist
        else:
            bg_hist = self.get_bg_hist(bg_data)
            reco = all_hist - bg_hist
        reco.name = 'reco_signal'
        reco.tex = r'\rm{reco_signal}'

        r_flat = roounfold._flatten_to_1d(reco)
        r_th1d = convert_to_th1d(r_flat, errors=True)

        # Find optimum value for regularisation parameter
        if self.params['optimize_reg'].value:
            chisq = None
            for r_idx in range(regularisation):
                unfold = RooUnfoldBayes(
                    response, r_th1d, r_idx+1
                )
                unfold.SetVerbose(0)
                idx_chisq = unfold.Chi2(self.t_th1d, 1)
                if chisq is None:
                    pass
                elif idx_chisq > chisq:
                    regularisation = r_idx
                    break
                chisq = idx_chisq

        # Unfold
        unfold = RooUnfoldBayes(
            response, r_th1d, regularisation
        )
        unfold.SetVerbose(0)

        unfolded_flat = unfold.Hreco(1)
        unfold_map = unflatten_thist(
            in_th1d=unfolded_flat,
            binning=self.true_binning,
            name=self.output_str,
            errors=True
        )

        # Efficiency correction
        if not unfold_eff:
            unfold_map *= inv_eff

        del r_th1d
        del unfold
        logging.info('Unfolded reco sum {0}'.format(
            np.sum(unp.nominal_values(unfold_map.hist))
        ))
        return unfold_map

    def unfold_real_data(self):
        logging.info('Unfolding real data')
        regularisation = int(self.params['regularisation'].m)
        unfold_bg = self.params['unfold_bg'].value
        unfold_eff = self.params['unfold_eff'].value

        raw_data_0 = self._data['nuall']
        if regularisation == 0:
            raise AssertionError('Regularisation is set to 0')

        # Get the inversed efficiency histogram
        if not unfold_eff:
            inv_eff = self.get_inv_eff()

        # Load response object from disk cache
        response = self.create_response()

        # Background Subtraction
        if unfold_bg:
            raw_data_1 = raw_data_0
        else:
            bg_hist = self.get_bg_hist()
            raw_data_1 = raw_data_0 - bg_hist

        r_flat = roounfold._flatten_to_1d(raw_data_1)
        r_th1d = convert_to_th1d(r_flat, errors=True)

        # Unfold
        unfold = RooUnfoldBayes(
            response, r_th1d, regularisation
        )
        unfold.SetVerbose(0)

        unfolded_flat = unfold.Hreco(1)
        unfold_map = unflatten_thist(
            in_th1d=unfolded_flat,
            binning=self.true_binning,
            name=self.output_str,
            errors=True
        )

        # Efficiency correction
        if not unfold_eff:
            unfold_map *= inv_eff

        del r_th1d
        del unfold
        logging.info('Unfolded reco sum {0}'.format(
            np.sum(unp.nominal_values(unfold_map.hist))
        ))
        return unfold_map

    def split_data(self):
        this_hash = hash_obj(
            [self.fit_hash, self.output_str, self._data.contains_muons,
             self._data.contains_noise], full_hash = self.full_hash
        )
        if self.split_data_hash == this_hash:
            return self._signal_data, self._bg_data, self._all_data

        if self.params['real_data'].value:
            return self._data, None, self._data

        trans_data = self._data.transform_groups(self.output_str)
        [trans_data[fig].pop('sample_weight') for fig in trans_data]
        bg_str = [fig for fig in trans_data if fig != self.output_str]
        if trans_data.contains_muons:
            trans_data['muons'].pop('sample_weight')
            bg_str.append('muons')
        if trans_data.contains_noise:
            trans_data['noise'].pop('sample_weight')
            bg_str.append('noise')

        signal_data = trans_data[self.output_str]
        bg_data = [trans_data[bg] for bg in bg_str]
        bg_data = reduce(Data._merge, bg_data)
        all_data = Data._merge(deepcopy(bg_data), signal_data)

        self._signal_data = signal_data
        self._bg_data = bg_data
        self._all_data = all_data
        self.split_data_hash = this_hash
        return signal_data, bg_data, all_data

    def load_gen_data(self):
        logging.debug('Loading generator level sample')
        unfold_pipeline_cfg = self.params['unfold_pipeline_cfg'].value
        if isinstance(unfold_pipeline_cfg, str):
            pipeline_cfg = from_file(unfold_pipeline_cfg)
            pipeline_hash = pipeline_cfg
            sa_cfg = from_file(
                pipeline_cfg.get('stage.data', 'param.data_sample_config')
            )
            template_maker = Pipeline(pipeline_cfg)
        elif isinstance(unfold_pipeline_cfg, Pipeline):
            pipeline_hash = unfold_pipeline_cfg.state_hash
            sa_cfg = from_file(
                unfold_pipeline_cfg.params['data_sample_config'].value
            )
            template_maker = unfold_pipeline_cfg
        gen_cfg = from_file(sa_cfg.get('neutrinos|gen_lvl', 'gen_cfg_file'))
        this_hash = hash_obj([gen_cfg, pipeline_hash, self.output_str],
                             full_hash = self.full_hash)
        if self.gen_data_hash == this_hash:
            return self._gen_data

        full_gen_data = template_maker.get_outputs()
        if not isinstance(full_gen_data, Data):
            raise AssertionError(
                'Output of pipeline is not a Data object, instead is type '
                '{0}'.format(type(full_gen_data))
            )
        trans_data = full_gen_data.transform_groups(self.output_str)
        gen_data = trans_data[self.output_str]

        self._gen_data = gen_data
        self.gen_data_hash = this_hash
        return gen_data

    def get_inv_eff(self, signal_data=None, gen_data=None):
        this_hash = hash_obj(
            [self.true_binning.hash, self.output_str, 'inv_eff'],
            full_hash = self.full_hash
        )
        assert len(set([signal_data is None, gen_data is None])) == 1
        if signal_data is None and gen_data is None:
            if self.inv_eff_hash == this_hash:
                logging.trace('Loading inv eff from mem cache')
                return self._inv_eff
            if this_hash in self.disk_cache:
                logging.debug('Loading inv eff histogram from disk cache.')
                inv_eff = self.disk_cache[this_hash]
            else:
                raise ValueError(
                    'inverse efficiency histogram with correct hash not found '
                    'in disk_cache'
                )
        else:
            this_hash = hash_obj(
                [this_hash, self.fit_hash], full_hash = self.full_hash
            )
            if self.inv_eff_hash == this_hash:
                logging.trace('Loading inv eff from mem cache')
                return self._inv_eff
            inv_eff = self._get_inv_eff(
                signal_data, gen_data, self.true_binning, self.output_str
            )

            if self.disk_cache is not None:
                if this_hash not in self.disk_cache:
                    logging.debug('Caching inv eff histogram to disk.')
                    self.disk_cache[this_hash] = inv_eff

        self.inv_eff_hash = this_hash
        self._inv_eff = inv_eff
        return inv_eff

    def create_response(self, reco_norm_data=None, true_norm_data=None, data=None):
        """Create the response object from the signal data."""
        unfold_bg = self.params['unfold_bg'].value
        unfold_eff = self.params['unfold_eff'].value
        unfold_unweighted = self.params['unfold_unweighted'].value
        this_hash = hash_obj(
            [self.reco_binning.hash, self.true_binning.hash, unfold_bg,
             unfold_eff, unfold_unweighted, self.output_str, 'response'],
            full_hash = self.full_hash
        )
        assert len(set([reco_norm_data is None,
                        true_norm_data is None,
                        data is None])) == 1
        if reco_norm_data is None and true_norm_data is None and data is None:
            if self.response_hash == this_hash:
                logging.trace('Loading response from mem cache')
                return self._response
            else:
                try:
                    del self._response
                except:
                    pass
            if this_hash in self.disk_cache:
                logging.debug('Loading response from disk cache.')
                response = self.disk_cache[this_hash]
            else:
                raise ValueError(
                    'response object with correct hash not found in disk_cache'
                )
        else:
            this_hash = hash_obj(
                [this_hash, self.fit_hash] + list(self.params.values),
                full_hash = self.full_hash
            )
            if self.response_hash == this_hash:
                logging.debug('Loading response from mem cache')
                return self._response
            else:
                try:
                    del self._response
                    del self.t_th1d
                except:
                    pass

            # Truth histogram also gets returned if response matrix is created
            response, self.t_th1d = self._create_response(
                reco_norm_data, true_norm_data, data, self.reco_binning,
                self.true_binning
            )

            if self.disk_cache is not None:
                if this_hash not in self.disk_cache:
                    logging.debug('Caching response object to disk.')
                    self.disk_cache[this_hash] = response

        self.response_hash = this_hash
        self._response = response
        return response

    def get_bg_hist(self, bg_data=None):
        """Histogram the bg data unless using real data, in which case load
        the bg hist from disk cache."""
        this_hash = hash_obj(
            [self.reco_binning.hash, self.output_str, 'bg_hist'],
            full_hash = self.full_hash
        )
        if bg_data is None:
            if self.bg_hist_hash == this_hash:
                logging.trace('Loading bg hist from mem cache')
                return self._bg_hist
            if this_hash in self.disk_cache:
                logging.debug('Loading bg hist from disk cache.')
                bg_hist = self.disk_cache[this_hash]
            else:
                raise ValueError(
                    'bg hist object with correct hash not found in disk_cache'
                )
        else:
            this_hash = hash_obj(
                [this_hash, self.fit_hash], full_hash = self.full_hash
            )
            if self.bg_hist_hash == this_hash:
                logging.trace('Loading bg hist from mem cache')
                return self._bg_hist
            bg_hist = self._histogram(
                events=bg_data,
                binning=self.reco_binning,
                weights=bg_data['pisa_weight'],
                errors=True,
                name='background',
                tex=r'\rm{background}'
            )

            if self.disk_cache is not None:
                if this_hash not in self.disk_cache:
                    logging.debug('Caching bg hist to disk.')
                    self.disk_cache[this_hash] = bg_hist

        self.bg_hist_hash = this_hash
        self._bg_hist = bg_hist
        return bg_hist

    @staticmethod
    def _get_inv_eff(signal_data, gen_data, binning, name):
        signal_map = roounfold._histogram(
            events=signal_data,
            binning=binning,
            weights=signal_data['pisa_weight'],
            errors=True,
            name=name
        )
        gen_map = roounfold._histogram(
            events=gen_data,
            binning=binning,
            weights=gen_data['pisa_weight'],
            errors=True,
            name='generator_lvl',
            tex=r'\rm{generator_lvl}'
        )
        i_mask = ~(signal_map.hist == 0.)
        inv_eff = unp.uarray(np.zeros(signal_map.hist.shape),
                             np.zeros(signal_map.hist.shape))
        inv_eff[i_mask] = gen_map.hist[i_mask] / signal_map.hist[i_mask]

        return inv_eff

    @staticmethod
    def _create_response(reco_norm_data, true_norm_data, data, reco_binning,
                         true_binning):
        """Create the response object from the signal data."""
        logging.debug('Creating response object.')

        reco_hist = roounfold._histogram(
            events=reco_norm_data,
            binning=reco_binning,
            weights=reco_norm_data['pisa_weight'],
            errors=True,
            name='reco_signal',
            tex=r'\rm{reco_signal}'
        )
        true_hist = roounfold._histogram(
            events=true_norm_data,
            binning=true_binning,
            weights=true_norm_data['pisa_weight'],
            errors=True,
            name='true_signal',
            tex=r'\rm{true_signal}'
        )
        r_flat = roounfold._flatten_to_1d(reco_hist)
        t_flat = roounfold._flatten_to_1d(true_hist)

        smear_matrix = roounfold._histogram(
            events=data,
            binning=reco_binning+true_binning,
            weights=data['pisa_weight'],
            errors=True,
            name='smearing_matrix',
            tex=r'\rm{smearing_matrix}'
        )
        smear_flat = roounfold._flatten_to_2d(smear_matrix)

        r_th1d = convert_to_th1d(r_flat, errors=True)
        t_th1d = convert_to_th1d(t_flat, errors=True)
        smear_th2d = convert_to_th2d(smear_flat, errors=True)

        response = RooUnfoldResponse(r_th1d, t_th1d, smear_th2d)
        del r_th1d
        del smear_th2d
        return response, t_th1d

    @staticmethod
    def _histogram(events, binning, weights=None, errors=False, **kwargs):
        """Histogram the events given the input binning."""
        logging.trace('Histogramming')

        bin_names = binning.names
        bin_edges = [edges.m for edges in binning.bin_edges]
        for name in bin_names:
            if name not in events:
                raise AssertionError('Input events object does not have '
                                     'key {0}'.format(name))

        sample = [events[colname] for colname in bin_names]
        hist, edges = np.histogramdd(
            sample=sample, weights=weights, bins=bin_edges
        )
        if errors:
            hist2, edges = np.histogramdd(
                sample=sample, weights=np.square(weights), bins=bin_edges
            )
            hist = unp.uarray(hist, np.sqrt(hist2))

        return Map(hist=hist, binning=binning, **kwargs)

    @staticmethod
    def _flatten_to_1d(in_map):
        assert isinstance(in_map, Map)

        bin_name = reduce(add, in_map.binning.names)
        num_bins = np.product(in_map.shape)
        binning = MultiDimBinning([OneDimBinning(
            name=bin_name, num_bins=num_bins, is_lin=True, domain=[0, num_bins]
        )])
        hist = in_map.hist.flatten()

        return Map(name=in_map.name, hist=hist, binning=binning)

    @staticmethod
    def _flatten_to_2d(in_map):
        assert isinstance(in_map, Map)
        shape = in_map.shape
        names = in_map.binning.names
        dims = len(shape)
        assert dims % 2 == 0

        nbins_a = np.product(shape[:dims//2])
        nbins_b = np.product(shape[dims//2:])
        names_a = reduce(lambda x, y: x+' '+y, names[:dims//2])
        names_b = reduce(lambda x, y: x+' '+y, names[dims//2:])

        binning = []
        binning.append(OneDimBinning(
            name=names_a, num_bins=nbins_a, is_lin=True, domain=[0, nbins_a]
        ))
        binning.append(OneDimBinning(
            name=names_b, num_bins=nbins_b, is_lin=True, domain=[0, nbins_b]
        ))
        binning = MultiDimBinning(binning)

        hist = in_map.hist.reshape(nbins_a, nbins_b)
        return Map(name=in_map.name, hist=hist, binning=binning)

    def validate_params(self, params):
        pq = pint.quantity._Quantity
        param_types = [
            ('real_data', bool),
            ('return_eff', bool),
            ('stat_fluctuations', pq),
            ('regularisation', pq),
            ('optimize_reg', bool),
            ('unfold_eff', bool),
            ('unfold_bg', bool),
            ('unfold_unweighted', bool)
        ]
        for p, t in param_types:
            val = params[p].value
            if not isinstance(val, t):
                raise TypeError(
                    'Param "%s" must be type %s but is %s instead'
                    % (p, type(t), type(val))
                )
        assert isinstance(params['unfold_pipeline_cfg'].value, str) or \
                isinstance(params['unfold_pipeline_cfg'].value, Pipeline)
