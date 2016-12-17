"""
The purpose of this stage is to apply background subtraction and
unfolding to the reconstructed variables.

This service in particular uses the RooUnfold implementation of Bayesian
unfolding.
"""
from operator import add
from copy import deepcopy
from itertools import product

import numpy as np
import pint
from uncertainties import unumpy as unp

from ROOT import TH1, TH1D, TH2D
from ROOT import RooUnfoldResponse, RooUnfoldBayes
from root_numpy import array2hist, hist2array
TH1.SetDefaultSumw2(False)

from pisa import ureg, Q_
from pisa.core.stage import Stage
from pisa.core.events import Data
from pisa.core.map import Map, MapSet
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.flavInt import NuFlavIntGroup, ALL_NUFLAVINTS
from pisa.utils.random_numbers import get_random_state
from pisa.utils.comparisons import normQuant
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile


class roounfold(Stage):
    """TODO(shivesh): docstring"""
    def __init__(self, params, input_names, output_names, reco_binning,
                 true_binning, error_method=None, disk_cache=None,
                 outputs_cache_depth=20, memcache_deepcopy=True,
                 debug_mode=None):
        self.sample_hash = None
        """Hash of input event sample."""
        self.random_state = None
        """Hash of random state."""
        self.response_hash = None
        """Hash of response object."""

        expected_params = (
            'create_response', 'stat_fluctuations', 'regularisation',
            'optimize_reg', 'unfold_bg', 'unfold_unweighted'
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

        signal = output_names.replace(' ', '').split(',')
        self._output_nu_group = []
        for name in signal:
            if 'muons' in name or 'noise' in name:
                raise AssertionError('Are you trying to unfold muons/noise?')
            else:
                self._output_nu_group.append(NuFlavIntGroup(name))

        if len(self._output_nu_group) > 1:
            raise AssertionError('Specified more than one NuFlavIntGroup as '
                                 'signal, {0}'.format(self._output_nu_group))
        self._output_nu_group = str(self._output_nu_group[0])

        if len(reco_binning.names) != len(true_binning.names):
            raise AssertionError('Number of dimensions in reco binning '
                                 'doesn'+"'"+'t match number of dimensions in '
                                 'true binning')
        if len(reco_binning.names) != 2:
            raise NotImplementedError('Bin dimensions != 2 not implemented')

        self.data_hash = None

        super(self.__class__, self).__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            input_names=clean_innames,
            output_names=self._output_nu_group,
            error_method=error_method,
            disk_cache=disk_cache,
            outputs_cache_depth=outputs_cache_depth,
            memcache_deepcopy=memcache_deepcopy,
            output_binning=true_binning,
            debug_mode=debug_mode
        )

        if disk_cache is not None:
            self.instantiate_disk_cache()

        self.include_attrs_for_hashes('sample_hash')
        self.include_attrs_for_hashes('random_state')

    @profile
    def _compute_outputs(self, inputs=None):
        """Compute histograms for output channels."""
        if not isinstance(inputs, Data):
            raise AssertionError('inputs is not a Data object, instead is '
                                 'type {0}'.format(type(inputs)))
        self.sample_hash = inputs.hash
        self._data = deepcopy(inputs)

        if not self.params['create_response'].value \
           and self.disk_cache is None:
            raise AssertionError('No disk_cache specified from which to load '
                                 'response object.')

        if self.params['optimize_reg'].value and \
           not self.params['create_response'].value:
            raise AssertionError('`create_response` must be set to True if '
                                 'the flag `optimize_reg` is set to True.')

        # TODO(shivesh): [   TRACE] None of the selections ['iron', 'nh'] found in this pipeline.
        # TODO(shivesh): Fix "smearing_matrix" memory leak
        # TODO(shivesh): Fix unweighted unfolding
        # TODO(shivesh): real data
        # TODO(shivesh): different algorithms
        # TODO(shivesh): efficiency correction in unfolding
        self.signal_data, self.background_data, self.all_data = \
            self.transform_data(self._data)

        response = self.create_response(self.signal_data, self.all_data)

        all_hist = self._histogram(
            events=self.all_data,
            binning=self.reco_binning,
            weights=self.all_data['pisa_weight'],
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

        if self.params['unfold_bg'].value:
            sig_reco = deepcopy(all_hist)
        else:
            bg_hist = self._histogram(
                events=self.background_data,
                binning=self.reco_binning,
                weights=self.background_data['pisa_weight'],
                errors=True,
                name='background',
                tex=r'\rm{background}'
            )
            sig_reco = all_hist - bg_hist
        sig_reco.name = 'reco_signal'
        sig_reco.tex = r'\rm{reco_signal}'

        sig_r_flat = roounfold._flatten_to_1d(sig_reco)
        sig_r_th1d = roounfold._convert_to_th1d(sig_r_flat, errors=True)

        regularisation = int(self.params['regularisation'].m)
        if regularisation == 0:
            sig_true = roounfold._histogram(
                events=self.signal_data,
                binning=self.true_binning,
                weights=self.signal_data['pisa_weight'],
                errors=True,
                name=self._output_nu_group
            )
            return MapSet([sig_true])

        if self.params['optimize_reg'].value:
            chisq = None
            for r_idx in xrange(regularisation):
                unfold = RooUnfoldBayes(
                    response, sig_r_th1d, r_idx+1
                )
                unfold.SetVerbose(0)
                idx_chisq = unfold.Chi2(self.sig_t_th1d, 1)
                if chisq is None:
                    pass
                elif idx_chisq > chisq:
                    regularisation = r_idx
                    break
                chisq = idx_chisq

        unfold = RooUnfoldBayes(
            response, sig_r_th1d, regularisation
        )
        unfold.SetVerbose(0)

        sig_unfolded_flat = unfold.Hreco(1)
        sig_unfold = self._unflatten_thist(
            in_th1d=sig_unfolded_flat,
            binning=self.true_binning,
            name=self._output_nu_group,
            errors=True
        )

        del sig_r_th1d
        del unfold
        logging.info('Unfolded reco sum {0}'.format(
            np.sum(unp.nominal_values(sig_unfold.hist))
        ))
        return MapSet([sig_unfold])

    def transform_data(self, data):
        this_hash = self.sample_hash
        if self.data_hash == this_hash:
            return self.signal_data, self.background_data, self.all_data
        trans_data = self._data.transform_groups(
            self._output_nu_group
        )

        background_str = [fig for fig in trans_data
                          if fig != self._output_nu_group]
        if trans_data.contains_muons:
            background_str.append('muons')

        signal_data = trans_data[self._output_nu_group]
        background_data = [trans_data[bg] for bg in background_str]
        background_data = reduce(Data._merge, background_data)
        all_data = Data._merge(
            deepcopy(background_data), signal_data
        )

        self.data_hash = self.sample_hash
        return signal_data, background_data, all_data

    def create_response(self, signal_data, all_data):
        """Create the response object from the signal data."""
        this_hash = hash_obj(normQuant(self.params))
        if self.response_hash == this_hash:
            return self._response
        else:
            try:
                del self._response
                del self.sig_t_th1d
            except:
                pass

        unfold_bg = self.params['unfold_bg'].value
        unfold_unweighted = self.params['unfold_unweighted'].value
        if self.params['create_response'].value:
            # Truth histogram gets returned if response matrix is created
            response, self.sig_t_th1d = self._create_response(
                signal_data, all_data, self.reco_binning, self.true_binning,
                unfold_bg, unfold_unweighted
            )
        else:
            # Cache based on binning, output names and event sample hash
            cache_params = [self.reco_binning, self.true_binning,
                            self.output_names, self._data.hash, unfold_bg]
            this_cache_hash = hash_obj(cache_params)

            if this_cache_hash in self.disk_cache:
                logging.info('Loading response object from cache.')
                response = self.disk_cache[this_cache_hash]
            else:
                raise ValueError('response object with correct hash not found '
                                 'in disk_cache')

        if self.disk_cache is not None:
            # Cache based on binning, output names and event sample hash
            cache_params = [self.reco_binning, self.true_binning,
                            self.output_names, self._data.hash]
            this_cache_hash = hash_obj(cache_params)
            if this_cache_hash not in self.disk_cache:
                logging.info('Caching response object to disk.')
                self.disk_cache[this_cache_hash] = response

        self.response_hash = this_hash
        self._response = response
        return response

    @staticmethod
    def _create_response(signal_data, all_data, reco_binning, true_binning,
                         unfold_bg, unfold_unweighted):
        """Create the response object from the signal data."""
        logging.debug('Creating response object.')

        if unfold_bg:
            reco_data = all_data
        else:
            reco_data = signal_data

        if unfold_unweighted:
            sig_weights = np.ones(signal_data['pisa_weight'].shape)
            reco_weights = np.ones(reco_data['pisa_weight'].shape)
        else:
            sig_weights = signal_data['pisa_weight']
            reco_weights = reco_data['pisa_weight']

        sig_reco = roounfold._histogram(
            events=reco_data,
            binning=reco_binning,
            weights=reco_weights,
            errors=True,
            name='reco_signal',
            tex=r'\rm{reco_signal}'
        )
        sig_true = roounfold._histogram(
            events=signal_data,
            binning=true_binning,
            weights=sig_weights,
            errors=True,
            name='true_signal',
            tex=r'\rm{true_signal}'
        )
        sig_r_flat = roounfold._flatten_to_1d(sig_reco)
        sig_t_flat = roounfold._flatten_to_1d(sig_true)

        smear_matrix = roounfold._histogram(
            events=signal_data,
            binning=reco_binning+true_binning,
            weights=sig_weights,
            errors=True,
            name='smearing_matrix',
            tex=r'\rm{smearing_matrix}'
        )
        smear_flat = roounfold._flatten_to_2d(smear_matrix)

        sig_r_th1d = roounfold._convert_to_th1d(sig_r_flat, errors=True)
        sig_t_th1d = roounfold._convert_to_th1d(sig_t_flat, errors=True)
        smear_th2d = roounfold._convert_to_th2d(smear_flat, errors=True)

        response = RooUnfoldResponse(sig_r_th1d, sig_t_th1d, smear_th2d)
        del sig_r_th1d
        del smear_th2d
        return response, sig_t_th1d

    @staticmethod
    def _histogram(events, binning, weights=None, errors=False, **kwargs):
        """Histogram the events given the input binning."""
        logging.debug('Histogramming')

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

        nbins_a = np.product(shape[:dims/2])
        nbins_b = np.product(shape[dims/2:])
        names_a = reduce(lambda x, y: x+' '+y, names[:dims/2])
        names_b = reduce(lambda x, y: x+' '+y, names[dims/2:])

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

    @staticmethod
    def _convert_to_th1d(in_map, errors=False):
        assert isinstance(in_map, Map)
        name = in_map.name
        assert len(in_map.shape) == 1
        n_bins = in_map.shape[0]
        edges = in_map.binning.bin_edges[0].m

        th1d = TH1D(name, name, n_bins, edges)
        array2hist(unp.nominal_values(in_map.hist), th1d)
        if errors:
            map_errors = unp.std_devs(in_map.hist)
            for idx in xrange(n_bins):
                th1d.SetBinError(idx+1, map_errors[idx])
        return th1d

    @staticmethod
    def _convert_to_th2d(in_map, errors=False):
        assert isinstance(in_map, Map)
        name = in_map.name
        n_bins = in_map.shape
        assert len(n_bins) == 2
        nbins_a, nbins_b = n_bins
        edges_a, edges_b = [b.m for b in in_map.binning.bin_edges]

        th2d = TH2D(name, name, nbins_a, edges_a, nbins_b, edges_b)
        array2hist(unp.nominal_values(in_map.hist), th2d)
        if errors:
            map_errors = unp.std_devs(in_map.hist)
            for x_idx, y_idx in product(*map(range, n_bins)):
                th2d.SetBinError(x_idx+1, y_idx+1, map_errors[x_idx][y_idx])
        return th2d

    @staticmethod
    def _unflatten_thist(in_th1d, binning, name='', errors=False, **kwargs):
        flat_hist = hist2array(in_th1d)
        if errors:
            map_errors = [in_th1d.GetBinError(idx+1)
                          for idx in xrange(len(flat_hist))]
            flat_hist = unp.uarray(flat_hist, map_errors)
        hist = flat_hist.reshape(binning.shape)
        return Map(hist=hist, binning=binning, name=name, **kwargs)

    def validate_params(self, params):
        pq = pint.quantity._Quantity
        assert isinstance(params['create_response'].value, bool)
        assert isinstance(params['stat_fluctuations'].value, pq)
        assert isinstance(params['regularisation'].value, pq)
        assert isinstance(params['optimize_reg'].value, bool)
        assert isinstance(params['unfold_bg'].value, bool)
        assert isinstance(params['unfold_unweighted'].value, bool)
