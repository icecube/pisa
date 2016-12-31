"""
The purpose of this stage is to apply background subtraction and
unfolding to the reconstructed variables.

This service in particular uses the RooUnfold implementation of Bayesian
unfolding.
"""
from operator import add
from copy import deepcopy

import numpy as np
import pint
from uncertainties import unumpy as unp

from ROOT import TH1
from ROOT import RooUnfoldResponse, RooUnfoldBayes
TH1.SetDefaultSumw2(False)

from pisa import ureg, Q_
from pisa.core.stage import Stage
from pisa.core.events import Data
from pisa.core.map import Map, MapSet
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.rooutils import convert_to_th1d, convert_to_th2d
from pisa.utils.rooutils import unflatten_thist
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
            'optimize_reg', 'unfold_eff', 'unfold_bg', 'unfold_unweighted'
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
        self.hist_hash = None

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
        signal_data, bg_data, all_data = self.split_data()

        # Return true map is regularisation is set to 0
        regularisation = int(self.params['regularisation'].m)
        if regularisation == 0:
            true = roounfold._histogram(
                events=signal_data,
                binning=self.true_binning,
                weights=signal_data['pisa_weight'],
                errors=True,
                name=self._output_nu_group
            )
            return MapSet([true])

        # Set the reco and true data based on cfg file settings
        unfold_eff = self.params['unfold_eff'].value
        unfold_bg = self.params['unfold_bg'].value
        unfold_unweighted = self.params['unfold_unweighted'].value
        reco_data = signal_data
        true_data = signal_data
        if unfold_bg:
            reco_data = all_data
        if unfold_eff:
            # TODO(shivesh)
            raise NotImplementedError
        if unfold_unweighted:
            reco_data = deepcopy(reco_data)
            true_data = deepcopy(true_data)
            reco_data['pisa_weight'] = np.ones(reco_data['pisa_weight'].shape)
            true_data['pisa_weight'] = np.ones(true_data['pisa_weight'].shape)

        # Create response object
        self.create_response(reco_data, true_data)

        # Make pseduodata
        all_hist = self._histogram(
            events=all_data,
            binning=self.reco_binning,
            weights=all_data['pisa_weight'],
            errors=False,
            name='all',
            tex=r'\rm{all}'
        )
        all_hist_poisson = deepcopy(all_hist)
        seed = int(self.params['stat_fluctuations'].m)
        if seed != 0:
            if self.random_state is None or seed != self.seed:
                self.seed = seed
                self.random_state = get_random_state(seed)
            all_hist_poisson = all_hist_poisson.fluctuate(
                'poisson', self.random_state
            )
        else:
            self.seed = None
            self.random_state = None
        all_hist_poisson.set_poisson_errors()

        # Background Subtraction
        if unfold_bg:
            reco = deepcopy(all_hist_poisson)
        else:
            bg_hist = self._histogram(
                events=bg_data,
                binning=self.reco_binning,
                weights=bg_data['pisa_weight'],
                errors=True,
                name='background',
                tex=r'\rm{background}'
            )
            reco = all_hist_poisson - bg_hist
        reco.name = 'reco_signal'
        reco.tex = r'\rm{reco_signal}'

        r_flat = roounfold._flatten_to_1d(reco)
        r_th1d = convert_to_th1d(r_flat, errors=True)

        if self.params['optimize_reg'].value:
            chisq = None
            for r_idx in xrange(regularisation):
                unfold = RooUnfoldBayes(
                    self.response, r_th1d, r_idx+1
                )
                unfold.SetVerbose(0)
                idx_chisq = unfold.Chi2(self.t_th1d, 1)
                if chisq is None:
                    pass
                elif idx_chisq > chisq:
                    regularisation = r_idx
                    break
                chisq = idx_chisq

        unfold = RooUnfoldBayes(
            self.response, r_th1d, regularisation
        )
        unfold.SetVerbose(0)

        unfolded_flat = unfold.Hreco(1)
        unfold_map = unflatten_thist(
            in_th1d=unfolded_flat,
            binning=self.true_binning,
            name=self._output_nu_group,
            errors=True
        )

        del r_th1d
        del unfold
        logging.info('Unfolded reco sum {0}'.format(
            np.sum(unp.nominal_values(unfold_map.hist))
        ))
        return MapSet([unfold_map])

    def split_data(self):
        this_hash = hash_obj(
            [self.sample_hash, self._output_nu_group,
             self._data.contains_muons]
        )
        if self.data_hash == this_hash:
            return self._signal_data, self._bg_data, self._all_data

        trans_data = self._data.transform_groups(
            self._output_nu_group
        )
        bg_str = [fig for fig in trans_data
                          if fig != self._output_nu_group]
        if trans_data.contains_muons:
            bg_str.append('muons')

        signal_data = trans_data[self._output_nu_group]
        bg_data = [trans_data[bg] for bg in bg_str]
        bg_data = reduce(Data._merge, bg_data)
        all_data = Data._merge(deepcopy(bg_data), signal_data)

        self._signal_data = signal_data
        self._bg_data = bg_data
        self._all_data = all_data
        self.data_hash = this_hash
        return signal_data, bg_data, all_data

    def create_response(self, reco_data, true_data):
        """Create the response object from the signal data."""
        this_hash = hash_obj(normQuant(self.params))
        if self.response_hash == this_hash:
            return self.response
        else:
            try:
                del self.response
                del self.t_th1d
            except:
                pass

        if self.params['create_response'].value:
            # Truth histogram gets returned if response matrix is created
            response, self.t_th1d = self._create_response(
                reco_data, true_data, self.reco_binning, self.true_binning
            )
        else:
            # Cache based on binning, output names and event sample hash
            cache_params = [self.reco_binning, self.true_binning,
                            self.output_names, self._data.hash]
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
        self.response = response

    @staticmethod
    def _create_response(reco_data, true_data, reco_binning, true_binning):
        """Create the response object from the signal data."""
        logging.debug('Creating response object.')

        reco_hist = roounfold._histogram(
            events=reco_data,
            binning=reco_binning,
            weights=reco_data['pisa_weight'],
            errors=True,
            name='reco_signal',
            tex=r'\rm{reco_signal}'
        )
        true_hist = roounfold._histogram(
            events=true_data,
            binning=true_binning,
            weights=true_data['pisa_weight'],
            errors=True,
            name='true_signal',
            tex=r'\rm{true_signal}'
        )
        r_flat = roounfold._flatten_to_1d(reco_hist)
        t_flat = roounfold._flatten_to_1d(true_hist)

        smear_matrix = roounfold._histogram(
            events=true_data,
            binning=reco_binning+true_binning,
            weights=true_data['pisa_weight'],
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

    def validate_params(self, params):
        pq = pint.quantity._Quantity
        assert isinstance(params['create_response'].value, bool)
        assert isinstance(params['stat_fluctuations'].value, pq)
        assert isinstance(params['regularisation'].value, pq)
        assert isinstance(params['optimize_reg'].value, bool)
        assert isinstance(params['unfold_eff'].value, bool)
        assert isinstance(params['unfold_bg'].value, bool)
        assert isinstance(params['unfold_unweighted'].value, bool)
