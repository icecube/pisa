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
from pisa.utils.flavInt import ALL_NUFLAVINTS, NuFlavIntGroup, FlavIntDataGroup
from pisa.utils.fileio import from_file
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
        self.split_data_hash = None
        """Hash of data after it has been separated."""
        self.hist_hash = None
        """Hash of histogrammed objects."""
        self.bg_hist_hash = None
        """Hash of background histogram for subtraction."""
        self.gen_data_hash = None
        """Hash of generator level events."""
        self.flux_hash = None
        """Hash of calculated flux values for generator level events."""

        expected_params = (
            'real_data', 'gen_cfg_file', 'flux_file',
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

        signal = output_names.replace(' ', '').split(',')
        self.output_str = []
        for name in signal:
            if 'muons' in name or 'noise' in name:
                raise AssertionError('Are you trying to unfold muons/noise?')
            else:
                self.output_str.append(NuFlavIntGroup(name))

        if len(self.output_str) > 1:
            raise AssertionError('Specified more than one NuFlavIntGroup as '
                                 'signal, {0}'.format(self.output_str))
        self.output_str = str(self.output_str[0])

        if len(reco_binning.names) != len(true_binning.names):
            raise AssertionError('Number of dimensions in reco binning '
                                 'doesn'+"'"+'t match number of dimensions in '
                                 'true binning')
        if len(reco_binning.names) != 2:
            raise NotImplementedError('Bin dimensions != 2 not implemented')

        super(self.__class__, self).__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            input_names=clean_innames,
            output_names=self.output_str,
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

        real_data = self.params['real_data'].value
        if real_data and self.disk_cache is None:
            raise AssertionError('No disk_cache specified from which to load '
                                 'response object.')

        if self.params['optimize_reg'].value and real_data:
            raise AssertionError('`real_data` must be set to False if '
                                 'the flag `optimize_reg` is set to True.')

        # TODO(shivesh): [   TRACE] None of the selections ['iron', 'nh'] found in this pipeline.
        # TODO(shivesh): Fix "smearing_matrix" memory leak
        # TODO(shivesh): Fix unweighted unfolding
        # TODO(shivesh): real data
        # TODO(shivesh): need to cache bg subtraction so it can be loaded when using real data
        # TODO(shivesh): different algorithms
        # TODO(shivesh): efficiency correction
        # TODO(shivesh): more logging
        signal_data, bg_data, all_data = self.split_data()

        # Load generator level data

        # Return true map is regularisation is set to 0
        regularisation = int(self.params['regularisation'].m)
        if regularisation == 0:
            true = roounfold._histogram(
                events=signal_data,
                binning=self.true_binning,
                weights=signal_data['pisa_weight'],
                errors=True,
                name=self.output_str
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

        # Create response object or load from disk cache
        response = self.create_response(reco_data, true_data)

        # Make pseduodata (if not using real data)
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
            if real_data:
                logging.warn(
                    '`stat_fluctuations` seed is set to a non-zero value '
                    'while `real_data` is True. Will not be applying any stat '
                    'fluctuations to the real data.'
                )
            else:
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

        if self.params['optimize_reg'].value:
            chisq = None
            for r_idx in xrange(regularisation):
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

        del r_th1d
        del unfold
        logging.info('Unfolded reco sum {0}'.format(
            np.sum(unp.nominal_values(unfold_map.hist))
        ))
        return MapSet([unfold_map])

    def split_data(self):
        this_hash = hash_obj(
            [self.sample_hash, self.output_str, self._data.contains_muons]
        )
        if self.split_data_hash == this_hash:
            return self._signal_data, self._bg_data, self._all_data

        if self.params['real_data'].value:
            return self._data, None, self._data

        trans_data = self._data.transform_groups(self.output_str)
        bg_str = [fig for fig in trans_data if fig != self.output_str]
        if trans_data.contains_muons:
            bg_str.append('muons')

        signal_data = trans_data[self.output_str]
        bg_data = [trans_data[bg] for bg in bg_str]
        bg_data = reduce(Data._merge, bg_data)
        all_data = Data._merge(deepcopy(bg_data), signal_data)

        self._signal_data = signal_data
        self._bg_data = bg_data
        self._all_data = all_data
        self.split_data_hash = this_hash
        return signal_data, bg_data, all_data

    # TODO(shivesh): load this here, or in sample.py??
    def load_gen_data(self):
        config = from_file(self.params['gen_cfg_file'].value)
        flux_file = self.params['flux_file'].value
        this_hash = hash_obj([config, flux_file])
        if self.gen_data_hash == this_hash:
            return self._gen_data

        gen_data = roounfold._load_gen_data(config)


    def create_response(self, reco_data, true_data):
        """Create the response object from the signal data."""
        this_hash = hash_obj(
            [self.sample_hash, normQuant(self.params), self.reco_binning.hash,
             self.true_binning.hash, self.output_str]
        )
        if self.response_hash == this_hash:
            return self._response
        else:
            try:
                del self._response
                del self.t_th1d
            except:
                pass

        if self.params['real_data'].value:
            if this_hash in self.disk_cache:
                logging.debug('Loading response object from cache.')
                response = self.disk_cache[this_hash]
            else:
                raise ValueError(
                    'response object with correct hash not found in disk_cache'
                )
        else:
            # Truth histogram also gets returned if response matrix is created
            response, self.t_th1d = self._create_response(
                reco_data, true_data, self.reco_binning, self.true_binning
            )

        if self.disk_cache is not None:
            if this_hash not in self.disk_cache:
                logging.debug('Caching response object to disk.')
                self.disk_cache[this_hash] = response

        self.response_hash = this_hash
        self._response = response

    def get_bg_hist(self, bg_data):
        """Histogram the bg data unless using real data, in which case load
        the bg hist from disk cache."""
        this_hash = hash_obj(
            [self.sample_hash, self.reco_binning.hash, self.output_str]
        )
        if self.bg_hist_hash == this_hash:
            return self._bg_hist
        if self.params['real_data'].value:
            if this_hash in self.disk_cache:
                logging.debug('Loading bg histogram from cache.')
                bg_hist = self.disk_cache[this_hash]
            else:
                raise ValueError(
                    'bg histogram with correct hash not found in disk_cache'
                )
        else:
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
                logging.debug('Caching bg histogram to disk.')
                self.disk_cache[this_hash] = bg_hist

        self.bg_hist_hash = this_hash
        self._bg_hist = bg_hist
        return bg_hist

    @staticmethod
    def _load_gen_data(config):
        name = config.get('general', 'name')
        logging.info('Extracting events from generator level sample '
                     '"{0}"'.format(name))
        def parse(string):
            return string.replace(' ', '').split(',')
        datadir = config.get('general', 'datadir')
        event_types = parse(config.get('general', 'event_type'))
        weights = parse(config.get('general', 'weights'))
        weight_units = config.get('general', 'weight_units')

        nu_data = []
        for idx, flav in enumerate(event_types):
            fig = NuFlavIntGroup(flav)
            all_flavints = fig.flavints()
            flav_fidg = FlavIntDataGroup(flavint_groups=fig)
            filepath = datadir + config.get(flav, 'filename')

            events = from_file(filepath)
            nu_mask = events['ptype'] > 0
            nubar_mask = events['ptype'] < 0
            cc_mask = events['interaction'] == 1
            nc_mask = events['interaction'] == 2

            if weights[idx] == 'None' or weights[idx] == '1':
                events['pisa_weight'] = \
                    np.ones(events['ptype'].shape) * ureg.dimensionless
            elif weights[idx] == '0':
                events['pisa_weight'] = \
                    np.zeros(events['ptype'].shape) * ureg.dimensionless
            else:
                events['pisa_weight'] = events[weights[idx]] * \
                        ureg(weight_units)

            if 'zenith' in events and 'coszen' not in events:
                events['coszen'] = np.cos(events['zenith'])
            if 'reco_zenith' in events and 'reco_coszen' not in events:
                events['reco_coszen'] = np.cos(events['reco_zenith'])

            for flavint in all_flavints:
                i_mask = cc_mask if flavint.isCC() else nc_mask
                t_mask = nu_mask if flavint.isParticle() else nubar_mask

                flav_fidg[flavint] = {var: events[var][i_mask & t_mask]
                                      for var in events.iterkeys()}
            nu_data.append(flav_fidg)
        nu_data = Data(
            reduce(add, nu_data),
            metadata={'name': name, 'sample': 'generator level'}
        )

        return nu_data


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
        assert isinstance(params['real_data'].value, bool)
        assert isinstance(params['gen_cfg_file'].value, basestring)
        assert isinstance(params['flux_file'].value, basestring)
        assert isinstance(params['stat_fluctuations'].value, pq)
        assert isinstance(params['regularisation'].value, pq)
        assert isinstance(params['optimize_reg'].value, bool)
        assert isinstance(params['unfold_eff'].value, bool)
        assert isinstance(params['unfold_bg'].value, bool)
        assert isinstance(params['unfold_unweighted'].value, bool)
