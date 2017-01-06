"""
This stage loads event information of simulations which have been
generated using Monte Carlo techniques.

This service in particular reads in from files having a similar
convention to the low energy event samples. More information about these
event samples can be found on
https://wiki.icecube.wisc.edu/index.php/IC86_Tau_Appearance_Analysis
https://wiki.icecube.wisc.edu/index.php/IC86_oscillations_event_selection
"""


from operator import add

import numpy as np

from pisa import ureg, Q_
from pisa.core.stage import Stage
from pisa.core.events import Data
from pisa.core.map import MapSet
from pisa.utils.flavInt import ALL_NUFLAVINTS, NuFlavIntGroup, FlavIntDataGroup
from pisa.utils.fileio import from_file
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile


__all__ = ['sample']


class sample(Stage):
    """data service to load in events from an event sample.

    Parameters
    ----------
    params: ParamSet of sequence with which to instantiate a ParamSet
        Parameters which set everything besides the binning

        Parameters required by this service are
            * data_sample_config : filepath
                Filepath to event sample configuration

            * dataset : string
                Pick which systematic set to use (or nominal)
                examples: 'nominal', 'neutrinos:dom_eff:1.05', 'muons:hole_ice:0.01'
                the nominal set will be used for the event types not specified

            * keep_criteria : None or string
                Apply a cut such as the only events which satisfy
                `keep_criteria` are kept.
                Any string interpretable as numpy boolean expression.

            * output_events_data : bool
                Flag to specify whether the service output returns a MapSet
                or the full information about each event

    output_binning : MultiDimBinning or convertible thereto
        The binning desired for the output maps.

    output_names : string
        Specifies the string representation of the NuFlavIntGroup(s) which will
        be produced as an output.

    error_method : None, bool, or string
        If None, False, or empty string, the stage does not compute errors for
        the transforms and does not apply any (additional) error to produce its
        outputs. (If the inputs already have errors, these are propagated.)

    debug_mode : None, bool, or string
        If None, False, or empty string, the stage runs normally.
        Otherwise, the stage runs in debug mode. This disables caching (forcing
        recomputation of any nominal transforms, transforms, and outputs).

    transforms_cache_depth
    outputs_cache_depth : int >= 0

    """
    def __init__(self, params, output_binning, output_names, error_method=None,
                 debug_mode=None, disk_cache=None, memcache_deepcopy=True,
                 transforms_cache_depth=20, outputs_cache_depth=20):
        self.sample_hash = None
        """Hash of event sample"""

        expected_params = (
            'data_sample_config', 'dataset', 'keep_criteria',
            'output_events_data'
        )

        self.neutrinos = False
        self.muons = False
        self.noise = False

        output_names = output_names.replace(' ', '').split(',')
        clean_outnames = []
        self._output_nu_groups = []
        for name in output_names:
            if 'muons' in name:
                self.muons = True
                clean_outnames.append(name)
            elif 'noise' in name:
                self.noise = True
                clean_outnames.append(name)
            elif 'all_nu' in name:
                self.neutrinos = True
                self._output_nu_groups = \
                    [NuFlavIntGroup(f) for f in ALL_NUFLAVINTS]
            else:
                self.neutrinos = True
                self._output_nu_groups.append(NuFlavIntGroup(name))

        if self.neutrinos:
            clean_outnames += [str(f) for f in self._output_nu_groups]

        super(self.__class__, self).__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            output_names=clean_outnames,
            error_method=error_method,
            debug_mode=debug_mode,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            output_binning=output_binning
        )

        self._compute_outputs()

    @profile
    def _compute_outputs(self, inputs=None):
        """Apply basic cuts and compute histograms for output channels."""
        logging.debug('Entering sample._compute_outputs')
        self.config = from_file(self.params['data_sample_config'].value)
        name = self.config.get('general', 'name')
        logging.trace('{0} sample sample_hash = '
                      '{1}'.format(name, self.sample_hash))
        self.load_sample_events()

        if self.params['keep_criteria'].value is not None:
            # TODO(shivesh)
            raise NotImplementedError(
                'needs check to make sure this works in a DistributionMaker'
            )
            self._data.applyCut(self.params['keep_criteria'].value)
            self._data.update_hash()

        if self.params['output_events_data'].value:
            return self._data

        outputs = []
        if self.neutrinos:
            trans_nu_data = self._data.transform_groups(
                self._output_nu_groups
            )
            for fig in trans_nu_data.iterkeys():
                outputs.append(trans_nu_data.histogram(
                    kinds       = fig,
                    binning     = self.output_binning,
                    weights_col = 'pisa_weight',
                    errors      = True,
                    name        = str(NuFlavIntGroup(fig)),
                ))

        if self.muons:
            outputs.append(self._data.histogram(
                kinds       = 'muons',
                binning     = self.output_binning,
                weights_col = 'pisa_weight',
                errors      = True,
                name        = 'muons',
                tex         = r'\rm{muons}'
            ))

        name = self.config.get('general', 'name')
        return MapSet(maps=outputs, name=name)

    def load_sample_events(self):
        """Load the event sample given the configuration file and output
        groups. Hash this object using both the configuration file and
        the output types."""
        hash_property = [self.config, self.neutrinos, self.muons,
                         self.params['dataset'].value]
        this_hash = hash_obj(hash_property, full_hash=self.full_hash)
        if this_hash == self.sample_hash:
            return

        name = self.config.get('general', 'name')
        def parse(string):
            return string.replace(' ', '').split(',')
        event_types = parse(self.config.get('general', 'event_type'))

        events = []
        if self.neutrinos:
            if 'neutrinos' not in event_types:
                raise AssertionError('`neutrinos` field not found in '
                                     'configuration file.')
            dataset = self.params['dataset'].value.lower()
            if 'neutrinos' not in dataset:
                dataset = 'nominal'
            nu_data = self.load_neutrino_events(
                config=self.config, dataset=dataset
            )
            events.append(nu_data)

        if self.muons:
            if 'muons' not in event_types:
                raise AssertionError('`muons` field not found in '
                                     'configuration file.')
            dataset = self.params['dataset'].value
            if 'muons' not in dataset:
                dataset = 'nominal'
            muon_events = self.load_muon_events(
                config=self.config, dataset=dataset
            )
            events.append(muon_events)
        self._data = reduce(add, events)

        self.sample_hash = this_hash
        self._data.metadata['sample_hash'] = this_hash
        self._data.update_hash()

    @staticmethod
    def load_neutrino_events(config, dataset):
        def parse(string):
            return string.replace(' ', '').split(',')

        nu_data = []
        if dataset == 'neutrinos:gen_lvl':
            gen_cfg      = from_file(config.get(dataset, 'gen_cfg_file'))
            name         = gen_cfg.get('general', 'name')
            datadir      = gen_cfg.get('general', 'datadir')
            event_types  = parse(gen_cfg.get('general', 'event_type'))
            weights      = parse(gen_cfg.get('general', 'weights'))
            weight_units = gen_cfg.get('general', 'weight_units')
            keep_keys    = parse(gen_cfg.get('general', 'keep_keys'))
            logging.info('Extracting dataset "{0}" from generator level '
                         'sample "{1}"'.format(dataset, name))

            for idx, flav in enumerate(event_types):
                fig = NuFlavIntGroup(flav)
                all_flavints = fig.flavints()
                events_file = datadir + gen_cfg.get(flav, 'filename')

                flav_fidg = sample.load_from_nu_file(
                    events_file, all_flavints, weights[idx], weight_units,
                    keep_keys
                )
                nu_data.append(flav_fidg)
        else:
            name         = config.get('general', 'name')
            flavours     = parse(config.get('neutrinos', 'flavours'))
            weights      = parse(config.get('neutrinos', 'weights'))
            weight_units = config.get('neutrinos', 'weight_units')
            sys_list     = parse(config.get('neutrinos', 'sys_list'))
            base_prefix  = config.get('neutrinos', 'baseprefix')
            keep_keys    = parse(config.get('neutrinos', 'keep_keys'))
            logging.info('Extracting dataset "{0}" from generator level '
                         'sample "{1}"'.format(dataset, name))
            if base_prefix == 'None':
                base_prefix = ''

            for idx, flav in enumerate(flavours):
                f = int(flav)
                all_flavints = NuFlavIntGroup(f, -f).flavints()
                if dataset == 'nominal':
                    prefixes = []
                    for sys in sys_list:
                        ev_sys = 'neutrinos:' + sys
                        nominal = config.get(ev_sys, 'nominal')
                        ev_sys_nom = ev_sys + ':' + nominal
                        prefixes.append(config.get(ev_sys_nom, 'file_prefix'))
                    if len(set(prefixes)) > 1:
                        raise AssertionError(
                            'Choice of nominal file is ambigous. Nominal '
                            'choice of systematic parameters must coincide '
                            'with one and only one file. Options found are: '
                            '{0}'.format(prefixes)
                        )
                    file_prefix = flav + prefixes[0]
                else:
                    file_prefix = flav + config.get(dataset, 'file_prefix')
                events_file = config.get('general', 'datadir') + \
                    base_prefix + file_prefix

                flav_fidg = sample.load_from_nu_file(
                    events_file, all_flavints, weights[idx], weight_units,
                    keep_keys
                )
                nu_data.append(flav_fidg)
        nu_data = Data(
            reduce(add, nu_data),
            metadata={'name': name, 'sample': dataset}
        )

        return nu_data

    @staticmethod
    def load_muon_events(config, dataset):
        def parse(string):
            return string.replace(' ', '').split(',')
        name         = config.get('general', 'name')
        weight       = config.get('muons', 'weight')
        weight_units = config.get('muons', 'weight_units')
        sys_list     = parse(config.get('muons', 'sys_list'))
        base_prefix  = config.get('muons', 'baseprefix')
        if base_prefix == 'None':
            base_prefix = ''

        if dataset == 'nominal':
            paths = []
            for sys in sys_list:
                ev_sys = 'muons:' + sys
                nominal = config.get(ev_sys, 'nominal')
                ev_sys_nom = ev_sys + ':' + nominal
                paths.append(config.get(ev_sys_nom, 'file_path'))
            if len(set(paths)) > 1:
                raise AssertionError(
                    'Choice of nominal file is ambigous. Nominal '
                    'choice of systematic parameters must coincide '
                    'with one and only one file. Options found are: '
                    '{0}'.format(paths)
                )
            file_path = paths[0]
        else:
            file_path = config.get(dataset, 'file_path')

        muons = from_file(file_path)

        if weight == 'None' or weight == '1':
            muons['pisa_weight'] = np.ones(muons['weights'].shape)
        elif weight == '0':
            muons['pisa_weight'] = np.zeros(muons['weights'].shape)
        else:
            muons['pisa_weight'] = muons[weight] * ureg(weight_units)

        # TODO(shivesh): implement alias feature in cfg file
        if 'zenith' in muons and 'coszen' not in muons:
            muons['coszen'] = np.cos(muons['zenith'])
        if 'reco_zenith' in muons and 'reco_coszen' not in muons:
            muons['reco_coszen'] = np.cos(muons['reco_zenith'])

        muon_dict = {'muons': muons}
        return Data(muon_dict, metadata={'name': name, 'mu_sample': dataset})

    @staticmethod
    def load_from_nu_file(events_file, all_flavints, weight, weight_units,
                          keep_keys):
        flav_fidg = FlavIntDataGroup(flavint_groups=all_flavints)

        events = from_file(events_file)
        if keep_keys == 'all':
            pass
        else:
            remove_keys = []
            for k_key in keep_keys:
                if k_key not in events:
                    raise AssertionError(
                        'Key "{0}" not found in the {1} file, which '
                        'contains keys {2}'.format(
                            k_key, events_file, events.keys()
                        )
                    )
            for d_key in events.iterkeys():
                if d_key not in keep_keys:
                    remove_keys.append(d_key)
            for r_k in remove_keys:
                del events[r_k]

        nu_mask = events['ptype'] > 0
        nubar_mask = events['ptype'] < 0
        cc_mask = events['interaction'] == 1
        nc_mask = events['interaction'] == 2

        if weight == 'None' or weight == '1':
            events['pisa_weight'] = \
                np.ones(events['ptype'].shape) * ureg.dimensionless
        elif weight == '0':
            events['pisa_weight'] = \
                np.zeros(events['ptype'].shape) * ureg.dimensionless
        else:
            events['pisa_weight'] = events[weight] * \
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
        return flav_fidg

    def validate_params(self, params):
        assert isinstance(params['data_sample_config'].value, basestring)
        assert isinstance(params['dataset'].value, basestring)
        assert params['keep_criteria'].value is None or \
            isinstance(params['keep_criteria'].value, basestring)
        assert isinstance(params['output_events_data'].value, bool)
