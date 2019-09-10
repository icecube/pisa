"""
This stage loads event information of simulations which have been
generated using Monte Carlo techniques.

This service in particular reads in from files having a similar
convention to the low energy event samples. More information about these
event samples can be found on
https://wiki.icecube.wisc.edu/index.php/IC86_Tau_Appearance_Analysis
https://wiki.icecube.wisc.edu/index.php/IC86_oscillations_event_selection
"""


from __future__ import absolute_import

from os import path
from copy import deepcopy
from functools import reduce
from operator import add
import re

import numpy as np

from pisa import ureg
from pisa.core.events import Data
from pisa.core.map import MapSet
from pisa.core.stage import Stage
from pisa.utils.fileio import from_file
from pisa.utils.flavInt import ALL_NUFLAVINTS, NuFlavIntGroup, FlavIntDataGroup
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile


__all__ = ['SEP', 'sample','split','parse_event_type_names']

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


SEP = '|'



#TODO Try to merge with pisa.utils.config_parser.split, but currently gives problems due to the lower case forcing
def split(string):
    return string.replace(' ', '').split(',')


#Function for parsing event type names from the config file
#Handle any aliases here, such as 'all_nu'
def parse_event_type_names(names,return_flags=False) :

    #Split into list if has not already been done
    if isinstance(names, str):
        names = split(names)

    #Parse the names
    parsed_names = []
    for name in names :
        if 'all_nu' in name:
            parsed_names.extend( [str(NuFlavIntGroup(f)) for f in ALL_NUFLAVINTS] )
        else :
            parsed_names.append(name)
    parsed_names = [ n.lower() for n in parsed_names ]

    #Set some flags
    muons = False
    noise = False
    neutrinos = False
    for name in parsed_names:
        if 'muons' in name:
            muons = True
        elif 'noise' in name:
            noise = True
        elif name.startswith("nu"):
            neutrinos = True
        else :
            raise ValueError("Unrecognised event type '%s' found"%name)

    if return_flags : return parsed_names,muons,noise,neutrinos
    else : return parsed_names


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
                examples: 'nominal', 'neutrinos|dom_eff|1.05', 'muons|hole_ice|0.01'
                the nominal set will be used for the event types not specified

            * keep_criteria : None or string
                Apply a cut such as the only events which satisfy
                `keep_criteria` are kept.
                Any string interpretable as numpy boolean expression.

    output_binning : MultiDimBinning or convertible thereto
        The binning desired for the output maps.

    output_names : string
        Specifies the string representation of the NuFlavIntGroup(s) which will
        be produced as an output.

    output_events : bool
        Flag to specify whether the service output returns a MapSet
        or the full information about each event

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
    def __init__(self, params, output_binning, output_names,
                 output_events=True, error_method=None, debug_mode=None,
                 disk_cache=None, memcache_deepcopy=True,
                 transforms_cache_depth=20, outputs_cache_depth=20,
                 fix_truth_variable_names=False):
        self.sample_hash = None
        """Hash of event sample"""

        expected_params = (
            'data_sample_config', 'dataset', 'keep_criteria',
        )

        output_names,self.muons,self.noise,self.neutrinos = parse_event_type_names(output_names,return_flags=True)
        self._output_nu_groups = [ NuFlavIntGroup(name) for name in output_names ]

        if not isinstance(output_events, bool):
            raise AssertionError(
                'output_events must be of type bool, instead it is supplied '
                'with type {0}'.format(type(output_events))
            )
        if output_events:
            output_binning = None
        self.output_events = output_events

        super().__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            output_names=output_names,
            error_method=error_method,
            debug_mode=debug_mode,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            output_binning=output_binning,
        )

        #User can specify that truth variables have their names prefixed with "truth_"
        self.fix_truth_variable_names = fix_truth_variable_names
        self.truth_variables = ["energy","coszen"]
        self.truth_variable_prefix = "true_"

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

        if self.output_events:
            return self._data

        outputs = []
        if self.neutrinos:
            trans_nu_data = self._data.transform_groups(
                self._output_nu_groups
            )
            for fig in trans_nu_data.keys():
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

        if self.noise:
            outputs.append(self._data.histogram(
                kinds       = 'noise',
                binning     = self.output_binning,
                weights_col = 'pisa_weight',
                errors      = True,
                name        = 'noise',
                tex         = r'\rm{noise}'
            ))

        name = self.config.get('general', 'name')
        return MapSet(maps=outputs, name=name)

    def load_sample_events(self):
        """Load the event sample given the configuration file and output
        groups. Hash this object using both the configuration file and
        the output types."""
        hash_property = [self.config, self.neutrinos, self.muons, self.noise,
                         self.params['dataset'].value]
        this_hash = hash_obj(hash_property, full_hash=self.full_hash)
        if this_hash == self.sample_hash:
            return

        name = self.config.get('general', 'name')
        event_types = split(self.config.get('general', 'event_type'))

        logging.info( "Event types in data sample '%s': %s" % (name,[str(e) for e in event_types]) )

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

        if self.noise:
            if 'noise' not in event_types:
                raise AssertionError('`noise` field not found in '
                                     'configuration file.')
            dataset = self.params['dataset'].value
            if 'noise' not in dataset:
                dataset = 'nominal'
            noise_events = self.load_noise_events(
                config=self.config, dataset=dataset
            )
            events.append(noise_events)
        self._data = reduce(add, events)

        #If requested, add fix the truth variable names
        if self.fix_truth_variable_names :
            for event_key in self._data.metadata["flavints_joined"] :
                for var in self.truth_variables :
                    if var in self._data[event_key] :
                        new_var = self.truth_variable_prefix + var
                        self._data[event_key][new_var] = self._data[event_key].pop(var)

        self.sample_hash = this_hash
        self._data.metadata['sample_hash'] = this_hash
        self._data.update_hash()

    @staticmethod
    def load_neutrino_events(config, dataset):

        nu_data = []
        if dataset == 'neutrinos%sgen_lvl' % SEP:
            gen_cfg      = from_file(config.get(dataset, 'gen_cfg_file'))
            name         = gen_cfg.get('general', 'name')
            datadir      = gen_cfg.get('general', 'datadir')
            event_types  = split(gen_cfg.get('general', 'event_type'))
            weights      = split(gen_cfg.get('general', 'weights'))
            weight_units = gen_cfg.get('general', 'weight_units')
            keep_keys    = split(gen_cfg.get('general', 'keep_keys'))
            aliases      = gen_cfg.items('aliases')
            logging.info('Extracting neutrino dataset "{0}" from generator '
                         'level sample "{1}"'.format(dataset, name))

            for idx, flav in enumerate(event_types):
                fig = NuFlavIntGroup(flav)
                all_flavints = fig.flavints
                events_file = datadir + gen_cfg.get(flav, 'filename')

                flav_fidg = sample.load_from_nu_file(
                    events_file, all_flavints, weights[idx], weight_units,
                    keep_keys, aliases
                )
                nu_data.append(flav_fidg)
        else:

            name         = config.get('general', 'name')
            flavours     = split(config.get('neutrinos', 'flavours'))
            weights      = split(config.get('neutrinos', 'weights'))
            weight_units = config.get('neutrinos', 'weight_units')
            sys_list     = split(config.get('neutrinos', 'sys_list'))
            base_prefix  = config.get('neutrinos', 'baseprefix')
            keep_keys    = split(config.get('neutrinos', 'keep_keys'))
            aliases      = config.items('neutrinos%saliases' % SEP)
            logging.info('Extracting neutrino dataset "{0}" from sample '
                         '"{1}"'.format(dataset, name))
            if base_prefix == 'None':
                base_prefix = ''

            for idx, flav in enumerate(flavours):
                f = int(flav)
                all_flavints = NuFlavIntGroup(f, -f).flavints
                if dataset == 'nominal':
                    prefixes = []
                    for sys in sys_list:
                        ev_sys = 'neutrinos%s%s' % (SEP, sys)
                        nominal = config.get(ev_sys, 'nominal')
                        ev_sys_nom = ev_sys + SEP + nominal
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
                events_file = path.join( config.get('general', 'datadir'), base_prefix + file_prefix )

                flav_fidg = sample.load_from_nu_file(
                    events_file, all_flavints, weights[idx], weight_units,
                    keep_keys, aliases
                )
                nu_data.append(flav_fidg)
        nu_data = Data(
            reduce(add, nu_data),
            metadata={'name': name, 'sample': dataset}
        )

        return nu_data

    @staticmethod
    def load_muon_events(config, dataset):
        name         = config.get('general', 'name')
        weight       = config.get('muons', 'weight')
        weight_units = config.get('muons', 'weight_units')
        sys_list     = split(config.get('muons', 'sys_list'))
        base_prefix  = config.get('muons', 'baseprefix')
        keep_keys    = split(config.get('muons', 'keep_keys'))
        aliases      = config.items('muons%saliases' % SEP)
        if base_prefix == 'None':
            base_prefix = ''

        if dataset == 'nominal':
            paths = []
            for sys in sys_list:
                ev_sys = 'muons%s%s' % (SEP, sys)
                nominal = config.get(ev_sys, 'nominal')
                ev_sys_nom = ev_sys + SEP + nominal
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
        logging.info('Extracting muon dataset "{0}" from sample '
                     '"{1}"'.format(dataset, name))

        muons = from_file(file_path)
        sample.strip_keys(keep_keys, muons)

        if weight == 'None' or weight == '1':
            muons['sample_weight'] = np.ones(muons['weights'].shape)
        elif weight == '0':
            muons['sample_weight'] = np.zeros(muons['weights'].shape)
        else:
            muons['sample_weight'] = muons[weight] * ureg(weight_units)
        muons['pisa_weight'] = deepcopy(muons['sample_weight'])

        for alias, expr in aliases:
            if alias in muons:
                logging.warning(
                    'Overwriting Data key {0} with aliased expression '
                    '{1}'.format(alias, expr)
                )
            muons[alias] = eval(re.sub(r'\<(.*?)\>', r"muons['\1']", expr))

        muon_dict = {'muons': muons}
        return Data(muon_dict, metadata={'name': name, 'mu_sample': dataset})

    @staticmethod
    def load_noise_events(config, dataset):
        name         = config.get('general', 'name')
        weight       = config.get('noise', 'weight')
        weight_units = config.get('noise', 'weight_units')
        sys_list     = split(config.get('noise', 'sys_list'))
        base_prefix  = config.get('noise', 'baseprefix')
        keep_keys    = split(config.get('noise', 'keep_keys'))
        aliases      = config.items('noise%saliases' % SEP)
        if base_prefix == 'None':
            base_prefix = ''

        if dataset == 'nominal':
            paths = []
            for sys in sys_list:
                ev_sys = 'noise%s%s' % (SEP, sys)
                nominal = config.get(ev_sys, 'nominal')
                ev_sys_nom = ev_sys + SEP + nominal
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
        logging.info('Extracting noise dataset "{0}" from sample '
                     '"{1}"'.format(dataset, name))

        noise = from_file(file_path)
        sample.strip_keys(keep_keys, noise)

        if weight == 'None' or weight == '1':
            noise['sample_weight'] = np.ones(noise['weights'].shape)
        elif weight == '0':
            noise['sample_weight'] = np.zeros(noise['weights'].shape)
        else:
            noise['sample_weight'] = noise[weight] * ureg(weight_units)
        noise['pisa_weight'] = deepcopy(noise['sample_weight'])

        for alias, expr in aliases:
            if alias in noise:
                logging.warning(
                    'Overwriting Data key {0} with aliased expression '
                    '{1}'.format(alias, expr)
                )
            noise[alias] = eval(re.sub(r'\<(.*?)\>', r"noise['\1']", expr))

        noise_dict = {'noise': noise}
        return Data(noise_dict,
                    metadata={'name': name, 'noise_sample': dataset})

    @staticmethod
    def load_from_nu_file(events_file, all_flavints, weight, weight_units,
                          keep_keys, aliases):
        flav_fidg = FlavIntDataGroup(flavint_groups=all_flavints)

        events = from_file(events_file)
        sample.strip_keys(keep_keys, events)

        nu_mask = events['ptype'] > 0
        nubar_mask = events['ptype'] < 0
        cc_mask = events['interaction'] == 1
        nc_mask = events['interaction'] == 2

        if weight == 'None' or weight == '1':
            events['sample_weight'] = \
                np.ones(events['ptype'].shape) * ureg.dimensionless
        elif weight == '0':
            events['sample_weight'] = \
                np.zeros(events['ptype'].shape) * ureg.dimensionless
        else:
            events['sample_weight'] = events[weight] * \
                ureg(weight_units)
        events['pisa_weight'] = deepcopy(events['sample_weight'])

        for alias, expr in aliases:
            if alias in events:
                logging.warning(
                    'Overwriting Data key {0} with aliased expression '
                    '{1}'.format(alias, expr)
                )
            events[alias] = eval(re.sub(r'\<(.*?)\>', r"events['\1']", expr))

        for flavint in all_flavints:
            i_mask = cc_mask if flavint.cc else nc_mask
            t_mask = nu_mask if flavint.particle else nubar_mask

            flav_fidg[flavint] = {var: events[var][i_mask & t_mask]
                                  for var in events.keys()}
        return flav_fidg

    @staticmethod
    def strip_keys(keep_keys, events):
        if keep_keys == ['all']:
            pass
        else:
            remove_keys = []
            for k_key in keep_keys:
                if k_key not in events:
                    raise AssertionError(
                        'Key "{0}" not found in file, which contains keys '
                        '{1}'.format(k_key, events.keys())
                    )
            for d_key in events.keys():
                if d_key not in keep_keys:
                    remove_keys.append(d_key)
            for r_k in remove_keys:
                del events[r_k]

    def validate_params(self, params):
        assert isinstance(params['data_sample_config'].value, str)
        assert isinstance(params['dataset'].value, str)
        assert params['keep_criteria'].value is None or \
            isinstance(params['keep_criteria'].value, str)
