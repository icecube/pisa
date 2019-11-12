#! /usr/bin/env python

"""
Events class for working with PISA events files and Data class for working
with arbitrary Monte Carlo and datasets
"""


from __future__ import absolute_import, division, print_function

from copy import deepcopy
from collections.abc import Iterable, Mapping, Sequence
from collections import OrderedDict

import h5py
import numpy as np
from numpy import inf, nan # pylint: disable=unused-import
from uncertainties import unumpy as unp

from pisa import ureg
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.core.map import Map, MapSet
from pisa.utils import resources
from pisa.utils.comparisons import normQuant, recursiveEquality
from pisa.utils.flavInt import (FlavIntData, FlavIntDataGroup,
                                flavintGroupsFromString, NuFlavIntGroup)
from pisa.utils.format import text2tex
from pisa.utils.hash import hash_obj
from pisa.utils.fileio import from_file
from pisa.utils import hdf
from pisa.utils.log import logging, set_verbosity


__all__ = ['Events', 'Data', 'test_Events', 'test_Data']

__author__ = 'J.L. Lanfranchi, S. Mandalia'

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


# TODO: test hash function (attr)
class Events(FlavIntData):
    """Container for storing events, including metadata about the events.

    Examples
    --------
    >>> from pisa.core.binning import OneDimBinning, MultiDimBinning

    >>> # Load events from a PISA HDF5 file
    >>> events = Events('events/events__vlvnt__toy_1_to_80GeV_spidx1.0_cz-1_to_1_1e2evts_set0__unjoined__with_fluxes_honda-2015-spl-solmin-aa.hdf5')

    >>> # Apply a simple cut
    >>> events = events.applyCut('(true_coszen <= 0.5) & (true_energy <= 70)')
    >>> np.max(events[fi]['true_coszen']) <= 0.5
    True

    >>> # Apply an "inbounds" cut via a OneDimBinning
    >>> true_e_binning = OneDimBinning(
    ...    name='true_energy', num_bins=80, is_log=True,
    ...    domain=[10, 60]*ureg.GeV
    ... )
    >>> events = events.keepInbounds(true_e_binning)
    >>> np.min(events[fi]['true_energy']) >= 10
    True

    >>> print([(k, events.metadata[k]) for k in sorted(events.metadata.keys())])
    [('cuts', ['analysis']),
      ('detector', 'pingu'),
      ('flavints_joined',
         ['nue_cc+nuebar_cc',
             'numu_cc+numubar_cc',
             'nutau_cc+nutaubar_cc',
             'nuall_nc+nuallbar_nc']),
      ('geom', 'v39'),
      ('proc_ver', '5.1'),
      ('runs', [620, 621, 622])]

    """
    def __init__(self, val=None):
        self.metadata = OrderedDict([
            ('detector', ''),
            ('geom', ''),
            ('runs', []),
            ('proc_ver', ''),
            ('cuts', []),
            ('flavints_joined', []),
        ])
        meta = {}
        data = FlavIntData()
        if isinstance(val, (str, h5py.Group)):
            data, meta = self.__load(val)
        elif isinstance(val, Events):
            meta = deepcopy(val.metadata)
            data = deepcopy(val)
        elif isinstance(val, dict):
            data = deepcopy(val)
        self.metadata.update(meta)
        self.validate(data)
        self.update(data)
        self.update_hash()

    def __str__(self):
        meta = [(str(k) + ' : ' + str(v)) for k, v in self.metadata.items()]
        #fields =
        return '\n'.join(meta)

    def __repr__(self):
        return str(self)

    @property
    def hash(self):
        """Hash value"""
        return self._hash

    def __hash__(self):
        return self.hash

    def update_hash(self):
        """Update the cached hash value"""
        self._hash = hash_obj(normQuant(self.metadata))

    @property
    def flavint_groups(self):
        """All flavor/interaction type groups (even singletons) present"""
        return sorted(flavintGroupsFromString(
            ','.join(self.metadata['flavints_joined'])
        ))

    @property
    def joined_string(self):
        """Concise string identifying _only_ joined flavints"""
        joined_groups = sorted(
            [NuFlavIntGroup(j) for j in self.metadata['flavints_joined']]
        )
        if len(joined_groups) == 0:
            return 'unjoined'
        return 'joined_G_' + '_G_'.join([str(g) for g in joined_groups])

    def meta_eq(self, other):
        """Test whether the metadata for this object matches that of `other`"""
        return recursiveEquality(self.metadata, other.metadata)

    def data_eq(self, other):
        """Test whether the data for this object matches that of `other`"""
        return recursiveEquality(self, other)

    def __eq__(self, other):
        return self.meta_eq(other) and self.data_eq(other)

    def __load(self, fname):
        fpath = resources.find_resource(fname)
        with h5py.File(fpath, 'r') as open_file:
            meta = dict(open_file.attrs)
            for k, v in meta.items():
                if hasattr(v, 'tolist'):
                    meta[k] = v.tolist()
            data = hdf.from_hdf(open_file)
        self.validate(data)
        return data, meta

    def save(self, fname, **kwargs):
        hdf.to_hdf(self, fname, attrs=self.metadata, **kwargs)

    def histogram(self, kinds, binning, binning_cols=None, weights_col=None,
                  errors=False, name=None, tex=None):
        """Histogram the events of all `kinds` specified, with `binning` and
        optionally applying `weights`.

        Parameters
        ----------
        kinds : string, sequence of NuFlavInt, or NuFlavIntGroup
        binning : OneDimBinning, MultiDimBinning or sequence of arrays (one array per binning dimension)
        binning_cols : string or sequence of strings
            Bin only these dimensions, ignoring other dimensions in `binning`
        weights_col : None or string
            Column to use for weighting the events
        errors : bool
            Whether to attach errors to the resulting Map
        name : None or string
            Name to give to resulting Map. If None, a default is derived from
            `kinds` and `weights_col`.
        tex : None or string
            TeX label to give to the resulting Map. If None, default is
            dereived from the `name` specified (or its value derived from
            `kinds` and `weights_col`).

        Returns
        -------
        Map : numpy ndarray with as many dimensions as specified by `binning`
            argument

        """
        # TODO: make able to take integer for `binning` and--in combination
        # with units in the Events columns--generate an appropriate
        # MultiDimBinning object, attach this and return the package as a Map.

        if not isinstance(kinds, NuFlavIntGroup):
            kinds = NuFlavIntGroup(kinds)
        if isinstance(binning_cols, str):
            binning_cols = [binning_cols]
        assert weights_col is None or isinstance(weights_col, str)

        # TODO: units of columns, and convert bin edges if necessary
        if isinstance(binning, OneDimBinning):
            binning = MultiDimBinning([binning])
        elif isinstance(binning, MultiDimBinning):
            pass
        elif (isinstance(binning, Iterable)
              and not isinstance(binning, Sequence)):
            binning = list(binning)
        elif isinstance(binning, Sequence):
            pass
        else:
            raise TypeError('Unhandled type %s for `binning`.' %type(binning))

        if isinstance(binning, Sequence):
            raise NotImplementedError(
                'Simle sequences not handled at this time. Please specify a'
                ' OneDimBinning or MultiDimBinning object for `binning`.'
            )
            #assert len(binning_cols) == len(binning)
            #bin_edges = binning

        # TODO: units support for Events will mean we can do `m_as(...)` here!
        bin_edges = [edges.magnitude for edges in binning.bin_edges]
        if binning_cols is None:
            binning_cols = binning.names
        else:
            assert set(binning_cols).issubset(set(binning.names))

        # Extract the columns' data into a list of array(s) for histogramming
        repr_flavint = kinds[0]
        sample = [self[repr_flavint][colname] for colname in binning_cols]
        err_weights = None
        hist_weights = None
        if weights_col is not None:
            hist_weights = self[repr_flavint][weights_col]
            if errors:
                err_weights = np.square(hist_weights)

        hist, edges = np.histogramdd(sample=sample,
                                     weights=hist_weights,
                                     bins=bin_edges)
        if errors:
            sumw2, edges = np.histogramdd(sample=sample,
                                          weights=err_weights,
                                          bins=bin_edges)
            hist = unp.uarray(hist, np.sqrt(sumw2))

        if name is None:
            if tex is None:
                tex = kinds.tex
                if weights_col is not None:
                    tex += r', \; {\rm weights=' + text2tex(weights_col) + r'}'

            name = str(kinds)
            if weights_col is not None:
                name += ', weights=' + weights_col

        if tex is None:
            tex = text2tex(name)

        return Map(name=name, hist=hist, binning=binning, tex=tex)

    def applyCut(self, keep_criteria):
        """Apply a cut by specifying criteria for keeping events. The cut must
        be successfully applied to all flav/ints in the events object before
        the changes are kept, otherwise the cuts are reverted.

        Parameters
        ----------
        keep_criteria : string
            Any string interpretable as numpy boolean expression.

        Examples
        --------
        Keep events with true energies in [1, 80] GeV (note that units are not
        recognized, so have to be handled outside this method)

        >>> events = events.applyCut("(true_energy >= 1) & (true_energy <= 80)")

        Do the opposite with "~" inverting the criteria

        >>> events = events.applyCut("~((true_energy >= 1) & (true_energy <= 80))")

        Numpy namespace is available for use via `np` prefix

        >>> events = events.applyCut("np.log10(true_energy) >= 0")

        """
        if keep_criteria in self.metadata['cuts']:
            logging.debug("Criteria '%s' have already been applied. Returning"
                          " events unmodified.", keep_criteria)
            return self

        # Nothing to do if no cuts specified
        if keep_criteria is None:
            return

        assert isinstance(keep_criteria, str)

        #Only get the flavints for which we have data
        flavints_to_process = self.flavints_present 
        flavints_processed = []
        remaining_data = {}
        for flavint in flavints_to_process:
            #Get the evets for this flavor/interaction
            data_dict = self[flavint]

            field_names = data_dict.keys()

            # TODO: handle unicode:
            #  * translate crit to unicode (easiest to hack but could be
            #    problematic elsewhere)
            #  * translate field names to ascii (probably should be done at
            #    the from_hdf stage?)

            # Replace simple field names with full paths into the data that
            # lives in this object
            crit_str = keep_criteria
            for field_name in field_names:
                crit_str = crit_str.replace(
                    field_name, 'self["%s"]["%s"]' %(flavint, field_name)
                )
            mask = eval(crit_str)
            remaining_data[flavint] = (
                {k : v[mask] for k, v in self[flavint].items()}
            )
            flavints_processed.append(flavint)

        remaining_events = Events()
        remaining_events.metadata.update(deepcopy(self.metadata))
        remaining_events.metadata['cuts'].append(keep_criteria)

        for flavint in flavints_processed:
            remaining_events[flavint] = deepcopy(remaining_data[flavint])

        return remaining_events

    def keepInbounds(self, binning):
        """Cut out any events that fall outside `binning`. Note that events
        that fall exactly on an outer edge are kept.

        Parameters
        ----------
        binning : OneDimBinning or MultiDimBinning

        Returns
        -------
        remaining_events : Events

        """
        try:
            binning = OneDimBinning(binning)
        except:
            pass
        if isinstance(binning, OneDimBinning):
            binning = [binning]
        binning = MultiDimBinning(binning)

        current_cuts = self.metadata['cuts']
        new_cuts = [dim.inbounds_criteria for dim in binning]
        unapplied_cuts = [c for c in new_cuts if c not in current_cuts]
        if not unapplied_cuts:
            logging.debug("All inbounds criteria '%s' have already been"
                          " applied. Returning events unmodified.", new_cuts)
            return self
        all_cuts = deepcopy(current_cuts) + unapplied_cuts

        # Create a single cut from all unapplied cuts
        keep_criteria = ' & '.join(['(%s)' % c for c in unapplied_cuts])

        # Do the cutting
        remaining_events = self.applyCut(keep_criteria=keep_criteria)

        # Replace the combined 'cuts' string with individual cut strings
        remaining_events.metadata['cuts'] = all_cuts

        return remaining_events


    @property
    def flavints_present(self):
        '''
        returns a tuple of the flavints that are present in the events
        '''

        flavints_present_list = []

        #Loop over a tuple of all possible flav/int combinations
        for flavint in self.flavints:

            # If a particular flavor/interaction combination is not present in the events, then
            # self[flavint] will be set to np.nan
            # Check this here, using a try block to catch exceptions throw if the data is actually
            # there (in which case it is a dict, and np.isnan will raise an exception as cannot
            # take a dit as input)
            found_data_for_this_flavint = True
            try:
                if np.isnan(self[flavint]):
                    found_data_for_this_flavint = False
            except TypeError: 
                pass
            if found_data_for_this_flavint:
                flavints_present_list.append(flavint)

        return tuple(flavints_present_list)


class Data(FlavIntDataGroup):
    """Container for storing events, including metadata about the events.

    Examples
    --------
        [('cuts', ['analysis']),
         ('detector', 'pingu'),
         ('flavints_joined',
            ['nue_cc+nuebar_cc',
                'numu_cc+numubar_cc',
                'nutau_cc+nutaubar_cc',
                'nuall_nc+nuallbar_nc']),
         ('geom', 'v39'),
         ('proc_ver', '5.1'),
         ('runs', [620, 621, 622])]

    """
    def __init__(self, val=None, flavint_groups=None, metadata=None):
        # TODO(shivesh): add noise implementation
        self.metadata = OrderedDict([
            ('name', ''),
            ('detector', ''),
            ('geom', ''),
            ('runs', []),
            ('proc_ver', ''),
            ('cuts', []),
            ('flavints_joined', []),
        ])
        self.contains_neutrinos = False
        self.contains_muons = False
        self.contains_noise = False

        # Get data and metadata from val
        meta = {}
        if isinstance(val, (str, h5py.Group)):
            data, meta = self.__load(val)
        elif isinstance(val, Data):
            data = val
            meta = val.metadata
        elif isinstance(val, (Mapping, FlavIntDataGroup)):
            data = val
            meta = None
        else:
            raise TypeError('Unrecognized `val` type %s' % type(val))

        # Check consistency of metadata from val and from input
        if meta is not None:
            if metadata is not None and meta != metadata:
                raise AssertionError('Input `metadata` does not match '
                                     'metadata inside `val`')
            self.metadata.update(meta)
        elif metadata is not None:
            self.metadata.update(metadata)

        # Find and deal with any muon data if it exists
        if self.metadata['flavints_joined'] == list([]):
            if 'muons' in data:
                self.muons = data.pop('muons')
        elif 'muons' in self.metadata['flavints_joined']:
            if 'muons' not in data:
                raise AssertionError('Metadata has muons specified but '
                                     'they are not found in the data')
            else:
                self.muons = data.pop('muons')
        elif 'muons' in data:
            raise AssertionError('Found muons in data but not found in '
                                 'metadata key `flavints_joined`')

        # Find and deal with any noise data if it exists
        if self.metadata['flavints_joined'] == list([]):
            if 'noise' in data:
                self.noise = data.pop('noise')
        elif 'noise' in self.metadata['flavints_joined']:
            if 'noise' not in data:
                raise AssertionError('Metadata has noise specified but '
                                     'they are not found in the data')
            else:
                self.noise = data.pop('noise')
        elif 'noise' in data:
            raise AssertionError('Found noise in data but not found in '
                                 'metadata key `flavints_joined`')

        # Instantiate a FlavIntDataGroup
        if data == dict():
            self._flavint_groups = []
        else:
            super().__init__(val=data, flavint_groups=flavint_groups)
            self.contains_neutrinos = True

        # Check consistency of flavints_joined
        if self.metadata['flavints_joined']:
            combined_types = []
            if self.contains_neutrinos:
                combined_types += [str(f) for f in self.flavint_groups]
            if self.contains_muons:
                combined_types += ['muons']
            if self.contains_noise:
                combined_types += ['noise']
            if set(self.metadata['flavints_joined']) != \
               set(combined_types):
                raise AssertionError(
                    '`flavint_groups` metadata does not match the '
                    'flavint_groups in the data\n{0} != '
                    '{1}'.format(set(self.metadata['flavints_joined']),
                                 set(combined_types))
                )
        else:
            self.metadata['flavints_joined'] = [str(f)
                                                for f in self.flavint_groups]
            if self.contains_muons:
                self.metadata['flavints_joined'] += ['muons']
            if self.contains_noise:
                self.metadata['flavints_joined'] += ['noise']

        self._hash = None
        self.update_hash()

    @property
    def hash(self):
        """Probabilistically unique identifier"""
        return self._hash

    @hash.setter
    def hash(self, val):
        self._hash = val

    def __hash__(self):
        return self.hash

    def update_hash(self):
        """Update the cached hash value"""
        self._hash = hash_obj(normQuant(self.metadata))

    @property
    def muons(self):
        """muon data"""
        # TODO: it seems more sensible to return None rather than raise
        # AttributeError, since the attribute `muons` absolutely exists, just
        # it contains no information... hence, `None` value.
        # Same for `neutrinos` property.
        if not self.contains_muons:
            raise AttributeError('No muons loaded in Data')
        return self._muons

    @muons.setter
    def muons(self, val):
        assert isinstance(val, dict)
        self.contains_muons = True
        self._muons = val

    @property
    def noise(self):
        if not self.contains_noise:
            raise AttributeError('No noise loaded in Data')
        return self._noise

    @noise.setter
    def noise(self, val):
        assert isinstance(val, dict)
        self.contains_noise = True
        self._noise = val

    @property
    def neutrinos(self):
        """neutrino data"""
        if not self.contains_neutrinos:
            raise AttributeError('No neutrinos loaded in Data')
        return dict(zip(self.keys(), self.values()))

    # TODO: make sure this returns all flavints, and not just joined (grouped)
    # flavints, as is the case for the Events object
    @property
    def names(self):
        """Names of flavints joined"""
        return self.metadata['flavints_joined']

    def meta_eq(self, other):
        """Test whether the metadata for this object matches that of `other`"""
        return recursiveEquality(self.metadata, other.metadata)

    def data_eq(self, other):
        """Test whether the data for this object matche that of `other`"""
        return recursiveEquality(self, other)

    def applyCut(self, keep_criteria):
        """Apply a cut by specifying criteria for keeping events. The cut must
        be successfully applied to all flav/ints in the events object before
        the changes are kept, otherwise the cuts are reverted.

        Parameters
        ----------
        keep_criteria : string
            Any string interpretable as numpy boolean expression.

        Returns
        -------
        remaining_events : Events
            An Events object with the remaining events (deepcopied) and with
            updated cut metadata including `keep_criteria`.

        Examples
        --------
        Keep events with true energies in [1, 80] GeV (note that units are not
        recognized, so have to be handled outside this method)

        >>> remaining = applyCut("(true_energy >= 1) & (true_energy <= 80)")

        Do the opposite with "~" inverting the criteria

        >>> remaining = applyCut("~((true_energy >= 1) & (true_energy <= 80))")

        Numpy namespace is available for use via `np` prefix

        >>> remaining = applyCut("np.log10(true_energy) >= 0")

        """
        # TODO(shivesh): function does not pass tests
        raise NotImplementedError

        if keep_criteria in self.metadata['cuts']:
            return

        assert isinstance(keep_criteria, str)

        fig_to_process = []
        if self.contains_neutrinos:
            fig_to_process += deepcopy(self.flavint_groups)
        if self.contains_muons:
            fig_to_process += ['muons']
        if self.contains_noise:
            fig_to_process += ['noise']

        logging.info("Applying cut to %s : %s" %(fig_to_process,keep_criteria))

        fig_processed = []
        remaining_data = {}
        for fig in fig_to_process:
            data_dict = self[fig]
            field_names = data_dict.keys()

            # TODO: handle unicode:
            #  * translate crit to unicode (easiest to hack but could be
            #    problematic elsewhere)
            #  * translate field names to ascii (probably should be done at
            #    the from_hdf stage?)

            # Replace simple field names with full paths into the data that
            # lives in this object
            crit_str = (keep_criteria)
            for field_name in field_names:
                crit_str = crit_str.replace(
                    field_name, 'self["%s"]["%s"]' % (fig, field_name)
                )
            mask = eval(crit_str)
            remaining_data[fig] = {k: v[mask]
                                   for k, v in self[fig].items()}
            fig_processed.append(fig)

        remaining_events = Events()
        remaining_events.metadata.update(deepcopy(self.metadata))
        remaining_events.metadata['cuts'].append(keep_criteria)
        for fig in fig_to_process:
            remaining_events[fig] = deepcopy(remaining_data.pop(fig))

        return remaining_events

    def keepInbounds(self, binning):
        """Cut out any events that fall outside `binning`. Note that events
        that fall exactly on the outer edge are kept.

        Parameters
        ----------
        binning : OneDimBinning or MultiDimBinning

        """
        if isinstance(binning, OneDimBinning):
            binning = [binning]
        else:
            assert isinstance(binning, MultiDimBinning)
        current_cuts = self.metadata['cuts']
        new_cuts = [dim.inbounds_criteria for dim in binning]
        unapplied_cuts = [c for c in new_cuts if c not in current_cuts]
        for cut in unapplied_cuts:
            self.applyCut(keep_criteria=cut)

    def transform_groups(self, flavint_groups):
        """Transform Data into a structure given by the input
        flavint_groups. Calls the corresponding inherited function.

        Parameters
        ----------
        flavint_groups : string, or sequence of strings or sequence of
                         NuFlavIntGroups

        Returns
        -------
        t_data : Data
        """
        t_fidg = super().transform_groups(flavint_groups)
        metadata = deepcopy(self.metadata)
        metadata['flavints_joined'] = [str(f) for f in t_fidg.flavint_groups]
        t_dict = dict(t_fidg)
        if self.contains_muons:
            metadata['flavints_joined'] += ['muons']
            t_dict['muons'] = deepcopy(self['muons'])
        if self.contains_noise:
            metadata['flavints_joined'] += ['noise']
            t_dict['noise'] = deepcopy(self['noise'])
        t_fidg = t_dict
        ret_obj = Data(t_fidg, metadata=metadata)
        ret_obj.update_hash()
        return ret_obj

    def digitize(self, kinds, binning, binning_cols=None):
        """Wrapper for numpy's digitize function."""
        if isinstance(kinds, str):
            kinds = [kinds]
        if 'muons' not in kinds and 'noise' not in kinds:
            kinds = self._parse_flavint_groups(kinds)
        kinds = kinds[0]

        if isinstance(binning_cols, str):
            binning_cols = [binning_cols]

        # TODO: units of columns, and convert bin edges if necessary
        if isinstance(binning, OneDimBinning):
            binning = MultiDimBinning([binning])
        elif isinstance(binning, MultiDimBinning):
            pass
        elif (isinstance(binning, Iterable)
              and not isinstance(binning, Sequence)):
            binning = list(binning)
        elif isinstance(binning, Sequence):
            pass
        else:
            raise TypeError('Unhandled type %s for `binning`.' % type(binning))

        if isinstance(binning, Sequence):
            raise NotImplementedError(
                'Simle sequences not handled at this time. Please specify a'
                ' OneDimBinning or MultiDimBinning object for `binning`.'
            )
            # assert len(binning_cols) == len(binning)
            # bin_edges = binning

        # TODO: units support for Data will mean we can do `m_as(...)` here!
        bin_edges = [edges.magnitude for edges in binning.bin_edges]
        if binning_cols is None:
            binning_cols = binning.names
        else:
            assert set(binning_cols).issubset(set(binning.names))

        hist_idxs = []
        for colname in binning_cols:
            sample = self[kinds][colname]
            hist_idxs.append(np.digitize(
                sample, binning[colname].bin_edges.m
            ))
        hist_idxs = np.vstack(hist_idxs).T

        return hist_idxs

    def histogram(self, kinds, binning, binning_cols=None, weights_col=None,
                  errors=False, name=None, tex=None, **kwargs):
        """Histogram the events of all `kinds` specified, with `binning` and
        optionally applying `weights`.

        Parameters
        ----------
        kinds : string, sequence of NuFlavInt, or NuFlavIntGroup
        binning : OneDimBinning, MultiDimBinning or sequence of arrays
            (one array per binning dimension)
        binning_cols : string or sequence of strings
            Bin only these dimensions, ignoring other dimensions in `binning`
        weights_col : None or string
            Column to use for weighting the events
        errors : bool
            Whether to attach errors to the resulting Map
        name : None or string
            Name to give to resulting Map. If None, a default is derived from
            `kinds` and `weights_col`.
        tex : None or string
            TeX label to give to the resulting Map. If None, default is
            dereived from the `name` specified or the derived default.
        **kwargs : Keyword args passed to Map object

        Returns
        -------
        Map : numpy ndarray with as many dimensions as specified by `binning`
            argument

        """
        # TODO: make able to take integer for `binning` and--in combination
        # with units in the Data columns--generate an appropriate
        # MultiDimBinning object, attach this and return the package as a Map.

        if isinstance(kinds, str):
            kinds = [kinds]
        if 'muons' not in kinds and 'noise' not in kinds:
            kinds = self._parse_flavint_groups(kinds)
        kinds = kinds[0]

        if isinstance(binning_cols, str):
            binning_cols = [binning_cols]
        assert weights_col is None or isinstance(weights_col, str)

        # TODO: units of columns, and convert bin edges if necessary
        if isinstance(binning, OneDimBinning):
            binning = MultiDimBinning([binning])
        elif isinstance(binning, MultiDimBinning):
            pass
        elif (isinstance(binning, Iterable)
              and not isinstance(binning, Sequence)):
            binning = list(binning)
        elif isinstance(binning, Sequence):
            pass
        else:
            raise TypeError('Unhandled type %s for `binning`.' % type(binning))

        if isinstance(binning, Sequence):
            raise NotImplementedError(
                'Simle sequences not handled at this time. Please specify a'
                ' OneDimBinning or MultiDimBinning object for `binning`.'
            )
            # assert len(binning_cols) == len(binning)
            # bin_edges = binning

        # TODO: units support for Data will mean we can do `m_as(...)` here!
        bin_edges = [edges.magnitude for edges in binning.bin_edges]
        if binning_cols is None:
            binning_cols = binning.names
        else:
            assert set(binning_cols).issubset(set(binning.names))

        # Extract the columns' data into a list of array(s) for histogramming
        sample = [self[kinds][colname] for colname in binning_cols]
        err_weights = None
        hist_weights = None
        if weights_col is not None:
            hist_weights = self[kinds][weights_col]
            if errors:
                err_weights = np.square(hist_weights)

        hist, edges = np.histogramdd(sample=sample,
                                     weights=hist_weights,
                                     bins=bin_edges)
        if errors:
            sumw2, edges = np.histogramdd(sample=sample,
                                          weights=err_weights,
                                          bins=bin_edges)
            hist = unp.uarray(hist, np.sqrt(sumw2))

        if name is None:
            if tex is None:
                try:
                    tex = kinds.tex
                # TODO: specify specific exception(s)
                except:
                    tex = r'{0}'.format(kinds)
                if weights_col is not None:
                    tex += r', \; {\rm weights} =' + text2tex(weights_col)

            name = str(kinds)
            if weights_col is not None:
                name += ', weights=' + weights_col

        if tex is None:
            tex = text2tex(name)

        return Map(name=name, hist=hist, binning=binning, tex=tex, **kwargs)

    def histogram_set(self, binning, nu_weights_col, mu_weights_col,
                      noise_weights_col, mapset_name, errors=False):
        """Uses the above histogram function but returns the set of all of them
        for everything in the Data object.

        Parameters
        ----------
        binning : OneDimBinning, MultiDimBinning
            The definition of the binning for the histograms.
        nu_weights_col : None or string
            The column in the Data object by which to weight the neutrino
            histograms. Specify None for unweighted histograms.
        mu_weights_col : None or string
            The column in the Data object by which to weight the muon
            histograms. Specify None for unweighted histograms.
        noise_weights_col : None or string
            The column in the Data object by which to weight the noise
            histograms. Specify None for unweighted histograms.
        mapset_name : string
            The name by which the resulting MapSet will be identified.
        errors : boolean
            A flag for whether to calculate errors on the histograms or not.
            This defaults to False.

        Returns
        -------
        MapSet : A MapSet containing all of the Maps for everything in this
                 Data object.

        """
        if not isinstance(binning, MultiDimBinning):
            if not isinstance(binning, OneDimBinning):
                raise TypeError('binning should be either MultiDimBinning or '
                                'OneDimBinning object. Got %s.' % type(binning))
        if nu_weights_col is not None:
            if not isinstance(nu_weights_col, str):
                raise TypeError('nu_weights_col should be a string. Got %s'
                                % type(nu_weights_col))
        if mu_weights_col is not None:
            if not isinstance(mu_weights_col, str):
                raise TypeError('mu_weights_col should be a string. Got %s'
                                % type(mu_weights_col))
        if not isinstance(errors, bool):
            raise TypeError('flag for whether to calculate errors or not '
                            'should be a boolean. Got %s.' % type(errors))
        outputs = []
        if self.contains_neutrinos:
            for fig in self.keys():
                outputs.append(
                    self.histogram(
                        kinds=fig,
                        binning=binning,
                        weights_col=nu_weights_col,
                        errors=errors,
                        name=str(NuFlavIntGroup(fig))
                    )
                )
        if self.contains_muons:
            outputs.append(
                self.histogram(
                    kinds='muons',
                    binning=binning,
                    weights_col=mu_weights_col,
                    errors=errors,
                    name='muons',
                    tex=r'\rm{muons}'
                )
            )
        if self.contains_noise:
            outputs.append(
                self.histogram(
                    kinds='noise',
                    binning=binning,
                    weights_col=mu_weights_col,
                    errors=errors,
                    name='noise',
                    tex=r'\rm{noise}'
                )
            )
        return MapSet(maps=outputs, name=mapset_name)

    def __load(self, fname):
        try:
            data, meta = from_file(fname, return_attrs=True)
        except TypeError:
            data = from_file(fname)
            meta = None
        return data, meta

    def __getitem__(self, arg):
        if isinstance(arg, str):
            arg = arg.strip().lower()
            if arg == 'muons':
                return self.muons
            if arg == 'noise':
                return self.noise
        tgt_obj = super().__getitem__(arg)
        return tgt_obj

    def __setitem__(self, arg, value):
        if isinstance(arg, str):
            arg = arg.strip().lower()
            if arg == 'muons':
                self.muons = value
                return
            if arg == 'noise':
                self.noise = value
                return
        super().__setitem__(arg, value)

    def __add__(self, other):
        muons = None
        noise = None
        assert isinstance(other, Data)

        metadata = {}
        for key in self.metadata:
            if key == 'flavints_joined':
                continue
            if key in other.metadata:
                if self.metadata[key] != other.metadata[key]:
                    raise AssertionError(
                        'Metadata mismatch, key {0}, {1} != '
                        '{2}'.format(key, self.metadata[key],
                                     other.metadata[key])
                    )
                else:
                    metadata[key] = deepcopy(self.metadata[key])
            else:
                metadata[key] = deepcopy(self.metadata[key])

        for key in other.metadata:
            if key == 'flavints_joined':
                continue
            if key in self.metadata:
                if other.metadata[key] != self.metadata[key]:
                    raise AssertionError(
                        'Metadata mismatch, key {0}, {1} != '
                        '{2}'.format(key, other.metadata[key],
                                     self.metadata[key])
                    )
                else:
                    metadata[key] = deepcopy(other.metadata[key])
            else:
                metadata[key] = deepcopy(other.metadata[key])

        if self.contains_muons:
            if other.contains_muons:
                muons = self._merge(deepcopy(self['muons']), other['muons'])
            else:
                muons = deepcopy(self['muons'])
        elif other.contains_muons:
            muons = deepcopy(other['muons'])

        if self.contains_noise:
            if other.contains_noise:
                noise = self._merge(deepcopy(self['noise']), other['noise'])
            else:
                noise = deepcopy(self['noise'])
        elif other.contains_noise:
            noise = deepcopy(other['noise'])

        if len(self.flavint_groups) == 0:
            if len(other.flavint_groups) == 0:
                a_fidg = FlavIntDataGroup(other)
        elif len(other.flavint_groups) == 0:
            a_fidg = FlavIntDataGroup(self)
        else:
            a_fidg = super().__add__(other)
        metadata['flavints_joined'] = [str(f) for f in a_fidg.flavint_groups]

        if muons is not None:
            a_dict = dict(a_fidg)
            metadata['flavints_joined'] += ['muons']
            a_dict['muons'] = muons
            a_fidg = a_dict
        if noise is not None:
            a_dict = dict(a_fidg)
            metadata['flavints_joined'] += ['noise']
            a_dict['noise'] = noise
            a_fidg = a_dict
        return Data(a_fidg, metadata=metadata)

    def __str__(self):
        meta = [(str(k) + ' : ' + str(v)) for k, v in self.metadata.items()]
        return '\n'.join(meta)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.meta_eq(other) and self.data_eq(other)


# pylint: disable=line-too-long
def test_Events():
    """Unit tests for Events class"""
    from pisa.utils.flavInt import NuFlavInt
    # Instantiate empty object
    events = Events()

    # Instantiate from PISA events HDF5 file
    events = Events('events/events__vlvnt__toy_1_to_80GeV_spidx1.0_cz-1_to_1_1e2evts_set0__unjoined__with_fluxes_honda-2015-spl-solmin-aa.hdf5')

    # Apply a simple cut
    events = events.applyCut('(true_coszen <= 0.5) & (true_energy <= 70)')
    for fi in events.flavints:
        assert np.max(events[fi]['true_coszen']) <= 0.5
        assert np.max(events[fi]['true_energy']) <= 70

    # Apply an "inbounds" cut via a OneDimBinning
    true_e_binning = OneDimBinning(
        name='true_energy', num_bins=80, is_log=True, domain=[10, 60]*ureg.GeV
    )
    events = events.keepInbounds(true_e_binning)
    for fi in events.flavints:
        assert np.min(events[fi]['true_energy']) >= 10
        assert np.max(events[fi]['true_energy']) <= 60

    # Apply an "inbounds" cut via a MultiDimBinning
    true_e_binning = OneDimBinning(
        name='true_energy', num_bins=80, is_log=True, domain=[20, 50]*ureg.GeV
    )
    true_cz_binning = OneDimBinning(
        name='true_coszen', num_bins=40, is_lin=True, domain=[-0.8, 0]
    )
    mdb = MultiDimBinning([true_e_binning, true_cz_binning])
    events = events.keepInbounds(mdb)
    for fi in events.flavints:
        assert np.min(events[fi]['true_energy']) >= 20
        assert np.max(events[fi]['true_energy']) <= 50
        assert np.min(events[fi]['true_coszen']) >= -0.8
        assert np.max(events[fi]['true_coszen']) <= 0

    # Now try to apply a cut that fails on one flav/int (since the field will
    # be missing) and make sure that the cut did not get applied anywhere in
    # the end (i.e., it is rolled back)
    sub_evts = events['nutaunc']
    sub_evts.pop('true_energy')
    events['nutaunc'] = sub_evts
    try:
        events = events.applyCut('(true_energy >= 30) & (true_energy <= 40)')
    except Exception:
        pass
    else:
        raise Exception('Should not have been able to apply the cut!')
    for fi in events.flavints:
        if fi == NuFlavInt('nutaunc'):
            continue
        assert np.min(events[fi]['true_energy']) < 30

    logging.info(
        '<< PASS : test_Events >> (note:'
        ' "[   ERROR] Events object is in an inconsistent state. Reverting cut'
        ' for all flavInts." message above **is expected**.)')


def test_Data():
    """Unit tests for Data class"""
    # Instantiate from LEESARD file - located in $PISA_RESOURCES
    file_loc = 'LEESARD/PRD_extend_finalLevel/12550.pckl'
    file_loc2 = 'LEESARD/PRD_extend_finalLevel/14550.pckl'
    f = from_file(file_loc)
    f2 = from_file(file_loc2)
    d = {'nue+nuebar': f}
    d2 = {'numu+numubar': f2}
    data = Data(d)
    data2 = Data(d2)
    logging.debug(str((data.keys())))

    muon_file = 'GRECO/new_style_files/Level7_muongun.12370_15.pckl'
    m = {'muons': from_file(muon_file)}
    m = Data(val=m)
    assert m.contains_muons
    assert not m.contains_neutrinos
    logging.debug(str((m)))
    data = data + m
    assert data.contains_neutrinos
    logging.debug(str((data)))
    if not data.contains_muons:
        raise Exception("data doesn't contain muons.")
    logging.debug(str((data.neutrinos.keys())))

    noise_file = 'GRECO/new_style_files/Level7_VuvuzelaPureNoise_V2.990015.pckl'
    n = {'noise': from_file(muon_file)}
    n = Data(val=n)
    assert n.contains_noise
    assert not n.contains_neutrinos
    logging.debug(str((n)))
    data = data + n
    assert data.contains_neutrinos
    logging.debug(str((data)))
    if not data.contains_noise:
        raise Exception("data doesn't contain noise.")
    logging.debug(str((data.neutrinos.keys())))

    # Apply a simple cut
    # data.applyCut('(zenith <= 1.1) & (energy <= 200)')
    # for fi in data.flavint_groups:
    #     assert np.max(data[fi]['zenith']) <= 1.1
    #     assert np.max(data[fi]['energy']) <= 200

    # Apply an "inbounds" cut via a OneDimBinning
    # e_binning = OneDimBinning(
    #     name='energy', num_bins=80, is_log=True, domain=[10, 200]*ureg.GeV
    # )
    # data.keepInbounds(e_binning)
    # for fi in data.flavint_groups:
    #     assert np.min(data[fi]['energy']) >= 10
    #     assert np.max(data[fi]['energy']) <= 200

    # Apply an "inbounds" cut via a MultiDimBinning
    # e_binning = OneDimBinning(
    #     name='energy', num_bins=80, is_log=True, domain=[20, 210]*ureg.GeV
    # )
    # cz_binning = OneDimBinning(
    #     name='zenith', num_bins=40, is_lin=True, domain=[0.1, 1.8*np.pi]
    # )
    # mdb = MultiDimBinning([e_binning, cz_binning])
    # data.keepInbounds(mdb)
    # for fi in data.flavint_groups:
    #     assert np.min(data[fi]['energy']) >= 20
    #     assert np.max(data[fi]['energy']) <= 210
    #     assert np.min(data[fi]['zenith']) >= 0.1
    #     assert np.max(data[fi]['zenith']) <= 1.8*np.pi

    # # Now try to apply a cut that fails on one flav/int (since the field will
    # # be missing) and make sure that the cut did not get applied anywhere in
    # # the end (i.e., it is rolled back)
    # sub_evts = data['nue+nuebar']
    # sub_evts.pop('energy')
    # data['nue+nuebar'] = sub_evts
    # try:
    #     data.applyCut('(energy >= 30) & (energy <= 40)')
    # except Exception:
    #     pass
    # else:
    #     raise Exception('Should not have been able to apply the cut!')
    # for fi in data.flavint_groups:
    #     if fi == NuFlavIntGroup('nue+nuebar'):
    #         continue
    #     assert np.min(data[fi]['energy']) < 30

    data.save('/tmp/test_FlavIntDataGroup.json')
    data.save('/tmp/test_FlavIntDataGroup.hdf5')
    data = Data('/tmp/test_FlavIntDataGroup.json')
    data = Data(val='/tmp/test_FlavIntDataGroup.hdf5')

    d3 = data + data2 + m
    logging.debug(str((d3)))
    d3_com = d3.transform_groups(['nue+nuebar+numu+numubar'])
    logging.debug(str((d3_com)))

    logging.info('<< PASS : test_Data >>')


if __name__ == "__main__":
    set_verbosity(1)
    test_Events()
    # TODO: following is removed until a test dataset can be introduced
    #test_Data()
