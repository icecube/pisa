#! /usr/bin/env python
#
# Events class for working with PISA events files
#
# author: Justin L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   October 24, 2015
#
"""
Events class for working with PISA events files and Data class for working
with arbitrary Monte Carlo and datasets

"""


from copy import deepcopy
from collections import Iterable, OrderedDict, Sequence

import h5py
import numpy as np
from uncertainties import unumpy as unp

from pisa import ureg
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.core.map import Map, MapSet
from pisa.utils import resources
from pisa.utils.comparisons import normQuant, recursiveEquality
from pisa.utils.flavInt import FlavIntData, NuFlavIntGroup, FlavIntDataGroup
from pisa.utils.format import text2tex
from pisa.utils.hash import hash_obj
from pisa.utils.fileio import from_file
from pisa.utils import hdf
from pisa.utils.log import logging, set_verbosity


__all__ = ['Events', 'Data',
           'test_Events', 'test_Data']


# TODO: test hash function (attr)
class Events(FlavIntData):
    """Container for storing events, including metadata about the events.

    Examples
    --------
    >>> from pisa.core.binning import OneDimBinning, MultiDimBinning

    >>> # Load events from a PISA HDF5 file
    >>> events = Events('events/pingu_v39/events__pingu__v39__runs_620-622__proc_v5.1__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5')

    >>> # Apply a simple cut
    >>> events.applyCut('(true_coszen <= 0.5) & (true_energy <= 70)')
    >>> np.max(events[fi]['true_coszen']) <= 0.5
    True

    >>> # Apply an "inbounds" cut via a OneDimBinning
    >>> true_e_binning = OneDimBinning(
    ...    name='true_energy', num_bins=80, is_log=True,
    ...    domain=[10, 60]*ureg.GeV
    ... )
    >>> events.keepInbounds(true_e_binning)
    >>> np.min(events[fi]['true_energy']) >= 10
    True

    >>> print [(k, events.metadata[k]) for k in sorted(events.metadata.keys())]
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
        if isinstance(val, basestring) or isinstance(val, h5py.Group):
            data, meta = self.__load(val)
        elif isinstance(val, Events):
            self.metadata = val.metadata
            data = val
        elif isinstance(val, dict):
            data = val
        self.metadata.update(meta)
        self.validate(data)
        self.update(data)
        self._hash = hash_obj(normQuant(self.metadata))

    def __str__(self):
        meta = [(str(k) + ' : ' + str(v)) for k, v in self.metadata.items()]
        #fields =
        return '\n'.join(meta)

    def __repr__(self):
        return str(self)

    @property
    def hash(self):
        return self._hash

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
        if isinstance(binning_cols, basestring):
            binning_cols = [binning_cols]
        assert weights_col is None or isinstance(weights_col, basestring)

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
        repr_flav_int = kinds[0]
        sample = [self[repr_flav_int][colname] for colname in binning_cols]
        err_weights = None
        hist_weights = None
        if weights_col is not None:
            hist_weights = self[repr_flav_int][weights_col]
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
                tex = kinds.tex()
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
        >>> applyCut("(true_energy >= 1) & (true_energy <= 80)")

        Do the opposite with "~" inverting the criteria
        >>> applyCut("~((true_energy >= 1) & (true_energy <= 80))")

        Numpy namespace is available for use via `np` prefix
        >>> applyCut("np.log10(true_energy) >= 0")

        """
        if keep_criteria in self.metadata['cuts']:
            return

        assert isinstance(keep_criteria, basestring)

        flavints_to_process = self.flavints()
        flavints_processed = []
        new_data = {}
        try:
            for flav_int in flavints_to_process:
                data_dict = self[flav_int]
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
                        field_name, 'self["%s"]["%s"]' %(flav_int, field_name)
                    )
                mask = eval(crit_str)
                new_data[flav_int] = {k: v[mask]
                                      for k, v in self[flav_int].iteritems()}
                flavints_processed.append(flav_int)
        except Exception:
            if (len(flavints_processed) > 0
                    and flavints_processed != flavints_to_process):
                logging.error('Events object is in an inconsistent state.'
                              ' Reverting cut for all flavInts.')
            raise
        else:
            for flav_int in flavints_to_process:
                self[flav_int] = new_data[flav_int]
                new_data[flav_int] = None
            self.metadata['cuts'].append(keep_criteria)

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


class Data(FlavIntDataGroup):
    """Container for storing events, including metadata about the events.

    Examples
    --------
    TODO(shivesh): docs
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

        # Get data and metadata from val
        meta = {}
        if isinstance(val, basestring) or isinstance(val, h5py.Group):
            data, meta = self.__load(val)
        elif isinstance(val, Data):
            data = val
            meta = val.metadata
        elif isinstance(val, dict) or isinstance(val, FlavIntDataGroup):
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

        # Instantiate a FlavIntDataGroup
        if data == dict():
            self._flavint_groups = []
        else:
            super(self.__class__, self).__init__(
                val=data, flavint_groups=flavint_groups
            )
            self.contains_neutrinos = True

        # Check consistency of flavints_joined
        if self.metadata['flavints_joined']:
            combined_types = []
            if self.contains_neutrinos:
                combined_types += [str(f) for f in self.flavint_groups]
            if self.contains_muons:
                combined_types += ['muons']
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

        self.update_hash()

    @property
    def hash(self):
        return self._hash

    @hash.setter
    def hash(self, val):
        self._hash = val

    def update_hash(self):
        self._hash = hash_obj(normQuant(self.metadata))

    @property
    def muons(self):
        if not self.contains_muons:
            raise AttributeError('No muons loaded in Data')
        return self._muons

    @muons.setter
    def muons(self, val):
        assert isinstance(val, dict)
        self.contains_muons = True
        self._muons = val

    @property
    def neutrinos(self):
        if not self.contains_neutrinos:
            raise AttributeError('No neutrinos loaded in Data')
        return dict(zip(self.keys(), self.values()))

    @property
    def names(self):
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


        Examples
        --------
        Keep events with true energies in [1, 80] GeV (note that units are not
        recognized, so have to be handled outside this method)
        >>> applyCut("(true_energy >= 1) & (true_energy <= 80)")

        Do the opposite with "~" inverting the criteria
        >>> applyCut("~((true_energy >= 1) & (true_energy <= 80))")

        Numpy namespace is available for use via `np` prefix
        >>> applyCut("np.log10(true_energy) >= 0")

        """
        if keep_criteria in self.metadata['cuts']:
            return

        assert isinstance(keep_criteria, basestring)

        fig_to_process = []
        if self.contains_neutrinos:
            fig_to_process += deepcopy(self.flavint_groups)
        if self.contains_muons:
            fig_to_process += ['muons']
        fig_processed = []
        new_data = {}
        try:
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
                new_data[fig] = {k: v[mask] for k, v in self[fig].iteritems()}
                fig_processed.append(fig)
        except:
            if len(fig_processed) > 0 and fig_processed != fig_to_process:
                logging.error('Data object is in an inconsistent state.'
                              ' Reverting cut for all flavInts.')
            raise
        else:
            for fig in fig_to_process:
                self[fig] = new_data[fig]
                new_data[fig] = None
            self.metadata['cuts'].append(keep_criteria)

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
        t_fidg = super(Data, self).transform_groups(flavint_groups)
        metadata = deepcopy(self.metadata)
        metadata['flavints_joined'] = [str(f) for f in t_fidg.flavint_groups]
        if self.contains_muons:
            metadata['flavints_joined'] += ['muons']
            t_dict = dict(t_fidg)
            t_dict['muons'] = self['muons']
            t_fidg = t_dict
        ret_obj = Data(t_fidg, metadata=metadata)
        ret_obj.update_hash()
        return ret_obj

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

        if isinstance(kinds, basestring):
            kinds = [kinds]
        if 'muons' not in kinds:
            kinds = self._parse_flavint_groups(kinds)
        kinds = kinds[0]

        if isinstance(binning_cols, basestring):
            binning_cols = [binning_cols]
        assert weights_col is None or isinstance(weights_col, basestring)

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
                    tex = kinds.tex()
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
                      mapset_name, errors=False):
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
            if not isinstance(nu_weights_col, basestring):
                raise TypeError('nu_weights_col should be a string. Got %s'
                                % type(nu_weights_col))
        if mu_weights_col is not None:
            if not isinstance(mu_weights_col, basestring):
                raise TypeError('mu_weights_col should be a string. Got %s'
                                % type(mu_weights_col))
        if not isinstance(errors, bool):
            raise TypeError('flag for whether to calculate errors or not '
                            'should be a boolean. Got %s.' % type(errors))
        outputs = []
        if self.contains_neutrinos:
            trans_nu_data = self.transform_groups(
                self.flavint_groups
            )
            for fig in trans_nu_data.iterkeys():
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
        return MapSet(maps=outputs, name=mapset_name)

    def __load(self, fname):
        try:
            data, meta = from_file(fname, return_attrs=True)
        except TypeError:
            data = from_file(fname)
            meta = None
        return data, meta

    def __getitem__(self, arg):
        if arg == 'muons':
            return self.muons
        tgt_obj = super(Data, self).__getitem__(arg)
        return tgt_obj

    def __setitem__(self, arg, value):
        if arg == 'muons':
            self.muons = value
            return
        super(Data, self).__setitem__(arg, value)

    def __add__(self, other):
        muons = None
        assert isinstance(other, Data)

        for key in self.metadata:
            if (key != 'flavints_joined' and
                    self.metadata[key] != other.metadata[key]):
                raise AssertionError(
                    'Metadata mismatch, key {0}, {1} != '
                    '{2}'.format(key, self.metadata[key],
                                 other.metadata[key])
                )
        metadata = deepcopy(self.metadata)

        if self.contains_muons:
            if other.contains_muons:
                muons = self._merge(deepcopy(self['muons']), other['muons'])
            else:
                muons = deepcopy(self['muons'])
        elif other.contains_muons:
            muons = deepcopy(other['muons'])

        if len(self.flavint_groups) == 0:
            if len(other.flavint_groups) == 0:
                a_fidg = FlavIntDataGroup(other)
        elif len(other.flavint_groups) == 0:
            a_fidg = FlavIntDataGroup(self)
        else:
            a_fidg = super(Data, self).__add__(other)
        metadata['flavints_joined'] = [str(f) for f in a_fidg.flavint_groups]

        if muons is not None:
            a_dict = dict(a_fidg)
            metadata['flavints_joined'] += ['muons']
            a_dict['muons'] = muons
            a_fidg = a_dict
        return Data(a_fidg, metadata=metadata)

    def __str__(self):
        meta = [(str(k) + ' : ' + str(v)) for k, v in self.metadata.items()]
        return '\n'.join(meta)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.meta_eq(other) and self.data_eq(other)


def test_Events():
    """Unit tests for Events class"""
    from pisa.utils.flavInt import NuFlavInt
    # Instantiate empty object
    events = Events()

    # Instantiate from PISA events HDF5 file
    events = Events('events/pingu_v39/events__pingu__v39__runs_620-622__proc_v5.1__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5')

    # Apply a simple cut
    events.applyCut('(true_coszen <= 0.5) & (true_energy <= 70)')
    for fi in events.flavints():
        assert np.max(events[fi]['true_coszen']) <= 0.5
        assert np.max(events[fi]['true_energy']) <= 70

    # Apply an "inbounds" cut via a OneDimBinning
    true_e_binning = OneDimBinning(
        name='true_energy', num_bins=80, is_log=True, domain=[10, 60]*ureg.GeV
    )
    events.keepInbounds(true_e_binning)
    for fi in events.flavints():
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
    events.keepInbounds(mdb)
    for fi in events.flavints():
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
        events.applyCut('(true_energy >= 30) & (true_energy <= 40)')
    except Exception:
        pass
    else:
        raise Exception('Should not have been able to apply the cut!')
    for fi in events.flavints():
        if fi == NuFlavInt('nutaunc'):
            continue
        assert np.min(events[fi]['true_energy']) < 30

    logging.info(
        '<< PASSED : test_Events >> (note:'
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

    muon_file = 'Level7_muongun.12370_15.pckl'
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

    # Apply a simple cut
    data.applyCut('(zenith <= 1.1) & (energy <= 200)')
    for fi in data.flavint_groups:
        assert np.max(data[fi]['zenith']) <= 1.1
        assert np.max(data[fi]['energy']) <= 200

    # Apply an "inbounds" cut via a OneDimBinning
    e_binning = OneDimBinning(
        name='energy', num_bins=80, is_log=True, domain=[10, 200]*ureg.GeV
    )
    data.keepInbounds(e_binning)
    for fi in data.flavint_groups:
        assert np.min(data[fi]['energy']) >= 10
        assert np.max(data[fi]['energy']) <= 200

    # Apply an "inbounds" cut via a MultiDimBinning
    e_binning = OneDimBinning(
        name='energy', num_bins=80, is_log=True, domain=[20, 210]*ureg.GeV
    )
    cz_binning = OneDimBinning(
        name='zenith', num_bins=40, is_lin=True, domain=[0.1, 1.8*np.pi]
    )
    mdb = MultiDimBinning([e_binning, cz_binning])
    data.keepInbounds(mdb)
    for fi in data.flavint_groups:
        assert np.min(data[fi]['energy']) >= 20
        assert np.max(data[fi]['energy']) <= 210
        assert np.min(data[fi]['zenith']) >= 0.1
        assert np.max(data[fi]['zenith']) <= 1.8*np.pi

    # Now try to apply a cut that fails on one flav/int (since the field will
    # be missing) and make sure that the cut did not get applied anywhere in
    # the end (i.e., it is rolled back)
    sub_evts = data['nue+nuebar']
    sub_evts.pop('energy')
    data['nue+nuebar'] = sub_evts
    try:
        data.applyCut('(energy >= 30) & (energy <= 40)')
    except Exception:
        pass
    else:
        raise Exception('Should not have been able to apply the cut!')
    for fi in data.flavint_groups:
        if fi == NuFlavIntGroup('nue+nuebar'):
            continue
        assert np.min(data[fi]['energy']) < 30

    data.save('/tmp/test_FlavIntDataGroup.json')
    data.save('/tmp/test_FlavIntDataGroup.hdf5')
    data = Data('/tmp/test_FlavIntDataGroup.json')
    data = Data(val='/tmp/test_FlavIntDataGroup.hdf5')

    d3 = data + data2 + m
    logging.debug(str((d3)))
    d3_com = d3.transform_groups(['nue+nuebar+numu+numubar'])
    logging.debug(str((d3_com)))

    logging.info('<< PASSED : test_Data >>')


if __name__ == "__main__":
    set_verbosity(1)
    test_Events()
    test_Data()
