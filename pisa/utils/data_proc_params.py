#!/usr/bin/env python

"""
DataProcParams class for importing, working with, and storing data processing
parameters (e.g., PINGU's V5 processing).
"""


from __future__ import absolute_import, division

from collections.abc import Mapping, Sequence
from collections import OrderedDict
from copy import deepcopy
import os
import re

import h5py
# Note that the form of the numpy import is intentional, so that cuts -- which
# are exectuted with `eval` -- will have access to numpy's namespace without
# explicit reference to numpy. It's a hack, but it works well.
from numpy import * # pylint: disable=wildcard-import, unused-wildcard-import, redefined-builtin
import numpy # pylint: disable=unused-import
import numpy as np # pylint: disable=reimported

from pisa.utils import jsons
from pisa.utils.flavInt import NuFlav, IntType, FlavIntData
from pisa.utils.log import logging
from pisa.utils import resources


__all__ = ['MULTI_PART_FIELDS', 'NU_PDG_CODES', 'DataProcParams']

__author__ = 'J.L. Lanfranchi'

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


MULTI_PART_FIELDS = [
    'I3MCTree',
]

NU_PDG_CODES = [-12, 12, -14, 14, -16, 16]


class DataProcParams(dict):
    """Class for importing, working with, and storing data processing
    parameters.

    Implements cutting and particle identification (PID) functionality that can
    be applied to MC/data that have the specified verion of processing applied
    to it.

    Parameters
    ----------
    data_proc_params : string or dict
        If string: looks for the corresponding JSON resource (file) and loads
          the contents as a data_proc_params dict
        If dict: taken to be data_proc_params dict
        The data_proc_params dict must follow the format described below.
    detector : string
        Converted to lower-case string which must be a detector key in
        data_proc_params dict
    proc_ver
        Converted to lower-case string which must be a proc_ver key in
        data_proc_params dict

    Notes
    -----
    All information describing the processing version is loaded from a JSON
    file with the following defined format:

    Note that the following common cuts are defined in this class and so
    needn't be defined in the JSON file:
                      '1' : Select particles
                     '-1' : Select anti-particles
                     'cc' : Select charged-current (CC) interactions
                     'nc' : Select neutral-current (NC) interactions
       'true_upgoing_zen' : Select true-upgoing events by zenith angle
    'true_upgoing_coszen' : Select true-upgoing events by cos(zenith) angle

    data_proc_params dictionary format (and same for corresponding JSON file):

    {
      # Specify the detector name, lower case

      "<lower-case detector name>": {


        # Specify the processing version, lower case
        # Examples in PINGU include "4", "5", and "5.1"

        "<lower-case processing version>": {


          # Mapping from standard names to path where these can be found in
          # source HDF5 file; separate HDF5 path nodes with a forward-slash.
          #
          # Fields that cuts or particles in the "cuts"/"pid" sections below
          # require (e.g., "cuts_step_1" for PINGU v5 processing), must be
          # added here so the code knows how to extract the info from the
          # source HDF5 file.
          #
          # Outputting a PID field to the destination PISA HDF5 file will *not*
          # be done if the "pid" field is omitted below.
          #
          # In addition to the below-named fields, "true_coszen" and
          # "reco_coszen" are generated from the data from the "true_zenith"
          # and "reco_zenith" fields, respectively. So any of those fields can
          # be used via the aforementioned names.

          "field_map": {
            "run": "<HDF5 path to corresponding node>",
            "nu_code": "<HDF5 path to corresponding node>",
            "true_energy": "<HDF5 path to corresponding node>",
            "true_zenith": "<HDF5 path to corresponding node>",
            "reco_energy": "<HDF5 path to corresponding node>",
            "reco_zenith": "<HDF5 path to corresponding node>",
            "one_weight": "<HDF5 path to corresponding node>",
            "generator_volume": "<HDF5 path to corresponding node>",
            "generator_radius": "<HDF5 path to corresponding node>",
            "detection_length": "<HDF5 path to corresponding node>",
            "interaction_type": "<HDF5 path to corresponding node>",
            "azimuth_min": "<HDF5 path to corresponding node>",
            "azimuth_max": "<HDF5 path to corresponding node>",
            "zenith_min": "<HDF5 path to corresponding node>",
            "zenith_max": "<HDF5 path to corresponding node>",
            "energy_log_min": "<HDF5 path to corresponding node>",
            "energy_log_max": "<HDF5 path to corresponding node>",
            "num_events_per_file": "<HDF5 path to corresponding node>",
            "sim_spectral_index": "<HDF5 path to corresponding node>",
            "pid": "<HDF5 path to corresponding node>",
          },


          # Mapping from file's nu code to PDG nu codes (only necessary if
          # nu_code values are not the PDG codes listed below)

          "nu_code_to_pdg_map": {
            "<source nue code>":        12,
            "<source nue_bar code>":   -12,
            "<source numu code>":       14,
            "<source numu_bar code>":  -14,
            "<source nutau code>":      16,
            "<source nutau_bar code>": -16
          },


          # Specify standard cuts

          "cuts": {


            # Cut name; "bjrej" and "analysis" listed here are just example
            # names for cuts one might specify

            "bgrej": {

              # Which of the fields in the field_map (and the derived fields
              # such as true_coszen and reco_coszen) are required for this cut?

              "fields": ["<field1>", "<field2>", ... ],

              # Expression for an event to pass the cut (not get thrown away);
              # see below for details on specifying an expression

              "pass_if": "<expression>"
            },

            "analysis": {
              "fields": ["<field1>", "<field2>", ... ],
              "pass_if": "<expression>"
            },

            "<cut name>": {
              "fields": ["<field1>", "<field2>", ... ],
              "pass_if": "<expression>"
            }
          },


          # Particle identification section

          "pid": {


            # Name of the particle (case-insensitive); e.g., in PINGU this
            # would be "trck" or "cscd"

            "<particle name 1>": {


              # Which of the fields in the field_map (and the derived fields
              # such as true_coszen and reco_coszen) are required for this cut?

              "field": [<field1>, <field2>, ...],


              # Expression for an event to be classified as this type of
              # particle; # see below for details on specifying an expression

              "criteria": "<expression>"
            }

            "<particle name 2>": {
              "field": [<field1>, <field2>, ...],
              "criteria": "<expression>"
            }
          }
        }
      }
    }

    Note that cuts "pass_if" and pid "criteria" expressions can make use of the
    numpy namespace and have access to any columns extracted from the source
    HDF5 file, by the standardized names given in the "field_map". For example,
    if the following "fields" are specified for a cut in the data_proc_params
    dict:
        ["cuts_step_1", "cuts_step_2"]
    then the following is a valid "pass_if" expression:
        "(reco_zenith > pi/2) & (cuts_step_1 == 1) & (cuts_step_2 == 1)"

    """
    def __init__(self, detector, proc_ver, data_proc_params=None):
        super().__init__()
        if data_proc_params is None:
            data_proc_params = 'events/data_proc_params.json'
        if isinstance(data_proc_params, str):
            ps = jsons.from_json(resources.find_resource(data_proc_params))
        elif isinstance(data_proc_params, dict):
            ps = data_proc_params
        else:
            raise TypeError('Unhandled data_proc_params type passed in arg: ' +
                            type(data_proc_params))
        self.detector = detector
        self.proc_ver = str(proc_ver)
        self.det_key = [k for k in ps.keys()
                        if k.lower() == self.detector.lower()][0]
        for key in ps[self.det_key].keys():
            lk = key.lower()
            lpv = self.proc_ver.lower()
            if lk == lpv or ('v'+lk == lpv) or (lk == 'v'+lpv):
                self.procver_key = key
                # This works for PINGU
            elif ('msu_'+lk == lpv) or (lk == 'msu_'+lpv):
                self.procver_key = key
            elif ('nbi_'+lk == lpv) or (lk == 'nbi_'+lpv):
                self.procver_key = key
                # Generalising for DeepCore and different selections
        ps = ps[self.det_key][self.procver_key]
        self.update(ps)

        self.trans_nu_code = False
        if 'nu_code_to_pdg_map' in self:
            self.trans_nu_code = True
            try:
                self.nu_code_to_pdg_map = {
                    int(code): pdg
                    for code, pdg in self['nu_code_to_pdg_map'].items()
                }
            except:
                self.nu_code_to_pdg_map = self['nu_code_to_pdg_map']

        # NOTE: the keys are strings so the particular string formatting is
        # important for indexing into the dict!

        # Add generic cuts
        self['cuts'].update({
            # Cut for particles only (no anti-particles)
            str(NuFlav(12).bar_code).lower():
                {'fields': ['nu_code'], 'pass_if': 'nu_code > 0'},
            # Cut for anti-particles only (no particles)
            str(NuFlav(-12).bar_code).lower():
                {'fields': ['nu_code'], 'pass_if': 'nu_code < 0'},
            # Cut for charged-current interactions only
            str(IntType('cc')).lower():
                {'fields': ['interaction_type'],
                 'pass_if': 'interaction_type == 1'},
            # Cut for neutral-current interactions only
            str(IntType('nc')).lower():
                {'fields': ['interaction_type'],
                 'pass_if': 'interaction_type == 2'},
            # True-upgoing cut usinng the zenith field
            'true_upgoing_zen':
                {'fields': ['true_zenith'], 'pass_if': 'true_zenith > pi/2'},
            # True-upgoing cut usinng the cosine-zenith field
            'true_upgoing_coszen':
                {'fields': ['true_coszen'], 'pass_if': 'true_coszen < 0'},
        })

        # Enforce rules on cuts:
        self.validate_cut_spec(self['cuts'])

    @staticmethod
    def validate_cut_spec(cuts):
        """Validate a cut specification dictionary"""
        for cutname, cutspec in cuts.items():
            # Cut names are lower-case strings with no surrounding whitespace
            assert isinstance(cutname, str)
            assert cutname == cutname.lower()
            assert cutname == cutname.strip()
            # Has appropriate keys (and no extra)
            assert len(cutspec) == 2
            assert 'fields' in cutspec
            assert 'pass_if' in cutspec
            assert not isinstance(cutspec['fields'], str)
            # 'fields' contains a sequence
            assert hasattr(cutspec['fields'], '__iter__') and \
                    not isinstance(cutspec['fields'], str)
            # 'pass_if' contains a string
            assert isinstance(cutspec['pass_if'], str)

    @staticmethod
    def validate_pid_spec(pids):
        """Validate a PID specification dictionary"""
        for particle_name, pidspec in pids.items():
            # Particle names are lower-case strings with no surrounding
            # whitespace
            assert isinstance(particle_name, str)
            assert particle_name == particle_name.lower()
            assert particle_name == particle_name.strip()
            # Has appropriate keys (and no extra)
            assert len(pidspec) == 2
            assert 'fields' in pidspec
            assert 'criteria' in pidspec
            assert not isinstance(pidspec['fields'], str)
            # 'fields' contains a sequence
            assert hasattr(pidspec['fields'], '__iter__') and \
                    not isinstance(pidspec['fields'], str)
            # 'criteria' contains a string
            assert isinstance(pidspec['criteria'], str)

    # TODO: prefix the field names with e.g. "$" such that anything that is
    # _not_ prefixed by this is not replaced. This allows for righer
    # expresssions (but also dangerous things...).
    @staticmethod
    def retrieve_expression(h5group, expression):
        """Retrieve data from an HDF5 group `h5group` according to
        `expresssion`. This can apply expressions with simple mathematical
        operators and numpy functions to multiple fields within the HDF5 file
        to derive the output. Python keywords are _not_ allowed, since they
        may alias with a name.

        Refer to any numpy functions by prefixing with either "np.<func>" or
        "numpy.<func>". In order to specify division, spaces must surround the
        forward slash, such that it isn't interpreted as a path.

        Nodes in the HDF5 hierarchy are separated by forward slashes ("/") in a
        path spec. We restrict valid HDF5 node names to contain the characters
        a-z, A-Z, 0-9, peroids ("."), and underscores ("_"). with the
        additional restriction that the node name must not start with a period
        or a number, and a path cannot start with a slash.


        Parameters
        ----------
        h5group : h5py Group
        expression : string
            Expression to evaluate.

        Returns
        -------
        result : result of evaluating `expression`

        Examples
        --------
        >>> retrieve_expression('np.sqrt(MCneutrino/x**2 + MCneutrino/y**2)')

        Indexing into the data arrays can also be performed, and numpy masks
        used as usual:

        >>> expr = 'I3MCTree/energy[I3MCTree/event == I3EventHeader[0]

        """
        h5path_re = re.compile(
            r'''
            ([a-z_]          # First character must be letter or underscore
             [a-z0-9_.]*     # 0 or more legal chars: letters, numbers, _, .
             (?:             # (Do not return the following group separately)
                [/]{0,1}     # Next character CAN be no or 1 front-slash
                [a-z0-9_.]+  # But a slash *must* be followed by legal chars
             )*              # Slash+chars pattern might not occur, or repeat
            )''', re.VERBOSE | re.IGNORECASE
        )
        numpy_re = re.compile(r'^(np|numpy)\.[a-z_.]+', re.IGNORECASE)

        eval_str = expression
        intermediate_data = {}
        for h5path in h5path_re.findall(expression):
            if numpy_re.match(h5path):
                continue
            intermediate_data[h5path] = DataProcParams.retrieve_node_data(
                h5group, h5path
            )
            eval_str = eval_str.replace(h5path,
                                        "intermediate_data['%s']"%h5path)

        try:
            result = eval(eval_str) # pylint: disable=eval-used
        except:
            logging.error('`expression` "%s" was translated into `eval_str`'
                          ' "%s" and failed to evaluate.',
                          expression, eval_str)
            raise

        return result

    @staticmethod
    def retrieve_node_data(h5group, address, allow_missing=False):
        """Retrieve data from an HDF5 group `group` at address `address`.
        Levels of hierarchy are separated by forward-slashes ('/').

        See h5py for further details on specifying a valid `address`.
        """
        subgroup = h5group
        for sub_addy in address.split('/'):
            try:
                subgroup = subgroup[sub_addy]
            except KeyError:
                if allow_missing:
                    return None
                raise
        return subgroup

    @staticmethod
    def populate_global_namespace(h5group, field_map, allow_missing=False):
        """Populate the Python global namespace with variables named as the
        keys in `field_map` and values loaded from the `h5group` at addresses
        specified by the corresponding values in `field_map`.
        """
        for var, h5path in field_map.items():
            try:
                value = DataProcParams.retrieve_node_data(
                    h5group, h5path, allow_missing=False
                )
            except KeyError:
                if allow_missing:
                    return None
                raise
            globals()[var] = value

    # TODO: make the following behave like `retrieve_expression` method which
    # does not rely on populating globals (just a dict, the name of which gets
    # substituted in where approprite to the expression) to work.
    @staticmethod
    def cut_bool_idx(h5group, cut_fields, keep_criteria):
        """Return numpy boolean indexing for data in `h5group` given a cut
        specified using `cut_fields` in the `h5group` and evaluation criteria
        `keep_criteria`

        Parameters
        ----------
        h5group : h5py node/entity
        cut_fields : field_map dict
        keep_criteria : string

        Returns
        -------
        bool_idx : numpy array (1=keep, 0=reject)

        """
        DataProcParams.populate_global_namespace(h5group, cut_fields)
        bool_idx = eval(keep_criteria) # pylint: disable=eval-used
        return bool_idx

    def get_data(self, h5, run_settings=None, flav=None):
        """Get data attached to an HDF5 node, returned as a dictionary.

        The returned dictionary's keys match those in the field_map and the
        dict's values are the data from the HDF5's nodes found at the addresses
        specified as values in the field_map

        Parameters
        ----------
        file_type : string, one of {'mc', 'data'}

        """
        not_fields_in_data = ['I3MCWeightDict', 'PrimaryNu', 'trueNeutrino']
        myfile = False
        try:
            if isinstance(h5, str):
                myfile = True
                h5 = h5py.File(os.path.expandvars(os.path.expanduser(h5)),
                               mode='r')
            data = OrderedDict()
            for name, path in self['field_map'].items():
                datum = self.retrieve_expression(h5, path)
                path_parts = path.split('/')
                if (file_type == 'data' and 'I3MCWeightDict' in path_parts
                        or 'PrimaryNu' in path_parts or 'trueNeutrino' in path_parts):
                    continue

                if path_parts[0] == 'I3MCTree' and path_parts[-1] != 'Event':
                    evts = self.retrieve_node_data(
                        h5, '/'.join(path_parts[:-1] + ['Event'])
                    )
                    pdgs = self.retrieve_node_data(
                        h5, '/'.join(path_parts[:-1] + ['pdg_encoding'])
                    )
                    energies = self.retrieve_node_data(
                        h5, '/'.join(path_parts[:-1] + ['energy'])
                    )

                    # Looping here is ugly and slow, but people don't make the
                    # Event field unique, so the only thing you can count on is
                    # that if the event number changes in sequence, you're in a
                    # different Event (for now, I think). The actual Event
                    # number can be repeated elsewhere, though.
                    #
                    # This makes for wonderfully reproducible results.
                    # </sardonic laughter>
                    new_datum = []
                    this_evt = np.nan
                    this_d = None
                    for d, evt, pdg, egy in zip(datum, evts, pdgs, energies):
                        if evt != this_evt:
                            if this_d is not None:
                                new_datum.append(this_d)
                            this_egy = -np.inf
                            this_d = None
                            this_evt = evt
                        if egy > this_egy and pdg in NU_PDG_CODES:
                            this_egy = egy
                            this_d = d
                    if this_d is not None:
                        new_datum.append(this_d)
                    datum = new_datum

                data[name] = np.array(datum)

        finally:
            if myfile and isinstance(h5, h5py.File):
                try:
                    h5.close()
                except: # TODO: specify exception type(s)!
                    pass

        self.interpret_data(data)
        # TODO: enable consistency checks here & implement in run_settings
        #if run_settings is not None:
        #    run_settings.consistency_checks(data, flav=flav)

        # TODO: implement flav filtering (or not? or more advanced filtering?)
        return data

    def interpret_data(self, data):
        """Perform mappings from non-standard to standard values (such as
        translating non-PDG neutrino flavor codes to PDG codes) and add
        fields expected to be useful (such as coszen, derived from zen fields).

        Attach / reattach the translated/new fields to the `data` object passed
        into this methd.

        """
        for k, v in data.items():
            if isinstance(v, Sequence):
                data[k] = v[0]

        if self.trans_nu_code:
            data['nu_code'] = [
                self.nu_code_to_pdg_map[code] for code in data['nu_code']
            ]
        if 'true_zenith' in data:
            data['true_coszen'] = np.cos(data['true_zenith'])
        if 'reco_zenith' in data:
            data['reco_coszen'] = np.cos(data['reco_zenith'])
        return data

    @staticmethod
    def subselect(data, fields, indices=None):
        if isinstance(data, FlavIntData):
            outdata = FlavIntData()
            for flavint in data.flavints:
                outdata[flavint] = DataProcParams.subselect(data[flavint],
                                                            fields=fields,
                                                            indices=indices)
        elif isinstance(data, Mapping):
            if indices is None:
                return {k:v for k, v in data.items() if k in fields}
            return {k:v[indices] for k, v in data.items() if k in fields}

    def apply_cuts(self, data, cuts, boolean_op='&', return_fields=None):
        """Perform `cuts` on `data` and return a dict containing
        `return_fields` from events that pass the cuts.

        Parameters
        ----------
        data : single-level dict or FlavIntData object
        cuts : string or dict, or sequence thereof
        boolean_op : string
        return_fields : string or sequence thereof
        """
        if isinstance(data, FlavIntData):
            outdata = FlavIntData()
            for flavint in data.flavints:
                outdata[flavint] = self.apply_cuts(
                    data[flavint], cuts=cuts, boolean_op=boolean_op,
                    return_fields=return_fields
                )
            return outdata

        if isinstance(cuts, (str, dict)):
            cuts = [cuts]

        # Default is to return all fields
        if return_fields is None:
            return_fields = data.keys()

        # If no cuts specified, return all data from specified fields
        if len(cuts) == 0:
            return self.subselect(data, return_fields)

        cut_strings = set()
        cut_fields = set()
        for cut in cuts:
            if isinstance(cut, dict):
                self.validate_cut_spec(cut)
            elif cut.lower() in self['cuts']:
                cut = self['cuts'][cut.lower()]
            else:
                raise Exception('Unrecognized or invalid cut: "'+str(cut)+'"')
            cut_strings.add(cut['pass_if'])
            cut_fields.update(cut['fields'])

        # Combine cut criteria strings together with boolean operation
        cut_string = boolean_op.join(['('+cs+')' for cs in cut_strings])

        # Load the fields necessary for the cut into the global namespace
        for field in set(cut_fields):
            globals()[field] = data[field]

        # Evaluate cuts, returning a boolean array
        try:
            bool_idx = eval(cut_string) # pylint: disable=eval-used
        except:
            logging.error('Failed to evaluate `cut_string` "%s"', cut_string)
            raise

        # Return specified (or all) fields, indexed by boolean array
        return {f:np.array(data[f])[bool_idx] for f in return_fields}
