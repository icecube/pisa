"""
PIDSpec class for importing, working with, and storing PID specifications.
"""


from __future__ import division

import collections
import os

import h5py
# Note that the form of the numpy import is intentional, so that cuts -- which
# are exectuted with `eval` -- will have access to numpy's namespace without
# explicit reference to numpy. It's a hack, but it works well.
import numpy as np
from numpy import *

import pisa.utils.flavInt as flavInt
from pisa.utils.jsons import from_json
import pisa.utils.resources as resources


__all__ = ['PIDSpec']

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


class PIDSpec(object):
    """Class for importing, applying, and storing particle identification (PID)
    specifications.

    Parameters
    ----------
    pid_specs : string or dict
        Specifications of PID categorization criteria and the fields required
        to apply those criteria
    detector, geom, proc_ver, pid_spec_ver : strings
        Detector name, version of detector's geometry, processing version
        applied, and PID specification version to use. These must all be found
        in the `pid_specs` file/dict.

    Notes
    -----
    All information describing the PID specifications is loaded from a
    file with the following defined format::

        {
          "<detector name>": {
            "<geometry version>": {
              "<processing version>": {
                "<pid version>": {
                  "field_map": {
                    "<field1>": "<path to field in I3 file>",
                    "<field2>": "<path to field in I3 file>",
                    ...
                  },
                  "criteria": {
                    "<signature 1>": "<criteria>",
                    "<signature 2>": "<criteria>",
                  }
                }
              }
            }
          }
        }

    """
    def __init__(self, detector, geom, proc_ver, pid_spec_ver=1,
                 pid_specs=None):
        geom = str(geom)
        proc_ver = str(proc_ver)
        pid_spec_ver = str(pid_spec_ver)

        if pid_specs is None:
            pid_specs = 'pid/pid_specifications.json'
        if isinstance(pid_specs, str):
            pid_specs = from_json(resources.find_resource(pid_specs))
        elif isinstance(pid_specs, collections.Mapping):
            pass
        else:
            raise TypeError('Unhandled `pid_specs` type: "%s"' %
                            type(data_proc_params))
        self.detector = detector
        self.proc_ver = proc_ver
        self.pid_spec_ver = str(pid_spec_ver)
        d = pid_specs
        all_k = []
        for wanted_key in [detector, geom, proc_ver, pid_spec_ver]:
            wanted_key = wanted_key.replace("'","").lower()
            for orig_dict_key, subdict in d.items():
                dict_key = orig_dict_key.replace("'","").lower()
                if (dict_key == wanted_key):
                    d = subdict
                    all_k.append(orig_dict_key)
        if len(all_k) != 4:
            raise ValueError(
                'Could not find %s'
                %str([detector, geom, proc_ver, pid_spec_ver])
            )
        self.pid_spec = pid_specs[all_k[0]][all_k[1]][all_k[2]][all_k[3]]

        # Enforce rules on PID spec:
        self.validatePIDSpec(self.pid_spec)

    @staticmethod
    def validatePIDSpec(pids):
        """Validate a PID specification"""
        # TODO: implement validation
        #for signature, pidspec in pids.items():
        #    # Particle names are lower-case strings with no surrounding
        #    # whitespace
        #    assert isinstance(signature, str)
        #    assert signature == signature.lower()
        #    assert signature == signature.strip()
        #    # Has appropriate keys (and no extra)
        #    assert len(pidspec) == 2
        #    assert 'fields' in pidspec
        #    assert 'criteria' in pidspec
        #    assert not isinstance(pidspec['fields'], str)
        #    # 'fields' contains a sequence
        #    assert hasattr(pidspec['fields'], '__iter__') and \
        #            not isinstance(pidspec['fields'], str)
        #    # 'criteria' contains a string
        #    assert isinstance(pidspec['criteria'], str)
        return

    def get_signatures(self):
        return sorted(self.pid_spec['criteria'].keys())

    def validate_signatures(self, signatures):
        if isinstance(signatures, str):
            signatures = [signatures]
        for signature in signatures:
            assert isinstance(signature, str), \
                    'Signature "%s" is not string.' % signature
            if signature not in self.pid_spec['criteria']:
                raise ValueError(
                    'Invalid signature "%s": Not avilable in PID specification;'
                    ' valid signatures are: %s' %
                    (signature, ', '.join(['"%s"'%vn for vn in valid_names]))
                )

    def applyPID(self, events, signatures=None, return_fields=None, cuts=None):
        """Divide events in `events` that fall into specified PID signature(s).

        Parameters
        ----------
        events : Events object or convertible thereto
        signatures : string, sequence of strings, or None
            Signatures to return events for.
            If sring or sequence of strings, return events for the specified
            signature(s); if None, use all signatures in the PID specification.
        return_fields : sequence of strings or None

        Returns
        -------
        Separated events FlavIntData object with each flavint leaf node
        populated by a dictionary formatted as follows::

            {
              '<signature1>': {
                '<return_field1>': array,
                '<return_field2>': array,
                 ...
              },
              ...
            }

        """
        # Interpret `signatures`
        if isinstance(signatures, str):
            signatures = [signatures]
        elif signatures is None:
            signatures = self.get_signatures()

        self.validate_signatures(signatures)

        if isinstance(return_fields, str):
            return_fields = [return_fields]

        separated_events = flavInt.FlavIntData()
        # Outer loop is over flavint, so that same data is processed multiply
        # for different PID signatures, rather than new data loaded for each
        # PID spec. (Theoretically faster.)
        for flavint in events.flavints:
            src = events[flavint]
            dest = separated_events[flavint] = {}
            if return_fields is None:
                fields_to_get = sorted(src.keys())
            else:
                fields_to_get = return_fields

            if len(fields_to_get) == 0:
                raise ValueError('No fields to get.')

            for signature in signatures:
                dest[signature] = {}
                pid_criteria = self.pid_spec['criteria'][signature]

                # Load the fields into global namespace that are necessary to
                # apply PID criteria
                for field in sorted(self.pid_spec['field_map'].keys()):
                    globals()[field] = src[field]

                # Evaluate PID criteria, returning a boolean array
                bool_idx = eval(pid_criteria)

                # Store all requested fields that meet the criteria
                for field in fields_to_get:
                    dest[signature][field] = src[field][bool_idx]

        return separated_events

    @staticmethod
    def aggregate(separated_events):
        agg_events = {}
        for flavint in separated_events.flavints:
            for sig, datadict in separated_events[flavint].items():
                if sig not in agg_events:
                    agg_events[sig] = {}
                for field, data in datadict.items():
                    if field not in agg_events[sig]:
                        agg_events[sig][field] = []
                    agg_events[sig][field].append(data)
        # Concatenate the collected data arrays for final output
        return {sig: {k:np.concatenate(v) for k,v in dat.items()}
                for sig,dat in  agg_events.items()}
