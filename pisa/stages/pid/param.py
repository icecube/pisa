"""
The purpose of this stage is to simulate event classification sorting the
reconstructed particles into PID-signature channels.

For each PID signature, the input map is transformed by the probability for
events in each of its bins to be ID'd as that signature. Therefore the ouptut
binning is similar to the input binning, but with the added 'pid' dimension,
which has as many bins as PID signatures.
"""


from __future__ import division

from collections.abc import Mapping
from collections import OrderedDict

import numpy as np
import scipy as sp

# NOTE: need both versions of the imported names, as eval strings can name
# numpy and scipy either ways
import numpy
import scipy

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.fileio import from_file
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile


__all__ = ['param']

__author__ = 'L. Schulte, J.L. Lanfranchi, S. Mandalia'

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


def load_pid_energy_param(source):
    """Load pid energy-dependent parameterisation from file or dictionary.

    Parameters
    ----------
    source : string or mapping
        If string, interprete as resource location of the file; if mapping, use
        directly.

    Returns
    -------
    pid_energy_param_dict : OrderedDict
        Keys are `NuFlavIntGroup`s and values are callables of one arg.

    """
    # Get the original dict
    if isinstance(source, str):
        orig_dict = from_file(source)
    elif isinstance(source, Mapping):
        orig_dict = source
    else:
        raise TypeError('`source` must either be string or mapping; got %s'
                        ' instead.' % type(source))

    # Build dict with flavintgroups as keys; subdict with signatures as keys
    # and callables as values
    pid_energy_param_dict = OrderedDict()

    for flavintgroup_str, subdict in orig_dict.items():
        flavintgroup = NuFlavIntGroup(flavintgroup_str)

        pid_energy_param_dict[flavintgroup] = OrderedDict()

        for signature, sig_param_spec in subdict.items():
            if isinstance(sig_param_spec, str):
                sig_param_func = eval(sig_param_spec)
                if not callable(sig_param_func):
                    raise ValueError(
                        'Group %s PID signature %s param spec "%s" does'
                        ' not evaluate to a callable.'
                        % (flavintgroup_str, signature, sig_param_spec)
                    )
            elif callable(sig_param_spec):
                sig_param_func = sig_param_spec
            else:
                raise TypeError(
                    'Group %s PID signature %s parameterization is a "%s"'
                    ' but must be a string or callable.'
                    % (flavintgroup_str, signature, type(sig_param_spec))
                )

            pid_energy_param_dict[flavintgroup][signature] = sig_param_func

    return pid_energy_param_dict


class param(Stage):
    """Parameterised MC PID based on an input json file containing functions
    describing the PID as a function of energy.

    Transforms an input map of the specified particle "signature" (aka ID) into
    a map of the track-like events ('track') and a map of the shower-like events
    ('cascade').

    Parameters
    ----------
    params : ParamSet or sequence with which to instantiate a ParamSet

        Parameters which set everything besides the binning.

        If str, interpret as resource location and load params from resource.
        If dict, set contained params. Format expected is
            {'<param_name>': <Param object or passable to Param()>}

        Parameters required by this service are
            * pid_energy_paramfile : dict or filepath
                json file or equivalent dict containing the PID functions for
                each flavor. The structure should be:
                  {
                    "numu_cc + numubar_cc": {
                      "track" : "lambda E: some function",
                      "cascade" : "lambda E: 1 - some function"
                    },
                    "nue_cc + nuebar_cc": {
                      "track" : "lambda E: some function",
                      "cascade" : "lambda E: 1 - some function"
                    },
                    "nutau_cc + nutaubar_cc": {
                      "track" : "lambda E: some function",
                      "cascade" : "lambda E: 1 - some function"
                    },
                    "nuall_nc + nuallbar_nc": {
                      "track" : "lambda E: some function",
                      "cascade" : "lambda E: 1 - some function"
                    }
                  }

    particles : string

    input_names : sequence of strings

    transform_groups

    sum_grouped_flavints : bool

    input_binning : MultiDimBinning
        Arbitrary number of dimensions accepted. Contents of the input
        `pid_events` parameter defines the possible binning dimensions. Name(s)
        of given binning(s) must match to a reco variable in `pid_events`.

    output_binning : MultiDimBinning

    error_method : None, bool, or string

    transforms_cache_depth : int >= 0

    outputs_cache_depth : int >= 0

    memcache_deepcopy : bool

    debug_mode : None, bool, or string
        Whether to store extra debug info for this service.


    Input Names
    ----------
    The `inputs` container must include objects with `name` attributes:
        * 'nue_cc'
        * 'nuebar_cc'
        * 'numu_cc'
        * 'numubar_cc'
        * 'nutau_cc'
        * 'nutaubar_cc'
        * 'nuall_nc'
        * 'nuallbar_nc'

    Output Names
    ----------
    The `outputs` container generated by this service will be objects with the
    following `name` attribute; pid is added as a binning dimension:
        * 'nue_cc'
        * 'nuebar_cc'
        * 'numu_cc'
        * 'numubar_cc'
        * 'nutau_cc'
        * 'nutaubar_cc'
        * 'nuall_nc'
        * 'nuallbar_nc'

    """
    def __init__(self, params, particles, input_names, transform_groups,
                 sum_grouped_flavints, input_binning, output_binning,
                 memcache_deepcopy, error_method, transforms_cache_depth,
                 outputs_cache_depth, debug_mode=None):
        assert particles in ['muons', 'neutrinos']
        self.particles = particles.strip().lower()
        """Whether stage is instantiated to process neutrinos or muons"""

        self.transform_groups = flavintGroupsFromString(transform_groups)
        """Particle/interaction types to group for computing transforms"""

        self.sum_grouped_flavints = sum_grouped_flavints

        # All of the following params (and no more) must be passed via
        # the `params` argument.
        expected_params = (
            'pid_energy_paramfile'
        )

        if isinstance(input_names, str):
            input_names = input_names.replace(' ', '').split(',')

        if self.particles == 'neutrinos':
            if self.sum_grouped_flavints:
                output_names = [str(g) for g in self.transform_groups]
            else:
                output_names = input_names
        elif self.particles == 'muons':
            raise NotImplementedError('%s not implemented.' % self.particles)

        super().__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            memcache_deepcopy=memcache_deepcopy,
            input_binning=input_binning,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('sum_grouped_flavints')
        self.include_attrs_for_hashes('transform_groups')

        self.signatures = output_binning.pid.bin_names
        """PID signatures that this stage generates"""

        # If no bin names are present, use the integer bin indices instead
        if self.signatures is None:
            self.signatures = list(range(len(output_binning.pid)))

        # Define the transform binnning...

        # Note that Numpy broadcasting rules start with last axis and work
        # inwards. We want the input map (say MxN) to automatically be
        # broadcast to multiply into each of L PID bins. Therefore, if
        # we _prepend_ the PID dimension to the transform, we have an MxN input
        # multiplying an LxMxN transform, and Numpy treats this as L separate
        # MxN by MxN multiplies which populate an output array of dimension
        # LxMxN... exactly what we want, and with maximal computational
        # efficiency (for Numpy to handle, at least). If the user's output
        # binning does not follow the same ordering, this is okay, as the
        # output is passed through the `rebin` function each time it is
        # computed, and this takes care of any axis swapping necessary.

        self.transform_output_binning = (
            self.output_binning.pid * self.input_binning
        )

        self.ebin_centers = (
            self.input_binning.reco_energy.weighted_centers.m_as('GeV')
        )

        self.pid_energy_param_dict = None
        self._pid_energy_param_hash = None

    def validate_binning(self):
        """Validate input and output binning"""
        required_input_binning_dims = 'reco_energy', 'reco_coszen'
        required_output_binning_dims = 'reco_energy', 'reco_coszen', 'pid'

        msg = ('%s binning must contain dimensions %s, but has dimensions %s'
               ' instead.')

        if set(self.input_binning.names) != set(required_input_binning_dims):
            raise ValueError(msg % ('Input', required_input_binning_dims,
                                    self.input_binning.names))

        if set(self.output_binning.names) != set(required_output_binning_dims):
            raise ValueError(msg % ('Output', required_output_binning_dims,
                                    self.input_binning.names))

        # While output binning will have a 'pid' dimension, the remaining
        # dimensions must be the same in both input and output binnings
        for dim in self.input_binning.dims:
            if dim != self.output_binning[dim.name]:
                raise NotImplementedError(
                    'Input and output dimensions %s are not equal, but stage'
                    ' %s / service %s does not implement binning up- or'
                    ' downsampling.'
                    % (dim.name, self.stage_name, self.service_name)
                )

    def load_pid_energy_param(self, source):
        """Load pid energy-dependent parameterisation from file or dictionary.

        Parameters
        ----------
        source : string
            Resource location of the file

        """
        this_hash = hash_obj(source)
        if (self._pid_energy_param_hash is not None
                and this_hash == self._pid_energy_param_hash):
            return

        # Invalidate the hash and clear the entry, so we aren't left in an
        # inconsistent state if any of the below fails
        self._pid_energy_param_hash = None
        self.pid_energy_param_dict = None

        # Call external function for basic loading and conversion
        pid_energy_param_dict = load_pid_energy_param(source)

        # Perform validation
        for flavintgroup, subdict in pid_energy_param_dict.items():
            if set(subdict.keys()) != set(self.signatures):
                raise ValueError(
                    'Expected PID specs for %s, but the energy PID'
                    ' parameterization for %s specifies %s instead.'
                    % (self.signatures, flavintgroup, subdict.keys())
                )

        # Transform groups are implicitly defined by keys
        implicit_transform_groups = pid_energy_param_dict.keys()

        # Make sure these match the transform groups specified for the stage
        if set(implicit_transform_groups) != set(self.transform_groups):
            raise ValueError(
                'Transform groups (%s) defined implicitly by `source` "%s" do'
                ' not match those defined as the stage\'s configured'
                ' `transform_groups` (%s).'
                % (implicit_transform_groups, source, self.transform_groups)
            )

        # Verify that each input name--which specifies a flavint or
        # flavintgroup--is wholly encapsulated by one of the transform
        # flavintgroups
        for name in self.input_names:
            if not any(name in group for group in implicit_transform_groups):
                raise ValueError(
                    'Input "%s" either not present in or spans multiple'
                    ' transform groups (transform_groups = %s)'
                    % (name, implicit_transform_groups)
                )

        self.pid_energy_param_dict = pid_energy_param_dict
        self._pid_energy_param_hash = this_hash

    @profile
    def _compute_nominal_transforms(self):
        """Compute new PID transforms."""
        logging.debug('Updating pid.param PID histograms...')

        self.load_pid_energy_param(self.params.pid_energy_paramfile.value)

        nominal_transforms = []
        for xform_flavints in self.transform_groups:
            logging.debug('Working on %s PID', xform_flavints)

            xform_array = np.empty(self.transform_output_binning.shape)

            subdict = self.pid_energy_param_dict[xform_flavints]
            for signature, sig_param_func in subdict.items():
                # Get the PID probabilities vs. energy at the energy bins'
                # (weighted) centers
                pid1d = sig_param_func(self.ebin_centers)

                # Broadcast this 1d array across the reco_coszen dimension
                # since it's independent of reco_coszen
                broadcasted_pid = self.transform_output_binning.broadcast(
                    pid1d, from_dim='reco_energy', to_dims='reco_coszen'
                )

                pid_indexer = (
                    self.transform_output_binning.indexer(pid=signature)
                )

                # Assign the broadcasted array to the correct PID bin
                xform_array[pid_indexer] = broadcasted_pid

            if self.sum_grouped_flavints:
                xform_input_names = []
                for input_name in self.input_names:
                    input_flavs = NuFlavIntGroup(input_name)
                    if set(xform_flavints).intersection(input_flavs):
                        xform_input_names.append(input_name)

                for output_name in self.output_names:
                    if output_name not in xform_flavints:
                        continue
                    xform = BinnedTensorTransform(
                        input_names=xform_input_names,
                        output_name=str(xform_flavints),
                        input_binning=self.input_binning,
                        output_binning=self.transform_output_binning,
                        xform_array=xform_array,
                        sum_inputs=self.sum_grouped_flavints
                    )
                    nominal_transforms.append(xform)

            else:
                for input_name in self.input_names:
                    if input_name not in xform_flavints:
                        continue
                    xform = BinnedTensorTransform(
                        input_names=input_name,
                        output_name=input_name,
                        input_binning=self.input_binning,
                        output_binning=self.transform_output_binning,
                        xform_array=xform_array,
                    )
                    nominal_transforms.append(xform)

        return TransformSet(transforms=nominal_transforms)

    def _compute_transforms(self):
        """There are no systematics in this stage, so the transforms are just
        the nominal transforms. Thus, this function just returns the nominal
        transforms, computed by `_compute_nominal_transforms`..
        """
        return self.nominal_transforms

    def validate_params(self, params):
        """Do checks on the parameters"""
        val = params.pid_energy_paramfile.value
        if not isinstance(val, (str, Mapping)):
            raise TypeError(
                'Expecting either a path to a file or a dictionary provided'
                ' as the store of the parameterisations. Got "%s".' % type(val)
            )
