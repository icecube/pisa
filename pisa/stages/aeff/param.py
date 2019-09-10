"""
This is an effective area stage designed for quick studies of how effective
areas affect experimental observables and sensitivities. In addition, it is
supposed to be easily reproducible as it may rely on (phenomenological)
functions or interpolated discrete data points, dependent on energy
(and optionally cosine zenith), and which can thus be used as reference or
benchmark scenarios.
"""


from __future__ import absolute_import, division

from collections.abc import Mapping
from collections import OrderedDict

import numpy as np
from scipy.interpolate import interp1d

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.stages.aeff.hist import (compute_transforms, populate_transforms,
                                   validate_binning)
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup
from pisa.utils.fileio import from_file
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging


__all__ = ['load_aeff_param', 'param']

__author__ = 'T.C. Arlen, T. Ehrhardt, S. Wren'

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


def load_aeff_param(source):
    """Load aeff parameterisation (energy- or coszen-dependent) from file
    or dictionary.

    Parameters
    ----------
    source : string or mapping
        Source of the parameterization. If string, treat as file path or
        resource location and load from the file; this must yield a mapping. If
        `source` is a mapping, it is used directly. See notes below on format.

    Returns
    -------
    aeff_params : OrderedDict
        Keys are stringified flavintgroups and values are the callables that
        produce aeff when called with energy or coszen values.

    Notes
    -----
    The mapping passed via `source` or loaded therefrom msut have the format:
        {
            <flavintgroup_string>: val,
            <flavintgroup_string>: val,
            ...
        }

    `flavintgroup_string`s must be parsable by
    pisa.utils.flavInt.NuFlavIntGroup. Note that the `transform_groups` defined
    in a pipeline config file using this must match the groupings defined
    above.

    `val`s can be one of the following:
        - Callable with one argument
        - String such that `eval(val)` yields a callable with one argument
        - Mapping with the format:
            {
                <"energy" or "coszen">: [sequence of values],
                "aeff": [sequence of values]A
            }
          the two sequences are used to form a linear interpolant callable that
          maps energy or coszen values to aeff values..

    """
    if not (source is None or isinstance(source, (str, Mapping))):
        raise TypeError('`source` must be string, mapping, or None')

    if isinstance(source, str):
        orig_dict = from_file(source)
    elif isinstance(source, Mapping):
        orig_dict = source
    else:
        raise TypeError('Cannot load aeff parameterizations from a %s'
                        % type(source))

    # Build dict of parameterizations (each a callable) per flavintgroup

    aeff_params = OrderedDict()
    for flavint_key, param_spec in orig_dict.items():
        flavintgroup = NuFlavIntGroup(flavint_key)

        if isinstance(param_spec, str):
            param_func = eval(param_spec)

        elif callable(param_spec):
            param_func = param_spec

        elif isinstance(param_spec, Mapping):
            is_energy = 'energy' in param_spec
            is_coszen = 'coszen' in param_spec

            valid = True
            if 'aeff' not in param_spec:
                valid = False
            elif not (is_energy or is_coszen):
                valid = False
            if not valid:
                raise ValueError(
                    'Expected keys of "aeff" and either "energy" or'
                    ' "coszen" to construct a spline. Got %s instead.'
                    ' Aeff param spec source: %s, flavintgroup %s'
                    % (param_spec.keys(), source, flavintgroup)
                )

            var = 'energy' if is_energy else 'coszen'
            x_vals = param_spec[var]
            aeff_vals = param_spec['aeff']

            # TODO: Could potentially add interp1d options to config
            param_func = interp1d(x_vals, aeff_vals, kind='linear',
                                  bounds_error=False, fill_value=0)

        else:
            raise TypeError(
                'Expected parameteriation spec to be either a string that'
                ' can be interpreted by eval or as a mapping of values'
                ' from which to construct a spline. Got "%s".'
                % type(param_spec)
            )

        aeff_params[flavintgroup] = param_func

    return aeff_params


class param(Stage):
    """Effective area service based on parameterisation functions stored in a
    .json file.
    Transforms an input map of a flux of a given flavour into maps of
    event rates for the two types of weak current (charged or neutral),
    according to energy and cosine zenith dependent effective areas specified
    by parameterisation functions.

    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameters:

        aeff_energy_paramfile
        aeff_coszen_paramfile
        livetime
        aeff_scale
        nutau_cc_norm

    particles : string
        Must be one of 'neutrinos' or 'muons' (though only neutrinos are
        supported at this time).

    input_names : None, string or sequence of strings
        If None, defaults are derived from `particles`.

    transform_groups : string
        Specifies which particles/interaction types to use for computing the
        transforms. (See Notes.)

    sum_grouped_flavints : bool
        Whether to sum the event-rate maps for the flavint groupings
        specified by `transform_groups`. If this is done, the output map names
        will be the group names (as well as the names of any flavor/interaction
        types not grouped together). Otherwise, the output map names will be
        the same as the input map names. Combining grouped flavints' is
        computationally faster and results in fewer maps, but it may be
        desirable to not do so for, e.g., debugging.

    input_binning : MultiDimBinning or convertible thereto
        Input binning is in true variables, with names prefixed by "true_".
        Each must match a corresponding dimension in `output_binning`.

    output_binning : MultiDimBinning or convertible thereto
        Output binning is in reconstructed variables, with names (traditionally
        in PISA but not necessarily) prefixed by "reco_". Each must match a
        corresponding dimension in `input_binning`.

    debug_mode : None, bool, or string
        Whether to store extra debug info for this service.

    transforms_cache_depth : int >= 0

    outputs_cache_depth : int >= 0

    memcache_deepcopy : bool

    """
    def __init__(self, params, particles, transform_groups,
                 sum_grouped_flavints, input_binning, output_binning,
                 memcache_deepcopy, transforms_cache_depth,
                 outputs_cache_depth, input_names=None, error_method=None,
                 debug_mode=None):
        assert particles in ['neutrinos', 'muons']
        self.particles = particles
        """Whether stage is instantiated to process neutrinos or muons"""

        self.transform_groups = flavintGroupsFromString(transform_groups)
        """Particle/interaction types to group for computing transforms"""

        assert isinstance(sum_grouped_flavints, bool)
        self.sum_grouped_flavints = sum_grouped_flavints

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = [
            'aeff_energy_paramfile', 'aeff_coszen_paramfile',
            'livetime', 'aeff_scale'
        ]
        if particles == 'neutrinos':
            expected_params.append('nutau_cc_norm')

        if isinstance(input_names, str):
            input_names = input_names.replace(' ', '').split(',')
        elif input_names is None:
            if particles == 'neutrinos':
                input_names = ('nue', 'nuebar', 'numu', 'numubar', 'nutau',
                               'nutaubar')

        if self.particles == 'neutrinos':
            # TODO: if sum_grouped_flavints, then the output names should be
            # e.g. 'nue_cc_nuebar_cc' and 'nue_nc_nuebar_nc' if nue and nuebar
            # are grouped... (?)
            if self.sum_grouped_flavints:
                output_names = [str(g) for g in self.transform_groups]
            else:
                input_flavints = NuFlavIntGroup(input_names)
                output_names = [str(fi) for fi in input_flavints]
        elif self.particles == 'muons':
            raise NotImplementedError
        else:
            raise ValueError('Particle type `%s` is not valid'
                             % self.particles)

        logging.trace('transform_groups = %s', self.transform_groups)
        logging.trace('output_names = %s', ' :: '.join(output_names))

        super().__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('transform_groups')

        self.ecen = self.input_binning.true_energy.weighted_centers.magnitude
        """input energy-binning weighted centers"""

        self.has_cz = False
        """Whether the stage has true_coszen input binning"""

        self.czcen = None
        """input coszen-binning weighted centers (or None if no coszen dim)"""

        if 'true_coszen' in self.input_binning.names:
            self.has_cz = True
            self.czcen = self.input_binning.true_coszen.weighted_centers.m_as('dimensionless')

        self._param_hashes = dict(energy=None, coszen=None)
        self.aeff_params = dict(energy=dict())
        if self.has_cz:
            self.aeff_params['coszen'] = None

    def _compute_nominal_transforms(self):
        """Compute parameterised effective area transforms"""
        energy_param_source = self.params.aeff_energy_paramfile.value
        coszen_param_source = self.params.aeff_coszen_paramfile.value

        energy_param_hash = hash_obj(energy_param_source)
        coszen_param_hash = hash_obj(coszen_param_source)

        load_energy = False
        load_coszen = False
        if (self._param_hashes['energy'] is None
                or energy_param_hash != self._param_hashes['energy']):
            load_energy = True

        if (self.has_cz
                and (self._param_hashes['coszen'] is None
                     or energy_param_hash != self._param_hashes)):
            load_coszen = True

        if energy_param_source is None:
            raise ValueError(
                'non-None energy parameterization params.aeff_energy_paramfile'
                ' must be provided'
            )
        if not self.has_cz and coszen_param_source is not None:
            raise ValueError(
                'true_coszen dimension was not found in the binning but a'
                ' coszen parameterisation file has been provided by'
                ' `params.aeff_coszen_paramfile`.'
            )

        if not (load_energy or load_coszen):
            return

        dims = ['energy', 'coszen']
        loads = [load_energy, load_coszen]
        sources = [energy_param_source, coszen_param_source]
        hashes = [energy_param_hash, coszen_param_hash]

        for dim, load, source, hash_ in zip(dims, loads, sources, hashes):
            if not load:
                continue
            self._param_hashes[dim] = None
            self.aeff_params[dim] = None
            params = load_aeff_param(source)

            # Transform groups are implicitly defined by the contents of the
            # `pid_energy_paramfile`'s keys
            implicit_transform_groups = params.keys()

            # Make sure these match transform groups specified for the stage
            if set(implicit_transform_groups) != set(self.transform_groups):
                raise ValueError(
                    'Transform groups (%s) defined implicitly by'
                    ' %s aeff parameterizations "%s"  do not match those'
                    ' defined as the stage\'s `transform_groups` (%s).'
                    % (implicit_transform_groups, dim, source,
                       self.transform_groups)
                )

            self.aeff_params[dim] = params
            self._param_hashes[dim] = hash_

        nominal_transforms = []
        for xform_flavints in self.transform_groups:
            logging.debug('Working on %s effective areas xform',
                          xform_flavints)

            energy_param_func = self.aeff_params['energy'][xform_flavints]
            coszen_param_func = None
            if self.aeff_params['coszen'] is not None:
                coszen_param_func = self.aeff_params['coszen'][xform_flavints]

            # Now calculate the 1D aeff along energy
            aeff_vs_e = energy_param_func(self.ecen)

            # NOTE/TODO: Below is taken from the PISA 2 implementation of this.
            # Almost certainly comes from the fact that the highest knot there
            # was 79.5 GeV with the upper energy bin edge being 80 GeV. There's
            # probably something better that could be done here...

            # Correct for final energy bin, since interpolation does not
            # extend to JUST right outside the final bin
            if aeff_vs_e[-1] == 0:
                aeff_vs_e[-1] = aeff_vs_e[-2]

            if self.has_cz:
                aeff_vs_e = self.input_binning.broadcast(
                    aeff_vs_e, from_dim='true_energy', to_dims='true_coszen'
                )

                if coszen_param_func is not None:
                    aeff_vs_cz = coszen_param_func(self.czcen)
                    # Normalize
                    aeff_vs_cz *= len(aeff_vs_cz) / np.sum(aeff_vs_cz)
                else:
                    aeff_vs_cz = np.ones(shape=len(self.czcen))

                cz_broadcasted = self.input_binning.broadcast(
                    aeff_vs_cz, from_dim='true_coszen', to_dims='true_energy'
                )
                aeff_transform = aeff_vs_e * cz_broadcasted
            else:
                aeff_transform = aeff_vs_e

            nominal_transforms.extend(
                populate_transforms(
                    service=self,
                    xform_flavints=xform_flavints,
                    xform_array=aeff_transform
                )
            )

        return TransformSet(transforms=nominal_transforms)

    def _compute_transforms(self):
        # Have to do this in case aeff_energy_paramfile or
        # aeff_coszen_paramfile changed
        self._compute_nominal_transforms()

        # Modify transforms according to other systematics by calling a
        # generic function from aeff.hist
        return compute_transforms(self)

    # Attach generic method from aeff.hist
    validate_binning = validate_binning
