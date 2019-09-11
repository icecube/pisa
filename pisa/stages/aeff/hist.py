"""
Effective areas histogramming service.

Histogram Monte Carlo events directly to derive the effective area transforms.
"""


from __future__ import division
import os

import numpy as np

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup
from pisa.utils.fileio import mkdir, to_file
from pisa.utils.log import logging
from pisa.utils.resources import find_resource


__all__ = ['validate_binning', 'compute_transforms', 'hist']

__author__ = 'J.L. Lanfranchi, P. Eller'

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


def validate_binning(service, must_have_dims=('true_energy',),
                     can_have_dims=('true_energy', 'true_coszen',
                                    'true_azimuth')):
    """Validate binning for effective areas.

    Input binning must contain `must_have_dims` and can have `can_have_dims`
    but no other dims are allowed. Output binning dimension names must match
    those of input binning.

    """
    for dim in must_have_dims:
        if dim not in service.input_binning:
            raise ValueError(
                'Input binning must contain "%s" dimension; got %s instead.'
                % (dim, service.input_binning.names)
            )

    excess_dims = set(service.input_binning.names).difference(
        set(can_have_dims)
    )
    if len(excess_dims) > 0:
        raise ValueError('Input binning has extra dimension(s): %s'
                         % sorted(excess_dims))

    if set(service.input_binning.names) != set(service.output_binning.names):
        raise ValueError(
            'Input binning dim names (%s) do not match output binning dim'
            ' names (%s).'
            % (service.input_binning.names, service.output_binning.names)
        )


def compute_transforms(service):
    """Compute effective area transforms, taking aeff systematics into account.

    Systematics are: `aeff_scale`, `livetime`, and `nutau_cc_norm`

    """
    aeff_scale = service.params.aeff_scale.m_as('dimensionless')
    livetime_s = service.params.livetime.m_as('sec')
    base_scale = aeff_scale * livetime_s

    logging.trace('livetime = %s --> %s sec',
                  service.params.livetime.value, livetime_s)

    if service.particles == 'neutrinos':
        if not hasattr(service, 'nutau_cc_norm_must_be_one'):
            service.nutau_cc_norm_must_be_one = False
            """If any flav/ints besides nutau_cc and nutaubar_cc are grouped
            with one or both of those for transforms, then a
            `nutau_cc_norm` != 1 cannot be applied."""

            nutaucc_and_nutaubarcc = set(NuFlavIntGroup('nutau_cc+nutaubar_cc'))
            for group in service.transform_groups:
                # If nutau_cc, nutaubar_cc, or both are the group and other flavors
                # are present, nutau_cc_norm must be one!
                group_set = set(group)
                if group_set.intersection(nutaucc_and_nutaubarcc) and \
                        group_set.difference(nutaucc_and_nutaubarcc):
                    service.nutau_cc_norm_must_be_one = True

        nutau_cc_norm = service.params.nutau_cc_norm.m_as('dimensionless')
        if nutau_cc_norm != 1 and service.nutau_cc_norm_must_be_one:
            raise ValueError(
                '`nutau_cc_norm` = %e but can only be != 1 if nutau CC and'
                ' nutaubar CC are separated from other flav/ints.'
                ' Transform groups are: %s'
                % (nutau_cc_norm, service.transform_groups)
            )

    if hasattr(service, 'sum_grouped_flavints'):
        sum_grouped_flavints = service.sum_grouped_flavints
    else:
        sum_grouped_flavints = False

    new_transforms = []
    for transform in service.nominal_transforms:
        this_scale = base_scale
        if service.particles == 'neutrinos':
            out_nfig = NuFlavIntGroup(transform.output_name)
            if 'nutau_cc' in out_nfig or 'nutaubar_cc' in out_nfig:
                this_scale *= nutau_cc_norm

        if this_scale != 1:
            aeff_transform = transform.xform_array * this_scale
        else:
            aeff_transform = transform.xform_array

        new_xform = BinnedTensorTransform(
            input_names=transform.input_names,
            output_name=transform.output_name,
            input_binning=transform.input_binning,
            output_binning=transform.output_binning,
            xform_array=aeff_transform,
            sum_inputs=sum_grouped_flavints
        )
        new_transforms.append(new_xform)

    return TransformSet(new_transforms)


def populate_transforms(service, xform_flavints, xform_array):
    """General function for populating a BinnedTensorTransform with a single
    aeff transform array, taking into account e.g. sum_grouped_flavints etc.

    Any rebinning is assumed to be performed outside of the transform, so the
    transform's `output_binning` is the same as its `input_binning`. This does
    _not_ mean that the stage's output binning needs to match its input
    binning, though, since a rebinning can occur after the transform is
    applied but before the maps are emitted from the stage.

    Note that, as certain assumptions (like the above) are made about input and
    outputs names and binning, this function should _only_ be applied to aeff
    services_ (unless very carefully considered).

    Parameters
    ----------
    service : Stage
        The aeff serivce

    xform_array : numpy.ndarray
        Raw transform array

    Returns
    -------
    transforms : list of BinnedTensorTransform

    """
    transforms = []

    # If combining grouped flavints:
    # Create a single transform for each group and assign all inputs
    # that contribute to the group as the single transform's inputs.
    # The actual sum of the input event rate maps will be performed by
    # the BinnedTensorTransform object upon invocation of the `apply`
    # method.
    if service.sum_grouped_flavints:
        xform_input_names = []
        for input_name in service.input_names:
            if set(NuFlavIntGroup(input_name)).isdisjoint(xform_flavints):
                continue
            xform_input_names.append(input_name)

        for output_name in service.output_names:
            if output_name not in xform_flavints:
                continue

            logging.trace('  inputs: %s, output: %s, xform: %s',
                          xform_input_names, output_name, xform_flavints)

            xform = BinnedTensorTransform(
                input_names=xform_input_names,
                output_name=output_name,
                input_binning=service.input_binning,
                output_binning=service.input_binning,
                xform_array=xform_array,
                sum_inputs=service.sum_grouped_flavints
            )
            transforms.append(xform)

    # If *not* combining grouped flavints:
    # Copy the transform for each input flavor, regardless if the
    # transform is computed from a combination of flavors.
    else:
        for input_name in service.input_names:
            # Since aeff "splits" neutrino flavors into
            # flavor+interaction types, need to check if the output
            # flavints are encapsulated by the input flavor(s).
            if set(NuFlavIntGroup(input_name)).isdisjoint(xform_flavints):
                continue

            for output_name in service.output_names:
                if (output_name not in NuFlavIntGroup(input_name)
                        or output_name not in xform_flavints):
                    continue

                logging.trace('  input: %s, output: %s, xform: %s',
                              input_name, output_name, xform_flavints)

                xform = BinnedTensorTransform(
                    input_names=input_name,
                    output_name=output_name,
                    input_binning=service.input_binning,
                    output_binning=service.input_binning,
                    xform_array=xform_array,
                    sum_inputs=service.sum_grouped_flavints
                )
                transforms.append(xform)

    return transforms


# TODO: the below logic does not generalize to muons, but probably should
# (rather than requiring an almost-identical version just for muons). For
# example, an input arg can dictate neutrino or muon, which then sets the
# input_names and output_names.

class hist(Stage):
    """Effective area.

    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameters:

        aeff_events
        livetime
        aeff_scale
        nutau_cc_norm
        transform_events_keep_criteria

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

    Notes
    -----
    See Conventions section in the documentation for more informaton on
    particle naming scheme in PISA.

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
            'aeff_events', 'livetime', 'aeff_scale',
            'transform_events_keep_criteria'
        ]
        if particles == 'neutrinos':
            expected_params.append('nutau_cc_norm')

        if isinstance(input_names, str):
            input_names = input_names.replace(' ', '').split(',')
        elif input_names is None:
            if particles == 'neutrinos':
                input_names = ('nue', 'nuebar', 'numu', 'numubar', 'nutau',
                               'nutaubar')

        # Define the names of objects expected in inputs and produced as
        # outputs
        if self.particles == 'neutrinos':
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

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
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

        # Can do these now that binning has been set up in call to Stage's init
        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('transform_groups')

    def _compute_nominal_transforms(self):
        self.load_events(self.params.aeff_events)
        self.cut_events(self.params.transform_events_keep_criteria)

        # Units must be the following for correctly converting a sum-of-
        # OneWeights-in-bin to an average effective area across the bin.
        comp_units = dict(true_energy='GeV', true_coszen=None,
                          true_azimuth='rad')

        # Select only the units in the input/output binning for conversion
        # (can't pass more than what's actually there)
        in_units = {dim: unit for dim, unit in comp_units.items()
                    if dim in self.input_binning}

        # TODO: use out_units for some kind of conversion?
        #out_units = {dim: unit for dim, unit in comp_units.items()
        #             if dim in self.output_binning}

        # These will be in the computational units
        input_binning = self.input_binning.to(**in_units)

        # Account for "missing" dimension(s) (dimensions OneWeight expects for
        # computation of bin volume), and accommodate with a factor equal to
        # the full range. See IceCube wiki/documentation for OneWeight for
        # more info.
        missing_dims_vol = 1
        if 'true_azimuth' not in input_binning:
            missing_dims_vol *= 2*np.pi
        if 'true_coszen' not in input_binning:
            missing_dims_vol *= 2

        if bool(self.debug_mode):
            outdir = os.path.join(find_resource('debug'),
                                  self.stage_name,
                                  self.service_name)
            mkdir(outdir)
            #hex_hash = hash2hex(kde_hash)

        bin_volumes = input_binning.bin_volumes(attach_units=False)
        norm_volumes = bin_volumes * missing_dims_vol

        nominal_transforms = []
        for xform_flavints in self.transform_groups:
            logging.debug('Working on %s effective areas xform',
                          xform_flavints)

            aeff_transform = self.events.histogram(
                kinds=xform_flavints,
                binning=input_binning,
                weights_col='weighted_aeff',
                errors=(self.error_method not in [None, False])
            )
            aeff_transform = aeff_transform.hist

            # Divide histogram by
            #   (energy bin width x coszen bin width x azimuth bin width)
            # volumes to convert from sums-of-OneWeights-in-bins to
            # effective areas. Note that volume correction factor for
            # missing dimensions is applied here.
            aeff_transform /= norm_volumes

            if self.debug_mode:
                outfile = os.path.join(
                    outdir, 'aeff_' + str(xform_flavints) + '.pkl'
                )
                to_file(aeff_transform, outfile)

            nominal_transforms.extend(
                populate_transforms(
                    service=self,
                    xform_flavints=xform_flavints,
                    xform_array=aeff_transform
                )
            )

        return TransformSet(transforms=nominal_transforms)

    # Generic methods

    validate_binning = validate_binning
    _compute_transforms = compute_transforms
