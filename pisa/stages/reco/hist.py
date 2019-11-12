"""
Create the transforms that map from true energy and coszen
to the reconstructed parameters. Provides reco event rate maps using these
transforms.
"""


from __future__ import division

import numpy as np

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup
from pisa.utils.log import logging


__all__ = ['hist']

__author__ = 'T.C. Arlen, S. Wren'

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


# TODO: the below logic does not generalize to muons, but probably should
# (rather than requiring an almost-identical version just for muons). For
# example, an input arg can dictate neutrino or muon, which then sets the
# input_names and output_names.

class hist(Stage):
    """
    From the simulation file, creates 4D histograms of
    [true_energy][true_coszen][reco_energy][reco_coszen] which act as
    2D pdfs for the probability that an event with (true_energy,
    true_coszen) will be reconstructed as (reco_energy,reco_coszen).

    From these histograms and the true event rate maps, calculates
    the reconstructed even rate templates.

    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameters:

        reco_events : string or Events
            PISA events file to use to derive transforms, or a string
            specifying the resource location of the same.

        reco_weights_name : None or string
            Column in the events file to use for Monte Carlo weighting of the
            events

        res_scale_ref : string
            One of "mean", "median", or "zero". This is the reference point
            about which resolutions are scaled. "zero" scales about the
            zero-error point (i.e., the bin midpoint), "mean" scales about the
            mean of the events in the bin, and "median" scales about the median
            of the events in the bin.

        e_res_scale : float
            A scaling factor for energy resolutions.

        cz_res_scale : float
            A scaling factor for coszen resolutions.

        e_reco_bias : float

        cz_reco_bias : float

        transform_events_keep_criteria : None, string, or sequence of strings

    particles : string
        Must be one of 'neutrinos' or 'muons' (though only neutrinos are
        supported at this time).

    input_names : string or list of strings
        Names of inputs expected. These should follow the standard PISA
        naming conventions for flavor/interaction types OR groupings
        thereof. Note that this service's outputs are named the same as its
        inputs. See Conventions section in the documentation for more info.

    transform_groups : string
        Specifies which particles/interaction types to combine together in
        computing the transforms. See Notes section for more details on how
        to specify this string

    sum_grouped_flavints : bool

    input_binning : MultiDimBinning or convertible thereto
        Input binning is in true variables, with names prefixed by "true_".
        Each must match a corresponding dimension in `output_binning`.

    output_binning : MultiDimBinning or convertible thereto
        Output binning is in reconstructed variables, with names (traditionally
        in PISA but not necessarily) prefixed by "reco_". Each must match a
        corresponding dimension in `input_binning`.

    transforms_cache_depth : int >= 0

    outputs_cache_depth : int >= 0

    memcache_deepcopy : bool

    debug_mode : None, bool, or string
        Whether to store extra debug info for this service.

    Notes
    -----
    The `transform_groups` string is interpreted (and therefore defined) by
    pisa.utils.flavInt.flavint_groups_string. E.g. commonly one might use:

    'nue_cc+nuebar_cc, numu_cc+numubar_cc, nutau_cc+nutaubar_cc, nuall_nc+nuallbar_nc'

    Any particle type not explicitly mentioned is taken as a singleton group.
    Plus signs add types to a group, while groups are separated by commas.
    Whitespace is ignored, so add whitespace for readability.

    """
    def __init__(self, params, particles, input_names, transform_groups,
                 sum_grouped_flavints, input_binning, output_binning,
                 error_method=None, transforms_cache_depth=20,
                 outputs_cache_depth=20, memcache_deepcopy=True,
                 debug_mode=None):
        assert particles in ['neutrinos', 'muons']
        self.particles = particles
        self.transform_groups = flavintGroupsFromString(transform_groups)
        self.sum_grouped_flavints = sum_grouped_flavints

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'reco_events', 'reco_weights_name',
            'transform_events_keep_criteria',
            'res_scale_ref', 'e_res_scale', 'cz_res_scale',
            'e_reco_bias', 'cz_reco_bias'
        )

        if isinstance(input_names, str):
            input_names = (''.join(input_names.split(' '))).split(',')

        # Define the names of objects expected in inputs and produced as
        # outputs
        if self.particles == 'neutrinos':
            if self.sum_grouped_flavints:
                output_names = [str(g) for g in self.transform_groups]
            else:
                output_names = input_names

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
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

        # Can do these now that binning has been set up in call to Stage's init
        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('transform_groups')
        self.include_attrs_for_hashes('sum_grouped_flavints')

    def validate_binning(self):
        input_basenames = set(self.input_binning.basenames)
        output_basenames = set(self.output_binning.basenames)
        for base_d in input_basenames:
            assert base_d in output_basenames, \
                   "'%s' in input but not in output binning!"%base_d

    def _compute_transforms(self):
        """Generate reconstruction "smearing kernels" by histogramming true and
        reconstructed variables from a Monte Carlo events file.

        The resulting transform is a 2N-dimensional histogram, where N is the
        dimensionality of the input binning. The transform maps the truth bin
        counts to the reconstructed bin counts.

        I.e., for the case of 1D input binning, the ith element of the
        reconstruction kernel will be a map showing the distribution of events
        over all the reco space from truth bin i. This will be normalised to
        the total number of events in truth bin i.

        Notes
        -----
        In the current implementation these histograms are made
        **UN**weighted. This is probably quite wrong...

        """
        e_res_scale = self.params.e_res_scale.value.m_as('dimensionless')
        cz_res_scale = self.params.cz_res_scale.value.m_as('dimensionless')
        e_reco_bias = self.params.e_reco_bias.value.m_as('GeV')
        cz_reco_bias = self.params.cz_reco_bias.value.m_as('dimensionless')
        res_scale_ref = self.params.res_scale_ref.value.strip().lower()
        assert res_scale_ref in ['zero'] # TODO: , 'mean', 'median']

        self.load_events(self.params.reco_events)
        self.cut_events(self.params.transform_events_keep_criteria)

        # Computational units must be the following for compatibility with
        # events file
        comp_units = dict(
            true_energy='GeV', true_coszen=None, true_azimuth='rad',
            reco_energy='GeV', reco_coszen=None, reco_azimuth='rad', pid=None
        )

        # Select only the units in the input/output binning for conversion
        # (can't pass more than what's actually there)
        in_units = {dim: unit for dim, unit in comp_units.items()
                    if dim in self.input_binning}
        out_units = {dim: unit for dim, unit in comp_units.items()
                     if dim in self.output_binning}

        # These binnings will be in the computational units defined above
        input_binning = self.input_binning.to(**in_units)
        output_binning = self.output_binning.to(**out_units)

        xforms = []
        for xform_flavints in self.transform_groups:
            logging.debug("Working on %s reco kernels" %xform_flavints)

            repr_flavint = xform_flavints[0]

            true_energy = self.events[repr_flavint]['true_energy']
            true_coszen = self.events[repr_flavint]['true_coszen']
            reco_energy = self.events[repr_flavint]['reco_energy']
            reco_coszen = self.events[repr_flavint]['reco_coszen']
            e_reco_err = reco_energy - true_energy
            cz_reco_err = reco_coszen - true_coszen

            if self.params.res_scale_ref.value.strip().lower() == 'zero':
                self.events[repr_flavint]['reco_energy'] = (
                    true_energy + e_reco_err * e_res_scale + e_reco_bias
                )
                self.events[repr_flavint]['reco_coszen'] = (
                    true_coszen + cz_reco_err * cz_res_scale + cz_reco_bias
                )

            # True (input) + reco {+ PID} (output)-dimensional histogram
            # is the basis for the transformation
            reco_kernel = self.events.histogram(
                kinds=xform_flavints,
                binning=input_binning * output_binning,
                weights_col=self.params.reco_weights_name.value,
                errors=(self.error_method not in [None, False])
            )
            # Extract just the numpy array to work with
            reco_kernel = reco_kernel.hist

            # This takes into account the correct kernel normalization:
            # What this means is that we have to normalise the reco map
            # to the number of events in the truth bin.
            #
            # I.e., we have N events from the truth bin which then become
            # spread out over the whole map due to reconstruction.
            # The normalisation is dividing this map by N.
            #
            # Previously this was hard-coded for 2 dimensions, but I have tried
            # to generalise it to arbitrary dimensionality.

            # Truth-only (N-dimensional) histogram will be used for
            # normalization (so transform is in terms of fraction-of-events in
            # input--i.e. truth--bin). Sum over the input dimensions.
            true_event_counts = self.events.histogram(
                kinds=xform_flavints,
                binning=input_binning,
                weights_col=self.params.reco_weights_name.value,
                errors=(self.error_method not in [None, False])
            )
            # Extract just the numpy array to work with
            true_event_counts = true_event_counts.hist

            # If there weren't any events in the input (true_*) bin, make this
            # bin have no effect -- i.e., populate all output bins
            # corresponding to the input bin with zeros via `nan_to_num`.
            with np.errstate(divide='ignore', invalid='ignore'):
                true_event_counts[true_event_counts == 0] = np.nan
                norm_factors = 1.0 / true_event_counts
                norm_factors = np.nan_to_num(norm_factors)

            # Numpy broadcasts lower-dimensional things to higher dimensions
            # from last dimension to first; if we simply mult the reco_kernel
            # by norm_factors, this will apply the normalization to the
            # __output__ dimensions rather than the input dimensions. Add
            # "dummy" dimensions to norm_factors where we want the "extra
            # dimensions": at the end.
            for dim in self.output_binning:
                norm_factors = np.expand_dims(norm_factors, axis=-1)

            # Apply the normalization to the kernels
            reco_kernel *= norm_factors

            assert np.all(reco_kernel >= 0), \
                    'number of elements less than 0 = %d' \
                    % np.sum(reco_kernel < 0)
            sum_over_axes = tuple(range(-len(self.output_binning), 0))
            totals = np.sum(reco_kernel, axis=sum_over_axes)
            assert np.all(totals <= 1+1e-14), 'max = ' + str(np.max(totals)-1)

            # Now populate this transform to each input for which it applies.

            if self.sum_grouped_flavints:
                xform_input_names = []
                for input_name in self.input_names:
                    input_flavs = NuFlavIntGroup(input_name)
                    if len(set(xform_flavints).intersection(input_flavs)) > 0:
                        xform_input_names.append(input_name)

                for output_name in self.output_names:
                    if output_name not in xform_flavints:
                        continue
                    xform = BinnedTensorTransform(
                        input_names=xform_input_names,
                        output_name=output_name,
                        input_binning=self.input_binning,
                        output_binning=self.output_binning,
                        xform_array=reco_kernel,
                        sum_inputs=self.sum_grouped_flavints
                    )
                    xforms.append(xform)
            else:
                # NOTES:
                # * Output name is same as input name
                # * Use `self.input_binning` and `self.output_binning` so maps
                #   are returned in user-defined units (rather than
                #   computational units, which are attached to the non-`self`
                #   versions of these binnings).
                for input_name in self.input_names:
                    if input_name not in xform_flavints:
                        continue
                    xform = BinnedTensorTransform(
                        input_names=input_name,
                        output_name=input_name,
                        input_binning=self.input_binning,
                        output_binning=self.output_binning,
                        xform_array=reco_kernel,
                    )
                    xforms.append(xform)

        return TransformSet(transforms=xforms)
