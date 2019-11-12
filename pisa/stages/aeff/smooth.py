"""
Smoothed-histogram effective areas stage
"""


from __future__ import division

import numpy as np
from uncertainties import unumpy as unp
from scipy import ndimage

from pisa.core.stage import Stage
from pisa.core.transform import TransformSet
from pisa.core.binning import OneDimBinning
from pisa.stages.aeff.hist import (compute_transforms, populate_transforms,
                                   validate_binning)
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.spline_smooth import spline_smooth


__all__ = ['smooth']

__author__ = 'J.L. Lanfranchi, P. Eller, M. Weiss'

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

# TODO: remove the input_names instantiation arg since these are computed
# from the `particles` arg?

class smooth(Stage):
    """Smooth each effective area transform by fitting splines

    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameters:

        aeff_events
        livetime
        aeff_scale
        aeff_e_smooth_factor
        aeff_cz_smooth_factor
        nutau_cc_norm
        transform_events_keep_criteria

    particles : string
        Must be one of 'neutrinos' or 'muons' (though only neutrinos are
        supported at this time).

    input_names : None, string or sequence of strings
        If None, defaults are derived from `particles`.

    error_method

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

    transforms_cache_depth : int >= 0

    outputs_cache_depth : int >= 0

    memcache_deepcopy : bool

    Notes
    -----
    See Conventions section in the documentation for more information on
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

        self.sum_grouped_flavints = sum_grouped_flavints

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = [
            'aeff_events', 'livetime', 'aeff_scale',
            'aeff_e_smooth_factor', 'aeff_cz_smooth_factor',
            'transform_events_keep_criteria',
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
            raise ValueError('Particle type `%s` is not valid' % self.particles)

        logging.trace('transform_groups = %s' %self.transform_groups)
        logging.trace('output_names = %s' %' :: '.join(output_names))

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

    def smooth(self, xform, errors, e_binning, cz_binning):
        """Smooth a 2d array

        with energy as the first dimension and CZ as the second
        First performing a gaussian smoothing to get rid of zeros
        then spline smooth

        Parameters
        ----------

        xform : 2d-array
        errors : 2d-array
        e_binning : OneDimBinning
        cz_binning : OneDimBinning

        """

        sumw2 = np.square(errors)

        # First, let's check if we have entire coszen values with zero, this is for example
        # the case for all nutau CC below 3.5 GeV

        non_zero_idx = 0
        while np.sum(xform[non_zero_idx, :]) == 0:
            non_zero_idx += 1

        # cut away these all-zero regions for now
        truncated_xform = xform[non_zero_idx:, :]
        truncated_sumw2 = sumw2[non_zero_idx:, :]
        truncated_e_binning = e_binning[non_zero_idx:]

        # Now lets extend that array at both energy ends
        # by about 10% of bins
        num_extension_bins_lower = int(np.floor(0.1*len(e_binning)))
        num_extension_bins_upper = 0 
        assert e_binning.is_lin or e_binning.is_log, 'Do not know how to extend arbitrary binning'

        # what will new bin edges be?
        bin_edges = truncated_e_binning.bin_edges.m
        if e_binning.is_log:
            bin_edges = np.log10(bin_edges)
        bin_edges = list(bin_edges)
        lower_edges = []
        upper_edges = []
        delta = bin_edges[1] - bin_edges[0]
        for i in range(num_extension_bins_lower):
            lower_edges.append(bin_edges[0] - (i+1)*delta)
        for i in range(num_extension_bins_upper):
            upper_edges.append(bin_edges[-1] + (i+1)*delta)
        new_edges = np.array(lower_edges[::-1] + bin_edges + upper_edges)
        if e_binning.is_log:
            new_edges = np.power(10, new_edges)

        extended_binning = OneDimBinning('true_energy', bin_edges=new_edges, is_lin=e_binning.is_lin, is_log=e_binning.is_log)

        # also extend that arrays
        # We do that by point-reflecting the values
        # so an array like  [0 1 2 3 4 ...] will become [-3 -2 -1 0 1 2 3 4 ...]

        #if non_zero_idx == 0:
        lower_bit = 2*truncated_xform[0, :] - np.flipud(truncated_xform[1:num_extension_bins_lower+1, :])
        #else:
        #    lower_bit =  - np.flipud(truncated_xform[1:num_extension_bins+1,:])
        upper_bit = 2*truncated_xform[-1, :] - np.flipud(truncated_xform[-num_extension_bins_upper-1:-1, :])
        extended_xform = np.concatenate((lower_bit, truncated_xform, upper_bit))

        # also handle the errors (which simply add up in quadrature)
        #if non_zero_idx == 0:
        lower_bit = truncated_sumw2[0, :] + np.flipud(truncated_sumw2[1:num_extension_bins_lower+1, :])
        #else:
        #    lower_bit = np.flipud(truncated_sumw2[1:num_extension_bins+1,:])
        upper_bit = truncated_sumw2[-1, :] + np.flipud(truncated_sumw2[-num_extension_bins_upper-1:-1, :])
        extended_sumw2 = np.concatenate((lower_bit, truncated_sumw2, upper_bit))

        # what's the stat. situation here?
        rel_error = np.square(errors)/xform
        rel_error = np.median(rel_error[xform != 0])
        logging.debug('Relative errors are ~ %.5f', rel_error)
        # how many zero bins?
        zero_fraction = np.count_nonzero(truncated_xform == 0)/float(truncated_xform.size)
        logging.debug('zero fraction is %.3f', zero_fraction)


        # now use gaussian smoothing on those
        # some black magic sigma values
        sigma_e = xform.shape[0] * np.sqrt(zero_fraction) * 0.02
        sigma_cz = xform.shape[1] * np.sqrt(zero_fraction) * 0.05
        sigma1 = (0, sigma_cz)
        sigma2 = (sigma_e, 0)
        smooth_extended_xform = ndimage.filters.gaussian_filter(extended_xform, sigma1, mode='reflect')
        smooth_extended_sumw2 = ndimage.filters.gaussian_filter(extended_sumw2, sigma1, mode='reflect')
        smooth_extended_xform = ndimage.filters.gaussian_filter(smooth_extended_xform, sigma2, mode='nearest')
        smooth_extended_sumw2 = ndimage.filters.gaussian_filter(smooth_extended_sumw2, sigma2, mode='nearest')
        smooth_extended_errors = np.sqrt(smooth_extended_sumw2)

        # for low stats, smooth less....otherwise leave factor alone
        smooth_corr_factor = max(1., (rel_error * 10000))

        # now spline smooth
        new_xform, _ = spline_smooth(array=smooth_extended_xform,
                                     spline_binning=extended_binning,
                                     eval_binning=e_binning,
                                     axis=0,
                                     smooth_factor=self.params.aeff_e_smooth_factor.value/smooth_corr_factor,
                                     k=3,
                                     errors=smooth_extended_errors)

        final_xform, _ = spline_smooth(array=new_xform,
                                       spline_binning=cz_binning,
                                       eval_binning=cz_binning,
                                       axis=1,
                                       smooth_factor=self.params.aeff_cz_smooth_factor.value/smooth_corr_factor,
                                       k=3,
                                       errors=None)

        # the final array has the right shape again, because we evaluated the splines only
        # on the real binning

        # don't forget to zero out the zero bins again
        final_xform[:non_zero_idx, :] *= 0

        # clip unphysical (negative)  values
        return final_xform.clip(0)


    @profile
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
        #out_units = {dim: unit for dim, unit in comp_units.items()
        #             if dim in self.output_binning}

        # These will be in the computational units
        input_binning = self.input_binning.to(**in_units)

        # Account for "missing" dimension(s) (dimensions OneWeight expects for
        # computation of bin volume), and accommodate with a factor equal to
        # the full range. See IceCube wiki/documentation for OneWeight for
        # more info.
        missing_dims_vol = 1
        # TODO: currently, azimuth required to *not* be part of input binning
        if 'true_azimuth' not in input_binning:
            missing_dims_vol *= 2*np.pi
        # TODO: Following is currently never the case, handle?
        if 'true_coszen' not in input_binning:
            missing_dims_vol *= 2

        nominal_transforms = []

        for xform_flavints in self.transform_groups:
            logging.info("Working on %s effective areas xform", xform_flavints)

            raw_hist = self.events.histogram(
                kinds=xform_flavints,
                binning=input_binning,
                weights_col='weighted_aeff',
                errors=True
            )
            raw_transform = unp.nominal_values(raw_hist.hist)
            raw_errors = unp.std_devs(raw_hist.hist)

            # Divide histogram by
            #   (energy bin width x coszen bin width x azimuth bin width)
            # volumes to convert from sums-of-OneWeights-in-bins to
            # effective areas. Note that volume correction factor for
            # missing dimensions is applied here.
            bin_volumes = input_binning.bin_volumes(attach_units=False)
            raw_transform /= (bin_volumes * missing_dims_vol)
            raw_errors /= (bin_volumes * missing_dims_vol)

            e_idx = input_binning.index('true_energy')
            if e_idx == 1:
                # transpose
                raw_transform = raw_transform.T
                raw_errors = raw_errors.T

            # Do the smoothing
            smooth_transform = self.smooth(raw_transform, raw_errors,
                                           input_binning['true_energy'],
                                           input_binning['true_coszen'])

            if e_idx == 1:
                # transpose back
                smooth_transform = smooth_transform.T

            nominal_transforms.extend(
                populate_transforms(
                    service=self,
                    xform_flavints=xform_flavints,
                    xform_array=smooth_transform
                )
            )

        return TransformSet(transforms=nominal_transforms)

    # Generic methods from aeff.hist

    validate_binning = validate_binning
    _compute_transforms = compute_transforms
