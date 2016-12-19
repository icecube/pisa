# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016


import os

import numpy as np

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.core.events import Events
from pisa.utils.flavInt import (ALL_NUFLAVINTS, flavintGroupsFromString,
                                IntType, NuFlavIntGroup)
from pisa.utils.fileio import mkdir, to_file
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import profile
from pisa.utils.resources import find_resource


__all__ = ['hist']


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

        self.sum_grouped_flavints = sum_grouped_flavints

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = [
            'aeff_events', 'livetime', 'aeff_scale',
            'transform_events_keep_criteria'
        ]
        if particles == 'neutrinos':
            expected_params.append('nutau_cc_norm')


        if isinstance(input_names, basestring):
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
        super(self.__class__, self).__init__(
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

        # Only works if energy is in input_binning
        if 'true_energy' not in self.input_binning:
            raise ValueError('Input binning must contain "true_energy"'
                             ' dimension, but does not.')

        # coszen and azimuth are both optional, but no further dimensions are
        excess_dims = set(self.input_binning.names).difference(
            comp_units.keys()
        )
        if len(excess_dims) > 0:
            raise ValueError('Input binning has extra dimension(s): %s'
                             %sorted(excess_dims))

        # Select only the units in the input/output binning for conversion
        # (can't pass more than what's actually there)
        in_units = {dim: unit for dim, unit in comp_units.items()
                    if dim in self.input_binning}
        out_units = {dim: unit for dim, unit in comp_units.items()
                     if dim in self.output_binning}

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

        nominal_transforms = []
        for xform_flavints in self.transform_groups:
            logging.debug("Working on %s effective areas xform" %xform_flavints)

            aeff_transform = self.remaining_events.histogram(
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
            bin_volumes = input_binning.bin_volumes(attach_units=False)
            aeff_transform /= (bin_volumes * missing_dims_vol)

            if self.debug_mode:
                outfile = os.path.join(outdir,
                                       'aeff_' + str(xform_flavints) + '.dill')
                to_file(aeff_transform, outfile)

            # If combining grouped flavints:
            # Create a single transform for each group and assign all flavors
            # that contribute to the group as the transform's inputs. Combining
            # the event rate maps will be performed by the
            # BinnedTensorTransform object upon invocation of the `apply`
            # method.
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
                        xform_array=aeff_transform,
                        sum_inputs=self.sum_grouped_flavints
                    )
                    nominal_transforms.append(xform)

            # If *not* combining grouped flavints:
            # Copy the transform for each input flavor, regardless if the
            # transform is computed from a combination of flavors.
            else:
                for input_name in self.input_names:
                    input_flavs = NuFlavIntGroup(input_name)
                    # Since aeff "splits" neutrino flavors into
                    # flavor+interaction types, need to check if the output
                    # flavints are encapsulated by the input flavor(s).
                    if len(set(xform_flavints).intersection(input_flavs)) == 0:
                        continue
                    for output_name in self.output_names:
                        if output_name not in xform_flavints:
                            continue
                        xform = BinnedTensorTransform(
                            input_names=input_name,
                            output_name=output_name,
                            input_binning=self.input_binning,
                            output_binning=self.output_binning,
                            xform_array=aeff_transform,
                            sum_inputs=self.sum_grouped_flavints
                        )
                        nominal_transforms.append(xform)

        return TransformSet(transforms=nominal_transforms)

    def _compute_transforms(self):
        """Compute new effective area transforms"""
        # Read parameters in in the units used for computation
        aeff_scale = self.params.aeff_scale.m_as('dimensionless')
        livetime_s = self.params.livetime.m_as('sec')
        logging.trace('livetime = %s --> %s sec'
                      %(self.params.livetime.value, livetime_s))

        if self.particles == 'neutrinos':
            nutau_cc_norm = self.params.nutau_cc_norm.m_as('dimensionless')
            if nutau_cc_norm != 1:
                assert NuFlavIntGroup('nutau_cc') in self.transform_groups
                assert NuFlavIntGroup('nutaubar_cc') in self.transform_groups

        new_transforms = []
        for xform_flavints in self.transform_groups:
            flav_names = [str(flav) for flav in xform_flavints.flavs()]
            aeff_transform = None
            for transform in self.nominal_transforms:
                if (transform.input_names[0] in flav_names
                        and transform.output_name in xform_flavints):
                    if aeff_transform is None:
                        scale = aeff_scale * livetime_s
                        if (self.particles == 'neutrinos' and
                                ('nutau_cc' in transform.output_name
                                 or 'nutaubar_cc' in transform.output_name)):
                            scale *= nutau_cc_norm
                        aeff_transform = transform.xform_array * scale
                    new_xform = BinnedTensorTransform(
                        input_names=transform.input_names,
                        output_name=transform.output_name,
                        input_binning=transform.input_binning,
                        output_binning=transform.output_binning,
                        xform_array=aeff_transform,
                        sum_inputs=self.sum_grouped_flavints
                    )
                    new_transforms.append(new_xform)

        return TransformSet(new_transforms)
