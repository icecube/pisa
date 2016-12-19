# authors: J.Lanfranchi/P.Eller/M.Weiss
# date:   March 20, 2016
"""
Smoothed-histogram effective areas stage

"""

import numpy as np
from scipy.interpolate import interp2d, splrep, splev
import uncertainties as unc
from uncertainties import unumpy as unp

from pisa import ureg
from pisa.core.binning import OneDimBinning
from pisa.core.map import Map
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup
from pisa.utils.log import logging
from pisa.utils.plotter import Plotter
from pisa.utils.profiler import profile


__all__ = ['smooth']


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
        interp_kind
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

    debug_mode : bool
        If true, this stage will:
        1. Make plots:
            1. raw Aeff transforms
            2. smoothed transforms
            3. interpolated transforms
            4. fractional difference between raw and smoothed transforms
            5. a pair of slices of a transform (to compare smoothing to
                original transform)
        2. Make some internal stage objects available as attributes:
            1. <stage>.raw_transforms : TransformSet
            2. <stage>.smooth_transforms : TransformSet
            3. <stage>.interp_transforms : TransformSet
            4. <stage>.plot_slice_comparison : function
                    A function to make a comparison plot between a slice of
                    a smoothed and an original transform.

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
            'aeff_e_smooth_factor', 'aeff_cz_smooth_factor', 'interp_kind',
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

    def slice_smooth(self, xform):
        """Returns a smoothed version of `xform`.

        Fits splines to `xform.xform_array`, parallel to energy direction,
        then parallel to coszen, using the smoothing factors found in params.

        Note that values in returned xform.array can be negative, which is
        non-physical. Consider using `np.clip` on output.

        The smoothing operation used here is based on
        pisa.utils.slice_smooth_aeff

        Parameters
        ----------
        xform : BinnedTensorTransform
            The transform to be smoothed. Must have dimensions true_energy and
            true_coszen.

        Returns
        -------
        A transform with a smoothed xform.array

        """
        # Dimensions and order used for computation
        dims = ['true_coszen', 'true_energy'] #, 'true_azimuth']
        # Order is simple for computation
        comp_dim_indices = range(len(dims))
        # Find order in the transform
        users_dim_indices = [xform.input_binning.index(d) for d in dims]

        hist = xform.xform_array
        xform_binning = xform.input_binning

        # Swap hist dim order to be [true_coszen, true_energy, true_azimuth]
        if comp_dim_indices != users_dim_indices:
            hist = np.moveaxis(
                hist,
                source=users_dim_indices,
                destination=comp_dim_indices
            )

        # Get smooth factors from stage parameters
        e_smooth_factor = \
                self.params.aeff_e_smooth_factor.value.m_as('dimensionless')
        cz_smooth_factor = \
                self.params.aeff_cz_smooth_factor.value.m_as('dimensionless')

        # Separate binning dimensions
        czbins = xform_binning.true_coszen
        ebins = xform_binning.true_energy

        # Set spline weights
        # If hist has uncertainties
        if isinstance(hist.flat.next(), (unc.core.AffineScalarFunc,
                                         unc.core.Variable)):
            error = unp.std_devs(hist)
            hist = unp.nominal_values(hist)
        # If no uncertainties
        else:
            error = None

        # Smooth cz-slices of hist
        smoothed_cz_slices = []
        for index in xrange(len(czbins)):
            # Get slice and slice error
            cz_slice = hist[index, :]
            if error is not None:
                cz_slice_err = error[index, :]

            # Remove extra dimensions
            s_cz_slice = np.squeeze(cz_slice)

            # Deal with problematic bin values in error
            if False: #error is not None:
                s_cz_slice_err = np.squeeze(cz_slice_err)

                zero_and_nan_indices = np.squeeze(
                    (s_cz_slice == 0) | (s_cz_slice != s_cz_slice) |
                    (s_cz_slice_err == 0) | (s_cz_slice_err != s_cz_slice_err)
                )
                min_err = np.min(s_cz_slice_err[s_cz_slice_err > 0])
                s_cz_slice_err[zero_and_nan_indices] = min_err
                weights = 1./np.array(s_cz_slice_err)
            else:
                weights = None

            # Fit spline to cz-slices
            cz_slice_spline = splrep(
                ebins.midpoints, s_cz_slice, w=weights,
                k=3, s=e_smooth_factor
            )

            # Sample cz-spline over ebin midpoints
            smoothed_cz_slice = splev(ebins.midpoints, cz_slice_spline)

            # Assert that there are no nan or inf values in smoothed cz-slice
            assert np.all(np.isfinite(smoothed_cz_slice))

            smoothed_cz_slices.append(smoothed_cz_slice)

        # Convert list of cz-slices to array
        smoothed_cz_slices = np.array(smoothed_cz_slices)

        # Iterate through e-slices
        smoothed_e_slices = []
        for e_slice_num in xrange(smoothed_cz_slices.shape[1]):
            e_slice = smoothed_cz_slices[:, e_slice_num]

            # Fit spline to e-slice
            e_slice_spline = splrep(
                czbins.midpoints, e_slice, w=None,
                k=3, s=cz_smooth_factor
            )

            # Evaluate spline at bin midpoints
            smoothed_aeff = splev(czbins.midpoints, e_slice_spline)

            smoothed_e_slices.append(smoothed_aeff)

        # Convert list of e-slices to array with cz as first index
        smoothed_hist = np.array(smoothed_e_slices).T

        # Reorder dimensions to restore user's original ordering
        if comp_dim_indices != users_dim_indices:
            smoothed_hist = np.moveaxis(
                smoothed_hist,
                source=comp_dim_indices,
                destination=users_dim_indices
            )

        # Reorder dims to original
        smooth_xform = BinnedTensorTransform(
            input_names=xform.input_names,
            output_name=xform.output_name,
            input_binning=xform.input_binning,
            output_binning=xform.output_binning,
            xform_array=smoothed_hist
        )

        return smooth_xform

    def interpolate_transform(self, xform, new_binning):
        """Interpolates `xform.xform_array` to `new_binning` using
        scipy.interpolate.interp2d. The degree of the interpolation
        is given by <stage>.params.interp_kind.

        Parameters
        ----------
        xform : BinnedTensorTransform
            The transform containing the array to be interpolated

        new_binning : MultiDimBinning
            The binning to which the transform should be interpolated

        Return
        ------
        BinnedTensorTransform
            Identical to `xform` except that it contains the interpolated
            transform and new_binning

        Notes
        -----
        The returned transform may contain negative (non-physical) values.
        Use np.clip to remove these if necessary.

        """
        if xform.input_binning.names != new_binning.names:
            raise ValueError(
                'At present it is required that interplation be carried out'
                ' in the same dimensions/order of dimensions as those exist in'
                ' the original trnsform. xform.input_binning.names: %s;'
                ' new_binning.names: %s' %(xform.input_binning.names,
                                           new_binning.names)
            )

        hist = xform.xform_array
        xform_binning = xform.input_binning

        interp_kind = self.params.interp_kind.value

        # Interpolation treats x as first axis and y as second (i.e., like i-j
        # axes), so compute accordingly by specifying the second binning
        # dimension first, and vice versa
        interpolant = interp2d(
            x=xform_binning.dims[1].midpoints,
            y=xform_binning.dims[0].midpoints,
            z=hist,
            kind=interp_kind, copy=True, fill_value=None
        )
        interp = interpolant(new_binning.dims[1].midpoints,
                             new_binning.dims[0].midpoints)

        interp_xform = BinnedTensorTransform(
            input_names=xform.input_names,
            output_name=xform.output_name,
            input_binning=new_binning,
            output_binning=new_binning,
            xform_array=interp
        )

        return interp_xform

    @profile
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
        output_binning = self.output_binning.to(**out_units)

        # Account for "missing" dimension(s) (dimensions OneWeight expects for
        # computation of bin volume), and accommodate with a factor equal to
        # the full range. See IceCube wiki/documentation for OneWeight for
        # more info.
        missing_dims_vol = 1
        if 'true_azimuth' not in input_binning:
            missing_dims_vol *= 2*np.pi
        if 'true_coszen' not in input_binning:
            missing_dims_vol *= 2

        # Make binning for smoothing
        # TODO Add support for azimuth
        assert 'true_coszen' in input_binning.names
        assert 'true_energy' in input_binning.names
        assert len(input_binning.names) == 2

        # TODO: define smoothed binning as oversampled version of input
        # binning, or define in the config file!

        smoothing_e_binning = OneDimBinning(
            name='true_energy', tex=r'E_\nu',
            is_log=True, num_bins=39, domain=[1, 80]*ureg.GeV
        )
        smoothing_cz_binning = OneDimBinning(
            name='true_coszen', tex=r'\cos\theta_\nu', is_lin=True,
            num_bins=40, domain=[-1, 1]*ureg(None)
        )
        smoothing_binning = smoothing_e_binning * smoothing_cz_binning
        smoothing_binning = smoothing_binning.reorder_dimensions(
            self.input_binning
        )

        nominal_transforms = []
        raw_transforms = []
        smooth_transforms = []
        interp_transforms = []
        for xform_flavints in self.transform_groups:
            logging.debug("Working on %s effective areas xform" %xform_flavints)

            aeff_transform = self.remaining_events.histogram(
                kinds=xform_flavints,
                binning=smoothing_binning,
                weights_col='weighted_aeff',
                errors=None
            )
            aeff_transform = aeff_transform.hist

            # Divide histogram by
            #   (energy bin width x coszen bin width x azimuth bin width)
            # volumes to convert from sums-of-OneWeights-in-bins to
            # effective areas. Note that volume correction factor for
            # missing dimensions is applied here.
            bin_volumes = smoothing_binning.bin_volumes(attach_units=False)
            aeff_transform /= (bin_volumes * missing_dims_vol)

            bin_counts = self.remaining_events.histogram(
                kinds=xform_flavints,
                binning=smoothing_binning,
                weights_col=None,
                errors=None
            )
            if self.error_method not in [None, False]:
                aeff_err = aeff_transform / np.sqrt(bin_counts)
                aeff_transform = unp.uarray(aeff_transform, aeff_err)

            # For each member of the group, save the raw aeff transform and
            # its smoothed and interpolated versions
            flav_names = [str(flav) for flav in xform_flavints.flavs()]

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
                    smooth_transform = self.slice_smooth(xform)
                    interp_transform = self.interpolate_transform(
                        smooth_transform, new_binning=self.input_binning
                    )
                    raw_transforms.append(xform)
                    smooth_transforms.append(smooth_transform)
                    interp_transforms.append(interp_transform)

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
                            input_binning=smoothing_binning,
                            output_binning=smoothing_binning,
                            xform_array=aeff_transform,
                        )
                        smooth_transform = self.slice_smooth(xform)
                        interp_transform = self.interpolate_transform(
                            smooth_transform, new_binning=self.input_binning
                        )
                        raw_transforms.append(xform)
                        smooth_transforms.append(smooth_transform)
                        interp_transforms.append(interp_transform)

        raw_transforms = TransformSet(transforms=raw_transforms)
        smooth_transforms = TransformSet(transforms=smooth_transforms)
        interp_transforms = TransformSet(transforms=interp_transforms)

        # Clip negative values
        for xform in smooth_transforms:
            xform.xform_array = xform.xform_array.clip(0)
        for xform in interp_transforms:
            xform.xform_array = xform.xform_array.clip(0)

        #
        # DEBUG MODE
        #
        if self.debug_mode:
            import matplotlib as mpl
            mpl.use('pdf')
            import matplotlib.pyplot as plt
            self.raw_transforms = raw_transforms
            self.smooth_transforms = smooth_transforms
            self.interp_transforms = interp_transforms

            #
            # Calculate fractional diff between smoothed and raw transforms
            #
            frac_diff_xforms = []
            values = []
            for raw, smooth in zip(raw_transforms, smooth_transforms):
                smooth_arr = unp.nominal_values(smooth.xform_array)
                raw_arr = unp.nominal_values(raw.xform_array)

                # Make sure you're comparing the right transforms
                assert smooth.input_names == raw.input_names

                # Calculate fractional difference (may have np.inf and np.nan)
                frac_diff = (smooth_arr - raw_arr) / raw_arr

                frac_diff_finite = frac_diff[np.isfinite(frac_diff) &
                                             ~np.isnan(frac_diff)]

                mean = np.mean(frac_diff_finite)
                stddev = np.std(frac_diff_finite)
                mad = np.median(np.abs(frac_diff_finite -
                                       np.median(frac_diff_finite)))
                med = np.median(frac_diff_finite)
                min_val = np.min(frac_diff_finite)
                max_val = np.max(frac_diff_finite)

                values.append(dict(mean=mean, std=stddev, mad=mad, med=med,
                                   min=min_val, max=max_val))

                # Make Transforms out of frac_diff (may contain inf and nans)
                frac_diff = BinnedTensorTransform(
                    input_names=smooth.input_names,
                    output_name=smooth.output_name,
                    input_binning=smooth.input_binning,
                    output_binning=smooth.output_binning,
                    xform_array=frac_diff
                )
                # Append to list of frac_diff transforms
                frac_diff_xforms.append(frac_diff)
            frac_diff_xforms = TransformSet(transforms=frac_diff_xforms)
            self.frac_diff_xforms = frac_diff_xforms

            #
            # Plot raw, smoothed, and interp transforms
            #
            plots = Plotter(stamp='Aeff Transforms')

            # Raw
            plots.init_fig()
            plots.plot_2d_array(raw_transforms, n_rows=2, n_cols=6,
                                cmap=mpl.cm.Paired)
            plots.dump('aeff_raw_transforms')

            # Smoothed
            plots.init_fig()
            plots.plot_2d_array(smooth_transforms, n_rows=2, n_cols=6,
                                cmap=mpl.cm.Paired)
            plots.dump('aeff_smooth_transforms')

            # Interpolated
            plots.init_fig()
            plots.plot_2d_array(interp_transforms, n_rows=2, n_cols=6,
                                cmap=mpl.cm.Paired)
            plots.dump('aeff_interp_transforms')

            #
            # Plot fractional difference and coszen slice comparison
            #

            # Fractional difference
            plots = Plotter(stamp='Comparison Between'+'\n'
                            'Smoothed and Original Aeff'+'\n'
                            r'Plotted value: $\frac{smoothed - orig}{orig}$')
            plots.init_fig()
            plots.log = False
            plots.plot_2d_array(frac_diff_xforms, n_rows=2, n_cols=6,
                                cmap=plt.get_cmap('bwr'), vmin=-1, vmax=1)
            # TODO better way to add text boxes to axes
            for i, _ in enumerate(frac_diff_xforms):
                plt.subplot(2, 6, i+1)
                fields = ['mean', 'std', 'mad', 'med', 'min', 'max']
                textstr = '\n'.join([(f + ' ={' + f + ':7.4f}')
                                     for f in fields]).format(**values[i])
                a_text = mpl.offsetbox.AnchoredText(textstr, loc=1,
                                                    frameon=False)
                plt.gca().add_artist(a_text)
            plots.dump('aeff_frac_diff_raw_smooth')

            # Smooth-vs-raw coszen slice comparison
            # TODO more interactive way of exploring slices
            def plot_slice_comparison(i_xform=0, i_cz=0,
                                      fname='aeff_cz_slice_comparison'):
                """Plot corresponding slices of a transform from
                smooth_transforms and raw_transforms.

                Parameters
                ----------
                i_xform : int
                    Index of the transform in the TransformSet
                i_cz : int
                    Index of the cz slice
                    """
                raw_xform = raw_transforms[i_xform]
                smooth_xform = smooth_transforms[i_xform]

                assert raw_xform.input_binning == smooth_xform.input_binning
                ebins = raw_xform.input_binning.true_energy
                czbin = raw_xform.input_binning.true_coszen[i_cz]
                binning = czbin * ebins

                nom_cz_slice = raw_xform.xform_array[i_cz]
                nom_cz_slice = nom_cz_slice.reshape((1, -1))
                nom_cz_slice = Map(name='raw coszen slice',
                                   hist=nom_cz_slice, binning=binning)
                smth_cz_slice = smooth_xform.xform_array[i_cz]
                smth_cz_slice = smth_cz_slice.reshape((1, -1))
                smth_cz_slice = Map(name='smooth coszen slice',
                                    hist=smth_cz_slice, binning=binning)

                plots = Plotter(stamp='Aeff transform smoothing comparison')
                plots.init_fig()
                plots.plot_1d_projection(nom_cz_slice, 'true_energy')
                plots.plot_1d_projection(smth_cz_slice, 'true_energy')
                plots.add_stamp('Smoothed-vs-original\n'
                                + str(czbin) + '\n'
                                + 'input_names: '+str(raw_xform.input_names)
                                +'\n'
                                + 'output_name: '+str(raw_xform.output_name))
                plots.dump('aeff_cz_slice_comparison')

            self.plot_slice_comparison = plot_slice_comparison
            plot_slice_comparison(i_xform=4, i_cz=39)

        return interp_transforms

    @profile
    def _compute_transforms(self):
        """Compute new effective areas transforms"""
        # Read parameters in in the units used for computation
        aeff_scale = self.params.aeff_scale.value.m_as('dimensionless')
        livetime_s = self.params.livetime.value.m_as('sec')
        logging.trace('livetime = %s --> %s sec'
                      %(self.params.livetime.value, livetime_s))

        if self.particles == 'neutrinos':
            nutau_cc_norm = self.params.nutau_cc_norm.m_as('dimensionless')
            if nutau_cc_norm != 1:
                assert NuFlavIntGroup('nutau_cc') in self.transform_groups
                assert NuFlavIntGroup('nutaubar_cc') in self.transform_groups

        new_transforms = []
        for xform_flavints in self.transform_groups:
            repr_flav_int = xform_flavints[0]
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
