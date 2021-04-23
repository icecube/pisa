"""
Classes and methods needed to do hypersurface interpolation over arbitrary parameters.
"""

__all__ = ['HypersurfaceInterpolator', 'fit_interpolated_hypersurfaces',
            'load_interpolated_hypersurfaces']

__author__ = 'T. Stuttard, A. Trettin'

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


import os
import collections
import copy

import numpy as np
from scipy import interpolate
from hypersurface import Hypersurface
from pisa import FTYPE, ureg
from pisa.utils import matrix
from pisa.utils.jsons import from_json, to_json
from pisa.core.pipeline import Pipeline
from pisa.core.binning import OneDimBinning, MultiDimBinning, is_binning
from pisa.core.map import Map
from pisa.core.param import Param, ParamSet
from pisa.utils.resources import find_resource
from pisa.utils.fileio import mkdir
from pisa.utils.log import logging, set_verbosity
from pisa.utils.comparisons import ALLCLOSE_KW
from uncertainties import ufloat, correlated_values
from uncertainties import unumpy as unp

class HypersurfaceInterpolator(object):
    """Factory for interpolated hypersurfaces.

    After being initialized with a set of hypersurface fits produced at different
    parameters, it uses interpolation to produce a Hypersurface object
    at a given point in parameter space using scipy's `RegularGridInterpolator`.
    
    The interpolation is piecewise-linear between points. All points must lie on a
    rectilinear ND grid.

    Parameters
    ----------
    interpolation_param_spec : dict
        Specification of interpolation parameter grid of the form::
            interpolation_param_spec = {
                'param1': {"values": [val1_1, val1_2, ...], "scales_log": True/False}
                'param2': {"values": [val2_1, val2_2, ...], "scales_log": True/False}
                ...
                'paramN': {"values": [valN_1, valN_2, ...], "scales_log": True/False}
            }
        where values are given as :obj:`Quantity`.
    hs_fits : list of dict
        list of dicts with hypersurfacesthat were fit at the points of the parameter mesh
        defined by interpolation_param_spec
    ignore_nan : bool
        Ignore empty bins in hypersurfaces. The intercept in those bins is set to 1 and
        all slopes are set to 0.

    Notes
    -----
    Be sure to give a support that covers the entire relevant parameter range and a
    good distance beyond! To prevent minimization failure from NaNs, extrapolation
    is used if hypersurfaces outside the support are requested but needless to say
    these numbers are unreliable.

    See Also
    --------
    scipy.interpolate.RegularGridInterpolator :
        class used for interpolation
    """

    def __init__(self, interpolation_param_spec, hs_fits, ignore_nan=True):
        self.ndim = len(interpolation_param_spec.keys())
        # key ordering is important to guarantee that dimensions stay consistent
        msg = "interpolation params must be specified as a dict with ordered keys"
        assert isinstance(interpolation_param_spec, collections.OrderedDict), msg
        for k, v in interpolation_param_spec.items():
            assert set(v.keys()) == {"values", "scales_log"}
            assert isinstance(v["values"], collections.Sequence)
        self.interp_param_spec = interpolation_param_spec
        reference_hs = hs_fits[0]["hs_fit"]
        # we are going to produce the hypersurface from a state that is the same
        # as the reference, only the coefficients and covariance matrices are
        # injected from the interpolation.
        self._reference_state = copy.deepcopy(reference_hs.serializable_state)
        # for cleanliness we wipe numbers from the original state
        self._reference_state["intercept_sigma"] = np.nan
        self._reference_state["fit_maps_norm"] = None
        self._reference_state["fit_maps_raw"] = None
        self._reference_state["fit_chi2"] = np.nan
        for param in self._reference_state['params'].values():
            param['fit_coeffts_sigma'] = np.full_like(
                param['fit_coeffts_sigma'], np.nan)
        # Instead of holding numbers, these coefficients and covariance matrices are
        # interpolator objects the produce them at the requested point.
        # The shape of fit_coeffts is [binning ..., fit coeffts]
        self.coeff_shape = reference_hs.fit_coeffts.shape
        self.coefficients = None
        # The shape of fit_cov_mat is [binning ..., fit coeffts, fit coeffts]
        self.covars_shape = reference_hs.fit_cov_mat.shape
        self.covars = None
        
        # We now need to massage the fit coefficients into the correct shape
        # for interpolation.
        # The dimensions of the interpolation parameters come first, the dimensions
        # of the hypersurface coefficients comes last.
        self.interp_shape = tuple(len(v["values"]) for v in self.interp_param_spec.values())
        # dimension is [interp_shape, binning..., fit coeffts]
        self._coeff_z = np.zeros(self.interp_shape + self.coeff_shape)
        # dimension is [interp_shape, binning..., fit coeffts, fit coeffts]
        self._covar_z = np.zeros(self.interp_shape + self.covars_shape)
        # Here we use the same indexing as below in `fit_hypersurfaces`
        for i, idx in enumerate(np.ndindex(self.interp_shape)):
            # As an additional safety measure, we check that the parameters are what
            # we expect to find at this index.
            expected_params = dict(
                (n, self.interp_param_spec[n]["values"][idx[j]])
                for j, n in enumerate(self.interp_param_spec.keys())
            )
            param_values = hs_fits[i]["param_values"]
            msg = ("The stored values where hypersurfaces were fit do not match those"
                   "in the interpolation grid.")
            assert np.all([expected_params[n].m == param_values[n].m
                           for n in self.interp_param_spec.keys()]), msg
            self._coeff_z[idx] = hs_fits[i]["hs_fit"].fit_coeffts
            self._covar_z[idx] = hs_fits[i]["hs_fit"].fit_cov_mat
        
        grid_coords = list(
            np.array([val.m for val in val_list["values"]])
            for val_list in self.interp_param_spec.values()
        )
        # If a parameter scales as log, we give the log of the parameter to the 
        # interpolator. We must not forget to do this again when we call the
        # interpolator later!
        for i, param_name in enumerate(self.interpolation_param_names):
            if self.interp_param_spec[param_name]["scales_log"]:
                grid_coords[i] = np.log10(grid_coords[i])
        self.coefficients = interpolate.RegularGridInterpolator(
            grid_coords,
            self._coeff_z,
            # while we do enable extrapolation, it is not at all reliable... we only
            # use it to avoid crashes/nans if the minimizer tests extreme values outside
            # the domain
            bounds_error=False, fill_value=None
        )
        self.covars = interpolate.RegularGridInterpolator(
            grid_coords,
            self._covar_z,
            bounds_error=False, fill_value=None
        )
        # In order not to spam warnings, we only want to warn about non positive
        # semi definite covariance matrices once for each bin. We store the bin
        # indeces for which the warning has already been issued.
        self.covar_bins_warning_issued = []
        self.ignore_nan = ignore_nan
    
    @property
    def interpolation_param_names(self):
        return list(self.interp_param_spec.keys())
    
    @property
    def param_names(self):
        return list(self._reference_state["params"].keys())
    
    @property
    def num_interp_params(self):
        return len(self.interp_param_spec.keys())
    
    def get_hypersurface(self, **param_kw):
        """
        Get a Hypersurface object with interpolated coefficients.

        Parameters
        ----------
        **param_kw
            Parameters are given as keyword arguments, where the names
            of the arguments must match the names of the parameters over
            which the hypersurfaces are interpolated. The values
            are given as :obj:`Quantity` objects with units.
        """
        assert set(param_kw.keys()) == set(self.interp_param_spec.keys()), "invalid parameters"
        # getting param magnitudes in the same units as the parameter specification
        x = np.array([
            param_kw[p].m_as(self.interp_param_spec[p]["values"][0].u)
            # we have checked that this is an OrderedDict so that the order of x is not
            # ambiguous here
            for p in self.interp_param_spec.keys()
        ])
        # if a parameter scales as log, we have to take the log here again
        for i, param_name in enumerate(self.interpolation_param_names):
            if self.interp_param_spec[param_name]["scales_log"]:
                x[i] = np.log10(x[i])
        
        state = copy.deepcopy(self._reference_state)
        # fit covariance matrices are stored directly in the state while fit coeffts
        # must be assigned with the setter method...
        # need squeeze here because the RegularGridInterpolator always puts another 
        # dimension around the output
        state["fit_cov_mat"] = np.squeeze(self.covars(x))
        assert state["fit_cov_mat"].shape == self.covars_shape
        for idx in np.ndindex(state['fit_cov_mat'].shape):
            if self.ignore_nan: continue
            assert np.isfinite(state['fit_cov_mat'][idx]), ("invalid cov matrix "
                f"element encountered at {param_kw} in loc {idx}")
        # check covariance matrices for symmetry, positive semi-definiteness
        for bin_idx in np.ndindex(state['fit_cov_mat'].shape[:-2]):
            m = state['fit_cov_mat'][bin_idx]
            if self.ignore_nan and np.any(~np.isfinite(m)):
                state['fit_cov_mat'][bin_idx] = np.identity(m.shape[0])
                m = state['fit_cov_mat'][bin_idx]
            assert np.allclose(
                m, m.T, rtol=ALLCLOSE_KW['rtol']*10.), f'cov matrix not symmetric in bin {bin_idx}'
            if not matrix.is_psd(m):
                state['fit_cov_mat'][bin_idx] = matrix.fronebius_nearest_psd(m)
                if not bin_idx in self.covar_bins_warning_issued:
                    logging.warn(
                        f'Invalid covariance matrix fixed in bin: {bin_idx}')
                    self.covar_bins_warning_issued.append(bin_idx)
        hypersurface = Hypersurface.from_state(state)
        coeffts = np.squeeze(self.coefficients(x))  # calls interpolator
        assert coeffts.shape == self.coeff_shape
        # check that coefficients exist and if not replace with default values
        for idx in np.ndindex(self.coeff_shape):
            if self.ignore_nan and ~np.isfinite(coeffts[idx]):
                coeffts[idx] = 1 if idx[-1] == 0 else 0  # set intercept to 1, slopes 0
            assert np.isfinite(coeffts[idx]), ("invalid coeff encountered at "
                f"{param_kw} in loc {idx}")
        # the setter method defined in the Hypersurface class takes care of
        # putting the coefficients in the right place in their respective parameters
        hypersurface.fit_coeffts = coeffts
        return hypersurface
    
    def make_slices(self, *xi):
        """Make slices of hypersurfaces for plotting.

        In some covariance matrices, the spline fits are corrected to make
        the matrix positive semi-definite. The slices produced by this function
        include all of those effects.

        Parameters
        ----------
        xi : list of ndarray
            Points at which the hypersurfaces are to be evaluated. The length of the 
            list must equal the number of parameters, each ndarray in the list must have
            the same shape (slice_shape).

        Returns
        -------
        coeff_slices : numpy.ndarray
            slices in fit coefficients. Size: (binning..., number of coeffs) + slice_shape
        covar_slices : numpy.ndarray
            slices in covariance matrix elements.
            Size: (binning..., number of coeffs, number of coeffs) + slice_shape
        """
        slice_shape = xi[0].shape
        for x in xi:
            assert x.shape == slice_shape
        assert len(xi) == self.num_interp_params
        coeff_slices = np.zeros(self.coeff_shape + slice_shape)
        covar_slices = np.zeros(self.covars_shape + slice_shape)
        for idx in np.ndindex(slice_shape):
            pars = collections.OrderedDict()
            for i, name in enumerate(self.interpolation_param_names):
                pars[name] = xi[i][idx]
            hs = self.get_hypersurface(**pars)
            slice_idx = (Ellipsis,) + idx
            coeff_slices[slice_idx] = hs.fit_coeffts
            covar_slices[slice_idx] = hs.fit_cov_mat
        return coeff_slices, covar_slices

    def plot_fits_in_bin(self, bin_idx, ax=None, n_steps=20, **param_kw):
        """
        Plot the coefficients as well as covariance matrix elements as a function
        of the interpolation parameters.

        Parameters
        ----------
            bin_idx : tuple
                index of the bin for which to plot the fits
            ax : 2D array of axes, optional
                axes into which to place the plots. If None (default),
                appropriate axes will be generated. Must have at least
                size (n_coeff, n_coeff + 1).
            n_steps : int, optional
                number of steps to plot between minimum and maximum
            **param_kw :
                Parameters to be fixed when producing slices. If the interpolation 
                is in N-D, then (N-2) parameters need to be fixed to produce 2D plots
                of the remaining 2 parameters and (N-1) need to be fixed to produce a
                1D slice.
        """
        plot_dim = self.ndim - len(param_kw.keys())
        assert plot_dim <= 2, "plotting only supported in 1D or 2D"
        import matplotlib.pyplot as plt
        n_coeff = self.coeff_shape[-1]
        hs_param_names = list(self._reference_state['params'].keys())
        hs_param_labels = ["intercept"] + [f"{p} p{i}" for p in hs_param_names
                                           for i in range(self._reference_state['params'][p]['num_fit_coeffts'])]
        if ax is None:
            fig, ax = plt.subplots(nrows=n_coeff, ncols=n_coeff+1,
                                   squeeze=False, sharex=True,
                                   figsize=(20, 10))
        # remember whether the plots need log scale or not, by default not
        x_is_log = False
        y_is_log = False
        
        # names of the variables we are plotting
        plot_names = set(self.interpolation_param_names) - set(param_kw.keys())
        if plot_dim == 1:
            x_name = list(plot_names)[0]
        else:
            x_name, y_name = list(plot_names)

        # in both 1D and 2D cases, we always plot at least an x-variable
        x_unit = self.interp_param_spec[x_name]["values"][0].u
        # we need the magnitudes here so that units are unambiguous when we make
        # the linspace/geomspace for plotting
        x_mags = [v.m_as(x_unit) for v in self.interp_param_spec[x_name]["values"]]
        if self.interp_param_spec[x_name]["scales_log"]:
            x_plot = np.geomspace(np.min(x_mags), np.max(x_mags), n_steps)
            x_is_log = True
        else:
            x_plot = np.linspace(np.min(x_mags), np.max(x_mags), n_steps)
        # we put the unit back later
        if plot_dim == 1:
            # To make slices, we need to set any variables we do not plot over to the
            # value given in param_kw.
            slice_args = []
            # We need to make sure that we give the values in the correct order!
            for n in self.interpolation_param_names:
                if n == x_name:
                    slice_args.append(x_plot * x_unit)
                elif n in param_kw.keys():
                    # again, insure that the same unit is used that went into the 
                    # interpolation
                    param_unit = self.interp_param_spec[n]["values"][0].u
                    slice_args.append(
                        np.full(x_plot.shape, param_kw[n].m_as(param_unit)) * param_unit
                    )
                else:
                    raise ValueError("parameter neither specified nor plotted")
            coeff_slices, covar_slices = self.make_slices(*slice_args)
        else:
            # if we are in 2D, we need to do the same procedure again for the y-variable
            y_unit = self.interp_param_spec[y_name]["values"][0].u
            y_mags = [v.m_as(y_unit) for v in self.interp_param_spec[y_name]["values"]]
            if self.interp_param_spec[y_name]["scales_log"]:
                # we add one step to the size in y so that transposition is unambiguous
                y_plot = np.geomspace(np.min(y_mags), np.max(y_mags), n_steps + 1)
                y_is_log = True
            else:
                y_plot = np.linspace(np.min(y_mags), np.max(y_mags), n_steps + 1)
            
            x_mesh, y_mesh = np.meshgrid(x_plot, y_plot)
            slice_args = []
            for n in self.interpolation_param_names:
                if n == x_name:
                    slice_args.append(x_mesh * x_unit)
                elif n == y_name:
                    slice_args.append(y_mesh * y_unit)
                elif n in param_kw.keys():
                    # again, insure that the same unit is used that went into the 
                    # interpolation
                    param_unit = self.interp_param_spec[n]["values"][0].u
                    slice_args.append(
                        np.full(x_mesh.shape, param_kw[n].m_as(param_unit)) * param_unit
                    )
                else:
                    raise ValueError("parameter neither specified nor plotted")
            coeff_slices, covar_slices = self.make_slices(*slice_args)

        # first column plots fit coefficients
        for i in range(n_coeff):
            z_slice = coeff_slices[bin_idx][i]
            if plot_dim == 1:
                ax[i, 0].plot(x_plot, z_slice, label='interpolation')
                # Plotting the original input points only works if the interpolation
                # is in 1D. If we are plotting a 1D slice from a 2D interpolation, this
                # does not work.
                # The number of fit points is the first dimension in self._coeff_z
                if plot_dim == self.ndim:
                    slice_idx = (Ellipsis,) + bin_idx + (i,)
                    ax[i, 0].scatter(x_mags, self._coeff_z[slice_idx],
                                     color='k', marker='x', label='fit points')
                ax[i, 0].set_ylabel(hs_param_labels[i])
            else:
                pc = ax[i, 0].pcolormesh(x_mesh, y_mesh, z_slice)
                cbar = plt.colorbar(pc, ax=ax[i, 0])
                cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))
                ax[i, 0].set_ylabel(y_name)
                ax[i, 0].set_xlabel(x_name)
            
            # later column plots the elements of the covariance matrix
            for j in range(0, n_coeff):
                z_slice = covar_slices[bin_idx][i, j]
                if plot_dim == 1:
                    ax[i, j+1].plot(x_plot, z_slice, label='interpolation')
                    # Same problem as above, only in 1D case can this be shown
                    # the number of points is the first dim in self._covar_z
                    if plot_dim == self.ndim:
                        coeff_idx = (Ellipsis,) + bin_idx + (i, j)
                        ax[i, j+1].scatter(x_mags, self._covar_z[coeff_idx],
                                           color='k', marker='x', label='fit points')
                else:
                    pc = ax[i, j+1].pcolormesh(x_mesh, y_mesh, z_slice)
                    cbar = plt.colorbar(pc, ax=ax[i, j+1])
                    cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))
                    ax[i, j+1].set_ylabel(y_name)
                    ax[i, j+1].set_xlabel(x_name)
        
        if plot_dim == 1:
            # in the 1D case, labels can be placed on the x and y axes
            for j in range(n_coeff+1):
                ax[-1, j].set_xlabel(x_name)
            ax[0, 0].set_title('coefficient')
            for j in range(n_coeff):
                ax[0, j+1].set_title(f'cov. {hs_param_labels[j]}')
        else:
            # in the 2D case, we need separate annotations
            rows = hs_param_labels
            cols = ["coefficient"] + [f"cov. {hl}" for hl in hs_param_labels]
            pad = 20
            for a, col in zip(ax[0], cols):
                a.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                           xycoords='axes fraction', textcoords='offset points',
                           size='x-large', ha='center', va='baseline')

            for a, row in zip(ax[:, 0], rows):
                a.annotate(row, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - pad, 0),
                           xycoords=a.yaxis.label, textcoords='offset points',
                           size='x-large', ha='right', va='center')
        for i, j in np.ndindex((n_coeff, n_coeff+1)):
            if x_is_log: ax[i, j].set_xscale("log")
            if y_is_log: ax[i, j].set_yscale("log")
            ax[i, j].grid()
            if plot_dim == 1:
                ax[i, j].legend()
            ax[i, j].relim()
            ax[i, j].autoscale_view()
            if not x_is_log:
                ax[i, j].ticklabel_format(style='sci', scilimits=(0, 0), axis="x")
            if not y_is_log:
                ax[i, j].ticklabel_format(style='sci', scilimits=(0, 0), axis="y")
        fig.tight_layout()
        if plot_dim == 2:
            fig.subplots_adjust(left=0.15, top=0.95)
        return fig


def fit_interpolated_hypersurfaces(
    nominal_dataset, sys_datasets, params, output_file, combine_regex=None,
    log=False, interpolation_param_spec=None, **hypersurface_fit_kw):
    '''
    A helper function that a user can use to fit hypersurfaces to a bunch of simulation
    datasets, and save the results to a file. Basically a wrapper of Hypersurface.fit,
    handling common pre-fitting tasks like producing mapsets from piplelines, merging
    maps from similar specifies, etc.

    Note that this supports fitting multiple hypersurfaces to the datasets, e.g. one per
    simulated species. Returns a dict with format: { map_0_key : map_0_hypersurface,
    ..., map_N_key : map_N_hypersurface, }

    Parameters
    ----------
    nominal_dataset : dict
        Definition of the nominal dataset. Specifies the pipleline with which the maps
        can be created, and the values of all systematic parameters used to produced the
        dataset.
        Format must be:
            nominal_dataset = {
                "pipeline_cfg" = <pipeline cfg file (either cfg file path or dict)>),
                "sys_params" = { param_0_name : param_0_value_in_dataset, ..., param_N_name : param_N_value_in_dataset }
            }
        Sys params must correspond to the provided HypersurfaceParam instances provided
        in the `params` arg.

    sys_datasets : list of dicts
        List of dicts, where each dict defines one of the systematics datasets to be
        fitted. The format of each dict is the same as explained for `nominal_dataset`

    params : list of HypersurfaceParams
        List of HypersurfaceParams instances that define the hypersurface. Note that
        this defined ALL hypersurfaces fitted in this function, e.g. only supports a
        single parameterisation for all maps (this is almost almost what you want).

    output_file : str
        Filename to store the output as.

    combine_regex : list of str, or None
        List of string regex expressions that will be used for merging maps. Used to
        combine similar species. Must be something that can be passed to the
        `MapSet.combine_re` function (see that functions docs for more details). Choose
        `None` is do not want to perform this merging.
    
    interpolation_param_spec : collections.OrderedDict or None
        Specification of parameter grid that hypersurfaces should be interpolated over.
        If None (default), just one set of hypersurfaces at the nominal parameters is
        produced. The dict should have the following form::
            interpolation_param_spec = {
                'param1': {"values": [val1_1, val1_2, ...], "scales_log": True/False}
                'param2': {"values": [val2_1, val2_2, ...], "scales_log": True/False}
                ...
                'paramN': {"values": [valN_1, valN_2, ...], "scales_log": True/False}
            }
        The hypersurfaces will be fit on an N-dimensional rectilinear grid over
        parameters 1 to N. The flag `scales_log` indicates that the interpolation over
        that parameter should happen in log-space.

    hypersurface_fit_kw : kwargs
        kwargs will be passed on to the calls to `Hypersurface.fit`
    '''

    # Take (deep) copies of lists/dicts to avoid modifying the originals
    # Useful for cases where this function is called in a loop (e.g. leave-one-out tests)
    nominal_dataset = copy.deepcopy(nominal_dataset)
    sys_datasets = copy.deepcopy(sys_datasets)
    params = copy.deepcopy(params)

    # Check types
    assert isinstance(sys_datasets, collections.Sequence)
    assert isinstance(params, collections.Sequence)
    assert isinstance(output_file, str)
    if interpolation_param_spec is not None:
        # there must not be any ambiguity between fitting the hypersurfaces and 
        # interpolating them later
        msg = "interpolation params must be specified as a dict with ordered keys"
        assert isinstance(interpolation_param_spec, collections.OrderedDict), msg
        for k, v in interpolation_param_spec.items():
            assert set(v.keys()) == {"values", "scales_log"}
            assert isinstance(v["values"], collections.Sequence)
    
    # Check output format and path
    assert output_file.endswith(".json") or output_file.endswith(".json.bz2")
    assert os.path.isdir(os.path.dirname(output_file)), "output directory does not exist"
    
    # Check formatting of datasets is as expected
    all_datasets = [nominal_dataset] + sys_datasets
    for dataset in all_datasets:
        assert isinstance(dataset, collections.Mapping)
        assert "pipeline_cfg" in dataset
        assert isinstance(dataset["pipeline_cfg"], (str, collections.Mapping))
        assert "sys_params" in dataset
        assert isinstance(dataset["sys_params"], collections.Mapping)

    # Check params
    assert len(params) >= 1
    for p in params:
        assert isinstance(p, HypersurfaceParam)

    # Report inputs
    msg = "Hypersurface fit details :\n"
    msg += f"  Num params            : {len(params)}\n"
    msg += f"  Num fit coefficients  : {sum([p.num_fit_coeffts for p in params])}\n"
    msg += f"  Num datasets          : 1 nominal + {len(sys_datasets)} systematics\n"
    msg += f"  Nominal values        : {nominal_dataset['sys_params']}\n"
    if interpolation_param_spec is None:
        msg += "No interpolation parameters are applied."
    else:
        msg += "Hypersurfaces are fit on the following grid:\n"
        msg += str(interpolation_param_spec)
    logging.info(msg)
    
    # Create the pipelines to load all the data in memory
    nominal_pipeline = Pipeline(nominal_dataset["pipeline_cfg"])
    sys_pipelines = [Pipeline(sys_dataset["pipeline_cfg"])
                     for sys_dataset in sys_datasets]

    def fit_hypersurface_set():
        """Fit a set of hypersurfaces.
        
        Returns an OrderedDict with one fitted hypersurface for each element in 
        `combine_regex`.
        """
        
        # Getting outputs after updating the parameters is very efficient, only the
        # stages for which parameters have changed are recomputed.
        nominal_dataset["mapset"] = nominal_pipeline.get_outputs()  # return_sum=False)
        for sys_dataset, sys_pipeline in zip(sys_datasets, sys_pipelines):
            sys_dataset["mapset"] = sys_pipeline.get_outputs()  # return_sum=False)

        # Merge maps according to the combine regex, if one was provided
        if combine_regex is not None:
            nominal_dataset["mapset"] = nominal_dataset["mapset"].combine_re(
                combine_regex)
            for sys_dataset in sys_datasets:
                sys_dataset["mapset"] = sys_dataset["mapset"].combine_re(
                    combine_regex)

        hypersurfaces = collections.OrderedDict()

        for map_name in nominal_dataset["mapset"].names:
            nominal_map = nominal_dataset["mapset"][map_name]
            nominal_param_values = nominal_dataset["sys_params"]

            sys_maps = [sys_dataset["mapset"][map_name]
                        for sys_dataset in sys_datasets]
            sys_param_values = [sys_dataset["sys_params"]
                                for sys_dataset in sys_datasets]

            hypersurface = Hypersurface(
                params=copy.deepcopy(params),
                initial_intercept=0. if log else 1.,  # Initial value for intercept
                log=log
            )

            hypersurface.fit(
                nominal_map=nominal_map,
                nominal_param_values=nominal_param_values,
                sys_maps=sys_maps,
                sys_param_values=sys_param_values,
                norm=True,
                **hypersurface_fit_kw
            )

            logging.debug("\nFitted hypersurface report:\n%s" % hypersurface)
            hypersurfaces[map_name] = hypersurface

        return hypersurfaces
    
    if interpolation_param_spec is None:
        output_data = fit_hypersurface_set()
    else:
        # We store all the hypersurfaces that we will later interpolate over into one
        # long list in addition to the interpolation param specifications.
        output_data = collections.OrderedDict(
            interpolation_param_spec=interpolation_param_spec,
            hs_fits=[]
        )
        # because we require this to be an OrderedDict, there is no ambiguity in the
        # construction of the mesh here
        param_names = list(interpolation_param_spec.keys())
        grid_shape = tuple(len(v["values"]) for v in interpolation_param_spec.values())
        
        for idx in np.ndindex(grid_shape):
            param_update = ParamSet()
            for i, n in enumerate(param_names):
                param = nominal_pipeline.params[n]
                param.value = interpolation_param_spec[n]["values"][idx[i]]
                param_update.extend(param)
            logging.info(f"Updating params with:\n{param_update}")
            for pl in sys_pipelines + [nominal_pipeline]:
                pl.update_params(param_update)
            
            # we explicitly store the param values used in this particular fit
            # as a way to cross-check when we actually do the interpolation
            param_values = dict((n, param_update[n].value) for n in param_names)
            hs_fit = fit_hypersurface_set()

            output_data["hs_fits"].append({
                "param_values": param_values,
                "hs_fit": hs_fit,
            })

    # Write to a json file
    to_json(output_data, output_file)

    logging.info("Fit results written to file: %s" % output_file)

    return output_file


def load_interpolated_hypersurfaces(input_file):
    '''
    Load a set of interpolated hypersurfaces from a file.

    Analogously to "load_hypersurfaces", this function returns a
    collection with a HypersurfaceInterpolator object for each Map.

    Parameters
    ----------
    input_file : str
        A JSON input file as produced by fit_hypersurfaces if interpolation params
        were given. It has the form::
            {
                interpolation_param_spec = {
                    'param1': {"values": [val1_1, val1_2, ...], "scales_log": True/False}
                    'param2': {"values": [val2_1, val2_2, ...], "scales_log": True/False}
                    ...
                    'paramN': {"values": [valN_1, valN_2, ...], "scales_log": True/False}
                },
                'hs_fits': [
                    <list of dicts where keys are map names such as 'nue_cc' and values
                    are hypersurface states>
                ]
            }

    Returns
    -------
    collections.OrderedDict
        dictionary with a :obj:`HypersurfaceInterpolator` for each map
    '''
    assert isinstance(input_file, str)

    if input_file.endswith("json") or input_file.endswith("json.bz2"):
        input_data = from_json(input_file)
        assert set(['interpolation_param_spec', 'hs_fits']).issubset(
            set(input_data.keys())), 'missing keys'
        map_names = None
        # input_data['hs_fits'] is a list of dicts, each dict contains "param_values"
        # and "hs_fit" 
        for hs_fit_dict in input_data['hs_fits']:
            # this is still not the actual Hypersurface, but a dict with the (linked)
            # maps and the HS fit for the map...
            hs_state_maps = hs_fit_dict["hs_fit"]
            if map_names is None:
                map_names = list(hs_state_maps.keys())
            else:
                assert set(map_names) == set(hs_state_maps.keys()), "inconsistent maps"
            # When data is recovered from JSON, the object states are not automatically
            # converted to the corresponding objects, so we need to do it manually here.
            for map_name in map_names:
                hs_state_maps[map_name] = Hypersurface.from_state(hs_state_maps[map_name])

        logging.info(f"Read hypersurface maps: {map_names}")
        
        # Now we have a list of dicts where the map names are on the lower level.
        # We need to convert this into a dict of HypersurfaceInterpolator objects.
        output = collections.OrderedDict()
        for m in map_names:
            hs_fits = [{"param_values": fd["param_values"], "hs_fit": fd['hs_fit'][m]} for fd in input_data['hs_fits']]
            output[m] = HypersurfaceInterpolator(input_data['interpolation_param_spec'], hs_fits)
    else:
        raise Exception("unknown file format")
    return output
