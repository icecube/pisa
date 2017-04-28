import numpy as np
from pisa.utils.log import logging
from scipy.interpolate import splrep, splev, interp1d

def spline_smooth(array, spline_binning, eval_binning, axis=0, smooth_factor=5, k=3, errors=None):
    """Fuction for spline-smoothing arrays

    It is smoothing in slices along one axis, assuming 2d arrays
    The smoothing is done by splines

    Parameters
    ----------

    array : 2d-array
        array to be smoothed
    spline_binning : OneDimBinning
        Binning of axis on which to construct the spline
        Must corrspond to the array dimension
    axis : int
        Index of the axis along which to smooth
    eval_binning : OneDimBinning
        Binning on which to evaluate the constructed spline
    smooth_factor : float
        smoothing factor for spline
    k : int
        spline degree
    errors : 2d-array or None
        uncertainties on the array

    Notes
    -----
    could be expanded to nd arrays to generalize it
    """
    # only working for 2d right now!
    if array.ndim != 2:
        raise ValueError('cannot do other dimensions than 2 right now, sorry')
    # points at which to evaluate splines
    spline_points = spline_binning.midpoints
    if axis == 1:
        # always smooth along axis=0
        array = array.T
        if errors is not None:
            errors = errors.T

    smoothed_slices = []
    interp_errors = None if errors is None else []
    for index in range(array.shape[1]):
        # take slice
        h_slice = array[:, index]
        if errors is None:
            weights = None
        else:
            h_errors = errors[:, index]
            #replace zeros errors minimum avrage value along other axis
            for i in range(len(h_errors)):
                if h_errors[i] == 0:
                    if np.sum(errors[i, :]) == 0:
                        h_errors[i] = 0
                        logging.warning('Detected row in array with all zero values, this can be problematic for spline smoothing!')
                    else:
                        h_errors[i] = np.min(errors[i, :][errors[i, :] != 0])
            weights = 1./h_errors
        # Fit spline to slices
        slice_spline = splrep(
            spline_points, h_slice, weights,
            k=k, s=smooth_factor,
        )
        # eval spline over midpoints
        smoothed_slice = splev(eval_binning.midpoints, slice_spline)
        # Assert that there are no nan or inf values in smoothed cz-slice
        assert np.all(np.isfinite(smoothed_slice))
        smoothed_slices.append(smoothed_slice)

        if errors is not None:
            erf = interp1d(spline_points, 1./weights, fill_value="extrapolate")
            new_errors = erf(eval_binning.midpoints)
            interp_errors.append(new_errors)

    # Convert list of slices to array
    smoothed_array = np.array(smoothed_slices)
    if errors is not None:
        interp_errors = np.array(interp_errors)
    # swap axes back if necessary
    if axis == 0:
        smoothed_array = smoothed_array.T
        if errors is not None:
            interp_errors = interp_errors.T

    return smoothed_array, interp_errors
