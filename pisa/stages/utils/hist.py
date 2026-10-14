"""
Stage to transform arrays with weights into actual `histograms`
that represent event counts
"""

import numpy as np

from pisa.core.stage import Stage
from pisa.core.translation import histogram
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.utils.profiler import profile
from pisa.utils.log import logging

__all__ = ['hist', 'init_test']


class hist(Stage):  # pylint: disable=invalid-name
    """Stage to histogram events.

    Parameters
    ----------
    unweighted : bool, default False
        Return un-weighted event counts in each bin
    apply_unc_weights : bool, default False
        The corresponding "unc_weights" (see notes) will be used to rescale
        the "weights". If, in addition, error_method="sumw2", they will be
        used in computing "errors" and "bin_unc2".

    Notes
    -----

    Expected container keys are::

        "weights", "unc_weights" (if `apply_unc_weights`)

    In case `calc_mode` is a :py:class:`~.core.binning.MultiDimBinning`, a transfer
    matrix containing fractions/probabilities is computed, which distributes weights
    from the `calc_mode` bins to the `apply_mode` bins proportionally. Hence, weights
    in `calc_mode` representation are expected to correspond to *summed* weights.
    In contrast, the translation mode for "unc_weights" is explicitly set to "average"
    by this service, i.e., before they are obtained in binned `calc_mode`.

    In case `error_method = "sumw2", variables "errors" and "bin_unc2" will be added
    to the containers. The variable "unc_weights" is not required for this, but is
    in case of `apply_unc_weights` and will then modify "errors" and "bin_unc2".
    """

    def __init__(
        self,
        apply_unc_weights=False,
        unweighted=False,
        **std_kwargs,
    ):
        expected_container_keys = [
            'weights'
        ]
        if apply_unc_weights:
            expected_container_keys.append('unc_weights')

        # apply_mode can be detected automatically for convenience
        supported_reps = {
            'calc_mode': [MultiDimBinning, "events"],
            'apply_mode': [None, MultiDimBinning],
        }
        # init base class
        super().__init__(
            expected_params=(),
            expected_container_keys=expected_container_keys,
            supported_reps=supported_reps,
            **std_kwargs,
        )

        self.apply_unc_weights = apply_unc_weights
        self.unweighted = unweighted

    def setup_function(self):

        if self.apply_mode is None:
            self.apply_mode = self.data["output_binning"]
        else:
            assert self.apply_mode == self.data["output_binning"]

        if isinstance(self.calc_mode, MultiDimBinning):

            # The two binnings must be exclusive
            assert len(set(self.calc_mode.names) & set(self.apply_mode.names)) == 0

            transform_binning = self.calc_mode + self.apply_mode

            # Create a transfer matrix: transform[i,j] = fraction of calc_bin_i's
            # events that go to apply_bin_j
            for container in self.data:
                self.data.representation = "events"
                # Get all binning variables in event-by-event representation
                sample = [container[name] for name in transform_binning.names]
                # Unweighted histogram: no. events in each [calc_bin_i, apply_bin_j]
                joint_counts = histogram(sample, None, transform_binning, averaged=False)
                # Automatically determine size of final dimension (apply_mode.size)
                joint_counts_reshaped = joint_counts.reshape(self.calc_mode.shape + (-1,))
                assert joint_counts_reshaped.shape[-1] == self.apply_mode.size
                # Sum along apply_bins to get total event no. per calc_bin
                calc_bin_totals = joint_counts_reshaped.sum(axis=-1, keepdims=True)
                # Normalize to these totals (replace NaN -> 0 in case of 0/0)
                # (-> 1 along apply_bin axis for each populated calc_bin, otherwise 0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    transform = joint_counts_reshaped / calc_bin_totals
                    transform = np.nan_to_num(transform)
                assert transform.shape == tuple(self.calc_mode.num_bins + [self.apply_mode.size])

                self.data.representation = self.calc_mode
                container["hist_transform"] = transform
                # calc_mode dimensions now flattened by Container.__add_data
                assert container["hist_transform"].shape == (self.calc_mode.size, self.apply_mode.size)

        elif self.calc_mode == "events":
            # For dimensions where the binning is irregular, we pre-compute the
            # index that each sample falls into and then bin regularly in the index.
            # For dimensions that are logarithmic, we add a linear binning in
            # the logarithm.
            dimensions = []
            for dim in self.apply_mode:
                if dim.is_irregular:
                    # create a new axis with digitized variable
                    varname = dim.name + "__" + self.apply_mode.name + "_idx"
                    new_dim = OneDimBinning(
                        varname, domain=[0, dim.num_bins], num_bins=dim.num_bins
                    )
                    dimensions.append(new_dim)
                    for container in self.data:
                        container.representation = "events"
                        x = container[dim.name] * dim.units
                        # Compute the bin index each sample would fall into, and
                        # shift by -1 such that samples below the binning range
                        # get assigned the index -1.
                        x_idx = np.searchsorted(dim.bin_edges, x, side="right") - 1
                        # To be consistent with numpy histogramming, we need to
                        # shift those values that are exactly at the uppermost edge
                        # down one index such that they are included in the highest
                        # bin instead of being treated as an outlier.
                        on_edge = x == dim.bin_edges[-1]
                        x_idx[on_edge] -= 1
                        container[varname] = x_idx
                elif dim.is_log:
                    # We don't compute the log of the variable just yet, this
                    # will be done later during `apply_function` using the
                    # representation mechanism.
                    new_dim = OneDimBinning(
                        dim.name, domain=np.log(dim.domain.m), num_bins=dim.num_bins
                    )
                    dimensions.append(new_dim)
                else:
                    dimensions.append(dim)
            self.data["regularized_output_binning"] = MultiDimBinning(dimensions)
            logging.debug(
                "Using regularized binning:\n%s", str(self.data["regularized_output_binning"])
            )

    @profile
    def apply_function(self):

        if isinstance(self.calc_mode, MultiDimBinning):

            if self.unweighted:
                raise NotImplementedError(
                    "Unweighted hist only implemented in event-wise calculation"
                )
            for container in self.data:

                container.representation = self.calc_mode
                if "astro_weights" in container.keys:
                    weights = container["weights"] + container["astro_weights"]
                else:
                    weights = container["weights"]
                if self.apply_unc_weights:
                    # These need to be bin-averaged, otherwise we are double counting
                    container.translation_modes["unc_weights"] = "average"
                    unc_weights = container["unc_weights"]
                else:
                    unc_weights = np.ones(weights.shape)
                logging.trace("Using 'unc_weights' histogram %s for '%s'",
                              unc_weights, container.name)
                transform = container["hist_transform"]

                weights_to_transform = unc_weights * weights
                hist = weights_to_transform @ transform
                logging.trace(
                    "Performed matrix multiplication of 'weights' with shape"
                    " %s with transform with shape %s to yield histogram with"
                    " shape %s.", weights_to_transform.shape, transform.shape,
                    hist.shape
                )

                if self.error_method == "sumw2":
                    sumw2 = np.square(weights_to_transform) @ transform
                    bin_unc2 = (np.square(unc_weights) * weights) @ transform

                container.representation = self.apply_mode
                container["weights"] = hist

                if self.error_method == "sumw2":
                    container["errors"] = np.sqrt(sumw2)
                    container["bin_unc2"] = bin_unc2

        elif self.calc_mode == "events":
            for container in self.data:
                container.representation = self.calc_mode
                sample = []
                dims_log = [d.is_log for d in self.apply_mode]
                dims_ire = [d.is_irregular for d in self.apply_mode]
                for dim, is_log, is_ire in zip(
                    self.data["regularized_output_binning"], dims_log, dims_ire
                ):
                    if is_log and not is_ire:
                        container.representation = "log_events"
                        sample.append(container[dim.name])
                    else:
                        container.representation = "events"
                        sample.append(container[dim.name])

                if self.unweighted:
                    if "astro_weights" in container.keys:
                        weights = np.ones_like(
                            container["weights"] + container["astro_weights"]
                        )
                    else:
                        weights = np.ones_like(container["weights"])
                else:
                    if "astro_weights" in container.keys:
                        weights = container["weights"] + container["astro_weights"]
                    else:
                        weights = container["weights"]
                if self.apply_unc_weights:
                    unc_weights = container["unc_weights"]
                else:
                    unc_weights = np.ones(weights.shape)
                logging.trace("Using 'unc_weights' array %s for '%s'",
                              unc_weights, container.name)

                full_weights = unc_weights * weights
                # The hist is now computed using a binning that is completely linear
                # and regular
                hist = histogram(
                    sample,
                    full_weights,
                    self.data["regularized_output_binning"],
                    averaged=False
                )

                if self.error_method == "sumw2":
                    sumw2 = histogram(sample, np.square(full_weights),
                        self.data["regularized_output_binning"], averaged=False)
                    bin_unc2 = histogram(sample, np.square(unc_weights) * weights,
                        self.data["regularized_output_binning"], averaged=False)

                container.representation = self.apply_mode
                container["weights"] = hist
                # Histogramming does not invalidate "events" representation
                container.validity["weights"][hash("events")] = True

                if self.error_method == "sumw2":
                    container["errors"] = np.sqrt(sumw2)
                    container["bin_unc2"] = bin_unc2


def init_test(**param_kwargs):
    """Instantiation example"""
    return hist(calc_mode='events')
