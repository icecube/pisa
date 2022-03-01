"""
Stage to transform arrays with weights into actual `histograms`
that represent event counts
"""

import numpy as np

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.core.translation import histogram
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.utils.profiler import profile, line_profile
from pisa.utils import vectorizer
from pisa.utils.log import logging


class hist(Stage):  # pylint: disable=invalid-name

    """stage to histogram events

    Parameters
    ----------
    unweighted : bool, optional
        Return un-weighted event counts in each bin.
    """
    def __init__(
        self,
        apply_unc_weights=False,
        unweighted=False,
        **std_kwargs,
    ):

        # init base class
        super().__init__(
            expected_params=(),
            **std_kwargs,
        )

        assert self.calc_mode is not None
        assert self.apply_mode is not None
        self.regularized_apply_mode = None
        self.apply_unc_weights = apply_unc_weights
        self.unweighted = unweighted

    def setup_function(self):

        assert isinstance(self.apply_mode, MultiDimBinning), (
            "Hist stage needs a binning as `apply_mode`, but is %s" % self.apply_mode
        )

        if isinstance(self.calc_mode, MultiDimBinning):

            # The two binning must be exclusive
            assert len(set(self.calc_mode.names) & set(self.apply_mode.names)) == 0

            transform_binning = self.calc_mode + self.apply_mode

            # go to "events" mode to create the transforms

            for container in self.data:
                self.data.representation = "events"
                sample = [container[name] for name in transform_binning.names]
                hist = histogram(sample, None, transform_binning, averaged=False)
                transform = hist.reshape(self.calc_mode.shape + (-1,))
                self.data.representation = self.calc_mode
                container["hist_transform"] = transform

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
            self.regularized_apply_mode = MultiDimBinning(dimensions)
            logging.debug(
                "Using regularized binning:\n" + str(self.regularized_apply_mode)
            )
        else:
            raise ValueError(f"unknown calc mode: {self.calc_mode}")

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
                    unc_weights = container["unc_weights"]
                else:
                    unc_weights = np.ones(weights.shape)
                transform = container["hist_transform"]

                hist = (unc_weights*weights) @ transform
                if self.error_method == "sumw2":
                    sumw2 = np.square(unc_weights*weights) @ transform
                    bin_unc2 = (np.square(unc_weights)*weights) @ transform

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
                    self.regularized_apply_mode, dims_log, dims_ire
                ):
                    if is_log and not is_ire:
                        container.representation = "log_events"
                        sample.append(container[dim.name])
                    else:
                        container.representation = "events"
                        sample.append(container[dim.name])

                if self.unweighted:
                    if "astro_weights" in container.keys:
                        weights = np.ones_like(container["weights"] + container["astro_weights"])
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
                
                # The hist is now computed using a binning that is completely linear
                # and regular
                hist = histogram(
                    sample,
                    unc_weights*weights,
                    self.regularized_apply_mode,
                    averaged=False
                )

                if self.error_method == "sumw2":
                    sumw2 = histogram(sample, np.square(unc_weights*weights), self.regularized_apply_mode, averaged=False)
                    bin_unc2 = histogram(sample, np.square(unc_weights)*weights, self.regularized_apply_mode, averaged=False)

                container.representation = self.apply_mode
                container["weights"] = hist

                if self.error_method == "sumw2":
                    container["errors"] = np.sqrt(sumw2)
                    container["bin_unc2"] = bin_unc2
