"""
Stage to transform arrays with weights into KDE maps
that represent event counts
"""
import numpy as np


from copy import deepcopy
from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils import vectorizer
from pisa.utils import kde_hist


class kde(Stage):
    """stage to KDE-map events

    Parameters
    ----------

    bw_method: string
        'scott' or 'silverman' (see kde module)
    coszen_name : string
        Binning name to identify the coszen bin that needs to undergo special
        treatment for reflection
    oversample : int
        Evaluate KDE at more points per bin, takes longer, but is more accurate
    stash_hists : bool
        Evaluate KDE only once and stash the result. This effectively ignores all changes
        from earlier stages, but greatly increases speed. Useful for muons where
        only over-all weight and detector systematic variations matter, which can both
        be applied on the histograms after this stage.
    bootstrap : bool
        Use the bootstrapping technique to estimate errors on the KDE histograms.
    linearize_log_dims : bool
        If True (default), calculate the KDE for a dimension that is binned
        logarithmically on the logarithm of the sample values. This generally results
        in better agreement of the total normalization of the KDE'd histograms to the
        sum of weights.

    Notes
    -----

    Make sure enough events are present with reco energy below and above the
    binning range, otherwise events will only "bleed out"

    """

    def __init__(
        self,
        bw_method="silverman",
        coszen_name="reco_coszen",
        oversample=10,
        coszen_reflection=0.25,
        alpha=0.1,
        stack_pid=True,
        stash_hists=False,
        bootstrap=False,
        bootstrap_niter=10,
        bootstrap_seed=None,
        linearize_log_dims=True,
        **std_kargs,
    ):

        self.bw_method = bw_method
        self.coszen_name = coszen_name
        self.oversample = int(oversample)
        self.coszen_reflection = float(coszen_reflection)
        self.alpha = float(alpha)
        self.stack_pid = stack_pid
        self.stash_hists = stash_hists
        self.stash_valid = False
        self.linearize_log_dims = linearize_log_dims
        self.bootstrap = bootstrap
        self.bootstrap_niter = int(bootstrap_niter)
        if bootstrap_seed is not None:
            self.bootstrap_seed = int(bootstrap_seed)
        else:
            self.bootstrap_seed = None

        if stash_hists:
            self.stashed_hists = None
            if self.bootstrap:
                self.stashed_errors = None

        # init base class
        super().__init__(
            expected_params=(),
            **std_kargs,
        )

        assert self.calc_mode == "events"
        self.regularized_apply_mode = None

    def setup_function(self):

        assert isinstance(
            self.apply_mode, MultiDimBinning
        ), f"KDE stage needs a binning as `apply_mode`, but is {self.apply_mode}"

        # For dimensions that are logarithmic, we add a linear binning in
        # the logarithm (but only if this feature is enabled)

        if not self.linearize_log_dims:
            self.regularized_apply_mode = self.apply_mode
            return

        dimensions = []
        for dim in self.apply_mode:
            if dim.is_lin:
                new_dim = deepcopy(dim)
            # We don't compute the log of the variable just yet, this
            # will be done later during `apply_function` using the
            # representation mechanism.
            # We replace the logarithmic binning with a linear binning in log-space
            elif dim.is_irregular:
                new_dim = OneDimBinning(
                    dim.name,
                    bin_edges=np.log(dim.bin_edges.m),
                )
            else:
                new_dim = OneDimBinning(
                    dim.name, domain=np.log(dim.domain.m), num_bins=dim.num_bins
                )
            dimensions.append(new_dim)

            self.regularized_apply_mode = MultiDimBinning(dimensions)
            logging.debug(
                "Using regularized binning:\n" + repr(self.regularized_apply_mode)
            )

    @profile
    def apply(self):
        # this is special, we want the actual event weights in the kde
        # therefor we're overwritting the apply function
        # normally in a stage you would implement the `apply_function` method
        # and not the `apply` method!

        for container in self.data:

            if self.stash_valid:
                self.data.representation = self.apply_mode
                # Making a copy of the stash so that subsequent stages will not manipulate
                # it.
                container["weights"] = self.stashed_hists[container.name].copy()
                if self.bootstrap:
                    container["errors"] = self.stashed_errors[container.name].copy()
                continue

            sample = []
            dims_log = [d.is_log for d in self.apply_mode]
            for dim, is_log in zip(self.regularized_apply_mode, dims_log):
                if is_log and self.linearize_log_dims:
                    container.representation = "log_events"
                    sample.append(container[dim.name])
                else:
                    container.representation = "events"
                    sample.append(container[dim.name])

            # Make sure that we revert back to "events" before extracting weights (could
            # otherwise end up in "log_events").
            container.representation = "events"

            sample = np.stack(sample).T
            weights = container["weights"]

            kde_kwargs = dict(
                sample=sample,
                binning=self.regularized_apply_mode,
                # weights=weights,
                bw_method=self.bw_method,
                coszen_name=self.coszen_name,
                coszen_reflection=self.coszen_reflection,
                alpha=self.alpha,
                oversample=self.oversample,
                use_cuda=False,
                stack_pid=self.stack_pid,
                # bootstrap=self.bootstrap,
                # bootstrap_niter=self.bootstrap_niter,
            )

            if self.bootstrap:
                from numpy.random import default_rng

                kde_maps = []
                rng = default_rng(self.bootstrap_seed)
                sample_size = container.size
                for i in range(self.bootstrap_niter):
                    # Indices of events are randomly chosen from the entire sample until
                    # we have a new sample of the same size.
                    # If we are stacking in PID, we will want to do this independently
                    # for each PID channel.

                    # We accumulate sample weights into one array.
                    sample_weights = np.zeros(sample_size)

                    if self.stack_pid:
                        binning = self.regularized_apply_mode
                        bin_edges = [b.bin_edges.m for b in binning]
                        pid_bin = binning.names.index("pid")
                        pid_bin_edges = bin_edges[pid_bin]

                        n_ch = len(pid_bin_edges) - 1

                        for pid_channel in range(n_ch):
                            # Get mask of events falling into this PID bin
                            pid_mask = (
                                sample[:, pid_bin] >= pid_bin_edges[pid_channel]
                            ) & (sample[:, pid_bin] < pid_bin_edges[pid_channel + 1])
                            pid_size = np.sum(pid_mask)
                            # Select indices of the appropriate size for just this PID
                            # channel
                            pid_sample_idx = rng.integers(pid_size, size=pid_size)
                            # Instead of manipulating all of the data arrays, we count
                            # how often each index was chosen and take that as a weight,
                            # i.e. an event that was selected twice will have a weight
                            # of 2.
                            pid_sample_weights = np.bincount(
                                pid_sample_idx, minlength=pid_size
                            )
                            sample_weights[pid_mask] += pid_sample_weights
                        # Ensure that we indeed conserved the number of events in each
                        # PID channel after all
                        for pid_channel in range(n_ch):
                            pid_mask = (
                                sample[:, pid_bin] >= pid_bin_edges[pid_channel]
                            ) & (sample[:, pid_bin] < pid_bin_edges[pid_channel + 1])
                            assert sum(sample_weights[pid_mask]) == sum(pid_mask)
                    else:
                        sample_idx = rng.integers(sample_size, size=sample_size)
                        # Instead of manipulating all of the data arrays, we count how
                        # often each index was chosen and take that as a weight, i.e. an
                        # event that was selected twice will have a weight of 2.
                        sample_weights = np.bincount(sample_idx, minlength=sample_size)

                    with np.errstate(invalid="raise"):
                        try:
                            kde_maps.append(
                                kde_hist.kde_histogramdd(
                                    weights=weights * sample_weights, **kde_kwargs
                                )
                            )
                        except FloatingPointError:
                            raise RuntimeError(
                                "Could not calculate KDE with the given sample. This can "
                                "happen if the bootstrap selects too few distinct events "
                                "in one of the PID channels."
                            )
                kde_maps = np.stack(kde_maps)
                kde_map = np.mean(kde_maps, axis=0)
                kde_errors = np.std(kde_maps, axis=0)
                kde_errors = np.ascontiguousarray(kde_errors.ravel())
            else:
                kde_map = kde_hist.kde_histogramdd(weights=weights, **kde_kwargs)
            kde_map = np.ascontiguousarray(kde_map.ravel())

            self.data.representation = self.apply_mode

            container["weights"] = kde_map
            if self.bootstrap:
                container["errors"] = kde_errors

            if self.stash_hists:
                if self.stashed_hists is None:
                    self.stashed_hists = {}
                    if self.bootstrap:
                        self.stashed_errors = {}
                # Making a copy is required because subsequent stages may change weights
                # in-place.
                self.stashed_hists[container.name] = kde_map.copy()
                if self.bootstrap:
                    self.stashed_errors[container.name] = kde_errors.copy()

        self.stash_valid = (
            self.stash_hists
        )  # valid is true if we are stashing, else not


# Placing a unit test here creates an import error due to the fact that the class
# defined above has the exact same name as the `kde` module that has to be imported to
# make it work. If this script is __main__, then we import `kde` (the stage) directly
# into the main scope and thus overshadow `kde` (the module).
# The unit test for this stage is therefore instead placed in
# pisa/pisa_tests/test_kde_stage.py
