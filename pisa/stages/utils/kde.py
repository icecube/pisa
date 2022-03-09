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
from pisa.utils.kde_hist import kde_histogramdd


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
        stack_pid=True,
        stash_hists=False,
        bootstrap=False,
        bootstrap_niter=10,
        **std_kargs,
    ):

        self.bw_method = bw_method
        self.coszen_name = coszen_name
        self.oversample = int(oversample)
        self.coszen_reflection = coszen_reflection
        self.stack_pid = stack_pid
        self.stash_hists = stash_hists
        self.stash_valid = False
        self.bootstrap = bootstrap
        self.bootstrap_niter = bootstrap_niter

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

        assert isinstance(self.apply_mode, MultiDimBinning), (
            f"KDE stage needs a binning as `apply_mode`, but is {self.apply_mode}"
        )

        # For dimensions that are logarithmic, we add a linear binning in
        # the logarithm.
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
                    dim.name,
                    domain=np.log(dim.domain.m),
                    num_bins=dim.num_bins
                )
            dimensions.append(new_dim)

            self.regularized_apply_mode = MultiDimBinning(dimensions)
            logging.debug("Using regularized binning:\n" + repr(self.regularized_apply_mode))

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
                if is_log:
                    container.representation = "log_events"
                    sample.append(container[dim.name])
                else:
                    container.representation = "events"
                    sample.append(container[dim.name])

            sample = np.stack(sample).T
            weights = container["weights"]

            kde_kwargs = dict(
                sample=sample,
                binning=self.regularized_apply_mode,
                weights=weights,
                bw_method=self.bw_method,
                coszen_name=self.coszen_name,
                coszen_reflection=self.coszen_reflection,
                oversample=self.oversample,
                use_cuda=False,
                stack_pid=self.stack_pid,
                bootstrap=self.bootstrap,
                bootstrap_niter=self.bootstrap_niter,
            )

            if self.bootstrap:
                kde_map, kde_errors = kde_histogramdd(**kde_kwargs)
                kde_errors = np.ascontiguousarray(kde_errors.ravel())
            else:
                kde_map = kde_histogramdd(**kde_kwargs)
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
