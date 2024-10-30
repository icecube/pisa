"""
Stage to transform arrays with weights into actual `histograms`
that represent event counts
"""

import numpy as np

from pisa.core.stage import Stage
from pisa.core.translation import histogram
from pisa.core.binning import MultiDimBinning, OneDimBinning, VarMultiDimBinning, EventSpecie
from pisa.utils.profiler import profile
from pisa.utils.log import logging
import re


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
        self.variable_binning = None
        self.apply_unc_weights = apply_unc_weights
        self.unweighted = unweighted

    def setup_function(self):

        assert isinstance(self.apply_mode, (MultiDimBinning, VarMultiDimBinning)), (
            "Hist stage needs a binning as `apply_mode`, but is %s" % self.apply_mode
        )

#         for container in self.data:
#             print('initial', container.keys)

        self.variable_binning = False
        if isinstance(self.apply_mode, VarMultiDimBinning):
            self.variable_binning = True

        if isinstance(self.calc_mode, MultiDimBinning):

            if self.variable_binning:
                raise NotImplementedError(
                    "MultiDimBinning to VarMultiDimBinning conversion is not implemented!"
                )

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

        elif isinstance(self.calc_mode, VarMultiDimBinning):
            raise NotImplementedError(
                "Using VarMultiDimBinning as calc_mode is not implemented!"
            )
#             assert len(set(self.calc_mode.names) & set(self.apply_mode.names)) == 0
#             for calc_mode_sp, apply_mode_sp in zip(self.calc_mode, self.apply_mode):
#                 assert len(set(calc_mode_sp.binning.names) & set(apply_mode_sp.binning.names)) == 0
            # TODO

        elif self.calc_mode == "events":
            # For dimensions where the binning is irregular, we pre-compute the
            # index that each sample falls into and then bin regularly in the index.
            # For dimensions that are logarithmic, we add a linear binning in
            # the logarithm.
            if not self.variable_binning:
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
                # simplify the code by making it a VarMultiDimBinning ...
                # self.regularized_apply_mode = MultiDimBinning(dimensions)
                self.regularized_apply_mode = VarMultiDimBinning([EventSpecie(
                    binning=MultiDimBinning(dimensions))])
                logging.debug(
                    "Using regularized binning:\n" + str(self.regularized_apply_mode)
                )
            else:
                # TODO: deal with repeat code
                print('calculating using var binning!!!')
                species = []
                for specie in self.apply_mode.species:
                    dimensions = []
                    for dim in specie.binning:
                        if dim.is_irregular:
                            varname = dim.name + "__" + specie.name + "_" + "_idx"
                            print(varname)
                            new_dim = OneDimBinning(
                                varname, domain=[0, dim.num_bins], num_bins=dim.num_bins
                            )
                            dimensions.append(new_dim)
                            for container in self.data:
                                container.representation = "events"
                                x = container[dim.name] * dim.units
                                x_idx = np.searchsorted(dim.bin_edges, x, side="right") - 1
                                on_edge = x == dim.bin_edges[-1]
                                x_idx[on_edge] -= 1
                                container[varname] = x_idx
                        elif dim.is_log:
                            new_dim = OneDimBinning(
                                dim.name, domain=np.log(dim.domain.m), num_bins=dim.num_bins
                            )
                            dimensions.append(new_dim)
                        else:
                            dimensions.append(dim)
                    species.append(EventSpecie(name=specie.name, selection=specie.selection,
                                              binning=MultiDimBinning(dimensions)))
                self.regularized_apply_mode = VarMultiDimBinning(species)
                logging.debug(
                    "Using regularized binning:\n" + str(self.regularized_apply_mode)
                )
            
        else:
            raise ValueError(f"unknown calc mode: {self.calc_mode}")

#         for container in self.data:
#             print('setup_final', container.keys)

    @profile
    def apply_function(self):

#         for container in self.data:
#             print('apply_initial', container.keys)

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
#                 print(container.keys)
                container.representation = self.calc_mode
                print(container.keys)
                hists = []
                errors = []
                bin_unc2s = []
                for specie, reg_specie in zip(self.apply_mode.species,
                                              self.regularized_apply_mode):
                    selection = specie.selection
                    if selection is None:
                        sel_mask = np.ones_like(container["weights"])
                    for var_name in container.keys: #TODO add check that masks don't overlap
                        # using word boundary \b to replace whole words only
                        print(type(var_name), var_name)
                        selection = re.sub(
                            r'\b{}\b'.format(var_name),
                            'container["%s"]' % (var_name),
                            selection
                        )
                    print(selection)
                    sel_mask = eval(selection)  # pylint: disable=eval-used
                    
                    sample = []
                    dims_log = [d.is_log for d in specie.binning]
                    dims_ire = [d.is_irregular for d in specie.binning]
                    for dim, is_log, is_ire in zip(
                        reg_specie.binning, dims_log, dims_ire
                    ):
                        
                        if is_log and not is_ire:
                            container.representation = "log_events"
                            sample.append(container[dim.name][sel_mask])
                        else:
                            container.representation = "events"
                            sample.append(container[dim.name][sel_mask])

                    if self.unweighted:
                        if "astro_weights" in container.keys:
                            weights = np.ones_like(
                                container["weights"][sel_mask] + container["astro_weights"][sel_mask]
                            )
                        else:
                            weights = np.ones_like(container["weights"][sel_mask])
                    else:
                        if "astro_weights" in container.keys:
                            weights = container["weights"][sel_mask] + container["astro_weights"][sel_mask]
                        else:
                            weights = container["weights"][sel_mask]
                    if self.apply_unc_weights:
                        unc_weights = container["unc_weights"][sel_mask]
                    else:
                        unc_weights = np.ones(weights.shape)

                    # The hist is now computed using a binning that is completely linear
                    # and regular
                    hist = histogram(
                        sample,
                        (unc_weights*weights),
                        MultiDimBinning(reg_specie.binning),
                        averaged=False
                    )
#                     print(hist)
                    hists.append(hist)

                    if self.error_method == "sumw2":
                        errors.append( np.sqrt(histogram(sample, np.square(unc_weights*weights),
                            reg_specie.binning, averaged=False)) )
                        bin_unc2s.append(histogram(sample, np.square(unc_weights)*weights,
                            reg_specie.binning, averaged=False))

                container.representation = self.apply_mode
                container["weights"] = self.apply_mode, hists
                print(container)

                if self.error_method == "sumw2":
                    container["errors"] = self.apply_mode, errors
                    container["bin_unc2"] = self.apply_mode, bin_unc2s
