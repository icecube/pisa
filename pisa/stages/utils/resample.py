"""
Stage to transform binned data from one binning to another while also dealing with
uncertainty estimates in a reasonable way. In particular, this allows up-sampling from a
more coarse binning to a finer binning.

The implementation is similar to that of the hist stage, hence the over-writing of
the `apply` method.
"""

from __future__ import absolute_import, print_function, division
from enum import Enum, auto

import numpy as np

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.utils.profiler import profile
from pisa.utils import vectorizer
from pisa.core import translation
from pisa.core.binning import MultiDimBinning
from pisa.utils.log import logging, set_verbosity


__all__ = ['ResampleMode', 'resample', 'init_test', 'test_resample']


class ResampleMode(Enum):
    """Enumerates sampling methods of the `resample` stage."""

    UP = auto()
    DOWN = auto()
    ARB = auto()

class resample(Stage):  # pylint: disable=invalid-name
    """
    Stage to resample weighted MC histograms from one binning to another.

    The origin binning is given as `calc_mode` and the output binning is given in
    `apply_mode`.

    Parameters
    ----------

    scale_errors : bool, optional
        If `True` (default), apply scaling to errors.
    """

    def __init__(
        self,
        scale_errors=True,
        **std_kwargs,
    ):

        expected_container_keys = [
            'weights',
        ]
        if scale_errors:
            expected_container_keys.append('errors')

        # init base class
        super().__init__(
            expected_params=(),
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )

        # This stage only makes sense when going binned to binned.
        assert isinstance(self.apply_mode, MultiDimBinning), "stage only produces binned output"
        assert isinstance(self.calc_mode, MultiDimBinning), "stage only produces binned output"

        self.scale_errors = scale_errors

        # The following tests whether `apply_mode` is a strict up-sample

        # TODO: Test for ability to resample in two steps
        # TODO: Update to new test nomenclature

        # ToDo: check logic after pisa4 upgrade
        if self.calc_mode.is_compat(self.apply_mode):
            self.rs_mode = ResampleMode.UP
        elif self.apply_mode.is_compat(self.calc_mode):
            self.rs_mode = ResampleMode.DOWN
        else:
            raise ValueError("Binnings are not compatible with each other for resample")

        # TODO: Implement downsampling
        # TODO: Implement arbitrary resampling
        if self.rs_mode == ResampleMode.DOWN:
            raise NotImplementedError("Downsampling not yet implemented.")
        if self.rs_mode == ResampleMode.ARB:
            raise NotImplementedError("Arbitrary resampling not yet implemented.")

    def setup_function(self):
        # Set up a container for intermediate storage of variances in input specs
        self.data.representation = self.calc_mode
        for container in self.data:
            container["variances"] = np.empty((container.size), dtype=FTYPE)
        # set up containers for the resampled output in the output specs
        self.data.representation = self.apply_mode
        for container in self.data:
            container["weights_resampled"] = np.empty((container.size), dtype=FTYPE)
            if self.scale_errors:
                container["vars_resampled"] = np.empty((container.size), dtype=FTYPE)
                container["errors_resampled"] = np.empty((container.size), dtype=FTYPE)

    @profile
    def apply(self):
        # DO NOT USE THIS STAGE AS YOUR TEMPLATE IF YOU ARE NEW TO PISA!
        # --------------------------------------------------------------
        #
        # We are overwriting the `apply` method rather than the `apply_function` method
        # because we are manipulating the data binning in a delicate way that doesn't
        # work with automatic rebinning.

        self.data.representation = self.calc_mode

        if self.scale_errors:
            for container in self.data:
                vectorizer.pow(
                    vals=container["errors"],
                    pwr=2,
                    out=container["variances"],
                )

        input_binvols = self.calc_mode.weighted_bin_volumes(attach_units=False).ravel()
        output_binvols = self.apply_mode.weighted_bin_volumes(attach_units=False).ravel()

        for container in self.data:

            self.data.representation = self.calc_mode
            weights_flat_hist = container["weights"]
            if self.scale_errors:
                vars_flat_hist = container["variances"]
            self.data.representation = self.apply_mode

            if self.rs_mode == ResampleMode.UP:
                # The `unroll_binning` function returns the midpoints of the bins in the
                # dimension `name`.
                fine_gridpoints = [
                    container.unroll_binning(name, self.apply_mode)
                    for name in self.apply_mode.names
                ]
                # We look up at which bin index of the input binning the midpoints of
                # the output binning can be found, and assign to each the content of the
                # bin of that index.
                container["weights_resampled"] = translation.lookup(
                    fine_gridpoints,
                    weights_flat_hist,
                    self.calc_mode
                )
                if self.scale_errors:
                    container["vars_resampled"] = translation.lookup(
                        fine_gridpoints,
                        vars_flat_hist,
                        self.calc_mode
                    )
                # These are the volumes of the bins we sample *from*
                origin_binvols = translation.lookup(
                    fine_gridpoints,
                    input_binvols,
                    self.calc_mode
                )
                # Finally, we scale the weights and variances by the ratio of the
                # bin volumes in place:
                container["weights_resampled"] = (
                    container["weights_resampled"] * output_binvols / origin_binvols
                )
                if self.scale_errors:
                    container["vars_resampled"] = (
                        container["vars_resampled"] * output_binvols / origin_binvols
                    )
            elif self.rs_mode == ResampleMode.DOWN:
                pass  # not yet implemented

            if self.scale_errors:
                container["errors_resampled"] = np.sqrt(container["vars_resampled"])

def init_test(**param_kwargs):
    """Initialisation example"""
    return resample() #FIXME


def test_resample():
    """Unit test for the resampling stage."""
    from pisa.core.distribution_maker import DistributionMaker
    from pisa.core.map import Map
    from pisa.utils.config_parser import parse_pipeline_config
    from pisa.utils.comparisons import ALLCLOSE_KW
    from collections import OrderedDict
    from copy import deepcopy
    from numpy.testing import assert_allclose

    example_cfg = parse_pipeline_config('settings/pipeline/example.cfg')
    reco_binning = example_cfg[('utils', 'hist')]['apply_mode']
    coarse_binning = reco_binning.downsample(reco_energy=2, reco_coszen=2)
    assert coarse_binning.is_compat(reco_binning)
    # replace binning of output with coarse binning
    example_cfg[('utils', 'hist')]['apply_mode'] = coarse_binning
    # New in PISA4: We explicitly tell the pipeline which keys and binning to use for
    # the output. We must manually set this to the same binning as the output from the
    # hist stage because otherwise it would attempt to automatically rebin everything.
    example_cfg['pipeline']['output_key'] = ('weights', 'errors')
    example_cfg['pipeline']['output_binning'] = coarse_binning
    # make another pipeline with an upsampling stage to the original binning
    upsample_cfg = deepcopy(example_cfg)
    resample_cfg = OrderedDict()
    resample_cfg['apply_mode'] = reco_binning
    resample_cfg['calc_mode'] = coarse_binning
    resample_cfg['scale_errors'] = True
    upsample_cfg[('utils', 'resample')] = resample_cfg
    # Here we want to take the resampled output to generate Maps from the pipeline
    upsample_cfg['pipeline']['output_key'] = ('weights_resampled', 'errors_resampled')
    upsample_cfg['pipeline']['output_binning'] = reco_binning

    example_maker = DistributionMaker([example_cfg])
    upsampled_maker = DistributionMaker([upsample_cfg])

    example_map = example_maker.get_outputs(return_sum=True)[0]
    example_map_upsampled = upsampled_maker.get_outputs(return_sum=True)[0]

    # First check: The upsampled map must have the same total count as the original map
    assert np.isclose(
        np.sum(example_map.nominal_values),
        np.sum(example_map_upsampled.nominal_values),
    )

    # Check consistency of modified chi-square
    # ----------------------------------------
    # When the assumption holds that events are uniformly distributed over the coarse
    # bins, the modified chi-square should not change from upscaling the maps. We test
    # this by making a fluctuated coarse map and then upsampling that map according to
    # the assumption by bin volumes. We should find that the modified chi-square between
    # the coarse map and the coarse fluctuated map is the same as the upsampled map and
    # the upsampled fluctuated map.

    # It doesn't matter precisely how we fluctuate it here, we just want any different
    # map...
    random_map_coarse = example_map.fluctuate(method='scaled_poisson', random_state=42)
    random_map_coarse.set_errors(None)

    # This bit is an entirely independent implementation of the upsampling. The count
    # in every bin is scaled according to the reatio of weighted bin volumes.
    upsampled_hist = np.zeros_like(example_map_upsampled.nominal_values)
    upsampled_errs = np.zeros_like(example_map_upsampled.nominal_values)
    up_binning = example_map_upsampled.binning

    coarse_hist = np.array(random_map_coarse.nominal_values)
    coarse_errors = np.array(random_map_coarse.std_devs)
    coarse_binning = random_map_coarse.binning

    for bin_idx in np.ndindex(upsampled_hist.shape):
        one_bin = up_binning[bin_idx]
        fine_bin_volume = one_bin.weighted_bin_volumes(
            attach_units=False,
        ).squeeze().item()
        # the following is basically an independent implementation of translate.lookup
        coarse_index = []  # index where the upsampled bin came from
        for dim in up_binning.names:
            x = one_bin[dim].weighted_centers[0].m  # middle point of the one bin
            bins = coarse_binning[dim].bin_edges.m  # coarse bin edges in that dim
            coarse_index.append(np.digitize(x, bins) - 1)  # index 1 means bin 0
        coarse_index = tuple(coarse_index)
        coarse_bin_volume = coarse_binning.weighted_bin_volumes(
            attach_units=False,
        )[coarse_index].squeeze().item()

        upsampled_hist[bin_idx] = coarse_hist[coarse_index]
        upsampled_hist[bin_idx] *= fine_bin_volume
        upsampled_hist[bin_idx] /= coarse_bin_volume

    # done, at last!
    random_map_upsampled = Map(
        name="random_upsampled",
        hist=upsampled_hist,
        binning=up_binning
    )
    random_map_upsampled.set_errors(None)

    # After ALL THIS, we get the same modified chi-square from the coarse and the
    # upsampled pair of maps. Neat, huh?
    assert_allclose(random_map_coarse.mod_chi2(example_map),
                    random_map_upsampled.mod_chi2(example_map_upsampled),
                    **ALLCLOSE_KW)
    logging.info('<< PASS : resample >>')

if __name__ == "__main__":
    set_verbosity(2)
    test_resample()
