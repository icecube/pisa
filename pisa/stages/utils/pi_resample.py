"""
Stage to transform binned data from one binning to another while also dealing with
uncertainty estimates in a reasonable way. In particular, this allows up-sampling from a
more coarse binning to a finer binning.

The implementation is similar to that of the pi_hist stage, hence the over-writing of
the `apply` method.
"""

from __future__ import absolute_import, print_function, division

import numpy as np
from enum import Enum, auto

from pisa import FTYPE
from pisa.core.pi_stage import PiStage
from pisa.utils.profiler import profile
from pisa.utils import vectorizer
from pisa.core import translation
from numba import SmartArray

from pisa.utils.log import logging, set_verbosity

class ResampleMode(Enum):
    """Enumerates sampling methods of the `pi_resample` stage."""

    UP = auto()
    DOWN = auto()
    ARB = auto()

class pi_resample(PiStage):  # pylint: disable=invalid-name
    """
    Stage to resample weighted MC histograms from one binning to another.
    
    Parameters
    ----------
    
    scale_errors : bool, optional
        If `True` (default), apply scaling to errors.
    """

    def __init__(
        self,
        scale_errors=True,
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        error_method=None,
        input_specs=None,
        calc_specs=None,
        output_specs=None,
    ):

        expected_params = ()
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_apply_keys = ("weights",)

        # what are keys added or altered in the calculation used during apply
        assert calc_specs is None
        if scale_errors:
            output_apply_keys = ("weights_resampled", "errors_resampled")
        else:
            output_apply_keys = ("weights_resampled",)
        
        map_output_key = "weights_resampled"
        map_output_error_key = "errors_resampled"
        
        # init base class
        super().__init__(
            data=data,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            error_method=error_method,
            input_specs=input_specs,
            calc_specs=calc_specs,
            output_specs=output_specs,
            input_apply_keys=input_apply_keys,
            output_apply_keys=output_apply_keys,
            map_output_key=map_output_key,
            map_output_error_key=map_output_error_key,
        )

        # This stage only makes sense when going binned to binned.
        assert self.input_mode == "binned", "stage only takes binned input"
        assert self.output_mode == "binned", "stage only produces binned output"
        
        self.scale_errors = scale_errors
        
        # The following tests whether `output_specs` is a strict up-sample
        # from `input_specs`, i.e. the bin edges of `output_specs` are a superset
        # of the bin edges of `input_specs`.
        
        # TODO: Test for ability to resample in two steps
        # TODO: Update to new test nomenclature
        if input_specs.is_compat(output_specs):
            self.rs_mode = ResampleMode.UP
        elif output_specs.is_compat(input_specs):
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
        self.data.data_specs = self.input_specs
        for container in self.data:
            container["variances"] = np.empty((container.size), dtype=FTYPE)
        # set up containers for the resampled output in the output specs
        self.data.data_specs = self.output_specs
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

        self.data.data_specs = self.input_specs
        if self.scale_errors:
            for container in self.data:
                vectorizer.pow(
                    vals=container["errors"],
                    pwr=2,
                    out=container["variances"],
                )

        input_binvols = SmartArray(
            self.input_specs.weighted_bin_volumes(attach_units=False).ravel()
        )
        output_binvols = SmartArray(
            self.output_specs.weighted_bin_volumes(attach_units=False).ravel()
        )
        
        for container in self.data:
            self.data.data_specs = self.input_specs
            # we want these to be SmartArrays, so no `.get(WHERE)`
            weights_flat_hist = container["weights"]
            if self.scale_errors:
                vars_flat_hist = container["variances"]
            self.data.data_specs = self.output_specs
            if self.rs_mode == ResampleMode.UP:
                # The `unroll_binning` function returns the midpoints of the bins in the
                # dimension `name`.
                fine_gridpoints = [
                    SmartArray(container.unroll_binning(name, self.output_specs))
                    for name in self.output_specs.names
                ]
                # We look up at which bin index of the input binning the midpoints of
                # the output binning can be found, and assign to each the content of the
                # bin of that index.
                container["weights_resampled"] = translation.lookup(
                    fine_gridpoints,
                    weights_flat_hist,
                    self.input_specs,
                )
                if self.scale_errors:
                    container["vars_resampled"] = translation.lookup(
                        fine_gridpoints,
                        vars_flat_hist,
                        self.input_specs,
                    )
                # These are the volumes of the bins we sample *from*
                origin_binvols = translation.lookup(
                    fine_gridpoints,
                    input_binvols,
                    self.input_specs,
                )
                # Finally, we scale the weights and variances by the ratio of the
                # bin volumes in place:
                vectorizer.imul(output_binvols, container["weights_resampled"])
                vectorizer.itruediv(origin_binvols, container["weights_resampled"])
                if self.scale_errors:
                    vectorizer.imul(output_binvols, container["vars_resampled"])
                    vectorizer.itruediv(origin_binvols, container["vars_resampled"])
            elif self.rs_mode == ResampleMode.DOWN:
                pass  # not yet implemented

            if self.scale_errors:
                vectorizer.sqrt(
                    vals=container["vars_resampled"], out=container["errors_resampled"]
                )
    
    
def test_pi_resample():
    """Unit test for the resampling stage."""
    from pisa.core.distribution_maker import DistributionMaker
    from pisa.core.map import Map
    from pisa.utils.config_parser import parse_pipeline_config
    from pisa.utils.log import set_verbosity, logging
    from pisa.utils.comparisons import ALLCLOSE_KW
    from collections import OrderedDict
    from copy import deepcopy
    
    example_cfg = parse_pipeline_config('settings/pipeline/example.cfg')
    reco_binning = example_cfg[('utils', 'pi_hist')]['output_specs']
    coarse_binning = reco_binning.downsample(reco_energy=2, reco_coszen=2)
    assert coarse_binning.is_compat(reco_binning)
    
    # replace binning of output with coarse binning
    example_cfg[('utils', 'pi_hist')]['output_specs'] = coarse_binning
    # make another pipeline with an upsampling stage to the original binning
    upsample_cfg = deepcopy(example_cfg)
    pi_resample_cfg = OrderedDict()
    pi_resample_cfg['input_specs'] = coarse_binning
    pi_resample_cfg['output_specs'] = reco_binning
    upsample_cfg[('utils', 'pi_resample')] = pi_resample_cfg

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
    assert np.allclose(random_map_coarse.mod_chi2(example_map),
                       random_map_upsampled.mod_chi2(example_map_upsampled),
                       **ALLCLOSE_KW,
                      )
    logging.info('<< PASS : pi_resample >>')

if __name__ == "__main__":
    set_verbosity(2)
    test_pi_resample()
