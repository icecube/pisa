"""
Module to test KDE bootstrapping. This could not be built into the KDE stage's script
itself, because the import of a class named `kde` directly in the main scope overshadows
the `kde` module and causes an import error.
"""

import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from pisa.utils.log import logging, set_verbosity, Levels
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.config_parser import parse_pipeline_config
from collections import OrderedDict

from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.core.param import Param, ParamSet
from pisa.core.prior import Prior
from pisa import ureg


__all__ = ["test_kde_bootstrapping", "test_kde_stash"]

__author__ = "A. Trettin"

__license__ = """Copyright (c) 2014-2022, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


PARAM_DEFAULTS = {"prior": None, "range": None, "is_fixed": True}
"""Defaults for stage parameters."""

TEST_BINNING = MultiDimBinning(
    [
        OneDimBinning(
            name="true_energy",
            is_log=True,
            num_bins=15,
            domain=[10, 100] * ureg.GeV,
            tex=r"E_{\rm true}",
        ),
        OneDimBinning(
            name="true_coszen",
            is_log=False,
            num_bins=16,
            domain=[-1, 0] * ureg.dimensionless,
            tex=r"\cos(\theta_z)",
        ),
    ]
)
"""Binning to use for test pipeline."""


class TEST_CONFIGS(object):
    """Default configurations for stages in a minimal test pipeline."""

    pipe_cfg = OrderedDict(
        pipeline={
            "name": "muons",
            "output_binning": TEST_BINNING,
            "output_key": ("weights"),
            "detector_name": None,
        }
    )
    event_generator_cfg = {
        "calc_mode": "events",
        "apply_mode": "events",
        "output_names": ["muon"],
        "params": ParamSet(
            [
                Param(name="n_events", value=1e3, **PARAM_DEFAULTS),
                Param(name="seed", value=0, **PARAM_DEFAULTS),
                Param(name="random", value=False, **PARAM_DEFAULTS),
            ]
        ),
    }
    aeff_cfg = {
        "calc_mode": "events",
        "apply_mode": "events",
        "params": ParamSet(
            [
                Param(name="livetime", value=12345 * ureg["seconds"], **PARAM_DEFAULTS),
                Param(name="weight_scale", value=1.0, **PARAM_DEFAULTS),
            ]
        ),
    }
    set_variance_cfg = {
        "calc_mode": TEST_BINNING,
        "apply_mode": TEST_BINNING,
        "divide_total_mc": True,
        # expected number of unweighted MC events including events that fall outside of
        # the analysis binning
        "expected_total_mc": 1000,
        "variance_scale": 0.1,
    }
    fix_error_cfg = {
        "calc_mode": TEST_BINNING,
        "apply_mode": TEST_BINNING,
    }
    kde_cfg = {
        "calc_mode": "events",
        "apply_mode": TEST_BINNING,
        "bootstrap": False,
        "bootstrap_seed": 0,
        "bootstrap_niter": 6,
        "linearize_log_dims": True,
        "stash_hists": False,
        "coszen_name": "true_coszen",
        "stack_pid": False,
        "oversample": 1,
    }


def test_kde_bootstrapping(verbosity=Levels.WARN):
    """Unit test for the kde stage."""

    set_verbosity(verbosity)

    test_cfg = deepcopy(TEST_CONFIGS.pipe_cfg)
    test_cfg[("data", "toy_event_generator")] = deepcopy(
        TEST_CONFIGS.event_generator_cfg
    )
    test_cfg[("aeff", "weight")] = deepcopy(TEST_CONFIGS.aeff_cfg)
    test_cfg[("utils", "kde")] = deepcopy(TEST_CONFIGS.kde_cfg)

    # get map, but without the linearization
    test_cfg[("utils", "kde")]["linearize_log_dims"] = False
    dmaker = DistributionMaker([test_cfg])
    map_baseline_no_linearization = dmaker.get_outputs(return_sum=True)[0]

    # get a baseline (with linearization, which we will use from here on out)
    test_cfg[("utils", "kde")]["linearize_log_dims"] = True
    dmaker = DistributionMaker([test_cfg])
    map_baseline = dmaker.get_outputs(return_sum=True)[0]
    logging.debug(f"Baseline KDE'd map:\n{map_baseline}")

    # assert that linearization make a difference at all
    total_no_lin = np.sum(map_baseline_no_linearization.nominal_values)
    total_with_lin = np.sum(map_baseline.nominal_values)
    assert not (total_no_lin == total_with_lin)
    # but also that the difference isn't huge (< 5% difference in total bin count)
    # --> This will fail if one forgets to *not* take the log when linearization
    #     is turned off, for example. In that case, most bins will be empty, because
    #     the binning would be lin while the KDE would be log.
    assert np.abs(total_no_lin / total_with_lin - 1.0) < 0.05
    # Make sure that different seeds produce different maps, and that the same seed will
    # produce the same map.
    # We enable bootstrapping now, without re-loading everything, to save time.
    dmaker.pipelines[0].output_key = ("weights", "errors")
    dmaker.pipelines[0].stages[-1].bootstrap = True

    map_seed0 = dmaker.get_outputs(return_sum=True)[0]
    dmaker.pipelines[0].stages[-1].bootstrap_seed = 1
    map_seed1 = dmaker.get_outputs(return_sum=True)[0]

    logging.debug(f"Map with seed 0 is:\n{map_seed0}")
    logging.debug(f"Map with seed 1 is:\n{map_seed1}")

    assert not map_seed0 == map_seed1

    dmaker.pipelines[0].stages[-1].bootstrap_seed = 0
    map_seed0_reprod = dmaker.get_outputs(return_sum=True)[0]

    assert map_seed0 == map_seed0_reprod

    logging.info("<< PASS : kde_bootstrapping >>")


def test_kde_stash(verbosity=Levels.WARN):
    """Unit test for the hist stashing feature.

    Hist stashing can greatly speed up fits as long as the only free parameters
    are in stages that work on the output histograms, rather than the individual
    events. In particular, it should be strictly equivalent to either scale all weights
    by a factor and then running the KDE, or to first calculate the KDE and then scale
    all the bin counts by the same factor. This test ensures that the order of
    operation really doesn't matter.

    This should apply also to the errors, independent of whether the bootstrapping
    method or the utils.set_variance stage was used to produce them.
    """

    import pytest
    from numpy.testing import assert_array_equal, assert_allclose

    set_verbosity(verbosity)

    def assert_correct_scaling(pipeline_cfg, fixed_errors=False):
        """Run the pipeline and assert that scaling by a factor of two is correct."""
        dmaker = DistributionMaker([pipeline_cfg])
        out = dmaker.get_outputs(return_sum="true")[0]
        dmaker.pipelines[0].params.weight_scale = 2.0
        out2 = dmaker.get_outputs(return_sum="true")[0]
        if fixed_errors:
            # this is special: We expect that the nominal counts are multiplied, but
            # that hte errors stay fixed (applies to set_variance errors)
            assert_array_equal(out.nominal_values * 2.0, out2.nominal_values)
            assert_array_equal(out.std_devs, out2.std_devs)
        else:
            assert out * 2.0 == out2

    ## KDE without errors

    # First aeff, then KDE
    test_cfg = deepcopy(TEST_CONFIGS.pipe_cfg)
    test_cfg[("data", "toy_event_generator")] = deepcopy(
        TEST_CONFIGS.event_generator_cfg
    )
    test_cfg[("aeff", "weight")] = deepcopy(TEST_CONFIGS.aeff_cfg)
    test_cfg[("utils", "kde")] = deepcopy(TEST_CONFIGS.kde_cfg)
    assert_correct_scaling(test_cfg)

    # First KDE, then aeff, with stashing
    test_cfg = deepcopy(TEST_CONFIGS.pipe_cfg)
    test_cfg[("data", "toy_event_generator")] = deepcopy(
        TEST_CONFIGS.event_generator_cfg
    )
    test_cfg[("utils", "kde")] = deepcopy(TEST_CONFIGS.kde_cfg)
    test_cfg[("aeff", "weight")] = deepcopy(TEST_CONFIGS.aeff_cfg)
    # turn on stashing
    test_cfg[("utils", "kde")]["stash_hists"] = True
    # Change aeff calculation to binned mode (i.e. multiply bin counts)
    test_cfg[("aeff", "weight")]["calc_mode"] = TEST_BINNING
    test_cfg[("aeff", "weight")]["apply_mode"] = TEST_BINNING
    assert_correct_scaling(test_cfg)

    ## KDE with bootstrap errors

    # First aeff, then KDE with bootstrap
    test_cfg = deepcopy(TEST_CONFIGS.pipe_cfg)
    test_cfg[("data", "toy_event_generator")] = deepcopy(
        TEST_CONFIGS.event_generator_cfg
    )
    test_cfg[("aeff", "weight")] = deepcopy(TEST_CONFIGS.aeff_cfg)
    test_cfg[("utils", "kde")] = deepcopy(TEST_CONFIGS.kde_cfg)
    # turn OFF stashing
    test_cfg[("utils", "kde")]["stash_hists"] = False
    # turn on bootstrapping
    test_cfg[("utils", "kde")]["bootstrap"] = True
    # return the errors
    test_cfg["pipeline"]["output_key"] = ("weights", "errors")
    assert_correct_scaling(test_cfg)

    # First KDE with stashed hists and bootstrap, then aeff
    test_cfg = deepcopy(TEST_CONFIGS.pipe_cfg)
    test_cfg[("data", "toy_event_generator")] = deepcopy(
        TEST_CONFIGS.event_generator_cfg
    )
    test_cfg[("utils", "kde")] = deepcopy(TEST_CONFIGS.kde_cfg)
    test_cfg[("aeff", "weight")] = deepcopy(TEST_CONFIGS.aeff_cfg)
    # turn on stashing
    test_cfg[("utils", "kde")]["stash_hists"] = True
    # turn on bootstrapping
    test_cfg[("utils", "kde")]["bootstrap"] = True
    # return the errors
    test_cfg["pipeline"]["output_key"] = ("weights", "errors")
    # need to change mode to binned
    test_cfg[("aeff", "weight")]["calc_mode"] = TEST_BINNING
    test_cfg[("aeff", "weight")]["apply_mode"] = TEST_BINNING
    assert_correct_scaling(test_cfg)

    ## KDE with errors calculated using set_variance stage

    # first aeff, then KDE and set_variance
    test_cfg = deepcopy(TEST_CONFIGS.pipe_cfg)
    test_cfg[("data", "toy_event_generator")] = deepcopy(
        TEST_CONFIGS.event_generator_cfg
    )
    test_cfg[("aeff", "weight")] = deepcopy(TEST_CONFIGS.aeff_cfg)
    test_cfg[("utils", "kde")] = deepcopy(TEST_CONFIGS.kde_cfg)
    test_cfg[("utils", "set_variance")] = deepcopy(TEST_CONFIGS.set_variance_cfg)
    # turn on stashing
    test_cfg[("utils", "kde")]["stash_hists"] = False
    # turn OFF bootstrapping
    test_cfg[("utils", "kde")]["bootstrap"] = False
    # return the errors
    test_cfg["pipeline"]["output_key"] = ("weights", "errors")
    # The set_variance stage only calculates errors the first time that the pipeline
    # is evaluated, these errors are stored and re-instated on any sub-sequent
    # evaluations. We expect therefore that only the nominal values scale.
    assert_correct_scaling(test_cfg, fixed_errors=True)

    # first KDE and set_variance, then aeff
    test_cfg = deepcopy(TEST_CONFIGS.pipe_cfg)
    test_cfg[("data", "toy_event_generator")] = deepcopy(
        TEST_CONFIGS.event_generator_cfg
    )
    test_cfg[("utils", "kde")] = deepcopy(TEST_CONFIGS.kde_cfg)
    test_cfg[("aeff", "weight")] = deepcopy(TEST_CONFIGS.aeff_cfg)
    # It is still important that the `set_variance` stage is *last*.
    test_cfg[("utils", "set_variance")] = deepcopy(TEST_CONFIGS.set_variance_cfg)
    # turn on stashing
    test_cfg[("utils", "kde")]["stash_hists"] = True
    # turn OFF bootstrapping
    test_cfg[("utils", "kde")]["bootstrap"] = False
    # return the errors
    test_cfg["pipeline"]["output_key"] = ("weights", "errors")
    # need to change mode to binned
    test_cfg[("aeff", "weight")]["calc_mode"] = TEST_BINNING
    test_cfg[("aeff", "weight")]["apply_mode"] = TEST_BINNING
    # We ensure that the behavior is the same as it has been when we were not stashing
    # the histograms and used set_variance.
    assert_correct_scaling(test_cfg, fixed_errors=True)

    # Using the wrong order (not putting set_variance last)
    test_cfg = deepcopy(TEST_CONFIGS.pipe_cfg)
    test_cfg[("data", "toy_event_generator")] = deepcopy(
        TEST_CONFIGS.event_generator_cfg
    )
    test_cfg[("utils", "kde")] = deepcopy(TEST_CONFIGS.kde_cfg)
    # If set_variance is not the last stage, this breaks. The reason is a slightly
    # silly design of set_variance. It should have been constructed such that the
    # total normalization is always divided out, but it wasn't. The way it is
    # constructed now, it is basically tuned by the scaling factor to work for the
    # given livetime and breaks immediately when that changes.
    test_cfg[("utils", "set_variance")] = deepcopy(TEST_CONFIGS.set_variance_cfg)
    test_cfg[("aeff", "weight")] = deepcopy(TEST_CONFIGS.aeff_cfg)
    # turn on stashing
    test_cfg[("utils", "kde")]["stash_hists"] = True
    # turn OFF bootstrapping
    test_cfg[("utils", "kde")]["bootstrap"] = False
    # return the errors
    test_cfg["pipeline"]["output_key"] = ("weights", "errors")
    # need to change mode to binned
    test_cfg[("aeff", "weight")]["calc_mode"] = TEST_BINNING
    test_cfg[("aeff", "weight")]["apply_mode"] = TEST_BINNING
    # With the wrong order, this will fail.
    # FIXME: If someone changes the behavior of set_variance in the future to be
    # more robust, they are welcome to change this unit test.
    with pytest.raises(AssertionError):
        assert_correct_scaling(test_cfg, fixed_errors=True)

    logging.info("<< PASS : kde_stash >>")


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-v", action="count", default=Levels.WARN, help="set verbosity level"
    )
    args = parser.parse_args()
    return args


def main():
    """Script interface to test_kde_bootstrapping"""

    args = parse_args()
    kwargs = vars(args)
    kwargs["verbosity"] = kwargs.pop("v")

    test_kde_bootstrapping(**kwargs)
    test_kde_stash(**kwargs)


if __name__ == "__main__":
    main()
