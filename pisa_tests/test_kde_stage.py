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

__all__ = ["test_kde_bootstrapping"]

__author__ = "A. Trettin"

__license__ = """Copyright (c) 2014-2020, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


def test_kde_bootstrapping(verbosity=Levels.WARN):
    """Unit test for the kde stage."""

    set_verbosity(verbosity)

    example_cfg = parse_pipeline_config("settings/pipeline/example.cfg")

    # We have to remove containers with too few events, otherwise the KDE fails simply
    # because too few distinct events are in one of the PID channels after bootstrapping.
    example_cfg[("data", "simple_data_loader")]["output_names"] = [
        "numu_cc",
        "numubar_cc",
    ]

    kde_stage_cfg = OrderedDict()
    kde_stage_cfg["apply_mode"] = example_cfg[("utils", "hist")]["apply_mode"]
    kde_stage_cfg["calc_mode"] = "events"
    kde_stage_cfg["bootstrap"] = False
    kde_stage_cfg["bootstrap_seed"] = 0
    kde_stage_cfg["bootstrap_niter"] = 5

    kde_pipe_cfg = deepcopy(example_cfg)

    # Replace histogram stage with KDE stage
    del kde_pipe_cfg[("utils", "hist")]
    kde_pipe_cfg[("utils", "kde")] = kde_stage_cfg

    # no errors in baseline since there is no bootstrapping enabled
    kde_pipe_cfg["pipeline"]["output_key"] = "weights"

    # get map, but without the linearization
    kde_pipe_cfg[("utils", "kde")]["linearize_log_dims"] = False
    dmaker = DistributionMaker([kde_pipe_cfg])
    map_baseline_no_linearization = dmaker.get_outputs(return_sum=True)[0]

    # get a baseline (with linearization, which we will use from here on out)
    kde_pipe_cfg[("utils", "kde")]["linearize_log_dims"] = True
    dmaker = DistributionMaker([kde_pipe_cfg])
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

if __name__ == "__main__":
    main()
