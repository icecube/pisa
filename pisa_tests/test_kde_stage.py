"""
Module to test KDE bootstrapping. This could not be built into the KDE stage's script
itself, because the import of a class named `kde` directly in the main scope overshadows
the `kde` module and causes an import error.
"""

from copy import deepcopy
from pisa.utils.log import logging, set_verbosity
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.config_parser import parse_pipeline_config
from collections import OrderedDict


def test_kde_bootstrapping():
    """Unit test for the kde stage."""

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

    # get a baseline
    dmaker = DistributionMaker([kde_pipe_cfg])
    map_baseline = dmaker.get_outputs(return_sum=True)[0]
    logging.debug(f"Baseline KDE'd map:\n{map_baseline}")

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


if __name__ == "__main__":
    set_verbosity(1)
    test_kde_bootstrapping()
