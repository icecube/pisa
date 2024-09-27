"""
Make bootstrap samples of data.

This stage allows one to resample datasets to estimate MC uncertainties without having
to decrease statistics. Bootstrap samples are produced by random selection with
replacement, which is implemented in this stage by an equivalent re-weighting of
events.
"""

from copy import deepcopy
from collections import OrderedDict

import numpy as np

from pisa.core.stage import Stage
from pisa.utils.log import logging, set_verbosity

__author__ = "A. Trettin"

__license__ = """Copyright (c) 2022, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


class bootstrap(Stage):  # pylint: disable=invalid-name
    """
    Stage to make bootstrap samples from input data.

    Parameters
    ----------
    seed : int, optional
        Seed for the random number generator.
    """

    def __init__(
        self,
        seed=None,
        **std_kwargs,
    ):
        # init base class
        super().__init__(
            expected_params=(),
            **std_kwargs,
        )

        assert self.calc_mode == "events"

        if seed is None:
            self.seed = None
        else:
            self.seed = int(seed)

    def setup_function(self):

        logging.debug(f"Setting up bootstrap with seed: {self.seed}")

        from numpy.random import default_rng

        rng = default_rng(self.seed)

        for container in self.data:
            sample_size = container["weights"].size
            # indices of events are randomly chosen from the entire sample until
            # we have a new sample of the same size
            sample_idx = rng.integers(sample_size, size=sample_size)
            # Instead of manipulating all of the data arrays, we count how often each
            # index was chosen and take that as a weight, i.e. an event that was selected
            # twice will have a weight of 2.
            sample_weights = np.bincount(sample_idx, minlength=sample_size)
            container["bootstrap_weights"] = sample_weights

    def apply_function(self):

        for container in self.data:
            container["weights"] *= container["bootstrap_weights"]


def insert_bootstrap_after_data_loader(cfg_dict, seed=None):
    """
    Given a pipeline configuration parsed with `parse_pipeline_config`, insert the
    bootstrap stage directly after the `simple_data_loader` stage and return the
    modified config dict.

    Parameters
    ----------
    cfg_dict : collections.OrderedDict
        Pipeline configuration in the form of an ordered dictionary.
    seed : int, optional
        Seed to be placed into the pipeline configuration.

    Returns
    -------
    collections.OrderedDict
        A deepcopy of the original input `cfg_dict` with the configuration of the
        bootstrap stage inserted after the data loader.
    """

    bootstrap_stage_cfg = OrderedDict()
    bootstrap_stage_cfg["apply_mode"] = "events"
    bootstrap_stage_cfg["calc_mode"] = "events"
    bootstrap_stage_cfg["seed"] = seed

    bootstrap_pipe_cfg = deepcopy(cfg_dict)

    # Important: Cannot mutate dict while iterating over it, instantiate list instead
    for k in list(bootstrap_pipe_cfg.keys()):
        bootstrap_pipe_cfg.move_to_end(k)
        if k == ("data", "simple_data_loader"):
            bootstrap_pipe_cfg[("utils", "bootstrap")] = bootstrap_stage_cfg

    return bootstrap_pipe_cfg


def test_bootstrap():
    """Unit test for the bootstrap stage."""

    from pisa.core.distribution_maker import DistributionMaker
    from pisa.utils.config_parser import parse_pipeline_config

    example_cfg = parse_pipeline_config("settings/pipeline/example.cfg")

    # We need to insert the bootstrap stage right after the data loading stage
    bootstrap_pipe_cfg = insert_bootstrap_after_data_loader(example_cfg, seed=0)

    logging.debug("bootstrapped pipeline stage order:")
    logging.debug(list(bootstrap_pipe_cfg.keys()))

    # get a baseline
    dmaker = DistributionMaker([example_cfg])
    map_baseline = dmaker.get_outputs(return_sum=True)[0]

    # Make sure that different seeds produce different maps, and that the same seed will
    # produce the same map.
    dmaker = DistributionMaker([bootstrap_pipe_cfg])
    map_seed0 = dmaker.get_outputs(return_sum=True)[0]

    # find key of bootstrap stage
    bootstrap_idx = 0
    for i, stage in enumerate(dmaker.pipelines[0].stages):
        if stage.__class__.__name__ == "bootstrap":
            bootstrap_idx = i

    # without re-loading the entire pipeline, we set the seed and call the setup function
    # to save time for the test
    dmaker.pipelines[0].stages[bootstrap_idx].seed = 1
    dmaker.pipelines[0].stages[bootstrap_idx].setup()

    map_seed1 = dmaker.get_outputs(return_sum=True)[0]

    assert not map_seed0 == map_seed1

    dmaker.pipelines[0].stages[bootstrap_idx].seed = 0
    dmaker.pipelines[0].stages[bootstrap_idx].setup()
    map_seed0_reprod = dmaker.get_outputs(return_sum=True)[0]

    assert map_seed0 == map_seed0_reprod

    # Quantify the variance of the resulting maps. They should be about the size of the
    # expectation from sum of weights-squared.

    nominal_values = []
    for i in range(100):
        dmaker.pipelines[0].stages[bootstrap_idx].seed = i
        dmaker.pipelines[0].stages[bootstrap_idx].setup()
        map_bootstrap = dmaker.get_outputs(return_sum=True)[0]
        nominal_values.append(map_bootstrap.nominal_values)

    nominal_values = np.stack(nominal_values)
    with np.errstate(divide="ignore", invalid="ignore"):
        # calculate the ratio between the bootstrap nominal and the baseline nominal
        bs_nom_ratios = np.mean(nominal_values, axis=0) / map_baseline.nominal_values
        # and the standard deviation ratio as well
        bs_std_ratios = np.std(nominal_values, axis=0) / map_baseline.std_devs
        # assert that both nominal and standard deviation match the expectation from
        # baseline up to a small error
        assert np.abs(np.nanmean(bs_nom_ratios) - 1.0) < 0.01
        # the standard deviations are a little harder to match in 100 samples
        assert np.abs(np.nanmean(bs_std_ratios) - 1.0) < 0.02

    logging.info("<< PASS : bootstrap >>")


if __name__ == "__main__":
    set_verbosity(1)
    test_bootstrap()
