"""
Stage to implement an ad-hoc systematic that corrects the discrepancy between data and
MC in one particular variable. This can be used to check how large the impact of such a
hypothetical systematic would be on the physics parameters of an analysis.
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.resources import find_resource
from pisa.utils.jsons import from_json
from pisa.core.binning import MultiDimBinning

__all__ = ['adhoc_sys', 'init_test']


class adhoc_sys(Stage):  # pylint: disable=invalid-name
    """
    Stage to re-weight events according to factors derived from post-fit data/MC
    comparisons. The comparisons are produced somewhere externally and stored as a JSON
    which encodes the binning that was used to make the comparison and the resulting
    scaling factors.

    Parameters
    ----------

    variable_name : str
        Name of the variable to correct data/MC agreement for. The variable must be
        loaded in the data loading stage and it must be present in the loaded JSON file.

    scale_file : str
        Path to the file which contains the binning and the scale factors. The JSON
        file must contain a dictionary in which, for each variable, a 1D binning and
        an array of factors. This file is produced externally from PISA.
    """
    def __init__(
        self,
        data=None,
        params=None,
        variable_name=None,
        scale_file=None,
        **std_kwargs,
    ):

        expected_params = ()

        expected_container_keys = [
            'weights', variable_name
        ]

        supported_reps = {
            'calc_mode': 'events',
            'apply_mode': 'events'
        }

        # init base class
        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            supported_reps=supported_reps,
            **std_kwargs,
        )

        assert self.calc_mode == "events"
        assert self.apply_mode == "events"

        self.scale_file = scale_file
        self.variable = variable_name

    def setup_function(self):
        scale_file = find_resource(self.scale_file)
        logging.info("Loading scaling factors from : %s", scale_file)

        scaling_dict = from_json(scale_file)
        scale_binning = MultiDimBinning(**scaling_dict[self.variable]["binning"])

        scale_factors = np.array(scaling_dict[self.variable]["scales"], dtype=FTYPE)
        logging.info(f"Binning for ad-hoc systematic: \n {str(scale_binning)}")
        logging.info(f"scaling factors of ad-hoc systematic:\n {str(scale_factors)}")
        self.data.representation = scale_binning
        for container in self.data:
            container["adhoc_scale_factors"] = scale_factors

    def apply_function(self):
        for container in self.data:
            container["weights"] *= container["adhoc_scale_factors"]


def init_test(**param_kwargs):
    """Instantiation example"""
    import os
    from pisa import CACHE_DIR
    from pisa.core.binning import OneDimBinning
    from pisa.utils.fileio import to_file
    from pisa.utils.random_numbers import get_random_state

    # create temporary scale file on the fly
    var = 'reco_length'
    bin_edges = [0, 0.5, 1.0]
    var_binning = MultiDimBinning(
        name='adhoc_sys_test_binning',
        dimensions=[OneDimBinning(name=var, bin_edges=bin_edges, is_lin=True)]
    )
    scales = get_random_state(0).random(len(bin_edges)-1).astype(dtype=FTYPE)
    scaling_dict = {var: {'binning':  var_binning, 'scales': scales}}
    scale_fpath = os.path.join(CACHE_DIR, 'test_scale_file.json')
    to_file(scaling_dict, scale_fpath, warn=False)

    return adhoc_sys(
        variable_name=var, scale_file=scale_fpath,
        calc_mode='events', apply_mode='events'
    )
