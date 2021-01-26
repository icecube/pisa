"""
Override errors and replace with manually chosen error fraction.
"""

import numpy as np
from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.profiler import profile
from pisa.utils.log import logging
from pisa.utils import vectorizer
from pisa.core.binning import MultiDimBinning


class set_variance(Stage):  # pylint: disable=invalid-name
    """
    Override errors and replace with manually chosen variance.
    """

    def __init__(
        self,
        variance_scale=1.0,
        variance_floor=None,
        expected_total_mc=None,
        divide_total_mc=False,
        **std_kwargs,
    ):

        # init base class
        super().__init__(
            expected_params=(),
            **std_kwargs,
        )
        
        assert isinstance(self.calc_mode, MultiDimBinning)
        assert isinstance(self.apply_mode, MultiDimBinning)

        self.variance_scale = variance_scale
        self.variance_floor = variance_floor
        assert self.variance_scale is not None
        
        self.expected_total_mc = int(expected_total_mc)
        self.divide_n = divide_total_mc
        if self.divide_n:
            assert self.expected_total_mc is not None
        self.total_mc = {}
        
    def setup_function(self):
        if self.divide_n:
            self.data.representation = "events"
            for container in self.data:
                self.total_mc[container.name] = container.size
                logging.debug(f"{container.size} mc events in container {container.name}")
        self.data.representation = self.calc_mode
        for container in self.data:
            container["manual_variance"] = np.empty((container.size), dtype=FTYPE)
            if "errors" not in container.keys:
                container["errors"] = np.empty((container.size), dtype=FTYPE)

    def apply_function(self):
        for container in self.data:
            vectorizer.sqrt(vals=container["manual_variance"], out=container["errors"])

    def compute_function(self):
        for container in self.data:
            vectorizer.assign(vals=container["weights"], out=container["manual_variance"])
            vectorizer.scale(
                vals=container["manual_variance"],
                scale=self.variance_scale,
                out=container["manual_variance"],
            )
            if self.divide_n:
                vectorizer.scale(
                    vals=container["manual_variance"],
                    scale=self.expected_total_mc/self.total_mc[container.name],
                    out=container["manual_variance"],
                )
            if self.variance_floor is not None:
                apply_floor(self.variance_floor, out=container["manual_variance"])

FX = "f4" if FTYPE == np.float32 else "f8"

def apply_floor(val, out):
    apply_floor_gufunc(FTYPE(val), out=out)

@guvectorize([f"({FX}, {FX}[:])"], "() -> ()", target=TARGET)
def apply_floor_gufunc(val, out):
    out[0] = val if out[0] < val else out[0]

def set_constant(val, out):
    set_constant_gufunc(FTYPE(val), out=out)

@guvectorize([f"({FX}, {FX}[:])"], "() -> ()", target=TARGET)
def set_constant_gufunc(val, out):
    out[0] = val
