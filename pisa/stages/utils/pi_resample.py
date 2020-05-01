"""
Stage to transform binned data from one binning to another while also dealing with
uncertainty estimates in a reasonable way. In particular, this allows up-sampling
from a more coarse binning to a finer binning.

The implementation is similar to that of the pi_hist stage, hence the over-writing
of the `apply` method.
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


class ResampleMode(Enum):
    """Enumerates sampling methods of the `pi_resample` stage."""

    UP = auto()
    DOWN = auto()


class pi_resample(PiStage):  # pylint: disable=invalid-name
    """
    Stage to resample events from one binning to another.
    """

    def __init__(
        self,
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
        if error_method in ["sumw2"]:
            output_apply_keys = ("weights", "errors")
        else:
            output_apply_keys = ("weights",)
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
        )

        # This stage only makes sense when going binned to binned.
        assert self.input_mode == "binned", "stage only takes binned input"
        assert self.output_mode == "binned", "stage only produces binned output"

        # The following tests whether `output_specs` is a strict up-sample
        # from `input_specs`, i.e. the bin edges of `output_specs` are a superset
        # of the bin edges of `input_specs`.
        if input_specs.is_compat(output_specs):
            self.rs_mode = ResampleMode.UP
        elif output_specs.is_compat(input_specs):
            self.rs_mode = ResampleMode.DOWN
        else:
            raise ValueError("Binnings are not compatible with each other for resample")

        # TODO: Implement downsampling.
        if self.rs_mode == ResampleMode.DOWN:
            raise NotImplementedError("Downsampling not yet implemented.")

    @profile
    def apply(self):
        # DO NOT USE THIS STAGE AS YOUR TEMPLATE IF YOU ARE NEW TO PISA!
        # --------------------------------------------------------------
        #
        # We are overwriting the `apply` method rather than the `apply_function` method
        # because we are manipulating the data binning in a delicate way that doesn't
        # work with automatic rebinning.

        self.data.data_specs = self.output_specs
        for container in self.data:
            # The built-in `binned_to_binned` method behaves as follows:
            # - When several bins are merged into one, the large bin contains the
            #   average of the smaller bins.
            # - When a bin is split into smaller bins, each of the smaller bins gets
            #   the same value as the large bin.
            # This first step is the same whether we sample up or down.
            container.binned_to_binned("weight", self.output_specs)
            if self.error_method in ["sumw2"]:
                container.binned_to_binned("weight_squared", self.output_specs)

            # We now have to scale the weights and squared weights according to the bin
            # volumes depending on the sampling mode.
            if self.rs_mode == ResampleMode.UP:
                # These are the volumes of the bins we sample *into*
                upsampled_binvols = SmartArray(
                    self.output_specs.bin_volumes(attach_units=False).ravel()
                )
                # These are the volumes of the bins we sample *from*
                coarse_volumes = SmartArray(
                    self.input_specs.bin_volumes(attach_units=False).ravel()
                )
                # For every upsampled bin, we need to know what the volume of the bin
                # was where it came from. First, we get the position of the midpoint of
                # each fine (output) bin:
                fine_gridpoints = [
                    # The `unroll_binning` function returns the midpoints of the bins
                    # in the dimension `name`.
                    SmartArray(container.unroll_binning(name, self.output_specs))
                    for name in self.output_specs.names
                ]
                # We look up at which bin index of the input binning the midpoints of
                # the output binning can be found, and assign to each the volume of the
                # bin of that index.
                origin_binvols = translation.lookup(
                    fine_gridpoints, coarse_volumes, self.input_specs
                )
                # Finally, we scale the weights and squared weights by the ratio of the
                # bin volumes in place:
                vectorizer.imul(upsampled_binvols, container["weight"])
                vectorizer.itruediv(origin_binvols, container["weight"])
                container["weight"].mark_changed()
                if self.error_method in ["sumw2"]:
                    vectorizer.imul(upsampled_binvols, container["weight_squared"])
                    vectorizer.itruediv(origin_binvols, container["weight_squared"])
                    container["weight_squared"].mark_changed()
            elif self.rs_mode == ResampleMode.DOWN:
                pass  # not yet implemented

            if self.error_method in ["sumw2"]:
                vectorizer.sqrt(
                    vals=container["weights_squared"], out=container["errors"]
                )
                container["errors"].mark_changed()
