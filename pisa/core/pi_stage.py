# Authors

"""
Stage class designed to be inherited by PISA Pi services, such that all basic
functionality is built-in.
"""


from __future__ import absolute_import, division

from numba import SmartArray

from pisa.core.base_stage import BaseStage
from pisa.core.binning import MultiDimBinning
from pisa.core.container import ContainerSet
from pisa.utils.log import logging
from pisa.utils.profiler import profile


__all__ = ["PiStage"]
__version__ = "Pi"
__author__ = "Philipp Eller (pde3@psu.edu)"


class PiStage(BaseStage):
    """
    PISA Pi stage base class. Should be used to implement PISA Pi stages

    Specialization should be done via subclasses.

    Parameters
    ----------
    data : ContainerSet or None
        object to be passed along

    input_names : None or list of strings

    output_names : None or list of strings

    input_specs : binning or 'events' or None
        Specify the inputs (i.e. what did the last stage output, or None)

    calc_specs : binning or 'events' or None
        Specify in what to do the calculation

    output_specs : binning or 'events' or None
        Specify how to generate the outputs

    input_cal_keys : tuple of str
        external keys of data the compute function needs

    output_calc_keys : tuple of str
        output keys of the calculation (not intermediate results)

    input_apply_keys : tuple of str
        keys needed by the apply function data (usually 'weights')

    output_apply_keys : tuple of str
        keys of the output data (usually 'weights')

    """

    def __init__(
        self,
        data=None,
        params=None,
        expected_params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        error_method=None,
        input_specs=None,
        calc_specs=None,
        output_specs=None,
        input_apply_keys=(),
        output_apply_keys=(),
        input_calc_keys=(),
        output_calc_keys=(),
    ):
        super().__init__(
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            error_method=error_method,
        )

        self.input_specs = input_specs
        self.calc_specs = calc_specs
        self.output_specs = output_specs
        self.data = data

        if isinstance(self.input_specs, MultiDimBinning):
            self.input_mode = "binned"
        elif self.input_specs == "events":
            self.input_mode = "events"
        elif self.input_specs is None:
            self.input_mode = None
        else:
            raise ValueError("Cannot understand `input_specs` %s" % input_specs)

        if isinstance(self.calc_specs, MultiDimBinning):
            self.calc_mode = "binned"
        elif self.calc_specs == "events":
            self.calc_mode = "events"
        elif self.calc_specs is None:
            self.calc_mode = None
        else:
            raise ValueError("Cannot understand `calc_specs` %s" % calc_specs)

        if isinstance(self.output_specs, MultiDimBinning):
            self.output_mode = "binned"
        elif self.output_specs == "events":
            self.output_mode = "events"
        elif self.output_specs is None:
            self.output_mode = None
        else:
            raise ValueError("Cannot understand `output_specs` %s" % output_specs)

        self.input_calc_keys = input_calc_keys
        self.output_calc_keys = output_calc_keys
        self.input_apply_keys = input_apply_keys
        self.output_apply_keys = output_apply_keys

        # make a string of the modes for convenience
        mode = ["N", "N", "N"]
        if self.input_mode == "binned":
            mode[0] = "B"
        elif self.input_mode == "events":
            mode[0] = "E"

        if self.calc_mode == "binned":
            mode[1] = "B"
        elif self.calc_mode == "events":
            mode[1] = "E"

        if self.output_mode == "binned":
            mode[2] = "B"
        elif self.output_mode == "events":
            mode[2] = "E"

        self.mode = "".join(mode)

        self.param_hash = None
        # cake compatibility
        self.outputs = None

    def setup(self):

        # check that data is a ContainerSet (downstream modules assume this)
        if self.data is not None:
            if not isinstance(self.data, ContainerSet):
                raise TypeError("`data` must be a `pisa.core.container.ContainerSet`")

        # check that the arrays in `data` is stored as numba `SmartArrays`
        # the downstream stages generally assume this
        # a common problem is if the user copies data before passing it to th stage then
        # a bug in SmartArray means the result is a numoy array, rather than a
        # SmartArray
        if self.data is not None:
            for container in self.data:
                for key, array in container.array_data.items():
                    if not isinstance(array, SmartArray):
                        raise TypeError(
                            "Array `%s` in `data` should be a `numba.SmartArray`, but"
                            " is a %s" % (key, type(array))
                        )

        # call the user-defined setup function
        self.setup_function()

        # invalidate param hash:
        self.param_hash = -1

    def setup_function(self):
        """Implement in services (subclasses of PiStage)"""
        pass

    @profile
    def compute(self):

        if len(self.params) == 0 and len(self.output_calc_keys) == 0:
            return

        # simplest caching algorithm: don't compute if params didn't change
        new_param_hash = self.params.values_hash
        if new_param_hash == self.param_hash:
            logging.trace("cached output")
            return

        self.data.data_specs = self.input_specs
        # convert any inputs if necessary:
        if self.mode[:2] == "EB":
            for container in self.data:
                for key in self.input_calc_keys:
                    container.array_to_binned(key, self.calc_specs)

        elif self.mode == "EBE":
            for container in self.data:
                for key in self.input_calc_keys:
                    container.binned_to_array(key)

        #elif self.mode == "BBE":
        #    for container in self.data:
        #        for key in self.input_calc_keys:
        #            container.binned_to_array(key)

        self.data.data_specs = self.calc_specs
        self.compute_function()
        self.param_hash = new_param_hash

        # convert any outputs if necessary:
        if self.mode[1:] == "EB":
            for container in self.data:
                for key in self.output_calc_keys:
                    container.array_to_binned(key, self.output_specs)

        elif self.mode[1:] == "BE":
            for container in self.data:
                for key in self.output_calc_keys:
                    container.binned_to_array(key)

    def compute_function(self):
        """Implement in services (subclasses of PiStage)"""
        pass

    @profile
    def apply(self):

        self.data.data_specs = self.input_specs
        # convert any inputs if necessary:
        if self.mode[0] + self.mode[2] == "EB":
            for container in self.data:
                for key in self.input_apply_keys:
                    container.array_to_binned(key, self.output_specs)

        # elif self.mode == 'BBE':
        #    pass

        elif self.mode[0] + self.mode[2] == "BE":
            for container in self.data:
                for key in self.input_apply_keys:
                    container.binned_to_array(key)

        # if self.input_specs is not None:
        #    self.data.data_specs = self.input_specs
        # else:
        self.data.data_specs = self.output_specs
        self.apply_function()

        if self.mode == "BBE":
            for container in self.data:
                for key in self.output_apply_keys:
                    container.binned_to_array(key)

    def apply_function(self):
        """Implement in services (subclasses of PiStage)"""
        pass

    def run(self, inputs=None):
        if not inputs is None:
            raise ValueError("PISA pi requires there not be any inputs.")
        self.compute()
        self.apply()
        return None

    def get_outputs(self, output_mode=None):
        """
        Get the outputs of the PISA stage
        Depending on `self.output_mode`, this may be a binned object, or the event container itself

        add option to force an output mode

        """

        if output_mode is not None:
            assert output_mode=='binned' or output_mode=='events','ERROR: user-specified output mode is unrecognized'
            self.output_mode = output_mode
        
        if self.output_mode == 'binned' and len(self.output_apply_keys) == 1:
            self.outputs = self.data.get_mapset(self.output_apply_keys[0])
        elif self.output_mode=='binned' and len(self.output_apply_keys) == 2 and 'errors' in self.output_apply_keys:
            other_key = [key for key in self.output_apply_keys if not key == 'errors'][0]
            self.outputs = self.data.get_mapset(other_key, error='errors')
        elif self.output_mode == "events" :
            self.outputs = self.data
        else:
            self.outputs = None
            logging.warning('Cannot create CAKE style output mapset')

        return self.outputs
