# Authors
"""
Stage class designed to be inherited by PISA Pi services, such that all basic
functionality is built-in.

"""
from __future__ import absolute_import

from pisa.core.base_stage import BaseStage
from pisa.core.binning import MultiDimBinning
from pisa.utils.log import logging
from pisa.utils.profiler import profile


__all__ = ['PiStage']
__version__ = 'Pi'
__author__ = 'Philipp Eller (pde3@psu.edu)'


class PiStage(BaseStage):
    """
    PISA stage base class. Should encompass all behaviors common to (almost)
    all stages.

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

    input_keys : tuple of str
        keys of input data the stage needs

    calc_keys : tuple of str
        output keys of the calculation (not intermediate results)

    output_keys : tuple of str
        keys of the output data (usually 'weights')

    """
    def __init__(self,
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
                 input_keys=(),
                 calc_keys=(),
                 output_keys=(),
                ):

        # init base class
        super(PiStage, self).__init__(params=params,
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
            self.input_mode = 'binned'
        elif self.input_specs == 'events':
            self.input_mode = 'events'
        elif self.input_specs is None:
            self.input_mode = None
        else:
            raise ValueError('Not understood input_specs %s'%input_specs)

        if isinstance(self.calc_specs, MultiDimBinning):
            self.calc_mode = 'binned'
        elif self.calc_specs == 'events':
            self.calc_mode = 'events'
        elif self.calc_specs is None:
            self.calc_mode = None
        else:
            raise ValueError('Not understood calc_specs %s'%calc_specs)

        if isinstance(self.output_specs, MultiDimBinning):
            self.output_mode = 'binned'
        elif self.output_specs == 'events':
            self.output_mode = 'events'
        elif self.output_specs is None:
            self.output_mode = None
        else:
            raise ValueError('Not understood output_specs %s'%output_specs)

        self.input_keys = input_keys
        self.calc_keys = calc_keys
        self.output_keys = output_keys

        self.param_hash = None

        # cake compatibility
        self.outputs = None

    def setup(self):
        self.setup_function()
        # invalidate param hash:
        self.param_hash = -1

    def setup_function(self):
        # to be implemented by stage
        pass

    @profile
    def compute(self):
        # simplest caching algorithm....just don't compute if params didn't change
        if len(self.params) > 0:
            new_param_hash = self.params.values_hash
            if not new_param_hash == self.param_hash:
                self.compute_function()
                self.param_hash = new_param_hash
            else:
                logging.trace('cached output')

    def compute_function(self):
        # to be implemented by stage
        pass

    @profile
    def apply(self):

        self.compute()

        self.data.data_specs = 'events'

        # make a string of the modes for convenience
        mode = ['N','N','N']
        if self.input_mode == 'binned':
            mode[0] = 'B'
        elif self.input_mode == 'events':
            mode[0] = 'E'
        if self.calc_mode == 'binned':
            mode[1] = 'B'
        elif self.calc_mode == 'events':
            mode[1] = 'E'
        if self.output_mode == 'binned':
            mode[2] = 'B'
        elif self.output_mode == 'events':
            mode[2] = 'E'
        mode = ''.join(mode)


        if mode == 'NNE':
            pass

        if mode[:2] == 'BB':
            self.data.data_specs = self.calc_specs

        if mode == 'BEE':
            for container in self.data:
                for key in self.input_keys:
                    container.binned_to_array(key)

        if mode == 'BEB':
            for container in self.data:
                for key in self.calc_keys:
                    container.array_to_binned(key, self.input_specs)
            self.data.data_specs = self.calc_specs

        if mode == 'EBB':
            for container in self.data:
                for key in self.input_keys:
                    container.array_to_binned(key, self.calc_specs)
            self.data.data_specs = self.calc_specs

        if mode == 'ENB':
            for container in self.data:
                for key in self.input_keys:
                    container.array_to_binned(key, self.output_specs)
            self.data.data_specs = self.output_specs

        if mode == 'EBE':
            for container in self.data:
                for key in self.calc_keys:
                    container.binned_to_array(key)

        # run apply function 
        self.apply_function()

        if self.data.data_mode == 'binned' and self.output_mode == 'events':
            for container in self.data:
                for key in self.output_keys:
                    container.binned_to_array(key)

        if self.data.data_mode == 'events' and self.output_mode == 'binned':
            for container in self.data:
                for key in self.output_keys:
                    container.array_to_binned(key, self.output_specs)

    def apply_function(self):
        # to be implemented by stage
        pass


    def get_outputs(self):
        '''
        function for cake style outputs
        '''
        # output keys need to be exactly 1 to generate pisa cake style mapset
        if len(self.output_keys) == 1:
            self.outputs = self.data.get_mapset(self.output_keys[0])
        else:
            assert len(self.output_keys) == 2 and 'errors' in self.output_keys, 'Cannot transfor this output into PISA style maps with output keys %s'%self.output_keys
            other_key = [key for key in self.output_keys if not key == 'errors'][0]
            self.outputs = self.data.get_mapset(other_key, error='errors')

        return self.outputs
