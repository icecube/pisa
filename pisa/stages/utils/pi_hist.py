
"""
Stage to transform arrays with weights into actual `histograms`
that represent event counts
"""
from __future__ import absolute_import, print_function, division
import numpy as np

from pisa import FTYPE
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils import vectorizer

class pi_hist(PiStage):
    """
    stage to histogram events

    Paramaters
    ----------

    None

    Notes
    -----

    """
    def __init__(self,
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
        input_apply_keys = ('weights',
                           )

        # what are keys added or altered in the calculation used during apply
        assert calc_specs is None
        if error_method in ['sumw2']:
            output_apply_keys = ('weights',
                                 'errors',
                                )
        else:
            output_apply_keys = ('weights',
                                )


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

        assert self.input_mode is not None
        assert self.output_mode == 'binned'

    def setup_function(self):
        # create the variables to be filled in `apply`
        if self.error_method in ['sumw2']:
            self.data.data_specs = self.input_specs
            for container in self.data:
                container['weights_squared'] = np.empty((container.size), dtype=FTYPE)
            self.data.data_specs = self.output_specs
            for container in self.data:
                container['errors'] = np.empty((container.size), dtype=FTYPE)

    @profile
    def apply(self):
        # this is special, we want the actual event weights in the histo
        # therefor we're overwritting the apply function
        # normally in a stage you would implement the `apply_function` method
        # and not the `apply` method!

        if self.input_mode == 'binned':
            self.data.data_specs = self.output_specs
            for container in self.data:
                # calcualte errors
                if self.error_method in ['sumw2']:
                    vectorizer.square(container['weights'], out=container['weights_squared'])
                    vectorizer.sqrt(container['weights_squared'], out=container['errors'])

        elif self.input_mode == 'events':
            for container in self.data:
                self.data.data_specs = self.input_specs
                # calcualte errors
                if self.error_method in ['sumw2']:
                    vectorizer.square(container['weights'], out=container['weights_squared'])
                self.data.data_specs = self.output_specs
                container.array_to_binned('weights', self.output_specs, averaged=False)
                if self.error_method in ['sumw2']:
                    container.array_to_binned('weights_squared', self.output_specs, averaged=False)
                    vectorizer.sqrt(container['weights_squared'], out=container['errors'])
