"""
Stage to transform arrays with weights into actual `histograms`
that represent event counts
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.core.translation import histogram
from pisa.utils.profiler import profile
from pisa.utils import vectorizer


class hist(Stage):  # pylint: disable=invalid-name
    """stage to histogram events"""
    def __init__(
        self,
        **std_kwargs,
        ):

        # init base class
        super().__init__(
            expected_params=(),
            **std_kwargs,
        )


    @profile
    def apply_function(self):

        self.data.representation = self.calc_mode

        if self.calc_mode == 'binned':
            raise NotImplementedError('Needs some care, broken in pisa4')
            self.data.representation = self.apply_mode
            for container in self.data:
                # calcualte errors
                if self.error_method in ['sumw2']:
                    vectorizer.pow(
                        vals=container['weights'],
                        pwr=2,
                        out=container['weights_squared'],
                    )
                    vectorizer.sqrt(
                        vals=container['weights_squared'], out=container['errors']
                    )

        elif self.calc_mode == 'events':
            for container in self.data:
                # calcualte errors

                container.representation = self.calc_mode
                sample = [container[name] for name in self.apply_mode.names]
                weights = container['weights']

                hist = histogram(sample, weights, self.apply_mode, averaged=False)

                if self.error_method == 'sumw2':
                    sumw2 = histogram(sample, np.square(weights), self.apply_mode, averaged=False)

                container.representation = self.apply_mode
                
                container['weights'] = hist

                if self.error_method == 'sumw2':
                    container['errors'] = np.sqrt(sumw2)
