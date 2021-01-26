
"""
Stage to transform arrays with weights into KDE maps
that represent event counts
"""
from __future__ import absolute_import, print_function, division
import numpy as np

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils import vectorizer
from pisa.utils.kde_hist import kde_histogramdd

class kde(Stage):
    """stage to KDE-map events

    Parameters
    ----------

    bw_method: string
        'scott' or 'silverman' (see kde module)
    coszen_name : string
        Binning name to identify the coszen bin that needs to undergo special
        treatment for reflection
    oversample : int
        Evaluate KDE at more points per bin, takes longer, but is more accurate

    Notes
    -----

    Make sure enough events are present with reco energy below and above the
    binning range, otherwise events will only "bleed out"

    """
    def __init__(self,
                 bw_method='silverman',
                 coszen_name='reco_coszen',
                 oversample=10,
                 **std_kargs,
                ):

        self.bw_method = bw_method
        self.coszen_name = coszen_name
        self.oversample = int(oversample)

        # init base class
        super().__init__(
            expected_params=(),
            **std_kargs,
        )

    @profile
    def apply(self):
        # this is special, we want the actual event weights in the kde
        # therefor we're overwritting the apply function
        # normally in a stage you would implement the `apply_function` method
        # and not the `apply` method!

        binning = self.apply_mode

        for container in self.data:
            self.data.representation = self.calc_mode
            sample = np.stack([container[n] for n in binning.names]).T
            weights = container['weights']

            kde_map = kde_histogramdd(sample=sample,
                            binning=binning,
                            weights=weights,
                            bw_method=self.bw_method,
                            coszen_name=self.coszen_name,
                            oversample=self.oversample,
                            use_cuda=False,
                            stack_pid=True)

            kde_map = np.ascontiguousarray(kde_map.ravel())

            self.data.representation = self.apply_mode
            container['weights'] = kde_map
