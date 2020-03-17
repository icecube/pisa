
"""
Stage to transform arrays with weights into KDE maps
that represent event counts
"""
from __future__ import absolute_import, print_function, division
import numpy as np

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils import vectorizer
from pisa.utils.kde_hist import kde_histogramdd

class pi_kde(PiStage):
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

        self.bw_method = bw_method
        self.coszen_name = coszen_name
        self.oversample = int(oversample)

        expected_params = ()
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_apply_keys = ('weights',
                           )

        # what are keys added or altered in the calculation used during apply
        assert calc_specs is None
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

        assert self.input_mode == 'events'
        assert self.output_mode == 'binned'

    @profile
    def apply(self):
        # this is special, we want the actual event weights in the kde
        # therefor we're overwritting the apply function
        # normally in a stage you would implement the `apply_function` method
        # and not the `apply` method!

        binning = self.output_specs

        for container in self.data:
            self.data.data_specs = self.input_specs
            sample = np.stack([container[n].get('host') for n in binning.names]).T
            weights = container['weights'].get('host')

            kde_map = kde_histogramdd(sample=sample,
                            binning=binning,
                            weights=weights,
                            bw_method=self.bw_method,
                            coszen_name=self.coszen_name,
                            oversample=self.oversample,
                            use_cuda=False,
                            stack_pid=True)

            kde_map = np.ascontiguousarray(kde_map.ravel())

            self.data.data_specs = self.output_specs
            container['weights'] = kde_map
            container['weights'].mark_changed('host')
