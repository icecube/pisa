"""
Stage to apply pre-calculated Genie uncertainties
"""

from __future__ import absolute_import, print_function, division

__all__ = ["genie_sys", "apply_genie_sys"] #, "SIGNATURE"

import re
import numpy as np

from pisa.core.stage import Stage
from pisa.utils.log import logging

class genie_sys(Stage): # pylint: disable=invalid-name
    """
    Stage to apply pre-calculated Genie systematics.

    Parameters
    ----------
    params
        Must contain ::

        parameters specified in interactions (dimensionless)

    Notes
    -----
    Requires the events have the following keys for each included interaction ::

        linear_fit_{name}
            Genie linear coefficient for interaction {name}
        quad_fit_{name}
            Genie quadratic coefficient for interaction {name}

    """
    def __init__(
        self,
        interactions="Genie_Ma_QE, Genie_Ma_RES",
        names="maccqe, maccres",
        **std_kwargs,
    ):
        interactions = re.split(r'\W+', interactions)
        names = re.split(r'\W+', names)
        assert len(interactions) == len(names), 'Specify a name for each interaction'

        expected_params = tuple(interactions)
        self.interactions = interactions
        self.names = names

        # init base class
        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )


    def setup_function(self):
        '''
        Check the range of the parameter in the analysis. Send a warning if these are beyond +- 2sigma
        '''
        for I in self.interactions:
            if self.params[I].range[0]<-2. or self.params[I].range[1]>2.:
                logging.warn(I+' parameter bounds have been set larger than the range used to produce interpolation points ([-2.,2]). This will void the warranty...')


    def apply_function(self):
        genie_params = []
        for I in self.interactions:
            exec("genie_params.append(self.params.%s.m_as('dimensionless'))"%(I))

        for container in self.data:
            linear_fits, quad_fits = [], []
            for i in range(len(self.interactions)):
                linear_fits.append(container['linear_fit_'+self.names[i]])
                quad_fits.append(container['quad_fit_'+self.names[i]])

            apply_genie_sys(
                genie_params,
                linear_fits,
                quad_fits,
                out=container['weights'],
            )

            #
            # In cases where the axial mass is extrapolated outside
            # the range of the points used in the interpolation, some
            # weights become negative. These are floored at 0.
            #
            container.mark_changed('weights')


# TODO: need to keep these comments?
#if FTYPE == np.float64:
#    SIGNATURE = '(f8, f8, f8, f8[:])'
#else:
#    SIGNATURE = '(f4, f4, f4, f4[:])'
#@guvectorize([SIGNATURE], '(),(),()->()', target=TARGET)
def apply_genie_sys(
    genie_params,
    linear_fits,
    quad_fits,
    out,
):
    factor = 1
    for i in range(len(genie_params)):
        factor *= 1. + (linear_fits[i] + quad_fits[i] * genie_params[i]) * genie_params[i]
    out *= np.maximum(0, factor)
