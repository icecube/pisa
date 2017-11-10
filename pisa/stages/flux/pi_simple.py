import numpy as np
from numba import guvectorize, SmartArray

from pisa import *
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import WHERE
from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet


class pi_simple(PiStage):
    """
    stage to apply Barr style flux uncertainties

    Paramaters
    ----------
    
    nue_numu_ratio : quantity (dimensionless)

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
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                 ):

        expected_params = ('nue_numu_ratio',
                            )
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_keys = ('weights',
                      'neutrino_nue_flux',
                      'neutrino_numu_flux',
                      )
        # what are keys added or altered in the calculation used during apply 
        calc_keys = ('flux_e',
                     'flux_mu',
                     )
        # what keys are added or altered for the outputs during apply
        output_keys = ('flux_e',
                       'flux_mu',
                       )

        # init base class
        super(pi_simple, self).__init__(data=data,
                                       params=params,
                                       expected_params=expected_params,
                                       input_names=input_names,
                                       output_names=output_names,
                                       debug_mode=debug_mode,
                                       input_specs=input_specs,
                                       calc_specs=calc_specs,
                                       output_specs=output_specs,
                                       input_keys=input_keys,
                                       calc_keys=calc_keys,
                                       output_keys=output_keys,
                                       )

        assert self.input_mode is not None
        assert self.calc_mode is not None
        assert self.output_mode is not None

    def setup(self):

        self.data.data_specs = self.calc_specs

        if self.calc_mode == 'binned':
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc'])
            self.data.link_containers('nubar', ['nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                                'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])
        for container in self.data:
            container['flux_e'] = np.empty((container.size), dtype=FTYPE)
            container['flux_mu'] = np.empty((container.size), dtype=FTYPE) 

        self.data.unlink_containers()

    def compute(self):

        self.data.data_specs = self.calc_specs
        nue_numu_ratio = self.params.nue_numu_ratio.value.m_as('dimensionless')

        if self.calc_mode == 'binned':
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc'])
            self.data.link_containers('nubar', ['nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                                'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])

        for container in self.data:
            apply_ratio_scale_vectorized(nue_numu_ratio,
                                         container['neutrino_nue_flux'].get(WHERE),
                                         out=container['flux_e'].get(WHERE),
                                         )
            container['flux_e'].mark_changed(WHERE)
            apply_ratio_scale_vectorized(1.,
                                         container['neutrino_numu_flux'].get(WHERE),
                                         out=container['flux_mu'].get(WHERE),
                                         )
            container['flux_e'].mark_changed(WHERE)

        self.data.unlink_containers()


# vectorized function to apply
# must be outside class
if FTYPE == np.float64:
    signature = '(f8, f8, f8[:])'
else:
    signature = '(f4, f4, f4[:])'

@guvectorize([signature], '(),()->()', target=TARGET)
def apply_ratio_scale_vectorized(ratio_scale, flux_in, out):
    out[0] = ratio_scale * flux_in
