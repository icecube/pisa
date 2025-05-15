"""
This stage corrects inelasticity distribution 
to remove the most obvious impact from charm 
production bug in GENIE.

Maria Liubarska
"""

import numpy as np
import pickle
from numba import guvectorize

from pisa.core.stage import Stage
from pisa.utils.resources import open_resource
from pisa.utils import vectorizer
from pisa.utils.fileio import from_file
from pisa.utils.log import logging
from pisa import FTYPE, TARGET

class correct_charm_y(Stage):
    """
    blah
    
    """
    def __init__(
            self,
            correct_charm=False,
            **std_kwargs,
    ):

        self.correct_charm = correct_charm
        expected_params = ()

        expected_container_keys = (
            'true_energy',
            'true_coszen',
            'bjorken_y',
            'weights',
        )

        # init base class
        super(correct_charm_y, self).__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

    def setup_function(self):

        if self.correct_charm:
            init_y_disrt_hists = from_file('cross_sections/charm_y_correction_2d_coszen_split.pckl')

            h_nucc_upg = init_y_disrt_hists['nu_cc_upg']
            h_nubarcc_upg = init_y_disrt_hists['nubar_cc_upg']
            h_nucc_oth = init_y_disrt_hists['nu_cc_oth']
            h_nubarcc_oth = init_y_disrt_hists['nubar_cc_oth']
            xed = init_y_disrt_hists['bins_lgE']
            yed = init_y_disrt_hists['bins_y']

            def eval_hist(lgE, y, coszen, nubar=False):

                ind_x = np.digitize(lgE, bins=xed)
                ind_y = np.digitize(y, bins=yed)

                # TODO: replace hardcoded number of bins?
                ind_x[ind_x==31] = 30
                ind_y[ind_y==31] = 30

                res = np.zeros_like(lgE)
                mask_upg = coszen<-0.9

#                 if coszen<-0.9:
#                     h_nucc = h_nucc_upg
#                     h_nubarcc = h_nubarcc_upg
#                 else:
#                     h_nucc = h_nucc_oth
#                     h_nubarcc = h_nubarcc_oth

                if not nubar:
                    res[mask_upg] = h_nucc_upg[ind_x[mask_upg]-1, ind_y[mask_upg]-1]
                    res[~mask_upg] = h_nucc_oth[ind_x[~mask_upg]-1, ind_y[~mask_upg]-1]
                else:
                    res[mask_upg] = h_nubarcc_upg[ind_x[mask_upg]-1, ind_y[mask_upg]-1]
                    res[~mask_upg] = h_nubarcc_oth[ind_x[~mask_upg]-1, ind_y[~mask_upg]-1]

                return res

            # create empty containers for weight corrections
            self.data.representation = self.apply_mode
            for container in self.data:

                # reweighting all CC events
                if container.name.endswith('_cc'):

                    container["charm_y_distr_corr"] = np.ones(container.size, dtype=FTYPE)

                    true_lg_energy = np.log10(container["true_energy"])
                    true_y = container['bjorken_y']
                    true_coszen = container['true_coszen']

                    apply_mask = true_y >= 0
#                     dis_mask = container['dis'] > 0
#                     apply_mask = apply_mask * dis_mask
                    container["apply_mask"] = apply_mask

                    lgE_min = 0 #2. 
                    valid_mask = true_lg_energy >= lgE_min
                    extrp_mask = ~valid_mask

                    valid_mask = valid_mask * apply_mask
                    extrp_mask = extrp_mask * apply_mask

                    # storing initial true inelasticity distribution 
                    # will later use to divide event weights by it
                    if container.name in ['nue_cc', 'numu_cc', 'nutau_cc']:
                        is_nubar = False
                    elif container.name in ['nuebar_cc', 'numubar_cc', 'nutaubar_cc']:
                        is_nubar = True
                    else:
                        raise ValueError('Incorrect container type "%s"' % container.name)

                    distr_valid_erange = eval_hist(true_lg_energy[valid_mask],
                                                   true_y[valid_mask], true_coszen[valid_mask], nubar=is_nubar)
                    distr_extrap_erange = eval_hist(np.ones_like(true_y[extrp_mask])*lgE_min,
                                                    true_y[extrp_mask], true_coszen[extrp_mask], nubar=is_nubar)

                    container["charm_y_distr_corr"][valid_mask] = distr_valid_erange
                    container["charm_y_distr_corr"][extrp_mask] = distr_extrap_erange 

    def apply_function(self):
        # modify weights
        if self.correct_charm:
            for container in self.data:
                # reweighting all CC events
                if container.name.endswith('_cc'):
                    modif_weights = container["weights"] * container["charm_y_distr_corr"]
                    container["weights"] = modif_weights

                    
