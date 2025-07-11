"""
This stage corrects inelasticity distribution 
to remove the most obvious impact from charm 
production bug in GENIE.

Maria Liubarska
"""

import numpy as np
from pisa.core.stage import Stage
from pisa.utils.fileio import from_file
from pisa import FTYPE

class correct_charm_y(Stage):
    """
    This stage reweights inelasticity distribution
    to remove the most obvious impact from charm
    production bug in GENIE 2.12.8. The bug effectively 
    kills most charm events.

    Parameters
    ----------
    nu_cc_container_keys : list of strings
        list of all neutrino charged current container keys

    nubar_cc_container_keys : list of strings
        list of all anti-neutrino charged current container keys

    Notes:
    ------
    THIS STAGE IS FOR MC TESTING ONLY - do *NOT* use in real analysis!!!


    References
    ----------
    Slides summarizing the bug (osc. call, 28 Nov 2022): 
    https://drive.google.com/file/d/16tF_ofjI-YTxnxSsuZD4W2X52xND1CAM/view?usp=drive_link

    Slides showing use for check with oscNext VS (osc. call, 19 Dec 2022): 
    https://drive.google.com/file/d/14h0WYWPWn7yS7wLQpyMAPTenfKfFAq3k/view?usp=drive_link

    """
    def __init__(
            self,
            nu_cc_container_keys = ['nue_cc', 'numu_cc', 'nutau_cc'],
            nubar_cc_container_keys = ['nuebar_cc', 'numubar_cc', 'nutaubar_cc'],
            **std_kwargs,
    ):

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
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )

        self.nucc_keys = nu_cc_container_keys
        self.nubarcc_keys = nubar_cc_container_keys
        self.all_cc_keys = self.nucc_keys + self.nubarcc_keys

    def setup_function(self):

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

            ind_x[ind_x==31] = 30
            ind_y[ind_y==31] = 30

            res = np.zeros_like(lgE)
            mask_upg = coszen<-0.9

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

            # reweighting all (anti-)neutrino CC events
            if container.name in self.nucc_keys:
                is_nubar = False
            elif container.name in self.nubarcc_keys:
                is_nubar = True
            else:
                continue

            container["charm_y_distr_corr"] = np.ones(container.size, dtype=FTYPE)

            true_lg_energy = np.log10(container["true_energy"])
            true_y = container['bjorken_y']
            true_coszen = container['true_coszen']

            apply_mask = true_y >= 0
            container["apply_mask"] = apply_mask

            lgE_min = 0
            valid_mask = true_lg_energy >= lgE_min
            extrp_mask = ~valid_mask

            valid_mask = valid_mask * apply_mask
            extrp_mask = extrp_mask * apply_mask

            # storing initial true inelasticity distribution 
            # will later use to divide event weights by it
            distr_valid_erange = eval_hist(true_lg_energy[valid_mask],
                                           true_y[valid_mask], true_coszen[valid_mask], nubar=is_nubar)
            distr_extrap_erange = eval_hist(np.ones_like(true_y[extrp_mask])*lgE_min,
                                            true_y[extrp_mask], true_coszen[extrp_mask], nubar=is_nubar)

            container["charm_y_distr_corr"][valid_mask] = distr_valid_erange
            container["charm_y_distr_corr"][extrp_mask] = distr_extrap_erange 

    def apply_function(self):
        # modify weights
        for container in self.data:
            # reweighting all (anti-)neutrino CC events
            if container.name in self.all_cc_keys:
                modif_weights = container["weights"] * container["charm_y_distr_corr"]
                container["weights"] = modif_weights

def init_test(**param_kwargs):
    """Instantiation example that enables actual testing on toy containers from test_services.py"""
    return correct_charm_y(
        nu_cc_container_keys=["test1_cc"],
        nubar_cc_container_keys=["test2_nc"]
    )
        
