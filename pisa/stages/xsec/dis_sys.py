"""
Stage to apply pre-calculated DIS uncertainties
Study done by Maria Liubarska & Juan Pablo Yanez, more information available here:
https://drive.google.com/open?id=1SRBgIyX6kleYqDcvop6m0SInToAVhSX6
ToDo: tech note being written, link here as soon as available
"""

from __future__ import absolute_import, print_function, division

__all__ = ["dis_sys", "apply_dis_sys"]

import numpy as np
from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.profiler import profile
from pisa.utils.fileio import from_file
from pisa.utils.numba_tools import WHERE
from pisa import ureg


class dis_sys(Stage): # pylint: disable=invalid-name
    """
    Stage to apply pre-calculated DIS systematics.

    Parameters
    ----------
    data
    params
        Must contain ::
            dis_csms : quantity (dimensionless)

    extrapolation_type : string
        choice of ['constant', 'linear', 'higher']

    extrapolation_energy_threshold : float
        Below what energy (in GeV) to extrapolate
        Defaults to 100. CSMS not considered reliable below 50-100 GeV

    Notes
    -----
    Requires the events have the following keys ::
        true_energy
            Neutrino energy in GeV
        bjorken_y
            Inelasticity
        dis
            1 if event is DIS, else 0

    """
    def __init__(
        self,
        extrapolation_type='constant',
        extrapolation_energy_threshold=100*ureg["GeV"],
        **std_kwargs,
    ):
        expected_params = (
            'dis_csms',
        )

        # init base class
        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

        self.extrapolation_type = extrapolation_type
        self.extrapolation_energy_threshold = extrapolation_energy_threshold

    @profile
    def setup_function(self):

        extrap_dict = from_file('cross_sections/tot_xsec_corr_Q2min1_isoscalar.pckl')

        # load splines
        wf_nucc = from_file('cross_sections/dis_csms_splines_flat_no_nucl_corr/NuMu_CC_flat.pckl')
        wf_nubarcc = from_file('cross_sections/dis_csms_splines_flat_no_nucl_corr/NuMu_Bar_CC_flat.pckl')
        wf_nunc = from_file('cross_sections/dis_csms_splines_flat_no_nucl_corr/NuMu_NC_flat.pckl')
        wf_nubarnc = from_file('cross_sections/dis_csms_splines_flat_no_nucl_corr/NuMu_Bar_NC_flat.pckl')

        # set this to events mode, as we need the per-event info to calculate these weights
        self.data.representation = 'events'

        lgE_min = np.log10(self.extrapolation_energy_threshold.m_as("GeV"))

        for container in self.data:

            # creat keys for external dict

            if container.name.endswith('_cc'):
                current = 'CC'
            elif container.name.endswith('_nc'):
                current = 'NC'
            else:
                raise ValueError('Can not determine whether container with name "%s" is pf type CC or NC based on its name'%container.name)
            nu = 'Nu' if container['nubar'] > 0 else 'NuBar'

            lgE = np.log10(container['true_energy'])
            bjorken_y = container['bjorken_y']
            dis = container['dis']

            w_tot = np.ones_like(lgE)

            valid_mask = lgE >= lgE_min
            extrapolation_mask = ~valid_mask

            #
            # Calculate variation of total cross section
            #

            poly_coef = extrap_dict[nu][current]['poly_coef']
            lin_coef = extrap_dict[nu][current]['linear']

            if self.extrapolation_type == 'higher':
                w_tot = np.polyval(poly_coef, lgE)
            else:
                w_tot[valid_mask] = np.polyval(poly_coef, lgE[valid_mask])

                if self.extrapolation_type == 'constant':
                    w_tot[extrapolation_mask] = np.polyval(poly_coef, lgE_min)  # note Numpy broadcasts
                elif self.extrapolation_type == 'linear':
                    w_tot[extrapolation_mask] = np.polyval(lin_coef, lgE[extrapolation_mask])
                else:
                    raise ValueError('Unknown extrapolation type "%s"'%self.extrapolation_type)

            # make centered arround 0, and set to 0 for all non-DIS events
            w_tot = (w_tot - 1) * dis
          
            container["dis_correction_total"] = w_tot
            container.mark_changed('dis_correction_total')

            #
            # Calculate variation of differential cross section
            #

            w_diff = np.ones_like(lgE)

            if current == 'CC' and container['nubar'] > 0:
                weight_func = wf_nucc
            elif current == 'CC' and container['nubar'] < 0:
                weight_func = wf_nubarcc
            elif current == 'NC' and container['nubar'] > 0:
                weight_func = wf_nunc
            elif current == 'NC' and container['nubar'] < 0:
                weight_func = wf_nubarnc

            w_diff[valid_mask] = weight_func.ev(lgE[valid_mask], bjorken_y[valid_mask])
            w_diff[extrapolation_mask] = weight_func.ev(lgE_min, bjorken_y[extrapolation_mask])

            # make centered arround 0, and set to 0 for all non-DIS events
            w_diff = (w_diff - 1) * dis
            
            container["dis_correction_diff"] = w_diff
            container.mark_changed('dis_correction_diff')
         
    @profile
    def apply_function(self):
        dis_csms = self.params.dis_csms.m_as('dimensionless')

        for container in self.data:
            apply_dis_sys(
                container['dis_correction_total'],
                container['dis_correction_diff'],
                FTYPE(dis_csms),
                out=container['weights'],
            )
            container.mark_changed('weights')


FX = 'f8' if FTYPE == np.float64 else 'f4'

@guvectorize([f'({FX}, {FX}, {FX}, {FX}[:])'], '(),(),()->()', target=TARGET)
def apply_dis_sys(
    dis_correction_total,
    dis_correction_diff,
    dis_csms,
    out,
):
    out[0] *= max(0, (1. + dis_correction_total * dis_csms) * (1. + dis_correction_diff * dis_csms) )
