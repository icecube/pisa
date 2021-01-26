"""
A stage to apply nutau cross-section uncertainties as implemented in
https://github.com/marialiubarska/nutau_xsec
It interpolates between different nutau CC cross section models as compared in this 
paper:
https://arxiv.org/pdf/1008.2984.pdf?fname=cm&font=TypeI
"""

import numpy as np
import pickle
from numba import guvectorize

from pisa.core.stage import Stage
from pisa.utils.resources import open_resource
from pisa.utils import vectorizer
from pisa import FTYPE, TARGET
from pisa.utils.numba_tools import WHERE


class nutau_xsec(Stage):
    """
    Nu_tau cross-section correction to interpolate between different nutau CC
    cross-section models. This requires the interpolated file produced by 
    Maria Liubarska: https://github.com/marialiubarska/nutau_xsec

    Parameters
    ----------
    xsec_file : (string)
        Path to pickled interpolated function. Default is included in PISA in 
        `pisa_examples/resources/cross_sections/interp_nutau_xsec_protocol2.pckl`

    params : ParamSet or sequence with which to instantiate a ParamSet.
        Expected params .. ::

            scale : quantity (dimensionless)
                Scaling between different cross-section models. The range [-1, 1]
                covers all models tested in the paper.

    """
    def __init__(
        self,
        xsec_file="cross_sections/interp_nutau_xsec_protocol2.pckl",
        **std_kwargs,
    ):

        expected_params = ("nutau_xsec_scale")

        # init base class
        super(nutau_xsec, self).__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

        self.xsec_file = xsec_file
    
    def setup_function(self):
        with open_resource(self.xsec_file, mode="rb") as fl:
            interp_dict = pickle.load(fl, encoding='latin1')
        interp_nutau = interp_dict["NuTau"]
        interp_nutaubar = interp_dict["NuTauBar"]
        
        self.data.representation = self.calc_mode
        for container in self.data:
            if container.name == "nutau_cc":
                energy = container["true_energy"]
                func = interp_nutau(energy)
                # Invalid values of the function occur below the tau production 
                # threshold. For those values, we put in negative infinity, which will
                # cause them to be clamped to zero when the weights are calculated.
                func[~np.isfinite(func)] = -np.inf
                container["nutau_xsec_func"] = func
            elif container.name == "nutaubar_cc":
                energy = container["true_energy"]
                func = interp_nutaubar(energy)
                func[~np.isfinite(func)] = -np.inf
                container["nutau_xsec_func"] = func

        self.data.representation = self.apply_mode
        for container in self.data:
            if container.name in ["nutau_cc", "nutaubar_cc"]:
                container["nutau_xsec_scale"] = np.empty(container.size, dtype=FTYPE)

    def compute_function(self):
        scale = self.params.nutau_xsec_scale.value.m_as('dimensionless')
        for container in self.data:
            if container.name in ["nutau_cc", "nutaubar_cc"]:
                calc_scale_vectorized(
                    container["nutau_xsec_func"],
                    FTYPE(scale),
                    out=container["nutau_xsec_scale"]
                )
    
    def apply_function(self):
        for container in self.data:
            if container.name in ["nutau_cc", "nutaubar_cc"]:
                container["weights"] *= container["nutau_xsec_scale"]

# vectorized function to calculate 1 + f(E)*scale
# must be outside class
if FTYPE == np.float64:
    FX = 'f8'
    IX = 'i8'
else:
    FX = 'f4'
    IX = 'i4'
signature = f'({FX}[:], {FX}, {FX}[:])'
@guvectorize([signature], '(),()->()', target=TARGET)
def calc_scale_vectorized(func, scale, out):
    # weights that would come out negative are clamped to zero
    if func[0] * scale > -1.:
        out[0] = 1. + func[0] * scale
    else:
        out[0] = 0.

                
