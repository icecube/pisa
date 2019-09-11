"""
PISA pi stage for the calculation of earth layers and survival probabilities.

The stage calculates first the depth of a water column that is mass-equivalent
to the path traversed by the neutrino through the earth. This is done
using the same Layers module that is also used for oscillation.
The survival probability is then calculated from the average cross-section
with protons and neutrons.
"""
from __future__ import absolute_import, print_function, division

import numpy as np

from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa import ureg
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.stages.osc.layers import Layers
from pisa.utils.numba_tools import WHERE
from pisa.utils import vectorizer
from pisa.utils.resources import find_resource

FLAV_BAR_STR_MAPPING = {
    (0, -1): "e_bar",
    (0, +1): "e",
    (1, -1): "mu_bar",
    (1, +1): "mu",
    (2, -1): "tau_bar",
    (2, +1): "tau",
}
"""
Mapping from flav and nubar container content to
the string for this neutrino in the ROOT file.
"""
        
class pi_earth_absorption(PiStage):
    """
    earth absorption PISA Pi class

    Paramaters
    ----------
    earth_model : str
        PREM file path
    xsec_file : str
        path to ROOT file containing cross-sections
    detector_depth : quantity (distance), optional
        detector depth
    prop_height : quantity (distance), optional
        height of neutrino production in the atmosphere
    
    Notes
    -----
    
    """
    def __init__(self,
                 earth_model,
                 xsec_file,
                 data=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                 detector_depth=2.*ureg.km,
                 prop_height=20.*ureg.km
                ):

        expected_params = ()
        input_names = ()
        output_names = ()

        input_apply_keys = ('weights',
                           )
        # The weights are simply scaled by the earth survival probability
        output_calc_keys = ('survival_prob',
                           )
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
            input_specs=input_specs,
            calc_specs=calc_specs,
            output_specs=output_specs,
            input_apply_keys=input_apply_keys,
            output_calc_keys=output_calc_keys,
            output_apply_keys=output_apply_keys,
        )

        assert self.input_mode is not None
        assert self.calc_mode is not None
        assert self.output_mode is not None

        self.layers = None
        self.xsroot = None
        self.earth_model = earth_model
        self.xsec_file = xsec_file
        self.detector_depth = detector_depth.m_as('km')
        self.prop_height = prop_height.m_as('km')
        # this does nothing for speed, but makes for convenient numpy style broadcasting
        # TODO: Use numba vectorization (not sure how that works with splines)
        self.calculate_xsections = np.vectorize(self.calculate_xsections)  
        
    def setup_function(self):
        import ROOT
        # setup the layers
        earth_model = find_resource(self.earth_model)
        self.layers = Layers(earth_model, self.detector_depth, self.prop_height)
        # This is a bit hacky, but setting the electron density to 1. 
        # gives us the total density of matter, which is what we want.
        self.layers.setElecFrac(1., 1., 1.)
        
        # setup cross-sections
        self.xsroot = ROOT.TFile(self.xsec_file)
        # set the correct data mode
        self.data.data_specs = self.calc_specs

        # --- calculate the layers ---
        if self.calc_mode == 'binned':
            # layers don't care about flavor
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc',
                                             'nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                             'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])

        for container in self.data:
            self.layers.calcLayers(container['true_coszen'].get(WHERE))
            container['densities'] = self.layers.density.reshape((container.size, self.layers.max_layers))
            container['distances'] = self.layers.distance.reshape((container.size, self.layers.max_layers))
            container['rho_int'] = np.empty((container.size), dtype=FTYPE)
        # don't forget to un-link everything again
        self.data.unlink_containers()

        # --- setup cross section and survival probability --- 
        if self.calc_mode == 'binned':
            # The cross-sections do not depend on nc/cc, so we can at least link those containers
            self.data.link_containers('nue', ['nue_cc', 'nue_nc'])
            self.data.link_containers('nuebar', ['nuebar_cc', 'nuebar_nc'])
            self.data.link_containers('numu', ['numu_cc', 'numu_nc'])
            self.data.link_containers('numubar', ['numubar_cc', 'numubar_nc'])
            self.data.link_containers('nutau', ['nutau_cc', 'nutau_nc'])
            self.data.link_containers('nutaubar', ['nutaubar_cc', 'nutaubar_nc'])
        for container in self.data:
            container['xsection'] = np.empty((container.size), dtype=FTYPE)
            container['survival_prob'] = np.empty((container.size), dtype=FTYPE)
        self.data.unlink_containers()
    
    @profile
    def compute_function(self):
        # --- calculate the integrated density in the layers ---
        if self.calc_mode == 'binned':
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc',
                                             'nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                             'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])

        for container in self.data:
            calculate_integrated_rho(container['distances'].get(WHERE),
                                     container['densities'].get(WHERE),
                                     out=container['rho_int'].get(WHERE)
                                    )
        # don't forget to un-link everything again
        self.data.unlink_containers()

        # --- calculate survival probability --- 
        if self.calc_mode == 'binned':
            # The cross-sections do not depend on nc/cc, so we can at least link those containers
            self.data.link_containers('nue', ['nue_cc', 'nue_nc'])
            self.data.link_containers('nuebar', ['nuebar_cc', 'nuebar_nc'])
            self.data.link_containers('numu', ['numu_cc', 'numu_nc'])
            self.data.link_containers('numubar', ['numubar_cc', 'numubar_nc'])
            self.data.link_containers('nutau', ['nutau_cc', 'nutau_nc'])
            self.data.link_containers('nutaubar', ['nutaubar_cc', 'nutaubar_nc'])
        for container in self.data:
            container['xsection'] = self.calculate_xsections(container['flav'],
                                                             container['nubar'],
                                                             container['true_energy'].get(WHERE)
                                                            )
            calculate_survivalprob(container['rho_int'].get(WHERE),
                                   container['xsection'].get(WHERE),
                                   out=container['survival_prob'].get(WHERE)
                                  )
            container['survival_prob'].mark_changed(WHERE)
        self.data.unlink_containers()
      
    @profile
    def apply_function(self):
        for container in self.data:
            vectorizer.multiply(container['survival_prob'], out=container['weights'])
            
    def calculate_xsections(self, flav, nubar, energy):
        '''Calculates the cross-sections on isoscalar targets.
        The result is returned in cm^2. The xsection on one 
        target is calculated by taking the xsection for O16
        and dividing it by 16. 
        '''
        flavor = FLAV_BAR_STR_MAPPING[(flav, nubar)]
        return (self.xsroot.Get('nu_'+flavor+'_O16').Get('tot_cc').Eval(energy)+
                self.xsroot.Get('nu_'+flavor+'_O16').Get('tot_nc').Eval(energy))*10**(-38)/16. # this gives cm^2

signatures = [
    '(f4[::1], f4[::1], f4[::1])',
    '(f8[::1], f8[::1], f8[::1])'
]

# TODO: make this work with the 'cuda' target. Right now, it seems like np.dot
# does not work or is used incorrectly.
@guvectorize(signatures, '(n),(n)->()', target=TARGET)
def calculate_integrated_rho(layer_dists, layer_densities, out):
    """Calculate density integrated over the path through all layers.
    Gives the length of a matter-equivalent water column in cm.
    
    Parameters
    ----------
    layer_dists : vector
        distance travelled through each layer
    layer_densities : vector
        densities of the layers
    out : scalar
        Result is stored here

    """
    out[0] = np.dot(layer_dists, layer_densities)*1e5 #distances are converted from km to cm

@guvectorize(signatures, '(),()->()', target=TARGET)
def calculate_survivalprob(int_rho, xsection, out):
    """Calculate survival probability given layer distances,
    layer densities and (pre-computed) cross-sections.
    
    Parameters
    ----------
    int_rho : scalar
        depth of mass equivalent water column in cm
    xsection : scalar
        cross-section per nucleon in cm^2
    out : scalar
        Result is stored here

    """
    Na = 6.022E23 # nuclei per cm^(-3)
    # The molar mass is 1 g for pure nuclei, so there is a hidden division
    # here by 1 g/mol. Also, int_rho is the depth of a mass equivalent
    # water column, where water has the density of 1 g/cm^3.
    # So the units work out to:
    # int_rho [cm] * 1 [g/cm^3] * xsection [cm^2] * 1 [mol/g] * Na [1/mol] = [ 1 ] (all units cancel)
    out[0] = np.exp(-int_rho[0]*xsection[0]*Na)

