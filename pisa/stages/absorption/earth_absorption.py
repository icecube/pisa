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
import math

from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa import ureg
from pisa.core.stage import Stage
from pisa.utils.profiler import profile
from pisa.stages.osc.layers import Layers
from pisa.utils.resources import find_resource

__author__ = 'A. Trettin'

__license__ = '''Copyright (c) 2020, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''

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


class earth_absorption(Stage):
    """
    earth absorption PISA Pi class

    Parameters
    ----------
    earth_model : str
        PREM file path
    xsec_file : str
        path to ROOT file containing cross-sections
    detector_depth : quantity (distance), optional
        detector depth
    prop_height : quantity (distance), optional
        height of neutrino production in the atmosphere

    """
    def __init__(
        self,
        earth_model,
        xsec_file,
        detector_depth=2.*ureg.km,
        prop_height=20.*ureg.km,
        **std_kwargs,
    ):

        # init base class
        super().__init__(
            expected_params=(),
            **std_kwargs,
        )


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
        self.data.representation = self.calc_mode

        # --- calculate the layers ---
        if self.data.is_map:
            # layers don't care about flavor
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc',
                                             'nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                             'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])

        for container in self.data:
            self.layers.calcLayers(container['true_coszen'])
            container['densities'] = self.layers.density.reshape((container.size, self.layers.max_layers))
            container['distances'] = self.layers.distance.reshape((container.size, self.layers.max_layers))
            container['rho_int'] = np.empty((container.size), dtype=FTYPE)
            
            container.mark_changed('densities')
            container.mark_changed('distances')
            container.mark_changed('rho_int')
        # don't forget to un-link everything again
        self.data.unlink_containers()

        # --- setup cross section and survival probability ---
        if self.data.is_map:
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
        if self.data.is_map:
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc',
                                             'nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                             'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])

        for container in self.data:
            calculate_integrated_rho(container['distances'],
                                     container['densities'],
                                     out=container['rho_int']
                                    )
            container.mark_changed('rho_int')
        # don't forget to un-link everything again
        self.data.unlink_containers()

        # --- calculate survival probability ---
        if self.data.is_map:
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
                                                             container['true_energy']
                                                            )
            container.mark_changed('xsection')
            calculate_survivalprob(container['rho_int'],
                                   container['xsection'],
                                   out=container['survival_prob']
                                  )
            container.mark_changed('survival_prob')
        self.data.unlink_containers()

    def apply_function(self):
        for container in self.data:
            container['weights'] *= container['survival_prob']

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
    out[0] = 0
    for i in range(len(layer_dists)):
        out[0] += layer_dists[i]*layer_densities[i]
    out[0] *= 1e5  # distances are converted from km to cm

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
    out = np.exp(-int_rho*xsection*Na)

