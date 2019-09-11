'''
Oscillation stage using nuSQuIDS
'''

# TODO Check if can speed up by linking containers in certain modes (see `pi_prob3`)
# TODO Update descriptions/docs

from __future__ import absolute_import, print_function, division

# TODO Clean these up, including numba

import math
import numpy as np
from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.stages.osc.pi_osc_params import OscParams
from pisa.stages.osc.layers import Layers
from pisa.stages.osc.prob3numba.numba_osc import propagate_array, fill_probs
from pisa.utils.numba_tools import WHERE
from pisa.stages.osc.pi_prob3 import apply_probs

from pisa.utils.resources import find_resource
from pisa.utils import vectorizer
from pisa.stages.osc.nusquids.nusquids_osc import NSQ_CONST, validate_calc_grid, compute_binning_constants, init_nusquids_prop, evolve_states, osc_probs, earth_model
from pisa import ureg


from scipy.interpolate import RectBivariateSpline

__all__ = ['pi_nusquids']

__author__ = 'T. Stuttard, T. Ehrhardt'


#TODO Probably just delete this...
#TODO Make into dedicated file, and document
#TODO Is this really worth having, or can I just implement directly in the code (e.g. is this just not possible to make general enough)?
#Make can make a geeneral E_coszen_spline_tool? The  also in e.g. flux?
'''
class OscSpline() :

    def __init__(self,energy_nodes,coszen_nodes) :
        #TODO Check inputs
        self.energy_nodes = energy_nodes
        self.coszen_nodes = coszen_nodes #TODO private?
        self.prob_e_buffer = np.full( self.shape, np.NaN )
        self.prob_mu_buffer = np.full( self.shape, np.NaN )

    @property
    def shape(self) :
        return ( len(self.energy_nodes), len(self.coszen_nodes))

    def generate_spline(self,prob_vs_energy_coszen_grid) :
        assert prob_vs_energy_coszen_grid.shape == self.shape, "Probability grid must have shape matching the energy-coszen grid nodes"
        return RectBivariateSpline( self.energy_nodes, self.coszen_nodes, prob_vs_energy_coszen_grid )
'''


class pi_nusquids(PiStage):
    """
    PISA Pi stage for weighting events due to the effect of neutrino oscillations,
    using nuSQuIDS as the oscillation probability calculator.

    Parameters
    ----------

    Uses the standard parameters as required by a PISA pi stage (see `pisa/core/pi_stage.py`)

    use_decoherence : bool
        set to true to include neutrino decoherence in the oscillation probability calculation
    num_decoherence_gamma : int
        number of decoherence gamma parameters to be considered in the decoherence model
        must be ether 1 or 3

    use_nsi : bool
        set to true to include Non-Standard Interactions (NSI) in the oscillation probability calculation

    num_neutrinos=3,
        use_spline=False,
                ):

    Expected contents of `params` ParamSet:
        detector_depth : float
        earth_model : PREM file path
        prop_height : quantity (dimensionless)
        YeI : quantity (dimensionless)
        YeO : quantity (dimensionless)
        YeM : quantity (dimensionless)
        theta12 : quantity (angle)
        theta13 : quantity (angle)
        theta23 : quantity (angle)
        deltam21 : quantity (mass^2)
        deltam31 : quantity (mass^2)
        deltacp : quantity (angle)
        rel_err : quantity (dimensionless)
        abs_err : quantity (dimensionless)

    Additional ParamSet params expected when using the `use_nsi` argument:
        TODO

    Additional ParamSet params expected when using the `use_decoherence` argument:
        n_energy : quantity (dimensionless)
        * If using `num_decoherence_gamma` == 1:
            gamma : quantity (energy)
        * If using `num_decoherence_gamma` == 3:
            gamma12 : quantity (energy)
            gamma13 : quantity (energy)
            gamma23 : quantity (energy)

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
                 use_decoherence=False,
                 num_decoherence_gamma=1,
                 use_nsi=False,
                 num_neutrinos=3,
                 use_spline=False,
                ):

        self.num_neutrinos = num_neutrinos
        self.use_nsi = use_nsi
        self.use_decoherence = use_decoherence
        self.use_spline = use_spline
        self.num_decoherence_gamma = num_decoherence_gamma

        # Define standard params
        expected_params = ['detector_depth',
                           'earth_model',
                           'prop_height',
                           'YeI',
                           'YeO',
                           'YeM',
                           'theta12',
                           'theta13',
                           'theta23',
                           'deltam21',
                           'deltam31',
                           'deltacp',
                           'rel_err',
                           'abs_err',
                          ]

        # Add decoherence parameters
        assert self.num_decoherence_gamma in [1,3], "Must choose either 1 or 3 decoherence gamma parameters"
        if self.use_decoherence :
            if self.num_decoherence_gamma == 1 :
                expected_params.extend(['gamma'])
            elif self.num_decoherence_gamma == 3 :
                expected_params.extend(['gamma21',
                                        'gamma31',
                                        'gamma32'])
            expected_params.extend(['n_energy'])

        # Add NSI parameters
        #TODO

        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_apply_keys = ('weights',
                            'sys_flux',
                           )
        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ('prob_e',
                            'prob_mu',
                           )
        # what keys are added or altered for the outputs during apply
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

        assert self.num_neutrinos == 3, "Only 3-flavor oscillations implemented right now" # TODO Add interface to nuSQuIDS 3+N handling
        assert self.use_nsi == False, "NSI support not yet implemented" # TODO
        assert not (self.use_nsi and self.use_decoherence), "NSI and decoherence not suported together, must use one or the other"

        assert self.input_mode is not None # TODO Need to test binned mode
        assert self.calc_mode == 'binned', "Must use a grid-based calculation for nuSQuIDS"
        assert self.output_mode is not None # TODO Need to test binned mode

        # Define new specs here for the points we evaluate probabilties at in nuSQuIDS
        # This is a bit different to a standard PISA stage since nuSQuIDS interally calcuates on a grid (which we set via calc_specs) 
        # and then returns interpolated values each time we evaluate
        self.eval_specs = self.input_specs #TODO Should this be output_specs?

        assert self.params.earth_model.value is not None, "Vacuum oscillations not currently supported when using nuSQuIDS with PISA"


    def setup_function(self):

        # set the correct data mode
        self.data.data_specs = self.eval_specs

        # check the calc binning
        validate_calc_grid(self.calc_specs)
        #TODO Check grid encompasses all events... (maybe needs to be in `compute_function`)

        # pad the grid to make sure we can later on evaluate osc. probs.
        # *anywhere* in between of the outermost bin edges
        self.en_calc_grid, self.cz_calc_grid = compute_binning_constants(self.calc_specs) #TODO Check what this actually does, and if I need it

        #TODO enforce all events within grid

        # set up initial states, get the nuSQuIDS "propagator" instances (one per flavor)
        self.ini_states, self.props = init_nusquids_prop(
            cz_nodes=self.cz_calc_grid,
            en_nodes=self.en_calc_grid,
            nu_flav_no=self.num_neutrinos,
            rel_err=self.params.rel_err.value.m_as('dimensionless'),
            abs_err=self.params.abs_err.value.m_as('dimensionless'),
            progress_bar=False,
            use_nsi=self.use_nsi,
            use_decoherence=self.use_decoherence,
        )

        # make an Earth model  #TODO handle vacuum option
        self.earth_atm = earth_model(YeI=self.params.YeI.value.m_as('dimensionless'), 
                                    YeM=self.params.YeM.value.m_as('dimensionless'), 
                                    YeO=self.params.YeO.value.m_as('dimensionless'),
                                    PREM_file=self.params.earth_model.value)

        #TODO Need to take prop_height and detector_depth into account

        # create oscillation parameter value holder (values actually set later)
        self.osc_params = OscParams()

        # setup empty arrays to hold the calculated probabilities #TODO SHould these be in the linked containers?
        for container in self.data:
            container['prob_e'] = np.full((container.size), np.NaN, dtype=FTYPE)
            container['prob_mu'] = np.full((container.size), np.NaN, dtype=FTYPE)

        
    @profile
    def compute_function(self):

        # set the correct data mode
        self.data.data_specs = self.eval_specs

        # update osc params
        self.osc_params.theta12 = self.params.theta12.value.m_as('rad')
        self.osc_params.theta13 = self.params.theta13.value.m_as('rad')
        self.osc_params.theta23 = self.params.theta23.value.m_as('rad')
        self.osc_params.dm21 = self.params.deltam21.value.m_as('eV**2')
        self.osc_params.dm31 = self.params.deltam31.value.m_as('eV**2')
        self.osc_params.deltacp = self.params.deltacp.value.m_as('rad')

        # update osc params specific to decoherence
        if self.use_decoherence :
            if self.num_decoherence_gamma == 1 :
                self.osc_params.gamma21 = self.params.gamma.value.m_as('eV')
                self.osc_params.gamma31 = self.params.gamma.value.m_as('eV')
                self.osc_params.gamma32 = self.params.gamma.value.m_as('eV')
            elif self.num_decoherence_gamma == 3 :
                self.osc_params.gamma21 = self.params.gamma21.value.m_as('eV')
                self.osc_params.gamma31 = self.params.gamma31.value.m_as('eV')
                self.osc_params.gamma32 = self.params.gamma32.value.m_as('eV')
            self.osc_params.n_energy = self.params.n_energy.m_as('dimensionless')

        
        # TODO sterile params
        '''
        self.osc_params.theta14 = np.deg2rad(0.0)
        self.osc_params.dm41 = 0.
        '''

        # TODO NSI params
        '''
        self.osc_params.eps_ee = 0.
        self.osc_params.eps_emu = 0.
        self.osc_params.eps_etau = 0.
        self.osc_params.eps_mumu = 0.
        self.osc_params.eps_mutau = 0.005
        self.osc_params.eps_tautau = 0.
        '''

        # evolve the states starting from initial ones, using the current state of the params
        evolve_states(
            cz_shape=self.cz_calc_grid.shape[0],
            propagators=self.props,
            ini_states=self.ini_states, # TODO Check these are not changed during `compute_function`
            nsq_earth_atm=self.earth_atm,
            osc_params=self.osc_params
        )

        # Loop over containers
        for container in self.data:

            # get the event energy and coszen values in nuSQuIDS units
            en_nusq = container["true_energy"].get(WHERE) * NSQ_CONST.GeV # GeV -> eV
            cz_nusq = container["true_coszen"].get(WHERE) # No conversion required

            # define the points where osc. probs. are to be evaluated in nuSQuIDS
            # this is either the events themselves, or on a grid if using a spline
            # this is just the energy/coszen values for each event, in the correct units
            if self.use_spline : #TODO once only
                en_eval_ax,cz_eval_ax = self.en_calc_grid,self.cz_calc_grid
                en_eval_grid, cz_eval_grid = np.meshgrid(en_eval_ax,cz_eval_ax,indexing="ij")
                en_eval,cz_eval = en_eval_grid.ravel(), cz_eval_grid.ravel()
            else :
                en_eval, cz_eval = en_nusq, cz_nusq

            # get the neutrino flavor (ignore the interaction)
            nuflav = container.name.replace("_cc","").replace("_nc","") # TODO Update this once we have the new events class which has helper functions for this kind of thing

            # Define the output arrays for the calculated oscillation probability.
            # Can directly use container arrays whe calculating event-wise, or need a 
            # buffer with one element per grid point if using a spline.
            if self.use_spline :
                prob_e_buff,prob_mu_buff = np.full_like(en_eval,np.NaN), np.full_like(cz_eval,np.NaN)
            else :
                prob_e_buff,prob_mu_buff = container['prob_e'].get(WHERE), container['prob_mu'].get(WHERE)

            # Get the oscillation probs, writing them to the container
            osc_probs(  # pylint: disable=unused-variable
                nuflav=nuflav, 
                propagators=self.props,
                true_energies=en_eval,
                true_coszens=cz_eval,
                prob_e=prob_e_buff,
                prob_mu=prob_mu_buff,
            )

            # If using splines, create the spline using the probabilitis computed at each grid point, and evaluate for each event
            if self.use_spline :
                prob_e_spline = RectBivariateSpline( en_eval_ax, cz_eval_ax, prob_e_buff.reshape(en_eval_grid.shape) )
                np.copyto( src=prob_e_spline.ev(en_nusq,cz_nusq), dst=container['prob_e'].get(WHERE) )
                prob_mu_spline = RectBivariateSpline( en_eval_ax, cz_eval_ax, prob_mu_buff.reshape(en_eval_grid.shape) )
                np.copyto( src=prob_mu_spline.ev(en_nusq,cz_nusq), dst=container['prob_mu'].get(WHERE) )

            container['prob_e'].mark_changed(WHERE)
            container['prob_mu'].mark_changed(WHERE)


    @profile
    def apply_function(self):

        # update the outputted weights
        for container in self.data:
            apply_probs(container['sys_flux'].get(WHERE),
                        container['prob_e'].get(WHERE),
                        container['prob_mu'].get(WHERE),
                        out=container['weights'].get(WHERE))
            container['weights'].mark_changed(WHERE)
