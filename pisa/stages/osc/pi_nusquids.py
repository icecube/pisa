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
from pisa.utils.resources import find_resource
from pisa.stages.osc.nusquids.nusquids_osc import NSQ_CONST, validate_calc_grid, compute_binning_constants, init_nusquids_prop, evolve_states, osc_probs, earth_model
from pisa import ureg


__all__ = ['pi_nusquids']

__author__ = 'T. Stuttard, T. Ehrhardt'


class pi_nusquids(PiStage):
    """
    PISA Pi stage for weighting events due to the effect of neutrino oscillations,
    using nuSQuIDS as the oscillation probability calculator.

    Parameters
    ----------

    Uses the standard parameters as required by a PISA pi stage (see `pisa/core/pi_stage.py`)

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

    Additional params expected when using the `use_nsi` argument:
        TODO

    Additional params expected when using the `use_decoherence` argument:
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
                 use_nsi=False,
                 num_neutrinos=3,
                ):

        self.num_neutrinos = num_neutrinos
        self.use_nsi = use_nsi
        self.use_decoherence = use_decoherence

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
        if self.use_decoherence :
            expected_params.extend(['gamma21',
                                    'gamma31',
                                    'gamma32'])

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
        super(pi_nusquids, self).__init__(data=data,
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

        # set up initial states, get the nuSQuIDS "propagator" instances
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
            self.osc_params.gamma21 = self.params.gamma21.value.m_as('eV')
            self.osc_params.gamma31 = self.params.gamma31.value.m_as('eV')
            self.osc_params.gamma32 = self.params.gamma32.value.m_as('eV')

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

        import datetime
        t1 = datetime.datetime.now()

        # evolve the states starting from initial ones, using the current state of the params
        evolve_states(
            cz_shape=self.cz_calc_grid.shape[0],
            propagators=self.props,
            ini_states=self.ini_states, # TODO Check these are not changed during `compute_function`
            nsq_earth_atm=self.earth_atm,
            osc_params=self.osc_params
        )

        t2 = datetime.datetime.now()

        # Loop over containers
        for container in self.data:

            # define the points where osc. probs. are to be evaluated
            # this is just the energy/coszen values for each event, in the correct units
            en_eval = container["true_energy"].get(WHERE) * NSQ_CONST.GeV #TODO Can I be more efficient here?
            cz_eval = container["true_coszen"].get(WHERE)

            # Get the neutrino flavor (ignore the interaction)
            nuflav = container.name.replace("_cc","").replace("_nc","") # TODO Update this once we have the new events class which has helper functions for this kind of thing

            # Get the oscillation probs, writing them to the container
            _,_ = osc_probs(  # pylint: disable=unused-variable
                nuflav=nuflav,
                propagators=self.props,
                true_energies=en_eval,
                true_coszens=cz_eval,
                prob_e=container['prob_e'].get(WHERE),
                prob_mu=container['prob_mu'].get(WHERE),
            )
            container['prob_e'].mark_changed(WHERE)
            container['prob_mu'].mark_changed(WHERE)

        t3 = datetime.datetime.now() #TODO REMOVE THIS LOT

        #print("+++ Time taken : evolve_states = %s : osc_probs = %s" % (t2-t1,t3-t2) )


    @profile
    def apply_function(self):

        # update the outputted weights
        for container in self.data:
            apply_probs(container['sys_flux'].get(WHERE),
                        container['prob_e'].get(WHERE),
                        container['prob_mu'].get(WHERE),
                        out=container['weights'].get(WHERE))
            container['weights'].mark_changed(WHERE)


# vectorized function to apply (flux * prob)
# must be outside class
# TODO Is this really necessary?
if FTYPE == np.float64:
    signature = '(f8[:], f8, f8, f8[:])'
else:
    signature = '(f4[:], f4, f4, f4[:])'
@guvectorize([signature], '(d),(),()->()', target=TARGET)
def apply_probs(flux, prob_e, prob_mu, out):
    out[0] *= (flux[0] * prob_e) + (flux[1] * prob_mu)
