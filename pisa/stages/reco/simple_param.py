# pylint: disable=not-callable

"""
Create reconstructed and PID variables based on truth information 
for MC events using simple parameterisations.
"""

#TODO In future this could be integrated with param.py (but, that meed updating from cake to pi first)


from __future__ import absolute_import, print_function, division

import math
import numpy as np

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import WHERE, myjit, ftype



__all__ = ["simple_param","simple_reco_energy_parameterization","simple_reco_cozen_parameterization","simple_pid_parameterization"]

__author__ = 'T. Stuttard'

__license__ = '''Copyright (c) 2014-2017, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


def get_visible_energy(particle_key,true_energy) :
    '''
    Simple way to estimate the amount of visible energy in the event.

    Right now considering cases with final state neutrinos, such as NC events, 
    and nutau CC events (where the tau decays to a tau neutrino).

    Neglecting the much lower losses due to muon decay for numu CC.
    Also neglecting fact that different particle types produce differing photon yields.

    I've tuned these by eye due to the biases seen in GRECO pegleg, which to first 
    order I'm assuming are due to this missing energy
    There is also a bias in numu CC in GRECO, but suspect this is due to containment
    or stochastics, but either way not reproducing this here.

    Parameters
    ----------
    particle_key : string
        Key identifiying the particle type, e.g. numu_cc, nutau_nc, muon, etc.

    true_energy : array
        True energy array.

    Returns
    -------
    visible_energy : array
        Estimated visible energy in each event

    '''

    #TODO Add some smearing

    visible_energy_mod = np.ones_like(true_energy)

#    nc_mask = interaction.astype(int) == 2
#    nutau_cc_mask = (np.abs(pdg_code.astype(int)) == 16) & (interaction.astype(int) == 1)

    nc_mask = np.full_like(true_energy, particle_key.endswith("_nc"), dtype=bool)
    nutau_cc_mask = np.full_like(true_energy, particle_key.startswith("nutau") and particle_key.endswith("_cc"), dtype=bool)
    atm_muon_mask = np.full_like(true_energy, particle_key == "muons", dtype=bool)
    visible_energy_mod[nc_mask] = 0.4 #TODO Calculate, for now just eye-balling GRECO
    visible_energy_mod[nutau_cc_mask] = 0.6 #TODO Calculate, for now just eye-balling GRECO
    visible_energy_mod[atm_muon_mask] = 0.1 #TODO Calculate, for now just eye-balling GRECO
    visible_energy = true_energy * visible_energy_mod
    return visible_energy


def simple_reco_energy_parameterization(particle_key,true_energy,random_state) :
    '''
    Function to produce a smeared reconstructed energy distribution.
    Use as a placeholder if real reconstructions are not currently available.
    Uses the true energy of the particle.

    Parameters
    ----------
    particle_key : string
        Key identifiying the particle type, e.g. numu_cc, nutau_nc, muon, etc.

    true_energy : array
        True energy array.

    random_state : np.random.RandomState
        User must provide the random state, meaning that reproducible results 
        can be obtained when calling multiple times.

    Returns
    -------
    reco_energy : array
        Reconstructed energy array.
    '''

    #TODO Make sigma an arg, and a parameter in the stage

    # Default random state with no fixed seed
    if random_state is None :
        random_state = np.random.RandomState()
        
    # Define an energy-dependent smearing based on the true energy
    # Define a different smearing for atmospheric muons, which behave a little differently 
    if particle_key == "muons" :
        sigma = true_energy / 8. #TODO Tune this value, just eye-balling something GRECO-like for now
    else :
        sigma = true_energy / 4. #TODO Tune this value, just eye-balling something GRECO-like for now

    # Get the visible energy
    visible_energy = get_visible_energy(particle_key,true_energy)

    # Now apply the smearing (AFTER the conversion to visible energy so that the smearing isn't suppressed by the bias)
    reco_energy = random_state.normal(visible_energy,sigma)

    # Ensure physical values
    reco_energy[reco_energy < 0.] = 0.

    return reco_energy


def simple_reco_cozen_parameterization(true_coszen,random_state) :
    '''
    Function to produce a smeared reconstructed cos(zenith) distribution.
    Use as a placeholder if real reconstructions are not currently available.
    Uses the true coszen of the particle as an input.
    Keep within the rotational bounds

    Parameters
    ----------
    true_coszen : array
        True cos(zenith angle) array.

    random_state : np.random.RandomState
        User must provide the random state, meaning that reproducible results 
        can be obtained when calling multiple times.

    Returns
    -------
    reco_coszen : array
        Reconstructed cos(zenith angle) array.
    '''

    #TODO Energy and PID dependence
    #TODO Include neutrino opening angle model: 30. deg / np.sqrt(true_energy)
    #TODO Make sigma an arg, and a parameter in the stage

    # Default random state with no fixed seed
    if random_state is None :
        random_state = np.random.RandomState()

    # Smear the cos(zenith)
    # Using a Gaussian smearing, indepedent of the true zenith angle
    sigma = 0.2
    reco_coszen = random_state.normal(true_coszen,sigma)

    # Enforce rotational bounds
    out_of_bounds_mask = reco_coszen > 1.
    reco_coszen[out_of_bounds_mask] = reco_coszen[out_of_bounds_mask] - ( 2. * (reco_coszen[out_of_bounds_mask] - 1.) )

    out_of_bounds_mask = reco_coszen < -1.
    reco_coszen[out_of_bounds_mask] = reco_coszen[out_of_bounds_mask] - ( 2. * (reco_coszen[out_of_bounds_mask] + 1.) )

    return reco_coszen


def logistic_function(a,b,c,x) :
    '''
    Logistic function as defined here: https://en.wikipedia.org/wiki/Logistic_function.
    Starts off slowly rising, before stteply rising, then plateaus.

    Parameters
    ----------
    a : float
        Normalisation (e.g. plateau height) 
    b : float
        Steepness of rise
    c : float 
        x value at half-height of curve
    x : array
        The continuous parameter

    Returns
    -------
    f(x) : array
        The results of applying the logistic function to x
    '''
    return a / (1 + np.exp( -b * (x-c) ) )


def has_muon(particle_key) :
    '''
    Function returning True if the particle type has muons in the final state
    This is numu CC and atmopsheric muons

    Parameters
    ----------
    particle_key : string
        Key identifiying the particle type, e.g. numu_cc, nutau_nc, muon, etc.


    Returns
    -------
    has_muon : bool
        Flag set to try if particle has muon in final state
    '''

    #TODO consider adding nutau CC where the tau decays to muons

    return ( (particle_key.startswith("numu") and particle_key.endswith("_cc")) or particle_key.startswith("muon") )


def simple_pid_parameterization(particle_key,true_energy,random_state,track_pid=100.,cascade_pid=5.) :
    '''
    Function to assign a PID based on truth information.
    Use as a placeholder if real reconstructions are not currently available.
    Uses the flavor and interaction type of the particle

    Approximating energy dependence using a logistic function.
    Tuned to roughly match GRECO (https://wiki.icecube.wisc.edu/index.php/IC86_Tau_Appearance_Analysis#Track_Length_as_Particle_ID)

    Parameters
    ----------
    particle_key : string
        Key identifiying the particle type, e.g. numu_cc, nutau_nc, muon, etc.

    true_energy : array
        True energy array.

    track_pid : float
        A PID value to assign to track-like events

    cascade_pid : float
        A PID value to assign to cascade-like events

    random_state : np.random.RandomState
        User must provide the random state, meaning that reproducible results 
        can be obtained when calling multiple times.

    Returns
    -------
    reco_energy : array
        Reconstructed energy array.
    '''

    # Default random state with no fixed seed
    if random_state is None :
        random_state = np.random.RandomState()

    # Track/cascade ID is energy dependent.
    # Considering energy-dependence, and assigning one dependence for events with muon 
    # tracks (numu CC, atmospheric muons) and another for all other events.

    # Define whether each particle is a track
    if ( particle_key.startswith("numu") and particle_key.endswith("_cc") ) :
        # numu CC, good track ID
        track_prob = logistic_function(0.9,0.5,5.,true_energy)
    elif particle_key == "muons" :
        # Atmospheric muons, totally random #TODO Can probably do better here, but this is broadly consistent with GRECO and given only super weird events survive it isn't so crazy
        track_prob = 0.5
    else :
        # Everything else is a cascade
        track_prob = logistic_function(0.3,0.05,10.,true_energy)
    track_mask = random_state.uniform(0.,1.,size=true_energy.size) < track_prob

    # Assign PID values
    pid = np.full_like(true_energy,np.NaN)
    pid[track_mask] = track_pid
    pid[~track_mask] = cascade_pid

    return pid


class simple_param(PiStage):
    """
    Stage to generate reconstructed parameters (energy, coszen, pid) using simple parameterizations.
    These are not fit to any input data, but are simple and easily understandable and require no 
    input reconstructed events.

    Can easily be tuned to any desired physics case, rught now repesent a DeepCore/ICU-like detector.

    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameters:

        perfect_reco : bool
            If True, use "perfect reco": reco == true, numu(bar)_cc -> tracks, rest to cascades
            If False, use the parametrised energy, coszen and pid functions

        track_pid : float
            The numerical 'pid' variable value to assign for tracks

        cascade_pid : float
            The numerical 'pid' variable value to assign for cascades

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

        expected_params = ( 
                        "perfect_reco",
                        "track_pid",
                        "cascade_pid",
                        )
        
        input_names = (
                    'true_energy',
                    'true_coszen',
                    )
        output_names = ()

        # what keys are added or altered for the outputs during apply
        output_apply_keys = (
                            'reco_energy',
                            'reco_coszen',
                            'pid',
                            )

        # init base class
        super(simple_param, self).__init__(data=data,
                                        params=params,
                                        expected_params=expected_params,
                                        input_names=input_names,
                                        output_names=output_names,
                                        debug_mode=debug_mode,
                                        input_specs=input_specs,
                                        calc_specs=calc_specs,
                                        output_specs=output_specs,
                                        output_apply_keys=output_apply_keys,
                                       )

        #TODO Suport other modes
        assert self.input_mode == "events"
        assert self.calc_mode is None
        assert self.output_mode == "events"


    def setup_function(self):

        #TODO Could add a number of discrete cases here that can be selected betweeen, e.g. ICU baseline (LoI?), DeepCore current best, etc...

        self.data.data_specs = self.input_specs

        # Get params
        perfect_reco = self.params.perfect_reco.value
        track_pid = self.params.track_pid.value.m_as("dimensionless")
        cascade_pid = self.params.cascade_pid.value.m_as("dimensionless")

        # If using random numbers, use a rando state with a fixed seed to make the 
        # same smearing for e.g. template and pseudodata (this achieves the same
        # as we would normally use if we had reco variales in the file).
        # Note that this doesn't affect other random numbers generated by other
        # calls to numpy.random throughout the code.
        random_state = np.random.RandomState(0)

        for container in self.data :

            # Get stuff that is used multiples times
            particle_key = container.name
            true_energy = container["true_energy"].get(WHERE)
            true_coszen = container["true_coszen"].get(WHERE)


            #
            # Get reco energy
            #

            # Create container if not already present
            if "reco_energy" not in container :
                container.add_array_data( "reco_energy", np.full_like(true_energy,np.NaN,dtype=FTYPE) )

            # Create the reco energy variable
            if perfect_reco :
                reco_energy = true_energy
            else :
                reco_energy = simple_reco_energy_parameterization(particle_key,true_energy,random_state=random_state)

            # Write to the container
            np.copyto( src=reco_energy, dst=container["reco_energy"].get("host") )
            container["reco_energy"].mark_changed()


            #
            # Get reco coszen
            #

            # Create container if not already present
            if "reco_coszen" not in container :
                container.add_array_data( "reco_coszen", np.full_like(true_coszen,np.NaN,dtype=FTYPE) )

            # Create the reco coszen variable
            if perfect_reco :
                reco_coszen = true_coszen
            else :
                reco_coszen = simple_reco_cozen_parameterization(true_coszen,random_state=random_state)

            # Write to the container
            np.copyto( src=reco_coszen, dst=container["reco_coszen"].get("host") )
            container["reco_coszen"].mark_changed()


            #
            # Create a PID variable
            #

            # Create container if not already present
            if "pid" not in container :
                container.add_array_data( "pid", np.full_like(true_energy,np.NaN,dtype=FTYPE) )

            # Create the PID variable
            if perfect_reco :
                pid_value = track_pid if has_muon(particle_key) else cascade_pid
                pid = np.full_like(true_energy,pid_value)
            else :
                pid = simple_pid_parameterization(particle_key,true_energy,track_pid=track_pid,cascade_pid=cascade_pid,random_state=random_state)

            # Write to the container
            np.copyto( src=pid, dst=container["pid"].get("host") )
            container["pid"].mark_changed()



