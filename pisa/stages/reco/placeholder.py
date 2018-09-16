# pylint: disable=not-callable
"""
Create placeholder reconstructed and PID variables in a file
These are tuned to sensible values for DeepCore/ICU-like detector
TODO Merge with param.py (needs updating from cake to pi)

Tom Stuttard
"""
from __future__ import absolute_import, print_function, division

import math
import numpy as np

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import WHERE, myjit, ftype


__all__ = ["placeholder","reco_energy_placeholder","reco_coszen_placeholder","pid_placeholder"]


def reco_energy_placeholder(deposited_energy,random_state=None) :
    '''
    Function to produce a smeared reconstructed energy distribution.
    Use as a placeholder if real reconstructions are not currently available.
    Uses the deposited energy of the particle.
    '''

    # Default random state with no fixed seed
    if random_state is None :
        random_state = np.random.RandomState()
        
    # Smear the deposited energy
    # Gaussian smearing that is dependent on the amount of deposited energy
    sigma = deposited_energy / 8. #TODO Tune this value, just eye-balling something GRECO-like for now
    reco_energy = np.random.normal(deposited_energy,sigma)

    # Ensure physics values
    reco_energy[reco_energy < 0.] = 0.

    return reco_energy


def reco_coszen_placeholder(true_coszen,random_state=None) :
    '''
    Function to produce a smeared reconstructed cos(zenith) distribution.
    Use as a placeholder if real reconstructions are not currently available.
    Uses the true coszen of the particle as an input.
    Keep within the rotational bounds
    '''

    #TODO Energy and PID dependence
    #TODO Include neutrino opening angle model: 30. deg / np.sqrt(true_energy)

    # Default random state with no fixed seed
    if random_state is None :
        random_state = np.random.RandomState()

    # Smear the cos(zenith)
    # Using a Gaussian smearing, indepedent of the true zenith angle
    sigma = 0.2
    reco_coszen = np.random.normal(true_coszen,sigma)

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

    Params: 
        a = normalisation (e.g. plateau height) 
        b = steepness of rise
        c = x value at half-height of curve
    '''
    return a / (1 + np.exp( -b * (x-c) ) )


def has_muon(particle_key) :
    '''
    Function returning True if the particle type has muons in the final state
    This is numu CC and atmopsheric muons
    TODO In future consider adding nutau CC where the tau decays to muons
    '''
    return ( (particle_key.startswith("numu") and particle_key.endswith("_cc")) or particle_key.startswith("muon") )


def pid_placeholder(particle_key,deposited_energy,track_pid=100.,cascade_pid=5.,random_state=None) :
    '''
    Function to assign a PID based on truth information.
    Use as a placeholder if real reconstructions are not currently available.
    Uses the flavor and interaction type of the particle

    Approximating energy dependence using a logistic function.
    Tuned to roughly match GRECO (https://wiki.icecube.wisc.edu/index.php/IC86_Tau_Appearance_Analysis#Track_Length_as_Particle_ID) 
    '''

    # Default random state with no fixed seed
    if random_state is None :
        random_state = np.random.RandomState()

    # Track/cascade ID is energy dependent.
    # Using deposited energy, and assigning one dependence for events with muon 
    # tracks (numu CC, atmospheric muons) and another for all other events.

    # Define whether each particle is a track
    if has_muon(particle_key) : # Maybe treat atmopsheric muons differently???
        track_prob = logistic_function(0.8,0.2,20.,deposited_energy)
    else :
        track_prob = logistic_function(0.3,0.05,10.,deposited_energy)
    track_mask = np.random.uniform(0.,1.,size=deposited_energy.size) < track_prob

    # Assign PID values
    pid = np.full_like(deposited_energy,np.NaN)
    pid[track_mask] = track_pid
    pid[~track_mask] = cascade_pid

    return pid


class placeholder(PiStage):
    """
    stage to generate placeholder reconstructed parameters

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

    Notes
    -----
    If using parameterised reco/pid, input file must contain 
    `deposited_energy` variable. This represents the charge in 
    the event in the detector region from detectable particles 
    (e.g. no neutrinos).

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
        
        input_names = ()
        output_names = ()

        # what keys are added or altered for the outputs during apply
        output_apply_keys = (
                            'reco_energy',
                            'reco_coszen',
                            'pid',
                            )

        # init base class
        super(placeholder, self).__init__(data=data,
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

        self.data.data_specs = self.input_specs

        #TODO Add useful cases here, e.g. perfect reco/PID, ICU baseline (LoI?), DeepCore current best, etc...

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
            deposited_energy = container["deposited_energy"].get(WHERE)


            #
            # Smear energy
            #

            # Create container if not already present
            if "reco_energy" not in container :
                container.add_array_data( "reco_energy", np.full_like(true_energy,np.NaN,dtype=FTYPE) )

            # Create the reco energy variable
            if perfect_reco :
                reco_energy = true_energy
            else :
                reco_energy = reco_energy_placeholder(deposited_energy,random_state=random_state)

            # Write to the container
            np.copyto( src=reco_energy, dst=container["reco_energy"].get("host") )
            container["reco_energy"].mark_changed()


            #
            # Smear coszen
            #

            # Create container if not already present
            if "reco_coszen" not in container :
                container.add_array_data( "reco_coszen", np.full_like(true_coszen,np.NaN,dtype=FTYPE) )

            # Create the reco coszen variable
            if perfect_reco :
                reco_coszen = true_coszen
            else :
                reco_coszen = reco_coszen_placeholder(true_coszen,random_state=random_state)

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
                pid = pid_placeholder(particle_key,true_energy,track_pid=track_pid,cascade_pid=cascade_pid,random_state=random_state)

            # Write to the container
            np.copyto( src=pid, dst=container["pid"].get("host") )
            container["pid"].mark_changed()



