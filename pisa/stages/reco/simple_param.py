# pylint: disable=not-callable

"""
Create reconstructed and PID variables based on truth information 
for MC events using simple parameterisations.
"""

#TODO In future this could be integrated with param.py (but, that meed updating from cake to pi first)


from __future__ import absolute_import, print_function, division

import math, fnmatch, collections
import numpy as np

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import WHERE, myjit, ftype



__all__ = ["simple_param","simple_reco_energy_parameterization","simple_reco_coszen_parameterization","simple_pid_parameterization"]

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


def dict_lookup_wildcard(dict_obj,key) :
    '''
    Find the object in a dict specified by a key, where the key may include wildcards

    Parameters
    ----------
    dict_obj : dict
        The dict (or dict-like) object to search 
    key : str
        The key to search for in the dict (may include wildcards)

    Returns
    -------
    key : str
        The str found in the dict matching the wildcard
    value : anything
        The value found corresponding to the the requested key
    '''

    assert isinstance(dict_obj,collections.Mapping)
    assert isinstance(key,str)

    matches = collections.OrderedDict([ (k,v) for k,v in dict_obj.items() if fnmatch.fnmatch(key,k) ])

    assert len(matches) > 0, "No match for '%s' found in dict" % key
    assert len(matches) < 2, "Multiple matches for '%s' found in dict : %s" % (key,matches.keys())

    return matches.keys()[0], matches.values()[0]


def logistic_function(a,b,c,x) :
    '''
    Logistic function as defined here: https://en.wikipedia.org/wiki/Logistic_function.
    Starts off slowly rising, before steeply rising, then plateaus.

    Parameters
    ----------
    a : float
        Normalisation (e.g. plateau height) 
    b : float
        Steepness of rise (larger value means steeper rise)
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



def visible_energy_correction(particle_key) :
    '''
    Simple way to estimate the amount of visible energy in the event.

    Right now considering cases with final state neutrinos, such as NC events, 
    and nutau CC events (where the tau decays to a tau neutrino).

    Neglecting the much lower losses due to muon decay for numu CC.
    Also neglecting fact that different particle types produce differing photon yields.

    I've tuned these by eye due to the biases seen in GRECO pegleg, which to first 
    order I'm assuming are due to this missing energy.
    There is also a bias in numu CC in GRECO, but suspect this is due to containment
    or stochastics, but either way not reproducing this here.

    Parameters
    ----------
    particle_key : string
        Key identifiying the particle type, e.g. numu_cc, nutau_nc, muon, etc.

    Returns
    -------
    visible_energy : array
        Estimated visible energy in each event

    '''

    #TODO Add some smearing

    # NC events have final state neutrino
    if particle_key.endswith("_nc") :
        return 0.4 #TODO tune, consider inelasticity numbers vs energy (including nu vs nubar)

    # nutau CC events have final state neutrino, but from a subsequent tau decay
    elif particle_key.startswith("nutau") and particle_key.endswith("_cc") :
        return 0.6 #TODO tune, consider decay spectrum vs energy

    # muons are all over the place since their "true_energy" doesn't reflect any real value in the ice (is energy as surface of generation cylinder)
    elif particle_key == "muons" :
        return 0.1 #TODO Should really store deposied energy in the frame instead

    # Everything else deposits full energy as visible energy
    else :
        return 1.


def energy_dependent_sigma(energy,energy_0,sigma_0,energy_power) :
    '''
    Returns an energy dependent sigma (standard deviation) value(s),
    with energy dependence defined as follows:

        sigma(E) = sigma(E=E0) * (E/E0)^n 

    Parameters
    ----------
    energy : array or float
        Energy value to evaluate sigma at 
    energy_0 : float
        Energy at which sigma_0 is defined
    sigma_0 : float 
        The value of sigma at energy_0
    energy_power : float
        Power/index fo the energy dependence

    Returns
    -------
    sigma(energy) : array or float
        The value of sigma at the specified energy (or energies)
    '''
    return sigma_0 * np.power(energy/energy_0,energy_power)


def simple_reco_energy_parameterization(particle_key,true_energy,params,random_state) :
    '''
    Function to produce a smeared reconstructed energy distribution.
    Resolution is particle- and energy-dependent
    Use as a placeholder if real reconstructions are not currently available.

    Parameters
    ----------
    particle_key : string
        Key identifiying the particle type, e.g. numu_cc, nutau_nc, muon, etc.

    true_energy : array
        True energy array.

    params : dict
        keys   : particle key (wilcards accepted)
        values : list : [ E0 (reference true_energy), median reco error at E0, index/power of energy dependence ]
        (example: params = {'nue*_cc':[10.,0.2,0.2],})

    random_state : np.random.RandomState
        User must provide the random state, meaning that reproducible results 
        can be obtained when calling multiple times.

    Returns
    -------
    reco_energy : array
        Reconstructed energy array.
    '''

    #TODO Update docs

    # Default random state with no fixed seed
    if random_state is None :
        random_state = np.random.RandomState()

    # Get the visible energy
    visible_energy = true_energy * visible_energy_correction(particle_key)

    # Grab the params for this particle type
    _,energy_dependent_sigma_params = dict_lookup_wildcard(dict_obj=params,key=particle_key)

    # Get the sigma of the "reco error" distribution (energy dependent)
    # Easier to use this than the "reco energy" directly
    energy_0 = energy_dependent_sigma_params[0]
    reco_error_sigma_0 = energy_dependent_sigma_params[1]
    energy_power = energy_dependent_sigma_params[2]
    reco_error_sigma = energy_dependent_sigma(visible_energy,energy_0,reco_error_sigma_0,energy_power)

    # Get the reco error
    reco_error = random_state.normal(np.zeros_like(reco_error_sigma),reco_error_sigma)

    # Compute the corresponding reco energy
    # Use visible energy since that is what really matters
    reco_energy = visible_energy * ( reco_error + 1. )

    # Ensure physical values
    reco_energy[reco_energy < 0.] = 0.

    return reco_energy


def simple_reco_coszen_parameterization(particle_key,true_energy,true_coszen,params,random_state) :
    '''
    Function to produce a smeared reconstructed cos(zenith) distribution.
    Resolution is particle- and energy-dependent
    Use as a placeholder if real reconstructions are not currently available.
    Keep within the rotational bounds

    Parameters
    ----------
    true_coszen : array
        True cos(zenith angle) array.

    true_energy : array
        True energy array.

    params : dict
        keys   : particle key (wilcards accepted)
        values : list : [ E0 (reference true_energy), median reco error at E0, index/power of energy dependence ]
        (example: params = {'nue*_cc':[10.,0.2,0.5],})

    random_state : np.random.RandomState
        User must provide the random state, meaning that reproducible results 
        can be obtained when calling multiple times.

    Returns
    -------
    reco_coszen : array
        Reconstructed cos(zenith angle) array.
    '''

    # Default random state with no fixed seed
    if random_state is None :
        random_state = np.random.RandomState()

    # Get the visible energy
    visible_energy = true_energy * visible_energy_correction(particle_key)

    # Grab the params for this particle type
    _,energy_dependent_sigma_params = dict_lookup_wildcard(dict_obj=params,key=particle_key)

    # Get the sigma of the "reco error" distribution (energy dependent)
    # Easier to use this than the "reco coszen" directly
    energy_0 = energy_dependent_sigma_params[0]
    reco_error_sigma_0 = energy_dependent_sigma_params[1]
    energy_power = energy_dependent_sigma_params[2]
    reco_error_sigma = energy_dependent_sigma(visible_energy,energy_0,reco_error_sigma_0,energy_power)

    # Get the reco error
    reco_error = random_state.normal(np.zeros_like(reco_error_sigma),reco_error_sigma)

    # Compute the corresponding reco coszen
    # Use visible energy since that is what really matters
    reco_coszen = true_coszen + reco_error 

    # Enforce rotational bounds
    out_of_bounds_mask = reco_coszen > 1.
    reco_coszen[out_of_bounds_mask] = reco_coszen[out_of_bounds_mask] - ( 2. * (reco_coszen[out_of_bounds_mask] - 1.) )

    out_of_bounds_mask = reco_coszen < -1.
    reco_coszen[out_of_bounds_mask] = reco_coszen[out_of_bounds_mask] - ( 2. * (reco_coszen[out_of_bounds_mask] + 1.) )

    return reco_coszen


def simple_pid_parameterization(particle_key,true_energy,params,track_pid,cascade_pid,random_state,) :
    '''
    Function to assign a PID based on truth information.
    Is particle-, interaction- and energy-dependent
    Approximating energy dependence using a logistic function.
    Can use as a placeholder if real reconstructions are not currently available.

    Parameters
    ----------
    particle_key : string
        Key identifiying the particle type, e.g. numu_cc, nutau_nc, muon, etc.

    true_energy : array
        True energy array.

    params : dict
        keys   : particle key (wilcards accepted)
        values : Logistic function params for track ID (list) : [ normalisation (plateau height), steepness of rise, true_energy at half-height ]
        (example: params = {'nue*_cc':[0.05,0.2,15.],})

    track_pid : float
        A PID value to assign to track-like events

    cascade_pid : float
        A PID value to assign to cascade-like events

    random_state : np.random.RandomState
        User must provide the random state, meaning that reproducible results 
        can be obtained when calling multiple times.

    Returns
    -------
    pid : array
        PID values.
    '''

    # Default random state with no fixed seed
    if random_state is None :
        random_state = np.random.RandomState()

    # Grab the params for this particle type
    _,logistic_func_params = dict_lookup_wildcard(dict_obj=params,key=particle_key)

    # Define whether each particle is a track
    track_prob = logistic_function(logistic_func_params[0],logistic_func_params[1],logistic_func_params[2],true_energy)
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
            If False, use the parametrised reco energy, coszen and pid functions

        reco_energy_params : dict
            Dict defining the `params` argument to `simple_reco_energy_parameterization`
            See `simple_reco_energy_parameterization` documentatio for more details

        reco_coszen_params : dict
            Dict defining the `params` argument to `simple_reco_coszen_parameterization`
            See `simple_reco_coszen_parameterization` documentatio for more details

        pid_track_params : dict
            Dict defining the `params` argument to `simple_pid_parameterization`
            See `simple_pid_parameterization` documentatio for more details

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
                        "perfect_reco", #TODO move these to constructor args?
                        "reco_energy_params",
                        "reco_coszen_params",
                        "pid_track_params",
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
        reco_energy_params = eval(self.params.reco_energy_params.value)
        reco_coszen_params = eval(self.params.reco_coszen_params.value)
        pid_track_params = eval(self.params.pid_track_params.value)
        track_pid = self.params.track_pid.value.m_as("dimensionless")
        cascade_pid = self.params.cascade_pid.value.m_as("dimensionless")

        # If using random numbers, use a random state with a fixed seed to make the 
        # same smearing for e.g. template and pseudodata (this achieves the same
        # as we would normally use if we had reco variales in the file).
        # Note that this doesn't affect other random numbers generated by other
        # calls to numpy.random throughout the code.
        random_state = np.random.RandomState(0) #TODO seed as arg/param

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
                reco_energy = simple_reco_energy_parameterization(
                    particle_key=particle_key,
                    true_energy=true_energy,
                    params=reco_energy_params,
                    random_state=random_state,
                )

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
                reco_coszen = simple_reco_coszen_parameterization(
                    particle_key=particle_key,
                    true_energy=true_energy,
                    true_coszen=true_coszen,
                    params=reco_coszen_params,
                    random_state=random_state,
                )

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
                pid = simple_pid_parameterization(
                    particle_key=particle_key,
                    true_energy=true_energy,
                    params=pid_track_params,
                    track_pid=track_pid,
                    cascade_pid=cascade_pid,
                    random_state=random_state,
                )

            # Write to the container
            np.copyto( src=pid, dst=container["pid"].get("host") )
            container["pid"].mark_changed()



