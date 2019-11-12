"""
This is an inout module for filling a 'Data' instance from a file
containg data from an 'Events' instance
The outputs should be consistent with the 'sample' stage
"""


from __future__ import absolute_import

from copy import deepcopy
from functools import reduce
from operator import add

from pisa.core.events import Data, Events
from pisa.core.stage import Stage
from pisa.utils.flavInt import ALL_NUFLAVINTS, NuFlavIntGroup, FlavIntDataGroup
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile


__all__ = ['SEP', 'events_to_data']


SEP = '|'



#TODO Try to merge with pisa.utils.config_parser.split, but currently gives problems due to the lower case forcing
def split(string):
    return string.replace(' ', '').split(',')

class events_to_data(Stage):
    """data service to load in events from an event events_to_data.

    Parameters
    ----------
    params: ParamSet of sequence with which to instantiate a ParamSet
        Parameters which set everything besides the binning

        Parameters required by this service are
            * data_events_to_data_config : filepath
                Filepath to event events_to_data configuration

            * dataset : string
                Pick which systematic set to use (or nominal)
                examples: 'nominal', 'neutrinos|dom_eff|1.05', 'muons|hole_ice|0.01'
                the nominal set will be used for the event types not specified

            * keep_criteria : None or string
                Apply a cut such as the only events which satisfy
                `keep_criteria` are kept.
                Any string interpretable as numpy boolean expression.

    output_binning : MultiDimBinning or convertible thereto
        The binning desired for the output maps.

    output_names : string
        Specifies the string representation of the NuFlavIntGroup(s) which will
        be produced as an output.

    output_events : bool
        Flag to specify whether the service output returns a MapSet
        or the full information about each event

    error_method : None, bool, or string
        If None, False, or empty string, the stage does not compute errors for
        the transforms and does not apply any (additional) error to produce its
        outputs. (If the inputs already have errors, these are propagated.)

    debug_mode : None, bool, or string
        If None, False, or empty string, the stage runs normally.
        Otherwise, the stage runs in debug mode. This disables caching (forcing
        recomputation of any nominal transforms, transforms, and outputs).

    transforms_cache_depth
    outputs_cache_depth : int >= 0

    """
    def __init__(self, params, output_names, events_file, output_binning=None,
                 output_events=True, error_method=None, debug_mode=None,
                 disk_cache=None, memcache_deepcopy=True,
                 transforms_cache_depth=20, outputs_cache_depth=20):

        self.sample_hash = None

        expected_params = ('dataset','keep_criteria') #TODO -> kwargs???

        self.events_file = events_file

        self.neutrinos = False
        self.muons = False
        self.noise = False

        output_names = output_names.replace(' ', '').split(',')
        clean_outnames = []
        self._output_nu_groups = []
        for name in output_names:
            if 'muons' in name:
                self.muons = True
                clean_outnames.append(name)
            elif 'noise' in name:
                self.noise = True
                clean_outnames.append(name)
            elif 'all_nu' in name:
                self.neutrinos = True
                self._output_nu_groups = \
                    [NuFlavIntGroup(f) for f in ALL_NUFLAVINTS]
            else:
                self.neutrinos = True
                self._output_nu_groups.append(NuFlavIntGroup(name))

        if self.neutrinos:
            clean_outnames += [str(f) for f in self._output_nu_groups]

        if not isinstance(output_events, bool):
            raise AssertionError(
                'output_events must be of type bool, instead it is supplied '
                'with type {0}'.format(type(output_events))
            )
        if output_events: #TODO Implement MapSet option or remove
            output_binning = None
        self.output_events = output_events

        super().__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            output_names=clean_outnames,
            error_method=error_method,
            debug_mode=debug_mode,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            output_binning=output_binning,
        )

        self._compute_outputs()

    @profile
    def _compute_outputs(self, inputs=None):

        """Apply basic cuts and compute histograms for output channels."""

        logging.debug('Entering events_to_data._compute_outputs')


        #Hashing 
        #TODO What should I hash??
        hash_property = [self.events_file,self.params['dataset'].value,self.output_names]
        this_hash = hash_obj(hash_property, full_hash=self.full_hash)
        #if this_hash == self.sample_hash: #TODO Fix this and replace...
        #    return

        #TODO Check there are no inputs

        #Fill an events instance from a file
        events = Events(self.events_file)

        #TODO Handle nominal, etc, etc datasets?

        #Extract the neutrino data from the 'Events' instance
        nu_data = []
        flav_fidg = FlavIntDataGroup(flavint_groups=events.flavints)
        for flavint in events.present_flavints :
            flav_fidg[flavint] = { var:events[flavint][var] for var in events[flavint].keys() }
        nu_data.append(flav_fidg)

        #Create the data instance, including the metadata
        #Note that there is no muon or noise data  in the 'Events'
        data = Data( reduce(add,nu_data) , metadata=deepcopy(events.metadata) )

        #Make cuts
        if self.params['keep_criteria'].value is not None:
            self._data.applyCut(self.params['keep_criteria'].value) #TODO Shivesh says this needs testing
            self._data.update_hash()

        #Update hashes
        self.sample_hash = this_hash
        data.metadata['sample_hash'] = this_hash
        data.update_hash()

        return data


    def validate_params(self, params):
        assert isinstance(params['dataset'].value, str)
        assert params['keep_criteria'].value is None or \
            isinstance(params['keep_criteria'].value, str)
