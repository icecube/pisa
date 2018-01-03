"""
Histogram Monte Carlo events directly 
"""

from __future__ import division
import numpy as np
from uncertainties import unumpy as unp

from pisa.core.stage import Stage
from pisa.core.map import Map, MapSet
from pisa.core.events import Events
from pisa.utils.log import logging
from pisa.utils.config_parser import split

class mc(Stage):
    """ MC

    A realy simple stage just to load a bunch of events into a MapSet
    That's it! As simple as it gets...

    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameters:

        mc_events
        mc_cuts
    """

    def __init__(self, params,
                 output_binning, combine_groups,
                 memcache_deepcopy,
                 outputs_cache_depth,
                 error_method=None,
                 debug_mode=None):
        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = [
            'mc_events_file', 
            'mc_cuts',
        ]

        self.combine_groups = eval(combine_groups)

        output_names = self.combine_groups.keys()

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
        super(self.__class__, self).__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            output_names=output_names,
            error_method=error_method,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

    def _compute_nominal_outputs(self):
        evts = Events(self.params.mc_events_file.value)
        if self.params.mc_cuts.value is not None:
            logging.info('Applying cut %s'%self.params.mc_cuts.value)
            evts = evts.applyCut(self.params.mc_cuts.value)
        
        maps = []
	bin_edges = [edges.magnitude for edges in self.output_binning.bin_edges]

        for key, val in self.combine_groups.items():
            # ddd some E-2 spectrum to weights...

            collect_maps = []
            for kind in split(val):

		sample = [evts[kind][colname] for colname in self.output_binning.names]

                #if '_cc' in kind:
                flux = 'neutrino_nue_flux' if 'nue' in kind else 'neutrino_numu_flux'
                hist_weights = evts[kind]['weighted_aeff'] * evts[kind][flux]
                #else:
                #    hist_weights = evts[kind]['weighted_aeff'] * 0.5 * (evts[kind]['neutrino_nue_flux'] + evts[kind]['neutrino_numu_flux'])

		hist, _ = np.histogramdd(sample=sample,
					     weights=hist_weights,
					     bins=bin_edges)
                if self.error_method == 'sumw2':
                    print 'adding errors'
                    error_weights = np.square(hist_weights)
                    sumw2, _ = np.histogramdd(sample=sample,
                                                 weights=error_weights,
                                                 bins=bin_edges)
                    hist = unp.uarray(hist, np.sqrt(sumw2))
		collect_maps.append(Map(name=kind, hist=hist, binning=self.output_binning))
            map = sum(collect_maps)
            map.name = key
            maps.append(map)
        self.out = MapSet(maps)

    def _compute_outputs(self, inputs=None):
        return self.out
