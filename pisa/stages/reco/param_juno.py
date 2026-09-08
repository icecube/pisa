"""
Create the transforms that map from true energy and coszen
to the reconstructed parameters. Provides reco event rate maps using these
transforms.
"""


from __future__ import division

from collections.abc import Mapping
from collections import OrderedDict
from copy import deepcopy
import itertools

import numpy as np
from scipy.stats import norm
from scipy import stats

from pisa import ureg
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.core.binning import basename
from pisa.utils.fileio import from_file
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.comparisons import recursiveEquality, EQUALITY_PREC, isscalar
from pisa.utils.profiler import profile


def load_reco_param(source):
    """Load reco parameterisation (energy-dependent) from file or dictionary.
    Parameters
    ----------
    source : string or mapping
        Source of the parameterization. If string, treat as file path or
        resource location and load from the file; this must yield a mapping. If
        `source` is a mapping, it is used directly. See notes below on format.
    Returns
    -------
    reco_params : OrderedDict
        Keys are stringified flavintgroups and values are dicts of strings
        representing the different reco dimensions and lists of distribution
        properties. These latter have a 'fraction', a 'dist' and a 'kwargs' key.
        The former two hold callables, while the latter holds a dict of
        key-callable pairs ('loc', 'scale'), which can be evaluated at the desired
        energies and passed into the respective `scipy.stats` distribution.
        The distributions for a given dimension will be superimposed according
        to their relative weights to form the reco kernels (via integration)
        when called with energy values (parameterisations are functions of
        energy only!).
    """
    if not (source is None or isinstance(source, (str, Mapping))):
        raise TypeError('`source` must be string, mapping, or None')

    if isinstance(source, str):
        orig_dict = from_file(source)

    elif isinstance(source, Mapping):
        orig_dict = source

    else:
        raise TypeError('Cannot load reco parameterizations from a %s'
                        % type(source))

    #valid_dimensions = ('coszen', 'energy')
    #required_keys = ('dist', 'fraction', 'kwargs')

    return orig_dict

class param_juno(Stage):
    """
    ----------
    params : ParamSet
        reco_paramfile 
    """
    def __init__(
        self,
        **std_kwargs
    ):
        expected_params = (
            'reco_paramfile',
        )

        expected_container_keys = (
            'true_energy',
            'true_coszen',
            'weights',
        )

        supported_reps = {
            'calc_mode' : [MultiDimBinning],
            'apply_mode' : ['events']
        }

        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            supported_reps=supported_reps,
            **std_kwargs,
        )

        reco_param_source = self.params.reco_paramfile.value
        self.reco_param = load_reco_param(reco_param_source)

    def setup_function(self):
        self.bin_edges = []
        for b in self.calc_mode.bin_edges[self.calc_mode.names.index("true_energy")]:
            self.bin_edges.append(b.magnitude)
        self.reweight = {}
        
        for container in self.data:
            
            if container.name in self.reco_param.keys():
                hist = getattr(stats, self.reco_param[container.name]['energy'][0]['dist'])
                loc_f = eval(self.reco_param[container.name]['energy'][0]['kwargs']['loc'])
                scale_f = eval(self.reco_param[container.name]['energy'][0]['kwargs']['scale'])
                
                if 'visible' in self.reco_param[container.name]['energy'][0]['kwargs']:
                    visible_f = eval(self.reco_param[container.name]['energy'][0]['kwargs']['visible'])
                    E_visible = visible_f(container['true_energy'])
                else:
                    E_visible = container['true_energy']

                loc = E_visible + loc_f(E_visible)
                scale = scale_f(E_visible)
                
                self.reweight[container.name] = np.zeros((len(self.bin_edges)-1, len(self.bin_edges)-1))
                for i in range(len(self.bin_edges)-1):
                    cdfs = hist(loc=loc[i], scale=scale[i]).cdf(self.bin_edges)
                    self.reweight[container.name][i] = cdfs[1:] - cdfs[:-1]

    def apply_function(self):
        
        for container in self.data:
            
            if container.name in self.reco_param.keys():
                reco_weight = np.zeros(container.size)
                initial_weight = np.copy(container['weights'])
                
                N_bins = len(self.bin_edges)-1
                N_reactors = int(container.size/(N_bins))
                for i in range(N_reactors):
                    reco_weight[i*N_bins:(i+1)*N_bins] = np.sum((initial_weight[i*N_bins:(i+1)*N_bins] * self.reweight[container.name].T).T, axis=0)

                container['weights'] = reco_weight
                container['true_weights'] = initial_weight

            container['reco_energy'] = container['true_energy'] 
            container['reco_coszen'] = container['true_coszen']


def init_test(**param_kwargs):
    """Instantiation example"""
    param_set = ParamSet([Param(name="reco_paramfile", value='reco/JUNO/reco_param.json',  **param_kwargs)])
    e_binning = OneDimBinning(name='true_energy', is_log=True, num_bins=10, domain=[0.002, 0.008]*ureg.GeV)
    cz_binning = OneDimBinning(name='true_coszen', is_lin=True, num_bins=1, domain=[-1, 1])
    binning = MultiDimBinning([cz_binning, e_binning])
    return param_juno(params=param_set, calc_mode=binning, apply_mode='events')
