"""
This is an effective area stage designed for quick studies of how effective
areas affect experimental observables and sensitivities. In addition, it is
supposed to be easily reproducible as it may rely on (phenomenological)
functions or interpolated discrete data points, dependent on energy
(and optionally cosine zenith), and which can thus be used as reference or
benchmark scenarios.
"""


from __future__ import absolute_import, division

from collections.abc import Mapping

import numpy as np

from pisa.core.stage import Stage
from pisa.utils.fileio import from_file
from pisa.utils.profiler import profile


__all__ = ['load_aeff_param', 'param']

__author__ = 'T.C. Arlen, T. Ehrhardt, S. Wren, J. Weldert'

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


def load_aeff_param(source):
    """Load aeff parameterisation (energy- or coszen-dependent) from file
    or dictionary.
    Parameters
    ----------
    source : string or mapping
        Source of the parameterization. If string, treat as file path or
        resource location and load from the file; this must yield a mapping. If
        `source` is a mapping, it is used directly. See notes below on format.
    Returns
    -------
    aeff_params : OrderedDict
        Keys are stringified flavintgroups and values are the callables that
        produce aeff when called with energy or coszen values.
    Notes
    -----
    The mapping passed via `source` or loaded therefrom must have the format:
        {
            <flavintgroup_string>: val,
            <flavintgroup_string>: val,
            ...
        }
    Note that the `transform_groups` (container.name) defined
    in a pipeline config file using this must match the groupings defined
    above.
    `val`s can be one of the following:
        - Callable with one argument
        - String such that `eval(val)` yields a callable with one argument
    """
    if not isinstance(source, (str, Mapping)):
        raise TypeError('`source` must be string or mapping')

    if isinstance(source, str):
        orig_dict = from_file(source)
    elif isinstance(source, Mapping):
        orig_dict = source
    else:
        raise TypeError('Cannot load aeff parameterizations from a %s'
                        % type(source))

    return orig_dict


class param(Stage): # pylint: disable=invalid-name
    """Effective area service based on parameterisation functions stored in a
    .json file.
    Transforms an input map of a flux of a given flavour (and interaction)
    into maps of event rates, according to energy and cosine zenith dependent 
    effective areas specified by parameterisation functions.
    Requires true_energy, true_coszen, and weights to be present in the container.
    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameters:
        aeff_energy_paramfile
        aeff_coszen_paramfile
        livetime
        aeff_scale
    """
    def __init__(
        self,
        **std_kwargs,
    ):
        expected_params = (
            'aeff_energy_paramfile',
            'aeff_coszen_paramfile',
            'livetime',
            'aeff_scale'
        )

        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

        self.energy_param = load_aeff_param(self.params.aeff_energy_paramfile.value)
        self.coszen_param = load_aeff_param(self.params.aeff_coszen_paramfile.value)

    @profile
    def apply_function(self):
        aeff_scale = self.params.aeff_scale.m_as('dimensionless')
        livetime_s = self.params.livetime.m_as('sec')

        for container in self.data:
            scale = aeff_scale * livetime_s * np.ones(container.size)
            if container.name in self.energy_param.keys():
                func = eval(self.energy_param[container.name])
                scale *= func(container['true_energy'])
            if container.name in self.coszen_param.keys():
                func = eval(self.coszen_param[container.name])
                scale *= func(container['true_coszen'])

            container['weights'] *= scale
            container.mark_changed('weights')
