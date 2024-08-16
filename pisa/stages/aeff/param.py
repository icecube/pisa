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
from scipy.interpolate import interp1d

from pisa.core.stage import Stage
from pisa.utils.fileio import from_file


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
        - Mapping with the format:
            {
                <"energy" or "coszen">: [sequence of values],
                "aeff": [sequence of values]
            }
          the two sequences are used to form a linear interpolant callable that
          maps energy or coszen values to aeff values. The effective area for any
          energy or coszen outside the bounds of the corresponding sequence is
          assumed to be 0.
    """
    if not isinstance(source, (str, Mapping)):
        raise TypeError('`source` must be string or mapping')

    if isinstance(source, str):
        aeff_dict = from_file(source)
    elif isinstance(source, Mapping):
        aeff_dict = source
    else:
        raise TypeError('Cannot load aeff parameterizations from a %s'
                        % type(source))

    for k in aeff_dict:
        func = aeff_dict[k]

        if isinstance(func, str):
            param_func = eval(func)

        elif callable(func):
            param_func = func

        elif isinstance(func, Mapping):
            is_energy = 'energy' in func
            is_coszen = 'coszen' in func

            if 'aeff' not in func:
                raise ValueError('No effective area values are provided for %s'%(k))
            if not (is_energy or is_coszen):
                raise ValueError('No energy or coszen values are provided for %s'%(k))

            var = 'energy' if is_energy else 'coszen'
            x_vals = func[var]
            aeff_vals = func['aeff']

            param_func = interp1d(x_vals, aeff_vals, kind='linear',
                                  bounds_error=False, fill_value=0)

        else:
            raise TypeError(
                'Expected parameteriation to be either a string that'
                ' can be interpreted by eval or as a mapping of values'
                ' from which to construct a spline. Got "%s".'
                % type(func)
            )

        aeff_dict[k] = param_func

    return aeff_dict


class param(Stage): # pylint: disable=invalid-name
    """Effective area service based on parameterisation functions stored in a
    .json file.
    Transforms an input map of a flux of a given flavour (and interaction)
    into maps of event rates, according to energy and cosine zenith dependent
    effective areas specified by parameterisation functions.

    Parameters
    ----------
    params : ParamSet or sequence with which to instantiate a ParamSet.
        Expected params are .. ::

            aeff_energy_paramfile : string
            aeff_coszen_paramfile : string
            livetime : Quantity [time]
            aeff_scale : Quantity [dimensionless]

        Expected container keys are .. ::

            "true_energy"
            "true_coszen"
            "weights"

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

        expected_container_keys = (
            'true_energy',
            'true_coszen',
            'weights',
        )

        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )

        self.energy_param = load_aeff_param(self.params.aeff_energy_paramfile.value)
        self.coszen_param = load_aeff_param(self.params.aeff_coszen_paramfile.value)

    def apply_function(self):
        aeff_scale = self.params.aeff_scale.m_as('dimensionless')
        livetime_s = self.params.livetime.m_as('sec')

        for container in self.data:
            scale = aeff_scale * livetime_s * np.ones(container.size)
            if container.name in self.energy_param.keys():
                func = self.energy_param[container.name]
                scale *= func(container['true_energy'])
            if container.name in self.coszen_param.keys():
                func = self.coszen_param[container.name]
                scale *= func(container['true_coszen'])

            container['weights'] *= scale
            container.mark_changed('weights')
