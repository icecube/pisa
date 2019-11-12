"""
TODO(shivesh): more docs
This oscillation service provides a wrapper for nuSQuIDS which is a
neutrino oscillation software using SQuIDS.

This service has a prerequisite package called nuSQuIDS, built with the python
bindings. This package along with its documentation can be found at:
https://github.com/arguelles/nuSQuIDS

For convenience the prerequisites of nuSQuIDS are listed below:
    - gsl (>= 1.15): http://www.gnu.org/software/gsl/
    - hdf5 with c bindings: http://www.hdfgroup.org/HDF5/
    - C++ compiler with C++11 support
    - SQUIDS (>= 1.2): https://github.com/jsalvado/SQuIDS/
additionally the following python bindings are needed:
    - boost (>= 1.54): http://www.boost.org/
    - numpy: http://www.numpy.org/
    - matplotlib: http://matplotlib.org/
"""


from __future__ import absolute_import, division

import multiprocessing

import numpy as np
from uncertainties import unumpy as unp

import nuSQUIDSpy as nsq

from pisa import FTYPE, ureg
from pisa.core.map import Map, MapSet
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile


__all__ = ['cake_nusquids']

__author__ = 'S. Mandalia'

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


class cake_nusquids(Stage):
    """osc service to provide oscillated fluxes via nuSQuIDS.

    Parameters
    ----------
    params: ParamSet of sequence with which to instantiate a ParamSet
        Parameters which set everything besides the binning

        Parameters required by this service are
        * oversample : ureg.Quantity
            For each bin, split evenly into a given number of sub-bins and
            evaluate the oscillation probabilities for each sub-bin. Then
            average over the sub-bins to obtain the oscillated result for the
            bin.

        * Oscillation related parameters:
            - deltacp
            - deltam21
            - deltam31
            - theta12
            - theta13
            - theta23

    input_names : string
        Specifies the string representation of the NuFlavIntGroup(s) that
        belong in the Data object passed to this service.

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

    outputs_cache_depth : int >= 0

    """
    def __init__(self, params, input_binning, output_binning,
                 error_method=None, debug_mode=None, disk_cache=None,
                 memcache_deepcopy=True, outputs_cache_depth=20):

        expected_params = (
            'oversample', 'deltacp', 'deltam21', 'deltam31', 'theta12',
            'theta13', 'theta23',
        )

        # Define the names of objects that are required by this stage (objects
        # will have the attribute `name`: i.e., obj.name)
        input_names = (
            'nue', 'numu', 'nuebar', 'numubar'
        )

        # Define the names of objects that get produced by this stage
        output_names = (
            'nue', 'numu', 'nutau', 'nuebar', 'numubar', 'nutaubar'
        )

        if input_binning != output_binning:
            raise AssertionError('Input binning must match output binning.')

        super().__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            debug_mode=debug_mode,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning,
        )

    @profile
    def _compute_outputs(self, inputs=None):
        """Compute histograms for output channels."""
        logging.debug('Entering nusquids._compute_outputs')
        if not isinstance(inputs, MapSet):
            raise AssertionError('inputs is not a MapSet object, instead '
                                 'is type {0}'.format(type(inputs)))
        # TODO(shivesh): oversampling
        # TODO(shivesh): more options
        # TODO(shivesh): static function
        # TODO(shivesh): hashing
        binning = self.input_binning.basename_binning
        binning = binning.reorder_dimensions(
            ('coszen', 'energy'), use_basenames=True
        )
        cz_binning = binning['coszen']
        en_binning = binning['energy']

        units = nsq.Const()

        interactions = False
        cz_min = cz_binning.bin_edges.min().m_as('radian')
        cz_max = cz_binning.bin_edges.max().m_as('radian')
        en_min = en_binning.bin_edges.min().m_as('GeV') * units.GeV
        en_max = en_binning.bin_edges.max().m_as('GeV') * units.GeV
        cz_centers = cz_binning.weighted_centers.m_as('radian')
        en_centers = en_binning.weighted_centers.m_as('GeV') * units.GeV
        cz_grid = np.array([cz_min] + cz_centers.tolist() + [cz_max])
        en_grid = np.array([en_min] + en_centers.tolist() + [en_max])
        nu_flavours = 3

        nuSQ = nsq.nuSQUIDSAtm(
            cz_grid, en_grid, nu_flavours, nsq.NeutrinoType.both,
            interactions
        )

        nuSQ.Set_EvalThreads(multiprocessing.cpu_count())

        theta12 = self.params['theta12'].value.m_as('radian')
        theta13 = self.params['theta13'].value.m_as('radian')
        theta23 = self.params['theta23'].value.m_as('radian')

        deltam21 = self.params['deltam21'].value.m_as('eV**2')
        deltam31 = self.params['deltam21'].value.m_as('eV**2')

        # TODO(shivesh): check if deltacp should be in radians
        deltacp = self.params['deltacp'].value.m_as('radian')

        nuSQ.Set_MixingAngle(0, 1, theta12)
        nuSQ.Set_MixingAngle(0, 2, theta13)
        nuSQ.Set_MixingAngle(1, 2, theta23)

        nuSQ.Set_SquareMassDifference(1, deltam21)
        nuSQ.Set_SquareMassDifference(2, deltam31)

        nuSQ.Set_CPPhase(0, 2, deltacp)

        nuSQ.Set_rel_error(1.0e-10)
        nuSQ.Set_abs_error(1.0e-10)

        # Pad the edges of the energy, coszen space to cover the entire grid range
        cz_shape = cz_binning.shape[0]+2
        en_shape = en_binning.shape[0]+2
        shape = (cz_shape, en_shape) + (2, 3)
        initial_state = np.full(shape, np.nan)

        def pad_inputs(x):
            return np.pad(unp.nominal_values(x.hist), 1, 'edge')
        # Third index is selecting nu(0), nubar(1)
        # Fourth index is selecting flavour nue(0), numu(1), nutau(2)
        initial_state[:, :, 0, 0] = pad_inputs(inputs['nue'])
        initial_state[:, :, 1, 0] = pad_inputs(inputs['nuebar'])
        initial_state[:, :, 0, 1] = pad_inputs(inputs['numu'])
        initial_state[:, :, 1, 1] = pad_inputs(inputs['numubar'])
        initial_state[:, :, 0, 2] = np.zeros(pad_inputs(inputs['nue']).shape)
        initial_state[:, :, 1, 2] = np.zeros(pad_inputs(inputs['nue']).shape)

        if np.any(np.isnan(initial_state)):
            raise AssertionError('nan entries in initial_state: '
                                 '{0}'.format(initial_state))
        nuSQ.Set_initial_state(initial_state, nsq.Basis.flavor)

        # TODO(shivesh): use verbosity level to set this
        nuSQ.Set_ProgressBar(True)
        nuSQ.EvolveState()

        os = self.params['oversample'].value.m
        os_binning = binning.oversample(os)
        os_cz_binning = os_binning['coszen']
        os_en_binning = os_binning['energy']
        os_cz_centers = os_cz_binning.weighted_centers.m_as('radians')
        os_en_centers = os_en_binning.weighted_centers.m_as('GeV')

        fs = {}
        for nu in self.output_names:
            fs[nu] = np.full(os_binning.shape, np.nan)

        for icz, cz_bin in enumerate(os_cz_centers):
            for ie, en_bin in enumerate(os_en_centers):
                en_bin_u = en_bin * units.GeV
                fs['nue'][icz][ie] = nuSQ.EvalFlavor(0, cz_bin, en_bin_u, 0)
                fs['nuebar'][icz][ie] = nuSQ.EvalFlavor(0, cz_bin, en_bin_u, 1)
                fs['numu'][icz][ie] = nuSQ.EvalFlavor(1, cz_bin, en_bin_u, 0)
                fs['numubar'][icz][ie] = nuSQ.EvalFlavor(1, cz_bin, en_bin_u, 1)
                fs['nutau'][icz][ie] = nuSQ.EvalFlavor(2, cz_bin, en_bin_u, 0)
                fs['nutaubar'][icz][ie] = nuSQ.EvalFlavor(2, cz_bin, en_bin_u, 1)

        out_binning = self.input_binning.reorder_dimensions(('coszen',
                                                             'energy'),
                                                            use_basenames=True)
        os_out_binning = out_binning.oversample(os)

        outputs = []
        for key in fs.keys():
            if np.any(np.isnan(fs[key])):
                raise AssertionError(
                    'Invalid value computed for {0} oscillated output: '
                    '\n{1}'.format(key, fs[key])
                )
            map = Map(name=key, binning=os_out_binning, hist=fs[key])
            map = map.downsample(os) / float(os)
            map = map.reorder_dimensions(self.input_binning)
            outputs.append(map)

        return MapSet(outputs)

    def validate_params(self, params):
        pq = ureg.Quantity
        param_types = [
            ('oversample', pq),
            ('theta12', pq),
            ('theta13', pq),
            ('theta23', pq),
            ('deltam21', pq),
            ('deltam31', pq),
            ('deltacp', pq),
        ]

        for p, t in param_types:
            val = params[p].value
            if not isinstance(val, t):
                raise TypeError(
                    'Param "%s" must be type %s but is %s instead'
                    % (p, type(t), type(val))
                )
