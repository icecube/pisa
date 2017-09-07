"""
TODO(shivesh): more docs
This oscillation service provides a wrapper for nuSQuIDS which is a
neutrino oscillation software using SQuIDS.

https://github.com/arguelles/nuSQuIDS
"""


from __future__ import absolute_import, division

from collections import OrderedDict

import numpy as np
from uncertainties import unumpy as unp

import nuSQUIDSpy as nsq

from pisa import FTYPE, ureg
from pisa.core.map import Map, MapSet
from pisa.core.stage import Stage
from pisa.utils.resources import open_resource
from pisa.utils.log import logging
from pisa.utils.profiler import profile


__all__ = ['nusquids']


class nusquids(Stage):
    """osc service to provide oscillated fluxes via nuSQuIDS.

    Parameters
    ----------
    params: ParamSet of sequence with which to instantiate a ParamSet
        Parameters which set everything besides the binning

        Parameters required by this service are
            * TODO(shivesh)

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
            'detector_depth', 'prop_height', 'deltacp', 'deltam21', 'deltam31',
            'theta12', 'theta13', 'theta23',
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

        super(nusquids, self).__init__(
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
            output_binning=output_binning
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
        binning = binning.reorder_dimensions(('coszen', 'energy'))
        cz_binning = binning['coszen']
        en_binning = binning['energy']

        units = nsq.Const()

        interactions = False
        en_min = en_binning.bin_edges.min().m_as('GeV') * units.GeV
        en_max = en_binning.bin_edges.max().m_as('GeV') * units.GeV
        cz_min = cz_binning.bin_edges.min().m_as('radian')
        cz_max = cz_binning.bin_edges.max().m_as('radian')
        en_grid = en_binning.weighted_centers.m_as('GeV') * units.GeV
        cz_grid = cz_binning.weighted_centers.m_as('radian')
        nu_flavours = 3

        nuSQ = nsq.nuSQUIDSAtm(
            cz_grid, en_grid, nu_flavours, nsq.NeutrinoType.both,
            interactions
        )

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

        initial_state = np.full(cz_binning.shape+en_binning.shape+(2, 3), np.nan)
        # Third index is selecting nu(0), nubar(1)
        # Fourth index is selecting flavour nue(0), numu(1), nutau(2)
        initial_state[:, :, 0, 0] = unp.nominal_values(inputs['nue'].hist)
        initial_state[:, :, 1, 0] = unp.nominal_values(inputs['nuebar'].hist)
        initial_state[:, :, 0, 1] = unp.nominal_values(inputs['numu'].hist)
        initial_state[:, :, 1, 1] = unp.nominal_values(inputs['numubar'].hist)
        initial_state[:, :, 0, 2] = np.zeros(inputs['nue'].hist.shape)
        initial_state[:, :, 1, 2] = np.zeros(inputs['nue'].hist.shape)

        if np.any(np.isnan(initial_state)):
            raise AssertionError('nan entries in initial_state: '
                                 '{0}'.format(initial_state))
        nuSQ.Set_initial_state(initial_state, nsq.Basis.flavor)

        nuSQ.Set_ProgressBar(True);
        nuSQ.EvolveState()

        fs = {'nue': np.full(binning.shape, np.nan),
              'nuebar': np.full(binning.shape, np.nan),
              'numu': np.full(binning.shape, np.nan),
              'numubar': np.full(binning.shape, np.nan),
              'nutau': np.full(binning.shape, np.nan),
              'nutaubar': np.full(binning.shape, np.nan)}
        for cz_idx, cz_bin in enumerate(cz_binning.weighted_centers.m_as('radians')):
            for e_idx, en_bin in enumerate(en_binning.weighted_centers.m_as('GeV')):
                en_bin_u = en_bin * units.GeV
                fs['nue'][cz_idx][e_idx] = nuSQ.EvalFlavor(0, cz_bin, en_bin_u, 0)
                fs['nuebar'][cz_idx][e_idx] = nuSQ.EvalFlavor(0, cz_bin, en_bin_u, 1)
                fs['numu'][cz_idx][e_idx] = nuSQ.EvalFlavor(1, cz_bin, en_bin_u, 0)
                fs['numubar'][cz_idx][e_idx] = nuSQ.EvalFlavor(1, cz_bin, en_bin_u, 1)
                fs['nutau'][cz_idx][e_idx] = nuSQ.EvalFlavor(2, cz_bin, en_bin_u, 0)
                fs['nutaubar'][cz_idx][e_idx] = nuSQ.EvalFlavor(2, cz_bin, en_bin_u, 1)

        outputs = []
        out_binning = self.input_binning.reorder_dimensions(('coszen', 'energy'))
        for key in fs.iterkeys():
            if np.any(np.isnan(fs[key])):
                raise AssertionError(
                    'Invalid value computed for {0} oscillated output: '
                    '{1}'.format(key, fs[key])
                )
            map = Map(name=key, binning=out_binning, hist=fs[key])
            map = map.reorder_dimensions(self.input_binning)
            outputs.append(map)

        return MapSet(outputs)

    def validate_params(self, params):
        pq = ureg.Quantity
        param_types = [
            ('detector_depth', pq),
            ('prop_height', pq),
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
