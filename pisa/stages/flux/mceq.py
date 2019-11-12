"""
This flux service provides flux tables from the MCEq (Matrix Cascade Equation)
package.

This service has a prerequisite package called MCEq.
This package along with its documentation can be found at:
https://github.com/afedynitch/MCEq
http://arxiv.org/abs/1503.00544

The authors of that software / the paper that it is based upon (A. Fedynitch,
R. Engel, T. K. Gaisser, F. Riehn, T. Stanev) request that anyone who uses
their work to produce results cite their work, so please do so if you make use
the `flux.mceq` service. The form of the citation that they request is found in
their documentation at
  http://mceq.readthedocs.io/en/latest/citations.html

For convenience the prerequisites of MCEq are listed below:
    - python-2.7
    - numpy
    - scipy
    - numba
    - matplotlib
    - jupyter notebook (optional, but needed for examples)
    - progressbar
Additional dependencies are required for the C implementation of the
NRLMSISE-00 atmosphere:
    - a C compiler (GNU gcc, for example)
    - make
    - ctypes
"""


from __future__ import absolute_import, division

from collections import OrderedDict

import scipy.interpolate as interpolate

import numpy as np

from MCEq.core import MCEqRun
import mceq_config
import CRFluxModels as pm

from pisa import ureg
from pisa.core.map import Map, MapSet
from pisa.core.stage import Stage
from pisa.utils.format import split
from pisa.utils.log import logging
from pisa.utils.profiler import profile


__all__ = ['mceq']

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


class mceq(Stage): # pylint: disable=invalid-name
    """flux service to calculate the flux tables by solving the matrix cascade
    equation using the MCEq package.


    Parameters
    ----------
    params: ParamSet of sequence with which to instantiate a ParamSet
        Parameters which set everything besides the binning

        Parameters required by this service are
            * interaction_model : str
                Hadronic interaction model

            * primary_model : str
                Primary flux model
                Default options are listed in the CRFluxModels docs:
                https://crfluxmodels.readthedocs.io/en/latest/index.html#module-CRFluxModels

            * density_model : str
                Density model of Earth's atmosphere
                Default options are listed in the MCEq docs:
                https://mceq.readthedocs.io/en/latest/physics.html#module-MCEq.density_profiles

            * location : str
                Location to evaluate the fluxes
                Default options are listed in the MCEq docs:
                https://mceq.readthedocs.io/en/latest/physics.html#MCEq.density_profiles.CorsikaAtmosphere.init_parameters

            * season : str
                Season in which to evaluate the fluxes
                Default options are listed in the MCEq docs:
                https://mceq.readthedocs.io/en/latest/physics.html#MCEq.density_profiles.CorsikaAtmosphere.init_parameters

            * smoothing : ureg.Quantity
                Amount of smoothing to apply to splines

    input_names : string
        Specifies the string representation of the NuFlavIntGroup(s) that
        belong in the Data object passed to this service.

    output_binning : MultiDimBinning or convertible thereto
        The binning desired for the output maps.

    output_names : string
        Specifies the string representation of the NuFlavIntGroup(s) which will
        be produced as an output.

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
    def __init__(self, params, output_binning, error_method=None,
                 debug_mode=None, disk_cache=None, memcache_deepcopy=True,
                 outputs_cache_depth=20):
        self.mceq_hash = None
        """Hash of primary flux."""

        expected_params = (
            'interaction_model', 'primary_model', 'density_model', 'location',
            'season', 'smoothing'
        )

        output_names = (
            'nue', 'numu', 'nuebar', 'numubar'
        )

        super().__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            output_names=output_names,
            error_method=error_method,
            debug_mode=debug_mode,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning,
        )

    @profile
    def _compute_outputs(self, inputs=None):
        """Compute histograms for output channels."""
        logging.debug('Entering mceq._compute_outputs')

        primary_model = split(self.params['primary_model'].value, ',')
        if len(primary_model) != 2:
            raise ValueError('primary_model is not of length 2, instead is of '
                             'length {0}'.format(len(primary_model)))
        primary_model[0] = eval('pm.'+primary_model[0])
        density_model = (
            self.params['density_model'].value, (self.params['location'].value,
                                                 self.params['season'].value)
        )

        mceq_run = MCEqRun(
            interaction_model=str(self.params['interaction_model'].value),
            primary_model=primary_model,
            theta_deg=0.0,
            density_model=density_model,
            **mceq_config.mceq_config_without(['density_model'])
        )

        # Power of energy to scale the flux (the results will be returned as E**mag * flux)
        mag = 0

        # Obtain energy grid (fixed) of the solution for the x-axis of the plots
        e_grid = mceq_run.e_grid

        # Dictionary for results
        flux = OrderedDict()
        for nu in self.output_names:
            flux[nu] = []

        binning = self.output_binning
        cz_binning = binning.dims[binning.index('coszen', use_basenames=True)]
        en_binning = binning.dims[binning.index('energy', use_basenames=True)]
        cz_centers = cz_binning.weighted_centers.m
        angles = (np.arccos(cz_centers)*ureg.radian).m_as('degrees')

        for theta in angles:
            mceq_run.set_theta_deg(theta)
            mceq_run.solve()

            flux['nue'].append(mceq_run.get_solution('total_nue', mag))
            flux['nuebar'].append(mceq_run.get_solution('total_antinue', mag))
            flux['numu'].append(mceq_run.get_solution('total_numu', mag))
            flux['numubar'].append(mceq_run.get_solution('total_antinumu', mag))

        for nu in flux.keys():
            flux[nu] = np.array(flux[nu])

        smoothing = self.params['smoothing'].value.m
        en_centers = en_binning.weighted_centers.m_as('GeV')
        spline_flux = self.bivariate_spline(
            flux, cz_centers, e_grid, smooth=smoothing
        )
        ev_flux = self.bivariate_evaluate(
            spline_flux, cz_centers, en_centers
        )

        for nu in ev_flux:
            ev_flux[nu] = ev_flux[nu] *  ureg('cm**-2 s**-1 sr**-1 GeV**-1')

        mapset = []
        for nu in ev_flux.keys():
            mapset.append(Map(name=nu, hist=ev_flux[nu], binning=binning))

        return MapSet(mapset)

    @staticmethod
    def bivariate_spline(flux_dict, cz_centers, en_centers, smooth=0.02):
        """Spline the flux."""
        logging.debug('Entering mceq.bivariate_spline')
        Cz, logE = np.meshgrid(cz_centers, np.log10(en_centers))

        spline_dict = OrderedDict()
        for nu in flux_dict.keys():
            log_flux = np.log10(flux_dict[nu]).T
            spline = interpolate.bisplrep(Cz, logE, log_flux, s=smooth)
            spline_dict[nu] = spline
        return spline_dict

    @staticmethod
    def bivariate_evaluate(spline_dict, czvals, evals):
        """Evaluate the bivariate spline to get the flux."""
        fluxes = OrderedDict()
        for nu in spline_dict.keys():
            fluxes[nu] = np.power(10., interpolate.bisplev(
                czvals, np.log10(evals), spline_dict[nu]
            ))
        return fluxes

    def validate_params(self, params):
        pq = ureg.Quantity
        param_types = [
            ('interaction_model', str),
            ('primary_model', str),
            ('density_model', str),
            ('location', str),
            ('season', str),
            ('smoothing', pq)
        ]

        for p, t in param_types:
            val = params[p].value
            if not isinstance(val, t):
                raise TypeError(
                    'Param "%s" must be type %s but is %s instead'
                    % (p, type(t), type(val))
                )
