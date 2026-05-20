"""
Implementation of DAEMON flux (https://arxiv.org/abs/2303.00022)
by Juan Pablo Yañez and Anatoli Fedynitch for use in PISA.

Module originally authored by Maria Liubarska, J.P. Yañez 2023
"""
# TODO:
# * does every parameter affect all fluxes (i.e., further caching potential)?
# * remove piecewise (manual) timings for production
# * any benefit from optimising/adapting arguments of fast_interp.interp2d?
# * why does there not seem to be any speed up from parallel (multi-threaded)
# interpolant evaluation in fast_interp mode even though true_energy.size >
# fast_interp.fast_interp.serial_cutoffs[2]?

import time

from daemonflux import Flux
from daemonflux import __version__ as daemon_version
import fast_interp

import numpy as np
from packaging.version import Version
from scipy import interpolate

from pisa import FTYPE, ureg
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.comparisons import ALLCLOSE_KW
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.random_numbers import get_random_state

__all__ = ['MIN_VERSION', 'ENERGY_GRID', 'daemon_flux', 'init_test']

MIN_VERSION = "0.8.0"
"""Minimum daemonflux package version for correct chi2 prior penalty"""

ENERGY_GRID = np.logspace(-1, 5, 500, dtype=FTYPE) * ureg.GeV
"""Default array of true neutrino energies at which to request fluxes
from daemonflux"""


class daemon_flux(Stage):  # pylint: disable=invalid-name
    """
    DAEMON flux stage

    Parameters
    ----------

    calibration_file: str, optional
        Path to the calibration file to be used

    use_fast_interp : bool (default: False)
        Whether to use numba-accelerated flux interpolation
        (experimental)

    energy_grid : array-like (default: None)
        Array of true neutrino energies at which to request fluxes
        from daemonflux. If `None`, the default :py:data:`ENERGY_GRID`
        will be used.

    params: ParamSet
        Must have parameters::

            daemon_K_158G : quantity (dimensionless)
            daemon_K_2P : quantity (dimensionless)
            daemon_K_31G : quantity (dimensionless)
            daemon_antiK_158G : quantity (dimensionless)
            daemon_antiK_2P : quantity (dimensionless)
            daemon_antiK_31G : quantity (dimensionless)
            daemon_n_158G : quantity (dimensionless)
            daemon_n_2P : quantity (dimensionless)
            daemon_p_158G : quantity (dimensionless)
            daemon_p_2P : quantity (dimensionless)
            daemon_pi_158G : quantity (dimensionless)
            daemon_pi_20T : quantity (dimensionless)
            daemon_pi_2P : quantity (dimensionless)
            daemon_pi_31G : quantity (dimensionless)
            daemon_antipi_158G : quantity (dimensionless)
            daemon_antipi_20T : quantity (dimensionless)
            daemon_antipi_2P : quantity (dimensionless)
            daemon_antipi_31G : quantity (dimensionless)
            daemon_GSF_1 : quantity (dimensionless)
            daemon_GSF_2 : quantity (dimensionless)
            daemon_GSF_3 : quantity (dimensionless)
            daemon_GSF_4 : quantity (dimensionless)
            daemon_GSF_5 : quantity (dimensionless)
            daemon_GSF_6 : quantity (dimensionless)

    Notes
    -----

    Expected container keys are::

        "true_energy", "true_coszen", "nubar"

    """

    def __init__(self, calibration_file=None, use_fast_interp=False, energy_grid=None,
                 **std_kwargs):
        self.cal_file = calibration_file
        if self.cal_file is not None:
            logging.debug('Requested custom DAEMON flux calibration file: %s',
                          self.cal_file)

        if Version(daemon_version) < Version(MIN_VERSION):
            logging.fatal(
                "Detected daemonflux version %s < %s, which will lead to"
                " incorrect penalty calculations. You must update your"
                " daemonflux package to use this stage. You can do it by"
                " running 'pip install daemonflux --upgrade'.",
                daemon_version, MIN_VERSION
            )
            raise RuntimeError(f'Detected daemonflux version < {MIN_VERSION}')

        # numba-accelerated flux interpolation
        self.fast_interp = use_fast_interp

        # create daemonflux Flux object
        self.flux_obj = Flux(location='IceCube', use_calibration=True,
                             cal_file=self.cal_file)

        # self.flux_obj.zenith_angles is list of strings of (approx.) uniformly-spaced
        # values in deg between 0° & 180° -> make ascending array first
        self.icangles_asc = np.array(sorted(map(float, self.flux_obj.zenith_angles),
                                       reverse=False), dtype=FTYPE)
        self.costheta_angles_asc = np.cos(np.deg2rad(self.icangles_asc))[::-1]

        if energy_grid is None:
            energy_grid = ENERGY_GRID.m_as("GeV")
        else:
            # user requested a custom grid
            if isinstance(energy_grid, str):
                energy_grid = eval(energy_grid)
            # required to be specified with energy units
            assert (hasattr(energy_grid, "magnitude") and
                    energy_grid.is_compatible_with("GeV"))
            energy_grid = energy_grid.m_as("GeV")
            if not isinstance(energy_grid, (list, np.ndarray)):
                energy_grid = [energy_grid]
            logging.debug("Requested custom energy grid (GeV) to pass to daemonflux: "
                          "%s", energy_grid)
        self.egrid = energy_grid

        if self.fast_interp:
            logging.debug("Using daemon_flux service with fast interpolation...")
            # Obtain uniform spacings in interpolation dimensions:
            # Cosine zenith we got from daemonflux
            costheta_deltas = self.costheta_angles_asc[1:] - self.costheta_angles_asc[:-1]
            # TODO: These deltas don't quite seem to agree at desired precision, looks
            # like numerical inaccuracy.
            #assert np.allclose(costheta_delta[0], costheta_deltas, **ALLCLOSE_KW)
            self.costheta_delta = costheta_deltas[0]

            if not OneDimBinning.is_bin_spacing_log_uniform(self.egrid):
                # Could in principle relax this, but then couldn't just assume that
                # log(energy) is uniform.
                raise ValueError(
                    "Energy grid required to have log-uniform spacing"
                    " (energy_grid=np.logspace(...)) when fast_interp=True."
                )
            # Energy dimension is made uniform by taking log
            self.egrid_log = np.log10(self.egrid)
            egrid_log_deltas = self.egrid_log[1:] - self.egrid_log[:-1]
            if not np.allclose(egrid_log_deltas[0], egrid_log_deltas, **ALLCLOSE_KW):
                raise ValueError("Need uniformly-spaced log-energy values!")
            self.egrid_log_delta = egrid_log_deltas[0]

        # get parameter names from daemonflux
        self.daemon_names = self.flux_obj.params.known_parameters

        # make parameter names pisa config compatible and add prefix
        self.daemon_params = ['daemon_' +
            p.replace('pi+','pi').replace('pi-','antipi')
            .replace('K+','K').replace('K-','antiK') for p in self.daemon_names
        ]

        # Add daemon_chi2 internal parameter to carry on chi2 penalty
        # from daemonflux (using covar. matrix)
        daemon_chi2 = Param(
            name='daemon_chi2',
            nominal_value=0., value=0.,
            prior=None, range=None, is_fixed=True
        )

        # Saving number of parameters into a internal param in order to check
        # that we don't have non-daemonflux params with 'daemon_' in their
        # name, which will make prior penalty calculation incorrect
        daemon_params_len = Param(
            name='daemon_params_len',
            nominal_value=len(self.daemon_names)+2,
            value=len(self.daemon_names)+2,
            prior=None, range=None, is_fixed=True
        )
        std_kwargs['params'].update([daemon_chi2, daemon_params_len])

        expected_params = tuple(
            self.daemon_params + ['daemon_chi2', 'daemon_params_len']
        )
        expected_container_keys = (
            'true_energy',
            'true_coszen',
            'nubar',
        )
        # event-by-event or binned fluxes; no apply_function
        supported_reps = {
            'calc_mode': ["events", MultiDimBinning],
            'apply_mode': [None],
        }

        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            supported_reps=supported_reps,
            **std_kwargs,
        )

    def setup_function(self):
        """
        Just prepare empty nue(bar) & numu(bar) atmos. flux arrays on setup.
        """
        for container in self.data:
            container['nu_flux'] = np.empty((container.size, 2), dtype=FTYPE)

    @profile
    def compute_function(self):
        """
        Compute nominal atmospheric fluxes of nue(bar) and numu(bar).
        """
        # get modified parameters (in units of sigma)
        modif_param_dict = {}
        for i, k in enumerate(self.daemon_params):
            modif_param_dict[self.daemon_names[i]] = self.params[k].value.m_as("dimensionless")

        # update chi2 parameter
        self.params['daemon_chi2'].value = self.flux_obj.chi2(modif_param_dict)

        # Compute flux maps
        flux_map_numu = self.make_2d_flux_map(
            particle='numu', params=modif_param_dict
        )
        flux_map_numubar = self.make_2d_flux_map(
            particle='antinumu', params=modif_param_dict
        )
        flux_map_nue = self.make_2d_flux_map(
            particle='nue', params=modif_param_dict
        )
        flux_map_nuebar = self.make_2d_flux_map(
            particle='antinue', params=modif_param_dict
        )

        ntot = 0
        start_t = time.time()
        # Obtain modified fluxes using provided parameters at desired points
        # in true_energy and true_coszen now
        for container in self.data:
            nubar = container['nubar']
            nue_flux = self.evaluate_flux_map(
                flux_map=flux_map_nuebar if nubar < 0 else flux_map_nue,
                true_energy=container['true_energy'],
                true_coszen=container['true_coszen']
            )
            numu_flux = self.evaluate_flux_map(
                flux_map=flux_map_numubar if nubar < 0 else flux_map_numu,
                true_energy=container['true_energy'],
                true_coszen=container['true_coszen']
            )
            container['nu_flux'][:, 0] = nue_flux
            container['nu_flux'][:, 1] = numu_flux
            container.mark_changed("nu_flux")
            ntot += len(container['true_energy'])
        stop_t = time.time()
        logging.info("PISA spline evaluation time (s, %d events): %s", ntot, f"{stop_t - start_t:.2e}")


    def make_2d_flux_map(self, particle='numuflux', params=None):
        """
        Create an interpolated 2d (energy-coszen) flux map using daemonflux.

        Parameters
        ----------
        particle : str (default: "numuflux")
            Type of flux to be returned. See `daemonflux.flux.Flux.quantities`.
        params : Dict[str, float], optional
            Dictionary of parameter values for off-baseline shifts.

        Returns
        -------
        fcn : interpolant
            Bivariate spline approximation of flux over energy-coszen grid.
            `scipy.interpolate.RectBivariateSpline` if not `fast_interp`,
            otherwise `fast_interp.interp2d`. See `daemonflux.Flux.flux()` for
            units.
        """
        if params is None:
            params = {}
        start_t = time.time()
        # Obtain flux from daemonflux: expects ascending zenith angles in deg
        # TODO: Why does flux_obj need these to be handed back to it again?
        flux_ref = self.flux_obj.flux(
            energy=self.egrid,
            zenith_deg=self.icangles_asc,
            quantity=particle,
            params=params
        )
        stop_t = time.time()
        logging.info("daemonflux %s flux generation duration (s): %s", particle, f"{stop_t - start_t:.2e}")
        # Now flip zenith angle dimension so we can interpolate in costheta
        # with increasing costheta
        flux_ref_lr = np.fliplr(flux_ref)

        start_t = time.time()
        # Return interpolant which can be evaluated later
        if not self.fast_interp:
            fcn = interpolate.RectBivariateSpline(
                x=self.egrid, y=self.costheta_angles_asc, z=flux_ref_lr
            )
        else:
            fcn = fast_interp.interp2d(
                a=[min(self.egrid_log), min(self.costheta_angles_asc)],
                b=[max(self.egrid_log), max(self.costheta_angles_asc)],
                h=[self.egrid_log_delta, self.costheta_delta],
                f=flux_ref_lr,
                k=3,
                p=[False, False],
                c=[True, True],
                e=[0, 0]
            )
        stop_t = time.time()
        logging.info("splining duration (s): %s", f"{stop_t - start_t:.2e}")
        return fcn


    def evaluate_flux_map(self, flux_map, true_energy, true_coszen):
        """
        Evaluate bivariate spline approximation of flux.

        Parameters
        ----------
        flux_map : scipy.interpolate.RectBivariateSpline
        true_energy : array_like
            True energies in GeV at which to evaluate
        true_coszen : array_like
            True coszens at which to evaluate

        Returns
        -------
        ndarray
            Flux in units of 1/(GeV m² s sr)
        """
        # flux unit conversion factor (see daemonflux.Flux.flux())
        uconv = true_energy**-3 * 1e4
        if not self.fast_interp:
            return flux_map.ev(true_energy, true_coszen) * uconv
        # Remember to transform into log-energy space again before evaluating
        return flux_map(np.log10(true_energy), true_coszen) * uconv


def init_test(**param_kwargs):
    """Initialisation example"""
    param_set = []
    random_state = get_random_state(random_state=666)
    for pname in Flux(location='IceCube', use_calibration=True).params.known_parameters:
        param = Param(
            name='daemon_' + pname.replace('pi+','pi').replace('pi-','antipi')
                 .replace('K+','K').replace('K-','antiK'),
            value=2 * random_state.rand() - 1,
            **param_kwargs
        )
        param_set.append(param)
    param_set = ParamSet(*param_set)
    return daemon_flux(params=param_set)
