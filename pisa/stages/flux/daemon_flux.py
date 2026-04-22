"""
Implementation of DAEMON flux (https://arxiv.org/abs/2303.00022)
by Juan Pablo Yañez and Anatoli Fedynitch for use in PISA.

Maria Liubarska, J.P. Yanez 2023
"""

from daemonflux import Flux
from daemonflux import __version__ as daemon_version

import numpy as np
from packaging.version import Version
from scipy import interpolate

from pisa import FTYPE
from pisa.core.binning import MultiDimBinning
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.random_numbers import get_random_state

__all__ = ['MIN_VERSION', 'daemon_flux', 'make_2d_flux_map',
           'evaluate_flux_map', 'init_test']

MIN_VERSION = "0.8.0"
"""Minimum daemonflux package version for correct chi2 prior penalty"""

class daemon_flux(Stage):  # pylint: disable=invalid-name
    """
    DAEMON flux stage

    Parameters
    ----------

    calibration_file: str, optional
        Path to the calibration file to be used
    
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

    def __init__(self, calibration_file=None, **std_kwargs):
        self.cal_file = calibration_file
        logging.debug('DAEMON flux calibration file: %s', self.cal_file)

        if Version(daemon_version) < Version(MIN_VERSION):
            logging.fatal(
                "Detected daemonflux version %s < %s, which will lead to"
                " incorrect penalty calculations. You must update your"
                " daemonflux package to use this stage. You can do it by"
                " running 'pip install daemonflux --upgrade'.",
                daemon_version, MIN_VERSION
            )
            raise RuntimeError('detected daemonflux version < 0.8.0')

        # create daemonflux Flux object
        self.flux_obj = Flux(location='IceCube', use_calibration=True,
                             cal_file=self.cal_file)

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
        self.data.representation = self.calc_mode

        # get modified parameters (in units of sigma)
        modif_param_dict = {}
        for i, k in enumerate(self.daemon_params):
            modif_param_dict[self.daemon_names[i]] = self.params[k].value.m_as("dimensionless")

        # update chi2 parameter
        self.params['daemon_chi2'].value = self.flux_obj.chi2(modif_param_dict)

        # compute flux maps
        flux_map_numu = make_2d_flux_map(
            flux_obj=self.flux_obj, particle='numu', params=modif_param_dict
        )
        flux_map_numubar = make_2d_flux_map(
            flux_obj=self.flux_obj, particle='antinumu', params=modif_param_dict
        )
        flux_map_nue = make_2d_flux_map(
            flux_obj=self.flux_obj, particle='nue', params=modif_param_dict
        )
        flux_map_nuebar = make_2d_flux_map(
            flux_obj=self.flux_obj, particle='antinue', params=modif_param_dict
        )

        # calc modified flux using provided parameters
        for container in self.data:
            nubar = container['nubar']
            nue_flux = evaluate_flux_map(
                flux_map=flux_map_nuebar if nubar<0 else flux_map_nue,
                true_energy=container['true_energy'],
                true_coszen=container['true_coszen']
            )
            numu_flux = evaluate_flux_map(
                flux_map=flux_map_numubar if nubar<0 else flux_map_numu,
                true_energy=container['true_energy'],
                true_coszen=container['true_coszen']
            )
            container['nu_flux'][:,0] = nue_flux
            container['nu_flux'][:,1] = numu_flux
            container.mark_changed("nu_flux")


def make_2d_flux_map(flux_obj,
                     particle='numuflux',
                     egrid=np.logspace(-1, 5, 500),
                     params=None,
                     ):
    """
    Create an interpolated 2d (energy-coszen) flux map using daemonflux.

    Parameters
    ----------
    flux_obj : daemonflux.Flux
    particle : str (default: "numuflux")
        Type of flux to be returned. See `daemonflux.flux.Flux.quantities`.
    egrid : float or np.ndarray (default: np.logspace(-1, 5, 500))
        True energy/energies in GeV at which to compute flux.
    params : Dict[str, float], optional
        Dictionary of parameter values for off-baseline shifts.

    Returns
    -------
    fcn : scipy.interpolate.RectBivariateSpline
        Bivariate spline approximation of flux over energy-coszen grid.
        See `daemonflux.Flux.flux()` for units.
    """
    if params is None:
        params = {}
    icangles = list(flux_obj.zenith_angles)
    icangles_array = np.array(icangles, dtype=float)
    mysort = icangles_array.argsort()
    icangles = np.array(icangles)[mysort][::-1]

    flux_ref = np.zeros([len(egrid), len(icangles)])
    costheta_angles = np.zeros(len(icangles))

    for index in range(len(icangles)):
        costheta_angles[index] = np.cos(np.deg2rad(float(icangles[index])))
        flux_ref[:,index] = flux_obj.flux(egrid, icangles[index], particle, params)

    fcn = interpolate.RectBivariateSpline(egrid,
                                          costheta_angles,
                                          flux_ref)
    return fcn


def evaluate_flux_map(flux_map, true_energy, true_coszen):
    """
    Evaluate bivariate spline approximation of flux.

    Parameters
    ----------
    flux_map : scipy.interpolate.RectBivariateSpline
    true_energy : Sequence
        List of true energies in GeV at which to evaluate.
    true_coszen : Sequence
        List of true coszens at which to evaluate.

    Returns
    -------
    np.ndarray
        Flux in units of 1/(GeV m² s sr)
    """
    # flux unit conversion factor (see See daemonflux.Flux.flux())
    uconv = true_energy**-3 * 1e4
    return flux_map.ev(true_energy, true_coszen) * uconv


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
