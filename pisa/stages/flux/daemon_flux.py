"""
Implementation of DAEMON flux (https://arxiv.org/abs/2303.00022) 
by Juan Pablo Yañez and Anatoli Fedynitch for use in PISA.

Maria Liubarska, J.P. Yanez 2023
"""

import numpy as np
from daemonflux import Flux
from daemonflux import __version__ as daemon_version

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.core.param import Param
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from numba import jit
from scipy import interpolate
from packaging.version import Version


class daemon_flux(Stage):
    """
    DAEMON flux stage
    
    Parameters
    ----------

    params: ParamSet
        Must have parameters: .. ::

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

    """

    def __init__(
        self,
        **std_kwargs,
    ):

        # first have to check daemonflux package version is >=0.8.0
        # (have to do this to ensure chi2 prior penalty is calculated correctly)
        if Version(daemon_version) < Version("0.8.0"):
            logging.fatal("Detected daemonflux version below 0.8.0! This will lead to incorrect penalty calculation. You must update your daemonflux package to use this stage. You can do it by running 'pip install daemonflux --upgrade'")
            raise Exception('detected daemonflux version < 0.8.0')

        # create daemonflux Flux object
        self.flux_obj = Flux(location='IceCube', use_calibration=True)

        # get parameter names from daemonflux
        self.daemon_names = self.flux_obj.params.known_parameters

        # make parameter names pisa config compatible and add prefix
        self.daemon_params = ['daemon_'+p.replace('pi+','pi').replace('pi-','antipi').replace('K+','K').replace('K-','antiK') 
                              for p in self.daemon_names]

        # add daemon_chi2 internal parameter to carry on chi2 penalty from daemonflux (using covar. matrix)
        daemon_chi2 = Param(name='daemon_chi2', nominal_value=0., 
                            value=0., prior=None, range=None, is_fixed=True)

        # saving number of parameters into a internal param in order to check that we don't have 
        # non-daemonflux params with 'daemon_' in their name, which will make prior penalty calculation incorrect
        daemon_params_len = Param(name='daemon_params_len', nominal_value=len(self.daemon_names)+2,
                                  value=len(self.daemon_names)+2, prior=None, range=None, is_fixed=True)

        std_kwargs['params'].update([daemon_chi2,daemon_params_len])

        # init base class
        super(daemon_flux, self).__init__(
            expected_params=tuple(self.daemon_params+['daemon_chi2','daemon_params_len']),
            **std_kwargs,
        )

    def setup_function(self):

        self.data.representation = self.calc_mode

        for container in self.data:
            container['nu_flux'] = np.empty((container.size, 2), dtype=FTYPE)

    @profile
    def compute_function(self):

        self.data.representation = self.calc_mode

        # get modified parameters (in units of sigma)
        modif_param_dict = {}
        for i,k in enumerate(self.daemon_params):
            modif_param_dict[self.daemon_names[i]] = getattr(self.params, k).value.m_as("dimensionless")

        # update chi2 parameter
        self.params['daemon_chi2'].value = self.flux_obj.chi2(modif_param_dict)

        # compute flux maps
        flux_map_numu    = make_2d_flux_map(self.flux_obj,
                                            particle = 'numu',
                                            params = modif_param_dict)
        flux_map_numubar = make_2d_flux_map(self.flux_obj,
                                            particle = 'antinumu',
                                            params = modif_param_dict)
        flux_map_nue     = make_2d_flux_map(self.flux_obj,
                                            particle = 'nue',
                                            params = modif_param_dict)
        flux_map_nuebar  = make_2d_flux_map(self.flux_obj,
                                            particle = 'antinue',
                                            params = modif_param_dict)
        

        # calc modified flux using provided parameters
        for container in self.data:
            nubar = container['nubar']

            nue_flux   = evaluate_flux_map(flux_map_nuebar if nubar>0 else flux_map_nue, 
                                           container['true_energy'],
                                           container['true_coszen'])

            numu_flux  = evaluate_flux_map(flux_map_numubar if nubar>0 else flux_map_numu, 
                                           container['true_energy'],
                                           container['true_coszen'])


            container['nu_flux'][:,0] = nue_flux
            container['nu_flux'][:,1] = numu_flux

            container.mark_changed("nu_flux")

@jit(forceobj=True)
def make_2d_flux_map(flux_obj,
                     particle = 'numuflux',
                     egrid = np.logspace(-1,5,500),
                     params = {},
                     ):

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

@jit(forceobj=True)
def evaluate_flux_map(flux_map, true_energy, true_coszen):

    uconv = true_energy**-3 * 1e4
    return flux_map.ev(true_energy, true_coszen) * uconv
            
