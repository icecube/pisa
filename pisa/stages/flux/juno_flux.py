"""
Define flux service for JUNO
"""

from __future__ import absolute_import

import numpy as np
import math

from pisa import ureg
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.core.map import Map, MapSet
from pisa.core.container import Container
from pisa.utils.hash import hash_obj
from pisa.utils.profiler import profile

__all__ = ['juno_flux', 'func', 'get_flux', 'init_test']


# Some fission parameter
f_U235 , f_U238 , f_Pu239 , f_Pu241 = 0.584 , 0.076 , 0.29 , 0.05
e_U235 , e_U238 , e_Pu239 , e_Pu241 = 202.36 , 205.99 , 211.12 , 214.26 

fe = f_U235 * e_U235 + f_U238 * e_U238 + f_Pu239 * e_Pu239 + f_Pu241 * e_Pu241

param = [['U235',3.217,-3.111,1.395,-0.369,0.04445,-0.002053],['U238',0.4883,0.1927,-0.1283,-0.006762,0.002233,-0.0001536],
         ['Pu239',6.413,-7.432,3.535,-0.882,0.1025,-0.00455],['Pu241',3.251,-3.204,1.428,-0.3675,0.04254,-0.001896]]

# the assumed flux function
def func(x, a1, a2, a3, a4, a5, a6):
    return np.exp(a1 + a2 * x + a3 * x**2 + a4 * x**3 + a5 * x**4 + a6 * x**5)
    
def get_flux(E,d,w): #E[MeV] d[km] w[GW]
    W = w * 6.2415*10**12 * 10**9 #MeV/s

    F_U235 = f_U235 * func(E,param[0][1],param[0][2],param[0][3],param[0][4],param[0][5],param[0][6])
    F_U238 = f_U238 * func(E,param[1][1],param[1][2],param[1][3],param[1][4],param[1][5],param[1][6])
    F_Pu239 = f_Pu239 * func(E,param[2][1],param[2][2],param[2][3],param[2][4],param[2][5],param[2][6])
    F_Pu241 = f_Pu241 * func(E,param[3][1],param[3][2],param[3][3],param[3][4],param[3][5],param[3][6])
    
    flux = W/fe * (F_U235 + F_U238 + F_Pu239 + F_Pu241) #1/s MeV
    flux = flux / (4 * math.pi * (d*100000)**2) #1/cm2 s MeV
        
    return flux


class juno_flux(Stage): # pylint: disable=invalid-name

    def __init__(
        self,
        output_names=None,
        NEvents=1,
        **std_kwargs,
    ):
        expected_params = (
            'used_NPPs', 
            'corr_react_uncer',
            'uncorr_react_uncer1', 
            'uncorr_react_uncer2', 
            'uncorr_react_uncer3', 
            'uncorr_react_uncer4',
            'uncorr_react_uncer5',
            'uncorr_react_uncer6',
            'uncorr_react_uncer7',
            'uncorr_react_uncer8', 
            'uncorr_react_uncer9',
            'uncorr_react_uncer10',
            'uncorr_react_uncer11',
            'uncorr_react_uncer12'
        )
        
        self.output_names = output_names
        self.NEvents = int(NEvents)

        super().__init__(
            expected_params=expected_params,
            expected_container_keys=(),
            **std_kwargs,
        )
        
    def setup_function(self):
        for output_name in self.output_names:
            if not output_name in self.data.names:
                container = Container(output_name)
                self.data.add_container(container)

        # Nuclear Power Plants  
        NPPs = [['YJ-C1',2.9,52.75,13],['YJ-C2',2.9,52.84,16],['YJ-C3',2.9,52.42,7],['YJ-C4',2.9,52.51,9],['YJ-C5',2.9,52.12,0]
        ,['YJ-C6',2.9,52.21,3],['TS-C1',4.6,52.76,14],['TS-C2',4.6,52.63,11],['TS-C3',4.6,52.32,5],['TS-C4',4.6,52.2,2]
        ,['DYB',17.4,215,18],['HZ',17.4,265,20]] #arXiv:1507.05613 ; name,power[GW],distance[km],bin
        used_NPPs = self.params.used_NPPs.value
    
        num_NPP = []
        for i in range(len(NPPs)):
            if NPPs[i][0] in used_NPPs:
                num_NPP.append(i)
        
        E_index = self.calc_mode.names.index("true_energy")
        cz_index = self.calc_mode.names.index("true_coszen")

        # energy binning
        num_e_bins = self.calc_mode.shape[E_index]
        e_min = (min(self.calc_mode.bin_edges[E_index])).magnitude
        e_range = (max(self.calc_mode.bin_edges[E_index]) - min(self.calc_mode.bin_edges[E_index])).magnitude
        e_step = float(e_range)/num_e_bins

        # coszen binning
        cz_bins = (self.calc_mode.bin_edges[cz_index][1:] + self.calc_mode.bin_edges[cz_index][:-1]) / 2

        # only one NPP vicarious for the first 10?
        if self.calc_mode.shape[cz_index] == 1: 
            num_NPP = [0]
            NPPs = [['all',36.0,52.5,0]]

        self.data.representation = "events"

        # Generating maps
        for output_name in self.output_names:
            Es, czs, flux, reactor = [], [], [], []

            for i in num_NPP:  # find NPP bins
                cz_bin = NPPs[i][3]
                cz = cz_bins[cz_bin] # cz at bin center

                for j in range(num_e_bins):
                    E = e_min + e_step/2. + j * e_step # E at bin center
                    Es.append(E)
                    czs.append(cz)

                    if output_name == 'nuebar_cc': # NPP emit only electron antineutrinos
                        fl = get_flux(E, NPPs[i][2], NPPs[i][1]) * e_step
                    else:
                        fl = 0
                    flux.append(fl)
                    reactor.append(i)

            self.data[output_name]['initial_weights'] = np.ones(len(flux)*self.NEvents) / self.NEvents
            self.data[output_name]['nom_nu_flux'] = np.stack([np.repeat(flux, self.NEvents), np.zeros(len(flux)*self.NEvents)], axis=1)
            self.data[output_name]['reactor'] = np.repeat(reactor, self.NEvents)
            self.data[output_name]['true_energy'] = np.repeat(Es, self.NEvents) * 1e-3 #MeV to GeV
            self.data[output_name]['true_coszen'] = np.repeat(czs, self.NEvents)
            self.data[output_name].set_aux_data('nubar', -1)
            self.data[output_name].set_aux_data('flav', 0)
            
    @profile
    def apply_function(self):
        self.data.representation = "events"
        
        # sys. parameter
        corr_react_uncer = self.params.corr_react_uncer.m_as('dimensionless')
        uncorr_react_uncer = np.array([self.params.uncorr_react_uncer1.m_as('dimensionless'),
                                       self.params.uncorr_react_uncer2.m_as('dimensionless'),
                                       self.params.uncorr_react_uncer3.m_as('dimensionless'),
                                       self.params.uncorr_react_uncer4.m_as('dimensionless'),
                                       self.params.uncorr_react_uncer5.m_as('dimensionless'),
                                       self.params.uncorr_react_uncer6.m_as('dimensionless'),
                                       self.params.uncorr_react_uncer7.m_as('dimensionless'),
                                       self.params.uncorr_react_uncer8.m_as('dimensionless'),
                                       self.params.uncorr_react_uncer9.m_as('dimensionless'),
                                       self.params.uncorr_react_uncer10.m_as('dimensionless'),
                                       self.params.uncorr_react_uncer11.m_as('dimensionless'),
                                       self.params.uncorr_react_uncer12.m_as('dimensionless'),
                                      ])
        
        for container in self.data:
            container['weights'] = np.copy(container['initial_weights'])
            container['nu_flux'] = np.copy(container['nom_nu_flux']) * corr_react_uncer * np.stack([uncorr_react_uncer[container['reactor']], np.zeros(container.size)], axis=1)


def init_test(**param_kwargs):
    """Instantiation example"""
    param_set = [Param(name='corr_react_uncer', value=1.0,  **param_kwargs)]
    for i in range(1,13):
        param_set.append(Param(name='uncorr_react_uncer'+str(i), value=1.0,  **param_kwargs))
    param_set.append(Param(name='used_NPPs', value="['YJ-C1','YJ-C2','YJ-C3','YJ-C4','YJ-C5','YJ-C6','TS-C1','TS-C2','DYB','HZ']",  **param_kwargs))
    param_set = ParamSet(*param_set)

    e_binning = OneDimBinning(name='true_energy', is_log=True, num_bins=10, domain=[2.0, 8.0]*ureg.MeV)
    cz_binning = OneDimBinning(name='true_coszen', is_lin=True, num_bins=1, domain=[-1, 1])
    binning = MultiDimBinning([cz_binning, e_binning])

    return juno_flux(output_names=['nue_cc', 'nuebar_cc', 'test1_cc', 'test2_nc'], params=param_set, calc_mode=binning)
