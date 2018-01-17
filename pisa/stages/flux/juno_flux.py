"""
Define flux service for JUNO
"""

from __future__ import absolute_import

import numpy as np
import math

from pisa import ureg
from pisa.core.stage import Stage
from pisa.core.map import Map, MapSet
from pisa.utils.hash import hash_obj

# Some fission parameter
f_U235 , f_U238 , f_Pu239 , f_Pu241 = 0.584 , 0.076 , 0.29 , 0.05
e_U235 , e_U238 , e_Pu239 , e_Pu241 = 202.36 , 205.99 , 211.12 , 214.26 

fe = f_U235 * e_U235 + f_U238 * e_U238 + f_Pu239 * e_Pu239 + f_Pu241 * e_Pu241

param = [['U235',3.217,-3.111,1.395,-0.369,0.04445,-0.002053],['U238',0.4883,0.1927,-0.1283,-0.006762,0.002233,-0.0001536]
         ,['Pu239',6.413,-7.432,3.535,-0.882,0.1025,-0.00455],['Pu241',3.251,-3.204,1.428,-0.3675,0.04254,-0.001896]]

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

    def __init__(self, params, output_binning, error_method,
                 outputs_cache_depth, memcache_deepcopy, disk_cache=None,
                 debug_mode=None):
        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
             'used_NPPs', 'corr_react_uncer', 'uncorr_react_uncer1', 'uncorr_react_uncer2', 'uncorr_react_uncer3', 
             'uncorr_react_uncer4', 'uncorr_react_uncer5', 'uncorr_react_uncer6', 'uncorr_react_uncer7', 'uncorr_react_uncer8', 
             'uncorr_react_uncer9', 'uncorr_react_uncer10', 'uncorr_react_uncer11', 'uncorr_react_uncer12'
        )

        # Define the names of objects that get produced by this stage
        output_names = (
            'nue', 'numu', 'nuebar', 'numubar'
        )

        # Invoke the init method from the parent class, which does a lot of
        # work for you. Note that we do not specify `input_names` here, since
        # there are no "inputs" used by this stage. (Of course there are
        # parameters, and files with info, but no maps or MC events are used
        # and transformed directly by this stage to produce its output.)
        super(juno_flux, self).__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

        # There might be other things to do at init time than what Stage does,
        # but typically this is not much... and it's almost always a good idea
        # to have "real work" defined in another method besides init, which can
        # then get called from init (so that if anyone else wants to do the
        # same "real work" after object instantiation, (s)he can do so easily
        # by invoking that same method).
    
    
    def _compute_outputs(self, inputs=None):
        
        # sys. parameter
        corr_react_uncer = self.params.corr_react_uncer.m_as('dimensionless')
        uncorr_react_uncer = [self.params.uncorr_react_uncer1.m_as('dimensionless'),
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
                              self.params.uncorr_react_uncer12.m_as('dimensionless')]
        
        # Nuclear Power Plants  
        NPPs = [['YJ-C1',2.9,52.75,13],['YJ-C2',2.9,52.84,16],['YJ-C3',2.9,52.42,7],['YJ-C4',2.9,52.51,9],['YJ-C5',2.9,52.12,0]
        ,['YJ-C6',2.9,52.21,3],['TS-C1',4.6,52.76,14],['TS-C2',4.6,52.63,11],['TS-C3',4.6,52.32,5],['TS-C4',4.6,52.2,2]
        ,['DYB',17.4,215,18],['HZ',17.4,265,20]] #arXiv:1507.05613 ; name,power[GW],distance[km],bin
        used_NPPs = self.params.used_NPPs.value
    
        num_NPP = []
        for i in range(len(NPPs)):
            if NPPs[i][0] in used_NPPs:
                num_NPP.append(i)
        
        if self.output_binning.shape[0] == 1: # only one NPP vicarious for the first 10
            num_NPP = [0]
            NPPs = [['all',36.0,52.5,0]]
 

        # Generating maps
        output_maps = []
        for output_name in self.output_names:
        
            hist = np.zeros(self.output_binning.shape)
            
            if output_name == 'nuebar': # NPP emit only electron antineutrinos
                # energy binning
                num_e_bins = self.output_binning.shape[1]
                e_min = (min(self.output_binning.bin_edges[1])).magnitude
                e_range = (max(self.output_binning.bin_edges[1]) - min(self.output_binning.bin_edges[1])).magnitude
                e_step = float(e_range)/num_e_bins
            
                for i in num_NPP:  # find NPP bins
                    cz_bin = NPPs[i][3]
                    
                    for j in range(num_e_bins):
                        E = e_min + e_step/2. + j * e_step # E at bin center
                
                        flux = get_flux(E,NPPs[i][2],NPPs[i][1]) * e_step * corr_react_uncer * uncorr_react_uncer[i]
                        hist[cz_bin][j] = flux

            # Put the "fluxes" into a Map object, give it the output_name
            m = Map(name=output_name, hist=hist, binning=self.output_binning)
        
            output_maps.append(m)

        # Combine the output maps into a single MapSet object to return.
        # The MapSet contains the varous things that are necessary to make
        # caching work and also provides a nice interface for the user to all
        # of the contained maps
        return MapSet(maps=output_maps, name='flux maps')

    #def validate_params(self, params):

