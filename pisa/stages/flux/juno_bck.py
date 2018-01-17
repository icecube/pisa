"""
Define flux service for JUNO background
"""

from __future__ import absolute_import
from scipy.interpolate import interp1d

import numpy as np

from pisa import ureg
from pisa.core.stage import Stage
from pisa.core.map import Map, MapSet
from pisa.utils.hash import hash_obj


Accidental = np.loadtxt('/fs02/stud12/homes/jawelder/JUNO/JUNO data/Accidental.csv',delimiter=',')
Fast_neutron = np.loadtxt('/fs02/stud12/homes/jawelder/JUNO/JUNO data/Fast_neutron.csv',delimiter=',')
Li_He = np.loadtxt('/fs02/stud12/homes/jawelder/JUNO/JUNO data/Li_He.csv',delimiter=',')
alpha_n = np.loadtxt('/fs02/stud12/homes/jawelder/JUNO/JUNO data/alpha_n.csv',delimiter=',')
Geo_neutrino = np.loadtxt('/fs02/stud12/homes/jawelder/JUNO/JUNO data/Geo_neutrino.csv',delimiter=',')


def get_xy(lis):
    x , y = [] , []
    
    for i in range(len(lis)):
        x.append(lis[i][0])
        y.append(lis[i][1])
    
    return x , y
        

class juno_bck(Stage): # pylint: disable=invalid-name

    def __init__(self, params, output_binning, error_method,
                 outputs_cache_depth, memcache_deepcopy, disk_cache=None,
                 debug_mode=None):
        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
             'Accidental', 'alpha_n', 'Fast_neutrons', 'Li_He', 'Geo_neutrinos', 'livetime'
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
        super(juno_bck, self).__init__(
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
        
        livetime = self.params.livetime.m_as('day')
        
        Acc = self.params.Accidental.value      # check which background should be in
        alp = self.params.alpha_n.value
        Fas = self.params.Fast_neutrons.value
        LiH = self.params.Li_He.value
        Geo = self.params.Geo_neutrinos.value
        
        x , y = get_xy(Accidental)       # interpolate background linaer
        Acci_int = interp1d(x,y)
        
        x , y = get_xy(Fast_neutron)
        Neut_int = interp1d(x,y)
        
        x , y = get_xy(Li_He)
        LiHe_int = interp1d(x,y)
        
        x , y = get_xy(alpha_n)
        Alph_int = interp1d(x,y)
        
        x , y = get_xy(Geo_neutrino)
        Geon_int = interp1d(x,y)
        
        
        # Generating maps
        output_maps = []
        for output_name in self.output_names:
        
            hist = np.zeros(self.output_binning.shape)
            
            if output_name == 'nuebar': # only electron antineutrino background
                # energy binning
                num_e_bins = self.output_binning.shape[1]
                e_min = (min(self.output_binning.bin_edges[1])).magnitude
                e_range = (max(self.output_binning.bin_edges[1]) - min(self.output_binning.bin_edges[1])).magnitude
                e_step = float(e_range)/num_e_bins
            
                    
                for j in range(num_e_bins):
                    E = e_min + e_step/2. + j * e_step # E at bin center
                
                    flux = 0
                    if Acc == True : flux += Acci_int(E) * (0.9/1.25334608174)        # Norm
                    if LiH == True : flux += LiHe_int(E) * (1.6/2.03905761931)
                    if Fas == True : flux += Neut_int(E) * (0.1/0.10137254836)
                    if alp == True : flux += Alph_int(E) * (0.05/0.0697352420074)
                    if Geo == True : flux += Geon_int(E) * (1.1/1.52561865621)
                        
                    flux = flux * livetime/2000. * e_step/0.031                       # livetime and binning
                    hist[0][j] = flux
                    
            # Put the "fluxes" into a Map object, give it the output_name
            m = Map(name=output_name, hist=hist, binning=self.output_binning)
        
            output_maps.append(m)

        # Combine the output maps into a single MapSet object to return.
        # The MapSet contains the varous things that are necessary to make
        # caching work and also provides a nice interface for the user to all
        # of the contained maps
        return MapSet(maps=output_maps, name='flux maps')

    #def validate_params(self, params):

