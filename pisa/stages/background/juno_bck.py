"""
Define service for JUNO background
"""

from __future__ import absolute_import
from scipy.interpolate import interp1d

import numpy as np

from pisa import ureg
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.core.map import Map, MapSet
from pisa.core.container import Container
from pisa.utils.hash import hash_obj
from pisa.utils.resources import find_resource
from pisa.utils.profiler import profile


Accidental = np.loadtxt(find_resource('flux/JUNO_bck/Accidental.csv'),delimiter=',')
Fast_neutron = np.loadtxt(find_resource('flux/JUNO_bck/Fast_neutron.csv'),delimiter=',')
Li_He = np.loadtxt(find_resource('flux/JUNO_bck/Li_He.csv'),delimiter=',')
alpha_n = np.loadtxt(find_resource('flux/JUNO_bck/alpha_n.csv'),delimiter=',')
Geo_neutrino = np.loadtxt(find_resource('flux/JUNO_bck/Geo_neutrino.csv'),delimiter=',')


def get_xy(lis):
    x , y = [] , []
    
    for i in range(len(lis)):
        x.append(lis[i][0])
        y.append(lis[i][1])
    
    return x , y
        

class juno_bck(Stage): # pylint: disable=invalid-name

    def __init__(
        self,
        **std_kwargs,
    ):
        expected_params = (
            'Accidental',
            'alpha_n',
            'Fast_neutrons',
            'Li_He',
            'Geo_neutrinos',
            'livetime'
        )

        super().__init__(
            expected_params=expected_params,
            expected_container_keys=(),
            **std_kwargs,
        )
    
    def setup_function(self):
        container = Container('background')
        self.data.add_container(container)
        
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

        Es, czs, flux = [], [], []

        E_index = self.calc_mode.names.index("true_energy")
        cz_index = self.calc_mode.names.index("true_coszen")
        
        num_e_bins = self.calc_mode.shape[E_index]
        e_min = (min(self.calc_mode.bin_edges[E_index])).magnitude * 1000 # GeV to MeV
        e_range = (max(self.calc_mode.bin_edges[E_index]) - min(self.calc_mode.bin_edges[E_index])).magnitude * 1000 # GeV to MeV
        e_step = float(e_range)/num_e_bins

        for j in range(num_e_bins):
            E = e_min + e_step/2. + j * e_step # E at bin center

            fl = 0
            if Acc == True : fl += Acci_int(E) * (0.9/1.25334608174)        # Norm
            if LiH == True : fl += LiHe_int(E) * (1.6/2.03905761931)
            if Fas == True : fl += Neut_int(E) * (0.1/0.10137254836)
            if alp == True : fl += Alph_int(E) * (0.05/0.0697352420074)
            if Geo == True : fl += Geon_int(E) * (1.1/1.52561865621)

            fl *= e_step/0.031                       # binning
            flux.append(fl)
            Es.append(E)
            czs.append(0)
            
        self.data['background']['flux'] = np.array(flux)
        self.data['background']['true_energy'] = np.array(Es) * 1e-3 #MeV to GeV
        self.data['background']['true_coszen'] = np.array(czs)
    
    @profile
    def apply_function(self):
        self.data.representation = "events"
        
        livetime = self.params.livetime.m_as('day')
        self.data['background']['weights'] = self.data['background']['flux'] * livetime/2000.


def init_test(**param_kwargs):
    """Instantiation example"""
    param_set = [
        Param(name="Accidental", value=True,  **param_kwargs),
        Param(name="alpha_n", value=True,  **param_kwargs),
        Param(name="Fast_neutrons", value=True,  **param_kwargs),
        Param(name="Li_He", value=True,  **param_kwargs),
        Param(name="Geo_neutrinos", value=True,  **param_kwargs),
        Param(name="livetime", value=10*ureg.s, **param_kwargs),
    ]
    param_set = ParamSet(*param_set)

    e_binning = OneDimBinning(name='true_energy', is_log=True, num_bins=10, domain=[0.002, 0.007]*ureg.GeV)
    cz_binning = OneDimBinning(name='true_coszen', is_lin=True, num_bins=1, domain=[-1, 1])
    binning = MultiDimBinning([cz_binning, e_binning])

    return juno_bck(params=param_set, calc_mode=binning)
