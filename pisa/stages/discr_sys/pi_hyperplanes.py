"""
PISA pi stage to apply hyperplane fits from discrete systematics parameterizations
"""
from __future__ import absolute_import, print_function, division

import numpy as np
from numba import guvectorize, cuda

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.fileio import from_file
from pisa.utils.numba_tools import WHERE, myjit
from pisa.utils import vectorizer


class pi_hyperplanes(PiStage):
    """
    stage to histogram events

    Paramaters
    ----------

    fit_results_file : str
    dom_eff : dimensionless quantity
    hole_ice : dimensionless quantity
    hole_ice_fwd : dimensionless quantity
    spiciness : dimensionless quantity

    Notes
    -----

    the fit_results_file must contain the following keys:
        sys_list : containing the order of the parameters
        fit_results : the resulting hyperplane coeffs from the fits, first
                      entry is constant, followed by the linear ones in the order
                      defined in `sys_list`
    """
    def __init__(self,
                 data=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 error_method=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                 links=None,
                ):

        expected_params = ('fit_results_file',
                           'dom_eff', 
                           'hole_ice', 
                           'hole_ice_fwd', 
                           'spiciness', 
                          )
        input_names = ()
        output_names = ()

        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ('hyperplane_scalefactors')
        # what keys are added or altered for the outputs during apply
        if error_method in ['sumw2']:
            output_apply_keys = ('weights',
                                 'errors',
                                )
            input_apply_keys = output_apply_keys
        else:
            output_apply_keys = ('weights',
                                )
            input_apply_keys = output_apply_keys

        # init base class
        super(pi_hyperplanes, self).__init__(data=data,
                                             params=params,
                                             expected_params=expected_params,
                                             input_names=input_names,
                                             output_names=output_names,
                                             debug_mode=debug_mode,
                                             error_method=error_method,
                                             input_specs=input_specs,
                                             calc_specs=calc_specs,
                                             output_specs=output_specs,
                                             input_apply_keys=input_apply_keys,
                                             output_apply_keys=output_apply_keys,
                                            )

        assert self.input_mode is not None
        assert self.calc_mode == 'binned'
        assert self.output_mode is not None

        self.sys_list = None

        self.links = eval(links)
    
    def setup_function(self):
        """Load the fit results from the file and make some check
        compatibility"""
        fit_results = from_file(self.params['fit_results_file'].value)

        self.data.data_specs = self.calc_specs

        if self.links is not None:
            for key, val in self.links.items():
                self.data.link_containers(key, val)

        for container in self.data:
            container['hyperplane_results'] = fit_results[container.name].reshape(container.size, -1)
            #container.add_bined_data('hyperplane_results', (self.calc_specs, fit_results[container.name]), flat=False)
            container['hyperplane_scalefactors'] = np.empty(container.size, dtype=FTYPE)

            #print(container['hyperplane_results'].shape)
            #print(container['hyperplane_scalefactors'].shape)

        self.sys_list = fit_results['sys_list'] 
        # check compatibility
        # ToDo: check binning hash
        assert set(self.sys_list) == set([sys for sys in self.params.names if not sys.endswith('_file')])

        self.data.unlink_containers()

    @profile
    def compute_function(self):
        self.data.data_specs = self.calc_specs
        if self.links is not None:
            for key, val in self.links.items():
                self.data.link_containers(key, val)

        # get parameters:
        param_values = []
        for sys in self.sys_list:
            param_values.append(self.params[sys].magnitude)
        param_values = np.array(param_values, dtype=FTYPE)

        for container in self.data:
            eval_hyperplane(param_values,
                            container['hyperplane_results'].get(WHERE),
                            out=container['hyperplane_scalefactors'].get(WHERE)
                           )
            container['hyperplane_scalefactors'].mark_changed(WHERE)

        self.data.unlink_containers()


    @profile
    def apply_function(self):
        for container in self.data:
            vectorizer.multiply(container['hyperplane_scalefactors'], container['weights'])
            if self.error_method == 'sumw2':
                vectorizer.multiply(container['hyperplane_scalefactors'], container['errors'])


# vectorized function to apply
# must be outside class
if FTYPE == np.float64:
    signature = '(f8[:], f8[:], f8[:])'
else:
    signature = '(f4[:], f4[:], f4[:])'
@guvectorize([signature], '(a),(b)->()', target=TARGET)
def eval_hyperplane(param_values, hyperplane_results, out):
    result = hyperplane_results[0]
    for i in range(param_values.size):
        result += hyperplane_results[i+1] * param_values[i]
    out[0] = result
