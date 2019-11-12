# author: P.Eller
# date:   March 20, 2016


import numpy as np

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.config_parser import split
from pisa.utils.fileio import from_file
from pisa.utils.profiler import profile
from pisa.utils.log import logging


# TODO: the below logic does not generalize to muons, but probably should
# (rather than requiring an almost-identical version just for muons). For
# example, an input arg can dictate neutrino or muon, which then sets the
# input_names and output_names.

def fit_fun(X,*P):
    '''
    Hyperplane function

    Parameters
    ----------
    
    X : array of points
    P : array of params
        with first being the offset followe by the slopes
    '''
    ret_val = P[0]
    for x,p in zip(X,P[1:]):
        ret_val += x*p
    return ret_val


class hyperplane(Stage):
    """
    Stage to apply externally created fits of discrete systematics sets
    that are a n-dimensional hyperplane for n systematics

    The slope files can be created with pisa/utils/fit_discrete_sys_nd.py

    """
    def __init__(self, params, input_binning, output_binning, input_names,
                 disk_cache=None, error_method=None,
                 transforms_cache_depth=20, outputs_cache_depth=20):
        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'fit_results_file',
            'dom_eff', 
            'hole_ice', 
            'hole_ice_fwd', 
            'spiciness', 
        )

        output_names = input_names

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
        super().__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            disk_cache=disk_cache,
            outputs_cache_depth=outputs_cache_depth,
            error_method=error_method,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning,
        )
        self.fit_results = None
        self.sys_list = None

    def load_discr_sys(self, sys_list):
        """Load the fit results from the file and make some check
        compatibility"""
        self.fit_results = from_file(self.params['fit_results_file'].value)
        if not set(self.input_names) == set(self.fit_results['map_names']):
            for name in self.input_names:
                if not name in self.fit_results['map_names']:
                    #check if there is somethingi uniquely compatible
                    compatible_names = [mapname in name for mapname in self.fit_results['map_names']]
                    if sum(compatible_names) == 1:
                        # compatible
                        compatible_name = self.fit_results['map_names'][compatible_names.index(True)]
                        self.fit_results[name] = self.fit_results[compatible_name]
                        logging.warning('Substituting hyperplane parameterization %s for %s'%(compatible_name,name))
                    else:
                        logging.error('No compatible map for %s found!'%name)
        assert set(sys_list) == set(self.fit_results['sys_list'])
        self.sys_list = self.fit_results['sys_list'] 

    def _compute_nominal_transforms(self):
        # TODO: what is the mysterious logic here?
        sys_list = [sys for sys in self.params.names if not
                  sys.endswith('_file')]
        if self.fit_results is None or sys_list != self.sys_list:
            self.load_discr_sys(sys_list)

    @profile
    def _compute_transforms(self):
        """For the current parameter values, evaluate the fit function and
        write the resulting scaling into an x-form array"""
        # TODO: use iterators to collapse nested loops
        transforms = []
        for input_name in self.input_names:
            transform = None
            sys_values = []
            for sys in self.sys_list:
                sys_values.append(self.params[sys].magnitude)
            fit_params = self.fit_results[input_name]
            shape = fit_params.shape[:-1]
            if transform is None:
                transform = np.ones(shape)
            for idx in np.ndindex(*shape):
                # At every point evaluate the function
                transform[idx] *= fit_fun(sys_values, *fit_params[idx])

            xform = BinnedTensorTransform(
                input_names=(input_name),
                output_name=input_name,
                input_binning=self.input_binning,
                output_binning=self.output_binning,
                xform_array=transform,
                error_method=self.error_method,
            )
            transforms.append(xform)
        return TransformSet(transforms)
