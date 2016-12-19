# author: P.Eller
# date:   March 20, 2016


import numpy as np

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.config_parser import split
from pisa.utils.fileio import from_file
from pisa.utils.profiler import profile


# TODO: the below logic does not generalize to muons, but probably should
# (rather than requiring an almost-identical version just for muons). For
# example, an input arg can dictate neutrino or muon, which then sets the
# input_names and output_names.


class polyfits(Stage):
    """
    Stage to apply externally created fits of discrete systematics sets
    The inputs are parameter value plus corresponding slope file per
    systematic

    i.e. for example the params
        dom_eff : Quantity
        dom_eff_file : path of corresponding fit file

    The slope files can be created with pisa/utils/fit_discrete_sys_pid.py

    """
    def __init__(self, params, input_binning, output_binning, input_names,
                 disk_cache=None, error_method=None,
                 transforms_cache_depth=20, outputs_cache_depth=20):
        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'dom_eff', 'dom_eff_file',
            'hole_ice_fwd', 'hole_ice_fwd_file',
            'hole_ice', 'hole_ice_file',
            #'reco_cz_res', 'reco_cz_res_file',
        )

        input_names = split(input_names)
        output_names = input_names

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
        super(self.__class__, self).__init__(
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
            output_binning=output_binning
        )
        self.fit_results = None
        self.pnames = None

    def load_discr_sys(self, pnames):
        """Load the fit results from the file and make some check
        compatibility"""
        self.fit_results = {}
        for pname in pnames:
            self.fit_results[pname] = from_file(
                self.params[pname+'_file'].value
            )
            assert self.input_names == self.fit_results[pname]['map_names']
        self.pnames = pnames

    def _compute_nominal_transforms(self):
        # TODO: what is the mysterious logic here?
        pnames = [pname for pname in self.params.names if not
                  pname.endswith('_file')]
        if self.fit_results is None or pnames != self.pnames:
            self.load_discr_sys(pnames)

    @profile
    def _compute_transforms(self):
        """For the current parameter values, evaluate the fit function and
        write the resulting scaling into an x-form array"""
        # TODO: use iterators to collapse nested loops
        transforms = []
        for name in self.input_names:
            transform = None
            for pname in self.pnames:
                p_value = (self.params[pname].magnitude -
                           self.fit_results[pname]['nominal'])
                fit_fun = eval(self.fit_results[pname]['function'])
                fit_params = self.fit_results[pname][name]
                shape = fit_params.shape[:-1]
                if transform is None:
                    transform = np.ones(shape)
                for idx in np.ndindex(*shape):
                    # At every point evaluate the function
                    transform[idx] *= fit_fun(p_value, *fit_params[idx])

            xform = BinnedTensorTransform(
                input_names=(name),
                output_name=name,
                input_binning=self.input_binning,
                output_binning=self.output_binning,
                xform_array=transform,
                error_method=self.error_method,
            )
            transforms.append(xform)
        return TransformSet(transforms)
