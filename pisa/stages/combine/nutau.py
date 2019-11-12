import numpy as np

from pisa import ureg, Q_
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.log import logging
from pisa.utils.format import split
from pisa.utils.profiler import profile


__all__ = ['nutau']

__author__ = 'P. Eller'

__license__ = '''Copyright (c) 2014-2017, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


class nutau(Stage):
    """
    Stage combining the different maps (flav int) into right now a single map
    and apply a scale factor for nutau events

    combine_groups: dict with output map names and what maps should be contained
    for example
      {
        'evts':
          'nue_cc, nuebar_cc, numu_cc, numubar_cc, nutau_cc, nutaubar_cc, nue_nc, nuebar_nc, numu_nc, numubar_nc, nutau_nc, nutaubar_nc'
      }

    Parameters
    ----------
    params : ParamSet or sequence with which to instantiate a ParamSet.
        Expected params are:

            nu_nc_norm : quantity (dimensionless)
                global scaling factor that is applied to all *_nc maps
            nutau_norm : quantity (dimensionless)
            nutau_cc_norm : quantity (dimensionless)

    """
    def __init__(self, params, input_binning, input_names, combine_groups,
                 disk_cache=None, memcache_deepcopy=True, error_method=None,
                 outputs_cache_depth=20, debug_mode=None):
        expected_params = (
            'nu_nc_norm',
            #'nutau_norm',
            #'nutau_cc_norm'
        )

        #input_names = split(input_names, sep=',')
        self.combine_groups = eval(combine_groups)
        for key, val in self.combine_groups.items():
            self.combine_groups[key] = split(val, sep=',')
        output_names = self.combine_groups.keys()

        super().__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=input_binning,
            input_binning=input_binning,
            debug_mode=debug_mode
        )

    @profile
    def _compute_transforms(self):
        dims = self.input_binning.names

        transforms = []
        for group, in_names in self.combine_groups.items():
            xform_shape = [len(in_names)] + [self.input_binning[d].num_bins for d in dims]

            xform = np.ones(xform_shape)
            input_names = self.input_names
            for i,name in enumerate(in_names):
                scale = 1.
                if '_nc' in name:
                    scale *= self.params.nu_nc_norm.value.m_as('dimensionless')
                #if 'nutau' in name:
                #    scale *= self.params.nutau_norm.value.m_as('dimensionless')
                #if name in ['nutau_cc','nutaubar_cc']:
                #    scale *= self.params.nutau_cc_norm.value.m_as('dimensionless')
                if scale != 1:
                    xform[i] *= scale

            transforms.append(
                BinnedTensorTransform(
                    input_names=in_names,
                    output_name=group,
                    input_binning=self.input_binning,
                    output_binning=self.output_binning,
                    xform_array=xform
                )
            )

        return TransformSet(transforms=transforms)
