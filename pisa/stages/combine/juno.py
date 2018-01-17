
import numpy as np

from pisa import ureg, Q_
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.log import logging
from pisa.utils.format import split
from pisa.utils.profiler import profile


__all__ = ['juno']


class juno(Stage):
    """

    """
    def __init__(self, params, input_binning, output_binning, input_names, output_names,
                 disk_cache=None, memcache_deepcopy=True, error_method=None,
                 outputs_cache_depth=20, debug_mode=None):
        expected_params = (
            'test',
        )

        input_names = split(input_names, sep=',')
        #output_name = output_name

        super(self.__class__, self).__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning,
            input_binning=input_binning,
            debug_mode=debug_mode
        )

    @profile
    def _compute_transforms(self):
        inshape = self.input_binning.shape
        outshape = self.output_binning.shape
        
        xforms = []
        xform_shape = (inshape[0] , inshape[1] , outshape[0] , outshape[1])
        xform = np.zeros(xform_shape)
        
        for i in range(inshape[0]):
            for j in range(inshape[1]):
                xform[i][j][0][j] = 1             # inshape[1] has to be equal to outshape[1]
                
        
        for input_name in self.input_names:
            for output_name in self.output_names:

                trafo = BinnedTensorTransform(
                    input_names=input_name,
                    output_name=output_name,
                    input_binning=self.input_binning,
                    output_binning=self.output_binning,
                    xform_array=xform,
                    sum_inputs=True
                        )
                xforms.append(trafo)

        return TransformSet(transforms=xforms)
