"""
This is a dummy oscillations service, provided as a template others can use to
build their own services.

This service makes use of transforms, but does *not* use nominal_transforms.

Note that this string, delineated by triple-quotes, is the "module-level
docstring," which you should write for your own services. Also, include all of
the docstrings (delineated by triple-quotes just beneath a class or method
definition) seen below, too! These all automatically get compiled into the PISA
documentation.
"""

from __future__ import absolute_import

import numpy as np

from pisa.core.binning import MultiDimBinning
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.hash import hash_obj

__all__ = ['dummy']

__author__ = 'J.L. Lanfranchi, P. Eller, J. Weldert'

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


class dummy(Stage): # pylint: disable=invalid-name
    """Example stage with maps as inputs and outputs, and no disk cache. E.g.,
    histogrammed oscillations stages will work like this.


    Parameters
    ----------
    params : ParamSet
        Parameters which set everything besides the binning.

    input_binning : MultiDimBinning
        The `inputs` must be a MapSet whose member maps (instances of Map)
        match the `input_binning` specified here.

    output_binning : MultiDimBinning
        The `outputs` produced by this service will be a MapSet whose member
        maps (instances of Map) will have binning `output_binning`.

    transforms_cache_depth : int >= 0
        Number of transforms (TransformSet) to store in the transforms cache.
        Setting this to 0 effectively disables transforms caching.

    outputs_cache_depth : int >= 0
        Number of outputs (MapSet) to store in the outputs cache. Setting this
        to 0 effectively disables outputs caching.


    Attributes
    ----------
    an_attr
    another_attr


    Methods
    -------
    foo
    bar
    bat
    baz


    Notes
    -----
    Blah blah blah ...

    """
    def __init__(self, params, input_binning, output_binning,
                 memcache_deepcopy, error_method, transforms_cache_depth,
                 outputs_cache_depth, debug_mode=None):
        # Here we provide the exhaustive list of params that can and must be
        # passed via the `params` argument.
        expected_params = (
            'earth_model',
            'YeI', 'YeM', 'YeO', 'deltacp', 'deltam21', 'deltam31',
            'detector_depth', 'prop_height', 'theta12', 'theta13',
            'theta23'
        )

        # Define the names of objects that are required by this stage (objects
        # will have the attribute "name": i.e., obj.name)
        input_names = (
            'nue', 'numu', 'nuebar', 'numubar'
        )

        # Define the names of objects that get produced by this stage
        output_names = (
            'nue', 'numu', 'nutau', 'nuebar', 'numubar', 'nutaubar'
        )

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
        super().__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            disk_cache=None,
            outputs_cache_depth=outputs_cache_depth,
            memcache_deepcopy=memcache_deepcopy,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning,
            debug_mode=debug_mode,
        )

        # There might be other things to do at init time than what Stage does,
        # but typically this is not much... and it's almost always a good idea
        # to have "real work" defined in another method besides init, which can
        # then get called from init (so that if anyone else wants to do the
        # same "real work" after object instantiation, (s)he can do so easily
        # by invoking that same method).

        # we could for example enable the stage to provide a weight for
        # a single event, which the user would tell it by deactivating
        # both the input and output binning
        self.calc_transforms = (input_binning is not None
                                and output_binning is not None)

        # we could want to keep track of binning related constants relevant
        # to this oscillation service's calculations
        if self.calc_transforms:
            self.compute_binning_constants()
        else:
            # this dummy service has no idea what it is supposed to do with
            # no binning provided, so abort
            raise ValueError("This service can only calculate binned"
                             " transforms! Please provide input and output"
                             " binning.")

    def compute_binning_constants(self):
        """Compute some constants related to the binning.
        Just for illustrating a few properties of the
        binning one might want to evaluate."""
        # Get the energy/coszen (ONLY) weighted centers here, since these
        # are actually used in the oscillations computation. All other
        # dimensions are ignored. Since these won't change so long as the
        # binning doesn't change, attache these to self.
        self.ecz_binning = MultiDimBinning([
            self.input_binning.true_energy.to('GeV'),
            self.input_binning.true_coszen.to('dimensionless')
        ])
        e_centers, cz_centers = self.ecz_binning.weighted_centers
        self.e_centers = e_centers.magnitude
        self.cz_centers = cz_centers.magnitude

        self.num_czbins = self.input_binning.true_coszen.num_bins
        self.num_ebins = self.input_binning.true_energy.num_bins

        self.e_dim_num = self.input_binning.names.index('true_energy')
        self.cz_dim_num = self.input_binning.names.index('true_coszen')

        # Illustrate how to find input binning dimensions which the transforms
        # created by this service will not depend on.
        self.extra_dim_nums = list(range(self.input_binning.num_dims))
        [self.extra_dim_nums.remove(d) for d in (self.e_dim_num,
                                                 self.cz_dim_num)]

    # In the following: methods called upon initialization of the `stage` parent
    # class that are commonly overriden (cf. `pisa/core/stage.py` for an
    # exhaustive list)
    def validate_binning(self):
        """This can be used to make sure the binning
        satisfies desired criteria."""
        # Our dummy service is set up such that it can only deal with 2D energy/
        # coszenith binning
        if set(self.input_binning.names) != set(['true_coszen', 'true_energy']):
            raise ValueError(
                "Input binning must be 2D true energy / coszenith binning. "
                "Got %s."%(self.input_binning.names)
            )

        # We do not handle rebinning (or oversampling), so we require the
        # output binning to correspond to the input binning
        assert self.input_binning == self.output_binning

    def validate_params(self, params):
        """This can be used to validate types, values, etc.
        of `params`."""
        # perform some action on `params`
        pass

    def create_dummy_osc_probs(self):
        """Here we generate the data structures that will be
        used as transforms representing oscillation
        probabilities."""
        # We have three neutrino flavors appearing through oscillations from
        # two in the initial flux, for both neutrinos and anti-neutrinos. These
        # are evaluated on a grid with dimension set by the input binning shape.
        xform_shape = [2] + list(self.input_binning.shape)
        # Here, the probabilities are random numbers between 0 and 1
        xform = np.random.rand(*xform_shape)
        return xform

    def _compute_transforms(self):
        """Compute new oscillation transforms."""
        # The seed is created from parameter values to produce different sets
        # of transforms for different sets of parameters
        seed = hash_obj(self.params.values, hash_to='int') % (2**32-1)
        np.random.seed(seed)

        # Read parameters in in the units used for computation, e.g.
        theta23 = self.params.theta23.m_as('rad')

        transforms = []
        for out_idx, output_name in enumerate(self.output_names):
            if out_idx < 3:
                # neutrinos (-> input names are neutrinos)
                input_names = self.input_names[0:2]
            else:
                # anti-neutrinos (-> input names are anti-neutrinos)
                input_names = self.input_names[2:4]

            # generate the "oscillation probabilities"
            xform = self.create_dummy_osc_probs()

            # create object of type `BinnedTensorTransform` and attach
            # to list of transforms with correct set of input names for the
            # output name in question
            transforms.append(
                BinnedTensorTransform(
                    input_names=input_names,
                    output_name=output_name,
                    # we have already made sure that input and output binnings
                    # are identical
                    input_binning=self.input_binning,
                    output_binning=self.output_binning,
                    xform_array=xform
                )
            )


        return TransformSet(transforms=transforms)
