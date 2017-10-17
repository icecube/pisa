# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016
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

import numpy as np

from pisa.core.binning import MultiDimBinning
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity


class dummy(Stage):
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
        # All of the following params (and no more) must be passed via the
        # `params` argument.
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
        super(self.__class__, self).__init__(
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
            debug_mode=debug_mode
        )

        # There might be other things to do at init time than what Stage does,
        # but typically this is not much... and it's almost always a good idea
        # to have "real work" defined in another method besides init, which can
        # then get called from init (so that if anyone else wants to do the
        # same "real work" after object instantiation, (s)he can do so easily
        # by invoking that same method).
        self.calc_transforms = (input_binning is not None
                                and output_binning is not None)
        if self.calc_transforms:
            self.compute_binning_constants()

    def compute_binning_constants(self):
        # Only works if energy and coszen are in input_binning
        if 'true_energy' not in self.input_binning \
                or 'true_coszen' not in self.input_binning:
            raise ValueError(
                'Input binning must contain both "true_energy" and'
                ' "true_coszen" dimensions.'
            )

        # Not handling rebinning (or oversampling)
        assert self.input_binning == self.output_binning

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

        self.extra_dim_nums = range(self.input_binning.num_dims)
        [self.extra_dim_nums.remove(d) for d in (self.e_dim_num,
                                                 self.cz_dim_num)]

    def _compute_transforms(self):
        """Compute new oscillation transforms."""
        # This is done just to produce different set of transforms for
        # different set of parameters
        seed = hash_obj(self.params.values, hash_to='int') % (2**32-1)
        np.random.seed(seed)

        # Read parameters in in the units used for computation, e.g.
        theta12 = self.params.theta12.m_as('rad')

        total_bins = int(len(self.e_centers)*len(self.cz_centers))
        # We use 18 since we have 3*3 possible oscillations for each of
        # neutrinos and antineutrinos.
        prob_list = np.random.random(total_bins*18)

       # Slice up the transform arrays into views to populate each transform
        dims = ['true_energy', 'true_coszen']
        xform_dim_indices = [0, 1]
        users_dim_indices = [self.input_binning.index(d) for d in dims]
        xform_shape = [2] + [self.input_binning[d].num_bins for d in dims]

        # TODO: populate explicitly by flavor, don't assume any particular
        # ordering of the outputs names!
        transforms = []
        for out_idx, output_name in enumerate(self.output_names):
            xform = np.empty(xform_shape)
            if out_idx < 3:
                # Neutrinos
                xform[0] = np.array([
                    prob_list[out_idx + 18*i*self.num_czbins
                              : out_idx + 18*(i+1)*self.num_czbins
                              : 18]
                    for i in range(0, self.num_ebins)
                ])
                xform[1] = np.array([
                    prob_list[out_idx+3 + 18*i*self.num_czbins
                              : out_idx+3 + 18*(i+1)*self.num_czbins
                              : 18]
                    for i in range(0, self.num_ebins)
                ])
                input_names = self.input_names[0:2]

            else:
                # Antineutrinos
                xform[0] = np.array([
                    prob_list[out_idx+6 + 18*i*self.num_czbins
                              : out_idx+6 + 18*(i+1)*self.num_czbins
                              : 18]
                    for i in range(0, self.num_ebins)
                ])
                xform[1] = np.array([
                    prob_list[out_idx+9 + 18*i*self.num_czbins
                              : out_idx+9 + 18*(i+1)*self.num_czbins
                              : 18]
                    for i in range(0, self.num_ebins)
                ])
                input_names = self.input_names[2:4]

            xform = np.moveaxis(
                xform,
                source=[0] + [i+1 for i in xform_dim_indices],
                destination=[0] + [i+1 for i in users_dim_indices]
            )
        
            transforms.append(
                BinnedTensorTransform(
                    input_names=input_names,
                    output_name=output_name,
                    input_binning=self.input_binning,
                    output_binning=self.output_binning,
                    xform_array=xform
                )
            )


        return TransformSet(transforms=transforms)
