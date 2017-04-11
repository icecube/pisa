# PISA author: Timothy C. Arlen
#
# CAKE authors: Thomas Ehrhardt
#               tehrhardt@icecube.wisc.edu
#               Steven Wren
#               steven.wren@icecube.wisc.edu
# date:         Oct 19, 2016
"""
This is an effective area stage designed for quick studies of how effective
areas affect experimental observables and sensitivities. In addition, it is
supposed to be easily reproducible as it may rely on (phenomenological)
functions or interpolated discrete data points, dependent on energy
(and optionally cosine zenith), and which can thus be used as reference or
benchmark scenarios.
"""


from collections import Mapping, OrderedDict

import numpy as np
from scipy.interpolate import interp1d

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.stages.aeff.hist import compute_transforms, validate_binning
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup
from pisa.utils.fileio import from_file
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging


# TODO: the below logic does not generalize to muons, but probably should
# (rather than requiring an almost-identical version just for muons). For
# example, an input arg can dictate neutrino or muon, which then sets the
# input_names and output_names. (cf. aeff.hist)

class param(Stage):
    """Effective area service based on parameterisation functions stored in a
    .json file.
    Transforms an input map of a flux of a given flavour into maps of
    event rates for the two types of weak current (charged or neutral),
    according to energy and cosine zenith dependent effective areas specified
    by parameterisation functions.

    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameters:

        aeff_energy_paramfile
        aeff_coszen_paramfile
        livetime
        aeff_scale
        nutau_cc_norm

    particles : string
        Must be one of 'neutrinos' or 'muons' (though only neutrinos are
        supported at this time).

    input_names : None, string or sequence of strings
        If None, defaults are derived from `particles`.

    transform_groups : string
        Specifies which particles/interaction types to use for computing the
        transforms. (See Notes.)

    sum_grouped_flavints : bool
        Whether to sum the event-rate maps for the flavint groupings
        specified by `transform_groups`. If this is done, the output map names
        will be the group names (as well as the names of any flavor/interaction
        types not grouped together). Otherwise, the output map names will be
        the same as the input map names. Combining grouped flavints' is
        computationally faster and results in fewer maps, but it may be
        desirable to not do so for, e.g., debugging.

    input_binning : MultiDimBinning or convertible thereto
        Input binning is in true variables, with names prefixed by "true_".
        Each must match a corresponding dimension in `output_binning`.

    output_binning : MultiDimBinning or convertible thereto
        Output binning is in reconstructed variables, with names (traditionally
        in PISA but not necessarily) prefixed by "reco_". Each must match a
        corresponding dimension in `input_binning`.

    debug_mode : None, bool, or string
        Whether to store extra debug info for this service.

    transforms_cache_depth : int >= 0

    outputs_cache_depth : int >= 0

    memcache_deepcopy : bool

    """
    def __init__(self, params, particles, transform_groups,
                 sum_grouped_flavints, input_binning, output_binning,
                 memcache_deepcopy, transforms_cache_depth,
                 outputs_cache_depth, input_names=None, error_method=None,
                 debug_mode=None):
        # whether stage is instantiated to process neutrinos or muons
        assert particles in ['neutrinos', 'muons']
        self.particles = particles
        """Whether stage is instantiated to process neutrinos or muons"""

        self.transform_groups = flavintGroupsFromString(transform_groups)
        """Particle/interaction types to group for computing transforms"""

        self.sum_grouped_flavints = sum_grouped_flavints

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = [
            'aeff_energy_paramfile', 'aeff_coszen_paramfile',
            'livetime', 'aeff_scale'
        ]
        if particles == 'neutrinos':
            expected_params.append('nutau_cc_norm')

        if isinstance(input_names, basestring):
            input_names = input_names.replace(' ', '').split(',')
        elif input_names is None:
            if particles == 'neutrinos':
                input_names = ('nue', 'nuebar', 'numu', 'numubar', 'nutau',
                               'nutaubar')

        if self.particles == 'neutrinos':
            # TODO: if sum_grouped_flavints, then the output names should be e.g.
            #       'nue_cc_nuebar_cc' and 'nue_nc_nuebar_nc' if nue and nuebar
            #       are grouped... (?)
            if self.sum_grouped_flavints:
                output_names = [str(g) for g in self.transform_groups]
            else:
                input_flavints = NuFlavIntGroup(input_names)
                output_names = [str(fi) for fi in input_flavints]
        elif self.particles == 'muons':
            raise NotImplementedError
        else:
            raise ValueError('Particle type `%s` is not valid' % self.particles)

        logging.trace('transform_groups = %s', self.transform_groups)
        logging.trace('output_names = %s', ' :: '.join(output_names))

        super(self.__class__, self).__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('transform_groups')

        self.ecen = self.input_binning.true_energy.weighted_centers.magnitude
        if 'true_coszen' in self.input_binning.names:
            self.czcen = self.input_binning.true_coszen.weighted_centers.magnitude
        else:
            self.czcen = None

        self._param_hashes = dict(energy=None, coszen=None)
        self.parameterizations = dict(energy=None, coszen=None)

    def load_aeff_dim_param(self, dim, source=None, hash=None):
        """Load aeff parameterisation (energy- or coszen-dependent) from file
        or dictionary.

        Parameters
        ----------
        dim : string, one of 'energy' or 'coszen'
            Dimension for which the parameteriztion is to be loaded.

        source : string, mapping, or None
            Source of the parameterization. If string, treat as file path or
            resource location and load from the file. If mapping, use directly.

        hash : int or None
            Hash of `source`; if None, the hash is computed.

        """
        # Validation on args
        valid_dims = ('energy', 'coszen')
        if dim not in valid_dims:
            raise ValueError('``dim` must be one of %s. Got "%s" instead.'
                             % (valid_dims, dim))
        if not (source is None or isinstance(source, (basestring, Mapping))):
            raise TypeError('`source` must be string, mapping, or None')

        if hash is None:
            hash = hash_obj(source)

        if isinstance(source, basestring):
            orig_dict = from_file(source)
        elif isinstance(source, Mapping):
            orig_dict = source
        elif source is None:
            orig_dict = {str(group): None for group in self.transform_groups}

        # TODO: Perform validation on the object's contents

        #  Build dict with flavintgroups as keys
        flavintgroup_dict = OrderedDict()
        for key, val in orig_dict.iteritems():
            flavintgroup_dict[NuFlavIntGroup(key)] = val

        # Transform groups are implicitly defined by the contents of the
        # `pid_energy_paramfile`'s keys
        implicit_transform_groups = flavintgroup_dict.keys()

        # Make sure these match the transform groups specified for the stage
        if set(implicit_transform_groups) != set(self.transform_groups):
            raise ValueError(
                'Transform groups (%s) defined implicitly by'
                ' %s aeff parameterization "%s" do not match those defined'
                ' as the stage `transform_groups` (%s).'
                % (implicit_transform_groups, dim, source,
                   self.transform_groups)
            )

        ## Verify that each input name--which specifies a flavor--has at least
        ## one corresponding flavint specified by the transform
        #for name in self.input_names:
        #    if not any(name in group for group in implicit_transform_groups):
        #        raise ValueError(
        #            'Input "%s" either not present in or spans multiple'
        #            ' transform groups (transform_groups = %s)'
        #            % (name, implicit_transform_groups)
        #        )

        self.parameterizations[dim] = flavintgroup_dict
        self._param_hashes[dim] = hash

    def _compute_nominal_transforms(self):
        """Compute parameterised effective area transforms"""
        eparam_val = self.params.aeff_energy_paramfile.value
        czparam_val = self.params.aeff_coszen_paramfile.value

        if eparam_val is None:
            raise ValueError(
                'non-None energy parameterization params.aeff_energy_paramfile'
                ' must be provided'
            )
        if czparam_val is not None and self.czcen is None:
            raise ValueError(
                'true_coszen was not found in the binning but a coszen'
                ' parameterisation file has been provided by'
                ' `params.aeff_coszen_paramfile`.'
            )

        self.load_aeff_dim_param(dim='energy', source=eparam_val)
        self.load_aeff_dim_param(dim='coszen', source=czparam_val)

        nominal_transforms = []
        for xform_flavints in self.transform_groups:
            logging.debug('Working on %s effective areas xform', xform_flavints)

            energy_param = self.parameterizations['energy'][xform_flavints]
            coszen_param = None
            if self.parameterizations['coszen'] is not None:
                coszen_param = self.parameterizations['coszen'][xform_flavints]

            if isinstance(energy_param, basestring):
                energy_param = eval(energy_param)
            elif isinstance(energy_param, Mapping):
                if set(energy_param.keys()) != set(['aeff', 'energy']):
                    raise ValueError(
                        'Expected values of energy and aeff from which to'
                        ' construct a spline. Got %s.' % energy_param.keys()
                    )
                evals = energy_param['energy']
                avals = energy_param['aeff']

                # Construct the spline from the values.
                # The bounds_error=False means that the spline will not throw
                # an error when a value outside of the range is requested.
                # Instead, a fill_value of zero will be returned, as specified.
                # Currently done linear. Could potentially add this to the
                # config file.
                energy_param = interp1d(evals, avals, kind='linear',
                                        bounds_error=False, fill_value=0)
            else:
                raise TypeError('Expected energy_param to be either a string'
                                ' that can be interpreted by eval or as a'
                                ' mapping of values from which to construct a'
                                ' spline. Got "%s".' % type(energy_param))

            if coszen_param is not None:
                if isinstance(coszen_param, basestring):
                    coszen_param = eval(coszen_param)
                else:
                    raise TypeError('coszen dependence currently only'
                                    ' supported as a lambda function provided'
                                    ' as a string. Got "%s".'
                                    % type(coszen_param))

            # Now calculate the 1D aeff along energy
            aeff_vs_e = energy_param(self.ecen)

            # Correct for final energy bin, since interpolation does not
            # extend to JUST right outside the final bin

            # NOTE: Taken from the PISA 2 implementation of this. Almost
            # certainly comes from the fact that the highest knot there was
            # 79.5 GeV with the upper energy bin edge being 80 GeV. There's
            # probably something better that could be done here...
            if aeff_vs_e[-1] == 0:
                aeff_vs_e[-1] = aeff_vs_e[-2]

            if 'true_coszen' in self.input_binning:
                aeff_vs_e = self.input_binning.broadcast(
                    aeff_vs_e, from_dim='true_energy', to_dims='true_coszen'
                )

                # Now add cz-dependence, if required
                if coszen_param is not None:
                    aeff_vs_cz = coszen_param(self.czcen)
                    # Normalize
                    aeff_vs_cz *= len(aeff_vs_cz)/np.sum(aeff_vs_cz)
                else:
                    aeff_vs_cz = np.ones(shape=len(self.czcen))

                cz_broadcasted = self.input_binning.broadcast(
                    aeff_vs_cz, from_dim='true_coszen',
                    to_dims='true_energy'
                )
                xform_array = aeff_vs_e * cz_broadcasted
            else:
                xform_array = aeff_vs_e

            # If combining grouped flavints:
            # Create a single transform for each group and assign all flavors
            # that contribute to the group as the transform's inputs. Combining
            # the event rate maps will be performed by the
            # BinnedTensorTransform object upon invocation of the `apply`
            # method.
            if self.sum_grouped_flavints:
                xform_input_names = []
                for input_name in self.input_names:
                    input_flavs = NuFlavIntGroup(input_name)
                    if len(set(xform_flavints).intersection(input_flavs)) > 0:
                        xform_input_names.append(input_name)

                for output_name in self.output_names:
                    if output_name not in xform_flavints:
                        continue
                    xform = BinnedTensorTransform(
                        input_names=xform_input_names,
                        output_name=output_name,
                        input_binning=self.input_binning,
                        output_binning=self.input_binning,
                        xform_array=xform_array,
                        sum_inputs=self.sum_grouped_flavints
                    )
                    nominal_transforms.append(xform)

            # If *not* combining grouped flavints:
            # Copy the transform for each input flavor, regardless if the
            # transform is computed from a combination of flavors.
            else:
                for input_name in self.input_names:
                    input_flavs = NuFlavIntGroup(input_name)
                    # Since aeff "splits" neutrino flavors into
                    # flavor+interaction types, need to check if the output
                    # flavints are encapsulated by the input flavor(s).
                    if len(set(xform_flavints).intersection(input_flavs)) == 0:
                        continue
                    for output_name in self.output_names:
                        if output_name not in xform_flavints:
                            continue
                        xform = BinnedTensorTransform(
                            input_names=input_name,
                            output_name=output_name,
                            input_binning=self.input_binning,
                            output_binning=self.input_binning,
                            xform_array=xform_array,
                            sum_inputs=self.sum_grouped_flavints
                        )
                        nominal_transforms.append(xform)

        return TransformSet(transforms=nominal_transforms)

    def _compute_transforms(self):
        eparam_val = self.params.aeff_energy_paramfile.value
        czparam_val = self.params.aeff_coszen_paramfile.value

        eparam_hash = hash_obj(eparam_val)
        czparam_hash = hash_obj(czparam_val)

        recompute_nominal = False
        if eparam_hash != self._param_hashes['energy']:
            self.load_aeff_dim_param(dim='energy', source=eparam_val,
                                     hash=eparam_hash)

            recompute_nominal = True

        if czparam_hash != self._param_hashes['coszen']:
            self.load_aeff_dim_param(dim='coszen', source=czparam_val,
                                     hash=czparam_hash)
            recompute_nominal = True

        if recompute_nominal:
            self._compute_nominal_transforms()

        # Modify transforms according to other systematics by calling a
        # generic function from aeff.hist
        return compute_transforms(self)

    # Generic method from aeff.hist

    validate_binning = validate_binning
