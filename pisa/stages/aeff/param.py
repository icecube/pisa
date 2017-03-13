# PISA author: Timothy C. Arlen
#
# CAKE author: Thomas Ehrhardt
#              tehrhardt@icecube.wisc.edu
# date:        Oct 19, 2016
"""
This is an effective area service designed for quick studies of how effective
areas affect experimental observables and sensitivities. In addition, it is
supposed to be easily reproducible as it solely relies on (phenomenological)
functions, dependent on energy (and optionally cosine zenith), and which can
thus be used as reference or benchmark scenarios.
"""
import copy
from itertools import product

import numpy as np
from scipy.interpolate import interp1d

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.core.events import Events
from pisa.utils.flavInt import ALL_NUFLAVINTS, flavintGroupsFromString, \
        IntType, NuFlavIntGroup
from pisa.utils.fileio import from_file
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import profile


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
        transform_events_keep_criteria

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
            'livetime', 'aeff_scale', 'transform_events_keep_criteria'
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
            if self.sum_grouped_flavints:
                output_names = [str(g) for g in self.transform_groups]
            else:
                input_flavints = NuFlavIntGroup(input_names)
                output_names = [str(fi) for fi in input_flavints]
        elif self.particles == 'muons':
            raise NotImplementedError
        else:
            raise ValueError('Particle type `%s` is not valid' % self.particles)

        logging.trace('transform_groups = %s' %self.transform_groups)
        logging.trace('output_names = %s' %' :: '.join(output_names))

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

        # Can do these now that binning has been set up in call to Stage's init
        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('transform_groups')

    def load_aeff_energy_param(self, aeff_energy_param):
        """
        Load aeff energy-dependent parameterisation from file or dictionary.
        """
        this_hash = hash_obj(aeff_energy_param)
        if (hasattr(self, '_energy_param_hash') and
            this_hash == self._energy_param_hash):
            return
        if isinstance(aeff_energy_param, basestring):
            energy_param_dict = from_file(aeff_energy_param)
        elif isinstance(aeff_energy_param, dict):
            energy_param_dict = aeff_energy_param
        else:
            raise ValueError(
                "Expecting either a path to a file or a dictionary provided as"
                " the store of the parameterisations. Got %s. Something is "
                "wrong."%(type(reco_param))
            )
        self.energy_param_dict = energy_param_dict
        self._energy_param_hash = this_hash

    def load_aeff_coszen_param(self, aeff_coszen_param):
        """
        Load aeff coszen-dependent parameterisation from file or dictionary.
        """
        if aeff_coszen_param is not None:
            this_hash = hash_obj(aeff_coszen_param)
            if (hasattr(self, '_coszen_param_hash') and
                this_hash == self._coszen_param_hash):
                return
            if isinstance(aeff_coszen_param, basestring):
                coszen_param_dict = from_file(aeff_coszen_param)
            elif isinstance(aeff_coszen_param, dict):
                coszen_param_dict = aeff_coszen_param
            else:
                raise ValueError("Got type %s for aeff_coszen_param when "
                                 "either basestring or dict was expected. "
                                 "Something is wrong."%type(aeff_coszen_param))
            self.coszen_param_dict = coszen_param_dict
            self._coszen_param_hash = this_hash
        else:
            self.coszen_param_dict = None
            self._coszen_param_hash = None

    def find_energy_param(self, flavstr):
        if flavstr not in self.energy_param_dict.keys():
            if 'nc' in flavstr:
                if 'bar' in flavstr:
                    if 'nuallbar_nc' in self.energy_param_dict.keys():
                        logging.warn("Could not find the %s transform group "
                                     "but did find a nuallbar version. Will "
                                     "proceed assuming this is to be used for"
                                     " all nubar_nc transforms."%flavstr)
                        energy_param = self.energy_param_dict['nuallbar_nc']
                    elif 'nuall_nc' in self.energy_param_dict.keys():
                        logging.warn("Could not find the %s transform group "
                                     "but did find a nuallbar version. Will "
                                     "proceed assuming this is to be used for"
                                     " all nubar_nc transforms and that "
                                     "therefore nu and nubar transform the "
                                     "same."%flavstr)
                        energy_param = self.energy_param_dict['nuall_nc']
                        raise ValueError(
                            "Transform group %s not found in energy aeff "
                            "parameterisations dictionary keys - %s and "
                            "neither an equivalent nuallbar_nc or nuall_nc "
                            "entry. Something is wrong."%(
                                flavstr,
                                self.energy_param_dict.keys()
                            )
                        )
                else:
                    if 'nuall_nc' in self.energy_param_dict.keys():
                        logging.warn("Could not find the %s transform "
                                     "group but did find a nuall version."
                                     " Will proceed assuming this is to "
                                     "be used for all nu_nc transforms."%(
                                         flavstr))
                        energy_param = self.energy_param_dict['nuall_nc']
                    else:
                        raise ValueError(
                            "Transform group %s not found in energy aeff "
                            "parameterisations dictionary keys - %s and "
                            "neither was an equivalent nuall_nc entry. "
                            "Something is wrong."%(
                                flavstr,
                                self.energy_param_dict.keys()
                            )
                        )
            elif 'bar' in flavstr:
                new_flavstr = flavstr.replace('bar','')
                if new_flavstr not in self.energy_param_dict.keys():
                    raise ValueError(
                        "Transform group %s not found in energy aeff "
                        "parameterisation dictionary keys - %s and neither "
                        "was an equivalent 'unbarred' one. Something is "
                        "wrong."%(
                            flavstr,
                            self.energy_param_dict.keys()
                        )
                    )
                else:
                    logging.warn("Could not find the %s transform group but "
                                 "did find an 'unbarred' version. Will "
                                 "proceed assuming that nu and nubar "
                                 "transform the same."%(flavstr))
            else:
                raise ValueError(
                    "Transform group %s not found in energy aeff "
                    "parameterisation dictionary keys - %s. Something is "
                    "wrong."%(
                        flavstr,
                        self.energy_param_dict.keys()
                    )
                )
        else:
            energy_param = self.energy_param_dict[flavstr]
            
        return energy_param

    def find_coszen_param(self, flavstr):
        if flavstr not in self.coszen_param_dict.keys():
            if 'nc' in flavstr:
                if 'bar' in flavstr:
                    if 'nuallbar_nc' in self.coszen_param_dict.keys():
                        logging.warn("Could not find the %s transform group "
                                     "but did find a nuallbar version. Will "
                                     "proceed assuming this is to be used "
                                     "for all nubar_nc transforms."%(
                                         flavstr))
                        coszen_param = self.coszen_param_dict['nuallbar_nc']
                    elif 'nuall_nc' in self.coszen_param_dict.keys():
                        logging.warn("Could not find the %s transform group "
                                     "but did find a nuall version. Will "
                                     "proceed assuming this is to be used "
                                     "for all nu_nc transforms and that nu "
                                     "and nubar transform the same."%(
                                         flavstr))
                        coszen_param = self.coszen_param_dict['nuall_nc']
                        
                    else:
                        raise ValueError(
                            "Transform group %s not found in coszen aeff "
                            "parameterisations dictionary keys - %s and "
                            "neither was a nuallbar_nc or nuall_nc entry. "
                            "Something is wrong."%(
                                flavstr,
                                self.coszen_param_dict.keys()
                            )
                        )
                else:
                    if 'nuall_nc' in self.coszen_param_dict.keys():
                        logging.warn("Could not find the %s transform group "
                                     "but did find a nuall version. Will "
                                     "proceed assuming this is to be used "
                                     "for all nu_nc transforms."%(
                                         flavstr))
                        coszen_param = self.coszen_param_dict['nuall_nc']
                    else:
                        raise ValueError(
                            "Transform group %s not found in coszen aeff "
                            "parameterisations dictionary keys - %s and "
                            "neither was an equivalent nuall_nc entry. "
                            "Something is wrong."%(
                                flavstr,
                                self.coszen_param_dict.keys()
                            )
                        )
            elif 'bar' in flavstr:
                new_flavints = flavstr.replace('bar','')
                if new_flavints not in self.coszen_param_dict.keys():
                    raise ValueError(
                        "Transform group %s (or 'unbarred' equivalent) not "
                        "found in coszen aeff parameterisation dictionary "
                        "keys - %s. Something is wrong."%(
                            flavstr,
                            self.coszen_param_dict.keys()
                        )
                    )
                else:
                    logging.warn("Could not find the %s transform group but "
                                 "did find the 'unbarred' version  %s instead."
                                 " Will proceed assuming this flavour behaves"
                                 " identical for nu and nubar."%(
                                     flavstr,
                                     new_flavints
                                 ))
                    coszen_param = self.coszen_param_dict[new_flavints]
            else:
                raise ValueError(
                    "Transform group %s not found in coszen aeff "
                    "parameterisation dictionary keys - %s. Something is "
                    "wrong."%(
                        flavstr,
                        self.coszen_param_dict.keys()
                    )
                )
        else:
            coszen_param = self.coszen_param_dict[flavstr]

        return coszen_param

    def _compute_nominal_transforms(self):
        """Compute parameterised effective area transforms"""
        # Require at least true energy in input_binning.
        if 'true_energy' not in self.input_binning:
            raise ValueError('Input binning must contain "true_energy"'
                             ' dimension, but does not.')

        # TODO: not handling rebinning in this stage or within Transform
        # objects; implement this! (and then this assert statement can go away)
        assert self.input_binning == self.output_binning

        # Right now this can only deal with 1D energy or 2D energy / coszenith
        # binning, so if azimuth is present then this will raise an exception.
        if 'reco_azimuth' in set(self.input_binning.names):
            raise ValueError(
                "Input binning cannot have azimuth present for this "
                "parameterised aeff service."
            )

        self.load_aeff_energy_param(self.params.aeff_energy_paramfile.value)
        self.load_aeff_coszen_param(self.params.aeff_coszen_paramfile.value)

        ecen = self.input_binning.true_energy.weighted_centers.magnitude
        if 'true_coszen' in self.input_binning.names:
            czcen = self.input_binning.true_coszen.weighted_centers.magnitude
        else:
            if self.params.aeff_coszen_paramfile.value is not None:
                raise ValueError("coszenith was not found in the binning but a"
                                 " coszenith parameterisation file has been "
                                 "provided in the configuration file. "
                                 "Something is wrong.")
            czcen = None

        nominal_transforms = []
        for xform_flavints in self.transform_groups:
            logging.debug("Working on %s effective areas xform" %xform_flavints)
            energy_param = self.find_energy_param(str(xform_flavints))
            if self.params.aeff_coszen_paramfile.value is not None:
                coszen_param = self.find_coszen_param(str(xform_flavints))
            else:
                coszen_param = None
            
            if isinstance(energy_param, basestring):
                energy_param = eval(energy_param)
            elif isinstance(energy_param, dict):
                if sorted(energy_param.keys()) != ['aeff','energy']:
                    raise ValueError("Expected values of energy and aeff from"
                                     " which to construct a spline. Got %s."%(
                                         energy_param.keys()))
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
                raise ValueError("Expected energy_param to be either a string"
                                 " that can be interpreted by eval or as a "
                                 "dict of values from which to construct a "
                                 "spline. Got %s. Something is wrong."%(
                                     type(energy_param)))
            if coszen_param is not None:
                if isinstance(coszen_param, basestring):
                    coszen_param = eval(coszen_param)
                else:
                    raise ValueError("coszen dependence currently only "
                                     "supported as a lambda function provided"
                                     " as a string in a json file. Got %s. "
                                     "Something is wrong."%(
                                         type(coszen_param)))
                
            # Now calculate the 1D aeff along energy
            aeff1d = energy_param(ecen)
            # Correct for final energy bin, since interpolation does not
            # extend to JUST right outside the final bin
            # Taken from the PISA 2 implementation of this. Almost certainly
            # comes from the fact that the highest knot there was 79.5 GeV with
            # the upper energy bin edge being 80.0 GeV. There's probably
            # something better that could be done here...
            if aeff1d[-1] == 0.0:
                aeff1d[-1] = aeff1d[-2]
            # Make this in to the right dimensionality.
            if 'true_coszen' in set(self.input_binning.names):
                aeff2d = np.repeat(aeff1d, len(czcen))
                aeff2d = np.reshape(aeff2d, (len(ecen), len(czcen)))
                # Now add cz-dependence, if required
                if coszen_param is not None:
                    cz_dep = coszen_param(czcen)
                    # Normalise
                    cz_dep *= len(cz_dep)/np.sum(cz_dep)
                    aeff2d *= cz_dep
                if self.input_binning.names[0] == 'true_energy':
                    aeff2d = aeff2d
                elif self.input_binning.names[0] == 'true_coszen':
                    aeff2d = aeff2d.T
                else:
                    raise ValueError(
                        "Got a name for the first bins that was unexpected - "
                        "%s. Something is wrong."%(
                            self.input_binning.names[0]
                        )
                    )
                xform_array = aeff2d
            else:
                xform_array = aeff1d

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
                        output_binning=self.output_binning,
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
                            output_binning=self.output_binning,
                            xform_array=xform_array,
                            sum_inputs=self.sum_grouped_flavints
                        )
                        nominal_transforms.append(xform)

        return TransformSet(transforms=nominal_transforms)

    def _compute_transforms(self):
        """
        Compute new parameterised effective area transforms. 
        Copied from the hist service since it's the same.
        """
        # Read parameters in in the units used for computation
        aeff_scale = self.params.aeff_scale.m_as('dimensionless')
        livetime_s = self.params.livetime.m_as('sec')
        logging.trace('livetime = %s --> %s sec'
                      %(self.params.livetime.value, livetime_s))

        if self.particles == 'neutrinos':
            nutau_cc_norm = self.params.nutau_cc_norm.m_as('dimensionless')
            if nutau_cc_norm != 1:
                assert NuFlavIntGroup('nutau_cc') in self.transform_groups
                assert NuFlavIntGroup('nutaubar_cc') in self.transform_groups

        new_transforms = []
        for xform_flavints in self.transform_groups:
            flav_names = [str(flav) for flav in xform_flavints.flavs()]
            aeff_transform = None
            for transform in self.nominal_transforms:
                if (transform.input_names[0] in flav_names
                        and transform.output_name in xform_flavints):
                    if aeff_transform is None:
                        scale = aeff_scale * livetime_s
                        if (self.particles == 'neutrinos' and
                                ('nutau_cc' in transform.output_name
                                 or 'nutaubar_cc' in transform.output_name)):
                            scale *= nutau_cc_norm
                        aeff_transform = transform.xform_array * scale
                    new_xform = BinnedTensorTransform(
                        input_names=transform.input_names,
                        output_name=transform.output_name,
                        input_binning=transform.input_binning,
                        output_binning=transform.output_binning,
                        xform_array=aeff_transform,
                        sum_inputs=self.sum_grouped_flavints
                    )
                    new_transforms.append(new_xform)

        return TransformSet(new_transforms)
