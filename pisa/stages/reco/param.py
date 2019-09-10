"""
Create the transforms that map from true energy and coszen
to the reconstructed parameters. Provides reco event rate maps using these
transforms.
"""


from __future__ import division

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import itertools

import numpy as np
from scipy.stats import norm
from scipy import stats

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.core.binning import basename
from pisa.utils.fileio import from_file
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.comparisons import recursiveEquality, EQUALITY_PREC, isscalar


__all__ = ['load_reco_param', 'param']

__author__ = 'L. Schulte, S. Wren, T. Ehrhardt'

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


def load_reco_param(source):
    """Load reco parameterisation (energy-dependent) from file or dictionary.

    Parameters
    ----------
    source : string or mapping
        Source of the parameterization. If string, treat as file path or
        resource location and load from the file; this must yield a mapping. If
        `source` is a mapping, it is used directly. See notes below on format.

    Returns
    -------
    reco_params : OrderedDict
        Keys are stringified flavintgroups and values are dicts of strings
        representing the different reco dimensions and lists of distribution
        properties. These latter have a 'fraction', a 'dist' and a 'kwargs' key.
        The former two hold callables, while the latter holds a dict of
        key-callable pairs ('loc', 'scale'), which can be evaluated at the desired
        energies and passed into the respective `scipy.stats` distribution.
        The distributions for a given dimension will be superimposed according
        to their relative weights to form the reco kernels (via integration)
        when called with energy values (parameterisations are functions of
        energy only!).

    Notes
    -----
    The mapping passed via `source` or loaded therefrom must have the format:
        {
            <flavintgroup_string>:
                {
                    <dimension_string>:[
                        {
                            "dist": dist_id,
                            "fraction": val,
                            "kwargs": {
                                "loc": val,
                                "scale": val,
                                ...
                            }
                        },
                    ...
                    ]
                },
            <flavintgroup_string>:
                ...
        }

    `flavintgroup_string`s must be parsable by
    pisa.utils.flavInt.NuFlavIntGroup. Note that the `transform_groups` defined
    in a pipeline config file using this must match the groupings defined
    above.

    `dimension_string`s denote the observables/dimensions whose reco error
    distribution is parameterised (`"energy"` or `"coszen"`).

    `dist_id` needs to be a string identifying a probability distribution/statistical
    function provided by `scipy.stats`. No implicit assumptions about the
    distribution will be made if the `"dist"` key is missing.

    `"fraction"` holds the relative weight of the distribution. For a given
    dimension, the sum of all fractions present must be 1.

    Valid kwargs for distributions must at least include `"loc"` and `"scale"` -
    these will be passed into the respective `scipy.stats` function.

    `val`s can be one of the following:
        - Callable with one argument
        - String such that `eval(val)` yields a callable with one argument
    """
    if not (source is None or isinstance(source, (str, Mapping))):
        raise TypeError('`source` must be string, mapping, or None')

    if isinstance(source, str):
        orig_dict = from_file(source)

    elif isinstance(source, Mapping):
        orig_dict = source

    else:
        raise TypeError('Cannot load reco parameterizations from a %s'
                        % type(source))

    valid_dimensions = ('coszen', 'energy')
    required_keys = ('dist', 'fraction', 'kwargs')

    # Build dict of parameterizations (each a callable) per flavintgroup
    reco_params = OrderedDict()
    for flavint_key, dim_dict in orig_dict.items():
        flavintgroup = NuFlavIntGroup(flavint_key)
        reco_params[flavintgroup] = {}
        for dimension in dim_dict.keys():
            dim_dist_list = []

            if not isinstance(dimension, str):
                raise TypeError("The dimension needs to be given as a string!"
                                " Allowed: %s."%valid_dimensions)

            if dimension not in valid_dimensions:
                raise ValueError("Dimension '%s' not recognised!"%dimension)

            for dist_dict in dim_dict[dimension]:
                dist_spec_dict = {}

                # allow reading in even if kwargs not present - computation of
                # transform will fail because "loc" and "scale" hard-coded
                # requirement
                for required in required_keys:
                    if required not in dist_dict:
                        raise ValueError("Found distribution property dict "
                                         "without required '%s' key for "
                                         "%s - %s!"
                                         %(required, flavintgroup, dimension))

                for k in dist_dict.keys():
                    if k not in required_keys:
                        logging.warning(
                            "Unrecognised key in distribution property dict: '%s'"%k
                        )

                dist_spec = dist_dict['dist']

                if not isinstance(dist_spec, str):
                    raise TypeError(" The resolution function needs to be"
                                    " given as a string!")

                if not dist_spec:
                    raise ValueError("Empty string found for resolution"
                                     " function!")

                try:
                    dist = getattr(stats, dist_spec.lower())
                except AttributeError:
                    try:
                        import scipy
                        sp_ver_str = scipy.__version__
                    except:
                        sp_ver_str = "N/A"
                    raise AttributeError("'%s' is not a valid distribution"
                                         " from scipy.stats (your scipy"
                                         " version: '%s')."
                                         %(dist_spec.lower(), sp_ver_str))
                logging.debug("Found %s - %s resolution function: '%s'"
                              %(flavintgroup, dimension, dist.name))

                dist_spec_dict['dist'] = dist

                frac = dist_dict['fraction']

                if isinstance(frac, str):
                    frac_func = eval(frac)

                elif callable(frac):
                    frac_func = frac

                else:
                    raise TypeError(
                        "Expected 'fraction' to be either a string"
                        " that can be interpreted by eval or a callable."
                        " Got '%s'." % type(frac)
                    )

                dist_spec_dict['fraction'] = frac_func

                kwargs = dist_dict['kwargs']

                if not isinstance(kwargs, dict):
                    raise TypeError(
                        "'kwargs' must hold a dictionary. Got '%s' instead."
                        % type(kwargs)
                    )

                dist_spec_dict['kwargs'] = kwargs
                for kwarg, kwarg_spec in kwargs.items():

                    if isinstance(kwarg_spec, str):
                        kwarg_eval = eval(kwarg_spec)

                    elif callable(kwarg_spec) or isscalar(kwarg_spec):
                        kwarg_eval = kwarg_spec

                    else:
                        raise TypeError(
                            "Expected kwarg '%s' spec to be either a string"
                            " that can be interpreted by eval, a callable or"
                            " a scalar. Got '%s'." % type(kwarg_spec)
                        )

                    dist_spec_dict['kwargs'][kwarg] = kwarg_eval

                dim_dist_list.append(dist_spec_dict)

            reco_params[flavintgroup][dimension] = dim_dist_list

    return reco_params

def get_physical_bounds(dim):
    """Returns the boundaries of the physical region for the various
    dimensions"""
    dim = basename(dim)

    if dim == "coszen":
        trunc_low = -1.
        trunc_high = 1.

    elif dim == "energy":
        trunc_low = 0.
        trunc_high = None

    elif dim == "azimuth":
        trunc_low = 0.
        trunc_high = 2*np.pi

    else:
        raise ValueError("No physical bounds for dimension '%s' available."%dim)

    return trunc_low, trunc_high

def get_trunc_cdf(rv, dim):
    """Returns the value of the distribution `rv`'s cdf at the physical
    boundaries in the requested dimension (e.g., coszen or energy)"""
    trunc_low, trunc_high = get_physical_bounds(dim=dim)
    cdf_low = rv.cdf(trunc_low) if trunc_low is not None else 0.
    cdf_high = rv.cdf(trunc_high) if trunc_high is not None else 1.

    return cdf_low, cdf_high

def truncate_and_renormalise_dist(rv, frac, bin_edges, dim):
    """Renormalises the part of the distribution `rv` which spans the physical
    domain to 1 (where `frac` is an overall normalisation factor of the
    distribution)"""
    cdf_low, cdf_high = get_trunc_cdf(rv=rv, dim=dim)
    cdfs = frac*rv.cdf(bin_edges)/(cdf_high-cdf_low)

    return cdfs

def truncate_and_renormalise_superposition(weighted_integrals_physical_domain,
                                           binwise_cdf_summed):
    """Renormalise the combined distribution - characterised by
    its binwise quantiles `binwise_cdf_summed` - resulting from a superposition
    of n invidiual distributions with relative weights n_i to integrate to 1
    over the physical domain. `weighted_integrals_physical_domain` is the list
    of n_i-weighted integrals of the constituting distributions."""
    return binwise_cdf_summed/np.sum(weighted_integrals_physical_domain)

def perform_coszen_flipback(cz_kern_cdf, flipback_mask, keep):
    """
    Performs the flipback by mirroring back in any probability quantiles
    that go beyond the physical bounds in coszen. Independent of whether
    the output binning is upgoing, downgoing or allsky, mirror back in
    any density that goes beyond -1 as well as +1.
    """
    flipback = np.where(flipback_mask)[0]

    flipup = flipback[:int(len(flipback)/2)]
    flipdown = flipback[int(len(flipback)/2):]
    no_flipback = np.where(np.logical_not(flipback_mask))[0]

    cz_kern_cdf = (cz_kern_cdf[flipup][::-1] + cz_kern_cdf[flipdown][::-1] +
                   cz_kern_cdf[no_flipback])[keep-int(len(flipback)/2)]

    return cz_kern_cdf

# TODO: the below logic does not generalize to muons, but probably should
# (rather than requiring an almost-identical version just for muons). For
# example, an input arg can dictate neutrino or muon, which then sets the
# input_names and output_names.

class param(Stage):
    """
    From the simulation file, creates 4D histograms of
    [true_energy][true_coszen][reco_energy][reco_coszen] which act as
    2D pdfs for the probability that an event with (true_energy,
    true_coszen) will be reconstructed as (reco_energy,reco_coszen).

    From these histograms and the true event rate maps, calculates
    the reconstructed even rate templates.

    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameters:

        reco_paramfile : string
            Source of the parameterization. File path or resource location; must
            yield a mapping. See `load_reco_param()`.

        res_scale_ref : string
            This is the reference point about which every resolution distribution
            is scaled. "zero" scales about the zero-error point (i.e., the bin
            midpoint). No other reference points implemented.

        e_res_scale : float
            A scaling factor for energy resolutions (cf. `res_scale_ref`).

        cz_res_scale : float
            A scaling factor for coszen resolutions (cf. `res_scale_ref`).

        e_reco_bias : float
            A shift of each energy resolution function's `loc` parameter.

        cz_reco_bias : float
            A shift of each coszen resolution function's `loc` parameter.

    particles : string
        Must be one of 'neutrinos' or 'muons' (though only neutrinos are
        supported at this time).

    input_names : string or list of strings
        Names of inputs expected. These should follow the standard PISA
        naming conventions for flavor/interaction types OR groupings
        thereof. Note that this service's outputs are named the same as its
        inputs. See Conventions section in the documentation for more info.

    transform_groups : string
        Specifies which particles/interaction types to combine together in
        computing the transforms. See Notes section for more details on how
        to specify this string

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

    only_physics_domain_sum : bool
        Set to `True` in order to truncate the superposition of distributions
        at the physical boundaries of cosine zenith and energy, and renormalise
        its area to 1, so as to not smear events into unphysical regions.

    only_physics_domain_distwise : bool
        Set to `True` in order to truncate the individual distributions
        at the physical boundaries of cosine zenith and energy, and renormalise
        their areas to 1, before they are superimposed according to their
        respective relative weights. Has the same effect as
        `only_physics_domain_sum` in the case of a single distribution in any
        dimension.

    coszen_flipback : bool
        Mirror back in "probability" that would otherwise leak into unphysical
        regions in cosine zenith. This will be applied to a cosine zenith range
        of [-3, +3], so there is no mirroring "back-and-forth" between the
        physical cosine zenith boundaries. Not compatible with either one of the
        preferred `only_physics_domain_*` options, which means no correction of
        the energy resolution functions will be made.

    transforms_cache_depth : int >= 0

    outputs_cache_depth : int >= 0

    memcache_deepcopy : bool

    debug_mode : None, bool, or string
        Whether to store extra debug info for this service.

    Notes
    -----
    The `transform_groups` string is interpreted (and therefore defined) by
    pisa.utils.flavInt.flavint_groups_string. E.g. commonly one might use:

    'nue_cc+nuebar_cc, numu_cc+numubar_cc, nutau_cc+nutaubar_cc, nuall_nc+nuallbar_nc'

    Any particle type not explicitly mentioned is taken as a singleton group.
    Plus signs add types to a group, while groups are separated by commas.
    Whitespace is ignored, so add whitespace for readability.

    """
    def __init__(self, params, particles, input_names, transform_groups,
                 sum_grouped_flavints, input_binning, output_binning,
                 only_physics_domain_sum, only_physics_domain_distwise,
                 coszen_flipback, error_method=None, transforms_cache_depth=20,
                 outputs_cache_depth=20, memcache_deepcopy=False,
                 debug_mode=None):
        assert particles in ['neutrinos', 'muons']
        self.particles = particles
        """Whether stage is instantiated to process neutrinos or muons"""

        self.transform_groups = flavintGroupsFromString(transform_groups)
        """Particle/interaction types to group for computing transforms"""

        assert isinstance(sum_grouped_flavints, bool)
        self.sum_grouped_flavints = sum_grouped_flavints

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'reco_paramfile',
            'res_scale_ref', 'e_res_scale', 'cz_res_scale',
            'e_reco_bias', 'cz_reco_bias'
        )

        if isinstance(input_names, str):
            input_names = (''.join(input_names.split(' '))).split(',')

        # Define the names of objects expected in inputs and produced as
        # outputs
        if self.particles == 'neutrinos':
            if self.sum_grouped_flavints:
                output_names = [str(g) for g in self.transform_groups]
            else:
                output_names = input_names
        elif self.particles == 'muons':
            raise NotImplementedError
        else:
            raise ValueError('Particle type `%s` is not valid'
                             % self.particles)

        logging.trace('transform_groups = %s', self.transform_groups)
        logging.trace('output_names = %s', ' :: '.join(output_names))

        if only_physics_domain_sum and only_physics_domain_distwise:
            raise ValueError(
                "Either choose truncation of the superposition at the"
                " physical boundaries or truncation of the individual"
                " distributions, but not both!"
                )

        self.only_physics_domain_sum = only_physics_domain_sum

        self.only_physics_domain_distwise = only_physics_domain_distwise

        only_physics_domain = (only_physics_domain_sum or
                               only_physics_domain_distwise)

        if only_physics_domain and coszen_flipback:
            raise ValueError(
                "Truncating parameterisations at physical boundaries"
                " and flipping back at coszen = +-1 at the same time is"
                " not allowed! Please decide on only one of these."
                )

        self.only_physics_domain = only_physics_domain
        """Whether one of the physics domain restrictions has been requested"""

        self.coszen_flipback = coszen_flipback
        """Whether to flipback coszen error distributions at +1 and -1"""

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
        super().__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            memcache_deepcopy=memcache_deepcopy,
            input_binning=input_binning,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

        self._reco_param_hash = None

        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('transform_groups')
        self.include_attrs_for_hashes('sum_grouped_flavints')
        self.include_attrs_for_hashes('only_physics_domain')
        self.include_attrs_for_hashes('only_physics_domain_sum')
        self.include_attrs_for_hashes('only_physics_domain_distwise')
        self.include_attrs_for_hashes('coszen_flipback')

    def validate_binning(self):
        # Right now this can only deal with 2D energy / coszenith binning
        # Code can probably be generalised, but right now is not
        if set(self.input_binning.names) != set(['true_coszen','true_energy']):
            raise ValueError(
                "Input binning must be 2D true energy / coszenith binning. "
                "Got %s."%(self.input_binning.names)
            )

        assert set(self.input_binning.basename_binning.names) == \
               set(self.output_binning.basename_binning.names), \
               "input and output binning must both be 2D in energy / coszenith!"

        if self.coszen_flipback is None:
            raise ValueError(
                        "coszen_flipback should be set to True or False since"
                        " coszen is in your binning."
                  )

        if self.coszen_flipback:
            coszen_output_binning = self.output_binning.basename_binning['coszen']

            if not coszen_output_binning.is_lin:
                raise ValueError(
                            "coszen_flipback is set to True but zenith output"
                            " binning is not linear - incompatible settings!"
                      )
            coszen_step_out = (coszen_output_binning.range.magnitude/
                               coszen_output_binning.size)

            if not recursiveEquality(int(1/coszen_step_out), 1/coszen_step_out):
                raise ValueError(
                            "coszen_flipback requires an integer number of"
                            " coszen output binning steps to fit into a range"
                            " of integer length."
                      )

    def check_reco_dist_consistency(self, dist_list):
        """Enforces correct normalisation of resolution functions."""
        logging.trace(" Verifying correct normalisation of resolution function.")
        # Obtain list of all distributions. The sum of their relative weights
        # should yield 1.
        frac_sum = np.zeros_like(dist_list[0]['fraction'])
        for dist_dict in dist_list:
            frac_sum += dist_dict['fraction']
        if not recursiveEquality(frac_sum, np.ones_like(frac_sum)):
            err_msg = ("Total normalisation of resolution function is off"
                       " (fractions do not add up to 1).")
            raise ValueError(err_msg)
        return True

    def evaluate_reco_param(self):
        """
        Evaluates the parameterisations for the requested binning and stores
        this in a useful way for eventually constructing the reco kernels.
        """
        evals = self.input_binning['true_energy'].weighted_centers.magnitude
        n_e = len(self.input_binning['true_energy'].weighted_centers.magnitude)
        n_cz = len(self.input_binning['true_coszen'].weighted_centers.magnitude)
        eval_dict = deepcopy(self.param_dict)
        for flavintgroup, dim_dict in eval_dict.items():
            for dim, dist_list in dim_dict.items():
                for dist_prop_dict in dist_list:
                    for dist_prop in dist_prop_dict.keys():
                        if dist_prop == 'dist':
                            continue
                        if callable(dist_prop_dict[dist_prop]):
                            func = dist_prop_dict[dist_prop]
                            vals = func(evals)
                            dist_prop_dict[dist_prop] =\
                                np.repeat(vals,n_cz).reshape((n_e,n_cz))
                        elif isinstance(dist_prop_dict[dist_prop], dict):
                            assert dist_prop == 'kwargs'
                            for kwarg in dist_prop_dict['kwargs'].keys():
                                func = dist_prop_dict['kwargs'][kwarg]
                                vals = func(evals)
                                dist_prop_dict['kwargs'][kwarg] =\
                                    np.repeat(vals,n_cz).reshape((n_e,n_cz))
                # Now check for consistency, to not have to loop over all dict
                # entries again at a later point in time
                self.check_reco_dist_consistency(dist_list)
        return eval_dict

    def make_cdf(self, bin_edges, enval, enindex, czindex, czval, dist_params):
        """
        General make function for the cdf needed to construct the kernels.
        """
        dim = "coszen" if czval is not None else "energy"

        weighted_physical_int = []
        binwise_cdfs = []
        for this_dist_dict in dist_params:
            dist_kwargs = {}
            for dist_prop, prop_vals in this_dist_dict['kwargs'].items():
                dist_kwargs[dist_prop] = prop_vals[enindex, czindex]
            frac = this_dist_dict['fraction'][enindex,czindex]

            # now add error to true parameter value
            dist_kwargs['loc'] += czval if czval is not None else enval
            rv = this_dist_dict['dist'](**dist_kwargs)
            cdfs = frac*rv.cdf(bin_edges)

            if self.only_physics_domain_sum:
                cdf_low, cdf_high = get_trunc_cdf(rv=rv, dim=dim)
                int_weighted_physical = frac*(cdf_high-cdf_low)
                weighted_physical_int.append(int_weighted_physical)

            if self.only_physics_domain_distwise:
                cdfs = truncate_and_renormalise_dist(
                           rv=rv, frac=frac, bin_edges=bin_edges, dim=dim
                       )

            binwise_cdfs.append(cdfs[1:] - cdfs[:-1])

        binwise_cdf_summed = np.sum(binwise_cdfs, axis=0)

        if self.only_physics_domain_sum:
            binwise_cdf_summed = \
                truncate_and_renormalise_superposition(
                    weighted_integrals_physical_domain=weighted_physical_int,
                    binwise_cdf_summed=binwise_cdf_summed
                )

        return binwise_cdf_summed

    def scale_and_shift_reco_dists(self):
        """
        Applies the scales and shifts to all the resolution functions.
        """
        e_res_scale = self.params.e_res_scale.value.m_as('dimensionless')
        cz_res_scale = self.params.cz_res_scale.value.m_as('dimensionless')
        e_reco_bias = self.params.e_reco_bias.value.m_as('GeV')
        cz_reco_bias = self.params.cz_reco_bias.value.m_as('dimensionless')
        eval_dict_mod = deepcopy(self.eval_dict)
        for flavintgroup in eval_dict_mod.keys():
            for (dim, dim_scale, dim_bias) in \
              (('energy', e_res_scale, e_reco_bias),
               ('coszen', cz_res_scale, cz_reco_bias)):
                for i,flav_dim_dist_dict in \
                  enumerate(eval_dict_mod[flavintgroup][dim]):
                    for param in flav_dim_dist_dict["kwargs"].keys():
                        if param == 'scale':
                            flav_dim_dist_dict["kwargs"][param] *= dim_scale
                        elif param == 'loc':
                            flav_dim_dist_dict["kwargs"][param] += dim_bias
        return eval_dict_mod
        
    def reco_scales_and_biases_applicable(self):
        """
        Wrapper function for applying the resolution scales and biases to all
        distributions. Performs consistency check, then calls the function
        that carries out the actual computations.
        """
        # these parameters are the ones to which res scales and biases will be
        # applied
        entries_to_mod = set(('scale', 'loc'))
        # loop over all sub-dictionaries with distribution parameters to check
        # whether all parameters to which the systematics will be applied are
        # really present, raise exception if not
        for flavintgroup in self.eval_dict.keys():
            for dim in self.eval_dict[flavintgroup].keys():
                for flav_dim_dist_dict in self.eval_dict[flavintgroup][dim]:
                    param_view = flav_dim_dist_dict["kwargs"].viewkeys()
                    if not entries_to_mod & param_view == entries_to_mod:
                        raise ValueError(
                        "Couldn't find all of "+str(tuple(entries_to_mod))+
                        " in chosen reco parameterisation, but required for"
                        " applying reco scale and bias. Got %s for %s %s."
                        %(flav_dim_dist_dict["kwargs"].keys(), flavintgroup, dim))
        return

    def extend_binning_for_coszen(self, ext_low=-3., ext_high=+3.):
        """
        Check whether `coszen_flipback` can be applied to the stage's
        coszen output binning and return an extended binning spanning [-3, +3]
        if that is the case.
        """
        logging.trace("Preparing binning for flipback of reco kernel at"
                      " coszen boundaries of physical range.")

        cz_edges_out = self.output_binning['reco_coszen'].bin_edges.magnitude
        coszen_range = self.output_binning['reco_coszen'].range.magnitude
        n_cz_out = self.output_binning['reco_coszen'].size
        coszen_step = coszen_range/n_cz_out
        # we need to check for possible contributions from (-3, -1) and
        # (1, 3) in coszen
        assert ext_high > ext_low
        ext_range = ext_high - ext_low
        extended = np.linspace(ext_low, ext_high, int(ext_range/coszen_step) + 1)

        # We cannot flipback if we don't have -1 & +1 as (part of extended)
        # bin edges. This could happen if 1 is a multiple of the output bin
        # size, but the original edges themselves are not a multiple of that
        # size.
        for bound in (-1., +1.):
            comp = [recursiveEquality(bound, e) for e in extended]
            assert np.any(comp)

        # Perform one final check: original edges subset of extended ones?
        for coszen in cz_edges_out:
            comp = [recursiveEquality(coszen, e) for e in extended]
            assert np.any(comp)

        # Binning seems fine - we can proceed
        ext_cent = (extended[1:] + extended[:-1])/2.
        flipback_mask = ((ext_cent < -1. ) | (ext_cent > +1.))
        keep = np.where((ext_cent > cz_edges_out[0]) &
                            (ext_cent < cz_edges_out[-1]))[0]
        cz_edges_out = extended
        logging.trace("  -> temporary coszen bin edges:\n%s"%cz_edges_out)

        return cz_edges_out, flipback_mask, keep

    def _compute_transforms(self):
        """
        Generate reconstruction "smearing kernels" by reading in a set of
        parameterisation functions from a json file. This should have the same
        dimensionality as the input binning i.e. if you have energy and
        coszenith input binning then the kernels provided should have both
        energy and coszenith resolution functions.

        Any superposition of distributions from scipy.stats is supported.
        """
        res_scale_ref = self.params.res_scale_ref.value.strip().lower()
        assert res_scale_ref in ['zero'] # TODO: , 'mean', 'median']

        reco_param_source = self.params.reco_paramfile.value

        if reco_param_source is None:
            raise ValueError(
                'non-None reco parameterization params.reco_paramfile'
                ' must be provided'
            )

        reco_param_hash = hash_obj(reco_param_source)

        if (self._reco_param_hash is None
                or reco_param_hash != self._reco_param_hash):
            reco_param = load_reco_param(reco_param_source)

            # Transform groups are implicitly defined by the contents of the
            # reco paramfile's keys
            implicit_transform_groups = reco_param.keys()

            # Make sure these match transform groups specified for the stage
            if set(implicit_transform_groups) != set(self.transform_groups):
                raise ValueError(
                    'Transform groups (%s) defined implicitly by'
                    ' %s reco parameterizations do not match those'
                    ' defined as the stage\'s `transform_groups` (%s).'
                    % (implicit_transform_groups, reco_param_source,
                       self.transform_groups)
                )

            self.param_dict = reco_param
            self._reco_param_hash = reco_param_hash

            self.eval_dict = self.evaluate_reco_param()
            self.reco_scales_and_biases_applicable()

        # everything seems to be fine, so rescale and shift distributions
        eval_dict = self.scale_and_shift_reco_dists()

        # Computational units must be the following for compatibility with
        # events file
        comp_units = dict(
            true_energy='GeV', true_coszen=None, true_azimuth='rad',
            reco_energy='GeV', reco_coszen=None, reco_azimuth='rad', pid=None
        )

        # Select only the units in the input/output binning for conversion
        # (can't pass more than what's actually there)
        in_units = {dim: unit for dim, unit in comp_units.items()
                    if dim in self.input_binning}
        out_units = {dim: unit for dim, unit in comp_units.items()
                     if dim in self.output_binning}

        # These binnings will be in the computational units defined above
        input_binning = self.input_binning.to(**in_units)
        output_binning = self.output_binning.to(**out_units)
        en_centers_in = self.input_binning['true_energy'].weighted_centers.magnitude
        en_edges_in = self.input_binning['true_energy'].bin_edges.magnitude
        cz_centers_in = self.input_binning['true_coszen'].weighted_centers.magnitude
        cz_edges_in = self.input_binning['true_coszen'].bin_edges.magnitude
        en_edges_out = self.output_binning['reco_energy'].bin_edges.magnitude
        cz_edges_out = self.output_binning['reco_coszen'].bin_edges.magnitude

        n_e_in = len(en_centers_in)
        n_cz_in = len(cz_centers_in)
        n_e_out = len(en_edges_out)-1
        n_cz_out = len(cz_edges_out)-1

        if self.coszen_flipback:
            cz_edges_out, flipback_mask, keep = \
                self.extend_binning_for_coszen(ext_low=-3., ext_high=+3.)

        xforms = []
        for xform_flavints in self.transform_groups:
            logging.debug("Working on %s reco kernel..." %xform_flavints)

            this_params = eval_dict[xform_flavints]
            reco_kernel = np.zeros((n_e_in, n_cz_in, n_e_out, n_cz_out))

            for (i,j) in itertools.product(range(n_e_in), range(n_cz_in)):
                e_kern_cdf = self.make_cdf(
                    bin_edges=en_edges_out,
                    enval=en_centers_in[i],
                    enindex=i,
                    czval=None,
                    czindex=j,
                    dist_params=this_params['energy']
                )
                cz_kern_cdf = self.make_cdf(
                    bin_edges=cz_edges_out,
                    enval=en_centers_in[i],
                    enindex=i,
                    czval=cz_centers_in[j],
                    czindex=j,
                    dist_params=this_params['coszen']
                )

                if self.coszen_flipback:
                    cz_kern_cdf = perform_coszen_flipback(
                                      cz_kern_cdf, flipback_mask, keep
                                  )

                reco_kernel[i,j] = np.outer(e_kern_cdf, cz_kern_cdf)

            # Sanity check of reco kernels - intolerable negative values?
            logging.trace(" Ensuring reco kernel sanity...")
            kern_neg_invalid = reco_kernel < -EQUALITY_PREC
            if np.any(kern_neg_invalid):
                raise ValueError("Detected intolerable negative entries in"
                                 " reco kernel! Min.: %.15e"
                                 % np.min(reco_kernel))

            # Set values numerically compatible with zero to zero
            np.where(
                (np.abs(reco_kernel) < EQUALITY_PREC), reco_kernel, 0
            )
            sum_over_axes = tuple(range(-len(self.output_binning), 0))
            totals = np.sum(reco_kernel, axis=sum_over_axes)
            totals_large = totals > (1 + EQUALITY_PREC)
            if np.any(totals_large):
                raise ValueError("Detected overflow in reco kernel! Max.:"
                                 " %0.15e" % (np.max(totals)))

            if self.input_binning.basenames[0] == "coszen":
                # The reconstruction kernel has been set up with energy as its
                # first dimension, so swap axes if it is applied to an input
                # binning where 'coszen' is the first
                logging.trace(" Swapping kernel dimensions since 'coszen' has"
                              " been requested as the first.")
                reco_kernel = np.swapaxes(reco_kernel, 0, 1)
                reco_kernel = np.swapaxes(reco_kernel, 2, 3)


            if self.sum_grouped_flavints:
                xform_input_names = []
                for input_name in self.input_names:
                    if set(NuFlavIntGroup(input_name)).isdisjoint(xform_flavints):
                        continue
                    xform_input_names.append(input_name)

                for output_name in self.output_names:
                    if output_name not in xform_flavints:
                        continue
                    xform = BinnedTensorTransform(
                        input_names=xform_input_names,
                        output_name=output_name,
                        input_binning=self.input_binning,
                        output_binning=self.output_binning,
                        xform_array=reco_kernel,
                        sum_inputs=self.sum_grouped_flavints
                    )
                    xforms.append(xform)
            # If *not* combining grouped flavints:
            # Copy the transform for each input flavor, regardless if the
            # transform is computed from a combination of flavors.
            else:
                for input_name in self.input_names:
                    if set(NuFlavIntGroup(input_name)).isdisjoint(xform_flavints):
                        continue
                    for output_name in self.output_names:
                        if (output_name not in NuFlavIntGroup(input_name)
                                or output_name not in xform_flavints):
                            continue
                        logging.trace('  input: %s, output: %s, xform: %s',
                                      input_name, output_name, xform_flavints)

                        xform = BinnedTensorTransform(
                            input_names=input_name,
                            output_name=output_name,
                            input_binning=self.input_binning,
                            output_binning=self.output_binning,
                            xform_array=reco_kernel,
                            sum_inputs=self.sum_grouped_flavints
                        )
                        xforms.append(xform)

        return TransformSet(transforms=xforms)
