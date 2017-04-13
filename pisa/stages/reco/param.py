# PISA author: Lukas Schulte
#              schulte@physik.uni-bonn.de
#
# CAKE authors: Steven Wren
#               steven.wren@icecube.wisc.edu
#               Thomas Ehrhardt
#               tehrhardt@icecube.wisc.edu
#
# date:   2017-03-08

"""
Create the transforms that map from true energy and coszen
to the reconstructed parameters. Provides reco event rate maps using these
transforms.

"""


from __future__ import division
from copy import deepcopy

from collections import Mapping, OrderedDict
import itertools
import numpy as np
from scipy.stats import norm
from scipy import stats

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.fileio import from_file
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.comparisons import recursiveEquality, EQUALITY_PREC


__all__ = ['load_reco_param', 'param']


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
        Keys are stringified flavintgroups and values are dicts of `scipy.stats`
        callables (keys) and lists of distribution property dicts (values,
        one list per key, one property dict per distribution-to-be-superimposed
        of a given type). These property dicts consist of string-callable pairs,
        and are eventually used to produce the reco kernels (via integration)
        when called with energy values (parameterisations are functions of
        energy only!).

    Notes
    -----
    The mapping passed via `source` or loaded therefrom must have the format:
        {
            <flavintgroup_string>:
                {
                    <dimension_string>:
                        {
                            "dist": dist_string,
                            <dist_property_string> : val,
                            ...
                        },
                    ...
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
    function provided by `scipy.stats`. If the `"dist"` key is missing, it will
    be attempted to proceed with a sum of two normal distributions
    (PISA2 default behaviour).

    Valid `dist_property_string`s are: ["loc", "scale", "fraction"] (refer to
    README for how these can be used in the case of multiple distributions that
    are superimposed).

    `val`s can be one of the following:
        - Callable with one argument
        - String such that `eval(val)` yields a callable with one argument
    """
    if not (source is None or isinstance(source, (basestring, Mapping))):
        raise TypeError('`source` must be string, mapping, or None')

    if isinstance(source, basestring):
        orig_dict = from_file(source)

    elif isinstance(source, Mapping):
        orig_dict = source

    else:
        raise TypeError('Cannot load reco parameterizations from a %s'
                        % type(source))

    # Build dict of parameterizations (each a callable) per flavintgroup

    reco_params = OrderedDict()
    for flavint_key, dim_dict in orig_dict.iteritems():
        flavintgroup = NuFlavIntGroup(flavint_key)
        reco_params[flavintgroup] = {}
        for dimension in dim_dict.iterkeys():
            dist_spec_dict = {}
            if 'dist' in dim_dict[dimension]:
                dist_spec = dim_dict[dimension]['dist']

                if not isinstance(dist_spec, basestring):
                    raise TypeError(" The resolution function needs to be"
                                    " given as a string!")

                if not dist_spec:
                    raise ValueError("Empty string found for resolution"
                                     " function! Cannot proceed.")

                logging.debug("Found %s %s resolution function: '%s'"
                              %(flavintgroup, dimension, dist_spec.lower()))

            else:
                # For backward compatibility, assume double Gauss if key
                # is not present.
                logging.warn("No resolution function specified for %s %s."
                             " Assuming sum of two Gaussians."
                             %(flavintgroup, dimension))
                dist_spec = "norm+norm"

            dist_spec_dict['dist'] = dist_spec.lower()

            for dist_prop, param_spec in dim_dict[dimension].iteritems():
                dist_prop = dist_prop.lower()

                if dist_prop == 'dist':
                    continue

                if isinstance(param_spec, basestring):
                    param_func = eval(param_spec)

                elif callable(param_spec):
                    param_func = param_spec

                else:
                    raise TypeError(
                        "Expected parameterization spec to be either a string"
                        " that can be interpreted by eval or a callable."
                        " Got '%s'." % type(param_spec)
                    )
                dist_spec_dict[dist_prop] = param_func

            dist_spec_dict_proc = process_reco_dist_params(dist_spec_dict)

            reco_params[flavintgroup][dimension] = dist_spec_dict_proc

    return reco_params

def process_reco_dist_params(reco_param_dict):
    """
    Ensure consistency between specified reconstruction function(s)
    and their corresponding parameters.
    """
    def select_dist_param_key(allowed, param_dict, unsel=None):
        """
        Evaluates whether 'param_dict' contains exactly
        one of the keys from 'allowed', and returns it if so.
        If none or more than one is found, raises exception.
        `unsel` (if a set) is updated with non-allowed/
        unselected keys.
        """
        logging.trace("  Searching for one of '%s'."%str(allowed))
        allowed_here = set(allowed)
        search_keys = set(param_dict.keys())
        search_found = allowed_here & search_keys
        diff = search_keys.difference(search_found)
        if len(search_found) == 0:
            raise ValueError("No parameter from "+
                             str(tuple(allowed_here))+" found!")
        elif len(search_found) > 1:
            raise ValueError("Please remove one of "+
                             str(tuple(allowed_here))+" !")
        param_str_sel = search_found.pop()
        logging.trace("  Found and selected '%s'."%param_str_sel)
        try:
            unsel.update(diff)
        except:
            pass
        return param_str_sel

    allowed_dist_params = ['loc','scale','fraction']
    # Prepare for detection of parameter ids that are never selected
    sometime_sel = []; sometime_unsel = set()
    # First, get list of distributions to be superimposed
    dists = reco_param_dict['dist'].split("+")
    ndist = len(dists)
    # Need to retain order of specification for correct assignment of
    # distributions' parameters
    dist_type_count = OrderedDict()
    for dist_type in dists:
        dist_type_count[dist_type] = dist_type_count.get(dist_type, 0) + 1
    reco_param_dict.pop('dist')
    dist_param_dict = {}
    tot_dist_count = 1
    # For each distribution type, find all distributions' 'scale' and 'loc'
    # parameterisations and store in a list of dictionaries
    # (with length `this_dist_type_count`)
    for dist_str, this_dist_type_count in dist_type_count.items():
        dist_str = "".join(dist_str.split())
        try:
            dist = getattr(stats, dist_str)
        except AttributeError:
            try:
                import scipy
                sp_ver_str = scipy.__version__
            except:
                sp_ver_str = "N/A"
                raise AttributeError("'%s' is not a valid distribution from"
                                     " scipy.stats (your scipy version: '%s')."
                                     %(dist_str, sp_ver_str))
        dist_param_dict[dist] = []
        for i in xrange(1, this_dist_type_count+1):
            logging.trace(" Collecting parameters for resolution"
                          " function #%d of type '%s'."%(i, dist_str))
            this_dist_dict = {}
            # Also explicitly require a 'fraction' to be present always
            for param in allowed_dist_params:
                if ndist == 1:
                    # There's greater flexibility in this case
                    allowed_here = (param, param+"_"+dist_str,
                                    param+"%s"%tot_dist_count,
                                    param+"_"+dist_str+"%s"%i)
                else:
                    allowed_here = (param+"%s"%tot_dist_count,
                                    param+"_"+dist_str+"%s"%i)
                param_str = select_dist_param_key(allowed_here,
                                                  reco_param_dict,
                                                  sometime_unsel)
                # Keep track of the parameter id that got selected
                sometime_sel += [param_str]
                # Select the corresponding entry
                this_dist_dict[param] = reco_param_dict[param_str]
            # Add to list of distribution properties for each distribution
            # of this type
            dist_param_dict[dist].append(this_dist_dict)
            tot_dist_count += 1
    # Find the parameter ids that are present in the parameterisation
    # dictionary, but which never got selected, and warn the user about those
    never_sel = sometime_unsel.difference(set(sometime_sel))
    if len(never_sel) > 0:
        logging.warn("Unused distribution parameter identifiers detected: "+
                     str(tuple(never_sel)))
    return dist_param_dict

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

        reco_events : string or Events
            PISA events file to use to derive transforms, or a string
            specifying the resource location of the same.

        reco_weights_name : None or string
            Column in the events file to use for Monte Carlo weighting of the
            events

        res_scale_ref : string
            One of "mean", "median", or "zero". This is the reference point
            about which resolutions are scaled. "zero" scales about the
            zero-error point (i.e., the bin midpoint), "mean" scales about the
            mean of the events in the bin, and "median" scales about the median
            of the events in the bin.

        e_res_scale : float
            A scaling factor for energy resolutions.

        cz_res_scale : float
            A scaling factor for coszen resolutions.

        e_reco_bias : float

        cz_reco_bias : float

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

    input_binning : MultiDimBinning or convertible thereto
        Input binning is in true variables, with names prefixed by "true_".
        Each must match a corresponding dimension in `output_binning`.

    output_binning : MultiDimBinning or convertible thereto
        Output binning is in reconstructed variables, with names (traditionally
        in PISA but not necessarily) prefixed by "reco_". Each must match a
        corresponding dimension in `input_binning`.

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
                 truncate_at_physical_bounds=True, coszen_flipback=None,
                 error_method=None, transforms_cache_depth=20,
                 outputs_cache_depth=20, memcache_deepcopy=True, debug_mode=None):
        assert particles in ['neutrinos', 'muons']
        self.particles = particles
        self.transform_groups = flavintGroupsFromString(transform_groups)
        self.sum_grouped_flavints = sum_grouped_flavints

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'reco_paramfile',
            'res_scale_ref', 'e_res_scale', 'cz_res_scale',
            'e_reco_bias', 'cz_reco_bias'
        )

        self.coszen_flipback = coszen_flipback
        self.truncate_at_physical_bounds = truncate_at_physical_bounds

        if self.truncate_at_physical_bounds and self.coszen_flipback:
            raise ValueError(
                        "Truncating parameterisations at physical boundaries"
                        " and flipping back at coszen = +-1 at the same time is"
                        " not allowed! Please decide on only one of these."
                  )

        if isinstance(input_names, basestring):
            input_names = (''.join(input_names.split(' '))).split(',')

        # Define the names of objects expected in inputs and produced as
        # outputs
        if self.particles == 'neutrinos':
            if self.sum_grouped_flavints:
                output_names = [str(g) for g in self.transform_groups]
            else:
                output_names = input_names

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
        super(self.__class__, self).__init__(
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

        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('transform_groups')
        self.include_attrs_for_hashes('sum_grouped_flavints')

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

    def check_reco_dist_consistency(self, dist_param_dict):
        """Enforces correct normalisation of resolution functions."""
        logging.trace(" Verifying correct normalisation of resolution function.")
        # Obtain list of all distributions (one list of dicts for a distribution
        # of a certain type). The sum of their relative weights should yield 1.
        dist_dicts = np.array(dist_param_dict.values())
        frac_sum = np.zeros_like(dist_dicts[0][0]['fraction'])
        for dist_type in dist_dicts:
            for dist_dict in dist_type:
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
        for flavintgroup, dim_dict in eval_dict.iteritems():
            for dim, dist_dict in dim_dict.iteritems():
                for dist, dist_prop_list in dist_dict.iteritems():
                    for dist_prop_dict in dist_prop_list:
                        for dist_prop in dist_prop_dict.iterkeys():
                            func = dist_prop_dict[dist_prop]
                            vals = func(evals)
                            dist_prop_dict[dist_prop] =\
                                np.repeat(vals,n_cz).reshape((n_e,n_cz))
                # Now check for consistency, to not have to loop over all dict
                # entries again at a later point in time
                self.check_reco_dist_consistency(dist_dict)
        return eval_dict

    def make_cdf(self, bin_edges, enval, enindex, czindex, czval, dist_params):
        """
        General make function for the cdf needed to construct the kernels.
        """
        for dist in dist_params.keys():
            binwise_cdfs = []
            for this_dist_dict in dist_params[dist]:
                loc = this_dist_dict['loc'][enindex,czindex]
                scale = this_dist_dict['scale'][enindex,czindex]
                frac = this_dist_dict['fraction'][enindex,czindex]
                # now add error to true parameter value
                loc = loc + czval if czval is not None else loc + enval
                # unfortunately, creating all dists of same type with
                # different parameters and evaluating cdfs doesn't seem
                # to work, so do it one-by-one
                rv = dist(loc=loc, scale=scale)
                if self.truncate_at_physical_bounds:
                    # truncate each distribution at the physical boundaries,
                    # i.e., renormalise so that integral between boundaries yields 1.
                    if czval is None:
                        # energy reco
                        trunc_low = 0.
                        trunc_high = None
                    else:
                        # coszen reco
                        trunc_low = -1.
                        trunc_high = 1.
                    cdf_low = rv.cdf(trunc_low)
                    cdf_high = rv.cdf(trunc_high) if trunc_high is not None else 1.
                    cdfs = frac*rv.cdf(bin_edges)/(cdf_high-cdf_low)
                else:
                    cdfs = frac*rv.cdf(bin_edges)
                binwise_cdfs.append(cdfs[1:] - cdfs[:-1])
            # the following would be nice:
            # cdfs = dist(loc=loc_list, scale=scale_list).cdf(bin_edges)
            # binwise_cdfs = [cdf[1:]-cdf[:-1] for cdf in cdfs]
        binwise_cdf_summed = np.sum(binwise_cdfs, axis=0)
        return binwise_cdf_summed

    def scale_and_shift_reco_dists(self):
        """
        Applies the scales and shifts to all the resolution functions.
        """
        e_res_scale = self.params.e_res_scale.value.m_as('dimensionless')
        cz_res_scale = self.params.cz_res_scale.value.m_as('dimensionless')
        e_reco_bias = self.params.e_reco_bias.value.m_as('GeV')
        cz_reco_bias = self.params.cz_reco_bias.value.m_as('dimensionless')
        for flavintgroup in self.eval_dict.keys():
            for (dim, dim_scale, dim_bias) in \
              (('energy', e_res_scale, e_reco_bias),
               ('coszen', cz_res_scale, cz_reco_bias)):
                for dist in self.eval_dict[flavintgroup][dim].keys():
                    for i,flav_dim_dist_dict in \
                      enumerate(self.eval_dict[flavintgroup][dim][dist]):
                        for param in flav_dim_dist_dict.keys():
                            if param == 'scale':
                                flav_dim_dist_dict[param] *= dim_scale
                            elif param == 'loc':
                                flav_dim_dist_dict[param] += dim_bias
        
    def apply_reco_scales_and_biases(self):
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
                for dist in self.eval_dict[flavintgroup][dim].keys():
                    for flav_dim_dist_dict in self.eval_dict[flavintgroup][dim][dist]:
                        param_view = flav_dim_dist_dict.viewkeys()
                        if not entries_to_mod & param_view == entries_to_mod:
                            raise ValueError(
                            "Couldn't find all of "+str(tuple(entries_to_mod))+
                            " in chosen reco parameterisation, but required for"
                            " applying reco scale and bias. Got %s for %s %s."
                            %(flav_dim_dist_dict.keys(), flavintgroup, dim))
        # everything seems to be fine, so rescale and shift distributions
        self.scale_and_shift_reco_dists()

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

        reco_param = load_reco_param(self.params['reco_paramfile'].value)

        # Transform groups are implicitly defined by the contents of the
        # `pid_energy_paramfile`'s keys
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
        self._param_hash = reco_param_hash

        self.eval_dict = self.evaluate_reco_param()
        self.apply_reco_scales_and_biases()

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
            logging.trace("Preparing binning for flipback of reco kernel at"
                          " coszen boundaries of physical range.")
            coszen_range = self.output_binning['reco_coszen'].range.magnitude
            coszen_step = coszen_range/n_cz_out
            # we need to check for possible contributions from (-3, -1) and
            # (1, 3) in coszen
            extended = np.linspace(-3., 3., int(6/coszen_step) + 1)
            # We cannot flipback if we don't have -1 & +1 as (part of extended)
            # bin edges. This could happen if 1 is a multiple of the output bin
            # size, but the original edges themselves are not a multiple of that
            # size.
            for edge in (-1., +1.):
                comp = [recursiveEquality(edge, e) for e in extended]
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

        xforms = []
        for xform_flavints in self.transform_groups:
            logging.debug("Working on %s reco kernel..." %xform_flavints)

            this_params = self.eval_dict[xform_flavints]
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
                    # independent of whether the output binning is upgoing,
                    # downgoing or allsky, we mirror back in any density that
                    # goes beyond -1 as well as +1
                    flipback = np.where(flipback_mask)[0]
                    cz_kern_cdf = \
                        (
                            cz_kern_cdf[flipback[:int(len(flipback)/2)]][::-1] +
                            cz_kern_cdf[flipback[int(len(flipback)/2):]][::-1] +
                            cz_kern_cdf[np.logical_not(flipback_mask)]
                        )[keep-int(len(flipback)/2)]

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
                    input_flavs = NuFlavIntGroup(input_name)
                    if len(set(xform_flavints).intersection(input_flavs)) > 0:
                        xform_input_names.append(input_name)

                for output_name in self.output_names:
                    if not output_name in xform_flavints:
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
            else:
                # NOTES:
                # * Output name is same as input name
                # * Use `self.input_binning` and `self.output_binning` so maps
                #   are returned in user-defined units (rather than
                #   computational units, which are attached to the non-`self`
                #   versions of these binnings).
                for input_name in self.input_names:
                    if input_name not in xform_flavints:
                        continue
                    xform = BinnedTensorTransform(
                        input_names=input_name,
                        output_name=input_name,
                        input_binning=self.input_binning,
                        output_binning=self.output_binning,
                        xform_array=reco_kernel,
                    )
                    xforms.append(xform)

        return TransformSet(transforms=xforms)
