#!/usr/bin/env python
"""
Hyperplane fitting scriot

Produce fit results for sets of discrete systematics (i.e. for example
several simulations for different DOM efficiencies)

The parameters and settings going into the fit are given by an external cfg
file (fit config).

n-dimensional MapSets are supported to be fitted with m-dimensional, linear
hyperplanes functions
"""
from __future__ import absolute_import, division

import os, copy, sys

from argparse import ArgumentParser
from collections import Mapping, Sequence, OrderedDict
from uncertainties import unumpy as unp
from uncertainties import ufloat

import numpy as np
from scipy.optimize import curve_fit

from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.utils.fileio import from_file, to_file
from pisa.utils.log import logging, set_verbosity

import matplotlib.pyplot as plt #TODO REMOVE


__all__ = ['parse_args', 'main']


def parse_args():
    """Parse arguments from command line.
    """
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument(
        '-f', '--fit-cfg', type=str,
        metavar='configfile', required=True,
        help='Settings for the hyperplane fit'
    )
    parser.add_argument(
        '-sp', '--set-param', type=str, default=None, action='append',
        help='''Currently *NOT* implemented. Set one or multiple parameters
        to a certain value (e.g. to test stability of parameterisation).'''
    )
    parser.add_argument(
        '--tag', type=str, default='deepcore',
        help='Tag for the filename.'
    )
    parser.add_argument(
        '-o', '--outdir', type=str, required=True,
        help='Set output directory'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='''Plot distribution of fit parameters, bin-by-bin variations
        from systematics sets, and the distribution of chi-square residuals
        together with a chi-square fit.'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    args = parser.parse_args()
    return args


def hyperplane_fun(x, *p):
    """Hyperplane fit function (just defines plane in n dimensions).

    Parameters
    ----------
    x : list
        nested list holding the different assumed values of each parameter
        in the second dimension (i.e., m values for m discrete sets)
        e.g. [ [vals for param 0], [vals for param 1], ... ]
    p : list
        list of fit function parameters values
        (one offset, n slopes, where n is the number of systematic parameters)

    Returns
    -------
    fun : list
        function value vector (one value in each systematics dimension)

    TODO Avoid duplication with pi_hyperplanes.eval_hyperplane

    """
    fun = p[0]
    for xval, pval in zip(x, p[1:]):
        fun += xval*pval
    return fun


def parse_fit_config(fit_cfg):
    """Perform sanity checks on and parse fit configuration file.

    Parameters
    ----------
    fit_cfg : str
        path to a fit configuration file

    Returns
    -------
    fit_cfg : PISAConfigParser
        parsed fit configuration
    sys_list : list of str
        parsed names of systematic parameters
    combine_regex : list of str
        parsed regular expressions for combining pipeline outputs

    """
    fit_cfg = from_file(fit_cfg)
    general_key = 'general'
    if not fit_cfg.has_section(general_key):
        raise KeyError(
            'Fit config is missing the "%s" section!' % general_key
        )
    sys_list_key = 'sys_list'
    if not sys_list_key in fit_cfg[general_key]:
        raise KeyError(
            'Fit config has to specify systematic parameters as'
            ' "%s" option in "%s" section (comma-separated list of names).'
            % (sys_list_key, general_key)
        )
    sys_list = fit_cfg.get(general_key, sys_list_key).replace(' ', '').split(',')
    logging.info('Found systematic parameters %s.' % sys_list) # pylint: disable=logging-not-lazy
    combine_regex_key = 'combine_regex'
    combine_regex = fit_cfg.get(general_key, combine_regex_key, fallback=None)
    if combine_regex:
        combine_regex = combine_regex.replace(' ', '').split(',')


    return fit_cfg, sys_list, combine_regex


def make_discrete_sys_distributions(fit_cfg):
    """Generate and store mapsets for different discrete systematics sets
    (with a single set characterised by a dedicated pipeline configuration)

    Parameters
    ----------
    fit_cfg : string
        path to a fit config file

    Returns
    -------
    nominal_mapset : MapSet
        mapset corresponding to the nominal set (as defined in fit settings)
    sys_list : list
        list of systematic parameter names (as given in fit settings)
    sys_param_points : list
        a list holding the values of the systematic parameters in sys_list
        for each discrete set (user is responsible to specify the correct
        values in the fit settings)
    sys_mapsets : list
        list of mapsets, one for each point in sys_param_points

    Notes
    -----
    The nominal mapset is also included in sys_mapsets. It is not treated
    any differently than the systematics variations.

    """


    '''
    # parse the fit config and get other things which we need further down
    fit_cfg, sys_list, combine_regex = parse_fit_config(fit_cfg)

    sys_param_points = []
    sys_mapsets = []
    nominal_mapset = None
    # retrieve sets:
    for section in fit_cfg.sections():

        if section == 'general':
            continue

        elif section.startswith('nominal_set:') or section.startswith('sys_set:'):
            sys_param_point = [float(x) for x in section.split(':')[1].split(',')]
            if section.startswith('nominal_set:') :
                nom_sys_param_points = copy.deepcopy(sys_param_point)
            point_str = ' | '.join(['%s=%.2f' % (param, val) for param, val in
                                    zip(sys_list, sys_param_point)])
            # this is what "characterises" a systematics set
            sys_set_specifier = 'pipeline_cfg'
            # retreive settings
            section_keys = fit_cfg[section].keys()
            diff = set(section_keys).difference(set([sys_set_specifier]))
            if diff:
                raise KeyError(
                    'Systematics sets in fit config must be specified via'
                    ' the "%s" key, and no more. Found "%s".'
                    % (sys_set_specifier, diff)
                )
            pipeline_cfg = fit_cfg.get(section, sys_set_specifier)
            if not len(sys_param_point) == len(sys_list):
                raise ValueError(
                    '%s "%s" specifies %d systematic parameter values'
                    ' (%s), but list of systematics is %s. Make sure'
                    ' number of values in section headers agree with'
                    ' number of systematic parameters.'
                    % (section[:section.find(':')], pipeline_cfg,
                       len(sys_param_point), sys_param_point,
                       sys_list)
                )
            # retreive maps
            logging.info( # pylint: disable=logging-not-lazy
                'Generating maps for discrete systematics point: %s. Using'
                ' pipeline config at %s.' % (point_str, pipeline_cfg)
            ) # pylint: disable=logging-not-lazy
            # make a dedicated distribution maker for each systematics set

            distribution_maker = DistributionMaker(pipeline_cfg)
            mapset = distribution_maker.get_outputs(return_sum=False)[0]
            if combine_regex:
                logging.info(
                    'Combining maps according to regular expression(s) %s'
                    % combine_regex
                )
                mapset = mapset.combine_re(combine_regex)
        else:
            raise ValueError(
                'Additional, unrecognized section in fit cfg. file: %s'
                % section
            )

        # add them to the right place
        if section.startswith('nominal_set:'):
            if nominal_mapset:
                raise ValueError(
                    'Found multiple nominal sets in fit cfg! There must be'
                    ' exactly one.'
                )
            nominal_mapset = mapset
        # we have already checked that the section is either for the nominal
        # or for the systematics variation sets, and the nominal set will be
        # treated just the same as the variations
        sys_mapsets.append(mapset)
        sys_param_points.append(sys_param_point)

    nsets = len(sys_mapsets)
    nsys = len(sys_list)
    if not nsets > nsys:
        logging.warn( # pylint: disable=logging-not-lazy
            'Fit will either fail or be unreliable since the number of'
            ' systematics sets to be fit is small (%d <= %d).'
            % (nsets, nsys + 1)
        )

    if not nominal_mapset:
        raise ValueError(
            'Could not find a nominal discrete systematics set in fit cfg.'
            ' There must be exactly one.'
        )

    return nominal_mapset, sys_list, sys_param_points, nom_sys_param_points, sys_mapsets




    '''

    # parse the fit config and get other things which we need further down
    fit_cfg, sys_list, combine_regex = parse_fit_config(fit_cfg)

    # prepare the data container
    input_data = OrderedDict()
    input_data["param_names"] = sys_list
    input_data["datasets"] = []

    # retrieve sets:
    found_nominal = False
    for section in fit_cfg.sections():

        # skip the general section
        if section == 'general':
            continue

        # Handle a dataset definitjon (could be nominal)
        elif section.startswith('nominal_set:') or section.startswith('sys_set:'):

            # Parse the list of parameter values
            sys_param_point = [float(x) for x in section.split(':')[1].split(',')]

            point_str = ' | '.join(['%s=%.2f' % (param, val) for param, val in
                                    zip(sys_list, sys_param_point)])

            # this is what "characterises" a systematics set
            sys_set_specifier = 'pipeline_cfg'

            # retreive settings
            section_keys = fit_cfg[section].keys()
            diff = set(section_keys).difference(set([sys_set_specifier]))
            if diff:
                raise KeyError(
                    'Systematics sets in fit config must be specified via'
                    ' the "%s" key, and no more. Found "%s".'
                    % (sys_set_specifier, diff)
                )
            pipeline_cfg = fit_cfg.get(section, sys_set_specifier)
            if not len(sys_param_point) == len(sys_list):
                raise ValueError(
                    '%s "%s" specifies %d systematic parameter values'
                    ' (%s), but list of systematics is %s. Make sure'
                    ' number of values in section headers agree with'
                    ' number of systematic parameters.'
                    % (section[:section.find(':')], pipeline_cfg,
                       len(sys_param_point), sys_param_point,
                       sys_list)
                )

            # retreive maps
            logging.info( # pylint: disable=logging-not-lazy
                'Generating maps for discrete systematics point: %s. Using'
                ' pipeline config at %s.' % (point_str, pipeline_cfg)
            ) # pylint: disable=logging-not-lazy
            # make a dedicated distribution maker for each systematics set
            distribution_maker = DistributionMaker(pipeline_cfg)
            print(pipeline_cfg)
            mapset = distribution_maker.get_outputs(return_sum=False)[0]
            if combine_regex:
                logging.info(
                    'Combining maps according to regular expression(s) %s'
                    % combine_regex
                )
                mapset = mapset.combine_re(combine_regex)

        # handle unexpected cfg file section
        else:
            raise ValueError(
                'Additional, unrecognized section in fit cfg. file: %s'
                % section
            )

        # handle the nominal dataset
        nominal = section.startswith('nominal_set:')
        if nominal :
            if found_nominal :
                raise ValueError(
                    'Found multiple nominal sets in fit cfg! There must be'
                    ' exactly one.'
                )
            found_nominal = nominal

        # Store the info
        dataset = OrderedDict()
        dataset["name"] = os.path.splitext(os.path.basename(pipeline_cfg))[0] #TODO Do something better here...
        dataset["param_values"] = sys_param_point
        dataset["mapset"] = mapset
        dataset["nominal"] = nominal
        input_data["datasets"].append(dataset)
    # perform some checks
    nsets = len(input_data["datasets"])
    nsys = len(sys_list)
    if not nsets > nsys:
        logging.warn( # pylint: disable=logging-not-lazy
            'Fit will either fail or be unreliable since the number of'
            ' systematics sets to be fit is small (%d <= %d).'
            % (nsets, nsys + 1)
        )

    if not found_nominal:
        raise ValueError(
            'Could not find a nominal discrete systematics set in fit cfg.'
            ' There must be exactly one.'
        )

    return input_data
    #'''


def norm_sys_distributions(input_data):
#def norm_sys_distributions(nominal_mapset, sys_mapsets):

    """Normalises systematics mapsets to the nominal mapset,
    performing error propagation.

    Parameters
    ----------
    nominal_mapset : MapSet
        the reference mapset at the nominal values of the systematics
    sys_mapsets : MapSet
        mapsets from variations of the systematics

    Returns
    -------
    norm_sys_maps : dict
        list of normalised maps (nominal + variations) for each event group

    """


    '''

    #
    # Get the input mapsets
    #

    nominal_mapset = [ dataset["mapset"] for dataset in input_data["datasets"] if dataset["nominal"] ]
    assert len(nominal_mapset) == 1
    nominal_mapset = nominal_mapset[0]

    sys_mapsets = [ dataset["mapset"] for dataset in input_data["datasets"] ]


    #
    # Prepare output data container
    #

    out_names = sorted(nominal_mapset.names)
    #norm_sys_maps = {map_name: [] for map_name in out_names}

    norm_sys_maps = OrderedDict([ (map_name,[]) for map_name in out_names ]) 


    #
    # loop over types of event
    #

    for map_name in out_names:

        logging.info('Normalizing "%s" maps.' % map_name) # pylint: disable=logging-not-lazy
        nominal_map = nominal_mapset[map_name]
        chan_norm_sys_maps = []

        #
        # loop over datasets
        #

        for sys_mapset in sys_mapsets:

            #
            # Normalise maps
            #

            # TODO: think about the best way to perform unc. propagation

            # calculate a normalised version of the systematic set histogram
            # need to handle cases where the nominal histogram has empty bins
            #norm_sys_map = sys_mapset[map_name].hist/nominal_map.nominal_values
            norm_sys_hist = copy.deepcopy(sys_mapset[map_name].hist)
            finite_mask = np.isfinite( sys_mapset[map_name].nominal_values / nominal_map.nominal_values )
            norm_sys_hist[finite_mask] = sys_mapset[map_name].hist[finite_mask] / nominal_map.nominal_values[finite_mask]
            norm_sys_hist[~finite_mask] = ufloat(np.NaN,np.NaN)

            #TODO Check for bins that are empty in the nominal hist but no in at least one of the sys sets, currently we do not support this...

            norm_sys_map = Map(name=sys_mapset[map_name].name, binning=sys_mapset[map_name].binning, hist=norm_sys_hist) #TODO Save the map

            chan_norm_sys_maps.append(norm_sys_hist)

        chan_norm_sys_maps = np.array(chan_norm_sys_maps)

        #print("+++ shape before = %s" % (chan_norm_sys_maps.shape,) )

        # move to last axis
        chan_norm_sys_maps = np.rollaxis(
            chan_norm_sys_maps, axis=0, start=len(chan_norm_sys_maps.shape)
        )

        #print("+++ shape after = %s" % (chan_norm_sys_maps.shape,) 

        norm_sys_maps[map_name] = chan_norm_sys_maps

    return norm_sys_maps
    '''


    #
    # Get the input mapsets
    #

    nominal_mapset = [ dataset["mapset"] for dataset in input_data["datasets"] if dataset["nominal"] ]
    assert len(nominal_mapset) == 1
    nominal_mapset = nominal_mapset[0]

    sys_mapsets = [ dataset["mapset"] for dataset in input_data["datasets"] ]

    for dataset_dict in input_data["datasets"] :
        dataset_dict["norm_mapset"] = []


    #
    # loop over types of event
    #

    for map_name in nominal_mapset.names:

        logging.info('Normalizing "%s" maps.' % map_name) # pylint: disable=logging-not-lazy
        nominal_map = nominal_mapset[map_name]
        chan_norm_sys_maps = []


        #
        # loop over datasets
        #

        for dataset_dict in input_data["datasets"] :

            #
            # Normalise maps
            #

            sys_mapset = dataset_dict["mapset"]

            # TODO: think about the best way to perform unc. propagation

            # calculate a normalised version of the systematic set histogram
            # need to handle cases where the nominal histogram has empty bins
            #norm_sys_map = sys_mapset[map_name].hist/nominal_map.nominal_values
            norm_sys_hist = copy.deepcopy(sys_mapset[map_name].hist)
            finite_mask = np.isfinite( sys_mapset[map_name].nominal_values / nominal_map.nominal_values )
            norm_sys_hist[finite_mask] = sys_mapset[map_name].hist[finite_mask] / nominal_map.nominal_values[finite_mask]
            norm_sys_hist[~finite_mask] = ufloat(np.NaN,np.NaN)

            #TODO Check for bins that are empty in the nominal hist but no in at least one of the sys sets, currently we do not support this...

            norm_sys_map = Map(name=sys_mapset[map_name].name, binning=sys_mapset[map_name].binning, hist=norm_sys_hist) #TODO Save the map
            dataset_dict["norm_mapset"].append(norm_sys_map)


    # Re-format
    for dataset_dict in input_data["datasets"] :
        dataset_dict["norm_mapset"] = MapSet( maps=dataset_dict["norm_mapset"], name=dataset_dict["mapset"].name )

    #return norm_sys_maps


#def fit_discrete_sys_distributions(
#        nominal_mapset, sys_list, sys_param_points, sys_mapsets, p0=None
#    ):
def fit_discrete_sys_distributions(input_data,p0=None) :

    """Fits a hyperplane to MapSets generated at given systematics parameters
    values.

    Parameters
    ----------
    nominal_mapset : MapSet
        nominal mapset, used for normalisation of the systematics variation
        mapsets
    sys_list : list
        list of systematic parameter names (just to put in output dictionary)
    sys_param_points : list
        a list holding the values of the systematic parameters in sys_list
        for each discrete set (passed as x values to the fitting function)
    sys_mapsets : list
        list of mapsets, one for each point in sys_param_points, should
        include the nominal mapset also
    p0 : list or dict
        Initial guess list (same initial guess for all maps) or dictionary
        (keys have to correspond to event groups/channels in maps)
        with one offset and len(sys_list) slopes. Default is list of ones.

    Returns
    -------
    fit_results : dict
        stores fit results, i.e., map names, fit parameters for each map,
        parameter covariances under 'pcov', the names of
        the systematic parameters, the hash of the binning
    chi2s : numpy array
        individual chi-square residuals between fit and data points
    binning : MultiDimBinning
        binning of all maps

    """

    #
    # Prepare a few things before fitting
    #

    # prepare an output data container
    fit_results = OrderedDict()
    fit_results["hyperplanes"] = OrderedDict()
    fit_results["norm_maps"] = OrderedDict()

    # store info from the input data in the fit results
    fit_results["datasets"] = input_data["datasets"]
    fit_results["param_names"] = input_data["param_names"]

    # get number of systematic parameters and datasets
    n_sys_params = len(fit_results["param_names"])
    n_datasets = len(fit_results["datasets"])

    # get number of params in hyperplane fit
    # this is one slope per systematic, plus a single intercept
    n_fit_params = 1 + len(fit_results["param_names"])

    # get binning info
    binning =  fit_results["datasets"][0]["mapset"][0].binning
    binning_shape = list(binning.shape)

    # normalise the systematics variations to the nominal distribution
    # with error propagation
    norm_sys_distributions(input_data)

    # re-organise normalised maps to be stored per event type (a list for each dataset)
    norm_sys_maps = OrderedDict()
    for map_name in input_data["datasets"][0]["norm_mapset"].names :
        norm_sys_maps[map_name] = [ dataset_dict["norm_mapset"][map_name] for dataset_dict in input_data["datasets"] ]

    # get an array of the systematic parameter points sampled across all datasets
    # transpose to get format compatible with scipy.optimize.curve_fit
    sys_param_points = np.asarray([ dataset_dict["param_values"] for dataset_dict in fit_results["datasets"] ]) #[datasets,params]
    sys_param_points_T = sys_param_points.T
    assert sys_param_points_T.shape[0] == n_sys_params
    assert sys_param_points_T.shape[1] == n_datasets

    # store some of this stuff 
    fit_results["sys_param_points"] = sys_param_points
    fit_results['binning_hash'] = binning.hash


    #
    # Prepare initial parameter guesses
    #

    if p0:
        if isinstance(p0, Mapping):
            p0_keys = sorted(p0.keys())
            map_keys = sorted(norm_sys_maps.keys())
            if not p0_keys == map_keys:
                raise KeyError(
                    'Initial guess mapping contains keys %s which are not the'
                    ' same as %s in maps.' % (p0_keys, map_keys)
                )
            for k, ini_guess in p0.items():
                assert len(ini_guess) == n_fit_params
        elif isinstance(p0, Sequence):
            assert len(p0) == n_fit_params
            p0 = {map_name: p0 for map_name in norm_sys_maps.keys()}
        else:
            raise TypeError(
                'Initial guess must be a mapping or a sequence. Found %s.'
                % type(p0)
            )
    else:
        p0 = {map_name: np.ones(n_fit_params) for map_name in norm_sys_maps.keys()}

    fit_results['p0'] = p0


    #
    # Loop over event types
    #

    for map_name, chan_norm_sys_maps in norm_sys_maps.items():

        logging.info( # pylint: disable=logging-not-lazy
            'Fitting "%s" maps with initial guess %s.'
            % (map_name, p0[map_name])
        )

        # store the normalised maps
        #fit_results["norm_maps"][map_name] = chan_norm_sys_maps #TODO

        # create a container for fit results for this event type
        fit_results["hyperplanes"][map_name] = OrderedDict()

        # initialise data arrays with NaNs
        fit_results["hyperplanes"][map_name]["fit_params"] = np.full(binning_shape+[n_fit_params], np.nan) #[bins..., hyperplane params]
        fit_results["hyperplanes"][map_name]["chi2s"] = np.full(binning_shape+[n_datasets], np.nan) #[bins..., datasets]
        fit_results["hyperplanes"][map_name]["cov_matrices"] = np.full(binning_shape+[n_fit_params,n_fit_params], np.nan) #[bins..., hyperplane params, hyperplane params]
        fit_results["hyperplanes"][map_name]["finite_mask"] = np.full(binning_shape+[n_datasets], np.nan) #[bins..., datasets]


        #
        # loop over bins
        #

        for idx in np.ndindex(*binning_shape):

            #print "idx = %s" % (idx,)

            # get the bin content, including uncertainty and mask indicating if the bin is finite
            # treat the bin content as y values in the fit, e.g. y(x0,...,xN) where N is the number of parameters
            # each of these 1D arrays has one element per input dataset
            y = np.asarray([ m.hist[idx] for m in chan_norm_sys_maps ])
            y_values = unp.nominal_values(y)
            y_sigma = unp.std_devs(y)
            finite_mask = np.isfinite(y_values) & np.isfinite(y_sigma)


            #
            # Perform hyperplane fit in this bin
            #

            # case 1: uncertainties are available in the bins
            if np.any(y_sigma[finite_mask]):

                # fit
                popt, pcov = curve_fit(
                    hyperplane_fun, sys_param_points_T[:,finite_mask], y_values[finite_mask],
                    sigma=y_sigma[finite_mask], p0=p0[map_name]
                )

                # calculate chi-square values comparing the input data and the fit results at 
                # each data point (e.g. per dataset, and of course in each bin)
                for point_idx in range(n_datasets):
                    point = sys_param_points_T[:, point_idx]
                    predicted = hyperplane_fun(point, *popt)
                    observed = y_values[point_idx]
                    sigma = y_sigma[point_idx]
                    chi2 = ((predicted - observed)/sigma)**2

            else:

                # if here, no errors are available for this bin

                # case 2: there are at least central values in the bins
                if np.any(y_values[finite_mask]):

                    # without error estimates each point has the same weight
                    # and we cannot get chi-square values (but can still fit)
                    logging.warn( # pylint: disable=logging-not-lazy
                        'No uncertainties for any of the normalised counts in bin'
                        ' %s ("%s") found. Fit is performed unweighted and no'
                        ' chisquare values will be available.' % (idx, map_name)
                    )

                    # fit
                    popt, pcov = curve_fit(
                        hyperplane_fun, sys_param_points_T[:,finite_mask], y_values, p0=p0[map_name]
                    )

                    # cannot calculate chi2 without uncertainties
                    chi2 = np.nan

                else :

                    # case 3: no data in this bin
                    # this is the worst case, where there are no central values or errors.
                    # most likely this came about because this bin is empty, which is not 
                    # necessarily an error.

                    # Store NaN for fit params and chi2
                    popt = np.full_like(p0[map_name],np.NaN) 
                    pcov = np.NaN #TODO Shape?
                    chi2 = np.nan


            # store the results for this bin
            fit_results["hyperplanes"][map_name]["fit_params"][idx] = popt
            fit_results["hyperplanes"][map_name]["chi2s"][idx] = chi2
            fit_results["hyperplanes"][map_name]["cov_matrices"][idx] = pcov
            fit_results["hyperplanes"][map_name]["finite_mask"][idx] = finite_mask

            #fit_results[map_name][idx] = popt
            #fit_results['pcov'][map_name][idx] = pcov


    #return fit_results, np.asarray(chi2s), binning
    return fit_results


def hyperplane(fit_cfg, set_param=None):
    """Wrapper around distribution generation and fitting functions.

    Parameters
    ----------
    fit_cfg : string
        path to a fit cfg file
    set_param : not implemented

    Returns
    -------
    nominal_mapset : MapSet
        nominal mapset, used for normalisation of the systematics variation
        mapsets
    sys_param_points : list
        a list holding the values of the systematic parameters
        for each discrete set (passed as x values to the fitting function)
    sys_mapsets : list
        list of mapsets, one for each point in sys_param_points, should
        include the nominal mapset also
    binning : MultiDimBinning
        binning of all maps
    hyperplane_fits : dict
        fit results
    chi2s : list
        chi-square values of fits

    """

    if set_param:
        raise NotImplementedError()

    '''
    nominal_mapset, sys_list, sys_param_points, nom_sys_param_points, sys_mapsets =\
        make_discrete_sys_distributions(fit_cfg=fit_cfg)

    hyperplane_fits, chi2s, binning = fit_discrete_sys_distributions(
        nominal_mapset=nominal_mapset,
        sys_list=sys_list,
        sys_param_points=sys_param_points,
        sys_mapsets=sys_mapsets
    )
    '''

    input_data = make_discrete_sys_distributions(fit_cfg=fit_cfg)
    fit_results = fit_discrete_sys_distributions(input_data=input_data)

    return input_data,fit_results

#    return (nominal_mapset, sys_param_points, nom_sys_param_points, sys_mapsets, binning,
#            hyperplane_fits, chi2s)


'''
def save_hyperplane_fits(hyperplane_fits, chi2s, sys_param_points,
        nom_sys_param_points, nominal_mapset, sys_mapsets, outdir, tag=None):
    """Store discrete systematics fits and chi-square values to a specified
    output location, with results identified by a tag.

    Parameters
    ----------
    hyperplane_fits : dict
        as output by fit_discrete_sys_distributions
    chi2s : list
        chi-square values
    outdir : string
        output directory
    tag : string
        identifier for filenames holding fit results

    """


    dim = len(hyperplane_fits['sys_list'])
    param_str = '_'.join(hyperplane_fits['sys_list'])

    # Store fits as JSON
    res_path = os.path.join(
        outdir,
        '%s_%dd_%s_hyperplane_fits.json' % (tag, dim, param_str)
    )
    to_file(hyperplane_fits, res_path)

    # Store chi2 as numpy array #TODO Store in the 'aux' JSON file below, or is it jusyt too inefficient?
    chi2s = np.array(chi2s)
    chi2s_path = os.path.join(
        outdir,
        '%s_%dd_%s_chi2s' % (tag, dim, param_str)
    )

    print("chi2: %s : %s" % (type(chi2s), chi2s.shape,) )
    sys.exit()

    np.save(chi2s_path, chi2s)

    # Store map sets and sys points
    input_data = OrderedDict()
    input_data["sys_param_points"] = sys_param_points
    input_data["nom_sys_param_points"] = nom_sys_param_points
    input_data["nominal_mapset"] = nominal_mapset
    input_data["sys_mapsets"] = sys_mapsets
    res_path = os.path.join(
        outdir,
        '%s_%dd_%s_hyperplane_fits_aux.json' % (tag, dim, param_str)
    )
    to_file(input_data, res_path)





    dim = len(hyperplane_fits['sys_list'])
    param_str = '_'.join(hyperplane_fits['sys_list'])

    # Store fits as JSON
    res_path = os.path.join(
        outdir,
        '%s_%dd_%s_hyperplane_fits.json' % (tag, dim, param_str)
    )
    to_file(hyperplane_fits, res_path)

    # Store chi2 as numpy array #TODO Store in the 'aux' JSON file below, or is it jusyt too inefficient?
    chi2s = np.array(chi2s)
    chi2s_path = os.path.join(
        outdir,
        '%s_%dd_%s_chi2s' % (tag, dim, param_str)
    )
    np.save(chi2s_path, chi2s)

    # Store map sets and sys points
    input_data = OrderedDict()
    input_data["sys_param_points"] = sys_param_points
    input_data["nom_sys_param_points"] = nom_sys_param_points
    input_data["nominal_mapset"] = nominal_mapset
    input_data["sys_mapsets"] = sys_mapsets
    res_path = os.path.join(
        outdir,
        '%s_%dd_%s_hyperplane_fits_aux.json' % (tag, dim, param_str)
    )
    to_file(input_data, res_path)

'''


def save_hyperplane_fits(input_data,fit_results,outdir,tag):
    """Store discrete systematics fits and chi-square values to a specified
    output location, with results identified by a tag.

    Parameters
    ----------
    hyperplane_fits : dict
        as output by fit_discrete_sys_distributions
    chi2s : list
        chi-square values
    outdir : string
        output directory
    tag : string
        identifier for filenames holding fit results

    """

    dim = len(input_data["param_names"])
    param_str = '_'.join(input_data["param_names"])

    #TODO Also store input_data

    #TODO REMOVE, need to store these as maps
    #TODO REMOVE, need to store these as maps
    #TODO REMOVE, need to store these as maps
    #fit_results.pop("norm_sys_maps")
    #TODO REMOVE, need to store these as maps
    #TODO REMOVE, need to store these as maps
    #TODO REMOVE, need to store these as maps

    # Store as JSON
    res_path = os.path.join(
        outdir,
        '%s__%dd__%s__hyperplane_fits.json' % (tag, dim, param_str)
    )
    to_file(fit_results,res_path)


def plot_hyperplane_fit_params(hyperplane_fits, names, binning, outdir=None,
                               tag=None):
    """Plot 2D distributions of fit parameters.

    Parameters
    ----------
    hyperplane_fits : dict
        fit results as returned by `hyperplane`
    names : list of strings
        lists of event groups/types whose fit results are to be plotted
    binning : MultiDimBinning
        binning as used in fits
    outdir : string
        path to output directory for plots
    tag : string
        identifier for fit results to put in filenames

    """
    import matplotlib as mpl
    mpl.use('pdf')
    from pisa.utils.plotter import Plotter

    sys_list = hyperplane_fits['sys_list']

    # there are no. of systematic params + 1 fit parameters
    for d in range(len(sys_list)+1):
        if d == 0:
            fit_param_id = 'offset'
        else:
            fit_param_id = 'slope_%s' % sys_list[d-1]
        maps = []
        for name in names:
            map_to_plot = Map(
                name='%s_raw' % name,
                hist=hyperplane_fits[name][..., d],
                binning=binning
            )
            maps.append(map_to_plot)
        maps = MapSet(maps)
        my_plotter = Plotter(
            stamp='',
            outdir=outdir,
            fmt='pdf',
            log=False,
            label=''
        )
        my_plotter.plot_2d_array(
            maps,
            fname='%s_%s_%ddfits'%(tag, fit_param_id, len(sys_list)),
        )


def plot_chisquare_values(chi2s, outfile, fit=True, fit_loc_scale=False,
                          bins=None, logy=True):
    """Fit and plot distribution of chi-square values.

    Parameters
    ----------
    chi2s : list
        flat list of chi-square values to fit and histogram
    outfile : str
        plot output file
    fit : bool
        whether to fit the list of chi-square values with
        a chi-square distribution
    fit_loc_scale : bool
        whether to allow the shape parameters "loc" and "scale"
        to float in the fit
    bins : sequence
        binning to employ for histogramming
    logy : bool
        employ a logarithmic y scale
    """
    import matplotlib as mpl
    mpl.use('agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    if fit_loc_scale and not fit:
        raise ValueError(
            'Use `fit_loc_scale` only when `fit` is True.'
        )

    # check there are some finite chi2 values to plot
    finite_mask = np.isfinite(chi2s)
    if finite_mask.size == 0 :
        logging.warning( # pylint: disable=logging-not-lazy
            'No finite chi2 values found, not plotting.'
        )
        return 

    logging.info('Histogramming %d chisquare values.' % len(chi2s)) # pylint: disable=logging-not-lazy
    if bins is None:
        bins = 100
    fig = plt.figure()
    n, bins, _ = plt.hist(
        chi2s[finite_mask], bins=bins, facecolor='firebrick', histtype='stepfilled',
        weights=np.ones_like(chi2s[finite_mask])/len(chi2s[finite_mask]),
        label='hyperplane residuals (%d)' % len(chi2s[finite_mask])
    )
    if fit:
        logging.info( # pylint: disable=logging-not-lazy
            'Performing chisquare fit with "loc" and "scale" %s.'
            % ('floating' if fit_loc_scale else 'fixed')
        )
        if fit_loc_scale:
            # fit for d.o.f., location and scale of distribution of values
            popt = stats.chi2.fit(chi2s[finite_mask])
            fit_rv = stats.chi2(*popt[:-2], loc=popt[-2], scale=popt[-1])
        else:
            popt = stats.chi2.fit(chi2s[finite_mask], floc=0, fscale=1)
            fit_rv = stats.chi2(df=popt[0])
        logging.info('Best fit parameters: %s' % list(popt)) # pylint: disable=logging-not-lazy
        # plot the binwise integrated fit pdf
        fit_cdf = fit_rv.cdf(bins[1:]) - fit_rv.cdf(bins[:-1])
        centers = (bins[1:] + bins[:-1])/2.
        plt.step(centers, fit_cdf, where='mid',
                 color='black', label=r'$\chi^2$ fit (%.2f d.o.f.)' % popt[0],
                 linestyle='solid')
    plt.xlabel(r'$\chi^2$', fontsize='x-large')
    plt.ylabel('AU', fontsize='x-large')
    if not logy:
        plt.ylim(0, 1.01*max(n))
    else:
        plt.yscale('log')
    plt.xlim(min(bins), max(bins))
    plt.legend(loc='best')
    plt.tight_layout()
    fig.savefig(outfile)


def plot_binwise_variations_with_fits(
        hyperplane_fits, sys_param_points,
        nominal_mapset, sys_mapsets, outdir, tag=None,
        subplot_kw=None, gridspec_kw=None, **fig_kw):
    """Bin-by-bin plots of count variations as function of
    systematic parameters together with projections of fit
    function along each dimension in each bin.

    Parameters
    ----------
    hyperplane_fits : dict
    sys_param_points : list
    nominal_mapset : MapSet
    sys_mapsets : list of MapSets
    outdir : str
    tag : str
    subplot_kw, gridspec_kw, fig_kw : dict
        keyword arguments passed on to plt.subplots

    """
    import matplotlib.pyplot as plt
    # normalise the systematics variations to the nominal distribution
    # with error propagation
    norm_sys_maps = norm_sys_distributions(nominal_mapset, sys_mapsets)
    binning = nominal_mapset[0].binning
    shape_map = binning.shape
    if not (len(shape_map) == 2 or len(shape_map) == 3):
        raise NotImplementedError(
            'Need 2d or 3d maps currently.'
        )
    nx, ny = shape_map[0], shape_map[1]
    try:
        nz = shape_map[2]
        is_3d = True
    except IndexError:
        nz = 0
        is_3d = False
    sys_list = hyperplane_fits['sys_list']
    if not len(sys_list) == 1:
        # TODO: allow for multi-sys. fits, but then project correctly
        logging.warn(
            'Plotting logic for more than 1 systematic not yet done. Returning.'
        )
        return
    sys_param_points = np.array(sys_param_points)
    for map_name, chan_norm_sys_maps in norm_sys_maps.items():
        logging.info( # pylint: disable=logging-not-lazy
            'Displaying binwise variations of "%s" maps.' % (map_name)
        )
        for i, sys_name in enumerate(sys_list):
            # these are all values of the param in question, irrespective
            # of the values of possible other parameters -> TODO: filter out
            # those for which others are at not at nominal
            sys_vals = sys_param_points[:, i]
            some_xs = np.linspace(min(sys_vals), max(sys_vals), 100)
            # make a new figure for each bin in the third dimension
            ziter = range(nz) if nz else [0] # pylint: disable=range-builtin-not-iterating
            for zind in ziter:
                zstr = '_%s_bin_%d_' % (binning.names[2], zind) if is_3d else '_'
                fig, ax2d = plt.subplots(
                    nrows=ny, ncols=nx, sharex='col', sharey='row',
                    subplot_kw=subplot_kw, gridspec_kw=gridspec_kw, **fig_kw
                )
                if is_3d:
                    title = map_name + ': %s' % zstr.replace('_', ' ').strip()
                    fig.suptitle(title, fontsize='xx-large')
                chan_norm_sys_maps_zind = chan_norm_sys_maps[:, :, zind]

                # each unique idx corresponds to one bin
                for idx in np.ndindex(*(shape_map[:2])):

                    # get the points to plot, handling NaN
                    y_values = unp.nominal_values(chan_norm_sys_maps_zind[idx])
                    y_sigma = unp.std_devs(chan_norm_sys_maps_zind[idx])
                    finite_mask = np.isfinite(y_values) & np.isfinite(y_sigma)

                    # plot the points
                    ax2d[idx[1], idx[0]].errorbar(
                        x=sys_vals[finite_mask], y=y_values[finite_mask], yerr=y_sigma[finite_mask], fmt='o',
                        mfc='firebrick', mec='firebrick', ecolor='firebrick'
                    )

                    # show points that were NaN
                    ax2d[idx[1], idx[0]].scatter(
                        sys_vals[~finite_mask], np.ones_like(sys_vals[~finite_mask]), marker='x',
                        c='black'
                    )

                    # TODO color/shade by chi2

                    # obtain the best fit parameters and plot the fit function
                    # for these
                    popt = np.array(hyperplane_fits[map_name][:, :, zind][idx])
                    y_opt = hyperplane_fun([some_xs], *popt)
                    ax2d[idx[1], idx[0]].plot(
                        some_xs, y_opt, color='firebrick', lw=2
                    )

                    # label the bin numbers
                    if idx[1] == len(ax2d) - 1:
                        ax2d[idx[1], idx[0]].set_xlabel(
                            '%s bin %d' % (binning.basenames[0], idx[0]),
                            fontsize='small', labelpad=10
                        )
                    if idx[0] == 0:
                        ax2d[idx[1], idx[0]].set_ylabel(
                            '%s bin %d' % (binning.basenames[1], idx[1]),
                            fontsize='small', labelpad=10
                        )

                    # formatting
                    ax2d[idx[1], idx[0]].grid(True)

                fig.text(0.5, 0.04, sys_name, ha='center', fontsize='xx-large')
                fig.text(0.04, 0.5, 'normalised count', va='center',
                         rotation='vertical', fontsize='xx-large')
                fname = ('%s_%s_%s%sbinwise_hyperplane.png'
                         % (tag, sys_name, map_name, zstr))
                fig.savefig(os.path.join(outdir, fname))


def main():
    """Main function to run discrete systematics fits from command line and
    possibly plot the results.
    """
    args = parse_args()
    set_verbosity(args.v)

#    nom_ms, sys_points, nom_sys_points, sys_ms, binning, fits, chi2s = hyperplane(
#        fit_cfg=args.fit_cfg,
#    )

    # Read in data and fit hyperplanes to it
    input_data,fit_results = hyperplane(fit_cfg=args.fit_cfg)

    # Save to disk
    save_hyperplane_fits(input_data=input_data,fit_results=fit_results,outdir=args.outdir,tag=args.tag)

    '''
    save_hyperplane_fits(
        hyperplane_fits=fits,
        chi2s=chi2s,
        sys_param_points=sys_points,
        nom_sys_param_points=nom_sys_points,
        nominal_mapset=nom_ms,
        sys_mapsets=sys_ms,
        outdir=args.outdir,
        tag=args.tag
    )
    '''

    if args.plot:
        plot_hyperplane_fit_params(
            hyperplane_fits=fits,
            names=nom_ms.names,
            binning=binning,
            outdir=args.outdir,
            tag=args.tag
        )
        plot_binwise_variations_with_fits(
            hyperplane_fits=fits,
            nominal_mapset=nom_ms,
            sys_param_points=sys_points,
            sys_mapsets=sys_ms,
            outdir=args.outdir,
            tag=args.tag,
            figsize=(16, 16)
        )
        plot_chisquare_values(
            chi2s=chi2s,
            outfile='%s/%s_%dd_%s_chi2s.png'
            % (args.outdir, args.tag, len(fits['sys_list']), '_'.join(fits['sys_list']))
        )


if __name__ == '__main__':
    main()
