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

#TODO Need to replace `fit_discrete_sys_ns.py` with this...


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
    input_data : OrderedDict
        container with the processed input data including MapSets 
        resulting from each input pipelines
    """

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


def norm_sys_distributions(input_data):

    """Normalises systematics mapsets to the nominal mapset,
    performing error propagation.

    Parameters
    ----------
    input_data : dict
        The data container returned by `make_discrete_sys_distributions`.
        Note that this is modified to add the normalised distrbutions 
        by this function.

    Returns
    -------
    Nothing is returned, instead `input_data` is modified)

    """

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


def fit_discrete_sys_distributions(input_data,p0=None) :

    """Fits a hyperplane to MapSets generated at given systematics parameters
    values.

    Parameters
    ----------
    input_data : OrderedDict
        The data container returned by `make_discrete_sys_distributions` 
        and modified by `norm_sys_distributions`.
    p0 : list or dict
        Initial guess list (same initial guess for all maps) or dictionary
        (keys have to correspond to event groups/channels in maps)
        with one offset and len(sys_list) slopes. Default is list of ones.

    Returns
    -------
    fit_results : OrderedDict
        Container of the hyerplane fit results + supporting data
    """

    #
    # Prepare a few things before fitting
    #

    # prepare an output data container
    fit_results = OrderedDict()
    fit_results["hyperplanes"] = OrderedDict()

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
                for point_idx in range(n_datasets): # Loop over datasets
                    point = sys_param_points[point_idx,:] # Get param values for this dataset
                    predicted = hyperplane_fun(point,*popt) # Predict counts in this bin accoridng to hyperplane for this dataset
                    observed = y_values[point_idx]
                    sigma = y_sigma[point_idx]
                    chi2 = ((predicted - observed)/sigma)**2 # Calc chi2 #TODO Is this correct?
                    chi2_idx = tuple(list(idx)+[point_idx])
                    fit_results["hyperplanes"][map_name]["chi2s"][chi2_idx] = chi2

            else:

                # if here, no uncertainties are available for this bin
                # note that cannot calculate chi2 without uncertainties

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


                else :

                    # case 3: no data in this bin
                    # this is the worst case, where there are no central values or errors.
                    # most likely this came about because this bin is empty, which is not 
                    # necessarily an error.

                    # Store NaN for fit params and chi2
                    popt = np.full_like(p0[map_name],np.NaN) 
                    pcov = np.NaN #TODO Shape?

            # store the results for this bin
            # note that chi2 is already stored above
            fit_results["hyperplanes"][map_name]["fit_params"][idx] = popt
            fit_results["hyperplanes"][map_name]["cov_matrices"][idx] = pcov #TODO need to np.copyto?
            fit_results["hyperplanes"][map_name]["finite_mask"][idx] = finite_mask

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

    input_data = make_discrete_sys_distributions(fit_cfg=fit_cfg)
    fit_results = fit_discrete_sys_distributions(input_data=input_data)

    return input_data,fit_results


def save_hyperplane_fits(input_data,fit_results,outdir,tag):
    """Store discrete systematics fits and chi-square values to a specified
    output location, with results identified by a tag.

    Parameters
    ----------
    input_data : dict
        input data container returned by `hyperplane` function
    fit_results : dict
        fit results data container returned by `hyperplane` function
    outdir : string
        output directory
    tag : string
        identifier for filenames holding fit results
    """

    # Get some strings to use when naming
    dim = len(input_data["param_names"])
    param_str = '_'.join(input_data["param_names"])

    # Store as JSON
    res_path = os.path.join(
        outdir,
        '%s__%dd__%s__hyperplane_fits.json' % (tag, dim, param_str)
    )
    to_file(fit_results,res_path)




def main():
    """Main function to run discrete systematics fits from command line and
    possibly plot the results.
    """
    args = parse_args()
    set_verbosity(args.v)

    # Read in data and fit hyperplanes to it
    input_data,fit_results = hyperplane(fit_cfg=args.fit_cfg)

    # Save to disk
    save_hyperplane_fits(input_data=input_data,fit_results=fit_results,outdir=args.outdir,tag=args.tag)


if __name__ == '__main__':
    main()
