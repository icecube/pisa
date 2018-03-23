#!/usr/bin/env python
"""
Hyperplane fitting scriot

Produce fit results for sets of disctrete systematics (i.e. for example
several simulations for different DOM efficiencies)

The parameters and settings going into the fit are given by an external cfg
file (fit config).

n-dimensional MapSets are supported to be fitted with m-dimensional, linear
hyperplanes functions
"""
from __future__ import absolute_import, division

from argparse import ArgumentParser
from uncertainties import unumpy as unp

import numpy as np
from scipy.optimize import curve_fit

from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.utils.fileio import from_file, to_file
from pisa.utils.log import logging, set_verbosity


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
        '-sp', '--set-param', type=str, default=None,
        help='Set a param to a certain value.',
        action='append'
    )
    parser.add_argument(
        '--tag', type=str, default='deepcore',
        help='Tag for the filename'
    )
    parser.add_argument(
        '-o', '--outdir', type=str, required=True,
        help='Set output directory'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='plot'
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
    p : list
        list of fit function parameters values
        (one offset, n slopes, where n is the number of systematic parameters)

    Returns
    -------
    fun : list
        function value vector (one value in each systematics dimension)

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

    return nominal_mapset, sys_list, sys_param_points, sys_mapsets


def fit_discrete_sys_distributions(
        nominal_mapset, sys_list, sys_param_points, sys_mapsets
    ):
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

    Returns
    -------
    outputs : dict
        stores fit results (fit parameters for each map name, the names of
        the systematic parameters, the hash of the binning)
    chi2s : list
        fit chi-square values
    binning : MultiDimBinning
        binning of all maps

    """
    out_names = sorted(nominal_mapset.names)
    for mapset in sys_mapsets:
        if not sorted(mapset.names) == out_names:
            raise ValueError(
                'The output names of at least two mapsets do not agree!'
            )
    # transpose to get successive values of the same param in the second dim.
    sys_param_points = np.array(sys_param_points).T
    # for every bin in the map we need to store 1 + n terms for n systematics,
    # i.e. 1 offset and n slopes
    n_params = 1 + sys_param_points.shape[0]
    logging.info('Number of params to fit: %d' % n_params) # pylint: disable=logging-not-lazy

    # do it for every map in the MapSet
    outputs = {}
    errors = {}
    chi2s = []
    binning = nominal_mapset[0].binning
    binning_hash = binning.hash
    for map_name in out_names:
        logging.info('Fitting "%s" maps.' % map_name) # pylint: disable=logging-not-lazy
        nominal_hist = unp.nominal_values(nominal_mapset[map_name].hist)
        sys_hists = []
        for sys_mapset in sys_mapsets:
            # normalize to nominal:
            sys_hist = sys_mapset[map_name].hist/nominal_hist
            sys_hists.append(sys_hist)

        # put them into an array
        sys_hists = np.array(sys_hists)
        # put that to the last axis
        sys_hists = np.rollaxis(sys_hists, 0, len(sys_hists.shape))

        this_binning = nominal_mapset[map_name].binning
        if not this_binning == binning:
            # sanity check
            raise ValueError(
                'There seem to be different binnings for different maps.'
                ' This should not be happening.'
            )

        shape_output = [d.num_bins for d in binning] + [n_params]
        shape_map = [d.num_bins for d in binning]

        outputs[map_name] = np.ones(shape_output)
        errors[map_name] = np.ones(shape_output)

        for idx in np.ndindex(*shape_map):
            y_values = unp.nominal_values(sys_hists[idx])
            y_sigma = unp.std_devs(sys_hists[idx])
            if np.any(y_sigma):
                popt, pcov = curve_fit(hyperplane_fun, sys_param_points, y_values,
                                       sigma=y_sigma, p0=np.ones(n_params))

                # calculate chi2 values:
                for point_idx in range(sys_param_points.shape[1]):
                    point = sys_param_points[:, point_idx]
                    predicted = hyperplane_fun(point, *popt)
                    observed = y_values[point_idx]
                    sigma = y_sigma[point_idx]
                    chi2 = ((predicted - observed)/sigma)**2
                    chi2s.append(chi2)

            else:
                popt, pcov = curve_fit(hyperplane_fun, sys_param_points, y_values,
                                       p0=np.ones(n_params))
            perr = np.sqrt(np.diag(pcov))
            for k, p in enumerate(popt):
                outputs[map_name][idx][k] = p
                errors[map_name][idx][k] = perr[k]

    # save the raw ones anyway
    outputs['sys_list'] = sys_list
    outputs['map_names'] = nominal_mapset.names
    outputs['binning_hash'] = binning_hash

    return outputs, chi2s, binning


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

    nominal_mapset, sys_list, sys_param_points, sys_mapsets = make_discrete_sys_distributions(
        fit_cfg=fit_cfg
    )

    hyperplane_fits, chi2s, binning = fit_discrete_sys_distributions(
        nominal_mapset=nominal_mapset,
        sys_list=sys_list,
        sys_param_points=sys_param_points,
        sys_mapsets=sys_mapsets
    )
    return nominal_mapset, sys_param_points, sys_mapsets, binning, hyperplane_fits, chi2s


def save_hyperplane_fits(hyperplane_fits, chi2s, outdir, tag=None):
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
    tag = '_' if not tag else '_%s_' % tag
    to_file(hyperplane_fits, '%s/nd_sysfits%sraw.json' % (outdir, tag))
    chi2s = np.array(chi2s)
    np.save('%s/nd_sysfits_%s_raw_chi2s' % (outdir, tag), chi2s)


def plot_hyperplane_fits(hyperplane_fits, names, binning, outdir=None, tag=None):
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
            fname='%s_%s_raw_ndfits'%(tag, fit_param_id),
        )


def main():
    """Main function to run discrete systematics fits from command line and
    possibly plot the results.
    """
    args = parse_args()
    set_verbosity(args.v)

    nom_ms, sys_points, sys_ms, binning, fits, chi2s = hyperplane( # pylint: disable=unused-variable
        fit_cfg=args.fit_cfg,
    )
    save_hyperplane_fits(
        hyperplane_fits=fits,
        chi2s=chi2s,
        outdir=args.outdir,
        tag=args.tag
    )
    if args.plot:
        plot_hyperplane_fits(
            hyperplane_fits=fits,
            names=nom_ms.names,
            binning=binning,
            outdir=args.outdir,
            tag=args.tag
        )

if __name__ == '__main__':
    main()
