#!/usr/bin/env python
"""
Hyperplane fitting scriot

Produce fit results for sets of disctrete systematics (i.e. for example
several simulations for different DOM efficiencies)

The parameters and settings going into the fit are given by an external cfg
file (fit settings).

n-dimensional MapSets are supported to be fitted with m-dimesnional, linear
hyperplanes functions
"""
from __future__ import absolute_import, division

from argparse import ArgumentParser
from uncertainties import unumpy as unp

import numpy as np
from scipy.optimize import curve_fit

from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.utils.config_parser import parse_quantity, parse_string_literal
from pisa.utils.fileio import from_file, to_file
from pisa.utils.log import logging, set_verbosity


__all__ = ['parse_args', 'main']


def parse_args():
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument(
        '-p', '--pipeline', type=str,
        metavar='configfile', required=True,
        help='Pipeline config for the generation of templates'
    )
    parser.add_argument(
        '-f', '--fit-settings', type=str,
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


def fit_fun(x, *p):
    # x: array of points
    # p: array of params, with first being the offset followed by the slopes
    ret_val = p[0]
    for xval, pval in zip(x, p[1:]):
        ret_val += xval*pval
    return ret_val


def collect_discrete_sys_distributions(pipeline, fit_settings):
    fit_cfg = from_file(fit_settings)
    sys_list = fit_cfg.get('general', 'sys_list').replace(' ', '').split(',')
    stop_idx = fit_cfg.getint('general', 'stop_after_stage')
    combine_regex = fit_cfg.get('general', 'combine_regex')
    combine_regex = combine_regex.replace(' ', '').split(',')
    print combine_regex
    sys_param_points = []
    sys_mapsets = []
    nominal_mapset = None
    # retrive sets:
    for section in fit_cfg.sections():
        if section == 'general':
            continue
        elif section.startswith('nominal_set:') or section.startswith('sys_set:'):
            sys_param_point = [float(x) for x in section.split(':')[1].split(',')]
            # Instantiate template maker
            distribution_maker = DistributionMaker(pipeline)
            # retreive settings
            for key, val in fit_cfg.items(section):
                if key.startswith('param.'):
                    _, pname = key.split('.')
                    param = distribution_maker.params[pname]
                    try:
                        value = parse_quantity(val)
                        param.value = value.n * value.units
                    except ValueError:
                        value = parse_string_literal(val)
                        param.value = value
                    param.set_nominal_to_current_value()
                    distribution_maker.update_params(param)
            # retreive maps
            mapset = distribution_maker.get_outputs(idx=stop_idx)[0]
            if combine_regex:
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

    if not nominal_mapset:
        raise ValueError(
            'Could not find a nominal discrete systematics set in fit cfg.'
            ' There must be exactly one.'
        )

    return nominal_mapset, sys_list, sys_param_points, sys_mapsets


def fit_discrete_sys_distributions(
        nominal_mapset, sys_list, sys_param_points, sys_mapsets
    ):
    out_names = sorted(nominal_mapset.names)
    for mapset in sys_mapsets:
        if not sorted(mapset.names) == out_names:
            raise ValueError(
                'The output names of at least two mapsets do not agree!'
            )

    sys_param_points = np.array(sys_param_points).T
    # for every bin in the map we need to store 1 + n terms for n systematics,
    # i.e. 1 offset and n slopes
    n_params = 1 + sys_param_points.shape[0]
    logging.info('Number of params to fit: %d' % n_params)

    # do it for every map in the MapSet
    outputs = {}
    errors = {}
    chi2s = []
    for map_name in out_names:
        logging.info('Fitting "%s" maps.' % map_name)
        nominal_hist = unp.nominal_values(nominal_mapset[map_name].hist)
        sys_hists = []
        for sys_mapset in sys_mapsets:
            # normalize to nominal:
            sys_hist = sys_mapset[map_name].hist/nominal_hist
            print np.max(sys_hist)
            sys_hists.append(sys_hist)

        # put them into an array
        sys_hists = np.array(sys_hists)
        # put that to the last axis
        sys_hists = np.rollaxis(sys_hists, 0, len(sys_hists.shape))

        binning = nominal_mapset[map_name].binning

        shape_output = [d.num_bins for d in binning] + [n_params]
        shape_map = [d.num_bins for d in binning]

        outputs[map_name] = np.ones(shape_output)
        errors[map_name] = np.ones(shape_output)

        for idx in np.ndindex(*shape_map):
            y_values = unp.nominal_values(sys_hists[idx])
            y_sigma = unp.std_devs(sys_hists[idx])
            if np.any(y_sigma):
                popt, pcov = curve_fit(fit_fun, sys_param_points, y_values,
                                       sigma=y_sigma, p0=np.ones(n_params))

                #calculate chi2 values:
                for point_idx in range(sys_param_points.shape[1]):
                    point = sys_param_points[:, point_idx]
                    predicted = fit_fun(point, *popt)
                    observed = y_values[point_idx]
                    sigma = y_sigma[point_idx]
                    chi2 = ((predicted - observed)/sigma)**2
                    chi2s.append(chi2)

            else:
                popt, pcov = curve_fit(fit_fun, sys_param_points, y_values,
                                       p0=np.ones(n_params))
            perr = np.sqrt(np.diag(pcov))
            for k, p in enumerate(popt):
                outputs[map_name][idx][k] = p
                errors[map_name][idx][k] = perr[k]

    # Save the raw ones anyway
    outputs['sys_list'] = sys_list
    outputs['map_names'] = nominal_mapset.names
    outputs['binning_hash'] = binning.hash

    return outputs, chi2s, binning

def save_hyperplane_fits(hyperplane_fits, chi2s, outdir, tag=None):
    tag = '_' if not tag else '_%s_' % tag
    to_file(hyperplane_fits, '%s/nd_sysfits%sraw.json' % (outdir, tag))
    chi2s = np.array(chi2s)
    np.save('%s/nd_sysfits_%s_raw_chi2s' % (outdir, tag), chi2s)


def plot_hyperplane_fits(hyperplane_fits, names, n_params, binning,
                         outdir=None, tag=None):
    import matplotlib as mpl
    mpl.use('pdf')
    from pisa.utils.plotter import Plotter

    for d in range(n_params):
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
            fname='%s_%s_raw_ndfits'%(tag, d),
        )


def hyperplane(fit_settings, pipeline, set_param=None):

    nominal_mapset, sys_list, sys_param_points, sys_mapsets = collect_discrete_sys_distributions(
        fit_settings=fit_settings,
        pipeline=pipeline
    )

    hyperplane_fits, chi2s, binning = fit_discrete_sys_distributions(
        nominal_mapset=nominal_mapset,
        sys_list=sys_list,
        sys_param_points=sys_param_points,
        sys_mapsets=sys_mapsets
    )
    return nominal_mapset, sys_list, binning, hyperplane_fits, chi2s


def main():
    args = parse_args()
    set_verbosity(args.v)

    nominal_mapset, sys_list, binning, hyperplane_fits, chi2s = hyperplane(
        fit_settings=args.fit_settings,
        pipeline=args.pipeline
    )
    save_hyperplane_fits(
        hyperplane_fits=hyperplane_fits,
        chi2s=chi2s,
        outdir=args.outdir,
        tag=args.tag
    )
    if args.plot:
        plot_hyperplane_fits(
            hyperplane_fits=hyperplane_fits,
            names=nominal_mapset.names,
            n_params=1+len(sys_list),
            binning=binning,
            outdir=args.outdir,
            tag=args.tag
        )

if __name__ == '__main__':
    main()
