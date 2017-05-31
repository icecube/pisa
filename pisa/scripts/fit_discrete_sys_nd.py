#!/usr/bin/env python
"""
Produce fit results for sets of disctrete systematics (i.e. for example
several simulations for different DOM efficiencies)

The parameters and settings going into the fit are given by an external cfg
file (fit settings).

n-dimensional MapSets are supported to be fitted with m-dimesnional
polynomials, that can either be forced to go through the nominal data point or
not
"""


from argparse import ArgumentParser
from uncertainties import unumpy as unp

import numpy as np
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit

from pisa.core.map import Map, MapSet
from pisa.core.pipeline import Pipeline
from pisa.utils.config_parser import parse_quantity, parse_string_literal
from pisa.utils.fileio import from_file, to_file
from pisa.utils.log import set_verbosity


__all__ = ['parse_args', 'main']


def parse_args():
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument(
        '-t', '--template-settings', type=str,
        metavar='configfile', required=True,
        help='settings for the generation of templates'
    )
    parser.add_argument(
        '-f', '--fit-settings', type=str,
        metavar='configfile', required=True,
        help='settings for the generation of templates'
    )
    parser.add_argument(
        '-sp', '--set-param', type=str, default='',
        help='Set a param to a certain value.'
    )
    parser.add_argument(
        '--tag', type=str, default='deepcore',
        help='Tag for the filename'
    )
    parser.add_argument(
        '-o', '--out-dir', type=str, required=True,
        help='Set output directory'
    )
    parser.add_argument(
        '-p', '--plot', action='store_true',
        help='plot'
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_verbosity(args.v)

    if args.plot:
        import matplotlib as mpl
        mpl.use('pdf')
        import matplotlib.pyplot as plt
        from pisa.utils.plotter import Plotter

    cfg = from_file(args.fit_settings)
    sys_list = cfg.get('general', 'sys_list').replace(' ', '').split(',')
    stop_idx = cfg.getint('general', 'stop_after_stage')

    # Instantiate template maker
    template_maker = Pipeline(args.template_settings)

    if not args.set_param == '':
        p_name, value = args.set_param.split("=")
        print "p_name,value= ", p_name, " ", value
        value = parse_quantity(value)
        value = value.n * value.units
        print "value ", value
        test = template_maker.params[p_name]
        print "test.value = ", test.value
        test.value = value
        print "test.value = ", test.value
        template_maker.update_params(test)

    sys_parameter_points = []
    sys_mapsets = []
    nominal_parameter_point = None
    nominal_mapset = None

    # retrive sets:
    for section in cfg.sections():
        if section == 'general':
            continue
        elif section.startswith('nominal_set:') or section.startswith('sys_set:'):
            parameter_point = [float(x) for x in section.split(':')[1].split(',')]
            # retreive settings
            for key, val in cfg.items(section):
                if key.startswith('param.'):
                    _, pname = key.split('.')
                    param = template_maker.params[pname]
                    try:
                        value = parse_quantity(val)
                        param.value = value.n * value.units
                    except ValueError:
                        value = parse_string_literal(val)
                        param.value = value
                    param.set_nominal_to_current_value()
                    template_maker.update_params(param)
            # Retreive maps
            mapset = template_maker.get_outputs(idx=stop_idx)
        else:
            raise ValueError('Addidional, unrecognized section in cfg file: %s'%section)

        # add them to the right place
        if section.startswith('nominal_set:'):
            assert(nominal_mapset is None), 'multiple nominal sets in cfg'
            nominal_mapset = mapset
            nominal_parameter_point = parameter_point
            # but the nominal_set also acts a a sys_set and will be treated no differently
            sys_mapsets.append(mapset)
            sys_parameter_points.append(parameter_point)
        elif section.startswith('sys_set:'):
            sys_mapsets.append(mapset)
            sys_parameter_points.append(parameter_point)

    assert(nominal_mapset is not None), 'no nominal set in cfg'
    
    sys_parameter_points = np.array(sys_parameter_points).T
    # for every bin in the map we need to store 1 + n terms for n systematics, i.e. 1 ofset and n slopes
    n_params = 1 + sys_parameter_points.shape[0]
    print n_params

    def fit_fun(X,*P):
        # X: array of points
        # P: array of params, with first being the offset followe by the slopes
        ret_val = P[0]
        for x,p in zip(X,P[1:]):
            ret_val += x*p
        return ret_val

    #do it for every map in the MapSet
    outputs = {}
    errors = {}
    for map_name in nominal_mapset.names:
        print 'working on %s'%map_name
        nominal_hist = nominal_mapset[map_name].hist
        sys_hists = []
        for sys_mapset in sys_mapsets:
            # normalize to nominal:
            sys_hists.append(sys_mapset[map_name].hist/unp.nominal_values(nominal_hist))

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
                popt, pcov = curve_fit(fit_fun, sys_parameter_points, y_values,
                                       sigma=y_sigma, p0=np.ones(n_params))
            else:
                popt, pcov = curve_fit(fit_fun, sys_parameter_points, y_values,
                                       p0=np.ones(n_params))
            perr = np.sqrt(np.diag(pcov))
            for k, p in enumerate(popt):
                outputs[map_name][idx][k] = p
                errors[map_name][idx][k] = perr[k]

    # Save the raw ones anyway
    outputs['sys_list'] = sys_list
    outputs['map_names'] = nominal_mapset.names
    outputs['binning_hash'] = binning.hash
    to_file(outputs, '%s/nd_sysfits_%s_raw.json'%(args.out_dir,
                                                 args.tag))

    if args.plot:
        for d in range(n_params):
            maps = []
            for name in nominal_mapset.names:
                maps.append(Map(name='%s_raw'%name, hist=outputs[name][...,d],
                                binning=binning))
            maps = MapSet(maps)
            my_plotter = Plotter(
                stamp='',
                outdir=args.out_dir,
                fmt='pdf',
                log=False,
                label=''
            )
            my_plotter.plot_2d_array(
                maps,
                fname='%s_%s_raw_ndfits'%(args.tag, d),
            )


if __name__ == '__main__':
    main()
