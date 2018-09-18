#!/usr/bin/env python

"""
Fit 1D polynomials to discrete systematics datasets

Produce fit results for sets of disctrete systematics (i.e. for example
several simulations for different DOM efficiencies)

The parameters and settings going into the fit are given by an external cfg
file (fit settings).

n-dimensional MapSets are supported to be fitted with m-th order 1-dimensional
polynomials (one per systematic) that can be forced to go through the nominal
data point
"""


from __future__ import absolute_import

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

__author__ = 'P. Eller'

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
        '-sp', '--set-param', type=str, default=None,
        help='Set a param to a certain value.',
        action='append'
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


    for sys in sys_list:
        # Parse info for given systematic
        nominal = cfg.getfloat(sys, 'nominal')
        degree = cfg.getint(sys, 'degree')
        force_through_nominal = cfg.getboolean(sys, 'force_through_nominal')
        runs = eval(cfg.get(sys, 'runs'))
        #print "runs ", runs
        smooth = cfg.get(sys, 'smooth')

        x_values = np.array(sorted(runs))

        # Build fit function
        if force_through_nominal:
            function = "lambda x, *p: np.polynomial.polynomial.polyval(x, [1.] + list(p))"
        else:
            function = "lambda x, *p: np.polynomial.polynomial.polyval(x, list(p))"
            # Add free parameter for constant term
            degree += 1
        fit_fun = eval(function)

        # Instantiate template maker
        template_maker = Pipeline(args.template_settings)

        if not args.set_param == '':
            for one_set_param in args.set_param:
                p_name, value = one_set_param.split("=")
                #print "p_name,value= ", p_name, " ", value
                value = parse_quantity(value)
                value = value.n * value.units
                param = template_maker.params[p_name]
                #print "old ", p_name, "value = ", param.value
                param.value = value
                #print "new ", p_name, "value = ", param.value
                template_maker.update_params(param)

        inputs = {}
        map_names = None
        # Get sys templates
        for run in runs:
            for key, val in cfg.items('%s:%s'%(sys, run)):
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
            template = template_maker.get_outputs(idx=stop_idx)
            if map_names is None: map_names = [m.name for m in template]
            inputs[run] = {}
            for m in template:
                inputs[run][m.name] = m.hist

        # Numpy acrobatics:
        arrays = {}
        for name in map_names:
            arrays[name] = []
            for x in x_values:
                arrays[name].append(
                    inputs[x][name] / unp.nominal_values(inputs[nominal][name])
                )
            a = np.array(arrays[name])
            arrays[name] = np.rollaxis(a, 0, len(a.shape))

        # Shift to get deltas
        x_values -= nominal

        # Binning object (assuming they're all the same)
        binning = template.maps[0].binning

        shape = [d.num_bins for d in binning] + [degree]
        shape_small = [d.num_bins for d in binning]

        outputs = {}
        errors = {}
        for name in map_names:
            # Now actualy perform some fits
            outputs[name] = np.ones(shape)
            errors[name] = np.ones(shape)


            for idx in np.ndindex(*shape_small):
                y_values = unp.nominal_values(arrays[name][idx])
                y_sigma = unp.std_devs(arrays[name][idx])
                if np.any(y_sigma):
                    popt, pcov = curve_fit(fit_fun, x_values, y_values,
                                           sigma=y_sigma, p0=np.ones(degree))
                else:
                    popt, pcov = curve_fit(fit_fun, x_values, y_values,
                                           p0=np.ones(degree))
                perr = np.sqrt(np.diag(pcov))
                for k, p in enumerate(popt):
                    outputs[name][idx][k] = p
                    errors[name][idx][k] = perr[k]

                # TODO(philippeller): the below block of code will fail

                # Maybe plot
                #if args.plot:
                #    fig_num = i + nx * j
                #    if fig_num == 0:
                #        fig = plt.figure(num=1, figsize=( 4*nx, 4*ny))
                #    subplot_idx = nx*(ny-1-j)+ i + 1
                #    plt.subplot(ny, nx, subplot_idx)
                #    #plt.snameter(x_values, y_values, color=plt_colors[name])
                #    plt.gca().errorbar(x_values, y_values, yerr=y_sigma,
                #                       fmt='o', color=plt_colors[name],
                #                       ecolor=plt_colors[name],
                #                       mec=plt_colors[name])
                #    # Plot nominal point again in black
                #    plt.snameter([0.0], [1.0], color='k')
                #    f_values = fit_fun(x_values, *popt)
                #    fun_plot, = plt.plot(x_values, f_values,
                #            color=plt_colors[name])
                #    plt.ylim(np.min(unp.nominal_values(arrays[name]))*0.9,
                #             np.max(unp.nominal_values(arrays[name]))*1.1)
                #    if i > 0:
                #        plt.setp(plt.gca().get_yticklabels(), visible=False)
                #    if j > 0:
                #        plt.setp(plt.gca().get_xticklabels(), visible=False)

        if smooth == 'gauss':
            for name in map_names:
                for d in range(degree):
                    outputs[name][...,d] = gaussian_filter(outputs[name][...,d],sigma=1)

        if smooth == 'gauss_pid':
            for name in map_names:
                split_idx = binning.names.index('pid')
                tot = len(binning)-1
                for d in range(degree):
                    for p in range(len(binning['pid'])):
                        outputs[name][...,p,d] = gaussian_filter(
                            np.swapaxes(outputs[name], split_idx, tot)[...,p,d],
                            sigma=1
                        )
                outputs[name] = np.swapaxes(outputs[name], split_idx, tot)

        # Save the raw ones anyway
        outputs['pname'] = sys
        outputs['nominal'] = nominal
        outputs['function'] = function
        outputs['map_names'] = map_names
        outputs['binning_hash'] = binning.hash
        to_file(outputs, '%s/%s_sysfits_%s_%s.json'%(args.out_dir, sys,
                                                     args.tag, smooth))

        if args.plot:
            for d in range(degree):
                maps = []
                for name in map_names:
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
                    fname='%s_%s_%s_%s'%(sys, args.tag, d, smooth),
                )


if __name__ == '__main__':
    main()
