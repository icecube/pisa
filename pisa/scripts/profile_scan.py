#!/usr/bin/env python

"""
Profile scan
"""


from __future__ import absolute_import

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import copy
from os.path import expanduser, expandvars, isfile

from pisa.analysis.analysis import Analysis
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.detectors import Detectors
from pisa.utils.fileio import from_file, to_file
from pisa.utils.log import logging, set_verbosity


__all__ = ['profile_scan', 'parse_args', 'main']

__author__ = 'T. Ehrhardt'

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


def profile_scan(data_settings, template_settings, param_names, steps,
                 only_points, no_outer, data_param_selections,
                 hypo_param_selections, profile, outfile, minimizer_settings,
                 metric, debug_mode, shared_params):
    """Perform a profile scan.

    Parameters
    ----------
    data_settings
    template_settings
    param_names
    steps
    only_points
    no_outer
    data_param_selections
    hypo_param_selections
    profile
    outfile
    minimizer_settings
    metric
    debug_mode
    shared_params

    Returns
    -------
    results
    analysis

    """
    outfile = expanduser(expandvars(outfile))
    if isfile(outfile):
        raise IOError('`outfile` "{}" already exists!'.format(outfile))

    minimizer_settings = from_file(minimizer_settings)

    hypo_maker = Detectors(template_settings, shared_params=shared_params)

    if data_settings is None:
        if (data_param_selections is None
                or data_param_selections == hypo_param_selections):
            data_maker = hypo_maker
        else:
            data_maker = copy(hypo_maker)
            data_maker.select_params(data_param_selections)
    else:
        data_maker = Detectors(data_settings, shared_params=shared_params)
        data_maker.select_params(data_param_selections)

    data_dist = data_maker.get_outputs(return_sum=True)

    analysis = Analysis()
    results = analysis.scan(
        data_dist=data_dist,
        hypo_maker=hypo_maker,
        hypo_param_selections=hypo_param_selections,
        metric=metric,
        param_names=param_names,
        steps=steps,
        only_points=only_points,
        outer=not no_outer,
        profile=profile,
        minimizer_settings=minimizer_settings,
        outfile=outfile,
        debug_mode=debug_mode
    )
    to_file(results, outfile)
    logging.info("Done.")

    return results, analysis


def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data-settings', type=str,
        metavar='CONFIGFILE', default=None,
        help='''Settings for the generation of "data" distributions; repeat
        this argument to specify multiple pipelines. If omitted, the same
        settings as specified for --template-settings are used to generate data
        distributions.'''
    )
    parser.add_argument(
        '--template-settings',
        metavar='CONFIGFILE', required=True, action='append',
        help='''Settings for generating template distributions; repeat
        this option to define multiple pipelines.'''
    )
    parser.add_argument(
        '-sp', '--shared-params', type=str, default=None,
        action='append',
        help='''Shared parameters for multi det analysis (repeat for multiple).
        These parameters have to be the same in all detectors, that contain the param.'''
    )
    parser.add_argument(
        '--param-names', type=str, nargs='+', required=True,
        help='''Provide a list of parameter names to scan.'''
    )
    parser.add_argument(
        '--steps', type=int, nargs='+', required=True,
        help='''Provide a number of steps for each parameter (in the same order
        as the parameter names).'''
    )
    parser.add_argument(
        '--only-points', type=int, nargs='+', required=False,
        help='''Provide a point or ranges of points to be scanned specified by
        one or an even number of integer numbers (might be useful if the
        analysis is to be split up into several smaller jobs). 0-indexing is
        assumed. Isn't applied to any single parameter, but to the whole set of
        points (with steps x steps - 1 corresponding to the last).'''
    )
    parser.add_argument(
        '--no-outer', action='store_true',
        help='''Do not scan points as outer product of inner sequences.'''
    )
    parser.add_argument(
        '--data-param-selections', type=str, required=False,
        help='''Selection of params to use in order to generate the data
        distributions.'''
    )
    parser.add_argument(
        '--hypo-param-selections', type=str, nargs='+', required=False,
        help='''Selection of params to use in order to generate the
        hypothesised Asimov distributions.'''
    )
    parser.add_argument(
        '--profile', action='store_true',
        help='''Run profile scan, i.e. optimise over remaining free
        parameters.'''
    )
    parser.add_argument(
        '--outfile', metavar='FILE',
        type=str, action='store', default='profile_scan.json',
        help='file to store the output'
    )
    parser.add_argument(
        '--minimizer-settings', type=str,
        metavar='JSONFILE', required=True,
        help='''Settings related to the minimizer used in the LLR analysis.'''
    )
    parser.add_argument(
        '--metric', type=str,
        choices=['llh', 'chi2', 'conv_llh', 'mod_chi2'], required=True,
        help='''Settings related to the minimizer used in the LLR analysis.'''
    )
    parser.add_argument(
        '--debug-mode', type=int, choices=[0, 1, 2], required=False, default=1,
        help='''How much information to keep in the output file. 0 for only
        essentials for a physics analysis, 1 for more minimizer history, 2 for
        whatever can be recorded.'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    args = parser.parse_args()
    kwargs = vars(args)
    set_verbosity(kwargs.pop('v'))

    return kwargs


def main():
    """Run profile_scan with args from command line"""
    return profile_scan(**parse_args())


if __name__ == '__main__':
    results, analysis = main()
