#!/usr/bin/env python

"""
Performs 1D scans over all of the systematics in a pipeline (or multiple
pipelines) and saves the output. This is to check the their likelihood spaces.
"""


from __future__ import absolute_import

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
from os.path import expanduser, expandvars, isfile, join

from pisa.analysis.analysis import Analysis
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.fileio import from_file, to_file, mkdir
from pisa.utils.log import logging, set_verbosity


__author__ = 'S. Wren, T. Ehrhardt, J. Lanfranchi'

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
    """Parse command line arguments and return as a dict.

    Returns
    -------
    kwargs

    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--template-settings',
        metavar='CONFIGFILE', required=True, action='append',
        help='''Settings for generating template distributions; repeat
        this option to define multiple pipelines.'''
    )
    parser.add_argument(
        '--steps', type=int, required=True,
        help='''Provide a number of steps to scan the likelihood space over.'''
    )
    parser.add_argument(
        '--hypo-param-selections', type=str, nargs='+', required=False,
        help='''Selection of params to use in order to generate the
        hypothesised Asimov distributions.'''
    )
    parser.add_argument(
        '--outdir', required=True,
        metavar='DIR', type=str,
        help='Directory into which to store results.'
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
    kwargs = vars(parser.parse_args())
    set_verbosity(kwargs.pop('v'))
    return kwargs


def scan_allsyst(template_settings, steps, hypo_param_selections, outdir,
                 minimizer_settings, metric, debug_mode):
    """Scan (separately) all systematics (i.e., non-fixed params).

    Parameters
    ----------
    template_settings
    steps
    hypo_param_selections
    outdir
    minimizer_settings
    metric
    debug_mode

    Returns
    -------
    restults : dict
        Keys are param names, values are the scan results

    """
    outdir = expanduser(expandvars(outdir))
    mkdir(outdir, warn=False)

    hypo_maker = DistributionMaker(template_settings)

    hypo_maker.select_params(hypo_param_selections)
    data_dist = hypo_maker.get_outputs(return_sum=True)

    minimizer_settings = from_file(minimizer_settings)

    analysis = Analysis()

    results = OrderedDict() # pylint: disable=redefined-outer-name
    for param in hypo_maker.params:
        if param.is_fixed:
            continue

        logging.info('Scanning %s', param.name)
        nominal_value = param.value

        outfile = join(
            outdir,
            '{:s}_{:d}_steps_{:s}_scan.json'.format(param.name, steps,
                                                    metric)
        )
        if isfile(outfile):
            raise IOError('`outfile` "{}" exists, not overwriting.'
                          .format(outfile))

        results[param.name] = analysis.scan(
            data_dist=data_dist,
            hypo_maker=hypo_maker,
            hypo_param_selections=hypo_param_selections,
            metric=metric,
            param_names=param.name,
            steps=steps,
            only_points=None,
            outer=True,
            profile=False,
            minimizer_settings=minimizer_settings,
            outfile=outfile,
            debug_mode=debug_mode
        )

        to_file(results[param.name], outfile)
        param.value = nominal_value

        logging.info('Done scanning param "%s"', param.name)

    logging.info('Done.')

    return results


def main():
    """Run scan_allsyst with arguments from command line"""
    return scan_allsyst(**parse_args())


if __name__ == '__main__':
    results = main() # pylint: disable=invalid-name
