#!/usr/bin/env python

# authors: S. Wren, T. Ehrhardt
# date:    December 3, 2016

"""
Performs 1D scans over all of the systematics in a pipeline (or multiple 
pipelines) and saves the output. This is to check the their likelihood spaces.

"""


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import os
import numpy as np

from pisa import ureg, Q_
from pisa.analysis.analysis import Analysis
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.config_parser import parse_pipeline_config
from pisa.utils.fileio import from_file, to_file
from pisa.utils.log import logging, set_verbosity


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-t', '--template-settings',
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
        '-o', '--outdir', required=True,
        metavar='DIR', type=str,
        help='Directory into which to store results.'
    )
    parser.add_argument(
        '-m', '--minimizer-settings', type=str,
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

    set_verbosity(args.v)

    hypo_maker = DistributionMaker(args.template_settings)

    hypo_maker.select_params(args.hypo_param_selections)
    data = hypo_maker.get_outputs(return_sum=True)

    analysis = Analysis()

    minimizer_settings = from_file(args.minimizer_settings)

    for param in hypo_maker.params:

        if not param.is_fixed:

            logging.info("Scanning %s"%param.name)
            nominal_value = param.value
            outfile = os.path.join(
                args.outdir,
                "%s_%i_steps_%s_scan.json"%(
                    param.name,
                    args.steps,
                    args.metric
                )
            )
            res = analysis.scan(
                data_dist=data,
                hypo_maker=hypo_maker,
                hypo_param_selections=args.hypo_param_selections,
                metric=args.metric,
                param_names=param.name,
                steps=args.steps,
                only_points=None,
                outer=True,
                profile=False,
                minimizer_settings=minimizer_settings,
                outfile=outfile,
                debug_mode=args.debug_mode
            )
            to_file(res, outfile)
            param.value = nominal_value
            logging.info("Done with %s"%param.name)
    logging.info("Done.")
