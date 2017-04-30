#!/usr/bin/env python

# authors: T. Ehrhardt
# date:    October 14, 2016

"""
Profile LLH/Chisquare Analysis

"""


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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
        '-d', '--data-settings', type=str,
        metavar='CONFIGFILE', default=None,
        help='''Settings for the generation of "data" distributions; repeat
        this argument to specify multiple pipelines. If omitted, the same
        settings as specified for --template-settings are used to generate data
        distributions.'''
    )
    parser.add_argument(
        '-t', '--template-settings',
        metavar='CONFIGFILE', required=True, action='append',
        help='''Settings for generating template distributions; repeat
        this option to define multiple pipelines.'''
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
        '--data-param-selection', type=str, required=False,
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
        '-o', '--outfile', metavar='FILE',
        type=str, action='store', default='out.json',
        help='file to store the output'
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
        '--find-best-fit', action='store_true',
        help='''Find the best fit point to the data. This is probably
        unnecessary when the data maker equals the hypo maker but is useful
        otherwise. This will be an unblinded fit checking both octants (if 
        theta23 is in the free params) with pprint turned on. Note that 
        profile must also be flagged else this will raise an error.'''
    )
    parser.add_argument(
        '--best-fit-outfile', metavar='FILE',
        type=str, action='store', default=None,
        help='''File to store the best fit output. If --find-best-fit is
        flagged then this must be specified.'''
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
    data_maker = hypo_maker if args.data_settings is None else \
                                        DistributionMaker(args.data_settings)

    data_maker.select_params(args.data_param_selection)
    data = data_maker.get_outputs(return_sum=True)

    analysis = Analysis()

    minimizer_settings = from_file(args.minimizer_settings)

    if args.find_best_fit:
        if not args.profile:
            raise ValueError(
                "You requested to find the best fit point to the data but "
                "not to profile the free systematic parameters. These are "
                "incompatible."
            )
        if args.best_fit_outfile is None:
            raise ValueError(
                "You requested to find the best fit point to the data but "
                "did not specify a file for which this to be saved in."
            )
        # An unblinded fit will be performed checking both octants (if theta23
        # is in the free params) with pprint turned on.
        logging.info("Finding the best fit to the data")
        best_fit, alternate_fits = analysis.fit_hypo(
            data_dist=data,
            hypo_maker=hypo_maker,
            hypo_param_selections=args.hypo_param_selections,
            metric=args.metric,
            minimizer_settings=minimizer_settings
        )
        to_file(best_fit, args.best_fit_outfile)

    res = analysis.scan(data_dist=data, hypo_maker=hypo_maker,
                        hypo_param_selections=args.hypo_param_selections,
                        metric=args.metric, param_names=args.param_names,
                        steps=args.steps, only_points=args.only_points,
                        outer=not args.no_outer, profile=args.profile,
                        minimizer_settings=minimizer_settings,
                        outfile=args.outfile, debug_mode=args.debug_mode)
    to_file(res, args.outfile)
    logging.info("Done.")
