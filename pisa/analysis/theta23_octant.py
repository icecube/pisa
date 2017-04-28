#!/usr/bin/env python

# authors: T. Ehrhardt
# date:    October 20, 2016

"""
Theta23 Octant/Maximal Mixing Analysis

"""

from pisa.analysis.analysis import Analysis
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.log import logging, set_verbosity


def get_metadata(ini_args):
    METADATA_K = ('data_settings', 'template_settings', 'data_param_selection',
                  'only_correct_nmo', 'only_x_nmo', 'minimizer_settings')
    ini_args_d = vars(ini_args)
    metadata_d = {k: ini_args_d.get(k, None) for k in METADATA_K}
    return metadata_d

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from itertools import product
    from copy import deepcopy

    import numpy as np

    from pisa import ureg, Q_
    from pisa.utils.fileio import from_file, to_file
    from pisa.utils.config_parser import parse_pipeline_config

    parser = ArgumentParser(description = '''Calculates the separation between
    data and the wrong octant of theta23 (+ theta23 = 45 deg) for given injected
    value(s) of theta23 and livetime.''',
    formatter_class = ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--theta23', type=float, required=True, nargs='+',
        help='Theta23 values to inject (in degrees).'
    )
    parser.add_argument(
        '--livetime', type=float, required=True, nargs='+',
         help='Livetimes to sample.'
    )
    parser.add_argument(
        '-d', '--data-settings', type=str,
        metavar='configfile', default=None,
        help='''Settings for the generation of "data" distributions; repeat
        this argument to specify multiple pipelines. If omitted, the same
        settings as specified for --template-settings are used to generate data
        distributions.'''
    )
    # TODO: implement eventually
    parser.add_argument(
        '--fluctuate-data', action='store_true',
        help='''*Not implemented yet*. Apply fluctuations to the data
        distribution. This should *not* be set for analyzing "real" (measured)
        data, and it is common to not use this feature even for Monte Carlo
        analysis.'''
    )
    parser.add_argument(
        '-t', '--template-settings',
        metavar='CONFIGFILE', required=True, action='append',
        help='''Settings for generating template distributions; repeat
        this option to define multiple pipelines.'''
    )
    parser.add_argument(
        '--data-param-selection', type=str, required=False,
        help='''Selection of params to use in order to generate the data
        distributions. Only one allowed (e.g. 'nh' for normal ordering data).'''
    )
    hypo_group = parser.add_mutually_exclusive_group()
    hypo_group.add_argument(
        '--only-correct-nmo', action='store_true', default=False,
        dest='only_correct_nmo',
        help='''Only assume NMO is correctly identified. By default, will
        try to minimize over both NMOs.'''
    )
    hypo_group.add_argument(
        '--only-cross-nmo', action='store_true', default=False,
        dest='only_x_nmo', help='''Only assume NMO is mis-identified.'''
    )
    parser.add_argument(
        '--theta23-only', action='store_true', help='''Only fit wrong-octant
        theta23, without minimizing over systematics.'''
    )
    parser.add_argument(
        '--no-theta23-prior', action='store_true', help='''Force removal of
        prior knowledge of theta23 in the fit.'''
    )
    parser.add_argument(
        '--theta23-range', type=str, required=False,
        help='''Range of theta23 in degrees to use in wrong-octant fits
        (will be split up into one lower- and one upper-octant range).
        Delimited length-2 list input. If none specified, range
        '[30 deg, 60 deg]' will be assumed.'''
    )
    parser.add_argument(
        '-m', '--minimizer-settings', type=str,
        metavar='JSONFILE', required=True,
        help='''Settings related to the minimizer used.'''
    )
    parser.add_argument(
        '--metric', type=str,
        choices=['llh', 'chi2', 'conv_llh', 'mod_chi2'], required=True,
        help='''Metric to be minimized.'''
    )
    parser.add_argument(
        '--debug-mode', type=int, choices=[0, 1, 2], required=False, default=1,
        help='''*Use 0 for `slsqp` minimizer settings file*.
        How much information to keep in the output file. 0 for only
        essentials for a physics analysis, 1 for more minimizer history, 2 for
        whatever can be recorded.'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='Set verbosity level.'
    )
    parser.add_argument(
        '-o', '--outfile', metavar='FILE',
        type=str, action='store', default='t23_octant.json',
        help='File to store the output to.'
    )

    def delete_keys(fit_dict, keys):
        for k in fit_dict:
            if k in keys:
                del fit_dict[k]

    args = parser.parse_args()

    set_verbosity(args.v)

    debug_mode = args.debug_mode
    if not debug_mode in (0, 1, 2):
        debug_mode = 2

    # set up results dict
    results_dict = {}
    results_dict["metadata"] = get_metadata(args)
    results_dict["truth_sampled"] = []
    results_dict["fits"] = {}

    # parse minimizer settings
    minimizer_settings = from_file(args.minimizer_settings)

    # set up distribution makers, but keep them completey separate instances so
    # we do not accidentally modify the hypo_maker's ParamSet when manipulating
    # the true parameter values
    hypo_maker = DistributionMaker(args.template_settings)
    data_settings = args.data_settings if args.data_settings is not None else \
                    args.template_settings
    data_maker = DistributionMaker(data_settings)

    # find the requested set(s) of parameters for the fits
    # depending on user input
    if args.data_param_selection == "nh":
        if args.only_correct_nmo: hypo_param_selections = "nh"
        elif args.only_x_nmo: hypo_param_selections = "ih"

    elif args.data_param_selection == "ih":
        if args.only_correct_nmo: hypo_param_selections = "ih"
        elif args.only_x_nmo: hypo_param_selections = "nh"

    elif args.data_param_selection is not None:
        raise ValueError("Data parameter selection '%s' not supported!" %
                                                    args.data_param_selection)

    # fits of both orderings will be performed if not explicitly negated
    if not args.only_correct_nmo and not args.only_x_nmo:
        hypo_param_selections = ["nh", "ih"]

    if isinstance(hypo_param_selections, basestring):
        hypo_param_selections = [hypo_param_selections]

    # get the fit ranges for theta23
    if args.theta23_range is not None:
        t23_range = [int(t23_bound) for t23_bound in
                                                args.theta23_range.split(',')]
        assert len(t23_range) == 2
        # require ranges for both octants for now, independent of whether true
        # values of theta23 lie in both octants or not
        assert t23_range[1] > 45.0
        assert t23_range[0] < 45.0
        t23_range_lo = (t23_range[0], 45.0)*ureg.deg
        t23_range_uo = (45.0, t23_range[1])*ureg.deg
    else:
        t23_range_lo = (30.0, 45.0)*ureg.deg
        t23_range_uo = (45.0, 60.0)*ureg.deg

    # select the requested set of parameters for the data distributions
    data_maker.select_params(args.data_param_selection)

    # describe type of data for logging purposes
    data_dist_type = "Asimov" if not args.fluctuate_data else "pseudo"

    # now create the (livetime, theta23) combinations to run the analysis for
    for lt, t23 in product(args.livetime, args.theta23):
        lt = lt*ureg.common_year
        # TODO: consistent units (range + values etc.)
        t23 = t23*ureg.deg
        results_dict["truth_sampled"].append({"livetime": lt, "theta23": t23})
        logging.info("Generating %s data distribution for true livetime='%s'"
                     " and true theta23='%s'"%(data_dist_type, lt, t23))
        data_maker.params.livetime.value = lt

        # extend range of theta23 to encompass injected value, as otherwise
        # validation will throw - this has no impact on anything that follows
        if t23 > data_maker.params.theta23.range[1]:
            data_maker.params.theta23.range = \
                (data_maker.params.theta23.range[0].to(t23.units), t23)
        elif t23 < data_maker.params.theta23.range[0]:
            data_maker.params.theta23.range = \
                (t23, data_maker.params.theta23.range[1].to(t23.units))

        # now set true theta23
        data_maker.params.theta23.value = t23

        # generate the data distributions
        data_dist = data_maker.get_outputs(return_sum=True)

        # loop over fit parameter selections
        for hypo_param_selection in hypo_param_selections:
            one_fit_dict = {'wrong octant': {}, 'maximal mixing': {}}
            if not hypo_param_selection in results_dict["fits"]:
                results_dict["fits"][hypo_param_selection] = []

            logging.info("Fit parameters: '%s'"%hypo_param_selection)
            hypo_maker.select_params(hypo_param_selection)

            # also make sure to set the identical livetime in the fit
            hypo_maker.params.livetime.value = lt

            if args.no_theta23_prior:
                # force removal of prior on theta23
                hypo_maker.params.theta23.prior = None

            # restrict range to reasonable wrong-octant interval
            # 45 degrees is part of both ranges
            if t23 < 45.0*ureg.deg:
                hypo_maker.params.theta23.range = t23_range_uo
                logging.info("Searching for best upper octant match to"
                             " theta23='%s' in range %s" %(t23, t23_range_uo))
            elif t23 >= 45.0*ureg.deg:
                hypo_maker.params.theta23.range = t23_range_lo
                logging.info("Searching for best lower octant match to"
                             " theta23='%s' in range %s" %(t23, t23_range_lo))

            # now we can update the seed (after resetting free parameters)
            # but we cannot just also reset theta23, as nominal and range
            # might clash
            hypo_maker.params.fix("theta23")
            hypo_maker.params.reset_free()
            hypo_maker.params.unfix("theta23")
            t23_wo_seed = 90*ureg.deg - t23
            hypo_maker.params.theta23.value = t23_wo_seed
            logging.debug("Starting fit at theta23='%s'"%t23_wo_seed)

            if args.theta23_only:
                # fix all parameters except theta23
                for p in hypo_maker.params.free:
                    if p.name != 'theta23': p.is_fixed = True

            # finally instantiate an `Analysis` object, which provides the
            # necessary fitting tools
            analysis = Analysis()
            # start at the wrong octant mirror point, and allow theta23 to vary
            bf, _ = analysis.fit_hypo(data_dist=data_dist, hypo_maker=hypo_maker,
                                   hypo_param_selections=hypo_param_selection,
                                   metric=args.metric, reset_free=False,
                                   check_octant=False,
                                   minimizer_settings=minimizer_settings,
                                   )
            # now check the match of the solution at maximal mixing, not allowing
            # theta23 but all the other systematics to vary
            logging.info("Checking fit of maximal mixing.")
            hypo_maker.params.theta23.value = 45*ureg.deg
            hypo_maker.params.fix("theta23")
            hypo_maker.params.reset_free()
            if not args.theta23_only:
                bf_max_mix, _ = analysis.fit_hypo(data_dist=data_dist,
                                   hypo_maker=hypo_maker,
                                   hypo_param_selections=hypo_param_selection,
                                   metric=args.metric, reset_free=False,
                                   check_octant=False,
                                   minimizer_settings=minimizer_settings,
                                   )
            # no need for a fit if args.theta23_only = True
            else:
                bf_max_mix = analysis.nofit_hypo(data_dist=data_dist,
                               hypo_maker=hypo_maker,
                               hypo_param_selections=hypo_param_selection,
                               hypo_asimov_dist=hypo_maker.get_outputs(return_sum=True),
                               metric=args.metric
                               )

            for bf_dict in (bf, bf_max_mix):
                # TODO: serialisation!
                bf_dict['params'] = deepcopy(bf_dict['params'].serializable_state)
                if 'minimizer_metadata' in bf_dict:
                    delete_keys(bf_dict['minimizer_metadata'],
                                keys=['hess_inv'])
                # furthermore, decide which information to retain based on chosen
                # debug mode
                if debug_mode == 0 or debug_mode == 1:
                    delete_keys(bf_dict, ['fit_history', 'hypo_asimov_dist'])
                if debug_mode == 0:
                    delete_keys(bf_dict, ['minimizer_metadata', 'minimizer_time'])
                try: bf_dict['hypo_asimov_dist'] = \
                        deepcopy(bf_dict['hypo_asimov_dist'].serializable_state)
                except: pass

            one_fit_dict['wrong octant'] = deepcopy(bf)
            one_fit_dict['maximal mixing'] = deepcopy(bf_max_mix)

            results_dict["fits"][hypo_param_selection].append(one_fit_dict)
            to_file(results_dict, args.outfile)
            # unfix theta23 again
            hypo_maker.params.theta23.is_fixed = False

    logging.info("Done.")

