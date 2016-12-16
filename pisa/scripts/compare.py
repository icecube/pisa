#! /usr/bin/env python
# author: J.L. Lanfranchi
# date:   October 8, 2016
"""
Compare two entities: Maps, map sets, pipelines, or distribution makers. One
kind can be compared against another, so long as the resulting map(s) have
equivalent names and binning. The result each entity specification is formatted
into a MapSet and stored to disk, so that e.g. re-running a DistributionMaker
is unnecessary to reproduce the results.
"""


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
import os

import numpy as np

from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.utils.fileio import mkdir, to_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.plotter import Plotter


__all__ = ['parse_args', 'main']


def parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--outdir', metavar='DIR', type=str, required=True,
        help='''Store output plots to this directory.'''
    )
    parser.add_argument(
        '--ref', type=str, required=True, action='append',
        help='''Pipeline settings config file that generates reference
        output, or a stored map or map set. Repeat --ref option for multiple
        pipelines, maps, or map sets'''
    )
    parser.add_argument(
        '--ref-abs', action='store_true',
        help='''Use the absolute value of the reference plot for
        comparisons.'''
    )
    parser.add_argument(
        '--ref-label', type=str, required=True,
        help='''Label for reference'''
    )
    parser.add_argument(
        '--ref-param-selections', type=str, required=False, default=None,
        action='append',
        help='''Param selections to apply to --ref pipeline config(s). Not
        applicable if --ref specifies stored map or map sets'''
    )
    parser.add_argument(
        '--test', type=str, required=True, action='append',
        help='''Pipeline settings config file that generates test
        output, or a stored map or map set. Repeat --test option for multiple
        pipelines, maps, or map sets'''
    )
    parser.add_argument(
        '--test-abs', action='store_true',
        help='''Use the absolute value of the test plot for
        comparisons.'''
    )
    parser.add_argument(
        '--test-label', type=str, required=True,
        help='''Label for test'''
    )
    parser.add_argument(
        '--test-param-selections', type=str, required=False, default=None,
        action='append',
        help='''Param selections to apply to --test pipeline config(s). Not
        applicable if --test specifies stored map or map sets'''
    )
    parser.add_argument(
        '--combine', type=str, default=None, action='append',
        help='''Combine by wildcard string, where string globbing (a la command
        line) uses asterisk for any number of wildcard characters. Use single
        quotes such that asterisks do not get expanded by the shell. Repeat the
        --combine option for multiple combine strings.'''
    )
    parser.add_argument(
        '--json', action='store_true',
        help='''Save output maps in compressed json (json.bz2) format.''' 
    )
    parser.add_argument(
        '--pdf', action='store_true',
        help='''Save plots in PDF format. If neither this nor --png is
        specified, no plots are produced.'''
    )
    parser.add_argument(
        '--png', action='store_true',
        help='''Save plots in PNG format. If neither this nor --pdf is
        specfied, no plots are produced.'''
    )
    parser.add_argument(
        '--diff-min', type=float, required=False, default=None,
        help='''Difference plot vmin; if you specify only one of --diff-min or
        --diff-max, symmetric limits are automatically used (min = -max).'''
    )
    parser.add_argument(
        '--diff-max', type=float, required=False, default=None,
        help='''Difference plot max; if you specify only one of --diff-min or
        --diff-max, symmetric limits are automatically used (min = -max).'''
    )
    parser.add_argument(
        '--fract-diff-min', type=float, required=False, default=None,
        help='''Fractional difference plot vmin; if you specify only one of
        --fract-diff-min or --fract-diff-max, symmetric limits are
        automatically used (min = -max).'''
    )
    parser.add_argument(
        '--fract-diff-max', type=float, required=False, default=None,
        help='''Fractional difference plot max; if you specify only one of
        --fract-diff-min or --fract-diff-max, symmetric limits are
        automatically used (min = -max).'''
    )
    parser.add_argument(
        '--asymm-min', type=float, required=False, default=None,
        help='''Asymmetry plot vmin; if you specify only one of --asymm-min or
        --asymm-max, symmetric limits are automatically used (min = -max).'''
    )
    parser.add_argument(
        '--asymm-max', type=float, required=False, default=None,
        help='''Fractional difference plot max; if you specify only one of
        --asymm-min or --asymm-max, symmetric limits are automatically used
        (min = -max).'''
    )
    parser.add_argument(
        '-v', action='count',
        help='Set verbosity level; repeat -v for higher level.'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_verbosity(args.v)

    ref_plot_label = args.ref_label
    if args.ref_abs and not args.ref_label.startswith('abs'):
        ref_plot_label = 'abs(%s)' % ref_plot_label
    test_plot_label = args.test_label
    if args.test_abs and not args.test_label.startswith('abs'):
        test_plot_label = 'abs(%s)' % test_plot_label

    plot_formats = []
    if args.pdf:
        plot_formats.append('pdf')
    if args.png:
        plot_formats.append('png')

    diff_symm = True
    if args.diff_min is not None and args.diff_max is None:
        args.diff_max = -args.diff_min
        diff_symm = False
    if args.diff_max is not None and args.diff_min is None:
        args.diff_min = -args.diff_max
        diff_symm = False

    fract_diff_symm = True
    if args.fract_diff_min is not None and args.fract_diff_max is None:
        args.fract_diff_max = -args.fract_diff_min
        fract_diff_symm = False
    if args.fract_diff_max is not None and args.fract_diff_min is None:
        args.fract_diff_min = -args.fract_diff_max
        fract_diff_symm = False

    asymm_symm = True
    if args.asymm_max is not None and args.asymm_min is None:
        args.asymm_min = -args.asymm_max
        asymm_symm = False
    if args.asymm_min is not None and args.asymm_max is None:
        args.asymm_max = -args.asymm_min
        asymm_symm = False

    args.outdir = os.path.expanduser(os.path.expandvars(args.outdir))
    mkdir(args.outdir)

    # Get the reference distribution(s) into the form of a test MapSet
    ref = None
    ref_source = None
    try:
        ref_dmaker = DistributionMaker(pipelines=args.ref)
    except:
        pass
    else:
        ref_source = 'DistributionMaker'
        if args.ref_param_selections is not None:
            ref_dmaker.select_params(args.ref_param_selections)
        ref = ref_dmaker.get_outputs(return_sum=True)

    if ref is None:
        try:
            ref = [Map.from_json(f) for f in args.ref]
        except:
            pass
        else:
            ref_source = 'Map'
            ref = MapSet(ref)

    if ref is None:
        assert args.ref_param_selections is None
        assert len(args.ref) == 1, 'Can only handle one MapSet'
        try:
            ref = MapSet.from_json(args.ref[0])
        except:
            pass
        else:
            ref_source = 'MapSet'

    if ref is None:
        raise ValueError(
            'Could not instantiate the reference DistributionMaker, Map, or'
            ' MapSet from ref value(s) %s' % args.ref
        )

    # Get the test distribution(s) into the form of a test MapSet
    test = None
    test_source = None
    try:
        test_dmaker = DistributionMaker(pipelines=args.test)
    except:
        pass
    else:
        test_source = 'DistributionMaker'
        if args.test_param_selections is not None:
            test_dmaker.select_params(args.test_param_selections)
        test = test_dmaker.get_outputs(return_sum=True)

    if test is None:
        try:
            test = [Map.from_json(f) for f in args.test]
        except:
            pass
        else:
            test_source = 'Map'
            test = MapSet(test)

    if test is None:
        assert args.test_param_selections is None
        assert len(args.test) == 1, 'Can only handle one MapSet'
        try:
            test = MapSet.from_json(args.test[0])
        except:
            pass
        else:
            test_source = 'MapSet'

    if test is None:
        raise ValueError(
            'Could not instantiate the test DistributionMaker, Map, or MapSet'
            ' from test value(s) %s' % args.test
        )

    if args.combine is not None:
        ref = ref.combine_wildcard(args.combine)
        test = test.combine_wildcard(args.combine)
        if isinstance(ref, Map):
            ref = MapSet([ref])
        if isinstance(test, Map):
            test = MapSet([test])

    # Save to disk the maps being plotted (excluding optional aboslute value
    # operations)
    if args.json:
        ref.to_json(os.path.join(
            args.outdir, 'maps__%s.json.bz2' % args.ref_label
        ))
        test.to_json(os.path.join(
            args.outdir, 'maps__%s.json.bz2' % args.test_label
        ))

    if set(test.names) != set(ref.names):
        raise ValueError(
            'Test map names %s do not match ref map names %s.'
            % (sorted(test.names), sorted(ref.names))
        )

    reordered_test = []
    new_ref = []
    diff_maps = []
    fract_diff_maps = []
    asymm_maps = []
    summary_stats = {}
    for ref_map in ref:
        test_map = test[ref_map.name].reorder_dimensions(ref_map.binning)
        if args.ref_abs:
            ref_map = abs(ref_map)
        if args.test_abs:
            test_map = abs(test_map)

        diff_map = test_map - ref_map
        fract_diff_map = (test_map - ref_map)/ref_map
        asymm_map = (test_map - ref_map)/ref_map**0.5

        new_ref.append(ref_map)
        reordered_test.append(test_map)
        diff_maps.append(diff_map)
        fract_diff_maps.append(fract_diff_map)
        asymm_maps.append(asymm_map)

        total_ref = np.sum(np.ma.masked_invalid(ref_map.nominal_values))
        total_test = np.sum(np.ma.masked_invalid(test_map.nominal_values))

        mean_ref = np.mean(np.ma.masked_invalid(ref_map.nominal_values))
        mean_test = np.mean(np.ma.masked_invalid(test_map.nominal_values))

        mean_diff = np.mean(np.ma.masked_invalid(diff_map.nominal_values))
        std_diff = np.std(np.ma.masked_invalid(diff_map.nominal_values))
        mean_fract_diff = np.mean(np.ma.masked_invalid(fract_diff_map.nominal_values))
        std_fract_diff = np.std(np.ma.masked_invalid(fract_diff_map.nominal_values))

        median_diff = np.median(np.ma.masked_invalid(diff_map.nominal_values))
        mad_diff = np.median(np.abs(np.ma.masked_invalid(diff_map.nominal_values)))
        median_fract_diff = np.median(np.ma.masked_invalid(fract_diff_map.nominal_values))
        mad_fract_diff = np.median(np.abs(np.ma.masked_invalid(fract_diff_map.nominal_values)))

        asymm = np.sqrt(np.sum(np.ma.masked_invalid(asymm_map.nominal_values)**2))

        summary_stats[test_map.name] = OrderedDict([
            ('total_ref', total_ref),
            ('total_test', total_test),
            ('mean_ref', mean_ref),
            ('mean_test', mean_test),
            ('mean_diff', mean_diff),
            ('std_diff', std_diff),
            ('mean_fract_diff', mean_fract_diff),
            ('std_fract_diff', std_fract_diff),
            ('median_diff', median_diff),
            ('mad_diff', mad_diff),
            ('median_fract_diff', median_fract_diff),
            ('mad_fract_diff', mad_fract_diff),
            ('asymm', asymm),
        ])

        logging.info('Map %s...' % ref_map.name)
        logging.info('  Pct Agreement: %+8.3f%s' % (100*mean_fract_diff, '%'))
        logging.info('  Totals:')
        logging.info('    Ref :' + ('%.2f' % total_ref).rjust(8))
        logging.info('    Test:' + ('%.2f' % total_test).rjust(8))
        logging.info('  Means:')
        logging.info('    Ref :' + ('%.2f' % mean_ref).rjust(8))
        logging.info('    Test:' + ('%.2f' % mean_test).rjust(8))
        logging.info('  Test - Ref, mean +/- std dev:')
        logging.info('    %.4e +/- %.4e' %(mean_diff, std_diff))
        logging.info('  Test - Ref, median +/- median-abs-dev:')
        logging.info('    %.4e +/- %.4e' %(median_diff, mad_diff))
        logging.info('  (Test - Ref) / Ref, mean +/- std dev:')
        logging.info('    %.4e +/- %.4e' %(mean_fract_diff, std_fract_diff))
        logging.info('  (Test - Ref) / Ref, median +/- median-abs-dev:')
        logging.info('    %.4e +/- %.4e' %(median_fract_diff, mad_fract_diff))
        logging.info('  (Test - Ref) / sqrt(Ref), sum in quadrature:')
        logging.info('    %.4e' %asymm)
        logging.info('')

    ref = MapSet(new_ref)
    test = MapSet(reordered_test)
    diff = MapSet(diff_maps)
    fract_diff = MapSet(fract_diff_maps)
    asymm = MapSet(asymm_maps)

    if args.json:
        diff.to_json(os.path.join(
            args.outdir,
            'diff__%s__%s.json.bz2' %(test_plot_label, ref_plot_label)
        ))
        fract_diff.to_json(os.path.join(
            args.outdir,
            'fract_diff__%s___%s.json.bz2' %(test_plot_label, ref_plot_label)
        ))
        asymm.to_json(os.path.join(
            args.outdir,
            'asymm__%s___%s.json.bz2' %(test_plot_label, ref_plot_label)
        ))
        to_file(
            summary_stats,
            os.path.join(
                args.outdir,
                'stats__%s__%s.json.bz2' %(test_plot_label, ref_plot_label)
            )
        )

    for plot_format in plot_formats:
        # Plot the raw distributions
        plotter = Plotter(stamp='', outdir=args.outdir, fmt=plot_format,
                          log=False, annotate=False,
                          symmetric=False,
                          ratio=False)
        plotter.plot_2d_array(ref, fname='distr__%s'
                              % ref_plot_label)
        plotter.plot_2d_array(test, fname='distr__%s'
                              % test_plot_label)

        # Plot the difference (test - ref)
        plotter = Plotter(stamp='', outdir=args.outdir, fmt=plot_format,
                          log=False, annotate=False,
                          symmetric=diff_symm,
                          ratio=False)
        plotter.label = '%s - %s' % (test_plot_label, ref_plot_label)
        plotter.plot_2d_array(
            test - ref,
            fname='diff__%s__%s' % (test_plot_label, ref_plot_label),
            cmap='RdBu',
            #vmin=args.diff_min, vmax=args.diff_max
        )

        # Plot the fractional difference (test - ref)/ref
        plotter = Plotter(stamp='', outdir=args.outdir, fmt=plot_format,
                          log=False,
                          annotate=False,
                          symmetric=fract_diff_symm,
                          ratio=True)
        plotter.label = '%s/%s - 1' % (test_plot_label, ref_plot_label)
        plotter.plot_2d_array(
            test/ref - 1.,
            fname='fract_diff__%s__%s' % (test_plot_label, ref_plot_label),
            cmap='RdBu',
            #vmin=args.fract_diff_min, vmax=args.fract_diff_max
        )

        # Plot the asymmetry (test - ref)/sqrt(ref)
        plotter = Plotter(stamp='', outdir=args.outdir, fmt=plot_format,
                          log=False,
                          annotate=False,
                          symmetric=asymm_symm,
                          ratio=True)
        plotter.label = '(%s - %s)/sqrt(%s)' % (test_plot_label,
                                                ref_plot_label, ref_plot_label)
        plotter.plot_2d_array(
            (test-ref)/ref**0.5,
            fname='asymm__%s__%s' % (test_plot_label, ref_plot_label),
            cmap='RdBu',
            #vmin=args.asymm_min, vmax=args.asymm_max
        )


if __name__ == '__main__':
    main()
