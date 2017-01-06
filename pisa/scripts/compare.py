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

# TODO: make use of `MapSet.compare()` method (and/or expand that until it is
# equally useful here)

from argparse import ArgumentParser
from collections import OrderedDict
import os

import numpy as np

from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.core.pipeline import Pipeline
from pisa.utils.fileio import mkdir, to_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.plotter import Plotter


__all__ = ['DISTRIBUTIONMAKER_SOURCE_STR', 'PIPELINE_SOURCE_STR',
           'MAP_SOURCE_STR', 'MAPSET_SOURCE_STR',
           'parse_args', 'compare', 'main']


DISTRIBUTIONMAKER_SOURCE_STR = (
    'DistributionMaker instantiated from multiple pipeline config files'
)
PIPELINE_SOURCE_STR = 'Pipeline instantiated from a pipelinen config file'
MAP_SOURCE_STR = 'Map stored on disk'
MAPSET_SOURCE_STR = 'MapSet stored on disk'

def parse_args():
    parser = ArgumentParser(description=__doc__)
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
        '--ref-param-selections', type=str, required=False,
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
        '--test-param-selections', type=str, required=False,
        action='append',
        help='''Param selections to apply to --test pipeline config(s). Not
        applicable if --test specifies stored map or map sets'''
    )
    parser.add_argument(
        '--combine', type=str, action='append',
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
        '--diff-min', type=float, required=False,
        help='''Difference plot vmin; if you specify only one of --diff-min or
        --diff-max, symmetric limits are automatically used (min = -max).'''
    )
    parser.add_argument(
        '--diff-max', type=float, required=False,
        help='''Difference plot max; if you specify only one of --diff-min or
        --diff-max, symmetric limits are automatically used (min = -max).'''
    )
    parser.add_argument(
        '--fract-diff-min', type=float, required=False,
        help='''Fractional difference plot vmin; if you specify only one of
        --fract-diff-min or --fract-diff-max, symmetric limits are
        automatically used (min = -max).'''
    )
    parser.add_argument(
        '--fract-diff-max', type=float, required=False,
        help='''Fractional difference plot max; if you specify only one of
        --fract-diff-min or --fract-diff-max, symmetric limits are
        automatically used (min = -max).'''
    )
    parser.add_argument(
        '--asymm-min', type=float, required=False,
        help='''Asymmetry plot vmin; if you specify only one of --asymm-min or
        --asymm-max, symmetric limits are automatically used (min = -max).'''
    )
    parser.add_argument(
        '--asymm-max', type=float, required=False,
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


def compare(outdir, ref, ref_label, test, test_label, asymm_max=None,
            asymm_min=None, combine=None, diff_max=None, diff_min=None,
            fract_diff_max=None, fract_diff_min=None, json=False, pdf=False,
            png=False, ref_abs=False, ref_param_selections=None,
            test_abs=False, test_param_selections=None):
    ref_plot_label = ref_label
    if ref_abs and not ref_label.startswith('abs'):
        ref_plot_label = 'abs(%s)' % ref_plot_label
    test_plot_label = test_label
    if test_abs and not test_label.startswith('abs'):
        test_plot_label = 'abs(%s)' % test_plot_label

    plot_formats = []
    if pdf:
        plot_formats.append('pdf')
    if png:
        plot_formats.append('png')

    diff_symm = True
    if diff_min is not None and diff_max is None:
        diff_max = -diff_min
        diff_symm = False
    if diff_max is not None and diff_min is None:
        diff_min = -diff_max
        diff_symm = False

    fract_diff_symm = True
    if fract_diff_min is not None and fract_diff_max is None:
        fract_diff_max = -fract_diff_min
        fract_diff_symm = False
    if fract_diff_max is not None and fract_diff_min is None:
        fract_diff_min = -fract_diff_max
        fract_diff_symm = False

    asymm_symm = True
    if asymm_max is not None and asymm_min is None:
        asymm_min = -asymm_max
        asymm_symm = False
    if asymm_min is not None and asymm_max is None:
        asymm_max = -asymm_min
        asymm_symm = False

    outdir = os.path.expanduser(os.path.expandvars(outdir))
    mkdir(outdir)

    # Get the reference distribution(s) into the form of a test MapSet
    reference = None
    ref_source = None
    if isinstance(ref, MapSet):
        reference = ref
        ref_source = MAPSET_SOURCE_STR

    if len(ref) == 1:
        try:
            ref_pipeline = Pipeline(config=ref[0])
        except:
            pass
        else:
            ref_source = PIPELINE_SOURCE_STR
            if ref_param_selections is not None:
                ref_pipeline.select_params(ref_param_selections)
            reference = ref_pipeline.get_outputs()
    else:
        try:
            ref_dmaker = DistributionMaker(pipelines=ref)
        except:
            pass
        else:
            ref_source = DISTRIBUTIONMAKER_SOURCE_STR
            if ref_param_selections is not None:
                ref_dmaker.select_params(ref_param_selections)
            reference = ref_dmaker.get_outputs()

    if reference is None:
        try:
            reference = [Map.from_json(f) for f in ref]
        except:
            pass
        else:
            ref_source = MAP_SOURCE_STR
            reference = MapSet(reference)

    if reference is None:
        assert ref_param_selections is None
        assert len(ref) == 1, 'Can only handle one MapSet'
        try:
            reference = MapSet.from_json(ref[0])
        except:
            pass
        else:
            ref_source = MAPSET_SOURCE_STR

    if reference is None:
        raise ValueError(
            'Could not instantiate the reference Pipeline, DistributionMaker,'
            ' Map, or MapSet from ref value(s) %s' % ref
        )
    ref = reference

    logging.info('Reference map(s) derived from a ' + ref_source)

    # Get the test distribution(s) into the form of a test MapSet
    test_dist = None
    test_source = None
    if isinstance(ref, MapSet):
        test_dist = test
        test_source = MAPSET_SOURCE_STR

    if len(test) == 1:
        try:
            test_pipeline = Pipeline(config=test[0])
        except:
            pass
        else:
            test_source = PIPELINE_SOURCE_STR
            if test_param_selections is not None:
                test_pipeline.select_params(test_param_selections)
            test_dist = test_pipeline.get_outputs()
    else:
        try:
            test_dmaker = DistributionMaker(pipelines=test)
        except:
            pass
        else:
            test_source = DISTRIBUTIONMAKER_SOURCE_STR
            if test_param_selections is not None:
                test_dmaker.select_params(test_param_selections)
            test_dist = test_dmaker.get_outputs()

    if test_dist is None:
        try:
            test_dist = [Map.from_json(f) for f in test]
        except:
            pass
        else:
            test_source = MAP_SOURCE_STR
            test_dist = MapSet(test_dist)

    if test_dist is None:
        assert test_param_selections is None
        assert len(test) == 1, 'Can only handle one MapSet'
        try:
            test_dist = MapSet.from_json(test[0])
        except:
            pass
        else:
            test_source = MAPSET_SOURCE_STR

    if test_dist is None:
        raise ValueError(
            'Could not instantiate the test Pipeline, DistributionMaker, Map,'
            ' or MapSet from test value(s) %s' % test
        )
    test = test_dist

    logging.info('Test map(s) derived from a ' + test_source)

    if combine is not None:
        ref = ref.combine_wildcard(combine)
        test = test.combine_wildcard(combine)
        if isinstance(ref, Map):
            ref = MapSet([ref])
        if isinstance(test, Map):
            test = MapSet([test])

    # Set the MapSet names according to args passed by user
    ref.name = ref_label
    test.name = test_label

    # Save to disk the maps being plotted (excluding optional aboslute value
    # operations)
    if json:
        refmaps_path = os.path.join(
            outdir, 'maps__%s.json.bz2' % ref_label
        )
        to_file(ref, refmaps_path)

        testmaps_path = os.path.join(
            outdir, 'maps__%s.json.bz2' % test_label
        )
        to_file(test, testmaps_path)

    if set(test.names) != set(ref.names):
        raise ValueError(
            'Test map names %s do not match ref map names %s.'
            % (sorted(test.names), sorted(ref.names))
        )

    # Alias to save keystrokes
    def masked(x):
        return np.ma.masked_invalid(x.nominal_values)

    reordered_test = []
    new_ref = []
    diff_maps = []
    fract_diff_maps = []
    asymm_maps = []
    summary_stats = {}
    for ref_map in ref:
        test_map = test[ref_map.name].reorder_dimensions(ref_map.binning)
        if ref_abs:
            ref_map = abs(ref_map)
        if test_abs:
            test_map = abs(test_map)

        diff_map = test_map - ref_map
        with np.errstate(divide='ignore', invalid='ignore'):
            fract_diff_map = (test_map - ref_map)/ref_map
            asymm_map = (test_map - ref_map)/ref_map**0.5
        abs_fract_diff_map = np.abs(fract_diff_map)

        new_ref.append(ref_map)
        reordered_test.append(test_map)
        diff_maps.append(diff_map)
        fract_diff_maps.append(fract_diff_map)
        asymm_maps.append(asymm_map)

        min_ref = np.min(masked(ref_map))
        max_ref = np.max(masked(ref_map))

        min_test = np.min(masked(test_map))
        max_test = np.max(masked(test_map))

        total_ref = np.sum(masked(ref_map))
        total_test = np.sum(masked(test_map))

        mean_ref = np.mean(masked(ref_map))
        mean_test = np.mean(masked(test_map))

        max_abs_fract_diff = np.max(masked(abs_fract_diff_map))
        mean_abs_fract_diff = np.mean(masked(abs_fract_diff_map))
        median_abs_fract_diff = np.median(masked(abs_fract_diff_map))

        mean_fract_diff = np.mean(masked(fract_diff_map))
        min_fract_diff = np.min(masked(fract_diff_map))
        max_fract_diff = np.max(masked(fract_diff_map))
        std_fract_diff = np.std(masked(fract_diff_map))

        mean_diff = np.mean(masked(diff_map))
        min_diff = np.min(masked(diff_map))
        max_diff = np.max(masked(diff_map))
        std_diff = np.std(masked(diff_map))

        median_diff = np.nanmedian(masked(diff_map))
        mad_diff = np.nanmedian(masked(np.abs(diff_map)))
        median_fract_diff = np.nanmedian(masked(fract_diff_map))
        mad_fract_diff = np.nanmedian(masked(np.abs(fract_diff_map)))

        min_asymm = np.min(masked(fract_diff_map))
        max_asymm = np.max(masked(fract_diff_map))

        total_asymm = np.sqrt(np.sum(masked(asymm_map)**2))

        summary_stats[test_map.name] = OrderedDict([
            ('min_ref', min_ref),
            ('max_ref', max_ref),
            ('total_ref', total_ref),
            ('mean_ref', mean_ref),

            ('min_test', min_test),
            ('max_test', max_test),
            ('total_test', total_test),
            ('mean_test', mean_test),

            ('max_abs_fract_diff', max_abs_fract_diff),
            ('mean_abs_fract_diff', mean_abs_fract_diff),
            ('median_abs_fract_diff', median_abs_fract_diff),

            ('min_fract_diff', min_fract_diff),
            ('max_fract_diff', max_fract_diff),
            ('mean_fract_diff', mean_fract_diff),
            ('std_fract_diff', std_fract_diff),
            ('median_fract_diff', median_fract_diff),
            ('mad_fract_diff', mad_fract_diff),

            ('min_diff', min_diff),
            ('max_diff', max_diff),
            ('mean_diff', mean_diff),
            ('std_diff', std_diff),
            ('median_diff', median_diff),
            ('mad_diff', mad_diff),

            ('min_asymm', min_asymm),
            ('max_asymm', max_asymm),
            ('total_asymm', total_asymm),
        ])

        logging.info('Map %s...' % ref_map.name)
        logging.info('  Ref map(s):')
        logging.info('    min   :' + ('%.2f' % min_ref).rjust(12))
        logging.info('    max   :' + ('%.2f' % max_ref).rjust(12))
        logging.info('    total :' + ('%.2f' % total_ref).rjust(12))
        logging.info('    mean  :' + ('%.2f' % mean_ref).rjust(12))
        logging.info('  Test map(s):')
        logging.info('    min   :' + ('%.2f' % min_test).rjust(12))
        logging.info('    max   :' + ('%.2f' % max_test).rjust(12))
        logging.info('    total :' + ('%.2f' % total_test).rjust(12))
        logging.info('    mean  :' + ('%.2f' % mean_test).rjust(12))
        logging.info('  Absolute fract. diff., abs((Test - Ref) / Ref):')
        logging.info('    max   : %.4e' %(max_abs_fract_diff))
        logging.info('    mean  : %.4e' %(mean_abs_fract_diff))
        logging.info('    median: %.4e' %(median_abs_fract_diff))
        logging.info('  Fractional difference, (Test - Ref) / Ref:')
        logging.info('    min   : %.4e' %(min_fract_diff))
        logging.info('    max   : %.4e' %(max_fract_diff))
        logging.info('    mean  : %.4e +/- %.4e' %(mean_fract_diff, std_fract_diff))
        logging.info('    median: %.4e +/- %.4e' %(median_fract_diff, mad_fract_diff))
        logging.info('  Difference, Test - Ref:')
        logging.info('    min   : %.4e' %(min_diff))
        logging.info('    max   : %.4e' %(max_diff))
        logging.info('    mean  : %.4e +/- %.4e' %(mean_diff, std_diff))
        logging.info('    median: %.4e +/- %.4e' %(median_diff, mad_diff))
        logging.info('  Asymmetry, (Test - Ref) / sqrt(Ref)')
        logging.info('    min   : %.4e' %(min_asymm))
        logging.info('    max   : %.4e' %(max_asymm))
        logging.info('    total : %.4e (sum in quadrature)' %total_asymm)
        logging.info('')

    ref = MapSet(new_ref)
    test = MapSet(reordered_test)
    diff = MapSet(diff_maps)
    fract_diff = MapSet(fract_diff_maps)
    asymm = MapSet(asymm_maps)

    if json:
        diff.to_json(os.path.join(
            outdir,
            'diff__%s__%s.json.bz2' %(test_plot_label, ref_plot_label)
        ))
        fract_diff.to_json(os.path.join(
            outdir,
            'fract_diff__%s___%s.json.bz2' %(test_plot_label, ref_plot_label)
        ))
        asymm.to_json(os.path.join(
            outdir,
            'asymm__%s___%s.json.bz2' %(test_plot_label, ref_plot_label)
        ))
        to_file(
            summary_stats,
            os.path.join(
                outdir,
                'stats__%s__%s.json.bz2' %(test_plot_label, ref_plot_label)
            )
        )

    for plot_format in plot_formats:
        # Plot the raw distributions
        plotter = Plotter(stamp='', outdir=outdir, fmt=plot_format,
                          log=False, annotate=False,
                          symmetric=False,
                          ratio=False)
        plotter.plot_2d_array(ref, fname='distr__%s'
                              % ref_plot_label)
        plotter.plot_2d_array(test, fname='distr__%s'
                              % test_plot_label)

        # Plot the difference (test - ref)
        plotter = Plotter(stamp='', outdir=outdir, fmt=plot_format,
                          log=False, annotate=False,
                          symmetric=diff_symm,
                          ratio=False)
        plotter.label = '%s - %s' % (test_plot_label, ref_plot_label)
        plotter.plot_2d_array(
            test - ref,
            fname='diff__%s__%s' % (test_plot_label, ref_plot_label),
            cmap='RdBu',
            #vmin=diff_min, vmax=diff_max
        )

        # Plot the fractional difference (test - ref)/ref
        plotter = Plotter(stamp='', outdir=outdir, fmt=plot_format,
                          log=False,
                          annotate=False,
                          symmetric=fract_diff_symm,
                          ratio=True)
        plotter.label = '%s/%s - 1' % (test_plot_label, ref_plot_label)
        plotter.plot_2d_array(
            test/ref - 1.,
            fname='fract_diff__%s__%s' % (test_plot_label, ref_plot_label),
            cmap='RdBu',
            #vmin=fract_diff_min, vmax=fract_diff_max
        )

        # Plot the asymmetry (test - ref)/sqrt(ref)
        plotter = Plotter(stamp='', outdir=outdir, fmt=plot_format,
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
            #vmin=asymm_min, vmax=asymm_max
        )


def main():
    args = vars(parse_args())
    set_verbosity(args.pop('v'))

    compare(**args)


main.__doc__ = __doc__


if __name__ == '__main__':
    main()
