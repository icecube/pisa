#! /usr/bin/env python
# author: S.Wren
# date:   March 20, 2016
"""
Runs the pipeline in event-by-event mode to see how much it agrees with OscFit.
This is also checked against the PISA 2 legacy mode.
Test data for comparing against should be in the tests/data directory.
A set of plots will be output in your output directory for you to check.
"""

from argparse import ArgumentParser

from pisa.core.pipeline import Pipeline
from pisa.utils.fileio import mkdir
from pisa.utils.log import logging, set_verbosity
from pisa.utils.tests import plot_cmp

FMT = 'png'

if __name__ == '__main__':
    parser = ArgumentParser(
        description='''Run a set of tests on the PISA 3 pipeline against
        the output from OscFit. If no test flags are specified, *all* tests will
        be run.'''
    )
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='''Run baseline tests i.e. the output of PISA 3
                        event by event and OscFit with a set of parameters
                        agreed upon before the tests were started.''')
    parser.add_argument('--continuous_systematics', action='store_true',
                        default=False,
                        help='''Run continuous systematics tests i.e. the
                        output of PISA 3 event by event and OscFit with
                        variations on the NOT discrete systematics. The
                        fiducial model was agreed upon before the tests were
                        started.''')
    parser.add_argument('--outdir', metavar='DIR', type=str, default=None,
                        required=False,
                        help='''Store all output plots to this directory. If
                        they don't exist, the script will make them, including
                        all subdirectories. If none is supplied no plots will
                        be saved.''')
    parser.add_argument('--ratio_threshold', type=float, default=1E-8,
                        help='''Sets the agreement threshold on the ratio test
                        plots. If this is not reached the tests will fail.''')
    parser.add_argument('--diff_threshold', type=float, default=2E-3,
                        help='''Sets the agreement threshold on the diff test
                        plots. If this is not reached the tests will fail. This
                        test is only important if any ratios return inf.''')
    parser.add_argument('-v', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()
    set_verbosity(args.v)

    if args.outdir is not None:
        mkdir(args.outdir)

    hist_cfg = 'tests/settings/smooth_test_aeff.hist.cfg'
    smooth_cfg = 'tests/settings/smooth_test_aeff.smooth.cfg'

    # Figure out which tests to do
    test_all = True

    # Perform internal tests
    if test_all:
        hist_pipeline = Pipeline(
            config=hist_cfg
        )
        smooth_pipeline = Pipeline(
            config=smooth_cfg
        )
        hist_maps = hist_pipeline.get_outputs()
        smooth_maps = smooth_pipeline.get_outputs()
        assert smooth_maps.names == hist_maps.names
        for map_name in smooth_maps.names:
            vmap = smooth_maps[map_name]
            hmap = hist_maps[map_name]
            comparisons = vmap.compare(hmap)
            for k in ['max_diff_ratio', 'max_diff', 'nanmatch', 'infmatch']:
                print '%s: %s = %s' %(map_name, k, comparisons[k])

            if args.outdir is not None:
                plot_cmp(new=vmap, ref=hmap, new_label='aeff.smooth',
                         ref_label='aeff.hist', plot_label=vmap.tex,
                         file_label=vmap.name, outdir=args.outdir,
                         ftype='png')
