#! /usr/bin/env python
# author: S.Wren
# date:   March 20, 2016
"""
Run the pipeline in event-by-event mode to see how much it agrees with OscFit.
This is also checked against the PISA 2 legacy mode.

Test data for comparing against should be in the tests/data directory. A set of
plots will be output in your output directory for you to check.
"""


from argparse import ArgumentParser
from copy import deepcopy
import os
import sys

from pisa import PYCUDA_AVAIL, ureg, Q_
from pisa.core.map import MapSet
from pisa.core.pipeline import Pipeline
from pisa.utils.config_parser import parse_pipeline_config
from pisa.utils.fileio import from_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.tests import print_agreement, check_agreement, plot_comparisons


__all__ = ['consistency_test', 'compare_baseline', 'compare_systematics', 'main']


def consistency_test(config, testname, outdir, pisa3file,
                     ratio_test_threshold, diff_test_threshold):
    """
    Compare baseline output of PISA 3 with an older version of itself
    (for self-consistency checks).
    """
    logging.debug('>> Doing PISA self-consistency test for %s'%testname)
    baseline_comparisons = MapSet(from_file(pisa3file)["maps"])
    ref_abv='PISAV3Ref'

    pipeline = Pipeline(config)
    outputs = pipeline.get_outputs()

    for nukey in baseline_comparisons.names:

        pisa3ref_map = baseline_comparisons[nukey]
        pisa3ref_map_to_plot = {}
        pisa3ref_map_to_plot['ebins'] = \
                pisa3ref_map.binning['reco_energy'].bin_edges.magnitude
        pisa3ref_map_to_plot['czbins'] = \
                pisa3ref_map.binning['reco_coszen'].bin_edges.magnitude
        pisa3ref_map_to_plot['map'] = pisa3ref_map.hist
        pisa3ref_map_to_plot['map'] = pisa3ref_map_to_plot['map'].T

        cake_map = outputs[nukey]
        cake_map_to_plot = {}
        cake_map_to_plot['ebins'] = \
                cake_map.binning['reco_energy'].bin_edges.magnitude
        cake_map_to_plot['czbins'] = \
                cake_map.binning['reco_coszen'].bin_edges.magnitude
        cake_map_to_plot['map'] = cake_map.hist

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=pisa3ref_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv=ref_abv, new_abv='PISAV3',
            outdir=outdir,
            subdir='oscfit',
            stagename='baseline',
            servicename='full_mc',
            name=nukey,
            texname=outputs[nukey].tex)

        check_agreement(testname='PISAV3Ref-V3Now:%s %s'%(testname, nukey),
                        thresh_ratio=ratio_test_threshold,
                        ratio=max_diff_ratio,
                        thresh_diff=diff_test_threshold,
                        diff=max_diff)

    return pipeline


def compare_baseline(config, testname, outdir, oscfitfile):
    """
    Compare baseline output of PISA 3 with OscFit.
    """
    logging.debug('>> Working on baseline comparisons between both fitters.')
    logging.debug('>>> Doing %s test.'%testname)
    baseline_comparisons = from_file(oscfitfile)
    ref_abv='OscFit'

    pipeline = Pipeline(config)
    outputs = pipeline.get_outputs()

    for nukey in baseline_comparisons.keys():

        baseline_map_to_plot = baseline_comparisons[nukey]

        try:
            cake_map = outputs[nukey]
            texname = outputs[nukey].tex
        except:
            cake_map = outputs.combine_wildcard('*_%s'%nukey)
            if nukey == 'trck':
                texname = r'\rm{trck}'
            elif nukey == 'cscd':
                texname = r'\rm{cscd}'
        cake_map_to_plot = {}
        cake_map_to_plot['ebins'] = \
                cake_map.binning['reco_energy'].bin_edges.magnitude
        cake_map_to_plot['czbins'] = \
                cake_map.binning['reco_coszen'].bin_edges.magnitude
        cake_map_to_plot['map'] = cake_map.hist

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=baseline_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv=ref_abv, new_abv='PISAV3',
            outdir=outdir,
            subdir='oscfit',
            stagename=testname,
            servicename='baseline',
            name=nukey,
            texname=texname)

        print_agreement(testname='OscFit-V3:%s %s'%(testname, nukey),
                        ratio=max_diff_ratio)

    return pipeline


def compare_systematics(baseline_oscfit, config, testname, outdir, oscfitfile):
    """
    Compare systematic variations of PISA 3 and OscFit.
    """
    logging.debug('>> Working on systematic comparisons between both fitters.')
    logging.debug('>>> Doing %s test.'%testname)
    baseline_comparisons = from_file(baseline_oscfit)
    systematic_comparisons = from_file(oscfitfile)
    ref_abv='OscFit'

    pipeline = Pipeline(config)
    outputs = pipeline.get_outputs()

    for nukey in systematic_comparisons.keys():

        systematic_map_to_plot = systematic_comparisons[nukey]
        systematic_map_to_plot['map'] = systematic_map_to_plot['map'] + baseline_comparisons[nukey]['map']

        try:
            cake_map = outputs[nukey]
            texname = outputs[nukey].tex
        except:
            cake_map = outputs.combine_wildcard('*_%s'%nukey)
            if nukey == 'trck':
                texname = r'\rm{trck}'
            elif nukey == 'cscd':
                texname = r'\rm{cscd}'
        cake_map_to_plot = {}
        cake_map_to_plot['ebins'] = \
                cake_map.binning['reco_energy'].bin_edges.magnitude
        cake_map_to_plot['czbins'] = \
                cake_map.binning['reco_coszen'].bin_edges.magnitude
        cake_map_to_plot['map'] = cake_map.hist

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=systematic_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv=ref_abv, new_abv='PISAV3',
            outdir=outdir,
            subdir='oscfit',
            stagename=testname,
            servicename='systematic',
            name=nukey,
            texname=texname)

        print_agreement(testname='OscFit-V3:%s %s'%(testname, nukey),
                        ratio=max_diff_ratio)

    return pipeline


def main():
    parser = ArgumentParser(
        description=__doc__ + '''\n\nIf no test flags are specified, *all*
        tests will be run.'''
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
    parser.add_argument('--outdir', metavar='DIR', type=str, required=True,
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
                        help='''Set verbosity level; default is 1, so only -vv
                        or -vvv has an effect.''')
    args = parser.parse_args()
    args.v = min(1, args.v)
    set_verbosity(args.v)

    if not PYCUDA_AVAIL:
        raise RuntimeError('This system does not have a GPU so these tests '
                           'cannot be performed. Aborting')

    # Figure out which tests to do
    test_all = True
    if args.baseline or args.continuous_systematics:
        test_all = False

    # Perform baseline tests
    if args.baseline or test_all:
        pisa3_settings = os.path.join(
            'tests', 'settings', 'oscfit_fullmc_test.cfg'
        )
        pisa3_config = parse_pipeline_config(pisa3_settings)
        # First ensure that we still agree with Philipp's original
        # implementation of the event by event method. Here we can expect
        # agreement to machine precision.
        pisa3file = os.path.join(
            'tests', 'data', 'oscfit', 'PISAEventByEvent1X600Baseline.json'
        )
        pisa3file = find_resource(pisa3file)
        pisa3_pipeline = consistency_test(
            config=deepcopy(pisa3_config),
            testname='full-mc',
            outdir=args.outdir,
            pisa3file=pisa3file,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold)
        # If the above was passed we can now compare with OscFit. Agreement
        # is expected to better than 1 part in 1000.
        oscfitfile = os.path.join(
            'tests', 'data', 'oscfit', 'OscFit1X600Baseline.json'
        )
        oscfitfile = find_resource(oscfitfile)
        pisa3_pipeline = compare_baseline(
            config=deepcopy(pisa3_config),
            oscfitfile=oscfitfile,
            outdir=args.outdir,
            testname='full-mc'
        )
        pisa3_settings = os.path.join(
            'tests', 'settings', 'oscfit_standard_test.cfg'
        )
        pisa3_config = parse_pipeline_config(pisa3_settings)
        oscfitfile = os.path.join(
            'tests', 'data', 'oscfit', 'OscFit1X600Baseline.json'
        )
        oscfitfile = find_resource(oscfitfile)
        pisa3_pipeline = compare_baseline(
            config=deepcopy(pisa3_config),
            oscfitfile=oscfitfile,
            outdir=args.outdir,
            testname='standard'
        )

    # Perform continuous systematic tests
    if args.continuous_systematics or test_all:
        pisa3_settings = os.path.join(
            'tests', 'settings', 'oscfit_fullmc_test.cfg'
        )
        pisa3_config = parse_pipeline_config(pisa3_settings)
        # Define names and variations of systematics expected
        # Names are different between PISA and OscFit in some cases
        continuous_systematics = {'Genie_Ma_QE': 'axm_qe',
                                  'Genie_Ma_RES': 'axm_res',
                                  'deltam31': 'dm31',
                                  'theta13': 'theta13',
                                  'theta23': 'theta23',
                                  'delta_index': 'gamma',
                                  'nutau_cc_norm': 'norm_tau',
                                  'nu_nc_norm': 'norm_nc',
                                  'nue_numu_ratio': 'norm_e',
                                  'aeff_scale': 'norm_nu'}
        variations = {'Genie_Ma_RES': 1.0,
                      'Genie_Ma_QE' : 1.0,
                      'deltam31': 0.2e-3*ureg.eV**2,
                      'theta13': 0.008 * ureg.rad,
                      'theta23': 0.1 * ureg.rad,
                      'delta_index': 0.1,
                      'nutau_cc_norm': 0.5,
                      'nu_nc_norm': 0.2,
                      'nue_numu_ratio': 0.05,
                      'aeff_scale': 0.12}
        texnames = {'Genie_Ma_QE': 'Genie-Ma-QE',
                    'Genie_Ma_RES': 'Genie-Ma-RES',
                    'deltam31': 'dm31',
                    'theta13': 'theta13',
                    'theta23': 'theta23',
                    'delta_index': 'delta-index',
                    'nutau_cc_norm': 'nutau-cc-norm',
                    'nu_nc_norm': 'nu-nc-norm',
                    'nue_numu_ratio': 'nue-numu-ratio',
                    'aeff_scale': 'aeff-scale'}
        baseline_oscfitfile = os.path.join(
            'tests', 'data', 'oscfit', 'OscFit1X600Baseline.json'
        )
        baseline_oscfitfile = find_resource(baseline_oscfitfile)
        for sys in continuous_systematics.keys():
            config_to_modify = deepcopy(pisa3_config)
            # Need the set of params so we can modify them
            k = [k for k in config_to_modify.keys() if k[0] == 'mc'][0]
            params = config_to_modify[k]['params'].params
            params[sys] = params[sys].value + variations[sys]

            oscfitfile = os.path.join(
                'tests', 'data', 'oscfit',
                'OscFit1X600Diff%s.json'%continuous_systematics[sys]
            )
            oscfitfile = find_resource(oscfitfile)
            pisa3_pipeline = compare_systematics(
                baseline_oscfit=baseline_oscfitfile,
                config=deepcopy(config_to_modify),
                oscfitfile=oscfitfile,
                outdir=args.outdir,
                testname='%s'%texnames[sys]
            )


if __name__ == '__main__':
    main()
