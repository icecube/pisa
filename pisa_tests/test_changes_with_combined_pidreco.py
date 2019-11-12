#! /usr/bin/env python

"""
Run a set of tests on the PISA 3 pipeline to check the effect of combining Reco
and PID in to a single stage. Output is tested against both the standard PISA
and a full event-by-event treatment from OscFit in various configurations.
"""


from __future__ import absolute_import, division

from argparse import ArgumentParser
from copy import deepcopy
import os
import numpy as np

from pisa import ureg, Q_
from pisa.core.pipeline import Pipeline
from pisa.utils.config_parser import parse_pipeline_config
from pisa.utils.fileio import from_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.tests import print_event_rates, plot_comparisons


__all__ = ['FMT',
           'compare_pisa_self', 'compare_5stage', 'compare_4stage',
           'do_comparisons', 'oversample_config',
           'main']

__author__ = 'S. Wren'

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


FMT = 'png'


def compare_pisa_self(config1, config2, testname1, testname2, outdir):
    """Compare baseline output of PISA 3 with a different version of itself"""
    logging.debug('>> Comparing %s with %s (both PISA)'%(testname1,testname2))

    pipeline1 = Pipeline(config1)
    outputs1 = pipeline1.get_outputs()
    pipeline2 = Pipeline(config2)
    outputs2 = pipeline2.get_outputs()

    if '5-stage' in testname1:
        cake1_trck_map = outputs1.combine_wildcard('*_trck')
        cake1_cscd_map = outputs1.combine_wildcard('*_cscd')
        cake1_trck_map_to_plot = {}
        cake1_trck_map_to_plot['ebins'] = \
            cake1_trck_map.binning['reco_energy'].bin_edges.magnitude
        cake1_trck_map_to_plot['czbins'] = \
            cake1_trck_map.binning['reco_coszen'].bin_edges.magnitude
        cake1_trck_map_to_plot['map'] = cake1_trck_map.hist
        cake1_trck_events = np.sum(cake1_trck_map_to_plot['map'])
        cake1_cscd_map_to_plot = {}
        cake1_cscd_map_to_plot['ebins'] = \
            cake1_cscd_map.binning['reco_energy'].bin_edges.magnitude
        cake1_cscd_map_to_plot['czbins'] = \
            cake1_cscd_map.binning['reco_coszen'].bin_edges.magnitude
        cake1_cscd_map_to_plot['map'] = cake1_cscd_map.hist
        cake1_cscd_events = np.sum(cake1_cscd_map_to_plot['map'])
    elif '4-stage' in testname1:
        cake1_both_map = outputs1.combine_wildcard('*')
        cake1_trck_map_to_plot = {}
        cake1_trck_map_to_plot['ebins'] = \
            cake1_both_map.binning['reco_energy'].bin_edges.magnitude
        cake1_trck_map_to_plot['czbins'] = \
            cake1_both_map.binning['reco_coszen'].bin_edges.magnitude
        cake1_trck_map_to_plot['map'] = \
            cake1_both_map.split(
                dim='pid',
                bin='trck'
            ).hist
        cake1_trck_events = np.sum(cake1_trck_map_to_plot['map'])
        cake1_cscd_map_to_plot = {}
        cake1_cscd_map_to_plot['ebins'] = \
            cake1_both_map.binning['reco_energy'].bin_edges.magnitude
        cake1_cscd_map_to_plot['czbins'] = \
            cake1_both_map.binning['reco_coszen'].bin_edges.magnitude
        cake1_cscd_map_to_plot['map'] = \
            cake1_both_map.split(
                dim='pid',
                bin='cscd'
            ).hist
        cake1_cscd_events = np.sum(cake1_cscd_map_to_plot['map'])
    else:
        raise ValueError("Should be comparing 4-stage or 5-stage PISAs.")

    if '5-stage' in testname2:
        cake2_trck_map = outputs2.combine_wildcard('*_trck')
        cake2_cscd_map = outputs2.combine_wildcard('*_cscd')
        cake2_trck_map_to_plot = {}
        cake2_trck_map_to_plot['ebins'] = \
            cake2_trck_map.binning['reco_energy'].bin_edges.magnitude
        cake2_trck_map_to_plot['czbins'] = \
            cake2_trck_map.binning['reco_coszen'].bin_edges.magnitude
        cake2_trck_map_to_plot['map'] = cake2_trck_map.hist
        cake2_trck_events = np.sum(cake2_trck_map_to_plot['map'])
        cake2_cscd_map_to_plot = {}
        cake2_cscd_map_to_plot['ebins'] = \
            cake2_cscd_map.binning['reco_energy'].bin_edges.magnitude
        cake2_cscd_map_to_plot['czbins'] = \
            cake2_cscd_map.binning['reco_coszen'].bin_edges.magnitude
        cake2_cscd_map_to_plot['map'] = cake2_cscd_map.hist
        cake2_cscd_events = np.sum(cake2_cscd_map_to_plot['map'])
    elif '4-stage' in testname2:
        cake2_both_map = outputs2.combine_wildcard('*')
        cake2_trck_map_to_plot = {}
        cake2_trck_map_to_plot['ebins'] = \
            cake2_both_map.binning['reco_energy'].bin_edges.magnitude
        cake2_trck_map_to_plot['czbins'] = \
            cake2_both_map.binning['reco_coszen'].bin_edges.magnitude
        cake2_trck_map_to_plot['map'] = \
            cake2_both_map.split(
                dim='pid',
                bin='trck'
            ).hist
        cake2_trck_events = np.sum(cake2_trck_map_to_plot['map'])
        cake2_cscd_map_to_plot = {}
        cake2_cscd_map_to_plot['ebins'] = \
            cake2_both_map.binning['reco_energy'].bin_edges.magnitude
        cake2_cscd_map_to_plot['czbins'] = \
            cake2_both_map.binning['reco_coszen'].bin_edges.magnitude
        cake2_cscd_map_to_plot['map'] = \
            cake2_both_map.split(
                dim='pid',
                bin='cscd'
            ).hist
        cake2_cscd_events = np.sum(cake2_cscd_map_to_plot['map'])
    else:
        raise ValueError("Should be comparing 4-stage or 5-stage PISAs.")

    max_diff_ratio, max_diff = plot_comparisons(
        ref_map=cake1_trck_map_to_plot,
        new_map=cake2_trck_map_to_plot,
        ref_abv=testname1,
        new_abv=testname2,
        outdir=outdir,
        subdir='recopidcombinedchecks',
        stagename=None,
        servicename='recopid',
        name='trck',
        texname=r'\rm{trck}',
        shorttitles=True,
        ftype=FMT
    )

    max_diff_ratio, max_diff = plot_comparisons(
        ref_map=cake1_cscd_map_to_plot,
        new_map=cake2_cscd_map_to_plot,
        ref_abv=testname1,
        new_abv=testname2,
        outdir=outdir,
        subdir='recopidcombinedchecks',
        stagename=None,
        servicename='recopid',
        name='cscd',
        texname=r'\rm{cscd}',
        shorttitles=True,
        ftype=FMT
    )

    print_event_rates(
        testname1=testname1,
        testname2=testname2,
        kind='trck',
        map1_events=cake1_trck_events,
        map2_events=cake2_trck_events
    )
    print_event_rates(
        testname1=testname1,
        testname2=testname2,
        kind='cscd',
        map1_events=cake1_cscd_events,
        map2_events=cake2_cscd_events
    )

    print_event_rates(
        testname1=testname1,
        testname2=testname2,
        kind='all',
        map1_events=cake1_trck_events+cake1_cscd_events,
        map2_events=cake2_trck_events+cake2_cscd_events
    )

    return pipeline2


def compare_5stage(config, testname, outdir, oscfitfile):
    """Compare 5 stage output of PISA 3 with OscFit."""
    logging.debug('>> Working on baseline comparisons between both fitters.')
    logging.debug('>>> Doing %s test.'%testname)
    baseline_comparisons = from_file(oscfitfile)
    ref_abv='OscFit'

    pipeline = Pipeline(config)
    outputs = pipeline.get_outputs()

    total_pisa_events = 0.0
    total_oscfit_events = 0.0

    for nukey in baseline_comparisons.keys():

        baseline_map_to_plot = baseline_comparisons[nukey]
        oscfit_events = np.sum(baseline_map_to_plot['map'])

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
        pisa_events = np.sum(cake_map_to_plot['map'])

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=baseline_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv=ref_abv,
            new_abv=testname,
            outdir=outdir,
            subdir='recopidcombinedchecks',
            stagename=None,
            servicename='baseline',
            name=nukey,
            texname=texname,
            shorttitles=True,
            ftype=FMT
        )

        print_event_rates(
            testname1=testname,
            testname2='OscFit',
            kind=nukey,
            map1_events=pisa_events,
            map2_events=oscfit_events
        )

        total_pisa_events += pisa_events
        total_oscfit_events += oscfit_events

    print_event_rates(
            testname1=testname,
            testname2='OscFit',
            kind='all',
            map1_events=total_pisa_events,
            map2_events=total_oscfit_events
        )

    return pipeline


def compare_4stage(config, testname, outdir, oscfitfile):
    """
    Compare 4 stage output of PISA 3 with OscFit.
    """
    logging.debug('>> Working on baseline comparisons between both fitters.')
    logging.debug('>>> Doing %s test.'%testname)
    baseline_comparisons = from_file(oscfitfile)
    ref_abv='OscFit'

    pipeline = Pipeline(config)
    outputs = pipeline.get_outputs()

    total_pisa_events = 0.0
    total_oscfit_events = 0.0

    for nukey in baseline_comparisons.keys():

        baseline_map_to_plot = baseline_comparisons[nukey]
        oscfit_events = np.sum(baseline_map_to_plot['map'])

        cake_map = outputs.combine_wildcard('*')
        cake_map_to_plot = {}
        cake_map_to_plot['ebins'] = \
                cake_map.binning['reco_energy'].bin_edges.magnitude
        cake_map_to_plot['czbins'] = \
                cake_map.binning['reco_coszen'].bin_edges.magnitude
        if nukey == 'trck':
            texname = r'\rm{trck}'
            cake_map_to_plot['map'] = \
                cake_map.split(
                    dim='pid',
                    bin='trck'
                ).hist
        elif nukey == 'cscd':
            texname = r'\rm{cscd}'
            cake_map_to_plot['map'] = \
                cake_map.split(
                    dim='pid',
                    bin='cscd'
                ).hist
        pisa_events = np.sum(cake_map_to_plot['map'])

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=baseline_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv=ref_abv,
            new_abv=testname,
            outdir=outdir,
            subdir='recopidcombinedchecks',
            stagename=None,
            servicename='baseline',
            name=nukey,
            texname=texname,
            shorttitles=True,
            ftype=FMT
        )

        print_event_rates(
            testname1=testname,
            testname2='OscFit',
            kind=nukey,
            map1_events=pisa_events,
            map2_events=oscfit_events
        )

        total_pisa_events += pisa_events
        total_oscfit_events += oscfit_events

    print_event_rates(
        testname1=testname,
        testname2='OscFit',
        kind='all',
        map1_events=total_pisa_events,
        map2_events=total_oscfit_events
    )

    return pipeline


def do_comparisons(config1, config2, oscfitfile,
                   testname1, testname2, outdir):
    pisa_recopid_pipeline = compare_pisa_self(
        config1=config1,
        config2=config2,
        testname1=testname1,
        testname2=testname2,
        outdir=outdir
    )
    pisa_standard_pipeline = compare_5stage(
        config=config1,
        testname=testname1,
        outdir=outdir,
        oscfitfile=oscfitfile
    )
    pisa_recopid_pipeline = compare_4stage(
        config=config2,
        testname=testname2,
        outdir=outdir,
        oscfitfile=oscfitfile
    )


def oversample_config(base_config, oversample):
    for stage in base_config.keys():
        for obj in base_config[stage].keys():
            if 'binning' in obj:
                if 'true' in base_config[stage][obj].names[0]:
                    base_config[stage][obj] = \
                            base_config[stage][obj].oversample(oversample)
    return base_config


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--oversampling', action='store_true', default=False,
                        help='''Run oversampling tests i.e. use a finer binning
                        through the truth stages in addition to the standard
                        tests. You must flag this if you want it.''')
    parser.add_argument('--weighting', type=str, default=None,
                        help='''Name of the weighting field to use in the
                        comparisons. This must correspond to a field in the
                        events files being used.''')
    parser.add_argument('--outdir', metavar='DIR', type=str, required=True,
                        help='''Store all output plots to this directory. If
                        they don't exist, the script will make them, including
                        all subdirectories.''')
    parser.add_argument('-v', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()
    set_verbosity(args.v)

    known_weights = [None, 'weighted_aeff']

    if args.weighting not in known_weights:
        logging.warning(
            '''%s weighting field not known to be in events file.
            Tests may not work in this case!'''%args.weighting
        )

    # Want these for all tests
    pisa_standard_settings = os.path.join(
        'tests', 'settings', 'recopid_full_pipeline_5stage_test.cfg'
    )
    pisa_standard_config = parse_pipeline_config(pisa_standard_settings)
    pisa_recopid_settings = os.path.join(
        'tests', 'settings', 'recopid_full_pipeline_4stage_test.cfg'
    )
    pisa_recopid_config = parse_pipeline_config(pisa_recopid_settings)

    # Add weighting to pipeline according to user input
    # Need to add it to both reco and PID for standard config
    reco_k = [k for k in pisa_standard_config.keys() \
              if k[0] == 'reco'][0]
    standard_reco_params = \
        pisa_standard_config[reco_k]['params'].params
    standard_reco_params.reco_weights_name.value = args.weighting
    pid_k = [k for k in pisa_standard_config.keys() \
             if k[0] == 'pid'][0]
    standard_pid_params = \
        pisa_standard_config[pid_k]['params'].params
    standard_pid_params.pid_weights_name.value = args.weighting
    # Just needs adding to reco for joined recopid config
    recopid_k = [k for k in pisa_recopid_config.keys() \
                 if k[0] == 'reco'][0]
    recopid_reco_params = \
        pisa_recopid_config[recopid_k]['params'].params
    recopid_reco_params.reco_weights_name.value = args.weighting

    # Load OscFit file for comparisons
    oscfitfile = os.path.join(
        'tests', 'data', 'oscfit', 'OscFit1X600Baseline.json'
    )

    # Rename in this instance now so it's clearer in logs and filenames
    if args.weighting == None:
        args.weighting = 'unweighted'

    logging.info("<<<< %s reco/pid Transformations >>>>"%args.weighting)
    # Perform baseline tests
    logging.info("<< No oversampling >>")
    do_comparisons(
        config1=deepcopy(pisa_standard_config),
        config2=deepcopy(pisa_recopid_config),
        oscfitfile=oscfitfile,
        testname1='5-stage-%s'%args.weighting,
        testname2='4-stage-%s'%args.weighting,
        outdir=args.outdir
    )

    # Perform oversampled tests
    if args.oversampling:
        oversamples = [5,10,20,50]
        for oversample in oversamples:
            pisa_standard_oversampled_config = oversample_config(
                base_config=deepcopy(pisa_standard_config),
                oversample=oversample
            )
            pisa_recopid_oversampled_config = oversample_config(
                base_config=deepcopy(pisa_recopid_config),
                oversample=oversample
            )
            logging.info("<< Oversampling by %i >>"%(oversample))
            do_comparisons(
                config1=deepcopy(pisa_standard_oversampled_config),
                config2=deepcopy(pisa_recopid_oversampled_config),
                oscfitfile=oscfitfile,
                testname1='5-stage-%s-Oversampled%i'%(args.weighting,
                                                      oversample),
                testname2='4-stage-%s-Oversampled%i'%(args.weighting,
                                                      oversample),
                outdir=args.outdir
            )

main.__doc__ = __doc__


if __name__ == '__main__':
    main()
