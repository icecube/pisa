#! /usr/bin/env python
# author: S.Wren
# date:   March 20, 2016
"""
Run a set of tests on the PISA pipeline against benchmark PISA 2 data. If no
test flags are specified, *all* tests will be run.

This script should always be run when you make any major modifications (and
prior to submitting a pull request) to be sure nothing has broken.

If you find this script does not work, please either fix it or report it! In
general, this will signify you have "changed" something, somehow in the basic
functionality which you should understand!

If an output directory is specified, a set of plots will be output for you to
visually inspect.
"""


from argparse import ArgumentParser
from copy import deepcopy
import os
import sys

import numpy as np

from pisa import FTYPE, PYCUDA_AVAIL
from pisa.core.map import Map, MapSet
from pisa.core.pipeline import Pipeline
from pisa.utils.fileio import from_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.config_parser import parse_pipeline_config
from pisa.utils.tests import check_agreement, plot_map_comparisons, pisa2_map_to_pisa3_map


__all__ = ['PID_FAIL_MESSAGE', 'PID_PASS_MESSAGE',
           'compare_flux', 'compare_osc', 'compare_aeff', 'compare_reco',
           'compare_pid', 'compare_flux_full', 'compare_osc_full',
           'compare_aeff_full', 'compare_reco_full', 'compare_pid_full',
           'parse_args', 'main']


PID_FAIL_MESSAGE = (
    "PISA 2 reference file has a known bug in its PID (fraction for each PID"
    " signature is taken as total of events ID'd for each signatures, and not"
    " total of all events, as it should be. Therefore, if PISA 3 / full"
    " pipeline PID agrees with PISA 2 (same) for DeepCore (which does not have"
    " mutually-exclusive PID definition), then PISA 3 is in error."
)

PID_PASS_MESSAGE = (
    "**NOTE** Ignore above FAIL message; PID passed, since PISA 3 *should"
    " disagree* with PISA 2 due to known bug in PISA 2 (doesn't normalize"
    " correctly if PID categories don't include all events)"
)


def compare_flux(config, servicename, pisa2file, systname,
                 outdir, ratio_test_threshold, diff_test_threshold):
    """Compare flux stages run in isolation with dummy inputs"""

    logging.debug('>> Working on flux stage comparisons')
    logging.debug('>>> Checking %s service'%servicename)
    test_service = servicename

    k = [k for k in config.keys() if k[0] == 'flux'][0]
    params = config[k]['params'].params

    if systname is not None:
        logging.debug('>>> Checking %s systematic'%systname)
        test_syst = systname
        try:
            params[systname] = params[systname].value + \
                    params[systname].prior.stddev
        except:
            params[systname] = 1.25*params[systname].value

        pisa2file = (pisa2file.split('.json')[0]
                     + '-%s%.2f.json' %(systname, params[systname].value))
        servicename += ('-%s%.2f' %(systname, params[systname].value))
    else:
        logging.debug('>>> Checking baseline')
        test_syst = 'baseline'

    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    outputs = stage.get_outputs()
    pisa2_comparisons = from_file(pisa2file)

    for nukey in pisa2_comparisons.keys():
        if 'nu' not in nukey:
            continue

        pisa_map_to_plot = pisa2_map_to_pisa3_map(
            pisa2_map = pisa2_comparisons[nukey],
            ebins_name = 'true_energy',
            czbins_name = 'true_coszen'
        )

        if '_' in nukey:
            if nukey.split('_')[1] == 'bar':
                new_nukey = ""
                for substr in nukey.split('_'):
                    new_nukey += substr
                nukey = new_nukey

        cake_map_to_plot = outputs[nukey]

        max_diff_ratio, max_diff = plot_map_comparisons(
            ref_map=pisa_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv='PISAV2', new_abv='PISAV3',
            outdir=outdir,
            subdir='flux',
            stagename='flux',
            servicename=servicename,
            name=nukey,
            texname=outputs[nukey].tex
        )

        check_agreement(
            testname='PISAV3-PISAV2 flux:%s %s %s'
                %(test_service, test_syst, nukey),
            thresh_ratio=ratio_test_threshold,
            ratio=max_diff_ratio,
            thresh_diff=diff_test_threshold,
            diff=max_diff
        )

    return pipeline


def compare_osc(config, servicename, pisa2file, systname,
                outdir, ratio_test_threshold, diff_test_threshold):
    """Compare osc stages run in isolation with dummy inputs"""

    logging.debug('>> Working on osc stage comparisons')
    logging.debug('>>> Checking %s service'%servicename)
    test_service = servicename

    k = [k for k in config.keys() if k[0] == 'osc'][0]
    params = config[k]['params'].params

    if systname is not None:
        logging.debug('>>> Checking %s systematic'%systname)
        test_syst = systname
        try:
            params[systname] = \
                    params[systname].value + \
                    params[systname].prior.stddev
        except:
            params[systname] = \
                    1.25*params[systname].value

        if params[systname].magnitude < 0.01:
            systval = '%e'%params[systname].magnitude
            systval = systval[0:4]
        else:
            systval = '%.2f'%params[systname].magnitude

        pisa2file = pisa2file.split('.json')[0] + \
                '-%s%s.json' %(systname, systval)
        servicename += '-%s%s' %(systname, systval)
    else:
        logging.debug('>>> Checking baseline')
        test_syst = 'baseline'

    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    input_maps = []
    for name in stage.input_names:
        hist = np.ones(stage.input_binning.shape)
        input_maps.append(
            Map(name=name, hist=hist, binning=stage.input_binning)
        )
    outputs = stage.get_outputs(
        inputs=MapSet(maps=input_maps, name='ones', hash=1)
    )
    pisa2_comparisons = from_file(pisa2file)

    for nukey in pisa2_comparisons.keys():
        if 'nu' not in nukey:
            continue

        pisa_map_to_plot = pisa2_map_to_pisa3_map(
            pisa2_map = pisa2_comparisons[nukey],
            ebins_name = 'true_energy',
            czbins_name = 'true_coszen'
        )

        if '_' in nukey:
            if nukey.split('_')[1] == 'bar':
                new_nukey = ""
                for substr in nukey.split('_'):
                    new_nukey += substr
                nukey = new_nukey

        cake_map_to_plot = outputs[nukey]

        max_diff_ratio, max_diff = plot_map_comparisons(
            ref_map=pisa_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv='PISAV2', new_abv='PISAV3',
            outdir=outdir,
            subdir='osc',
            stagename='osc',
            servicename=servicename,
            name=nukey,
            texname=outputs[nukey].tex
        )

        check_agreement(
            testname='PISAV3-PISAV2 osc:%s %s %s'
                %(test_service, test_syst, nukey),
            thresh_ratio=ratio_test_threshold,
            ratio=max_diff_ratio,
            thresh_diff=diff_test_threshold,
            diff=max_diff
        )

    return pipeline


def compare_aeff(config, servicename, pisa2file, systname,
                 outdir, ratio_test_threshold, diff_test_threshold):
    """Compare aeff stages run in isolation with dummy inputs"""

    logging.debug('>> Working on aeff stage comparisons')
    logging.debug('>>> Checking %s service'%servicename)
    test_service = servicename

    k = [k for k in config.keys() if k[0] == 'aeff'][0]
    params = config[k]['params'].params

    if systname is not None:
        logging.debug('>>> Checking %s systematic'%systname)
        test_syst = systname
        try:
            params[systname] = \
                    params[systname].value + \
                    params[systname].prior.stddev
        except:
            params[systname] = \
                    1.25*params[systname].value

        pisa2file = pisa2file.split('.json')[0] + \
                '-%s%.2f.json' \
                %(systname, params[systname].value)
        servicename += '-%s%.2f' \
                %(systname, params[systname].value)
    else:
        logging.debug('>>> Checking baseline')
        test_syst = 'baseline'

    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    input_maps = []
    for name in stage.input_names:
        hist = np.ones(stage.input_binning.shape)
        input_maps.append(
            Map(name=name, hist=hist, binning=stage.input_binning)
        )
    outputs = stage.get_outputs(inputs=MapSet(maps=input_maps, name='ones',
                                              hash=1))
    pisa2_comparisons = from_file(pisa2file)
    for nukey in pisa2_comparisons.keys():
        if 'nu' not in nukey:
            continue

        for intkey in pisa2_comparisons[nukey].keys():
            if '_' in nukey:
                if nukey.split('_')[1] == 'bar':
                    new_nukey = ""
                    for substr in nukey.split('_'):
                        new_nukey += substr
            else:
                new_nukey = nukey
            cakekey = new_nukey + '_' + intkey
            pisa_map_to_plot = pisa2_map_to_pisa3_map(
                pisa2_map = pisa2_comparisons[nukey][intkey],
                ebins_name = 'true_energy',
                czbins_name = 'true_coszen'
            )

            cake_map_to_plot = outputs[cakekey]

            max_diff_ratio, max_diff = plot_map_comparisons(
                ref_map=pisa_map_to_plot,
                new_map=cake_map_to_plot,
                ref_abv='PISAV2', new_abv='PISAV3',
                outdir=outdir,
                subdir='aeff',
                stagename='aeff',
                servicename=servicename,
                name=cakekey,
                texname=outputs[cakekey].tex,
            )

            check_agreement(
                testname='PISAV3-PISAV2 aeff:%s %s %s'
                    %(test_service, test_syst, cakekey),
                thresh_ratio=ratio_test_threshold,
                ratio=max_diff_ratio,
                thresh_diff=diff_test_threshold,
                diff=max_diff
            )

    return pipeline


def compare_reco(config, servicename, pisa2file, systname,
                 outdir, ratio_test_threshold,
                 diff_test_threshold):
    """Compare reco stages run in isolation with dummy inputs"""
    logging.debug('>> Working on reco stage comparisons')
    logging.debug('>>> Checking %s service'%servicename)
    test_service = servicename

    k = [k for k in config.keys() if k[0] == 'reco'][0]
    params = config[k]['params'].params

    if systname is not None:
        logging.debug('>>> Checking %s systematic'%systname)
        test_syst = systname
        try:
            params[systname] = \
                    params[systname].value + \
                    params[systname].prior.stddev
        except:
            params[systname] = \
                    1.25*params[systname].value

        pisa2file = pisa2file.split('.json')[0] + \
                '-%s%.2f.json' \
                %(systname, params[systname].value)
        servicename += '-%s%.2f' \
                %(systname, params[systname].value)
    else:
        logging.debug('>>> Checking baseline')
        test_syst = 'baseline'
        
    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    input_maps = []
    for name in stage.input_names:
        hist = np.ones(stage.input_binning.shape)
        if 'nc' in name:
            # NC is combination of three flavours
            hist *= 3.0
        input_maps.append(
            Map(name=name, hist=hist, binning=stage.input_binning)
        )
    outputs = stage.get_outputs(inputs=MapSet(maps=input_maps, name='ones',
                                              hash=1))
    nue_nuebar_cc = outputs.combine_re(r'nue(bar){0,1}_cc')
    numu_numubar_cc = outputs.combine_re(r'numu(bar){0,1}_cc')
    nutau_nutaubar_cc = outputs.combine_re(r'nutau(bar){0,1}_cc')
    nuall_nuallbar_nc = outputs.combine_re(r'nu.*_nc')

    modified_cake_outputs = {
        'nue_cc': nue_nuebar_cc,
        'numu_cc': numu_numubar_cc,
        'nutau_cc': nutau_nutaubar_cc,
        'nuall_nc': nuall_nuallbar_nc
    }

    pisa2_comparisons = from_file(pisa2file)

    for nukey in pisa2_comparisons.keys():
        if 'nu' not in nukey:
            continue

        pisa_map_to_plot = pisa2_map_to_pisa3_map(
            pisa2_map = pisa2_comparisons[nukey],
            ebins_name = 'reco_energy',
            czbins_name = 'reco_coszen'
        )

        if '_' in nukey:
            if nukey.split('_')[1] == 'bar':
                new_nukey = ""
                for substr in nukey.split('_'):
                    new_nukey += substr
                nukey = new_nukey

        cake_map_to_plot = modified_cake_outputs[nukey]

        max_diff_ratio, max_diff = plot_map_comparisons(
            ref_map=pisa_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv='PISAV2', new_abv='PISAV3',
            outdir=outdir,
            subdir='reco',
            stagename='reco',
            servicename=servicename,
            name=nukey,
            texname=outputs[nukey].tex
        )

        check_agreement(
            testname='PISAV3-PISAV2 reco:%s %s %s' %(test_service, test_syst,
                                                     nukey),
            thresh_ratio=ratio_test_threshold,
            ratio=max_diff_ratio,
            thresh_diff=diff_test_threshold,
            diff=max_diff
        )

    return pipeline


def compare_pid(config, servicename, pisa2file, outdir, ratio_test_threshold,
                diff_test_threshold):
    """Compare pid stages run in isolation with dummy inputs"""
    logging.debug('>> Working on pid stage comparisons')
    logging.debug('>>> Checking %s service'%servicename)
    test_service = servicename

    logging.debug('>>> Checking baseline')
    test_syst = 'baseline'
    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    input_maps = []
    for name in stage.input_names:
        hist = np.ones(stage.input_binning.shape)
        # Input names still has nu and nubar separated.
        # PISA 2 is not expecting this
        hist *= 0.5
        input_maps.append(
            Map(name=name, hist=hist, binning=stage.input_binning)
        )
    outputs = stage.get_outputs(inputs=MapSet(maps=input_maps, name='ones',
                                              hash=1))

    pisa2_comparisons = from_file(pisa2file)
    pisa_trck = pisa2_map_to_pisa3_map(
        pisa2_map = pisa2_comparisons['trck'],
        ebins_name = 'reco_energy',
        czbins_name = 'reco_coszen'
    )
    pisa_cscd = pisa2_map_to_pisa3_map(
        pisa2_map = pisa2_comparisons['cscd'],
        ebins_name = 'reco_energy',
        czbins_name = 'reco_coszen'
    )

    # Do check if output is exactly the same as old PISA 2 with track and
    # cascade separated or is new PISA 3 where PID is a binning dimension.
    bins_old = [('trck' in s) or ('cscd' in s) for s in outputs.names]
    if np.all(np.array(bins_old)):
        # In this case we can get the two PID maps using the wildcard.
        cake_trck = outputs.combine_wildcard('*_trck')
        cake_cscd = outputs.combine_wildcard('*_cscd')

    else:
        # If not we must split along the PID dimension.
        total_outputs = outputs.combine_wildcard('*')
        cake_trck = total_outputs.slice(pid=1).squeeze()
        cake_cscd = total_outputs.slice(pid=0).squeeze()

    max_diff_ratio, max_diff= plot_map_comparisons(
        ref_map=pisa_cscd,
        new_map=cake_cscd,
        ref_abv='PISAV2', new_abv='PISAV3',
        outdir=outdir,
        subdir='pid',
        stagename='pid',
        servicename=servicename,
        name='cscd',
        texname=r'{\rm cscd}'
    )

    check_agreement(
        testname='PISAV3-PISAV2 pid:%s %s cscd'
        %(test_service, test_syst),
        thresh_ratio=ratio_test_threshold,
        ratio=max_diff_ratio,
        thresh_diff=diff_test_threshold,
        diff=max_diff
    )

    max_diff_ratio, max_diff = plot_map_comparisons(
        ref_map=pisa_trck,
        new_map=cake_trck,
        ref_abv='PISAV2', new_abv='PISAV3',
        outdir=outdir,
        subdir='pid',
        stagename='pid',
        servicename=servicename,
        name='trck',
        texname=r'{\rm trck}'
    )

    check_agreement(
        testname='PISAV3-PISAV2 pid:%s %s trck'
        %(test_service, test_syst),
        thresh_ratio=ratio_test_threshold,
        ratio=max_diff_ratio,
        thresh_diff=diff_test_threshold,
        diff=max_diff
    )

    return pipeline


def compare_flux_full(cake_maps, pisa_maps, outdir, ratio_test_threshold,
                      diff_test_threshold):
    """Compare a fully configured pipeline (with stages flux, osc, aeff, reco,
    and pid) through the flux stage.

    """
    logging.debug('>> Working on full pipeline comparisons')
    logging.debug('>>> Checking to end of flux stage')
    test_service = 'honda'

    for nukey in pisa_maps.keys():
        if 'nu' not in nukey:
            continue

        pisa_map_to_plot = pisa2_map_to_pisa3_map(
            pisa2_map = pisa_maps[nukey],
            ebins_name = 'true_energy',
            czbins_name = 'true_coszen',
            is_log=False
        )

        if '_' in nukey:
            if nukey.split('_')[1] == 'bar':
                new_nukey = ""
                for substr in nukey.split('_'):
                    new_nukey += substr
                nukey = new_nukey

        cake_map_to_plot = cake_maps[nukey]

        max_diff_ratio, max_diff = plot_map_comparisons(
            ref_map=pisa_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv='PISAV2', new_abv='PISAV3',
            outdir=outdir,
            subdir='fullpipeline',
            stagename='flux',
            servicename=test_service,
            name=nukey,
            texname=cake_maps[nukey].tex
        )

        check_agreement(
            testname='PISAV3-PISAV2 full pipeline through flux:%s %s'
                %(test_service, nukey),
            thresh_ratio=ratio_test_threshold,
            ratio=max_diff_ratio,
            thresh_diff=diff_test_threshold,
            diff=max_diff
        )


def compare_osc_full(cake_maps, pisa_maps, outdir, ratio_test_threshold,
                     diff_test_threshold):
    """Compare a fully configured pipeline (with stages flux, osc, aeff, reco,
    and pid) through the osc stage.

    """
    logging.debug('>> Working on full pipeline comparisons')
    logging.debug('>>> Checking to end of osc stage')
    test_service = 'prob3cpu'

    for nukey in pisa_maps.keys():
        if 'nu' not in nukey:
            continue

        pisa_map_to_plot = pisa2_map_to_pisa3_map(
            pisa2_map = pisa_maps[nukey],
            ebins_name = 'true_energy',
            czbins_name = 'true_coszen',
            is_log=False
        )

        if '_' in nukey:
            if nukey.split('_')[1] == 'bar':
                new_nukey = ""
                for substr in nukey.split('_'):
                    new_nukey += substr
                nukey = new_nukey

        cake_map_to_plot = cake_maps[nukey]

        max_diff_ratio, max_diff = plot_map_comparisons(
            ref_map=pisa_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv='PISAV2', new_abv='PISAV3',
            outdir=outdir,
            subdir='fullpipeline',
            stagename='osc',
            servicename=test_service,
            name=nukey,
            texname=cake_maps[nukey].tex
        )

        check_agreement(
            testname='PISAV3-PISAV2 full pipeline through osc:%s %s'
                %(test_service, nukey),
            thresh_ratio=ratio_test_threshold,
            ratio=max_diff_ratio,
            thresh_diff=diff_test_threshold,
            diff=max_diff
        )


def compare_aeff_full(cake_maps, pisa_maps, outdir, ratio_test_threshold,
                      diff_test_threshold):
    """Compare a fully configured pipeline (with stages flux, osc, aeff, reco,
    and pid) through the aeff stage.

    """
    logging.debug('>> Working on full pipeline comparisons')
    logging.debug('>>> Checking to end of aeff stage')
    test_service = 'hist_1X585'

    for nukey in pisa_maps.keys():
        if 'nu' not in nukey:
            continue

        for intkey in pisa_maps[nukey].keys():
            if '_' in nukey:
                if nukey.split('_')[1] == 'bar':
                    new_nukey = ""
                    for substr in nukey.split('_'):
                        new_nukey += substr
            else:
                new_nukey = nukey
            cakekey = new_nukey + '_' + intkey
            pisa_map_to_plot = pisa2_map_to_pisa3_map(
                pisa2_map = pisa_maps[nukey][intkey],
                ebins_name = 'true_energy',
                czbins_name = 'true_coszen',
                is_log=False
            )

            cake_map_to_plot = cake_maps[cakekey]

            max_diff_ratio, max_diff = plot_map_comparisons(
                ref_map=pisa_map_to_plot,
                new_map=cake_map_to_plot,
                ref_abv='PISAV2', new_abv='PISAV3',
                outdir=outdir,
                subdir='fullpipeline',
                stagename='aeff',
                servicename=test_service,
                name=cakekey,
                texname=cake_maps[cakekey].tex,
            )

            check_agreement(
                testname='PISAV3-PISAV2 full pipeline through aeff:hist %s' %nukey,
                thresh_ratio=ratio_test_threshold,
                ratio=max_diff_ratio,
                thresh_diff=diff_test_threshold,
                diff=max_diff
            )


def compare_reco_full(cake_maps, pisa_maps, outdir, ratio_test_threshold,
                      diff_test_threshold):
    """Compare a fully configured pipeline (with stages flux, osc, aeff, reco,
    and pid) through the reco stage.

    """
    logging.debug('>> Working on full pipeline comparisons')
    logging.debug('>>> Checking to end of reco stage')
    test_service = 'hist_1X585'

    nue_nuebar_cc = cake_maps.combine_wildcard(r'nue*cc')
    numu_numubar_cc = cake_maps.combine_wildcard(r'numu*cc')
    nutau_nutaubar_cc = cake_maps.combine_wildcard(r'nutau*cc')
    nuall_nuallbar_nc = cake_maps.combine_wildcard(r'nu*nc')
    if 'pid' in nue_nuebar_cc.binning:
        nue_nuebar_cc = nue_nuebar_cc.sum('pid', keepdims=False)
        numu_numubar_cc = numu_numubar_cc.sum('pid', keepdims=False)
        nutau_nutaubar_cc = nutau_nutaubar_cc.sum('pid', keepdims=False)
        nuall_nuallbar_nc = nuall_nuallbar_nc.sum('pid', keepdims=False)

    modified_cake_outputs = {
        'nue_cc': nue_nuebar_cc,
        'numu_cc': numu_numubar_cc,
        'nutau_cc': nutau_nutaubar_cc,
        'nuall_nc': nuall_nuallbar_nc
    }

    for nukey in pisa_maps.keys():
        if 'nu' not in nukey:
            continue

        pisa_map_to_plot = pisa2_map_to_pisa3_map(
            pisa2_map = pisa_maps[nukey],
            ebins_name = 'reco_energy',
            czbins_name = 'reco_coszen',
            is_log=False
        )

        if '_' in nukey:
            if nukey.split('_')[1] == 'bar':
                new_nukey = ""
                for substr in nukey.split('_'):
                    new_nukey += substr
                nukey = new_nukey

        cake_map_to_plot = modified_cake_outputs[nukey]

        if 'nc' in nukey:
            if 'bar' in nukey:
                texname = r'\bar{\nu} NC'
            else:
                texname = r'\nu NC'
        else:
            texname = cake_maps[nukey].tex

        max_diff_ratio, max_diff = plot_map_comparisons(
            ref_map=pisa_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv='PISAV2', new_abv='PISAV3',
            outdir=outdir,
            subdir='fullpipeline',
            stagename='reco',
            servicename=test_service,
            name=nukey,
            texname=texname
        )

        check_agreement(
            testname='PISAV3-PISAV2 full pipeline through reco:%s %s'
                %(test_service, nukey),
            thresh_ratio=ratio_test_threshold,
            ratio=max_diff_ratio,
            thresh_diff=diff_test_threshold,
            diff=max_diff
        )


def compare_pid_full(cake_maps, pisa_maps, outdir, ratio_test_threshold,
                     diff_test_threshold):
    """Compare a fully configured pipeline (with stages flux, osc, aeff, reco,
    and pid) through the pid stage.

    """
    logging.debug('>> Working on full pipeline comparisons')
    logging.debug('>>> Checking to end of pid stage')
    test_service = 'hist_1X585'

    try:
        cake_trck = cake_maps.combine_wildcard('*_trck')
        cake_cscd = cake_maps.combine_wildcard('*_cscd')
    except ValueError:
        total = cake_maps.combine_wildcard('*')
        cake_cscd = total[0,:,:]
        cake_trck = total[1,:,:]

    pisa_trck = pisa2_map_to_pisa3_map(
        pisa2_map = pisa_maps['trck'],
        ebins_name = 'reco_energy',
        czbins_name = 'reco_coszen',
        is_log=False
    )
    pisa_cscd = pisa2_map_to_pisa3_map(
        pisa2_map = pisa_maps['cscd'],
        ebins_name = 'reco_energy',
        czbins_name = 'reco_coszen',
        is_log=False
    )

    max_diff_ratio, max_diff = plot_map_comparisons(
        ref_map=pisa_cscd,
        new_map=cake_cscd,
        ref_abv='PISAV2', new_abv='PISAV3',
        outdir=outdir,
        subdir='fullpipeline',
        stagename='pid',
        servicename=test_service,
        name='cscd',
        texname=r'{\rm cscd}'
    )
    check_agreement(
        testname='PISAV3-PISAV2 full pipeline through pid:%s cscd'
            %(test_service),
        thresh_ratio=ratio_test_threshold,
        ratio=max_diff_ratio,
        thresh_diff=diff_test_threshold,
        diff=max_diff
    )

    max_diff_ratio, max_diff = plot_map_comparisons(
        ref_map=pisa_trck,
        new_map=cake_trck,
        ref_abv='PISAV2', new_abv='PISAV3',
        outdir=outdir,
        subdir='fullpipeline',
        stagename='pid',
        servicename=test_service,
        name='trck',
        texname=r'{\rm trck}'
    )
    check_agreement(
        testname='PISAV3-PISAV2 full pipeline through pid:%s trck'
            %(test_service),
        thresh_ratio=ratio_test_threshold,
        ratio=max_diff_ratio,
        thresh_diff=diff_test_threshold,
        diff=max_diff
    )


def parse_args():
    if FTYPE == np.float32:
        dflt_ratio_threshold = 5e-4
    elif FTYPE == np.float64:
        dflt_ratio_threshold = 1e-7
    else:
        raise ValueError('FTYPE=%s from const.py not handled' % FTYPE)

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--flux', action='store_true',
                        help='''Run flux tests i.e. the interpolation methods
                        and the flux systematics.''')
    parser.add_argument('--osc-prob3cpu', action='store_true',
                        help='''Run osc tests i.e. the oscillograms with one
                        sigma deviations in the parameters.''')
    parser.add_argument('--osc-prob3gpu', action='store_true',
                        help='''Run GPU-based osc tests i.e. the oscillograms
                        with one sigma deviations in the parameters.''')
    parser.add_argument('--aeff', action='store_true',
                        help='''Run effective area tests i.e. the different
                        transforms with the aeff systematics.''')
    parser.add_argument('--reco', action='store_true',
                        help='''Run reco tests i.e. the different reco kernels
                        and their systematics.''')
    parser.add_argument('--pid', action='store_true',
                        help='''Run PID tests i.e. the different pid kernels
                        methods and their systematics.''')
    parser.add_argument('--full', action='store_true',
                        help='''Run full pipeline tests for the baseline i.e.
                        all stages simultaneously rather than each in
                        isolation.''')
    parser.add_argument('--outdir', metavar='DIR', type=str,
                        help='''Store all output plots to this directory. If
                        they don't exist, the script will make them, including
                        all subdirectories. If --outdir is not supplied, no
                        plots will be saved.''')
    parser.add_argument('--ratio_threshold', type=float,
                        default=dflt_ratio_threshold,
                        help='''Sets the agreement threshold on the ratio test
                        plots. If this is not reached the tests will fail.''')
    parser.add_argument('--diff_threshold', type=float, default=2E-3,
                        help='''Sets the agreement threshold on the diff test
                        plots. If this is not reached the tests will fail. This
                        test is only important if any ratios return inf.''')
    parser.add_argument('--ignore-cuda-errors', action='store_true',
                        help='''Ignore errors if pycuda cannot be used. If
                        pycuda can be used, however, errors will still be
                        raised as exceptions.''')
    parser.add_argument('-v', action='count', default=None,
                        help='''Set verbosity level; default is 1, so only -vv
                        or -vvv has an effect.''')
    args = parser.parse_args()
    args.v = min(1, args.v)
    return args


def main():
    args = parse_args()
    set_verbosity(args.v)

    # Figure out which tests to do
    test_all = True
    if args.flux or args.osc_prob3cpu or args.osc_prob3gpu or args.aeff or \
            args.reco or args.pid or args.full:
        test_all = False

    # Perform flux tests
    if args.flux or test_all:
        flux_settings = os.path.join(
            'tests', 'settings', 'pisa2_flux_honda_test.cfg'
        )
        flux_config = parse_pipeline_config(flux_settings)

        k = [k for k in flux_config.keys() if k[0] == 'flux'][0]
        params = flux_config[k]['params'].params

        params.flux_file.value = 'flux/honda-2015-spl-solmax-aa.d'
        params.flux_mode.value = 'integral-preserving'

        for syst in [None, 'atm_delta_index', 'nue_numu_ratio',
                     'nu_nubar_ratio', 'energy_scale']:
            pisa2file = os.path.join(
                'tests', 'data', 'flux', 'PISAV2IPHonda2015SPLSolMaxFlux.json'
            )
            pisa2file = find_resource(pisa2file)
            flux_pipeline = compare_flux(
                config=deepcopy(flux_config),
                servicename='IP_Honda',
                pisa2file=pisa2file,
                systname=syst,
                outdir=args.outdir,
                ratio_test_threshold=args.ratio_threshold,
                diff_test_threshold=args.diff_threshold
            )

        params.flux_mode.value = 'bisplrep'
        pisa2file = os.path.join(
            'tests', 'data', 'flux', 'PISAV2bisplrepHonda2015SPLSolMaxFlux.json'
        )
        pisa2file = find_resource(pisa2file)
        flux_pipeline = compare_flux(
            config=deepcopy(flux_config),
            servicename='bisplrep_Honda',
            pisa2file=pisa2file,
            systname=None,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )

    # Perform CPU-based oscillations tests
    if args.osc_prob3cpu or test_all:
        osc_settings = os.path.join(
            'tests', 'settings', 'pisa2_osc_prob3cpu_test.cfg'
        )
        osc_config = parse_pipeline_config(osc_settings)
        for syst in [None, 'theta12', 'theta13', 'theta23', 'deltam21',
                     'deltam31']:
            pisa2file = os.path.join(
                'tests', 'data', 'osc', 'PISAV2OscStageProb3Service.json'
            )
            pisa2file = find_resource(pisa2file)
            osc_pipeline = compare_osc(
                config=deepcopy(osc_config),
                servicename='prob3cpu',
                pisa2file=pisa2file,
                systname=syst,
                outdir=args.outdir,
                ratio_test_threshold=args.ratio_threshold,
                diff_test_threshold=args.diff_threshold
            )

    # Test for CUDA being present
    if not PYCUDA_AVAIL:
        msg = 'No CUDA support found, so GPU-based services cannot run.'
        if args.osc_prob3gpu or test_all:
            if args.ignore_cuda_errors:
                logging.warn(msg)
            else:
                raise ImportError(msg)

    # Perform GPU-based oscillations tests
    if (args.osc_prob3gpu or test_all) and PYCUDA_AVAIL:
        osc_settings = os.path.join(
            'tests', 'settings', 'pisa2_osc_prob3gpu_test.cfg'
        )
        osc_config = parse_pipeline_config(osc_settings)
        for syst in [None, 'theta12', 'theta13', 'theta23', 'deltam21',
                     'deltam31']:
            pisa2file = os.path.join(
                'tests', 'data', 'osc', 'PISAV2OscStageProb3Service.json'
            )
            pisa2file = find_resource(pisa2file)
            osc_pipeline = compare_osc(
                config=deepcopy(osc_config),
                servicename='prob3gpu',
                pisa2file=pisa2file,
                systname=syst,
                outdir=args.outdir,
                ratio_test_threshold=args.ratio_threshold,
                diff_test_threshold=args.diff_threshold
            )

    # Perform effective-area tests
    if args.aeff or test_all:
        aeff_settings = os.path.join(
            'tests', 'settings', 'pisa2_aeff_hist_test.cfg'
        )
        aeff_config = parse_pipeline_config(aeff_settings)

        k = [k for k in aeff_config.keys() if k[0] == 'aeff'][0]
        params = aeff_config[k]['params'].params

        params.aeff_events.value = os.path.join(
            'events', 'deepcore_ic86', 'MSU', '1XXXX', 'UnJoined',
            'DC_MSU_1X585_unjoined_events_mc.hdf5'
        )
        pisa2file = os.path.join(
            'tests', 'data', 'aeff', 'PISAV2AeffStageHist1X585Service.json'
        )
        pisa2file = find_resource(pisa2file)
        for syst in [None, 'aeff_scale']:
            aeff_pipeline = compare_aeff(
                config=deepcopy(aeff_config),
                servicename='hist_1X585',
                pisa2file=pisa2file,
                systname=syst,
                outdir=args.outdir,
                ratio_test_threshold=args.ratio_threshold,
                diff_test_threshold=args.diff_threshold
            )    
        aeff_settings = os.path.join(
            'tests', 'settings', 'pisa2_aeff_param_test.cfg'
        )
        aeff_config = parse_pipeline_config(aeff_settings)

        k = [k for k in aeff_config.keys() if k[0] == 'aeff'][0]
        params = aeff_config[k]['params'].params

        pisa2file = os.path.join(
            'tests', 'data', 'aeff', 'PISAV2AeffStageParam1X60Service.json'
        )
        pisa2file = find_resource(pisa2file)
        for syst in [None, 'aeff_scale']:
            aeff_pipeline = compare_aeff(
                config=deepcopy(aeff_config),
                servicename='param_1X60',
                pisa2file=pisa2file,
                systname=syst,
                outdir=args.outdir,
                ratio_test_threshold=args.ratio_threshold,
                diff_test_threshold=args.diff_threshold
            )

    # Perform reconstruction tests
    if args.reco or test_all:
        reco_settings = os.path.join(
            'tests', 'settings', 'pisa2_reco_hist_test.cfg'
        )
        reco_config = parse_pipeline_config(reco_settings)

        k = [k for k in reco_config.keys() if k[0] == 'reco'][0]
        params = reco_config[k]['params'].params

        params.reco_weights_name.value = None
        params.reco_events.value = os.path.join(
            'events', 'deepcore_ic86', 'MSU', '1XXXX', 'Joined',
            'DC_MSU_1X585_joined_nu_nubar_events_mc.hdf5'
        )
        pisa2file = os.path.join(
            'tests', 'data', 'reco', 'PISAV2RecoStageHist1X585Service.json'
        )
        pisa2file = find_resource(pisa2file)
        reco_pipeline = compare_reco(
            config=deepcopy(reco_config),
            servicename='hist_1X585',
            pisa2file=pisa2file,
            systname=None,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )

        params.reco_events.value = os.path.join(
            'events', 'deepcore_ic86', 'MSU', '1XXX', 'Joined',
            'DC_MSU_1X60_joined_nu_nubar_events_mc.hdf5'
        )
        pisa2file = os.path.join(
            'tests', 'data', 'reco', 'PISAV2RecoStageHist1X60Service.json'
        )
        pisa2file = find_resource(pisa2file)
        reco_pipeline = compare_reco(
            config=deepcopy(reco_config),
            servicename='hist_1X60',
            pisa2file=pisa2file,
            systname=None,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )
        reco_settings = os.path.join(
            'tests', 'settings', 'pisa2_reco_param_test.cfg'
        )
        reco_config = parse_pipeline_config(reco_settings)
        pisa2file = os.path.join(
            'tests', 'data', 'reco', 'PISAV2RecoStageParam1X60Service.json'
        )
        pisa2file = find_resource(pisa2file)
        for syst in [None, 'e_res_scale', 'cz_res_scale']:
            reco_pipeline = compare_reco(
                config=deepcopy(reco_config),
                servicename='param_1X60',
                pisa2file=pisa2file,
                systname=syst,
                outdir=args.outdir,
                ratio_test_threshold=args.ratio_threshold,
                diff_test_threshold=args.diff_threshold
            )

    # Perform PID tests
    if args.pid or test_all:
        pid_settings = os.path.join(
            'tests', 'settings', 'pisa2_pid_hist_test.cfg'
        )
        pid_config = parse_pipeline_config(pid_settings)

        k = [k for k in pid_config.keys() if k[0] == 'pid'][0]
        params = pid_config[k]['params'].params

        pisa2file = os.path.join(
            'tests', 'data', 'pid', 'PISAV2PIDStageHistV39Service.json'
        )
        pisa2file = find_resource(pisa2file)
        pid_pipeline = compare_pid(
            config=deepcopy(pid_config),
            servicename='hist_V39',
            pisa2file=pisa2file,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )
        params.pid_events.value = os.path.join(
            'events', 'deepcore_ic86', 'MSU', '1XXXX', 'Joined',
            'DC_MSU_1X585_joined_nu_nubar_events_mc.hdf5'
        )
        pisa2file = os.path.join(
            'tests', 'data', 'pid', 'PISAV2PIDStageHist1X585Service.json'
        )
        pisa2file = find_resource(pisa2file)
        try:
            pid_pipeline = compare_pid(
                config=deepcopy(pid_config),
                servicename='hist_1X585',
                pisa2file=pisa2file,
                outdir=args.outdir,
                ratio_test_threshold=args.ratio_threshold,
                diff_test_threshold=args.diff_threshold
            )
        except ValueError:
            logging.info(PID_PASS_MESSAGE)
        else:
            raise ValueError(PID_FAIL_MESSAGE)
        pid_settings = os.path.join(
            'tests', 'settings', 'pisa2_pid_param_test.cfg'
        )
        pid_config = parse_pipeline_config(pid_settings)
        pisa2file = os.path.join(
            'tests', 'data', 'pid', 'PISAV2PIDStageParam1X60Service.json'
        )
        pisa2file = find_resource(pisa2file)
        pid_pipeline = compare_pid(
            config=deepcopy(pid_config),
            servicename='param_1X60',
            pisa2file=pisa2file,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )
    ## Perform reco+PID tests
    #if args.recopid or test_all:
    #    pid_settings = os.path.join(
    #        'tests', 'settings', 'recopid_test.cfg'
    #    )
    #    recopid_config = parse_pipeline_config(pid_settings)

    #    k = [k for k in recopid_config.keys() if k[0] == 'pid'][0]
    #    params = recopid_config[k]['params'].params

    #    pisa2file = os.path.join(
    #        'tests', 'data', 'pid', 'PISAV2PIDStageHistV39Service.json'
    #    )
    #    pisa2file = find_resource(pisa2file)
    #    pid_pipeline = compare_pid(
    #        config=deepcopy(recopid_config),
    #        servicename='hist_V39',
    #        pisa2file=pisa2file,
    #        outdir=args.outdir,
    #        ratio_test_threshold=args.ratio_threshold,
    #        diff_test_threshold=args.diff_threshold
    #    )
    #    params.pid_events.value = os.path.join(
    #        'events', 'deepcore_ic86', 'MSU', '1XXXX', 'Joined',
    #        'DC_MSU_1X585_joined_nu_nubar_events_mc.hdf5'
    #    )
    #    params.pid_weights_name.value = 'weighted_aeff'
    #    params.pid_ver.value = 'msu_mn8d-mn7d'
    #    pisa2file = os.path.join(
    #        'tests', 'data', 'pid', 'PISAV2PIDStageHist1X585Service.json'
    #    )
    #    pisa2file = find_resource(pisa2file)
    #    pid_pipeline = compare_pid(
    #        config=deepcopy(recopid_config),
    #        servicename='hist_1X585',
    #        pisa2file=pisa2file,
    #        outdir=args.outdir,
    #        ratio_test_threshold=args.ratio_threshold,
    #        diff_test_threshold=args.diff_threshold
    #    )

    # Perform full-pipeline tests
    if args.full or test_all:
        full_settings = os.path.join(
            'tests', 'settings', 'pisa2_full_pipeline_test.cfg'
        )
        pipeline = Pipeline(full_settings)
        pipeline.get_outputs()

        pisa2file = os.path.join(
            'tests', 'data', 'full',
            'PISAV2FullDeepCorePipeline-IPSPL2015SolMax-Prob3CPUNuFit2014-AeffHist1X585-RecoHist1X585-PIDHist1X585.json'
        )
        pisa2file = find_resource(pisa2file)
        pisa2_comparisons = from_file(pisa2file)
        # Through flux stage comparisons
        compare_flux_full(
            pisa_maps=pisa2_comparisons[0],
            cake_maps=pipeline['flux'].outputs,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )
        # Through osc stage comparisons
        compare_osc_full(
            pisa_maps=pisa2_comparisons[1],
            cake_maps=pipeline['osc'].outputs,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )
        # Through aeff stage comparisons
        compare_aeff_full(
            pisa_maps=pisa2_comparisons[2],
            cake_maps=pipeline['aeff'].outputs,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )
        # Through reco stage comparisons
        compare_reco_full(
            pisa_maps=pisa2_comparisons[3],
            cake_maps=pipeline['reco'].outputs,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )
        # Through PID stage comparisons
        try:
            compare_pid_full(
                pisa_maps=pisa2_comparisons[4],
                cake_maps=pipeline['pid'].outputs, # use reco here
                outdir=args.outdir,
                ratio_test_threshold=args.ratio_threshold,
                diff_test_threshold=args.diff_threshold
            )
        except ValueError:
            logging.info(PID_PASS_MESSAGE)
        else:
            raise ValueError(PID_FAIL_MESSAGE)


if __name__ == '__main__':
    main()
