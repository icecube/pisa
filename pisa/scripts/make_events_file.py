#!/usr/bin/env python

"""
Take simulated (and reconstructed) HDF5 file(s) (as converted from I3 by
icecube.hdfwriter.I3HDFTableService) as input and writes out a simplified HDF5
file for use with PISA.
"""


from __future__ import absolute_import, division

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
from copy import deepcopy
import os

import numpy as np

from pisa.core.events import Events
from pisa.utils.data_proc_params import DataProcParams
from pisa.utils.format import list2hrlist
from pisa.utils.fileio import expand, mkdir, to_file
from pisa.utils.flavInt import (FlavIntData, NuFlav, NuFlavIntGroup,
                                ALL_NUFLAVINTS, ALL_NUINT_TYPES, xlateGroupsStr)
from pisa.utils.log import logging, set_verbosity
from pisa.utils.mcSimRunSettings import DetMCSimRunsSettings
from pisa.utils.resources import find_resource


__all__ = ['EXAMPLE', 'CMSQ_TO_MSQ', 'EXTRACT_FIELDS', 'OUTPUT_FIELDS',
           'powerLawIntegral', 'makeEventsFile',
           'parse_args', 'main']

__author__ = 'J.L. Lanfranchi'

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


EXAMPLE = """
Example command-line usage:

make_events_file.py
    --det "PINGU"
    --proc "V5.1"
    --run 390 ~/data/390/icetray_hdf5/*.hdf5
    --run 389 ~/data/389/icetray_hdf5/*.hdf5
    --run 388 ~/data/388/icetray_hdf5/*.hdf5
    -vv
    --outdir /tmp/events/
"""


CMSQ_TO_MSQ = 1.0e-4
"""Conversion factor: convert from centimeters^2 to meters^2"""

# Default fields to extract from source HDF5 files during processing
# Note that *_coszen is generated from *_zenith
EXTRACT_FIELDS = (
    'true_energy',
    'true_coszen',
    'reco_energy',
    'reco_coszen',
    'one_weight',
    'interaction_prob',
    'pid',
)

# Default fields to output to destination PISA events HDF5 file
OUTPUT_FIELDS = (
    'true_energy',
    'true_coszen',
    'reco_energy',
    'reco_coszen',
    #'mc_weight',
    #'mc_weight_per_gev_per_sr',
    'weighted_aeff',
    'pid',
)


def powerLawIntegral(E0, E1, gamma):
    import sympy as sym
    E = sym.Symbol('E')
    I = sym.integrate(E**(-gamma), E)
    return I.evalf(subs={E: E1}) - I.evalf(subs={E: E0})


def makeEventsFile(data_files, detector, proc_ver, cut, outdir,
                   run_settings=None, data_proc_params=None, join=None,
                   cust_cuts=None, extract_fields=EXTRACT_FIELDS,
                   output_fields=OUTPUT_FIELDS):
    r"""Take the simulated and reconstructed HDF5 file(s) (as converted from I3
    by icecube.hdfwriter.I3HDFTableService) as input and write out a simplified
    PISA-standard-format HDF5 file for use in aeff, reco, and/or PID stages.

    Parameters
    ----------
    data_files : dict
        File paths for finding data files for each run, formatted as:
            {
                <string run>: <list of file paths>,
                <string run>: <list of file paths>,
                ...
                <string run>: <list of file paths>,
            }

    detector : string
        Name of the detector (e.g. IceCube, DeepCore, PINGU, etc.) as found in
        e.g. mc_sim_run_settings.json and data_proc_params.json files.

    proc_ver
        Version of processing applied to the events, as found in e.g.
        data_proc_params.json.

    cut
        Name of a standard cut to use; must be specified in the relevant
        detector/processing version node of the data processing parameters
        (file from which the data_proc_params object was instantiated)

    outdir
        Directory path in which to store resulting files; will be generated if
        it does not already exist (including any parent directories that do not
        exist)

    run_settings : string or MCSimRunSettings
        Resource location of mc_sim_run_settings.json or an MCSimRunSettings
        object instantiated therefrom.

    data_proc_params : string or DataProcParams
        Resource location of data_proc_params.json or a DataProcParams object
        instantiated therefrom.

    join
        String specifying any flavor/interaction types (flavInts) to join
        together. Separate flavInts with commas (',') and separate groups
        with semicolons (';'). E.g. an acceptable string is:
            'numucc+numubarcc; nuall bar NC, nuall NC'

    cust_cuts
        dict with a single DataProcParams cut specification or list of same
        (see help for DataProcParams for detailed description of cut spec)

    extract_fields : None or iterable of strings
        Field names to extract from source HDF5 file. If None, extract all
        fields.

    output_fields : None or iterable of strings
        Fields to include in the generated PISA-standard-format events HDF5
        file; note that if 'weighted_aeff' is not preent, effective area will
        not be computed. If None, all fields will be written.

    Notes
    -----
    Compute "weighted_aeff" field:

    Within each int type (CC or NC), ngen should be added together;
    events recorded of that int type then get their one_weight divided by the
    total *for that int type only* to obtain the "weighted_aeff" for that
    event (even if int types are being grouped/joined together).

    This has the effect that within a group, ...
      ... and within an interaction type, effective area is a weighted
      average of that of the flavors being combined. E.g. for CC,

                     \sum_{run x}\sum_{flav y} (Aeff_{x,y} * ngen_{x,y})
          Aeff_CC = ----------------------------------------------------- ,
                          \sum_{run x}\sum_{flav y} (ngen_{x,y})

      ... and then across interaction types, the results of the above for
      each int type need to be summed together, i.e.:

          Aeff_total = Aeff_CC + Aeff_NC

    Note that each grouping of flavors is calculated with the above math
    completely independently from other flavor groupings specified.

    See Justin Lanfranchi's presentation on the PINGU Analysis call,
    2015-10-21, for more details:
      https://wikispaces.psu.edu/download/attachments/282040606/meff_report_jllanfranchi_v05_2015-10-21.pdf

    """
    if isinstance(run_settings, str):
        run_settings = DetMCSimRunsSettings(
            find_resource(run_settings),
            detector=detector
        )
    assert isinstance(run_settings, DetMCSimRunsSettings)
    assert run_settings.detector == detector

    if isinstance(data_proc_params, str):
        data_proc_params = DataProcParams(
            detector=detector,
            proc_ver=proc_ver,
            data_proc_params=find_resource(data_proc_params)
        )
    assert data_proc_params.detector == detector
    assert data_proc_params.proc_ver == proc_ver

    runs = sorted(data_files.keys())

    all_flavs = []
    flavs_by_run = {}
    run_norm_factors = {}
    bin_edges = set()

    runs_by_flavint = FlavIntData()
    for flavint in runs_by_flavint.flavints:
        runs_by_flavint[flavint] = []

    #ngen_flavint_by_run = {run:FlavIntData() for run in runs}
    ##ngen_per_flav_by_run = {run:FlavIntData() for run in runs}
    #eint_per_flav_by_run = {run:FlavIntData() for run in runs}
    #for run in runs:
    #    flavints_in_run = run_settings.get_flavints(run=run)
    #    e_range = run_settings.get_energy_range(run)
    #    gamma = run_settings.get_spectral_index(run)
    #    for flavint in flavints_in_run:
    #        runs_by_flavint[flavint].append(run)
    #        ngen_flav = run_settings.get_num_gen(
    #            run=run, flav_or_flavint=flavint, include_physical_fract=True
    #        )
    #        #runs_by_flavint[flavint].append(run)
    #        #this_flav = flavint.
    #        #xsec_fract_en_wtd_avg[run][flavint] = \
    #        ngen_flavint_by_run[run][flavint] = \
    #                xsec.get_xs_ratio_integral(
    #                    flavintgrp0=flavint,
    #                    flavintgrp1=flavint.flav,
    #                    e_range=e_range,
    #                    gamma=gamma,
    #                    average=True
    #                )
    #    xsec_ver = run_settings.get_xsec_version(run=run)
    #    if xsec_ver_ref is None:
    #        xsec_ver_ref = xsec_ver
    #    # An assumption of below logic is that all MC is generated using the
    #    # same cross sections version.
    #    #
    #    # TODO / NOTE:
    #    # It would be possible to combine runs with different cross sections so
    #    # long as each (flavor, interaction type) cross sections are
    #    # weighted-averaged together using weights
    #    #   N_gen_{n,flav+inttype} * E_x^{-gamma_n} /
    #    #       ( \int_{E_min_n}^{E_max_n} E^{-\gamma_n} dE )
    #    # where E_x are the energy sample points specified in the cross
    #    # sections (and hence these must also be identical across all cross
    #    # sections that get combined, unless interpolation is performed).
    #    assert xsec_ver == xsec_ver_ref
    #    #ngen_weighted_energy_integral[str(run)] = powerLawIntegral(
    #    #flavs_by_run[run] = run_settings.flavs(run)
    ##flavs_present =

    detector_geom = run_settings[runs[0]]['geom']

    # Create Events object to store data
    evts = Events()
    evts.metadata.update({
        'detector': run_settings.detector,
        'proc_ver': data_proc_params.proc_ver,
        'geom': detector_geom,
        'runs': runs,
    })

    cuts = []
    if isinstance(cust_cuts, dict):
        cust_cuts = [cust_cuts]
    if cut is not None:
        evts.metadata['cuts'].append(cut)
        cuts.append(cut)
    if cust_cuts is not None:
        for ccut in cust_cuts:
            evts.metadata['cuts'].append('custom: ' + ccut['pass_if'])
            cuts.append(ccut)

    orig_outdir = outdir
    outdir = expand(outdir)
    logging.info('Output dir spec\'d: %s', orig_outdir)
    if outdir != orig_outdir:
        logging.info('Output dir expands to: %s', outdir)
    mkdir(outdir)

    detector_label = str(data_proc_params.detector)
    proc_label = 'proc_' + str(data_proc_params.proc_ver)

    # What flavints to group together
    if join is None or join == '':
        grouped = []
        ungrouped = [NuFlavIntGroup(k) for k in ALL_NUFLAVINTS]
        groups_label = 'unjoined'
        logging.info('Events in the following groups will be joined together:'
                     ' (none)')
    else:
        grouped, ungrouped = xlateGroupsStr(join)
        evts.metadata['flavints_joined'] = [str(g) for g in grouped]
        groups_label = 'joined_G_' + '_G_'.join([str(g) for g in grouped])
        logging.info('Events in the following groups will be joined together: '
                     + '; '.join([str(g) for g in grouped]))

    # Find any flavints not included in the above groupings
    flavint_groupings = grouped + ungrouped
    if len(ungrouped) == 0:
        ungrouped = ['(none)']
    logging.info('Events of the following flavints will NOT be joined'
                 'together: ' + '; '.join([str(k) for k in ungrouped]))

    # Enforce that flavints composing groups are mutually exclusive
    for grp_n, flavintgrp0 in enumerate(flavint_groupings[:-1]):
        for flavintgrp1 in flavint_groupings[grp_n+1:]:
            assert len(set(flavintgrp0).intersection(set(flavintgrp1))) == 0

    flavintgrp_names = [str(flavintgrp) for flavintgrp in flavint_groupings]

    # Instantiate storage for all intermediate destination fields;
    # The data structure looks like:
    #   extracted_data[group #][interaction type][field name] = list of data
    if extract_fields is None:
        extracted_data = [
            {
                inttype: {} for inttype in ALL_NUINT_TYPES
            }
            for _ in flavintgrp_names
        ]
    else:
        extracted_data = [
            {
                inttype: {field: [] for field in extract_fields}
                for inttype in ALL_NUINT_TYPES
            }
            for _ in flavintgrp_names
        ]

    # Instantiate generated-event counts for destination fields; count
    # CClseparately from NC because aeff's for CC & NC add, whereas
    # aeffs intra-CC should be weighted-averaged (as for intra-NC)
    ngen = [
        {inttype: {} for inttype in ALL_NUINT_TYPES}
        for _ in flavintgrp_names
    ]

    # Loop through all of the files, retrieving the events, filtering,
    # and recording the number of generated events pertinent to
    # calculating aeff
    filecount = {}
    detector_geom = None
    bad_files = []
    for run, fnames in data_files.items():
        file_count = 0
        for fname in fnames:
            # Retrieve data from all nodes specified in the processing
            # settings file
            logging.trace('Trying to get data from file %s', fname)
            try:
                data = data_proc_params.get_data(
                    fname, run_settings=run_settings
                )
            except (ValueError, KeyError, IOError):
                logging.warning('Bad file encountered: %s', fname)
                bad_files.append(fname)
                continue

            file_count += 1

            # Check to make sure only one run is present in the data
            runs_in_data = set(data['run'])
            assert len(runs_in_data) == 1, 'Must be just one run in data'

            #run = int(data['run'][0])
            if not run in filecount:
                filecount[run] = 0
            filecount[run] += 1
            rs_run = run_settings[run]

            # Record geom; check that geom is consistent with other runs
            if detector_geom is None:
                detector_geom = rs_run['geom']
            assert rs_run['geom'] == detector_geom, \
                    'All runs\' geometries must match!'

            # Loop through all flavints spec'd for run
            for run_flavint in rs_run['flavints']:
                barnobar = run_flavint.bar_code
                int_type = run_flavint.intType

                # Retrieve this-interaction-type- & this-barnobar-only events
                # that also pass cuts. (note that cut names are strings)
                intonly_cut_data = data_proc_params.apply_cuts(
                    data,
                    cuts=cuts+[str(int_type), str(barnobar)],
                    return_fields=extract_fields
                )

                # Record the generated count and data for this run/flavor for
                # each group to which it's applicable
                for grp_n, flavint_group in enumerate(flavint_groupings):
                    if not run_flavint in flavint_group:
                        continue

                    # Instantiate a field for particles and antiparticles,
                    # keyed by the output of the bar_code property for each
                    if not run in ngen[grp_n][int_type]:
                        ngen[grp_n][int_type][run] = {
                            NuFlav(12).bar_code: 0,
                            NuFlav(-12).bar_code: 0,
                        }

                    # Record count only if it hasn't already been recorded
                    if ngen[grp_n][int_type][run][barnobar] == 0:
                        # Note that one_weight includes cc/nc:total fraction,
                        # so DO NOT specify the full flavint here, only flav
                        # (since one_weight does NOT take bar/nobar fraction,
                        # it must be included here in the ngen computation)
                        flav_ngen = run_settings.get_num_gen(run=run,
                                                             barnobar=barnobar)
                        ngen[grp_n][int_type][run][barnobar] = flav_ngen

                    # Append the data. Note that extracted_data is:
                    # extracted_data[group n][int_type][extract field name] =
                    #   list
                    if extract_fields is None:
                        for f in intonly_cut_data.keys():
                            if f not in extracted_data[grp_n][int_type]:
                                extracted_data[grp_n][int_type][f] = []
                            extracted_data[grp_n][int_type][f].extend(
                                intonly_cut_data[f]
                            )
                    else:
                        for f in extract_fields:
                            extracted_data[grp_n][int_type][f].extend(
                                intonly_cut_data[f]
                            )
        logging.info('File count for run %s: %d', run, file_count)
    to_file(bad_files, '/tmp/bad_files.json')

    if ((output_fields is None
         and (extract_fields is None or 'one_weight' in extract_fields))
            or 'weighted_aeff' in output_fields):
        fmtfields = (' '*12+'flavint_group',
                     'int type',
                     '     run',
                     'part/anti',
                     'part/anti count',
                     'aggregate count')
        fmt_n = [len(f) for f in fmtfields]
        fmt = '  '.join([r'%'+str(n)+r's' for n in fmt_n])
        lines = '  '.join(['-'*n for n in fmt_n])
        logging.info(fmt, fmtfields)
        logging.info(lines)
        for grp_n, flavint_group in enumerate(flavint_groupings):
            for int_type in set([fi.intType for fi in
                                 flavint_group.flavints]):
                ngen_it_tot = 0
                for run, run_counts in ngen[grp_n][int_type].items():
                    for barnobar, barnobar_counts in run_counts.items():
                        ngen_it_tot += barnobar_counts
                        logging.info(
                            fmt, flavint_group.simple_str(), int_type,
                            str(run), barnobar, int(barnobar_counts),
                            int(ngen_it_tot)
                        )
                # Convert data to numpy array
                if extract_fields is None:
                    for field in extracted_data[grp_n][int_type].keys():
                        extracted_data[grp_n][int_type][field] = \
                                np.array(extracted_data[grp_n][int_type][field])
                else:
                    for field in extract_fields:
                        extracted_data[grp_n][int_type][field] = \
                                np.array(extracted_data[grp_n][int_type][field])
                # Generate weighted_aeff field for this group / int type's data
                extracted_data[grp_n][int_type]['weighted_aeff'] = \
                        extracted_data[grp_n][int_type]['one_weight'] \
                        / ngen_it_tot * CMSQ_TO_MSQ

    # Report file count per run
    for run, count in filecount.items():
        logging.info('Files read, run %s: %d', run, count)
        ref_num_i3_files = run_settings[run]['num_i3_files']
        if count != ref_num_i3_files:
            logging.warning(
                'Run %s, Number of files read (%d) != number of '
                'source I3 files (%d), which may indicate an error.',
                run, count, ref_num_i3_files
            )

    # Generate output data
    for flavint in ALL_NUFLAVINTS:
        int_type = flavint.intType
        for grp_n, flavint_group in enumerate(flavint_groupings):
            if not flavint in flavint_group:
                logging.trace('flavint %s not in flavint_group %s, passing.',
                              flavint, flavint_group)
                continue
            else:
                logging.trace(
                    'flavint %s **IS** in flavint_group %s, storing.',
                    flavint, flavint_group
                )
            if output_fields is None:
                evts[flavint] = extracted_data[grp_n][int_type]
            else:
                evts[flavint] = {f: extracted_data[grp_n][int_type][f]
                                 for f in output_fields}

    # Generate file name
    numerical_runs = []
    alphanumerical_runs = []
    for run in runs:
        try:
            int(run)
            numerical_runs.append(int(run))
        except ValueError:
            alphanumerical_runs.append(str(run))
    run_labels = []
    if len(numerical_runs) > 0:
        run_labels.append(list2hrlist(numerical_runs))
    if len(alphanumerical_runs) > 0:
        run_labels += sorted(alphanumerical_runs)
    run_label = 'runs_' + ','.join(run_labels)
    geom_label = '' + detector_geom
    fname = 'events__' + '__'.join([
        detector_label,
        geom_label,
        run_label,
        proc_label,
        groups_label,
    ]) + '.hdf5'

    outfpath = os.path.join(outdir, fname)
    logging.info('Writing events to %s', outfpath)

    # Save data to output file
    evts.save(outfpath)


def parse_args():
    """Get command line arguments"""
    parser = ArgumentParser(
        description=__doc__ + EXAMPLE,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--det',
        metavar='DETECTOR',
        type=str,
        required=True,
        help='''Detector, e.g. "PINGU" or "DeepCore". This is used as the
        top-most key in run_settings.json and data_proc_params.json files.'''
    )
    parser.add_argument(
        '--proc',
        metavar='PROC_VER',
        type=str,
        required=True,
        help='''Processing version applied to simulation; processing versions
        are defined with respect to each geometry version. See
        data_proc_params.json file for definitions (or edit that file to add
        more).'''
    )

    parser.add_argument(
        '--run',
        metavar='RUN_ID H5_FILE0 H5_FILE1 ...',
        type=str,
        nargs="+",
        action='append',
        required=True,
        help='nue HDF5 file(s)'
    )

    parser.add_argument(
        '--outdir',
        metavar='DIR',
        type=str,
        required=True,
        help='directory into which to store resulting HDF5 file'
    )

    parser.add_argument(
        '--run-settings',
        metavar='JSON_FILE',
        type=str,
        default='events/mc_sim_run_settings.json',
        help='JSON file with reference run settings'
    )

    parser.add_argument(
        '--data-proc-params',
        metavar='JSON_FILE',
        type=str,
        default='events/data_proc_params.json',
        help='JSON file with reference processing settings'
    )

    # NOTE:
    # Removed --join in favor of forcing standard events groupings to be output
    # all at once, to ensure all files get generated all the time. Also
    # need to implement validation for consistent events file usage in PISA for
    # template settings file (i.e., if different files are specified for
    # different PISA stages, ensure they all come from the same detector,
    # geometry, and processing versions and have events groupings that do not
    # lead to erroneous conclusions for the stages they're specified for)

    #parser.add_argument(
    #    '--join',
    #    const='nuecc,nuebarcc;numucc,numubarcc;nutaucc,nutaubarcc;'
    #          'nuallnc,nuallbarnc',
    #    default='',
    #    action='store',
    #    nargs='?',
    #    type=str,
    #    help= \
    #    '''Optionally join flavors together to increase statistics for Aeff
    #    and/or resolutions (aeff and reco stages, respectively). Specifying the
    #    --join option without an argument joins together: nu_x &
    #    nu_x_bar CC events together (one set for each of x=e, x=mu, and x=tau),
    #    and joins nuall NC & nuallbar NC events tegether. If a string
    #    argument is supplied, this specifies custom groups to join together
    #    instead. The string must be a semicolon-separated list each field of
    #    which itself a comma-separated list of event "flavints" (flavor and
    #    interaction type) to grup together. Any event flavint not included in
    #    that string will be found individually, i.e., not joined together with
    #    any other flavors'''
    #)

    parser.add_argument(
        '--cut',
        metavar='CUT_NAME',
        type=str,
        help='''Name of pre-defined cut to apply. See the specified
        --data-proc-params file for definitions for the detector and processing
        version you're working with (note that the names of cuts and what these
        entail varies by detector and processing version)'''
    )

    parser.add_argument(
        '--ccut-pass-if',
        metavar='CRITERIA',
        type=str,
        default='',
        help= \
        '''Custom cut: String containing criteria for passing a cut, using
        field names specified by the --ccut-fields argument. Standard Python-
        and numpy-namespace expressions are allowed as well, since this string
        is passed to 'eval'. E.g.:
        --ccut-fields="z:MCNeutrino/zenith,l6:my_l6/value" \
        --ccut-pass-if="(l6 == 1) & (z > pi/2)" '''
    )
    parser.add_argument(
        '--ccut-fields',
        metavar='FIELDS',
        type=str,
        default='',
        help='''Custom cut: String of comma-separated fields, each containing
        colon-separated (variable name : HDF5 address) tuples. For example,
        specifying:
        --ccut-fields="l5:my_l5/value,l6:my_l6/value"
        allows for a custom cut to be defined via --ccut-pass-if="(l5 == 1) &
        (l6 == 1)"'''
    )

    parser.add_argument(
        '--no-aeff',
        action='store_true',
        help='''Do not compute or include the 'weighted_aeff' field in the
        generated PISA events HDF5 file, disallowing use of the file for
        effective area parameterizations or the Monte Carlo aeff stage'''
    )

    parser.add_argument(
        '--no-pid',
        action='store_true',
        help='''Do not include the 'pid' field in the generated PISA events
        HDF5 file, disallowing use of the file for PID parameterizations or the
        Monte Carlo PID stage'''
    )

    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='set verbosity level'
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_verbosity(args.verbose)

    runs_files = OrderedDict()
    for run_info in args.run:
        runs_files[run_info[0]] = run_info[1:]

    det = args.det.strip()
    proc = args.proc.strip()
    run_settings = DetMCSimRunsSettings(
        find_resource(args.run_settings),
        detector=det
    )
    data_proc_params = DataProcParams(
        detector=det,
        proc_ver=proc,
        data_proc_params=find_resource(args.data_proc_params)
    )

    logging.info('Using detector %s, processing version %s.', det, proc)

    extract_fields = deepcopy(EXTRACT_FIELDS)
    output_fields = deepcopy(OUTPUT_FIELDS)

    if args.no_pid:
        extract_fields = [f for f in extract_fields if f != 'pid']
        output_fields = [f for f in output_fields if f != 'pid']

    if args.no_aeff:
        output_fields = [f for f in output_fields if f != 'weighted_aeff']

    # Add any custom cuts specified on command line
    ccut = None
    if args.ccut_pass_if:
        ccut = {
            'pass_if': args.ccut_pass_if,
            'fields': args.ccut_fields.split(',')
        }

    # One events file will be produced for each of The following flavint
    # groupings
    groupings = [
        # No events joined together
        None,
        # CC events unjoined; join nuall NC and nuallbar NC separately (used
        # for generating aeff param service's aeff parameterizations)
        'nuallnc,nuallbarnc',
        # CC events paried by flav--anti-flav; nuallNC+nuallbarNC all joined
        # together; used for reco services (MC and vbwkde)
        'nuecc+nuebarcc,numucc+numubarcc,nutaucc+nutaubarcc,nuallnc+nuallbarnc',
    ]

    # Create the events files
    for grouping in groupings:
        makeEventsFile(
            data_files=runs_files,
            detector=args.det,
            proc_ver=args.proc,
            cut=args.cut,
            outdir=args.outdir,
            run_settings=run_settings,
            data_proc_params=data_proc_params,
            join=grouping,
            cust_cuts=ccut,
            extract_fields=None, #extract_fields,
            output_fields=None, #output_fields,
        )


if __name__ == "__main__":
    main()
