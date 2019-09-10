"""
A class for doing postprocessing.
"""


from __future__ import absolute_import

from collections import OrderedDict
import os
import re
import sys
import numpy as np
from scipy.stats import spearmanr

from pisa import ureg
from pisa.analysis.hypo_testing import Labels
from pisa.utils.fileio import from_file, mkdir, nsort, to_file
from pisa.utils.log import logging

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


class Postprocessor(object):
    """Class to contain all of the functions that are used by the various
    postprocessing scripts.


    Parameters
    ----------
    analysis_type : string
        Name for the type of analysis this was run e.g. hypo_testing,
        profile_scan etc.
    test_type : string
        Name for the type of test then done. This may be none, but may also be
        something along the line of `systematic_tests` etc.
    logdir : string
        Path to logging directory where output files of hypo_testing analysis
        are stored. This should contain e.g. the "config_summary.json" file.
    fluctuate_fid : bool
        Whether the trials you're interested in applied fluctuations to the
        fiducial-fit Asimov distributions. `fluctuate_fid` False is
        equivalent to specifying an Asimov analysis (so long as the metric
        used was chi-squared).
    fluctuate_data : bool
        Whether the trials you're interested in applied fluctuations to the
        (toy) data. This is invalid if actual data was processed.
    scan_file : string 
        Path to output file of profile_scan.
    best_fit_file : string
        Path to best fit output from profile_scan so the best fit point can
        be easily labelled on the output plots.
    extra_points : sequence
        List of extra points to be added to the plots. The exact form of
        these will depend on the postprocessing being done. For example,
        in the hypo_testing analysis postprocessing this could be a set of
        LLR points to put on the LLR plots as vertical lines. For the Asimov
        analyses they could be significances derived from the LLR analysis
        with appropriate errors.
    extra_points_labels : sequence
        A sequence of the same length as the extra points above so they can
        all be identified. 
    other_contours : sequence
        A sequence of paths to files containing other contours to be
        plotted on top of the current contour when using the profile
        scan postprocessing.
    projection_files : sequence
        A list of paths to files containing 1D projections of your multi-D
        profile scan. One must be passed for each dimension in your scan.
    pseudo_experiments : string
        Path to a logdir containing the output from hypo_testing analysis so
        the pseudo-experiment results can be added on to the profile_scan
        contour like a 2D scatter plot.
    inj_param_units : string
        The units used in the hypo_testing injparamscan for plots.

    Note that a single `logdir` can have different kinds of analyses run
    and results be logged within, so `fluctuate_fid` and `fluctuate_data`
    allows these to be separated from one another.
    """

    def __init__(self, analysis_type, detector, selection,
                 outdir, formats, test_type=None, logdir=None,
                 fluctuate_fid=None, fluctuate_data=None,
                 scan_file=None, best_fit_file=None,
                 extra_points=None, extra_points_labels=None,
                 other_contours=None, projection_files=None,
                 pseudo_experiments=None, inj_param_units=None):
        expected_analysis_types = ['hypo_testing', 'profile_scan', None]
        if analysis_type not in expected_analysis_types:
            raise ValueError(
                "Postprocessing only implemented for analyses of type %s "
                "but have been asked to process %s."%(
                    expected_analysis_types, analysis_type)
            )
        if analysis_type == 'hypo_testing':
            expected_test_types = ['analysis', 'injparamscan', 'systtests']
        elif analysis_type == 'profile_scan':
            expected_test_types = [None]
        elif analysis_type is None:
            expected_test_types = [None]
        if test_type not in expected_test_types:
            raise ValueError(
                "Postprocessing only implemented for %s analyses of test "
                "type %s but have been asked to process %s."%(
                    analysis_type, expected_test_types, test_type)
            )
        # Things to store for all postprocessing
        self.analysis_type = analysis_type
        self.test_type = test_type
        self.detector = detector
        self.selection = selection
        self.outdir = outdir
        self.formats = formats
        self.store_extra_points(
            extra_points=extra_points,
            extra_points_labels=extra_points_labels
        )
        self.fluctuate_fid = fluctuate_fid
        self.fluctuate_data = fluctuate_data
        # Things to initialise for hypo_testing
        if analysis_type == 'hypo_testing':
            self.test_type = test_type
            if test_type == 'injparamscan':
                self.logdirs = logdir
            else:
                self.logdir = logdir
            self.expected_pickles = [
                'data_sets.pckl',
                'all_params.pckl'
            ]
            if test_type == 'analysis' or test_type == 'injparamscan':
                self.expected_pickles.append(
                    'minimiser_info.pckl'
                )
            if test_type == 'injparamscan' or test_type == 'systtests':
                self.expected_pickles.append(
                    'labels.pckl'
                )
            if test_type == 'injparamscan':
                labels_list = []
                data_sets_list = []
                wh_to_th_list = []
                th_to_wh_list = []
                self.inj_param_vals = None
                self.inj_param_name = None
                for logdir in self.logdirs:
                    self.logdir = logdir
                    self.extract_trials()
                    labels_list.append(self.labels)
                    data_sets_list.append(self.data_sets)
                    wh_to_th_list.append(self.wh_to_th)
                    th_to_wh_list.append(self.th_to_wh)
                    self.get_inj_param_vals()
                    self.set_inj_param_units(inj_param_units=inj_param_units)
                self.labels = labels_list
                self.data_sets = data_sets_list
                self.wh_to_th = wh_to_th_list
                self.th_to_wh = th_to_wh_list
            else:
                self.extract_trials()
            if test_type == 'analysis':
                self.extract_fid_data()
                self.extract_data()
            if test_type == 'systtests':
                self.organise_baseline_data()
        # Things to initialise for profile_scan
        elif analysis_type == 'profile_scan':
            self.scan_file_dict = from_file(scan_file)
            if best_fit_file is not None:
                self.best_fit_dict = from_file(best_fit_file)
            else:
                self.best_fit_dict = None
            self.get_scan_steps()
            if projection_files is not None:
                if len(self.all_bin_cens) != 2:
                    raise ValueError(
                        "Can only deal with projection files for 2D scans."
                    )
                self.projection_dicts = []
                for projection_file in projection_files:
                    self.projection_dicts.append(from_file(projection_file))
            else:
                self.projection_dicts = None
            if other_contours is not None:
                if len(self.all_bin_cens) != 2:
                    raise ValueError(
                        "Can only deal with extra contours for 2D scans."
                    )
                self.contour_dicts = []
                for other_contour in other_contours:
                    self.contour_dicts.append(from_file(other_contour))
            else:
                self.contour_dicts = None
            if pseudo_experiments is not None:
                self.logdir = pseudo_experiments
                self.expected_pickles = [
                    'data_sets.pckl',
                    'all_params.pckl',
                    'minimiser_info.pckl'
                ]
                self.extract_trials()
                self.extract_fid_data()
                self.extract_data()
            else:
                self.logdir = None
            self.get_scan_data()

    def store_extra_points(self, extra_points, extra_points_labels):
        """Store the extra points to self"""
        if extra_points is not None:
            if extra_points_labels is not None:
                if len(extra_points) != len(extra_points_labels):
                    raise ValueError(
                        'You must specify at least one label for each set of '
                        'extra points. Got %i label(s) for %s set(s) of '
                        'extra points.'%(
                            len(extra_points), len(extra_points_labels)
                        )
                    )
            else:
                raise ValueError(
                    'You have specified %i set(s) of extra points but no '
                    'labels to go with them.'%len(extra_points)
                )
        else:
            if extra_points_labels is not None:
                raise ValueError(
                    'You have specified %i label(s) for extra points but no'
                    ' set(s) of extra points.'%len(extra_points_labels)
                )
        self.extra_points = extra_points
        self.extra_points_labels = extra_points_labels

    def add_deltam32_sin2theta23(self):
        """Will add deltam32 and sin2theta23 to be plotted,
        given that this is the more standard way of
        presenting these results."""

        # Get the deltam21 value used in the fits
        deltam21 = self.scan_file_dict['results'][0][
            'params']['deltam21']['value'][0]

        # Sort the bins
        for i, bin_name in enumerate(self.all_bin_names):
            if bin_name == 'theta23':
                self.all_bin_edges[i] = np.power(np.sin(
                    self.all_bin_edges[i]*ureg(
                        self.all_bin_units[i]
                    ).to('radians').magnitude), 2)
                self.all_bin_cens[i] = np.power(np.sin(
                    self.all_bin_cens[i]*ureg(
                        self.all_bin_units[i]
                    ).to('radians').magnitude), 2)
            elif bin_name == 'deltam31':
                self.all_bin_edges[i] = self.all_bin_edges[i] - deltam21
                self.all_bin_cens[i] = self.all_bin_cens[i] - deltam21
                ## Correct best fit, if needed
                if self.best_fit_data is not None:
                    self.best_fit_data['deltam32'] = {}
                    self.best_fit_data['deltam32']['units'] = \
                        self.best_fit_data['deltam31']['units']
                    self.best_fit_data['deltam32']['val'] = \
                        self.best_fit_data['deltam31']['val'] - deltam21
        # Correcting best fit/projection theta23 is easier
        if self.best_fit_data is not None:
            self.best_fit_data['sin2theta23'] = {}
            self.best_fit_data['sin2theta23']['units'] = \
                self.best_fit_data['theta23']['units']
            self.best_fit_data['sin2theta23']['val'] = \
                np.power(np.sin(self.best_fit_data['theta23']['val']*ureg(
                    self.all_bin_units[0]).to('radians').magnitude), 2)

        ## Correct projections, if needed
        if self.projection_data is not None:
            ## Correct bins
            new_proj_bin_names = []
            for i, proj_name in enumerate(self.proj_bin_names):
                ## Projection is a function of theta23
                if proj_name == 'theta23':
                    new_proj_bin_names.append('sin2theta23')
                    ## Correct bins
                    self.proj_bin_edges[i] = np.power(np.sin(
                        self.proj_bin_edges[i]*ureg(
                            self.proj_bin_units[i]
                        ).to('radians').magnitude), 2)
                    self.proj_bin_cens[i] = np.power(np.sin(
                        self.proj_bin_cens[i]*ureg(
                            self.proj_bin_units[i]
                        ).to('radians').magnitude), 2)
                    self.projection_data[i]['deltam32'] = {}
                    self.projection_data[i]['deltam32']['units'] = \
                        self.projection_data[i]['deltam31']['units']
                    self.projection_data[i]['deltam32']['vals'] = \
                        np.array(self.projection_data[i][
                            'deltam31']['vals']) - deltam21
                    del self.projection_data[i]['deltam31']
                ## Projection is a function of deltam31
                if proj_name == 'deltam31':
                    new_proj_bin_names.append('deltam32')
                    ## Correct bins
                    self.proj_bin_edges[i] = self.proj_bin_edges[i] - \
                        deltam21
                    self.proj_bin_cens[i] = self.proj_bin_cens[i] - \
                        deltam21
                    ## Need to also correct the theta23 fits
                    self.projection_data[i]['sin2theta23'] = {}
                    self.projection_data[i]['sin2theta23']['units'] = \
                        'dimensionless'
                    self.projection_data[i]['sin2theta23']['vals'] = \
                        np.power(np.sin(
                            np.array(
                                self.projection_data[i]['theta23']['vals']
                            )*ureg(
                                self.projection_data[i]['theta23']['units']
                            ).to('radians').magnitude), 2)
                    del self.projection_data[i]['theta23']

            self.proj_bin_names = new_proj_bin_names

        ## Correct pseudos, if needed
        if self.logdir is not None:
            for injkey in self.values.keys():
                for fhkey in self.values[injkey].keys():
                    self.values[injkey][fhkey]['sin2theta23'] = {}
                    self.values[injkey][fhkey]['sin2theta23'][
                        'units'] = 'dimensionless'
                    self.values[injkey][fhkey]['sin2theta23']['vals'] = \
                        np.power(np.sin(
                            np.array(
                                self.values[injkey][fhkey]['theta23']['vals']
                            )*ureg(
                                self.values[injkey][fhkey]['theta23']['units']
                            ).to('radians').magnitude), 2)
                    self.values[injkey][fhkey]['deltam32'] = {}
                    self.values[injkey][fhkey]['deltam32']['units'] = \
                        self.values[injkey][fhkey]['deltam31']['units']
                    self.values[injkey][fhkey]['deltam32']['vals'] = \
                        np.array(self.values[injkey][fhkey][
                            'deltam31']['vals']) - deltam21

    #### Hypo testing Specific Postprocessing functions ####

    def extract_trials(self):
        """Extract and aggregate analysis results."""
        self.logdir = os.path.expanduser(os.path.expandvars(self.logdir))
        logdir_content = os.listdir(self.logdir)
        # For the standard hypo_testing analysis, this logdir_content
        # will contain what we need it to and so we can proceed to
        # extract the trials.
        if self.test_type == 'analysis':
            if 'config_summary.json' in logdir_content:
                self.organise_trials(logdir_content=logdir_content)
            else:
                raise ValueError(
                    'config_summary.json cannot be found in the specified '
                    'logdir. It should have been created as part of the '
                    'output of hypo_testing.py and so this postprocessing '
                    'cannot be performed.'
                )
        elif self.test_type == 'injparamscan':
            pickle_there = self.check_pickle_files(logdir_content)
            if pickle_there:
                self.load_from_pickle()
            else:
                toy_names = []
                scan_variables = []
                for folder in logdir_content:
                    if '.pckl' not in folder and 'Plots' not in folder:
                        bits = folder.split('toy')[1].split('_')
                        toy_name = bits[1]
                        toy_names.append(
                            toy_name
                        )
                        scan_variable = None
                        add_bit = True
                        for bit in bits:
                            try:
                                float(bit)
                                add_bit = False
                            except:
                                if not (bit == '') and not (bit == toy_name):
                                    if add_bit:
                                        if scan_variable is None:
                                            scan_variable = bit
                                        else:
                                            scan_variable += '_%s'%bit
                        scan_variables.append(scan_variable)
                toy_names = np.array(toy_names)
                scan_variables = np.array(scan_variables)
                # Require all to be the same injected truth model
                if not np.alltrue(toy_names == toy_names[0]):
                    raise ValueError(
                        'Not all output is for the same injected truth '
                        'hypothesis. Got %s'%set(toy_names)
                    )
                # Require all to be scanning the same variable
                if not np.alltrue(scan_variables == scan_variables[0]):
                    raise ValueError(
                        'Not all output is for the same scanned parameter. '
                        'Got %s'%set(scan_variables)
                    )
                self.labels = {}
                self.all_params = {}
                self.data_sets = {}
                self.minimiser_info = {}
                for scandir in logdir_content:
                    if '.pckl' not in scandir and 'Plots' not in scandir:
                        self.scandir = os.path.join(self.logdir, scandir)
                        scandir_content = os.listdir(self.scandir)
                        if 'config_summary.json' in scandir_content:
                            self.extract_scans()
                        else:
                            raise ValueError(
                                'config_summary.json cannot be found in the '
                                'specified scandir, %s. It should have been '
                                'created as part of the output of '
                                'hypo_testing.py and so this postprocessing '
                                'cannot be performed.'%self.scandir
                            )
                # Pickle at the end so all of the scans are in the output
                self.pickle_data()
            self.organise_scans()
        elif self.test_type == 'systtests':
            pickle_there = self.check_pickle_files(logdir_content)
            if pickle_there:
                self.load_from_pickle()
            else:
                self.labels = {}
                self.all_params = {}
                self.data_sets = {}
                for systdir in logdir_content:
                    if '.pckl' not in systdir and 'Plots' not in systdir:
                        self.systdir = os.path.join(self.logdir, systdir)
                        systdir_content = os.listdir(self.systdir)
                        if 'config_summary.json' in systdir_content:
                            self.extract_systtests()
                        else:
                            raise ValueError(
                                'config_summary.json cannot be found in the '
                                'specified directory, %s. It should have been '
                                'created as part of the output of '
                                'hypo_testing.py and so this postprocessing '
                                'cannot be performed.'%self.systdir
                            )
                # Pickle at the end so all of the truths/systematics
                # are in the output
                self.pickle_data()

    def organise_trials(self, logdir_content):
        """This will actually go in to the directory where the trials 
        are and pull out the fit results."""
        config_summary_fpath = os.path.join(
            self.logdir,
            'config_summary.json'
        )
        cfg = from_file(config_summary_fpath)
        self.data_is_data = cfg['data_is_data']
        # Get naming scheme
        self.labels = Labels(
            h0_name=cfg['h0_name'],
            h1_name=cfg['h1_name'],
            data_name=cfg['data_name'],
            data_is_data=self.data_is_data,
            fluctuate_data=self.fluctuate_data,
            fluctuate_fid=self.fluctuate_fid
        )
        # Look for the pickle files in the directory to indicate
        # that this data may have already been processed.
        pickle_there = self.check_pickle_files(logdir_content)
        if pickle_there:
            self.load_from_pickle()
        # Else we must extract it
        else:
            if self.data_is_data and self.fluctuate_data:
                raise ValueError('Analysis was performed on data, so '
                                 '`fluctuate_data` is not supported.')
            # Get starting params
            self.get_starting_params(cfg=cfg)
            # Find all relevant data dirs, and from each extract the
            # fiducial fit(s) information contained
            self.get_data()
            self.pickle_data()

    def extract_scans(self):
        """This will actually go in to all of the scan directories and
        pull out the fit results."""
        config_summary_fpath = os.path.join(
            self.scandir,
            'config_summary.json'
        )
        cfg = from_file(config_summary_fpath)
        
        self.data_is_data = cfg['data_is_data']
        if self.data_is_data:
            raise ValueError('Analysis should NOT have been performed '
                             'on data since this script should only '
                             'process output from MC studies.')

        # Get naming scheme
        labels = Labels(
            h0_name=cfg['h0_name'],
            h1_name=cfg['h1_name'],
            data_name=cfg['data_name'],
            data_is_data=self.data_is_data,
            fluctuate_data=self.fluctuate_data,
            fluctuate_fid=self.fluctuate_fid
        )
        bits = labels.dict['data_name'].split('_')
        bare_truth = bits[0]
        injparam = None
        for bit in bits:
            if not (bit == bare_truth):
                if injparam is None:
                    injparam = bit
                else:
                    injparam += '_%s'%bit
        self.labels[injparam] = labels
        # Get starting params
        self.get_starting_params(cfg=cfg, injparam=injparam)
        self.get_data(injparam=injparam)

    def organise_scans(self):
        """Will organise the Asimov scans in to the metrics and the params
        with the appropriate wrong hypothesis to true hypothesis and true
        hypothesis to wrong hypothesis fits. Also extracts the minimiser
        info and saves it to the same object."""
        self.wh_to_th = {}
        self.th_to_wh = {}

        wh_to_th_metrics = []
        th_to_wh_metrics = []
        wh_to_th_params = {}
        th_to_wh_params = {}
        wh_to_th_minim_info = {}
        th_to_wh_minim_info = {}
        wh_to_th_minim_info['time'] = []
        wh_to_th_minim_info['iterations'] = []
        wh_to_th_minim_info['funcevals'] = []
        wh_to_th_minim_info['status'] = []
        th_to_wh_minim_info['time'] = []
        th_to_wh_minim_info['iterations'] = []
        th_to_wh_minim_info['funcevals'] = []
        th_to_wh_minim_info['status'] = []

        for injparam in sorted(self.data_sets.keys()):
            injlabels = self.labels[injparam].dict
            for injkey in self.data_sets[injparam].keys():
                h0_metric_val = self.data_sets[injparam][injkey][
                    'h0_fit_to_toy_%s_asimov'
                    %(injlabels['data_name'])]['metric_val']
                h1_metric_val = self.data_sets[injparam][injkey][
                    'h1_fit_to_toy_%s_asimov'
                    %(injlabels['data_name'])]['metric_val']
                if h1_metric_val > h0_metric_val:
                    bestfit = 'h0'
                    altfit = 'h1'
                else:
                    bestfit = 'h1'
                    altfit = 'h0'

                wh_to_th_fit = self.data_sets[injparam][injkey][
                    '%s_fit_to_%s_fid'%(altfit, bestfit)]['fid_asimov']
                th_to_wh_fit = self.data_sets[injparam][injkey][
                    '%s_fit_to_%s_fid'%(bestfit, altfit)]['fid_asimov']

                wh_to_th_metrics.append(wh_to_th_fit['metric_val'])
                th_to_wh_metrics.append(th_to_wh_fit['metric_val'])

                for systkey in wh_to_th_fit['params'].keys():
                    if systkey not in wh_to_th_params.keys():
                        wh_to_th_params[systkey] = []
                    wh_to_th_params[systkey].append(
                        wh_to_th_fit['params'][systkey]
                    )
                for systkey in th_to_wh_fit['params'].keys():
                    if systkey not in th_to_wh_params.keys():
                        th_to_wh_params[systkey] = []
                    th_to_wh_params[systkey].append(
                        th_to_wh_fit['params'][systkey]
                    )

                wh_to_th_minim_info['time'].append(
                    self.minimiser_info[injparam][injkey][
                        '%s_fit_to_%s_fid'%(altfit, bestfit)
                    ]['fid_asimov']['minimizer_time'])
                wh_to_th_minim_info['iterations'].append(
                    self.minimiser_info[injparam][injkey][
                        '%s_fit_to_%s_fid'%(altfit, bestfit)
                    ]['fid_asimov']['minimizer_metadata']['nit'])
                wh_to_th_minim_info['funcevals'].append(
                    self.minimiser_info[injparam][injkey][
                        '%s_fit_to_%s_fid'%(altfit, bestfit)
                    ]['fid_asimov']['minimizer_metadata']['nfev'])
                wh_to_th_minim_info['status'].append(
                    self.minimiser_info[injparam][injkey][
                        '%s_fit_to_%s_fid'%(altfit, bestfit)
                    ]['fid_asimov']['minimizer_metadata']['status'])
                
                th_to_wh_minim_info['time'].append(
                    self.minimiser_info[injparam][injkey][
                        '%s_fit_to_%s_fid'%(bestfit, altfit)
                    ]['fid_asimov']['minimizer_time'])
                th_to_wh_minim_info['iterations'].append(
                    self.minimiser_info[injparam][injkey][
                        '%s_fit_to_%s_fid'%(bestfit, altfit)
                    ]['fid_asimov']['minimizer_metadata']['nit'])
                th_to_wh_minim_info['funcevals'].append(
                    self.minimiser_info[injparam][injkey][
                        '%s_fit_to_%s_fid'%(bestfit, altfit)
                    ]['fid_asimov']['minimizer_metadata']['nfev'])
                th_to_wh_minim_info['status'].append(
                    self.minimiser_info[injparam][injkey][
                        '%s_fit_to_%s_fid'%(bestfit, altfit)
                    ]['fid_asimov']['minimizer_metadata']['status'])

        wh_to_th_params['bestfit'] = bestfit
        wh_to_th_params['altfit'] = altfit
        th_to_wh_params['bestfit'] = bestfit
        th_to_wh_params['altfit'] = altfit

        self.wh_to_th['metrics'] = wh_to_th_metrics
        self.th_to_wh['metrics'] = th_to_wh_metrics
        self.wh_to_th['params'] = wh_to_th_params
        self.th_to_wh['params'] = th_to_wh_params
        self.wh_to_th['minim_info'] = wh_to_th_minim_info
        self.th_to_wh['minim_info'] = th_to_wh_minim_info

    def extract_systtests(self):
        """This will actually go in to all of the systtest directories and
        pull out the fit results."""
        config_summary_fpath = os.path.join(
            self.systdir,
            'config_summary.json'
        )
        cfg = from_file(config_summary_fpath)

        self.data_is_data = cfg['data_is_data']
        if self.data_is_data:
            raise ValueError('Analysis should NOT have been performed '
                             'on data since this script should only '
                             'process output from MC studies.')

        # Get naming scheme
        labels = Labels(
            h0_name=cfg['h0_name'],
            h1_name=cfg['h1_name'],
            data_name=cfg['data_name'],
            data_is_data=self.data_is_data,
            fluctuate_data=self.fluctuate_data,
            fluctuate_fid=self.fluctuate_fid
        )
        # Special extraction for the baseline case of all systematics
        if 'full_syst_baseline' in labels.dict['data_name']:
            trueordering = labels.dict['data_name'].split('_')[0]
            systematic = 'full_syst_baseline'
            direction = None
        else:
            if 'fixed' in labels.dict['h0_name']:
                if 'inj' in labels.dict['data_name']:
                    testtype = 'fixwrong'
                    trueordering = labels.dict['data_name'].split('_')[0]
                    direction = labels.dict['data_name'].split('_')[-2]
                    systematic = labels.dict['data_name'].split(
                        '%s_inj_'%trueordering
                    )[-1].split('_%s_wrong'%direction)[0]
                else:
                    testtype = 'nminusone'
                    trueordering = labels.dict['data_name'].split('_')[0]
                    hypo = labels.dict['h0_name'].split('_')[0]
                    direction = None
                    systematic = labels.dict['h0_name'].split(
                        '%s_fixed_'%hypo
                    )[-1].split('_baseline')[0]
            else:
                testtype = 'fitwrong'
                trueordering = labels.dict['data_name'].split('_')[0]
                direction = labels.dict['data_name'].split('_')[-2]
                systematic = labels.dict['data_name'].split(
                    '%s_inj_'%trueordering
                )[-1].split('_%s_wrong'%direction)[0]
        trueordering = 'toy_%s_asimov'%trueordering
        if trueordering not in self.labels.keys():
            self.labels[trueordering] = {}
            self.all_params[trueordering] = {}
            self.data_sets[trueordering] = {}
        if systematic not in self.labels[trueordering].keys():
            self.labels[trueordering][systematic] = {}
            self.all_params[trueordering][systematic] = {}
            self.data_sets[trueordering][systematic] = {}
        if direction is not None:
            if direction not in self.labels[trueordering][systematic].keys():
                self.labels[trueordering][systematic][direction] = labels
                self.all_params[trueordering][systematic][direction] = {}
                self.data_sets[trueordering][systematic][direction] = {}
        else:
            self.labels[trueordering][systematic] = labels
        # Get starting params
        self.get_starting_params(cfg=cfg, trueordering=trueordering,
                                 systematic=systematic, direction=direction)
        self.get_data(trueordering=trueordering, systematic=systematic,
                      direction=direction)

    def systtest_fit_extract(self, fit_data, datakey, labels):
        """Function to extract the fit information relevant to the
        systematic tests for the given datakey. The labels are needed to
        check that the truth was recovered as it may not be in these
        systematic tests. This should be the .dict of the Labels object."""
        # Find which hypothesis is the best fit.
        h0_fit = fit_data['h0_fit_to_%s'%datakey]
        h1_fit = fit_data['h1_fit_to_%s'%datakey]
        if h0_fit['metric_val'] > h1_fit['metric_val']:
            bestfit = 'h1'
            altfit = 'h0'
        else:
            bestfit = 'h0'
            altfit = 'h1'
        # Extract the relevant fits
        best_to_alt_key = '%s_fit_to_%s_fid'%(bestfit,altfit)
        best_to_alt_fit = fit_data[best_to_alt_key]['fid_asimov']
        relevant_fit_data = {}
        relevant_fit_data['best_to_alt_fit'] = best_to_alt_fit
        relevant_fit_data['alt_to_best_fit'] = \
            fit_data['%s_fit_to_%s'%(altfit,datakey)]
        relevant_fit_data['best_to_best_fit'] = \
            fit_data['%s_fit_to_%s'%(bestfit,datakey)]
        # Since this is an MC study in Asimov, this _should_ also be
        # the injected truth. But, it is possible that if a
        # systematically wrong hypothesis has been injected that
        # this is NOT the case.
        truth_recovered = labels['%s_name'%bestfit] in labels['data_name'] or \
                          labels['data_name'] in labels['%s_name'%bestfit]
        relevant_fit_data['truth_recovered'] = truth_recovered
        return relevant_fit_data

    def make_systtest_plots(self):
        """Makes the plots showing the significances from
        the systematic tests"""
        for injkey in self.data_sets.keys():
            data = {}
            for testsyst in self.data_sets[injkey].keys():
                data[testsyst] = {}
                if len(self.data_sets[injkey][testsyst].keys()) == 2:
                    # This is the case of doing off-baseline tests
                    testdatakey = injkey.split('_asimov')[0] + \
                                  '_inj_%s_%s_wrong_asimov'%(testsyst,'nve')
                    fitted_syst = self.data_sets[injkey][testsyst]['nve'][
                        testdatakey]['h0_fit_to_%s'%testdatakey][
                            'params'].keys()
                    # If the number of fitted systematics equals the
                    # total number of systematics then the wrong thing
                    # was injected with the fitter being allowed to
                    # correct for it.
                    total_syst = self.data_sets[injkey].keys()
                    if len(fitted_syst) == len(total_syst):
                        self.systtest_type = 'fitwrong'
                        raise ValueError(
                            "Postprocessing for systematic tests where "
                            "one systematic is injected at a systematically"
                            " wrong value and the fitter is allowed to "
                            "correct for it is not implemented yet."
                        )
                    elif len(fitted_syst) == len(total_syst)-1:
                        self.systtest_type = 'fixwrong'
                        raise ValueError(
                            "Postprocessing for systematic tests where "
                            "one systematic is injected at a systematically"
                            " wrong value without allowing the fitter to "
                            "correct for it is not implemented yet."
                        )
                    else:
                        raise ValueError(
                            "The number of fitted systematics, %i, in "
                            "the test should be either equal to the "
                            "total number of systematics, %i, or 1 "
                            "less than it."%(len(fitted_syst), len(total_syst))
                        )
                else:
                    # Otherwise it's a standard N-1 test
                    self.systtest_type = 'nminusone'
                    data[testsyst] = self.systtest_fit_extract(
                        fit_data=self.data_sets[injkey][testsyst][injkey],
                        datakey=injkey,
                        labels=self.labels[injkey][testsyst].dict
                    )
            if self.systtest_type == 'nminusone':
                self.make_nminusone_systtests_plots(
                    data=data,
                    injkey=injkey
                )
            else:
                self.make_injwrong_systtests_plots(
                    data=data,
                    injkey=injkey
                )

    def make_injwrong_systtests_plots(self, data, injkey):
        """Make the systematic test plots where a systematically
        wrong hypothesis is injected."""
        # TODO
        raise ValueError(
            "Postprocessing for systematic tests where a systematically "
            "wrong hypothesis is injected is not implemented yet."
        )

    def make_nminusone_systtests_plots(self, data, injkey):
        """Make the N-1 test plots showing the importance of
        the systematics."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True

        mkdir(self.outdir)

        maintitle = self.make_main_title(end='N-1 Systematic Test')
        subtitle = r"For True %s\end{center}"%(
            self.tex_axis_label(injkey.split('_')[1])
        )

        if not self.baseline_data[injkey]['truth_recovered']:
            raise ValueError(
                "The truth was NOT recovered in the baseline fit."
            )
        baseline_significance = self.deltachi2_significance(
            wh_to_th_metrics=self.baseline_data[injkey][
                'alt_to_best_fit']['metric_val'],
            th_to_wh_metrics=self.baseline_data[injkey][
                'best_to_alt_fit']['metric_val']
        )
        alt_to_best_metrics = []
        best_to_alt_metrics = []
        truth_recovered = []
        for testsyst in data.keys():
            alt_to_best_metrics.append(
                data[testsyst]['alt_to_best_fit']['metric_val']
            )
            best_to_alt_metrics.append(
                data[testsyst]['best_to_alt_fit']['metric_val']
            )
            truth_recovered.append(
                data[testsyst]['truth_recovered']
            )
        alt_to_best_metrics = np.array(alt_to_best_metrics)
        best_to_alt_metrics = np.array(best_to_alt_metrics)
        significances = self.deltachi2_significance(
            wh_to_th_metrics=alt_to_best_metrics,
            th_to_wh_metrics=best_to_alt_metrics
        )
        truth_recovered = np.array(truth_recovered)
        if not np.all(truth_recovered):
            systnames = []
            for systname in np.array(data.keys())[significances.argsort()]:
                systnames.append(systname)
            systnames = np.array(systnames)
            raise ValueError(
                "Truth not recovered in tests where %s were fixed."%
                systnames[np.logical_not(truth_recovered)]
            )
        systnames = []
        for systname in np.array(data.keys())[significances.argsort()]:
            systnames.append(self.tex_axis_label(systname))
        vrange = max(significances)-min(significances)
        ylims = [min(significances)-0.1*vrange, max(significances)+0.1*vrange]
        self.make_1d_graph(
            xvals=np.linspace(0.5,len(significances)-0.5,len(significances)),
            yvals=significances[significances.argsort()],
            xlabel='Fixed Systematic',
            nicexlabel=False,
            xunits=None,
            ylabel=r'Asimov Significance ($\sigma$)',
            niceylabel=False,
            yunits=None,
            marker='x',
            xlims=[0,len(significances)],
            linestyle='',
            color=self.plot_colour(injkey.split('_')[1]),
            ylims=ylims
        )
        plt.xticks(
            np.linspace(0.5,len(significances)-0.5,len(significances)),
            systnames,
            rotation=45,
            horizontalalignment='right'
        )
        plt.grid(axis='y', zorder=0, linestyle='--')
        plt.title(maintitle+r"\\"+subtitle, fontsize=16)
        plt.tight_layout()
        save_end = "nminusone_systematic_test"
        self.save_plot(
            outdir=self.outdir,
            end=save_end,
            truth=injkey.split('_')[1]
        )
        # Add a line showing the baseline significance
        plt.axhline(
            baseline_significance,
            linestyle='--',
            label='Baseline Asimov Significance',
            color='k',
            lw=2
        )
        plt.legend(loc='upper left')
        save_end = "nminusone_systematic_test_w_baseline"
        self.save_plot(
            outdir=self.outdir,
            end=save_end,
            truth=injkey.split('_')[1]
        )
        plt.close()
        # Do another plot with the baseline subtracted
        corrected_significances = significances - baseline_significance
        vrange = max(corrected_significances)-min(corrected_significances)
        ylims = [min(corrected_significances)-0.1*vrange,
                 max(corrected_significances)+0.1*vrange]
        self.make_1d_graph(
            xvals=np.linspace(0.5,len(significances)-0.5,len(significances)),
            yvals=corrected_significances[significances.argsort()],
            xlabel='Fixed Systematic',
            nicexlabel=False,
            xunits=None,
            ylabel=r'Change in Asimov Significance ($\sigma$)',
            niceylabel=False,
            yunits=None,
            marker='x',
            xlims=[0,len(significances)],
            linestyle='',
            color=self.plot_colour(injkey.split('_')[1]),
            ylims=ylims
        )
        plt.xticks(
            np.linspace(0.5,len(significances)-0.5,len(significances)),
            systnames,
            rotation=45,
            horizontalalignment='right'
        )
        plt.grid(axis='y', zorder=0, linestyle='--')
        plt.title(maintitle+r"\\"+subtitle, fontsize=16)
        plt.tight_layout()
        save_end = "nminusone_systematic_test_baseline_subtracted"
        self.save_plot(
            outdir=self.outdir,
            end=save_end,
            truth=injkey.split('_')[1]
        )
        plt.close()

    def organise_baseline_data(self):
        """Store the baseline fits in the systematic tests to self."""
        self.baseline_data = {}
        for injkey in self.data_sets.keys():
            data = {}
            baseline_result = self.data_sets[injkey].pop('full_syst_baseline')
            datakey = baseline_result.keys()[0]
            baseline_data = self.systtest_fit_extract(
                fit_data=baseline_result[datakey],
                datakey=datakey,
                labels=self.labels[injkey]['full_syst_baseline'].dict
            )
            self.baseline_data[injkey] = baseline_data

    def extract_fit(self, fpath, keys=None):
        """Extract fit info from a file.

        Parameters
        ----------
        fpath : string
            Path to the file

        keys : None, string, or iterable of strings
            Keys to extract. If None, all keys are extracted.

        """
        try:
            info = from_file(fpath)
        except:
            raise RuntimeError("Cannot read from file located at %s."%fpath)
        if keys is None:
            return info
        if isinstance(keys, str):
            keys = [keys]
        for key in info.keys():
            if key not in keys:
                info.pop(key)
        return info

    def get_hypo_from_fiducial_hypo_key(self, fhkey):
        """Returns the hypo from the fiducial/fit-hypothesis key"""
        return fhkey.split('_')[0]

    def get_fid_from_fiducial_hypo_key(self, fhkey):
        """Returns the fid from the fiducial/fit-hypothesis key"""
        return fhkey.split('_')[-2]

    def extract_paramval(self, injparams, systkey, fhkey=None,
                         paramlabel=None, smalllabel=False):
        """Extract a value from a set of parameters and modify it based on the
        hypothesis/fiducial fit being considered. The label associated with this
        is then modified accordingly."""
        paramval = float(injparams[systkey].split(' ')[0])
        if (fhkey is None) or (paramlabel is None):
            if not ((fhkey is None) and (paramlabel is None)):
                raise ValueError(
                    "Either both fhkey and paramlabel must be"
                    " None or they must both be specified."
                )
            return paramval
        else:
            hypo = self.get_hypo_from_fiducial_hypo_key(fhkey=fhkey)
            hypo_label = self.labels.dict['%s_name'%hypo]
            if systkey == 'deltam31':
                if 'no' in hypo_label:
                    if np.sign(paramval) != 1:
                        paramval = -1*float(injparams[systkey].split(' ')[0])
                        if not smalllabel:
                            paramlabel += r' ($\times-1$)'
                elif 'io' in hypo_label:
                    if np.sign(paramval) != -1:
                        paramval = -1*float(injparams[systkey].split(' ')[0])
                        if not smalllabel:
                            paramlabel += r' ($\times-1$)'

            if (np.abs(paramval) < 1e-2) and (paramval != 0.0):
                paramlabel += ' = %.2e'%paramval
            else:
                paramlabel += ' = %.3g'%paramval

            return paramval, paramlabel

    def extract_fid_data(self):
        """Take the data sets returned by the `extract_trials` and extract the
        data on the fiducial fits."""
        fid_values = {}
        for injkey in self.data_sets.keys():
            fid_values[injkey] = {}
            for datakey in self.data_sets[injkey]:
                if ('toy' in datakey) or ('data' in datakey):
                    fid_values[injkey][datakey] \
                        = self.data_sets[injkey].pop(datakey)
        self.fid_values = fid_values

    def extract_gaussian(self, prior_string, units):
        """Parses the string for the Gaussian priors that comes from the
        config summary file in the logdir. This should account for dimensions
        though has only been proven with "deg" and "ev ** 2"."""
        if units == 'dimensionless':
            parse_string = ('gaussian prior: stddev=(.*)'
                            ' , maximum at (.*)')
            bits = re.match(
                parse_string,
                prior_string,
                re.M|re.I
            )
            stddev = float(bits.group(1))
            maximum = float(bits.group(2))
        else:
            try:
                # This one works for deg and other single string units
                parse_string = ('gaussian prior: stddev=(.*) (.*)'
                                ', maximum at (.*) (.*)')
                bits = re.match(
                    parse_string,
                    prior_string,
                    re.M|re.I
                )
                stddev = float(bits.group(1))
                maximum = float(bits.group(3))
            except:
                # This one works for ev ** 2 and other triple string units
                parse_string = ('gaussian prior: stddev=(.*) (.*) (.*) (.*)'
                                ', maximum at (.*) (.*) (.*) (.*)')
                bits = re.match(
                    parse_string,
                    prior_string,
                    re.M|re.I
                )
                stddev = float(bits.group(1))
                maximum = float(bits.group(5))

        return stddev, maximum

    ######## Hypo testing Analysis Specific Postprocessing functions ########

    def extract_data(self):
        """Take the data sets returned by `extract_trials` and turn them in to a
        format used by all of the plotting functions."""
        values = {}
        for injkey in self.data_sets.keys():
            values[injkey] = {}
            alldata = self.data_sets[injkey]
            paramkeys = alldata['params'].keys()
            for datakey in alldata.keys():
                if not datakey == 'params':
                    values[injkey][datakey] = {}
                    values[injkey][datakey]['metric_val'] = {}
                    values[injkey][datakey]['metric_val']['vals'] = []
                    for paramkey in paramkeys:
                        values[injkey][datakey][paramkey] = {}
                        values[injkey][datakey][paramkey]['vals'] = []
                    trials = alldata[datakey]
                    for trial_num in trials.keys():
                        trial = trials[trial_num]
                        values[injkey][datakey]['metric_val']['vals'] \
                            .append(trial['metric_val'])
                        values[injkey][datakey]['metric_val']['type'] \
                            = trial['metric']
                        values[injkey][datakey]['metric_val']['units'] \
                            = 'dimensionless'
                        param_vals = trial['params']
                        for param_name in param_vals.keys():
                            val, units = self.parse_pint_string(
                                pint_string=param_vals[param_name]
                            )
                            values[injkey][datakey][param_name]['vals'] \
                                .append(float(val))
                            values[injkey][datakey][param_name]['units'] \
                                = units
        self.values = values

    def purge_outlying_trials(self, trial_nums, thresh=5.0):
        """Look at the values of the metric and find any deemed to be from a
        failed job. That is, the value of the metric falls very far outside of
            the rest of the values.

        Notes
        -----
        Interestingly, I only saw a need for this with my true NO jobs, where I
        attempted to run some jobs in fp32 mode. No jobs were needed to be
        removed for true IO, where everything was run in fp64 mode. So if
        there's a need for this function in your analysis it probably points
        at some more serious underlying problem.

        References:
        ----------
        Taken from stack overflow:

            http://stackoverflow.com/questions/22354094/pythonic-way-\
            of-detecting-outliers-in-one-dimensional-observation-data

        which references:

            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect
            and Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
        """
        for injkey in self.values.keys():
            for fit_key in self.values[injkey].keys():
                points = np.array(self.values[injkey][
                    fit_key]['metric_val']['vals'])
                if len(points.shape) == 1:
                    points = points[:, None]
                median = np.median(points, axis=0)
                diff = np.sum((points - median)**2, axis=-1)
                diff = np.sqrt(diff)
                med_abs_deviation = np.median(diff)
                modified_z_score = 0.6745 * diff / med_abs_deviation
                good_trials = modified_z_score < thresh
                if not np.all(good_trials):
                    bad_trials = np.where(not good_trials)[0]
                    logging.warning(
                        'Outlier(s) detected for %s in trial(s) %s. Will be '
                        'removed. If you think this should not happen, please '
                        'change the value of the threshold used for the '
                        'decision (currently set to %.2e).'%(
                            fit_key, trial_nums[bad_trials], thresh
                        )
                    )
                    for fitkey in self.values[injkey].keys():
                        for param in self.values[injkey][fitkey].keys():
                            new_vals = np.delete(
                                np.array(self.values[injkey][
                                    fitkey][param]['vals']),
                                bad_trials
                            )
                            self.values[injkey][
                                fitkey][param]['vals'] = new_vals

    def get_resulting_hypo_params(self, injkey):
        """Returns the sets of h0 and h1 fits to the data"""
        h0_params = self.fid_values[injkey][
            'h0_fit_to_%s'%(self.labels.dict['data'])]['params']
        h1_params = self.fid_values[injkey][
            'h1_fit_to_%s'%(self.labels.dict['data'])]['params']
        return h0_params, h1_params

    def get_inj_param_vals(self):
        """Adds the set of injected parameters to self. If this is not
        None then it ensures that this matches what is already there
        i.e. when specifying multiple scan directories."""
        inj_params = self.data_sets.keys()
        inj_param_vals = []
        for inj_param in inj_params:
            inj_param_vals.append(float(inj_param.split('_')[-1]))
        if self.inj_param_vals is None:
            self.inj_param_name = inj_params[0].split(
                '_%.4f'%inj_param_vals[0]
            )[0]
            self.inj_param_vals = sorted(inj_param_vals)
        else:
            inj_param_name = inj_params[0].split(
                '_%.4f'%inj_param_vals[0]
            )[0]
            inj_param_vals = sorted(inj_param_vals)
            if self.inj_param_name != inj_param_name:
                raise ValueError(
                    "You have requested to plot multiple scan directories "
                    "that do not seem to over the same injected scan "
                    "parameter. Have %s in self but %s in the current "
                    "directory."%(self.inj_param_name, inj_param_name)
                )
            if self.inj_param_vals != inj_param_vals:
                raise ValueError(
                    "You have requested to plot multiple scan directories "
                    "that do not seem to over the same injected scan "
                    "range. Have %s in self but %s in the current "
                    "directory."%(self.inj_param_vals, inj_param_vals)
                )

    def set_inj_param_units(self, inj_param_units):
        """Sets the inj param units in to self. Typically it will do this by
        finding one of the fit parameters and extracting it. This requires
        that it is in the fitted parameters and that you scanned in the same
        units as was fit. If this is not the case you must set it in the
        arguments to the script."""
        if inj_param_units is None:
            # If the inj param units are not specified they should
            # be found from the units attached to the fitted parameters.
            # There are two known exceptions.
            if self.inj_param_name == 'sin2theta23':
                self.inj_param_units = 'dimensionless'
            elif self.inj_param_name == 'deltam3l':
                self.inj_param_units = 'electron_volt ** 2'
            else:
                if self.inj_param_name not in self.wh_to_th['params'].keys():
                    raise ValueError(
                        "The injected parameter %s could not be found in "
                        "the fitted parameters: %s. Please use the script"
                        " argument to set the injected parameter units "
                        "manually"%(self.inj_param_name,
                                    self.wh_to_th['params'].keys())
                    )
                else:
                    val, inj_param_units = self.parse_pint_string(
                        pint_string=self.wh_to_th['params'][
                            self.inj_param_name][0]
                    )
                    
        self.inj_param_units = inj_param_units

    def get_injected_params(self):
        """Return the injected params, if they exist"""
        if 'data_params' in self.all_params.keys():
            if self.all_params['data_params'] is not None:
                data_params = {}
                for pkey in self.all_params['data_params'].keys():
                    data_params[pkey] = \
                        self.all_params['data_params'][pkey]['value']
            else:
                data_params = None
        else:
            data_params = None
        return data_params

    def deltachi2_significance(self, wh_to_th_metrics,
                                          th_to_wh_metrics,
                                          truth_recovered=None):
        """Calculates the Asimov significance from the sets of metrics."""
        if isinstance(wh_to_th_metrics, list):
            wh_to_th_metrics = np.array(wh_to_th_metrics)
            th_to_wh_metrics = np.array(th_to_wh_metrics)
        num = wh_to_th_metrics + th_to_wh_metrics
        denom = 2 * np.sqrt(wh_to_th_metrics)
        significances = num/denom
        if truth_recovered is not None:
            truth_multiplier = []
            for tr in truth_recovered:
                if tr:
                    truth_multiplier.append(1.0)
                else:
                    truth_multiplier.append(-1.0)
            truth_multiplier = np.array(truth_multiplier)
            significances *= truth_multiplier
        return significances
        
    def make_asimov_significance_plots(self):
        """Makes the Asimov significance plots."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        outdir = os.path.join(self.outdir, 'Significances')
        mkdir(outdir)
        maintitle = self.make_main_title(
            end='Asimov Analysis Significances',
            end_center=True
        )

        # Want to ensure the resulting y range can show all of the plots
        # Therefore find the max and min sig of the whole set of data_sets
        maxsig = None
        minsig = None
        # xrange is easier
        hrange = self.inj_param_vals[-1]-self.inj_param_vals[0]
        xlims = [self.inj_param_vals[0]-0.1*hrange,
                 self.inj_param_vals[-1]+0.1*hrange]

        for i in range(len(self.data_sets)):

            significances = self.deltachi2_significance(
                wh_to_th_metrics=self.wh_to_th[i]['metrics'],
                th_to_wh_metrics=self.th_to_wh[i]['metrics']
            )

            truth = self.labels[i][
                self.labels.keys()[0]].dict['data_name'].split('_')[0]
            plotlabel = 'True %s'%self.tex_axis_label(truth)
            
            self.make_1d_graph(
                xvals=self.inj_param_vals,
                yvals=significances,
                xlabel=self.inj_param_name,
                xunits=self.inj_param_units,
                ylabel=None,
                yunits=None,
                marker=self.marker_style(truth),
                color=self.plot_colour(truth),
                plotlabel=plotlabel,
                xlims=xlims
            )

            if maxsig is None:
                maxsig = max(significances)
            else:
                maxsig = max(maxsig, max(significances))
            if minsig is None:
                minsig = min(significances)
            else:
                minsig = min(minsig, min(significances))

        # Give a more descriptive y-axis label if only one thing being plotted
        if len(self.data_sets) == 1:
            alt = self.labels[
                self.labels.keys()[0]].dict['%s_name'%(
                    self.wh_to_th[0]['params']['altfit'])].split('_')[0]
            plt.ylabel(r'%s from %s Significance $\left(\sigma\right)$'%(
                self.tex_axis_label(truth),
                self.tex_axis_label(alt)
            ))
        else:
            plt.ylabel(r'Significance $\left(\sigma\right)$', fontsize=24)

        vrange = maxsig - minsig
        plt.ylim(minsig-0.1*vrange, maxsig+0.2*vrange)
        plt.title(maintitle, fontsize=16)
        plt.legend(loc='best')
        plt.tight_layout()
        save_end = "%s_asimov_significances"%(self.inj_param_name)
        self.save_plot(outdir=outdir, end=save_end, truth=truth)
        if self.extra_points is not None:
            yminextra, ymaxextra = self.add_extra_points()
            yminall = min(yminextra, minsig)
            ymaxall = max(ymaxextra, maxsig)
            vrange = ymaxall - yminall
            if yminall == 0:
                plt.ylim(yminall, ymaxall+0.2*vrange)
            else:
                plt.ylim(yminall-0.1*vrange, ymaxall+0.3*vrange)
            plt.legend(loc='upper left')
            save_end = "%s_asimov_significances_w_extra_points"%(
                self.inj_param_name)
            self.save_plot(outdir=outdir, end=save_end, truth=truth)
        plt.close()

    def make_asimov_fit_parameter_plots(self, combined=False):
        """Makes the plots of the fitted parameters in the injected
        parameter scans. The combined argument will plot these on the
        same canvas for the same hypothesis fit."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        
        if combined:
            outdir = os.path.join(self.outdir, 'CombinedBestFits')
        else:
            outdir = os.path.join(self.outdir, 'IndividualBestFits')
        mkdir(outdir)
        
        maintitle = self.make_main_title(
            end='Asimov Analysis'
        )

        hrange = self.inj_param_vals[-1]-self.inj_param_vals[0]
        xlims = [self.inj_param_vals[0]-0.1*hrange,
                 self.inj_param_vals[-1]+0.1*hrange]

        th = self.labels[self.labels.keys()[0]].dict[
            '%s_name'%self.th_to_wh[0]['params']['bestfit']]
        wh = self.labels[self.labels.keys()[0]].dict[
            '%s_name'%self.th_to_wh[0]['params']['altfit']]

        th_to_wh_label = "%s fit to %s fiducial"%(
            self.tex_axis_label(th),
            self.tex_axis_label(wh)
        )
        wh_to_th_label = "%s fit to %s fiducial"%(
            self.tex_axis_label(wh),
            self.tex_axis_label(th)
        )
        fitlabels = [th_to_wh_label, wh_to_th_label]

        subtitle = "True %s Best Fit Parameters\end{center}"%(self.tex_axis_label(th))

        # Set up multi-plot if needed
        if combined:
            num_rows = self.get_num_rows(
                data=self.th_to_wh[0]['params'],
                omit_metric=False
            )
            plt.figure(figsize=(20, 5*num_rows+2))
            subplotnum = 1
        else:
            subplotnum = None

        for param in self.th_to_wh[0]['params'].keys():
            if param not in ['bestfit', 'altfit']:
                ymax = None
                ymin = None
                for fit, fitname, fitlabel in zip(
                        [self.th_to_wh, self.wh_to_th],
                        ['th_to_wh', 'wh_to_th'],
                        fitlabels):
                    vals = []
                    for param_val in fit[0]['params'][param]:
                        val, units = self.parse_pint_string(
                            pint_string=param_val
                        )
                        if param == 'deltam31':
                            vals.append(np.abs(float(val)))
                        else:
                            vals.append(float(val))
                    # Specify the subplot, if necessary
                    if combined:
                        plt.subplot(num_rows, 4, subplotnum)
                    self.make_1d_graph(
                        xvals=self.inj_param_vals,
                        yvals=vals,
                        xlabel=self.inj_param_name,
                        xunits=self.inj_param_units,
                        ylabel=param,
                        yunits=units,
                        marker=self.marker_style(fitname),
                        color=self.plot_colour(fitname),
                        plotlabel=fitlabel,
                        xlims=xlims
                    )

                    if ymax is None:
                        ymax = max(vals)
                    else:
                        ymax = max(ymax, max(vals))
                    if ymin is None:
                        ymin = min(vals)
                    else:
                        ymin = min(ymin, min(vals))

                yrange = ymax - ymin
                plt.ylim(ymin-0.1*yrange, ymax+0.2*yrange)
                plt.legend(loc='upper left')
                # Advance the subplot number, if necessary
                if combined:
                    subplotnum += 1
                # Else, save/close this plot
                else:
                    plt.title(r'%s \\ %s'%(maintitle,subtitle))
                    plt.tight_layout()
                    save_end = "%s_%s_best_fit_values"%(self.inj_param_name,
                                                    param)
                    self.save_plot(outdir=outdir, end=save_end, truth=th)
                    plt.close()
        # Save the whole canvas, if necessary
        if combined:
            plt.suptitle(r'%s \\ %s'%(maintitle,subtitle), fontsize=36)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            save_end = "%s_all_best_fit_values"%(self.inj_param_name)
            self.save_plot(outdir=outdir, end=save_end, truth=th)
            plt.close()                

    def make_scatter_plots(self, combined=False,
                           singlesyst=False, matrix=False):
        """Make scatter plots."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        if matrix:
            if combined or singlesyst:
                raise ValueError(
                    "Function must be used to plot the correlation "
                    "matrix or the scatter plots, but not both at "
                    "the same time."
                )
            try:
                import matplotlib.patheffects as PathEffects
                logging.warning(
                    "PathEffects could be imported, so the correlation values"
                    " will be written on the bins. This is slow."
                )
                pe = True
            except ImportError:
                logging.warning(
                    "PathEffects could not be imported, so the correlation"
                    " values will not be written on the bins.")
                pe = False
            outdir = os.path.join(self.outdir, 'CorrelationMatrices')
            maintitle = self.make_main_title(end='Correlation Coefficients')
        else:
            if combined:
                outdir = os.path.join(self.outdir, 'CombinedScatterPlots')
                maintitle = self.make_main_title(end='Correlation Plots')
            else:
                outdir = os.path.join(self.outdir, 'IndividualScatterPlots')
                maintitle = self.make_main_title(end='Correlation Plot')
        mkdir(outdir)
        # These arguments to the scattering plot must be none
        # for the case of individual plots.
        if not combined:
            num_rows = None
            subplotnum = None
            plot_cor = True

        for injkey in self.values.keys():
            for fhkey in self.values[injkey].keys():
                systs = []
                for systkey in self.values[injkey][fhkey].keys():
                    if not systkey == 'metric_val':
                        systs.append(systkey)
                fittitle = self.make_fit_title(
                    fhkey=fhkey,
                    trials=self.num_trials
                )
                # Set up container for correlation coefficients
                # containers, if necessary
                if matrix:
                    all_corr_lists = []
                # Set up multi-plot, if necessary
                ## Need a square of size numsyst x numsyst for all combined
                if combined and (not singlesyst):
                    # Systematic number is one less than number
                    # of keys since this also contains the metric_val entry
                    systnum = len(self.values[injkey][fhkey].keys())-1
                    plt.figure(figsize=(3.5*(systnum-1), 3.5*(systnum-1)))
                    subplotnum = (systnum-1)*(systnum-1)+1
                    # Set up container to know which correlations
                    # have already been plotted
                    plottedsysts = []
                    num_rows = None
                    plot_cor = False
                for xsystkey in systs:
                    # Set up container for correlation
                    # coefficients if necessary
                    if matrix:
                        all_corr_values = []
                    if combined and (not singlesyst):
                        plottedsysts.append(xsystkey)
                    # Set up multi-plot, if necessary
                    ## One subplot for each systematic
                    if combined and singlesyst:
                        num_rows = self.get_num_rows(
                            data=self.values[injkey][fhkey],
                            omit_metric=False
                        )
                        plt.figure(figsize=(20, 5*num_rows+2))
                        subplotnum = 1
                        plot_cor = True
                    for ysystkey in systs:
                        if matrix:
                            rho, pval = self.get_correlation_coefficient(
                                xdata=self.values[injkey][fhkey][
                                    xsystkey]['vals'],
                                ydata=self.values[injkey][fhkey][
                                    ysystkey]['vals'],
                                xsystkey=xsystkey,
                                ysystkey=ysystkey
                            )
                            all_corr_values.append(rho)

                        if not ysystkey == xsystkey:

                            if combined and (not singlesyst):
                                # Subplotnum counts backwards in the case of
                                # putting all correlations on one canvas.
                                subplotnum -= 1
                                # Don't repeat plotted systematics
                                if ysystkey not in plottedsysts:
                                    do_plot = True
                                    plt.subplot(
                                        systnum-1,
                                        systnum-1,
                                        subplotnum
                                    )
                                else:
                                    do_plot = False
                            # Don't plot the scatters when making the matrices
                            elif matrix:
                                do_plot = False
                            # Plot is always wanted in other cases
                            else:
                                do_plot = True

                            # Set up subplot, if necessary
                            if combined and singlesyst:
                                plt.subplot(num_rows, 4, subplotnum)
                            if do_plot:
                                self.make_2d_scatter_plot(
                                    xdata=self.values[injkey][fhkey][
                                        xsystkey]['vals'],
                                    ydata=self.values[injkey][fhkey][
                                        ysystkey]['vals'],
                                    xlabel=xsystkey,
                                    xunits=self.values[injkey][fhkey][
                                        xsystkey]['units'],
                                    ylabel=ysystkey,
                                    yunits=self.values[injkey][fhkey][
                                        ysystkey]['units'],
                                    title=maintitle+r'\\'+fittitle,
                                    num_rows=num_rows,
                                    subplotnum=subplotnum,
                                    plot_cor=plot_cor
                                )
                            # Advance the subplot number, if necessary
                            if combined and singlesyst:
                                subplotnum += 1
                            # Save/close this plot, if necessary
                            if not combined and not matrix:
                                self.save_plot(
                                    fhkey=fhkey,
                                    outdir=outdir,
                                    end='%s_%s_scatter_plot'%(
                                        xsystkey,
                                        ysystkey
                                    )
                                )
                                plt.close()
                    # Store the list of correlation values for plotting
                    if matrix:
                        all_corr_lists.append(all_corr_values)
                    # Save/close this plot, if necessary
                    if combined and singlesyst:
                        plt.suptitle(maintitle+r'\\'+fittitle, fontsize=36)
                        plt.tight_layout()
                        plt.subplots_adjust(top=0.9)
                        self.save_plot(
                            fhkey=fhkey,
                            outdir=outdir,
                            end='%s_scatter_plots'%(
                                xsystkey
                            )
                        )
                        plt.close()

                if matrix:
                    texsysts = []
                    for syst in systs:
                        texsysts.append(self.tex_axis_label(syst))
                    all_corr_nparray = np.ma.masked_invalid(
                        np.array(all_corr_lists)
                    )
                    self.make_2d_hist_plot(
                        zvals=all_corr_nparray,
                        xbins=np.linspace(-0.5, len(systs)-0.5, len(systs)+1),
                        ybins=np.linspace(-0.5, len(systs)-0.5, len(systs)+1),
                        xlabel=None,
                        xunits=None,
                        ylabel=None,
                        yunits=None,
                        zlabel='correlation_coefficients',
                        zunits=None,
                        xticks=texsysts,
                        yticks=texsysts,
                        cmap=plt.cm.RdBu
                    )
                    plt.subplots_adjust(
                        bottom=0.30,
                        left=0.27,
                        right=0.95,
                        top=0.88
                    )
                    plt.title(maintitle+r'\\'+fittitle, fontsize=16)
                    self.save_plot(
                        fhkey=fhkey,
                        outdir=outdir,
                        end='correlation_matrix'
                    )
                    if pe:
                        self.add_annotation_to_2d_hist(
                            annotations=all_corr_nparray
                        )
                        self.save_plot(
                            fhkey=fhkey,
                            outdir=outdir,
                            end='correlation_matrix_values'
                        )
                    plt.close()
                if combined and (not singlesyst):
                    plt.suptitle(maintitle+r'\\'+fittitle, fontsize=120)
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.9)
                    self.save_plot(
                        fhkey=fhkey,
                        outdir=outdir,
                        end='all_scatter_plots'
                    )
                    plt.close()

    def make_posterior_plots(self, combined=False):
        """Make posterior plots. With combined=False they will be saved
        each time but with combined=True they will be saved on a single
        canvas for each fiducial/hypothesis combination."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True

        if combined:
            outdir = os.path.join(self.outdir, 'CombinedPosteriors')
            maintitle = self.make_main_title(end='Posteriors')
        else:
            outdir = os.path.join(self.outdir, 'IndividualPosteriors')
            maintitle = self.make_main_title(end='Posterior')
        mkdir(outdir)

        for injkey in self.values.keys():
            for fhkey in self.values[injkey].keys():
                # Set up multi-plot if needed
                if combined:
                    num_rows = self.get_num_rows(
                        data=self.values[injkey][fhkey],
                        omit_metric=False
                    )
                    plt.figure(figsize=(20, 5*num_rows+2))
                    subplotnum = 1
                else:
                    subplotnum = None
                # Loop through the systematics
                for systkey in self.values[injkey][fhkey].keys():
                    fittitle = self.make_fit_title(
                        fhkey=fhkey,
                        trials=self.num_trials
                    )
                    systunits = self.values[injkey][fhkey][systkey]['units']
                    if systkey == 'metric_val':
                        xlabel = self.tex_axis_label(
                            self.values[injkey][fhkey][systkey]['type']
                        )
                    else:
                        xlabel = self.tex_axis_label(systkey)
                    if not systunits == 'dimensionless':
                        xlabel += r' (%s)'%self.tex_axis_label(systunits)
                    # Specify the subplot, if necessary
                    if combined:
                        plt.subplot(num_rows, 4, subplotnum)
                    self.make_1d_hist_plot(
                        data=np.array(
                            self.values[injkey][fhkey][systkey]['vals']
                        ),
                        xlabel=xlabel,
                        title=maintitle+r'\\'+fittitle,
                        ylabel='Number of Trials',
                        subplotnum=subplotnum
                    )
                    # Add the details i.e. injected/fiducial lines and priors
                    plt.ylim(0, 1.35*plt.ylim()[1])
                    if not systkey == 'metric_val':
                        self.add_inj_fid_lines(
                            injkey=injkey,
                            systkey=systkey,
                            fhkey=fhkey
                        )
                        self.add_prior_region(
                            injkey=injkey,
                            systkey=systkey,
                            fhkey=fhkey
                        )
                        plt.legend(
                            loc='upper left',
                            fontsize=12,
                            framealpha=1.0
                        )
                    plt.subplots_adjust(
                        left=0.10,
                        right=0.90,
                        top=0.85,
                        bottom=0.11
                    )
                    # Advance the subplot number, if necessary
                    if combined:
                        subplotnum += 1
                    # Else, save/close this plot
                    else:
                        self.save_plot(
                            fhkey=fhkey,
                            outdir=outdir,
                            end='%s_posterior'%systkey
                        )
                        plt.close()
                # Save the whole canvas, if necessary
                if combined:
                    plt.suptitle(maintitle+r'\\'+fittitle, fontsize=36)
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.9)
                    self.save_plot(
                        fhkey=fhkey,
                        outdir=outdir,
                        end='posteriors'
                    )
                    plt.close()

    def make_overlaid_posterior_plots(self, combined=False):
        """Make overlaid posterior plots. Overlaid here means that
        a plot will be made with each of the h0 and h1 returned
        values on the same plot for each of the fiducial h0 and h1
        pseudos. With combined=False they will be saved each time but
        with combined=True they will be saved on a single canvas for
        each fiducial hypothesis."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True

        if combined:
            outdir = os.path.join(self.outdir, 'CombinedOverlaidPosteriors')
            maintitle = self.make_main_title(end='Posteriors')
        else:
            outdir = os.path.join(self.outdir, 'IndividualOverlaidPosteriors')
            maintitle = self.make_main_title(end='Posterior')
        mkdir(outdir)

        hypos = ['h0', 'h1']
        hcolors = ['limegreen', 'darkviolet']
        hlabels = ['Hypo %s'%self.tex_axis_label(self.labels.dict['h0_name']),
                   'Hypo %s'%self.tex_axis_label(self.labels.dict['h1_name'])]

        for injkey in self.values.keys():
            for fid in hypos:
                # Need just one the hypo/fid combinations to feed in
                # to things at this stage
                dummy_fhkey = 'h0_fit_to_%s_fid'%fid
                # Set up multi-plot if needed
                if combined:
                    num_rows = self.get_num_rows(
                        data=self.values[injkey][dummy_fhkey],
                        omit_metric=False
                    )
                    plt.figure(figsize=(20, 5*num_rows+2))
                    subplotnum = 1
                else:
                    subplotnum = None
                # Loop through the systematics
                for systkey in self.values[injkey][dummy_fhkey].keys():
                    fittitle = self.make_fit_title(
                        fid=fid,
                        hypo='both',
                        trials=self.num_trials
                    )
                    systunits = self.values[injkey][
                        dummy_fhkey][systkey]['units']
                    if systkey == 'metric_val':
                        xlabel = self.tex_axis_label(
                            self.values[injkey][dummy_fhkey][systkey]['type']
                        )
                    else:
                        xlabel = self.tex_axis_label(systkey)
                    if not systunits == 'dimensionless':
                        xlabel += r' (%s)'%self.tex_axis_label(systunits)
                    # Specify the subplot, if necessary
                    if combined:
                        plt.subplot(num_rows, 4, subplotnum)
                    # Get binning
                    datamin = None
                    datamax = None
                    for hypo in hypos:
                        fhkey = '%s_fit_to_%s_fid'%(hypo, fid)
                        data = np.array(
                            self.values[injkey][fhkey][systkey]['vals']
                        )
                        if datamin is None:
                            datamin = data.min()
                        else:
                            datamin = min(datamin, data.min())
                        if datamax is None:
                            datamax = data.max()
                        else:
                            datamax = max(datamax, data.max())
                    datarange = datamax - datamin
                    databins = np.linspace(datamin - 0.1*datarange,
                                           datamax + 0.1*datarange,
                                           21)
                    for hypo, hcolor, hlabel in zip(hypos, hcolors, hlabels):
                        fhkey = '%s_fit_to_%s_fid'%(hypo, fid)
                        self.make_1d_hist_plot(
                            data=np.array(
                                self.values[injkey][fhkey][systkey]['vals']
                            ),
                            bins=databins,
                            xlabel=xlabel,
                            title=maintitle+r'\\'+fittitle,
                            ylabel='Number of Trials',
                            subplotnum=subplotnum,
                            alpha=0.5,
                            color=hcolor,
                            label=hlabel,
                            histtype='step',
                            lw=2
                        )
                    plt.ylim(0, 1.35*plt.ylim()[1])
                    plt.legend(
                        loc='upper left',
                        fontsize=12,
                        framealpha=1.0
                    )
                    plt.subplots_adjust(
                        left=0.10,
                        right=0.90,
                        top=0.85,
                        bottom=0.11
                    )
                    # Advance the subplot number, if necessary
                    if combined:
                        subplotnum += 1
                    # Else, save/close this plot
                    else:
                        self.save_plot(
                            fid=fid,
                            hypo='both',
                            outdir=outdir,
                            end='%s_posterior'%systkey
                        )
                        plt.close()
                # Save the whole canvas, if necessary
                if combined:
                    plt.suptitle(maintitle+r'\\'+fittitle, fontsize=36)
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.9)
                    self.save_plot(
                        fid=fid,
                        hypo='both',
                        outdir=outdir,
                        end='posteriors'
                    )
                    plt.close()

    def make_prior_label(self, kind, stddev=None, maximum=None,
                         smalllabel=False):
        """Makes a label for showing priors on plots"""
        if kind == 'gaussian':
            if (stddev is None) or (maximum is None):
                raise ValueError(
                    "A gaussian prior must be defined with "
                    "both a maximum and a standard deviation."
                )
            if (np.abs(stddev) < 1e-2) and (stddev != 0.0):
                if smalllabel:
                    priorlabel = (r'GP ' + \
                                  r'($%.3e\pm%.3e$)'%(maximum, stddev))
                else:
                    priorlabel = (r'Gaussian Prior ' + \
                                  r'($%.3e\pm%.3e$)'%(maximum, stddev))
            else:
                if smalllabel:
                    priorlabel = (r'GP ' + \
                                  r'($%.3g\pm%.3g$)'%(maximum, stddev))
                else:
                    priorlabel = (r'Gaussian Prior ' + \
                                  r'($%.3g\pm%.3g$)'%(maximum, stddev))
        else:
            raise ValueError(
                "Only gaussian priors are currently implemented. Got %s."%kind
            )
        return priorlabel

    def add_prior_region(self, systkey, injkey=None, fhkey=None,
                         smalllabel=False):
        """Add a shaded region to show the 1 sigma band of the prior"""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        # TODO - Deal with non-gaussian priors
        # Adding priors to 1D scan plots
        if (injkey is None) and (fhkey is None):
            if self.data[systkey]['prior']['kind'] == 'gaussian':
                stddev = self.data[systkey]['prior']['stddev'][0]
                maximum = self.data[systkey]['prior']['max_at'][0]
                currentylim = plt.ylim()
                priorlabel = self.make_prior_label(
                    kind='gaussian',
                    stddev=stddev,
                    maximum=maximum,
                    smalllabel=smalllabel
                )
                plt.axhspan(
                    maximum-stddev,
                    maximum+stddev,
                    color='k',
                    label=priorlabel,
                    alpha=0.2,
                    zorder=5
                )
                # Reset ylimits if prior makes it go far off
                if plt.ylim()[0] < currentylim[0]:
                    plt.ylim(currentylim[0], plt.ylim()[1])
                if plt.ylim()[1] > currentylim[1]:
                    plt.ylim(plt.ylim()[0], currentylim[1])
        # Adding priors to posterior plots in hypo_testing
        else:
            if (injkey is None) or (fhkey is None):
                raise ValueError(
                    "injkey and fhkey must either be both "
                    "None or both specified."
                )
            hypo = self.get_hypo_from_fiducial_hypo_key(fhkey=fhkey)
            wanted_params = self.all_params['%s_params'%hypo]
            for param in wanted_params.keys():
                if param == systkey:
                    if 'gaussian' in wanted_params[param]['prior']:
                        stddev, maximum = self.extract_gaussian(
                            prior_string=wanted_params[param]['prior'],
                            units=self.values[injkey][fhkey][systkey]['units']
                        )
                        currentxlim = plt.xlim()
                        priorlabel = self.make_prior_label(
                            kind='gaussian',
                            stddev=stddev,
                            maximum=maximum,
                            smalllabel=smalllabel
                        )
                        plt.axvspan(
                            maximum-stddev,
                            maximum+stddev,
                            color='k',
                            label=priorlabel,
                            ymax=0.1,
                            alpha=0.5,
                            zorder=5
                        )
                        # Reset xlimits if prior makes it go far off
                        if plt.xlim()[0] < currentxlim[0]:
                            plt.xlim(currentxlim[0], plt.xlim()[1])
                        if plt.xlim()[1] > currentxlim[1]:
                            plt.xlim(plt.xlim()[0], currentxlim[1])

    def add_inj_fid_lines(self, injkey, systkey, fhkey, smalllabel=False):
        """Add lines to show the injected and fiducial fit lines
        where appropriate"""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        h0_params, h1_params = self.get_resulting_hypo_params(
            injkey=injkey
        )
        data_params = self.get_injected_params()
        # Add injected and hypothesis fit lines
        if data_params is not None:
            if systkey in data_params.keys():
                if smalllabel:
                    injval, injlabelproper = self.extract_paramval(
                        injparams=data_params,
                        systkey=systkey,
                        fhkey=fhkey,
                        paramlabel='IV',
                        smalllabel=True
                    )
                else:
                    injval, injlabelproper = self.extract_paramval(
                        injparams=data_params,
                        systkey=systkey,
                        fhkey=fhkey,
                        paramlabel='Injected Value'
                    )
                plt.axvline(
                    injval,
                    color='r',
                    linewidth=2,
                    label=injlabelproper,
                    zorder=5
                )
            else:
                injval = None
        else:
            injval = None
        if self.get_fid_from_fiducial_hypo_key(fhkey=fhkey) == 'h0':
            if smalllabel:
                fitval, fitlabelproper = self.extract_paramval(
                    injparams=h0_params,
                    systkey=systkey,
                    fhkey=fhkey,
                    paramlabel='%s FF'%self.tex_axis_label(
                        self.labels.dict['h0_name'],
                        smalllabel=True
                    ),
                    smalllabel=True
                )
            else:
                fitval, fitlabelproper = self.extract_paramval(
                    injparams=h0_params,
                    systkey=systkey,
                    fhkey=fhkey,
                    paramlabel='%s Fiducial Fit'%self.tex_axis_label(
                        self.labels.dict['h0_name']
                    )
                )
        elif self.get_fid_from_fiducial_hypo_key(fhkey=fhkey) == 'h1':
            if smalllabel:
                fitval, fitlabelproper = self.extract_paramval(
                    injparams=h1_params,
                    systkey=systkey,
                    fhkey=fhkey,
                    paramlabel='%s FF'%self.tex_axis_label(
                        self.labels.dict['h1_name'],
                        smalllabel=True
                    ),
                    smalllabel=True
                )
            else:
                fitval, fitlabelproper = self.extract_paramval(
                    injparams=h1_params,
                    systkey=systkey,
                    fhkey=fhkey,
                    paramlabel='%s Fiducial Fit'%self.tex_axis_label(
                        self.labels.dict['h1_name']
                    )
                )
        else:
            raise ValueError("I got a hypothesis %s. Expected h0 or h1 only."
                             %self.get_fid_from_fiducial_hypo_key(fhkey=fhkey))
        if injval is not None:
            if fitval != injval:
                plt.axvline(
                    fitval,
                    color='g',
                    linewidth=2,
                    label=fitlabelproper,
                    zorder=5
                )
        else:
            plt.axvline(
                fitval,
                color='g',
                linewidth=2,
                label=fitlabelproper,
                zorder=5
            )

    def make_fit_information_plots(self):
        """Make plots of the number of iterations and time taken with the
        minimiser. This is a good cross-check of pseudo-experiments that
        the minimiser did not end abruptly since you would see significant
        pile-up if it did."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True

        outdir = os.path.join(self.outdir, 'MinimiserPlots')
        mkdir(outdir)
        maintitle = self.make_main_title(end='Minimiser Information')
        if self.test_type == 'analysis':
            for injkey in self.minimiser_info.keys():
                for fhkey in self.minimiser_info[injkey].keys():
                    if self.minimiser_info[injkey][fhkey] is not None:
                        minimiser_times = []
                        minimiser_iterations = []
                        minimiser_funcevals = []
                        minimiser_status = []
                        for trial in self.minimiser_info[injkey][fhkey].keys():
                            bits = self.minimiser_info[injkey][fhkey][
                                trial]['minimizer_time'].split(' ')
                            minimiser_times.append(
                                float(bits[0])
                            )
                            minimiser_iterations.append(
                                int(self.minimiser_info[injkey][fhkey][trial][
                                    'minimizer_metadata']['nit'])
                            )
                            minimiser_funcevals.append(
                                int(self.minimiser_info[injkey][fhkey][trial][
                                    'minimizer_metadata']['nfev'])
                            )
                            minimiser_status.append(
                                int(self.minimiser_info[injkey][fhkey][trial][
                                    'minimizer_metadata']['status'])
                            )
                        fittitle = self.make_fit_title(
                            fhkey=fhkey,
                            trials=self.num_trials
                        )
                        data_to_plot = [
                            minimiser_times,
                            minimiser_iterations,
                            minimiser_funcevals,
                            minimiser_status
                        ]
                        data_to_plot_ends = [
                            'minimiser_times',
                            'minimiser_iterations',
                            'minimiser_funcevals',
                            'minimiser_status'
                        ]
                        for plot_data, plot_end in zip(data_to_plot,
                                                       data_to_plot_ends):
                            self.make_1d_hist_plot(
                                data=plot_data,
                                xlabel=self.tex_axis_label(plot_end),
                                title=maintitle+r'\\'+fittitle,
                                ylabel='Number of Trials'
                            )
                            self.save_plot(
                                fhkey=fhkey,
                                outdir=outdir,
                                end=plot_end
                            )
                            plt.close()
        elif self.test_type == 'injparamscan':
            trueoutdir = os.path.join(outdir, 'TrueToWrongFits')
            mkdir(trueoutdir)
            wrongoutdir = os.path.join(outdir, 'WrongToTrueFits')
            mkdir(wrongoutdir)
            for odir, fits in zip([trueoutdir, wrongoutdir],
                                  [self.th_to_wh, self.th_to_wh]):
                # Times have a unit so must be handled differently
                minimiser_times = []
                for time in fits[0]['minim_info']['time']:
                    val, units = self.parse_pint_string(
                        pint_string=time
                    )
                    minimiser_times.append(val)
                # Otherwise they're easy
                minimiser_iterations = fits[0]['minim_info']['iterations']
                minimiser_funcevals = fits[0]['minim_info']['funcevals']
                minimiser_status = fits[0]['minim_info']['status']

                data_to_plot = [
                    minimiser_times,
                    minimiser_iterations,
                    minimiser_funcevals,
                    minimiser_status
                ]
                data_to_plot_ends = [
                    'minimiser_times',
                    'minimiser_iterations',
                    'minimiser_funcevals',
                    'minimiser_status'
                ]

                truth = self.labels[
                    self.labels.keys()[0]].dict['data_name'].split('_')[0]
                plotlabel = 'True %s'%self.tex_axis_label(truth)
                hrange = self.inj_param_vals[-1]-self.inj_param_vals[0]
                xlims = [self.inj_param_vals[0]-0.1*hrange,
                         self.inj_param_vals[-1]+0.1*hrange]

                for plot_data, plot_end in zip(data_to_plot,
                                               data_to_plot_ends):

                
                    vrange = float(max(plot_data)) - float(min(plot_data))
                    ylims = [float(min(plot_data))-0.1*vrange,
                             float(max(plot_data))+0.2*vrange]

                    self.make_1d_graph(
                        xvals=self.inj_param_vals,
                        yvals=plot_data,
                        xlabel=self.inj_param_name,
                        xunits=self.inj_param_units,
                        ylabel=plot_end,
                        yunits=None,
                        marker=self.marker_style(truth),
                        color=self.plot_colour(truth),
                        plotlabel=plotlabel,
                        xlims=xlims,
                        ylims=ylims
                    )

                    plt.title(maintitle, fontsize=16)
                    plt.legend(loc='best')
                    plt.tight_layout()
                    if 'TrueToWrongFits' in odir:
                        save_end = "true_to_wrong_fits_" + plot_end
                    else:
                        save_end = "wrong_to_true_fits_" + plot_end
                    self.save_plot(outdir=odir, end=save_end, truth=truth)
                    plt.close()

    def add_extra_points(self, ymax=None):
        """Add extra points specified by self.extra_points and label them
        with self.extra_points_labels`"""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True

        if self.test_type == 'analysis':
            if ymax is None:
                raise ValueError(
                    'A maximum y value must be provided to add extra '
                    'points to the hypo_testing analysis plots.'
                )
            linelist = []
            for point, label in zip(self.extra_points,
                                    self.extra_points_labels):
                if isinstance(point, str):
                    if os.path.isfile(point):
                        point = np.genfromtxt(point)
                    try:
                        point = eval(point)
                    except:
                        raise ValueError(
                            'Provided point, %s, was not either a '
                            'path to a file or a string which could '
                            'be parsed by eval()' % point
                        )
                if not isinstance(point, float):
                    raise ValueError(
                        'Expecting a single point here to add to the'
                        ' plot and got %s instead.' % point
                    )
                plt.axvline(
                    point,
                    color=self.plot_colour(label),
                    linestyle=self.plot_style(label),
                    ymax=ymax,
                    lw=2,
                    label=self.tex_axis_label(label)
                )
                linelist.append(self.tex_axis_label(label))
            return linelist
        elif self.test_type == 'injparamscan':
            ymin = None
            ymax = None
            for pointset, label in zip(self.extra_points,
                                    self.extra_points_labels):
                if os.path.isfile(pointset):
                    pointset = np.genfromtxt(pointset)
                else:
                    try:
                        pointset = eval(pointset)
                    except:
                        raise ValueError(
                            "Provided pointset, %s, was not either a "
                            "path to a file or a string which could "
                            "be parsed by eval()"%pointset
                        )
                x = []
                y = []
                yerr = []
                for point in pointset:
                    x.append(point[0])
                    y.append(point[1])
                    # For no errors
                    if len(point) == 2:
                        yerr.append(0.0)
                    # For symmetric errors
                    elif len(point) == 3:
                        yerr.append(point[2])
                    # For asymmetric errors
                    elif len(point) == 4:
                        if len(yerr) == 0:
                            yerr.append([])
                            yerr.append([])
                        yerr[0].append(point[2])
                        yerr[1].append(point[3])
                    else:
                        raise ValueError(
                            "Number of entries found for each point was "
                            "not what was expected. Should be at least "
                            "(x,y) but may also be (x,y,yerr) or "
                            "(x,y,yuperr,ydownerr). Got a set with %i "
                            "numbers."%len(point)
                        )
                x = np.array(x)
                y = np.array(y)
                yerr = np.array(yerr)
                if ymin is not None:
                    if len(yerr) == 2:
                        ymin = min(ymin, min(y-yerr[0]))
                        ymax = max(ymax, max(y+yerr[1]))
                    else:
                        ymin = min(ymin, min(y-yerr))
                        ymax = max(ymax, max(y+yerr))
                else:
                    if len(yerr) == 2:
                        ymin = min(y-yerr[0])
                        ymax = max(y+yerr[1])
                    else:
                        ymin = min(y-yerr)
                        ymax = max(y+yerr)
                plt.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    linestyle='None',
                    marker=self.marker_style(label),
                    markersize=10,
                    color=self.plot_colour(label),
                    label='%s'%(self.tex_axis_label(label))
                )
            return ymin, ymax

    def calc_p_value(self, llrdist, critical_value, greater=True,
                     median_p_value=False, llrbest=None):
        """Calculate the p-value for the given dataset based on the given
        critical value with an associated error.

        The calculation involves asking in how many trials the test statistic
        was "beyond" the critical value. The threshold of beyond will depend
        on whether the given distribution is the best fit or the alternate fit.
        The default is a "greater than" criterion, which can be switched by
        setting the "greater" argument to false.

        In the case of median_p_values the error calculation must also account
        for the uncertainty on the median, and so one must pass the
        distribution from which this was calculated so the error can be
        estimated with bootstrapping."""
        if greater:
            misid_trials = float(np.sum(llrdist > critical_value))
        else:
            misid_trials = float(np.sum(llrdist < critical_value))
        p_value = misid_trials/self.num_trials
        if median_p_value:
            # Quantify the uncertainty on the median by bootstrapping
            sampled_medians = []
            for _ in range(1000):
                sampled_medians.append(
                    np.median(
                        np.random.choice(
                            llrbest,
                            size=len(llrbest),
                            replace=True
                        )
                    )
                )
            sampled_medians = np.array(sampled_medians)
            median_error = np.std(sampled_medians)/np.sqrt(self.num_trials)
            # Add relative errors in quadrature
            wdenom = misid_trials+median_error*median_error
            wterm = wdenom/(misid_trials*misid_trials)
            nterm = 1.0/self.num_trials
            unc_p_value = p_value * np.sqrt(wterm + nterm)
            return p_value, unc_p_value, median_error
        else:
            unc_p_value = np.sqrt(misid_trials*(1-p_value))/self.num_trials
            return p_value, unc_p_value

    def plot_llr_histograms(self, llrarrays, llrhistmax, binning, colors,
                            labels, best_name, alt_name, critical_value,
                            critical_label, critical_height, llrhist,
                            critical_color='k', plot_scaling_factor=1.55,
                            greater=True, cls=False):
        """Plot the llr histograms. The `greater` argument is intended to be
        used the same as in the p value function above."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True

        for llrarray, label, color in zip(llrarrays, labels, colors):
            plt.hist(
                llrarray,
                bins=binning,
                color=color,
                histtype='step',
                lw=2,
                label=label
            )
        plt.xlabel(r'Log-Likelihood Ratio', size='18', labelpad=18)
        plt.ylabel(r'Number of Trials (per %.2f)'%(binning[1]-binning[0]),
                   size='18')
        # Nicely scale the plot
        plt.ylim(0, plot_scaling_factor*llrhistmax)
        # Add labels to show which side means what...
        xlim = plt.gca().get_xlim()
        plt.text(
            xlim[0]-0.05*(xlim[1]-xlim[0]),
            -0.09*plot_scaling_factor*llrhistmax,
            r'\begin{flushleft} $\leftarrow$ Prefers %s\end{flushleft}'%(
                self.tex_axis_label(alt_name)),
            color='k',
            size='large'
        )
        plt.text(
            xlim[1]+0.05*(xlim[1]-xlim[0]),
            -0.09*plot_scaling_factor*llrhistmax,
            r'\begin{flushright} Prefers %s $\rightarrow$ \end{flushright}'%(
                self.tex_axis_label(best_name)),
            color='k',
            size='large',
            horizontalalignment='right'
        )
        # Add the critical value with the desired height and colour.
        if critical_value is not None:
            plt.axvline(
                critical_value,
                color=critical_color,
                ymax=critical_height,
                lw=2,
                label=critical_label
            )
            if llrhist is not None:
                if cls:
                    for hist, color in zip(llrhist, colors):
                        finehist = np.repeat(hist, 100)
                        finebinning = np.linspace(binning[0],
                                                  binning[-1],
                                                  (len(binning)-1)*100+1)
                        finebinwidth = finebinning[1]-finebinning[0]
                        finebincens = np.linspace(
                            finebinning[0]+finebinwidth/2.0,
                            finebinning[-1]-finebinwidth/2.0,
                            len(finebinning)-1
                        )
                        where = (finebincens > critical_value)
                        plt.fill_between(
                            finebincens,
                            0,
                            finehist,
                            where=where,
                            color=color,
                            hatch='x',
                            edgecolor='k',
                            lw=0,
                            alpha=0.3
                        )
                else:
                    # Create an object so that a hatch can be drawn over the
                    # region of interest to the p-value.
                    finehist = np.repeat(llrhist, 100)
                    finebinning = np.linspace(binning[0], binning[-1],
                                              (len(binning)-1)*100+1)
                    finebinwidth = finebinning[1]-finebinning[0]
                    finebincens = np.linspace(finebinning[0]+finebinwidth/2.0,
                                              finebinning[-1]-finebinwidth/2.0,
                                              len(finebinning)-1)
                    # Draw the hatch. This is between the x-axis (0) and the
                    # finehist object made above. The "where" tells is to only
                    # draw above the critical value. To make it just the hatch,
                    # color is set to none and hatch is set to X. Also, so that
                    # it doesn't have a border we set linewidth to zero.
                    if greater:
                        where = (finebincens > critical_value)
                    else:
                        where = (finebincens < critical_value)
                    plt.fill_between(
                        finebincens,
                        0,
                        finehist,
                        where=where,
                        color='k',
                        hatch='X',
                        edgecolor='k',
                        lw=0,
                        alpha=0.3
                    )
        plt.subplots_adjust(left=0.10, right=0.90, top=0.9, bottom=0.15)

    def make_llr_plots(self):
        """Make llr plots.

        Takes the data and makes llr distributions. These are then saved to the
        requested outdir within a folder labelled "LLRDistributions". The
        extra_points and extra_points_labels arguments can be used to specify
        extra points to be added to the plot for e.g. other fit llr values.
        """
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True

        outdir = os.path.join(self.outdir, 'LLRDistributions')
        mkdir(outdir)

        for injkey in self.values.keys():

            data = self.values[injkey]
            metric_type = data['h0_fit_to_h0_fid']['metric_val']['type']
            metric_type_pretty = self.tex_axis_label(metric_type)
            h0_fid_metric = self.fid_values[injkey][
                'h0_fit_to_%s'%self.labels.dict['data']
            ][
                'metric_val'
            ]
            h1_fid_metric = self.fid_values[injkey][
                'h1_fit_to_%s'%self.labels.dict['data']
            ][
                'metric_val'
            ]

            h0_fit_to_h0_fid_metrics = np.array(
                data['h0_fit_to_h0_fid']['metric_val']['vals']
            )
            h1_fit_to_h0_fid_metrics = np.array(
                data['h1_fit_to_h0_fid']['metric_val']['vals']
            )
            h0_fit_to_h1_fid_metrics = np.array(
                data['h0_fit_to_h1_fid']['metric_val']['vals']
            )
            h1_fit_to_h1_fid_metrics = np.array(
                data['h1_fit_to_h1_fid']['metric_val']['vals']
            )

            # In the case of likelihood, the maximum metric is the better fit.
            # With chi2 metrics the opposite is true, and so we must multiply
            # everything by -1 in order to apply the same treatment.
            if 'chi2' in metric_type:
                logging.info('Converting chi2 metric to likelihood equivalent.')
                h0_fid_metric *= -1
                h1_fid_metric *= -1
                h0_fit_to_h0_fid_metrics *= -1
                h1_fit_to_h0_fid_metrics *= -1
                h0_fit_to_h1_fid_metrics *= -1
                h1_fit_to_h1_fid_metrics *= -1

            if h1_fid_metric < h0_fid_metric:
                bestfit = 'h0'
                altfit = 'h1'
                critical_value = h0_fid_metric-h1_fid_metric
            else:
                bestfit = 'h1'
                altfit = 'h0'
                critical_value = h1_fid_metric-h0_fid_metric

            if bestfit == 'h0':
                llrbest = h0_fit_to_h0_fid_metrics - h1_fit_to_h0_fid_metrics
                llralt = h0_fit_to_h1_fid_metrics - h1_fit_to_h1_fid_metrics
            else:
                llrbest = h1_fit_to_h1_fid_metrics - h0_fit_to_h1_fid_metrics
                llralt = h1_fit_to_h0_fid_metrics - h0_fit_to_h0_fid_metrics

            minllr = min(min(llrbest), min(llralt))
            maxllr = max(max(llrbest), max(llralt))
            rangellr = maxllr - minllr
            # Special case for low numbers of trials. Here, the plot
            # can't really be interpreted but the numbers printed on
            # it can still be useful, so we need to make something.
            if self.num_trials < 100:
                binning = np.linspace(minllr - 0.1*rangellr,
                                      maxllr + 0.1*rangellr,
                                      10)
            elif self.num_trials < 2000:
                binning = np.linspace(minllr - 0.1*rangellr,
                                      maxllr + 0.1*rangellr,
                                      int(self.num_trials/40))
            else:
                binning = np.linspace(minllr - 0.1*rangellr,
                                      maxllr + 0.1*rangellr,
                                      50)
            binwidth = binning[1]-binning[0]

            llrbesthist, llrbestbinedges = np.histogram(llrbest, bins=binning)
            llralthist, llraltbinedges = np.histogram(llralt, bins=binning)

            llrhistmax = max(max(llrbesthist), max(llralthist))

            best_median = np.median(llrbest)

            if self.labels.dict['data_name'] == '':
                inj_name = "data"
            else:
                inj_name = "true %s"%self.tex_axis_label(
                    self.labels.dict['data_name']
                )
            best_name = self.labels.dict['%s_name'%bestfit]
            alt_name = self.labels.dict['%s_name'%altfit]

            # Calculate p values
            ## First for the preferred hypothesis based on the fiducial fit
            crit_p_value, unc_crit_p_value = self.calc_p_value(
                llrdist=llralt,
                critical_value=critical_value
            )
            ## Then for the alternate hypothesis based on the fiducial fit
            alt_crit_p_value, alt_unc_crit_p_value = self.calc_p_value(
                llrdist=llrbest,
                critical_value=critical_value
            )
            ## Combine these to give a cls value based on arXiv:1407.5052
            cls_value = (1 - alt_crit_p_value) / (1 - crit_p_value)
            unc_cls_value = cls_value * np.sqrt(
                np.power(alt_unc_crit_p_value/alt_crit_p_value, 2.0) + \
                np.power(unc_crit_p_value/crit_p_value, 2.0)
            )
            ## Then for the preferred hypothesis based on the median. That
            ## is, the case of a median experiment from the distribution
            ## under the preferred hypothesis.
            med_p_value, unc_med_p_value, median_error = self.calc_p_value(
                llrdist=llralt,
                critical_value=best_median,
                median_p_value=True,
                llrbest=llrbest
            )

            if metric_type == 'llh':
                plot_title = (r"\begin{center}"\
                              +"%s %s Event Selection "%(self.detector,
                                                         self.selection)\
                              +r"\\"+" llr Distributions for %s (%i trials)"%(
                                  inj_name, self.num_trials)\
                              +r"\end{center}")

            else:
                plot_title = (r"\begin{center}"\
                              +"%s %s Event Selection "%(self.detector,
                                                         self.selection)\
                              +r"\\"+" %s \"llr\" Distributions for "
                              %(metric_type_pretty)\
                              +"%s (%i trials)"%(inj_name,
                                                 self.num_trials)\
                              +r"\end{center}")

            # Factor with which to make everything visible
            plot_scaling_factor = 1.55

            # In case of median plot, draw both best and alt histograms
            ## Set up the labels for the histograms
            llr_labels = [
                r"%s Pseudo-Experiments - "%(self.tex_axis_label(best_name)) + \
                r"$\log\left[\mathcal{L}\left(\mathcal{H}_{%s}"%(best_name) + \
                r"\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                    alt_name),
                r"%s Pseudo-Experiments - "%(self.tex_axis_label(alt_name)) + \
                r"$\log\left[\mathcal{L}\left(\mathcal{H}_{%s}"%(best_name) + \
                r"\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                    alt_name)
            ]
            self.plot_llr_histograms(
                llrarrays=[llrbest, llralt],
                llrhistmax=llrhistmax,
                binning=binning,
                colors=['r', 'b'],
                labels=llr_labels,
                best_name=best_name,
                alt_name=alt_name,
                critical_value=best_median,
                critical_label=r"%s Median = $%.4f\pm%.4f$"%(
                    self.tex_axis_label(best_name),
                    best_median,
                    median_error),
                critical_height=float(max(llrbesthist))/float(
                    plot_scaling_factor*llrhistmax),
                llrhist=llralthist
            )
            plt.legend(loc='upper left')
            plt.title(plot_title)
            # Write the p-value on the plot
            plt.figtext(
                0.15,
                0.66,
                r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.4f\pm%.4f$"%(
                    best_name, med_p_value, unc_med_p_value),
                color='k',
                size='xx-large'
            )
            self.save_plot(
                outdir=outdir,
                end='%s_llrDistribution_median_%i_Trials'%(
                    metric_type, self.num_trials)
            )
            # Add the extra points if they exist
            if self.extra_points is not None:
                plt.legend(loc='upper left', fontsize=11)
                curleg = plt.gca().get_legend()
                linelist = self.add_extra_points(
                    ymax=float(max(llrbesthist))/float(
                        plot_scaling_factor*llrhistmax)
                )
                handles, labels = plt.gca().get_legend_handles_labels()
                newhandles = []
                for l, h in zip(labels, handles):
                    if l in linelist:
                        newhandles.append(h)
                newleg = plt.legend(
                    handles=newhandles,
                    loc='upper right',
                    fontsize=11
                )
                plt.gca().add_artist(newleg)
                plt.gca().add_artist(curleg)
                self.save_plot(
                    outdir=outdir,
                    end='%s_llrDistribution_median_w_extra_points_%i_Trials'%(
                        metric_type, self.num_trials)
                )
            plt.close()

            # Make some debugging plots
            ## Set up the labels for the histograms
            llr_labels = [
                r"%s Pseudo-Experiments - "%(self.tex_axis_label(best_name)) + \
                r"$\log\left[\mathcal{L}\left(\mathcal{H}_{%s}"%(best_name) + \
                r"\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                    alt_name),
                r"%s Pseudo-Experiments - "%(self.tex_axis_label(alt_name)) + \
                r"$\log\left[\mathcal{L}\left(\mathcal{H}_{%s}"%(best_name) + \
                r"\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                    alt_name)
            ]
            self.plot_llr_histograms(
                llrarrays=[llrbest, llralt],
                llrhistmax=llrhistmax,
                binning=binning,
                colors=['r', 'b'],
                labels=llr_labels,
                best_name=best_name,
                alt_name=alt_name,
                critical_value=best_median,
                critical_label=r"%s Median = $%.4f\pm%.4f$"%(
                    self.tex_axis_label(best_name),
                    best_median,
                    median_error),
                critical_height=float(max(llrbesthist))/float(
                    plot_scaling_factor*llrhistmax),
                llrhist=None
            )
            plt.legend(loc='upper left')
            plt.title(plot_title)
            self.save_plot(
                outdir=outdir,
                end='%s_llrDistribution_median_both_fit_dists_%i_Trials'%(
                    metric_type, self.num_trials)
            )
            plt.close()
            ## Set up the labels for the histograms
            llr_labels = [
                r"%s Pseudo-Experiments - "%(self.tex_axis_label(best_name)) + \
                r"$\log\left[\mathcal{L}\left(\mathcal{H}_{%s}" + \
                r"\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                    alt_name),
            ]
            self.plot_llr_histograms(
                llrarrays=[llrbest],
                llrhistmax=llrhistmax,
                binning=binning,
                colors=['r'],
                labels=llr_labels,
                best_name=best_name,
                alt_name=alt_name,
                critical_value=None,
                critical_label=None,
                critical_height=None,
                llrhist=None
            )
            plt.legend(loc='upper left')
            plt.title(plot_title)
            self.save_plot(
                outdir=outdir,
                end='%s_llrDistribution_best_fit_dist_%i_Trials'%(
                    metric_type, self.num_trials)
            )
            plt.close()
            ## Set up the labels for the histograms
            llr_labels = [
                r"%s Pseudo-Experiments - "%(self.tex_axis_label(best_name)) + \
                r"$\log\left[\mathcal{L}\left(\mathcal{H}_{%s}"%(best_name) + \
                r"\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                    alt_name),
            ]
            self.plot_llr_histograms(
                llrarrays=[llrbest],
                llrhistmax=llrhistmax,
                binning=binning,
                colors=['r'],
                labels=llr_labels,
                best_name=best_name,
                alt_name=alt_name,
                critical_value=best_median,
                critical_label=r"%s Median = $%.4f\pm%.4f$"%(
                    self.tex_axis_label(best_name),
                    best_median,
                    median_error),
                critical_height=float(max(llrbesthist))/float(
                    plot_scaling_factor*llrhistmax),
                llrhist=None
            )
            plt.legend(loc='upper left')
            plt.title(plot_title)
            self.save_plot(
                outdir=outdir,
                end='%s_llrDistribution_median_best_fit_dist_%i_Trials'%(
                    metric_type, self.num_trials)
            )
            plt.close()
            ## Set up the labels for the histograms
            llr_labels = [
                r"%s Pseudo-Experiments - "%(self.tex_axis_label(alt_name)) + \
                r"$\log\left[\mathcal{L}\left(\mathcal{H}_{%s}"%(best_name) + \
                r"\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                    alt_name)
            ]
            self.plot_llr_histograms(
                llrarrays=[llralt],
                llrhistmax=llrhistmax,
                binning=binning,
                colors=['b'],
                labels=llr_labels,
                best_name=best_name,
                alt_name=alt_name,
                critical_value=None,
                critical_label=None,
                critical_height=None,
                llrhist=None
            )
            plt.legend(loc='upper left')
            plt.title(plot_title)
            self.save_plot(
                outdir=outdir,
                end='%s_llrDistribution_alt_fit_dist_%i_Trials'%(
                    metric_type, self.num_trials)
            )
            plt.close()
            ## Set up the labels for the histograms
            llr_labels = [
                r"%s Pseudo-Experiments - "%(self.tex_axis_label(alt_name)) + \
                r"$\log\left[\mathcal{L}\left(\mathcal{H}_{%s}" + \
                r"\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                    alt_name)
            ]
            self.plot_llr_histograms(
                llrarrays=[llralt],
                llrhistmax=llrhistmax,
                binning=binning,
                colors=['b'],
                labels=llr_labels,
                best_name=best_name,
                alt_name=alt_name,
                critical_value=best_median,
                critical_label=r"%s Median = $%.4f\pm%.4f$"%(
                    self.tex_axis_label(best_name),
                    best_median,
                    median_error),
                critical_height=float(max(llrbesthist))/float(
                    plot_scaling_factor*llrhistmax),
                llrhist=llralthist
            )
            plt.legend(loc='upper left')
            plt.title(plot_title)
            # Write the p-value on the plot
            plt.figtext(
                0.15,
                0.66,
                r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.4f\pm%.4f$"%(
                    best_name, med_p_value, unc_med_p_value),
                color='k',
                size='xx-large'
            )
            self.save_plot(
                outdir=outdir,
                end='%s_llrDistribution_median_alt_fit_dist_%i_Trials'%(
                    metric_type, self.num_trials)
            )
            plt.close()

            # In case of critical plot, draw just alt histograms
            ## Set up the label for the histogram
            llr_labels = [
                r"%s Pseudo-Experiments - "%(self.tex_axis_label(alt_name)) + \
                r"$\log\left[\mathcal{L}\left(\mathcal{H}_{%s}"%(best_name) + \
                r"\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                    alt_name)
            ]
            self.plot_llr_histograms(
                llrarrays=[llralt],
                llrhistmax=llrhistmax,
                binning=binning,
                colors=['b'],
                labels=llr_labels,
                best_name=best_name,
                alt_name=alt_name,
                critical_value=critical_value,
                critical_label=r"Critical Value = %.4f"%(critical_value),
                critical_height=float(max(llrbesthist))/float(
                    plot_scaling_factor*llrhistmax),
                llrhist=llralthist
            )
            plt.legend(loc='upper left')
            plt.title(plot_title)
            # Write the p-value on the plot
            plt.figtext(
                0.15,
                0.70,
                r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.4f\pm%.4f$"%(
                    best_name, crit_p_value, unc_crit_p_value),
                color='k',
                size='xx-large'
            )
            self.save_plot(
                outdir=outdir,
                end='%s_llrDistribution_critical_%i_Trials'%(
                    metric_type, self.num_trials)
            )
            plt.close()

            # Make a second critical plot for the alt hypothesis, so we draw the
            # preferred hypothesis
            ## Set up the label for the histogram
            llr_labels = [
                r"%s Pseudo-Experiments - "%(self.tex_axis_label(best_name)) + \
                r"$\log\left[\mathcal{L}\left(\mathcal{H}_{%s}"%(best_name) + \
                r"\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                    alt_name)
            ]
            self.plot_llr_histograms(
                llrarrays=[llrbest],
                llrhistmax=llrhistmax,
                binning=binning,
                colors=['r'],
                labels=llr_labels,
                best_name=best_name,
                alt_name=alt_name,
                critical_value=critical_value,
                critical_label=r"Critical Value = %.4f"%(critical_value),
                critical_height=float(max(llrbesthist))/float(
                    plot_scaling_factor*llrhistmax),
                llrhist=llrbesthist
            )
            plt.legend(loc='upper left')
            plt.title(plot_title)
            # Write the p-value on the plot
            plt.figtext(
                0.15,
                0.70,
                r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.4f\pm%.4f$"%(
                    alt_name, alt_crit_p_value, alt_unc_crit_p_value),
                color='k',
                size='xx-large'
            )
            self.save_plot(
                outdir=outdir,
                end='%s_llrDistribution_critical_alt_%i_Trials'%(
                    metric_type, self.num_trials)
            )
            plt.close()

            # Lastly, show both exclusion regions and then the joined cls value
            ## Set up the labels for the histograms
            llr_labels = [
                r"%s Pseudo-Experiments - "%(self.tex_axis_label(best_name)) + \
                r"$\log\left[\mathcal{L}\left(\mathcal{H}_{%s}"%(best_name) + \
                r"\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                    alt_name),
                r"%s Pseudo-Experiments - "%(self.tex_axis_label(alt_name)) + \
                r"$\log\left[\mathcal{L}\left(\mathcal{H}_{%s}"%(best_name) + \
                r"\right)/\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"%(
                    alt_name)
            ]
            self.plot_llr_histograms(
                llrarrays=[llrbest, llralt],
                llrhistmax=llrhistmax,
                binning=binning,
                colors=['r', 'b'],
                labels=llr_labels,
                best_name=best_name,
                alt_name=alt_name,
                critical_value=critical_value,
                critical_label=r"Critical Value = %.4f"%(critical_value),
                critical_height=float(max(llrbesthist))/float(
                    plot_scaling_factor*llrhistmax),
                llrhist=[llrbesthist, llralthist],
                cls=True,
            )
            plt.legend(loc='upper left')
            plt.title(plot_title)
            # Write the p-values on the plot
            plt.figtext(
                0.50,
                0.66,
                r"$\mathrm{CL}_{s}\left(\mathcal{H}_{%s}"%(best_name) + \
                r"\right)= %.4f\pm%.4f$"%(cls_value, unc_cls_value),
                horizontalalignment='center',
                color='k',
                size='xx-large'
            )
            plt.figtext(
                0.12,
                0.55,
                r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.2f\pm%.2f$"%(
                    alt_name, alt_crit_p_value, alt_unc_crit_p_value),
                bbox=dict(facecolor='none', edgecolor='red', boxstyle='round'),
                horizontalalignment='left',
                color='k',
                size='x-large'
            )
            plt.figtext(
                0.88,
                0.55,
                r"$\mathrm{p}\left(\mathcal{H}_{%s}\right) = %.2f\pm%.2f$"%(
                    best_name, crit_p_value, unc_crit_p_value),
                horizontalalignment='right',
                bbox=dict(facecolor='none', edgecolor='blue', boxstyle='round'),
                color='k',
                size='x-large'
            )
            self.save_plot(
                outdir=outdir,
                end='%s_llrDistribution_CLs_%i_Trials'%(
                    metric_type, self.num_trials)
            )
            plt.close()

    def make_fiducial_fit_files(self):
        """Make tex files which can be then be compiled in to tables
        showing the two fiducial fits and, if applicable, how they
        compare to what was injected."""
        outdir = os.path.join(self.outdir, 'FiducialFits')
        mkdir(outdir)

        paramfilename = self.make_tex_name(end="fiducial_fits")
        paramfile = os.path.join(outdir, paramfilename)
        self.texfile = open(paramfile, 'w')
        self.write_latex_preamble()

        for injkey in self.fid_values.keys():
            for tabletype in ["fiducial_fit_params", "fiducial_fit_metrics"]:
                self.setup_latex_table(
                    tabletype=tabletype,
                    injected=('data_params' in self.all_params.keys())
                )
                self.do_latex_table_middle(
                    tabletype=tabletype,
                    injkey=injkey
                )
                self.texfile.write(r"\n")
                self.end_latex_table(tabletype=tabletype)

        self.texfile.write(r"\n")
        self.texfile.write(r"\end{document}\n")

    def write_latex_preamble(self):
        """Write latex preamble needed to make, in my often-wrong opinion,
        nice-looking tex files."""
        self.texfile.write(r"\n")
        self.texfile.write(r"\documentclass[a4paper,12pt]{article}\n")
        self.texfile.write(r"\usepackage{tabu}\n")
        self.texfile.write(r"\usepackage{booktabs}\n")
        self.texfile.write(r"\usepackage[font=small,labelsep=space]{caption}\n")
        self.texfile.write(r"\usepackage[margin=2.5cm]{geometry}\n")
        self.texfile.write(r"\setlength{\\topmargin}{1.0cm}\n")
        self.texfile.write(r"\setlength{\\textheight}{22cm}\n")
        self.texfile.write(r"\usepackage{fancyhdr}\n")
        self.texfile.write(r"\pagestyle{fancy}\n")
        self.texfile.write(r"\\fancyhf{}\n")
        self.texfile.write(r"\\fancyhead[R]{\leftmark}\n")
        self.texfile.write(r"\usepackage{multirow}\n")
        self.texfile.write(r"\n")
        self.texfile.write(r"\\begin{document}\n")
        self.texfile.write(r"\n")

    def setup_latex_table(self, tabletype, injected):
        """Set up the beginning of the table for the tex output files.
        Currently will make tables for the output fiducial fit params
        and the chi2 values only."""
        self.texfile.write(r"\\renewcommand{\\arraystretch}{1.6}\n")
        self.texfile.write(r"\n")
        self.texfile.write(r"\\begin{table}[t!]\n")
        self.texfile.write(r"  \\begin{center}\n")
        if tabletype == 'fiducial_fit_params':
            if injected:
                nextline = r"    \\begin{tabu} to 1.0\\textwidth "
                nextline += r"{| X[2.0,c] | X[1,c] | X[1,c] | X[1,c]"
                nextline += r" | X[1,c] | X[1,c] | X[1,c] | X[1,c] |}\n"
                self.texfile.write(nextline)
                self.texfile.write(r"    \hline\n")
                nextline = r"    \multirow{2}{*}{\\textbf{Parameter}} "
                nextline += r"& \multirow{2}{*}{\\textbf{Inj}} "
                nextline += r"& \multicolumn{3}{c|}{h0} "
                nextline += r"& \multicolumn{3}{c|}{h1} "
                nextline += r"\\\\ \cline{3-8}"
                self.texfile.write(nextline)
                nextline = r"    & & Prior & Fit & \(\Delta\) "
                nextline += r"& Prior & Fit & \(\Delta\) \\\\ \hline\n"
                self.texfile.write(nextline)
            else:
                nextline = r"    \\begin{tabu} to 1.0\\textwidth "
                nextline += r"{| X[c] | X[c] | X[c] |}\n"
                self.texfile.write(nextline)
                self.texfile.write(r"    \hline\n")
                self.texfile.write(r"    Parameter & h0 & h1 \\\\ \hline\n")
        elif tabletype == 'fiducial_fit_metrics':
            nextline = r"    \\begin{tabu} to 1.0\\textwidth "
            nextline += r"{| X[c] | X[c] | X[c] |}\n"
            self.texfile.write(nextline)
            self.texfile.write(r"    \hline\n")
            self.texfile.write(r"    h0 & h1 & $\Delta$ \\\\ \hline\n")
        else:
            raise ValueError(
                "This function is only for making fit metric or fit "
                "param tables in LaTeX. Got type %s"%tabletype
            )

    def do_latex_table_middle(self, tabletype, injkey):
        """Adds the actual content to the latex tables."""
        if tabletype == 'fiducial_fit_params':
            h0_params, h1_params = self.get_resulting_hypo_params(
                injkey=injkey
            )
            data_params = self.get_injected_params()
            for param in h0_params.keys():
                # Get the units for this parameter
                val, param_units = self.parse_pint_string(
                    pint_string=h0_params[param]
                )
                # Get priors if they exists
                if 'gaussian' in self.all_params['h0_params'][param]['prior']:
                    h0stddev, h0maximum = self.extract_gaussian(
                        prior_string=self.all_params['h0_params'][
                            param]['prior'],
                        units=param_units
                    )
                else:
                    h0stddev = None
                    h0maximum = None
                if 'gaussian' in self.all_params['h1_params'][param]['prior']:
                    h1stddev, h1maximum = self.extract_gaussian(
                        prior_string=self.all_params['h1_params'][
                            param]['prior'],
                        units=param_units
                    )
                else:
                    h1stddev = None
                    h1maximum = None
                # Include injected parameter, fitted parameters and
                # differences with appropriate formatting.
                if data_params is not None:
                    tableline = "      "
                    tableline += "%s "%self.tex_axis_label(param)
                    if param == 'deltam31':
                        tableline += r" / $10^{-3}$ "
                    if param_units != 'dimensionless':
                        tableline += "(%s) &"%self.tex_axis_label(param_units)
                    else:
                        tableline += "&"
                    if param in data_params.keys():
                        dataval = self.extract_paramval(
                            injparams=data_params,
                            systkey=param
                        )
                        if param == 'deltam31':
                            dataval *= 1000.0
                        if (np.abs(dataval) < 1e-2) and (dataval != 0.0):
                            tableline += "%.2e &"%dataval
                        else:
                            tableline += "%.3g &"%dataval
                    # If no injected parameter, show this and the
                    # deltas with a line
                    else:
                        dataval = '--'
                        tableline += "%s &"%dataval
                    h0val = self.extract_paramval(
                        injparams=h0_params,
                        systkey=param
                    )
                    if param == 'deltam31':
                        h0val *= 1000.0
                    tableline += self.format_table_line(
                        val=h0val,
                        dataval=dataval,
                        stddev=h0stddev,
                        maximum=h0maximum
                    )
                    h1val = self.extract_paramval(
                        injparams=h1_params,
                        systkey=param
                    )
                    if param == 'deltam31':
                        h1val *= 1000.0
                    tableline += self.format_table_line(
                        val=h1val,
                        dataval=dataval,
                        stddev=h1stddev,
                        maximum=h1maximum,
                        last=True
                    )
                    tableline += r" \\\\ \hline\n"
                    self.texfile.write(tableline)
                # If no injected parameters it's much simpler
                else:
                    h0val = self.extract_paramval(
                        injparams=h0_params,
                        systkey=param
                    )
                    h1val = self.extract_paramval(
                        injparams=h1_params,
                        systkey=param
                    )
                    if (np.abs(h0val) < 1e-2) and (h0val != 0.0):
                        self.texfile.write(r"    %s & %.2e & %.2e\n"%(
                            self.tex_axis_label(param), h0val, h1val))
                    else:
                        self.texfile.write(r"    %s & %.3g & %.3g\n"%(
                            self.tex_axis_label(param), h0val, h1val))
        elif tabletype == "fiducial_fit_metrics":
            h0_fid_metric = self.fid_values[injkey][
                'h0_fit_to_%s'%(self.labels.dict['data'])]['metric_val']
            h1_fid_metric = self.fid_values[injkey][
                'h1_fit_to_%s'%(self.labels.dict['data'])]['metric_val']

            # Need the type of metric here. Doesn't matter which
            # fit that comes from so just choose h0_fit_to_h0_fid
            # since it will always exist.
            metric_type = self.values[injkey][
                'h0_fit_to_h0_fid']['metric_val']['type']
            # In the case of likelihood, the maximum metric is the better fit.
            # With chi2 metrics the opposite is true, and so we must multiply
            # everything by -1 in order to apply the same treatment.
            if 'chi2' not in metric_type:
                logging.info(
                    "Converting likelihood metric to chi2 equivalent."
                )
                h0_fid_metric *= -1
                h1_fid_metric *= -1

            # If truth is known, report the fits the correct way round
            if self.labels.dict['data_name'] is not None:
                if self.labels.dict['data_name'] in \
                    self.labels.dict['h0_name']:
                    delta = h1_fid_metric-h0_fid_metric
                elif self.labels.dict['data_name'] in \
                    self.labels.dict['h1_name']:
                    delta = h0_fid_metric-h1_fid_metric
                else:
                    logging.warning(
                        "Truth is known but could not be identified in "
                        "either of the hypotheses. The difference between"
                        " the best fit metrics will just be reported as "
                        "positive and so will not necessarily reflect if "
                        "the truth was recovered."
                    )
                    if h1_fid_metric > h0_fid_metric:
                        delta = h0_fid_metric-h1_fid_metric
                    else:
                        delta = h1_fid_metric-h0_fid_metric
            # Else just report it as delta between best fits
            else:
                if h1_fid_metric > h0_fid_metric:
                    delta = h0_fid_metric-h1_fid_metric
                else:
                    delta = h1_fid_metric-h0_fid_metric
            # Write this in the file
            newline = "    %.3g "%h0_fid_metric
            newline += "& %.3g "%h1_fid_metric
            newline += "& %.3g "%delta
            newline += r"\\\\ \hline\n"
            self.texfile.write(newline)
        else:
            raise ValueError(
                "This function is only for adding the content to metric"
                " or fit param tables in LaTeX. Got type %s"%tabletype
            )

    def end_latex_table(self, tabletype):
        """End the table and the whole document for the tex output files."""
        self.texfile.write(r"    \end{tabu}\n")
        self.texfile.write(r"  \end{center}\n")
        self.texfile.write(r"  \\vspace{-10pt}\n")
        newline = r"  \caption{shows the fiducial fit "
        if tabletype == "fiducial_fit_params":
            newline += "parameters"
        elif tabletype == "fiducial_fit_metrics":
            newline += "metrics"
        else:
            raise ValueError(
                "This function is only for ending fit metric or fit "
                "param tables in LaTeX. Got type %s"%tabletype
            )
        if self.detector is not None:
            newline += " obtained with the %s"%self.detector
            if self.selection is not None:
                newline += " %s sample"%self.selection
        if self.selection is not None:
            newline += " obtained with the %s"%self.selection
        newline += " for h0 of %s"%self.tex_axis_label(
            self.labels.dict['h0_name']
        )
        newline += " and h1 of %s."%self.tex_axis_label(
            self.labels.dict['h1_name']
        )
        if self.labels.dict['data_name'] == '':
            newline += " The truth is %s."%self.tex_axis_label(
                self.labels.dict['data_name']
            )
        else:
            newline += " This is from an analysis performed on data."
        newline += "}\n"
        self.texfile.write(newline)
        newline = r"  \label{tab:"
        if self.detector is not None:
            newline += self.detector
        if self.selection is not None:
            newline += self.selection
        newline += "%stable}\n"%tabletype
        self.texfile.write(newline)
        self.texfile.write(r"\end{table}\n")

    def format_table_line(self, val, dataval, stddev=None,
                          maximum=None, last=False):
        """Formatting the numbers to look nice is awkard so do it in its own
        function"""
        line = ""
        if stddev is not None:
            if (np.abs(stddev) < 1e-2) and (stddev != 0.0):
                line += r'$%.2e\pm%.2e$ &'%(maximum, stddev)
            else:
                line += r'$%.3g\pm%.3g$ &'%(maximum, stddev)
        else:
            if maximum is not None:
                raise ValueError("Both stddev and maximum should be None or "
                                 "specified")
            else:
                line += "-- &"
        if (np.abs(val) < 1e-2) and (val != 0.0):
            line += "%.2e"%val
        else:
            line += "%.3g"%val
        if dataval is not None:
            line += " &"
            if isinstance(dataval, str):
                line += "%s"%dataval
            else:
                delta = val - dataval
                if (np.abs(delta) < 1e-2) and (delta != 0.0):
                    line += "%.2e"%delta
                else:
                    line += "%.3g"%delta
        if not last:
            line += " &"
        return line

    def check_analysis_pickle_files(self):
        """Checks the pickles in the case of hypo_testing analysis"""
        # Make sure that there have been no more trials run since this
        # last processing. To do this, get the number of output files
        for basename in nsort(os.listdir(self.logdir)):
            m = self.labels.subdir_re.match(basename)
            if m is None or 'pckl' in basename:
                continue
            # Here is the output directory which contains the files
            subdir = os.path.join(self.logdir, basename)
            # Account for failed jobs. Get the set of file numbers that
            # exist for all h0 and h1 combinations
            self.get_set_file_nums(
                filedir=subdir
            )
            # Take one of the pickle files to see how many data
            # entries it has.
            data_sets = from_file(os.path.join(self.logdir,
                                               'data_sets.pckl'))
            # Take the first data key and then the h0 fit to h0 fid
            # which should always exist. The length of this is then
            # the number of trials in the pickle files.
            if 'h0_fit_to_h0_fid' in data_sets[data_sets.keys()[0]].keys():
                pckl_trials = len(data_sets[data_sets.keys()[0]][
                    'h0_fit_to_h0_fid'].keys())
                # The number of pickle trials should match the number of
                # trials derived from the output directory.
                if self.num_trials == pckl_trials:
                    logging.info(
                        'Found files I assume to be from a previous run of'
                        ' this processing script containing %i trials. If '
                        'this seems incorrect please delete the files: '
                        'data_sets.pckl, all_params.pckl and labels.pckl '
                        'from the logdir you have provided.'%pckl_trials
                    )
                    pickle_there = True
                else:
                    logging.info(
                        'Found files I assume to be from a previous run of'
                        ' this processing script containing %i trials. '
                        'However, based on the number of json files in the '
                        'output directory there should be %i trials in '
                        'these pickle files, so they will be regenerated.'%(
                            pckl_trials, self.num_trials)
                    )
                    pickle_there = False
            else:
                logging.info(
                    'Found files I assume to be from a previous run of'
                    ' this processing script which do not seem to '
                    'contain any trials, so they will be regenerated.'
                )
                pickle_there = False
                
        return pickle_there

    def check_injparamscan_pickle_files(self):
        """Checks the pickles in the case of injparamscan analysis"""
        # Make sure that there have been no more new scan points run since this
        # last processing. To do this, get the number of output directories
        # Compare this to the number in the pickle files.
        self.num_scan_points = 0
        for basename in nsort(os.listdir(self.logdir)):
            if 'pckl' in basename:
                continue
            basename_content = nsort(
                os.listdir(os.path.join(self.logdir, basename))
            )
            # This means it is a directory containing something a scan point
            if 'config_summary.json' in basename_content:
                self.num_scan_points += 1
        data_sets = from_file(os.path.join(self.logdir,
                                           'data_sets.pckl'))
        if len(data_sets.keys()) == self.num_scan_points:
            logging.info(
                'Found files I assume to be from a previous run of'
                ' this processing script containing %i scan points. If '
                'this seems incorrect please delete the files: '
                'data_sets.pckl, all_params.pckl and labels.pckl '
                'from the logdir you have provided.'%self.num_scan_points
            )
            pickle_there = True
        else:
            logging.info(
                'Found files I assume to be from a previous run of'
                ' this processing script containing %i scan points. '
                'However, based on the number of json files in the '
                'output directory there should be %i scan points in '
                'these pickle files, so they will be regenerated.'%(
                    len(data_sets.keys()), self.num_scan_points)
            )
            pickle_there = False
            
        return pickle_there

    def check_systtests_pickle_files(self):
        """Checks the pickles in the case of systtests analysis"""
        # Make sure that there have been no more new scan points run since this
        # last processing. To do this, get the number of output directories
        # Compare this to the number in the pickle files.
        self.num_systematics = {}
        for basename in nsort(os.listdir(self.logdir)):
            if 'pckl' in basename:
                continue
            basename_content = nsort(
                os.listdir(os.path.join(self.logdir, basename))
            )
            # This means it is a directory containing something useful
            if 'config_summary.json' in basename_content:
                bits = basename.split('toy_')[-1].split('_')
                toyname = None
                add_bit = True
                for bit in bits:
                    if bit == '' or bit == 'inj':
                        add_bit = False
                    if add_bit:
                        if toyname is None:
                            toyname = bit
                        else:
                            toyname += '_%s'%bit
                if '_full_syst_baseline' in toyname:
                    toyname = toyname.split('_full_syst_baseline')[0]
                toyname = 'toy_%s_asimov'%toyname
                if toyname not in self.num_systematics.keys():
                    self.num_systematics[toyname] = 0
                if 'wrong' in basename:
                    # Only want to include each systematic once, but
                    # they will have two directions.
                    if 'pve' in basename:
                        self.num_systematics[toyname] += 1
                else:
                    self.num_systematics[toyname] += 1
        data_sets = from_file(os.path.join(self.logdir,
                                           'data_sets.pckl'))
        if sorted(data_sets.keys()) != sorted(self.num_systematics.keys()):
            logging.info(
                'Found files I assume to be from a previous run of'
                ' this processing script containing these truths: %s. '
                'However, based on the directories in the overall '
                'output directory there should be these truths: %s, so '
                'they will be regenerated.'%(
                    sorted(data_sets.keys()),
                    sorted(self.num_systematics.keys())
                )
            )
        pickle_there = True
        for toyname in sorted(self.num_systematics.keys()):
            if len(data_sets[toyname].keys()) != self.num_systematics[toyname]:
                pickle_there = False
        if pickle_there:
            logging.info(
                'Found files I assume to be from a previous run of'
                ' this processing script containing %i sytematics. If '
                'this seems incorrect please delete the files: '
                'data_sets.pckl, all_params.pckl and labels.pckl '
                'from the logdir you have provided.'%(
                    self.num_systematics[self.num_systematics.keys()[0]])
            )
        else:
            logging.info(
                'Found files I assume to be from a previous run of'
                ' this processing script containing %i systematics. '
                'However, based on the number of directories in the overall '
                'output directory there should be %i systematics in '
                'these pickle files, so they will be regenerated.'%(
                    len(data_sets[data_sets.keys()[0]].keys()),
                    self.num_systematics[self.num_systematics.keys()[0]]
                )
            )
            pickle_there = False

        return pickle_there

    def check_pickle_files(self, logdir_content):
        """Checks for the expected pickle files in the output directory based
        on the analysis and test type. If they are there, it is made sure that
        they are the most up to date they can be. If not, they are regenerated.
        If they're not even there, then this returns false and the full
        extract_trials happens, at the end of which these pickle files will be
        generated for future use."""
        if np.all(np.array(
                [s in logdir_content for s in self.expected_pickles])):
            # Processed output files are there
            if self.test_type == 'analysis':
                pickle_there = self.check_analysis_pickle_files()
            elif self.test_type == 'injparamscan':
                pickle_there = self.check_injparamscan_pickle_files()
            elif self.test_type == 'systtests':
                pickle_there = self.check_systtests_pickle_files()
        else:
            logging.info(
                'Did not find all of the files - %s - expected to indicate '
                'this data has already been extracted.'%self.expected_pickles
            )
            pickle_there = False

        return pickle_there

    def get_set_file_nums(self, filedir, injparam=None, trueordering=None,
                          systematic=None, direction=None):
        """This function returns the set of file numbers that exist for all h0
        and h1 combination. This is needed to account for any failed or
        non-transferred jobs. i.e. for trial X you may not have all of the
        necessary fit files so it must be ignored."""
        file_nums = OrderedDict()
        if injparam is not None:
            wanted_labels = self.labels[injparam]
        elif trueordering is not None:
            if direction is not None:
                wanted_labels = self.labels[trueordering][systematic][direction]
            else:
                wanted_labels = self.labels[trueordering][systematic]
        else:
            wanted_labels = self.labels
        for fname in nsort(os.listdir(filedir)):
            for x in ['0', '1']:
                for y in ['0', '1']:
                    k = 'h{x}_fit_to_h{y}_fid'.format(x=x, y=y)
                    r = wanted_labels.dict[k + '_re']
                    m = r.match(fname)
                    if m is None:
                        continue
                    if self.fluctuate_fid:
                        fid_label = int(m.groupdict()['fid_ind'])
                    else:
                        fid_label = wanted_labels.fid
                    if k not in file_nums:
                        file_nums[k] = []
                    file_nums[k].append(fid_label)
                    break

        set_file_nums = []
        for hypokey in file_nums.keys():
            if len(set_file_nums) == 0:
                set_file_nums = set(file_nums[hypokey])
            else:
                set_file_nums = set_file_nums.intersection(file_nums[hypokey])
        self.set_file_nums = set_file_nums
        self.num_trials = len(set_file_nums)

    def get_starting_params(self, cfg, injparam=None, trueordering=None,
                            systematic=None, direction=None):
        """Extracts the h0, h1 and data (if possible) params from the config
        summary file."""
        all_params = {}
        all_params['h0_params'] = {}
        all_params['h1_params'] = {}
        parse_string = ('(.*)=(.*); prior=(.*),'
                        ' range=(.*), is_fixed=(.*),'
                        ' is_discrete=(.*); help="(.*)"')
        if not self.data_is_data:
            all_params['data_params'] = {}
            for param_string in cfg['data_params']:
                bits = re.match(parse_string, param_string, re.M|re.I)
                if bits.group(5) == 'False':
                    all_params['data_params'][bits.group(1)] = {}
                    all_params['data_params'][bits.group(1)]['value'] \
                        = bits.group(2)
                    all_params['data_params'][bits.group(1)]['prior'] \
                        = bits.group(3)
                    all_params['data_params'][bits.group(1)]['range'] \
                        = bits.group(4)
        else:
            all_params['data_params'] = None
        for param_string in cfg['h0_params']:
            bits = re.match(parse_string, param_string, re.M|re.I)
            if bits.group(5) == 'False':
                all_params['h0_params'][bits.group(1)] = {}
                all_params['h0_params'][bits.group(1)]['value'] \
                    = bits.group(2)
                all_params['h0_params'][bits.group(1)]['prior'] \
                    = bits.group(3)
                all_params['h0_params'][bits.group(1)]['range'] \
                    = bits.group(4)
        for param_string in cfg['h1_params']:
            bits = re.match(parse_string, param_string, re.M|re.I)
            if bits.group(5) == 'False':
                all_params['h1_params'][bits.group(1)] = {}
                all_params['h1_params'][bits.group(1)]['value'] \
                    = bits.group(2)
                all_params['h1_params'][bits.group(1)]['prior'] \
                    = bits.group(3)
                all_params['h1_params'][bits.group(1)]['range'] \
                    = bits.group(4)
        if injparam is not None:
            self.all_params[injparam] = all_params
        elif trueordering is not None:
            if direction is not None:
                self.all_params[trueordering][systematic][
                    direction] = all_params
            else:
                self.all_params[trueordering][systematic] = all_params
        else:
            self.all_params = all_params

    def get_data(self, injparam=None, trueordering=None,
                 systematic=None, direction=None):
        """Get all of the data from the logdir"""
        data_sets = OrderedDict()
        minimiser_info = OrderedDict()
        if injparam is not None:
            content = nsort(os.listdir(self.scandir))
        elif trueordering is not None:
            content = nsort(os.listdir(self.systdir))
        else:
            content = nsort(os.listdir(self.logdir))
        for basename in content:
            if injparam is not None:
                m = self.labels[injparam].subdir_re.match(basename)
                wanted_labels = self.labels[injparam]
            elif trueordering is not None:
                if direction is not None:
                    m = self.labels[trueordering][systematic][
                        direction].subdir_re.match(basename)
                    wanted_labels = self.labels[trueordering][
                        systematic][direction]
                else:
                    m = self.labels[trueordering][
                        systematic].subdir_re.match(basename)
                    wanted_labels = self.labels[trueordering][systematic]
            else:
                m = self.labels.subdir_re.match(basename)
                wanted_labels = self.labels
            if m is None or 'pckl' in basename:
                continue

            if self.fluctuate_data:
                data_ind = int(m.groupdict()['data_ind'])
                dset_label = data_ind
            else:
                dset_label = wanted_labels.data_prefix
                if not wanted_labels.data_name in [None, '']:
                    dset_label += '_' + wanted_labels.data_name
                if not wanted_labels.data_suffix in [None, '']:
                    dset_label += '_' + wanted_labels.data_suffix

            lvl2_fits = OrderedDict()
            lvl2_fits['h0_fit_to_data'] = None
            lvl2_fits['h1_fit_to_data'] = None
            minim_info = OrderedDict()
            minim_info['h0_fit_to_data'] = None
            minim_info['h1_fit_to_data'] = None

            if injparam is not None:
                subdir = os.path.join(self.scandir, basename)
            elif trueordering is not None:
                subdir = os.path.join(self.systdir, basename)
            else:
                subdir = os.path.join(self.logdir, basename)

            # Account for failed jobs. Get the set of file numbers that
            # exist for all h0 an h1 combinations
            self.get_set_file_nums(
                filedir=subdir,
                injparam=injparam,
                trueordering=trueordering,
                systematic=systematic,
                direction=direction
            )
            fnum = None
            
            for fnum, fname in enumerate(nsort(os.listdir(subdir))):
                fpath = os.path.join(subdir, fname)
                for x in ['0', '1']:
                    k = 'h{x}_fit_to_data'.format(x=x)
                    if fname == wanted_labels.dict[k]:
                        lvl2_fits[k] = self.extract_fit(fpath, 'metric_val')
                        break
                    # Also extract fiducial fits if needed
                    if 'toy' in dset_label:
                        ftest = ('hypo_%s_fit_to_%s'
                                 %(wanted_labels.dict['h{x}_name'.format(x=x)],
                                   dset_label))
                    elif dset_label == 'data':
                        ftest = ('hypo_%s_fit_to_data'
                                 %(wanted_labels.dict['h{x}_name'.format(x=x)]))
                    if ftest in fname:
                        k = 'h{x}_fit_to_{y}'.format(x=x, y=dset_label)
                        lvl2_fits[k] = self.extract_fit(
                            fpath,
                            ['metric_val', 'params']
                        )
                        break
                    k = 'h{x}_fit_to_{y}'.format(x=x, y=dset_label)
                    for y in ['0', '1']:
                        k = 'h{x}_fit_to_h{y}_fid'.format(x=x, y=y)
                        r = wanted_labels.dict[k + '_re']
                        m = r.match(fname)
                        if m is None:
                            continue
                        if self.fluctuate_fid:
                            fid_label = int(m.groupdict()['fid_ind'])
                        else:
                            fid_label = wanted_labels.fid
                        if k not in lvl2_fits:
                            lvl2_fits[k] = OrderedDict()
                            minim_info[k] = OrderedDict()
                        if fid_label in self.set_file_nums:
                            lvl2_fits[k][fid_label] = self.extract_fit(
                                fpath,
                                ['metric', 'metric_val', 'params']
                            )
                            minim_info[k][fid_label] = self.extract_fit(
                                fpath,
                                ['minimizer_metadata', 'minimizer_time']
                            )
                        break

            if fnum is None:
                raise ValueError('No files?')

            data_sets[dset_label] = lvl2_fits
            minimiser_info[dset_label] = minim_info
            data_sets[dset_label]['params'] = self.extract_fit(
                fpath,
                ['params']
            )['params']

        if injparam is not None:
            self.data_sets[injparam] = data_sets
            self.minimiser_info[injparam] = minimiser_info
        elif trueordering is not None:
            if direction is not None:
                self.data_sets[trueordering][systematic][direction] = data_sets
            else:
                self.data_sets[trueordering][systematic]= data_sets
        else:
            self.data_sets = data_sets
            self.minimiser_info = minimiser_info

    def pickle_data(self):
        """Will pickle the data for easy access later."""
        if 'data_sets.pckl' in self.expected_pickles:
            to_file(
                self.data_sets,
                os.path.join(self.logdir, 'data_sets.pckl')
            )
        if 'all_params.pckl' in self.expected_pickles:
            to_file(
                self.all_params,
                os.path.join(self.logdir, 'all_params.pckl')
            )
        if 'labels.pckl' in self.expected_pickles:
            to_file(
                self.labels,
                os.path.join(self.logdir, 'labels.pckl')
            )
        if 'minimiser_info.pckl' in self.expected_pickles:
            to_file(
                self.minimiser_info,
                os.path.join(self.logdir, 'minimiser_info.pckl')
            )

    def load_from_pickle(self):
        """Load from the pickle files created by the function above in a
        previous run of this script."""
        if 'data_sets.pckl' in self.expected_pickles:
            self.data_sets = from_file(
                os.path.join(self.logdir, 'data_sets.pckl')
            )
        if 'all_params.pckl' in self.expected_pickles:
            self.all_params = from_file(
                os.path.join(self.logdir, 'all_params.pckl')
            )
        if 'minimiser_info.pckl' in self.expected_pickles:
            self.minimiser_info = from_file(
                os.path.join(self.logdir, 'minimiser_info.pckl')
            )
        if 'labels.pckl' in self.expected_pickles:
            self.labels = from_file(
                os.path.join(self.logdir, 'labels.pckl')
            )

    def parse_binning_string(self, binning_string):
        """Returns a dictionary that can be used to instantiate a binning
        object from the output of having run str on the original binning
        object."""
        if 'MultiDimBinning' in binning_string:
            raise ValueError(
                'This function is designed to work with OneDimBinning'
                ' objects. You should separate the MultiDimBinning '
                'string in to the separate OneDimBinning strings '
                'before calling this function and then reconnect them'
                ' in to the MultiDimBinning object after.'
            )
        if 'OneDimBinning' not in binning_string:
            raise ValueError(
                'String expected to have OneDimBinning in it.'
                ' Got %s'%binning_string
            )
        binning_dict = {}
        if '1 bin ' in binning_string:
            raise ValueError('Singular bin case not dealt with yet')
        elif 'irregularly' in binning_string:
            parse_string = (r'OneDimBinning\((.*), (.*) irregularly-sized' + \
                            r' bins with edges at \[(.*)\] (.*)\)')
            a = re.match(parse_string, binning_string, re.M|re.I)
            # Match should come out None is the bins don't have units
            if a is None:
                parse_string = (r'OneDimBinning\((.*), (.*) ' + \
                                r'irregularly-sized bins with ' + \
                                r'edges at \[(.*)\]\)')
                a = re.match(parse_string, binning_string, re.M|re.I)
            else:
                binning_dict['units'] = a.group(4)
            binning_dict['name'] = a.group(1).strip('\'')
            binning_dict['num_bins'] = int(a.group(2))
            binning_dict['bin_edges'] = [float(i) for i in \
                                         a.group(3).split(', ')]
        elif 'logarithmically' in binning_string:
            parse_string = (r'OneDimBinning\((.*), (.*) ' + \
                            r'logarithmically-uniform ' + \
                            r'bins spanning \[(.*), (.*)\] (.*)\)')
            a = re.match(parse_string, binning_string, re.M|re.I)
            # Match should come out None is the bins don't have units
            if a is None:
                parse_string = (r'OneDimBinning\((.*), (.*) logarithmically' + \
                                r'-uniform bins spanning \[(.*), (.*)\]\)')
                a = re.match(parse_string, binning_string, re.M|re.I)
            else:
                binning_dict['units'] = a.group(5)
            binning_dict['name'] = a.group(1).strip('\'')
            binning_dict['num_bins'] = int(a.group(2))
            binning_dict['domain'] = [float(a.group(3)), float(a.group(4))]
            binning_dict['is_log'] = True
        elif 'equally-sized' in binning_string:
            parse_string = (r'OneDimBinning\((.*), (.*) equally-sized ' + \
                            r'bins spanning \[(.*) (.*)\] (.*)\)')
            a = re.match(parse_string, binning_string, re.M|re.I)
            # Match should come out None is the bins don't have units
            if a is None:
                parse_string = (r'OneDimBinning\((.*), (.*) equally-sized ' + \
                                r'bins spanning \[(.*), (.*)\]\)')
                a = re.match(parse_string, binning_string, re.M|re.I)
            else:
                binning_dict['units'] = a.group(5)
            binning_dict['name'] = a.group(1).strip('\'')
            binning_dict['num_bins'] = int(a.group(2))
            binning_dict['domain'] = [float(a.group(3)), float(a.group(4))]
            binning_dict['is_lin'] = True

        self.add_tex_to_binning(binning_dict)
        return binning_dict

    def add_tex_to_binning(self, binning_dict):
        """Will add a tex to binning dictionaries parsed with the above
        function."""
        if 'reco' in binning_dict['name']:
            sub_string = 'reco'
        elif 'true' in binning_dict['name']:
            sub_string = 'true'
        else:
            sub_string = None
        if 'energy' in binning_dict['name']:
            binning_dict['tex'] = r'$E_{%s}$'%sub_string
        elif 'coszen' in binning_dict['name']:
            binning_dict['tex'] = r'$\cos\theta_{Z,%s}$'%sub_string

    def parse_pint_string(self, pint_string):
        """Will return the value and units from a string with attached
        pint-style units. i.e. the string "0.97 dimensionless" would return a
        value of 0.97 and units of dimensionless. Both will return as
        strings."""
        val = pint_string.split(' ')[0]
        units = pint_string.split(val+' ')[-1]
        return val, units

    def get_num_rows(self, data, omit_metric=False):
        """Calculates the number of rows for multiplots based on the number of
        systematics."""
        if omit_metric:
            num_rows = int((len(data.keys())-1)/4)
        else:
            num_rows = int(len(data.keys())/4)
        if len(data.keys())%4 != 0:
            num_rows += 1
        return num_rows

    #### Profile Scan Specific Postprocessing Functions ####

    def get_scan_steps(self, scandict=None):
        """Gets the bin centres, edges, names and units used in the
        profile scan. This will default to the one passed in the infile
        unless you overwrite scandict."""
        if scandict is None:
            scandict = self.scan_file_dict
            return_vals = False
        else:
            return_vals = True
        all_steps = scandict['steps']
        all_bin_cens = []
        all_bin_units = []
        for step_variable in all_steps.keys():
            bin_cens = []
            if isinstance(all_steps[step_variable][0][1], list):
                if len(all_steps[step_variable][0][1]) == 0:
                    logging.warning(
                        "No units have been found for the scan "
                        "parameter. Making it dimensionless."
                    )
                    all_bin_units.append('dimensionless')
                else:
                    all_bin_units.append(all_steps[step_variable][0][1][0][0])
            else:
                all_bin_units.append('dimensionless')
            for val in all_steps[step_variable]:
                if val[0] not in bin_cens:
                    bin_cens.append(val[0])
            all_bin_cens.append(bin_cens)
        all_bin_edges = []
        for bin_cens in all_bin_cens:
            bin_width = bin_cens[1]-bin_cens[0]
            bin_edges = np.linspace(bin_cens[0]-bin_width/2.0,
                                    bin_cens[-1]+bin_width/2.0,
                                    len(bin_cens)+1)
            all_bin_edges.append(bin_edges)
        if return_vals:
            return (np.array(all_bin_cens), np.array(all_bin_edges),
                    all_steps.keys(), all_bin_units)
        else:
            self.all_bin_cens = np.array(all_bin_cens)
            self.all_bin_edges = np.array(all_bin_edges)
            self.all_bin_names = all_steps.keys()
            self.all_bin_units = all_bin_units

    def get_scan_data(self):
        """Gets the data i.e. best fit metric and params over the scan. If a
        best fit was supplied it will also be extracted. If projections were
        supplied they will be stored after it has been verified they correspond
        to the variables in the 2D scan. Also stores the metric name to self."""
        self.metric_name = self.scan_file_dict['results'][0]['metric']
        data = {}
        data['metric_vals'] = []
        for result in self.scan_file_dict['results']:
            data['metric_vals'].append(result['metric_val'])
            for param_key in result['params'].keys():
                if not result['params'][param_key]['is_fixed']:
                    if param_key not in data.keys():
                        data[param_key] = {}
                        data[param_key]['vals'] = []
                        data[param_key]['units'] = \
                            result['params'][param_key]['prior']['units']
                        data[param_key]['prior'] = \
                            result['params'][param_key]['prior']
                    data[param_key]['vals'].append(
                        result['params'][param_key]['value'][0]
                    )

        if self.best_fit_dict is not None:
            best_fit_data = {}
            best_fit_data['metric_val'] = self.best_fit_dict['metric_val']
            for param_key in self.best_fit_dict['params'].keys():
                if not self.best_fit_dict['params'][param_key]['is_fixed']:
                    best_fit_data[param_key] = {}
                    best_fit_data[param_key]['val'] = \
                        self.best_fit_dict['params'][param_key]['value'][0]
                    best_fit_data[param_key]['units'] = \
                        self.best_fit_dict['params'][param_key]['value'][1]
            # Make a list of shifted metrics based on this best fit point
            data['shifted_metric_vals'] = []
            for val in data['metric_vals']:
                data['shifted_metric_vals'].append(
                    val-best_fit_data['metric_val']
                )
        else:
            best_fit_data = None

        if self.projection_dicts is not None:
            self.proj_bin_names = []
            self.proj_bin_edges = []
            self.proj_bin_cens = []
            self.proj_bin_units = []
            self.projection_data = []
            for projection_dict in self.projection_dicts:
                projection_data = {}
                proj_bin_cens, proj_bin_edges, \
                    proj_bin_names, proj_bin_units = \
                        self.get_scan_steps(scandict=projection_dict)
                if len(proj_bin_names) != 1:
                    raise ValueError(
                        "Projection files should be 1D scans. "
                        "Got %i."%len(proj_bin_names)
                    )
                if proj_bin_names[0] not in self.all_bin_names:
                    raise ValueError(
                        "Projection file was over %s which is "
                        "not in the 2D scan over %s."%(
                            proj_bin_names[0], self.all_bin_names)
                    )
                else:
                    self.proj_bin_names.append(proj_bin_names[0])
                    self.proj_bin_edges.append(proj_bin_edges[0])
                    self.proj_bin_cens.append(proj_bin_cens[0])
                    self.proj_bin_units.append(proj_bin_units[0])
                projection_data['metric_vals'] = []
                for result in projection_dict['results']:
                    projection_data['metric_vals'].append(result['metric_val'])
                    for param_key in result['params'].keys():
                        if not result['params'][param_key]['is_fixed']:
                            if param_key not in projection_data.keys():
                                projection_data[param_key] = {}
                                projection_data[param_key]['vals'] = []
                                projection_data[param_key]['units'] = \
                                    result['params'][
                                        param_key]['prior']['units']
                                projection_data[param_key]['prior'] = \
                                    result['params'][param_key]['prior']
                            projection_data[param_key]['vals'].append(
                                result['params'][param_key]['value'][0]
                            )
                if best_fit_data is not None:
                    projection_data['shifted_metric_vals'] = []
                    for val in projection_data['metric_vals']:
                        projection_data['shifted_metric_vals'].append(
                            val-best_fit_data['metric_val']
                        )
                self.projection_data.append(projection_data)
        else:
            self.projection_data = None

        if self.contour_dicts is not None:
            for contour_dict in self.contour_dicts:
                if not sorted(self.all_bin_names) == \
                   sorted(contour_dict['vars']):
                    special_vars = sorted(['sin2theta23', 'deltam32'])
                    special_bins = sorted(['theta23', 'deltam31'])
                    good_contour = \
                        (sorted(self.all_bin_names) == special_bins) and \
                        (sorted(contour_dict['vars']) == special_vars)
                else:
                    good_contour = True
                if not good_contour:
                    raise ValueError(
                        "Contour variables - %s - do not match "
                        "the scan variables - %s."%(
                            contour_dict['vars'], self.all_bin_names
                        )
                    )

        self.data = data
        self.best_fit_data = best_fit_data

    def get_best_fit(self, xlabel=None, ylabel=None):
        """Extracts the best fit values from the best fit dictionary
        if it is not None"""
        if self.best_fit_data is not None:
            if xlabel is not None:
                best_fit_x = self.best_fit_data[xlabel]['val']
                if ylabel is not None:
                    best_fit_y = self.best_fit_data[ylabel]['val']
                    self.best_fit_point = [best_fit_x, best_fit_y]
                else:
                    self.best_fit_point = best_fit_x
            elif ylabel is None:
                raise ValueError(
                    "You have not specified a x parameter name but have"
                    " specified a y parameter name - %s."%ylabel
                )
        else:
            self.best_fit_point = None

    def sort_scan_data(self, data_key, onedimensional=False):
        """Sorts the scan data and gets the labels and such"""
        if data_key == 'metric_vals':
            label = self.metric_name
            units = 'dimensionless'
            vals = np.array(self.data[data_key])
        elif data_key == 'shifted_metric_vals':
            if not onedimensional:
                label = 'contour'
            else:
                label = 'delta_'+self.metric_name
            units = 'dimensionless'
            vals = np.array(self.data[data_key])
        else:
            label = data_key
            units = self.data[data_key]['units']
            vals = np.array(self.data[data_key]['vals'])

        if not onedimensional:
            vals = np.array(np.split(vals, len(self.all_bin_cens[0])))
        return label, units, vals

    def sort_projection_data(self, data_key, xlabel, ylabel):
        """Gets the projection data stored in self and assigns
        it as "X" or "Y" based on the passed x and y labels."""
        if self.projection_data is not None:
            for i, proj_data in enumerate(self.projection_data):
                if self.proj_bin_names[i] == xlabel:
                    xxvals = self.proj_bin_cens[i]
                    if 'metric_vals' in data_key:
                        xyvals = proj_data[data_key]
                    else:
                        xyvals = proj_data[data_key]['vals']
                elif self.proj_bin_names[i] == ylabel:
                    yxvals = self.proj_bin_cens[i]
                    if 'metric_vals' in data_key:
                        yyvals = proj_data[data_key]
                    else:
                        yyvals = proj_data[data_key]['vals']
                else:
                    raise ValueError(
                        "Got projection variable %s which does "
                        "not match either of %s or %s"%(
                            self.proj_bin_names[i], xlabel, ylabel)
                    )
        else:
            xxvals = None
            xyvals = None
            yxvals = None
            yyvals = None
        return xxvals, xyvals, yxvals, yyvals

    def plot_1d_scans(self, xvals=None, xlabel=None, xunits=None):
        """Makes the 1D scan plots. The x values as well as their
        labels/units can be specified here, or else they will be generated
        from what is stored in self"""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        if xvals is None:
            xvals = self.all_bin_cens[0]
        if xlabel is None:
            xlabel = self.all_bin_names[0]
        if xunits is None:
            xunits = self.all_bin_units[0]
        self.get_best_fit(xlabel=xlabel)
        title_end = "%s Parameter Scan"%(
            self.tex_axis_label(xlabel)
        )
        maintitle = self.make_main_title(end_center=True, end=title_end)

        for data_key in self.data.keys():
            ylabel, yunits, yvals = self.sort_scan_data(
                data_key=data_key,
                onedimensional=True
            )
            self.make_1d_graph(
                xvals=xvals,
                yvals=yvals,
                xlabel=xlabel,
                xunits=xunits,
                ylabel=ylabel,
                yunits=yunits
            )
            if 'metric' not in data_key:
                self.add_prior_region(systkey=ylabel)
            if self.best_fit_data is not None:
                bestfitlabel = 'Best Fit %s'%(self.tex_axis_label(xlabel))
                bestfitlabel += ' = %.3f'%(self.best_fit_point)
                if (xunits is not None) and (not xunits == 'dimensionless'):
                    bestfitlabel += ' %s'%(self.tex_axis_label(xunits))
                plt.axvline(
                    self.best_fit_point,
                    linestyle='-',
                    color='k',
                    linewidth=2,
                    label=bestfitlabel
                )
                plt.legend(loc='upper left')

            plt.title(maintitle, fontsize=16)
            plt.tight_layout()
            save_end = "%s_1D_%s_scan_%s_values"%(
                xlabel, self.metric_name, ylabel)
            self.save_plot(outdir=self.outdir, end=save_end)
            plt.close()

    def plot_2d_scans(self, xbins=None, xlabel=None, xunits=None, xcens=None,
                      ybins=None, ylabel=None, yunits=None, ycens=None):
        """Makes the 2D scan plots. The x and y bins as well as their
        labels/units can be specified here, or else they will be generated
        from what is stored in self"""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        if xbins is None:
            xbins = self.all_bin_edges[0]
        if xlabel is None:
            xlabel = self.all_bin_names[0]
        if xunits is None:
            xunits = self.all_bin_units[0]
        if xcens is None:
            xcens = self.all_bin_cens[0]
        if ybins is None:
            ybins = self.all_bin_edges[1]
        if ylabel is None:
            ylabel = self.all_bin_names[1]
        if yunits is None:
            yunits = self.all_bin_units[1]
        if ycens is None:
            ycens = self.all_bin_cens[1]

        self.get_best_fit(xlabel=xlabel, ylabel=ylabel)
        title_end = "%s / %s Parameter Scan"%(
            self.tex_axis_label(xlabel),
            self.tex_axis_label(ylabel)
        )
        maintitle = self.make_main_title(end_center=True, end=title_end)

        for data_key in self.data.keys():
            zlabel, zunits, zvals = self.sort_scan_data(data_key=data_key)
            if zlabel == 'contour':
                # Contour plots need bin centers...
                self.make_2d_hist_plot(
                    zvals=zvals,
                    xbins=xcens,
                    ybins=ycens,
                    xlabel=xlabel,
                    xunits=xunits,
                    ylabel=ylabel,
                    yunits=yunits,
                    zlabel=zlabel,
                    zunits=zunits
                )
                plt.figtext(
                    0.05,
                    0.05,
                    r"ICECUBE INTERNAL",
                    color='r',
                    size='xx-large'
                )
                plt.grid(zorder=0, linestyle='--')
            else:
                self.make_2d_hist_plot(
                    zvals=zvals,
                    xbins=xbins,
                    ybins=ybins,
                    xlabel=xlabel,
                    xunits=xunits,
                    ylabel=ylabel,
                    yunits=yunits,
                    zlabel=zlabel,
                    zunits=zunits
                )
            if self.best_fit_data is not None:
                plt.plot(
                    self.best_fit_point[0],
                    self.best_fit_point[1],
                    marker='x',
                    linestyle='None',
                    color='k',
                    markersize=10
                )
            plt.title(maintitle, fontsize=16)
            plt.tight_layout()
            save_end = "%s_%s_2D_%s_scan_%s"%(
                xlabel, ylabel, self.metric_name, zlabel)
            if zlabel != "contour":
                save_end += "_values"
            self.save_plot(outdir=self.outdir, end=save_end)
            if zlabel == 'contour':
                if self.logdir is not None:
                    for injkey in self.values.keys():
                        for fhkey in self.values[injkey].keys():
                            self.add_pseudo_experiments(
                                xlabel=xlabel,
                                ylabel=ylabel,
                                injkey=injkey,
                                fhkey=fhkey
                            )
                            save_end = "%s_%s_2D_%s_scan_contour"%(
                                xlabel, ylabel, self.metric_name)
                            save_end += "_w_%s_%s_pseudos"%(
                                injkey, fhkey)
                            self.save_plot(outdir=self.outdir, end=save_end)
                            plt.close()
                            # Need to re-make contour plot for both next
                            # pseudos and the next plots.
                            self.make_2d_hist_plot(
                                zvals=zvals,
                                xbins=xcens,
                                ybins=ycens,
                                xlabel=xlabel,
                                xunits=xunits,
                                ylabel=ylabel,
                                yunits=yunits,
                                zlabel=zlabel,
                                zunits=zunits
                            )
                            plt.figtext(
                                0.05,
                                0.05,
                                r"ICECUBE INTERNAL",
                                color='r',
                                size='xx-large'
                            )
                            plt.grid(zorder=0, linestyle='--')
                            if self.best_fit_data is not None:
                                plt.plot(
                                    self.best_fit_point[0],
                                    self.best_fit_point[1],
                                    marker='x',
                                    linestyle='None',
                                    color='k',
                                    markersize=10
                                )
                            plt.title(maintitle, fontsize=16)
                            plt.tight_layout()
                if self.contour_dicts is not None:
                    self.add_other_contours(
                        xlabel=xlabel,
                        ylabel=ylabel
                    )
                    plt.subplots_adjust(top=0.80)
                    plt.title('')
                    plt.legend(
                        bbox_to_anchor=(0., 1.02, 1., .102),
                        loc=3,
                        ncol=1,
                        mode="expand",
                        borderaxespad=0.,
                        fontsize=12
                    )
                    save_end += "_w_other_contours"
                    self.save_plot(outdir=self.outdir, end=save_end)
            plt.close()
            # Plot again with projections, if necessary
            xxvals, xyvals, yxvals, yyvals = self.sort_projection_data(
                data_key=data_key,
                xlabel=xlabel,
                ylabel=ylabel
            )
            if ((xxvals is not None) and (xyvals is not None)) or \
               ((yxvals is not None) and (yyvals is not None)):
                if zlabel == 'contour':
                    # Contour plots need bin centers...
                    mainplot = self.make_2d_hist_plot(
                        zvals=zvals,
                        xbins=xcens,
                        ybins=ycens,
                        xlabel=xlabel,
                        xunits=xunits,
                        ylabel=ylabel,
                        yunits=yunits,
                        zlabel=zlabel,
                        zunits=zunits,
                        xxvals=xxvals,
                        xyvals=xyvals,
                        yxvals=yxvals,
                        yyvals=yyvals
                    )
                    mainplot.grid(zorder=0, linestyle='--')
                    plt.figtext(
                        0.40,
                        0.15,
                        r"ICECUBE INTERNAL",
                        color='r',
                        size='xx-large'
                    )
                else:
                    mainplot = self.make_2d_hist_plot(
                        zvals=zvals,
                        xbins=xbins,
                        ybins=ybins,
                        xlabel=xlabel,
                        xunits=xunits,
                        ylabel=ylabel,
                        yunits=yunits,
                        zlabel=zlabel,
                        zunits=zunits,
                        xxvals=xxvals,
                        xyvals=xyvals,
                        yxvals=yxvals,
                        yyvals=yyvals
                    )
                if self.best_fit_data is not None:
                    mainplot.plot(
                        self.best_fit_point[0],
                        self.best_fit_point[1],
                        marker='x',
                        linestyle='None',
                        color='k',
                        markersize=10
                    )
                plt.subplots_adjust(
                    left=0.35,
                    right=0.95,
                    top=0.95,
                    bottom=0.13
                )
                save_end = "%s_%s_2D_%s_scan_%s"%(
                    xlabel, ylabel, self.metric_name, zlabel)
                if zlabel != "contour":
                    save_end += "_values"
                save_end += "_w_1D_projections"
                self.save_plot(outdir=self.outdir, end=save_end)
                plt.close()

    def add_pseudo_experiments(self, xlabel, ylabel, injkey, fhkey):
        """Will add the pseudo experiments contained in
        self.values[injkey][fhkey] on to whatever is currently in
        plt. The idea is to overlay them on top of contours, so it
        will find the appropriate dimensions from xlabel and ylabel."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        xdata = self.values[injkey][fhkey][xlabel]
        ydata = self.values[injkey][fhkey][ylabel]
        self.make_2d_scatter_plot(
            xdata=xdata['vals'],
            ydata=ydata['vals'],
            plot_cor=False,
            set_range=False
        )

    def make_other_contour(self, contour_vals, xlabel, ylabel,
                           contour_dict, do_label=1):
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        """Makes the actual other contour plot. The do_label argument sets
        whether to label it or not. This allows multiple contours to have
        only one label e.g. NoVA with a contour either side of maximal."""
        xvals = []
        yvals = []
        for vals in contour_vals:
            for i, var in enumerate(contour_dict['vars']):
                if var == 'deltam32':
                    vals[i] /= 1000
                if var == xlabel:
                    xvals.append(vals[i])
                elif var == ylabel:
                    yvals.append(vals[i])
                else:
                    raise ValueError(
                        "Got a variable - %s - that was not %s or %s"
                        "as expected"%(var, xlabel, ylabel)
                    )
        if do_label == 1:
            plabel = contour_dict['label']
        else:
            plabel = None
        plt.plot(
            xvals,
            yvals,
            linestyle=contour_dict['linestyle'],
            label=plabel,
            color=contour_dict['color'],
            linewidth=2,
            zorder=1
        )
        if 'best_fit' in contour_dict.keys():
            for i, var in enumerate(contour_dict['vars']):
                if var == 'deltam32':
                    contour_dict['best_fit'][i] /= 1000.0
                if var == xlabel:
                    xval = contour_dict['best_fit'][i]
                else:
                    yval = contour_dict['best_fit'][i]
            plt.plot(
                xval,
                yval,
                linestyle='',
                marker='o',
                color=contour_dict['color']
            )
        xlim = plt.gca().get_xlim()
        if min(xvals) < xlim[0]:
            xmin = 0.9*min(xvals)
        else:
            xmin = xlim[0]
        if max(xvals) > xlim[1]:
            xmax = 1.1*max(xvals)
        else:
            xmax = xlim[1]
        plt.xlim(xmin, xmax)
        ylim = plt.gca().get_ylim()
        if min(yvals) < ylim[0]:
            ymin = 0.9*min(yvals)
        else:
            ymin = ylim[0]
        if max(yvals) > ylim[1]:
            ymax = 1.1*max(yvals)
        else:
            ymax = ylim[1]
        plt.ylim(ymin, ymax)

    def add_other_contours(self, xlabel, ylabel):
        """Adds the other contours stored in self.contours_dict to the
        plot if they exist and if the variables match."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        for contour_dict in self.contour_dicts:
            if (xlabel in contour_dict['vars']) and \
               (ylabel in contour_dict['vars']):
                if isinstance(contour_dict['contour'], dict):
                    for i, ckey in enumerate(contour_dict['contour'].keys()):
                        self.make_other_contour(
                            contour_vals=contour_dict['contour'][ckey],
                            xlabel=xlabel,
                            ylabel=ylabel,
                            contour_dict=contour_dict,
                            do_label=i
                        )
                else:
                    self.make_other_contour(
                        contour_vals=contour_dict['contour'],
                        xlabel=xlabel,
                        ylabel=ylabel,
                        contour_dict=contour_dict,
                        do_label=1
                    )

    #### Generic Functions Relating to Plotting ####

    def make_data_label(self):
        """Makes a label for the data accounting for detector and
        selection. If these are not set it will default to IceCube."""
        data_label = ""
        if self.detector is not None:
            data_label += "%s "%self.detector
        if self.selection is not None:
            data_label += "%s Event Selection"%self.selection
        if data_label == "":
            data_label = "IceCube"
        return data_label

    def make_main_title(self, end, end_center=False):
        """Make main title accounting for detector and selection. Set
        end_center to true if you will not be using this with a
        corresponding fit title"""
        main_title = r"\begin{center}"
        if self.detector is not None:
            main_title += "%s "%self.detector
        if self.selection is not None:
            main_title += "%s Event Selection "%self.selection
        main_title += end
        if end_center:
            main_title += r"\end{center}"
        return main_title

    def make_fit_title(self, trials, fid=None, hypo=None,
                       fhkey=None, begin_center=False):
        """Make fit title to go with the main title. Set begin_center to
        true if you will not be using this with a corresponding main title."""
        fittitle = ""
        if begin_center:
            fittitle += r"\begin{center}"
        if hasattr(self, 'labels'):
            if self.labels.dict['data_name'] == '':
                fittitle += "Data, "
            else:
                fittitle += "True %s, "%self.labels.dict['data_name']
        if ((fid is not None) and (hypo is not None)) and (fhkey is not None):
            raise ValueError(
                "Got a fid, hypo and fhkey specified. Please use fid "
                "and hypo OR fhkey (from which fid and hypo will be "
                "extracted) but not both."
            )
        if fid is not None:
            fittitle += "Fiducial Fit %s, "%self.labels.dict['%s_name'%fid]
        if hypo is not None:
            if hypo == 'both':
                fittitle += "Both (%s/%s) Hypotheses "%(
                    self.labels.dict['h0_name'], self.labels.dict['h1_name'])
            else:
                fittitle += "Hypothesis %s "%self.labels.dict['%s_name'%hypo]
        if fhkey is not None:
            hypo = self.get_hypo_from_fiducial_hypo_key(fhkey=fhkey)
            fid = self.get_fid_from_fiducial_hypo_key(fhkey=fhkey)
            fittitle += "Fiducial Fit %s, "%self.labels.dict['%s_name'%fid]
            fittitle += "Hypothesis %s "%self.labels.dict['%s_name'%hypo]
        if trials is not None:
            fittitle += "(%i Trials)"%trials
        fittitle += r"\end{center}"
        return fittitle

    def make_1d_hist_plot(self, data, xlabel, title, ylabel, bins=10,
                          histtype='bar', color='darkblue', alpha=0.9,
                          xlabelsize='18', ylabelsize='18',
                          titlesize=16, label=None, subplots_adjust=True,
                          subplotnum=None, lw=1, subplotcols=4, normed=False):
        """Generic 1D histogram plotting function. Set subplots_adjust to
        True if the title goes over two lines and you need the plot to
        account for this."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        plt.grid(axis='y', zorder=0)
        plt.hist(
            data,
            bins=bins,
            histtype=histtype,
            color=color,
            alpha=alpha,
            zorder=3,
            label=label,
            lw=lw,
            normed=normed
        )
        plt.xlabel(xlabel, size=xlabelsize)
        if subplotnum is not None:
            if (subplotnum-1)%subplotcols == 0:
                plt.ylabel(ylabel, size=ylabelsize)
        else:
            plt.ylabel(ylabel, size=ylabelsize)
            plt.title(title, fontsize=titlesize)
        if subplots_adjust:
            plt.subplots_adjust(left=0.10, right=0.90, top=0.85, bottom=0.11)

    def make_1d_graph(self, xvals, yvals, xlabel, xunits,
                      ylabel, yunits, nicexlabel=True,
                      niceylabel=True ,xlims='edges', ylims=None,
                      linestyle='-', color='darkblue', alpha=0.9,
                      xlabelsize='18', ylabelsize='18', marker=None,
                      plotlabel=None, subplotnum=None, linewidth=1):
        """Generic 1D graph plotting function. The x limits will be set as
        the edges of the xvals unless overwritten. Set this to None to
        leave it as matplotlib dictates. The y limits will be left alone
        unless overwritten."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        plt.plot(
            xvals,
            yvals,
            linestyle=linestyle,
            color=color,
            alpha=alpha,
            marker=marker,
            label=plotlabel,
            linewidth=linewidth
        )
        if xlims is not None:
            if xlims == 'edges':
                plt.xlim(xvals[0], xvals[-1])
            else:
                plt.xlim(xlims)
        if xlabel is not None:
            if nicexlabel:
                xlabel = self.make_label(xlabel, xunits)
            plt.xlabel(xlabel, fontsize=xlabelsize)
        if ylims is not None:
            if ylims[0] == ylims[1]:
                plt.ylim(ylims[0]-0.1, ylims[0]+0.1)
            else:
                plt.ylim(ylims)
        if ylabel is not None:
            if subplotnum is not None:
                if (subplotnum-1)%4 == 0:
                    if niceylabel:
                        ylabel = self.make_label(ylabel, yunits)
                    plt.ylabel(ylabel, fontsize=ylabelsize)
            else:
                if niceylabel:
                    ylabel = self.make_label(ylabel, yunits)
                plt.ylabel(ylabel, fontsize=ylabelsize)

    def make_2d_hist_plot(self, zvals, xbins, ybins, xlabel,
                          ylabel, zlabel, xunits=None, yunits=None,
                          zunits=None, cmap=None, xticks=None,
                          yticks=None, xxvals=None, xyvals=None,
                          yxvals=None, yyvals=None):
        """Generic 2D histogram-style plotting function. Set zlabel to contour
        to make a contour plot instead of a histogram. cmap will be Blues
        unless explicitly overwritten. If any of the units
        are set None then the make_label function will just apply
        self.tex_axis_label to the string passed in either xlabel, ylabel
        or zlabel. Set xxvals/xyvals and yxvals/yyvals to add 1D projections
        to the edges of the plots."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        if (xxvals is not None) or (xyvals is not None):
            if not ((xxvals is not None) and (xyvals is not None)):
                raise ValueError(
                    "When specifying projections, both xx and "
                    "xy vals must be specified."
                )
        if (yxvals is not None) or (yxvals is not None):
            if not ((yxvals is not None) and (yyvals is not None)):
                raise ValueError(
                    "When specifying projections, both yx and "
                    "yy vals must be specified."
                )
            fig, axes = plt.subplots(
                nrows=2,
                ncols=2,
                gridspec_kw={
                    'width_ratios': [4, 1],
                    'height_ratios': [1, 4],
                    'wspace': 0.025,
                    'hspace': 0.025
                }
            )
            if zlabel == 'contour':
                X, Y = np.meshgrid(xbins, ybins)
                im = axes[1, 0].contour(
                    X,
                    Y,
                    zvals.T,
                    levels=[0, 4.605],
                    colors="k",
                    linewidths=3,
                    origin="lower"
                )
            else:
                if cmap is None:
                    cmap = "Blues"
                im = axes[1, 0].pcolormesh(xbins, ybins, zvals.T, cmap=cmap)
                cax = fig.add_axes([0.15, 0.13, 0.03, 0.595])
                nice_zlabel = self.make_label(zlabel, zunits)
                cb = fig.colorbar(im, cax=cax)
                cb.set_label(
                    label=nice_zlabel,
                    fontsize=24
                )
                cb.ax.yaxis.set_ticks_position('left')
                cb.ax.yaxis.set_label_position('left')
            axes[0, 1].set_visible(False)
            axes[1, 0].set_xlim(xbins[0], xbins[-1])
            axes[1, 0].set_ylim(ybins[0], ybins[-1])
            axes[0, 0].plot(xxvals, xyvals)
            if zlabel == 'contour':
                axes[0, 0].set_ylim(0.0, 2.0)
                axes[0, 0].set_ylabel(
                    self.tex_axis_label('delta_%s'%self.metric_name)
                )
            axes[0, 0].set_xlim(xbins[0], xbins[-1])
            axes[0, 0].tick_params(
                axis='x',
                which='both',
                bottom='off',
                top='off',
                labelbottom='off'
            )
            axes[0, 0].grid(zorder=0, linestyle='--')
            axes[1, 1].plot(yyvals, yxvals)
            if zlabel == 'contour':
                axes[1, 1].set_xlim(0.0, 2.0)
                axes[1, 1].set_xlabel(
                    self.tex_axis_label('delta_%s'%self.metric_name)
                )
            axes[1, 1].set_ylim(ybins[0], ybins[-1])
            axes[1, 1].tick_params(
                axis='y',
                which='both',
                left='off',
                right='off',
                labelleft='off'
            )
            axes[1, 1].grid(zorder=0, linestyle='--')
            if xlabel is not None:
                nice_xlabel = self.make_label(xlabel, xunits)
                axes[1, 0].set_xlabel(
                    nice_xlabel,
                    fontsize=24
                )
            if ylabel is not None:
                nice_ylabel = self.make_label(ylabel, yunits)
                axes[1, 0].set_ylabel(
                    nice_ylabel,
                    fontsize=24
                )
            return axes[1, 0]
        else:
            if zlabel == 'contour':
                X, Y = np.meshgrid(xbins, ybins)
                im = plt.contour(
                    X,
                    Y,
                    zvals.T,
                    levels=[0, 4.605],
                    colors="k",
                    linewidths=3,
                    origin="lower"
                )
                # Save contour data to a file
                contour_data = {}
                contour_data['label'] = self.make_data_label()
                contour_data['contour'] = im.allsegs[1][0]
                if self.best_fit_data is not None:
                    contour_data['best_fit'] = self.best_fit_point
                contour_data['vars'] = [xlabel, ylabel]
                contour_data['color'] = 'k'
                contour_data['linestyle'] = '-'
                contour_file = "%s_%s_2D_%s_scan_contour_data.json"%(
                    xlabel, ylabel, self.metric_name)
                to_file(
                    contour_data,
                    os.path.join(self.outdir, contour_file),
                    warn=False
                )
            else:
                if cmap is None:
                    cmap = "Blues"
                im = plt.pcolormesh(xbins, ybins, zvals.T, cmap=cmap)
                nice_zlabel = self.make_label(zlabel, zunits)
                plt.colorbar(im).set_label(
                    label=nice_zlabel,
                    fontsize=24
                )
            plt.xlim(xbins[0], xbins[-1])
            plt.ylim(ybins[0], ybins[-1])
            if xlabel is not None:
                nice_xlabel = self.make_label(xlabel, xunits)
                plt.xlabel(
                    nice_xlabel,
                    fontsize=24
                )
            if ylabel is not None:
                nice_ylabel = self.make_label(ylabel, yunits)
                plt.ylabel(
                    nice_ylabel,
                    fontsize=24
                )
            if xticks is not None:
                if len(xticks) != (len(xbins)-1):
                    raise ValueError(
                        "Got %i ticks for %i bins."%(len(xticks), len(xbins)-1)
                    )
                plt.xticks(
                    np.arange(len(xticks)),
                    xticks,
                    rotation=45,
                    horizontalalignment='right'
                )
            if yticks is not None:
                if len(yticks) != (len(ybins)-1):
                    raise ValueError(
                        "Got %i ticks for %i bins."%(len(yticks), len(ybins)-1)
                    )
                plt.yticks(
                    np.arange(len(xticks)),
                    yticks,
                    rotation=0
                )

    def add_annotation_to_2d_hist(self, annotations):
        """Adds annotations to bins of 2D hist. Expects to be able
        to import PathEffects and will fail if it can't."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        try:
            import matplotlib.patheffects as PathEffects
        except:
            raise ImportError()

        for i in range(0, len(annotations)):
            for j in range(0, len(annotations[0])):
                plt.text(i, j, '%.2f'%annotations[i][j],
                         fontsize='7',
                         verticalalignment='center',
                         horizontalalignment='center',
                         color='w',
                         path_effects=[PathEffects.withStroke(
                             linewidth=2.5,
                             foreground='k'
                         )])

    def make_2d_scatter_plot(self, xdata, ydata, xlabel=None, xunits=None,
                             ylabel=None, yunits=None, title=None,
                             subplotnum=None, num_rows=None,
                             plot_cor=True, set_range=True):
        """Generic 2D scatter plotting function."""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True

        if not set_range:
            xlim = plt.gca().get_xlim()
            ylim = plt.gca().get_ylim()

        plt.scatter(xdata, ydata)

        # Adjust ranges unless told otherwise
        if set_range:
            if isinstance(xdata, list):
                hrange = max(xdata) - min(xdata)
                if hrange != 0.0:
                    plt.xlim(min(xdata)-0.1*hrange,
                             max(xdata)+0.1*hrange)
            elif isinstance(xdata, np.ndarray):
                hrange = xdata.max() - xdata.min()
                if hrange != 0.0:
                    plt.xlim(xdata.min()-0.1*hrange,
                             xdata.max()+0.1*hrange)
            if isinstance(ydata, list):
                vrange = max(ydata) - min(ydata)
                if vrange != 0.0:
                    plt.ylim(min(ydata)-0.1*vrange,
                             max(ydata)+0.3*vrange)
            elif isinstance(ydata, np.ndarray):
                vrange = ydata.max() - ydata.min()
                if vrange != 0.0:
                    plt.ylim(ydata.min()-0.1*vrange,
                             ydata.max()+0.3*vrange)
        else:
            plt.xlim(xlim)
            plt.ylim(ylim)
        if plot_cor:
            # Calculate correlation and annotate
            rho, pval = self.get_correlation_coefficient(
                xdata=xdata,
                ydata=ydata,
                xsystkey=xlabel,
                ysystkey=ylabel
            )
            if (len(set(xdata)) != 1) and (len(set(ydata)) != 1):
                if subplotnum is not None:
                    if num_rows is None:
                        raise ValueError(
                            "Need to know the number of rows in "
                            "order to correctly place the correlation "
                            "annotation on the subplot"
                        )
                    row = int((subplotnum-1)/4)
                    xtext = 0.25*0.25+((subplotnum-1)%4)*0.25
                    ytext = 0.88-(1.0/num_rows)*0.9*row
                    plt.figtext(
                        xtext,
                        ytext,
                        'Correlation = %.2f'%rho,
                        fontsize='large'
                    )
                else:
                    plt.figtext(
                        0.15,
                        0.80,
                        'Correlation = %.2f'%rho,
                        fontsize=16
                    )

        # Set labels, if required
        if xlabel is not None:
            nice_xlabel = self.make_label(xlabel, xunits)
            plt.xlabel(nice_xlabel, fontsize=16)
        if ylabel is not None:
            nice_ylabel = self.make_label(ylabel, yunits)
            plt.ylabel(nice_ylabel, fontsize=16)
        if subplotnum is None and (title is not None):
            plt.title(title, fontsize=16)

    def get_correlation_coefficient(self, xdata, ydata, xsystkey, ysystkey):
        """Calculate the correlation coefficient between x and y"""
        if len(set(xdata)) == 1:
            logging.warning(
                "Parameter %s appears to not have been varied. "
                "i.e. all of the values in the set are the "
                "same. This will lead to NaN in the correlation "
                "calculation and so it will not be done."%xsystkey
            )
        if len(set(ydata)) == 1:
            logging.warning(
                "Parameter %s appears to not have been varied. "
                "i.e. all of the values in the set are the "
                "same. This will lead to NaN in the correlation "
                "calculation and so it will not be done."%ysystkey
            )
        if (len(set(xdata)) != 1) and (len(set(ydata)) != 1):
            rho, pval = spearmanr(xdata, ydata)
        else:
            rho = np.nan
            pval = 0
        return rho, pval

    def save_plot(self, outdir, end, fid=None, hypo=None,
                  fhkey=None, truth=None):
        """Save plot as each type of file format specified in self.formats"""
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        save_name = ""
        if hasattr(self, 'labels') and \
           not self.analysis_type == 'profile_scan':
            if isinstance(self.labels, dict):
                wanted_labels = self.labels[self.labels.keys()[0]]
            else:
                wanted_labels = self.labels
            if truth is None:
                if wanted_labels.dict['data_name'] == '':
                    save_name += "data_"
                else:
                    save_name += "true_%s_"%wanted_labels.dict['data_name']
            else:
                save_name += "true_%s_"%truth    
        if self.detector is not None:
            save_name += "%s_"%self.detector
        if self.selection is not None:
            save_name += "%s_"%self.selection
        if ((fid is not None) and (hypo is not None)) and (fhkey is not None):
            raise ValueError(
                "Got a fid, hypo and fhkey specified. Please use fid "
                "and hypo OR fhkey (from which fid and hypo will be "
                "extracted) but not both."
            )
        if fid is not None:
            save_name += "fid_%s_"%wanted_labels.dict['%s_name'%fid]
        if hypo is not None:
            if hypo == 'both':
                save_name += "both_hypos_%s_%s_"%(
                    wanted_labels.dict['h0_name'],
                    wanted_labels.dict['h1_name']
                )
            else:
                save_name += "hypo_%s_"%wanted_labels.dict['%s_name'%hypo]
        if fhkey is not None:
            hypo = self.get_hypo_from_fiducial_hypo_key(fhkey=fhkey)
            fid = self.get_fid_from_fiducial_hypo_key(fhkey=fhkey)
            save_name += "fid_%s_"%wanted_labels.dict['%s_name'%fid]
            save_name += "hypo_%s_"%wanted_labels.dict['%s_name'%hypo]
        save_name += end
        for fileformat in self.formats:
            full_save_name = save_name + '.%s'%fileformat
            plt.savefig(os.path.join(outdir, full_save_name))

    def make_tex_name(self, end):
        """Make file name for tex output files"""
        tex_name = ""
        if hasattr(self, 'labels'):
            if self.labels.dict['data_name'] == '':
                tex_name += "data_"
            else:
                tex_name += "true_%s_"%self.labels.dict['data_name']
        if self.detector is not None:
            tex_name += "%s_"%self.detector
        if self.selection is not None:
            tex_name += "%s_"%self.selection
        tex_name += end
        tex_name += ".tex"
        return tex_name

    #### General Style Functions ####

    def make_label(self, label, units):
        """Appends units to a label for plotting."""
        nice_label = self.tex_axis_label(label)
        if not (units == 'dimensionless') and \
           (units is not None) and (not units == []):
            nice_label += ' (%s)'%self.tex_axis_label(units)
        return nice_label

    def tex_axis_label(self, label, smalllabel=False):
        """Takes the labels used in the objects and turns them in to something
        nice for plotting. This can never truly be exhaustive, but it
        definitely does the trick. If something looks ugly add it to this
        function!"""
        if isinstance(label, list):
            label = label[0]
        if not isinstance(label, str):
            raise ValueError("Label must be a string. Got %s of "
                             "type %s"%(label, type(label)))
        label = label.lower()
        pretty_labels = {}
        pretty_labels["atm_muon_scale"] = r"Muon Background Scale"
        pretty_labels["nue_numu_ratio"] = r"$\nu_e/\nu_{\mu}$ Ratio"
        pretty_labels["nu_nc_norm"] = r"$\nu$ NC Scale"
        pretty_labels["nu_nubar_ratio"] = r"$\nu/\bar{\nu}$ Ratio"
        pretty_labels["barr_uphor_ratio"] = r"Barr Up/Horizontal Ratio"
        pretty_labels["barr_nu_nubar_ratio"] = r"Barr $\nu/\bar{\nu}$ Ratio"
        pretty_labels["barr_uphor"] = r"Barr Up/Horizontal Ratio"
        pretty_labels["barr_nu_nubar"] = r"Barr $\nu/\bar{\nu}$ Ratio"
        pretty_labels["delta_index"] = r"Atmospheric Index Change"
        pretty_labels["theta13"] = r"$\theta_{13}$"
        pretty_labels["theta23"] = r"$\theta_{23}$"
        pretty_labels["deltacp"] = r"$\delta_{\mathrm{CP}}$"
        pretty_labels["gamma"] = r"$\Gamma$"
        pretty_labels["sin2theta23"] = r"$\sin^2\theta_{23}$"
        pretty_labels["deltam31"] = r"$\Delta m^2_{31}$"
        pretty_labels["deltam32"] = r"$\Delta m^2_{32}$"
        pretty_labels["deltam3l"] = r"$\Delta m^2_{3l}$"
        pretty_labels["aeff_scale"] = r"$A_{\mathrm{eff}}$ Scale"
        pretty_labels["energy_scale"] = r"Energy Scale"
        pretty_labels["genie_ma_qe"] = r"GENIE $M_{A}^{QE}$"
        pretty_labels["genie_ma_res"] = r"GENIE $M_{A}^{Res}$"
        pretty_labels["dom_eff"] = r"DOM Efficiency"
        pretty_labels["hole_ice"] = r"Hole Ice"
        pretty_labels["hole_ice_fwd"] = r"Hole Ice Forward"
        pretty_labels["degree"] = r"$^\circ$"
        pretty_labels["radians"] = r"rads"
        pretty_labels["radian"] = r"rads"
        pretty_labels["electron_volt ** 2"] = r"$\mathrm{eV}^2$"
        pretty_labels["electron_volt"] = r"$\mathrm{eV}^2$"
        pretty_labels["gigaelectron_volt"] = r"$\mathrm{GeV}$"
        pretty_labels["llh"] = r"Likelihood"
        pretty_labels["conv_llh"] = r"Convoluted Likelihood"
        pretty_labels["chi2"] = r"$\chi^2$"
        pretty_labels["mod_chi2"] = r"Modified $\chi^2$"
        pretty_labels["delta_llh"] = r"$\Delta$ Likelihood"
        pretty_labels["delta_conv_llh"] = r"$\Delta$ Convoluted Likelihood"
        pretty_labels["delta_chi2"] = r"$\Delta\chi^2$"
        pretty_labels["delta_mod_chi2"] = r"$\Delta$ $\chi^2_{\mathrm{mod}}$"
        if smalllabel:
            pretty_labels["no"] = r"NO"
            pretty_labels["io"] = r"IO"
        else:
            pretty_labels["no"] = r"Normal Ordering"
            pretty_labels["io"] = r"Inverted Ordering"
        pretty_labels["nomsw"] = r"Normal Ordering, Matter Oscillations"
        pretty_labels["iomsw"] = r"Inverted Ordering, Matter Oscillations"
        pretty_labels["novacuum"] = r"Normal Ordering, Vacuum Oscillations"
        pretty_labels["iovacuum"] = r"Inverted Ordering, Vacuum Oscillations"
        pretty_labels["msw"] = r"Matter Oscillations"
        pretty_labels["vacuum"] = r"Vacuum Oscillations"
        pretty_labels["no,llr"] = r"LLR Method"
        pretty_labels["no,llr,nufitpriors"] = r"LLR Method, Nu-Fit Priors"
        pretty_labels["io,llr"] = r"llr Method"
        pretty_labels["io,llr,nufitpriors"] = r"LLR Method, Nu-Fit Priors"
        pretty_labels["nue"] = r"$\nu_e$"
        pretty_labels["nuebar"] = r"$\bar{\nu}_e$"
        pretty_labels["numu"] = r"$\nu_{\mu}$"
        pretty_labels["numubar"] = r"$\bar{\nu}_{\mu}$"
        pretty_labels["second"] = r"s"
        pretty_labels["seconds"] = r"s"
        pretty_labels["atm_delta_index"] = r"Atmospheric Index Change"
        pretty_labels["pve"] = r"Positive"
        pretty_labels["nve"] = r"Negative"
        pretty_labels["fitwrong"] = r"Sensitivity Stability"
        pretty_labels["fixwrong"] = r"Fitting Relevance"
        pretty_labels["nminusone"] = r"Hidden Potential"
        pretty_labels["minimiser_times"] = r"Minimiser Time (seconds)"
        pretty_labels["minimiser_iterations"] = r"Minimiser Iterations"
        pretty_labels["minimiser_funcevals"] = r"Minimiser Function Evaluations"
        pretty_labels["minimiser_status"] = r"Minimiser Status"
        pretty_labels["correlation_coefficients"] = r"Correlation Coefficients"
        pretty_labels["true no, llr"] = r"True Normal Ordering, LLR"
        pretty_labels["true io, llr"] = r"True Inverted Ordering, LLR"
        pretty_labels["e_res_scale"] = r"Energy Resolution Scale"
        pretty_labels["cz_res_scale"] = r"$\cos\theta_Z$ Resolution Scale"
        pretty_labels["livetime"] = r"Livetime"
        pretty_labels["julian_year"] = r"Years"
        if label not in pretty_labels.keys():
            logging.warning("I have no nice label for %s. Returning as is."%label)
            return label
        return pretty_labels[label]

    def plot_colour(self, label):
        """Will return a standard colour scheme which can be used for e.g.
        specific truths or specific ice models etc."""
        label = label.lower()
        pretty_colours = {}
        # SPIce HD
        pretty_colours['544'] = 'maroon'
        pretty_colours['545'] = 'goldenrod'
        pretty_colours['548'] = 'blueviolet'
        pretty_colours['549'] = 'forestgreen'
        # H2
        ## DOM Efficiency Sets
        pretty_colours['551'] = 'cornflowerblue'
        pretty_colours['552'] = 'cornflowerblue'
        pretty_colours['553'] = 'cornflowerblue'
        pretty_colours['554'] = 'mediumseagreen'
        pretty_colours['555'] = 'mediumseagreen'
        pretty_colours['556'] = 'mediumseagreen'
        ## Hole Ice Sets
        pretty_colours['560'] = 'olive'
        pretty_colours['561'] = 'olive'
        pretty_colours['564'] = 'darkorange'
        pretty_colours['565'] = 'darkorange'
        pretty_colours['572'] = 'teal'
        pretty_colours['573'] = 'teal'
        ## Dima Hole Ice Set without RDE
        pretty_colours['570'] = 'mediumvioletred'
        ## Baseline
        pretty_colours['585'] = 'slategrey'
        # Systematics
        pretty_colours['aeff_scale'] = 'maroon'
        pretty_colours['atm_muon_scale'] = 'goldenrod'
        pretty_colours['deltam31'] = 'blueviolet'
        pretty_colours['theta23'] = 'forestgreen'
        pretty_colours['hole_ice_fwd'] = 'mediumvioletred'
        pretty_colours['dom_eff'] = 'cornflowerblue'
        pretty_colours['genie_ma_qe'] = 'mediumseagreen'
        pretty_colours['genie_ma_res'] = 'olive'
        pretty_colours['hole_ice'] = 'darkorange'
        pretty_colours['nue_numu_ratio'] = 'teal'
        pretty_colours['theta13'] = 'fuchsia'
        pretty_colours['barr_nu_nubar'] = 'thistle'
        pretty_colours['barr_uphor'] = 'orchid'
        pretty_colours['delta_index'] = 'navy'
        # Mass ordering
        pretty_colours['no'] = 'r'
        pretty_colours['io'] = 'b'
        # Asimov fits
        pretty_colours['th_to_wh'] = 'darkviolet'
        pretty_colours['wh_to_th'] = 'deepskyblue'
        colourlabel = None
        for colourkey in pretty_colours.keys():
            if (colourkey in label) or (colourkey == label):
                colourlabel = pretty_colours[colourkey]
        if colourlabel is None:
            logging.debug("I do not have a colour scheme for your label %s. "
                          "Returning black."%label)
            colourlabel = 'k'
        return colourlabel

    def plot_style(self, label):
        """Will return a standard line style for plots similar to above."""
        label = label.lower()
        pretty_styles = {}
        # H2
        ## DOM Efficiency Sets
        pretty_styles['552'] = '--'
        pretty_styles['553'] = '-.'
        pretty_styles['555'] = '--'
        pretty_styles['556'] = '-.'
        ## Hole Ice Sets
        pretty_styles['561'] = '--'
        pretty_styles['565'] = '--'
        pretty_styles['572'] = '--'
        pretty_styles['573'] = '-.'
        colourstyle = None
        for colourkey in pretty_styles.keys():
            if colourkey in label:
                colourstyle = pretty_styles[colourkey]
        if colourstyle is None:
            logging.debug("I do not have a style for your label %s. "
                          "Returning standard."%label)
            colourstyle = '-'
        return colourstyle

    def marker_style(self, label):
        """Will return a standard marker style for plots similar to above."""
        label = label.lower()
        pretty_markers = {}
        # NMO
        pretty_markers['no'] = 'o'
        pretty_markers['io'] = 'o'
        # LLR
        pretty_markers['true no, llr'] = 'D'
        pretty_markers['true io, llr'] = 'D'
        # Asimov Fits
        pretty_markers['th_to_wh'] = 'o'
        pretty_markers['wh_to_th'] = 'o'
        # MSW
        pretty_markers['msw'] = '^'
        markerstyle = None
        for markerkey in pretty_markers.keys():
            # Look for exact match
            if markerkey == label:
                markerstyle = pretty_markers[markerkey]
            # Look for partial match
            if markerstyle is None:
                if markerkey in label:
                    markerstyle = pretty_markers[markerkey]
        if markerstyle is None:
            logging.debug("I do not have a marker for your label %s. "
                          "Returning standard."%label)
            markerstyle = 'x'
        return markerstyle
