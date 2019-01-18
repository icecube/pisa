#!/usr/bin/env python

"""
Parameter scan
"""

from __future__ import absolute_import

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os.path import expanduser, expandvars, isfile
import sys

import numpy as np

from pisa import ureg
from pisa.analysis.analysis import Analysis
from pisa.core.detectors import Detectors
from pisa.utils.fileio import from_file, to_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.stats import ALL_METRICS


__all__ = ['parameter_scan', 'parse_args', 'main']

__author__ = 'J.Weldert'

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


def parameter_scan(data_settings, template_settings, param_name, steps, Min, Max,
                   unit, data_param_selection,hypo_param_selection, profile, outfile,
                   minimizer_settings, metric, change_data, shared_params):
    """Perform a profile scan.

    Parameters
    ----------
    data_settings
    template_settings
    param_name
    steps
    Min
    Max
    unit
    data_param_selection
    hypo_param_selection
    profile
    outfile
    minimizer_settings
    metric
    shared_params
    change_data

    Returns
    -------
    results
    analysis
    """
    
    outfile = expanduser(expandvars(outfile))
    if isfile(outfile):
        raise IOError('`outfile` "{}" already exists!'.format(outfile))
        
    if Min > Max: 
        raise ValueError('Min > Max')
        
    if shared_params == None: shared_params = []
    if param_name in shared_params and not change_data:
        raise AttributeError(param_name + ' will be fixed, so don`t specify' 
                             ' it as shared parameter.')

    if minimizer_settings != None:
        minimizer_settings = from_file(minimizer_settings)
    if minimizer_settings == None and profile:
        raise AttributeError('Need minimizer-settings for profile scan.')

    hypo_maker = Detectors(template_settings,shared_params=shared_params)
    hypo_maker.select_params(hypo_param_selection)

    if data_settings is None:
        data_maker = Detectors(template_settings,shared_params=shared_params)
    else:
        data_maker = Detectors(data_settings,shared_params=shared_params)
    data_maker.select_params(data_param_selection)

    if data_maker.det_names != hypo_maker.det_names:
        raise AttributeError('Different detectors (sequence) in data and hypo.')
    
    param_values = np.linspace(Min,Max,steps)
    if unit == None: unit = data_maker.params[param_name].u

    if change_data:
        reset_free = False
    else:
        reset_free = True
    
    analysis = Analysis()
    results = []
    for pv in param_values:
        logging.info('Working on ' + param_name + ' = %s', pv)
        
        for i, dm in enumerate(hypo_maker._distribution_makers):
            if param_name in dm.params.names:
                para = dm.params
                if param_name == 'deltam31' and change_data:
                    para[param_name].value = -pv * unit
                else:
                    para[param_name].value = pv * unit
                if not change_data: para.fix(param_name)
                dm.update_params(para)

        for i, dm in enumerate(data_maker._distribution_makers):
            if param_name in dm.params.names and change_data:
                para = dm.params
                para[param_name].value = pv * unit
                dm.update_params(para)

        data = data_maker.get_outputs(return_sum=True)
        
        if profile:
            try:
                result = analysis.fit_hypo(
                                    data_dist=data,
                                    hypo_maker=hypo_maker,
                                    hypo_param_selections=hypo_param_selection,
                                    metric=metric,
                                    minimizer_settings=minimizer_settings,
                                    reset_free=reset_free,
                                    pprint=False
                                    )
                result.pop('fit_history')
                result['minimizer_metadata'].pop('hess_inv', None)
                results.append(result)
            except:
                e = str(sys.exc_info()[1])
                message = 'Error while working on ' + param_name + ' = ' + str(pv)
                logging.warning(message)
                logging.warning(e)
                results.append([message,e])

        else:
            hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)
            results.append(analysis.nofit_hypo(
                                data_dist=data,
                                hypo_maker=hypo_maker,
                                hypo_param_selections=hypo_param_selection,
                                hypo_asimov_dist=hypo_asimov_dist,
                                metric=metric,
                                )
                            )

    to_file(results, outfile)    
    logging.info("Done.")

    return results, analysis


def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data-settings', type=str,
        metavar='CONFIGFILE', default=None,
        help='''Settings for the generation of "data" distributions; repeat
        this argument to specify multiple pipelines. If omitted, the same
        settings as specified for --template-settings are used to generate data
        distributions.'''
    )
    parser.add_argument(
        '--template-settings',
        metavar='CONFIGFILE', required=True, action='append',
        help='''Settings for generating template distributions; repeat
        this option to define multiple pipelines.'''
    )
    parser.add_argument(
        '--shared-params', type=str, default=None,
        action='append',
        help='''Shared parameters for multi det analysis (repeat for multiple).'''
    )
    parser.add_argument(
        '--param-name', type=str, required=True,
        help='''Name of the parameter to be scanned.'''
    )
    parser.add_argument(
        '--steps', type=int, required=True,
        help='''Provide a number of steps for the parameter.'''
    )
    parser.add_argument(
        '--min', type=float, required=True,
        help='''Provide a min value.'''
    )
    parser.add_argument(
        '--max', type=float, required=True,
        help='''Provide a max value.'''
    )
    parser.add_argument(
        '--unit', type=str, default=None,
        help='''Provide a unit.'''
    )
    parser.add_argument(
        '--data-param-selection', type=str, required=True,
        help='''Selection of params to use in order to generate the data
        distributions.'''
    )
    parser.add_argument(
        '--hypo-param-selection', type=str, required=True,
        help='''Selection of params to use in order to generate the
        hypothesised Asimov distributions.'''
    )
    parser.add_argument(
        '--profile', action='store_true',
        help='''Run profile scan, i.e. optimise over remaining free
        parameters.'''
    )
    parser.add_argument(
        '--outfile', metavar='FILE',
        type=str, action='store', default='param_scan.json',
        help='file to store the output'
    )
    parser.add_argument(
        '--minimizer-settings', type=str,
        metavar='JSONFILE', default=None,
        help='''Settings related to the minimizer used in the LLR analysis.'''
    )
    parser.add_argument(
        '--metric',
        type=str, required=True, metavar='METRIC', action='append',
        choices=sorted(ALL_METRICS),
        help='''Name of metric(s) to use for optimizing the fit. Must be one of
        %s. Repeat this argument if you want to use different metrics for
        different detectors. If only one metric is specified, all detectors will
        use the same. Otherwise you have to specify one metric for each detector
        (even if two use the same) and pay attention to the order.'''
        % (ALL_METRICS,)
    )
    parser.add_argument(
        '--change-data', action='store_true',
        help='''Set for parameter scan in true hierarchy (data).'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    args = parser.parse_args()
    kwargs = vars(args)
    set_verbosity(kwargs.pop('v'))

    return kwargs


def main():
    """Run livetime_scan with args from command line"""
    return parameter_scan(**parse_args())


if __name__ == '__main__':
    results, analysis = main()
