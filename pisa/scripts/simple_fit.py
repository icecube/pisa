#!/usr/bin/env python

"""
A simple and bare, no-nonsense fitting script.
"""


from __future__ import absolute_import

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
from copy import deepcopy
from os.path import expanduser, expandvars, isfile

# Import numpy and define np=numpy to allow `eval` to work with either
import numpy

from pisa import ureg
from pisa.analysis.analysis import Analysis
from pisa.analysis.hypo_testing import setup_makers_from_pipelines
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.fileio import from_file, to_file


__all__ = ['simple_fit', 'parse_args', 'main']

__author__ = 'T. Ehrhardt'

__license__ = '''Copyright (c) 2014-2018, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


np = numpy # pylint: disable=invalid-name
units = ureg # pylint: disable=invalid-name


def simple_fit(init_args_d, return_outputs=False):
    """Load the analysis class and use it to do a simple fit of some
    some hypo parameters to pseudo(sic!)-data
    """

    # only care about h0_maker and data_maker
    setup_makers_from_pipelines(init_args_d=init_args_d,ref_maker_names=['h0', 'data'])

    data_dist = init_args_d['data_maker'].get_outputs(return_sum=True)
    hypo_maker = init_args_d['h0_maker']
    metric = init_args_d['metric']
    other_metrics = init_args_d.get('other_metrics', None)
    hypo_param_selections = None
    fit_settings = init_args_d['fit_settings']
    outfile = init_args_d['outfile']

    analysis = Analysis()

    fit_res = analysis.fit_hypo(
        data_dist=data_dist,
        hypo_maker=hypo_maker,
        hypo_param_selections=hypo_param_selections,
        metric=metric,
        other_metrics=other_metrics,
        fit_settings=fit_settings
    )[0][0]


    serialize = ['metric', 'metric_val', 'params', 'fit_time',
                 'detailed_metric_info', 'fit_metadata', #'fit_history',
                 'num_distributions_generated', 'hypo_asimov_dist',
                 ]

    fit_info = OrderedDict()
    for k, v in fit_res.iteritems():
        if k not in serialize:
            continue
        if k == 'params':
            d = OrderedDict()
            for param in v: # record *all* hypo parameter values
                d[param.name] = str(param.value)
            v = d
        if k == 'fit_metadata':
            for k2 in ['hess_inv']:
                if k2 in v:
                    try:
                        v[k2] = v[k2].todense()
                    except AttributeError:
                        v[k2] = v[k2]
        if isinstance(v, ureg.Quantity):
            v = str(v)
        fit_info[k] = v

    if outfile:
        to_file(fit_info, outfile)

    if return_outputs:
        return fit_res
