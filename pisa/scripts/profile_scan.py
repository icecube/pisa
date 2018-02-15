#!/usr/bin/env python

"""
Profile scan
"""


from __future__ import absolute_import

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from os.path import expanduser, expandvars, isfile

# Import numpy and define np=numpy to allow `eval` to work with either
import numpy

#from pisa.analysis.analysis import Analysis
from pisa import ureg
from pisa.analysis.hypo_testing import HypoTesting, setup_makers_from_pipelines,\
                                       collect_maker_selections, select_maker_params
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.fileio import from_file, to_file
from pisa.utils.log import logging, set_verbosity


__all__ = ['profile_scan', 'parse_args', 'main']

__author__ = 'T. Ehrhardt'

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


np = numpy # pylint: disable=invalid-name


def profile_scan(return_outputs=False):
    """Load the HypoTesting class and use it to do an Asimov test across the
    space of some hypo parameters.

    The user will define the parameter and pass a numpy-interpretable string to
    set the range of values. For example, one could scan over the space of
    theta23 by using a string such as `"np.linspace(0.35, 0.65, 31)*ureg.rad"`.
    """
    # NOTE: import here to avoid circular refs
    from pisa.scripts.analysis import parse_args
    init_args_d = parse_args(description=profile_scan.__doc__,
                             command=profile_scan)

    # only care about h0_maker and data_maker
    setup_makers_from_pipelines(init_args_d=init_args_d, ref_maker_names=['h0', 'data'])

    # process param selections for each of h0 and data
    collect_maker_selections(init_args_d=init_args_d, maker_names=['h0', 'data'])

    # so HypoTesting won't be unhappy, even though we don't care about h1
    init_args_d['h1_maker'] = init_args_d['h0_maker']

    # apply param selections to h0 and data distribution makers
    select_maker_params(init_args_d=init_args_d, maker_names=['h0', 'data'])

    # Remove final parameters that don't want to be passed to HypoTesting
    param_names = init_args_d.pop('param_name')
    scan_vals_lists = init_args_d.pop('scan_vals')
    scan_vals = [eval(scan_vals_list) for scan_vals_list in scan_vals_lists]
    outer = not init_args_d.pop('no_outer')
    profile = not init_args_d.pop('no_profile')
    store_intermediate = init_args_d.pop('store_intermediate')

    hypo_testing = HypoTesting(**init_args_d)

    scan_res = hypo_testing.hypo_scan(
        param_names = param_names,
        values = scan_vals,
        outer = outer,
        profile = profile
    )
