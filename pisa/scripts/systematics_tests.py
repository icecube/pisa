"""
Peform systematics tests based on command line args. Meant to be called from
`pisa.scripts.analysis` as a subcommand
"""


from __future__ import absolute_import, division

from pisa.analysis.hypo_testing import HypoTesting, setup_makers_from_pipelines,\
                                       collect_maker_selections, select_maker_params
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.log import logging


__all__ = ['systematics_tests']

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


def systematics_tests(init_args_d, return_outputs=False):
    """Load the HypoTesting class and use it to do a systematic study
    in Asimov.

    This will take some input pipeline configuration and then turn each one of
    the systematics off in turn, doing a new hypothesis test each time. The
    user will have the option to fix this systematic to either the baseline or
    some shifted value (+/- 1 sigma, or appropriate). One also has the ability
    in the case of the latter to still fit with this systematically incorrect
    hypothesis.
    """

    # NOTE: Removing extraneous args that won't get passed to instantiate the
    # HypoTesting object via dictionary's `pop()` method.
    inject_wrong = init_args_d.pop('inject_wrong')
    fit_wrong = init_args_d.pop('fit_wrong')
    only_syst = init_args_d.pop('only_syst')
    do_baseline = not init_args_d.pop('skip_baseline')
    if fit_wrong:
        if not inject_wrong:
            raise ValueError('You have specified to fit the systematically'
                             ' wrong hypothesis but have not specified to'
                             ' actually generate a systematically wrong'
                             ' hypothesis. If you want to flag "fit_wrong"'
                             ' please also flag "inject_wrong"')
        else:
            logging.info('Injecting a systematically wrong hypothesis while'
                         ' also allowing the minimiser to attempt to correct'
                         ' for it.')
    else:
        if inject_wrong:
            logging.info('Injecting a systematically wrong hypothesis but'
                         ' NOT allowing the minimiser to attempt to correct'
                         ' for it. Hypothesis maker will be FIXED at the'
                         ' baseline value.')
        else:
            logging.info('A standard N-1 test will be performed where each'
                         ' systematic is fixed to the baseline value'
                         ' one-by-one.')

    # only have a single distribution maker, the h0_maker
    # first set up all distribution makers the same
    setup_makers_from_pipelines(init_args_d=init_args_d, ref_maker_names=['h0'])

    # process param selections for each of h0, h1, and data
    collect_maker_selections(init_args_d=init_args_d, maker_names=['h0', 'h1', 'data'])

    # apply h0 selections to data if data doesn't have selections defined
    if init_args_d['data_param_selections'] is None:
        init_args_d['data_param_selections'] = \
            init_args_d['h0_param_selections']
        init_args_d['data_name'] = init_args_d['h0_name']

    if init_args_d['h1_param_selections'] is None or init_args_d['h1_param_selections'] == init_args_d['h0_param_selections']:
        # this will mean hypothesis testing will only work
        # with a single hypothesis
        init_args_d['h1_maker'] = None
        # just to be clear
        init_args_d['h1_name'] = init_args_d['h0_name']

    # apply param selections to data distribution maker if applicable
    select_maker_params(init_args_d=init_args_d, maker_names=['data'])

    if only_syst is not None:
        for syst in only_syst:
            if syst not in init_args_d['h0_maker'].params.free.names:
                raise ValueError(
                    'Systematic test requested to be performed on systematic'
                    ' %s but it does not appear in the free parameters of the'
                    ' pipeline passed to the script - %s.'
                    % (syst, init_args_d['h0_maker'].params.free.names)
                )
        logging.info(
            'Performing chosen systematic test on just the following'
            ' systematics - %s.', only_syst
        )

    # Instantiate the analysis object
    hypo_testing = HypoTesting(**init_args_d)
    # Everything is set up so do the tests
    outputs = hypo_testing.asimov_syst_tests( # pylint: disable=redefined-outer-name
        inject_wrong=inject_wrong,
        fit_wrong=fit_wrong,
        only_syst=only_syst,
        do_baseline=do_baseline,
        h0_name=init_args_d['h0_name'],
        h1_name=init_args_d['h1_name'],
        data_name=init_args_d['data_name']
    )

    if return_outputs:
        return outputs
