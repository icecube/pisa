"""
Performe injected parameter scan based on command line args. Ment to be called
from `pisa.scripts.analysis` as a subcommand.
"""


from __future__ import absolute_import, division

# Import numpy and define np=numpy to allow `eval` to work with either
import numpy

from pisa import ureg
from pisa.analysis.hypo_testing import HypoTesting
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.prior import Prior
from pisa.utils.log import logging
from pisa.utils.scripting import normcheckpath


__all__ = ['inj_param_scan']

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


np = numpy # pylint: disable=invalid-name


def inj_param_scan(return_outputs=False):
    """Load the HypoTesting class and use it to do an Asimov test across the
    space of one of the injected parameters.

    The user will define the parameter and pass a numpy-interpretable string to
    set the range of values. For example, one could scan over the space of
    theta23 by using a string such as `"numpy.linspace(0.35, 0.65, 31)"` which
    will then be evaluated to figure out a space of theta23 to inject and run
    Asimov tests.
    """
    # NOTE: import here to avoid circular refs
    from pisa.scripts.analysis import parse_args
    init_args_d = parse_args(description=inj_param_scan.__doc__,
                             command=inj_param_scan)

    # Normalize and convert `*_pipeline` filenames; store to `*_maker`
    # (which is argument naming convention that HypoTesting init accepts).
    # For this test, pipeline is required so we don't need the try arguments
    # or the checks on it being None
    filenames = init_args_d.pop('pipeline')
    filenames = sorted(
        [normcheckpath(fname) for fname in filenames]
    )
    init_args_d['h0_maker'] = filenames
    # However, we do need them for the selections, since they can be different
    for maker in ['h0', 'h1', 'data']:
        ps_name = maker + '_param_selections'
        ps_str = init_args_d[ps_name]
        if ps_str is None:
            ps_list = None
        else:
            ps_list = [x.strip().lower() for x in ps_str.split(',')]
        init_args_d[ps_name] = ps_list

    init_args_d['data_maker'] = init_args_d['h0_maker']
    init_args_d['h1_maker'] = init_args_d['h0_maker']
    init_args_d['h0_maker'] = DistributionMaker(init_args_d['h0_maker'])
    init_args_d['h1_maker'] = DistributionMaker(init_args_d['h1_maker'])
    init_args_d['h1_maker'].select_params(init_args_d['h1_param_selections'])
    init_args_d['data_maker'] = DistributionMaker(init_args_d['data_maker'])
    if init_args_d['data_param_selections'] is None:
        init_args_d['data_param_selections'] = \
            init_args_d['h0_param_selections']
        init_args_d['data_name'] = init_args_d['h0_name']
    init_args_d['data_maker'].select_params(
        init_args_d['data_param_selections']
    )

    # Remove final parameters that don't want to be passed to HypoTesting
    param_name = init_args_d.pop('param_name')
    inj_vals = eval(init_args_d.pop('inj_vals'))
    inj_units = init_args_d.pop('inj_units')
    force_prior = init_args_d.pop('use_inj_prior')

    # Instantiate the analysis object
    hypo_testing = HypoTesting(**init_args_d)

    logging.info(
        'Scanning over %s between %.4f and %.4f with %i vals',
        param_name, min(inj_vals), max(inj_vals), len(inj_vals)
    )
    # Modify parameters if necessary
    if param_name == 'sin2theta23':
        requested_vals = inj_vals
        inj_vals = np.arcsin(np.sqrt(inj_vals))
        logging.info(
            'Converting to theta23 values. Equivalent range is %.4f to %.4f'
            ' radians, or %.4f to %.4f degrees',
            min(inj_vals), max(inj_vals),
            min(inj_vals)*180/np.pi, max(inj_vals)*180/np.pi
        )
        test_name = 'theta23'
        inj_units = 'radians'

    elif param_name == 'deltam31':
        raise ValueError('Need to implement a test where it ensures the sign '
                         'of the requested values matches those in truth and '
                         'the hypo makers (else it makes no sense). For now, '
                         'please select deltam3l instead.')

    elif param_name == 'deltam3l':
        # Ensure all values are the same sign, else it doesn't make any sense
        if not np.alltrue(np.sign(inj_vals)):
            raise ValueError("Not all requested values to inject are the same "
                             "sign. This doesn't make any sense given that you"
                             " have requested to inject different values of "
                             "deltam3l.")
        logging.info('Parameter requested was deltam3l - will convert assuming'
                     ' that this is always the largest of the two splittings '
                     'i.e. deltam3l = deltam31 for deltam3l > 0 and deltam3l '
                     '= deltam32 for deltam3l < 0.')
        inj_sign = np.sign(inj_vals)[0]
        requested_vals = inj_vals
        test_name = 'deltam31'
        deltam21_val = hypo_testing.data_maker.params['deltam21'].value.to(
            inj_units
        ).magnitude
        if inj_sign == 1:
            no_inj_vals = requested_vals
            io_inj_vals = (requested_vals - deltam21_val) * -1.0
        else:
            io_inj_vals = requested_vals
            no_inj_vals = (requested_vals * -1.0) + deltam21_val
        inj_vals = []
        for no_inj_val, io_inj_val in zip(no_inj_vals, io_inj_vals):
            o_vals = {}
            o_vals['nh'] = no_inj_val
            o_vals['ih'] = io_inj_val
            inj_vals.append(o_vals)

    else:
        test_name = param_name
        requested_vals = inj_vals

    unit_inj_vals = []
    for inj_val in inj_vals:
        if isinstance(inj_val, dict):
            o_vals = {}
            for ivkey in inj_val.keys():
                o_vals[ivkey] = inj_val[ivkey]*ureg(inj_units)
            unit_inj_vals.append(o_vals)
        else:
            unit_inj_vals.append(inj_val*ureg(inj_units))
    inj_vals = unit_inj_vals

    # Extend the ranges of the distribution makers so that they reflect the
    # range of the scan. This is a pain if there are different values depending
    # on the ordering. Need to extend the ranges of both values in the
    # hypothesis maker since the hypotheses may minimise over the ordering,
    # and could then go out of range.

    # Also, some parameters CANNOT go negative or else things won't work.
    # To account for this, check if parameters lower value was positive and,
    # if so, enforce that it is positive now.
    if isinstance(inj_vals[0], dict):
        # Calculate ranges for both parameters
        norangediff = max(no_inj_vals) - min(no_inj_vals)
        norangediff = norangediff*ureg(inj_units)
        norangetuple = (min(no_inj_vals)*ureg(inj_units) - 0.5*norangediff,
                        max(no_inj_vals)*ureg(inj_units) + 0.5*norangediff)
        iorangediff = max(io_inj_vals) - min(io_inj_vals)
        iorangediff = iorangediff*ureg(inj_units)
        iorangetuple = (min(io_inj_vals)*ureg(inj_units) - 0.5*iorangediff,
                        max(io_inj_vals)*ureg(inj_units) + 0.5*iorangediff)
        # Do it for both hierarchies
        for hierarchy, rangetuple in zip(['nh', 'ih'],
                                         [norangetuple, iorangetuple]):
            hypo_testing.set_param_ranges(
                selection=hierarchy,
                test_name=test_name,
                rangetuple=rangetuple,
                inj_units=inj_units
            )
        # Select the proper params again
        hypo_testing.h0_maker.select_params(init_args_d['h0_param_selections'])
        hypo_testing.h1_maker.select_params(init_args_d['h1_param_selections'])
    # Otherwise it's way simpler...
    else:
        rangediff = max(inj_vals) - min(inj_vals)
        rangetuple = (min(inj_vals) - 0.5*rangediff,
                      max(inj_vals) + 0.5*rangediff)
        hypo_testing.set_param_ranges(
            selection=None,
            test_name=test_name,
            rangetuple=rangetuple,
            inj_units=inj_units
        )

    if hypo_testing.data_maker.params[test_name].prior is not None:
        if hypo_testing.data_maker.params[test_name].prior.kind != 'uniform':
            if force_prior:
                logging.warning(
                    'Parameter to be scanned, %s, has a %s prior that you have'
                    ' requested to be left on. This will likely make the'
                    ' results wrong.',
                    test_name,
                    hypo_testing.data_maker.params[test_name].prior.kind
                )
            else:
                logging.info(
                    'Parameter to be scanned, %s, has a %s prior.This will be'
                    ' changed to a uniform prior (i.e. no prior) for this'
                    ' test.',
                    test_name,
                    hypo_testing.data_maker.params[test_name].prior.kind
                )
                uniformprior = Prior(kind='uniform')
                hypo_testing.h0_maker.params[test_name].prior = uniformprior
                hypo_testing.h1_maker.params[test_name].prior = uniformprior
    else:
        if force_prior:
            raise ValueError('Parameter to be scanned, %s, does not have a'
                             ' prior but you have requested to force one to be'
                             ' left on. Something is potentially wrong.'
                             % test_name)
        else:
            logging.info('Parameter to be scanned, %s, does not have a prior.'
                         ' So nothing needs to be done.', test_name)

    # Everything is set up. Now do the scan.
    outputs = hypo_testing.asimov_inj_param_scan( # pylint: disable=redefined-outer-name
        param_name=param_name,
        test_name=test_name,
        inj_vals=inj_vals,
        requested_vals=requested_vals,
        h0_name=init_args_d['h0_name'],
        h1_name=init_args_d['h1_name'],
        data_name=init_args_d['data_name']
    )

    if return_outputs:
        return outputs
