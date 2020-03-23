"""
Common tools related to (no specific method of) fitting.
"""

from __future__ import absolute_import

import ast
from collections.abc import Mapping, Sequence
from copy import deepcopy
import numpy as np
from pisa import Q_
from pisa.utils.config_parser import parse_fit_config, parse_quantity
from pisa.utils.log import logging

# TODO: get rid of scanning method
def apply_fit_settings(fit_settings, free_params):
    """Validate fit settings (cf. `config_parser.parse_fit_config`) against
    a `DistributionMaker`'s set of free parameters. Ensure that params,
    ranges, test points, seeds etc. are compatible with free parameter specs.
    Returns a modified fit_settings dict.
    Parameters
    ----------
    fit_settings : str, dict, or ConfigParser
        Either path to fit cfg file or dictionary of fit settings as generated
        by `config_parser.parse_fit_config`
    free_params : ParamSet
        free parameters to which the fit settings are to be applied
    """
    if not isinstance(fit_settings, dict):
        fit_settings = parse_fit_config(fit_settings)
    processed_fit_settings = deepcopy(fit_settings)
    fit_methods = fit_settings.keys()
    # TODO: don't require all possible methods to be present
    # assert set(fit_methods) == set(ANALYSIS_METHODS)

    # wildcard can only occur once in fit_settings; all parameters not
    # specified will be treated by the method which has the wildcard
    wildcard = '*'

    method_with_wildcard = [
        fit_method for fit_method in fit_methods
        if wildcard in fit_settings[fit_method]['params']
    ]
    assert len(method_with_wildcard) <= 1

    params_with_fit_method = [
        pname for fit_method in fit_methods
        for pname in fit_settings[fit_method]['params'].keys()
        if pname != wildcard
    ]

    for pname in params_with_fit_method:
        # require to be a free parameter
        if not pname in free_params.names:
            raise ValueError(
                'Parameter "%s" present in fit settings but not among free'
                ' parameters "%s". Please ensure consistency.'
                % (pname, free_params.names)
            )

    # remaining
    params_remaining = [pname for pname in free_params.names
                        if pname not in params_with_fit_method]

    if method_with_wildcard:
        method_with_wildcard = method_with_wildcard[0]
        processed_fit_settings[method_with_wildcard]['params'].pop(wildcard)
        # these need the fit method settings defaults
        defaults = processed_fit_settings[method_with_wildcard].pop('defaults')
        processed_fit_settings[method_with_wildcard]['params'].update(
            {premain: defaults for premain in params_remaining}
        )
    elif params_remaining:
        raise ValueError(
            'Cannot tell how to fit the free parameters %s. Please provide'
            ' these in the fit settings or use the wildcard "%s".'
            % (params_remaining, wildcard)
        )

    # remove any 'default' entry (shouldn't be here in the first place)
    # for the other fit methods which don't make use of wildcard
    for fit_method in fit_methods:
        processed_fit_settings[fit_method].pop('defaults', None)

    # make the scan/pull values for each parameter
    # (can apply identical treatment)
    # These are constructed from a range and an nvalues integer, where
    # the range can be given as "nominal+/-<scale>*nominal.
    # TODO: compare fit ranges to free_params ranges, precedence?
    for fit_method in ['pull', 'scan']:
        new_values_d = {'params': [], 'values': []}
        if not fit_method in processed_fit_settings:
            processed_fit_settings[fit_method] = new_values_d
            continue
        for pname, sett_d in processed_fit_settings[fit_method]['params'].items():
            # keys correspond to individual parameter fit settings
            found_keys = set(sett_d.keys())
            # a certain combination of keys/options is allowed
            if fit_method == 'scan':
                valid_keys = [set(('nvalues', 'range')), set(('values',))]
            else:
                valid_keys = [set(('lin_range',))]
            # one of the combinations must be an exact hit
            if not found_keys in valid_keys:
                raise KeyError(
                    'Only recognised key (combinations) for %s method: %s. You'
                    ' tried to set %s for parameter "%s".'
                    % (fit_method, valid_keys, found_keys, pname)
                )
            # interpret and process the fields which we have depending on the
            # fit method in question
            if fit_method == 'scan':
                try:
                    nvals = int(sett_d['nvalues'])
                    prange = sett_d['range']
                except:
                    nvals, prange = None, None
                    values = sett_d['values']
            else:
                # pull method
                nvals = 2
                prange = sett_d['lin_range']

            # record the units of the target parameter and do consistency checks
            target_units = free_params[pname].units
            if prange is None:
                # this means we must have the case of scanning and scan values
                # being specified directly
                values = eval(values)
                if not isinstance(values, Q_):
                    raise TypeError(
                        'Please specify scan values for param "%s" with units'
                        ' (convertible to: "%s").' % (pname, target_units)
                    )
                try:
                    values.ito(target_units)
                except:
                    logging.error(
                        'The units ("%s") specified for parameter "%s" are not'
                        ' compatible with those ("%s") of the corresponding'
                        ' parameter in the `ParamSet` of free parameters.'
                        % (values.units, pname, target_units)
                    )
                    raise
            else:
                if fit_method == 'pull':
                    prange = eval(prange)
                prange_parsed = False
                # need to convert from range and nvalues to linearly spaced
                # values themselves
                if isinstance(prange, str):
                    if '+/-' in prange:
                        search_start = prange.find('+/-')+len('+/-')
                        # two cases:
                        # 1. range around nominal
                        # 2. range around some arbitrary value
                        if prange.startswith('nominal'):
                            # 1
                            # two cases:
                            # 1. range as fraction of nominal around nominal
                            # 2. absolute range around nominal
                            if prange.endswith('nominal'):
                                # 1
                                scale_nom = float(
                                    prange[search_start:prange.find('*')]
                                )
                                half_range = scale_nom * free_params[pname].nominal_value
                            else:
                                # 2
                                half_range = prange[search_start:]
                                try:
                                    half_range = parse_quantity(half_range).nominal_value
                                except:
                                    logging.error(
                                        'Parameter "%s": could not parse "%s"'
                                        ' into a pint/uncertainty quantity.'
                                        % (pname, half_range)
                                    )
                                    raise
                        else:
                            # 2
                            raise NotImplementedError(
                                'Parameter "%s": Range spec needs to be of'
                                ' format "nominal+/-..."' % pname
                            )
                        nom = free_params[pname].nominal_value
                        prange = [nom - half_range, nom + half_range]
                        prange_parsed = True
                    else:
                        prange = ast.literal_eval(prange)
                if isinstance(prange, Sequence) and not prange_parsed:
                    #if not isinstance(prange, Sequence):
                    #    raise TypeError(
                    #        'Range specified for parameter "%s" is not'
                    #        ' a sequence but of "%s".' % (pname, type(prange))
                    #    )
                    if not len(prange) == 2:
                        raise ValueError(
                            'Range "%s" specified for parameter "%s" is not'
                            ' of length 2.' % (prange, pname)
                        )
                    for val in prange:
                        val = Q_(val)
                        """
                        try:
                            # FIXME: find a way to convert from string to list
                            # when units are present, don't just assume
                            # the vals are given in the target units
                            val = val * target_units
                            #val.ito(target_units)
                        except:
                            logging.error(
                                'The units ("%s") specified for parameter "%s" are'
                                ' not compatible with those ("%s") of the'
                                ' corresponding parameter in the `ParamSet` of free'
                                ' parameters.' % (val.units, pname, target_units)
                            )
                            raise
                        """
                    prange_parsed = True
                if not prange_parsed:
                    raise TypeError(
                        'Range "%s" specified for parameter "%s" is of "%s" which'
                        ' is unhandled.' % (prange, pname, type(prange))
                    )
                values = np.linspace(prange[0], prange[1], nvals) * target_units

            new_values_d['params'].append(pname)
            new_values_d['values'].append(values)

        # overwrite entry with the new dict
        processed_fit_settings[fit_method] = new_values_d

    new_minimize_settings_d = {'global': None, 'local': None, 'params': []}
    if 'minimize' in processed_fit_settings:
        minimize_settings = processed_fit_settings['minimize']
        # all we do here for now is move 'global' and 'local' minimizer cfg
        # out from under one param to its own entry (assuming all minimizer cfg's
        # are identical) and make a simple list of params
        # TODO: could also allow for min. ranges to be specified in fit settings,
        # or for more complex things such as seeds
        for pname, sett_d in minimize_settings['params'].items():
            for opt in ['local', 'global']:
                # ensure necessary condition for this being parsed min. cfg
                assert isinstance(sett_d[opt], Mapping) or sett_d[opt] is None
                # also ensure all are identical
                if new_minimize_settings_d[opt]:
                    assert sett_d[opt] == new_minimize_settings_d[opt]
                new_minimize_settings_d[opt] = sett_d[opt]
            new_minimize_settings_d['params'].append(pname)

    processed_fit_settings['minimize'] = new_minimize_settings_d

    return processed_fit_settings
