# author: P.Eller
#         pde3+pisa@psu.edu
#
# date:   2016-04-28
"""
Parse a ConfigFile object into a dict containing an item for every analysis
stage, that itself contains all necessary instantiation arguments/objects for
that stage. for en example config file, please consider
`$PISA/pisa/resources/settings/pipeline/example.cfg`

Config File Structure:
===============

The config file is expected to contain the following sections::
    #include file_x.cfg
    #include file_y.cfg

    [pipeline]
    order = stageA:serviceA, stageB:serviceB

    [binning]
    binning1.order = axis1, axis2
    binning1.axis1 = {'num_bins':40, 'is_log':True,
                      'domain':[1,80] * units.GeV, 'tex': r'A_1'}
    binning1.axis2 = {'num_bins':10, 'is_lin':True,
                      'domain':[1,5], 'tex': r'A_2'}

    [stage:stageA]
    input_binning = bining1
    output_binning = binning1
    error_method = None
    debug_mode = False

    param.p1 = 0.0 +/- 0.5 * units.deg
    param.p1.fixed = False
    param.p1.range = nominal + [-2.0, +2.0] * sigma

    [stage:stageB]
    ...

* `#include` statements can be used to include other cfg files; these must be
    the first line(s) of the file.

* `pipeline` is the top most section that defines the hierarchy of stages and
    what services to be instantiated.

* `binning` can contain different binning definitions, that are then later
    referred to from within the stage sections.

* `stage` one such section per stage:service is necessary. It contains some
    options that are common for all stages (`binning`, `error_method` and
    `debug_mode`) as well as all the necessary arguments and parameters for a
    given stage.


Param definitions:
------------------

Every key in a stage section that starts with `param.name` is interpreted and
parsed into a PISA param object. These can be strings (e.g. a filename - don't
use any quotation marks) or quantities. The later case expects an expression
that can be converted by the `parse_quantity` function. The `+/-` notation will
be interpreted as a gaussian prior for the quantity. Units can be added by `*
unit.soandso`.

Additional arguments to a parameter are passed in with the `.` notation, for
example `param.name.fixed = False`, which makes it a free parameter in the fit
(by default a parameter is fixed unless specified like this).

A range must be given for a free parameter. Either as absolute range `[x,y]` or
in conjunction with the keywords `nominal` (= nominal parameter value) and
`sigma` if the param was specified with the `+/-` notation.

`.prior` is another argument, that can take the values `uniform` or `spline`,
for the latter case a `.prior.data` will be expected, pointing to the spline
data file.

N.B.:
+++++
Params that have the same name in multiple stages of the pipeline are
instantiated as references to a single param in memory, so updating one updates
all of them.

Note that this mechanism of synchronizing parameters holds only within the
scope of a single pipeline; synchronization of parameters across pipelines is
done by adding the pipelines to a single DistributionMaker object and updating
params through the DistributionMaker's update_params method.

If you DO NOT want parameters to be synchronized, provide a unique_id for them.
This is imply done by setting `.unique_id`

Param selector:
---------------

A special mechanism allows the user to specify multiple, different values for
the same param via the param selector method. This can be used for example for
hypothesis testing, there for hypothesis A a param takes a certain value, while
for hypothesis B a different value.

A given param, say `foo`, then needs two definitions like the following,
assuming we name our selections `A` and `B`:

param.A.foo = 1
param.B.foo = 2

The default param selector needs to be spcified under section `pipeline` as e.g.

param_selections = A

Which will default the value of 1 for param `foo`. An instatiated pipeline can
dynamically switch to another selection after instantiation.

Multiple different param selectors are allowed in a single config. In the
default selection they must be separated by commas.

"""


# TODO: Make interoperable with pisa.utils.resources. I.e., able to work with
# Python package resources, not just filesystem files.
# TODO: Docstrings, Philipp!
# TODO: add try: except: blocks around class instantiation calls to give
# maximally useful error info to the user (spit out a good message, but then
# re-raise the exception)


from __future__ import absolute_import, division

from collections import OrderedDict
from ConfigParser import ConfigParser, SafeConfigParser
import re

import numpy as np
from uncertainties import ufloat, ufloat_fromstr

from pisa import ureg

from pisa.utils.fileio import from_file
from pisa.utils.log import logging, set_verbosity


__all__ = ['PARAM_RE', 'PARAM_ATTRS', 'units',
           'parse_quantity', 'parse_string_literal', 'split',
           'interpret_param_subfields', 'parse_param', 'parse_pipeline_config',
           'BetterConfigParser']


PARAM_RE = re.compile(
    r'^param\.(?P<subfields>(([^.\s]+)(\.|$))+)',
    re.IGNORECASE
)

PARAM_ATTRS = ['range', 'prior', 'fixed']

# Define names that users can specify in configs such that the eval of those
# strings works.
numpy = np # pylint: disable=invalid-name
inf = np.inf # pylint: disable=invalid-name
units = ureg # pylint: disable=invalid-name


def parse_quantity(string):
    """Parse a string into a pint/uncertainty quantity.

    Parameters
    ----------
    string : string

    Examples
    --------
    >>> print parse_quantity('1.2 +/- 0.7 * units.meter')
    TODO

    """
    value = string.replace(' ', '')
    if 'units.' in value:
        value, unit = value.split('units.')
    else:
        unit = None
    value = value.rstrip('*')
    if '+/-' in value:
        value = ufloat_fromstr(value)
    else:
        value = ufloat(float(value), 0)
    value *= ureg(unit)
    return value


def parse_string_literal(string):
    """Evaluate a string with certain special values, or return the string. Any
    further parsing must be done outside this module, as this is as specialized
    as we're willing to be in assuming/interpreting what a string is supposed
    to mean.

    Parameters
    ----------
    string : string

    Returns
    -------
    bool, None, or str

    Examples
    --------
    >>> print parse_string_literal('true')
    True

    >>> print parse_string_literal('False')
    False

    >>> print parse_string_literal('none')
    None

    >>> print parse_string_literal('something else')
    'something else'

    """
    if string.strip().lower() == 'true':
        return True
    if string.strip().lower() == 'false':
        return False
    if string.strip().lower() == 'none':
        return None
    return string


def split(string, sep=','):
    """Parse a string containing a comma-separated list as a Python list of
    strings. Each resulting string is forced to be lower-case and surrounding
    whitespace is stripped.

    Parameters
    ----------
    string : string
        The string to be split

    sep : string
        Separator to look for

    Returns
    -------
    list of strings

    Examples
    --------
    >>> print split(' One, TWO, three ')
    ['one', 'two', 'three']

    >>> print split('one:two:three', sep=':')
    ['one', 'two', 'three']

    """
    return [x.strip().lower() for x in str.split(string, sep)]


def interpret_param_subfields(subfields, selector=None, pname=None, attr=None):
    infodict = dict(subfields=subfields, selector=selector, pname=pname,
                    attr=attr)

    # Everything has been parsed
    if not infodict['subfields']:
        return infodict

    # If only one field, this must be the param's name, and we're done
    if len(infodict['subfields']) == 1:
        infodict['pname'] = infodict['subfields'].pop()
        return interpret_param_subfields(**infodict)

    # Look for and remove attr field and any subsequent fields
    attr_indices = []
    for n, field in enumerate(infodict['subfields']):
        if field in PARAM_ATTRS:
            attr_indices.append(n)

    # TODO: not clear what's being done here; also, would slicing be more clear
    # than iterating & calling pop()?
    if len(attr_indices) == 1:
        attr_idx = attr_indices[0]
        infodict['attr'] = [
            infodict['subfields'].pop(attr_idx)
            for _ in range(attr_idx, len(infodict['subfields']))
        ]
        return interpret_param_subfields(**infodict)

    elif len(attr_indices) > 1:
        raise ValueError('Found multiple attrs in config name "%s"' %pname)

    if len(infodict['subfields']) == 2:
        infodict['pname'] = infodict['subfields'].pop()
        infodict['selector'] = infodict['subfields'].pop()
        return interpret_param_subfields(**infodict)

    raise ValueError('Unable to parse param subfields %s'
                     %infodict['subfields'])


def parse_param(config, section, selector, fullname, pname, value):
    # Note: imports placed here to avoid circular imports
    from pisa.core.param import Param
    from pisa.core.prior import Prior
    # TODO: Are these defaults actually a good idea? Should all be explicitly
    # specified?
    kwargs = dict(name=pname, is_fixed=True, prior=None, range=None)
    try:
        value = parse_quantity(value)
        kwargs['value'] = value.n * value.units
    except ValueError:
        value = parse_string_literal(value)
        kwargs['value'] = value

    # Search for explicit attr specifications
    if config.has_option(section, fullname + '.fixed'):
        kwargs['is_fixed'] = config.getboolean(section, fullname + '.fixed')

    if config.has_option(section, fullname + '.unique_id'):
        kwargs['unique_id'] = config.get(section, fullname + '.unique_id')

    if config.has_option(section, fullname + '.range'):
        range_ = config.get(section, fullname + '.range')
        # Note: `nominal` and `sigma` are called out in the `range_` string
        if 'nominal' in range_:
            nominal = value.n * value.units # pylint: disable=unused-variable
        if 'sigma' in range_:
            sigma = value.s * value.units # pylint: disable=unused-variable
        range_ = range_.replace('[', 'np.array([')
        range_ = range_.replace(']', '])')
        kwargs['range'] = eval(range_).to(value.units) # pylint: disable=eval-used

    if config.has_option(section, fullname + '.prior'):
        if config.get(section, fullname + '.prior') == 'uniform':
            kwargs['prior'] = Prior(kind='uniform')
        elif config.get(section, fullname + '.prior') == 'jeffreys':
            kwargs['prior'] = Prior(kind='jeffreys', A=kwargs['range'][0], B=kwargs['range'][1])
        elif config.get(section, fullname + '.prior') == 'spline':
            priorname = pname
            if selector is not None:
                priorname += '_' + selector
            data = config.get(section, fullname + '.prior.data')
            data = from_file(data)
            data = data[priorname]
            knots = ureg.Quantity(np.asarray(data['knots']), data['units'])
            knots = knots.to(value.units)
            coeffs = np.asarray(data['coeffs'])
            deg = data['deg']
            kwargs['prior'] = Prior(kind='spline', knots=knots, coeffs=coeffs,
                                    deg=deg)
        elif 'gauss' in config.get(section, fullname + '.prior'):
            raise Exception('Please use new style +/- notation for gaussian'
                            ' priors in config')
        else:
            raise Exception('Prior type unknown')

    elif hasattr(value, 's') and value.s != 0:
        kwargs['prior'] = Prior(kind='gaussian', mean=value.n * value.units,
                                stddev=value.s * value.units)

    try:
        param = Param(**kwargs)
    except:
        logging.error('Failed to instantiate new Param object with kwargs %s',
                      kwargs)
        raise

    return param


def parse_pipeline_config(config):
    """Parse pipeline config.

    Parameters
    ----------
    config : string or ConfigParser

    Returns
    -------
    stage_dicts : OrderedDict
        Keys are (stage_name, service_name) tuples and values are OrderedDicts
        with keys the argnames and values the arguments' values. Some known arg
        values are parsed out fully into Python objects, while the rest remain
        as strings that must be used or parsed elsewhere.

    """
    # Note: imports placed here to avoid circular imports
    from pisa.core.binning import MultiDimBinning, OneDimBinning
    from pisa.core.param import ParamSelector

    if isinstance(config, basestring):
        config = from_file(config)
    elif isinstance(config, ConfigParser):
        pass
    else:
        raise TypeError(
            '`config` must either be a string or ConfigParser. Got %s instead.'
            % type(config)
        )

    # Create binning objects
    binning_dict = {}
    for name, value in config.items('binning'):
        if name.endswith('.order'):
            order = split(config.get('binning', name))
            binning, _ = split(name, sep='.')
            bins = []
            for bin_name in order:
                kwargs = eval( # pylint: disable=eval-used
                    config.get('binning', binning + '.' + bin_name)
                )
                bins.append(OneDimBinning(bin_name, **kwargs))
            binning_dict[binning] = MultiDimBinning(bins)

    # Pipeline section
    section = 'pipeline'

    # Get and parse the order of the stages (and which services implement them)
    order = [split(x, ':') for x in split(config.get(section, 'order'))]

    param_selections = []
    if config.has_option(section, 'param_selections'):
        param_selections = split(config.get(section, 'param_selections'))

    # Parse [stage:<stage_name>] sections and store to stage_dicts
    stage_dicts = OrderedDict()
    for stage, service in order:
        section = 'stage:%s' %stage

        # Instantiate dict to store args to pass to this stage
        service_kwargs = OrderedDict()

        param_selector = ParamSelector(selections=param_selections)
        service_kwargs['params'] = param_selector

        n_params = 0
        for fullname, value in config.items(section):
            # See if this matches a param specification
            param_match = PARAM_RE.match(fullname)
            if param_match is not None:
                n_params += 1

                param_match_dict = param_match.groupdict()
                param_subfields = param_match_dict['subfields'].split('.')

                # Figure out what the dotted fields represent...
                infodict = interpret_param_subfields(subfields=param_subfields)

                # If field is an attr, skip since these are located manually
                if infodict['attr'] is not None:
                    continue

                # Check if this param already exists in a previous stage; if
                # so, make sure there are no specs for this param, but just a
                # link to previous the param object that is already
                # instantiated.
                for kw in stage_dicts.values():
                    # Stage did not get a `params` argument from config
                    if not kw.has_key('params'):
                        continue

                    # Retrieve the param from the ParamSelector
                    try:
                        param = kw['params'].get(
                            name=infodict['pname'],
                            selector=infodict['selector']
                        )
                    except KeyError:
                        continue

                    # Make sure there are no other specs (in this section) for
                    # the param defined defined in previous section
                    for a in PARAM_ATTRS:
                        if config.has_option(section, '%s.%s' %(fullname, a)):
                            raise ValueError("Parameter spec. '%s' of '%s' "
                                             "found in section '%s', but "
                                             "parameter exists in previous "
                                             "stage!"%(a, fullname, section))

                    break

                # Param *not* found in a previous stage (i.e., no explicit
                # `break` encountered in `for` loop above); therefore must
                # instantiate it.
                else:
                    param = parse_param(
                        config=config,
                        section=section,
                        selector=infodict['selector'],
                        fullname=fullname,
                        pname=infodict['pname'],
                        value=value
                    )

                param_selector.update(param, selector=infodict['selector'])

            # If it's not a param spec but contains 'binning', assume it's a
            # binning spec
            elif 'binning' in fullname:
                service_kwargs[fullname] = binning_dict[value]

            # Otherwise it's some other stage instantiation argument; identify
            # this by its full name and try to interpret and instantiate a
            # Python object using the string
            else:
                try:
                    value = parse_quantity(value)
                    value = value.n * value.units
                except ValueError:
                    value = parse_string_literal(value)
                service_kwargs[fullname] = value

        # If no params actually specified in config, remove 'params' from the
        # service's keyword args
        if n_params == 0:
            service_kwargs.pop('params')

        # Store the service's kwargs to the stage_dicts
        stage_dicts[(stage, service)] = service_kwargs

    return stage_dicts


class BetterConfigParser(SafeConfigParser):
    def __init__(self, *args, **kwargs):
        SafeConfigParser.__init__(self, *args, **kwargs)

    def read(self, filenames):
        from pisa.utils.resources import find_resource
        if isinstance(filenames, basestring):
            filenames = [filenames]
        new_filenames = [find_resource(fn) for fn in filenames]

        # Preprocessing for include statements
        processed_filenames = []
        # loop until we cannot find any more includes
        while True:
            processed_filenames.extend(new_filenames)
            new_filenames = self.recursive_filenames(new_filenames)
            rec_incs = set(new_filenames).intersection(processed_filenames)
            if any(rec_incs):
                raise ValueError('Recursive include statements found for %s'
                                 % ', '.join(rec_incs))
            if not new_filenames:
                break
        # call read with complete files list
        SafeConfigParser.read(self, processed_filenames)

    def recursive_filenames(self, filenames):
        new_filenames = []
        for filename in filenames:
            new_filenames.extend(self.process_file_and_includes(filename))
        return new_filenames

    @staticmethod
    def process_file_and_includes(filename):
        from pisa.utils.resources import find_resource
        processed_filenames = []
        with open(filename) as f:
            for line in f.readlines():
                if line.startswith('#include '):
                    inc_file = line[9:].rstrip()
                    inc_file = find_resource(inc_file)
                    processed_filenames.append(inc_file)
                    logging.debug('including file %s in cfg', inc_file)
                else:
                    break
        return processed_filenames

    def get(self, section, option, raw=True, vars=None): # pylint: disable=redefined-builtin
        result = SafeConfigParser.get(self, section, option,
                                                   raw=raw, vars=vars)
        result = self.__replace_sectionwide_templates(result)
        return result

    def items(self, section, raw=True, vars=None): # pylint: disable=redefined-builtin
        config_list = SafeConfigParser.items(
            self, section=section, raw=raw, vars=vars
        )
        result = [(key, self.__replace_sectionwide_templates(value))
                  for key, value in config_list]
        return result

    def optionxform(self, optionstr):
        """Enable case sensitive options in .ini/.cfg files."""
        return optionstr

    def __replace_sectionwide_templates(self, data):
        """Replace <section|option> with get(section, option) recursively."""
        result = data
        findExpression = re.compile(r"((.*)\<!(.*)\|(.*)\!>(.*))*")
        groups = findExpression.search(data).groups()

        # If expression not matched
        if groups != (None, None, None, None, None):
            result = self.__replace_sectionwide_templates(groups[1])
            result += self.get(groups[2], groups[3])
            result += self.__replace_sectionwide_templates(groups[4])
        return result


def test_parse_pipeline_config():
    """Unit test for function `parse_pipeline_config`"""
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--pipeline', metavar='CONFIGFILE',
        default='settings/pipeline/example.cfg',
        help='File containing settings for the pipeline.'
    )
    parser.add_argument(
        '-v', action='count', default=0,
        help='Set verbosity level. Minimum is forced to level 1 (info)'
    )
    args = parser.parse_args()
    args.v = max(1, args.v)
    set_verbosity(args.v)

    # Load via BetterConfigParser
    config0 = BetterConfigParser()
    config0.read(args.pipeline)
    _ = parse_pipeline_config(config0)

    # Load directly
    config = parse_pipeline_config(args.pipeline)

    for key, vals in config.items():
        logging.debug('%s: %s', key, vals)
    logging.info('<< PASS : test_parse_pipeline_config >>')


if __name__ == '__main__':
    test_parse_pipeline_config()
