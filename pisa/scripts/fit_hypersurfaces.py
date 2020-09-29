#!/usr/bin/env python

"""
Fit a hypersurface to discrete systematics datasets -- as specified by a fit config
file -- and produce a fit file usable by, e.g., the `discr_sys.hypersurfaces` service.

For more details, see `pisa/utils/hypersurface.pu`

A script for making plots from the fit results produced by this file can be found in the
Fridge (private repository) at

  fridge/analysis/common/scripts/plotting/plot_hypersurface_fits.py

See example fit config files in directory `pisa_examples/resources/discr_sys/`.


Config file syntax
==================

"general" section
-----------------
You must define a "general" section which must have at least options "sys_list" and 
"sys_func_list" and can optionally include options "units" and "combine_regex". You 
can add more options to the "general" section, but they will be ignored by PISA unless
they are explicitly referenced elsewhere in your config file. E.g., a "general" 
section with all three required and optional options (and no more) ::

  [general]
  sys_list = dom_eff, hole_ice, hole_ice_fwd
  sys_func_list = linear, exponential, linear
  units = dimensionless, dimensionless, dimensionless
  combine_regex = ["nue.*_cc", "numu.*_cc", "nutau.*_cc", ".*_nc"]

If "units" is not specified, units default to "dimensionless" for all systematics. If
specified, there must be the same number of comma-separated units strings
(interpret-able by the Pint module) as there are options in the "sys_list".

Note that "combine_regex" should be Python-evaulatable to a string or a sequence of
strings. Old versions of this script allowed values like ::

  combine_regex = nue.*_cc,numu.*_cc,nutau.*_cc,.*_nc

but unambiguously parsing such a single comma-separated string of one or more regexes
(which themselves can contian commas) is probably impossible, so this syntax is
deprecated (a warning is emitted if it is detected) and should be avoided.

"apply_to_all_sets" section
---------------------------
You can optionally include an "apply_to_all_sets" section in your config that, true to
its name, defines options applied to all systematic sets sections in the file. E.g., ::

  [apply_to_all_sets]
  pipeline_cfg = settings/pipeline/nutau_mc_baseline.cfg
  remove [discr_sys.hypersurfaces] =
  set [data.simple_data_loader] data_dict = {
      'true_energy': 'true_energy',
      'true_coszen': 'true_coszen',
      'reco_energy': 'reco_energy',
      'reco_coszen': 'reco_coszen',
      'pid': 'pid',
      'weighted_aeff': 'weighted_aeff',
      'dunkman_L5': 'dunkman_L5',
      }

The above sets "pipeline_cfg" for all discr sets, removes the "discr_sys.hypersurfaces"
service, and redefines the "data_dict" in the "data.simple_data_loader" section. Any
options defined in the discrete set sections of the config will start with this as their
configuration. Details of the above syntax are described more below.

"nominal_set" and "sys_set" sections
------------------------------------
All other required sections in the config file describe the discrete sets and must
define a "pipeline_cfg" for that systematics set (whether explicitly in the section or
via the "apply_to_all_sets" section). A systematics set section either starts with
"nominal_set" (for one and only one set) or "sys_set" (for all remaining sets), followed
by a colon and magnitudes of the values of each systematic specified in "general"
section / "sys_list" option (in the same order as defined there). E.g., continuing the
example config file defined in the above examples, ::

  [nominal_set : 1.00 , 25 , 0.0]
  set [data.simple_data_loader] events_file = /path/to/nominal_set.hdf5

  [sys_set : 0.88 , 22 , 0.1]
  set [data.simple_data_loader] events_file = /path/to/sys_set_1.hdf5

  [sys_set : 1.12 , 28 , 0.2]
  set [data.simple_data_loader] events_file = /path/to/sys_set_2.hdf5

and so forth. Note that all magnitudes must be specified in the same units specified in
"general" section / "units" option, or if that option is not specified, all magnitudes
must represent dimensionless quantities. Also note that the "pipeline_cfg" is already
defined in the above "apply_to_all_sets" section, so each of these sys set sections
simply swap out the events_file in that pipeline config file for the appropriate events
file.

Syntax for modifying the specified "pipeline_cfg"
-------------------------------------------------
The following syntax is interpreted if specified in "apply_to_all_sets", "sys_set",
and/or "nominal_set" sections. Note an "apply_to_all_sets" section must specify a
"pipeline_cfg" for any of the following syntax to be used.

Define or redefine an option via the "set" keyword followed by the section in square
brackets, the option, either equals (=) or colon (:), and finally the value to set the
option to. E.g., "events_file" in config section "data.simple_data_loader" is set to
"settings/pipeline/example.cfg" via ::

  set [data.simple_data_loader] events_file = settings/pipeline/example.cfg

the section is created if it doesn't exist and the option "events_file" is added to that
section if it doesn't already exist.

You can create a new section (if it doesn't already exist) via the "set" keyword
followed by the section in square brackets and either equals (=) or colon (:). Anything
following the equals/colon is ignored. E.g. to add the "data.simple_data_loader" section
(if it doesn't already exist), ::

  set [data.simple_data_loader] =

Notes on whitespace
-------------------
Whitespace is ignored in section names and in lists, so the following are interpreted
equivalently ::

  [ sys_set : 0.88 , 22 , 0.1 ]
  [sys_set:0.88,22,0.1]

Likewise, the following are all equivalent ::

  sys_list = aa,bb,cc
  sys_list = aa , bb , cc

"""

from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
from ast import literal_eval
from collections import OrderedDict
from collections.abc import Mapping, Sequence
import copy
from io import StringIO
from os.path import join
import re

import numpy as np
from scipy.optimize import curve_fit
from uncertainties import unumpy as unp
from uncertainties import ufloat

from pisa import ureg
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.utils.fileio import mkdir, from_file, to_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.hypersurface import Hypersurface, HypersurfaceParam, get_hypersurface_file_name


__all__ = [
    # constants
    "GENERAL_SECTION_NAME",
    "APPLY_ALL_SECTION_NAME",
    "COMBINE_REGEX_OPTION",
    "SYS_LIST_OPTION",
    "SYS_FUNC_LIST_OPTION",
    "UNITS_OPTION",
    "UNITS_SPECIFIER",
    "SYS_SET_OPTION",
    "SET_OPTION_RE",
    "REMOVE_OPTION_RE",
    # functions
    "parse_args",
    "parse_fit_config",
    "load_and_modify_pipeline_cfg",
    "create_hypersurfaces",
    "main",
]

__author__ = "P. Eller, T. Stuttard, T. Ehrhardt, J.L. Lanfranchi"

__license__ = """Copyright (c) 2014-2018, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


GENERAL_SECTION_NAME = "general"
"""general section name"""

APPLY_ALL_SECTION_NAME = "apply_to_all_sets"
"""section name that defines pipeline config / options for all discr sets"""

COMBINE_REGEX_OPTION = "combine_regex"
"""Option in general section to specify map(s) to combine before performing fit"""

SYS_LIST_OPTION = "sys_list"
"""Option in general section to specify discrete systematics"""

SYS_FUNC_LIST_OPTION = "sys_func_list"
"""Option in general section to specify functional form of the discrete systematics"""

UNITS_OPTION = "units"
"""Option in general section to specify units of discrete systematics"""

UNITS_SPECIFIER = "units."
"""User is allowed to use e.g. <UNITS_SPECIFIER>meter or simply meter in UNITS_OPTION"""

NOMINAL_SET_PFX = "nominal_set"
"""the section name with this followed by a colon indicates the nominal set"""

SYS_SET_PFX = "sys_set"
"""section names with this followed by a colon indicate non-nominal systematics sets"""

SYS_SET_OPTION = "pipeline_cfg"
"""systematics set config file is specified by this option"""

SET_OPTION_RE = re.compile(r"\s*set\s*\[(.*)\]\s*(\S*.*)")
"""defining a section and/or an option within a section in a pipeline configs is
specified by following this pattern"""

REMOVE_OPTION_RE = re.compile(r"\s*remove\s*\[(.*)\]\s*(\S*.*)")
"""modifications to pipeline configs are specified by options following this pattern"""


def parse_args():
    """Parse arguments from command line.

    Returns
    -------
    args : namespace

    """
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "-f",
        "--fit-cfg",
        type=str,
        metavar="configfile",
        required=True,
        help="Settings for the hypersurface fit",
    )
    parser.add_argument(
        "--tag", type=str, default="deepcore", help="Tag for the filename."
    )
    parser.add_argument(
        "-o", "--outdir", type=str, required=True, help="Set output directory"
    )
    parser.add_argument("-v", action="count", default=None, help="set verbosity level")
    args = parser.parse_args()
    return args


def parse_fit_config(fit_cfg):
    """Perform sanity checks on and parse fit configuration file.

    Parameters
    ----------
    fit_cfg : str
        path to a fit configuration file

    Returns
    -------
    fit_cfg : PISAConfigParser
        parsed fit configuration
    sys_list : list of str
        parsed names of systematic parameters
    sys_func_list : list of str
        parsed names of systematic parameter functional forms
    units_list : list of str
        units corresponding to each discrete systematic
    combine_regex : list of str
        each string is a regular expression for combining pipeline outputs; see
        :func:`pisa.core.map.MapSet.combine_regex` for details.

    """
    fit_cfg = from_file(fit_cfg)
    no_ws_section_map = {s.strip(): s for s in fit_cfg.sections()}

    if GENERAL_SECTION_NAME not in no_ws_section_map.values():
        raise KeyError('Fit config is missing the "%s" section!' % GENERAL_SECTION_NAME)

    general_section = fit_cfg[GENERAL_SECTION_NAME]

    if SYS_LIST_OPTION not in general_section:
        raise KeyError(
            "Fit config has to specify systematic parameters as"
            ' "%s" option in "%s" section (comma-separated list of names).'
            % (SYS_LIST_OPTION, GENERAL_SECTION_NAME)
        )

    if SYS_FUNC_LIST_OPTION not in general_section:
        raise KeyError(
            "Fit config has to specify systematic parameter functional forms as"
            ' "%s" option in "%s" section (comma-separated list of names).'
            % (SYS_FUNC_LIST_OPTION, GENERAL_SECTION_NAME)
        )

    sys_list = [s.strip() for s in general_section[SYS_LIST_OPTION].split(",")]

    sys_func_list = [s.strip() for s in general_section[SYS_FUNC_LIST_OPTION].split(",")]

    if UNITS_OPTION in general_section:
        units_list = []
        units_specs = (
            general_section[UNITS_OPTION].replace(UNITS_SPECIFIER, "").split(",")
        )
        for units_spec in units_specs:
            # Make sure units are interpret-able by Pint
            try:
                ureg.Unit(units_spec)
            except:
                logging.error(
                    'Unit "%s" specified by "%s" option in "general" section is not'
                    "interpret-able by Pint",
                    units_spec,
                    UNITS_OPTION,
                )
                raise
            units_list.append(units_spec)
    else:
        units_list = ["dimensionless" for s in sys_list]
        logging.warning(
            "No %s option found in %s section; assuming systematic parameters are"
            " dimensionless",
            UNITS_OPTION,
            GENERAL_SECTION_NAME,
        )

    if len(units_list) != len(sys_list):
        raise ValueError(
            '{} units specified by "{}" option but {} systematics specified by "{}"'
            "option; must be same number of each.".format(
                len(units_list), UNITS_OPTION, len(sys_list), SYS_LIST_OPTION
            )
        )

    logging.info(
        "Found systematic parameters %s",
        ["{} ({})".format(s, u) for s, u in zip(sys_list, units_list)],
    )

    combine_regex = general_section.get(COMBINE_REGEX_OPTION, None)
    if combine_regex:
        try:
            combine_regex = literal_eval(combine_regex)
        except (SyntaxError, ValueError):
            logging.warning(
                'Deprecated syntax for "combine_re" (make into a Python-evaluatable'
                "sequence of strings instead) :: combine_regex = %s",
                combine_regex,
            )
            combine_regex = [r.strip() for r in combine_regex.split(",")]

    if APPLY_ALL_SECTION_NAME in no_ws_section_map:
        apply_all_section = fit_cfg[no_ws_section_map[APPLY_ALL_SECTION_NAME]]
        for no_ws_sname, sname in no_ws_section_map.items():
            if not (
                no_ws_sname.startswith(NOMINAL_SET_PFX)
                or no_ws_sname.startswith(SYS_SET_PFX)
            ):
                continue
            sys_set_section = fit_cfg[sname]
            for option, val in apply_all_section.items():
                sys_set_section[option] = val

    return fit_cfg, sys_list, sys_func_list, units_list, combine_regex


def load_and_modify_pipeline_cfg(fit_cfg, section):
    """Load and modify the pipeline config file as specified in that section of the fit
    config.

    Parameters
    ----------
    fit_cfg : pisa.utils.config_parser.PISAConfigParser
        any subclass of :class:`configparser.RawConfigParser` should work as well

    section : str
        name of the section to extract from the `fit_cfg`

    Returns
    -------
    pipeline_cfg : pisa.utils.config_parser.PISAConfigParser
        pipeline config

    pipeline_cfg_path : str
        path to the pipeline config as it is specified in the fit config

    """
    pipeline_cfg_path = fit_cfg.get(section, SYS_SET_OPTION)
    other_options = fit_cfg.options(section)
    other_options.remove(SYS_SET_OPTION)

    pipeline_cfg = from_file(pipeline_cfg_path)

    # Get a no-whitespace version of the section names
    section_map = {s.strip(): s for s in pipeline_cfg.sections()}

    for option in other_options:
        set_match = SET_OPTION_RE.match(option)
        remove_match = REMOVE_OPTION_RE.match(option) if not set_match else None
        if set_match:
            section_spec, set_option = set_match.groups()
            no_ws_section_spec = section_spec.strip()
            set_option = set_option.strip()
            if no_ws_section_spec not in section_map:
                logging.debug(
                    'Adding section [%s] to in-memory copy of pipeline config "%s"',
                    section_spec,
                    pipeline_cfg_path,
                )
                pipeline_cfg.add_section(section_spec)
                section_map[no_ws_section_spec] = section_spec
            if set_option:
                set_value = fit_cfg.get(section, option).strip()
                logging.debug(
                    'Setting section [%s] option "%s = %s" in in-memory'
                    ' copy of pipeline config "%s"',
                    section_spec,
                    set_option,
                    set_value,
                    pipeline_cfg_path,
                )
                pipeline_cfg.set(section_map[no_ws_section_spec], set_option, set_value)
        elif remove_match:
            section_spec, remove_option = remove_match.groups()
            no_ws_section_spec = section_spec.strip()
            remove_option = remove_option.strip()
            if no_ws_section_spec in section_map:
                if remove_option:
                    logging.debug(
                        'Removing section [%s] option "%s" from in-memory copy of'
                        ' pipeline config "%s"',
                        section_spec,
                        remove_option,
                        pipeline_cfg_path,
                    )
                    pipeline_cfg.remove_option(
                        section_map[no_ws_section_spec], remove_option
                    )
                else:
                    logging.debug(
                        "Removing section [%s] from in-memory copy of pipeline config"
                        ' "%s"',
                        section_spec,
                        pipeline_cfg_path,
                    )
                    pipeline_cfg.remove_section(section_map[no_ws_section_spec])
            else:
                logging.warning(
                    "Told to remove section [%s] but section does not exist in"
                    ' pipline config "%s"',
                    section_spec,
                    pipeline_cfg_path,
                )
        else:
            raise ValueError("Unhandled option in fit config: {}".format(option))

    return pipeline_cfg, pipeline_cfg_path


def create_hypersurfaces(fit_cfg):
    """Generate and store mapsets for different discrete systematics sets
    (with a single set characterised by a dedicated pipeline configuration)

    Parameters
    ----------
    fit_cfg : string
        Path to a fit config file

    Returns
    -------
    hypersurfaces : OrderedDict
        Container with the fitted hypersurface for each map type

    """

    #
    # Parse fit config file
    #

    parsed_fit_cfg, sys_list, sys_func_list, units_list, combine_regex = parse_fit_config(fit_cfg)


    #
    # Create the hypersurface params
    #

    # Loop over the param names and functional forms and create the params
    #TODO Add option to support initial param guesses
    params = [ HypersurfaceParam(name=param_name, func_name=param_func_name) for param_name, param_func_name in zip(sys_list, sys_func_list) ]


    #
    # Parse defintion of each dataset
    #

    fit_cfg_txt_buf = StringIO()
    parsed_fit_cfg.write(fit_cfg_txt_buf)
    fit_cfg_txt = fit_cfg_txt_buf.getvalue()

    nominal_pipeline_cfg = None
    nominal_param_values = None
    sys_pipeline_cfgs = []
    sys_param_values = []

    # Loop over config
    for section in parsed_fit_cfg.sections():

        no_ws_section = section.strip()

        section_prefix = no_ws_section.split(":")[0].strip()
        is_nominal = section_prefix == NOMINAL_SET_PFX
        is_dataset = is_nominal or section_prefix == SYS_SET_PFX

        if is_dataset:

            # Parse the list of systematics parameter values from the section name
            sys_param_point = tuple(float(x) for x in section.split(":")[1].split(","))

            if len(sys_param_point) != len(sys_list):
                raise ValueError(
                    "Section heading [{}] specifies {:d} systematic"
                    " parameter values, but there are {:d} systematics".format(
                        section, len(sys_param_point), len(sys_list)
                    )
                )

            # Parse the config file
            parsed_pipeline_cfg, pipeline_cfg_path = load_and_modify_pipeline_cfg(
                fit_cfg=parsed_fit_cfg, section=section
            )

            # Store
            if is_nominal :
                assert nominal_pipeline_cfg is None, "Found multiple nominal dataset definitions"
                nominal_pipeline_cfg = parsed_pipeline_cfg
                nominal_param_values = sys_param_point
            else :
                sys_pipeline_cfgs.append(parsed_pipeline_cfg)
                sys_param_values.append(sys_param_point)

        # In this loop, nothing to do for general & apply_to_all_sets sections
        elif no_ws_section in (GENERAL_SECTION_NAME, APPLY_ALL_SECTION_NAME):
            pass

        # Do not allow any other sections in the config
        else:
            raise ValueError("Invalid section in fit config file: [%s]" % section)

    # Check found stuff
    assert nominal_pipeline_cfg is not None, "No nominal dataset definition found"
    assert len(sys_pipeline_cfgs) > 0, "No systematics dataset definitions found"

    # Re-format params into a dict, including the param names
    nominal_param_values = { name:val for name, val in zip(sys_list,nominal_param_values) }
    sys_param_values = [ { name:val for name, val in zip(sys_list,s) } for s in sys_param_values ]


    #
    # Create mapsets
    #

    # Get the nominal mapset
    nominal_dist_maker = DistributionMaker(nominal_pipeline_cfg)
    nominal_mapset = nominal_dist_maker.get_outputs(return_sum=False)[0]

    # Get the systematics mapsets
    sys_mapsets = []
    for sys_pipeline_cfg in sys_pipeline_cfgs :
        sys_dist_maker = DistributionMaker(sys_pipeline_cfg)
        sys_mapset = sys_dist_maker.get_outputs(return_sum=False)[0]
        sys_mapsets.append(sys_mapset)

    # Combine maps according to the provided regex, if one was provided
    if combine_regex:
        logging.info(
            "Combining maps according to regular expression(s) %s", combine_regex
        )
        nominal_mapset = nominal_mapset.combine_re(combine_regex)
        sys_mapsets = [ s.combine_re(combine_regex) for s in sys_mapsets ]


    #
    # Fit the hypersurface
    #

    hypersurfaces = OrderedDict()

    # Fit one per map, so loop over them
    for map_name in nominal_mapset.names :

        # Create the hypersurface
        hypersurface = Hypersurface( 
            params=params,
            initial_intercept=1., # Initial value for intercept
        )

        # Get just the requested map
        nominal_map = nominal_mapset[map_name]
        sys_maps = [ s[map_name] for s in sys_mapsets ]

        # Perform fit
        hypersurface.fit(
            nominal_map=nominal_map,
            nominal_param_values=nominal_param_values,
            sys_maps=sys_maps,
            sys_param_values=sys_param_values,
            norm=True,
        )

        # Store the result
        hypersurfaces[map_name] = hypersurface

    # Done
    return hypersurfaces



def main():
    """Perform a hypersurface fit to discrete systematics sets."""

    # Get args
    args = parse_args()
    set_verbosity(args.v)

    # Read in data and fit hypersurfaces to it
    hypersurfaces = create_hypersurfaces(fit_cfg=args.fit_cfg)

    # Store as JSON
    mkdir(args.outdir)
    arbitrary_hypersurface = list(hypersurfaces.values())[0]
    output_path = join( args.outdir, get_hypersurface_file_name(arbitrary_hypersurface, args.tag) )
    to_file(hypersurfaces, output_path)


if __name__ == "__main__":
    main()
