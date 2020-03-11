#!/usr/bin/env python

"""
Convert PISA config files from format used up until July 2017 to the new config
file format.
"""


from __future__ import absolute_import

from argparse import ArgumentParser
import os
import re
import sys

from configparser import MissingSectionHeaderError

from pisa.utils.config_parser import PISAConfigParser
from pisa.utils.resources import find_resource


__all__ = ['OLD_SUB_RE', 'parse_args', 'main']

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


OLD_SUB_RE = re.compile(
    r'''
        <!          # Start with <! delimiter
          ([^|]+)   # section is characters that are not pipe
          \|        # pipe as divider
          (.+?)     # at least one character (non-greedy match)
        !>          # end with !> delimiter
    ''', re.VERBOSE
)

OLD_STAGE_SECTION_RE = re.compile(
    r'''
        \[           # Start with [
            \s*      # optional whitespace
            stage    # must start with "stage"
            \s*      # optional whitespace
            :        # colon as separator
            \s*      # optional whitespace
            (\S+)    # non-whitespace key
            \s*      # optional whitespace
        \]           # end with ]
    ''', re.VERBOSE
)

OLD_STAGE_VARIABLE_RE = re.compile(
    r'''
    \${            # Start with ${
        \s*        # optional whitespace (technically illegal, will remove)
        stage      # variable must be named "stage"
        \s*        # optional whitespace (technially illegal, will remove)
        :          # colon separator
        \s*        # optional whitespace (technically illegal, will remove)
        ([^ }]+)   # any character besides space or endbrace
        \s*        # optional whitespace (technically illegal, will remove)
    }
    ''', re.VERBOSE
)

OLD_ORDER_RE = re.compile(r'order\s*(?:=|:)\s*(\S.*)\s*')
ORDER_SUBSTRING_RE = re.compile(r'\s*(\S+)\s*(?::|\.)\s*(\S+)\s*')
JSON_NO_BZ2_RE = re.compile(r'(\.json)(?!\.bz2)')

SECTION_NAME_WITH_COLON_RE = re.compile(
    r'''
    ^\s*                  # only possibly whitespace before opening bracket
    \[                    # open bracket
        (                 # capture text found within these parens
            (?:           # do not capture text within these parens
                [^\]]*:   # non-"]" followed by ":'
            )+            # stop no-capture group; one or more of this pattern
            [^\]]*        # any number of non-"]" before closing bracket
        )                 # stop capturing text;
    \]                    # closing bracket
    ''', re.VERBOSE
)
NEW_SECTION_RE = re.compile(
    r'''
    (
        ^\s*           # only possibly whitespace before opening bracket
        \[             # open bracket
            ([^\]]*)   # capture anything besides "]"
        \]             # closing bracket
    )                  # stop capturing text;
    ''', re.VERBOSE
)
NEW_VARIABLE_RE = re.compile(
    r'''
    (               # start capturing text
        \${         # start with ${
            ([^}]*) # any character besides closing brace
        }           # closing brace
    )               # stop capturing text
    ''', re.VERBOSE
)

OTHER_SECTION_NAME_SEPARATOR = '|'
"""If section names (and any corresponding variable names) use colons as
separators (and we haven't converted the colons to periods), the colons will be
replaced with this character instead."""

MULTI_COLON_VAR_RE = re.compile(
    r'''
    (                # Capture
    \${              # start with ${
        (?:          # do not capture this group
            [^}]*:   # anything besides closing brace
        ){2,}        # sto non-capturing group; must be 2 or more of these
        [^}]*        # any amount of non-closing-brace characters
    }                # closing brace
    )                # stop capture
    ''', re.VERBOSE
)

PARAM_RE_STR = r'''
    param                             # must start with "param"
    (\.[_A-Za-z][_a-zA-Z0-9]*){0,1}   # optional param sel. (valid Python name)
    \.                                # period before param name
    %s                                # get actual param name from elsewhere
'''
NSI_PARAM_RE_MAP = {
    e_ij: re.compile(PARAM_RE_STR % e_ij, re.VERBOSE)
    for e_ij in ['eps_ee', 'eps_emu', 'eps_etau', 'eps_mumu', 'eps_mutau',
                 'eps_tautau']
}

INCLUDE_AS = {
    'settings/binning/example.cfg': 'binning',
    'settings/binning/pingu_nutau.cfg': 'binning',
    'settings/osc/loiv2.cfg': 'osc',
    'settings/osc/nufitv20.cfg': 'osc',
    'settings/osc/nufitv22.cfg': 'osc',
    'settings/osc/earth.cfg': 'earth',
}

JSON_TO_JSON_BZ2 = [
    'PISAV2IPHonda2015SPLSolMaxFlux-atm_delta_index0.05',
    'PISAV2IPHonda2015SPLSolMaxFlux-atm_delta_index0.05',
    'PISAV2IPHonda2015SPLSolMaxFlux-nu_nubar_ratio1.10',
    'PISAV2IPHonda2015SPLSolMaxFlux-nue_numu_ratio1.03',
    'PISAV2IPHonda2015SPLSolMaxFlux',
    'PISAV2bisplrepHonda2015SPLSolMaxFlux'
]
JSON_NO_BZ2_RE_LIST = [
    re.compile(r'%s(\.json)(?!\.bz2)' % n) for n in JSON_TO_JSON_BZ2
]

NAMES_CHANGED_MAP = {
    'nutau_holice_domeff_fits_mc.ini': 'nutau_holice_domeff_fits_mc.cfg'
}


def replace_substitution(match):
    """Replace old substitution syntax ``<!section|key!>`` with new syntax
    ``${section:key}``. Also, convert any colons in `section` or `key` to
    ``OTHER_SECTION_NAME_SEPARATOR``.

    Parameters
    ----------
    match : re._pattern_type

    Returns
    -------
    repl : string
        Replacement text

    """
    substrs = match.groups()
    loc = substrs[0].find('stage:')
    if loc >= 0:
        postloc = loc + len('stage:')
        s0 = (
            substrs[0][:loc]
            + 'stage:'
            + substrs[0][postloc:].replace(':', OTHER_SECTION_NAME_SEPARATOR)
        )
    else:
        s0 = substrs[0].replace(':', OTHER_SECTION_NAME_SEPARATOR)

    s1 = substrs[1].replace(':', OTHER_SECTION_NAME_SEPARATOR)

    return '${%s:%s}' % (s0, s1)


def replace_order(match):
    """Replace e.g.
        ``  order = flux:honda , osc : prob3cpu,aeff :hist, reco : hist``
    with
        ``  order = flux.honda, osc.prob3cpu, aeff.hist, reco.hist``

    Parameters
    ----------
    match : re._pattern_type

    Returns
    -------
    repl : string
        Replacement text, starting with ``order`` (i.e., retain whitespace
        preceding the word "order").

    """
    new_substrings = []
    for old_substring in match.groups()[0].split(','):
        new_substrings.append(
            ORDER_SUBSTRING_RE.sub(
                repl=lambda m: '%s.%s' % m.groups(),
                string=old_substring.strip()
            )
        )
    return 'order = %s' % ', '.join(new_substrings)


def append_include_as(include_match):
    """Convert ``#include x`` to ``#include x as y``, where appropriate; also,
    convert incorrect "as" statements. See INCLUDE_AS dict for mapping from
    resource to its "as" target.

    Parameters
    ----------
    include_match : re._pattern_type
        Match produced by INCLUDE_RE.match(string)

    Returns
    -------
    repl : string
        Replacement text for whatever comes after the "#include "

    """
    include_text = include_match.groups()[0]
    include_as_match = PISAConfigParser.INCLUDE_AS_RE.match(include_text)

    as_section = None
    if include_as_match:
        gd = include_as_match.groupdict()
        resource = gd['file']
        as_section = gd['as']
    else:
        resource = include_text
        if resource in INCLUDE_AS.keys():
            as_section = INCLUDE_AS[resource]

    if as_section is None:
        repl = '#include ' + resource
    else:
        repl = '#include %s as %s' % (resource, as_section)

    return repl


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        'config', nargs=1, type=str,
        help='''Pipeline config file to convert. Note that new file will have
        `.new` suffix appended to filename so you can double check the
        conversion before overwriting the original file.'''
    )
    parser.add_argument(
        '--validate-only', action='store_true',
        help='''Do not write an output file, but raise an exception if the file
        _would_ be modified. I.e., specifying this flag makes the script behave
        as a (rough) validator for config files.'''
    )
    return parser.parse_args()


def main():
    """Do the conversion."""
    args = parse_args()
    in_fpath = os.path.expanduser(os.path.expandvars(args.config[0]))
    out_fpath = in_fpath + '.new'

    with open(in_fpath, 'r') as infile:
        orig_contents = infile.readlines()

    osc_stage_header_line = None
    section_names_with_colons = {}
    new_contents = []
    for lineno, orig_line in enumerate(orig_contents, start=1):
        # Remove trailing whitespace, including newline character(s)
        new_line = orig_line.rstrip()

        # Empty (or whitespace-only) line is trivial
        if new_line == '':
            new_contents.append(new_line)
            continue

        # Replace text substitution ("config file variables") syntax
        new_line = OLD_SUB_RE.sub(repl=replace_substitution, string=new_line)

        # Replace stage headers. E.g.
        #     ``  [ stage :stage_name ]``
        # is replaced by
        #     ``  [stage.stage_name]``
        # I.e. retain any whitespace before (and after... though this is
        # already removed) the brackets but swap colon for period and remove
        # whitespace within the brackets.
        new_line = OLD_STAGE_SECTION_RE.sub(
            repl=lambda m: '[stage.%s]' % m.groups(),
            string=new_line
        )

        # Replace stage:key variables. E.g. what should now look like
        #     ``  ${ stage : key }  ``
        # should look like
        #     ``  ${stage.key}  ``
        new_line = OLD_STAGE_VARIABLE_RE.sub(
            repl=lambda m: '${stage.%s}' % m.groups(),
            string=new_line
        )

        stripped = new_line.strip()

        # Replace order string
        if stripped.startswith('order'):
            new_line = OLD_ORDER_RE.sub(repl=replace_order, string=new_line)
        # Record line on which the [stage.osc] section occurs (if any)
        elif stripped == '[stage.osc]':
            osc_stage_header_line = lineno - 1

        # Convert ``#include x`` to ``#include x as y``, where appropriate
        new_line = PISAConfigParser.INCLUDE_RE.sub(
            repl=append_include_as,
            string=new_line
        )

        # Convert JSON filenames to .json.bz2 that are now bzipped
        if '.json' in new_line:
            for json_re in JSON_NO_BZ2_RE_LIST:
                new_line = json_re.sub(repl='.json.bz2', string=new_line)

        # Replace changed names
        for orig_name, new_name in NAMES_CHANGED_MAP.items():
            new_line = new_line.replace(orig_name, new_name)

        # Search for any colons used in section names. This is illegal, as a
        # section name can be used as a variable where the syntax is
        #   ``${section_name:key}``
        # so any colons in section_name will make the parser choke.
        for match in SECTION_NAME_WITH_COLON_RE.finditer(new_line):
            section_name_with_colons = match.groups()[0]
            if NEW_VARIABLE_RE.match(section_name_with_colons):
                if section_name_with_colons.count(':') > 1:
                    raise ValueError(
                        'Multiple colons in new-style variable, line %d:\n'
                        '>> Original line:\n%s\n>> New line:\n%s\n'
                        % (lineno, orig_line, new_line)
                    )
                else:
                    continue
            section_name_without_colons = section_name_with_colons.replace(
                ':', OTHER_SECTION_NAME_SEPARATOR
            )
            section_names_with_colons[section_name_with_colons] = (
                section_name_without_colons
            )

        new_contents.append(new_line)

    #for item in  section_names_with_colons.items():
    #    print '%s --> %s' % item

    # Go back through and replace colon-sparated section names with
    # ``OTHER_SECTION_NAME_SEPARATOR``-separated section names
    all_names_to_replace = section_names_with_colons.keys()
    def replace_var(match):
        """Closure to replace variable names"""
        whole_string, var_name = match.groups()
        if var_name in all_names_to_replace:
            return '${%s}' % section_names_with_colons[var_name]
        return whole_string

    def replace_section_name(match):
        """Closure to replace section names"""
        whole_string, section_name = match.groups()
        if section_name in all_names_to_replace:
            return whole_string.replace(
                section_name, section_names_with_colons[section_name]
            )
        return whole_string

    for lineno, new_line in enumerate(new_contents, start=1):
        if not new_line:
            continue

        new_line = NEW_VARIABLE_RE.sub(repl=replace_var, string=new_line)
        new_line = NEW_SECTION_RE.sub(repl=replace_section_name,
                                      string=new_line)

        #new_line = NEW_SECTION_RE.sub(repl=replace_colon_names,
        #                              string=new_line)

        #for with_colons, without_colons in section_names_with_colons:
        #    new_line = new_line.replace(with_colons, without_colons)

        # Check for multiple colons in a variable (which is illegal)
        if MULTI_COLON_VAR_RE.findall(new_line):
            raise ValueError(
                'Multiple colons in variable, line %d:\n>> Original'
                ' line:\n%s\n>> New line:\n%s\n'
                % (lineno, orig_contents[lineno - 1], new_line)
            )

        new_contents[lineno - 1] = new_line

    # Parse the new config file with the PISAConfigParser to see if NSI
    # parameters are defined in the `stage.osc` section (if the latter is
    # present). If needed, insert appropriate #include in the section
    pcp = PISAConfigParser()
    missing_section_header = False
    try:
        pcp.read_string(('\n'.join(new_contents) + '\n').decode('utf-8'))
    except MissingSectionHeaderError:
        missing_section_header = True
        pcp.read_string(
            (
                '\n'.join(['[dummy section header]'] + new_contents) + '\n'
            ).decode('utf-8')
        )

    if 'stage.osc' in pcp:
        keys_containing_eps = [k for k in pcp['stage.osc'].keys()
                               if '.eps_'.encode('utf-8') in k]

        nsi_params_present = []
        nsi_params_missing = []
        for nsi_param, nsi_param_re in NSI_PARAM_RE_MAP.items():
            found = None
            for key_idx, key in enumerate(keys_containing_eps):
                if nsi_param_re.match(key):
                    found = key_idx
                    nsi_params_present.append(nsi_param)

            if found is None:
                nsi_params_missing.append(nsi_param)
            else:
                # No need to search this key again
                keys_containing_eps.pop(found)

        if set(nsi_params_present) == set(NSI_PARAM_RE_MAP.keys()):
            all_nsi_params_defined = True
        elif set(nsi_params_missing) == set(NSI_PARAM_RE_MAP.keys()):
            all_nsi_params_defined = False
        else:
            raise ValueError('Found a subset of NSI params defined; missing %s'
                             % str(nsi_params_missing))

        # NOTE: since for now the contents of nsi_null.cfg are commented out
        # (until merging NSI branch), the above check will say NSI params are
        # missing if the #include statement was made. So check to see if
        # settings/osc/nsi_null.cfg _has_ been included (we can't tell what
        # section it is in, but we'll have to just accept that).
        #
        # We will probably want to remove this stanza as soon as NSI brnach is
        # merged, since this is imprecise and can introduce other weird corner
        # cases.
        rsrc_loc = find_resource('settings/osc/nsi_null.cfg')
        for file_iter in pcp.file_iterators:
            if rsrc_loc in file_iter.fpaths_processed:
                all_nsi_params_defined = True

        if not all_nsi_params_defined and osc_stage_header_line is None:
            raise ValueError(
                "Found a stage.osc section without NSI params defined (using"
                " PISAConfigParser) but could not find the line of the"
                " `[stage.osc]` header. This could occur if `[stage.osc]` came"
                " from an `#include`'d file. You can manually define the NSI"
                " parameters in this file or in the included file e.g. as"
                " found in `settings/osc/nsi_null.cfg` or simply add the"
                " statement ``#include settings/osc/nsi_null.cfg`` to either"
                " file (so long as that statement it falls within the"
                " stage.osc section)."
            )

        # Add ``#include settings/osc/nsi_null.cfg`` at top of stage.osc
        # section if a stage.osc section is present and no NSI params were
        # specified in that section
        if not all_nsi_params_defined:
            # Add an #include to set all NSI parameters to 0
            new_contents.insert(
                osc_stage_header_line + 1,
                '#include settings/osc/nsi_null.cfg'
            )
            # Add an extra blank line after the #include line
            new_contents.insert(osc_stage_header_line + 2, '')

    if not new_contents:
        raise ValueError('Empty file after conversion; quitting.')

    # Note that newlines are added but no join is performed for comparison
    # against `orig_contents`
    new_contents = [line + '\n' for line in new_contents]

    # Now for validation, try to parse the new config file with the
    # PISAConfigParser
    pcp = PISAConfigParser()
    if missing_section_header:
        pcp.read_string(
            (
                ''.join(['[dummy section header]\n'] + new_contents) + '\n'
            ).decode('utf-8')
        )
    else:
        pcp.read_string((''.join(new_contents)).decode('utf-8'))

    if new_contents == orig_contents:
        sys.stdout.write('Nothing modified in the original file (ok!).\n')
        return

    if args.validate_only:
        raise ValueError(
            'Original config file "%s" would be modfied (and so may be'
            ' invalid). Re-run this script without the --validate-only flag to'
            ' produce an appropriately-converted config file.'
            % args.config[0]
        )

    sys.stdout.write('Writing modified config file to "%s"\n' % out_fpath)
    with open(out_fpath, 'w') as outfile:
        outfile.writelines(new_contents)


if __name__ == '__main__':
    main()
