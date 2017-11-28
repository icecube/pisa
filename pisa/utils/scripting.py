"""
Tools for scripts
"""


from __future__ import absolute_import

from argparse import ArgumentParser
import inspect
from os.path import isdir, isfile
import sys

from pisa.utils.resources import find_resource


__all__ = ['get_script', 'normcheckpath', 'parse_command']

__author__ = 'J.L. Lanfranchi, S. Wren'


def get_script():
    """If called from a script, this will return its filename"""
    return inspect.stack()[-1][1]


# TODO: make this work with Python package resources, not merely absolute
# paths! ... e.g. hash on the file or somesuch?
def normcheckpath(path, checkdir=False):
    """Normalize a path and check that it exists.

    Useful e.g. so that different path specifications that resolve to the same
    location will resolve to the same string.

    Parameters
    ----------
    path : string
        Path to check

    checkdir : bool
        Whether `path` is expected to be a directory

    Returns
    -------
    normpath : string
        Normalized path

    """
    normpath = find_resource(path)
    if checkdir:
        kind = 'dir'
        check = isdir
    else:
        kind = 'file'
        check = isfile

    if not check(normpath):
        raise IOError('Path "%s" which resolves to "%s" is not a %s.'
                      %(path, normpath, kind))
    return normpath


def parse_command(command_depth, commands, description, usage,
                  return_outputs=False):
    """Parse command line argument for a (...(sub))command and call the
    coresponding function.

    Parameters
    ----------
    command_depth : int
        The command is expected at this position in the command line, relative
        to the script itself. E.g., `command_depth`=0 would access the
        command in
          <script> <command>
        while `command_depth`=1 would access the subcommand in
          <script> <command> <subcommand>

    commands : Mapping
        Keys should be commands (all lower-case, no whitespace) and values
        should be callables that take the argument `return_outputs` (a bool).

    description : string
        Description of the overall script

    usage : string
        Usage that should include the valid (sub)commands that can be called

    return_outputs : bool
        Passed to the callable corresponding with the (sub)command

    Returns
    -------
    outputs : None or object
        If `return_outputs` is True, some output _could_ be returned; if
        `return_outputs` is False, then no output is returned (i.e., None).

    """
    name = 'sub'*command_depth + 'command'
    parser = ArgumentParser(description=description, usage=usage)
    parser.add_argument('command', help='{} to run'.format(name.capitalize()))
    args = parser.parse_args(sys.argv[1+command_depth:2+command_depth])
    command = args.command.strip().lower()
    if command not in commands.keys():
        raise ValueError(
            'The {} issued, "{}", is not valid; expecting one of {}'
            .format(name, args.command, commands.keys())
        )
    return commands[command](return_outputs=return_outputs)
