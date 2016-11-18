#
# resources.py
#
# Tools to obtain one of the input resources files needed for PISA.
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-06-10

import os
from pkg_resources import resource_filename
import sys


__all__ = ['find_resource', 'open_resource']


# TODO: make this work with Python package resources, not just file paths (so
# we can distribute PISA as an egg).
def find_resource(resourcename, fail=True):
    """Try to find a resource (file or directory).

    First check if `resourcename` is an absolute path, then relative to the
    $PISA environment variable if it is set. Otherwise, look in the resources
    directory of the pisa installation.

    Parameters
    ----------
    resourcename : str
        File or directory name to locate. The `resourcename` can include an
        absolute or relative (to $PWD or within the $PISA/resources directory)
        path. E.g., 'directory/resourcename'.

    fail : bool
        If True, raise IOError if resourcename not found
        If False, return None if resourcename not found

    Returns
    -------
    String if `resource` is found (relative path to the file or directory); if
    not found and `fail` is False, returns None.

    Raises
    ------
    IOError if `resource` is not found and `fail` is True.

    """
    import pisa.utils.log as log

    log.logging.trace('Attempting to locate `resourcename` "%s"' %resourcename)
    # 1) Check for absolute path or path relative to current working
    #    directory
    log.logging.trace('Checking absolute or path relative to cwd...')
    rsrc_path = os.path.expandvars(os.path.expanduser(resourcename))
    if os.path.isfile(rsrc_path) or os.path.isdir(rsrc_path):
        log.logging.debug('Found %s at %s' % (resourcename, rsrc_path))
        return rsrc_path

    # 2) Check if $PISA is set in environment, and look relative to that
    log.logging.trace('Checking environment for $PISA...')
    if 'PISA' in os.environ:
        rpath = os.path.expandvars(os.path.expanduser(os.environ['PISA']))
        log.logging.trace('Searching resource path PISA=%s' % rpath)

        rsrc_path = os.path.join(rpath, resourcename)
        if os.path.isfile(rsrc_path) or os.path.isdir(rsrc_path):
            log.logging.debug('Found %s at %s' % (resourcename, rsrc_path))
            return rsrc_path

    # TODO: use resource_string or resource_stream instead, so that this work
    # swith egg distributions

    # 3) Look inside the installed pisa package
    log.logging.trace('Searching package resources...')
    rsrc_path = resource_filename('pisa', 'resources/' + resourcename)
    if os.path.isfile(rsrc_path) or os.path.isdir(rsrc_path):
        rsrc_path = os.path.relpath(rsrc_path)
        log.logging.debug('Found %s at %s' % (resourcename, rsrc_path))
        return rsrc_path

    # Nowhere to be found
    msg = 'Could not find resource "%s"' % resourcename
    if fail:
        raise IOError(msg)
    log.logging.debug(msg)
    return None


def open_resource(filename, mode='r'):
    """Find the resource file (see find_resource), open it and return a file
    handle.
    """
    return open(find_resource(filename, fail=True), mode=mode)

