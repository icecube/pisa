"""
Tools to obtain resource files needed for PISA, whether the resource is located
in the filesystem or with the installed PISA package.
"""


from __future__ import absolute_import

from os import environ
from os.path import exists, expanduser, expandvars, join
import sys

import pkg_resources


__all__ = ['RESOURCES_SUBDIRS', 'find_resource', 'open_resource', 'find_path']

__author__ = 'S. Boeser, J.L. Lanfranchi'

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


RESOURCES_SUBDIRS = ['data', 'scripts', 'settings']


def find_resource(resource, fail=True):
    """Try to find a resource (file or directory).

    First check if `resource` is an absolute path, then check relative to
    the paths specified by the PISA_RESOURCES environment variable (if it is
    defined). Otherwise, look in the resources directory of the PISA
    installation.

    Note that the PISA_RESOURCES environment variable can contain multiple
    paths, each separated by a colon. Due to using colons as separators,
    however, the paths themselves can *not* contain colons.

    Note also if PISA is packaged as archive distribution (e.g. zipped egg),
    this method extracts the resource (if directory, the entire contents are
    extracted) to a temporary cache directory. Therefore, it is preferable to
    use the `open_resource` method directly, and avoid this method if possible.


    Parameters
    ----------
    resource : str
        Resource path; can be path relative to CWD, path relative to
        PISA_RESOURCES environment variable (if defined), or a package resource
        location relative to the `pisa_examples/resources` sub-directory. Within
        each path specified in PISA_RESOURCES and within the
        `pisa_examples/resources` dir, the sub-directories 'data', 'scripts',
        and 'settings' are checked for `resource` _before_ the base directories
        are checked. Note that the **first** result found is returned.

    fail : bool
        If True, raise IOError if resource not found
        If False, return None if resource not found


    Returns
    -------
    String if `resource` is found (relative path to the file or directory); if
    not found and `fail` is False, returns None.


    Raises
    ------
    IOError if `resource` is not found and `fail` is True.

    """
    # NOTE: this import needs to be here -- and not at top -- to avoid circular
    # imports
    import pisa.utils.log as log

    log.logging.trace('Attempting to find resource "%s"', resource)

    # 1) Check for file in filesystem at absolute path or relative to
    #    PISA_RESOURCES environment var
    resource_path = find_path(resource, fail=False)
    if resource_path is not None:
        return resource_path

    # TODO: use resource_string or resource_stream instead, so that this works
    # with egg distributions

    # 2) Look inside the installed pisa package
    log.logging.trace('Searching package resources...')
    resource_spec = ('pisa_examples', 'resources/' + resource)
    if pkg_resources.resource_exists(*resource_spec):
        resource_path = pkg_resources.resource_filename(*resource_spec)
        log.logging.debug('Found resource "%s" in PISA package at "%s"',
                          resource, resource_path)
        return resource_path

    for subdir in RESOURCES_SUBDIRS + [None]:
        if subdir is None:
            augmented_path = resource
        else:
            augmented_path = '/'.join([subdir, resource])

        resource_spec = ('pisa_examples', 'resources/' + augmented_path)
        if pkg_resources.resource_exists(*resource_spec):
            resource_path = pkg_resources.resource_filename(*resource_spec)
            log.logging.debug('Found resource "%s" in PISA package at "%s"',
                              resource, resource_path)
            return resource_path

    # 3) If you get here, the resource is nowhere to be found
    msg = ('Could not find resource "%s" in filesystem OR in PISA package.'
           % resource)
    if fail:
        raise IOError(msg)
    log.logging.debug(msg)


def open_resource(resource, mode='r'):
    """Find the resource file (see find_resource), open it, and return a file
    handle.


    Parameters
    ----------
    resource : str
        Resource path; can be path relative to CWD, path relative to
        PISA_RESOURCES environment variable (if defined), or a package resource
        location relative to PISA's `pisa_examples/resources` sub-directory.
        Within each path specified in PISA_RESOURCES and within the
        `pisa_examples/resources` dir, the sub-directories 'data', 'scripts',
        and 'settings' are checked for `resource` _before_ the base directories
        are checked. Note that the **first** result found is returned.

    mode : str
        'r', 'w', or 'rw'; only 'r' is valid for package resources (as these
        cannot be written)


    Returns
    -------
    binary stream object (which behaves identically to a file object)


    See Also
    --------
    find_resource
        Locate a file or directory (in fileystem) or a package
        resource

    find_path
        Locate a file or directory in the filesystem

        Open a (file) package resource and return stream object.

    Notes
    -----
    See help for pkg_resources module / resource_stream method for more details
    on handling of package resources.

    """
    # NOTE: this import needs to be here -- and not at top -- to avoid circular
    # imports
    import pisa.utils.log as log

    log.logging.trace('Attempting to open resource "%s"', resource)

    # 1) Check for file in filesystem at absolute path or relative to
    #    PISA_RESOURCES environment var
    fs_exc_info = None
    try:
        resource_path = find_path(resource, fail=True)
    except IOError:
        fs_exc_info = sys.exc_info()
    else:
        log.logging.debug('Opening resource "%s" from filesystem at "%s"',
                          resource, resource_path)
        return open(resource_path, mode=mode)

    # 2) Look inside the installed pisa package; this should error out if not
    #    found
    log.logging.trace('Searching package resources...')
    pkg_exc_info = None
    for subdir in RESOURCES_SUBDIRS + [None]:
        if subdir is None:
            augmented_path = resource
        else:
            augmented_path = '/'.join([subdir, resource])
        try:
            resource_spec = ('pisa_examples', 'resources/' + augmented_path)
            stream = pkg_resources.resource_stream(*resource_spec)
            # TODO: better way to check if read mode (i.e. will 'r' miss
            # anything that can be specified to also mean "read mode")?
            if mode.strip().lower() != 'r':
                del stream
                raise IOError(
                    'Illegal mode "%s" specified. Cannot open a PISA package'
                    ' resource in anything besides "r" (read-only) mode.' %mode
                )
        except IOError:
            pkg_exc_info = sys.exc_info()
        else:
            log.logging.debug('Opening resource "%s" from PISA package.',
                              resource)
            return stream

    if fs_exc_info is not None:
        if pkg_exc_info is not None:
            msg = ('Could not locate resource "%s" in filesystem OR in'
                   ' installed PISA package.' %resource)
            raise IOError(msg)
        raise fs_exc_info[0](fs_exc_info[1]).with_traceback(fs_exc_info[2])
    raise pkg_exc_info[0](pkg_exc_info[1]).with_traceback(pkg_exc_info[2])


def find_path(pathspec, fail=True):
    """Find a file or directory in the filesystem (i.e., something that the
    operating system can locate, as opposed to Python package resources, which
    can be located within a package and therefore hidden from the filesystem).

    Parameters
    ----------
    pathspec : string
    fail : bool

    Returns
    -------
    None (if not found) or string (absolute path to file or dir if found)

    """
    # NOTE: this import needs to be here -- and not at top -- to avoid circular
    # imports
    import pisa.utils.log as log

    # 1) Check for absolute path or path relative to current working
    #    directory
    log.logging.trace('Checking absolute or path relative to cwd...')
    resource_path = expandvars(expanduser(pathspec))
    if exists(resource_path):
        log.logging.debug('Found "%s" at "%s"', pathspec, resource_path)
        return resource_path

    # 2) Check if $PISA_RESOURCES is set in environment; if so, look relative
    #    to that
    log.logging.trace('Checking environment for $PISA_RESOURCES...')
    if 'PISA_RESOURCES' in environ:
        pisa_resources = environ['PISA_RESOURCES']
        log.logging.trace('Searching resource path PISA_RESOURCES=%s',
                          pisa_resources)
        resource_paths = pisa_resources.split(':')
        for resource_path in resource_paths:
            if not resource_path:
                continue
            resource_path = expandvars(expanduser(resource_path))
            # Look in all default sub-dirs for the pathspec
            augmented_paths = [join(resource_path, subdir, pathspec)
                               for subdir in RESOURCES_SUBDIRS]
            # Also look in the base dir specified for the pathspec
            augmented_paths.append(join(resource_path, pathspec))

            for augmented_path in augmented_paths:
                if exists(augmented_path):
                    log.logging.debug('Found path "%s" at %s',
                                      pathspec, augmented_path)
                    return augmented_path

    # 3) If you get here, the file is nowhere to be found
    msg = 'Could not find path "%s"' % pathspec
    if fail:
        raise IOError(msg)
    log.logging.trace(msg)
    return None
