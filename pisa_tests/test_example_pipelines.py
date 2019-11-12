#! /usr/bin/env python

"""
Look in the PISA installation's pipeline settings directory for any example
pipeline configs (*example*.cfg) and run all of them to ensure that their
functionality remains intact. Note that this only tests that they run but does
not test that the generated outputs are necessarily correct (this is up to the
user).
"""


from __future__ import absolute_import

from argparse import ArgumentParser
import glob
import sys
import re
from traceback import format_exception

from pisa.core.pipeline import Pipeline
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource


__all__ = ['test_example_pipelines', 'parse_args', 'main']

__author__ = 'S. Wren, J.L. Lanfranchi'

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


def test_example_pipelines(ignore_gpu=False, ignore_root=False,
                           ignore_missing_data=False):
    """Run example pipelines.

    Parameters
    ----------
    ignore_gpu : bool
        Do not count errors initializing a GPU as failures

    ignore_root : bool
        Do not count errors importing ROOT as failures

    ignore_missing_data : bool
        Do not count errors due to missing data files as failures

    """
    # Set up the lists of strings needed to search the error messages for
    # things to ignore e.g. cuda stuff and ROOT stuff
    root_err_strings = ['ROOT', 'Roo', 'root', 'roo']
    cuda_err_strings = ['cuda']
    missing_data_string = ('Could not find resource "(.*)" in'
                           ' filesystem OR in PISA package.')

    example_directory = find_resource('settings/pipeline')
    settings_files = glob.glob(example_directory + '/*example*.cfg')

    num_configs = len(settings_files)
    failure_count = 0
    skip_count = 0

    for settings_file in settings_files:
        allow_error = False
        msg = ''
        try:
            logging.info('Instantiating pipeline from file "%s" ...',
                         settings_file)
            pipeline = Pipeline(settings_file)
            logging.info('    retrieving outputs...')
            _ = pipeline.get_outputs()

        except ImportError as err:
            exc = sys.exc_info()
            if any(errstr in err.message for errstr in root_err_strings) and \
              ignore_root:
                skip_count += 1
                allow_error = True
                msg = ('    Skipping pipeline, %s, as it has ROOT dependencies'
                       ' (ROOT cannot be imported)'%settings_file)
            elif any(errstr in err.message for errstr in cuda_err_strings) and \
              ignore_gpu:
                skip_count += 1
                allow_error = True
                msg = ('    Skipping pipeline, %s, as it has cuda dependencies'
                       ' (pycuda cannot be imported)'%settings_file)
            else:
                failure_count += 1

        except IOError as err:
            exc = sys.exc_info()
            match = re.match(missing_data_string, err.message, re.M|re.I)
            if match is not None and ignore_missing_data:
                skip_count += 1
                allow_error = True
                msg = ('    Skipping pipeline, %s, as it has data that cannot'
                       ' be found in the local PISA environment'%settings_file)
            else:
                failure_count += 1

        except: # pylint: disable=bare-except
            exc = sys.exc_info()
            failure_count += 1

        else:
            exc = None

        finally:
            if exc is not None:
                if allow_error:
                    logging.warning(msg)
                else:
                    logging.error(
                        '    FAILURE! %s failed to run. Please review the'
                        ' error message below and fix the problem. Continuing'
                        ' with any other configs now...', settings_file
                    )
                    for line in format_exception(*exc):
                        for sub_line in line.splitlines():
                            logging.error(' '*4 + sub_line)
            else:
                logging.info('    Seems fine!')

    if skip_count > 0:
        logging.warning(
            '%d of %d example pipeline config files were skipped',
            skip_count, num_configs
        )

    if failure_count > 0:
        msg = ('<< FAIL : test_example_pipelines : (%d of %d EXAMPLE PIPELINE'
               ' CONFIG FILES FAILED) >>' % (failure_count, num_configs))
        logging.error(msg)
        raise Exception(msg)

    logging.info('<< PASS : test_example_pipelines >>')


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--ignore-gpu', action='store_true', default=False,
        help='''Skip the pipelines which require a gpu to run. You will
        need to flag this if your system does not have a gpu else it
        will fail.'''
    )
    parser.add_argument(
        '--ignore-root', action='store_true', default=False,
        help='''Skip the pipelines which require ROOT to run. You will
        need to flag this if your system does not have an installation
        of ROOT that your python can find else it will fail.'''
    )
    parser.add_argument(
        '--ignore-missing-data', action='store_true', default=False,
        help='''Skip the pipeline which fail because you do not have the
        necessary data files in the right locations for your local PISA
        installation. This is NOT recommended and you should probably acquire
        the missing datafiles somehow.'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    args = parser.parse_args()
    return args


def main():
    """main"""
    args = parse_args()
    kwargs = vars(args)
    set_verbosity(kwargs.pop('v'))
    test_example_pipelines(**kwargs)


if __name__ == '__main__':
    main()
