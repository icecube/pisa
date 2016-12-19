#! /usr/bin/env python
# author: S.Wren
# date:   November 15, 2016
"""
Look in the PISA installation's pipeline settings directory for any example
pipeline configs (*example*.cfg) and run all of them to ensure that their
functionality remains intact. Note that this only tests that they run but does
not test that the generated outputs are necessarily correct (this is up to the
user).
"""


from argparse import ArgumentParser
import glob
import os
import sys
from traceback import format_exception

from pisa.core.pipeline import Pipeline
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource


__all__ = ['parse_args', 'main']


def parse_args():
    parser = ArgumentParser(description=__doc__)
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
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_verbosity(args.v)

    example_directory = os.path.join(
        'settings', 'pipeline'
    )
    example_directory = find_resource(example_directory)
    settings_files = glob.glob(example_directory + '/*example*.cfg')

    num_configs = len(settings_files)
    failure_count = 0
    skip_count = 0
    for settings_file in settings_files:
        # aeff smooth stage is currently broken. So ignore for now.
        if 'aeffsmooth' in settings_file:
            skip_count += 1
            logging.warn('Skipping "%s" as it is currently expected to be'
                         ' broken.' % settings_file)
            settings_files.remove(settings_file)

    for settings_file in settings_files:
        allow_error = False
        msg = ''
        try:
            logging.info('Instantiating pipeline from file "%s" ...'
                         %settings_file)
            pipeline = Pipeline(settings_file)
            logging.info('    retrieving outputs...')
            _ = pipeline.get_outputs()

        except ImportError as err:
            exc = sys.exc_info()
            if 'ROOT' in err.message and args.ignore_root:
                skip_count += 1
                allow_error = True
                msg = ('    Skipping pipeline as it has ROOT dependencies'
                       ' (ROOT cannot be imported)')
            elif 'cuda' in err.message and args.ignore_gpu:
                skip_count += 1
                allow_error = True
                msg = ('    Skipping pipeline as it has cuda dependencies'
                       ' (pycuda cannot be imported)')
            else:
                failure_count += 1

        except:
            exc = sys.exc_info()
            failure_count += 1

        else:
            exc = None

        finally:
            if exc is not None:
                if allow_error:
                    logging.warn(msg)
                else:
                    logging.error(
                        '    FAILURE! %s failed to run. Please review the'
                        ' error message below and fix the problem. Continuing'
                        ' with any other configs now...' % settings_file
                    )
                    for line in format_exception(*exc):
                        for sub_line in line.splitlines():
                            logging.error(' '*4 + sub_line)
            else:
                logging.info('    Seems fine!')

    if skip_count > 0:
        logging.warn('%d of %d example pipeline config files were skipped'
                     % (skip_count, num_configs))

    if failure_count > 0:
        msg = ('<< FAIL : test_example_pipelines : (%d of %d EXAMPLE PIPELINE'
               ' CONFIG FILES FAILED) >>' % (failure_count, num_configs))
        logging.error(msg)
        raise Exception(msg)

    logging.info('<< PASS : test_example_pipelines >>')


if __name__ == '__main__':
    main()
