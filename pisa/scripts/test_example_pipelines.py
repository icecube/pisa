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

    for settings_file in settings_files:
        # aeff smooth stage is currently broken. So ignore for now.
        if 'smooth' in settings_file:
            settings_files.remove(settings_file)

    for settings_file in settings_files:
        try:
            logging.info('Testing %s'%settings_file)
            logging.info('Instantiating pipeline...')
            pipeline = Pipeline(settings_file)
            logging.info('Retrieving outputs...')
            outputs = pipeline.get_outputs()
            logging.info('Seems fine!')
        except ImportError as err:
            if 'ROOT' in err.message:
                if args.ignore_root:
                    logging.info('Skipping pipeline as it has ROOT '
                                 'dependencies')
                else:
                    logging.error(sys.exc_info())
                    logging.error('%s does not work. Error trying to import '
                                  'ROOT - use the flag "--ignore-root" to '
                                  'skip ROOT dependent pipelines.'
                                  %settings_file)
                    raise
            elif 'cuda' in err.message:
                if args.ignore_gpu:
                    logging.info('Skipping pipeline as it has GPU '
                                 'dependencies')
                else:
                    logging.error(sys.exc_info())
                    logging.error('%s does not work. Error trying to import '
                                  'CUDA - use the flag "--ignore-gpu" to skip '
                                  'GPU dependent pipelines.' % settings_file)
                    raise
            else:
                logging.error(sys.exc_info())
                raise ValueError('%s does not work. Please review the error '
                                 'message above and fix the problem.'
                                 %settings_file)
        except:
            logging.error(sys.exc_info())
            raise ValueError('%s does not work. Please review the error '
                             'message above and fix the problem.'
                             %settings_file)


if __name__ == '__main__':
    main()
