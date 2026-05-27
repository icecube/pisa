#!/usr/bin/env python3
'''Benchmark pipeline execution times, either for preset or custom ones'''

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import sys

import cpuinfo

from pisa.core.distribution_maker import Pipeline
from pisa.utils.log import Levels, logging, set_verbosity

PIPELINE_CFGS_TO_TEST = ["settings/pipeline/IceCube_3y_neutrinos_daemon.cfg",
"settings/pipeline/IceCube_3y_neutrinos.cfg", "settings/pipeline/IceCube_3y_muons.cfg"
]

NTEMPLATES = 100
"""Number of random Asimov templates to produce by default (no caching)"""

PFX = "[B] "
"""Prefix each line output by this script to clearly delineate output from this
script vs. output from test functions being run"""


def parse_args():
    """Parse command-line arguments"""
    parser = ArgumentParser(description='''Benchmark time it takes to run preset '''
        f'''pipeline configurations ({PIPELINE_CFGS_TO_TEST}) or custom ones.''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-n', type=int, default=NTEMPLATES,
         help='No. of random (=no caching) Asimov templates to produce.'
    )
    parser.add_argument(
        '-p', type=str, action='append', default=None,
        help='''Custom pipelines to benchmark instead of the preset ones. '''
        '''Repeat for multiple.'''
    )
    return parser.parse_args()


def main():
    """Function to run when script is executed"""
    args = parse_args()

    set_verbosity(Levels.INFO)
    logging.info("%sPython build: %s", PFX, sys.version)
    for key, val in cpuinfo.get_cpu_info().items():
        logging.info("%s%s = %s", PFX, key, val)

    test_cfgs = PIPELINE_CFGS_TO_TEST if args.p is None else args.p
    for cfg in test_cfgs:
        logging.info("%sObtaining timings for pipeline %s...", PFX, cfg)
        set_verbosity(Levels.WARN)
        pipeline = Pipeline(cfg, profile=True)

         # Randomize all of the free parameter values n times
         # and get the corresponding outputs
        for seed in range(args.n):
            pipeline.params.randomize_free(random_state=seed)
            _ = pipeline.get_outputs()

        set_verbosity(Levels.INFO)
        logging.info("%s%s:", PFX, os.path.basename(cfg))
        pipeline.report_profile(detailed=False)
        print("\n")


if __name__ == '__main__':
    main()
