#!/usr/bin/env python3
'''Benchmark pipeline execution times, either for preset or custom ones'''

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import datetime
import os
from pathlib import Path
import sys

import cpuinfo
import numpy as np

from pisa import TARGET, PISA_NUM_THREADS
from pisa.core.distribution_maker import Pipeline
from pisa.utils.fileio import to_file
from pisa.utils.log import Levels, logging, set_verbosity

PIPELINE_CFGS_TO_TEST = ["settings/pipeline/IceCube_3y_neutrinos_daemon.cfg",
"settings/pipeline/IceCube_3y_neutrinos.cfg", "settings/pipeline/IceCube_3y_muons.cfg"
]

NTEMPLATES = 50
"""Number of random Asimov templates to produce by default (no caching)"""

PFX = "[B] "
"""Prefix each line output by this script to clearly delineate output from this
script vs. output from test functions being run"""


def get_get_outputs_time(pipeline):
    """
    Just extract minimum+average get_outputs() execution time from a profiled pipeline.
    
    Note that first call to get_outputs() typically includes compilation overhead
    (numba JIT). We thus just exclude it.
    
    Parameters
    ----------
    pipeline : pisa.core.pipeline.Pipeline
        pipeline instance with profile=True that has been run via get_outputs()
    
    Returns
    -------
    minim, avg, maxim : float
        Minimum, average, maximum get_outputs() time in seconds.
    
    Raises
    ------
    ValueError
        If the pipeline has no recorded get_outputs times.
    """
    times = pipeline._get_outputs_times # pylint: disable=protected-access

    if not times:
        raise ValueError(
            "No get_outputs() times recorded. Ensure pipeline was created with "
            "profile=True and get_outputs() was called."
        )

    # Return minimum and average time across all calls (skip first if multiple
    # to avoid contribution from overhead)
    if len(times) > 1:
        minim, avg, maxim = (float(np.min(times[1:])),
            float(np.mean(times[1:])), float(np.max(times[1:]))
        )
    else:
        minim = avg = maxim = float(times[0])
    return minim, avg, maxim


def create_benchmark_result(pipeline_config_name, target, nthreads, time_s, range_s):
    """
    Create a single benchmark result entry in github-action-benchmark format.
    
    Parameters
    ----------
    pipeline_config_name : str
        Name/path of the pipeline config being benchmarked
    target : str
        PISA_TARGET value ('cpu' or 'parallel')
    nthreads : int
        PISA_NUM_THREADS value
    time_s : float
        Execution time metric (average) in seconds
    range_s : float
        Execution time variation in seconds
    
    Returns
    -------
    dict
        Benchmark entry suitable for github-action-benchmark JSON output
    """
    cfg_basename = Path(pipeline_config_name).stem
    name = f"{cfg_basename} ({target}, nthreads={nthreads})"

    return {
        "name": name,
        "value": time_s,
        "unit": "s",
        "range": range_s,
        "extra": f"target={target}, nthreads={nthreads}"
    }


def write_benchmark_json(results, output_path, commit_sha=None, commit_msg=None):
    """
    Write benchmark results to JSON file in github-action-benchmark format.
    Adds timestamp and possibly commit information for improved manual tracking.
    
    Parameters
    ----------
    results : list of dict
        List of benchmark result dictionaries (from create_benchmark_result)
    output_path : str or Path
        Path where JSON file should be written
    commit_sha : str, optional
        Git commit SHA (for provenance tracking)
    commit_msg : str, optional
        Git commit message (for provenance tracking)
    
    Returns
    -------
    Path
        Path to written JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
        "results": results,
    }

    if commit_sha:
        data["commit_sha"] = commit_sha
    if commit_msg:
        data["commit_msg"] = commit_msg

    to_file(data, output_path)

    return output_path


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

    # Collect benchmark results for JSON output
    benchmark_results = []

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

        # Extract min./avg.execution times
        min_time, avg_time, max_time = get_get_outputs_time(pipeline)

        # Create the benchmark results, using just the average
        result_avg = create_benchmark_result(
            pipeline_config_name=cfg,
            target=TARGET,
            nthreads=PISA_NUM_THREADS,
            time_s=avg_time,
            range_s=max_time - min_time
        )
        benchmark_results.append(result_avg)

    # Prepare JSON output
    output_dir = Path("benchmark_results")
    output_file = ( output_dir /
        f"results_target_{TARGET}_nthreads_{PISA_NUM_THREADS}.json"
    )

    # Enrich with commit information
    commit_sha = os.environ.get("GITHUB_SHA")
    commit_msg = os.environ.get("GITHUB_COMMIT_MSG")

    write_benchmark_json(
        benchmark_results,
        output_file,
        commit_sha=commit_sha,
        commit_msg=commit_msg
    )
    logging.info("%sBenchmark results written to: %s", PFX, output_file)


if __name__ == '__main__':
    main()
