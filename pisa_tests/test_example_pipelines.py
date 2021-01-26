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

from pisa.core.pipeline import Pipeline
from pisa.utils.log import Levels, logging, set_verbosity
from pisa.utils.resources import find_resource


__all__ = ["test_example_pipelines", "parse_args", "main"]

__author__ = "S. Wren, J.L. Lanfranchi"

__license__ = """Copyright (c) 2014-2020, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


def test_example_pipelines(path="settings/pipeline", verbosity=Levels.WARN):
    """Run pipelines from any "*.cfg" config files found at `path`."""
    path = find_resource(path)
    settings_files = glob.glob(path + "/*.cfg")

    failures = []
    successes = []

    for settings_file in settings_files:
        try:
            # NOTE: Force output of info on which settings file is being
            # instantiated and run, as warnings emitted by individual stages
            # are not as useful if we don't know which pipeline config is being
            # run

            set_verbosity(Levels.INFO)
            logging.info(f'Instantiating Pipeline with "{settings_file}" ...')

            set_verbosity(Levels.WARN)
            pipeline = Pipeline(settings_file)

            set_verbosity(Levels.INFO)
            logging.info(f'Running Pipeline instantiated from "{settings_file}" ...')

            set_verbosity(Levels.WARN)
            pipeline.get_outputs()

        except Exception as err:
            failures.append(settings_file)

            msg = f"<< FAILURE IN PIPELINE : {settings_file} >>"
            set_verbosity(verbosity)
            logging.error("=" * len(msg))
            logging.error(msg)
            logging.error("=" * len(msg))

            # Reproduce the error with full output

            set_verbosity(Levels.TRACE)
            try:
                pipeline = Pipeline(settings_file)
                pipeline.get_outputs()
            except Exception:
                pass

            set_verbosity(Levels.TRACE)
            logging.exception(err)

            set_verbosity(verbosity)
            logging.error("#" * len(msg))

        else:
            successes.append(settings_file)

        finally:
            set_verbosity(verbosity)

    # Summarize results

    set_verbosity(verbosity)
    logging.info(
        "<< EXAMPLE PIPELINES : "
        f"{len(successes)} succeeded and {len(failures)} failed >>"
    )

    # Exit with error if any failures

    if failures:
        raise Exception(
            f"{len(failures)} example pipeline(s) failed:\n  "
            + ", ".join(f'"{f}"' for f in failures)
        )


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument("--path", default="settings/pipeline")
    parser.add_argument(
        "-v", action="count", default=Levels.WARN, help="set verbosity level"
    )
    args = parser.parse_args()
    return args


def main():
    """Script interface to test_example_pipelines"""
    args = parse_args()
    kwargs = vars(args)
    kwargs["verbosity"] = kwargs.pop("v")
    test_example_pipelines(**kwargs)


if __name__ == "__main__":
    main()
