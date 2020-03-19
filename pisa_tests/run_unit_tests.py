#!/usr/bin/env python
# pylint: disable=exec-used, eval-used


"""
Find and run PISA unit test functions
"""


from argparse import ArgumentParser
from os import walk
from os.path import dirname, expanduser, expandvars, isfile, join, relpath

import pisa
from pisa.utils.fileio import nsort_key_func
from pisa.utils.log import Levels, logging, set_verbosity


# TODO: is it a problem to leave already-imported modules imported? I.e., any
# issues we might miss from not doing a "fresh" import of a module?


__all__ = ["PISA_PATH", "run_unit_tests", "find_unit_tests", "find_unit_tests_in_file"]


__author__ = "J.L. Lanfranchi"

__license__ = """Copyright (c) 2020, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


PISA_PATH = dirname(pisa.__file__)


def run_unit_tests(path=PISA_PATH, verbosity=Levels.WARN):
    """Run all tests found at `path` (or recursively below if `path` is a
    directory).

    Each module is imported and each test function is run initially with
    `set_verbosity(verbosity)`, but if an exception is caught, the module is
    re-imported or the test function is re-run with
    `set_verbosity(Levels.TRACE)`, then the traceback from the (original)
    exception emitted is displayed.

    Parameters
    ----------
    path : str
        Path to file or directory

    Raises
    ------
    Exception
        If any import or test fails

    """
    path = expanduser(expandvars(path))

    tests = find_unit_tests(path)

    module_pypaths_succeeded = []
    module_pypaths_failed = []
    test_pypaths_succeeded = []
    test_pypaths_failed = []

    for rel_file_path, test_func_names in tests.items():
        pypath = ["pisa"] + rel_file_path[:-3].split("/")
        parent_pypath = ".".join(pypath[:-1])
        module = pypath[-1].replace(".", "_")
        module_pypath = f"{parent_pypath}.{module}"

        try:
            cmd = f"from {parent_pypath} import {module}"

            set_verbosity(verbosity)
            logging.info(f'exec("{cmd}") ...')

            set_verbosity(Levels.WARN)
            exec(cmd)

        except Exception as err:
            module_pypaths_failed.append(module_pypath)

            set_verbosity(verbosity)
            msg = f"<< FAILURE IMPORTING : {module_pypath} >>"
            logging.error("=" * len(msg))
            logging.error(msg)
            logging.error("=" * len(msg))

            # Reproduce the failure with full output
            set_verbosity(Levels.TRACE)
            try:
                exec(f"from {parent_pypath} import {module}")
            except Exception:
                pass

            set_verbosity(Levels.TRACE)
            logging.exception(err)

            set_verbosity(verbosity)
            logging.error("#" * len(msg))

            continue

        else:
            module_pypaths_succeeded.append(module_pypath)

        for test_func_name in test_func_names:
            test_pypath = f"{module_pypath}.{test_func_name}"

            try:
                func_pypath = f"{module}.{test_func_name}"

                set_verbosity(verbosity)
                logging.debug(f"Retrieving function: {func_pypath} ...")

                set_verbosity(Levels.WARN)
                test_func = eval(func_pypath)

                set_verbosity(verbosity)
                logging.debug(f"Running function: {func_pypath}() ...")

                set_verbosity(Levels.WARN)
                test_func()

            except Exception as err:
                test_pypaths_failed.append(test_pypath)

                set_verbosity(verbosity)
                msg = f"<< FAILURE RUNNING : {test_pypath} >>"
                logging.error("=" * len(msg))
                logging.error(msg)
                logging.error("=" * len(msg))

                # Reproduce the error with full output

                set_verbosity(Levels.TRACE)
                try:
                    test_func = eval(f"{module}.{test_func_name}")
                    test_func()
                except Exception:
                    pass

                set_verbosity(Levels.TRACE)
                logging.exception(err)

                set_verbosity(verbosity)
                logging.error("#" * len(msg))

            else:
                test_pypaths_succeeded.append(test_pypath)

    # Summarize results

    n_import_successes = len(module_pypaths_succeeded)
    n_import_failures = len(module_pypaths_failed)
    n_test_successes = len(test_pypaths_succeeded)
    n_test_failures = len(test_pypaths_failed)

    set_verbosity(verbosity)
    logging.info(
        f"<< IMPORT TESTS : {n_import_successes} modules loaded,"
        f" {n_import_failures} modules failed to load >>"
    )
    logging.info(
        f"<< UNIT TESTS : {n_test_successes} tests succeeded,"
        f" {n_test_failures} tests failed >>"
    )

    # Exit with error if any failures (import or unit test)

    if module_pypaths_failed or test_pypaths_failed:
        msgs = []
        if module_pypaths_failed:
            msgs.append(
                f"{n_import_failures} module(s) failed to import:\n  "
                + ", ".join(module_pypaths_failed)
            )
        if test_pypaths_failed:
            msgs.append(
                f"{n_test_failures} unit test(s) failed:\n  "
                + ", ".join(test_pypaths_failed)
            )
        raise Exception("\n".join(msgs))


def find_unit_tests(path):
    """Find .py file(s) and any tests to run within them, starting at `path`
    (which can be a single file or a directory, which is recursively searched
    for .py files)

    Parameters
    ----------
    path : str
        Path to a file or directory

    Returns
    -------
    tests : dict
        Each key is the path to the .py file relative to PISA_PATH and each
        value is a list of the "test_*" function names within that file (empty
        if no such functions are found)

    """
    path = expanduser(expandvars(path))

    tests = {}
    if isfile(path):
        return find_unit_tests_in_file(path)

    for dirpath, dirs, files in walk(path, followlinks=True):
        files.sort(key=nsort_key_func)
        dirs.sort(key=nsort_key_func)

        for filename in files:
            if not filename.endswith(".py"):
                continue
            filepath = join(dirpath, filename)
            filerelpath = relpath(filepath, start=PISA_PATH)
            tests[filerelpath] = find_unit_tests_in_file(filepath)

    return tests


def find_unit_tests_in_file(filepath):
    """Find test functions defined by "def test_*" within a file at `filepath`

    Parameters
    ----------
    filepath : str
         Path to python file

    Returns
    -------
    tests : list of str

    """
    tests = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            tokens = line.split()
            if tokens and tokens[0] == "def" and tokens[1].startswith("test_"):
                funcname = tokens[1].split("(")[0].strip()
                tests.append(funcname)
    return tests


def main(description=__doc__):
    """Script interface to `run_unit_tests` function"""
    parser = ArgumentParser(description=description)
    parser.add_argument("--path", default=PISA_PATH)
    parser.add_argument(
        "-v", action="count", default=Levels.WARN, help="set verbosity level"
    )
    kwargs = vars(parser.parse_args())
    kwargs["verbosity"] = kwargs.pop("v")
    run_unit_tests(**kwargs)


if __name__ == "__main__":
    main()
