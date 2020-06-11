#!/usr/bin/env python


"""
Find and run PISA unit test functions
"""

from __future__ import absolute_import

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from importlib import import_module
from os import walk
from os.path import dirname, isfile, join, relpath
import platform
import socket
import sys

import cpuinfo
import numpy as np

import pisa
from pisa.utils.fileio import expand, nsort_key_func
from pisa.utils.log import Levels, logging, set_verbosity

pycuda, nbcuda = None, None  # pylint: disable=invalid-name
if pisa.TARGET == "cuda":
    try:
        import pycuda
    except Exception:
        pass

    # See TODO below
    # try:
    #    from numba import cuda as nbcuda
    # except Exception:
    #    pass


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

# TODO: add timing for imports & unit test; faster => more used, more useful

PISA_PATH = expand(dirname(pisa.__file__), absolute=True, resolve_symlinks=True)

# TODO: get optional & required automatically (e.g., from setup.py?)
OPTIONAL_MODULES = (
    "pandas",
    "emcee",
    "pycuda",
    "pycuda.driver",
    "ROOT",
    "libPyROOT",
    "MCEq",
    "nuSQUIDSpy",
)
"""Okay if imports or test_* functions fail due to these not being import-able"""

REQUIRED_MODULES = (
    "pisa",
    "pip",
    "setuptools",
    "numpy",
    "decorator",
    "kde",
    "h5py",
    "iminuit",
    "line_profiler",
    "matplotlib",
    "numba",
    "numpy",
    "pint",
    "scipy",
    "simplejson",
    "tables",
    "uncertainties",
    "llvmlite",
    "cpuinfo",
    "sympy",
    "cython",
)

PFX = "[T] "
"""Prefix each line output by this script to clearly delineate output from this
script vs. output from test functions being run"""


def run_unit_tests(
    path=PISA_PATH, allow_missing=OPTIONAL_MODULES, verbosity=Levels.WARN
):
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

    allow_missing : None or sequence of str

    verbosity : int in pisa.utils.log.Levels

    Raises
    ------
    Exception
        If any import or test fails not in `allow_missing`

    """
    set_verbosity(verbosity)
    logging.info("%sPlatform information:", PFX)
    logging.info("%s  HOSTNAME = %s", PFX, socket.gethostname())
    logging.info("%s  FQDN = %s", PFX, socket.getfqdn())
    logging.info("%s  OS = %s %s", PFX, platform.system(), platform.release())
    for key, val in cpuinfo.get_cpu_info().items():
        logging.info("%s  %s = %s", PFX, key, val)
    logging.info(PFX)
    logging.info("%sModule versions:", PFX)
    for module_name in REQUIRED_MODULES + OPTIONAL_MODULES:
        try:
            module = import_module(module_name)
        except ImportError:
            if module_name in REQUIRED_MODULES:
                raise
            ver = "optional module not installed or not import-able"
        else:
            if hasattr(module, "__version__"):
                ver = module.__version__
            else:
                ver = "?"
        logging.info("%s  %s : %s", PFX, module_name, ver)
    logging.info(PFX)

    path = expand(path, absolute=True, resolve_symlinks=True)
    if allow_missing is None:
        allow_missing = []
    elif isinstance(allow_missing, str):
        allow_missing = [allow_missing]

    tests = find_unit_tests(path)

    module_pypaths_succeeded = []
    module_pypaths_failed = []
    module_pypaths_failed_ignored = []
    test_pypaths_succeeded = []
    test_pypaths_failed = []
    test_pypaths_failed_ignored = []

    for rel_file_path, test_func_names in tests.items():
        pypath = ["pisa"] + rel_file_path[:-3].split("/")
        parent_pypath = ".".join(pypath[:-1])
        module_name = pypath[-1].replace(".", "_")
        module_pypath = f"{parent_pypath}.{module_name}"

        try:
            set_verbosity(verbosity)
            logging.info(PFX + f"importing {module_pypath}")

            set_verbosity(Levels.WARN)
            module = import_module(module_pypath, package=parent_pypath)

        except Exception as err:
            if (
                isinstance(err, ImportError)
                and hasattr(err, "name")
                and err.name in allow_missing  # pylint: disable=no-member
            ):
                err_name = err.name  # pylint: disable=no-member
                module_pypaths_failed_ignored.append(module_pypath)
                logging.warning(
                    f"{PFX}module {err_name} failed to import wile importing"
                    f" {module_pypath}, but ok to ignore"
                )
                continue

            module_pypaths_failed.append(module_pypath)

            set_verbosity(verbosity)
            msg = f"<< FAILURE IMPORTING : {module_pypath} >>"
            logging.error(PFX + "=" * len(msg))
            logging.error(PFX + msg)
            logging.error(PFX + "=" * len(msg))

            # Reproduce the failure with full output
            set_verbosity(Levels.TRACE)
            try:
                import_module(module_name, package=parent_pypath)
            except Exception:
                pass

            set_verbosity(Levels.TRACE)
            logging.exception(err)

            set_verbosity(verbosity)
            logging.error(PFX + "#" * len(msg))

            continue

        else:
            module_pypaths_succeeded.append(module_pypath)

        for test_func_name in test_func_names:
            test_pypath = f"{module_pypath}.{test_func_name}"

            try:
                set_verbosity(verbosity)
                logging.debug(PFX + f"getattr({module}, {test_func_name})")

                set_verbosity(Levels.WARN)
                test_func = getattr(module, test_func_name)

                # Run the test function
                set_verbosity(verbosity)
                logging.info(PFX + f"{test_pypath}()")

                set_verbosity(Levels.WARN)
                test_func()

            except Exception as err:
                if (
                    isinstance(err, ImportError)
                    and hasattr(err, "name")
                    and err.name in allow_missing  # pylint: disable=no-member
                ):
                    err_name = err.name  # pylint: disable=no-member
                    test_pypaths_failed_ignored.append(module_pypath)
                    logging.warning(
                        PFX
                        + f"{test_pypath} failed because module {err_name} failed to"
                        + f" load, but ok to ignore"
                    )
                    continue

                test_pypaths_failed.append(test_pypath)

                set_verbosity(verbosity)
                msg = f"<< FAILURE RUNNING : {test_pypath} >>"
                logging.error(PFX + "=" * len(msg))
                logging.error(PFX + msg)
                logging.error(PFX + "=" * len(msg))

                # Reproduce the error with full output

                set_verbosity(Levels.TRACE)
                try:
                    test_func = getattr(module, test_func_name)
                    with np.printoptions(
                        precision=np.finfo(pisa.FTYPE).precision + 2,
                        floatmode="fixed",
                        sign=" ",
                        linewidth=200,
                    ):
                        test_func()
                except Exception:
                    pass

                set_verbosity(Levels.TRACE)
                logging.exception(err)

                set_verbosity(verbosity)
                logging.error(PFX + "#" * len(msg))

            else:
                test_pypaths_succeeded.append(test_pypath)

            finally:
                # remove references to the test function, e.g. to remove refs
                # to pycuda / numba.cuda contexts so these can be closed
                try:
                    del test_func
                except NameError:
                    pass

        # NOTE: Until we get all GPU code into Numba, need to unload pycuda
        # and/or numba.cuda contexts before a module requiring the other one is
        # to be imported.
        # NOTE: the following causes a traceback to be emitted at the very end
        # of the script, regardless of the exception catching here.
        if (
            pisa.TARGET == "cuda"
            and pycuda is not None
            and hasattr(pycuda, "autoinit")
            and hasattr(pycuda.autoinit, "context")
        ):
            try:
                pycuda.autoinit.context.detach()
            except Exception:
                pass

        # Attempt to unload the imported module
        # TODO: pipeline, etc. fail as isinstance(service, (Stage, PiStage)) is False
        #if module_pypath in sys.modules and module_pypath != "pisa":
        #    del sys.modules[module_pypath]
        #del module

        # TODO: crashes program; subseqeunt calls in same shell crash(!?!?)
        # if pisa.TARGET == 'cuda' and nbcuda is not None:
        #    try:
        #        nbcuda.close()
        #    except Exception:
        #        pass

    # Summarize results

    n_import_successes = len(module_pypaths_succeeded)
    n_import_failures = len(module_pypaths_failed)
    n_import_failures_ignored = len(module_pypaths_failed_ignored)
    n_test_successes = len(test_pypaths_succeeded)
    n_test_failures = len(test_pypaths_failed)
    n_test_failures_ignored = len(test_pypaths_failed_ignored)

    set_verbosity(verbosity)
    logging.info(
        PFX + f"<< IMPORT TESTS : {n_import_successes} imported,"
        f" {n_import_failures} failed,"
        f" {n_import_failures_ignored} failed to import but ok to ignore >>"
    )
    logging.info(
        PFX + f"<< UNIT TESTS : {n_test_successes} succeeded,"
        f" {n_test_failures} failed,"
        f" {n_test_failures_ignored} failed but ok to ignore >>"
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

        # Note the extra newlines before the exception to make it stand out;
        # and newlines after the exception are due to the pycuda error message
        # that is emitted when we call pycuda.autoinit.context.detach()
        sys.stdout.flush()
        sys.stderr.write("\n\n\n")
        raise Exception("\n".join(msgs) + "\n\n\n")


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
    path = expand(path, absolute=True, resolve_symlinks=True)

    tests = {}
    if isfile(path):
        filerelpath = relpath(path, start=PISA_PATH)
        tests[filerelpath] = find_unit_tests_in_file(path)
        return tests

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
    filepath = expand(filepath, absolute=True, resolve_symlinks=True)
    assert isfile(filepath), str(filepath)
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
    parser = ArgumentParser(
        description=description, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=PISA_PATH,
        help="""Specify a specific path to a file or directory in which to find
        and run unit tests""",
    )
    parser.add_argument(
        "--allow-missing",
        nargs="+",
        default=list(OPTIONAL_MODULES),
        help="""Allow ImportError (or subclasses) for these modules""",
    )
    parser.add_argument(
        "-v", action="count", default=Levels.WARN, help="set verbosity level"
    )
    kwargs = vars(parser.parse_args())
    kwargs["verbosity"] = kwargs.pop("v")
    run_unit_tests(**kwargs)


if __name__ == "__main__":
    main()
