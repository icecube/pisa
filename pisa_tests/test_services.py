#! /usr/bin/env python

"""
Try to simply run every existing service by automatically deriving as many
sensible test-configuration parameters as possible. A generic services's
test cannot be triggered from within a given service itself, because
sensibly initialising the instance itself (init params, expected params)
is part of the problem. Also, with this external script we can avoid
requesting the implementation of a test function within each service's
module.
"""

from __future__ import absolute_import

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from importlib import import_module
from numpy import linspace
from os import walk
from os.path import isfile, join, relpath

from pisa.core.container import Container, ContainerSet
from pisa.utils.fileio import expand, nsort_key_func
from pisa.utils.log import Levels, logging, set_verbosity
from pisa_tests.run_unit_tests import PISA_PATH


__all__ = ["STAGES_PATH", "test_services", "find_services", "find_services_in_file"]

__author__ = "T. Ehrhardt"

__license__ = """Copyright (c) 2014-2024, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""



STAGES_PATH = join(PISA_PATH, "stages")

# TODO: define hopeless cases? define services whose tests may fail?
# optional modules from unit tests?
SKIP_SERVICES = (
    'likelihood.generalized_llh_params',
)


def find_services(path):
    """Modelled after `run_unit_tests.find_unit_tests`"""
    path = expand(path, absolute=True, resolve_symlinks=True)

    services = {}
    if isfile(path):
        filerelpath = relpath(path, start=PISA_PATH)
        services[filerelpath] = find_services_in_file(path)
        return services

    for dirpath, dirs, files in walk(path, followlinks=True):
        files.sort(key=nsort_key_func)
        dirs.sort(key=nsort_key_func)

        for filename in files:
            if not filename.endswith(".py"):
                continue
            filepath = join(dirpath, filename)
            filerelpath = relpath(filepath, start=PISA_PATH)
            services[filerelpath] = find_services_in_file(filepath)

    return services


def find_services_in_file(filepath):
    """Modelled after `run_unit_tests.find_unit_tests_in_file`"""
    filepath = expand(filepath, absolute=True, resolve_symlinks=True)
    assert isfile(filepath), str(filepath)
    services = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            tokens = line.split()
            if tokens and tokens[0] == "class" and "(Stage)" in tokens[1]:
                service_name = tokens[1].split("(")[0].strip()
                services.append(service_name)
    return services


def test_services(
    path=STAGES_PATH, 
    skip_services=SKIP_SERVICES,
    verbosity=Levels.WARN,
):
    """Modelled after `run_unit_tests.run_unit_tests`"""
    services = find_services(path=path)

    set_verbosity(verbosity)
    for rel_file_path, service_names in services.items():
        if not service_names:
            continue
        assert len(service_names) == 1, 'Only specify one stage per file.'

        pypath = ["pisa"] + rel_file_path[:-3].split("/")
        parent_pypath = ".".join(pypath[:-1])
        module_name = pypath[-1].replace(".", "_")
        module_pypath = f"{parent_pypath}.{module_name}"

        if module_pypath[12:] in skip_services:
            logging.warning(f"{module_pypath[12:]} will be ignored in service test.")
            continue

        try:
            exec('from ' + module_pypath + ' import init_test')
            test = init_test()
        except:
            try:
                module = import_module(module_pypath, package=parent_pypath)
                test = getattr(module, service_names[0])()
            except:
                logging.warning(f"{module_pypath[12:]} can not be initialized and has no init_test function specified.")
                continue

        test.calc_mode = 'events'
        test.apply_mode = 'events'

        container1 = Container('test1')
        container2 = Container('test2')
        for k in test.expected_container_keys:
            container1[k] = linspace(0, 1, 10)
            container2[k] = linspace(0, 1, 10)
        test.data = ContainerSet('data', [container1, container2])

        try:
            test.setup()
            test.run()
            logging.info(f"{module_pypath[12:]} passed the test.")
        except:
            logging.error(f"{module_pypath[12:]} failed the test.")


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-v", action="count", default=Levels.WARN, help="set verbosity level"
    )
    args = parser.parse_args()
    return args


def main():
    """Script interface to test_services"""
    args = parse_args()
    kwargs = vars(args)
    kwargs["verbosity"] = kwargs.pop("v")
    test_services(**kwargs)
    logging.info(f'services testing done')


if __name__ == "__main__":
    main()
