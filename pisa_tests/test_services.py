#! /usr/bin/env python3

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
import os
from os import walk
from os.path import isfile, join, relpath
import sys
from typing import List, Dict, Tuple

import numpy as np

from pisa import FTYPE, ITYPE, ureg
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.container import Container, ContainerSet
from pisa.core.stage import Stage
from pisa.utils.random_numbers import get_random_state
from pisa.utils.fileio import expand, nsort_key_func
from pisa.utils.log import Levels, logging, set_verbosity
from pisa_tests.run_unit_tests import PISA_PATH, OPTIONAL_MODULES


__all__ = [
    "STAGES_PATH",
    "INIT_TEST_NAME",
    "TEST_BINNING",
    "SKIP_SERVICES",
    "AUX_DATA_KEYS",
    "test_services",
    "find_services",
    "find_services_in_file",
    "get_stage_dot_service_from_module_pypath",
    "add_test_inputs",
    "is_allowed_import_error",
    "set_service_attributes"
]

__author__ = "T. Ehrhardt, J. Weldert"

__license__ = """Copyright (c) 2014-2026, The IceCube Collaboration

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
INIT_TEST_NAME = "init_test"
"""Assumed name of custom function in each module which returns an
example instance of the service in question"""
TEST_BINNING = MultiDimBinning([
    OneDimBinning(name='reco_energy', is_log=True, num_bins=3, domain=[0.1, 1]*ureg.GeV),
    OneDimBinning(name='reco_coszen', is_lin=True, num_bins=3, domain=[0.1, 1]),
    OneDimBinning(name='pid', is_lin=True, num_bins=3, domain=[0.1, 1]),
])
AUX_DATA_KEYS = ['nubar', 'flav']

SKIP_SERVICES = (
    'osc.external', 'flux.airs'
)
"""If no other way, add hopeless cases in <stage>.<service> format"""

PFX = "[S] "
"""Prefix each line output by this script to clearly delineate output from this
script vs. output from test functions being run"""


def find_services(path):
    """Modelled after `run_unit_tests.find_unit_tests`"""
    path = expand(path, absolute=True, resolve_symlinks=True)

    services = {}
    if isfile(path):
        filerelpath = relpath(path, start=PISA_PATH)
        services[filerelpath] = find_services_in_file(path)
        return services

    for dirpath, dirs, files in walk(path, followlinks=True):
        if dirpath.endswith(".ipynb_checkpoints") or dirpath.endswith("__pycache__"):
            continue

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


def get_stage_dot_service_from_module_pypath(module_pypath):
    """Assumes `module_pypath` starts with pisa.stages and we
    have one directory per stage, which contains all services
    implementing that stage."""
    return module_pypath[12:]


def add_test_inputs(service, empty=False):
    """Try to come up with sensible test input data for the `Stage`
    instance `service`"""
    random_state = get_random_state(0)
    if not empty:
        name1 = 'test1_cc'
        name2 = 'test2_nc'
        container1 = Container(name1)
        container2 = Container(name2)
        keys = set(service.expected_container_keys +
                   ['reco_energy', 'reco_coszen', 'pid', 'weights']
                  )
        for k in keys:
            if k in AUX_DATA_KEYS:
                container1.set_aux_data(k, ITYPE(1))
                container2.set_aux_data(k, ITYPE(1))
            elif k in ['nu_flux', 'nu_flux_nominal', 'nubar_flux_nominal']:
                container1[k] = random_state.random((10, 2)).astype(dtype=FTYPE)
                container2[k] = random_state.random((10, 2)).astype(dtype=FTYPE)
            elif k.endswith('mask'):
                container1[k] = np.ones(10, dtype=ITYPE)
                container2[k] = np.zeros(10, dtype=ITYPE)
            else:
                container1[k] = np.linspace(0.1, 1, 10, dtype=FTYPE)
                container2[k] = np.linspace(0.1, 1, 10, dtype=FTYPE)
        service.data = ContainerSet('data', [container1, container2])
        service.data["output_binning"] = TEST_BINNING
        service.data["regularized_output_binning"] = TEST_BINNING
    else:
        logging.debug(PFX + "Creating empty test inputs...")
        service.data = ContainerSet('data')


def set_service_attributes(service, stage_dot_service):
    """For services requiring special treatment before `setup()` or `run()`"""
    if stage_dot_service == 'osc.external':
        # TODO: `osc_prob` callable and `external_params` required by callable have to
        # be set after init, where call signature is
        # self.osc_prob(energies, distances, self.external_params, is_anti, densities, densities_neutron_weighted)...
        #service.osc_prob = ....
        #service.external_params = ...
        pass


def run_service_test(service):
    """Try to set up and run the `Stage` instance `service`"""
    service.setup()
    service.run()


def is_allowed_import_error(err, module_pypath, allow_missing=OPTIONAL_MODULES):
    if (
        isinstance(err, ImportError)
        and hasattr(err, "name")
        and err.name in allow_missing
    ):
        err_name = err.name
        logging.warning(
            PFX + f"module {err_name} failed to import while importing "
            f"{module_pypath}, but ok to ignore"
        )
        return True
    return False


def test_services(
    path=STAGES_PATH,
    init_test_name=INIT_TEST_NAME,
    skip_services=SKIP_SERVICES,
    allow_missing=OPTIONAL_MODULES,
    verbosity=Levels.WARN,
    output_matrix=None
):
    """Modelled after `run_unit_tests.run_unit_tests`

    Parameters
    ----------
    path : str (default: STAGES_PATH)
    init_test : str (default: INIT_TEST_NAME)
    skip_services : array-like (default: SKIP_SERVICES)
    allow_missing : array-like (default: OPTIONAL_MODULES)
    verbosity : int (default: Levels.WARN)
    output_matrix : str, optional
        If provided, write service implementation matrix to this file path.
        Format controlled by file extension (.md or .csv accepted).
    """
    if allow_missing is None:
        allow_missing = []
    elif isinstance(allow_missing, str):
        allow_missing = [allow_missing]

    services = find_services(path=path)

    ntries = 0
    nsuccesses = 0
    stage_dot_services_failed_ignored = []
    stage_dot_services_failed = []
    services_info = [] # for collecting service metadata
    set_verbosity(verbosity)

    for rel_file_path, service_names in services.items():
        if not service_names:
            continue

        pypath = ["pisa"] + rel_file_path[:-3].split("/")
        parent_pypath = ".".join(pypath[:-1])
        module_name = pypath[-1].replace(".", "_")
        module_pypath = f"{parent_pypath}.{module_name}"

        if not len(service_names) == 1:
            raise ValueError(
                '%d > 1 services detected in file %s!'
                % (len(service_names), module_pypath)
            )

        service_name = service_names[0]
        stage_dot_service = get_stage_dot_service_from_module_pypath(module_pypath)

        # check whether we should skip testing this service for some reason
        if stage_dot_service in skip_services:
            logging.warning(
                PFX + f"{stage_dot_service} requested to be ignored in"
                " service test."
            )
            continue

        logging.info(PFX + f"Starting test for service {stage_dot_service}...")
        ntries += 1

        # if service module import successful, try to initialise the service
        try:
            module = import_module(module_pypath, package=parent_pypath)
        except Exception as err:
            if is_allowed_import_error(err, module_pypath, allow_missing):
                stage_dot_services_failed_ignored.append(stage_dot_service)
                continue

            stage_dot_services_failed.append(stage_dot_service)

            set_verbosity(verbosity)
            msg = PFX + f"<< FAILURE IMPORTING : {module_pypath} >>"
            logging.error("=" * len(msg))
            logging.error(msg)
            logging.error("=" * len(msg))

            set_verbosity(Levels.TRACE)
            logging.exception(err)

            set_verbosity(verbosity)
            continue

        if not hasattr(module, init_test_name):
            try:
                # Without a dedicated `init_test` function, we just try to
                # instantiate the service with std. Stage kwargs
                service = getattr(module, service_name)(in_standalone_mode=True)
            except Exception as err:
                logging.error(
                    PFX + f"{stage_dot_service} has no {init_test_name} function "
                     "and could not be instantiated with standard kwargs only.\n"
                     "msg: %s" % err
                )
                stage_dot_services_failed.append(stage_dot_service)
                continue
        else:
            try:
                # Exploit presence of init_test (TODO: switch order with above?)
                param_kwargs = {'prior': None, 'range': None, 'is_fixed': True}
                service = getattr(module, init_test_name)(**param_kwargs)
            except Exception as err:
                logging.error(
                    PFX + f"{stage_dot_service} has an {init_test_name} function "
                    "which failed to instantiate the service with msg:\n %s." % err
                )
                stage_dot_services_failed.append(stage_dot_service)
                continue

        if not isinstance(service, Stage):
            # Could be that init test function exists and runs but doesn't
            # actually return anything useful
            service_type = type(service)
            logging.error(
                PFX + f"Did not get an initialised `Stage` instance for "
                f"{stage_dot_service} but {service_type}!"
            )
            stage_dot_services_failed.append(stage_dot_service)
            continue

        # Collect service metadata
        try:
            services_info.append(collect_service_info(service))
        except Exception as e:
            logging.warning(
                PFX + f"Failed to collect metadata for {stage_dot_service}: {e}"
            )

        if service.data is None:
            # For data services, setup usually adds to `data` attribute
            # (`Pipeline.setup()` assigns empty `ContainerSet`)
            # TODO: Can/should init ever already result in populated `data` attribute?
            try:
                add_test_inputs(
                    service=service,
                    empty=stage_dot_service.split('.')[0] == 'data'
                )
            except Exception as err:
                logging.error(
                    PFX + f"Failed to assign test inputs for "
                    f"{stage_dot_service} with msg:\n {err}"
                )
                stage_dot_services_failed.append(stage_dot_service)
                continue

        #if service.data is not None: # we should never be in this state here
        if service.calc_mode is None and None not in service.supported_reps['calc_mode']:
            logging.debug(PFX + "Setting calc_mode ...")
            try:
                service.calc_mode = 'events'
            except ValueError:
                service.calc_mode = TEST_BINNING
            except ImportError as err:
                if is_allowed_import_error(err, stage_dot_service, allow_missing):
                    stage_dot_services_failed_ignored.append(stage_dot_service)
                    continue
            except Exception as err:
                logging.error(
                    PFX + "Failed to set `calc_mode` for "
                    f"{stage_dot_service} with msg:\n {err}"
                )
                stage_dot_services_failed.append(stage_dot_service)
                continue

        if service.apply_mode is None and None not in service.supported_reps['apply_mode']:
            logging.debug(PFX + "Setting apply_mode ...")
            try:
                service.apply_mode = 'events'
            except ValueError:
                service.apply_mode = TEST_BINNING
            except ImportError as err:
                if is_allowed_import_error(err, stage_dot_service, allow_missing):
                    stage_dot_services_failed_ignored.append(stage_dot_service)
                    continue
            except Exception as err:
                logging.error(
                    PFX + "Failed to set `apply_mode` for "
                    f"{stage_dot_service} with msg:\n {err}"
                )
                stage_dot_services_failed.append(stage_dot_service)
                continue

        try:
            set_service_attributes(service, stage_dot_service)
        except Exception as err:
            logging.error(
                PFX + "Failed to set attributes for "
                f"{stage_dot_service} with msg:\n {err}"
            )
            stage_dot_services_failed.append(stage_dot_service)
            continue

        try:
            logging.debug(PFX + "Setting up and running service...")
            run_service_test(service)
            logging.info(PFX + f"{stage_dot_service} passed the test.")
            nsuccesses += 1
        except Exception as err:
            if is_allowed_import_error(err, stage_dot_service, allow_missing):
                stage_dot_services_failed_ignored.append(stage_dot_service)
                continue
            logging.error(
                PFX + f"{stage_dot_service} failed to setup or run "
                f"with msg:\n {err}."
            )
            stage_dot_services_failed.append(stage_dot_service)
            continue

    logging.info(
        PFX + f"{nsuccesses} out of {ntries} tested services passed the test."
    )
    nfail = ntries - nsuccesses
    nfail_ignored = len(stage_dot_services_failed_ignored)
    logging.info(
        PFX + f"{nfail_ignored} out of {nfail} failures have been ignored."
    )
    nfail_remain = nfail - nfail_ignored

    # Output matrix file if requested
    if output_matrix and services_info:
        rootname, ext = os.path.splitext(output_matrix)
        logging.info(PFX + f"Generating service matrix ({ext} format)...")
        try:
            if ext.lower() == ".csv":
                matrix_content = generate_csv_matrix(services_info)
            else:
                if ext.lower() != ".md":
                    logging.warning("Unknown service matrix format requested. "
                        "Will produce markdown format instead.")
                matrix_content = generate_markdown_matrix(services_info)

            with open(output_matrix, 'w') as f:
                f.write(matrix_content)
            logging.info(PFX + f"Service matrix written to {output_matrix}")

        except Exception as e:
            logging.error(PFX + f"Failed to generate service matrix: {e}")

    if nfail_remain > 0:
        sys.stdout.flush()
        sys.stderr.write("\n\n\n")
        raise Exception(
            PFX + f"{nfail_remain} failures still need to be addressed:\n" +
            ", ".join(stage_dot_services_failed) + "\n\n\n"
        )


def format_supported_reps(reps_list) -> str:
    """Format supported representations for display in the matrix."""
    if not reps_list:
        return "None"

    formatted = []
    for rep in reps_list:
        if rep is None:
            formatted.append("None")
        elif isinstance(rep, str):
            formatted.append(f'"{rep}"')
        elif rep is MultiDimBinning:
            formatted.append("MultiDimBinning")
        else:
            formatted.append(str(rep))

    if len(formatted) == 1:
        return formatted[0]
    return ", ".join(formatted)


def collect_service_info(service) -> Dict:
    """Extract service implementation info from an instantiated Stage."""
    return {
        'stage_name': service.stage_name,
        'service_name': service.service_name,
        'has_setup': service.has_setup,
        'has_compute': service.has_compute,
        'has_apply': service.has_apply,
        'calc_mode_support': format_supported_reps(service.supported_reps.get('calc_mode', [])),
        'apply_mode_support': format_supported_reps(service.supported_reps.get('apply_mode', [])),
    }


def generate_service_notes(info: Dict) -> str:
    """Auto-detect caching issues and other possible oddities
    and produce corresponding notes."""
    notes = []
    if not info['has_apply']:
        notes.append("⚠️ **Implicit caching ([#821](https://github.com/icecube/pisa/issues/821))**")
    if not info['has_setup'] and not info['has_compute'] and not info['has_apply']:
        notes.append("init. only")
    elif not info['has_apply'] and not info['has_compute']:
        notes.append("setup only")
    notes = ", ".join(notes)
    return notes


#TODO: document expected container keys also?
def generate_markdown_matrix(services_info: List[Dict]) -> str:
    """Generate markdown table from collected service information."""
    # Sort by stage_name, then service_name
    services_info = sorted(
        services_info,
        key=lambda x: (x['stage_name'], x['service_name'])
    )

    lines = [
        "## Service implementation reference",
        "This [auto-generated](https://github.com/icecube/pisa/blob/master/pisa_tests/test_services.py) table"
        " documents which methods each service implements, which representations it supports, as well as any"
        " deviant behavior or properties to be aware of.",
        "It currently only contains services that can be **instantiated without importing optional PISA dependencies**.",
        "",
        "**Legend:**",
        "- **has_setup/compute/apply**: whether the service overrides setup/compute/apply_function (✓ = yes, ✗ = no)",
        "- **calc_mode support**: allowed representations during setup/compute steps",
        "- **apply_mode support**: allowed representations during apply step",
        "- **Notes**: special behaviors, restrictions, problems, etc.",
        "",
        "| Service | has_setup | has_compute | has_apply | calc_mode support | apply_mode support | Notes |",
        "|---------|:---------:|:-----------:|:---------:|:-----------------:|:------------------:|-------|",
    ]

    for info in services_info:
        setup_icon = "✓" if info['has_setup'] else "✗"
        compute_icon = "✓" if info['has_compute'] else "✗"
        apply_icon = "✓" if info['has_apply'] else "✗"

        service_full_name = f"**{info['stage_name']}.{info['service_name']}**"
        notes = generate_service_notes(info)

        line = (
            f"| {service_full_name} | {setup_icon} | {compute_icon} | {apply_icon} | "
            f"{info['calc_mode_support']} | {info['apply_mode_support']} | {notes} |"
        )
        lines.append(line)

    return "\n".join(lines)


def generate_csv_matrix(services_info: List[Dict], filepath: str = None) -> str:
    """Generate CSV table from collected service information (for external processing)."""
    import csv
    from io import StringIO

    # Sort by stage_name, then service_name
    services_info = sorted(
        services_info,
        key=lambda x: (x['stage_name'], x['service_name'])
    )

    output = StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            'Service', 'has_setup', 'has_compute', 'has_apply',
            'calc_mode_support', 'apply_mode_support', 'Notes'
        ]
    )

    writer.writeheader()
    for info in services_info:
        setup_icon = "Yes" if info['has_setup'] else "No"
        compute_icon = "Yes" if info['has_compute'] else "No"
        apply_icon = "Yes" if info['has_apply'] else "No"

        notes = generate_service_notes(info)

        writer.writerow({
            'Service': f"{info['stage_name']}.{info['service_name']}",
            'has_setup': setup_icon,
            'has_compute': compute_icon,
            'has_apply': apply_icon,
            'calc_mode_support': info['calc_mode_support'],
            'apply_mode_support': info['apply_mode_support'],
            'Notes': notes,
        })

    result = output.getvalue()
    output.close()
    return result


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-v", action="count", default=Levels.WARN, help="set verbosity level"
    )
    parser.add_argument(
        "--output-matrix",
        metavar="FILE",
        default=None,
        help="""If desired, write a service implementation overview table to
        this file, giving either .md (markdown) or .csv (comma-separated values)
        file extension to indicate the desired format for the output table."""
    )
    args = parser.parse_args()
    return args


def main():
    """Script interface to test_services"""
    args = parse_args()
    kwargs = vars(args)
    kwargs["verbosity"] = kwargs.pop("v")
    kwargs["output_matrix"] = kwargs.pop("output_matrix")
    test_services(**kwargs)
    logging.info(PFX + 'Services testing done.')


if __name__ == "__main__":
    main()
