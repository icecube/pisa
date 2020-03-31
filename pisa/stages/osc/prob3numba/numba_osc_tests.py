#!/usr/bin/env python
# pylint: disable = invalid-name


"""
Tests for prob3numba code
"""


from __future__ import absolute_import, print_function, division


__all__ = [
    "TEST_DATA_DIR",
    "FINFO_FTYPE",
    "AC_KW",
    "PRINTOPTS",
    "A2S_KW",
    "MAT_DOT_MAT_SUBSCR",
    "DEFAULTS",
    "TEST_CASES",
    "auto_populate_test_case",
    "test_prob3numba",
    "run_test_case",
    "stability_test",
    "execute_func",
    "compare_numeric",
    "check",
    "ary2str",
    "main",
]


from argparse import ArgumentParser
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from inspect import getmodule, signature
from os.path import join

import numpy as np
from numba import SmartArray

from pisa import FTYPE
from pisa.utils.comparisons import ALLCLOSE_KW
from pisa.utils.fileio import expand, from_file, to_file
from pisa.utils.log import Levels, logging, set_verbosity
from pisa.utils.numba_tools import WHERE
from pisa.utils.resources import find_resource
from pisa.stages.osc.prob3numba.numba_osc_hostfuncs import (
    CX,
    FX,
    IX,
    # propagate_scalar_vacuum,
    propagate_scalar,
    propagate_array,
    get_transition_matrix_hostfunc,
    get_transition_matrix_massbasis_hostfunc,
    get_H_vac_hostfunc,
    get_H_mat_hostfunc,
    get_dms_hostfunc,
    get_product_hostfunc,
    convert_from_mass_eigenstate_hostfunc,
)
from pisa.stages.osc.nsi_params import (
    StdNSIParams,
    VacuumLikeNSIParams,
)

TEST_DATA_DIR = find_resource("osc/numba_osc_tests_data")

FINFO_FTYPE = np.finfo(FTYPE)

AC_KW = dict(atol=FINFO_FTYPE.resolution * 10, rtol=ALLCLOSE_KW["rtol"] * 100)

PRINTOPTS = dict(
    precision=FINFO_FTYPE.precision + 2, floatmode="fixed", sign=" ", linewidth=200
)

A2S_KW = dict(precision=PRINTOPTS["precision"], separator=", ")

MAT_DOT_MAT_SUBSCR = "in,nj->ij"
"""matrix dot matrix subscripts for use by `numpy.einsum`"""

# ---------------------------------------------------------------------------- #
# Define relevant values for testing purposes (from nufit3.2, from intermediate
# calculations performed here, or arbitary values).
#
# NOTE: !!DO NOT CHANGE!! (unless a function is incorrect) tests rely on these
# ---------------------------------------------------------------------------- #

DEFAULTS = dict(
    energy=1,  # GeV
    state=1,
    nubar=1,
    rho=1,  # moles of electrons / cm^3
    baseline=1,  # km
    mat_pot=np.diag([1, 0, 0]).astype(np.complex128),
    layer_distances=np.logspace(0, 2, 10),  # km
    layer_densities=np.linspace(0.5, 3, 10),  # g/cm^3
    # osc params: defaults are nufit 3.2 normal ordering values
    t12=np.deg2rad(33.62),
    t23=np.deg2rad(47.2),
    t13=np.deg2rad(8.54),
    dcp=np.deg2rad(234),
    dm21=7.40e-5,
    dm31=2.494e-3,
)

# define non-0 NSI parameters for non-vacuum NSI
# roughly based on best fit params from Thomas Ehrhardts 3y DRAGON analysis
nsi_params = StdNSIParams()
nsi_params.eps_emu_magn = 0.07
nsi_params.eps_emu_phase = np.deg2rad(340)
nsi_params.eps_etau_magn = 0.06
nsi_params.eps_etau_phase = np.deg2rad(35)
nsi_params.eps_mutau_magn = 0.003
nsi_params.eps_mutau_phase = np.deg2rad(175)
mat_pot_std_nsi_no = np.diag([1, 0, 0]).astype(np.complex128) + nsi_params.eps_matrix

# Vacuum-like NSI parameters
nsi_params = VacuumLikeNSIParams()
nsi_params.eps_prime = 0.1
mat_pot_vac_nsi_no = np.diag([1, 0, 0]).astype(np.complex128) + nsi_params.eps_matrix

TEST_CASES = dict(
    nufit32_no=dict(),  # nufit 3.2 normal ordering (also overall) best-fit
    nufit32_no_nubar=dict(nubar=-1),  # NO but anti-neutrinos
    nufit32_no_E1TeV=dict(energy=1e3),  # NO but e=1 TeV
    nufit32_no_blearth=dict(
        baseline=6371e3 * 2,
        layer_distances=(
            6371e3
            * 2
            * DEFAULTS["layer_distances"]
            / np.sum(DEFAULTS["layer_distances"])
        ),
    ),
    nufit32_io=dict(  # nufit 3.2 best-fit params for inverted ordering
        t12=np.deg2rad(33.62),
        t23=np.deg2rad(48.1),
        t13=np.deg2rad(8.58),
        dcp=np.deg2rad(278),
        dm21=7.40e-5,
        dm31=-2.465e-3,
    ),
    nufit32_std_nsi_no=dict(  # nufit 3.2 normal ordering with non-0 standard NSI parameters
        mat_pot=mat_pot_std_nsi_no,
    ),
    nufit32_vac_nsi_no=dict(  # nufit 3.2 normal ordering with non-0 vacuum NSI parameters
        mat_pot=mat_pot_vac_nsi_no,
    ),
)


def auto_populate_test_case(tc, defaults):
    """Populate defaults and construct dm / PMNS matrices if they aren't
    present in a test case.

    Parameters
    ----------
    test_case : mutable mapping

    defaults : mapping

    """
    for key, val in defaults.items():
        if key not in tc:
            tc[key] = val

    # Construct dm and PMNS matrices derived from test case values, if
    # these were not already specified

    if "dm" not in tc:  # construct Delta m^2 matrix if not present
        if "dm32" not in tc:  # NO case; Delta m^2_32/eV^2
            tc["dm32"] = tc["dm31"] - tc["dm21"]

        if "dm31" not in tc:  # IO case; Delta m^2_32/eV^2
            tc["dm31"] = tc["dm32"] + tc["dm21"]

        tc["dm"] = np.array(
            [
                [0, -tc["dm21"], -tc["dm31"]],
                [tc["dm21"], 0, -tc["dm32"]],
                [tc["dm31"], tc["dm32"], 0],
            ]
        )

    if "pmns" not in tc:  # construct PMNS matrix if not present
        c12, s12 = np.cos(tc["t12"]), np.sin(tc["t12"])
        c23, s23 = np.cos(tc["t23"]), np.sin(tc["t23"])
        c13, s13 = np.cos(tc["t13"]), np.sin(tc["t13"])

        tc["pmns"] = (
            np.array([[1, 0, 0], [0, c23, s23], [0, -s23, c23]])
            @ np.array(
                [
                    [c13, 0, s13 * np.exp(-1j * tc["dcp"])],
                    [0, 1, 0],
                    [-s13 * np.exp(1j * tc["dcp"]), 0, c13],
                ]
            )
            @ np.array([[c12, s12, 0], [-s12, c12, 0], [0, 0, 1]])
        )


def test_prob3numba(ignore_fails=False, define_as_ref=False):
    """Run all unit test cases for prob3numba code"""

    # Pull first test case to test calling `propagate_array`
    tc_name, tc = next(iter(TEST_CASES.items()))
    tc_ = deepcopy(tc)
    logging.info(
        "Testing call and return shape of `propagate_array` with test case '%s'",
        tc_name,
    )

    # Test simple broadcasting over `nubars` and `energies` where both have
    # same shape, as this is the "typical" use case
    input_shape = (4, 5)

    # Without broadcasting, a single probability matrix is 3x3
    prob_array_shape = (3, 3)

    # Broadcasted shape
    out_shape = input_shape + prob_array_shape

    nubars = np.full(shape=input_shape, fill_value=tc_["nubar"], dtype=IX)
    energies = np.full(shape=input_shape, fill_value=tc_["energy"], dtype=FX)

    # Fill with NaN to ensure all elements are assinged a value
    probabilities = SmartArray(np.full(shape=out_shape, fill_value=np.nan, dtype=FX))

    propagate_array(
        SmartArray(tc_["dm"].astype(FX)).get(WHERE),
        SmartArray(tc_["pmns"].astype(CX)).get(WHERE),
        SmartArray(tc_["mat_pot"].astype(CX)).get(WHERE),
        SmartArray(nubars).get(WHERE),
        SmartArray(energies).get(WHERE),
        SmartArray(tc_["layer_densities"].astype(FX)).get(WHERE),
        SmartArray(tc_["layer_distances"].astype(FX)).get(WHERE),
        # output:
        probabilities.get(WHERE),
    )
    probabilities.mark_changed(WHERE)
    probabilities = probabilities.get("host")

    # Check that all probability matrices have no NaNs and are equal to one
    # another
    ref_probs = probabilities[0, 0]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            probs = probabilities[i, j]
            assert np.all(np.isfinite(probs))
            assert np.all(probs == ref_probs)

    # Run all test cases
    for tc_name, tc in TEST_CASES.items():
        run_test_case(
            tc_name, tc, ignore_fails=ignore_fails, define_as_ref=define_as_ref
        )


def run_test_case(tc_name, tc, ignore_fails=False, define_as_ref=False):
    """Run one test case"""
    logging.info("== TEST CASE : %s ==", tc_name)

    st_test_kw = dict(ignore_fails=ignore_fails, define_as_ref=define_as_ref)
    tf_sfx = f"__{tc_name}__{FX}.pkl"

    # Copy contents of test case, so if a function modifies these
    # (accidentally), it doesn't affect the outcome of further tests
    tc_ = deepcopy(tc)
    test, ref = stability_test(
        func=convert_from_mass_eigenstate_hostfunc,
        func_kw=dict(
            state=tc_["state"],
            mix_nubar=tc_["pmns"] if tc_["nubar"] > 0 else tc_["pmns"].conj().T,
            # output:
            psi=np.ones(shape=3, dtype=CX),
        ),
        ref_path=join(TEST_DATA_DIR, f"convert_from_mass_eigenstate_hostfunc{tf_sfx}"),
        **st_test_kw,
    )
    logging.debug("\npsi = %s", ary2str(test["psi"]))

    tc_ = deepcopy(tc)
    test, ref = stability_test(
        func=get_H_vac_hostfunc,
        func_kw=dict(
            mix_nubar=tc_["pmns"] if tc_["nubar"] > 0 else tc_["pmns"].conj().T,
            mix_nubar_conj_transp=(
                tc_["pmns"].conj().T if tc_["nubar"] > 0 else tc_["pmns"]
            ),
            dm_vac_vac=tc_["dm"],
            # output:
            H_vac=np.ones(shape=(3, 3), dtype=CX),
        ),
        ref_path=join(TEST_DATA_DIR, f"get_H_vac_hostfunc{tf_sfx}"),
        **st_test_kw,
    )
    logging.debug("\nH_vac = %s", ary2str(test["H_vac"]))
    # keep for use by `get_transition_matrix_hostfunc`
    H_vac_ref = ref["H_vac"]

    tc_ = deepcopy(tc)
    test, ref = stability_test(
        func=get_H_mat_hostfunc,
        func_kw=dict(
            rho=tc_["rho"],
            mat_pot=tc_["mat_pot"],
            nubar=tc_["nubar"],
            # output:
            H_mat=np.ones(shape=(3, 3), dtype=CX),
        ),
        ref_path=join(TEST_DATA_DIR, f"get_H_mat_hostfunc{tf_sfx}"),
        **st_test_kw,
    )
    logging.debug("\nH_mat = %s", ary2str(test["H_mat"]))
    # keep for use by `get_dms_hostfunc`,
    # `get_transition_matrix_massbasis_hostfunc`, `get_product_hostfunc``
    H_mat_ref = ref["H_mat"]

    # tc_ = deepcopy(tc)
    # test, ref = stability_test(
    #     func=propagate_scalar_vacuum,
    #     func_kw=dict(
    #         dm=tc_["dm"],
    #         mix=tc_["pmns"],
    #         nubar=tc_["nubar"],
    #         energy=tc_["energy"],
    #         distances=tc_["layer_distances"],
    #         # output:
    #         probability=np.ones(shape=(3, 3), dtype=FX),
    #     ),
    #     ref_path=join(TEST_DATA_DIR, f"propagate_scalar_vacuum{tf_sfx}"),
    #     **st_test_kw,
    # )
    # logging.debug("\nvac_prob = %s", ary2str(test["probability"]))
    # # check unitarity
    # # TODO: << BUG? >> these fail even in double precision!
    # check(
    #     test=np.sum(test["probability"], axis=0),
    #     ref=np.ones(3),
    #     label=(
    #         f"{tc_name} :: propagate_scalar_vacuum :: sum(vacuum probability, axis=0)"
    #     ),
    #     ignore_fails=True,
    # )
    # check(
    #     test=np.sum(test["probability"], axis=1),
    #     ref=np.ones(3),
    #     label=(
    #         f"{tc_name} :: propagate_scalar_vacuum :: sum(vacuum probability, axis=1)"
    #     ),
    #     ignore_fails=True,
    # )

    tc_ = deepcopy(tc)
    test, ref = stability_test(
        func=propagate_scalar,
        func_kw=dict(
            dm=tc_["dm"],
            mix=tc_["pmns"],
            mat_pot=tc_["mat_pot"],
            nubar=tc_["nubar"],
            energy=tc_["energy"],
            densities=tc_["layer_densities"],
            distances=tc_["layer_distances"],
            # output:
            probability=np.ones(shape=(3, 3), dtype=FX),
        ),
        ref_path=join(TEST_DATA_DIR, f"propagate_scalar{tf_sfx}"),
        **st_test_kw,
    )
    logging.debug("\nmat_prob = %s", ary2str(test["probability"]))
    # check unitarity
    check(
        test=np.sum(test["probability"], axis=0),
        ref=np.ones(3),
        label=f"{tc_name} :: propagate_scalar:: sum(matter probability, axis=0)",
        ignore_fails=ignore_fails,
    )
    check(
        test=np.sum(test["probability"], axis=1),
        ref=np.ones(3),
        label=f"{tc_name} :: propagate_scalar :: sum(matter probability, axis=1)",
        ignore_fails=ignore_fails,
    )

    tc_ = deepcopy(tc)
    test, ref = stability_test(
        func=get_transition_matrix_hostfunc,
        func_kw=dict(
            nubar=tc_["nubar"],
            energy=tc_["energy"],
            rho=tc_["rho"],
            baseline=tc_["baseline"],
            mix_nubar=tc_["pmns"] if tc_["nubar"] > 0 else tc_["pmns"].conj().T,
            mix_nubar_conj_transp=(
                tc_["pmns"].conj().T if tc_["nubar"] > 0 else tc_["pmns"]
            ),
            mat_pot=tc_["mat_pot"],
            H_vac=H_vac_ref,
            dm=tc_["dm"],
            # output:
            transition_matrix=np.ones(shape=(3, 3), dtype=CX),
        ),
        ref_path=join(TEST_DATA_DIR, f"get_transition_matrix_hostfunc{tf_sfx}"),
        **st_test_kw,
    )
    logging.debug("\ntransition_matrix = %s", ary2str(test["transition_matrix"]))
    # check unitarity
    check(
        test=np.sum(np.abs(test["transition_matrix"]) ** 2, axis=0),
        ref=np.ones(3),
        label=(
            f"{tc_name}"
            " :: get_transition_matrix_hostfunc"
            ":: sum(|transition_matrix|^2, axis=0)"
        ),
        ignore_fails=ignore_fails,
    )
    check(
        test=np.sum(np.abs(test["transition_matrix"]) ** 2, axis=1),
        ref=np.ones(3),
        label=(
            f"{tc_name}"
            " :: get_transition_matrix_hostfunc"
            " :: sum(|transition_matrix|^2, axis=1)"
        ),
        ignore_fails=ignore_fails,
    )

    tc_ = deepcopy(tc)

    # Compute H_full as used in `numba_osc_kernels` to call `get_dms` from
    # `get_transition_matrix`
    H_full_ref = H_vac_ref / (2 * tc_["energy"]) + H_mat_ref
    test, ref = stability_test(
        func=get_dms_hostfunc,
        func_kw=dict(
            energy=tc_["energy"],
            H_mat=H_full_ref,
            dm_vac_vac=tc_["dm"],
            # outputs:
            dm_mat_mat=np.ones(shape=(3, 3), dtype=CX),
            dm_mat_vac=np.ones(shape=(3, 3), dtype=CX),
        ),
        ref_path=join(TEST_DATA_DIR, f"get_dms_hostfunc{tf_sfx}"),
        **st_test_kw,
    )
    logging.debug("\ndm_mat_mat = %s", ary2str(test["dm_mat_mat"]))
    logging.debug("\ndm_mat_vac = %s", ary2str(test["dm_mat_vac"]))
    # keep for use by `get_transition_matrix_massbasis_hostfunc`, `get_product_hostfunc`
    dm_mat_mat_ref = ref["dm_mat_mat"]
    dm_mat_vac_ref = ref["dm_mat_vac"]

    tc_ = deepcopy(tc)

    # Compute same intermediate result `H_mat_mass_eigenstate_basis` as in
    # `numba_osc_kernels.get_transition_matrix` which calls
    # `get_transition_matrix_massbasis`
    mix_nubar = tc_["pmns"] if tc_["nubar"] > 0 else tc_["pmns"].conj().T
    mix_nubar_conj_transp = tc_["pmns"].conj().T if tc_["nubar"] > 0 else tc_["pmns"]
    tmp = np.einsum(MAT_DOT_MAT_SUBSCR, H_mat_ref, mix_nubar)
    H_mat_mass_eigenstate_basis = np.einsum(
        MAT_DOT_MAT_SUBSCR, mix_nubar_conj_transp, tmp
    )

    test, ref = stability_test(
        func=get_transition_matrix_massbasis_hostfunc,
        func_kw=dict(
            baseline=tc_["baseline"],
            energy=tc_["energy"],
            dm_mat_vac=dm_mat_vac_ref,
            dm_mat_mat=dm_mat_mat_ref,
            H_mat_mass_eigenstate_basis=H_mat_mass_eigenstate_basis,
            # output:
            transition_matrix=np.ones(shape=(3, 3), dtype=CX),
        ),
        ref_path=join(
            TEST_DATA_DIR, f"get_transition_matrix_massbasis_hostfunc{tf_sfx}"
        ),
        **st_test_kw,
    )
    logging.debug("\ntransition_matrix_mb = %s", ary2str(test["transition_matrix"]))
    check(
        test=np.sum(np.abs(test["transition_matrix"]) ** 2, axis=0),
        ref=np.ones(3),
        label=(
            f"{tc_name}"
            " :: get_transition_matrix_massbasis_hostfunc"
            " :: sum(|transition_matrix (mass basis)|^2), axis=0)"
        ),
        ignore_fails=ignore_fails,
    )
    check(
        test=np.sum(np.abs(test["transition_matrix"]) ** 2, axis=1),
        ref=np.ones(3),
        label=(
            f"{tc_name}"
            " :: get_transition_matrix_massbasis_hostfunc"
            " :: sum(|transition_matrix (mass basis)|^2), axis=1)"
        ),
        ignore_fails=ignore_fails,
    )

    tc_ = deepcopy(tc)
    test, ref = stability_test(
        func=get_product_hostfunc,
        func_kw=dict(
            energy=tc_["energy"],
            dm_mat_vac=dm_mat_vac_ref,
            dm_mat_mat=dm_mat_mat_ref,
            H_mat_mass_eigenstate_basis=H_mat_ref,
            # output:
            product=np.ones(shape=(3, 3, 3), dtype=CX),
        ),
        ref_path=join(TEST_DATA_DIR, f"product_hostfunc{tf_sfx}"),
        **st_test_kw,
    )
    logging.debug("\nproduct = %s", ary2str(test["product"]))


def stability_test(func, func_kw, ref_path, ignore_fails=False, define_as_ref=False):
    """basic stability test of a Numba CPUDispatcher function (i.e., function
    compiled via @jit / @njit)"""
    func_name = func.py_func.__name__
    logging.info("stability testing `%s`", func_name)
    ref_path = expand(ref_path)

    test = execute_func(func=func, func_kw=func_kw)

    if define_as_ref:
        to_file(test, ref_path)

    # Even when we define the test case as ref, round-trip to/from file to
    # ensure that doesn't corrupt the values
    ref = from_file(ref_path)

    check(test=test, ref=ref, label=func_name, ignore_fails=ignore_fails)

    return test, ref


def execute_func(func, func_kw):
    """Run `func` with *func_kw.values() where `outputs` specify names in
    `func_kw` taken to be outputs of the function; for these, mark changed.
    Retrieve both input and output values as Numpy arrays on the host and
    aggregate together in a single dict before returning.

    Parameters
    ----------
    func : numba CPUDispatcher or CUDADispatcher
    func_kw : OrderedDict

    Returns
    -------
    ret_dict : OrderedDict
        Keys are arg names and vals are type-"correct" values; all arrays are
        converted to host Numpy arrays

    """
    py_func = func.py_func
    func_name = ".".join([getmodule(py_func).__name__, py_func.__name__])
    arg_names = list(signature(py_func).parameters.keys())
    if hasattr(func, "signatures"):
        arg_types = func.signatures[0]
    else:
        arg_types = func.compiled.argument_types

    # Convert types; wrap arrays with SmartArray and place on device (if necessary)

    missing = set(arg_names).difference(func_kw.keys())
    excess = set(func_kw.keys()).difference(arg_names)
    if missing or excess:
        msgs = []
        if missing:
            msgs.append(f"missing kwargs {missing}")
        if excess:
            msgs.append(f"excess kwargs {excess}")
        raise KeyError(f"{func_name}:" + ", ".join(msgs))

    typed_args = OrderedDict()
    for arg_name, arg_type in zip(arg_names, arg_types):
        val = func_kw[arg_name]
        if arg_type.name.startswith("array"):
            arg_val = SmartArray(val.astype(arg_type.dtype.key))
            arg_val = arg_val.get("host")
        else:
            arg_val = arg_type(val)
        typed_args[arg_name] = arg_val

    # Call the host function with typed args

    try:
        func(*list(typed_args.values()))
    except Exception:
        logging.error("Failed running `%s` with args %s", func_name, str(typed_args))
        raise

    # All arrays converted to Numpy host arrays

    ret_dict = OrderedDict()
    for key, val in typed_args.items():
        if isinstance(val, SmartArray):
            val.mark_changed(WHERE)
            val = val.get("host")
        ret_dict[key] = val

    return ret_dict


def compare_numeric(test, ref, label=None, ac_kw=deepcopy(AC_KW), ignore_fails=False):
    """Compare scalars or numpy ndarrays.

    Parameters
    ----------
    test : scalar or numpy.ndarray
    ref : scalar or numpy.ndarray
    label : str or None, optional
    ac_kw : mapping, optional
        Keyword args to pass via **ac_kw to `numpy.isclose` / `numpy.allclose`
    ignore_fails : bool, optional

    Returns
    -------
    rslt : bool

    """
    pfx = f"{label} :: " if label else ""
    with np.printoptions(**PRINTOPTS):
        if np.isscalar(test):
            if np.isclose(test, ref, **ac_kw):
                return True

            msg = f"{pfx}test: {test} != ref: {ref}"
            if ignore_fails:
                logging.warning(msg)
            else:
                logging.error(msg)
            return False

        # Arrays
        if np.allclose(test, ref, **ac_kw):
            return True

        diff = test - ref
        msg = f"{pfx}test:" f"\n{(test)}\n!= ref:\n{(ref)}" f"\ndiff:\n{(diff)}"

        if not np.all(ref == 1):
            nzmask = ref != 0
            zmask = ref == 0
            fdiff = np.empty_like(ref)
            fdiff[nzmask] = diff[nzmask] / ref[nzmask]
            fdiff[zmask] = np.nan
            msg += f"\nfractdiff:\n{(fdiff)}"

        if ignore_fails:
            logging.warning(msg)
        else:
            logging.error(msg)

        return False


def check(test, ref, label=None, ac_kw=deepcopy(AC_KW), ignore_fails=False):
    """Check that `test` matches `ref` (closely enough).

    Parameters
    ----------
    test
    ref
    ac_kw : mapping, optional
        Kwargs to `np.allclose`, as used by
        `pisa.utils.comparisons.recursiveEquality`
    ignore_fails : bool, optional
        If True and comparison fails, do not raise AssertionError

    Raises
    ------
    AssertionError
        If `test` is not close enough to `ref` and ignore_fails is False

    """
    same = True
    with np.printoptions(**PRINTOPTS):
        if isinstance(test, Mapping):
            if not label:
                label = ""
            else:
                label = label + ": "

            for key, val in test.items():
                same &= compare_numeric(
                    test=val,
                    ref=ref[key],
                    label=label + f"key: '{key}'",
                    ac_kw=ac_kw,
                    ignore_fails=ignore_fails,
                )
        else:
            same &= compare_numeric(
                test=test, ref=ref, label=label, ac_kw=ac_kw, ignore_fails=ignore_fails
            )
    if not ignore_fails and not same:
        assert False
    return same


def ary2str(array):
    """Convert a numpy ndarray to string easy to copy back into code"""
    return "np.array(" + np.array2string(array, **A2S_KW) + ")"


# Augment test cases with defaults / contruct arrays where necessary
for TC in TEST_CASES.values():
    auto_populate_test_case(tc=TC, defaults=DEFAULTS)


def main(description=__doc__):
    """Script interface for `test_prob3numba` function"""
    parser = ArgumentParser(description=description)
    parser.add_argument("--ignore-fails", action="store_true")
    parser.add_argument("--define-as-ref", action="store_true")
    parser.add_argument("-v", action="count", default=Levels.WARN)
    kwargs = vars(parser.parse_args())
    set_verbosity(kwargs.pop("v"))
    test_prob3numba(**kwargs)


if __name__ == "__main__":
    main()
