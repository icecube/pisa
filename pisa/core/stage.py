"""
Stage class designed to be inherited by PISA services, such that all basic
functionality is built-in.
"""

from __future__ import absolute_import, division

from copy import deepcopy
from collections import OrderedDict
from collections.abc import Mapping
import inspect
import numpy as np
from tabulate import tabulate
from time import time

from pisa.core.binning import MultiDimBinning
from pisa.core.container import ContainerSet
from pisa.utils.log import logging
from pisa.utils.format import arg_to_tuple
from pisa.utils.profiler import profile
from pisa.core.param import ParamSelector
from pisa.utils.format import arg_str_seq_none
from pisa.utils.hash import hash_obj


__all__ = ["Stage"]
__author__ = "Philipp Eller, J. Lanfranchi"


class Stage():
    """
    PISA stage base class. Should be used to implement PISA Pi stages

    Specialization should be done via subclasses.

    Parameters
    ----------
    data : ContainerSet or None
        object to be passed along

    params : ParamSelector, dict of ParamSelector kwargs, ParamSet, or object instantiable to ParamSet

    expected_params : list of strings
        List containing required `params` names.

    debug_mode : None, bool, or string
        If None, False, or empty string, the stage runs normally.

        Otherwise, the stage runs in debug mode. This disables caching (forcing
        recomputation of any nominal transforms, transforms, and outputs).
        Services that subclass from the `Stage` class can then implement
        further custom behavior when this mode is set by reading the value of
        the `self.debug_mode` attribute.

    calc_mode : pisa.core.binning.MultiDimBinning, str, or None
        Specify in what to do the calculation

    apply_mode : pisa.core.binning.MultiDimBinning, str, or None
        Specify in what to do the application

    """

    def __init__(
        self,
        data=None,
        params=None,
        expected_params=None,
        debug_mode=None,
        error_method=None,
        calc_mode=None,
        apply_mode=None,
        profile=False,
    ):
        # Allow for string inputs, but have to populate into lists for
        # consistent interfacing to one or multiple of these things
        expected_params = arg_str_seq_none(expected_params, "expected_params")

        module_path = self.__module__.split(".")

        self.stage_name = module_path[-2]
        """Name of the stage (e.g. flux, osc, aeff, reco, pid, etc."""

        self.service_name = module_path[-1]
        """Name of the specific service implementing the stage."""

        self.expected_params = expected_params
        """The full set of parameters (by name) that must be present in
        `params`"""

        self._source_code_hash = None

        """Last-computed outputs; None if no outputs have been computed yet."""

        self._attrs_to_hash = set([])
        """Attributes of the stage that are to be included in its hash value"""

        self.full_hash = True
        """Whether to do full hashing if true, otherwise do fast hashing"""

        param_selector_keys = set(
            ["regular_params", "selector_param_sets", "selections"]
        )
        if isinstance(params, Mapping) and set(params.keys()) == param_selector_keys:
            self._param_selector = ParamSelector(**params)
        elif isinstance(params, ParamSelector):
            self._param_selector = params
        else:
            self._param_selector = ParamSelector(regular_params=params)

        # Get the params from the ParamSelector, validate, and set as the
        # params object for this stage
        p = self._param_selector.params
        self._check_params(p)
        self.validate_params(p)
        self._params = p

        if bool(debug_mode):
            self._debug_mode = debug_mode
        else:
            self._debug_mode = None


        self.calc_mode = calc_mode
        self.apply_mode = apply_mode
        self.data = data

        self._error_method = error_method

        self.param_hash = None

        self.profile = profile
        self.setup_times = []
        self.calc_times = []
        self.apply_times = []


    def __repr__(self):
        return 'Stage "%s"'%(self.__class__.__name__)

    def report_profile(self, detailed=False):
        for stage in self.stages:
            stage.report_profile(detailed=detailed)

    def report_profile(self, detailed=False):
        def format(times):
            tot = np.sum(times)
            n = len(times)
            ave = 0. if n == 0 else tot/n
            return 'Total time %.5f s, n calls: %i, time/call: %.5f s'%(tot, n, ave)

        print(self.stage_name, self.service_name)
        print('- setup: ', format(self.setup_times))
        if detailed:
            print('         Individual runs: ', ', '.join(['%i: %.3f s' % (i, t) for i, t in enumerate(self.setup_times)]))
        print('- calc:  ', format(self.calc_times))
        if detailed:
            print('         Individual runs: ', ', '.join(['%i: %.3f s' % (i, t) for i, t in enumerate(self.calc_times)]))
        print('- apply: ', format(self.apply_times))
        if detailed:
            print('         Individual runs: ', ', '.join(['%i: %.3f s' % (i, t) for i, t in enumerate(self.apply_times)]))

    def select_params(self, selections, error_on_missing=False):
        """Apply the `selections` to contained ParamSet.

        Parameters
        ----------
        selections : string or iterable
        error_on_missing : bool

        """
        try:
            self._param_selector.select_params(selections, error_on_missing=True)
        except KeyError:
            msg = "Not all of the selections %s found in this stage." % (selections,)
            if error_on_missing:
                # logging.error(msg)
                raise
            logging.trace(msg)
        else:
            logging.trace(
                "`selections` = %s yielded `params` = %s" % (selections, self.params)
            )

    def _check_params(self, params):
        """Make sure that `expected_params` is defined and that exactly the
        params specified in self.expected_params are present.

        """
        assert self.expected_params is not None
        exp_p, got_p = set(self.expected_params), set(params.names)
        if exp_p == got_p:
            return
        excess = got_p.difference(exp_p)
        missing = exp_p.difference(got_p)
        err_strs = []
        if len(excess) > 0:
            err_strs.append("Excess params provided: %s" % ", ".join(sorted(excess)))
        if len(missing) > 0:
            err_strs.append("Missing params: %s" % ", ".join(sorted(missing)))
        raise ValueError(
            "Expected parameters: %s;\n" % ", ".join(sorted(exp_p))
            + ";\n".join(err_strs)
        )

    @property
    def params(self):
        """Params"""
        return self._params

    @property
    def param_selections(self):
        """Param selections"""
        return sorted(deepcopy(self._param_selector.param_selections))

    @property
    def source_code_hash(self):
        """Hash for the source code of this object's class.

        Not meant to be perfect, but should suffice for tracking provenance of
        an object stored to disk that were produced by a Stage.
        """
        if self._source_code_hash is None:
            self._source_code_hash = hash_obj(
                inspect.getsource(self.__class__), full_hash=self.full_hash
            )
        return self._source_code_hash

    @property
    def hash(self):
        """Combines source_code_hash and params.hash for checking/tagging
        provenance of persisted (on-disk) objects."""
        objects_to_hash = [self.source_code_hash, self.params.hash]
        for attr in sorted(self._attrs_to_hash):
            objects_to_hash.append(
                hash_obj(getattr(self, attr), full_hash=self.full_hash)
            )
        return hash_obj(objects_to_hash, full_hash=self.full_hash)

    def __hash__(self):
        return self.hash

    def include_attrs_for_hashes(self, attrs):
        """Include a class attribute or attributes to be included when
        computing hashes (for all that apply: nominal transforms, transforms,
        and/or outputs).

        This is a convenience that allows some customization of hashing (and
        hence caching) behavior without having to override the hash-computation
        methods (`_derive_nominal_transforms_hash`, `_derive_transforms_hash`,
        and `_derive_outputs_hash`).

        Parameters
        ----------
        attrs : string or sequence thereof
            Name of the attribute(s) to include for hashes. Each must be an
            existing attribute of the object at the time this method is
            invoked.

        """
        if isinstance(attrs, str):
            attrs = [attrs]

        # Validate that all are actually attrs before setting any
        for attr in attrs:
            assert isinstance(attr, str)
            if not hasattr(self, attr):
                raise ValueError(
                    '"%s" not an attribute of the class; not'
                    " adding *any* of the passed attributes %s to"
                    " attrs to hash." % (attr, attrs)
                )

        # Include the attribute names
        for attr in attrs:
            self._attrs_to_hash.add(attr)

    @property
    def debug_mode(self):
        """Read-only attribute indicating whether or not the stage is being run
        in debug mode. None indicates non-debug mode, while non-none value
        indicates a debug mode."""
        return self._debug_mode

    def validate_params(self, params):  # pylint: disable=unused-argument, no-self-use
        """Override this method to test if params are valid; e.g., check range
        and dimensionality. Invalid params should be indicated by raising an
        exception; no value should be returned."""
        return

    @property
    def error_method(self):
        """Read-only attribute indicating whether or not the stage will compute
        errors for its transforms and outputs (whichever is applicable). Errors
        on inputs are propagated regardless of this setting."""
        return self._error_method

    @property
    def is_map(self):
        return self.data.is_map

    def setup(self):

        # check that data is a ContainerSet (downstream modules assume this)
        if self.data is not None:
            if not isinstance(self.data, ContainerSet):
                raise TypeError("`data` must be a `pisa.core.container.ContainerSet`")

        if self.calc_mode is not None:
            self.data.representation = self.calc_mode

        # call the user-defined setup function
        if self.profile:
            start_t = time()
            self.setup_function()
            end_t = time()
            self.setup_times.append(end_t - start_t)
        else:
            self.setup_function() 
            
        # invalidate param hash:
        self.param_hash = -1

    def setup_function(self):
        """Implement in services (subclasses of Stage)"""
        pass

    def compute(self):

        # simplest caching algorithm: don't compute if params didn't change
        new_param_hash = self.params.values_hash
        if new_param_hash == self.param_hash:
            logging.trace("cached output")
            return

        if self.calc_mode is not None:
            self.data.representation = self.calc_mode

        if self.profile:
            start_t = time()
            self.compute_function()
            end_t = time()
            self.calc_times.append(end_t - start_t)
        else:
            self.compute_function()
        self.param_hash = new_param_hash

    def compute_function(self):
        """Implement in services (subclasses of Stage)"""
        pass

    def apply(self):

        if self.apply_mode is not None:
            self.data.representation = self.apply_mode

        if self.profile:
            start_t = time()
            self.apply_function()
            end_t = time()
            self.apply_times.append(end_t - start_t)
        else:
            self.apply_function()


    def apply_function(self):
        """Implement in services (subclasses of Stage)"""
        pass

    def run(self):
        self.compute()
        self.apply()
        return None

