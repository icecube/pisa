"""
Stage class designed to be inherited by PISA services, such that all basic
functionality is built-in.
"""

from __future__ import absolute_import, division

from copy import deepcopy
from collections.abc import Mapping
import inspect
from time import time

from pisa.core.binning import MultiDimBinning
from pisa.core.container import Container, ContainerSet
from pisa.utils.format import format_times
from pisa.utils.log import logging
from pisa.core.param import ParamSelector
from pisa.utils.format import arg_str_seq_none
from pisa.utils.hash import hash_obj


__all__ = ["Stage"]
__author__ = "Philipp Eller, J. Lanfranchi"


class Stage():
    """
    PISA stage base class.

    Specialization should be done via subclasses.

    Parameters
    ----------
    data : ContainerSet or None
        object to be passed along

    params : ParamSelector, dict of ParamSelector kwargs, ParamSet, or object instantiable to ParamSet

    expected_params : list of strings
        List containing required `params` names.

    expected_container_keys: list of strings
        List containing required container keys.

    debug_mode : None, bool, or string
        If None, False, or empty string, the stage runs normally.

        Otherwise, the stage runs in debug mode. This disables caching
        (TODO: where or how?).
        Services that subclass from the `Stage` class can then implement
        further custom behavior when this mode is set by reading the value of
        the `self.debug_mode` attribute.

    error_method : None or string (not enforced)
        An option to define one or more dedicated error calculation methods
        for the stage transforms or outputs

    supported_reps : dict
        Dictionary containing the representations allowed for calc_mode and
        apply_mode. If nothing is specified, Container.array_representations
        plus MultiDimBinning is assumed. Should have keys `calc_mode` and/or
        `apply_mode`, they will be created if not there.

    calc_mode : pisa.core.binning.MultiDimBinning, str, or None
        Specify the default data representation for `setup()` and `compute()`

    apply_mode : pisa.core.binning.MultiDimBinning, str, or None
        Specify the default data representation for `apply()`

    profile : bool
        If True, perform timings for the setup, compute, and apply functions.

    in_standalone_mode : bool
        If True, assume stage is not part of a pipeline. Affects whether
        `setup()` can be automatically rerun whenever `calc_mode` is
        changed.

    """

    def __init__(
        self,
        data=None,
        params=None,
        expected_params=None,
        expected_container_keys=None,
        debug_mode=None,
        error_method=None,
        supported_reps=None,
        calc_mode=None,
        apply_mode=None,
        profile=False,
        in_standalone_mode=False,
    ):
        # Allow for string inputs, but have to populate into lists for
        # consistent interfacing to one or multiple of these things
        expected_params = arg_str_seq_none(
            inputs=expected_params, name="expected_params"
        )

        # dito
        expected_container_keys = arg_str_seq_none(
            inputs=expected_container_keys, name="expected_container_keys"
        )

        module_path = self.__module__.split(".")

        self.stage_name = module_path[-2]
        """Name of the stage (flux, osc, aeff, reco, pid, etc.)"""

        self.service_name = module_path[-1]
        """Name of the specific service implementing the stage."""

        self.expected_params = expected_params
        """The full set of parameters (by name) that must be present in
        `params`"""

        self.expected_container_keys = expected_container_keys
        """The full set of keys that is expected to be present in each
        container within `data`"""

        self._source_code_hash = None
        """Hash of the source code"""

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

        self._check_params(p, p.has_derived)
        self.validate_params(p)
        self._params = p

        if bool(debug_mode):
            self._debug_mode = debug_mode
        else:
            self._debug_mode = None

        if supported_reps is None:
            supported_reps = {}
        assert isinstance(supported_reps, Mapping)
        if 'calc_mode' not in supported_reps:
            supported_reps['calc_mode'] = list(Container.array_representations) + [MultiDimBinning]
        if 'apply_mode' not in supported_reps:
            supported_reps['apply_mode'] = list(Container.array_representations) + [MultiDimBinning]
        self.supported_reps = supported_reps

        self._check_representation(rep=calc_mode, mode='calc_mode', allow_None=True)
        self._calc_mode = calc_mode

        self._check_representation(rep=apply_mode, mode='apply_mode', allow_None=True)
        self._apply_mode = apply_mode

        self._error_method = error_method

        self.param_hash = None
        """Hash of stage params. Also serves as an indicator of whether `setup()`
        has already been called."""

        self.profile = profile
        """Whether to perform timings"""

        self.setup_times = []
        self.calc_times = []
        self.apply_times = []

        self.in_standalone_mode = in_standalone_mode
        """Whether stage is standalone or part of a pipeline"""

        self.data = data
        """Data based on which stage may make computations and which it may
        modify"""

    def __repr__(self):
        return 'Stage "%s"'%(self.__class__.__name__)

    def report_profile(self, detailed=False, **format_num_kwargs):
        """Report timing information on calls to setup, compute, and apply
        """
        print(self.stage_name, self.service_name)
        for func_str, times in [
            ('- setup:   ', self.setup_times),
            ('- compute: ', self.calc_times),
            ('- apply:   ', self.apply_times)
        ]:
            print(func_str,
                format_times(times=times,
                             nindent_detailed=len(func_str) + 1,
                             detailed=detailed, **format_num_kwargs)
            )

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

    def _check_params(self, params, ignore_excess=False):
        """Make sure that `expected_params` is defined and that exactly the
        params specified in self.expected_params are present.

        An exception is made for having excess parameters if `ignore_excess` is True
        This is useful for stages with `DerivedParams` that are used, but not explicitly accessed by the stage
        """
        assert self.expected_params is not None
        exp_p, got_p = set(self.expected_params), set(params.names)
        if exp_p == got_p:
            return
        excess = got_p.difference(exp_p)
        missing = exp_p.difference(got_p)
        err_strs = []
        if len(missing) > 0:
            err_strs.append("Missing params: %s" % ", ".join(sorted(missing)))


        if len(excess) > 0:
            if ignore_excess: #excess isn't a problem
                if len(err_strs)==0: # return if there aren't any problems already
                    return
            else:
                err_strs.append("Excess params provided: %s" % ", ".join(sorted(excess)))

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
    def calc_mode(self):
        """calc_mode"""
        return self._calc_mode

    @calc_mode.setter
    def calc_mode(self, value):
        """Set `calc_mode` after checking the validity of `value`, and,
        in standalone mode, rerun `setup()` if has already been executed at
        least once."""
        if value != self.calc_mode:
            self._check_representation(rep=value, mode='calc_mode')
            self._calc_mode = value
            if self.in_standalone_mode and self.param_hash is not None:
                # Only in standalone mode: repeat setup automatically only if
                # setup has already been run; first reset `data` to pre-setup state
                # TODO: test this scenario
                self.data = deepcopy(self._original_data)
                self.setup()
            # In non-standalone (i.e., pipeline) mode, the user has to make sure
            # they setup the pipeline again

    @property
    def data(self):
        """data"""
        return self._data

    @data.setter
    def data(self, value):
        """Keep a copy of any possible pre-setup `data` around whenever
        we are in standalone mode, and update it whenever `data`
        is updated before any call to `setup()`."""
        if self.param_hash is None and self.in_standalone_mode:
            self._original_data = deepcopy(value)
        self._data = value

    @property
    def apply_mode(self):
        """apply_mode"""
        return self._apply_mode

    @apply_mode.setter
    def apply_mode(self, value):
        """Set `apply_mode` after checking the validity of `value`"""
        if value != self.apply_mode:
            self._check_representation(rep=value, mode='apply_mode')
            self._apply_mode = value

    def _check_representation(self, rep, mode, allow_None=False):
        if isinstance(rep, str) and rep not in self.supported_reps[mode]:
            raise ValueError(
                f"{mode} {rep} is not supported by {self.stage_name}"
                f".{self.service_name}"
            )
        if (not isinstance(rep, str) and type(rep) not in self.supported_reps[mode]
            and (rep is not None or not allow_None)):
            raise ValueError(
                f"{mode} {type(rep)} is not supported by {self.stage_name}"
                f".{self.service_name}"
            )

    def _check_exp_keys_in_data(self, error_on_missing=False):
        """Make sure that `expected_container_keys` is defined and that
        they are present in all containers among `self.data`, independent of
        representation.

        Parameters
        ----------
        error_on_missing : bool
            Whether to raise error upon missing keys (default: False)

        Returns
        -------
        bool

        """
        if self.data is None:
            # nothing to do, we were probably directly instantiated with data
            # set to None
            return
        if self.expected_container_keys is None:
            raise ValueError(
                'Service %s.%s is not specifying expected container keys.'
                % (self.stage_name, self.service_name)
            )
        exp_k = set(self.expected_container_keys)
        got_k = set(self.data.get_shared_keys(rep_indep=True))
        missing = exp_k.difference(got_k)
        if len(missing) > 0:
            err_str = "Service %s.%s," % (self.stage_name, self.service_name)
            err_str += " expected container keys: %s," % ", ".join(sorted(exp_k))
            err_str += " but containers are missing keys: %s" % ", ".join(sorted(missing))
            if error_on_missing:
                raise ValueError(err_str)
            # TODO: warning could be confusing until we have made setup-time
            # check foolproof
            #logging.warn(err_str)
        return

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
        """Include a class attribute or attributes in the hash
        computation.

        This is a convenience that allows some customization of hashing (and
        hence caching) behavior without having to override the hash-computation
        method.

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
        """See ContainerSet.is_map for documentation"""
        return self.data.is_map

    def setup(self):

        # check that data is a ContainerSet (downstream modules assume this)
        if self.data is not None:
            if not isinstance(self.data, ContainerSet):
                raise TypeError("`data` must be a `pisa.core.container.ContainerSet`")

            self._check_exp_keys_in_data(error_on_missing=False)

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
