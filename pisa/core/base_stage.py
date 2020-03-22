"""
Stage base class designed to be inherited by PISA core stages (i.e., the PISA cake base
class, `stage` , and the PISA pi base class, `pi_stage`) such that all basic
functionality is built-in.
"""

from __future__ import absolute_import

__all__ = ["BaseStage"]

from collections.abc import Mapping
from copy import deepcopy
import inspect

from pisa.core.param import ParamSelector
from pisa.utils.format import arg_str_seq_none
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging


class BaseStage(object):
    """
    PISA base stage base class. Should encompass all behaviors common to (almost)
    all stages.

    Specialization should be done via subclasses.

    Parameters
    ----------
    params : ParamSelector, dict of ParamSelector kwargs, ParamSet, or object instantiable to ParamSet

    expected_params : list of strings
        List containing required `params` names.

    input_names : None or list of strings

    output_names : None or list of strings

    debug_mode : None, bool, or string
        If None, False, or empty string, the stage runs normally.

        Otherwise, the stage runs in debug mode. This disables caching (forcing
        recomputation of any nominal transforms, transforms, and outputs).
        Services that subclass from the `Stage` class can then implement
        further custom behavior when this mode is set by reading the value of
        the `self.debug_mode` attribute.

    error_method : None, bool, or string
        If None, False, or empty string, the stage does not compute errors for
        the transforms and does not apply any (additional) error to produce its
        outputs. (If the inputs already have errors, these are propagated.)

        Otherwise, this specifies the method by which the stage should compute
        errors for the transforms to be applied in producing outputs from the
        stage.

    Notes
    -----
    The following methods can be overridden in derived classes where
    applicable:
        validate_params
            Perform validation on any parameters.

    """

    def __init__(
        self,
        params=None,
        expected_params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        error_method=None,
    ):
        # Allow for string inputs, but have to populate into lists for
        # consistent interfacing to one or multiple of these things
        expected_params = arg_str_seq_none(expected_params, "expected_params")
        input_names = arg_str_seq_none(input_names, "input_names")
        output_names = arg_str_seq_none(output_names, "output_names")

        module_path = self.__module__.split(".")

        self.stage_name = module_path[-2]
        """Name of the stage (e.g. flux, osc, aeff, reco, pid, etc."""

        self.service_name = module_path[-1]
        """Name of the specific service implementing the stage."""

        self.expected_params = expected_params
        """The full set of parameters (by name) that must be present in
        `params`"""

        self._input_names = [] if input_names is None else input_names
        self._output_names = [] if output_names is None else output_names

        self._source_code_hash = None

        self.outputs = None
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

        if bool(error_method):
            self._error_method = error_method
        else:
            self._error_method = None

        self.inputs = None

    def setup(self):
        """Override in inheriting class"""
        pass

    def run(self, inputs=None):  # pylint: disable=unused-argument, no-self-use
        """Override in inheriting class"""
        return None

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
    def input_names(self):
        """Names of input objects (e.g. names of input maps)"""
        return deepcopy(self._input_names)

    @property
    def output_names(self):
        """Names of output objects (e.g. names of output maps)"""
        return deepcopy(self._output_names)

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
