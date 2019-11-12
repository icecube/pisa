"""
Stage base class designed to be inherited by PISA services, such that all basic
functionality is built-in.
"""


from __future__ import absolute_import, division, print_function

import os

from pisa import CACHE_DIR
from pisa.core.base_stage import BaseStage
from pisa.core.events import Events
from pisa.core.map import Map, MapSet
from pisa.core.param import Param
from pisa.core.transform import TransformSet
from pisa.utils.cache import DiskCache, MemoryCache
from pisa.utils.comparisons import normQuant
from pisa.utils.fileio import mkdir
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.resources import find_resource


__all__ = ["Stage"]
__version__ = "Cake"

__author__ = "J.L. Lanfranchi"

__license__ = """Copyright (c) 2014-2018, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


# TODO: mode for not propagating errors. Probably needs hooks here, but meat of
# implementation would live inside map.py and/or transform.py.


class Stage(BaseStage):
    """
    PISA stage base class. Should encompass all behaviors common to (almost)
    all stages.

    Specialization should be done via subclasses.

    Parameters
    ----------
    use_transforms : bool (required)
        Whether or not this stage takes inputs to be transformed (and hence
        implements transforms).

    input_names : None or list of strings

    output_names : None or list of strings

    disk_cache : None, bool, string, or DiskCache
      * If None or False, no disk cache is available.
      * If True, a disk cache is generated at the path
        `CACHE_DIR/<stage_name>/<service_name>.sqlite` where CACHE_DIR is
        defined in pisa.__init__
      * If string, this is interpreted as a path. If an absolute path is
        provided (e.g. "/home/myuser/mycache.sqlite'), this locates the disk
        cache file exactly, while a relative path (e.g.,
        "relative/dir/mycache.sqlite") is taken relative to the CACHE_DIR; the
        aforementioned example will be turned into
        `CACHE_DIR/relative/dir/mycache.sqlite`.
      * If a DiskCache object is passed, it will be used directly

    memcache_deepcopy : bool
        Whether to deepcopy objects prior to storing to the memory cache and
        upon loading these objects from the memory cache. Setting to True
        ensures no modification of mutable objects stored to a memory cache
        will affect other logic relying on that object remaining unchanged.
        However, this comes at the cost of more memory used and slower
        operations.

    outputs_cache_depth : int >= 0

    transforms_cache_depth : int >= 0

    input_binning : None or interpretable as MultiDimBinning

    output_binning : None or interpretable as MultiDimBinning


    Notes
    -----
    The following methods can be overridden in derived classes where
    applicable:
        _derive_nominal_transforms_hash
        _derive_transforms_hash
        _derive_nominal_outputs_hash
        _derive_outputs_hash
        _compute_nominal_transforms
            This is called during initialization to compute what are termed
            "nominal" transforms -- i.e, transforms with all systematic
            parameters set to their nominal values, such that they have no
            effect on the transform. It is optional to use this stage, but if
            it *is* used, then the result will be cached to memory (and
            optionally to disk cache, if one is provided) for future use. A
            nominal transform is useful when systematic parameters merely have
            the effect of modifying the nominal transform, rather than
            requiring a complete recomputation of the transform.
        _compute_nominal_outputs
            same as nominal transforms, but for outputs (e.g. used for
            non-input stages)
        _compute_transforms
            Do the actual work to produce the stage's transforms. For stages
            that specify use_transforms=False, this method is never called.
        _compute_outputs
            Do the actual work to compute the stage's output. Default
            implementation is to call self.transforms.apply(inputs); override
            if no transforms are present or if more needs to be done to
            compute outputs than this.
        validate_params
            Perform validation on any parameters.

    """

    def __init__(
        self,
        use_transforms,
        params=None,
        expected_params=None,
        input_names=None,
        output_names=None,
        error_method=None,
        disk_cache=None,
        memcache_deepcopy=True,
        transforms_cache_depth=10,
        outputs_cache_depth=0,
        input_binning=None,
        output_binning=None,
        debug_mode=None,
    ):
        # Allow for string inputs, but have to populate into lists for
        # consistent interfacing to one or multiple of these things

        logging.warning('This is a cake-style PISA stage, which is DEPRECATED!')

        self.use_transforms = use_transforms
        """Whether or not stage uses transforms"""

        self._events_hash = None

        self.input_binning = input_binning
        self.output_binning = output_binning
        self.validate_binning()

        # init base class!
        super().__init__(
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            error_method=error_method,
        )

        # Storage of latest transforms and outputs; default to empty
        # TransformSet and None, respectively.
        self.transforms = TransformSet([])
        """A stage that takes to-be-transformed inputs and has had these
        transforms computed stores them here. Before computation, `transforms`
        is an empty TransformSet; a stage that does not make use of these (such
        as a no-input stage) has an empty TransformSet."""

        self.memcache_deepcopy = memcache_deepcopy

        self.transforms_cache_depth = int(transforms_cache_depth)

        self.transforms_cache = None
        """Memory cache object for storing transforms"""

        self.nominal_transforms_cache = None
        """Memory cache object for storing nominal transforms"""

        self.full_hash = True
        """Whether to do full hashing if true, otherwise do fast hashing"""

        self.transforms_cache = MemoryCache(
            max_depth=self.transforms_cache_depth,
            is_lru=True,
            deepcopy=self.memcache_deepcopy,
        )
        self.nominal_transforms_cache = MemoryCache(
            max_depth=self.transforms_cache_depth,
            is_lru=True,
            deepcopy=self.memcache_deepcopy,
        )

        self.outputs_cache_depth = int(outputs_cache_depth)

        self.outputs_cache = None
        """Memory cache object for storing outputs (excludes sideband
        objects)."""

        self.outputs_cache = None
        if self.outputs_cache_depth > 0:
            self.outputs_cache = MemoryCache(
                max_depth=self.outputs_cache_depth,
                is_lru=True,
                deepcopy=self.memcache_deepcopy,
            )

        self.disk_cache = disk_cache
        """Disk cache object"""

        self.disk_cache_path = None
        """Path to disk cache file for this stage/service (or None)."""

        # Include each attribute here for hashing if it is defined and its
        # value is not None
        default_attrs_to_hash = [
            "input_names",
            "output_names",
            "input_binning",
            "output_binning",
        ]
        self._attrs_to_hash = set([])
        for attr in default_attrs_to_hash:
            if not hasattr(self, attr):
                continue
            val = getattr(self, attr)
            if val is None:
                continue
            try:
                self.include_attrs_for_hashes(attr)
            except ValueError():
                pass

        self.events = None
        self.nominal_transforms = None

        # Define useful flags and values for debugging behavior after running

        self.nominal_transforms_loaded_from_cache = None
        """Records which cache nominal transforms were loaded from, or None."""

        self.nominal_transforms_computed = False
        """Records whether nominal transforms were (re)computed."""

        self.transforms_loaded_from_cache = None
        """Records which cache transforms were loaded from, or None."""

        self.transforms_computed = False
        """Records whether transforms were (re)computed."""

        self.nominal_outputs_computed = False
        """Records whether nominal outputs were (re)computed."""

        self.outputs_loaded_from_cache = None
        """Records which cache outputs were loaded from, or None."""

        self.outputs_computed = False
        """Records whether outputs were (re)computed."""

        self.nominal_transforms_hash = None
        self.transforms_hash = None
        self.nominal_outputs_hash = None
        self.outputs_hash = None
        self.instantiate_disk_cache()

    @profile
    def get_nominal_transforms(self, nominal_transforms_hash):
        """Load a cached transform from the nominal transform memory cache
        (which is backed by a disk cache, if one is specified) if the nominal
        transform is in the cache, or else recompute it and store to the
        cache(s).

        This method calls the `_compute_nominal_transforms` method, which by
        default does nothing.

        However, if you want to use the nominal transforms feature, override
        the `_compute_nominal_transforms` method and fill in the logic there.

        Deciding whether to invoke the `_compute_nominal_transforms` method or
        to load the nominal transforms from cache is done here, so you needn't
        think about any of this within the `_compute_nominal_transforms`
        method.

        Returns
        -------
        nominal_transforms, hash

        """
        # Reset flags
        self.nominal_transforms_loaded_from_cache = None
        self.nominal_transforms_computed = False

        if nominal_transforms_hash is None:
            nominal_transforms_hash = self._derive_nominal_transforms_hash()

        nominal_transforms = None
        # Quick way to avoid further logic is if hash value is None
        if nominal_transforms_hash is None:
            self.nominal_transforms_hash = None
            self.nominal_transforms = None
            return self.nominal_transforms, self.nominal_transforms_hash

        recompute = True
        # If hash found in memory cache, load nominal transforms from there
        if (
            nominal_transforms_hash in self.nominal_transforms_cache
            and self.debug_mode is None
        ):
            nominal_transforms = self.nominal_transforms_cache[nominal_transforms_hash]
            self.nominal_transforms_loaded_from_cache = "memory"
            recompute = False

        # Otherwise try to load from an extant disk cache
        elif self.disk_cache is not None and self.debug_mode is None:
            try:
                nominal_transforms = self.disk_cache[nominal_transforms_hash]
            except KeyError:
                pass
            else:
                self.nominal_transforms_loaded_from_cache = "disk"
                recompute = False
                # Save to memory cache
                self.nominal_transforms_cache[
                    nominal_transforms_hash
                ] = nominal_transforms

        if recompute:
            self.nominal_transforms_computed = True
            nominal_transforms = self._compute_nominal_transforms()
            if nominal_transforms is None:
                # Invalidate hash value since found transforms
                nominal_transforms_hash = None
            else:
                nominal_transforms.hash = nominal_transforms_hash
                self.nominal_transforms_cache[
                    nominal_transforms_hash
                ] = nominal_transforms
                if self.disk_cache is not None:
                    self.disk_cache[nominal_transforms_hash] = nominal_transforms

        self.nominal_transforms = nominal_transforms
        self.nominal_transforms_hash = nominal_transforms_hash
        return nominal_transforms, nominal_transforms_hash

    @profile
    def get_transforms(self, transforms_hash=None, nominal_transforms_hash=None):
        """Load a cached transform (keyed on hash of parameter values) if it
        is in the cache, or else compute a new transform from currently-set
        parameter values and store this new transform to the cache.

        This calls the private method _compute_transforms (which must be
        implemented in subclasses if the nominal transform feature is desired)
        to generate a new transform if the nominal transform is not found in
        the nominal transform cache.

        Notes
        -----
        The hash used here is only meant to be valid within the scope of a
        session; a hash on the full parameter set used to generate the
        transform *and* the version of the generating software is required for
        non-volatile storage.

        """
        # Reset flags
        self.transforms_loaded_from_cache = None
        self.transforms_computed = False

        # TODO: store nominal transforms to the transforms cache as well, but
        # derive the hash value the same way as it is done for transforms,
        # to avoid needing to apply no systematics to the nominal transforms
        # to get the (identical) transforms?
        # Problem: assumes the nominal transform is the same as the transforms
        # that will result, which *might* not be true (though it seems it will
        # usually be so)

        # Compute nominal transforms; if feature is not used, this doesn't
        # actually do much of anything. To do more than this, override the
        # `_compute_nominal_transforms` method.
        _, nominal_transforms_hash = self.get_nominal_transforms(
            nominal_transforms_hash=nominal_transforms_hash
        )

        # Generate hash from param values
        if transforms_hash is None:
            transforms_hash = self._derive_transforms_hash(
                nominal_transforms_hash=nominal_transforms_hash
            )
        logging.trace("transforms_hash: %s" % str(transforms_hash))

        # Load and return existing transforms if in the cache
        if (
            self.transforms_cache is not None
            and transforms_hash in self.transforms_cache
            and self.debug_mode is None
        ):
            self.transforms_loaded_from_cache = "memory"
            logging.trace("loading transforms from cache.")
            transforms = self.transforms_cache[transforms_hash]

        # Otherwise: compute transforms, set hash, and store to cache
        else:
            self.transforms_computed = True
            logging.trace("computing transforms.")
            transforms = self._compute_transforms()
            transforms.hash = transforms_hash
            if self.transforms_cache is not None:
                self.transforms_cache[transforms_hash] = transforms

        self.check_transforms(transforms)
        self.transforms = transforms
        return transforms

    @profile
    def get_nominal_outputs(self, nominal_outputs_hash):
        """Load a cached output from the nominal outputs memory cache
        (which is backed by a disk cache, if one is specified) if the nominal
        outout is in the cache, or else recompute it and store to the
        cache(s).

        This method calls the `_compute_nominal_outputs` method, which by
        default does nothing.

        However, if you want to use the nominal outputs feature, override
        the `_compute_nominal_outputs` method and fill in the logic there.

        Deciding whether to invoke the `_compute_nominal_outputs` method or
        to load the nominal outputs from cache is done here, so you needn't
        think about any of this within the `_compute_nominal_outputs`
        method.

        Returns
        -------
        nominal_outputs, hash

        """
        if nominal_outputs_hash is None:
            nominal_outputs_hash = self._derive_nominal_outputs_hash()

        if (
            self.nominal_outputs_hash is None
            or self.nominal_outputs_hash != nominal_outputs_hash
        ):
            self._compute_nominal_outputs()
            self.nominal_outputs_hash = nominal_outputs_hash

    # for PI compatibility
    def run(self, inputs=None):
        return self.get_outputs(inputs=inputs)

    @profile
    def get_outputs(self, inputs=None):
        """Top-level function for computing outputs. Use this method to get
        outputs if you live outside this stage/service.

        Caching is handled here, so if the output hash returned by
        `_derive_outputs_hash` is in `outputs_cache`, it is simply returned.
        Otherwise, the `_compute_outputs` private method is invoked to do the
        actual work of computing outputs.

        Parameters
        ----------
        inputs : None or Mapping
            Any inputs to be transformed, plus any sideband objects that are to
            be passed on (untransformed) to subsequent stages.

        See also
        --------
        Overloadable methods called directly from this:
            _derive_outputs_hash
            _compute_outputs

        """
        # Reset flags
        self.outputs_loaded_from_cache = None
        self.outputs_computed = False

        # TODO: store nominal outputs to the outputs cache as well, but
        # derive the hash value the same way as it is done for outputs,
        # to avoid needing to apply no systematics to the nominal outputs
        # to get the (identical) outputs?
        # Problem: assumes the nominal transform is the same as the outputs
        # that will result, which *might* not be true (though it seems it will
        # usually be so)

        # Keep inputs for internal use and for inspection later
        self.inputs = inputs

        outputs_hash, transforms_hash, nominal_transforms_hash = (
            self._derive_outputs_hash()
        )

        # Compute nominal outputs; if feature is not used, this doesn't
        # actually do much of anything. To do more than this, override the
        # `_compute_nominal_outputs` method.
        self.get_nominal_outputs(nominal_outputs_hash=nominal_transforms_hash)

        logging.trace("outputs_hash: %s" % outputs_hash)

        if (
            self.outputs_cache is not None
            and outputs_hash is not None
            and outputs_hash in self.outputs_cache
            and self.debug_mode is None
        ):
            self.outputs_loaded_from_cache = "memory"
            logging.trace("Loading outputs from cache.")
            outputs = self.outputs_cache[outputs_hash]
        else:
            logging.trace("Need to compute outputs...")

            if self.use_transforms:
                self.get_transforms(
                    transforms_hash=transforms_hash,
                    nominal_transforms_hash=nominal_transforms_hash,
                )

            logging.trace("... now computing outputs.")
            outputs = self._compute_outputs(inputs=self.inputs)
            self.check_outputs(outputs)

            if isinstance(outputs, (Map, MapSet)):
                outputs = outputs.rebin(self.output_binning)

            outputs.hash = outputs_hash
            self.outputs_computed = True

            # Store output to cache
            if self.outputs_cache is not None and outputs_hash is not None:
                self.outputs_cache[outputs_hash] = outputs

        # Keep outputs for inspection later
        self.outputs = outputs

        # Attach sideband objects (i.e., inputs not specified in
        # `self.input_names`) to the "augmented" output object
        if self.inputs is None:
            names_in_inputs = set()
        else:
            names_in_inputs = set(self.inputs.names)
        unused_input_names = names_in_inputs.difference(self.input_names)

        if len(unused_input_names) == 0:
            return outputs

        # TODO: update logic for Data object, generic sideband objects
        # Create a new output container different from `outputs` but copying
        # the contents, for purposes of attaching the sideband objects found.
        if isinstance(outputs, MapSet):
            augmented_outputs = MapSet(outputs)
            for name in unused_input_names:
                augmented_outputs.append(inputs[name])

            return augmented_outputs
        else:
            raise TypeError(
                "Outputs are %s, but must currently be a MapSet in"
                " the case that the input includes sideband"
                " objects." % type(outputs)
            )

    def check_transforms(self, transforms):
        """Check that transforms' inputs and outputs match those specified
        for this service.

        Parameters
        ----------
        transforms

        Raises
        ------
        ValueError if transforms' inputs/outputs don't match stage spec

        """
        assert set(transforms.input_names) == set(self.input_names), (
            "Transforms' inputs: "
            + str(transforms.input_names)
            + "\nStage inputs: "
            + str(self.input_names)
        )

        assert set(transforms.output_names) == set(self.output_names), (
            "Transforms' outputs: "
            + str(transforms.output_names)
            + "\nStage outputs: "
            + str(self.output_names)
        )

    def check_outputs(self, outputs):
        """Check that the output names are those expected"""
        if set(outputs.names) != set(self.output_names):
            raise ValueError(
                "'{}' : Outputs found do not match expected outputs for this stage:\n"
                "  Outputs found: {}\n"
                "  Expected stage outputs: {}".format(
                    self.stage_name, outputs.names, self.output_names
                )
            )

    def load_events(self, events):
        """Load events from path given by `events`. Stored as `self.events`.

        Parameters
        ----------
        events : string or Events object
            If string, load events from that location. If Events object,
            deepcopy to obtain `self.events`

        """
        if isinstance(events, Param):
            events = events.value
        elif isinstance(events, str):
            events = find_resource(events)
        this_hash = hash_obj(events, full_hash=self.full_hash)
        if self._events_hash is not None and this_hash == self._events_hash:
            return
        logging.debug("Extracting events from Events obj or file: %s", events)
        events_obj = Events(events)
        events_hash = this_hash

        self.events = events_obj
        self._events_hash = events_hash

    def cut_events(self, keep_criteria):
        """Apply a cut to `self.events`, keeping only events that pass
        `keep_criteria`.

        Parameters
        ----------
        keep_criteria : string
             See pisa.core.Events.applyCut for more info on specifying this.

        """
        if isinstance(keep_criteria, Param):
            keep_criteria = keep_criteria.value

        if keep_criteria is not None:
            events = self.events.applyCut(keep_criteria=keep_criteria)
            events_hash = hash_obj(events, full_hash=self.full_hash)

            self.events = events
            self._events_hash = events_hash

    def instantiate_disk_cache(self):
        """Instantiate a disk cache for use by the stage."""
        if isinstance(self.disk_cache, DiskCache):
            self.disk_cache_path = self.disk_cache.path
            return

        if self.disk_cache is False or self.disk_cache is None:
            self.disk_cache = None
            self.disk_cache_path = None
            return

        if isinstance(self.disk_cache, str):
            dirpath, filename = os.path.split(
                os.path.expandvars(os.path.expanduser(self.disk_cache))
            )
            if os.path.isabs(dirpath):
                self.disk_cache_path = os.path.join(dirpath, filename)
            else:
                self.disk_cache_path = os.path.join(CACHE_DIR, dirpath, filename)
        elif self.disk_cache is True:
            dirs = [CACHE_DIR, self.stage_name]
            dirpath = os.path.expandvars(os.path.expanduser(os.path.join(*dirs)))
            if self.service_name is not None and self.service_name != "":
                filename = self.service_name + ".sqlite"
            else:
                filename = "generic.sqlite"
            mkdir(dirpath, warn=False)
            self.disk_cache_path = os.path.join(dirpath, filename)
        else:
            raise ValueError("Don't know what to do with a %s." % type(self.disk_cache))

        self.disk_cache = DiskCache(self.disk_cache_path, max_depth=10, is_lru=False)

    def _derive_outputs_hash(self):
        """Derive a hash value that unique identifies the outputs that will be
        generated based upon the current state of the stage.

        This implementation hashes together:
        * Input and output binning objects' hash values (if either input or
          output binning is not None)
        * Current params' values hash
        * Hashes from any input objects with names in `self.input_names`

        If any of the above objects is specified but returns None for its hash
        value, the entire output hash is invalidated, and None is returned.

        """
        id_objects = []

        # If stage uses inputs, grab hash from the inputs container object
        if self.outputs_cache is not None and len(self.input_names) > 0:
            inhash = self.inputs.hash
            logging.trace("inputs.hash = %s" % inhash)
            id_objects.append(inhash)

        # If stage uses transforms, get hash from the transforms
        transforms_hash = None
        if self.use_transforms:
            transforms_hash, nominal_transforms_hash = self._derive_transforms_hash()
            id_objects.append(transforms_hash)
            logging.trace("derived transforms hash = %s" % id_objects[-1])

        # Otherwise, generate sub-hash on binning and param values here
        else:
            transforms_hash, nominal_transforms_hash = None, None

            if self.outputs_cache is not None:
                id_subobjects = []
                # Include all parameter values
                id_subobjects.append(self.params.values_hash)

                # Include additional attributes of this object
                for attr in sorted(self._attrs_to_hash):
                    val = getattr(self, attr)
                    if hasattr(val, "hash"):
                        attr_hash = val.hash
                    elif self.full_hash:
                        norm_val = normQuant(val)
                        attr_hash = hash_obj(norm_val, full_hash=self.full_hash)
                    else:
                        attr_hash = hash_obj(val, full_hash=self.full_hash)
                    id_subobjects.append(attr_hash)

                # Generate the "sub-hash"
                if any([(h is None) for h in id_subobjects]):
                    sub_hash = None
                else:
                    sub_hash = hash_obj(id_subobjects, full_hash=self.full_hash)
                id_objects.append(sub_hash)

        # If any hashes are missing (i.e, None), invalidate the entire hash
        if self.outputs_cache is None or any([(h is None) for h in id_objects]):
            outputs_hash = None
        else:
            outputs_hash = hash_obj(id_objects, full_hash=self.full_hash)

        return outputs_hash, transforms_hash, nominal_transforms_hash

    def _derive_transforms_hash(self, nominal_transforms_hash=None):
        """Compute a hash that uniquely identifies the transforms that will be
        produced from the current configuration. Note that this hash needs only
        to be valid for this run (i.e., it is a volatile hash).

        This implementation returns a hash from the current parameters' values.

        """
        id_objects = []
        h = self.params.values_hash
        logging.trace("self.params.values_hash = %s" % h)
        id_objects.append(h)

        # Grab any provided nominal transforms hash, or derive it again
        if nominal_transforms_hash is None:
            nominal_transforms_hash = self._derive_nominal_transforms_hash()
        # If a valid hash has been gotten, include it
        if nominal_transforms_hash is not None:
            id_objects.append(nominal_transforms_hash)

        for attr in sorted(self._attrs_to_hash):
            val = getattr(self, attr)
            if hasattr(val, "hash"):
                attr_hash = val.hash
            elif self.full_hash:
                norm_val = normQuant(val)
                attr_hash = hash_obj(norm_val, full_hash=self.full_hash)
            else:
                attr_hash = hash_obj(val, full_hash=self.full_hash)
            id_objects.append(attr_hash)

        # If any hashes are missing (i.e, None), invalidate the entire hash
        if any([(h is None) for h in id_objects]):
            transforms_hash = None
        else:
            transforms_hash = hash_obj(id_objects, full_hash=self.full_hash)

        return transforms_hash, nominal_transforms_hash

    def _derive_nominal_transforms_hash(self):
        """Derive a hash to uniquely identify the nominal transform. This
        should be unique across processes and invocations bacuase the nominal
        transforms can be non-volatile (cached to disk) and must still be
        valid given their hash value upon loading from disk in the future.

        This implementation uses the nominal parameter values' hash
        combined with the source code hash to generate the final nominal
        transforms hash.

        Notes
        -----
        The hashing scheme implemented here might be sufficiently unique for
        many cases, but override this method in services according to the
        following guidelines:

        * Stages that use a nominal transform should override this method if
          the hash is more accurately computed differently from here.

        * Stages that use transforms but do not use nominal transforms can
          override this method with a simpler version that simply returns None
          to save computation time (if this method is found to be a significant
          performance hit). (This method is called each time an output
          is computed if `self.use_transforms == True`.)

        * Stages that use no transforms (i.e., `self.use_transforms == False`)
          will not call any built-in methods related to transforms, so
          overriding this method is irrelevant to such stages.

        If this method *is* overridden (and not just to return None), since the
        nominal transform may be stored to a disk cache, make sure that
        `self.source_code_hash` is included in the objects used to compute the
        final hash value. Even if all parameters are the same, a nominal
        transform stored to disk is ***invalid if the source code changes***,
        and `_derive_nominal_transforms_hash` must reflect this.

        """
        id_objects = []
        id_objects.append(self.params.nominal_values_hash)
        for attr in sorted(self._attrs_to_hash):
            val = getattr(self, attr)
            if hasattr(val, "hash"):
                attr_hash = val.hash
            elif self.full_hash:
                norm_val = normQuant(val)
                attr_hash = hash_obj(norm_val, full_hash=self.full_hash)
            else:
                attr_hash = hash_obj(val, full_hash=self.full_hash)
            id_objects.append(attr_hash)
        id_objects.append(self.source_code_hash)

        # If any hashes are missing (i.e, None), invalidate the entire hash
        if any([(h is None) for h in id_objects]):
            nominal_transforms_hash = None
        else:
            nominal_transforms_hash = hash_obj(id_objects, full_hash=self.full_hash)
        return nominal_transforms_hash

    def _derive_nominal_outputs_hash(self):
        return self._derive_nominal_transforms_hash()

    def _compute_nominal_transforms(self):  # pylint: disable=no-self-use
        """Stages that start with a nominal transform and use systematic
        parameters to modify the nominal transform in order to obtain the final
        transforms should override this method for deriving the nominal
        transform."""
        return None

    def _compute_transforms(self):  # pylint: disable=no-self-use
        """Stages that apply transforms to inputs should override this method
        for deriving the transform. No-input stages should leave this as-is."""
        return TransformSet([])

    def _compute_nominal_outputs(self):  # pylint: disable=no-self-use
        return None

    @profile
    def _compute_outputs(self, inputs):
        """Override this method for no-input stages which do not use transforms.
        Input stages that compute a TransformSet needn't override this, as the
        work for computing outputs is done by the TransfromSet below."""
        return self.transforms.apply(inputs)

    def validate_binning(self):  # pylint: disable=no-self-use
        """Override this method to test if the input and output binning
        (e.g., dimensionality, domains, separately or in combination)
        conform to the transform applied by the stage."""
        return
