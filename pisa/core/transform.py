"""
Transform as base class for transformations, TransformSet for sets of Transform
objects, and BinnedTensorTransform as an implementation of Transform for
defining and applying linear transforms.
"""


from __future__ import absolute_import, division

from collections.abc import Sequence
from collections import OrderedDict
from copy import deepcopy
from functools import wraps
import importlib
import inspect
import os
import shutil
import sys
import tempfile

import numpy as np
from uncertainties import unumpy as unp

from pisa import ureg, HASH_SIGFIGS
from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet, rebin
from pisa.utils.comparisons import normQuant, recursiveEquality
from pisa.utils.hash import hash_obj
from pisa.utils import jsons
from pisa.utils.log import logging, set_verbosity


__all__ = ['TransformSet', 'Transform', 'BinnedTensorTransform']

__author__ = 'J.L. Lanfranchi, P. Eller'

__license__ = '''Copyright (c) 2014-2017, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


# TODO: Include option for propagating/not propagating errors, so that while
# e.g. a minimizer runs to match templates to "data," the overhead is not
# incurred. But this then requires -- if the user does want errors -- for a
# final iteration after a match has been found where all outputs are
# re-computed but with the propagate_errors option set to True. The output
# caches must all then "miss" so that actual output including error is
# computed.

# TODO: use a generic container, *not* a MapSet to store sets of maps for
# inputs and outputs (in fact I think there should be no MapSet object at all,
# so we can trnsparently handle e.g. events alongside maps where one is a
# sideband object for the other in a given stage, but which is which should be
# irrelevant).

## TODO: numba implementation of BinnedTensorTransform faster or not?
#import numba
#@numba.jit("float64[:,:] (float64[:,:], float64[:,:,:,:])",
#           nopython=True, nogil=True, cache=True)
#def map2d_kernel4d(x, xform):
#    I = x.shape[0]
#    J = x.shape[1]
#    out = np.empty((I,J))
#    for i1 in range(I):
#        for j1 in range(J):
#            out[i1,j1] = 0.0
#            for i0 in range(I):
#                for j0 in range(J):
#                    out[i1,j1] += x[i1,j1] * xform[i0,j0,i1,j1]
#    return out

# TODO: add tex attr to TransformSet

# TODO: Add Sequence capabilities to TransformSet (e.g. it'd be nice to have at
# least append, extend, ...)
TRANS_SET_SLOTS = ('name', 'hash', 'transforms', '__iter__',
                   'nonvolatile_hash', 'input_names', 'num_inputs',
                   'output_names', 'apply', '__getattribute__',
                   'to_json', 'from_json')
class TransformSet(object):
    """
    Set of Transform objects.

    Parameters
    ----------
    transforms
    name

    """
    def __init__(self, transforms, name=None, hash=None): # pylint: disable=redefined-builtin
        self._transforms = transforms
        self.name = name
        self.hash = hash

    @property
    def serializable_state(self):
        """OrderedDict : State of the object in a format that is serializable"""
        state = OrderedDict()
        state['transforms'] = [
            (t.__module__, t.__class__.__name__, t.serializable_state)
            for t in self
        ]
        state['name'] = self.name
        return state

    @property
    def hashable_state(self):
        """OrderedDict : State of the objec that can be used for hashing"""
        state = OrderedDict()
        state['transforms'] = [
            (t.__module__, t.__class__.__name__, t.hashable_state)
            for t in self
        ]
        state['name'] = self.name
        return state

    def __iter__(self):
        return iter(self._transforms)

    def __len__(self):
        return len(self._transforms)

    def __eq__(self, other):
        if not isinstance(other, TransformSet):
            return False
        return recursiveEquality(self.hashable_state, other.hashable_state)

    def to_json(self, filename, **kwargs):
        """Serialize the state to a JSON file that can be instantiated as a new
        object later.

        Parameters
        ----------
        filename : str
            Filename; must be either a relative or absolute path (*not
            interpreted as a PISA resource specification*)
        **kwargs
            Further keyword args are sent to `pisa.utils.jsons.to_json()`

        See Also
        --------
        from_json : Intantiate new object from the file written by this method
        pisa.utils.jsons.to_json

        """
        jsons.to_json(self.serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, resource):
        """Instantiate a new TransformSet object from a JSON file.

        Parameters
        ----------
        resource : str
            A PISA resource specification (see pisa.utils.resources)

        See Also
        --------
        to_json
        pisa.utils.jsons.to_json

        """
        state = jsons.from_json(resource)
        transforms = []
        for module, classname, transform_state in state['transforms']:
            clsmembers = inspect.getmembers(sys.modules[__name__],
                                            inspect.isclass)
            # First try to get a class within this module/namespace
            classes = [c[1] for c in clsmembers if c[0] == classname]
            if len(classes) > 0:
                class_ = classes[0]
            # Otherwise try to import the module recorded in the JSON file
            else:
                module = importlib.import_module(module)
                # and then get the class
                class_ = getattr(module, classname)
            transforms.append(class_(**transform_state))
        state['transforms'] = transforms
        # State is a dict, so instantiate with double-asterisk syntax
        return cls(**state)

    @property
    def hash(self):
        """int : Hash for entire set of transforms"""
        hashes = self.hashes
        if len(hashes) > 0:
            if all([(h is not None and h == hashes[0]) for h in hashes]):
                return hashes[0]
            if all([(h is not None) for h in hashes]):
                return hash_obj(hashes)
        return None

    @hash.setter
    def hash(self, val):
        if val is not None:
            for xform in self:
                setattr(xform, 'hash', val)

    @property
    def hashes(self):
        """list of int : List of hashes, one per transform in the set"""
        return [t.hash for t in self]

    # TODO: implement a non-volatile hash that includes source code hash in
    # addition to self.hash from the contained transforms
    #@property
    #def nonvolatile_hash(self):
    #    return hash_obj((self.source_code_hash,

    @property
    def input_names(self):
        """Inputs to the transform must include inputs with these names"""
        input_names = set()
        for x in self:
            input_names.update(x.input_names)
        return sorted(input_names)

    @property
    def num_inputs(self):
        """int : number of inputs"""
        return len(self.input_names)

    @property
    def num_outputs(self):
        """int : number of outputs"""
        return len(self)

    @property
    def output_names(self):
        """list of str : names of outputs"""
        output_names = []
        for x in self:
            output_names.append(x.output_name)
        return output_names

    def get(self, input_names, output_name):
        """Retrieve the transform that maps `input_names` to `output_name`.
        Note that `input_names` can be a single name.

        Parameters
        ----------
        input_names : string or sequence thereof
        output_name : string

        Returns
        -------
        transform : Transform object

        """
        if isinstance(input_names, str):
            input_names = [input_names]
        for transform in self:
            if set(input_names) == set(transform.input_names) \
               and output_name == transform.output_name:
                return transform

    def apply(self, inputs):
        """Apply each transform to `inputs`; return computed outputs.

        Parameters
        -----------
        inputs : sequence of objects

        Returns
        -------
        outputs : container with computed outputs (no sideband objects)

        """
        output_names = []
        outputs = []

        # If any outputs have the same name, add them together to form a single
        # output for that name
        for xform in self:
            output = xform.apply(inputs)
            name = output.name
            try:
                idx = output_names.index(name)
                outputs[idx] = outputs[idx] + output
                outputs[idx].name = name
            except ValueError:
                outputs.append(output)
                output_names.append(name)

        # Automatically attach a sensible hash (this may be overwritten, but
        # the below should be a reasonable hash in most cases)
        if inputs.hash is None or self.hash is None:
            hash_ = None
        else:
            hash_ = hash_obj((inputs.hash, self.hash))

        # TODO: what to set for map set's name, tex, etc. ?
        return MapSet(maps=outputs, hash=hash_)

    def __getattr__(self, attr):
        if attr in TRANS_SET_SLOTS:
            return super().__getattribute__(attr)
        # TODO: return maps based upon name?
        #if attr in
        return TransformSet([getattr(t, attr) for t in self], name=self.name)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self._transforms[idx]
        # TODO: search for transform by output name?
        #if isinstance(idx, str):
        raise IndexError('`idx` %s not found in map set.' %idx)


class Transform(object):
    """
    Base class for building a transform.

    """
    # TODO: get rid of the tex attribute, or add back in the name attribute?

    # Attributes that __setattr__ will allow setting
    _slots = ('_input_names', '_output_name', '_tex', '_hash', '_hash',
              'error_method')
    # Attributes that should be retrieved to fully describe state
    _state_attrs = ('input_names', 'output_name', 'tex', 'hash', 'error_method')

    def __init__(self, input_names, output_name, input_binning=None,
                 output_binning=None, tex=None, hash=None, error_method=None): # pylint: disable=redefined-builtin
        # Convert to sequence of single string if a single string was passed
        # for uniform interfacing
        if isinstance(input_names, str):
            input_names = [input_names]
        else:
            input_names = [name for name in input_names]
        self._input_names = input_names

        assert isinstance(output_name, str)
        self._output_name = output_name

        if input_binning is not None:
            if not isinstance(input_binning, MultiDimBinning):
                if isinstance(input_binning, Sequence):
                    input_binning = MultiDimBinning(input_binning)
                else:
                    input_binning = MultiDimBinning(**input_binning)
            self._input_binning = input_binning
        else:
            self._input_binning = None

        if output_binning is not None:
            if not isinstance(output_binning, MultiDimBinning):
                if isinstance(output_binning, Sequence):
                    output_binning = MultiDimBinning(output_binning)
                else:
                    output_binning = MultiDimBinning(**output_binning)
            self._output_binning = output_binning
        else:
            self._output_binning = None

        self._tex = tex
        self._hash = hash
        if bool(error_method):
            self._error_method = error_method
        else:
            self._error_method = None

    @property
    def serializable_state(self):
        """OrderedDict : State of the object in a format that is serializable"""
        state = OrderedDict()
        state['input_names'] = self.input_names
        state['output_name'] = self.output_name
        state['input_binning'] = self.input_binning.serializable_state
        state['output_binning'] = self.output_binning.serializable_state
        state['tex'] = self.tex
        state['error_method'] = self.error_method
        state['hash'] = self.hash
        return state

    @property
    def hashable_state(self):
        """OrderedDict : State of the object that can be used for hashing"""
        state = OrderedDict()
        state['input_names'] = self.input_names
        state['output_name'] = self.output_name
        state['input_binning'] = self.input_binning.hashable_state
        state['output_binning'] = self.output_binning.hashable_state
        state['tex'] = self.tex
        state['error_method'] = self.error_method
        return state

    def to_json(self, filename, **kwargs):
        """Serialize the state to a JSON file that can be instantiated as a new
        object later.

        Parameters
        ----------
        filename : str
            Filename; must be either a relative or absolute path (*not
            interpreted as a PISA resource specification*)
        **kwargs
            Further keyword args are sent to `pisa.utils.jsons.to_json()`

        See Also
        --------
        from_json : Intantiate new object from the file written by this method
        pisa.utils.jsons.to_json

        """
        jsons.to_json(self.serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, resource):
        """Instantiate a new Map object from a JSON file.

        The format of the JSON is generated by the `Map.to_json` method, which
        converts a Map object to basic types and then numpy arrays are
        converted in a call to `pisa.utils.jsons.to_json`.

        Parameters
        ----------
        resource : str
            A PISA resource specification (see pisa.utils.resources)

        See Also
        --------
        to_json
        pisa.utils.jsons.to_json

        """
        state = jsons.from_json(resource)
        # State is a dict for Map, so instantiate with double-asterisk syntax
        return cls(**state)

    @property
    def hash(self):
        """int : hash of obejct's state"""
        return self._hash

    @hash.setter
    def hash(self, val):
        self._hash = val

    @property
    def input_names(self):
        """list of str : input names"""
        return self._input_names

    @property
    def num_inputs(self):
        """int : number of inputs"""
        return len(self.input_names)

    @property
    def output_name(self):
        """str : name of output"""
        return self._output_name

    @property
    def input_binning(self):
        """MultiDimBinning : binning specification for inputs"""
        return self._input_binning

    @property
    def output_binning(self):
        """MultiDimBinning : binning specification for output"""
        return self._output_binning

    @property
    def tex(self):
        """str : TeX label"""
        return self._tex

    @tex.setter
    def tex(self, val):
        assert val is None or isinstance(val, str)
        self._tex = val

    @property
    def error_method(self):
        """str or None : method specified for computing errors"""
        return self._error_method

    def apply(self, inputs):
        """Apply the transform to inputs to obtain the output.

        Parameters
        ----------
        inputs

        Returns
        -------
        output

        """
        output = self._apply(inputs)
        # TODO: tex, etc.?
        output.name = self.output_name
        return output

    def _apply(self, inputs):
        """Override this method in subclasses"""
        raise NotImplementedError('Override this method in subclasses')

    def validate_transform(self, xform):
        """Override this method in subclasses"""
        raise NotImplementedError('Override this method in subclasses')

    def validate_input(self, inputs):
        """Override this method in subclasses"""
        raise NotImplementedError('Override this method in subclasses')


def _new_obj(original_function):
    """Decorator to deepcopy unaltered states into new object

    Parameters
    ----------
    original_function : callable
        Callable must return None or a Mapping with some or all of
        _state_attrs defined.

    Returns
    -------
    new_function : callable
        New function that wraps `original_function`. `new_function` returns
        an instantiated object with _state_attrs from the current object
        but updated with any _state_attrs defined in the output from the
        call to `original_function`.

    """
    @wraps(original_function)
    def new_function(self, *args, **kwargs):
        """Wrapper function for returning an instantiated object based on
        current state of an object but with updates to those states based
        on the output from calling the original function.

        (Note that this docstring is here for reference within the
        sourcecode, but is overwritten at run-time by the @wraps
        decorator.)

        """
        new_state = OrderedDict()
        state_updates = original_function(self, *args, **kwargs)
        for slot in self._state_attrs: # pylint: disable=protected-access
            if slot in state_updates:
                new_state[slot] = state_updates[slot]
            else:
                new_state[slot] = deepcopy(getattr(self, slot))
        return BinnedTensorTransform(**new_state)
    return new_function


# TODO: integrate uncertainties module in with this so that a transform can
#       introduce (augment) error of an input Map for producing a more accurate
#       estimate of the error in the output map.
class BinnedTensorTransform(Transform):
    """
    BinnedTensorTransform implementing common transforms used in PISA:
        1) Element-by-element multiplicaton. E.g., to transform an N x M map,
           the transform array is also M x N.

        2) Smearing kernels, e.g. to characterize resolutions. For a transform
           indended to be applied to a 2D map of dimensions (M x N), for each
           of the M x N bins of the input map, there is one M x N kernel that
           represents how that bin gets "smeared out" to the entire output map.
           The effect of each input bin is added up to yield in the end a single
           M x N output map.


    Parameters
    ----------
    input_names : string or sequence thereof
        Names of maps expected in the input MapSet. See Notes for how multiple
        inputs are ito be indexed in the `xform_array`.

    output_name : string
        Name of Map that will be generated.

    input_binning : MultiDimBinning
        Binning required for inputs maps.

    output_binning : MultiDimBinning
        Binning used for generated output maps.

    xform_array : numpy ndarray
        The actual transform's numerical values. Shape must be in accordance
        with `input_binning` and `output_binning` to accommodate the type
        of transform being implemented. See Notes for more detail on allowed
        shapes.

    sum_inputs : bool
        If true, add inputs together if multiple are specified. Otherwise,
        stack input maps along the first dimension (increases dimensionality of
        the input array by one).

    tex : string
        TeX label for e.g. automatic plot labelling.

    hash : immutable object (usually integer)
        A hash value the user can attach

    error_method : None, bool, or string
        Define the method for error propaation on unumpy arrays

    output_name : string

    input_binning : MultiDimBinning

    output_binning : MultiDimBinning

    xform_array : numpy ndarray

    error_array : None or numpy ndarray

    params_hash : immutable object (usually integer)


    Notes
    -----
    For an input map that is M_ebins x N_czbins, the transform must either be
    2-dimensional of shape (M x N) or 4-dimensional of shape (M x N x M x N).
    The latter case can be thought of as a 2-dimensional (M x N) array, each
    element of which is a 2-dimensional (M x N) array, and is currently used
    for the reconstruction stage's convolution kernels where there is one
    (M_ebins x N_czbins)-size kernel for each (energy, coszen) bin.

    There can be extra objects in `inputs` that are not used by this transform
    ("sideband" objects, which are simply ignored here). If multiple input maps
    are used by the transform, if `sum_inputs` is Ture, they are summed;
    otherwise, they are combined via numpy.stack((map0, map1, ... ), axis=0)
    I.e., the first dimension of the input sent to the transform has a length
    the same number of input maps requested by the transform.

    """
    _slots = tuple(list(Transform._slots) +
                   ['_input_binning', '_output_binning', '_xform_array'])

    _state_attrs = tuple(list(Transform._state_attrs) +
                         ['input_binning', 'output_binning', 'xform_array'])

    def __init__(self, input_names, output_name, input_binning, output_binning,
                 xform_array, sum_inputs=False, error_array=None, tex=None,
                 error_method=None, hash=None): # pylint: disable=redefined-builtin
        super().__init__(
            input_names=input_names, output_name=output_name,
            input_binning=input_binning, output_binning=output_binning,
            tex=tex, hash=hash, error_method=error_method
        )
        self._xform_array = None
        self.xform_array = xform_array
        self.sum_inputs = sum_inputs
        if error_array is not None:
            self.set_errors(error_array)

    @property
    def serializable_state(self):
        """OrderedDict : State of the object in a format that is serializable"""
        state = super().serializable_state
        state['xform_array'] = self.nominal_values
        state['error_array'] = self.std_devs
        return state

    @property
    def hashable_state(self):
        """OrderedDict : State of the objec that can be used for hashing"""
        state = super().hashable_state
        state['xform_array'] = normQuant(self.nominal_values,
                                         sigfigs=HASH_SIGFIGS)
        state['error_array'] = normQuant(self.std_devs, sigfigs=HASH_SIGFIGS)
        return state

    def set_errors(self, error_array):
        """Manually define the error with an array the same shape as the
        contained histogram. Can also remove errors by passing None.

        Parameters
        ----------
        error_array : None or ndarray
            Standard deviations to apply to `self.xform_array`; shapes must
            match. If None is passed, any errors present are removed, making
            `self.xform_array` a bare numpy array.

        """
        if error_array is None:
            super().__setattr__(
                '_xform_array', self.nominal_values
            )
            return
        assert error_array.shape == self.xform_array.shape
        super().__setattr__(
            '_xform_array',
            unp.uarray(self.xform_array, np.ascontiguousarray(error_array))
        )

    @property
    def xform_array(self):
        """Numpy ndarray containing raw transform"""
        return self._xform_array

    @xform_array.setter
    def xform_array(self, x):
        self.validate_transform(self.input_binning, self.output_binning, x)
        self._xform_array = np.ascontiguousarray(x)

    @property
    def nominal_values(self):
        """numpy.ndarray, same shape as `xform_array` : Nominal transform
        values (i.e. with uncertainties stripped off)"""
        return unp.nominal_values(self.xform_array)

    @property
    def std_devs(self):
        """numpy.ndarray, same shape as `xform_array` : Stndard deviation of
        each transform values value (i.e. uncertainties)"""
        return unp.std_devs(self.xform_array)

    @_new_obj
    def __abs__(self):
        return dict(xform_array=np.abs(self.xform_array))

    @_new_obj
    def __add__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=self.xform_array + other.xform_array)
        return dict(xform_array=self.xform_array + other)

    @_new_obj
    def __div__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=self.xform_array / other.xform_array)
        return dict(xform_array=self.xform_array / other)

    def __eq__(self, other):
        if not isinstance(other, BinnedTensorTransform):
            return False
        return recursiveEquality(self.hashable_state, other.hashable_state)

    @_new_obj
    def __mul__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=self.xform_array * other.xform_array)
        return dict(xform_array=self.xform_array * other)

    @_new_obj
    def __ne__(self, other):
        return not self.__eq__(other)

    @_new_obj
    def __neg__(self):
        return dict(xform_array=-self.xform_array)

    @_new_obj
    def __pow__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=self.xform_array ** other.xform_array)
        return dict(xform_array=self.xform_array ** other)

    @_new_obj
    def __radd__(self, other):
        return self + other

    @_new_obj
    def __rdiv__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=other.xform_array / self.xform_array)
        return dict(xform_array=other / self.xform_array)

    @_new_obj
    def __rmul__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=other.xform_array * self.xform_array)
        return dict(xform_array=other * self.xform_array)

    @_new_obj
    def __rsub__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=other.xform_array - self.xform_array)
        return dict(xform_array=other - self.xform_array)

    @_new_obj
    def sqrt(self):
        """Square root of the transform"""
        return dict(xform_array=np.sqrt(self.xform_array))

    @_new_obj
    def __sub__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=self.xform_array - other.xform_array)
        return dict(xform_array=self.xform_array - other)

    # TODO: validate transform...; also, this def changes number of arguments,
    # which should be avoided if possible
    def validate_transform(self, input_binning, output_binning, xform_array): # pylint: disable=arguments-differ
        """Superficial validation that the transform being set is reasonable.

        As of now, only checks shape.

        Expected transform shape is:
            (
             <input binning n_ebins>,
             <input binning n_czbins>,
             {if num_inputs > 1: <num_inputs>,}
             <output binning n_ebins>,
             <output binning n_czbins>,
             {if num_outputs > 1: <num_outputs>}
            )

        """
        #in_dim = [] if self.num_inputs == 1 else [self.num_inputs]
        #out_dim = [] if self.num_outputs == 1 else [self.num_outputs]
        #assert xform_array.shape == tuple(list(input_binning.shape) + in_dim +
        #                                  list(output_binning.shape) + out_dim)
        pass

    def validate_input(self, inputs):
        if inputs is None:
            raise ValueError('No inputs provided.')

        for input_name in self.input_names:
            assert input_name in inputs, \
                    'Input "%s" expected; got: %s.' \
                    % (input_name, inputs.names)

    # TODO: make _apply work with multiple inputs (i.e., concatenate
    # these into a higher-dimensional array) and make logic for applying
    # element-by-element multiply and tensordot generalize to any dimension
    # given the (concatenated) input dimension and the dimension of the
    # transform kernel

    def _apply(self, inputs):
        """Apply transforms to input maps to compute output maps.

        Parameters
        ----------
        inputs : MapSet
            Container class that must contain (at least) the maps to be
            transformed.

        Returns
        -------
        output : Map
            Result of applying the transform to the input map(s).

        """
        self.validate_input(inputs)

        # TODO: In the multiple inputs / single output case and depending upon
        # the dimensions of the transform, for efficiency purposes we should
        # make sure that an operation is not carried out like
        #
        #   (input0 [*] transform) + (input1 [*] transform) = output
        #
        # but instead is performed more efficiently as
        #
        #   (input0 + input1) [*] transform = output
        #
        # where [*] is some linear operation, like element-by-element
        # multiplication, a dot product, etc.
        #
        # E.g., for a 1D dot product (dimensionality-reducing linear operation)
        # with M_in inputs and N_el elements in each
        # vector, for the first (less efficient) formulation, there are
        #
        #   (N_el-1)*M_in + (M_in-1) = (M_in*N_el - 1) adds
        #
        # and
        #
        #   (N_el*M_in) multiplies
        #
        # while for the second (more efficient) formulation, there are
        #
        #   (M_in-1)*N_el + (N_el-1) = (M_in*N_el - 1) adds
        #
        # and
        #
        #   (N_el) multiplies
        #
        # so the benefit is a reduction of the number of multiplies necessary
        # by a factor of the number of inputs being "combined" (the number of
        # adds stays the same).

        # TODO: make sure all of these operations are compatible with
        # uncertainties module!

        names = self.input_names
        in0 = inputs[names[0]]

        if self.num_inputs == 1:
            input_array = (in0.rebin(self.input_binning)).hist

        # Stack inputs, sum inputs, *then* rebin (if necessary)
        elif self.sum_inputs:
            input_array = np.sum([inputs[n].hist for n in names], axis=0)
            input_array = rebin(input_array, orig_binning=in0.binning,
                                new_binning=self.input_binning)

        # Rebin (if necessary) then stack
        else:
            input_array = [(inputs[n].rebin(self.input_binning)).hist
                           for n in names]
            input_array = np.stack(input_array, axis=0)

        # TODO: is logic kosher here?

        # Transform same shape: element-by-element multiplication
        if self.xform_array.shape == input_array.shape:
            if (isinstance(self.error_method, str) and
                    self.error_method.strip().lower() == 'fixed'):
                # don't scale errors here
                output = unp.uarray(
                    unp.nominal_values(input_array) * self.xform_array,
                    unp.std_devs(input_array)
                )
            else:
                output = input_array * self.xform_array

            # If multiple inputs were concatenated together, and we did not sum
            # these inputs together, we need to sum the results together now.

            # TODO: generalize this and the above operation (and possibly speed
            # this up) by formulating a simple inner product above and avoiding
            # an explicit sum here?
            if self.num_inputs > 1 and not self.sum_inputs:
                output = np.sum(output, axis=0)

        # TODO: Check that
        #   len(xform.shape) == 2*len(input_array.shape)
        # and then check that
        #   xform.shape == (input_array.shape, input_array.shape) (roughly)
        # and then apply tensordot appropriately for this generic case...

        # TODO: why does this fail for different input/output binning, but
        # below tensordot works?
        #elif len(self.xform_array.shape) == 4 and len(input_array.shape) == 2:
        #    output = map2d_kernel4d(input_array, self.xform_array)

        elif len(self.xform_array.shape) == 2*len(input_array.shape):
            output = np.tensordot(input_array, self.xform_array,
                                  axes=([0, 1], [0, 1]))

        elif (input_array.shape ==
              self.xform_array.shape[0:len(input_array.shape)]):
            axes = np.arange(len(input_array.shape))
            output = np.tensordot(input_array, self.xform_array,
                                  axes=(axes, axes))

        elif (input_array.shape ==
              self.xform_array.shape[-len(input_array.shape):]):
            output = input_array * self.xform_array

        else:
            raise ValueError(
                'Unhandled shapes for input(s) "%s": %s and'
                ' transform: %s.'
                %(', '.join(self.input_names), input_array.shape,
                  self.xform_array.shape)
            )

        output = Map(name=self.output_name,
                     hist=output,
                     binning=self.output_binning)

        return output


def test_BinnedTensorTransform():
    """Unit tests for BinnedTensorTransform class"""
    binning = MultiDimBinning([
        dict(name='energy', is_log=True, domain=(1, 80)*ureg.GeV, num_bins=10),
        dict(name='coszen', is_lin=True, domain=(-1, 0), num_bins=5)
    ])

    nue_map = Map(
        name='nue',
        binning=binning,
        hist=np.random.random(binning.shape)
    )
    nue_map.set_poisson_errors()
    numu_map = Map(
        name='numu',
        binning=binning,
        hist=np.random.random(binning.shape)
    )
    numu_map.set_poisson_errors()
    inputs = MapSet(
        name='inputs',
        maps=[nue_map, numu_map],
    )

    xform0 = BinnedTensorTransform(
        input_names='nue',
        output_name='nue',
        input_binning=binning,
        output_binning=binning,
        xform_array=2*np.ones(binning.shape)
    )

    xform1 = BinnedTensorTransform(
        input_names=['numu'],
        output_name='numu',
        input_binning=binning,
        output_binning=binning,
        xform_array=3*np.ones(binning.shape)
    )

    xform2 = BinnedTensorTransform(
        input_names=['nue', 'numu'],
        output_name='nue_numu',
        input_binning=binning,
        output_binning=binning,
        xform_array=np.stack([2*np.ones(binning.shape),
                              3*np.ones(binning.shape)], axis=0)
    )
    assert np.all((xform2 + 2).xform_array - xform2.xform_array == 2)

    testdir = tempfile.mkdtemp()
    try:
        for i, t in enumerate([xform0, xform1, xform2]):
            t_file = os.path.join(testdir, str(i) + '.json')
            t.to_json(t_file)
            t_ = BinnedTensorTransform.from_json(t_file)
            assert t_ == t, 't=\n%s\nt_=\n%s' %(t, t_)
    finally:
        shutil.rmtree(testdir, ignore_errors=True)

    logging.info('<< PASS : test_BinnedTensorTransform >>')

    xforms = TransformSet(
        name='scaling',
        transforms=[xform0, xform1, xform2],
        hash=9
    )

    assert xforms.hash == 9
    xforms.hash = -20
    assert xforms.hash == -20

    _ = xforms.apply(inputs)

    # TODO: get this working above, then test here!
    #xforms2 = xforms * 2

    testdir = tempfile.mkdtemp()
    try:
        for i, t in enumerate([xforms]):
            t_filename = os.path.join(testdir, str(i) + '.json')
            t.to_json(t_filename)
            t_ = TransformSet.from_json(t_filename)
            assert t_ == t, 't=\n%s\nt_=\n%s' %(t.transforms, t_.transforms)
    finally:
        shutil.rmtree(testdir, ignore_errors=True)

    logging.info('<< PASS : test_TransformSet >>')


if __name__ == "__main__":
    set_verbosity(1)
    test_BinnedTensorTransform()
