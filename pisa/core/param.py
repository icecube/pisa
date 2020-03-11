"""
Define Param, ParamSet, and ParamSelector classes for handling parameters, sets
of parameters, and being able to discretely switch between sets of parameter
values.
"""


from __future__ import absolute_import, division

from collections.abc import Iterable, Mapping, Sequence
from collections import OrderedDict
from copy import deepcopy
from functools import total_ordering
from operator import setitem
from os.path import join
from shutil import rmtree
import sys
import tempfile

import numpy as np
import pint
from six import string_types

from pisa import ureg
from pisa.core.prior import Prior
from pisa.utils import jsons
from pisa.utils.comparisons import (
    interpret_quantity, isbarenumeric, normQuant, recursiveEquality
)
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.random_numbers import get_random_state
from pisa.utils.stats import ALL_METRICS, CHI2_METRICS, LLH_METRICS


__all__ = [
    'Param',
    'ParamSet',
    'ParamSelector',
    'test_Param',
    'test_ParamSet',
    'test_ParamSelector',
]

__author__ = 'J.L. Lanfranchi'

__license__ = '''Copyright (c) 2014-2019, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


# TODO: Make property "frozen" or "read_only" so params in param set e.g.
# returned by a template maker -- which updating the values of will NOT have
# the effect the user might expect -- will be explicitly forbidden?

# TODO: units: pass in attached to values, or as separate kwarg? If e.g.
# value hasn't been set yet, then there's no implicit units to reference
# when setting the prior, range, and possibly other things (which all need to
# at least have compatible units)
@total_ordering
class Param:
    """Parameter class to store any kind of parameters

    Parameters
    ----------
    name : string

    value : scalar, bool, pint Quantity (value with units), string, or None

    prior : pisa.prior.Prior or instantiable thereto

    range : sequence of two scalars or Pint quantities, or None

    is_fixed : bool

    unique_id : string, optional
        If None is provided (default), `unique_id` is set to `name`

    is_discrete : bool, optional
        Default is False

    nominal_value : same type as `value`, optional
        If None (default), set to same as `value`

    tex : None or string

    help : string

    Notes
    -----
    In the case of a free (`is_fixed`=False) parameter, a valid range for the
    parameter should be spicfied and a prior must be assigned to compute llh
    and chi2 values.

    Examples
    --------
    >>> from pisa import ureg
    >>> from pisa.core.prior import Prior
    >>> gaussian = Prior(kind='gaussian', mean=10*ureg.meter,
    ...                  stddev=1*ureg.meter)
    >>> x = Param(name='x', value=1.5*ureg.foot, prior=gaussian,
    ...           range=[-10, 60]*ureg.foot, is_fixed=False, is_discrete=False)
    >>> x.value
    <Quantity(1.5, 'foot')>
    >>> print(x.prior_llh)
    -45.532515919999994
    >>> print(x.to('m'))

    >>> x.value = 10*ureg.m
    >>> print(x.value)
    <Quantity(32.8083989501, 'foot')>
    >>> x.ito('m')
    >>> print(x.value)

    >>> x.prior_llh
    -1.5777218104420236e-30
    >>> p.nominal_value

    >>> x.reset()
    >>> print(x.value)


    """
    # pylint: disable=protected-access
    _slots = (
        'name',
        'unique_id',
        'value',
        'prior',
        '_prior',
        'range',
        'is_fixed',
        'is_discrete',
        'nominal_value',
        'tex',
        '_rescaled_value',
        '_nominal_value',
        '_tex',
        'help',
        '_value',
        '_range',
        '_units',
        'normalize_values',
    )
    _state_attrs = (
        'name',
        'unique_id',
        'value',
        'prior',
        'range',
        'is_fixed',
        'is_discrete',
        'nominal_value',
        'tex',
        'help',
    )

    def __init__(self, name, value, prior, range, is_fixed, unique_id=None,
                 is_discrete=False, nominal_value=None, tex=None, help=''):
        self._range = None
        self._tex = None
        self._value = None
        self._units = None
        self._nominal_value = None
        self._prior = None

        self.value = value
        self.name = name
        self.unique_id = unique_id if unique_id is not None else name
        self.tex = tex
        self.help = help
        self.range = range
        self.prior = prior
        self.is_fixed = is_fixed
        self.is_discrete = is_discrete
        self.nominal_value = value if nominal_value is None else nominal_value
        self.normalize_values = False

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return recursiveEquality(self.state, other.state)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.name < other.name

    def __setattr__(self, attr, val):
        if attr not in self._slots:
            raise AttributeError('Invalid attribute: %s' % (attr,))
        object.__setattr__(self, attr, val)

    def __str__(self):
        return '%s=%s; prior=%s, range=%s, is_fixed=%s,' \
                ' is_discrete=%s; help="%s"' \
                % (self.name, self.value, self.prior, self.range,
                   self.is_fixed, self.is_discrete, self.help)

    def validate_value(self, value):
        if self.range is not None:
            if self.is_discrete:
                if value not in self.range:
                    raise ValueError(
                        'Param %s has a value %s which is not in the range of '
                        '%s'%(self.name, value, self.range)
                    )
            else:
                value_not_big_enough = value < min(self.range)
                value_not_small_enough = value > max(self.range)
                if value_not_big_enough or value_not_small_enough:
                    raise ValueError(
                        'Param %s has a value %s which is not in the range of '
                        '%s'%(self.name, value, self.range)
                    )

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        # Strings, bools, and `None` are simply used as-is; otherwise, enforce
        # input have units (or default to units of `dimensionless`)
        if not (val is None or isinstance(val, string_types) or isinstance(val, bool)):
            # A number with no units actually has units of "dimensionless"
            val = interpret_quantity(val, expect_sequence=False)

        if self._value is not None:
            if hasattr(self._value, 'units'):
                assert hasattr(val, 'units'), \
                        'Passed values must have units if the param has units'
                val = val.to(self._value.units)
            self.validate_value(val)
        self._value = val
        if hasattr(self._value, 'units'):
            self._units = self._value.units
        else:
            self._units = ureg.Unit('dimensionless')

    @property
    def magnitude(self):
        return self._value.magnitude

    @property
    def m(self):  # pylint: disable=invalid-name
        return self._value.magnitude

    def m_as(self, u):
        return self._value.m_as(u)

    @property
    def dimensionality(self):
        return self._value.dimensionality

    @property
    def units(self):
        return self._units

    @property
    def u(self):  # pylint: disable=invalid-name
        return self._units

    @property
    def range(self):
        if self._range is None:
            return None
        return tuple(self._range)

    @range.setter
    def range(self, values):
        if values is None:
            self._range = None
            return

        if len(values) != 2:
            raise ValueError(
                "Must provide a lower and upper bound; got {} value(s)".format(
                    len(values)
                )
            )

        new_vals = []
        for val in values:
            val = interpret_quantity(val, expect_sequence=False)

            # NOTE: intentionally using type() instead of isinstance() here.
            # Not sure if this could be converted to isinstance(), though.
            if not type(val) == type(self.value): # pylint: disable=unidiomatic-typecheck
                raise TypeError('Value "%s" has type %s but must be of type'
                                ' %s.' % (val, type(val), type(self.value)))

            if isinstance(self.value, ureg.Quantity):
                if self.dimensionality != val.dimensionality:
                    raise ValueError('Value "%s" units "%s" incompatible with'
                                     ' units "%s".'
                                     % (val, val.units, self.units))

            new_vals.append(val)

        self._range = new_vals

    # TODO: make discrete values rescale to integers 0, 1, ...
    @property
    def _rescaled_value(self):
        if self.is_discrete:
            return self.value
        srange = self.range
        if srange is None:
            raise ValueError('Cannot rescale without a range specified'
                             ' for parameter %s' % self)
        srange = self.range
        srange0 = srange[0].m
        srange1 = srange[1].m
        return (self._value.m - srange0) / (srange1 - srange0)

    @_rescaled_value.setter
    def _rescaled_value(self, rval):
        srange = self.range
        if srange is None:
            raise ValueError('Cannot rescale without a range specified'
                             ' for parameter %s' % self)
        if rval < 0 or rval > 1:
            raise ValueError(
                '%s: `rval`=%.15e, but cannot be outside [0, 1]'
                % (self.name, rval)
            )
        srange0 = srange[0].m
        srange1 = srange[1].m
        self._value = (srange0 + (srange1 - srange0)*rval) * self._units

    @property
    def tex(self):
        return self._tex

    @tex.setter
    def tex(self, t):
        self._tex = t if t is not None else r'{\rm %s}' % self.name

    @property
    def nominal_value(self):
        return self._nominal_value

    @nominal_value.setter
    def nominal_value(self, value):
        if not (value is None or isinstance(value, bool) or isinstance(value, string_types)):
            value = interpret_quantity(value, expect_sequence=False)
        self.validate_value(value)
        self._nominal_value = value

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        if value is None:
            prior = value
        elif isinstance(value, Prior):
            prior = value
        elif isinstance(value, Mapping):
            prior = Prior(**value)
        else:
            raise TypeError("Unhandled prior type {}".format(type(value)))
        self._prior = prior

    @property
    def state(self):
        state = OrderedDict()
        for attr in self._state_attrs:
            val = getattr(self, attr)
            if hasattr(val, 'state'):
                val = val.state
            setitem(state, attr, val)
        return state

    @property
    def serializable_state(self):
        return self.state

    def reset(self):
        """Reset the parameter's value to its nominal value."""
        self.value = self.nominal_value

    def set_nominal_to_current_value(self):
        """Define the nominal value to the parameter's current value."""
        self.nominal_value = self.value

    def randomize(self, random_state=None):
        """Randomize the parameter's value according to a uniform random
        distribution within the parameter's defined limits.

        Parameters
        ----------
        random_state : None, int, or RandomState
            Object to use for random state. None defaults to the global random
            state (this is discouraged, as results are not reproducible). An
            integer acts as a seed to `numpy.random.seed()`, and RandomState is
            a `numpy.random.RandomState` object.

        """
        random = get_random_state(random_state)
        rand = random.rand()
        self._rescaled_value = rand

    def prior_penalty(self, metric):
        """Return the prior penalty according to `metric`.

        Parameters
        ----------
        metric : str
            Metric to use for evaluating the prior penalty.

        Returns
        -------
        penalty : float prior penalty value

        """
        assert isinstance(metric, str)
        metric = metric.strip().lower()
        if metric not in ALL_METRICS:
            raise ValueError('Metric "%s" is invalid; must be one of %s'
                             %(metric, ALL_METRICS))
        if self.prior is None:
            return 0
        if metric in LLH_METRICS:
            logging.trace('self.value: %s' %self.value)
            logging.trace('self.prior: %s' %self.prior)
            return self.prior.llh(self.value)
        elif metric in CHI2_METRICS:
            return self.prior.chi2(self.value)
        else:
            raise ValueError('Unrecognized `metric` "%s"' %str(metric))

    def to(self, units): # pylint: disable=invalid-name
        """Return an equivalent copy of param but in units of `units`.

        Parameters
        ----------
        units : string or pint.Unit

        Returns
        -------
        Param : copy of this param, but in specified `units`.

        See also
        --------
        ito
        Pint.to
        Pint.ito

        """
        new_param = Param(**deepcopy(self.state))
        new_param.ito(units)
        return new_param

    def ito(self, units):
        """Convert this param (in place) to have units of `units`.

        Parameters
        ----------
        units : string or pint.Unit

        Returns
        -------
        None

        See also
        --------
        to
        Pint.to
        Pint.ito

        """
        self._value.ito(units)

    @property
    def prior_llh(self):
        return self.prior_penalty(metric='llh')

    @property
    def prior_chi2(self):
        return self.prior_penalty(metric='chi2')

    @property
    def hash(self):
        """int : hash of full state"""
        if self.normalize_values:
            return hash_obj(normQuant(self.state))
        return hash_obj(self.state)

    def __hash__(self):
        return self.hash

    def to_json(self, filename, **kwargs):
        """Serialize the state to a JSON file that can be instantiated as a new
        object later.
        """
        jsons.to_json(self.serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, filename):
        """Instantiate a new Param from a JSON file"""
        state = jsons.from_json(filename=filename)
        return cls(**state)


# TODO: temporary modification of parameters via "with" syntax?
class ParamSet(Sequence):
    r"""Container class for a set of parameters. Most methods are passed
    through to contained params.

    Parameters
    ----------
    *args : one or more Param objects or sequences thereof

    Examples
    --------
    >>> from pisa import ureg
    >>> from pisa.core.prior import Prior

    >>> e_prior = Prior(kind='gaussian', mean=10*ureg.GeV, stddev=1*ureg.GeV)
    >>> cz_prior = Prior(kind='uniform', llh_offset=-5)
    >>> reco_energy = Param(name='reco_energy', value=12*ureg.GeV,
    ...                     prior=e_prior, range=[1, 80]*ureg.GeV,
    ...                     is_fixed=False, is_discrete=False,
    ...                     tex=r'E^{\rm reco}')
    >>> reco_coszen = Param(name='reco_coszen', value=-0.2, prior=cz_prior,
    ...                     range=[-1, 1], is_fixed=True, is_discrete=False,
    ...                     tex=r'\cos\,\theta_Z^{\rm reco}')
    >>> param_set = ParamSet(reco_energy, reco_coszen)
    >>> print(param_set)
    reco_coszen=-2.0000e-01 reco_energy=+1.2000e+01 GeV

    >>> print(param_set.free)
    reco_energy=+1.2000e+01 GeV

    >>> print(param_set.reco_energy.value)
    12 gigaelectron_volt

    >>> print([p.prior_llh for p in param_set])
    [-5.0, -2]

    >>> print(param_set.priors_llh)
    -7.0

    >>> print(param_set.values_hash)
    3917547535160330856

    >>> print(param_set.free.values_hash)
    -7121742559130813936

    """
    # pylint: disable=protected-access
    def __init__(self, *args):
        # Unpack the input args into a flat list of params
        param_sequence_ = []
        for arg_num, arg in enumerate(args, start=1):
            if isinstance(arg, (Mapping, Param)):
                param_sequence_.append(arg)
            elif isinstance(arg, Iterable):
                param_sequence_.extend(arg)
            else:
                raise TypeError(
                    "Unhandled type {}, arg #{}: {}".format(type(arg), arg_num, arg)
                )

        param_sequence = []
        for param in param_sequence_:
            if isinstance(param, Mapping):
                param = Param(**param)
            if isinstance(param, Param):
                param_sequence.append(param)
            else:
                raise TypeError("Unhandled type {}: {}".format(type(param), param))

        # Disallow duplicated params
        all_names = [p.name for p in param_sequence]
        unique_names = set(all_names)
        if len(unique_names) != len(all_names):
            duplicates = set(x for x in all_names if all_names.count(x) > 1)
            raise ValueError('Duplicate definitions found for param(s): ' +
                             ', '.join(str(e) for e in duplicates))

        # Elements of list must be Param type
        assert all([isinstance(x, Param) for x in param_sequence]), \
                'All params must be of type "Param"'

        self._params = param_sequence
        self.normalize_values = False

    @property
    def serializable_state(self):
        return [p.state for p in self]

    @property
    def _by_name(self):
        return {obj.name: obj for obj in self._params}

    def index(self, value):
        """Return an integer index to the Param in this ParamSet indexed by
        `value`. This does not look up a param's `value` property but looks for
        param by name, integer index, or matching object.

        Parameters
        ----------
        value : int, str or Param
            The object to return an index for. If int, the integer is returned
            (so long as it's in the valid range). If str, return index of param
            with matching `name` attribute. If Param object, return index of an
            equivalent Param in this set.

        Returns
        -------
        idx : int index to a corresponding param in this ParamSet

        Raises
        ------
        ValueError : if `value` does not correspond to a param in this ParamSet

        """
        idx = -1
        if isinstance(value, int):
            idx = value
        elif value in self.names:
            idx = self.names.index(value)
        elif isinstance(value, Param) and value in self._params:
            idx = self._params.index(value)
        if idx < 0 or idx >= len(self):
            raise ValueError('%s not found in ParamSet' % (value,))
        return idx

    def replace(self, new):
        """Replace an existing param with `new` param, where the existing param
        must have the same `name` attribute as `new`.

        Parameters
        ----------
        new : Param
            New param to use instead of current param.

        Raises
        ------
        ValueError : if `new.name` does not match an existing param's name

        """
        idx = self.index(new.name)
        self._params[idx] = new

    def set_values(self, new_params):
        """Set values in current ParamSet to those defined in new ParamSet

        Parameters
        ----------
        new_params : ParamSet
            ParamSet containing set of values to change current ParamSet to.
        """
        if self.names != new_params.names:
            raise ValueError('ParamSet names do not match. Currently have %s '
                             'and want to change to a set containing %s.'
                             % (self.names, new_params.names))
        for new_param in new_params:
            self[new_param.name].value = new_param.value

    def fix(self, x):
        """Set param(s) to be fixed in value (and hence not modifiable by e.g.
        a minimizer).

        Note that the operation is atomic: If `x` is a sequence of indexing
        objects, if _any_ index in `x` cannot be found, _no_ other params
        specified in `x` will be set to be fixed.

        Any params specified in `x` that are already fixed simply remain so.

        Parameters
        ----------
        x : int, str, Param, or iterable thereof
            Object or sequence to index into params to define which to affix.
            See `index` method for valid objects to use for indexing into the
            ParamSet.

        Raises
        ------
        ValueError : if any index cannot be found

        """
        if isinstance(x, (Param, int, str)):
            x = [x]
        indices = set()
        for obj in x:
            indices.add(self.index(obj))
        for idx in indices:
            self[idx].is_fixed = True

    def unfix(self, x):
        """Set param(s) to be free (and hence  modifiable by e.g. a minimizer).

        Note that the operation is atomic: If `x` is a sequence of indexing
        objects, if _any_ index in `x` cannot be found, _no_ other params
        specified in `x` will be set to be free.

        Any params specified in `x` that are already free simply remain so.

        Parameters
        ----------
        x : int, str, Param, or iterable thereof
            Object or sequence to index into params to define which to affix.
            See `index` method for valid objects to use for indexing into the
            ParamSet.

        Raises
        ------
        ValueError : if any index cannot be found

        """
        if isinstance(x, (Param, int, str)):
            x = [x]
        indices = set()
        for obj in x:
            indices.add(self.index(obj))
        for idx in indices:
            self[idx].is_fixed = False

    def update(self, obj, existing_must_match=False, extend=True):
        """Update this param set using `obj`.

        Default behavior is similar to Python's dict.update, but this can be
        modified via `existing_must_match` and `extend`.

        Parameters
        ----------
        obj : Param, ParamSet, or sequence thereof
            Param or container with params to update and/or extend this param
            set
        existing_must_match : bool
            If True, raises ValueError if param values passed in that already
            exist in this param set have differing values.
        extend : bool
            If True, params not in this param set are appended.

        """
        if isinstance(obj, (Sequence, ParamSet)):
            for param in obj:
                self.update(param, existing_must_match=existing_must_match,
                            extend=extend)
            return
        if not isinstance(obj, Param):
            raise ValueError('`obj`="%s" is not a Param' % (obj))
        param = obj
        if param.name in self.names:
            if existing_must_match and (param != self[param.name]):
                raise ValueError(
                    'Param "%s" specified as\n\n%s\n\ncontradicts'
                    ' internally-stored version:\n\n%s'
                    %(param.name, param.state, self[param.name].state)
                )
            self.replace(param)
        elif extend:
            self._params.append(param)

    def extend(self, obj):
        """Append param(s) in `obj` to this param set, but ensure params in
        `obj` that are already in this param set match. Params with same name
        attribute are not duplicated.

        (Convenience method or calling `update` method with
        existing_must_match=True and extend=True.)

        """
        self.update(obj, existing_must_match=True, extend=True)

    def update_existing(self, obj):
        """Only existing params in this set are updated by that(those) param(s)
        in obj.

        Convenience method for calling `update` with
        existing_must_match=False and extend=False.

        """
        self.update(obj, existing_must_match=False, extend=False)

    def __len__(self):
        return len(self._params)

    def __setitem__(self, i, val):
        if isinstance(i, int):
            self._params[i].value = val
        elif isinstance(i, str):
            self._by_name[i].value = val

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._params[i]
        elif isinstance(i, str):
            return self._by_name[i]

    def __getattr__(self, attr):
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            t, v, tb = sys.exc_info()
            try:
                return self[attr]
            except KeyError:
                raise t(v).with_traceback(tb)

    def __setattr__(self, attr, val):
        try:
            params = super().__getattribute__('_params')
            param_names = [p.name for p in params]
        except AttributeError:
            params = []
            param_names = []
        try:
            idx = param_names.index(attr)
        except ValueError:
            super().__setattr__(attr, val)
        else:
            # `attr` (should be) param name
            if isinstance(val, Param):
                assert val.name == attr
                self._params[idx] = val
            elif isinstance(val, ureg.Quantity) or isbarenumeric(val):
                self._params[idx].value = val
            else:
                raise ValueError('Cannot set param "%s" to `val`=%s'
                                 %(attr, val))

    def __iter__(self):
        return iter(self._params)

    def __str__(self):
        numfmt = '%+.4e'
        strings = []
        for p in self:
            string = p.name + '='
            if isinstance(p.value, ureg.Quantity):
                string += numfmt %p.m
                full_unit_str = str(p.u)
                if full_unit_str in [str(ureg('electron_volt ** 2').u)]:
                    unit = ' eV2'
                elif full_unit_str in [str(ureg.deg)]:
                    unit = ' deg'
                elif full_unit_str in [str(ureg.rad)]:
                    unit = ' rad'
                else:
                    unit = ' ' + format(p.u, '~')
                string += unit
            else:
                try:
                    string += numfmt %p.value
                except TypeError:
                    string += '%s' %p.value
            strings.append(string.strip())
        return ' '.join(strings)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return recursiveEquality(self.state, other.state)

    def priors_penalty(self, metric):
        """Return the aggregate prior penalty for all params at their current
        values.

        Parameters
        ----------
        metric : str
            Metric to use for evaluating the prior.

        Returns
        -------
        penalty : float sum of all parameters' prior values

        """
        return np.sum([obj.prior_penalty(metric=metric)
                       for obj in self._params])

    def priors_penalties(self, metric):
        """Return the prior penalties for each param at their current values.

        Parameters
        ----------
        metric : str
            Metric to use for evaluating the prior.

        Returns
        -------
        penalty : list of float prior values, one for each param

        """
        return [obj.prior_penalty(metric=metric) for obj in self._params]

    def reset_all(self):
        """Reset both free and fixed parameters to their nominal values."""
        self.values = self.nominal_values

    def reset_free(self):
        """Reset only free parameters to their nominal values."""
        self.free.reset_all()

    def set_nominal_by_current_values(self):
        """Define the nominal values as the parameters' current values."""
        self.nominal_values = self.values

    def randomize_free(self, random_state=None):
        """Randomize any free parameters with according to a uniform random
        distribution within the parameters' defined limits.

        Parameters
        ----------
        random_state : None, int, or RandomState
            Object to use for random state. None defaults to the global random
            state (this is discouraged, as results are not reproducible). An
            integer acts as a seed to `numpy.random.seed()`, and RandomState is
            a `numpy.random.RandomState` object.

        """
        random = get_random_state(random_state)
        n = len(self.free)
        rand = random.rand(n)
        self.free._rescaled_values = rand

    @property
    def _rescaled_values(self):
        """Parameter values rescaled to be in the range [0, 1], based upon
        their defined range."""
        return tuple(param._rescaled_value for param in self._params)

    @_rescaled_values.setter
    def _rescaled_values(self, vals):
        assert len(vals) == len(self)
        for param, val in zip(self._params, vals):
            param._rescaled_value = val

    @property
    def tex(self):
        return r',\;'.join([obj.tex for obj in self._params])

    @property
    def fixed(self):
        return ParamSet([obj for obj in self._params if obj.is_fixed])

    @property
    def free(self):
        return ParamSet([obj for obj in self._params if not obj.is_fixed])

    @property
    def continuous(self):
        return ParamSet([obj for obj in self._params if not obj.is_discrete])

    @property
    def discrete(self):
        return ParamSet([obj for obj in self._params if obj.is_discrete])

    @property
    def are_fixed(self):
        return tuple(obj.is_fixed for obj in self._params)

    @property
    def are_discrete(self):
        return tuple(obj.is_discrete for obj in self._params)

    @property
    def names(self):
        return tuple(obj.name for obj in self._params)

    @property
    def values(self):
        return tuple(obj.value for obj in self._params)

    @values.setter
    def values(self, values):
        assert len(values) == len(self._params)
        for i, val in enumerate(values):
            setattr(self._params[i], 'value', val)

    @property
    def name_val_dict(self):
        d = OrderedDict()
        for name, val in zip(self.names, self.values):
            d[name] = val
        return d

    @property
    def is_nominal(self):
        return np.all([(v0 == v1)
                       for v0, v1 in zip(self.values, self.nominal_values)])

    @property
    def nominal_values(self):
        return [obj.nominal_value for obj in self._params]

    @nominal_values.setter
    def nominal_values(self, values):
        assert len(values) == len(self._params)
        for i, val in enumerate(values):
            setattr(self._params[i], 'nominal_value', val)

    @property
    def priors(self):
        return tuple(obj.prior for obj in self._params)

    @priors.setter
    def priors(self, values):
        assert len(values) == len(self._params)
        for i, val in enumerate(values):
            setattr(self._params[i], 'prior', val)

    @property
    def priors_llh(self):
        return np.sum([obj.prior_llh for obj in self._params])

    @property
    def priors_chi2(self):
        return np.sum([obj.prior_chi2 for obj in self._params])

    @property
    def ranges(self):
        return tuple(obj.range for obj in self._params)

    @ranges.setter
    def ranges(self, values):
        assert len(values) == len(self._params)
        for i, val in enumerate(values):
            setattr(self._params[i], 'range', val)

    @property
    def state(self):
        return tuple(obj.state for obj in self._params)

    @property
    def values_hash(self):
        """int : hash only on the current param values (not full state)"""
        if self.normalize_values:
            return hash_obj(normQuant(self.values))
        return hash_obj(self.values)

    @property
    def nominal_values_hash(self):
        """int : hash only on the nominal param values"""
        if self.normalize_values:
            return hash_obj(normQuant(self.nominal_values))
        return hash_obj(self.nominal_values)

    @property
    def hash(self):
        """int : full state hash"""
        if self.normalize_values:
            return hash_obj(normQuant(self.state))
        return hash_obj(self.state)

    def __hash__(self):
        return self.hash

    def to_json(self, filename, **kwargs):
        """Serialize the state to a JSON file that can be instantiated as a new
        object later.
        """
        jsons.to_json(self.serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, filename):
        """Instantiate a new ParamSet from a JSON file"""
        state = jsons.from_json(filename=filename)
        if isinstance(state, Sequence):
            return cls(*state)
        if isinstance(state, Mapping):
            return cls(*tuple(state.values()))
        raise TypeError(
            'Unhandled type loaded from "{}": {}\n{}'.format(
                filename, type(state), state
            )
        )


class ParamSelector:
    """
    Parameters
    ----------
    regular_params : ParamSet or instantiable thereto

    selector_param_sets : None, dict, or sequence of dict
        Dict(s) format:
            {
              '<name1>': <ParamSet or instantiable thereto>,
              '<name2>': <ParamSet or instantiable thereto>,
              ...
            }
        The names are what must be specified in `selections` to select the
        corresponding ParamSets. Params specified in any of the ParamSets
        within `selector_param_sets cannot be in `regular_params`.

    selections : None, string, or sequence of strings
        One string is required per

    Notes
    -----
    Params specified in `regular_params` are enforced to be mutually exclusive
    with params in the param sets specified by `selector_param_sets`.

    """
    # pylint: disable=protected-access
    def __init__(self, regular_params=None, selector_param_sets=None,
                 selections=None):
        self._current_params = ParamSet()
        self._regular_params = ParamSet()
        self._selector_params = {}
        self._selections = []

        if regular_params is not None:
            self.update(regular_params, selector=None)

        if selector_param_sets is not None:
            for selector, params in selector_param_sets.items():
                selector = selector.strip().lower()
                params = ParamSet(params)
                self._selector_params[selector] = params

        self.select_params(selections=selections, error_on_missing=False)

    def select_params(self, selections=None, error_on_missing=False):
        if selections is None:
            return self.select_params(selections=self._selections,
                                      error_on_missing=error_on_missing)

        if isinstance(selections, str):
            selections = selections.split(',')

        assert isinstance(selections, Sequence)

        distilled_selections = []
        for selection in selections:
            if selection is None:
                continue
            if not isinstance(selection, str):
                raise ValueError(
                    "Selection should be a str. Got %s instead."%(
                        type(selection))
                )
            selection = selection.strip().lower()
            try:
                if selection not in self._selector_params:
                    raise KeyError(
                        'No selection "%s" available; valid selections are %s'
                        ' (case-insensitive).'
                        %(selection, self._selector_params.keys())
                    )
                self._current_params.update(self._selector_params[selection])
            except KeyError:
                if error_on_missing:
                    raise
            distilled_selections.append(selection)

        self._selections = sorted(distilled_selections)

        return self._current_params

    @property
    def params(self):
        return self._current_params

    @property
    def param_selections(self):
        return deepcopy(self._selections)

    def __iter__(self):
        return iter(self._current_params)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if not recursiveEquality(self._selections, other._selections):
            return False
        if not recursiveEquality(self._regular_params, other._regular_params):
            return False
        if not recursiveEquality(self._selector_params, other._selector_params):
            return False
        return True

    def update(self, p, selector=None):
        try:
            p = ParamSet(p)
        except:
            logging.error('Could not instantiate a ParamSet with `p` of type'
                          ' %s, value = %s', type(p), p)
            raise

        if selector is None:
            self._regular_params.update(p)
            self._current_params.update(p)
        else:
            assert isinstance(selector, str)
            selector = selector.strip().lower()
            if selector not in self._selector_params:
                self._selector_params[selector] = ParamSet()
            self._selector_params[selector].update(p)

            # Re-select current selectiosn in case the update modifies these
            self.select_params(error_on_missing=False)

    def get(self, name, selector=None):
        if selector is None:
            return self._regular_params[name]
        try:
            return self._selector_params[selector][name]
        except KeyError:
            return self._regular_params[name]


def test_Param():
    """Unit tests for Param class"""
    # pylint: disable=unused-variable
    from scipy.interpolate import splrep

    temp_dir = tempfile.mkdtemp()

    counter = [0]
    def check_json(param, name=""):
        fpath = join(temp_dir, "foo{}.json".format(counter[0]))
        counter[0] += 1
        param.to_json(fpath)
        loaded = Param.from_json(fpath)
        if not (
            recursiveEquality(param, loaded)
            and recursiveEquality(param.serializable_state, loaded.serializable_state)
        ):
            msg = (
                "{}: failed to save / load identical param:\n"
                "original = {:s}\n"
                "loaded   = {:s}"
                .format(name, str(param), str(loaded))
            )
            logging.error(msg)
            raise ValueError(msg)

    try:
        uniform = Prior(kind='uniform', llh_offset=1.5)
        gaussian = Prior(kind='gaussian', mean=10*ureg.meter, stddev=1*ureg.meter)
        param_vals = np.linspace(-10, 10, 100) * ureg.meter
        llh_vals = (param_vals.magnitude)**2
        linterp_m = Prior(kind='linterp', param_vals=param_vals, llh_vals=llh_vals)
        linterp_nounits = Prior(kind='linterp', param_vals=param_vals.m,
                                llh_vals=llh_vals)

        param_vals = np.linspace(-10, 10, 100)
        llh_vals = param_vals**2
        knots, coeffs, deg = splrep(x=param_vals, y=llh_vals)

        spline = Prior(kind='spline', knots=knots, coeffs=coeffs, deg=deg)

        # Param with units, prior with compatible units
        p0 = Param(name='c', value=1.5*ureg.foot, prior=gaussian,
                   range=[-1, 2]*ureg.mile, is_fixed=False, is_discrete=False,
                   tex=r'\int{\rm c}')
        check_json(p0, "p0")

        # Param with no units, prior with no units
        p1 = Param(name='c', value=1.5, prior=spline, range=[1, 2],
                   is_fixed=False, is_discrete=False, tex=r'\int{\rm c}')
        check_json(p1, "p1")

        # Param with no units, prior with units
        try:
            p2 = Param(name='c', value=1.5, prior=linterp_m,
                       range=[1, 2], is_fixed=False, is_discrete=False,
                       tex=r'\int{\rm c}')
            _ = p2.prior_llh
            logging.debug(str(p2))
            logging.debug(str(linterp_m))
            logging.debug('p2.units: %s', p2.units)
            logging.debug('p2.prior.units: %s', p2.prior.units)
        except (TypeError, pint.DimensionalityError):
            pass
        else:
            assert False

        # Param with units, prior with no units
        try:
            p2 = Param(name='c', value=1.5*ureg.meter, prior=spline, range=[1, 2],
                       is_fixed=False, is_discrete=False, tex=r'\int{\rm c}')
            _ = p2.prior_llh
        except (ValueError, TypeError, AssertionError):
            pass
        else:
            assert False
        try:
            p2 = Param(name='c', value=1.5*ureg.meter, prior=linterp_nounits,
                       range=[1, 2], is_fixed=False, is_discrete=False,
                       tex=r'\int{\rm c}')
            _ = p2.prior_llh
        except (ValueError, TypeError, AssertionError):
            pass
        else:
            assert False

        # Param, range, prior with no units
        p2 = Param(name='c', value=1.5, prior=linterp_nounits,
                   range=[1, 2], is_fixed=False, is_discrete=False,
                   tex=r'\int{\rm c}')
        _ = p2.prior_llh

        # Param, prior with no units, range with units
        try:
            p2 = Param(name='c', value=1.5, prior=linterp_nounits,
                       range=[1, 2]*ureg.m, is_fixed=False, is_discrete=False,
                       tex=r'\int{\rm c}')
            _ = p2.prior_llh
            logging.debug(str(p2))
            logging.debug(str(linterp_nounits))
            logging.debug('p2.units: %s', p2.units)
            logging.debug('p2.prior.units: %s', p2.prior.units)
        except (ValueError, TypeError, AssertionError):
            pass
        else:
            assert False

        nom0 = p2.nominal_value
        val0 = p2.value
        p2.value = p2.value * 1.01
        val1 = p2.value
        assert p2.value != val0
        assert p2.value == val0 * 1.01
        assert p2.value != nom0
        assert p2.nominal_value == nom0

        p2.reset()
        assert p2.value == nom0
        assert p2.nominal_value == nom0

        p2.value = val1
        p2.set_nominal_to_current_value()
        assert p2.nominal_value == p2.value
        assert p2.nominal_value == val1, \
                '%s should be %s' %(p2.nominal_value, val1)

        # Test deepcopy
        param2 = deepcopy(p2)
        assert param2 == p2

    finally:
        rmtree(temp_dir)

    logging.info('<< PASS : test_Param >>')


# TODO: add tests for reset() and reset_all() methods
def test_ParamSet():
    """Unit tests for ParamSet class"""
    # pylint: disable=attribute-defined-outside-init
    p0 = Param(name='c', value=1.5, prior=None, range=[1, 2],
               is_fixed=False, is_discrete=False, tex=r'\int{\rm c}')
    p1 = Param(name='a', value=2.5, prior=None, range=[1, 5],
               is_fixed=False, is_discrete=False, tex=r'{\rm a}')
    p2 = Param(name='b', value=1.5, prior=None, range=[1, 2],
               is_fixed=False, is_discrete=False, tex=r'{\rm b}')
    param_set = ParamSet(p0, p1, p2)
    logging.debug(str((param_set.values)))
    logging.debug(str((param_set[0])))
    param_set[0].value = 1
    logging.debug(str((param_set.values)))

    param_set.values = [1.5, 5, 1]
    logging.debug(str((param_set.values)))
    logging.debug(str((param_set.values[0])))
    logging.debug(str((param_set[0].value)))

    logging.debug(str(('priors:', param_set.priors)))
    logging.debug(str(('names:', param_set.names)))

    logging.debug(str((param_set['a'])))
    logging.debug(str((param_set['a'].value)))
    logging.debug(str((param_set['a'].range)))
    try:
        param_set['a'].value = 33
    except:
        pass
    else:
        assert False, 'was able to set value outside of range'
    logging.debug(str((param_set['a'].value)))

    logging.debug(str((param_set['c'].is_fixed)))
    param_set['c'].is_fixed = True
    logging.debug(str((param_set['c'].is_fixed)))
    logging.debug(str((param_set.are_fixed)))
    param_set.fix('a')
    logging.debug(str((param_set.are_fixed)))
    param_set.unfix('a')
    logging.debug(str((param_set.are_fixed)))
    param_set.unfix([0, 1, 2])
    logging.debug(str((param_set.are_fixed)))

    fixed_params = param_set.fixed
    logging.debug(str((fixed_params.are_fixed)))
    free_params = param_set.free
    logging.debug(str((free_params.are_fixed)))
    logging.debug(str((param_set.free.values)))

    logging.debug(str((param_set.values_hash)))
    logging.debug(str((param_set.fixed.values_hash)))
    logging.debug(str((param_set.free.values_hash)))

    logging.debug(str((param_set[0].state)))
    logging.debug(str((param_set.hash)))
    logging.debug(str((param_set.fixed.hash)))
    logging.debug(str((param_set.free.hash)))

    logging.debug(str(('fixed:', param_set.fixed.names)))
    logging.debug(str(('fixed, discrete:', param_set.fixed.discrete.names)))
    logging.debug(str(('fixed, continuous:',
                       param_set.fixed.continuous.names)))
    logging.debug(str(('free:', param_set.free.names)))
    logging.debug(str(('free, discrete:', param_set.free.discrete.names)))
    logging.debug(str(('free, continuous:', param_set.free.continuous.names)))
    logging.debug(str(('continuous, free:', param_set.continuous.free.names)))
    logging.debug(str(('free, continuous hash:',
                       param_set.free.continuous.values_hash)))
    logging.debug(str(('continuous, free hash:',
                       param_set.continuous.free.values_hash)))

    logging.debug(str((param_set['b'].prior_llh)))
    logging.debug(str((param_set.priors_llh)))
    logging.debug(str((param_set.free.priors_llh)))
    logging.debug(str((param_set.fixed.priors_llh)))

    logging.debug(str((param_set[0].prior_chi2)))
    logging.debug(str((param_set.priors_chi2)))

    # Test that setting attributes works
    e_prior = Prior(kind='gaussian', mean=10*ureg.GeV, stddev=1*ureg.GeV)
    cz_prior = Prior(kind='uniform', llh_offset=-5)
    reco_energy = Param(name='reco_energy', value=12*ureg.GeV,
                        prior=e_prior, range=[1, 80]*ureg.GeV,
                        is_fixed=False, is_discrete=False,
                        tex=r'E^{\rm reco}')
    reco_coszen = Param(name='reco_coszen', value=-0.2, prior=cz_prior,
                        range=[-1, 1], is_fixed=True, is_discrete=False,
                        tex=r'\cos\,\theta_Z^{\rm reco}')
    reco_coszen_fail = Param(name='reco_coszen_fail', value=-0.2,
                             prior=cz_prior, range=[-1, 1], is_fixed=True,
                             is_discrete=False,
                             tex=r'\cos\,\theta_Z^{\rm reco}')
    reco_coszen2 = Param(name='reco_coszen', value=-0.9, prior=cz_prior,
                         range=[-1, 1], is_fixed=True, is_discrete=False,
                         tex=r'\cos\,\theta_Z^{\rm reco}')
    param_set = ParamSet([reco_energy, reco_coszen])
    # Try setting a param with a differently-named param
    try:
        param_set.reco_coszen = reco_coszen_fail
    except:
        pass
    else:
        assert False

    try:
        param_set.reco_coszen = 30
    except:
        pass
    else:
        assert False

    param_set.reco_coszen = reco_coszen2
    assert param_set.reco_coszen is reco_coszen2
    assert param_set['reco_coszen'] is reco_coszen2
    assert param_set.reco_coszen.value == -0.9
    param_set.reco_coszen = -1.0
    assert param_set.reco_coszen.value == -1.0
    param_set.reco_coszen = -1
    assert param_set.reco_coszen.value == -1.0
    param_set.reco_coszen = ureg.Quantity("0.1 dimensionless")
    assert param_set.reco_coszen.value == 0.1 == ureg.Quantity("0.1 dimensionless")
    param_set.reco_energy = ureg.Quantity("10.1 GeV")
    assert param_set.reco_energy.value == ureg.Quantity("10.1 GeV")

    # Test deepcopy
    param_set2 = deepcopy(param_set)
    logging.debug(str((param_set)))
    logging.debug(str((param_set2)))
    assert param_set2 == param_set

    # Test case added as a result of issue #543
    # https://github.com/IceCubeOpenSource/pisa/issues/543
    param_set = ParamSet(
        [
            Param(
                name="param0",
                value=1*ureg.dimensionless,
                prior=None,
                range=(-1, 2)*ureg.dimensionless,
                is_fixed=False,
            ),
            Param(
                name="param1",
                value=1,
                prior=None,
                range=(-1, 2),
                is_fixed=False,
            ),
            Param(
                name="param2",
                value=1,
                prior=None,
                range=(-1.1, 1.1),
                is_fixed=False,
            ),
            Param(
                name="param3",
                value=2.0*ureg.m/ureg.s,
                prior=None,
                range=(-1, 10)*ureg.cm/ureg.ns,
                is_fixed=True,
            ),
            Param(
                name="param4",
                value=2.1*ureg.m/ureg.s,
                prior=None,
                range=(-1.1, 1.1)*ureg.cm/ureg.ns,
                is_fixed=True,
            ),
        ]
    )
    temp_dir = tempfile.mkdtemp()
    try:
        fpath = join(temp_dir, "foo.json")
        param_set.to_json(fpath)
        param_set2 = ParamSet.from_json(fpath)
    finally:
        rmtree(temp_dir)
    assert recursiveEquality(param_set, param_set2)
    assert recursiveEquality(param_set.serializable_state, param_set2.serializable_state)

    logging.info('<< PASS : test_ParamSet >>')


def test_ParamSelector():
    """Unit tests for ParamSelector class"""
    # pylint: disable=attribute-defined-outside-init, no-member, invalid-name
    p0 = Param(name='a', value=1.5, prior=None, range=[1, 2],
               is_fixed=False, is_discrete=False, tex=r'\int{\rm c}')
    p1 = Param(name='b', value=2.5, prior=None, range=[1, 5],
               is_fixed=False, is_discrete=False, tex=r'{\rm a}')
    p20 = Param(name='c', value=1.5, prior=None, range=[1, 2],
                is_fixed=False, is_discrete=False, tex=r'{\rm b}')
    p21 = Param(name='c', value=2.0, prior=None, range=[1, 2],
                is_fixed=False, is_discrete=False, tex=r'{\rm b}')
    p22 = Param(name='c', value=1.0, prior=None, range=[1, 2],
                is_fixed=False, is_discrete=False, tex=r'{\rm b}')
    p30 = Param(name='d', value=-1.5, prior=None, range=[-1, -2],
                is_fixed=False, is_discrete=False, tex=r'{\rm b}')
    p31 = Param(name='d', value=-2.0, prior=None, range=[-1, -2],
                is_fixed=False, is_discrete=False, tex=r'{\rm b}')
    p40 = Param(name='e', value=-15, prior=None, range=[-10, -20],
                is_fixed=False, is_discrete=False, tex=r'{\rm b}')
    p41 = Param(name='e', value=-20, prior=None, range=[-10, -20],
                is_fixed=False, is_discrete=False, tex=r'{\rm b}')
    ps30_40 = ParamSet(p30, p40)
    param_selector = ParamSelector(
        regular_params=[p0, p1],
        selector_param_sets={'p20': p20, 'p21': p21, 'p22': p22,
                             'p30_40': ps30_40, 'p31_41': [p31, p41]},
        selections=['p20', 'p30_40']
    )
    params = param_selector.params
    assert params.a.value == 1.5
    assert params.b.value == 2.5
    assert params.c.value == 1.5
    assert params.d.value == -1.5
    assert params.e.value == -15

    # Modify a param's value from the selector's params
    params.c = 1.8
    # Make sure that took
    assert params['c'].value == 1.8
    # Make sure the original param was also modified (i.e., that it's the exact
    # object that was populated to the param_selector's params)
    assert p20.value == 1.8

    param_selector.select_params('p21')
    # Make sure 'c' is changed using all ways to access 'c'
    assert param_selector.params.c.value == 2.0
    assert param_selector.params['c'].value == 2.0
    assert params['c'].value == 2.0
    assert params.c.value == 2.0
    # Make sure original params have values previous to selection
    assert p20.value == 1.8
    assert p21.value == 2.0

    # Change the newly-selected param's value
    params.c = 1.9
    # Make sure that took
    assert params['c'].value == 1.9
    # Make sure the original param was also modified (i.e., that it's the exact
    # object that was populated to the param_selector's params)
    assert p21.value == 1.9

    param_selector.select_params('p31_41')
    assert params['d'].value == -2
    assert params['e'].value == -20
    params.e = -19.9
    assert p41.value == -19.9

    # Test the update method

    p5 = Param(name='f', value=120, prior=None, range=[0, 1000],
               is_fixed=True, is_discrete=False, tex=r'{\rm b}')
    p60 = Param(name='g', value=-1, prior=None, range=[-10, 10],
                is_fixed=False, is_discrete=False, tex=r'{\rm b}')
    p61 = Param(name='g', value=-2, prior=None, range=[-10, 10],
                is_fixed=False, is_discrete=False, tex=r'{\rm b}')

    # Update with a "regular" param that doesn't exist yet
    param_selector.update(p=p5, selector=None)
    assert params.f.value == 120

    # Update with a new "selector" param with selector that's currently
    # selected
    param_selector.update(p=p61, selector='p31_41')
    assert params.g.value == -2
    p = param_selector.get(name='g', selector='p31_41')
    assert p.value == -2

    # Update with a new "selector" param with selector that's _not_ currently
    # selected
    param_selector.update(p=p60, selector='p30_40')

    # Selected param value shouldn't have changed
    assert params.g.value == -2

    # ... but the param should be in the object
    p = param_selector.get(name='g', selector='p30_40')
    assert p.value == -1

    # ... and selecting it should now set current param to its value
    param_selector.select_params('p30_40')
    assert params.g.value == -1

    # Double check that the other one didn't change
    p = param_selector.get(name='g', selector='p31_41')
    assert p.value == -2

    # Use update to overwrite existing params...

    p402 = Param(name='e', value=-11, prior=None, range=[0, -20],
                 is_fixed=False, is_discrete=False, tex=r'{\rm b}')
    p412 = Param(name='e', value=-22, prior=None, range=[0, -100],
                 is_fixed=False, is_discrete=False, tex=r'{\rm b}')

    # Update param that exists already and is selected
    param_selector.update(p=p402, selector='p30_40')
    assert params.e.value == -11

    # Make sure original param wasn't overwritten (just not in param_selector)
    assert p40.value == -15

    # Update param that exists already but is not selected
    param_selector.update(p=p412, selector='p31_41')
    assert params.e.value == -11
    p = param_selector.get('e', selector='p31_41')
    assert p.value == -22
    param_selector.select_params('p31_41')
    assert params.e.value == -22

    # Test deepcopy
    param_selector2 = deepcopy(param_selector)
    assert param_selector2 == param_selector

    logging.info('<< PASS : test_ParamSelector >>')


if __name__ == "__main__":
    set_verbosity(1)
    test_Param()
    test_ParamSet()
    test_ParamSelector()
