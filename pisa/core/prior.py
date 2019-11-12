"""
Prior class for use in pisa.core.Param objects
"""


from __future__ import absolute_import, division

from collections.abc import Iterable
from collections import OrderedDict
from numbers import Number
from operator import setitem

import numpy as np
from scipy.interpolate import splev, splrep, interp1d
from scipy.optimize import fminbound

import pint
from pisa import ureg
from pisa.utils.comparisons import isbarenumeric, recursiveEquality
from pisa.utils.fileio import from_file
from pisa.utils.log import logging, set_verbosity


__all__ = ['Prior', 'plot_prior', 'get_prior_bounds', 'test_Prior',
           'test_Prior_plot']

__author__ = 'J.L. Lanfranchi'

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


# TODO: uniform prior should take a constant, such that e.g. discrete parameter
# values when run separately will return valid comparisons across the
# discretely-chosen values (with different uniform priors)

# TODO: use units "natively" (not via strings) internal to the object; only
# serializing to json should convert to strings (and deserializing should
# convert from strings to Units objects)

# TODO: add a "to" and/or "ito" method for converting units akin to those
# methods in Pint quantities.
class Prior(object):
    """Prior information for a parameter. Defines the penalty (in
    log-likelihood (llh)) for a parameter being at a given value (within the
    prior's valid parameter range). Chi-squared penalties can also be returned
    (but the *definition* of a prior here is always in terms of llh).

    Note that since this is a penalty, the more negative the prior's log
    likelihood, the greater the penalty and the less likely the parameter's
    value is.

    Valid parameters and properties of the object differ based upon what `kind`
    of prior is specified.

    Parameters
    ----------
    kind='uniform', llh_offset=...
        Uniform prior, no preference for any position relative to the valid
        range, which is taken to be [-inf, +inf] [x-units].

    kind='gaussian', mean=..., stddev=...
        Gaussian prior, defining log likelihood penalty for parameter being at
        any particular position. Valid range is [-inf, +inf] [x-units].

    kind='linterp', param_vals=..., llh_vals=...
        Linearly-interpolated prior. Note that "corners" in linear
        interpolation may cause difficulties for some minimizers.

    kind='spline', knots=..., coeffs=..., deg=...
        Smooth spline interpolation.

    Properties
    ----------
    kind
    max_at
    max_at_str
    state
    valid_range

    Additional properties are defined based on `kind`:
    kind='uniform':
        llh_offset

    kind='gaussian':
        mean
        stddev

    kind='linterp':
        param_vals
        llh_vals

    kind='spline':
        knots
        coeffs
        deg

    Methods
    -------
    chi2
    llh

    Notes
    -----
    If the parameter the prior is being applied to has units, the prior's
    "x"-values specification must have compatible units.

    If you implement a new prior, it ***must*** raise an exception if methods
    `llh` or `chi2` are called with a parameter value outside the prior's valid
    range, so subtle bugs aren't introduced that appear as an issue in e.g. the
    minimizer.

    Examples
    --------
    For spline prior: knots, coeffs, and deg can be found by, e.g.,
    scipy.interpolate.splrep; evaluation of spline priors is carried out
    internally by scipy.interpolate.splev, so an exact match to the output of
    the spline prior can be produced as follows:

    >>> from scipy.interpolate import splrep, splev
    >>> # Generate sample points
    >>> param_vals = np.linspace(-10, 10, 100)
    >>> llh_vals = param_vals**2
    >>> # Define spline interpolant
    >>> knots, coeffs, deg = splrep(param_vals, llh_vals)
    >>> # Instantiate spline prior
    >>> prior = Prior(kind='spline', knots=knots, coeffs=coeffs, deg=deg)
    >>> # Generate sample points for interpolation
    >>> param_upsamp = np.linspace(-10, 10, 1000)
    >>> # Evaluation of spline using splev
    >>> llh_upsamp = splev(param_upsamp, tck=(knots, coeffs, deg), ext=2)
    >>> # Check that evaluation of spline matches call to prior.llh()
    >>> all(prior.llh(param_upsamp) == llh_upsamp)
    True

    """
    def __init__(self, kind, **kwargs):
        self._state_attrs = ['kind', 'max_at', 'units', 'valid_range']
        self.units = None
        kind = kind.lower() if isinstance(kind, str) else kind

        self.chi2 = lambda x: -2*self.llh(x)
        # Dispatch the correct initialization method
        if kind in [None, 'none', 'uniform']:
            self.__init_uniform(**kwargs)
        elif kind == 'gaussian':
            self.__init_gaussian(**kwargs)
        elif kind == 'linterp':
            self.__init_linterp(**kwargs)
        elif kind == 'spline':
            self.__init_spline(**kwargs)
        elif kind == 'jeffreys':
            self.__init_jeffreys(**kwargs)
        else:
            raise TypeError('Unknown Prior kind `' + str(kind) + '`')

    @property
    def units_str(self):
        if self.units is None:
            return ''
        return ' ' + format(ureg(self.units).units, '~').strip()

    def __str__(self):
        return self._str(self)

    def __repr__(self):
        return '<' + str(self.__class__) + ' ' + self.__str__() + '>'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return recursiveEquality(self.state, other.state)

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def state(self):
        state = OrderedDict()
        for attr in self._state_attrs:
            setitem(state, attr, getattr(self, attr))
        return state

    def __init_uniform(self, llh_offset=0):
        self._state_attrs.extend(['llh_offset'])
        self.kind = 'uniform'
        self.llh_offset = llh_offset
        def llh(x):
            return 0.*self.__strip(x) + self.llh_offset
        self.llh = llh
        self.max_at = np.nan
        self.max_at_str = 'no maximum'
        self.valid_range = (-np.inf * ureg(self.units),
                            np.inf * ureg(self.units))
        self._str = lambda s: 'uniform prior, llh_offset=%s' %self.llh_offset

    def __init_jeffreys(self, A, B):
        """Calculate jeffreys prior as defined in Sivia p.125"""
        self.kind = 'jeffreys'
        if isinstance(A, Number):
            A = A * ureg.dimensionless
        if isinstance(B, Number):
            B = B * ureg.dimensionless
        assert A.dimensionality == B.dimensionality
        self._state_attrs.extend(['A', 'B'])
        if isinstance(A, ureg.Quantity):
            self.units = str(A.units)
            assert isinstance(B, ureg.Quantity), '%s' %type(B)
            B = B.to(self.units)
        self.A = A
        self.B = B
        def llh(x):
            x = self.__strip(self.__convert(x))
            A = self.__strip(self.A)
            B = self.__strip(self.B)
            return - np.log(x) + np.log(np.log(B)-np.log(A))
        self.llh = llh
        self.max_at = self.A
        self.max_at_str = self.__stringify(self.max_at)
        self.valid_range = (self.A * ureg(self.units),
                            self.B * ureg(self.units))
        self._str = lambda s: "jeffreys' prior, range [%s,%s]"%(self.A, self.B)

    def __init_gaussian(self, mean, stddev):
        if isinstance(mean, Number):
            mean = mean * ureg.dimensionless
        if isinstance(stddev, Number):
            stddev = stddev * ureg.dimensionless
        assert mean.dimensionality == stddev.dimensionality
        self._state_attrs.extend(['mean', 'stddev'])
        self.kind = 'gaussian'
        if isinstance(mean, ureg.Quantity):
            self.units = str(mean.units)
            assert isinstance(stddev, ureg.Quantity), \
                    str(type(stddev))
            stddev = stddev.to(self.units)
        self.mean = mean
        self.stddev = stddev
        def llh(x):
            x = self.__strip(self.__convert(x))
            m = self.__strip(self.mean)
            s = self.__strip(self.stddev)
            return -(x-m)**2 / (2*s**2)
        self.llh = llh
        self.max_at = self.mean
        self.max_at_str = self.__stringify(self.max_at)
        self.valid_range = (-np.inf * ureg(self.units),
                            np.inf * ureg(self.units))
        self._str = lambda s: 'gaussian prior: stddev=%s%s, maximum at %s%s' \
                %(self.__stringify(self.stddev), self.units_str,
                  self.__stringify(self.mean), self.units_str)

    def __init_linterp(self, param_vals, llh_vals):
        if not isinstance(param_vals, ureg.Quantity):
            param_vals = param_vals * ureg.dimensionless
        self._state_attrs.extend(['param_vals', 'llh_vals'])
        self.kind = 'linterp'
        if isinstance(param_vals, ureg.Quantity):
            self.units = str(param_vals.units)
        self.interp = interp1d(param_vals, llh_vals, kind='linear', copy=True,
                               bounds_error=True, assume_sorted=False)
        self.param_vals = param_vals
        self.llh_vals = llh_vals
        def llh(x):
            x = self.__strip(self.__convert(x))
            return self.interp(x)
        self.llh = llh
        self.max_at = self.param_vals[self.llh_vals == np.max(self.llh_vals)]
        self.max_at_str = ', '.join([self.__stringify(v) for v in self.max_at])
        self.valid_range = (np.min(self.param_vals) * ureg(self.units),
                            np.max(self.param_vals) * ureg(self.units))
        self._str = lambda s: 'linearly-interpolated prior: valid in [%s, %s]%s, maxima at (%s)%s' \
                %(self.__stringify(np.min(self.param_vals)),
                  self.__stringify(np.max(self.param_vals)), self.units_str,
                  self.max_at_str, self.units_str)

    def __init_spline(self, knots, coeffs, deg, units=None):
        if not isinstance(knots, ureg.Quantity):
            if units is None:
                knots = knots * ureg.dimensionless
            else:
                knots = ureg.Quantity(np.asarray(knots), units)
        self._state_attrs.extend(['knots', 'coeffs', 'deg', 'units'])
        self.kind = 'spline'
        if isinstance(knots, ureg.Quantity):
            self.units = str(knots.units)
        self.knots = knots
        self.coeffs = coeffs
        self.deg = deg
        def llh(x):
            x = self.__strip(self.__convert(x))
            return splev(x, tck=(self.__strip(self.knots), coeffs, deg), ext=2)
        self.llh = llh
        self.max_at = fminbound(
            func=self.__attach_units_to_args(self.chi2),
            x1=np.min(self.__strip(self.knots)),
            x2=np.max(self.__strip(self.knots)),
        )
        if self.units is not None:
            self.max_at = self.max_at * ureg(self.units)
        self.max_at_str = self.__stringify(self.max_at)
        self.valid_range = (np.min(self.knots) * ureg(self.units),
                            np.max(self.knots) * ureg(self.units))
        self._str = lambda s: 'spline prior: deg=%d, valid in [%s, %s]%s; max at %s%s' \
                %(self.deg, self.__stringify(np.min(self.knots)),
                  self.__stringify(np.max(self.knots)), self.units_str,
                  self.max_at_str, self.units_str)

    def __check_units(self, param_val):
        if self.units is None:
            if (isinstance(param_val, ureg.Quantity)
                    and param_val.dimensionality
                    != ureg.dimensionless.dimensionality):
                raise TypeError('Passed a value with units (%s), but this'
                                ' prior has no units.' %param_val.units)
        else:
            if not isinstance(param_val, ureg.Quantity):
                raise TypeError('Passed a value without units, but this prior'
                                ' has units (%s).' %self.units)
            if param_val.dimensionality != ureg(self.units).dimensionality:
                raise TypeError('Passed a value with units (%s);'
                                ' incompatible with prior units (%s)'
                                %(param_val.units, self.units))

    def __convert(self, x):
        if self.units is None:
            if (isinstance(x, ureg.Quantity)
                    and x.dimensionality != ureg.dimensionless.dimensionality):
                raise TypeError('No units on prior, so cannot understand'
                                ' passed value (with units): %s' %x)
            return x
        if not isinstance(x, ureg.Quantity):
            raise TypeError('Units %s must be present on param values (got'
                            ' %s, type %s instead).'
                            % (self.units, x, type(x)))
        return x.to(self.units)

    @staticmethod
    def __strip(x):
        if isinstance(x, ureg.Quantity):
            return x.magnitude
        return x

    def __stringify(self, x):
        if self.units is not None:
            x = x.to(self.units).magnitude
        return format(x, '0.4e')

    # TODO: proper function wrapping, including @wraps decorator
    def __attach_units_to_args(self, func):
        def newfunc(*args):
            if self.units is None:
                return func(*args)
            u = ureg(self.units)
            unitized_args = tuple([u*arg for arg in args])
            return func(*unitized_args)
        return newfunc


def plot_prior(obj, param=None, x_xform=None, ax1=None, ax2=None, **plt_kwargs):
    """Plot prior for param from template settings, params, or prior filename
    or dict.

    Arguments
    ---------
    obj : str or dict
        if str, interpret as path from which to load a dict
        if (nested) dict, (innermost) must be dict of prior properties :
            either supply `param` to choose which parameter's prior in `obj`
            to plot, or prior dict, in which case `param` need not be specified
    param
        Param name to plot; necessary if obj is either pipeline settings or
        params dict
    x_xform
        Transform to apply to x-values. E.g., to plot against sin^2 theta, use
        x_xform = lambda x: np.sin(x)**2
    ax1, ax2
        Axes onto which to plot LLH and chi-squared, respectively. If none are
        provided, new figures & axes will be created.
    plt_kwargs
        Keyword arguments to pass on to the plot function

    Returns
    -------
    ax1, ax2
        The axes onto which plots were drawn (ax1 = LLH, ax2 = chi^2)
    """
    import matplotlib as mpl
    mpl.use('pdf')
    import matplotlib.pyplot as plt
    if isinstance(obj, str):
        obj = from_file(obj)
    if param is not None and param in obj:
        obj = obj[param]
    if 'prior' in obj:
        obj = obj['prior']

    prior = Prior(**obj)
    logging.info('Plotting Prior: %s', prior)
    x0 = prior.valid_range[0]
    x1 = prior.valid_range[1]
    if prior.kind == 'gaussian':
        x0 = max(x0, prior.max_at - 5*prior.stddev)
        x1 = min(x1, prior.max_at + 5*prior.stddev)
    if np.isinf(x0):
        x0 = -1
    if np.isinf(x1):
        x1 = +1
    # if prior.units is None, will result in dimensionless quantity
    x = ureg.Quantity(np.linspace(x0, x1, 5000), prior.units)

    llh = prior.llh(x)
    chi2 = prior.chi2(x)

    if x_xform is not None:
        x = x_xform(x)

    if ax1 is None:
        f = plt.figure()
        ax1 = f.add_subplot(111)
    if ax2 is None:
        f = plt.figure()
        ax2 = f.add_subplot(111)

    ax1.plot(x, llh, **plt_kwargs)
    ax2.plot(x, chi2, **plt_kwargs)

    ax1.set_title(str(prior), fontsize=8, y=1.02)
    ax2.set_title(str(prior), fontsize=8, y=1.02)
    ax1.set_xlabel(param)
    ax2.set_xlabel(param)
    ax1.set_ylabel('LLH')
    ax2.set_ylabel(r'$\Delta\chi^2$')

    return ax1, ax2


def get_prior_bounds(obj, param=None, stddev=1.0):
    """Obtain confidence regions for CL corresponding to given number of
    stddevs from parameter prior.

    Parameters
    ----------
    obj : string or Mapping
        if str, interpret as path from which to load a dict
        if dict, can be:
            template settings dict; must supply `param` to choose which to plot
            params dict; must supply `param` to choose which to plot
            prior dict

    param : Param
        Name of param for which to get bounds;
        necessary if obj is either template settings or params

    stddev : float or Iterable of floats
        number of stddevs


    Returns
    -------
    bounds : OrderedDict
        A dictionary mapping the passed `stddev` values to the corresponding
        bounds

    """
    if isbarenumeric(stddev):
        stddev = [stddev]
    elif isinstance(stddev, Iterable):
        stddev = list(stddev)

    bounds = OrderedDict()
    for s in stddev:
        bounds[s] = []

    if isinstance(obj, str):
        obj = from_file(obj)

    if 'params' in obj:
        obj = obj['params']
    if param is not None and param in obj:
        obj = obj[param]
    if 'prior' in obj:
        obj = obj['prior']

    prior = Prior(**obj)

    logging.debug('Getting confidence region from prior: %s', prior)
    x0 = prior.valid_range[0]
    x1 = prior.valid_range[1]
    x = ureg.Quantity(np.linspace(x0, x1, 10000), prior.units)
    chi2 = prior.chi2(x)
    for (i, xval) in enumerate(x[:-1]):
        for s in stddev:
            chi2_level = s**2
            if chi2[i] > chi2_level and chi2[i+1] < chi2_level:
                bounds[s].append(xval)
            elif chi2[i] < chi2_level and chi2[i+1] > chi2_level:
                bounds[s].append(x[i+1])
    return bounds


# TODO enumerate all the cases rather than picking just a few.

# pylint: disable=unused-variable
def test_Prior():
    """Unit tests for Prior class"""
    uniform = Prior(kind='uniform', llh_offset=1.5)
    gaussian = Prior(kind='gaussian', mean=10, stddev=1)
    x = np.linspace(-10, 10, 100)
    y = x**2
    linterp = Prior(kind='linterp', param_vals=x*ureg.meter,
                    llh_vals=y)
    param_vals = np.linspace(-10, 10, 100)
    llh_vals = x**2
    knots, coeffs, deg = splrep(param_vals, llh_vals)
    spline = Prior(kind='spline', knots=knots*ureg.foot, coeffs=coeffs,
                   deg=deg)
    param_upsamp = np.linspace(-10, 10, 1000)*ureg.foot
    llh_upsamp = splev(param_upsamp, tck=(knots, coeffs, deg), ext=2)
    assert all(spline.llh(param_upsamp) == llh_upsamp)

    # Asking for param value outside of range should fail
    try:
        linterp.llh(-1000*ureg.mile)
    except ValueError:
        pass
    else:
        assert False

    try:
        linterp.chi2(-1000*ureg.km)
    except ValueError:
        pass
    else:
        assert False

    try:
        spline.llh(-1000*ureg.meter)
    except ValueError:
        pass
    else:
        assert False

    try:
        spline.chi2(+1000*ureg.meter)
    except ValueError:
        pass
    else:
        assert False

    # Asking for param value when units were used should fail
    try:
        spline.llh(10)
    except TypeError:
        pass
    else:
        assert False

    # ... or vice versa
    try:
        gaussian.llh(10*ureg.meter)
    except (TypeError, pint.DimensionalityError):
        pass
    else:
        assert False

    logging.info('<< PASS : test_Prior >>')

# TODO: FIX ME
def test_Prior_plot(ts_fname, param_name='theta23'):
    """Produce plots roughly like NuFIT's 1D chi-squared projections.

    Parameters
    ----------
    ts_fname : string
    param_name : string


    Returns
    -------
    ax1, ax2 : Matplotlib.axis
        The plot axes are returned for further manipulation

    """
    import matplotlib as mpl
    mpl.use('pdf')
    import matplotlib.pyplot as plt
    stddev = [1, 2, 3, 4, 5]
    chi2 = [s**2 for s in stddev]

    ts = from_file(ts_fname)
    f1 = plt.figure(1) #,figsize=(8,14),dpi=60)
    f2 = plt.figure(2) #,figsize=(8,14),dpi=60)
    f1.clf()
    f2.clf()
    ax1 = f1.add_subplot(111)
    ax2 = f2.add_subplot(111)

    # Defaults
    x_xform = None
    xlabel = param_name
    xlim = None
    ylim = 0, 15

    # Special cases
    if param_name == 'theta12':
        x_xform = lambda x: np.sin(x)**2
        xlabel = r'$\sin^2\theta_{12}$'
        xlim = 0.2, 0.42
    elif param_name == 'theta23':
        x_xform = lambda x: np.sin(x)**2
        xlabel = r'$\sin^2\theta_{23}$'
        xlim = 0.26, 0.74
    elif param_name == 'theta13':
        x_xform = lambda x: np.sin(x)**2
        xlabel = r'$\sin^2\theta_{13}$'
        xlim = 0.012, 0.032
    elif param_name == 'deltam21':
        x_xform = lambda x: x*1e5
        xlabel = r'$\Delta m^2_{21} \; {\rm[10^{-5}\;eV^2]}$'
        xlim = 6.5, 8.7
    elif param_name == 'deltam31':
        x_xform = lambda x: np.abs(x)*1e3
        xlabel = r'$|\Delta m^2_{31}| \; {\rm[10^{-3}\;eV^2]}$'
        xlim = 2.15, 2.8
    elif param_name == 'deltacp':
        xlabel = r'$\delta_{\rm CP} \; {\rm [deg]}$'
    plot_prior(select_hierarchy(ts['params'], normal_hierarchy=True),
               param=param_name,
               x_xform=x_xform, ax1=ax1, ax2=ax2,
               color='r', label=r'${\rm NH}$')
    plot_prior(select_hierarchy(ts['params'], normal_hierarchy=False),
               param=param_name,
               x_xform=x_xform, ax1=ax1, ax2=ax2,
               color='b', linestyle='--', label=r'${\rm IH}$')

    ax1.set_ylim([-0.5*y for y in ylim[::-1]])
    ax2.set_ylim(ylim)
    plt.tight_layout()

    for ax in [ax1, ax2]:
        ax.legend(loc='best', frameon=False)
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.grid(which='both', b=True)
        ax.set_title('')

    for c2 in chi2:
        ax2.plot(xlim, [c2, c2], 'k-', lw=1.0, alpha=0.4)

    return ax1, ax2


if __name__ == '__main__':
    set_verbosity(1)
    test_Prior()
