"""
Classes to store and handle the evaluation of splines.
"""


from __future__ import division

import inspect
from collections.abc import Mapping, Sequence

from pisa.core.map import Map, MapSet
from pisa.core.binning import MultiDimBinning
from pisa.utils import flavInt
from pisa.utils.profiler import profile


__all__ = ['Spline', 'CombinedSpline']

__author__ = 'S. Mandalia'

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


class Spline(object):
    """Encapsulation of spline evaluation and other operations.

    Provides methods to evaluate the spline object over a given binning.

    Parameters
    ----------
    name : string
        Name for the spline object. Used to identify the object.

    tex : None or string
        TeX string that can be used for e.g. plotting.

    spline:
        Splines used for evaluation.

    eval_spl: function
        Function prescribing how to obtain values for the input spline object
        from a given binning.

    validate_spl: function
        Function performing validation test on a given binning used to evaluate
        the spline.

    hash : None, or immutable object (typically an integer)
        Hash value to attach to the spline.
    """
    # TODO(shivesh): required?
    # TODO(shivesh): hashing?
    _state_attrs = ('name', 'tex', 'spline', 'hash')

    def __init__(self, name, spline, eval_spl, tex=None, validate_spl=None,
                 hash=None):
        # Set Read/write attributes via their defined setters
        self.name = name
        self._spline = spline
        self._hash = hash

        if tex is None:
            tex = flavInt.NuFlavIntGroup(name).tex

        # Ensure eval_spl has correct structure
        eval_args = inspect.getargspec(eval_spl).args
        if len(eval_args) < 2:
            raise ValueError('Evaluation function does not contain the '
                             'minimum number of input parameters (2)\n'
                             'Input function keywords: {0}'.format(eval_args))
        if 'spline' not in eval_args[0]:
            raise ValueError('Evaluation function does not contain the ' +
                             "'spline'" + ' keyword as its first argument\n'
                             'Input function keywords: {0}'.format(eval_args))
        if 'binning' not in eval_args[1]:
            raise ValueError('Evaluation function does not contain the ' +
                             "'binning'" + ' keyword as its second '
                             'argument\nInput function keywords: '
                             '{0}'.format(eval_args))
        self._eval_spl = eval_spl

        # Ensure validate_spl has correct structure
        validate_args = inspect.getargspec(validate_spl).args
        if len(validate_args) != 1:
            raise ValueError('Binning validation function contains more than '
                             'the maximum number of input parameters (1)\n'
                             'Input function keywords: '
                             '{0}'.format(validate_args))
        if'binning' not in validate_args[0]:
            raise ValueError('Binning validation function does not contain '
                             'the ' + "'binning'" + ' keyword argument\n'
                             'Input function keywords: {0}'.format(eval_args))
        self._validate_spl = validate_spl

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        assert isinstance(value, str)
        self._name = value

    @property
    def tex(self):
        return self._tex

    @tex.setter
    def tex(self, value):
        assert isinstance(value, str)
        self._tex = value

    @property
    def spline(self):
        return self._spline

    @property
    def hash(self):
        return self._hash

    @hash.setter
    def hash(self, value):
        """Hash must be an immutable type (i.e., have a __hash__ method)"""
        assert hasattr(value, '__hash__')
        self._hash = value

    def get_map(self, binning, **kwargs):
        """Return a map of the spline evaluated at the centers of the
        given binning.
        """
        if not isinstance(binning, MultiDimBinning):
            if isinstance(binning, Sequence):
                binning = MultiDimBinning(dimensions=binning)
            elif isinstance(binning, Mapping):
                binning = MultiDimBinning(**binning)
            else:
                raise ValueError('Do not know what to do with `binning`=%s of'
                                 ' type %s' % (binning, type(binning)))
        if self._validate_spl is not None:
            self._validate_spl(binning)
        return self._eval_spl(self.spline, binning, name=self.name, **kwargs)

    @profile
    def get_integrated_map(self, binning, bw_units=None, **kwargs):
        """Get the spline map integrated over the input binning values
        in output units specified by `bw_units`.
        """
        spline_map = self.get_map(binning, **kwargs)

        binning = binning.to(**bw_units)
        bin_widths = binning.bin_volumes(attach_units=False)

        return spline_map * bin_widths

    def __hash__(self):
        if self.hash is not None:
            return self.hash
        raise ValueError('No hash defined.')


class CombinedSpline(flavInt.FlavIntData):
    """Contained class for operating on Spline objects for various neutrino
    flavours.

    Inherits from FlavIntData object. Provides methods to allow
    evaluation of the splines for all neutrino flavours.

    Parameters
    --------
    inSpline : Spline or tuple of Spline
        Spline objects with `name` entry corresponding to a neutrino flavour
        `nue`, `numu`, `nuebar`, `numubar` and also corresponding to an
        interaction type `cc` and `nc` if the flag `interactions` is True.

    interactions: Bool
        Default = True
        Flag to specifiy whether to store flavours or flavour+interaction
        signatures.

    """
    def __init__(self, inSpline, interactions=True, ver=None):
        super().__init__()
        self.interactions = interactions

        if isinstance(inSpline, Spline):
            inSpline = [inSpline]
        if not all(isinstance(x, Spline) for x in inSpline):
            raise TypeError('Argument/object unhandled type: '
                            '{0}'.format(type(inSpline)))

        if interactions:
            self._spline_dict = {flavInt.NuFlavInt(flavint.name): flavint
                                 for flavint in inSpline}
            self._spline_data = {flavInt.NuFlavInt(flavint.name): None
                                 for flavint in inSpline}
        else:
            self._spline_dict = {flavInt.NuFlav(flav.name): flav
                                 for flav in inSpline}
            self._spline_data = {flavInt.NuFlav(flav.name): None
                                 for flav in inSpline}
        self._update_data_dict()

    def return_mapset(self, **kwargs):
        """Return a MapSet of stored spline maps."""
        for signature in self._spline_data.keys():
            if not isinstance(self._spline_data[signature], Map):
                raise ValueError('Error: map {0} has not yet been '
                                 'computed'.format(signature))
        maps = [self._spline_data[signature]
                for signature in self._spline_data.keys()]
        return MapSet(maps=maps, **kwargs)

    def get_spline(self, signature, centers, **kwargs):
        """Return the spline of a given signature and bins."""
        signature = self._validate_NuFlav(signature)
        return self._spline_dict[signature].get_spline(centers, **kwargs)

    def get_map(self, signature, binning, **kwargs):
        """Return a map of spline values for a given signature and
        binning.
        """
        signature = self._validate_NuFlav(signature)
        return self._spline_dict[signature].get_map(binning, **kwargs)

    def get_integrated_map(self, signature, binning, **kwargs):
        """Return a map of spline values for a given signature integrated
        over the input binning.
        """
        signature = self._validate_NuFlav(signature)
        return self._spline_dict[signature].get_integrated_map(
            binning, **kwargs
        )

    def compute_maps(self, binning, **kwargs):
        """Compute the map of spline values for a given signature and binning,
        then store it internally.
        """
        for signature in self._spline_data.keys():
            self._spline_data[signature] = self.get_map(
                signature, binning, **kwargs
            )
        self._update_data_dict()

    def compute_integrated_maps(self, binning, **kwargs):
        """Compute the map of spline values for a given signature integrated
        over the input binning, then store it internally.
        """
        for signature in self._spline_data.keys():
            self._spline_data[signature] = self.get_integrated_map(
                signature, binning, **kwargs
            )
        self._update_data_dict()

    def scale_map(self, signature, value):
        """Scale a specific spline map by an input value."""
        signature = self._validate_signature(signature)
        if not isinstance(self._spline_data[signature], Map):
            raise ValueError('Error: maps have not yet been computed')
        self._spline_data[signature] *= value
        self._update_data_dict()

    def scale_maps(self, value):
        """Scale the stored spline maps by an input value."""
        for signature in self._spline_data.keys():
            self._spline_data[signature] *= value
        self._update_data_dict()

    def reset(self):
        """Reset the flux maps to the original input maps."""
        for signature in self._spline_data.keys():
            self._spline_data[signature] = None
        self._update_data_dict()

    # TODO(shivesh): too slow!
    @staticmethod
    def validate_spline(spline):
        """Validate spline data."""
        return
        # spline = flavInt.FlavIntData(spline)
        # for k in flavInt.ALL_NUFLAVINTS:
        #     f = spline[k]
        #     if f is not None:
        #         assert np.sum(np.isnan(f.hist)) == 0
        # for k in flavInt.ALL_NUFLAVS:
        #     f = spline[k]
        #     if f is not None:
        #         assert np.all(f['nc'] == f['cc'])

    def _update_data_dict(self):
        assert set(self._spline_dict.keys()) == set(self._spline_data.keys())
        with flavInt.BarSep('_'):
            spline = {str(f): {str(it): None for it in flavInt.ALL_NUINT_TYPES}
                      for f in flavInt.ALL_NUFLAVS}
            for x in self._spline_data.keys():
                for y in flavInt.ALL_NUINT_TYPES:
                    if self.interactions:
                        spline[str(flavInt.NuFlav(x))][str(y)] = \
                                self._spline_data[x]
                    else:
                        spline[str(x)][str(y)] = self._spline_data[x]
        super().validate(spline)
        self.validate_spline(spline)
        self.update(spline)

    def __getattr__(self, attr):
        try:
            if self.interactions:
                sign = str(flavInt.NuFlavInt(attr))
            else:
                sign = str(flavInt.NuFlav(attr))
        except:
            raise ValueError('{0} is not a value signature'.format(attr))
        for signature in self._spline_data.keys():
            if self._spline_data[signature].name == sign:
                return self._spline_data[signature]
        return super().__getattribute__(sign)

    def _validate_NuFlav(self, signature):
        if self.interactions:
            if not isinstance(signature, flavInt.NuFlavInt):
                signature = flavInt.NuFlavInt(signature)
        else:
            if not isinstance(signature, flavInt.NuFlav):
                signature = flavInt.NuFlav(signature)
        if signature not in self._spline_dict:
            raise ValueError('signature {0} not loaded, choices are: '
                             '{1}'.format(signature, self._spline_dict.keys()))
        return signature

    def __add__(self, spline):
        if isinstance(spline, Spline):
            inSpline = self._spline_dict.values() + [spline]
            return CombinedSpline(inSpline)
        elif isinstance(spline, CombinedSpline):
            inSpline = self._spline_dict.values() + \
                spline._spline_dict.values()
            return CombinedSpline(inSpline)
        else:
            raise TypeError('Argument/object unhandled type: '
                            '{0}'.format(type(spline)))


def test_Spline():
    # TODO(shivesh): tests
    pass


if __name__ == '__main__':
    from pisa.utils.log import set_verbosity
    set_verbosity(3)
    test_Spline()
