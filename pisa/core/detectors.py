#! /usr/bin/env python
"""
Detector class definition and a simple script to generate, save, and
plot distributions for different detectors from pipeline config file(s).

"""

from __future__ import absolute_import

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
import inspect
from itertools import izip, product
import os

import numpy as np

from pisa import ureg
from pisa.core.map import MapSet
from pisa.core.pipeline import Pipeline
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.param import ParamSet
from pisa.utils.config_parser import PISAConfigParser
from pisa.utils.fileio import expand, mkdir, to_file
from pisa.utils.hash import hash_obj
from pisa.utils.log import set_verbosity, logging
from pisa.utils.random_numbers import get_random_state


__all__ = ['Detectors', 'test_Detectors', 'parse_args', 'main']


class Detectors(object):
    """Container for one or more distribution makers, that belong to different detectors.

    Parameters
    ----------
    pipelines : Pipeline or convertible thereto, or iterable thereof
        A new pipline is instantiated with each object passed. Legal objects
        are already-instantiated Pipelines and anything interpret-able by the
        Pipeline init method.

    Notes
    -----
    Free params with the same name in two pipelines are updated at the same
    time so long as you use the `update_params`, `set_free_params`, or
    `_set_rescaled_free_params` methods. Also use `select_params` to select
    params across all pipelines (if a pipeline does not have one or more of
    the param selectors specified, those param selectors have no effect in
    that pipeline).

    `_*_rescaled_*` properties and methods are for interfacing with a
    minimizer, where values are linearly mapped onto the interval [0, 1]
    according to the parameter's allowed range. Avoid interfacing with these
    except if using a minimizer, since, e.g., units are stripped and values and
    intervals are non-physical.

    """
    def __init__(self, distribution_makers, label=None):
        self.label = None
        self._source_code_hash = None

        self._distribution_makers = []
        if isinstance(distribution_makers, (basestring, PISAConfigParser, OrderedDict,
                                  DistributionMaker)):
            distribution_makers = [distribution_makers]

        for distribution_maker in distribution_makers:
            if not isinstance(distribution_maker, DistributionMaker):
                distribution_maker = DistributionMaker(distribution_maker)
            self._distribution_makers.append(distribution_maker)
        #for distribution_maker in self:
        #    distribution_maker.select_params(self.param_selections,
        #                           error_on_missing=False)

    def __iter__(self):
        return iter(self._distribution_makers)

    def get_outputs(self, return_sum=False, **kwargs):
        """Compute and return the outputs.

        Parameters
        ----------
        return_sum : bool
            If True, add up all Maps in all MapSets returned by all pipelines.
            The result will be a single Map contained in a MapSet.
            If False, return a list where each element is the full MapSet
            returned by each pipeline in the DistributionMaker.

        **kwargs
            Passed on to each pipeline's `get_outputs1` method.

        Returns
        -------
        MapSet if `return_sum=True` or list of MapSets if `return_sum=False`

        """
        outputs = [distribution_maker.get_outputs(return_sum=return_sum,**kwargs) for distribution_maker in self]
        return outputs

    def update_params(self, params):
        for distribution_maker in self:
            distribution_maker.update_params(params)

    def select_params(self, selections, error_on_missing=True):
        successes = 0
        if selections is not None:
            for distribution_maker in self:
                try:
                    distribution_maker.select_params(selections, error_on_missing=True)
                except KeyError:
                    pass
                else:
                    successes += 1

            if error_on_missing and successes == 0:
                raise KeyError(
                    'None of the stages from any pipeline in any distribution'
                    ' maker has all of the selections %s available.'
                    %(selections,)
                )
        else:
            for distribution_maker in self:
                possible_selections = distribution_maker.param_selections
                if possible_selections:
                    logging.warn("Although you didn't make a parameter "
                                 "selection, the following were available: %s."
                                 " This may cause issues.",
                                 possible_selections)

    @property
    def distribution_makers(self):
        return self._distribution_makers

    @property
    def params(self):
        params = ParamSet()
        for distribution_maker in self:
            params.extend(distribution_maker.params)
        return params

    @property
    def param_selections(self):
        selections = set()
        for distribution_maker in self:
            selections.update(distribution_maker.param_selections)
        return sorted(selections)

    @property
    def source_code_hash(self):
        """Hash for the source code of this object's class.

        Not meant to be perfect, but should suffice for tracking provenance of
        an object stored to disk that were produced by a Stage.
        """
        if self._source_code_hash is None:
            self._source_code_hash = hash_obj(inspect.getsource(self.__class__))
        return self._source_code_hash

    @property
    def hash(self):
        return hash_obj([self.source_code_hash] + [p.hash for p in self])

    def set_free_params(self, values):
        """Set free parameters' values.

        Parameters
        ----------
        values : list of quantities

        """
        for name, value in izip(self.params.free.names, values):
            for distribution_maker in self:
                if name in distribution_maker.params.free.names:
                    distribution_maker.params[name] = value
                elif name in distribution_maker.params.names:
                    raise AttributeError(
                        'Trying to set value for "%s", a parameter that is'
                        ' fixed in at least one pipeline of' %name
                    )

    def randomize_free_params(self, random_state=None):
        if random_state is None:
            random = np.random
        else:
            random = get_random_state(random_state)
        n = len(self.params.free)
        rand = random.rand(n)
        self._set_rescaled_free_params(rand)

    def reset_all(self):
        """Reset both free and fixed parameters to their nominal values."""
        for d in self:
            d.params.reset_all()

    def reset_free(self):
        """Reset only free parameters to their nominal values."""
        for d in self:
            d.params.reset_free()

    def set_nominal_by_current_values(self):
        """Define the nominal values as the parameters' current values."""
        for d in self:
            d.params.set_nominal_by_current_values()

    def _set_rescaled_free_params(self, rvalues):
        """Set free param values given a simple list of [0,1]-rescaled,
        dimensionless values

        """
        names = self.params.free.names
        for distribution_maker in self:
            for name, rvalue in izip(names, rvalues):
                if name in distribution_maker.params.free.names:
                    distribution_maker.params[name]._rescaled_value = rvalue
                elif name in distribution_maker.params.names:
                    raise AttributeError(
                        'Trying to set value for "%s", a parameter that is'
                        ' fixed in at least one pipeline of' %name
                    )


def test_Detectors():
    """Unit tests for Detectors"""
    #
    # Test: select_params and param_selections
    #

    hierarchies = ['nh', 'ih']
    materials = ['iron', 'pyrolite']

    t23 = dict(
        ih=49.5 * ureg.deg,
        nh=42.3 * ureg.deg
    )
    YeO = dict(
        iron=0.4656,
        pyrolite=0.4957
    )

    # Instantiate with two pipelines: first has both nh/ih and iron/pyrolite
    # param selectors, while the second only has nh/ih param selectors.
    dm1 = DistributionMaker(['tests/settings/test_Pipeline.cfg',
                            'tests/settings/test_Pipeline2.cfg'])
    dm2 = DistributionMaker(['tests/settings/test_Pipeline.cfg',
                            'tests/settings/test_Pipeline2.cfg'])
    det = Detectors(dm1,dm2)

    current_mat = 'pyrolite'
    current_hier = 'nh'

    for new_hier, new_mat in product(hierarchies, materials):
        new_YeO = YeO[new_mat]

        assert det.param_selections == sorted([current_hier, current_mat]), \
                str(det.params.param_selections)
        assert det.params.theta23.value == t23[current_hier], \
                str(det.params.theta23)
        assert det.params.YeO.value == YeO[current_mat], str(det.params.YeO)

        # Select just the hierarchy
        det.select_params(new_hier)
        assert det.param_selections == sorted([new_hier, current_mat]), \
                str(det.param_selections)
        assert det.params.theta23.value == t23[new_hier], \
                str(det.params.theta23)
        assert det.params.YeO.value == YeO[current_mat], \
                str(det.params.YeO)

        # Select just the material
        det.select_params(new_mat)
        assert det.param_selections == sorted([new_hier, new_mat]), \
                str(det.param_selections)
        assert det.params.theta23.value == t23[new_hier], \
                str(det.params.theta23)
        assert det.params.YeO.value == YeO[new_mat], \
                str(det.params.YeO)

        # Reset both to "current"
        det.select_params([current_mat, current_hier])
        assert det.param_selections == sorted([current_hier, current_mat]), \
                str(det.param_selections)
        assert det.params.theta23.value == t23[current_hier], \
                str(det.params.theta23)
        assert det.params.YeO.value == YeO[current_mat], \
                str(det.params.YeO)

        # Select both hierarchy and material
        det.select_params([new_mat, new_hier])
        assert det.param_selections == sorted([new_hier, new_mat]), \
                str(det.param_selections)
        assert det.params.theta23.value == t23[new_hier], \
                str(det.params.theta23)
        assert det.params.YeO.value == YeO[new_mat], \
                str(det.params.YeO)

        current_hier = new_hier
        current_mat = new_mat


def parse_args():
    """Get command line arguments"""
    parser = ArgumentParser(
        description='''Generate, store, and plot distributions from different
        pipeline configuration file(s) for one or more detectors.''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-p', '--pipeline', type=str, required=True,
        metavar='CONFIGFILE', action='append',
        help='''Settings file for each pipeline (repeat for multiple).'''
    )
    parser.add_argument(
        '--select', metavar='PARAM_SELECTIONS', nargs='+', default=None,
        help='''Param selectors (separated by spaces) to use to override any
        defaults in the config file.'''
    )
    parser.add_argument(
        '--return-sum', action='store_true',
        help='''Return a sum of the MapSets output by the distribution maker's
        pipelines as a single map (as opposed to a list of MapSets, one per
        pipeline)'''
    )
    parser.add_argument(
        '--outdir', type=str, action='store',
        help='Directory into which to store the output'
    )
    parser.add_argument(
        '--pdf', action='store_true',
        help='''Produce pdf plot(s).'''
    )
    parser.add_argument(
        '--png', action='store_true',
        help='''Produce png plot(s).'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='Set verbosity level'
    )
    args = parser.parse_args()
    return args


def main(return_outputs=False):
    """Main; call as script with `return_outputs=False` or interactively with
    `return_outputs=True`"""
    from pisa.utils.plotter import Plotter
    args = parse_args()
    set_verbosity(args.v)
    plot_formats = []
    if args.pdf:
        plot_formats.append('pdf')
    if args.png:
        plot_formats.append('png')
        
    pipelines = args.pipeline
    distribution_makers , Names = [] , []
    for pipeline in pipelines:
        name = Pipeline(pipeline)._detector_name
        if name in Names:
            distribution_makers[Names.index(name)].append(pipeline)
        else:
            distribution_makers.append([pipeline])
            Names.append(name)
    
    if None in Names and len(Names) > 1:
        raise NameError('One of the pipelines has no detector_name.')

    for i in range(len(distribution_makers)):
        distribution_makers[i] = DistributionMaker(pipelines=distribution_makers[i])
    detectors = Detectors(distribution_makers=distribution_makers)
        
    if args.select is not None:
        detectors.select_params(args.select)

    outputs = detectors.get_outputs(return_sum=args.return_sum)
    if args.outdir:
        # TODO: unique filename: append hash (or hash per pipeline config)
        fname = 'detectors_outputs.json.bz2'
        mkdir(args.outdir)
        fpath = expand(os.path.join(args.outdir, fname))
        to_file(outputs, fpath)

    if args.outdir and plot_formats:
        my_plotter = Plotter(
            outdir=args.outdir,
            fmt=plot_formats, log=False,
            annotate=False
        )
        for num, output in enumerate(outputs):
            if args.return_sum:
                my_plotter.plot_2d_array(
                    output,
                    fname=Names[num]
                )
            else:
                for out in output:
                    my_plotter.plot_2d_array(
                        out,
                        fname=Names[num]
                    )

    if return_outputs:
        return detectors, outputs


if __name__ == '__main__':
    detectors, outputs = main(return_outputs=True)
