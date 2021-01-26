#! /usr/bin/env python

"""
DistributionMaker class definition and a simple script to generate, save, and
plot a distribution from pipeline config file(s).
"""

from __future__ import absolute_import

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
from collections.abc import Mapping
import inspect
from itertools import product
import os
from tabulate import tabulate

import numpy as np

from pisa import ureg
from pisa.core.map import MapSet
from pisa.core.pipeline import Pipeline
from pisa.core.param import ParamSet
from pisa.utils.config_parser import PISAConfigParser
from pisa.utils.fileio import expand, mkdir, to_file
from pisa.utils.hash import hash_obj
from pisa.utils.log import set_verbosity, logging
from pisa.utils.random_numbers import get_random_state


__all__ = ['DistributionMaker', 'test_DistributionMaker', 'parse_args', 'main']

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


class DistributionMaker(object):
    """Container for one or more pipelines; the outputs from all contained
    pipelines are added together to create the distribution.

    Parameters
    ----------
    pipelines : Pipeline or convertible thereto, or iterable thereof
        A new pipline is instantiated with each object passed. Legal objects
        are already-instantiated Pipelines and anything interpret-able by the
        Pipeline init method.

    label : str or None, optional
        A label for the DistributionMaker.

    set_livetime_from_data : bool, optional
        If a (data) pipeline is found with the attr `metadata` and which has
        the contained key "livetime", this livetime is used to set the livetime
        on all pipelines which have param `params.livetime`. If multiple such
        data pipelines are found and `set_livetime_from_data` is True, all are
        checked for consistency (you should use multiple `Detector`s if you
        have incompatible data sets).

    profile : bool
        timing of inidividual pipelines / stages

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
    def __init__(self, pipelines, label=None, set_livetime_from_data=True, profile=False):

        self.label = label
        self._source_code_hash = None
        self.metadata = OrderedDict()

        self._profile = profile

        self._pipelines = []
        if isinstance(pipelines, (str, PISAConfigParser, OrderedDict,
                                  Pipeline)):
            pipelines = [pipelines]

        for pipeline in pipelines:
            if not isinstance(pipeline, Pipeline):
                pipeline = Pipeline(pipeline, profile=profile)
            self._pipelines.append(pipeline)

        data_run_livetime = None
        if set_livetime_from_data:
            #
            # Get livetime metadata if defined in any stage in any pipeline
            #
            for pipeline_idx, pipeline in enumerate(self):
                for stage_idx, stage in enumerate(pipeline):
                    if not (
                        hasattr(stage, "metadata")
                        and isinstance(stage.metadata, Mapping)
                        and "livetime" in stage.metadata
                    ):
                        continue

                    if data_run_livetime is None:
                        data_run_livetime = stage.metadata["livetime"]

                    if stage.metadata["livetime"] != data_run_livetime:
                        raise ValueError(
                            "Pipeline index {}, stage index {} has data"
                            " livetime = {}, in disagreement with"
                            " previously-found livetime = {}".format(
                                pipeline_idx,
                                stage_idx,
                                stage.metadata["livetime"],
                                data_run_livetime,
                            )
                        )

            # Save the last livetime found inside the pipeline's metadata
            # TODO: implement metadata in the pipeline class instead
            self.metadata['livetime'] = data_run_livetime
            #
            # Set param `params.livetime` for any pipelines that have it
            #
            if data_run_livetime is not None:

                data_run_livetime *= ureg.sec

                for pipeline_idx, pipeline in enumerate(self):

                    if "livetime" not in pipeline.params.names:
                        continue

                    pipeline.params.livetime.is_fixed = True

                    if pipeline.params.livetime != data_run_livetime:

                        logging.warning(
                            "Pipeline index %d has params.livetime = %s, in"
                            " disagreement with data livetime = %s defined by"
                            " data. The pipeline's livetime param will be"
                            " reset to the latter value and set to be fixed"
                            " (if it is not alredy).",
                            pipeline_idx,
                            pipeline.params.livetime.value,
                            data_run_livetime,
                        )
                        pipeline.params.livetime = data_run_livetime


        #for pipeline in self:
        #    pipeline.select_params(self.param_selections,
        #                           error_on_missing=False)

        # Make sure that all the pipelines have the same detector name (or None)
        self.detector_name = 'no_name'
        for p in self._pipelines:
            name = p.detector_name
            if name != self.detector_name and self.detector_name != 'no_name':
                raise NameError(
                    'Different detector names in distribution_maker pipelines'
                )

            self.detector_name = name

        # set parameters with an identical name to the same object
        # otherwise we get inconsistent behaviour when setting repeated params
        # See Isues #566 and #648
        all_parans = self.params
        for pipeline in self:
            pipeline.update_params(all_parans, existing_must_match=True, extend=False)

    def __repr__(self):
        return self.tabulate(tablefmt="presto")

    def _repr_html_(self):
        return self.tabulate(tablefmt="html")

    def tabulate(self, tablefmt="plain"):
        headers = ['pipeline number', 'name', 'detector name', 'output_binning', 'output_key', 'profile']
        colalign=["right"] + ["center"] * (len(headers) -1 )
        table = []
        for i, p in enumerate(self.pipelines):
            table.append([i, p.name, p.detector_name, p.output_binning, p.output_key, p.profile])
        return tabulate(table, headers, tablefmt=tablefmt, colalign=colalign)

    def __iter__(self):
        return iter(self._pipelines)

    @property
    def profile(self):
        return self._profile

    @profile.setter
    def profile(self, value):
        for pipeline in self.pipelines:
            pipeline.profile = value
        self._profile = value


    def run(self):
        for pipeline in self:
            pipeline.run()

    def setup(self):
        """Setup (reset) all pipelines"""
        for p in self:
            p.setup()

    def get_outputs(self, return_sum=False, sum_map_name='total',
                    sum_map_tex_name='Total', **kwargs):
        """Compute and return the outputs.

        Parameters
        ----------
        return_sum : bool
            If True, add up all Maps in all MapSets returned by all pipelines.
            The result will be a single Map contained in a MapSet.
            If False, return a list where each element is the full MapSet
            returned by each pipeline in the DistributionMaker.


        **kwargs
            Passed on to each pipeline's `get_outputs` method.

        Returns
        -------
        MapSet if `return_sum=True` or list of MapSets if `return_sum=False`

        """

        outputs = [pipeline.get_outputs(**kwargs) for pipeline in self] # pylint: disable=redefined-outer-name

        if return_sum:
            
            # Case where the output of a pipeline is a mapSet
            if isinstance(outputs[0], MapSet):
                outputs = sum([sum(x) for x in outputs]) # This produces a Map
                outputs.name = sum_map_name
                outputs.tex = sum_map_tex_name
                outputs = MapSet(outputs) # final output must be a MapSet

            # Case where the output of a pipeline is a dict of different MapSets
            elif isinstance(outputs[0], OrderedDict):
                output_dict = OrderedDict()
                for key in outputs[0].keys():
                    output_dict[key] = sum([sum(A[key]) for A in outputs]) # This produces a Map objects
                    output_dict[key].name = sum_map_name
                    output_dict[key].tex = sum_map_tex_name
                    output_dict[key] = MapSet(output_dict[key])

                outputs = output_dict

        return outputs

    def update_params(self, params):
        for pipeline in self:
            pipeline.update_params(params)

    def select_params(self, selections, error_on_missing=True):
        successes = 0
        if selections is not None:
            for pipeline in self:
                try:
                    pipeline.select_params(selections, error_on_missing=True)
                except KeyError:
                    pass
                else:
                    successes += 1

            if error_on_missing and successes == 0:
                raise KeyError(
                    'None of the stages from any pipeline in this distribution'
                    ' maker has all of the selections %s available.'
                    %(selections,)
                )
        else:
            for pipeline in self:
                possible_selections = pipeline.param_selections
                if possible_selections:
                    logging.warning(
                        "Although you didn't make a parameter "
                        "selection, the following were available: %s."
                        " This may cause issues.", possible_selections
                    )

    @property
    def pipelines(self):
        return self._pipelines

    @property
    def params(self):
        params = ParamSet()
        for pipeline in self:
            params.extend(pipeline.params)
        return params

    @property
    def param_selections(self):
        selections = set()
        for pipeline in self:
            selections.update(pipeline.param_selections)
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

    @property
    def num_events_per_bin(self):
        '''
        returns an array of bin indices where none of the 
        pipelines have MC events

        assumes that all pipelines have the same binning output specs

        number of events is taken out of the last stage of the pipeline
        '''
        num_bins = self.pipelines[0].stages[-1].output_specs.tot_num_bins
        num_events_per_bin = np.zeros(num_bins)

        for p in self.pipelines:
            assert p.stages[-1].output_specs.tot_num_bins==num_bins, 'ERROR: different pipelines have different binning'

            for c in p.stages[-1].data:
                for index in range(num_bins):
                    index_mask = c.array_data['bin_{}_mask'.format(index)].get('host')
                    current_weights = c.array_data['weights'].get('host')[index_mask]
                    n_weights = current_weights.shape[0]
                    num_events_per_bin[index] += n_weights

        return num_events_per_bin
    

    @property
    def empty_bin_indices(self):
        '''Find indices where there are no events present
        '''
        empty_counts = self.num_events_per_bin == 0
        indices = np.where(empty_counts)[0]
        return indices
    

    def set_free_params(self, values):
        """Set free parameters' values.

        Parameters
        ----------
        values : list of quantities

        """
        for name, value in zip(self.params.free.names, values):
            for pipeline in self:
                if name in pipeline.params.free.names:
                    pipeline.params[name] = value
                elif name in pipeline.params.names:
                    raise AttributeError(
                        'Trying to set value for "%s", a parameter that is'
                        ' fixed in at least one pipeline' %name
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
        for p in self:
            p.params.reset_all()

    def reset_free(self):
        """Reset only free parameters to their nominal values."""
        for p in self:
            p.params.reset_free()

    def set_nominal_by_current_values(self):
        """Define the nominal values as the parameters' current values."""
        for p in self:
            p.params.set_nominal_by_current_values()

    def _set_rescaled_free_params(self, rvalues):
        """Set free param values given a simple list of [0,1]-rescaled,
        dimensionless values

        """
        names = self.params.free.names
        for pipeline in self:
            for name, rvalue in zip(names, rvalues):
                if name in pipeline.params.free.names:
                    pipeline.params[name]._rescaled_value = rvalue # pylint: disable=protected-access
                elif name in pipeline.params.names:
                    raise AttributeError(
                        'Trying to set value for "%s", a parameter that is'
                        ' fixed in at least one pipeline' %name
                    )


def test_DistributionMaker():
    """Unit tests for DistributionMaker"""
    #
    # Test: select_params and param_selections
    #

    # TODO: make test config file with materials param selector, then uncomment
    # removed tests below

    hierarchies = ['nh', 'ih']
    #materials = ['iron', 'pyrolite']
    materials = []

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
    dm = DistributionMaker(
        ['settings/pipeline/example.cfg', 'settings/pipeline/example.cfg']
    )

    #current_mat = 'iron'
    current_hier = 'nh'

    for new_hier, new_mat in product(hierarchies, materials):
        #assert dm.param_selections == sorted([current_hier, current_mat]), \
        #        str(dm.param_selections)
        assert dm.param_selections == [current_hier], str(dm.param_selections)
        assert dm.params.theta23.value == t23[current_hier], str(dm.params.theta23)
        #assert dm.params.YeO.value == YeO[current_mat], str(dm.params.YeO)

        # Select just the hierarchy
        dm.select_params(new_hier)
        #assert dm.param_selections == sorted([new_hier, current_mat]), \
        #        str(dm.param_selections)
        assert dm.param_selections == [new_hier], str(dm.param_selections)
        assert dm.params.theta23.value == t23[new_hier], str(dm.params.theta23)
        #assert dm.params.YeO.value == YeO[current_mat], str(dm.params.YeO)

        ## Select just the material
        #dm.select_params(new_mat)
        #assert dm.param_selections == sorted([new_hier, new_mat]), \
        #        str(dm.param_selections)
        #assert dm.params.theta23.value == t23[new_hier], \
        #        str(dm.params.theta23)
        #assert dm.params.YeO.value == YeO[new_mat], \
        #        str(dm.params.YeO)

        # Reset both to "current"
        #dm.select_params([current_mat, current_hier])
        dm.select_params(current_hier)
        #assert dm.param_selections == sorted([current_hier, current_mat]), \
        #        str(dm.param_selections)
        assert dm.param_selections == [current_hier], str(dm.param_selections)
        assert dm.params.theta23.value == t23[current_hier], str(dm.params.theta23)
        #assert dm.params.YeO.value == YeO[current_mat], str(dm.params.YeO)

        ## Select both hierarchy and material
        #dm.select_params([new_mat, new_hier])
        #assert dm.param_selections == sorted([new_hier, new_mat]), \
        #        str(dm.param_selections)
        #assert dm.params.theta23.value == t23[new_hier], \
        #        str(dm.params.theta23)
        #assert dm.params.YeO.value == YeO[new_mat], \
        #        str(dm.params.YeO)

        #current_hier = new_hier
        #current_mat = new_mat


def parse_args():
    """Get command line arguments"""
    parser = ArgumentParser(
        description='''Generate, store, and plot a distribution from pipeline
        configuration file(s).''',
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

    distribution_maker = DistributionMaker(pipelines=args.pipeline) # pylint: disable=redefined-outer-name
    if args.select is not None:
        distribution_maker.select_params(args.select)

    outputs = distribution_maker.get_outputs(return_sum=args.return_sum) # pylint: disable=redefined-outer-name
    if args.outdir:
        # TODO: unique filename: append hash (or hash per pipeline config)
        fname = 'distribution_maker_outputs.json.bz2'
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
            my_plotter.plot_2d_array(
                output,
                fname='dist_output_%d' % num
            )

    if return_outputs:
        return distribution_maker, outputs


if __name__ == '__main__':
    distribution_maker, outputs = main(return_outputs=True) # pylint: disable=invalid-name
