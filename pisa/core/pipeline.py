#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016
"""
Implementation of the Pipeline object, and a simple script to instantiate and
run a pipeline (the outputs of which can be plotted and stored to disk).

"""


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
from importlib import import_module
from itertools import product
from inspect import getsource
import os

import numpy as np

from pisa import ureg
from pisa.core.events import Data
from pisa.core.map import Map, MapSet
from pisa.core.param import ParamSet
from pisa.core.stage import Stage
from pisa.core.transform import TransformSet
from pisa.utils.config_parser import parse_pipeline_config
from pisa.utils.betterConfigParser import BetterConfigParser
from pisa.utils.fileio import mkdir
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import profile


__all__ = ['Pipeline',
           'test_Pipeline', 'parse_args', 'main']


# TODO: should we check that the output binning of a previous stage produces
# the inputs required by the current stage, or that the aggregate outputs that
# got produced by previous stages (less those that got consumed in other
# previous stages) hold what the current stage requires for inputs... or
# should we not assume either will check out, since it's possible that the
# stage requires sideband objects that are to be introduced at the top of the
# pipeline by the user (and so there's no way to verify that all inputs are
# present until we see what the user hands the pipeline as its top-level
# input)? Alternatively, the lack of apparent inputs for a stage could show
# a warning message. Or we just wait to see if it fails when the user runs the
# code.

# TODO: return an OrderedDict instead of a list if the user requests
# intermediate results? Or simply use the `outputs` attribute of each stage to
# dynamically access this?

class Pipeline(object):
    """Instantiate stages according to a parsed config object; excecute
    stages.

    Parameters
    ----------
    config : string, OrderedDict, or BetterConfigParser
        If string, interpret as resource location; send to the
          config_parser.parse_pipeline_config() function to get a config
          OrderedDict.
        If OrderedDict, use directly as pipeline configuration.

    """
    def __init__(self, config):
        self._stages = []
        if isinstance(config, (basestring, BetterConfigParser)):
            config = parse_pipeline_config(config=config)
        assert isinstance(config, OrderedDict)
        self._config = config
        self._init_stages()
        self._source_code_hash = None

    def index(self, stage_id):
        """Return the index in the pipeline of `stage_id`.

        Parameters
        ----------
        stage_id : string or int
            Name of the stage, or stage number (0-indexed)

        Returns
        -------
        idx : integer stage number (0-indexed)

        Raises
        ------
        ValueError : if `stage_id` not in pipeline.

        """
        assert isinstance(stage_id, (int, basestring))
        idx = None
        for stage_num, stage in enumerate(self):
            if stage_id in [stage_num, stage.stage_name]:
                return stage_num
        raise ValueError('No stage named "%s".' %stage_name)

    def __len__(self):
        return len(self._stages)

    def __iter__(self):
        return iter(self._stages)

    def __getitem__(self, idx):
        if isinstance(idx, basestring):
            return self.stages[self.index(idx)]

        if isinstance(idx, (int, slice)):
            return self.stages[idx]

        raise ValueError('Cannot locate stage "%s" in pipeline. Stages'
                         ' available are %s.' %(idx, self.stage_names))

    def __getattr__(self, attr):
        for stage in self:
            if stage.stage_name == attr:
                return stage
        raise AttributeError('"%s" is not a stage in this pipeline or "%s" is'
                             ' a property of Pipeline that failed to execute.'
                             %(attr, attr))

    def _init_stages(self):
        """Stage factory: Instantiate stages specified by self.config.

        Conventions required for this to work:
            * Stage and service names must be lower-case
            * Service implementations must be found at Python path
              `pisa.stages.<stage_name>.<service_name>`
            * `service` cannot be an instantiation argument for a service

        """
        self._stages = []
        for stage_num, ((stage_name, service_name), settings) \
                in enumerate(self.config.items()):
            try:
                logging.debug('instantiating stage %s / service %s'
                              %(stage_name, service_name))

                # Import service's module
                logging.trace('Importing: pisa.stages.%s.%s' %(stage_name,
                                                               service_name))
                module = import_module(
                    'pisa.stages.%s.%s' %(stage_name, service_name)
                )

                # Get service class from module
                cls = getattr(module, service_name)

                # Instantiate service
                service = cls(**settings)
                if not isinstance(service, Stage):
                    raise TypeError(
                        'Trying to create service "%s" for stage #%d (%s),'
                        ' but object %s instantiated from class %s is not a'
                        ' %s type but instead is of type %s.'
                        %(service_name, stage_num, stage_name, service, cls,
                          Stage, type(service))
                    )

                # Append service to pipeline
                self._stages.append(service)

            except:
                logging.error(
                    'Failed to initialize stage #%d (stage=%s, service=%s).'
                    %(stage_num, stage_name, service_name)
                )
                raise

        for stage in self:
            stage.select_params(self.param_selections, error_on_missing=False)

    @profile
    def get_outputs(self, inputs=None, idx=None,
                    return_intermediate=False):
        """Run the pipeline to compute its outputs.

        Parameters
        ----------
        inputs : None or MapSet # TODO: other container(s)
            Optional inputs to send to the first stage of the pipeline.
        idx : None, string, or int
            Specification of which stage(s) to run. If None is passed, all
            stages will be run. If a string is passed, all stages are run up to
            and including the named stage. If int is passed, all stages are run
            up to but *not* including `idx`. Numbering follows Python
            conventions (i.e., is 0-indexed).
        return_intermediate : bool
            If True,

        Returns
        -------
        outputs : list or MapSet
            MapSet output by final stage if `return_intermediate` is False, or
            list of MapSets output by each stage if `return_intermediate` is
            True.

        """
        intermediate = []
        i = 0
        if isinstance(idx, basestring):
            idx = self.stage_names.index(idx) + 1
        if idx is not None:
            if idx < 0:
                raise ValueError('Integer `idx` must be >= 0')
            idx += 1
        assert len(self) > 0
        for stage in self.stages[:idx]:
            logging.debug('>> Working on stage "%s" service "%s"'
                          %(stage.stage_name, stage.service_name))
            try:
                logging.trace('>>> BEGIN: get_outputs')
                outputs = stage.get_outputs(inputs=inputs)
                logging.trace('>>> END  : get_outputs')
            except:
                logging.error('Error occurred computing outputs in stage %s /'
                              ' service %s ...' %(stage.stage_name,
                                                  stage.service_name))
                raise

            logging.trace('outputs: %s' %(outputs,))

            if return_intermediate:
                intermediate.append(outputs)

            inputs = outputs

        if return_intermediate:
            return intermediate

        return outputs

    def update_params(self, params):
        [stage.params.update_existing(params) for stage in self]

    def select_params(self, selections, error_on_missing=False):
        successes = 0
        for stage in self:
            try:
                stage.select_params(selections, error_on_missing=True)
            except KeyError:
                pass
            else:
                successes += 1

        if error_on_missing and successes == 0:
            raise KeyError(
                'None of the selections %s was found in any stage in this'
                ' pipeline.' %(selections,)
            )

    @property
    def params(self):
        params = ParamSet()
        [params.extend(stage.params) for stage in self]
        return params

    @property
    def param_selections(self):
        selections = set()
        [selections.update(stage.param_selections) for stage in self]
        return sorted(selections)

    @property
    def stages(self):
        return [s for s in self]

    @property
    def stage_names(self):
        return [s.stage_name for s in self]

    @property
    def config(self):
        return self._config

    @property
    def source_code_hash(self):
        """Hash for the source code of this object's class.

        Not meant to be perfect, but should suffice for tracking provenance of
        an object stored to disk that were produced by a Stage.

        """
        if self._source_code_hash is None:
            self._source_code_hash = hash_obj(getsource(self.__class__))
        return self._source_code_hash

    @property
    def state_hash(self):
        return hash_obj([self.source_code_hash] + [s.state_hash for s in self])


def test_Pipeline():
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
    pipeline = Pipeline('tests/settings/test_Pipeline.cfg')

    current_mat = 'iron'
    current_hier = 'nh'

    for new_hier, new_mat in product(hierarchies, materials):
        new_YeO = YeO[new_mat]

        assert pipeline.param_selections == sorted([current_hier, current_mat]), str(pipeline.params.param_selections)
        assert pipeline.params.theta23.value == t23[current_hier], str(pipeline.params.theta23)
        assert pipeline.params.YeO.value == YeO[current_mat], str(pipeline.params.YeO)

        # Select just the hierarchy
        pipeline.select_params(new_hier)
        assert pipeline.param_selections == sorted([new_hier, current_mat]), str(pipeline.param_selections)
        assert pipeline.params.theta23.value == t23[new_hier], str(pipeline.params.theta23)
        assert pipeline.params.YeO.value == YeO[current_mat], str(pipeline.params.YeO)

        # Select just the material
        pipeline.select_params(new_mat)
        assert pipeline.param_selections == sorted([new_hier, new_mat]), str(pipeline.param_selections)
        assert pipeline.params.theta23.value == t23[new_hier], str(pipeline.params.theta23)
        assert pipeline.params.YeO.value == YeO[new_mat], str(pipeline.params.YeO)

        # Reset both to "current"
        pipeline.select_params([current_mat, current_hier])
        assert pipeline.param_selections == sorted([current_hier, current_mat]), str(pipeline.param_selections)
        assert pipeline.params.theta23.value == t23[current_hier], str(pipeline.params.theta23)
        assert pipeline.params.YeO.value == YeO[current_mat], str(pipeline.params.YeO)

        # Select both hierarchy and material
        pipeline.select_params([new_mat, new_hier])
        assert pipeline.param_selections == sorted([new_hier, new_mat]), str(pipeline.param_selections)
        assert pipeline.params.theta23.value == t23[new_hier], str(pipeline.params.theta23)
        assert pipeline.params.YeO.value == YeO[new_mat], str(pipeline.params.YeO)

        current_hier = new_hier
        current_mat = new_mat


def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='''Instantiate and run a pipeline from a config file.
        Optionally store the resulting distribution(s) and plot(s) to disk.'''
    )
    parser.add_argument(
        '--settings', metavar='CONFIGFILE', type=str,
        required=True,
        help='File containing settings for the pipeline.'
    )
    parser.add_argument(
        '--only-stage', metavar='STAGE', type=str,
        help='''Test stage: Instantiate a single stage in the pipeline
        specification and run it in isolation (as the sole stage in a
        pipeline). If it is a stage that requires inputs, these can be
        specified with the --infile argument, or else dummy stage input maps
        (numpy.ones(...), matching the input binning specification) are
        generated for testing purposes. See also --infile and --transformfile
        arguments.'''
    )
    parser.add_argument(
        '--stop-after-stage', metavar='STAGE', type=int,
        help='''Test stage: Instantiate a pipeline up to and including
        STAGE, but stop there.'''
    )
    parser.add_argument(
        '-d', '--dir', metavar='DIR', type=str,
        help='''Store all output files (data and plots) to this directory.
        Directory will be created (including missing parent directories) if it
        does not exist already. If no dir is provided, no outputs will be
        saved.'''
    )
    #parser.add_argument(
    #    '-o', '--outname', metavar='FILENAME', type=str,
    #    default='out.json',
    #    help='''Filename for storing output data.'''
    #)
    parser.add_argument(
        '--intermediate', action='store_true',
        help='''Store all intermediate outputs, not just the final stage's
        outputs.'''
    )
    parser.add_argument(
        '--transforms', action='store_true',
        help='''Store all transforms (for stages that use transforms).'''
    )
    parser.add_argument(
        '-i', '--inputs-file', metavar='FILE', type=str,
        help='''File from which to read inputs to be fed to the pipeline.'''
    )
    # TODO: optionally store the transform sets from each stage
    #parser.add_argument(
    #    '-T', '--transform-file', metavar='FILE', type=str,
    #    help='''File into which to store transform(s) from the pipeline.'''
    #)
    parser.add_argument(
        '--pdf', action='store_true',
        help='''Produce pdf plot(s).'''
    )
    parser.add_argument(
        '--png', action='store_true',
        help='''Produce png plot(s).'''
    )
    parser.add_argument(
        '--annotate', action='store_true',
        help='''Annotate plots with counts per bin'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    args = parser.parse_args()
    return args


def main():
    from pisa.utils.plotter import Plotter

    args = parse_args()
    set_verbosity(args.v)

    try:
        args.only_stage = int(args.only_stage)
    except (TypeError, ValueError):
        pass

    if args.dir:
        mkdir(args.dir)
    else:
        if args.pdf or args.png:
            raise ValueError('No --dir provided, so cannot save images.')

    # Instantiate the pipeline
    pipeline = Pipeline(args.settings)

    for run in xrange(1):
        logging.info('')
        logging.info('## STARTING RUN %d ............' % run)
        logging.info('')
        #pipeline.params.free.values = [p.value*1.01 for p in pipeline.params.free]
        if args.only_stage is None:
            stop_idx = args.stop_after_stage
            if isinstance(stop_idx, basestring):
                stop_idx = pipeline.index(stop_idx)
            if stop_idx is not None:
                stop_idx += 1
            indices = slice(0, stop_idx)
            outputs = pipeline.get_outputs(idx=args.stop_after_stage)
        else:
            assert args.stop_after_stage is None
            idx = pipeline.index(args.only_stage)
            stage = pipeline[idx]
            indices = slice(idx, idx+1)
            # create dummy inputs
            if hasattr(stage, 'input_binning'):
                logging.info('building dummy input')
                input_maps = []
                for name in stage.input_names:
                    hist = np.ones(stage.input_binning.shape)
                    input_maps.append(
                        Map(name=name, hist=hist, binning=stage.input_binning)
                    )
                inputs = MapSet(maps=input_maps, name='ones', hash=1)
            else:
                inputs = None
            outputs = stage.get_outputs(inputs=inputs)
        logging.info('')
        logging.info('## ............ finished RUN %d' % run)
        logging.info('')

    for stage in pipeline[indices]:
        if not args.dir:
            break
        stg_svc = stage.stage_name + '__' + stage.service_name
        fbase = os.path.join(args.dir, stg_svc)
        if args.intermediate or stage == pipeline[-1]:
            stage.outputs.to_json(fbase + '__output.json.bz2')
        if args.transforms and stage.use_transforms:
            stage.transforms.to_json(fbase + '__transforms.json.bz2')

        formats = OrderedDict(png=args.png, pdf=args.pdf)
        if isinstance(stage.outputs, Data):
            outputs = stage.outputs.histogram_set(
                binning=stage.output_binning,
                nu_weights_col='pisa_weight',
                mu_weights_col='pisa_weight',
                mapset_name=stg_svc,
                errors=True
            )
        elif isinstance(stage.outputs, (MapSet, TransformSet)):
            outputs = stage.outputs
        outputs_2d = []
        for output in outputs:
            if len(output.binning) == 2:
                outputs_2d.append(output)
            elif len(output.binning) == 3:
                if 'pid' in output.binning.names:
                    logging.warn("Script is set up to only plot 2D maps. Your "
                                 "outputs are %iD. These will be reduced to "
                                 "multiple 2D maps for the PID dimension."
                                 %len(output.binning))
                    pid_names = output.binning['pid'].bin_names
                    if pid_names is None:
                        logging.warn("There are no names given for the PID "
                                     "bins, thus they will just be numbered.")
                        pid_names = [x for x in range(
                            0,
                            output.binning['pid'].num_bins
                        )]
                    for pid_name in pid_names:
                        outputs_2d.append(
                            output.split(
                                dim='pid',
                                bin=pid_name
                            )
                        )
                else:
                    raise ValueError("Script is set up to only plot 2D maps. "
                                     "Your outputs are %iD."
                                     %len(output.binning))
            else:
                raise ValueError("Script is set up to only plot 2D maps. Your "
                                 "outputs are %iD."%len(output.binning))
        if not len(outputs_2d) == 0:
            outputs = MapSet(maps=outputs_2d, name=outputs.name)
        for fmt, enabled in formats.items():
            if not enabled:
                continue
            my_plotter = Plotter(stamp='event rate',
                                 outdir=args.dir,
                                 fmt=fmt, log=False,
                                 annotate=args.annotate)
            my_plotter.ratio = True
            my_plotter.plot_2d_array(outputs,
                                     fname=stg_svc+'__output',
                                     cmap='OrRd')

    return pipeline, outputs


if __name__ == '__main__':
    pipeline, outputs = main()
