#! /usr/bin/env python

"""
Implementation of the Pipeline object, and a simple script to instantiate and
run a pipeline (the outputs of which can be plotted and stored to disk).
"""


from __future__ import absolute_import

from argparse import ArgumentParser
from collections import OrderedDict
from configparser import NoSectionError
from copy import deepcopy
from importlib import import_module
from itertools import product
from inspect import getsource
import os
import traceback

import numpy as np

from pisa import ureg
from pisa.core.events import Data
from pisa.core.map import Map, MapSet
from pisa.core.param import ParamSet
from pisa.core.base_stage import BaseStage
from pisa.core.stage import Stage
from pisa.core.pi_stage import PiStage
from pisa.core.transform import TransformSet
from pisa.core.container import ContainerSet
from pisa.utils.config_parser import PISAConfigParser, parse_pipeline_config
from pisa.utils.fileio import mkdir
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import profile


__all__ = ["Pipeline", "test_Pipeline", "parse_args", "main"]

__author__ = "J.L. Lanfranchi, P. Eller"

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
    config : string, OrderedDict, or PISAConfigParser
        If string, interpret as resource location; send to the
        `config_parser.parse_pipeline_config()` method to get a config
        OrderedDict. If `OrderedDict`, use directly as pipeline configuration.

    """

    def __init__(self, config):
        if isinstance(config, (str, PISAConfigParser)):
            config = parse_pipeline_config(config=config)
        elif isinstance(config, OrderedDict):
            pass
        else:
            raise TypeError(
                "`config` passed is of type %s but must be string,"
                " PISAConfigParser, or OrderedDict" % type(config).__name__
            )

        self.pisa_version = None

        self._stages = []
        self._detector_name = config.pop('detector_name', None)
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
        assert isinstance(stage_id, (int, str))
        for stage_num, stage in enumerate(self):
            if stage_id in [stage_num, stage.stage_name]:
                return stage_num
        raise ValueError('No stage "%s" found in the pipeline.' % stage_id)

    def __len__(self):
        return len(self._stages)

    def __iter__(self):
        return iter(self._stages)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.stages[self.index(idx)]

        if isinstance(idx, (int, slice)):
            return self.stages[idx]

        raise ValueError(
            'Cannot locate stage "%s" in pipeline. Stages'
            " available are %s." % (idx, self.stage_names)
        )

    def __getattr__(self, attr):
        for stage in self:
            if stage.stage_name == attr:
                return stage
        raise AttributeError(
            '"%s" is neither a stage in this pipeline nor an attribute/property'
            " of the `Pipeline` object." % attr
        )

    def _init_stages(self):
        """Stage factory: Instantiate stages specified by self.config.

        Conventions required for this to work:
            * Stage and service names must be lower-case
            * Service implementations must be found at Python path
              `pisa.stages.<stage_name>.<service_name>`
            * `service` cannot be an instantiation argument for a service

        """
        stages = []
        data = ContainerSet("events")
        for stage_num, ((stage_name, service_name), settings) in enumerate(
            self.config.items()
        ):
            try:
                logging.debug(
                    "instantiating stage %s / service %s", stage_name, service_name
                )

                # Import service's module
                module_path = f"pisa.stages.{stage_name}.{service_name}"
                logging.trace("Importing service module: %s", module_path)
                module = import_module(module_path)

                # Get service class from module
                service_cls = getattr(module, service_name)

                # Instantiate service
                logging.trace(
                    "initializing stage.service %s.%s with settings %s"
                    % (stage_name, service_name, settings)
                )
                try:
                    service = service_cls(**settings)
                except Exception:
                    logging.error(
                        "Failed to instantiate stage.service %s.%s with settings %s",
                        stage_name,
                        service_name,
                        settings.keys(),
                    )
                    raise

                cake_stage = isinstance(service, Stage)
                pi_stage = isinstance(service, PiStage)

                if not (cake_stage or pi_stage):
                    logging.debug("is BaseStage? %s", isinstance(service, BaseStage))
                    logging.debug("is Stage (cake)? %s", isinstance(service, Stage))
                    logging.debug("is PiStage? %s", isinstance(service, PiStage))

                    raise TypeError(
                        'Trying to create service "%s" for stage #%d (%s),'
                        " but object %s instantiated from class %s is not a"
                        " PISA Stage type but instead is of type %s."
                        % (
                            service_name,
                            stage_num,
                            stage_name,
                            service,
                            service_cls,
                            type(service),
                        )
                    )

                # first stage can determine type of pipeline
                if self.pisa_version is None:
                    self.pisa_version = "cake" if cake_stage else "pi"

                elif self.pisa_version == "cake" and pi_stage:
                    raise TypeError(
                        "Trying to use the PISA Pi Stage in " "a PISA cake pipeline."
                    )

                elif self.pisa_version == "pi" and cake_stage:
                    raise TypeError(
                        "Trying to use the PISA cake Stage in " "a PISA Pi pipeline."
                    )

                # Append service to pipeline

                if self.pisa_version == "pi":
                    service.data = data
                # add events object

                # run setup on service
                service.setup()

                stages.append(service)

            except:
                logging.error(
                    "Failed to initialize stage #%d (stage=%s, service=%s).",
                    stage_num,
                    stage_name,
                    service_name,
                )
                raise

        param_selections = set()
        for service in stages:
            param_selections.update(service.param_selections)
        param_selections = sorted(param_selections)

        previous_stage = None
        for stage in stages:
            stage.select_params(param_selections, error_on_missing=False)
            if previous_stage is not None:
                prev_has_binning = (
                    hasattr(previous_stage, "output_binning")
                    and previous_stage.output_binning is not None
                )
                this_has_binning = (
                    hasattr(stage, "input_binning") and stage.input_binning is not None
                )
                if this_has_binning != prev_has_binning:
                    raise ValueError(
                        'hasattr(%s, "output_binning") is %s but'
                        ' hasattr(%s, "input_binning") is %s.'
                        % (
                            previous_stage.stage_name,
                            prev_has_binning,
                            stage.stage_name,
                            this_has_binning,
                        )
                    )
                if this_has_binning:
                    is_compat = stage.input_binning.is_compat(
                        previous_stage.output_binning
                    )
                    if not is_compat:
                        logging.error(
                            "Stage %s output binning: %s",
                            previous_stage.stage_name,
                            previous_stage.output_binning,
                        )
                        logging.error(
                            "Stage %s input binning: %s",
                            stage.stage_name,
                            stage.input_binning,
                        )
                        raise ValueError(
                            "%s stage's output binning is incompatible with"
                            " %s stage's input binning."
                            % (previous_stage.stage_name, stage.stage_name)
                        )
            previous_stage = stage

        self._stages = stages

    # TODO: handle other container(s)
    @profile
    def get_outputs(self, inputs=None, idx=None, return_intermediate=False, output_mode=None, force_standard_output=True):
        """Run the pipeline to compute its outputs.

        Parameters
        ----------
        inputs : None or MapSet
            Optional inputs to send to the first stage of the pipeline.

        idx : None, string, or int
            Specification of which stage(s) to run. If None is passed, all
            stages will be run. If a string is passed, all stages are run up to
            and including the named stage. If int is passed, all stages are run
            up to and including `idx`. Numbering follows Python
            conventions (i.e., is 0-indexed).

        return_intermediate : bool
            Return list containing outputs from each stage in the pipeline.

        output_mode: string
            force an output mode to the stage

        force_standard_output: bool
            if requesting a binned output, object returned will be a list of MapSets
            if set to False, object returned will be a list of Dict. each Dict item is 
            a MapSet


        Returns
        -------
        outputs : list or pisa.core.map.MapSet
            If `return_intermediate` is `False`, returns `MapSet` output by
            final stage. If `return_intermediate` is `True`, returns `list` of
            `MapSet`s output by each stage.

        """
        intermediate = []

        if isinstance(idx, str):
            idx = self.stage_names.index(idx)

        if idx is not None:
            if idx < 0:
                raise ValueError("Integer `idx` must be >= 0")
            idx += 1

        if len(self) == 0:
            raise ValueError("No stages in the pipeline to run")

        for stage in self.stages[:idx]:
            name = "{}.{}".format(stage.stage_name, stage.service_name)
            logging.debug(
                '>> Working on stage "%s" service "%s"',
                stage.stage_name,
                stage.service_name,
            )
            try:
                logging.trace(">>> BEGIN: {}.run(...)".format(name))
                outputs = stage.run(inputs=inputs) # pylint: disable=redefined-outer-name
                if return_intermediate:
                    if outputs is None:  # e.g. for PISA pi
                        outputs = stage.get_outputs(output_mode=output_mode, force_standard_output=force_standard_output)
                    intermediate.append(outputs)
                logging.trace(">>> END  : {}.run(...)".format(name))
            except:
                logging.error(
                    "Error occurred computing outputs in stage %s /" " service %s ...",
                    stage.stage_name,
                    stage.service_name,
                )
                raise
            logging.trace("outputs: %s" % (outputs,))
            inputs = outputs

        if outputs is None:  # e.g. for PISA pi
            outputs = stage.get_outputs(output_mode=output_mode, force_standard_output=force_standard_output)

        if return_intermediate:
            return intermediate

        return outputs

    def update_params(self, params):
        """Update params for the pipeline.

        Note that any param in `params` in excess of those that already exist
        in the pipeline's stages will have no effect.

        Parameters
        ----------
        params : ParamSet
            Parameters to be updated

        """
        for stage in self:
            stage.params.update_existing(params)

    def select_params(self, selections, error_on_missing=False):
        """Select a set of alternate param values/specifications.

        Parameters
        -----------
        selections : string or iterable of strings
        error_on_missing : bool

        Raises
        ------
        KeyError if `error_on_missing` is `True` and any of `selections` does
            not exist in any stage in the pipeline.

        """
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
                "None of the stages in this pipeline has all of the"
                " selections %s available." % (selections,)
            )

    @property
    def params(self):
        """pisa.core.param.ParamSet : pipeline's parameters"""
        params = ParamSet()
        for stage in self:
            params.extend(stage.params)
        return params

    @property
    def param_selections(self):
        """list of strings : param selections collected from all stages"""
        selections = set()
        for stage in self:
            selections.update(stage.param_selections)
        return sorted(selections)

    @property
    def stages(self):
        """list of Stage : stages in the pipeline"""
        return [s for s in self]

    @property
    def stage_names(self):
        """list of strings : names of stages in the pipeline"""
        return [s.stage_name for s in self]

    @property
    def config(self):
        """Deepcopy of the OrderedDict used to instantiate the pipeline"""
        return deepcopy(self._config)

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
    def hash(self):
        """int : Hash of the state of the pipeline. This hashes together a hash
        of the Pipeline class's source code and a hash of the state of each
        contained stage."""
        return hash_obj([self.source_code_hash] + [stage.hash for stage in self])

    def __hash__(self):
        return self.hash


def test_Pipeline():
    """Unit tests for Pipeline class"""
    # pylint: disable=line-too-long

    # TODO: make a test config file with hierarchy AND material selector,
    # uncomment / add in tests commented / removed below

    #
    # Test: select_params and param_selections
    #

    hierarchies = ["nh", "ih"]
    #materials = ["iron", "pyrolite"]
    materials = []

    t23 = dict(ih=49.5 * ureg.deg, nh=42.3 * ureg.deg)
    #YeO = dict(iron=0.4656, pyrolite=0.4957)

    # Instantiate with two pipelines: first has both nh/ih and iron/pyrolite
    # param selectors, while the second only has nh/ih param selectors.
    pipeline = Pipeline("settings/pipeline/example.cfg")  # pylint: disable=redefined-outer-name

    #current_mat = "iron"
    current_hier = "nh"

    for new_hier, new_mat in product(hierarchies, materials):
        #_ = YeO[new_mat]

        assert pipeline.param_selections == sorted([current_hier]), str(
            pipeline.param_selections
        )
        assert pipeline.params.theta23.value == t23[current_hier], str(
            pipeline.params.theta23
        )
        #assert pipeline.params.YeO.value == YeO[current_mat], str(pipeline.params.YeO)

        # Select just the hierarchy
        pipeline.select_params(new_hier)
        assert pipeline.param_selections == sorted([new_hier]), str(
            pipeline.param_selections
        )
        assert pipeline.params.theta23.value == t23[new_hier], str(
            pipeline.params.theta23
        )
        #assert pipeline.params.YeO.value == YeO[current_mat], str(pipeline.params.YeO)

        ## Select just the material
        #pipeline.select_params(new_mat)
        #assert pipeline.param_selections == sorted([new_hier, new_mat]), str(
        #    pipeline.param_selections
        #)
        assert pipeline.params.theta23.value == t23[new_hier], str(
            pipeline.params.theta23
        )
        #assert pipeline.params.YeO.value == YeO[new_mat], str(pipeline.params.YeO)

        # Reset both to "current"
        pipeline.select_params([current_hier])
        assert pipeline.param_selections == sorted([current_hier]), str(
            pipeline.param_selections
        )
        assert pipeline.params.theta23.value == t23[current_hier], str(
            pipeline.params.theta23
        )
        #assert pipeline.params.YeO.value == YeO[current_mat], str(pipeline.params.YeO)

        ## Select both hierarchy and material
        #pipeline.select_params([new_hier])
        #assert pipeline.param_selections == sorted([new_hier, new_mat]), str(
        #    pipeline.param_selections
        #)
        #assert pipeline.params.theta23.value == t23[new_hier], str(
        #    pipeline.params.theta23
        #)
        #assert pipeline.params.YeO.value == YeO[new_mat], str(pipeline.params.YeO)

        #current_hier = new_hier
        #current_mat = new_mat


def parse_args():
    """Parse command line arguments if `pipeline.py` is called as a script."""
    parser = ArgumentParser(
        # formatter_class=ArgumentDefaultsHelpFormatter,
        description="""Instantiate and run a pipeline from a config file.
        Optionally store the resulting distribution(s) and plot(s) to disk."""
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-p",
        "--pipeline",
        metavar="CONFIGFILE",
        type=str,
        required=True,
        help="File containing settings for the pipeline.",
    )

    parser.add_argument(
        "-a",
        "--arg",
        metavar="SECTION ARG=VAL",
        nargs="+",
        action="append",
        help="""Set config arg(s) manually. SECTION can be e.g.
        "stage:<stage_name>" (like "stage:flux", "stage:reco", etc.),
        "pipeline", and so forth. Arg values specified here take precedence
        over those in the config file, but note that the sections specified
        must already exist in the config file.""",
    )
    parser.add_argument(
        "--select",
        metavar="PARAM_SELECTIONS",
        nargs="+",
        default=None,
        help="""Param selectors (separated by spaces) to use to override any
        defaults in the config file.""",
    )
    parser.add_argument(
        "--inputs",
        metavar="FILE",
        type=str,
        help="""File from which to read inputs to be fed to the pipeline.""",
    )
    parser.add_argument(
        "--only-stage",
        metavar="STAGE",
        type=str,
        help="""Test stage: Instantiate a single stage in the pipeline
        specification and run it in isolation (as the sole stage in a
        pipeline). If it is a stage that requires inputs, these can be
        specified with the --infile argument, or else dummy stage input maps
        (numpy.ones(...), matching the input binning specification) are
        generated for testing purposes.""",
    )
    parser.add_argument(
        "--stop-after-stage",
        metavar="STAGE",
        help="""Instantiate a pipeline up to and including STAGE, but stop
        there. Can specify a stage by index in the pipeline config (e.g., 0, 1,
        etc.) or name (e.g., flux, osc, etc.)""",
    )
    parser.add_argument(
        "--outdir",
        metavar="DIR",
        type=str,
        help="""Store all output files (data and plots) to this directory.
        Directory will be created (including missing parent directories) if it
        does not exist already. If no dir is provided, no outputs will be
        saved.""",
    )
    parser.add_argument(
        "--intermediate",
        action="store_true",
        help="""Store all intermediate outputs, not just the final stage's
        outputs.""",
    )
    parser.add_argument(
        "--transforms",
        action="store_true",
        help="""Store all transforms (for stages that use transforms).""",
    )
    # TODO: optionally store the transform sets from each stage
    # parser.add_argument(
    #    '-T', '--transform-file', metavar='FILE', type=str,
    #    help='''File into which to store transform(s) from the pipeline.'''
    # )
    parser.add_argument("--pdf", action="store_true", help="""Produce pdf plot(s).""")
    parser.add_argument("--png", action="store_true", help="""Produce png plot(s).""")
    parser.add_argument(
        "--annotate", action="store_true", help="""Annotate plots with counts per bin"""
    )
    parser.add_argument(
        "-v",
        action="count",
        default=None,
        help="""Set verbosity level. Repeat for increased verbosity. -v is
        info-level, -vv is debug-level and -vvv is trace-level output.""",
    )
    args = parser.parse_args()
    return args


def main(return_outputs=False):
    """Run unit tests if `pipeline.py` is called as a script."""
    from pisa.utils.plotter import Plotter

    args = parse_args()
    set_verbosity(args.v)

    # Even if user specifies an integer on command line, it comes in as a
    # string. Try to convert to int (e.g. if `'1'` is passed to indicate the
    # second stage), and -- if successful -- use this as `args.only_stage`.
    # Otherwise, the string value passed will be used (e.g. `'osc'` could be
    # passed).
    try:
        only_stage_int = int(args.only_stage)
    except (ValueError, TypeError):
        pass
    else:
        args.only_stage = only_stage_int

    if args.outdir:
        mkdir(args.outdir)
    else:
        if args.pdf or args.png:
            raise ValueError("No --outdir provided, so cannot save images.")

    # Most basic parsing of the pipeline config (parsing only to this level
    # allows for simple strings to be specified as args for updating)
    bcp = PISAConfigParser()
    bcp.read(args.pipeline)

    # Update the config with any args specified on command line
    if args.arg is not None:
        for arg_list in args.arg:
            if len(arg_list) < 2:
                raise ValueError(
                    'Args must be formatted as: "section arg=val". Got "%s"'
                    " instead." % " ".join(arg_list)
                )
            section = arg_list[0]
            remainder = " ".join(arg_list[1:])
            eq_split = remainder.split("=")
            newarg = eq_split[0].strip()
            value = ("=".join(eq_split[1:])).strip()
            logging.debug(
                'Setting config section "%s" arg "%s" = "%s"', section, newarg, value
            )
            try:
                bcp.set(section, newarg, value)
            except NoSectionError:
                logging.error(
                    'Invalid section "%s" specified. Must be one of %s',
                    section,
                    bcp.sections(),
                )
                raise

    # Instantiate the pipeline
    pipeline = Pipeline(bcp)  # pylint: disable=redefined-outer-name

    if args.select is not None:
        pipeline.select_params(args.select, error_on_missing=True)

    if args.only_stage is None:
        stop_idx = args.stop_after_stage
        try:
            stop_idx = int(stop_idx)
        except (TypeError, ValueError):
            pass
        if isinstance(stop_idx, str):
            stop_idx = pipeline.index(stop_idx)
        outputs = pipeline.get_outputs(
            idx=stop_idx
        )  # pylint: disable=redefined-outer-name
        if stop_idx is not None:
            stop_idx += 1
        indices = slice(0, stop_idx)
    else:
        assert args.stop_after_stage is None
        idx = pipeline.index(args.only_stage)
        stage = pipeline[idx]
        indices = slice(idx, idx + 1)

        # Create dummy inputs if necessary
        inputs = None
        if hasattr(stage, "input_binning"):
            logging.warning(
                "Stage requires input, so building dummy"
                " inputs of random numbers, with random state set to the input"
                " index according to alphabetical ordering of input names and"
                " filled in alphabetical ordering of dimension names."
            )
            input_maps = []
            tmp = deepcopy(stage.input_binning)
            alphabetical_binning = tmp.reorder_dimensions(sorted(tmp.names))
            for input_num, input_name in enumerate(sorted(stage.input_names)):
                # Create a new map with all 3's; name according to the input
                hist = np.full(shape=alphabetical_binning.shape, fill_value=3.0)
                input_map = Map(
                    name=input_name, binning=alphabetical_binning, hist=hist
                )

                # Apply Poisson fluctuations to randomize the values in the map
                input_map.fluctuate(method="poisson", random_state=input_num)

                # Reorder dimensions according to user's original binning spec
                input_map.reorder_dimensions(stage.input_binning)
                input_maps.append(input_map)
            inputs = MapSet(maps=input_maps, name="ones", hash=1)

        outputs = stage.run(inputs=inputs)

    for stage in pipeline[indices]:
        if not args.outdir:
            break
        stg_svc = stage.stage_name + "__" + stage.service_name
        fbase = os.path.join(args.outdir, stg_svc)
        if args.intermediate or stage == pipeline[indices][-1]:
            stage.outputs.to_json(fbase + "__output.json.bz2")
        if args.transforms and stage.use_transforms:
            stage.transforms.to_json(fbase + "__transforms.json.bz2")

        # also only plot if args intermediate or last stage
        if args.intermediate or stage == pipeline[indices][-1]:
            formats = OrderedDict(png=args.png, pdf=args.pdf)
            if isinstance(stage.outputs, Data):
                # TODO(shivesh): plots made here will use the most recent
                # "pisa_weight" column and so all stages will have identical plots
                # (one workaround is to turn on "memcache_deepcopy")
                # TODO(shivesh): intermediate stages have no output binning
                if stage.output_binning is None:
                    logging.debug("Skipping plot of intermediate stage %s", stage)
                    continue
                outputs = stage.outputs.histogram_set(
                    binning=stage.output_binning,
                    nu_weights_col="pisa_weight",
                    mu_weights_col="pisa_weight",
                    noise_weights_col="pisa_weight",
                    mapset_name=stg_svc,
                    errors=True,
                )
            elif isinstance(stage.outputs, (MapSet, TransformSet)):
                outputs = stage.outputs

            try:
                for fmt, enabled in formats.items():
                    if not enabled:
                        continue
                    my_plotter = Plotter(
                        stamp="Event rate",
                        outdir=args.outdir,
                        fmt=fmt,
                        log=False,
                        annotate=args.annotate,
                    )
                    my_plotter.ratio = True
                    my_plotter.plot_2d_array(
                        outputs, fname=stg_svc + "__output", cmap="RdBu"
                    )
            except ValueError as exc:
                logging.error(
                    "Failed to save plot to format %s. See exception" " message below",
                    fmt,
                )
                traceback.format_exc()
                logging.exception(exc)
                logging.warning("I can't go on, I'll go on.")

    if return_outputs:
        return pipeline, outputs


if __name__ == "__main__":
    pipeline, outputs = main(return_outputs=True)  # pylint: disable=invalid-name
