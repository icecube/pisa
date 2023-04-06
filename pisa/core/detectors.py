#! /usr/bin/env python
"""
Detector class definition and a simple script to generate, save, and
plot distributions for different detectors from pipeline config file(s).
A detector is represented by a DistributionMaker.

DistributionMaker: A single detector
Detectors: A sequence of detectors
"""

from __future__ import absolute_import

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
import inspect
from itertools import product
import os
from tabulate import tabulate
from copy import deepcopy

import numpy as np

from pisa import ureg
from pisa.core.map import MapSet
from pisa.core.pipeline import Pipeline
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.param import ParamSet, Param
from pisa.utils.config_parser import PISAConfigParser
from pisa.utils.fileio import expand, mkdir, to_file
from pisa.utils.hash import hash_obj
from pisa.utils.log import set_verbosity, logging, Levels
from pisa.utils.random_numbers import get_random_state


__all__ = ['Detectors', 'test_Detectors', 'parse_args', 'main']


class Detectors(object):
    """Container for one or more distribution makers, that represent different detectors.
    
    Parameters
    ----------
    pipelines : Pipeline or convertible thereto, or iterable thereof
        A new pipline is instantiated with each object passed. Legal objects
        are already-instantiated Pipelines and anything interpret-able by the
        Pipeline init method.
        
    shared_params : Parameter to be treated the same way in all the
        distribution_makers that contain them.
    """
    def __init__(self, pipelines, label=None, set_livetime_from_data=True, profile=False, shared_params=None):
        self.label = label
        self._source_code_hash = None
        
        self._profile = profile
        
        if shared_params == None:
            self.shared_params = []
        else:
            self.shared_params = shared_params

        if isinstance(pipelines, (str, PISAConfigParser, OrderedDict,
                                  Pipeline)):
            pipelines = [pipelines]
        
        self._distribution_makers , self.det_names = [] , []
        for pipeline in pipelines:
            if not isinstance(pipeline, Pipeline):
                pipeline = Pipeline(pipeline)
            
            name = pipeline.detector_name
            if name in self.det_names:
                self._distribution_makers[self.det_names.index(name)].append(pipeline)
            else:
                self._distribution_makers.append([pipeline])
                self.det_names.append(name)
    
        if None in self.det_names and len(self.det_names) > 1:
            raise NameError('At least one of the used pipelines has no detector_name.')

        for i, pipelines in enumerate(self._distribution_makers):
            self._distribution_makers[i] = DistributionMaker(pipelines=pipelines, 
                                                             set_livetime_from_data=set_livetime_from_data,
                                                             profile=profile
                                                            )
            
        for sp in self.shared_params:
            n = 0
            for distribution_maker in self._distribution_makers:
                if sp in distribution_maker.params.free.names:
                    n += 1
            if n < 2:
                raise NameError('Shared param %s only a free param in less than 2 detectors.' % sp)
                
    def __repr__(self):
        return self.tabulate(tablefmt="presto")

    def _repr_html_(self):
        return self.tabulate(tablefmt="html")

    def tabulate(self, tablefmt="plain"):
        headers = ['DistributionMaker number', 'detector name', 'output_binning', 'output_key', 'profile']
        colalign=["right"] + ["center"] * (len(headers) -1 )
        table = []
        for i, d in enumerate(self.distribution_makers):
            p = d.pipelines[0] #assuming binning and key are the same for all pipelines in DistributionMaker
            table.append([i, d.detector_name, p.output_binning, p.output_key, d.profile])
        return tabulate(table, headers, tablefmt=tablefmt, colalign=colalign)
            
    def __iter__(self):
        return iter(self._distribution_makers)
    
    @property
    def profile(self):
        return self._profile

    @profile.setter
    def profile(self, value):
        for d in self.distribution_makers:
            d.profile = value
        self._profile = value


    def run(self):
        for distribution_maker in self:
            distribution_maker.run()

    def setup(self):
        """Setup (reset) all distribution makers"""
        for d in self:
            d.setup()

    def get_outputs(self, **kwargs):
        """Compute and return the outputs.

        Parameters
        ----------
        **kwargs
            Passed on to each distribution_maker's `get_outputs` method.

        Returns
        -------
        List of MapSets if `return_sum=True` or list of lists of MapSets if `return_sum=False`

        """
        outputs = [distribution_maker.get_outputs(**kwargs) for distribution_maker in self]
        return outputs

    def update_params(self, params):
        for distribution_maker in self:
            distribution_maker.update_params(params)

        #if None in self.det_names: return # No detector names

        if isinstance(params,Param): params = ParamSet(params) # just for the following

        for p in params.names: # now update params with det_names inside
            for i, det_name in enumerate(self.det_names):
                if det_name in p:
                    cp = deepcopy(params[p])
                    cp.name = cp.name.replace('_'+det_name, "")
                    self._distribution_makers[i].update_params(cp)

    def select_params(self, selections, error_on_missing=True):
        for distribution_maker in self:
            distribution_maker.select_params(selections=selections, error_on_missing=error_on_missing)
            
    @property
    def distribution_makers(self):
        return self._distribution_makers

    @property
    def params(self):
        """Returns a ParamSet including all params of all detectors. First the shared params
        (if there are some), then all the "single detector" params. If two detectors use a
        parameter with the same name (but not shared), the name of the detector is added to the
        parameter name (except for the first detector).
        """        
        params = ParamSet()
        for p_name in self.shared_params:
            for distribution_maker in self:
                try:
                    params.extend(distribution_maker.params[p_name])
                    break  # shared param found, can continue with the next shared param
                except:
                    continue # shared param was not in this DistributionMaker, so search in the next one
                    
        for distribution_maker in self:
            for param in distribution_maker.params:
                if param.name in self.shared_params:
                    continue # shared param is already in param set, can continue with the next param
                elif param.name in params.names: # two parameters with the same name but not shared 
                    # add detector name to the parameter name
                    changed_param = deepcopy(param)
                    changed_param.name = param.name + '_' + distribution_maker.detector_name
                    params.extend(changed_param)
                else:
                    params.extend(param)
        return params

    @property
    def shared_param_ind_list(self):
        """ A list of lists (one for each detector) containing the position of the shared 
        params in the free params of the DistributionMaker (that belongs to the detector)
        together with their position in the shared parameter list.
        """
        if not self.shared_params: return []

        shared_param_ind_list = []
        for distribution_maker in self:
            free_names = distribution_maker.params.free.names
            spi = []
            for p_name in free_names:
                if p_name in self.shared_params:
                    spi.append((free_names.index(p_name),self.shared_params.index(p_name)))
            shared_param_ind_list.append(spi)
        return shared_param_ind_list
            
    @property
    def param_selections(self):
        selections = None
        for distribution_maker in self:
            if selections != None and sorted(distribution_maker.param_selections) != selections:
                raise ('Different param_selections for different detectors.')
            selections = sorted(distribution_maker.param_selections)
        return selections

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
        return hash_obj([self.source_code_hash] + [d.hash for d in self])

    @property
    def num_events_per_bin(self):
        '''
        returns list of arrays of bin indices where none of the 
        pipelines in the respective distribution maker have MC events
        '''
        num_events_per_bin = []
        for d in self.distribution_makers:
            num_events_per_bin.append(d.num_events_per_bin)

        return num_events_per_bin
    

    @property
    def empty_bin_indices(self):
        '''Find indices where there are no events present
        '''
        num_events_per_bin = self.num_events_per_bin
        
        indices = []
        for i, d in enumerate(self.distribution_makers):
            empty_counts = num_events_per_bin[i] == 0
            indices.append(np.where(empty_counts)[0])
        return indices


    def set_free_params(self, values):
        """Set free parameters' values.

        Parameters
        ----------
        values : a list of quantities

        """
        for dist_maker in self:
            dist_values = []
            for dist_name in dist_maker.params.free.names:
                for name, value in zip(self.params.free.names, values):
                    if name == dist_name:
                        v = value
                    if name == dist_name + '_' + dist_maker.detector_name:
                        v = value
                dist_values.append(v)
            dist_maker.set_free_params(dist_values)

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
            d.reset_all()

    def reset_free(self):
        """Reset only free parameters to their nominal values."""
        for d in self:
            d.reset_free()

    def set_nominal_by_current_values(self):
        """Define the nominal values as the parameters' current values."""
        for d in self:
            d.set_nominal_by_current_values()

    def _set_rescaled_free_params(self, rvalues):
        """Set free param values given a simple list of [0,1]-rescaled,
        dimensionless values
        """
        if not isinstance(rvalues,list):
            rvalues = list(rvalues)
        
        if self.shared_params == []:
            for d in self:
                rp = []
                for j in range(len(d.params.free)):
                    rp.append(rvalues.pop(0))
                d._set_rescaled_free_params(rp)
                
        else:
            sp = [] # first get the shared params
            for i in range(len(self.shared_params)):
                sp.append(rvalues.pop(0))
            spi = self.shared_param_ind_list

            for i in range(len(self._distribution_makers)):
                rp = []
                for j in range(len(self._distribution_makers[i].params.free) - len(spi[i])):
                    rp.append(rvalues.pop(0))
                for j in range(len(spi[i])):
                    rp.insert(spi[i][j][0],sp[spi[i][j][1]])
                self._distribution_makers[i]._set_rescaled_free_params(rp)


def test_Detectors(verbosity=Levels.WARN):
    from pisa.analysis.analysis import update_param_values_detector

    """Run a combination of two DeepCore detectors."""
    p1_nu = Pipeline("settings/pipeline/IceCube_3y_neutrinos.cfg")
    p1_mu = Pipeline("settings/pipeline/IceCube_3y_muons.cfg")
    p1_nu.detector_name, p1_mu.detector_name = 'detector1', 'detector1'

    p2_nu = Pipeline("settings/pipeline/IceCube_3y_neutrinos.cfg")
    p2_mu = Pipeline("settings/pipeline/IceCube_3y_muons.cfg")
    p2_nu.detector_name, p2_mu.detector_name = 'detector2', 'detector2'
    
    # Initializing
    try:
        set_verbosity(Levels.INFO)
        logging.info(f'Initializing Detectors')
        
        set_verbosity(Levels.WARN)
        model = Detectors([p1_nu, p1_mu, p2_nu, p2_mu], shared_params=['deltam31', 'theta13', 'theta23', 'nue_numu_ratio', 'Barr_uphor_ratio', 'Barr_nu_nubar_ratio', 'delta_index', 'nutau_norm', 'nu_nc_norm', 'opt_eff_overall', 'opt_eff_lateral', 'opt_eff_headon', 'ice_scattering', 'ice_absorption', 'atm_muon_scale'])
        
    except Exception as err:
        msg = f"<< Error when initializing the Detectors >>"
        set_verbosity(verbosity)
        logging.error("=" * len(msg))
        logging.error(msg)
        logging.error("=" * len(msg))
        
        set_verbosity(Levels.TRACE)
        logging.exception(err)

        set_verbosity(verbosity)
        logging.error("#" * len(msg))
        
    else:
        set_verbosity(verbosity)
        logging.info("<< Successfully initialized Detectors >>")
        
    finally:
        set_verbosity(verbosity)
    
    # Get outputs
    try:
        set_verbosity(Levels.INFO)
        logging.info(f'Running Detectors (takes a bit)')

        set_verbosity(Levels.WARN)
        model.get_outputs()

    except Exception as err:
        msg = f"<< Error when running the Detectors >>"
        set_verbosity(verbosity)
        logging.error("=" * len(msg))
        logging.error(msg)
        logging.error("=" * len(msg))
        
        set_verbosity(Levels.TRACE)
        logging.exception(err)

        set_verbosity(verbosity)
        logging.error("#" * len(msg))
    
    else:
        set_verbosity(verbosity)
        logging.info("<< Successfully ran Detectors >>")
    
    finally:
        set_verbosity(verbosity)
    
    # Change parameters
    set_verbosity(Levels.INFO)
    logging.info(f'Change parameters')

    set_verbosity(Levels.WARN)
    model.reset_free()
    model.params.opt_eff_lateral.value = 20 # shared parameter
    model.params.aeff_scale.value = 2       # only changes value for detector1

    update_param_values_detector(model, model.params)
    
    o0 = model.distribution_makers[0].params.opt_eff_lateral.value.magnitude
    o1 = model.distribution_makers[1].params.opt_eff_lateral.value.magnitude
    a0 = model.distribution_makers[0].params.aeff_scale.value.magnitude
    a1 = model.distribution_makers[1].params.aeff_scale.value.magnitude

    if not o0 == 20 or not o1 == 20:
        msg = f"<< Error when changing shared parameter >>"
        set_verbosity(verbosity)
        logging.error("=" * len(msg))
        logging.error(msg)
        logging.error("=" * len(msg))
    
    elif not a0 == 2 or not a1 == 1:
        msg = f"<< Error when changing non-shared parameter >>"
        set_verbosity(verbosity)
        logging.error("=" * len(msg))
        logging.error(msg)
        logging.error("=" * len(msg))
        
    else:
        set_verbosity(verbosity)
        logging.info("<< Successfully changed parameters >>")
    
    # Unit test
    set_verbosity(Levels.INFO)
    logging.info(f'Unit test')
    
    set_verbosity(Levels.WARN)
    hierarchies = ['nh', 'ih']
    t23 = dict(
        ih=49.5 * ureg.deg,
        nh=42.3 * ureg.deg
    )
    current_hier = 'nh'
    
    for new_hier in hierarchies:
        #assert model.param_selections == [current_hier], str(model.param_selections)
        assert model.params.theta23.value == t23[current_hier], str(model.params.theta23)

        # Select the hierarchy
        model.select_params(new_hier)
        #assert model.param_selections == [new_hier], str(model.param_selections)
        assert model.params.theta23.value == t23[new_hier], str(model.params.theta23)
        
        # Reset to "current"
        model.select_params(current_hier)
        #assert model.param_selections == [current_hier], str(model.param_selections)
        assert model.params.theta23.value == t23[current_hier], str(model.params.theta23)
        
    set_verbosity(verbosity)
    logging.info("<< Successfully ran unit test >>")
    
    # Done
    logging.info(f'Detectors class test done')


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
        '--shared_params', type=str, default=None,
        action='append',
        help='''Shared parameters for multi det analysis (repeat for multiple).'''
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
        
    detectors = Detectors(args.pipeline,shared_params=args.shared_params)
    Names = detectors.det_names
    if args.select is not None:
        detectors.select_params(args.select)

    outputs = detectors.get_outputs(return_sum=args.return_sum)

    #outputs = outputs[0].fluctuate(
     #               method='poisson', random_state=get_random_state([0, 0, 0]))

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
