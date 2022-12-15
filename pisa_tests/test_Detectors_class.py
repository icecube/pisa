#! /usr/bin/env python

"""
Tests the Detectors class
"""


from __future__ import absolute_import

from argparse import ArgumentParser
import glob

from pisa.core.pipeline import Pipeline
from pisa.core.detectors import Detectors
from pisa.utils.log import Levels, logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.analysis.analysis import update_param_values_detector


__all__ = ["test_Detectors", "parse_args", "main"]

__author__ = "J. Weldert"

__license__ = """Copyright (c) 2014-2022, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


def test_Detectors(verbosity=Levels.WARN):
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



def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-v", action="count", default=Levels.WARN, help="set verbosity level"
    )
    args = parser.parse_args()
    return args


def main():
    """Script interface to test_Detectors"""
    args = parse_args()
    kwargs = vars(args)
    kwargs["verbosity"] = kwargs.pop("v")
    test_Detectors(**kwargs)
    logging.info(f'Detectors class test done')


if __name__ == "__main__":
    main()
