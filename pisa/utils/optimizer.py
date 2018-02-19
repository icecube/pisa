#! /usr/bin/env python

"""
Implementation of an optimizer base class, which has all basic
functionality built in.
"""


from __future__ import absolute_import

from collections import OrderedDict

import numpy as np

from pisa.core.param import ParamSet
from pisa.utils.config_parser import PISAConfigParser, parse_optimizer_config
from pisa.utils.fit import Fit
from pisa.utils.log import logging
from pisa.utils.profiler import profile


__all__ = ['Optimizer', 'test_Optimizer']

__author__ = 'T. Ehrhardt'

__license__ = '''Copyright (c) 2014-2018, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


class Optimizer(object):
    """Instantiate an optimizer according to an instantiated config object;
    perform an optimization.

    Parameters
    ----------
    config : string, OrderedDict, or PISAConfigParser
        If string, interpret as resource location; send to the
        `config_parser.parse_optimizer_config()` method to get a config
        OrderedDict. If `OrderedDict`, use directly as optimizer configuration.

    data_dist : MapSet
        Data distribution(s). These are what the hypothesis is tasked to
        best describe during the optimization process.

    hypo_maker : DistributionMaker or instantiable thereto
        Generates the expectation distribution under a particular
        hypothesis. This typically has (but is not required to have) some
        free parameters which can be modified by the minimizer to optimize
        the `metric`.

    hypo_param_selections : None, string, or sequence of strings
        A pipeline configuration can have param selectors that allow
        switching a parameter among two or more values by specifying the
        corresponding param selector(s) here. This also allows for a single
        instance of a DistributionMaker to generate distributions from
        different hypotheses.

    pprint : bool
        Whether to show live-update of minimizer progress.

    blind : bool
        Whether to carry out a blind analysis. This hides actual parameter
        values from display and disallows these (as well as Jacobian,
        Hessian, etc.) from ending up in logfiles.

    Notes
    -----
    The following methods can be overridden in derived classes where
    applicable:
        _run_optimization
        _validate_result
        _print_progress
    """

    def __init__(self, config, data_maker, hypo_maker,
                 data_param_selections, hypo_param_selections,
                 blind, pprint):

        if isinstance(config, (basestring, PISAConfigParser)):
            config = parse_optimizer_config(config=config)
        elif isinstance(config, OrderedDict):
            pass
        else:
            raise TypeError(
                '`config` passed is of type %s but must be string,'
                ' PISAConfigParser, or OrderedDict' % type(config).__name__
            )

        self._config = config

        self.data_maker = data_maker
        self.hypo_maker = hypo_maker

        self.data_param_selections = data_param_selections
        self.hypo_param_selections = hypo_param_selections

        self.full_hash = True
        """Whether to do full hashing if true, otherwise do fast hashing"""

        self.blind = blind
        """Whether to perform a blind fit"""

        self.pprint = pprint
        """Whether to print progress"""

        self.params_to_fit = ParamSet()
        """Records the parameters to be fit"""

        self.result = Fit()
        """Records the fit result"""

        self.tmp = Fit()
        """Intermediate fit status that can be reported during the process"""

        self.is_result_cleansed = False
        """Whether the fit result has been 'cleansed' for blindness"""

        # Define useful flags and values for debugging behavior after running

        self.done = False
        """Whether the fit has been computed"""

        self.success = None
        """Whether the fit was successful"""

        self.cache = None
        """Memory cache object for storing fits"""

        self.result_hash = None
        #self.result_cleansed_hash = None

    @profile
    def run_optimizatin(self):
        """This method calls the `_run_optimization` method, which by
        default does nothing.

        However, if you want to implement your own optimizer,
        override the `_run_optimization` method and fill in the
        logic there.
        """
        self.result = self._run_optimization()
        self.done = True
        self.cleanse_result()

    def _run_optimization(self): # pylint: disable=no-self-use
        """Derived optimizers should override this method.
        """
        return None

    def cleanse_result(self):
        if self.blind:
            logging.info("Getting cleansed optimization results because"
                         "blindness was requested...")
            # requires a `Fit` object to have a `cleansed` attribute
            # leave no trace of the non-blind result 
            self.result = self.result.cleansed
            self.is_result_cleansed = True

    def validate_result(self):
        self.success = self._validate_result()

    def _validate_result(self): # pylint: disable=no-self-use
        """Derived optimizers should override this method."""
        return None


def test_Optimizer():
    pass


if __name__ == '__main__':
    test_Optimizer()
