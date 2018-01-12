#!/usr/bin/env python

"""
Class Timer which provides a context for timing blocks of code.

See Also: pisa.utils.profile module, which contains decorators for timing
functions and methods.
"""


from __future__ import absolute_import, division

from time import sleep, time

import numpy as np

from pisa.utils.format import timediff
from pisa.utils.log import logging, set_verbosity


__all__ = ['Timer', 'test_Timer']

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


# TODO: add unit tests!

class Timer(object):
    """Simple timer context (i.e. designed to be used via `with` sematics).

    Parameters
    ----------
    label
    verbose
    fmt_args : None or Mapping
        Passed to `timediff` via **fmt_args as optional format parameters.
        See that function for details of valid arguments

    """
    def __init__(self, label=None, verbose=False, fmt_args=None):
        self.label = label
        self.verbose = verbose
        self.fmt_args = fmt_args if fmt_args is not None else {}
        self.start = np.nan
        self.end = np.nan
        self.secs = np.nan
        self.msecs = np.nan

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000
        if self.verbose:
            formatted = timediff(dt_sec=self.secs, **self.fmt_args)
            logging.info('Elapsed time: ' + formatted)


def test_Timer():
    """Unit tests for Timer class"""
    with Timer(verbose=True):
        sleep(0.1)
    logging.info('<< PASS : test_Timer >>')


if __name__ == '__main__':
    set_verbosity(3)
    test_Timer()
