"""
This module sets up the logging system by looking for a "logging.json"
configuration file. It will search (in this order) the local directory, $PISA
and finally the package resources. The loggers found in there will be lifted
to the module namespace.

Currently, we have three loggers
* logging: generic for what is going on  (typically: `opening file x` or
  `doing this now` messages)
* physics: for any physics output that might be interesting
  (`have x many events`, `the flux is ...`)
* tprofile: for how much time it takes to run some step (in the format of
  `time : start bla`, `time : stop bla`)
"""


from __future__ import absolute_import

import json
import logging as logging_module
import logging.config as logging_config
from pkg_resources import resource_stream


__all__ = ['logging', 'physics', 'tprofile', 'set_verbosity']

__author__ = 'S. Boeser'


def initialize_logging():
    """Intializing PISA logging"""
    # Add a trace level
    logging_module.TRACE = 5
    logging_module.addLevelName(logging_module.TRACE, 'TRACE')
    def trace(self, message, *args, **kws):
        """Trace-level logging"""
        self.log(logging_module.TRACE, message, *args, **kws)
    logging_module.Logger.trace = trace
    logging_module.RootLogger.trace = trace
    logging_module.trace = logging_module.root.trace

    # Get the logging configuration
    logconfig = json.load(
        resource_stream(
            'pisa_example_resources', 'settings/logging/logging.json'
        )
    )

    # Setup the logging system with this config
    logging_config.dictConfig(logconfig)

    thandler = logging_module.StreamHandler()
    tformatter = logging_module.Formatter(
        fmt=logconfig['formatters']['profile']['format']
    )
    thandler.setFormatter(tformatter)

    # Capture warnings
    logging_module.captureWarnings(True)

    _logging = logging_module.getLogger('pisa')
    _physics = logging_module.getLogger('pisa.physics')
    _tprofile = logging_module.getLogger('pisa.tprofile')
    # TODO: removed following line due to dupllicate logging messages. Probably
    # should fix this in a better manner...
    #_tprofile.handlers = [thandler]

    return _logging, _physics, _tprofile


def set_verbosity(verbosity):
    """Overwrite the verbosity level for the root logger
    Verbosity should be an integer with the levels just below.
    """
    # Ignore if no verbosity is given
    if verbosity is None:
        return

    # define verbosity levels
    levels = {0: logging_module.WARN,
              1: logging_module.INFO,
              2: logging_module.DEBUG,
              3: logging_module.TRACE}

    if verbosity not in levels:
        raise ValueError(
            '`verbosity` specified is %s but must be one of %s.'
            %(verbosity, levels.keys())
        )

    # Overwrite the root logger with the verbosity level
    logging.setLevel(levels[verbosity])
    tprofile.setLevel(levels[verbosity])


# Make the loggers public
logging, physics, tprofile = initialize_logging() # pylint: disable=invalid-name
