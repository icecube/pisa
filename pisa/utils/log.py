# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-10-17
"""
This module sets up the logging system by looking for a "logging.json"
configuration file. It will search (in this order) the local directory, $PISA
and finally the package resources. The loggers found in there will be lifted
to the module namespace.

Currently, we have three loggers
* root: generic for what is going on  (typically: `opening file x` or
  `doing this now` messages)
* physics: for any physics output that might be interesting
  (`have x many events`, `the flux is ...`)
* tprofile: for how much time it takes to run some step (in the format of
  `time : start bla`, `time : stop bla`)

The last one is temporary and should be replaced by a proper profiler.
"""


import logging
import logging.config


__all__ = ['logging', 'set_verbosity']


# Add a trace level
logging.TRACE = 5
logging.addLevelName(logging.TRACE, "TRACE")
def trace(self, message, *args, **kws):
    self.log(logging.TRACE, message, *args, **kws)
logging.Logger.trace = trace
logging.RootLogger.trace = trace
logging.trace = logging.root.trace

# Don't move these up, as "trace" might be used in them
import simplejson as json
from pkg_resources import resource_stream #pisa.utils.resources as resources

# Get the logging configuration
logconfig = json.load(
    resource_stream('pisa', 'resources/settings/logging/logging.json'),
)

# Setup the logging system with this config
logging.config.dictConfig(logconfig)

thandler = logging.StreamHandler()
tformatter = logging.Formatter(fmt=logconfig['formatters']['profile']['format'])
thandler.setFormatter(tformatter)

#capture warnings
logging.captureWarnings(True)

# Make the loggers public
# In case they haven't been defined, this will just inherit from the root logger
physics = logging.getLogger('physics')
tprofile = logging.getLogger('profile')
tprofile.handlers = [thandler]


def set_verbosity(verbosity):
    """Overwrite the verbosity level for the root logger
    Verbosity should be an integer with the levels just below.
    """
    # Ignore if no verbosity is given
    if verbosity is None:
        return

    # define verbosity levels
    levels = {0: logging.WARN,
              1: logging.INFO,
              2: logging.DEBUG,
              3: logging.TRACE}

    if verbosity not in levels:
        raise ValueError(
            '`verbosity` specified is %s but must be one of %s.'
            %(verbosity, levels.keys())
        )

    # Overwrite the root logger with the verbosity level
    logging.root.setLevel(levels[verbosity])
    tprofile.setLevel(levels[verbosity])
