"""
Define globals available to all modules in PISA
"""


from __future__ import absolute_import

from collections import namedtuple, OrderedDict
import os
import sys
import warnings

import numba
from numba import jit as numba_jit

from numpy import (
    array, inf, nan,
    float32, float64,
    intp, int8, int16, int32, int64,
    uintp, uint8, uint16, uint32, uint64,
    complex64, complex128
)
# The alias only exists on Linux x86_64, see
# https://numpy.org/devdocs/reference/arrays.scalars.html#numpy.clongdouble
from numpy import clongdouble as complex256
import numpy as np
from pint import UnitRegistry

from ._version import get_versions


__author__ = 'J.L. Lanfranchi'

__license__ = '''Copyright (c) 2014-2026, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


# TODO: pisa.core names are _not_ included here, but possibly should be...?
__all__ = [
    # Versioneer needs this
    '__version__',

    # Utilities that must be accessed centrally for consistency
    'ureg', 'Q_',

    # Utilities that should be accessed centrally to avoid hassle
    'numba_jit',

    # Python standard library and Numpy names so that `eval(repr(x)) == x` for
    # all types defined in PISA (i.e. passes round trip test)
    'array', 'inf', 'nan', 'namedtuple', 'OrderedDict',
    'float32', 'float64',
    'intp', 'int8', 'int16', 'int32', 'int64',
    'uintp', 'uint8', 'uint16', 'uint32', 'uint64',
    'complex64', 'complex128', 'complex256',

    # Constants
    'NUMBA_CUDA_AVAIL',
    'TARGET',
    'OMP_NUM_THREADS',
    'PISA_NUM_THREADS',
    'PISA_HIST_THREADING',
    'FTYPE',
    'CTYPE',
    'ITYPE',
    'HASH_SIGFIGS',
    'EPSILON',
    'C_FTYPE',
    'C_PRECISION_DEF',
    'CACHE_DIR',
]


__version__ = get_versions()['version']
"""PISA version is automatically constructed from versioneer/git info"""


ureg = UnitRegistry()
"""Single Pint unit registry that should be used by all PISA code"""

Q_ = ureg.Quantity
"""Shortcut for Quantity that uses central PISA Pint unit regeistry"""


# Default value for CACHE_DIR
CACHE_DIR = '~/.cache/pisa'
"""Root directory for storing PISA cache files"""

# message later on written to stderr for user convenience
ini_msgs = []

# PISA users can define cache directory directly via PISA_CACHE_DIR env var;
# PISA_CACHE_DIR has priority over XDG_CACHE_HOME, so it is checked first
if 'PISA_CACHE_DIR' in os.environ:
    CACHE_DIR = os.environ['PISA_CACHE_DIR']

# Free Standards Group (freedesktop.org) defines the standard override for
# '~/.cache' to be set by XDG_CACHE_HOME env var; more info at
# https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
elif 'XDG_CACHE_HOME' in os.environ:
    CACHE_DIR = os.path.join(os.environ['XDG_CACHE_HOME'], 'pisa')

CACHE_DIR = os.path.expanduser(os.path.expandvars(CACHE_DIR))


# If `NUMBA_CACHE_DIR` env var is not set and `PISA_CACHE_DIR` is, then use
# `PISA_CACHE_DIR/numba` for caching Numba's compiled objects
if 'NUMBA_CACHE_DIR' not in os.environ:
    os.environ['NUMBA_CACHE_DIR'] = os.path.join(CACHE_DIR, 'numba')


# Default to single thread, then try to read from env
OMP_NUM_THREADS = 1
"""Number of threads OpenMP is allocated"""

if 'OMP_NUM_THREADS' in os.environ:
    OMP_NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
    assert OMP_NUM_THREADS >= 1

NUMBA_CUDA_AVAIL = False # pylint: disable=invalid-name
def dummy_func(x):
    """Decorate to to see if Numba actually works"""
    x += 1

try:
    from numba import cuda # pylint: disable=ungrouped-imports
    assert cuda.gpus, 'No GPUs detected'
    cuda.jit('void(float64)')(dummy_func)
except Exception:
    pass
else:
    NUMBA_CUDA_AVAIL = True # pylint: disable=invalid-name
finally:
    if 'cuda' in globals() or 'cuda' in locals():
        #if NUMBA_CUDA_AVAIL:
        #    cuda.close()
        del cuda
del dummy_func

# Default values for float, complex types
FTYPE = np.float64
"""Global floating-point data type. C, CUDA, and Numba datatype definitions are
derived from this"""

CTYPE = np.complex128
"""Global complex-valued floating-point data type. C, CUDA, and Numba datatype
definitions are derived from this"""

# Set FTYPE from environment variable PISA_FTYPE, if it is defined
FLOAT32_STRINGS = ['single', 'float32', 'fp32', '32', 'f4']
FLOAT64_STRINGS = ['double', 'float64', 'fp64', '64', 'f8']
if 'PISA_FTYPE' in os.environ:
    PISA_FTYPE = os.environ['PISA_FTYPE']
    #ini_msgs.append('PISA_FTYPE env var is defined as: "%s"' % PISA_FTYPE)
    if PISA_FTYPE.strip().lower() in FLOAT32_STRINGS:
        FTYPE = np.float32
        CTYPE = np.complex64
    elif PISA_FTYPE.strip().lower() in FLOAT64_STRINGS:
        FTYPE = np.float64
        CTYPE = np.complex128
    else:
        raise ValueError(
            'Environment var PISA_FTYPE="%s" is unrecognized.\n'
            '--> For single precision set PISA_FTYPE to one of %s\n'
            '--> For double precision set PISA_FTYPE to one of %s\n'
            %(PISA_FTYPE, FLOAT32_STRINGS, FLOAT64_STRINGS)
        )
ITYPE = np.int32 if FTYPE == np.float32 else np.int64
del FLOAT32_STRINGS, FLOAT64_STRINGS

# set default target
TARGET = 'cpu' # pylint: disable=invalid-name

cpu_targets = ['cpu', 'numba']
parallel_targets = ['parallel', 'multicore']
gpu_targets = ['cuda', 'gpu', 'numba-cuda']

if 'PISA_TARGET' in os.environ:
    PISA_TARGET = os.environ['PISA_TARGET']
    ini_msgs.append('PISA_TARGET env var is defined as: "%s"' % PISA_TARGET)
    try_target = PISA_TARGET.strip().lower()
    if try_target in gpu_targets:
        if NUMBA_CUDA_AVAIL:
            TARGET = 'cuda' # pylint: disable=invalid-name
        else:
            raise ValueError(
                'Environment var PISA_TARGET="%s" set, even though numba-cuda'
                ' is not available\n'%(PISA_TARGET)
            )
    elif try_target in cpu_targets or try_target in parallel_targets:
        if try_target in cpu_targets:
            TARGET = 'cpu' # pylint: disable=invalid-name
        elif try_target in parallel_targets:
            TARGET = 'parallel' # pylint: disable=invalid-name
    else:
        raise ValueError(
            'Environment var PISA_TARGET="%s" is unrecognized.\n'
            '--> For cpu set PISA_TARGET to one of %s\n'
            '--> For parallel set PISA_TARGET to one of %s\n'
            '--> For gpu set PISA_TARGET to one of %s\n'
            %(PISA_TARGET, cpu_targets, parallel_targets, gpu_targets)
        )
    del try_target

del cpu_targets, gpu_targets, parallel_targets

# Default to single thread, then try to read from env
PISA_NUM_THREADS = 1
"""Global limit for number of threads"""

if 'PISA_NUM_THREADS' in os.environ:
    PISA_NUM_THREADS = int(os.environ['PISA_NUM_THREADS'])
    assert PISA_NUM_THREADS >= 1
    if TARGET == 'cpu' and PISA_NUM_THREADS > 1:
        sys.stderr.write("[WARNING] PISA_NUM_THREADS > 1 will be ignored when "
                         "PISA_TARGET is not `parallel`.\n")
        PISA_NUM_THREADS = 1
elif TARGET == 'parallel':
    PISA_NUM_THREADS = numba.config.NUMBA_NUM_THREADS # pylint: disable=no-member

if TARGET == 'cpu':
    # making sure that we can definitely rely on the fact that the number of threads
    # will be 1 if the TARGET is `cpu` (some stages might do that)
    assert PISA_NUM_THREADS == 1

# initialize numba thread count immediately
numba.set_num_threads(PISA_NUM_THREADS)

# final choice for OpenMP number of threads
OMP_NUM_THREADS = min(PISA_NUM_THREADS, OMP_NUM_THREADS)

PISA_HIST_THREADING = 'off' # pylint: disable=invalid-name
"""Granular control of strategy for threading in PISA (fast-)histogram operations.
Choices:

- 'off' (default): Disable threading completely (as it adds significant
  overhead to typical applications).
- 'auto': Use :py:data:`PISA_NUM_THREADS` when `TARGET='parallel'`, else no threading.
- N (integer > 0): Use N threads for histogram operations regardless of `TARGET`.

Set via `PISA_HIST_THREADING` environment variable."""

if 'PISA_HIST_THREADING' in os.environ:
    pisa_hist_threading = os.environ['PISA_HIST_THREADING'].strip().lower()
    if pisa_hist_threading in ('auto', ''):
        PISA_HIST_THREADING = 'auto' # pylint: disable=invalid-name
    elif pisa_hist_threading == '0' or pisa_hist_threading in ('false', 'off', 'no'):
        PISA_HIST_THREADING = 'off' # pylint: disable=invalid-name
    else:
        try:
            PISA_HIST_THREADING = int(pisa_hist_threading) # pylint: disable=invalid-name
            if PISA_HIST_THREADING <= 0:
                raise ValueError('must be > 0')
        except ValueError as exc:
            raise ValueError(
                f'Environment variable PISA_HIST_THREADING="{pisa_hist_threading}" '
                'unrecognized:\n'
                '  For "auto" mode (PISA_NUM_THREADS) set to "auto" or empty string.\n'
                '  To disable histogram threading set to "0", "false", "no", or "off".\n'
                '  For N threads (precedence over PISA_NUM_THREADS) set to integer > 0.'
            ) from exc

# Define HASH_SIGFIGS to set hashing precision based on FTYPE above; value here
# is default (i.e. for FTYPE == np.float64)
HASH_SIGFIGS = 12 # pylint: disable=invalid-name
"""Round to this many significant figures for hashing numbers, such that
machine precision doesn't cause effectively equivalent numbers to hash
differently."""

if FTYPE == np.float32:
    HASH_SIGFIGS = 5 # pylint: disable=invalid-name

EPSILON = 10**(-HASH_SIGFIGS)
"""Best precision considering HASH_SIGFIGS (which is chosen kinda ad-hoc but
based on by FTYPE)"""


# Derive #define consts for dynamically-compiled C (and also C++ and CUDA) code
# to use.
#
# To use these in code, put in the C/C++/CUDA the following at the TOP of your
# code:
#
#   from pisa import FTYPE, C_FTYPE, C_PRECISION_DEF
#
#   ...
#
#   dynamic_source_code = '''
#   #define fType %(C_FTYPE)s
#   #define %(C_PRECISION_DEF)s
#
#   ...
#
#   ''' % dict(C_FTYPE=C_FTYPE, C_PRECISION_DEF=C_PRECISION_DEF)
#
if FTYPE == np.float32:
    C_FTYPE = 'float'
    C_PRECISION_DEF = 'SINGLE_PRECISION'
    ftype_msg = 'PISA is running in single precision (FP32) mode' # pylint: disable=invalid-name
elif FTYPE == np.float64:
    C_FTYPE = 'double'
    C_PRECISION_DEF = 'DOUBLE_PRECISION'
    ftype_msg = 'PISA is running in double precision (FP64) mode' # pylint: disable=invalid-name
else:
    raise ValueError('FTYPE must be one of `np.float32` or `np.float64`. Got'
                     f' {FTYPE} instead.')
ini_msgs.append(ftype_msg)
del ftype_msg

if TARGET is None:
    target_msg = 'numba is not present' # pylint: disable=invalid-name
elif TARGET == 'cpu':
    target_msg = 'numba is running on CPU (single core)' # pylint: disable=invalid-name
elif TARGET == 'parallel':
    target_msg = f'numba is running on CPU (multicore) with {PISA_NUM_THREADS} cores' # pylint: disable=invalid-name
elif TARGET == 'cuda':
    target_msg = 'numba is running on GPU' # pylint: disable=invalid-name
else:
    raise ValueError('TARGET must be one of `None`, "cpu", "parallel", or "cuda".'
                     f' Got {TARGET} instead.')
ini_msgs.append(target_msg)
del target_msg

if PISA_HIST_THREADING == 'auto':
    hist_threading_msg = 'PISA histogram threading is in auto mode (PISA_NUM_THREADS)' # pylint: disable=invalid-name
elif PISA_HIST_THREADING == 'off':
    hist_threading_msg = 'PISA histogram threading is disabled' # pylint: disable=invalid-name
else:
    hist_threading_msg = f'PISA histogram threading will use {PISA_HIST_THREADING} thread(s)' # pylint: disable=invalid-name
ini_msgs.append(hist_threading_msg)
del hist_threading_msg

sys.stderr.write("<< "+"; ".join(ini_msgs)+" >>\n")
del ini_msgs

# Clean up imported names
del os, sys, np, UnitRegistry, get_versions
