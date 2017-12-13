# Global variables

Global variables are defined upon initialization of the `pisa` package (`pisa/__init__.py`) and are available to all of its modules. Here we keep track of which global
variables (including constants) are available, what their purpose is, and by which stage(s) they are used.

## Global constants

| Constant | Description |
| --- | --- |
| `PYCUDA_AVAIL` | Availability of pycuda |
| `NUMBA_AVAIL` | Availability of numba |
| `NUMBA_CUDA_AVAIL` | Availability of cuda through numba (`False` if no GPUs are detected in any case) |
| `OMP_NUM_THREADS` | Number of threads allocated to OpenMP |
| `FTYPE` | Global floating-point data type |
| `HASH_SIGFIGS` | Number of significant digits used for hashing numbers, depends on `FTYPE` |
| `EPSILON` | Best numerical precision, derived from `HASH_SIGFIGS` |
| `C_FTYPE` | Corresponding numerical C type to `FTYPE`  |
| `C_PRECISION_DEF` | C precision of floating point calculations, derived from `FTYPE` |
| `CACHE_DIR` | Root directory for storing PISA cache files |
