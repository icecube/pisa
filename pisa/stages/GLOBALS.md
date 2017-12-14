# Global variables

Global variables are defined upon initialization of the `pisa` package (`pisa/__init__.py`) and are available to all of its modules. Here we keep track of which global
variables (including constants) are available, what their purpose is, and by which stage(s) they are used.

## Global constants overview

| Constant           | Description                                                                      | Default                    | Overwritten by environment variable(s) ( in this order) |
| ---                | ---                                                                              | ---                        | ---                               |
| `PYCUDA_AVAIL`     | Availability of pycuda                                                           | `False` (unless installed) |                                   |
| `NUMBA_AVAIL`      | Availability of numba                                                            | `False` (unless installed) |                                   |
| `NUMBA_CUDA_AVAIL` | Availability of cuda through numba (`False` if no GPUs are detected in any case) | `False` (unless installed) |                                   |
| `OMP_NUM_THREADS`  | Number of threads allocated to OpenMP                                            | `1`                        | `OMP_NUM_THREADS`                 |
| `FTYPE`            | Global floating-point data type                                                  | `np.float64`               | `PISA_FTYPE`                      |
| `HASH_SIGFIGS`     | Number of significant digits used for hashing numbers, depends on `FTYPE`        | `12`                       |                                   |
| `EPSILON`          | Best numerical precision, derived from `HASH_SIGFIGS`                            | `10**(-HASH_SIGFIGS)`      |                                   |
| `C_FTYPE`          | Numerical C type corresponding to `FTYPE`                                        | `'double'`                 |                                   |
| `C_PRECISION_DEF`  | C precision of floating point calculations, derived from `FTYPE`                 | `'DOUBLE_PRECISION'`       |                                   |
| `CACHE_DIR`        | Root directory for storing PISA cache files                                      | `'~/.cache/pisa'`          | `PISA_CACHE_DIR`, `XDG_CACHE_HOME`|

## Global constants usage

:heavy_check_mark:: implements
:black_square_button:: does not implement but does not fail
:heavy_exclamation_mark:: fails

| Constant | `PYCUDA_AVAIL`  | `NUMBA_AVAIL` | `NUMBA_CUDA_AVAIL` | `OMP_NUM_THREADS` | `FTYPE` |
| ---      | ---             | ---           | ---                | ---               | ---     |
| `aeff.hist`    | | | | | | | | | | |
| `aeff.param`   | | | | | | | | | | |
| `aeff.smooth`  | | | | | | | | | | |
| `osc.prob3cpu` | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: | :black_square_button: |
| `osc.prob3gpu` | :heavy_exclamation_mark: | :black_square_button: | :black_square_button: | :black_square_button: | :heavy_check_mark:    |
