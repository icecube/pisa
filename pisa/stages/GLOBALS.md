# Global constants

Global variables and constants are defined upon initialization of the `pisa` package (`pisa/__init__.py`) and are available to all of its modules.
They can be imported via `from pisa import <constant>`.

Here we keep track of which global constants are available, what their purpose is, and by which stage(s) they are used.

## Description

| Constant           | Description                                                               | Default                                                               | Overwritten by environment variables (priority indicated where necessary) |
| ------------------ | ------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `NUMBA_AVAIL`      | Availability of Numba                                                     | `False` (unless installed)                                            |                                                                           |
| `NUMBA_CUDA_AVAIL` | Availability of Numba's CUDA interface                                    | `False` (unless installed and CUDA-capable GPU available)             |                                                                           |
| `TARGET`           | Numba compilation target                                                  | `cpu` if `NUMBA_AVAIL`, `gpu` if `NUMBA_CUDA_AVAIL`, `None` otherwise | `PISA_TARGET`                                                             |
| `OMP_NUM_THREADS`  | Number of threads allocated to OpenMP                                     | `1`                                                                   | `OMP_NUM_THREADS`                                                         |
| `FTYPE`            | Global floating point data type                                           | `np.float64`                                                          | `PISA_FTYPE`                                                              |
| `HASH_SIGFIGS`     | Number of significant digits used for hashing numbers, depends on `FTYPE` | `12(5)` for `FTYPE=np.float64(32)`                                    |                                                                           |
| `EPSILON`          | Best numerical precision, derived from `HASH_SIGFIGS`                     | `10**(-HASH_SIGFIGS)`                                                 |                                                                           |
| `C_FTYPE`          | C floating point type corresponding to `FTYPE`                            | `'double'('single')` for `FTYPE=np.float64(32)`                       |                                                                           |
| `C_PRECISION_DEF`  | C precision of floating point calculations, derived from `FTYPE`          | `'DOUBLE_PRECISION'('SINGLE_PRECISION')` for `FTYPE=np.float64(32)`   |                                                                           |
| `CACHE_DIR`        | Root directory for storing PISA cache files                               | `'~/.cache/pisa'`                                                     | 1.`PISA_CACHE_DIR`, 2.`XDG_CACHE_HOME/pisa`                               |

## Usage
The table below depicts which services make use of a select set of global constants.
Note that the table entries are derived from both the module files themselves (where the services are defined) but also from any `pisa.utils` objects they make use of.
Constants which are implicitly used by all services via `pisa.core` objects (e.g. `HASH_SIGFIGS`, `CACHE_DIR`) are not shown.
Also note that where a service implements `FTYPE` and relies on C extension code, the simultaneous implementation of `C_FTYPE` and `C_PRECISION_DEF` is implied.

**Legend**
- :heavy_check_mark:: implements
- :black_square_button:: does not implement but does not fail (i.e., ignores)
- :heavy_exclamation_mark:: implements and fails if `False` (i.e., depends)

|                            | `NUMBA_AVAIL`            | `NUMBA_CUDA_AVAIL`    | `OMP_NUM_THREADS`     | `FTYPE`               |
| :------------------------: | :----------------------: | :-------------------: | :-------------------: | :-------------------: |
| `aeff.hist`                | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `aeff.param`               | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `aeff.pi_aeff`             | :heavy_exclamation_mark: | :heavy_check_mark:    | :black_square_button: | :heavy_check_mark:    |
| `aeff.smooth`              | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `combine.nutau`            | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `data.data`                | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `data.events_to_data`      | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `data.icc`                 | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `data.sample`              | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `data.simple_data_loader`  | :heavy_exclamation_mark: | :heavy_check_mark:    | :black_square_button: | :heavy_check_mark:    |
| `data.toy_event_generator` | :heavy_exclamation_mark: | :heavy_check_mark:    | :black_square_button: | :heavy_check_mark:    |
| `discr_sys.fit`            | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `discr_sys.hyperplane`     | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `discr_sys.pi_hyperplanes` | :heavy_exclamation_mark: | :heavy_check_mark:    | :black_square_button: | :heavy_check_mark:    |
| `discr_sys.polyfits`       | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `flux.dummy`               | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `flux.honda`               | :heavy_check_mark:       | :black_square_button: | :black_square_button: | :black_square_button: |
| `flux.mceq`                | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `flux.pi_barr_simple`      | :heavy_exclamation_mark: | :heavy_check_mark:    | :black_square_button: | :heavy_check_mark:    |
| `osc.pi_prob3`             | :heavy_exclamation_mark: | :heavy_check_mark:    | :black_square_button: | :heavy_check_mark:    |
| `pid.hist`                 | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `pid.param`                | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `pid.smooth`               | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `reco.hist`                | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `reco.hist`                | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `reco.param`               | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `reco.vbwkde`              | :heavy_check_mark:       | :heavy_check_mark:    | :heavy_check_mark:    | :heavy_check_mark:    |
| `unfold.roounfold`         | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `utils.pi_hist`            | :heavy_exclamation_mark: | :heavy_check_mark:    | :black_square_button: | :heavy_check_mark   : |
| `xsec.genie`               | :black_square_button:    | :black_square_button: | :black_square_button: | :black_square_button: |
| `xsec.genie_sys`           | :heavy_exclamation_mark: | :heavy_check_mark:    | :black_square_button: | :heavy_check_mark:    |
