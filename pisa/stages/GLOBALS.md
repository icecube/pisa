# Global constants

Global variables and constants are defined upon initialization of the `pisa` package (`pisa/__init__.py`) and are available to all of its modules.
They can be imported via `from pisa import <constant>`.

Here we keep track of which global constants are available, what their purpose is, and by which stages they are used.

## Description

| Constant              | Description                                                               | Default                                                               | Overwritten by environment variables (priority indicated where necessary) |
| -------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `NUMBA_CUDA_AVAIL`    | Availability of Numba's CUDA interface                                    | `False` (unless CUDA-capable GPU available)                           |                                                                           |
| `TARGET`              | Numba compilation target                                                  | `cpu`                                                                 | `PISA_TARGET` (GPU target only possible if `NUMBA_CUDA_AVAIL`)            |
| `OMP_NUM_THREADS`     | Number of threads allocated to OpenMP                                     | `1`                                                                   | `OMP_NUM_THREADS`                                                         |
| `PISA_NUM_THREADS`    | Global limit for number of threads (also upper limit for `OMP_NUM_THREADS`) | `1` (`numba.config.NUMBA_NUM_THREADS`) for `TARGET='cpu'` (`'parallel'`) | `PISA_NUM_THREADS`                                                         |
| `PISA_HIST_THREADING` | Multi-threading mode for PISA (fast) histogramming                        | `'off'`                                                               | `PISA_HIST_THREADING`                                                     |
| `FTYPE`               | Global floating-point data type                                           | `np.float64`                                                          | `PISA_FTYPE`                                                              |
| `CTYPE`               | Global complex-valued floating-point data type                            | `np.complex128` (`np.complex64`) for `FTYPE=np.float64(32)`           |                                                                           |
| `ITYPE`               | Global integer data type                                                  | `np.int64` (`np.int32`) for `FTYPE=np.float64(32)`                    |                                                                           |
| `HASH_SIGFIGS`        | Number of significant digits used for hashing numbers, depends on `FTYPE` | `12(5)` for `FTYPE=np.float64(32)`                                    |                                                                           |
| `EPSILON`             | Best numerical precision, derived from `HASH_SIGFIGS`                     | `10**(-HASH_SIGFIGS)`                                                 |                                                                           |
| `C_FTYPE`             | C floating-point type corresponding to `FTYPE`                            | `'double'` (`'float'`) for `FTYPE=np.float64(32)`                     |                                                                           |
| `C_PRECISION_DEF`     | C precision of floating-point calculations, derived from `FTYPE`          | `'DOUBLE_PRECISION'` (`'SINGLE_PRECISION'`) for `FTYPE=np.float64(32)`|                                                                           |
| `CACHE_DIR`           | Root directory for storing PISA cache files                               | `'~/.cache/pisa'`                                                     | 1.`PISA_CACHE_DIR`, 2.`XDG_CACHE_HOME/pisa`                               |

## Usage

The table below depicts which services make use of a select set of global constants.
Note that the table entries are derived both from the module files themselves (where the services are defined) and from any `pisa.utils` objects they make use of (e.g., reliance on "PISA-tailored" jit in `numba_tools`).
Constants which are implicitly used by all services via `pisa.core` objects (e.g. `HASH_SIGFIGS`, `CACHE_DIR`) are not shown.
Also note that where a service implements `FTYPE` and relies on C extension code, the simultaneous implementation of `C_FTYPE` and `C_PRECISION_DEF` is implied.

**Legend**
- ✓: implements
- ✗: does not implement but does not fail (i.e., ignores)

|                            | `TARGET` | `PISA_NUM_THREADS` | `FTYPE` | `PISA_HIST_THREADING` |
| :------------------------: | :------: | :----------------: | :-----: | :-------------------: |
| `absorption.earth_absorption` | ✗ | ✗ | ✓ | ✗ |
| `aeff.aeff`                   | ✗ | ✗ | ✗ | ✗ |
| `aeff.weight`                 | ✗ | ✗ | ✗ | ✗ |
| `aeff.weight_hnl`             | ✗ | ✗ | ✗ | ✗ |
| `background.atm_muons`        | ✗ | ✗ | ✗ | ✗ |
| `cont_sys.snowstorm_hist`     | ✓ | ✓ | ✓ | ✓ |
| `data.csv_data_hist`          | ✗ | ✗ | ✓ | ✗ |
| `data.csv_icc_hist`           | ✗ | ✗ | ✓ | ✗ |
| `data.csv_loader`             | ✗ | ✗ | ✓ | ✗ |
| `data.freedom_hdf5_loader`    | ✗ | ✗ | ✓ | ✗ |
| `data.grid`                   | ✗ | ✗ | ✓ | ✗ |
| `data.licloader_weighter`     | ✗ | ✗ | ✓ | ✗ |
| `data.meows_loader`           | ✗ | ✗ | ✓ | ✗ |
| `data.simple_data_loader`     | ✗ | ✗ | ✓ | ✗ |
| `data.simple_signal`          | ✗ | ✗ | ✓ | ✗ |
| `data.sqlite_loader`          | ✗ | ✗ | ✓ | ✗ |
| `data.toy_event_generator`    | ✗ | ✗ | ✓ | ✗ |
| `discr_sys.hypersurfaces`     | ✗ | ✗ | ✓ | ✗ |
| `discr_sys.ultrasurfaces`     | ✗ | ✗ | ✓ | ✗ |
| `flux.airs`                   | ✗ | ✗ | ✓ | ✗ |
| `flux.astrophysical`          | ✗ | ✗ | ✓ | ✗ |
| `flux.barr_simple`            | ✓ | ✓ | ✓ | ✗ |
| `flux.daemon_flux`            | ✗ | ✗ | ✓ | ✗ |
| `flux.hillasg`                | ✗ | ✗ | ✓ | ✗ |
| `flux.honda_ip`               | ✗ | ✗ | ✓ | ✗ |
| `flux.mceq_barr`              | ✓ | ✓ | ✓ | ✗ |
| `flux.mceq_barr_red`          | ✓ | ✗ | ✓ | ✗ |
| `likelihood.generalized_llh_params`   | ✗ | ✗ | ✓ | ✗ |
| `osc.decoherence`             | ✓ | ✓ | ✓ | ✗ |
| `osc.globes`                  | ✓ | ✓ | ✓ | ✗ |
| `osc.nusquids`                | ✗ | ✓ | ✓ | ✗ |
| `osc.prob3`                   | ✓ | ✓ | ✓ | ✗ |
| `osc.two_nu_osc`              | ✓ | ✓ | ✓ | ✗ |
| `reco.resolutions`            | ✗ | ✗ | ✗ | ✗ |
| `reco.simple_param`           | ✗ | ✗ | ✓ | ✗ |
| `utils.add_indices`           | ✗ | ✗ | ✗ | ✗ |
| `utils.adhoc_sys`             | ✗ | ✗ | ✓ | ✗ |
| `utils.bootstrap`             | ✗ | ✗ | ✗ | ✗ |
| `utils.fix_error`             | ✗ | ✗ | ✓ | ✗ |
| `utils.hist`                  | ✓ | ✓ | ✓ | ✓ |
| `utils.kde`                   | ✗ | ✗ | ✗ | ✗ |
| `utils.kfold`                 | ✗ | ✗ | ✓ | ✗ |
| `utils.resample`              | ✓ | ✓ | ✓ | ✗ |
| `utils.set_variance`          | ✓ | ✓ | ✓ | ✗ |
| `xsec.dis_sys`                | ✓ | ✓ | ✓ | ✗ |
| `xsec.genie_sys`              | ✗ | ✗ | ✗ | ✗ |
| `xsec.nutau_xsec`             | ✓ | ✓ | ✓ | ✗ |

