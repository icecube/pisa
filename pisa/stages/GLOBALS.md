# Global constants

Global variables and constants are defined upon initialization of the `pisa` package (`pisa/__init__.py`) and are available to all of its modules.
They can be imported via `from pisa import <constant>`.

Here we keep track of which global constants are available, what their purpose is, and by which stages they are used.

## Description

| Constant           | Description                                                               | Default                                                               | Overwritten by environment variables (priority indicated where necessary) |
| ------------------ | ------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `NUMBA_CUDA_AVAIL` | Availability of Numba's CUDA interface                                    | `False` (unless CUDA-capable GPU available)                           |                                                                           |
| `TARGET`           | Numba compilation target                                                  | `cpu`                                                                 | `PISA_TARGET` (GPU target only possible if `NUMBA_CUDA_AVAIL`)                                                              |
| `OMP_NUM_THREADS`  | Number of threads allocated to OpenMP                                     | `1`                                                                   | `OMP_NUM_THREADS`                                                         |
| `PISA_NUM_THREADS` | Global limit for number of threads (also upper limit for `OMP_NUM_THREADS`) | `1` (`numba.config.NUMBA_NUM_THREADS`) for `TARGET='cpu'` (`'parallel'`) | `PISA_NUM_THREADS`                                                         |
| `FTYPE`            | Global floating-point data type                                           | `np.float64`                                                          | `PISA_FTYPE`                                                              |
| `CTYPE`            | Global complex-valued floating-point data type                            | `np.complex128` (`np.complex64`) for `FTYPE=np.float64(32)`            |                                                                  |
| `ITYPE`            | Global integer data type                                                  | `np.int64` (`np.int32`) for `FTYPE=np.float64(32)`                    |                                                                  |
| `HASH_SIGFIGS`     | Number of significant digits used for hashing numbers, depends on `FTYPE` | `12(5)` for `FTYPE=np.float64(32)`                                    |                                                                           |
| `EPSILON`          | Best numerical precision, derived from `HASH_SIGFIGS`                     | `10**(-HASH_SIGFIGS)`                                                 |                                                                           |
| `C_FTYPE`          | C floating-point type corresponding to `FTYPE`                            | `'double'` (`'float'`) for `FTYPE=np.float64(32)`                       |                                                                           |
| `C_PRECISION_DEF`  | C precision of floating-point calculations, derived from `FTYPE`          | `'DOUBLE_PRECISION'` (`'SINGLE_PRECISION'`) for `FTYPE=np.float64(32)`   |                                                                           |
| `CACHE_DIR`        | Root directory for storing PISA cache files                               | `'~/.cache/pisa'`                                                     | 1.`PISA_CACHE_DIR`, 2.`XDG_CACHE_HOME/pisa`                               |

## Usage
The table below depicts which services make use of a select set of global constants.
Note that the table entries are derived both from the module files themselves (where the services are defined) and from any `pisa.utils` objects they make use of (in particular, reliance on "PISA-tailored" jit in `numba_tools`).
Constants which are implicitly used by all services via `pisa.core` objects (e.g. `HASH_SIGFIGS`, `CACHE_DIR`) are not shown.
Also note that where a service implements `FTYPE` and relies on C extension code, the simultaneous implementation of `C_FTYPE` and `C_PRECISION_DEF` is implied.

**Legend**
- :heavy_check_mark:: implements
- :heavy_minus_sign:: does not implement but does not fail (i.e., ignores)

|                            | `TARGET`    | `PISA_NUM_THREADS`     | `FTYPE`               |
| :------------------------: | :-------------------: | :-------------------: | :-------------------: |
| `absorption.earth_absorption` | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `aeff.aeff`                   | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: |
| `aeff.weight`                 | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: |
| `aeff.weight_hnl`             | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: |
| `background.atm_muons`        | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: |
| `data.csv_data_hist`          | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `data.csv_icc_hist`           | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `data.csv_loader`             | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `data.freedom_hdf5_loader`    | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `data.grid`                   | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `data.licloader_weighter`     | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `data.meows_loader`           | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `data.simple_data_loader`     | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `data.simple_signal`          | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `data.sqlite_loader`          | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `data.toy_event_generator`    | :heavy_minus_sign:    | :heavy_minus_sign: | :heavy_check_mark: |
| `discr_sys.hypersurfaces`     | :heavy_minus_sign:    | :heavy_minus_sign: | :heavy_check_mark:    |
| `discr_sys.ultrasurfaces`     | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `flux.airs`                   | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `flux.astrophysical`          | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `flux.barr_simple`            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| `flux.daemon_flux`            | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `flux.hillasg`                | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `flux.honda_ip`               | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `flux.mceq_barr`              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| `flux.mceq_barr_red`          | :heavy_check_mark:    | :heavy_minus_sign: | :heavy_check_mark:    |
| `likelihood.generalized_llh_params`   | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `osc.decoherence`             | :heavy_check_mark:    | :heavy_check_mark: | :heavy_check_mark:    |
| `osc.globes`                  | :heavy_check_mark:    | :heavy_check_mark: | :heavy_check_mark:    |
| `osc.nusquids`                | :heavy_minus_sign:    | :heavy_check_mark: | :heavy_check_mark:    |
| `osc.prob3`                   | :heavy_check_mark:    | :heavy_check_mark: | :heavy_check_mark:    |
| `osc.two_nu_osc`              | :heavy_check_mark:    | :heavy_check_mark: | :heavy_check_mark:    |
| `reco.resolutions`            | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: |
| `reco.simple_param`           | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_check_mark: |
| `utils.add_indices`           | :heavy_minus_sign:    | :heavy_minus_sign: | :heavy_minus_sign:    |
| `utils.adhoc_sys`             | :heavy_minus_sign:    | :heavy_minus_sign: | :heavy_check_mark:    |
| `utils.bootstrap`             | :heavy_minus_sign:    | :heavy_minus_sign: | :heavy_minus_sign:    |
| `utils.fix_error`             | :heavy_minus_sign:    | :heavy_minus_sign: | :heavy_check_mark:    |
| `utils.hist`                  | :heavy_minus_sign:    | :heavy_minus_sign: | :heavy_minus_sign:    |
| `utils.kde`                   | :heavy_minus_sign:    | :heavy_minus_sign: | :heavy_minus_sign:    |
| `utils.kfold`                 | :heavy_minus_sign:    | :heavy_minus_sign: | :heavy_check_mark:    |
| `utils.resample`              | :heavy_check_mark:    | :heavy_check_mark: | :heavy_check_mark:    |
| `utils.set_variance`          | :heavy_check_mark:    | :heavy_check_mark: | :heavy_check_mark:    |
| `xsec.dis_sys`                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| `xsec.genie_sys`              | :heavy_minus_sign:    | :heavy_minus_sign: | :heavy_minus_sign:    |
| `xsec.nutau_xsec`             | :heavy_check_mark:    | :heavy_check_mark: | :heavy_check_mark:    |
