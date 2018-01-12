# pisa.scripts

Analysis and helper scripts (executables).


## Directory listing

| File/directory                        | Description
| ------------------------------------- | -----------
| `__init__.py`                         | Make `pisa/scripts` directory behave as a Python module
| `README.md`                           | This file
| `add_flux_to_events_file.py`          | Reweighted-MC analyses need (or at least go much faster if) fluxes attached to each event; that's what this does
| `analysis.py`                         | Run analyses
| `analysis_postprocess.py`             | Postprocess data produced by `analysis.py`
| `compare.py`                          | Compare maps, map sets, pipelines, ... at the command line, producing plots and summary text
| `convert_config_format.py`            | Convert old PISA config format to new format
| `discrete_hypo_test.py`               | 
| `fit_discrete_sys.py`                 | 
| `inj_param_scan.py`                   | 
| `make_asymmetry_plots.py`             | 
| `make_events_file.py`                 | Generate a PISA events (HDF5) file from an IceTray-produced HDF5 file
| `make_nufit_theta23_spline_priors.py` | 
| `make_systematic_variation_plots.py`  | 
| `make_toy_events.py`                  | Generate a PISA events (HDF5) file from the toy very large volume neutrino telescope (VLVNT) detector model
| `profile_scan.py`                     | Script that uses the `scan` function of the `Analysis` class to sample likelihood surfaces over an arbitrary number of dimensions. The scan parameters are set by the user and can include minimisation over a set of systematics.
| `scan_allsyst.py`                     | Similar to `profile_scan.py`, but scans each free param separately (i.e., no profile)
| `smooth_pid.py`                       | Produce smooth PID parameterizations given a PISA events file (for use with the `stages.pid.smooth` service)
| `systematics_tests.py`                | 
| `test_flux_weights.py`                | 
