# Stage: Discrete Systematics

This stage applies parameterized systematics to the templates. 

## Services

Only one service is currently supported in PISA.

### fits

This service applies the results obtained from fits to discrete samples.

The fitting parameters are at the moment extracted by an external
script, that saves them in a json file, see below.

Any parameterized systematic needs to be added to the `[stage:sys]` section of the pipeline config. There the associated nuisance parameters (can be
N different ones), e.g. `hole_ice` are specified together with a parameter
`hole_ice_file` pointing to the `.json` file with the fit info.

### generating the fit values

To generate the fit file, the script `$PISA/pisa/utils/fit_discrerte_sys.py` can
be executed together with a special configuration file. An example as used in
the nutau analysis is found under `$PISA/resources/settings/discrete_sys_settings/nutau_holice_domeff_fits.ini`

This config file specifies the discrete datasets for the fits, here an example:

```
[dom_eff]
nominal = 1.0
degree = 1
force_through_nominal = True
smooth = gauss
; discrete sets for param values
runs = [1.0, 0.88, 0.94, 0.97, 1.03, 1.06, 1.12]
```

That means the systematic `dom_eff` is parametrized from 7 discrete datasets, with the nominal point being at `dom_eff=1.0`, parametrized with a linear fit that is forced through the nominal point, and gaussian smoothing is applied.

All 7 datasets must be specified in a separate section.

At the moment different fits are generated for `cscd` and `trck` maps only (they are added together for the fit). Systematics listed under `sys_list` are considered in the fit. This will generate N different `.json` for N systematics. All the info from the fit, including the fit function itself is stored in that file. Plotting is also available via `-p/--plot' and is HIGHLY recomended to inspect the fit results.
