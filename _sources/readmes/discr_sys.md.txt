# Stage: Discrete Systematics

These stages apply parameterized systematics to the templates.

## Services

### Hypersurfaces

#### fits

This service applies the results obtained from fits to discrete samples.

The fitting parameters are at the moment extracted by an external
script, that saves them in a json file, see below.

Any parameterized systematic needs to be added to the `[stage:sys]` section of the pipeline config.
There the associated nuisance parameters (can be N different ones), e.g. `hole_ice` are specified together with a parameter `hole_ice_file` pointing to the `.json` file with the fit info.

#### generating the fit values

To generate the fit file, the script `$PISA/pisa/scripts/fit_discrerte_sys.py` (command-line alias `pisa-fit_discrete_sys`) can be executed together with a special configuration file.

This config file specifies the discrete datasets for the fits, here an example:

```ini
[dom_eff]
nominal = 1.0
degree = 1
force_through_nominal = True
smooth = gauss
# discrete sets for param values
runs = [1.0, 0.88, 0.94, 0.97, 1.03, 1.06, 1.12]
```

That means the systematic `dom_eff` is parametrized from 7 discrete datasets, with the nominal point being at `dom_eff=1.0`, parametrized with a linear fit that is forced through the nominal point, and gaussian smoothing is applied.

All 7 datasets must be specified in a separate section.

At the moment different fits are generated for `cscd` and `trck` maps only (they are added together for the fit).
Systematics listed under `sys_list` are considered in the fit.
This will generate N different `.json` for N systematics.
All the info from the fit, including the fit function itself is stored in that file.
Plotting is also available via `-p/--plot' and is HIGHLY recomended to inspect the fit results.

### Ultrasurfaces

This is the novel treatment of detector systematics via likelihood-free inference. It assigns gradients to _every event_ to allow event-by-event re-weighting as a function of detector uncertainties in a way that is fully decoupled from flux and oscillation effects.

Once ready, the results are stored in a `.feather` file containing all events of the nominal MC set and their associated gradients.

### Preparation

The scripts producing the gradients are located in `$FRIDGE_DIR/analysis/oscnext_ultrasurfaces`. To produce the gradient feather file, we first need to convert PISA HDF5 files to `.feather` using the `pisa_to_feather.py` script. We need to pass the input file, the output file, and a flag setting the sample (variable names) to be used (either `--verification-sample`, `--flercnn-sample`, `--flercnn-hnl-sample`, `--upgrade-sample`, or no additional flag for the Retro sample).

```
python pisa_to_feather.py -i /path/to/pisa_hdf5/oscnext_genie_0151.hdf5 -o /path/to/pisa_hdf5/oscnext_genie_0151.feather {"--verification-sample", "--flercnn-sample", ""}
```
After converting all files and setting the appropriate paths in `$FRIDGE_DIR/analysis/oscnext_ultrasurfaces/datasets/data_loading.py`, we produce gradients in two steps.

**First**: Calculate event-wise probabilities with (assuming we `cd`'d into `$FRIDGE_DIR/analysis/oscnext_ultrasurfaces/knn`)

(Note here that this needs to be run with an earlier version of sklearn, due to deprecation of some used functions, e.g. use: `scikit-learn = 1.1.2`)

```
python calculate_knn_probs.py --data-sample {"verification", "flercnn", "flercnn_hnl", "retro"} --root-dir /path/to/pisa_feather/ --outfile /path/to/ultrasurface_fits/genie_all_bulkice_pm10pc_knn_200pc.feather --neighbors-per-class 200 --datasets 0000 0001 0002 0003 0004 0100 0101 0102 0103 0104 0105 0106 0107 0109 0151 0500 0501 0502 0503 0504 0505 0506 0507 --jobs 24
```

**Second**: Calculate the gradients that best fit the probabilities with:

```
python calculate_grads.py --input /path/to/ultrasurface_fits/genie_all_bulkice_pm10pc_knn_200pc.feather --output /path/to/ultrasurface_grads_vs/genie_all_bulkice_pm10pc_knn_200pc_poly2.feather --include-systematics dom_eff hole_ice_p0 hole_ice_p1 bulk_ice_abs bulk_ice_scatter --poly-features 2 --jobs 24
```

### Usage

The gradients are stored in a `.feather` file containing all events of the nominal MC set and their associated gradients. The Ultrasurface PISA stage needs to be pointed to the location of this file. In the unblinding version of this analysis, the file is

```
/path/to/ultrasurface_grads_vs/genie_all_bulkice_pm10pc_knn_200pc_poly2.feather
```
