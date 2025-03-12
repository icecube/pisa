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

Treatment of detector systematics via likelihood-free inference. Polynomial coefficients, assigned to every event, allow continuous re-weighting as a function of detector uncertainties in a way that is fully decoupled from flux and oscillation effects. The results are stored in a feather file containing all events of the nominal MC set and their associated polynomial coefficients.

To use this in a PISA analysis pipeline, you will need to set up an ultrasurface config file looking like this:

```ini
[discr_sys.ultrasurfaces]

calc_mode = events
apply_mode = events

# DOM efficiency
param.dom_eff = 1.0 +/- 0.1
param.dom_eff.fixed = False
param.dom_eff.range = [0.8, 1.2] * units.dimensionless
param.dom_eff.tex = \epsilon_{\rm{DOM}}

# hole ice scattering
param.hole_ice_p0 = +0.101569
param.hole_ice_p0.fixed = False
param.hole_ice_p0.range = [-0.6, 0.5] * units.dimensionless
param.hole_ice_p0.prior = uniform
param.hole_ice_p0.tex = \rm{hole \, ice}, \: p_0

# hole ice forward
param.hole_ice_p1 = -0.049344
param.hole_ice_p1.fixed = False
param.hole_ice_p1.range = [-0.2, 0.2] * units.dimensionless
param.hole_ice_p1.prior = uniform
param.hole_ice_p1.tex = \rm{hole \, ice}, \: p_1

# bulk ice absorption
param.bulk_ice_abs = 1.0
param.bulk_ice_abs.fixed = False
param.bulk_ice_abs.range = [0.85, 1.15] * units.dimensionless
param.bulk_ice_abs.prior = uniform
param.bulk_ice_abs.tex = \rm{ice \, absorption}

# bulk ice scattering
param.bulk_ice_scatter = 1.05
param.bulk_ice_scatter.fixed = False
param.bulk_ice_scatter.range = [0.90, 1.20] * units.dimensionless
param.bulk_ice_scatter.prior = uniform
param.bulk_ice_scatter.tex = \rm{ice \, scattering}

# These nominal points are the nominal points that were used to fit the gradients
# and might not agree with the nominal points of the parameter prior.
nominal_points = {"dom_eff": 1.0, "hole_ice_p0": 0.101569, "hole_ice_p1": -0.049344, "bulk_ice_abs": 1.0, "bulk_ice_scatter": 1.0}

fit_results_file = /path/to/ultrasurface_fits/genie_all_knn_200pc_weight_weighted_aeff_poly_2.feather
```

Here you specify the detector systematic parameters to be varied in the fit, with their nominal values and allowed ranges. Additionally, you have to specify the nominal point at which the ultrasurfaces were fit (`nominal_points`), since this might be different from the nominal point used in your analysis. Finally, you have to point to the file where the polynomial coefficients are stored (`fit_results_file`).

Your pipeline's order could then look like this:

```ini
order = data.simple_data_loader, flux.honda_ip, flux.mceq_barr, osc.prob3, xsec.genie_sys, xsec.dis_sys, aeff.aeff, discr_sys.ultrasurfaces, utils.hist
```

It's important to include the ultrasurface stage **before** the histogramming stage, unlike it's done for the hypersurfaces. Now you should be good to go.
