# Stage: Discrete Systematics

These stages apply parameterized systematics to the templates.

## Services

### hypersurfaces

This service applies the results obtained from fits to bin-count expectations given "discrete" MC event simulation samples,
i.e. one per choice of detector-response settings (such as a particular overall DOM efficiency correction).

Fits may be performed for different assumptions of non-detector model parameters, such as oscillation parameters, and then interpolated by this service.
The fitting parameters are computed and stored in a fit results file by the script `$PISA/pisa/scripts/fit_hypersurfaces.py` (command-line alias `pisa-fit_hypersurfaces`).
This has to be executed together with a dedicated fit configuration file (see script's documentation).

The method itself is described and illustrated in the paper ["Measurement of Atmospheric Neutrino Mixing with Improved IceCube DeepCore
Calibration and Data Processing"](https://inspirehep.net/literature/2653713) (Sec. VI A).

The "hyperplanes" predecessor method presented in the ["Measurement of atmospheric tau neutrino appearance
with IceCube DeepCore"](https://inspirehep.net/literature/1714067) (Sec. V E) is also incorporated.

### csv_hypersurfaces

Service to load and apply (pre-generated) hypersurfaces stored in csv files. This file format is usually used for data releases. Currently, the service expects hypersurfaces in the form of `pisa_examples/resources/events/hs_test.csv`.

### ultrasurfaces

Treatment of detector systematics via likelihood-free inference as described in https://inspirehep.net/literature/2656204.
Polynomial coefficients, assigned to every event, allow continuous re-weighting as a function of detector uncertainties in a way that is fully decoupled
from flux and oscillation effects. The results are stored in a feather file containing all events of the nominal MC set and their associated polynomial coefficients.

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

Here you specify the detector systematic parameters to be varied in the fit, with their nominal values and allowed ranges.
Additionally, you have to specify the nominal point at which the ultrasurfaces were fit (`nominal_points`), since this might be different from the nominal point used in your analysis.
Finally, you have to point to the file where the polynomial coefficients are stored (`fit_results_file`).

Your pipeline's order could then look like this:

```ini
order = data.simple_data_loader, flux.honda_ip, flux.mceq_barr, osc.prob3, xsec.genie_sys, xsec.dis_sys, aeff.aeff, discr_sys.ultrasurfaces, utils.hist
```

It's important to include the ultrasurface stage **before** the histogramming stage, unlike it's done for the hypersurfaces. Now you should be good to go.
