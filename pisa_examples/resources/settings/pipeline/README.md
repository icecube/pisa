# Pipeline Settings

This directory contains example pipeline configuration files using some of the stages and services available in PISA.

## File listing

| Configuration file                 | Description
| ---------------------------        | -----------
| `example.cfg`                      | produces toy atmospheric neutrino distributions for a generic neutrino telescope
| `fast_example.cfg`                 | same as above but performs oscillation-probability calculation on a coarser grid
| `varbin_example.cfg`               | same as above but demonstrates how to use different binnings for different event (sub-)selections
| `osc_example.cfg`                  | an untypical pipeline with unit numu(bar) fluxes whose outputs are atmospheric numu(bar)->nux(bar) oscillation probabilities on a grid
| `IceCube_3y_data.cfg`              | produces the observed event distribution of an IceCube oscillation sample released in https://icecube.wisc.edu/data-releases/2019/05/three-year-high-statistics-neutrino-oscillation-samples
| `IceCube_3y_muons.cfg`             | produces the data-driven muon background distribution of the same sample
| `IceCube_3y_neutrinos.cfg`         | produces the MC neutrino distributions of the same sample
| `IceCube_3y_neutrinos_daemon.cfg`  | same as above but uses DAEMONFLUX instead of Honda

