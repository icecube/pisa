# pisa.resources

Location to store default resources for PISA.

Note that for resources outside of the standard resources, you can use a separate directory and set the `PISA_RESOURCES` environment variable in your shell pointing to that directory.


## Directory Listing

* `aeff/` - Contains files used by the `aeff` stages e.g. parameterised effective areas.
* `backgrounds/` - Contains files relating to the backgrounds in analyses.
* `cache/` - Where the stages will store their cache by default.
* `cross_sections/` - Contains files relating to the implementation of cross-sections in the `xsec` stage.
* `debug/` - Where the output of running stages in debug mode will go.
* `events/` - Contains HDF5 files with neutrino simulations.
* `flux/` - Contains files used by the `flux` stages e.g. the Honda tables.
* `osc/` - Contains files used by the `osc` stages e.g. the Earth density files.
* `pid/` - Contains files used by the `pid` stages e.g. parametersied PID probabilities. 
* `priors/` - Contains any non-standard priors that can be used in analyses.
* `reco/` - Contains files used by the `reco` stages e.g. parameterised reconstruction kernels.
* `settings/` - Containing settings files used by the pipelines.
* `tests/` - Contains all files relating to the tests in `$PISA/tests`.
* `__init__.py` - File that makes `resources` directory behave as a Python module