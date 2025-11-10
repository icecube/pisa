# pisa.stages

Directories are PISA stages, and within each directory can be found the services implementing the respective stage.

## Directory Listing

* `absorption/` - A stage for neutrino flux absorption in the Earth.
* `aeff/` - All stages relating to effective area transforms.
* `background/` - A stage for modifying some nominal (background) MC muon flux due to systematics.
* `data/` - All stages relating to the handling of data.
* `discr_sys/` - All stages relating to the handling of discrete systematics.
* `flux/` - All stages relating to the atmospheric neutrino flux.
* `likelihood/` - A stage that pre-computes some quantities needed for the "generalized likelihood"
* `osc/` - All stages relating to neutrino oscillations. 
* `reco/` - All stages relating to applying reconstruction kernels (including pid).
* `utils/` - All "utility" stages (not representing physics effects).
* `xsec/` - All stages relating to cross sections.
* `GLOBALS.md` - File that describes globally available variables within PISA together with the services that implement them.
* `__init__.py` - File that makes the `stages` directory behave as a Python module.
