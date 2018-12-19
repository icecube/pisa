# pisa.stages

Directories are PISA stages, and within each directory can be found the services implementing the respective stage.


## Directory Listing

* `aeff/` - All stages relating to effective area transforms.
* `combine/` - A stage for combining maps together and applying appropriate scaling factors. 
* `data/` - All stages relating to the handling of data.
* `discr_sys/` - All stages relating to the handling of discrete systematics.
* `flux/` - All stages relating to the atmospheric neutrino flux.
* `osc/` - All stages relating to neutrino oscillations. 
* `pid/` - All stages relating to particle identification.
* `reco/` - All stages relating to applying reconstruction kernels.
* `unfold/` - All stages relating to the unfolding of parameters from data.
* `xsec/` - All stages relating to cross sections.
* `__init__.py` - File that makes the `stages` directory behave as a Python module.
