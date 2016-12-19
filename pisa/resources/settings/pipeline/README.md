# Pipeline Settings

This directory should contain everything needed to help you set up your pipeline for whatever analysis you want to do. In here are the following configs:

`examples` - contains the following examples:
  * `example.cfg` - Demonstrates the simplest `hist` pipeline. Here, `reco` and `pid` are joined by using PID as a binning dimension.
  * `example_aeffsmooth.cfg` - Demonstrates a pipeline with smoothing in the `aeff` stage.
  * `example_deepcore.cfg` - Demonstrates the simplest `hist` pipeline but using DeepCore Monte Carlo. Here, `reco` and `pid` are separate.
  * `example_gpu.cfg` - Demonstrates the simplest `hist` pipeline but with the oscillations stage running on a gpu.
  * `example_vbwkde.cfg` - Demonstrates the use of the variable-bandwidth kernel density estimation for computing the `reco` transformations.
  * `example_xsec.cfg` - Demonstrates the use of the `xsec` stage instead of the `aeff` stage.