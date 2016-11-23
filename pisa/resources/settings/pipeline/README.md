# Pipeline Settings

This directory should contain everything needed to help you set up your pipeline for whatever analysis you want to do. In here are three directories:

1. `examples` - contains the following examples:
    * `example.cfg` - Demonstrates the simplest `hist` pipeline.
    * `example_aeffsmooth.cfg` - Demonstrates a pipeline with smoothing in the `aeff` stage.
    * `example_deepcore.cfg` - Demonstrates the simplest `hist` pipeline but using DeepCore Monte Carlo.
    * `example_gpu.cfg` - Demonstrates the simplest `hist` pipeline but with the oscillations stage running on a gpu.
    * `example_recopid.cfg` - Demonstrates a parameterised pipeline with the `reco` and `pid` stages joined in to one by using PID as a binning dimension.
    * `example_vbwkde.cfg` - Demonstrates the use of the variable-bandwidth kernel density estimation for computing the `reco` transformations.
    * `example_xsec.cfg` - Demonstrates the use of the `xsec` stage instead of the `aeff` stage.
2. `deepcore` - contains pipelines that have been used for analyses with DeepCore.
3. `pingu` - contains pipelines that have been used for analyses with PINGU. Within this directory are further directories for different PINGU configurations.