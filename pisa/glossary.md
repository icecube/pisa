# PISA Terminology
A selection of key terms and concepts in PISA is compiled below.

* **Detectors**: A collection of one or more *DistributionMakers*, where each corresponds to one detector/experiment. The output of `Detectors` typically is a sequence of *MapSets*, one per `DistributionMaker` (detector).

* **DistributionMaker**: A collection of one or more *pipelines*; this produces the events distributions we see (in the case of data) or that we expect to see (in the case of Monte Carlo). The output of a `DistributionMaker` typically is a `MapSet`, produced by summing over outputs of all pipelines in the `DistributionMaker`.

* **Map**: N-dimensional histogram, most often in energy and cosine of the zenith angle (coszen). However, the number of dimensions and the binning in each are completely configurable.

* **MapSet**: Set of maps, with convenience methods for working with each.

* **Pipeline**: A single sequence of *stages* and the *services* implementing them for processing a single data type. E.g.:
  * There might be one pipeline for processing atmospheric neutrinos and a separate pipeline for processing atmospheric muons.
  * A separate and possibly completely independent set of pipelines can be defined to produce the pseudodata or the observed data distribution.

* **Pipeline settings**: The collection of all parameters required (and no more) to instantiate all stages (and which service to use for each) in a single `Pipeline`.

* **Quantity**: A number or array *with units attached*. See [units and uncertainties](units_and_uncertainties.md).

* **Resource**: A file with settings, simulated events, parameterizations, metadata, etc. that is used by one of the services, a `DistributionMaker`, an analysis script, .... Example resources are found in `$PISA/pisa_examples/resources`, where a subdirectory may exist for each stage (and several directories exist for resources used for other purposes). For PISA to be able to detect your personal resources anywhere else, include all your custom resource locations in your command shell's environment variable `PISA_RESOURCES`.

* **Reweighted Monte Carlo (MC) analysis**: Each stage of the analysis simulates the effects of physics and detector systematics by directly modifying the MC events' characteristics (e.g., their importance weights and reconstructed properties). After applying all such effects, only in the last step are the MC events histogrammed.

* **Service**: A particular *implementation* of a stage is called a ***service***. Each service is a python `.py` file that lives inside its stage's directory in `$PISA/pisa/stages/<stage name>/`.

* **Stage**: Each `Stage` represents a critical part of the process by which we can eventually detect e.g. neutrinos. For example, atmospheric neutrinos that pass through the earth will oscillate partially into different flavors prior to reaching the detector. This part of the process is modelled by the **oscillations** stage. Other characteristic stages are `data` (for initially loading events available in arbitrary formats into a pipeline), `flux`, or `xsec` (interaction cross section). Stages are directories in the `$PISA/pisa/stages` directory.

* **Stage modes**: Each `Stage` defines two modes which determine how the data (e.g. neutrino MC events) handed to it is represented during `setup()`/`compute()` and during `apply()`. Two common representations are individual events and grids/histograms.
