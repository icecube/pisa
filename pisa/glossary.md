# PISA Terminology

* **DistributionMaker**: A collection of one or more *pipelines*; this produces the events distributions we see (in the case of data) or that we expect to see (in the case of Monte Carlo). The output of a DistributionMaker is a sequence of MapSets, one produced per pipeline in the DistributionMaker.

* **Map**: N-dimensional histogram, most often in energy and cosine of the zenith angle (coszen)--but the number of dimensions and the binning in each are completely configurable in PISA). Note that the binning of a map is accessed via `map.binning`.

* **MapSet**: Container or set of `Map`, with convenience methods for working with the contained maps.

* **Pipeline**: A single sequence of stages and the services implementing them for processing a single data type. E.g.:
  * There might be defined a pipeline for processing neutrinos and a separate pipeline for processing muons.
  * For neutrino mass ordering measurements, the set of pipelines to produce "template" distributions might include one neutrino pipeline for normal ordering and one neutrino pipeline for inverted ordering, in addition to a single pipeline for muons.
    * If parameters to produce the "data" distribution used to match templates to differ from the nominal values for the template pipelines, a separate set of the aforementioned pipelines can be defined to produce the "data" distribution.

* **Pipeline settings**: The collection of all parameters required (and no more) to instantiate all stages (and which service to use for each) in a single pipeline.

* **Quantity**: A number or array *with units attached*. See (units_and_uncertainties).

* **Resource**: A file with settings, simulated events, parameterizations, metadata, etc. that is used by one of the services, the template maker, the minimizer, .... Example resources are found in `$PISA/pisa_examples/resources` directory, where a subdirectory exists for each stage (and several directories exist for resources used for other purposes). Set (in your command shell) the environment variable PISA\_RESOURCES to your personal PISA resources directory location to access your own resources.

* **Reweighted Monte Carlo (MC) analysis**: Each stage of the analysis simulates the effects of physics and detector systematics by directly modifying the MC events' characteristics (e.g., their importance weights and reconstructed properties). After applying all such effects, only in the last step are the MC events histogrammed. Contrast with *parameterized-MC analysis* (defined above), where it is histograms that are modified throughout the analysis to reflect the effects of physics and detector systematics.

* **Service**: A particular *implementation* of a stage is called a ***service***. For example, the effective areas stage (`aeff`) has services `mc` and `slice_smooth`. Each service is a python `.py` file that lives inside its stage's directory. E.g., the effective area `mc` service is at `$PISA/pisa/stages/aeff/mc.py`.

* **Stage**: Each stage represents a critical part of the process by which we can eventually detect neutrinos. For example, atmospheric neutrinos that pass through the earth will oscillate partially into different flavors prior to reaching the IceCube/PINGU detector. This part of the process is called the **oscillations** stage. There are currently defined the stages `flux` (for neutrino flux), `osc` (neutrino oscillations), `aeff` (detector effective area), `reco` (reconstruction resolutions), and `pid` (particle identification; e.g., tracks vs. cascades). Stages are directories in the `$PISA/pisa/stages` directory.
