# Core object hierarchy
This is a summary of the hierarchy of the objects instantiated by the pisa.core
classes. Indentation indicates that the object lives in the class above.

At the top level is `Analysis`, which is where the actual 'fitting' or 'scanning' etc
happens.

Some examples are given to facilitate understanding.

* [Analysis](/pisa/core/analysis.py)
  * [DistributionMaker](/pisa/core/distribution_maker.py) A (e.g. to produce pseudo data distribution)
  * [DistributionMaker](/pisa/core/distribution_maker.py) B (that may be fitted to a distribution from DistributionMaker A, for example)
    * [ParamSet](/pisa/core/param.py)
      * [Param](/pisa/core/param.py)
      * [Param](/pisa/core/param.py) ...
    * [Pipeline](/pisa/core/pipeline.py) a (e.g. muons from data (inv. corridor cut))
    * [Pipeline](/pisa/core/pipeline.py) b (e.g. neutrino MC)
    * [Pipeline](/pisa/core/pipeline.py) ...
      * [ParamSet](/pisa/core/param.py)
      * [Stage](/pisa/core/stage.py) s0: flux / service honda (inherits from Stage class)
      * [Stage](/pisa/core/stage.py) s1: osc / service prob3cpu (inherits from Stage)
      * [Stage](/pisa/core/stage.py) s2: aeff / service hist (inherits from Stage)
      * [Stage](/pisa/core/stage.py) ...
        * [ParamSet](/pisa/core/param.py)
          * [Param](/pisa/core/param.py) foo (e.g. energy_scale)
          * [Param](/pisa/core/param.py) bar (e.g. honda_flux_file)
          * [Param](/pisa/core/param.py) ...
            * [Prior](/pisa/core/prior.py) (e.g. a gaussian prior with given mu and sigma) 
        * [Events](/pisa/core/events.py) (if used by stage)
        * [TransformSet](/pisa/core/transform.py) (if applicable)
          * [Transform](/pisa/core/transform.py) t0 : BinnedTensorTransform (inherits from Transform)
          * [Transform](/pisa/core/transform.py) t1 : BinnedTensorTransform (inherits from Transform)
            * [MultiDimBinning](/pisa/core/binning.py) input_binning
              * [OneDimBinning](/pisa/core/binning.py) d0 (e.g. true/reco_energy)
              * [OneDimBinning](/pisa/core/binning.py) d1 (e.g. true/reco_coszen)
            * [MultiDimBinning](/pisa/core/binning.py) output_binning
        * [MapSet](/pisa/core/map.py) as input to / output from stage
          * [Map](/pisa/core/map.py) m0 (e.g. numu_cc)
          * [Map](/pisa/core/map.py) m1 (e.g. numu_nc)
          * [Map](/pisa/core/map.py) ...
            * [MultiDimBinning](/pisa/core/binning.py)
              * [OneDimBinning](/pisa/core/binning.py) d0 (e.g. true/reco_energy)
              * [OneDimBinning](/pisa/core/binning.py) d1 (e.g. true/reco_coszen)
              * [OneDimBinning](/pisa/core/binning.py) ...

# $PISA/pisa/core

* `images/` - Contains images designed to illustrate the architecture of a stage, how it fits in to the pipeline and the overall distribution maker.
* `__init__.py`
* `binning.py` - A class for dealing with binning so that features like the name, units, whether it is logarithmic etc. can be stored.
* `distribution_maker.py` - A class for dealing with how to go from a pipeline to actual output that can be fed to an analysis or whatever is needed. This can deal with combining multiple pipelines.
* `events.py` - A storage container for the raw events that can be then used to create _something_ e.g. a `Map` or a `Transform`.
* `map.py` - A storage container for the output histograms so that much more information can be easily kept with it e.g. name, binning, errors, units etc.
* `param.py` - Defines the behaviour of parameters i.e. the name, expected values, any priors, whether they are to be varied in an analysis etc.
* `pipeline.py` - Gives structure to the stages and organises them based on pipeline `cfg` files. 
* `prior.py` - Defines the generic form of a prior i.e. the type and how the penalty is calculated.
* `stage.py` - Defines the generic form of a stage i.e. loading from pipelines, how to cache results etc.
* `transform.py` - Defines the generic form of the transforms used to get between stages i.e. take the input (if any), do _something_ to it and then get an output. 
