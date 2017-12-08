# pisa.core

Module containing all core PISA objects


## Summary of Core PISA Objects

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


## Directory Listing

| File/directory          | Description
| ----------------------- | -----------
| `images/`               | Images to include in docs
| `__init__.py`           | Make the `pisa/core` directory behave as Python module `pisa.core`
| `binning.py`            | Define `OneDimBinning` and `MultiDimBinning` classes to define and work with various types of binning specifications
| `distribution_maker.py` | Define `DistributionMaker` class for producing distributions from one or more `Pipeline`s; also can run this file as a script to produce a distribution
| `events.py`             | Define `Events` and `Data` classes as containers for events
| `map.py`                | Define `Map` class for defining/creating a single histogram and `MapSet` for working with a set of `Map`s (histograms)
| `param.py`              | Define `Param` class for working with experimental parameters (name, expected values, priors, whether they are to be varied in an analysis etc.) and define `ParamSet` for working with a set of `Param`s, and finally `ParamSelector` to allow switching between discrete versions of individual `Param`s
| `pipeline.py`           | Define `Pipeline` class that holds one or more `Stage`s that are meant to be chained together to poduce a distribution; also can run this file as a script to produce a distribtuion
| `prior.py`              | Define `Prior` class to give to `Param`s for defining prior knowledge on them
| `stage.py`              | Define `Stage` class which is the base class for implementing a service in PISA
| `transform.py`          | Define  `Transform` as a generic object for use in a service for transforming the outputs of a previous stage; `BinnedTensorTransform` is a subclass of this implementing arbitrary-dimensional tensor transforms; `TransformSet` is a container for multiple `Transform`s
