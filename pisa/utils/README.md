# pisa.utils
Basic utilities underlying much of the work of a data analyzer and one developing for PISA.

These range from statistical functions to file readers/writers and from data smoothing techniques to logging messages.
See the directory listing below for a brief description of the utilities contained in each file.


## Directory Listing

| File/directory                | Description
| ----------------------------- | -----------
| `images/`                     | Images to appear in documentation
| `__init__.py`                 | Empty file that makes this directory behave as a Python module
| `barlow.py`                   | Likelihood class that implementes Barlow (and Poisson) likelihood
| `cache.py`                    | Caching to memory and disk
| `comparisons.py`              | Comparing things in sensible ways (e.g. floating point numbers, numbers with units, etc.)
| `confInterval.py`             | Compute confidence intervals
| `config_parser.py`            | Parsing of PISA-format config files
| `coords.py`                   | Coordinate systems and transformations
| `cross_sections.py`           | CrossSections class for importing, working with, and storing neutrino cross sections from GENIE
| `cross_sections.md`           | More help for CrossSections
| `cuda_utils.h`                | Header file with utils for CUDA coding
| `data_proc_params.py`         | DataProcParams class for importing, working with, and storing data processing parameters (e.g., PINGU's V5 processing).
| `fileio.py`                   | Load from and store to files in PISA-standard ways
| `flavInt.py`                  | Neutrino flavors and interaction types classes
| `flavor_interaction_types.md` | More help on flavInt module
| `flux_weights.py`             | Flux calculation utils
| `format.py`                   | Nice formatting for numbers, timestamps, etc.
| `gaussians.py`                | Sum-of-1D Gaussians functions for use by e.g. KDE tools
| `gaussians_cython.pyx`        | Cython implementation of sum-of-1D Gaussians
| `gpu_hist.py`                 | Histogramming for GPU arrays
| `hash.py`                     | Hashing PISA and other Python objects
| `hdf.py`                      | Read from and store to HDF5 files in PISA standard format
| `hdfchain.py`                 | Chain multiple HDF5 files together but read as one file
| `jsons.py`                    | Read from and store to JSON files in PISA standard format
| `kde_hist.py`                 | Populate histograms by first applying KDE to the samples
| `log.py`                      | Use for logging messages (to screen etc.)
| `mcSimRunSettings.py`         | Simulation run settings storage
| `parallel.py`                 | Thin wrapper for running functions in parallel
| `PIDSpec.py`                  | Particle Identification (PID) specification
| `plotter.py`                  | PISA standard plotting library
| `postprocess.py`              | Postprocessing of analysis results
| `profiler.py`                 | Profile code line-by-line or by function, utilizing PISA logging mechanisms
| `random_numbers.py`           | Standard way to (re)produce random numbers in PISA
| `README.md`                   | Overview of the `utils` module
| `resources.py`                | Access to PISA package resources, whether a simple file or stored within the packaged version of the software
| `rooutils.py`                 | Convenience functions when interacting with ROOT
| `spline.py`                   | Classes to store and handle the evaluation and manipulation of splines
| `spline_smooth.py`            | Smooth an array by splining it and resampling from the spline
| `stats.md`                    | Detailed description of the stats module
| `stats.py`                    | Statistics
| `tests.py`                    | Functions to help compare and plot differences between PISA 2 and PISA 3 maps
| `test_gaussians.py`           | Unittests for functions that live in the gaussians.pyx Cython module
| `timing.py`                   | Simple Timer class that can be used as a Python context for timing code blocks
| `vbwkde.py`                   | 1D variable bandwidth (adaptive) KDE using the Improved Sheather Jones bandwidth criteria (see external `kde` module for multi-dimenional adaptive KDE based on simpler bandwidth criteria)