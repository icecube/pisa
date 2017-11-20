# pisa.analysis

Primary tools for performing analyses in PISA.


## Directory Listing

* `__init__.py` - Makes `analysis` directory behave as a Python module
* `analysis.py` - Code containing an implementation of a generic `Analysis` class from which any other analysis can inherit.
* `hypo_testing.py` - Code containing an implementation of a `HypoTesting` class for doing hypothesis testing. Inherits from `Analysis`.
* `nutau_analysis.py`
* `profile_llh_analysis.py`
* `profile_scan.py` - Script that uses the `scan` function of the `Analysis` class to sample likelihood surfaces over an arbitrary number of dimensions. The scan parameters are set by the user and can include minimisation over a set of systematics.
* `scan_allsyst.py`
* `theta23_NMO_2.py`
* `theta23_octant.py`