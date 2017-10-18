# $PISA/pisa/analysis

* `__init__.py`
* `analysis.py` - Code containing an implementation of a generic `Analysis` class from which any other analysis can inherit.
* `hypo_testing.py` - Code containing an implementation of a `HypoTesting` class for doing hypothesis testing. Inherits from `Analysis`.
* `profile_scan.py` - Script that uses the `scan` function of the `Analysis` class to sample likelihood surfaces over an arbitrary number of dimensions. The scan parameters are set by the user and can include minimisation over a set of systematics.
* `scan_allsyst.py` - similar to `profile_scan.py`, but scans all free pipeline parameters one-by-one (no profile)
