# pisa.analysis

Primary tools for performing analyses in PISA.


## Directory Listing

| File/directory    | Description
| ----------------- | -----------
| `__init__.py`     | Makes `analysis` directory behave as a Python module
| `analysis.py`     | Defines `Analysis` class giving generic tools for performing analyses; sub-class this for specialized analyses
| `hypo_testing.py` | Script and class `HypoTesting` for performing hypothesis tests; the class inherits from `Analysis`
| `profile_scan.py` | Script that uses the `scan` function of the `Analysis` class to sample likelihood surfaces over an arbitrary number of dimensions. The scan parameters are set by the user and can include minimisation over a set of systematics.
| `scan_allsyst.py` | Similar to `profile_scan.py`, but scans each free param separately (i.e., no profile)
