# pisa.analysis

Primary tools for performing analyses in PISA.


## Directory Listing

| File/directory    | Description
| ----------------- | -----------
| `__init__.py`     | Makes `analysis` directory behave as a Python module
| `analysis.py`     | Defines `Analysis` class giving generic tools for performing analyses; sub-class this for specialized analyses
| `hypo_testing.py` | Script and class `HypoTesting` for performing hypothesis tests; the class inherits from `Analysis`
