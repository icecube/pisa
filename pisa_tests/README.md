# tests

PISA validation and unit tests


## Directory Listing

| File/directory                          | Description
| --------------------------------------- | -----------
| `__init__.py`                           | Make this directory behave as a Python module
| `test_changes_with_combined_pidreco.py` | python script for testing how merging `reco` and `pid` in to a single stage affected the distributions. This is compared to PISA 2 reference files as well as some from OscFit.
| `test_command_lines.sh`                 | Bash (shell) script to run all PISA unit tests
| `test_consistency_with_oscfit.py`       | script for testing how consistent the MC re-weighting treatment in PISA is with OscFit.
| `test_consistency_with_pisa2.py`        | Python script for testing the consistency of current PISA 3 services with reference files produced by PISA 2.
| `test_example_pipelines.py`             | Python script for testing that the pipelines contained in `$PISA/pisa/resources/settings/pipeline/` are all functional.