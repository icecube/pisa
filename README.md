# PISA
[Introduction](pisa/README.md) |
[Installation](INSTALL.md) |
[Documentation](http://icecube.wisc.edu/%7Epeller/pisa_docs/index.html) |
[Terminology](pisa/glossary.md) |
[License](LICENSE) |
[Contributors](CONTRIBUTORS.md) |
[Others' work](EXTERNAL_ATTRIBUTION.md)

PISA (PINGU Simulation and Analysis) is software written to analyze the results (or expected results) of an experiment based on Monte Carlo simulation.

In particular, PISA was written for the [IceCube Neutrino Observatory](https://icecube.wisc.edu/), including the [DeepCore](https://arxiv.org/abs/1109.6096) and the proposed [PINGU](https://arxiv.org/abs/1401.2046) low-energy in-fill arrays.
However, any such experiment—or any experiment at all—can make use of PISA for analyzing expected and actual results.

PISA was originally developed to cope with low-statistics Monte Carlo (MC) for PINGU when iterating on multiple proposed geometries by using parameterizations of the MC and operate on histograms of the data rather than directly reweighting the MC (as is traditionally done in high-energy Physics experiments).
However, PISA's methods apply equally well to high-MC situations, and PISA also performs traditional reweighted-MC analysis as well.

## Directory listing

| File/directory     | Description
| ------------------ | -----------
| `docs/`            | Sphinx auto-generated documentation
| `images/`          | Images to include in documentation
| `pisa/`            | Source code
| `tests/`           | Scripts for running physics and unit tests
| `.gitattributes`   |
| `.gitignore`       | GIT ignores files matching these specifications
| `INSTALL.md`       | How to install PISA
| `MANIFEST.in`      |
| `README.md`        | Brief overview of PISA
| `pylintrc`         | PISA coding conventions for use with pylint
| `requirements.txt` | Hard dependencies, use with `pip install -r requirements.txt`
| `setup.cfg`        | Setup file for `versioneer`
| `setup.py`         | Python setup file, allowing e.g. `pip` installation
| `versioneer.py`    | Automatic versioning
