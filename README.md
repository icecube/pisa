# PISA

PINGU Simulation and Analysis (PISA) is software written for performing analyses based upon Monte Carlo simulations of the [IceCube neutrino observatory](https://icecube.wisc.edu/), including the [DeepCore](https://arxiv.org/abs/1109.6096) and proposed [PINGU](https://arxiv.org/abs/1401.2046) low-energy in-fill arrays (as well as other similar neutrino detectors).

PISA was originally developed to cope with low-statistics Monte Carlo (MC) for PINGU using parameterizations of the MC, but its methods should apply equally as well to high-MC situations, and the PISA architecture is general enough to easily accomoodate traditional reweighted-MC-style analyses.

Visit the project Wiki for the latest documentation (not yet folded into the codebase):
* [PISA Cake wiki](https://github.com/jllanfranchi/pisa/wiki)

A compilation of the README files in to a set of documentation using Sphinx can be found at: http://icecube.wisc.edu/~peller/pisa_docs/index.html

# $PISA

* `docs/` - Directory containing the output of Sphinx auto-generating the documentation.
* `images/` - Directory containing the git fork button.
* `pisa/` - Directory containing all of the PISA code.
* `tests/` - Contains all the python scripts for tests on the performance of PISA.
* `.gitattributes`
* `.gitignore`
* `INSTALL.md`
* `MANIFEST.in`
* `README.md` - This file!
* `pylintrc`
* `requirements.txt`
* `setup.cfg`
* `setup.py`
* `versioneer.py`
