## Installation Guide

### Quickstart

_Note that terminal commands below are intended for the bash shell. You'll have to translate if you use a different shell._

1. Obtain a github user ID if you don’t have one already<br>
    https://github.com
    * Sign up for Github education pack for many features for free, too<br>
        https://education.github.com/pack
1. Fork PISA on github so you have your own copy to work from<br>
    https://github.com/IceCubeOpenSource/pisa#fork-destination-box
1. _(optional)_ Set up shared-key ssh access to github so you don’t have to enter passwords<br>
    https://help.github.com/articles/connecting-to-github-with-ssh
1. In your terminal, define a directory for PISA sourcecode to live in. For example:<br>
    `export PISA="~/src/pisa"`
    * Add this line to your ~/.bashrc file so you can refer to the `$PISA` variable without doing this everytime.
1. Create the directory<br>
    `mkdir -p $PISA`
1. Clone the PISA repo to your local computer (at command line)
    * If you set up shared-key auth above<br>
    `git clone git@github.com:<YOUR GITHUB USER ID HERE>/pisa.git $PISA`
    * Otherwise<br>
    `git clone https://github.com/<YOUR GITHUB USER ID HERE>/pisa.git $PISA`
1. Install the ***Python 3.7 / 64 bit*** Anaconda or Miniconda python distribution for either Mac or Linux (as your user, _not_ as root), if you don’t have it already
    * Anaconda (full-featured Python distribution, ~500 MB)<br>
        https://www.anaconda.com/download
    * Miniconda (just the essentials, ~40 MB)<br>
        https://conda.io/miniconda.html
1. Create a new conda environment, ideally with a python version compatible with the python requirements below (cf. conda's getting started guide).
1. Active your new conda environment.
1. Install PISA including optional packages and development tools (`develop`), if desired<br>
    `pip install -e $PISA[develop] -vvv`
1. Run a quick test: generate templates in the staged mode<br>
`$PISA/pisa/core/pipeline.py --pipeline settings/pipeline/example.cfg  --outdir /tmp/pipeline_output --intermediate --pdf -v`

See [github.com/IceCubeOpenSource/pisa/wiki/installation_specific_examples](https://github.com/IceCubeOpenSource/pisa/wiki/installation_specific_examples) for users' recipes for installing PISA under various circumstances.
Please add notes there and/or add your own recipe if your encounter a unique installation issue.
Also, for instructions on running PISA on Open Science Grid (OSG) nodes, see [github.com/IceCubeOpenSource/pisa/wiki/Running-PISA-on-GRID-nodes-with-access-to-CVMFS](https://github.com/jllanfranchi/pisa/wiki/Running-PISA-on-GRID-nodes-with-access-to-CVMFS)

The following sections delve into deeper detail on installing PISA.


### Python Distributions

Obtaining Python and Python packages, and handling interdependencies in those packages tends to be easiest if you use a Python distribution, such as [Anaconda](https://www.continuum.io/downloads) or [Canopy](https://www.enthought.com/products/canopy).
Although the selection of maintained packages is smaller than if you use the `pip` command to obtain packages from the Python Package Index (PyPi), you can still use `pip` with these distributions.

The other advantage to these distributions is that they easily install without system administrative privileges (and install in a user directory) and come with the non-Python binary libraries upon which many Python modules rely, making them ideal for setup on e.g. clusters.

* **Note**: Make sure that your `PATH` variable points to e.g. `<anaconda_install_dr>/bin` and *not* your system Python directory. To check this, type: `echo $PATH`; to udpate it, add `export PATH=<anaconda_install_dir>/bin:$PATH` to your .bashrc file.
* Python 3.7.x can also be found from the Python website [python.org/downloads](https://www.python.org/downloads/) or pre-packaged for almost any OS.


### Required Dependencies

To install PISA, you'll need to have the following non-python requirements.
Note that these are not installed automatically, and you must install them yourself prior to installing PISA.
Also note that Python, HDF5, and pip support come pre-packaged or as `conda`-installable packages in the Anaconda Python distribution.
* [python](http://www.python.org) — version 3.7.x required: numba 0.45, which is required for SmartArray, does not allow python 3.8 or higher
  * Anaconda: built in
* [pip](https://pip.pypa.io) version >= 1.8 required
  * Anaconda:<br>
    `conda install pip`
* [git](https://git-scm.com)
  * In Ubuntu,<br>
    `sudo apt install git`
* [hdf5](http://www.hdfgroup.org/HDF5) — install with `--enable-cxx` option
  * In Ubuntu,<br>
    `sudo apt install libhdf5-10`
* [llvm](http://llvm.org) Compiler needed by Numba. This is automatically installed in Anaconda alongside `numba`.
  * Anaconda<br>
    `conda install numba=0.45.1`

Required Python modules that are installed automatically when you use the `pip` command detailed later:
* [decorator](https://pypi.python.org/pypi/decorator)
* [h5py](http://www.h5py.org)
* [iminuit](): Python interface to the MINUIT2 C++ package, used for proper covariance matrices during minimization
* [kde](https://github.com/IceCubeOpenSource/kde)
  * You can install the `kde` module manually if it fails to install automatically:
    * Including CUDA support:<br>
      `pip install git+https://github.com/icecubeopensource/kde.git#egg=kde[cuda]`
    * Without CUDA support:<br>
      `pip install git+https://github.com/icecubeopensource/kde.git#egg=kde`
* [line_profiler](https://pypi.python.org/pypi/line_profiler): detailed profiling output<br>
  * if automatic pip installation of line_profiler fails, you may want to try `conda install line_profiler` if you are using anaconda
* [matplotlib>=3.0](http://matplotlib.org) >= 3.0 required
* [numba==0.45.1](http://numba.pydata.org) Just-in-time compilation of decorated Python functions to native machine code via LLVM. This package is required to use PISA pi; also in cake it can accelerate certain routines significantly. If not using Anaconda to install, you must have LLVM installed already on your system (see above).
* [numpy](http://www.numpy.org) version >= 1.17 required
* [pint>=0.8.1](https://pint.readthedocs.org) >= 0.8.1 required
  * if automatic pip installation of pint fails, you may want to try `conda install pint` if you are using anaconda
* [scipy](http://www.scipy.org) version >= 0.17 required
* [setuptools](https://setuptools.readthedocs.io) version >= 18.5 required
* [simplejson](https://github.com/simplejson/simplejson) version >= 3.2.0 required
* [tables](http://www.pytables.org)
* [uncertainties](https://pythonhosted.org/uncertainties)
* [py-cpuinfo](https://pypi.org/project/py-cpuinfo) retrieve detailed CPU and architecture info for documenting in tests / obtaining results
* [sympy](https://www.sympy.org/en/) Used for testing `nsi_params.py` 


### Optional Dependencies

Optional dependencies. Some of these must be installed manually prior to installing PISA, and some will be installed automatically by pip, and this seems to vary from system to system. Therefore you can first try to run the installation, and just install whatever pip says it needed, or just use apt, pip, and/or conda to install the below before running the PISA installation.

* [MCEq](http://github.com/afedynitch/MCEq) Required for `flux.mceq` service.
* [nuSQuiDS](https://github.com/arguelles/nuSQuIDS) Required for `osc.nusquids` service.
* [pandas](https://pandas.pydata.org/) Required for datarelease (csv) stages.
* [OpenMP](http://www.openmp.org) Intra-process parallelization to accelerate code on on multi-core/multi-CPU computers.
  * Available from your compiler: gcc supports OpenMP 4.0 and Clang >= 3.8.0 supports OpenMP 3.1. Either version of OpenMP should work, but Clang has yet to be tested for its OpenMP support.
* [Pylint](http://www.pylint.org): Static code checker and style analyzer for Python code. Note that our (more or less enforced) coding conventions are codified in the pylintrc file in PISA, which will automatically be found and used by Pylint when running on code within a PISA package.<br>
  * Installed alongside PISA if you specify option `['develop']` to `pip`
* [recommonmark](http://recommonmark.readthedocs.io/en/latest/) Translator to allow markdown docs/docstrings to be used; plugin for Sphinx. (Required to compile PISA's documentation.)
  * Installed alongside PISA if you specify option `['develop']` to `pip`
* [ROOT >= 6.12.04 with PyROOT](https://root.cern.ch) Necessary for `xsec.genie`, `unfold.roounfold` and `absorption.pi_earth_absorption` services, and to read ROOT cross section files in the `crossSections` utils module. Due to a bug in ROOT's python support (documented here https://github.com/IceCubeOpenSource/pisa/issues/430), you need at least version 6.12.04 of ROOT.
* [Sphinx](http://www.sphinx-doc.org/en/stable/) version >= 1.3
  * Installed alongside PISA if you specify option `['develop']` to `pip`
* [versioneer](https://github.com/warner/python-versioneer) Automatically get versions from git and make these embeddable and usable in code. Note that the install process is unique since it first places `versioneer.py` in the PISA root directory, and then updates source files within the repository to provide static and dynamic version info.
  * Installed alongside PISA if you specify option `['develop']` to `pip`
* [black](https://github.com/ambv/black) Format your Python code, _automatically_, with typically very nice results!
  * Note this only works in Python3


### Obtain PISA sourcecode

#### Develop PISA: Fork then clone

If you wish to modify PISA and contribute your code changes back to the PISA project (*highly recommended!*), fork `IceCubeOpenSource/pisa` from Github.
*(How to work with the `cake` branch of PISA will be detailed below.)*

Forking creates your own version of PISA within your Github account.
You can freely create your own *branch*, modify the code, and then *add* and *commit* changes to that branch within your fork of PISA.
When you want to share your changes with `IceCubeOpenSource/pisa`, you can then submit a *pull request* to `IceCubeOpenSource/pisa` which can be merged by the PISA administrator (after the code is reviewed and tested, of course).

* Navigate to the [PISA github page](https://github.com/IceCubeOpenSource/pisa) and fork the repository by clicking on the ![fork](images/ForkButton.png) button.
* Clone the repository into the `$PISA` directory via one of the following commands (`<github username>` is your Github username):
  * either SSH access to repo:<br>
`git clone git@github.com:<github username>/pisa.git $PISA
`
  * or HTTPS access to repo:<br>
`git clone https://github.com/<github username>/pisa.git $PISA`


#### Using but not developing PISA: Just clone

If you just wish to pull changes from github (and not submit any changes back), you can just clone the sourcecode without creating a fork of the project.

* Clone the repository into the `$PISA` directory via one of the following commands:
  * either SSH access to repo:<br>
`git clone git@github.com:IceCubeOpenSource/pisa.git $PISA`
  * or HTTPS access to repo:<br>
`git clone https://github.com/IceCubeOpenSource/pisa.git $PISA`


### Ensure a clean install using virtualenv or conda env

It is absolutely discouraged to install PISA as a root (privileged) user.
PISA is not vetted for security vulnerabilities, so should always be installed and run as a regular (unprivileged) user.

It is suggested (but not required) that you install PISA within a virtual environment (or in a conda env if you're using Anaconda or Miniconda Python distributions).
This minimizes cross-contamination by PISA of a system-wide (or other) Python installation with conflicting required package versions, guarantees that you can install PISA as an unprivileged user, guarantees that PISA's dependencies are met, and allows for multiple versions of PISA to be installed simultaneously (each in a different virtualenv / conda env).

Note that it is also discouraged, but you _can_ install PISA as an unprivileged user using your system-wide Python install with the `--user` option to `pip`.
This is not quite as clean as a virtual environment, and the issue with coflicting package dependency versions remains.


### Install PISA

```bash
pip install -e $PISA[develop] -vvv
```
Explanation:
* First, note that this is ***not run as administrator***. It is discouraged to do so (and has not been tested this way).
* `-e $PISA` (or equivalently, `--editable $PISA`): Installs from source located at `$PISA` and  allows for changes to the source code within to be immediately propagated to your Python installation.
Within the Python library tree, all files under `pisa` are links to your source code, so changes within your source are seen directly by the Python installation. Note that major changes to your source code (file names or directory structure changing) will require re-installation, though, for the links to be updated (see below for the command for re-installing).
* `[develop]` Specify optional dependency groups. You can omit any or all of these if your system does not support them or if you do not need them.
* `-vvv` Be maximally verbose during the install. You'll see lots of messages, including warnings that are irrelevant, but if your installation fails, it's easiest to debug if you use `-vvv`.
* If a specific compiler is set by the `CC` environment variable (`export CC=<path>`), it will be used; otherwise, the `cc` command will be run on the system for compiling C-code.

__Notes:__
* You can work with your installation using the usual git commands (pull, push, etc.). However, these ***won't recompile*** any of the extension (i.e. pyx, _C/C++_) libraries. See below for how to reinstall PISA when you need these to recompile.


### Reinstall PISA

Sometimes a change within PISA requires re-installation (particularly if a compiled module changes, the below forces re-compilation).

```bash
pip install -e $PISA[develop] --force-reinstall -vvv
```

Note that if files change names or locations, though, the above can still not be enough.
In this case, the old files have to be removed manually (along with any associated `.pyc` files, as Python will use these even if the `.py` files have been removed).


### Compile the documentation

To compile a new version of the documentation to html (pdf and other formats are available by replacing `html` with `pdf`, etc.):
```bash
cd $PISA && sphinx-apidoc -f -o docs/source pisa
```

In case code structure has changed, rebuild the apidoc by executing
```bash
cd $PISA/docs && make html
```


### Test PISA

#### Unit Tests

Throughout the codebase there are `test_*.py` files and `test_*` functions within various `*.py` files that represent unit tests.
Unit tests are designed to ensure that the basic mechanisms of objects' functionality work.

These are all run, plus additional tests (takes about 15-20 minutes on a laptop) with the command
```bash
$PISA/pisa_tests/test_command_lines.sh
```

### Run a basic analysis

To make sure that an analysis can be run, try running an Asimov analysis of neutrino mass ordering (NMO) with the following (this takes about one minute on a laptop; note, though, that the result is not terribly accurate due to the use of coarse binning and low Monte Carlo statistics):
```bash
export PISA_FTYPE=fp64
$PISA/pisa/analysis/hypo_testing.py --logdir /tmp/nmo_test analysis \
    --h0-pipeline settings/pipeline/example.cfg \
    --h0-param-selections="ih" \
    --h1-param-selections="nh" \
    --data-param-selections="nh" \
    --data-is-mc \
    --min-method slsqp \
    --metric=chi2 \
    --pprint -v
```

The above command sets the null hypothesis (h0) to be the inverted hierarchy (ih) and the hypothesis to be tested (h1) to the normal hierarchy (nh).
Meanwhile, the Asimov dataset is derived from the normal hierarchy.

The significance for distinguishing NH from IH in this case (with the crude but fast settings specified) is shown by typing the follwoing command (which should output something close to 4.3):
```bash
hypo_testing_postprocess.py --asimov --detector "pingu_v39" --dir /tmp/nmo_test/hypo*
```
