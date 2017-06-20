## Installation Guide

### Quickstart

In Ubuntu you can peform the following steps to perform a clean, full (all optional dependencies included), editable install of PISA on your local PC; test the installation; and run a quick analysis.

```bash
# Install required and optional system packages
sudo apt-get install git swig libhdf5-10 llvm-3.9-dev python2.7 python-pip virtualenv

# Create and activate a virtual environment: so fresh and so clean
virtualenv my_virtual_env
source my_virtual_env/bin/activate

# Create a parent directory for sourcecode
mkdir my_virtual_env/src

# Obtain the PISA sourcecode (must have access to the WIPACrepo/pisa repo)
git clone https://github.com/jllanfranchi/pisa.git --branch cake my_virtual_env/src/pisa

# Install PISA and its python package dependencies (optional dependency
# categories are in brackets). Note that sometimes an install issue with the
# below command can be overcome by simply re-running a second time; also, add
# -vvv to any pip command that fails for verbose output for debugging issues
pip install -e my_virtual_env/src/pisa/[cuda,numba,develop] \
    -r my_virtual_env/src/pisa/requirements.txt

# Define the precision you want GPU code to run in (single or double)
export PISA_FTYPE=single

# Run the physics tests (append --ignore-cuda-errors if no CUDA support)
test_consistency_with_pisa2.py -v

# EXAMPLE: Run a Monte Carlo pipeline to produce, store, and plot its expected
# distributions at the output of each stage
pipeline.py --settings settings/pipeline/example.cfg \
    --outdir /tmp/pipeline_output --intermediate --pdf -v

# EXAMPLE: Run the Asimov NMO analysis; leave off "_gpu" to run CPU-only
# version
hypo_testing.py --logdir /tmp/test \
    --h0-pipeline settings/pipeline/example_gpu.cfg \
    --h0-param-selections="ih" \
    --h1-param-selections="nh" \
    --data-param-selections="nh" \
    --data-is-mc \
    --minimizer-settings settings/minimizer/bfgs_settings_fac1e11_eps1e-4_mi20.json \
    --metric=chi2 \
    --pprint -v

# Display the significance for distinguishing hypothesis h1 from h0
hypo_testing_postprocess.py --asimov --logdir /tmp/test/*

# Leave the virtual environment (run the `source...` command above to re-enter
# the virtual environment at a later time)
deactivate
```


### Python Distributions

Obtaining Python and Python packages, and handling interdependencies in those packages tends to be easiest if you use a Python distribution, such as [Anaconda](https://www.continuum.io/downloads) or [Canopy](https://www.enthought.com/products/canopy).
Although the selection of maintained packages is smaller than if you use the `pip` command to obtain packages from the Python Package Index (PyPi), you can still use `pip` with these distributions.

The other advantage to these distributions is that they easily install without system administrative privileges (and install in a user directory) and come with the non-Python binary libraries upon which many Python modules rely, making them ideal for setup on e.g. clusters.

* **Note**: Make sure that your `PATH` variable points to e.g. `<anaconda_install_dr>/bin` and *not* your system Python directory. To check this, type: `echo $PATH`; to udpate it, add `export PATH=<anaconda_install_dir>/bin:$PATH` to your .bashrc file.
* Python 2.7.x can also be found from the Python website [https://www.python.org/downloads](https://www.python.org/downloads/) or pre-packaged for almost any OS.


### Required Dependencies

To install PISA, you'll need to have the following non-python requirements.
Note that these are not installed automatically, and you must install them yourself prior to installing PISA.
Also note that Python, SWIG, HDF5, and pip support come pre-packaged or as `conda`-installable packages in the Anaconda Python distribution.
* [python](http://www.python.org) — version 2.7.x required (tested with 2.7.11)
  * Anaconda: built in
  * Otherwise, if on Linux it will be pre-packaged; in Ubuntu:<br>
    `sudo apt-get install python2.7`
* [pip](https://pip.pypa.io) version >= 1.8 required
  * Anaconda:<br>
    `conda install pip`
  * In Ubuntu:<br>
    `sudo apt-get install python-pip`
* [git](https://git-scm.com)
  * In Ubuntu,<br>
    `sudo apt-get install git`
* [swig](http://www.swig.org)
  * In Ubuntu,<br>
    `sudo apt-get install swig`
* [hdf5](http://www.hdfgroup.org/HDF5) — install with `--enable-cxx` option
  * In Ubuntu,<br>
    `sudo apt-get install libhdf5-10`

Required Python modules that are installed automatically when you use the pip command detailed later:
* [cython](http://cython.org)
* [decorator](https://pypi.python.org/pypi/decorator)
* [dill](http://trac.mystic.cacr.caltech.edu/project/pathos/wiki/dill.html)
* [h5py](http://www.h5py.org)
* [line_profiler](https://pypi.python.org/pypi/line_profiler): detailed profiling output<br>
* [matplotlib](http://matplotlib.org) >= 2.0 required
* [numpy](http://www.numpy.org) version >= 1.11.0 required
* [pint](https://pint.readthedocs.org) >= 0.8 required
* [scipy](http://www.scipy.org) version >= 0.17 required
* [setuptools](https://setuptools.readthedocs.io) version >= 0.18 required
* [simplejson](https://github.com/simplejson/simplejson) version >= 3.2.0 required
* [tables](http://www.pytables.org)
* [uncertainties](https://pythonhosted.org/uncertainties)


### Optional Dependencies

Optional dependencies. Some of these must be installed manually prior to installing PISA, and some will be installed automatically by pip, and this seems to vary from system to system. Therefore you can first try to run the installation, and just install whatever pip says it needed, or just use apt, pip, and/or conda to install the below before running the PISA installation.

* [llvm](http://llvm.org) Compiler needed by Numba. This is automatically installed in Anaconda alongside `numba`, but must be installed manually on your system otherwise.
  * Anaconda<br>
    `conda install numba`
  * In Ubuntu,<br>
    `sudo apt install llvm-3.9-dev`
* [virtualenv](https://virtualenv.pypa.io/en/stable/) Use virtual environments to e.g. create a "clean" installation and/or to have multiple multiple versions installed, one version per virtual environment. To speed up installation (at the cost of a less "clean" environment), you can specify the `--system-site-packages` option to `virtualenv` to make use of already-installed Python packages.
  * Anaconda<br>
    `conda install virtualenv`
  * Otherwise,<br>
    `pip install virtualenv`
* [OpenMP](http://www.openmp.org) Intra-process parallelization to accelerate code on on multi-core/multi-CPU computers.
  * Available from your compiler: gcc supports OpenMP 4.0 and Clang >= 3.8.0 supports OpenMP 3.1. Either version of OpenMP should work, but Clang has yet to be tested for its OpenMP support.
* [PyROOT](https://root.cern.ch/pyroot) Necessary to read ROOT cross sections files; must install ROOT on your system in addition to PyROOT. There is no `pip` package for either.
  * Ubuntu 15.x and 16.04:<br>
    `sudo apt-get install root-system libroot-bindings-python*`
* [numba](http://numba.pydata.org) Just-in-time compilation of decorated Python functions to native machine code via LLVM. This can accelerate certain routines significantly. If not using Anaconda to install, you must have LLVM installed already on your system (see above).
  * Installed alongside PISA if you specify option `['numba']` to `pip`
* [Pylint](http://www.pylint.org): Static code checker and style analyzer for Python code. Note that our (more or less enforced) coding conventions are codified in the pylintrc file in PISA, which will automatically be found and used by Pylint when running on code within a PISA package.<br>
  * Installed alongside PISA if you specify option `['develop']` to `pip`
* [PyCUDA](https://mathema.tician.de/software/pycuda): run certain routines on Nvidia CUDA GPUs (must have compute 2.0 or greater capability)<br>
  * Installed alongside PISA if you specify option `['cuda']` to `pip`
* [Sphinx](http://www.sphinx-doc.org/en/stable/) version >= 1.3
  * Installed alongside PISA if you specify option `['develop']` to `pip`
* [recommonmark](http://recommonmark.readthedocs.io/en/latest/) Translator to allow markdown docs/docstrings to be used; plugin for Sphinx. (Required to compile PISA's documentation.)
  * Installed alongside PISA if you specify option `['develop']` to `pip`
* [versioneer](https://github.com/warner/python-versioneer) Automatically get versions from git and make these embeddable and usable in code. Note that the install process is unique since it first places `versioneer.py` in the PISA root directory, and then updates source files within the repository to provide static and dynamic version info.
  * Installed alongside PISA if you specify option `['develop']` to `pip`


### Set up your environment

* Create a "parent" directory (the directory into which you wish for the PISA sourcecode to live).
Note that subsequent steps will create a directory named `pisa` within the parent directory you've chosen, so you don't need to create the actual `pisa` directory yourself.
A common choice for a parent dir would be a directory named `src` in your home folder: `$HOME/src`.
```
mkdir -p <parent dir>
```

* To make life easier in the future (and to make these instructons easy to follow), define the environment variable `PISA`.
E.g., for the bash shell, edit your `.bashrc` file and add the line
```
export PISA=<parent dir>/pisa
```
Load this variable into your *current* environment by sourcing your `.bashrc` file:
```bash
. ~/.bashrc
```
(it will be loaded autmatically for all new shells).


### Github setup

1. Create your own [github account](https://github.com/)
1. Obtain access to the `jllanfranchi/pisa` repository by emailing (as a verifiable IceCube member) your **Github username** to Justin Lanfranchi (jll1062+pisa@phys.psu.edu) and copy John Kelley (jkelley@icecube.wisc.edu)


#### SSH vs. HTTPS access to repository

You can interact with Github repositories either via SSH (which allows password-less operation) or HTTPS (which gets through firewalls that don't allow for SSH).
To choose one or the other just requires a different form of the repsitory's URL (the URL can be modified later to change method of access if you change your mind).


##### Set up password-less access (SSH)

If you use the SSH URL, you can avoid passwords altogether by uploading your public key to Github:

1. [Check for an existing SSH key](https://help.github.com/articles/checking-for-existing-ssh-keys/)
1. [Generate a new SSH key if none already exists](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/)
1. [Add your SSH key to github account](https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account)
1. [Test the ssh connection](https://help.github.com/articles/testing-your-ssh-connection)


##### Set up password caching (SSH or HTTPS)

Git 1.7.10 and later allows storing your password for some time in memory so you aren't asked every time you interact with Github via the command line.
Follow instructions [here](https://help.github.com/articles/caching-your-github-password-in-git/).
This is particularly useful for HTTPS or if you use SSH but do not wish to store a key pair on the computer/server you use.


### Obtain PISA sourcecode

#### Developing PISA: Forking

If you wish to modify PISA and contribute your code changes back to the PISA project (*highly recommended!*), fork `jllanfranchi/pisa` from Github.
*(How to work with the `cake` branch of PISA will be detailed below.)*

Forking creates your own version of PISA within your Github account.
You can freely create your own *branch*, modify the code, and then *add* and *commit* changes to that branch within your fork of PISA.
When you want to share your changes with `jllanfranchi/pisa`, you can then submit a *pull request* to `jllanfranchi/pisa` which can be merged by the PISA administrator (after the code is reviewed and tested, of course).

* Navigate to the [PISA github page](https://github.com/jllanfranchi/pisa) and fork the repository by clicking on the ![fork](images/ForkButton.png) button.
* Clone the repository into the `$PISA` directory via one of the following commands (`<github username>` is your Github username):
  * either SSH access to repo:<br>
`git clone git@github.com:<github username>/pisa.git $PISA
`
  * or HTTPS access to repo:<br>
`git clone https://github.com/<github username>/pisa.git $PISA`


#### Using but not developing PISA: Cloning

If you just wish to pull changes from github (and not submit any changes back), you can just clone the sourcecode without creating a fork of the project.

* Clone the repository into the `$PISA` directory via one of the following commands:
  * either SSH access to repo:<br>
`git clone git@github.com:jllanfranchi/pisa.git $PISA`
  * or HTTPS access to repo:<br>
`git clone https://github.com/jllanfranchi/pisa.git $PISA`


### Ensuring a Clean Install: Using Virtualenv

It is absolutely discouraged to install PISA as a root (privileged) user.
PISA is not vetted for security vulnerabilities, so should always be installed and run as a regular (unprivileged) user.

It is suggested (but not required) that you install PISA within a virtual environment.
This minimizes contamination by PISA of a system-wide (or other) Python installation, guarantees that you can install PISA as an unprivileged user, guarantees that PISA's dependencies are met, and allows for multiple versions of PISA to be installed simultaneously.
(The latter is accomplished by creating multiple virtual environments and installing a version of PISA into each.)

Note that you can also install PISA as an unprivileged user (besides using `virtualenv`) in the following ways:
1. Use a user-installed Python distribution such as Anaconda or Canopy. Version conflicts with other packages can still be an issue, though, so installing or upgrading another package separately from PISA could break your PISA installation.
1. Install with the `--user` option to `pip`. This is not quite as clean as a virtual environment, and the issue with coflicting package dependency versions remains.

To create a new virtual environment with no extra Python packages:
```bash
virtualenv my_virtual_env
```
this is the cleanest option (as little opportunity for version conflicts to arise), but the rest of the installation could take additional time as dependencies will have to be downloaded and installed to the virtual environment that are already installed in the system Python.

To include system-installed packages in your virtual environment instead:
```bash
virtualenv my_virtual_env --system-site-packages
```

Next, "activate" the virtual environment (`PATH` will now point to your Python executable there):
```bash
source my_virtual_env/bin/activate
```

At this point, proceed to the instructions for installing or running PISA.
When you want to leave the virtual environment, type
```bash
deactivate
```

If you ever want to remove a virtual environment, deactivate it (if it's activated) and simply remove the directory into which it was installed.
For the above example with `my_virtual_env` installed to the current directory:
```bash
rm -rf my_virtual_env
```


### Install PISA

```bash
pip install -e $PISA[cuda,numba,develop] -r $PISA/requirements.txt
```
Explanation of the above command:
* First, note that this is ***not run as administrator***. It is discouraged to do so (and has not been tested this way).
* `-e $PISA` or `--editable $PISA`: Installs from source located at `$PISA` and  allows for changes to the source code within to be immediately propagated to your Python installation.
Within the Python library tree, all files under `pisa` are glorified links to the corresponding files in your source directory, so changes within your source are seen directly by the Python installation.
* `[cuda,numba,develop]` Specify optional dependency groups. You can omit any or all of these if your system does not support them or if you do not need them.
* `-r $PISA/requirements.txt`: Specifies the file containing PISA's dependencies for `pip` to install prior to installing PISA.
This file lives at `$PISA/requirements.txt`.
* If a specific compiler is set by the `CC` environment variable, it will be used, otherwise the `cc` command will be invoked on the system. Note that there have been some problems using `clang` under OSX, since versions of Clang older than 3.8.0 do not support `OpenMP`. Upgrade Clang or use `gcc` (if installed) and specify its executable with `CC` environment variable.

__Notes:__
* You can work with your installation using the usual git commands (pull, push, etc.). However, these ***won't recompile*** any of the extension (i.e. pyx, _C/C++_) libraries. See below for how to handle this case.
* To test if your system compiled the gaussins.pyx Cython file with OpenMP threading support (and to see the speedup you can get using multiple cores for this module), you can run the following test script:
  `python $PISA/pisa/utils/test_gaussians.py --speed`
  The output should show speedups for increasing numbers of threads; if not, then it's likely that OpenMP did not compile on your system.


### Reinstall PISA

Sometimes a change within PISA requires re-installation (particularly if a compiled module changes, the below forces re-compilation).

```bash
pip install --editable $PISA -r $PISA/requirements.txt --force-reinstall
```

Note that if files change names or locations, though, the above can still not be enough.
In this case, the old files have to be removed manually (along with any associated `.pyc` files, as Python will use these even if the `.py` files have been removed).

### Compiling the Documentation

To compile a new version of the documentation to html (pdf and other formats are available by replacing `html` with `pdf`, etc.):
```bash
cd $PISA && sphinx-apidoc -f -o docs/source pisa
```

In case code structure has changed, rebuild the apidoc by executing
```bash
cd $PISA/docs && make html
```


### Testing PISA

#### Unit Tests

Throughout the codebase there are `test_*.py` files and `test_*` functions within various `*.py` files that represent unit tests.
Unit tests are designed to ensure that the basic mechanisms of objects' functionality work.

These are all run, plus additional tests (takes about 15-20 minutes on a laptop) with the command
```bash
$PISA/tests/test_command_lines.sh
```

#### Physics Tests

Physics tests ensure that the distributions produced by services are consistent with previous or known-good results.
One example is the script that checks consistency of the results in the current PISA against results previously obtained with PISA 2.2:
```bash
test_consistency_with_pisa2.py -v
```

### Running a Basic Analysis

To make sure that an analysis can be run, try running an Asimov analysis of neutrino mass ordering (NMO) with the following (this takes about one minute on a laptop; note, though, that the result is not terribly accurate due to the use of coarse binning and low Monte Carlo statistics):
```bash
export PISA_FTYPE=fp64
hypo_testing.py --logdir /tmp/nmo_test \
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
