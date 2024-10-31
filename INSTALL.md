# Installation Guide

## Instruction to install PISA using Anaconda or Miniconda

The following commands are intended for execution in a terminal. If you encounter any problems with the installation, please file an issue and/or ask on slack.

1. Install the essential dependencies
   
   Choose either Anaconda (~4.4 GB) OR Miniconda (~480 MB)

   **Anaconda** (https://docs.anaconda.com/anaconda/), e.g.
   ```bash
   mkdir -p PATH_TO_ANACONDA/anaconda3
   wget https://repo.anaconda.com/archive/Anaconda3-<INSTALLER_VERSION>-Linux-x86_64.sh -O PATH_TO_ANACONDA/anaconda3/anaconda.sh
   bash PATH_TO_ANACONDA/anaconda3/anaconda.sh -b -u -p PATH_TO_ANACONDA/anaconda3
   rm PATH_TO_ANACONDA/anaconda3/anaconda.sh
   ```
   You can view a list of available installer versions at https://repo.anaconda.com/archive/.
   
   **Miniconda** (https://docs.anaconda.com/miniconda/), e.g.
   ```bash
   mkdir -p PATH_TO_ANACONDA/miniconda3
   wget https://repo.anaconda.com/miniconda/Miniconda3-<INSTALLER_VERSION>-Linux-x86_64.sh -O PATH_TO_ANACONDA/miniconda3/miniconda.sh
   bash PATH_TO_ANACONDA/miniconda3/miniconda.sh -b -u -p PATH_TO_ANACONDA/miniconda3
   rm PATH_TO_ANACONDA/miniconda3/miniconda.sh
   ```
   You can view a list of available installer versions at https://repo.anaconda.com/miniconda/.
   
   **If you install on the cobalts, rather use /data/user/YOURNAME/ than $HOME/ for PATH_TO_ANACONDA**
   
   **Note** that you will also need git. It is already installed on the cobalts, ask your local system administrator if it is not on your local machine. The other non-python requirements are listed [here](https://github.com/icecube/pisa/blob/master/INSTALL.md#required-dependencies) and should already come with the conda environment.

   Other required libraries will be installed automatically during the setup (listed in https://github.com/icecube/pisa/blob/master/setup.py). If something is missing, you can install it via pip afterwards.

2. Create an environment for the installation of PISA (after activating anaconda). You can choose your preferred version of python >= 3.10. E.g. use:

```bash
source PATH_TO_ANACONDA/miniconda3/bin/activate
conda create -n NAME_OF_YOUR_PISA_ENV python=3.10
```

3. Activate the newly created environment

```bash
conda activate NAME_OF_YOUR_PISA_ENV
```

4. Clone the PISA repository from github (https://github.com/icecube/pisa.git). You can also create your own fork first. For more information on how to obtain the pisa source code see [obtain-pisa-sourcecode](https://github.com/icecube/pisa/blob/master/INSTALL.md#obtain-pisa-sourcecode)

Define a directory for PISA sourcecode to live in.
```bash
export PISA="PATH_WHERE_PISA_SHOULD_LIVE/pisa
```
Add this line to your `~/.bashrc` file so you can refer to the `$PISA` variable without doing this everytime. 
PATH_WHERE_PISA_SHOULD_LIVE could for example be the same as PATH_TO_ANACONDA.

The clone the source code
```bash
mkdir -p $PISA
git clone https://github.com/icecube/pisa.git $PISA
```

5. Install PISA with the following command

```bash
pip install -e $PISA[develop] -vvv
```

**Explanation:**
   * First, note that this is ***not run as administrator***. It is discouraged to do so (and has not been tested this way).
   * `-e $PISA` (or equivalently, `--editable $PISA`): Installs from source located at `$PISA` and  allows for changes to the source code within to be immediately propagated to your Python installation.
   Within the Python library tree, all files under `pisa` are links to your source code, so changes within your source are seen directly by the Python installation. Note that major changes to your source code (file names or directory structure changing) will require re-installation, though, for the links to be updated (see below for the command for re-installing).
   * `[develop]` Specify optional dependency groups. You can omit any or all of these if your system does not support them or if you do not need them.
   * `-vvv` Be maximally verbose during the install. You'll see lots of messages, including warnings that are irrelevant, but if your installation fails, it's easiest to debug if you use `-vvv`.
   * If a specific compiler is set by the `CC` environment variable (`export CC=<path>`), it will be used; otherwise, the `cc` command will be run on the system for compiling C-code.

**Note** that you can work with your installation using the usual git commands (pull, push, etc.). However, these ***won't recompile*** any of the extension (i.e. pyx, _C/C++_) libraries. See below for how to reinstall PISA when you need these to recompile.


### Reinstall PISA

Sometimes a change within PISA requires re-installation (particularly if a compiled module changes, the below forces re-compilation).

```bash
pip install -e $PISA[develop] --force-reinstall -vvv
```

Note that if files change names or locations, though, the above can still not be enough.
In this case, the old files have to be removed manually (along with any associated `.pyc` files, as Python will use these even if the `.py` files have been removed).
them.


### Test PISA

 First activate your python environment (if not already active)

  ```bash
  source PATH_TO_ANACONDA/miniconda3/bin/activate
  conda activate NAME_OF_YOUR_PISA_ENV
  ```

#### Unit Tests

Throughout the codebase there are `test_*.py` files and `test_*` functions within various `*.py` files that represent unit tests.
Unit tests are designed to ensure that the basic mechanisms of objects' functionality work.

These are all run, plus additional tests (takes about 15-20 minutes on a laptop) with the command
```bash
$PISA/pisa_tests/test_command_lines.sh
```

#### Running a simple script

```bash
$PISA/pisa/core/pipeline.py --pipeline settings/pipeline/example.cfg  --outdir /tmp/pipeline_output --intermediate --pdf -v
```


## Instructions to install PISA using cvmfs and virtual environment

Assuming you already are in the directory where you want to store fridge/pisa source files and the python virtualenv and build pisa. You also need access to github through your account.


#### Clone into the fridge and pisa (ideally your own fork):

```
git clone git@github.com:icecube/wg-oscillations-fridge.git ./fridge

git clone git@github.com:USERNAME/pisa.git ./pisa
```


#### Load cvmfs environment:

```
unset OS_ARCH; eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/setup.sh`
```

`which python` should now point to `/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/bin/python`

**Note:** It's not tested whether this works with a cvmfs version newer than `py3-v4.2.1`.


#### Create virtual environment:

```
python -m venv ./YOUR_PISA_PY3_VENV
```


#### Start the virtual environment and install pisa from source:

```
source ./YOUR_PISA_PY3_VENV/bin/activate
```

(should now show that you are in the environment)

```
pip install -e pisa
```


#### Modify your environment by adding lines to `./YOUR_PISA_PY3_VENV/bin/activate`

Every time you want to use pisa now, you need to activate your virtual environment by running `source ./YOUR_PISA_PY3_VENV/bin/activate`. In order to set some useful environment variables you might want to add the following lines (or more if needed) to the end of the `./YOUR_PISA_PY3_VENV/bin/activate` script:

```
# PISA source
export PISA="/data/user/USERNAME/PATH_TO_PISA"

# set some custom environment variables and load fridge
export PISA_RESOURCES="/data/user/USERNAME/PATH_TO_FRIDGE/analysis/common"
export PISA_RESOURCES=$PISA_RESOURCES:"/data/user/USERNAME/PATH_TO_FRIDGE/analysis"

source "/data/user/USERNAME/PATH_TO_FRIDGE/setup_fridge.sh"

# Add this project to the python path
export PYTHONPATH=$FRIDGE_DIR:$PYTHONPATH

# Madison
export PISA_RESOURCES=/data/ana/LE:$PISA_RESOURCES
export PISA_RESOURCES=/data/ana/BSM/HNL/:$PISA_RESOURCES

export PISA_RESOURCES=$FRIDGE_DIR:$FRIDGE_DIR/analysis:$FRIDGE_DIR/analysis/YOUR_ANALYSIS/settings:$FRIDGE_DIR/analysis/YOUR_ANALYSIS/analysis:$FRIDGE_DIR/analysis/common:$PISA_RESOURCES

export PISA_CACHE=~/cache/
export PISA_FTYPE=fp64
export HDF5_USE_FILE_LOCKING='FALSE'
```


#### Install any additional packages that you might want

`pip install PACKAGE` (for example `jupyter`)


## Additional information


### Python Distributions

Obtaining Python and Python packages, and handling interdependencies in those packages tends to be easiest if you use a Python distribution, such as [Anaconda](https://www.continuum.io/downloads) or [Canopy](https://www.enthought.com/products/canopy).
Although the selection of maintained packages is smaller than if you use the `pip` command to obtain packages from the Python Package Index (PyPi), you can still use `pip` with these distributions.

The other advantage to these distributions is that they easily install without system administrative privileges (and install in a user directory) and come with the non-Python binary libraries upon which many Python modules rely, making them ideal for setup on e.g. clusters.

**Note**: Make sure that your `PATH` variable points to e.g. `<anaconda_install_dr>/bin` and *not* your system Python directory. To check this, type: `echo $PATH`; to udpate it, add `export PATH=<anaconda_install_dir>/bin:$PATH` to your .bashrc file.


### Required Dependencies

To install PISA, you'll need to have the following non-python requirements.
Note that these are not installed automatically, and you must install them yourself prior to installing PISA.
Also note that Python, HDF5, and pip support come pre-packaged or as `conda`-installable packages in the Anaconda Python distribution.
* [python](http://www.python.org) — version 3.x
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
    `conda install numba`


### Optional Dependencies

Optional dependencies. Some of these must be installed manually prior to installing PISA, and some will be installed automatically by pip, and this seems to vary from system to system. Therefore you can first try to run the installation, and just install whatever pip says it needed, or just use apt, pip, and/or conda to install the below before running the PISA installation.

* [LeptonWeighter](https://github.com/icecube/leptonweighter) Required for the `data.licloader_weighter` service. 
* [MCEq](http://github.com/afedynitch/MCEq) Required for `flux.mceq` service.
* [daemonflux](https://github.com/mceq-project/daemonflux) Recuired for `flux.daemon_flux` service.
* [nuSQuiDS](https://github.com/arguelles/nuSQuIDS) Required for `osc.nusquids` service.
* [pandas](https://pandas.pydata.org/) Required for datarelease (csv) stages.
* [OpenMP](http://www.openmp.org) Intra-process parallelization to accelerate code on on multi-core/multi-CPU computers.
  * Available from your compiler: gcc supports OpenMP 4.0 and Clang >= 3.8.0 supports OpenMP 3.1. Either version of OpenMP should work, but Clang has yet to be tested for its OpenMP support.
* [Photospline](https://github.com/icecube/photospline) Required for the `flux.airs` service. 
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


### Compile the documentation

To compile a new version of the documentation to html (pdf and other formats are available by replacing `html` with `pdf`, etc.):
```bash
cd $PISA && sphinx-apidoc -f -o docs/source pisa
```

In case code structure has changed, rebuild the apidoc by executing
```bash
cd $PISA/docs && make html
```
