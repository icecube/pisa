# Installation Guide

_Note that all terminal commands below are intended for the bash shell. You'll have to translate if you use a different shell._
## Quick start
This guide will enable you to _use_ PISA within about five minutes. If you are more interested in contributing to PISA's development, please refer to the [advanced installation guide](#advanced-installation-guide) instead.

1. Install the latest Miniforge Python distribution for either Mac or Linux (as your user, _not_ as root)<br>
    https://conda-forge.org/download/<br>
    * In case you declined to update your shell profile to automatically initialize conda, activate the base environment as prompted at the end.
1. In your terminal, create and activate a new conda environment, with a Python version compatible with the Python requirements below<br>
    ```bash
    conda create -n <ENV NAME> python=3.10
    conda activate <ENV NAME>
    ```
1. If your system doesn't already have it, install [git](https://git-scm.com) into this environment. (We use `mamba` as a drop-in replacement for the `conda` package manager.)
     ```bash
     mamba install git
     ```
1. Define a directory for the PISA source code (~ 500 MB) to live in, and create the directory. For example:<br>
    ```bash
    export PISA=<PATH TO PISA>/pisa
    mkdir -p $PISA
    ```
1. Clone the PISA repository to your local computer<br>
    ```bash
    git clone https://github.com/icecube/pisa.git $PISA
    ```
1. Install PISA with default packages only and without development tools<br>
     ```bash
     pip install -e $PISA -vvv
     ```
1. Run a quick test<br>
   ```bash
   pisa-distribution_maker --pipeline settings/pipeline/IceCube_3y_neutrinos.cfg --outdir <OUTPUT PATH> --pdf
   ```
   This command should have created the folder `<OUTPUT PATH>` containing a pdf with output maps for different neutrino types and interactions.

## Advanced installation guide

### Preparation

To ensure that you can contribute to PISA's development, first obtain a GitHub user ID if you don’t have one already.<br>
<details>
  <summary>optional sign up for GitHub education pack for many features for free</summary>
  https://education.github.com/pack
</details>

Fork PISA on GitHub so you have your own copy of the repository to work on, from which you can create pull requests:<br>
https://github.com/icecube/pisa/fork

If you like, set up passwordless ssh access to GitHub:<br>
https://help.github.com/articles/connecting-to-github-with-ssh

In your terminal, define a directory for PISA source code to live in, e.g.,<br>
```bash
export PISA=<PATH TO PISA>/pisa
```
<details>
  <summary>suggested paths</summary>


   Since the PISA source occupies several hundreds of MB of drive space: on one of IceCube's Cobalt nodes, consider setting `<PATH_TO_PISA>` to `/data/user/<USERNAME>` instead of e.g. `$HOME`.
</details>
Also add this line to your `~/.bashrc` file so you can refer to the `$PISA` variable without doing this every time.

Create the above directory:<br>
```bash
mkdir -p $PISA
```

Below we describe two different ways of setting up the PISA Python environment:<br>

> [!NOTE]
> Miniforge or the venv virtual environment folder into which all Python packages will be installed can be placed wherever you please. (Note that there are [disadvantages](https://pybit.es/articles/a-better-place-to-put-your-python-virtual-environments/) to putting either in your local PISA repository's top-level directory `$PISA`.) However, on one of IceCube's Cobalt nodes, again consider `/data/user/<USERNAME>` instead of e.g. `$HOME`.

The [first (default)](#default-miniforge-distribution) obtains Python and Python packages, as well as any non-Python binary libraries upon which many Python libraries rely, from the [Miniforge](https://conda-forge.org/docs/user/introduction/) distribution. This makes it ideal for setup on e.g. clusters, but also works well for your personal computer.<br>

The [second (alternative)](#alternative-cvmfs-and-venv) assumes you have access to IceCube's CernVM-FS (CVMFS) repository and would like to use one of the Python installations it provides as the "base" of a [venv](https://docs.python.org/3/library/venv.html). Our instructions have only been tested for the [`py3-v4.2.1` distribution](https://docs.icecube.aq/icetray/main/info/cvmfs.html#py3-v4-2).

### Default: Miniforge distribution

Install the latest Miniforge Python distribution for either Mac or Linux (as your user, _not_ as root) from https://conda-forge.org/download/.
<details>
  <summary>command suggestions</summary>
   
   ```bash
   mkdir -p <PATH TO MINIFORGE>/miniforge3
   wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash "Miniforge3-$(uname)-$(uname -m).sh" -p <PATH TO MINIFORGE>/miniforge3 -u
   rm "Miniforge3-$(uname)-$(uname -m).sh"
   ```

   **Notes:**
   * To perform SHA-256 checksum verification of the Miniforge installer, download the installer (`.sh`) for your platform whose name contains the release version and the corresponding `.sha256` checksum file from https://github.com/conda-forge/miniforge/releases/latest, then execute ```sha256sum -c "Miniforge3-<RELEASE VERSION>-$(uname)-$(uname -m).sh.sha256"```.
   * You can decline having your shell profile updated to automatically initialize conda. In this case
  ```bash
   eval "$(<PATH TO MINIFORGE>/miniforge3/bin/conda shell.bash hook)"
  ```
   will activate the base environment as prompted at the end of the Miniforge installation script. Doing so is required to proceed with this installation and whenever PISA is used again. The successful activation is indicated by the shell prompt `(base)`. An overview of the packages in the base environment can be gained via `mamba/conda list`.
</details>

It is recommended to keep the base environment stable. Therefore, create and activate a new conda environment, with a Python version compatible with the Python requirements below:<br>
 ```bash
 conda create -n <ENV NAME> python=3.10
 conda activate <ENV NAME>
 ```
A shell prompt with `<ENV NAME>` name in parentheses should now confirm the successful activation.

### Alternative: CVMFS and venv

Load the CVMFS environment:<br>
```bash
unset OS_ARCH; eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/setup.sh`
```
<details>
  <summary>on one of IceCube's Cobalt nodes</summary>
   
   Verify that `which python` outputs `/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/bin/python`.
</details>

```bash
python -m venv /PATH/TO/<VENV NAME>
```

Activate the virtual environment:<br>
```bash
source /PATH/TO/<VENV NAME>/bin/activate
```
A shell prompt with the virtual environment's name in parentheses should now confirm the successful activation.
 
### Common final steps: clone, install and test PISA

Install [git](https://git-scm.com) if you [don't have it](#required-dependencies) already.

Next, clone the PISA repository to your local computer. On the command line,
<details>
  <summary>with ssh authentication</summary>
   
  ```bash
  git clone git@github.com:<YOUR GITHUB USER ID>/pisa.git $PISA
  ```
</details>

<details>
  <summary>without</summary>
   
   ```bash
   git clone https://github.com/<YOUR GITHUB USER ID>/pisa.git $PISA
   ```
   See https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories#cloning-with-https-urls if you have issues authenticating in this case.
</details>


You can now proceed to install PISA, either

with default packages only and without development tools
```bash
pip install -e $PISA -vvv
```
or, if desired, including optional packages and development tools
```bash
pip install -e $PISA[develop] -vvv
```

> [!NOTE]
>  * The installation is ***not run as administrator***. It is discouraged to do so (and has not been tested this way).
>  * `-e $PISA` (or equivalently, `--editable $PISA`): Installs from source located at `$PISA` and  allows for changes to the source code within to be immediately propagated to your Python installation.
>   Within the Python library tree, all files under `pisa` are links to your source code, so changes within your source are seen directly by the Python installation. Major changes to your source code (file names or directory structure changing) will require re-installation, though, for the links to be updated (see below for the command for re-installing).
> * `[develop]` Specify optional dependency groups. You can omit any or all of these if your system does not support them or if you do not need them.
> * `-vvv` Be maximally verbose during the install. You'll see lots of messages, including warnings that are irrelevant, but if your installation fails, it's easiest to debug if you use `-vvv`.
> * If a specific compiler is set by the `CC` environment variable (`export CC=<path>`), it will be used; otherwise, the `cc` command will be run on the system for compiling C-code.
>  You can work with your installation using the usual git commands (pull, push, etc.). However, these ***won't recompile*** any of the extension (i.e. pyx, _C/C++_) libraries. See below for how to re-install PISA when you need these to recompile.

If the installation went smoothly, you are now ready to run a quick test<br>
```bash
pisa-distribution_maker --pipeline settings/pipeline/IceCube_3y_neutrinos.cfg --outdir <OUTPUT PATH> --pdf
```
This command should have created the folder `<OUTPUT PATH>` containing a pdf with output maps for different neutrino types and interactions.

## Additional information

### Ensure a clean install using venv or conda env

It is absolutely discouraged to install PISA as a root (privileged) user.
PISA is not vetted for security vulnerabilities, so should always be installed and run as a regular (unprivileged) user.

It is suggested (but not required) that you install PISA within a virtual environment (or in a conda env if you're using Anaconda, Miniconda, or Miniforge Python distributions).
This minimizes cross-contamination by PISA of a system-wide (or other) Python installation with conflicting required package versions, guarantees that you can install PISA as an unprivileged user, that PISA's dependencies are met, and allows for multiple versions of PISA to be installed simultaneously (each in a different venv / conda env).

Note that it is also discouraged, but you _can_ install PISA as an unprivileged user using your system-wide Python install with the `--user` option to `pip`.
This is not quite as clean as a virtual environment, and the issue with conflicting package dependency versions remains.

### Re-installation

Sometimes a change within PISA requires re-installation (particularly if a compiled module changes, the below forces re-compilation).

```bash
pip install -e $PISA[develop] --force-reinstall -vvv
```

**Note** that if files change names or locations, though, the above can still not be enough.
In this case, the old files have to be removed manually (along with any associated `.pyc` files, as Python will use these even if the `.py` files have been removed).

### Required Dependencies

With the exception of `Python` itself (and possibly `git`), the installation methods outlined above should not demand the _manual_ prior installation of any Python or non-Python requirements for PISA.
Support for all of these comes pre-packaged or as `conda`/`mamba`-installable packages in the Miniforge Python distribution.
* [python](http://www.python.org) — version >= 3.8 and <= 3.10 required
  * Miniforge & CVMFS: built in
* [pip](https://pip.pypa.io) version >= 1.8 and <= 25 required
  * Miniforge & CVMFS: built in
* [git](https://git-scm.com)
  * Miniforge: `mamba install git`
  * or system wide, e.g. in Ubuntu<br>
    `sudo apt install git`
  * it is already installed on IceCube's Cobalt nodes

Required Python modules whose installation is taken care of by pip are specified in [setup.py](setup.py).

### Optional Dependencies

Some of the following optional dependencies must be installed manually prior to installing PISA, and some will be installed automatically by pip, and this seems to vary from system to system. Therefore you can first try to run the installation, and just install whatever pip says it needed, or just use apt, pip, or conda/mamba to install the below before running the PISA installation.

* [emcee](https://github.com/dfm/emcee) Required for MCMC sampling functionality in the `llh_client`& `llh_server` utils modules and the `analysis` module.
* [GLoBES wrapper](https://github.com/atrettin/GLoBES_wrapper) Required for `osc.globes` service.
* [LeptonWeighter](https://github.com/icecube/leptonweighter) Required for `data.licloader_weighter` service.
* [MCEq](https://github.com/afedynitch/MCEq) Required for `create_barr_sys_tables_mceq.py` script.
* [nuSQuiDS](https://github.com/arguelles/nuSQuIDS) Required for `osc.nusquids` service.
* [OpenMP](https://openmp.org) Intra-process parallelization to accelerate code on on multi-core/multi-CPU computers.
  * Available from your compiler: gcc supports OpenMP 4.0 and Clang >= 3.8.0 supports OpenMP 3.1. Either version of OpenMP should work, but Clang has yet to be tested for its OpenMP support.
* [Photospline](https://github.com/icecube/photospline) Required for `flux.airs` service.
* [Pylint](https://pylint.org) Static code checker and style analyzer for Python code. Note that our (more or less enforced) coding conventions are codified in the pylintrc file in PISA, which will automatically be found and used by Pylint when running on code within a PISA package.
  * Installed alongside PISA if you specify option `['develop']` to `pip`
* [Pytest](https://docs.pytest.org/) Python testing framework. Used by a couple unit tests.
  * Installed alongside PISA if you specify option `['develop']` to `pip`
* [recommonmark](http://recommonmark.readthedocs.io/en/latest/) Translator to allow markdown docs/docstrings to be used; plugin for Sphinx. (Required to compile PISA's documentation.)
  * Installed alongside PISA if you specify option `['develop']` to `pip`
* [ROOT >= 6.12.04 with PyROOT](https://root.cern.ch) Required for `absorption.earth_absorption` service, and to read ROOT cross section files in the `crossSections` utils module. Due to a bug in ROOT's Python support (documented here https://github.com/IceCubeOpenSource/pisa/issues/430), you need at least version 6.12.04 of ROOT.
* [Sphinx >= 1.3](https://www.sphinx-doc.org)
  * Installed alongside PISA if you specify option `['develop']` to `pip`
* [Read the Docs Sphinx Theme](https://github.com/readthedocs/sphinx_rtd_theme)
  * Installed alongside PISA if you specify option `['develop']` to `pip`
* [versioneer](https://github.com/python-versioneer/python-versioneer) Automatically get versions from git and make these embeddable and usable in code. Note that the install process is unique since it first places `versioneer.py` in the PISA root directory, and then updates source files within the repository to provide static and dynamic version info.
  * Installed alongside PISA if you specify option `['develop']` to `pip`
* [black](https://github.com/psf/black) Format your Python code, _automatically_, with typically very nice results!

### Compile the documentation

In case you installed the optional "develop" dependencies, you can compile a (new) version of the documentation to html via
```bash
cd $PISA && sphinx-apidoc -f -o docs/source pisa
```

In case code structure has changed, rebuild the API documentation by executing
```bash
cd $PISA/docs && make html
```
(Run `make help`  to check which other documentation formats are available.)
