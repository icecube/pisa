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
1. Define a directory for PISA source code to live in, and create the directory. For example:<br>
    ```bash
    export PISA=~/src/pisa
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

If you like, set up passwordless ssh access to github:<br>
https://help.github.com/articles/connecting-to-github-with-ssh

In your terminal, define a directory for PISA source code to live in, e.g.,<br>
```bash
export PISA=~/src/pisa
```

Also add this line to your `~/.bashrc` file so you can refer to the `$PISA` variable without doing this every time.

Create the above directory:<br>
```bash
mkdir -p $PISA
```

Below we describe two different ways of setting up the PISA Python environment:<br>

The [first (default)](#default-miniforge-distribution) obtains Python and Python packages, as well as any non-Python binary libraries upon which many Python libraries rely, from the [Miniforge](https://conda-forge.org/docs/user/introduction/) distribution. This makes it ideal for setup on e.g. clusters, but also works well for your personal computer.<br>

The [second (alternative)](#alternative-cvmfs-and-virtualenv) assumes you have access to IceCube's CernVM-FS (CVMFS) repository and would like to use one of its Python and software distributions. Our instructions have only been tested for the [`py3-v4.2.1` distribution](https://docs.icecube.aq/icetray/main/info/cvmfs.html#py3-v4-2).

<details>
  <summary>in case of installation one of IceCube's Cobalt nodes</summary>
   
   Consider using `/data/user/<USERNAME>` instead of e.g. `$HOME` as installation location below.
</details>


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

### Alternative: CVMFS and virtualenv

Load the CVMFS environment:<br>
```bash
unset OS_ARCH; eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/setup.sh`
```
<details>
  <summary>on one of IceCube's Cobalt nodes</summary>
   
   Verify that `which python` outputs `/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/bin/python`.
</details>

After switching to the directory where you want to install PISA, create the virtual environment:<br>
```bash
python -m venv ./<VENV DIRECTORY NAME>
```

Activate the virtual environment:<br>
```bash
source ./<VENV DIRECTORY NAME>/bin/activate
```
A shell prompt with the virtual environment's directory name in parentheses should now confirm the successful activation.
 
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

<details>
  <summary>with default packages only and without development tools</summary>
   
  ```bash
  pip install -e $PISA -vvv
  ```
</details>

or, if desired,

<details>
  <summary>including optional packages and development tools</summary>
   
  ```bash
  pip install -e $PISA[develop] -vvv
  ```
</details>

If the installation went smoothly, you are now ready to run a quick test<br>
```bash
pisa-distribution_maker --pipeline settings/pipeline/IceCube_3y_neutrinos.cfg --outdir <OUTPUT PATH> --pdf
```
This command should have created the folder `<OUTPUT PATH>` containing a pdf with output maps for different neutrino types and interactions.

## Additional information

### Required Dependencies

With the exception of `Python` itself (and possibly `git`), the installation methods outlined above should not demand the _manual_ prior installation of any Python or non-Python requirements for PISA.
Support for all of these comes pre-packaged or as `conda`/`mamba`-installable packages in the Miniforge Python distribution.
* [python](http://www.python.org) — version >= 3.6 and <= 3.10 required
  * Miniforge & CVMFS: built in
* [pip](https://pip.pypa.io) version >= 1.8 and <= 25 required
  * Miniforge & CVMFS: built in
* [git](https://git-scm.com)
  * Miniforge: `mamba install git`
  * or system wide, e.g. in Ubuntu<br>
    `sudo apt install git`
  * it is already installed on IceCube's Cobalt nodes

### Installation
* First, note that the installation is ***not run as administrator***. It is discouraged to do so (and has not been tested this way).
* `-e $PISA` (or equivalently, `--editable $PISA`): Installs from source located at `$PISA` and  allows for changes to the source code within to be immediately propagated to your Python installation.
   Within the Python library tree, all files under `pisa` are links to your source code, so changes within your source are seen directly by the Python installation. Note that major changes to your source code (file names or directory structure changing) will require re-installation, though, for the links to be updated (see below for the command for re-installing).
* `[develop]` Specify optional dependency groups. You can omit any or all of these if your system does not support them or if you do not need them.
* `-vvv` Be maximally verbose during the install. You'll see lots of messages, including warnings that are irrelevant, but if your installation fails, it's easiest to debug if you use `-vvv`.
* If a specific compiler is set by the `CC` environment variable (`export CC=<path>`), it will be used; otherwise, the `cc` command will be run on the system for compiling C-code.

**Note** that you can work with your installation using the usual git commands (pull, push, etc.). However, these ***won't recompile*** any of the extension (i.e. pyx, _C/C++_) libraries. See below for how to reinstall PISA when you need these to recompile.


### Re-installation

Sometimes a change within PISA requires re-installation (particularly if a compiled module changes, the below forces re-compilation).

```bash
pip install -e $PISA[develop] --force-reinstall -vvv
```

**Note** that if files change names or locations, though, the above can still not be enough.
In this case, the old files have to be removed manually (along with any associated `.pyc` files, as Python will use these even if the `.py` files have been removed).


### Compile the documentation

In case you installed the optional "develop" dependencies: compile a new version of the documentation to html via
```bash
cd $PISA && sphinx-apidoc -f -o docs/source pisa
```

In case code structure has changed, rebuild the apidoc by executing
```bash
cd $PISA/docs && make html
```
(Run `make help`  to check which other documentation formats are available.)
