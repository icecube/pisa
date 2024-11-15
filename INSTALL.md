# Installation Guide

_Note that all terminal commands below are intended for the bash shell. You'll have to translate if you use a different shell._
## Quick start
This guide will enable you to _use_ PISA within about five minutes. If you are more interested in contributing to PISA's development, please refer to the [advanced installation guide](#advanced-installation-guide) instead.

1. Install [git](https://git-scm.com) if you don't have it already
    * In Ubuntu,<br>
       ```bash
      sudo apt install git
       ```
1. In your terminal, define a directory for PISA source code to live in, and create the directory. For example:<br>
    ```bash
    export PISA=~/src/pisa
    mkdir -p $PISA
    ```
1. Clone the PISA repository to your local computer<br>
    ```bash
    git clone https://github.com/icecube/pisa.git $PISA
    ```
1. Install the latest Miniforge python distribution for either Mac or Linux (as your user, _not_ as root)<br>
    https://conda-forge.org/download/<br>
    * In case you declined to update your shell profile to automatically initialize conda, activate the base environment as prompted at the end.
1. Create and activate a new conda environment, with a python version compatible with the python requirements below. Using mamba as a drop-in replacement for conda:<br>
    ```bash
    mamba create -n <ENV NAME HERE> python=3.10
    mamba activate <ENV NAME HERE>
    ```
1. Install PISA with default packages only and without development tools<br>
     ```bash
     pip install -e $PISA -vvv
     ```
1. Run a quick test<br>
   ```bash
   pisa-distribution_maker --pipeline settings/pipeline/IceCube_3y_neutrinos.cfg --outdir <OUTPUT PATH HERE> --pdf
   ```
   This command should have created the folder `<OUTPUT PATH HERE>` containing a pdf with output maps for different neutrino types and interactions.

## Advanced installation guide

### Preparation

To ensure that you can contribute to PISA's development, first obtain a GitHub user ID if you don’t have one already, and optionally sign up for GitHub education pack for many features for free, too:<br>
https://education.github.com/pack

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

Install [git](https://git-scm.com) if you don't have it already. On, e.g., Ubuntu: `sudo apt install git`.

Next, clone the PISA repository to your local computer. On the command line,
  * if you set up ssh authentication above<br>
      ```bash
       git clone git@github.com:<YOUR GITHUB USER ID HERE>/pisa.git $PISA
      ```
  * otherwise<br>
      ```bash
      git clone https://github.com/<YOUR GITHUB USER ID HERE>/pisa.git $PISA
      ```

Below we describe two different sets of pre-installation steps:<br>

The [first (default)](#default-miniforge-distribution) obtains Python and Python packages, as well as any non-Python binary libraries upon which many Python libraries rely, from the Miniforge distribution. This makes it ideal for setup on e.g. clusters, but also works well for your personal computer.<br>

The [second (alternative)](#alternative-cvmfs-and-virtualenv) assumes you have access to IceCube's cvmfs repository and would like to use one of its Python and software distributions. Our instructions have only been tested for the [`py3-v4.2.1` distribution](https://docs.icecube.aq/icetray/main/info/cvmfs.html#py3-v4-2). 

### Default: Miniforge distribution

Install the latest Miniforge python distribution for either Mac or Linux (as your user, _not_ as root) from https://conda-forge.org/download/.
1. _(optional)_ If you declined to update your shell profile to automatically initialize conda, activate the base environment as prompted at the end
1. Create and activate a new conda environment, with a python version compatible with the python requirements below. We suggest using mamba as a drop-in replacement for conda, for example<br>
    ```bash
    mamba create -n <YOUR ENV NAME HERE> python=3.10
    mamba activate <YOUR ENV NAME HERE>
    ```

### Alternative: CVMFS and virtualenv

Switch to the directory where you want to install PISA and create a virtual python environment (`virtualenv`).<br>
Load the CVMFS environment:<br>
```bash
unset OS_ARCH; eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/setup.sh`
```
On the cobalt machines of the IceCube collaboration, make sure that `which python` now outputs `/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/bin/python`.

Create the virtual environment:<br>
```bash
python -m venv ./<YOUR VENV NAME>
```

Load the virtual environment:<br>
```bash
source ./<YOUR VENV NAME>/bin/activate
```
The shell should now indicate that you are in the environment.
 
### Final step: install and test PISA
You can now proceed to install PISA either with default packages only and without development tools<br>
```bash
pip install -e $PISA -vvv
```
or, if desired, including optional packages and development tools<br>
```bash
pip install -e $PISA[develop] -vvv
```

If the installation went smoothly, you are now ready to run a quick test<br>
```bash
pisa-distribution_maker --pipeline settings/pipeline/IceCube_3y_neutrinos.cfg --outdir <TEST OUTPUT PATH HERE> --pdf
```
This command should have created the folder `<TEST OUTPUT PATH HERE>` containing a pdf with output maps for different neutrino types and interactions.

## Additional information

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
