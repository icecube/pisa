# Installation Guide

## Quick start

_Note that terminal commands below are intended for the bash shell. You'll have to translate if you use a different shell._

1. _(optional)_ Obtain a github user ID if you don’t have one already. (Otherwise, you will only have read access.)<br>
    https://github.com
    * Sign up for Github education pack for many features for free, too<br>
        https://education.github.com/pack
1. _(optional)_ Fork PISA on github so you have your own copy to work from<br>
    https://github.com/icecube/pisa/fork
1. _(optional)_ Set up passwordless ssh access to github<br>
    https://help.github.com/articles/connecting-to-github-with-ssh
1. In your terminal, define a directory for PISA source code to live in. For example:<br>
    `export PISA=~/src/pisa`
    * Add this line to your `~/.bashrc` file so you can refer to the `$PISA` variable without doing this every time.
1. Create the directory<br>
    `mkdir -p $PISA`
1. Clone the PISA repository to your local computer (on the command line)
    * If you forked PISA above<br>
      * If you set up ssh authentication above<br>
         `git clone git@github.com:<YOUR GITHUB USER ID HERE>/pisa.git $PISA`
      * Otherwise<br>
         `git clone https://github.com/<YOUR GITHUB USER ID HERE>/pisa.git $PISA`
    * If you didn't fork PISA<br>
      * If you set up ssh authentication above<br>
         `git clone git@github.com:icecube/pisa.git $PISA`
      * Otherwise<br>
         `git clone https://github.com/icecube/pisa.git $PISA`
1. Install the latest Miniforge python distribution for either Mac or Linux (as your user, _not_ as root), if you don’t have it already
    https://conda-forge.org/download/
1. _(optional)_ If you declined to update your shell profile to automatically initialize conda, activate the base environment as prompted at the end
1. Create and activate a new conda environment, with a python version compatible with the python requirements below. We suggest using mamba as a drop-in replacement for conda, for example<br>
    ```bash
    mamba create -n <YOUR ENV NAME HERE> python=3.10
    mamba activate <YOUR ENV NAME HERE>
    ```
1. Install PISA
    * either with default packages only and without development tools<br>
     `pip install -e $PISA -vvv`
    * or, if desired, including optional packages and development tools<br>
     `pip install -e $PISA[develop] -vvv`
1. Run a quick test<br>
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
