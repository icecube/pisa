# Stage: Flux

The purpose of this stage is to get the expected flux produced in the atmosphere for different particles at different energies and angles.
It is therefore possible to reweight a dataset using the effective area to investigate uncertainties from using different flux models.
The neutrino files come from several different measurements made at different positions on the earth and currently can only use azimuth-averaged data.
The ability to load azimuth-dependent data has been implemented, but it is not immediately obvious how one should treat these at the energy range relevant for IceCube studies.

## Services

Only one service is currently supported in PISA for dealing with atmospheric neutrino flux tables: `honda`.

### honda

This service implements the atmospheric neutrino tables produced by the Honda group.
Details of the supported files can be found in the Notes section of the docstrings in the file.
Both 2D and 3D files can currently be loaded, but interpolation is only supported with the 2D files.
Currently there are two interpolation choices:

* 'bisplrep' - A simple b-spline representation. This is quick.
* 'integral-preserving' - A slower, but more accurate choice.

The details of these interpolation methods can be found in the Notes section of the docstrings in the file.
For some more information on the second of these choices, and why it is a more accurate choice than the first, please see the following link:

[NuFlux on the IceCube wiki](https://wiki.icecube.wisc.edu/index.php/NuFlux)

Since this is a link on the IceCube wiki, you will need the access permissions for this page.

### daemonflux

Implementation of DAEMONFLUX based on [https://arxiv.org/abs/2303.00022]. 
For the example use see jupyter notebook `pisa_examples/test_daemonflux_stage.ipynb`, as well as an example config file at `pisa_examples/resources/settings/pipeline/IceCube_3y_neutrinos_daemon.cfg`. 

Important: [daemonflux](https://github.com/mceq-project/daemonflux) pyhon package is required dependency for this stage.