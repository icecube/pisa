# Stage: Flux

The sole purpose of this stage used to be to get the expected flux produced in the atmosphere for different particles at different energies and angles.
It is therefore possible to reweight a dataset using the effective area to investigate uncertainties from using different flux models.
The neutrino files come from several different measurements made at different positions on the earth and currently can only use azimuth-averaged data.
The ability to load azimuth-dependent data has been implemented, but it is not immediately obvious how one should treat these at the energy range relevant for IceCube studies.

Now, an extended set of services implement different types of flux expectations and systematic modifications thereof.

## Services (incomplete)

### honda_ip

This service implements an integral-preserving interpolation of the atmospheric neutrino tables produced by the Honda group.
Only 2D files (azimuth averaged) can currently be loaded.

For some more information on the integral-preserving interpolation, and why it is an accurate choice, please see the following link:

[NuFlux on the IceCube wiki](https://wiki.icecube.wisc.edu/index.php/NuFlux)

Since this is a link on the IceCube wiki, you will need the access permissions for this page.

### daemonflux

Implementation of DAEMONFLUX based on [https://arxiv.org/abs/2303.00022]. 
For the example use see jupyter notebook `pisa_examples/test_daemonflux_stage.ipynb`, as well as an example config file at `pisa_examples/resources/settings/pipeline/IceCube_3y_neutrinos_daemon.cfg`. 

Important: [daemonflux](https://github.com/mceq-project/daemonflux) pyhon package is required dependency for this stage.
