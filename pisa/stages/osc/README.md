# Stage: Oscillations

The purpose of this stage is to modify the modelled atmospheric neutrino flux by applying the appropriate oscillation probability in each energy and zenith bin through the earth, for each flavour.
At the end of the stage, the oscillated neutrino flux for each flavour is given.


### prob3

This service calculates the neutrino oscillation probabilities based on the original `Prob3` code, which at its core, relies on a 3-flavour analytic solution to the neutrino oscillation propagation over constant matter density.
For a realistic earth model, small enough constant density layers are chosen to accuratly describe the matter density through the earth.
The `Prob3` code, initially written by the Super-Kamiokande collaboration in C/C++, is publicly available here: http://www.phy.duke.edu/~raw22/public/Prob3++/

We use a custom re-implementation in python/numba.

To use this service, one must set values for all of the 3-flavour oscillation parameters: `theta12`, `theta13`, `theta23`, `deltam21`, `deltam31` and `deltacp`.
One can set hierarchy-dependent versions in the settings file by adding `.nh` before the name in the `params` (see `theta23` in the example pipeline settings file for more details).

The other advantage of `Prob3` is that it can fully take in to account matter effects in a very quick way.
It is fed a model of the density of the Earth via the `earth_model` parameter which contains a discretised model of the Earth density.
The two columns in the file are the radii of the boundaries and then densities in the sections.
PISA has the 4, 10, 12 and 59 layer versions of the PReliminary Earth reference Model (PREM), the source of which can be found here:

http://www.sciencedirect.com/science/article/pii/0031920181900467

One also has control over 3 values of the electron fraction: `YeI`, `YeM` and `YeO`, which are the values in the inner core, mantle and outer core respectively.
Finally, in PISA one can set the detector depth (since the standard `Prob3` has the detector at the surface) and the height above the Earth's surface from which to begin propagation of the neutrinos.
