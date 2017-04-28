# Stage 3: Effective Area

The purpose of this stage is the conversion of the incoming neutrino flux at the detector into an event count. This includes the interaction of the neutrinos, as well as the triggering and event selection criteria (filters), so that the resulting flux is at _analysis level_. 

The total event __counts__ in each bin is simply calculated as the product of 

![Events](images/events.png)


## Services

Three effective area services are surported:
  * The simplest builds effective areas directly from histogrammed MC events.
  * Another __smooths__ these effective areas from histogrammed MC events.
  * Lastly, the effective areas can be constructed from parameterisations.

### hist

This service takes the input MC events in HDF5 format. It then reads the events weights for each flavour and interaction type from the datafile and creates histogram of the effective area. The structure of the datafile is
```
flavour / int_type / value
```
where
  *  `flavour` is one of `nue, nue_bar, numu, numu_bar, nutau, nutau_bar`
  *  `int_type` is one of `cc` or `nc`
  *  `values` is one of 
    * `weighted_aeff`: the effective area weight per event (see below)
    * `true_energy` : the true energy of the event
    * `true_coszen` : the true cos(zenith) of the event

**NOTE:** the `weighted_aeff` is directly obtained from the `OneWeight` in `IceTray`; specifically it is calculated as
 
![Weights](images/weight.png)

where 
  * `OneWeight` is the `OneWeight` stored per event in `.i3` data files
  * `N_events` is the number of events **per data file**
  * `N_files` is the total number of data files included
  * the factor 2 reflects that in IceCube simulation both `nu` and `nubar` are simulated in the same runs, while we have seperate effective areas for each of them.

To obtain the effective area, these weights are histrogrammed in the given binning as a function of cos(zenith) and energy. To obtain the effective area in each bin _i_, the sum of weights in this bin is divided with the solid angle and energy range covered by the bin. 

![AeffMC](images/aeffmc.png)

### smooth

This service starts out from the exact same histograms as in the hist stage, btu then applies smoothing to them. For sparse MC samples, often 'holes' appear in the histograms, meaning bins with zero events. To aboid this, the historgams and their according uncertainties are smeared out with gaussian kernels, along both energy and coszen axes consecutively.
The then smoothed version are the input to spline comnstructions. Using cubic splines we achieve a smoothing that can in addition be steared by smoothing parameters in the stage's cfg file (higer value, more aggresive smoothing...too much smoothing will yield wrong results, take caution).
The splines are also performed in two iterations, first in the energy dimension and afterwards in coszen.

Since some processes, like nutau CC interactions, have hard cutoffs below a certain energy (here around 3.5 GeV), rows that are completely zero over all coszen bins are automatically excluded from the smoothing and remain zero.

The Energy range is extended for the smoothing part using a point-reflection at the edges. This is to be able to have the tails of splines under control for example. These are just helper points that are discarded in the output.


### param

This service uses pre-made datatables to describe the energy dependence of the effective area, while the cos(zenith) dependence is described as a functional form (i.e parametrization).
* **Energy dependence**: This is stored in a `.json` file which will be a dictionary of the form:
    ```
    {
      flav_int : {
        "energy": [
	  list of energy values
	],
	"aeff": [
	  list of effective area values
	]
      },
      etc...
    }
    ```

   where `energy` is in GeV and `aeff` is in m^2. These lists are interpolated using 1D linear interpolation with all ranges outside the interpolation range set to 0. Note that the provided `flav_int` can have either `_nc` for each flavour or just pass a `nuall_nc` (and another `nuallbar_nc`) instead. In the `params` for this service the location of this file is stored under `aeff_energy_paramfile`.
   
* **cos(zenith) dependence**: This is stored in a `.json` file which will be a dictionary of the form:
    ```
    {
      flav_int : lambda cz function,
      etc...
    }
    ```
    The function strings are evaluated using `eval` to return python function objects. The provided `flav_int` has the option to not include the `bar` versions and the service will just use the standard version instead. Note this is ONLY for the coszenith dependence, since the normalisation comes from the energy dependence and so will account for the difference between `nu` and `nubar` there. In the `params` for this service the location of this file is stored under `aeff_coszen_paramfile`.

The total effective area is calculated for each bin _i_ by evaluating the both parametrization functions at the bin centers in energy and cos(zenith) and multiplying them, where the cos(zenith) functions are also normalized to unity over the energy range. 

![AeffPar](images/aeffpar.png)


