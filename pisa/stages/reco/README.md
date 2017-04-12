# Stage 4: Reconstruction

The purpose of this stage is to apply an assumed detector resolution to the incoming flux to convert the events' "true" energies and directions into the "reconstructed" ones.

## Services

Three services are currently supported in PISA for dealing with converting from truth to reconstructed variables: `hist`, `vbwkde` and `param`.

### hist

The `hist` service (and indeed all `hist` services in PISA) takes Monte Carlo events as inputs and directly histograms them to create the transform.
For the reconstruction stage this transform takes the form of a 2N-dimensional histogram (N is the number of dimensions in your analysis) which matches from truth to reconstructed variables.
In it, each bin along the truth direction is a distribution over all of the reconstructed space.

Consider the 1-dimensional case for simplicity where we have a reconstruction kernel which transforms from true cosZenith to reconstructed cosZenith.
Say we have 10 bins in between -1 and 0 at truth level and an equivalent 10 bins at reco level.
The first bin of the reconstruction kernel will contain a map over -1 to 0 in reconstructed cosZenith that is all of the events from the truth bin -1 to -0.9.
Thus it tells us the contribution to the reconstructed distribution from every individual truth bin.
These maps in reconstructed space are normalised to the total number of events from that truth bin.

For this service, one must specify where to pull these Monte Carlo events from.
Typically this should be one of the files in the `resources/events/` directory.
One can also specify the choice of weights to be used in constructing these histograms.
Note that this must be one of the columns in the events file, or you can set it to `None` to produce the histograms unweighted.

### vbwkde


### param

The `param` service takes parametric resolution functions dependent on true energy as input to create the transform. More general comments from the `hist` section above on the reconstruction kernel also apply here.

For this service, one must specify the location of the file containing the function definitions (`reco_paramfile`). The file should contain a dictionary of valid combinations of flavour-interaction keys, with each of these pointing to another dictionary of "dimension" keys, typically `"coszen"` and `"energy"`. Each of these then points to a dictionary which holds the concrete definitions. This should have a `"dist"` entry specifying the types of the distributions building up the total resolution function (i.e. `"distID1+distID2+...+distIDN"` or simply `"distID"` for only a single distribution), where each distribution identifier needs to be available from `scipy.stats`. The total resolution function is interpreted as the weighted sum of the individual distributions (signified by the `"+"`). A "fraction" determines the relative weight of each distribution, while "scale" and "loc" keys serve the purpose of rescaling and recentering it with respect to its "standard" form (cf. scipy docs).

Here is how to assign these properties to the individual distributions: you can either append a "global" index, i.e., use `"scale2"` (`"loc2"`, `"fraction2"`) for denoting `"distID2"` in `"distID1+distID2+...+distIDN"`, or you can append the distribution identifier together with the "distribution type" index to the desired property, e.g. `"scale_norm2"` (and correspondingly for "loc" and "fraction") for setting the properties of the second distribution of type `scipy.stats.norm` from the left in, say, `"poisson+norm+norm"`. Combinations of the two nomenclatures presented here are also allowed.

Setting `coszen_flipback` to `True` or `False` determines whether events otherwise lost at the boundaries of the cosine zenith range should be mirrored back in (only possible with linear cosine zenith binning) or not.

#### Systematics
* `e_res_scale`, `cz_res_scale`, `e_reco_bias`, `cz_reco_bias`, all applied to all distributions involved, with `res_scale_ref` determining how the rescaling is done (currently only mode `"zero"` supported)
