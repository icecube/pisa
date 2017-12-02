# Stage: Reconstruction

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
An example is provided in the `pisa_examples/resources/events/` directory.
One can also specify the choice of weights to be used in constructing these histograms.
Note that this must be one of the columns in the events file, or you can set it to `None` to produce the histograms unweighted.

### vbwkde


### param

The `param` service takes parametric resolution functions dependent on true energy as input to create the transform. More general comments from the `hist` section above on the reconstruction kernel also apply here.

For this service, one must specify the location of the file containing the function definitions (`reco_paramfile`). The file should contain a dictionary of valid combinations of flavour-interaction keys, with each of these pointing to another dictionary of "dimension" keys, typically `"coszen"` and `"energy"`. Each of these then points to a list of dictionaries which hold the concrete definitions. The total resolution function is interpreted as the linear superposition of the individual distributions, where each distribution identifier needs to be available from `scipy.stats`. The total resolution function is interpreted as the weighted sum of the individual distributions. A "fraction" determines the relative weight of each distribution, while kwargs "scale" and "loc" serve the purpose of rescaling and recentering it with respect to its "standard" form (cf. scipy docs).

#### Treatment of physical boundaries

The parameterisations (either single distributions or superpositions thereof) will typically have a non-negligible fraction extending out into non-physical regions, i.e., below zero energy or -1 and +1 in cosine zenith, while MC events from a realistic simulation will have reconstruction error distributions truncated at these boundaries. This service provides three options for mitigating this problem:
When `coszen_flipback` is set to `True`, any probability leaking out into the unphysical cosine zenith range is mirrored back in (only possible with linear cosine zenith binning). This can result in effective coszen error distributions that differ drastically from the original parameterisation. A more shape-conserving option is to set either `only_physics_domain_sum` or `only_physics_domain_distwise` to `True`. In these two cases, those parts of the distributions spanning the physical range (in energy and cosine zenith) are rescaled so that the integral over the physical range returns 1. There is only a difference between the two in the case where two or more distributions are to be superimposed: while `only_physics_domain_distwise` first truncates and renormalises the constituting distributions one-by-one, `only_physics_domain_sum` truncates and renormalises the superimposed distribution (and thus leading to an integral which is exactly 1). In general, all options will yield compatible results if the parameterisations only have negligible non-physical contributions. Not setting one of these three options to `True` will lead to the parameterisations being taken at face-value, and no correction for spill-over into non-physical regions will be applied.

#### Systematics
* `e_res_scale`, `cz_res_scale`, `e_reco_bias`, `cz_reco_bias`, all applied to all distributions involved for a particular dimension, with `res_scale_ref` determining how the rescaling is done (currently only mode `"zero"` supported)
