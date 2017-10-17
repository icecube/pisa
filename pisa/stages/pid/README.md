# Stage: Particle ID

The purpose of this stage is to simulate the event classification of
PINGU, sorting the reconstructed nue CC, numu CC, nutau CC, and NC
events into the track and cascade channels.

## Services

There exists two services for this particular stage: `hist` and `param`.

### hist
This service utilises pre-computed particle ID scores in its determination of
how events should be classified given certain output channels e.g. as
track-like or cascade-like. This has the advantage that one can utilise much
more sophisticated classification methods such as multivariate analysis (MVA)
techniques however, the computation time to generate these scores grows
exponentially with the complexity of these techniques. Once they are
calculated, the pid score gives a single value which quantifies the likelihood
of a given event being classified as a certain channel e.g. track-like.
Specifications given as input to this service give the pid score value which is
used as the minimum cut-off to distinguish an event as belonging to this
channel, the events which have pid score's under this cut-off value are instead
classified as belonging to the other channel(s).

### param
This service uses parameterisations to know what fraction of events to classify in to each channel. Currently this is supported as a function of energy only. The information is stored in a `.json` file of the form:

```
{
  flav_int: {
    "trck": lambda E function,
    "cscd": 1 - trck function
  },
  etc...
}
```

The provided flav_int are expected to be `numu_cc`, `nue_cc`, `nutau_cc` and `nuall_nc`. It is expected that the difference between `nu` and `nubar` is negligible. The lambda functions are parsed using `eval` and so the syntax used in your functions must be loaded in the `param.py` file and must also match that i.e. since `numpy` is imported as `np` it must be written as such in your `.json` file.

Related links:
* [2013-11-20, Status of Particle Identification on PINGU, JP](https://wikispaces.psu.edu/download/attachments/173476942/20131120_jpamdandre_PINGUPID.pdf?version=1&modificationDate=1384959568000&api=v2)
* [2014-03-26, PID update, JP](https://wikispaces.psu.edu/download/attachments/194447201/20140326_jpamdandre_PIDinFrame.pdf?version=1&modificationDate=1395806349000&api=v2)
