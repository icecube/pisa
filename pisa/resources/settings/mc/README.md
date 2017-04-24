# Sample Settings

This directory contains configuration files for use in the `sample` service in
the `mc` stage. The goal of these configuration files is to make it easy to
load in simple pickle/hdf5 files into PISA for analysis. For processing of more
complex filetypes (e.g. I3 files) see `pisa/scripts/make_events_file.py`.

## Layout

The configuration file is split up into separate sections. The only compulsory
section is the `general` section. This section contains:
- `name`: name of the event sample
- `datadir`: filepath to the event sample
- `event_type`: a comma separated list listing the event topologies contained
  in the event sample. Most likely this will be something like "neutrinos,
  muons, noise".

Each listing in `event_type` needs to contain its own section. Currently the
following `event_type`s are implemented in the `sample` service:
### neutrinos

The `neutrinos` section contains the following:
- `flavours`: list of pdg codes of the neutrino flavours which have event
  samples available.
- `weights`: list of keys inside the event sample file to use for the weight
  respective to each `flavours`.
- `weight_units`: units of `weights`.
- `sys_list`: list of discrete systematic sets available.
- `baseprefix`: global prefix of the event sample filename.
- `basesuffix`: global suffix of the event sample filename.
- `keep_keys`: list of keys from the event sample to keep. 'all' keeps all the
  keys

The `neutrinos:aliases` section contains aliases. You can refer to existing
keys inside the left and right angle brackets, for example:
```
coszen = np.cos(<zenith>)
```

The `neutrinos:gen_lvl` section is for generator level (level 0) samples.

Then, a section exists for each entry in `sys_list`. Each section contains the
following:
- `nominal`: the nominal (baseline) value of the systematic set.
- `runs`: an array which gives the value of the systematic used for the
  generation of the systematically shifted sets.

For each `runs` a section exists in the format `neutrinos:sys_list:runs`, e.g.
`neutrinos:dom_eff:0.90`. Inside this section contains:
- `file_prefix`: the location of the file for this particular systematic set.

### muons

Very similar to the `neutrinos` section. Contains the following:
- `weight`: key inside the event sample file to use for the weight
- `weight_units`: units of `weight`.
- `sys_list`: list of discrete systematic sets available.
- `baseprefix`: global prefix of the event sample filename.
- `basesuffix`: global suffix of the event sample filename.
- `keep_keys`: list of keys from the event sample to keep. 'all' keeps all the
  keys

The `muons:aliases` section contains aliases. You can refer to existing
keys inside the left and right angle brackets.

Then, a section exists for each entry in `sys_list`. Each section contains the
following:
- `nominal`: the nominal (baseline) value of the systematic set.
- `runs`: an array which gives the value of the systematic used for the
  generation of the systematically shifted sets.

For each `runs` a section exists in the format `muons:sys_list:runs`, e.g.
`muons:dom_eff:0.9`. Inside this section contains:
- `file_prefix`: the location of the file for this particular systematic set.

### noise

Identical to the `muons` section.
