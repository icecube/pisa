# Flavor and interaction types

***NuFlav***, ***IntType***, ***NuFlavInt***, and ***NuFlavIntGroup*** are objects designed to easily work with neutrino flavors, interaction types, and combinations thereof. Source code: *pisa/utils/flavInt.py*

***FlavIntData*** and ***CombinedFlavIntData*** are container objects for data particular to neutrino flavors+interaction types. The former stores one datum for each flavor/interaction type combination, whereas the latter allows one to store a single data set for each grouping of flavor/interaction types. Source code: *pisa/utils/flavInt.py*

***Events*** subclasses *FlavIntData* but also contains standard metatadata pertinent to events. Source code: *pisa/core/events.py*

## Examples using flavor/interaction type objects

```python
import numpy as np
from pisa.utils import flavInt

# Create a flavor
numu = flavInt.NuFlav(14) # from code
numu = flavInt.NuFlav('nu_mu') # from str
numubar = -numu # flip parity

# Interaction type
cc = flavInt.IntType('Cc') # permissive
nc = flavInt.IntType(2) # from code

# A NuflavInt combines flavor and interaction type
numubarcc = flavInt.NuFlavInt('numu_bar_cc')
numucc = flavInt.NuFlavInt(numu, cc)
numubarcc = -numucc

numubarcc.flav # -> 'numubar'
numubarcc.int_type # -> 'cc'
numubar.particle # -> False
numubarcc.cc # -> True

# TeX string (nice for plots!)
numubarcc.tex # -> r'{{\bar\nu_\mu} \, {\rm CC}}'

# NuFlavIntGroup
nkg = flavInt.NuFlavIntGroup('numucc,numubarcc')
nkg = numucc + numubarcc # works!
nkg.flavints # -> (numu_cc, numubar_cc)
nkg.particles # -> (numu_cc,)
nkg -= numucc # remove a flavInt
nkg.flavints # -> (numubar_cc,)

# No intType spec=BOTH intTypes
nkg = flavInt.NuFlavIntGroup('numu')
nkg.flavints # -> (numu_cc, numu_nc)
nkg.cc_flavints # -> (numu_cc,)

# Loop over all particle CC flavInts
for fi in flavInt.NuFlavIntGroup('nuallcc'):
    print(fi)

# String, TeX
nkg = flavInt.NuFlavIntGroup('nuallcc')
print(nkg) # -> 'nuall_cc'
nkg.tex # -> r'{\nu_{\rm all}} \, {\rm CC}'

```

## Examples of using FlavIntData container object

```python
import numpy as np
from pisa.utils.flavInt import *

# Old way to instantiate an empty flav/int-type nested dict
import itertools
oldfidat = {}
flavs = ['nue', 'numu', 'nutau']
int_types = ['cc', 'nc']
for baseflav, int_type in itertools.product(flavs, int_types):
    for mID in ['', '_bar']:
        flav = baseflav + mID
        if not flav in oldfidat:
            oldfidat[flav] = {}
        oldfidat[flav][int_type] = None

# New way to instantiate, using a FlavIntData object
fidat = flavInt.FlavIntData()

# Old way to iterate over the old nested dicts
for baseflav, int_type in itertools.product(flavs, int_types):
    for mID in ['', '_bar']:
        flav = baseflav + mID
        oldfidat[flav][int_type] = {'map': np.arange(0,100)}
        m = oldfidat[flav][int_type]['map']

# New way to iterate through a FlavIntData object
for flavint in ALL_NUFLAVINTS:
    # Store data to flavint node
    fidat[flavint] = {'map': np.arange(0,100)}
    
    # Retrieve flavint node, then get the 'map' data
    m1 = fidat[flavint]['map']

    # Retrieve the 'map' data in one call (syntax works for all
    # nested dicts, where each subsequent arg is interpreted
    # as a key to a further-nested dict)
    m2 = fidat[flavint]['map']
    assert np.alltrue(m1 == m2)

# But if you really like nested dictionaries, you can still access a
# FlavIntData object as if it where a nested dict of
# `[flavor][interaction type]`:
fidat['nue_bar']['cc']['map']

# But you can access the element directly, and you can use the full string, and
# you don't have to use lower-case, '_' infix between 'numu' and 'bar', or know
# the exact structure of data container being used!
fidat[' numu bar  CC']['map']

# Get the entire branch starting at 'numu'
# (i.e., includes both interaction types)
fidat['numu'] # -> {'cc': ..., 'nc': ...}

# Save data to a bzip2-compressed JSON file
fidat.save('data.json.bz2')

# Save it to a HDF5 file (recognizes 'h5', 'hdf', and 'hdf5' extensions)
fidat.save('data.hdf5')

# Load, instantiate new object
fidat2 = flavInt.FlavIntData('data.json')

# Comparisons: intelligently recurses through the
# structure and any nested sub-structures
# (lists, dicts) when "==" is used
print(fidat == fidat2) # -> True

# There is a function for doing this in `pisa.utils.comparisons` that works
# with (almost) any nested object, including FlavIntData objects::
from pisa.utils.comparisons import recursiveEquality
print(recursiveEquality(fidat, fidat2)) # -> True
```

## Examples of using Events container object

```python
from pisa.core.events import Events

ev = Events('events/events__vlvnt__toy_1_to_80GeV_spidx1.0_cz-1_to_1_1e2evts_set0__unjoined__with_fluxes_honda-2015-spl-solmin-aa.hdf5')

print(ev.metadata)
```
Result:
```python
OrderedDict([('detector', ''), ('geom', ''), ('runs', []), ('proc_ver', ''), ('cuts', []), ('flavints_joined', [])])
```
