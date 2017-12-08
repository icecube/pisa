# Flavor and interaction types

***NuFlav***, ***IntType***, ***NuFlavInt***, and ***NuFlavIntGroup*** are objects designed to easily work with neutrino flavors, interaction types, and combinations thereof. Source code: *pisa/utils/flavInt.py*

***FlavIntData*** and ***CombinedFlavIntData*** are container objects for data particular to neutrino flavors+interaction types. The former stores one datum for each flavor/interaction type combination, whereas the latter allows one to store a single data set for each grouping of flavor/interaction types. Source code: *pisa/utils/flavInt.py*

***Events*** subclasses *FlavIntData* but also contains standard metatadata pertinent to events. Source code: *pisa/core/events.py*

## Examples using flavor/interaction type objects
```python
import numpy as np
import pisa.core.events as events
import pisa.utils.flavInt as flavInt

# Create a flavor
numu = flavInt.NuFlav(14) # from code
numu = flavInt.NuFlav('nu_mu') # from str
numubar = -numu # flip parity

# Interaction type
cc = flavInt.IntType('Cc') # permissive
nc = flavInt.IntType(2) # from code

# A NuflavInt combines flavor and interaction type
numucc = flavInt.NuFlavInt('numu_bar_cc')
numucc = flavInt.NuFlavInt(numu, cc)
numubarcc = -numucc

numubarcc.flav # -> 'numubar'
numubarcc.intType # -> 'cc'
numubar.isParticle # -> False
numubarcc.isCC # -> True

# TeX string (nice for plots!)
numubarcc.tex # -> '{{\bar\nu_\mu} \, {\rm CC}}'

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
nkg.ccFlavInts # -> (numu_cc,)

# Loop over all particle CC flavInts
for fi in flavInt.NuFlavIntGroup('nuallcc'):
    print fi

# String, TeX
nkg = flavInt.NuFlavIntGroup('nuallcc')
print nkg # -> 'nuall_cc'
nkg.tex # -> '{\nu_{\rm all}} \, {\rm CC}'

```

## Examples of using FlavIntData container object
```python
import numpy as np
from flavInt import *

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
for flavint in flavInt.ALL_NUFLAVINTS:
    # Store data to flavint node
    fidat.set(flavint, {'map': np.arange(0,100)})
    
    # Retrieve flavint node, then get the 'map' data
    m1 = (fidat.get(flavint))['map']

    # Retrieve the 'map' data in one call (syntax works for all
    # nested dicts, where each subsequent arg is interpreted
    # as a key to a further-nested dict)
    m2 = fidat.get(flavint, 'map')
    assert np.alltrue(m1 == m2)

# But if you really like nested dictionaries, you can still access a
# FlavIntData object as if it where a nested dict:
fidat['nue_bar']['cc']['map']

# ...easier to use set() and get() though! Can use strings, but the
# interface is awesome now: you don't have to know that you need
# lower-case, '_' infix between 'numu' and 'bar', or structure of
# data structure being used!
fidat.get('numu bar CC', 'map')

# Get the entire branch starting at 'numu'
# (i.e., includes both interaction types)
fidat.get('numu') # -> {'cc': ..., 'nc': ...}

# Save it to a JSON file
fidat.save('data.json')

# Save it to a HDF5 file (recognizes 'h5', 'hdf', and 'hdf5' extensions)
fidat.save('data.hdf5')

# Load, instantiate new object
fidat2 = flavInt.FlavIntData('data.json')

# Comparisons: intelligently recurses through the
# structure and any nested sub-structures
# (lists, dicts) when "==" is used
print fidat == fidat2 # -> True

# There is a function for doing this in utils.Utils.py:
import utils.utils as utils
print utils.recursiveEquality(nested_obj1, nested_obj2) # -> True
```

## Examples of using Events container object
```python
from pisa.core.events import Events

ev = Events('events/events__vlvnt__toy_1_to_80GeV_spidx1.0_cz-1_to_1_1e2evts_set0__unjoined__with_fluxes_honda-2015-spl-solmin-aa.hdf5')

print ev.metadata
```
Result:
```
{'cuts': array(['analysis'], 
       dtype='|S8'),
 'detector': 'pingu',
 'geom': 'v36',
 'kinds_joined': array(['nuall_nc', 'nuallbar_nc'], 
       dtype='|S11'),
 'proc_ver': '5',
 'runs': array([388, 389, 390])}
```

## Examples of using CombinedFlavIntData object
```python
from pisa.utils.flavInt import CombinedFlavIntData as CFID

cfidat = CFID(flavint_groupings='nuecc+nuebarcc; numucc+numubarcc;'
              'nutaucc+nutaubarcc; nuallnc; nuallbarnc')

# Iterate through the defined flavInt groupings
[cfidat.set(grp, np.random.rand(10)) for grp in cfidat.keys()]

# Save to JSON file
cfidat.save('/tmp/cfidat.json') # save to JSON file

# Load from JSON file
cfidat2 = CFID('/tmp/cfidat.json')

# Make an independent (deep) copy
cfidat3 = CFID(cfidat2)

# Force all data to be equal
[cfidat2.set(k, np.arange(10)) for k in cfidat2.keys()]

# Eliminate redundancy (should be just one unique dataset!)
cfidat2.deduplicate()
len(cfidat2) # -> 1
cfidat2.keys() # -> ['nuall+nuallbar']
```
