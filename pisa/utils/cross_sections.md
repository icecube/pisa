# Stage: Cross Sections

Cross sections used to generate Monte Carlo simulations are stored in $PISA/pisa.utils/cross_sections/cross_sections.json. New cross sections can be added to this file and existing cross sections can be loaded, plotted, and worked with via the class $PISA/utils/crossSections.py:CrossSections.


## cross_sections.json format
```python
{
	"<cross section version>": {
		"energy": [e0, e1, ..., eN],
		"nue": {
			"cc": [xs(e0), xs(e1), ..., xs(eN)],
			"nc": [...]
		},
		"nue_bar": { ... },
		"numu": { ... },
		"numu_bar": { ... },
		"nutau": { ... },
		"nutau_bar": { ... },
	},
	"<cross section version>": {
	 	...
	},
	...
}
```

## CrossSections class
Subclasses flavInt.FlavIntData but adds the energies at which cross sections are defined.

## Examples of working with cross sections
```python
import pisa.utils.crossSections as crossSections

# Load GENIE 2.6.4 cross sections from the default file in PISA
xs = crossSections.CrossSections(ver='genie_2.6.4')

# Plot (cross section/energy) as a function of energy to verify they are what
# you expect
xs.plot()

# Equivalent methods for getting the energy samples; input is flexible
energy_vals = xs.get('e')
energy_vals = xs.get('E')
energy_vals = xs.get('energy')

# Retrieve values for cross sections at the above-returned energies...
xs_vals_nuecc = xs.get('nuecc')
xs_vals_nuecc = xs.get('nue cc')
xs_vals_nuecc = xs.get('nuecc')

# Retrieve both CC and NC cross sections as a dictionary
xs_vals_nue = xs.get('nue')
```
