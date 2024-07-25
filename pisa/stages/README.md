# pisa.stages

Directories are PISA stages, and within each directory can be found the services implementing the respective stage.

## Anatomy of a typical stage

### Constructor
The constructor of the stage should look like the following:

```python
class mystage(Stage):
  """
  Docstring (mandatory!)
  
  What is the purpose of this stage? Short 1-2 sentence description
  
  Parameters:
  -----------
  
  params :
    Expected params are .. ::
      a : dimensionless Quantity
      b : dimensionless Quantity
  
  something_else : float
    Description
    
  Notes:
  ------
  
  More info, references, etc...
  
  """
  def __init__(self, something_else, **std_kwargs):

    expected_params = ('a', 'b')
    
    super().__init__(expected_params=expected_params, **std_kwargs)
    
    self.foo = something_else
```

The constructor arguments are passed in via the stage config file, which in this case would need to look something like:

 ```ini
 [stage_dir.mystage]

calc_mode = ...
apply_mode = ...

something_else = 42.

params.a = 13.
params.b = 27.3 +/- 3.2
```

The `std_kwargs` can only contain `data, params, expected_params, debug_mode, error_mode, calc_mode, apply_mode, profile`, of which `data` and `params` will be automatically populated.


### Methods

The stages can implement three standard methods:
* `setup_function` : executed upon instantiation (and in the future potentially for more)
* `compute_function` : executed in a run if parameters changed (and first run)
* `apply_function` : executed in every run

The data representation is set by default to `calc_mode` for the first two, and `apply_mode` in the latter, but can be changed by the user freely.

An example of such an above function could look like:

```python

def apply_function(self):
  b_magnitude = self.params.b.m_as('dimensionless')
  
  for container in self.data:
    container['weights'] *= b_magnitude
```
**N.B.:** If you use in-place array operations on your containers (e.g. `container['weights'][mask] = 0.0`, you need to mark thses changes via `container.mark_changed('weights')`)

## Directory Listing

* `absorption/` - A stage for neutrino flux absorption in the Earth.
* `aeff/` - All stages relating to effective area transforms.
* `background/` - A stage for modifying some nominal (background) MC muon flux due to systematics.
* `data/` - All stages relating to the handling of data.
* `discr_sys/` - All stages relating to the handling of discrete systematics.
* `flux/` - All stages relating to the atmospheric neutrino flux.
* `likelihood/` - A stage that pre-computes some quantities needed for the "generalized likelihood"
* `osc/` - All stages relating to neutrino oscillations. 
* `pid/` - All stages relating to particle identification.
* `reco/` - All stages relating to applying reconstruction kernels.
* `utils/` - All "utility" stages (not representing physics effects).
* `xsec/` - All stages relating to cross sections.
* `GLOBALS.md` - File that describes globally available variables within PISA that needs a significant overhaul (TODO).
* `__init__.py` - File that makes the `stages` directory behave as a Python module.
