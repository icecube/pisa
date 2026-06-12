# Creating a service

## Background info and naming conventions
A ***service*** is an implementation of a ***stage***. As far as directory structure goes:
```
$PISA/pisa/stages/<stage name>/<service name>.py
```
The stage names should be short and lower-case. The directories in here represent the existing stages.

Services should also have a lower-case file name (use underscores if multiple words), and the class inside the file should have the same name (including being all lower-case).
Note that class names being all-lower-case runs counter to the Python conventions we use elsewhere, but this rare exception to Python conventions was determined to be a necessary compromise for technical reasons.
Since the directory structure unambiguously identifies what stage a service implements, the stage name should not be included in the service name.

For example, the services
```bash
$PISA/pisa/stages/flux/mceq_barr.py
$PISA/pisa/stages/osc/prob3.py 
```
can be simultaneously (and unambiguously) invoked in the same Python script via
```python
from pisa.stages.flux import mceq_barr as flux_mceq_barr
from pisa.stages.osc import prob3 as osc_prob3
```

## Anatomy of a typical stage

### Module docstring

Just beneath an authorship comment at the top of the file, you should place the module docstring:
```python
"""
This is a dummy oscillations service, provided as a template others can use to
build their own services.
...
"""
```
Include a module docstring for your service, too, if there's general information (particularly about how this service addresses the physics/functionality expected for this stage of the analysis) that is higher-level than the
"what is this and how do you use it" information that you should put in the class and method docstrings. Here you can wax poetic about your reasoning behind the service, or some lessons for others implementing something similar,
to-do's, etc., etc. Basically, anything that doesn't really fit in one of the other docstring types can be put here.

### Imports

Beneath the module docstring, you place the import statements. First is a section for Python built-ins (none are imported in this example, so there are none), then a section for external modules, and finally a section for PISA modules.
Note that each "section" is separated by a blank line and the ordering of the imports in each is alphabetical.
```python
import numpy as np

from pisa import FTYPE, ureg
from pisa.core.stage import Stage
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resource import find_resource
```

### Class definition

The `class` statement appears next, after two blank lines:
```python
class dummy(Stage):
```
declares `dummy` as a Python class that inherits from the `Stage` class.
***Every service should inherit from `Stage`***. The name you give the class must be ***lower-case*** and must ***match the filename exactly*** (excluding the .py file extension).
This convention is used in the pipeline config parser for automatically importing and instantiating services, so it must be followed.

### Class docstring

The next docstring is for the class, and it follows the NumPy convention for formatting (which is essential for it to be used with the automatic documentation tools).
Document the parameters (aka keyword arguments) specified by the `__init__` method in a section titled `Parameters`.
Include a `Notes` section to discuss things like details of implementation that the user should be aware of that are too lengthy to put in the brief descriptions in the `Parameters` section.
Each section has dashes underneath the section name (same length as the name), and is separated by the above section by one or two empty lines.
Refer to the [general conventions](https://github.com/icecube/pisa/blob/master/pisa/general_conventions.md) for references where you can find out more about writing docstrings.

```python
    """Docstring (mandatory!)

    What is the purpose of this stage? Short 1-2 sentence description


    Parameters
    ----------
    params : ParamSet
      Must have parameters::

        a : dimensionless Quantity
        b : dimensionless Quantity

    something_else : float
        Description

    some_arg_with_default : bool (default: True)
        Description


    Notes
    -----
    If you want to use both foo and bar, blah blah.
    Does not implement setup/compute/apply because ...
    Only allowed apply_mode is ...
    More info, etc...

    Expected container keys are::

        'true_energy', 'true_coszen', 'weights'


    References
    ----------
    Einstein, Albert. "The photoelectric effect." Ann. Phys 17.132 (1905): 4.
    """
```

Note that the "Parameters" section has parameter names optionally followed by ` : ` and then a list of types that the parameter can be.
This turns out to be quite useful for both developers and users, so include what type(s) your code expects here!
The further-indented line(s) beneath the `name : type` specification then describe the parameters (arguments) for instantiating the class.
This same convention is followed for method docstrings.
How to determine which parameters belong to the `ParamSet` (`params`) and which ones serve as class instantiation arguments is outlined below.

### The `__init__` method definition
The definition of the init method is how the class must be called to be instantiated.
```python
    def __init__(self, something_else, some_arg_with_default=True, **std_kwargs):

        expected_params = ('a', 'b')
        expected_container_keys = ('true_energy', 'true_coszen', 'weights')
        supported_reps = {
            'calc_mode': MultiDimBinning,
            'apply_mode': 'events'
        }

        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            supported_reps=supported_reps,
            **std_kwargs
        )

        self.foo = something_else
        """Description"""
        self.bar = some_arg_with_default
        """Description"""
```
The `std_kwargs` can only contain `data`, `params`, `expected_params`, `expected_container_keys`, `debug_mode`, `error_method`, `supported_reps`, `calc_mode`, `apply_mode`, `profile`, and `in_standalone_mode`. Of these, `data` and `params` will be automatically populated when a `Pipeline` containing the service is instantiated.

In turn, a minimal example for the code instantiating the service will look something like
```python
from pisa.stages.osc.dummy import dummy as osc_dummy
instantiated_dummy = osc_dummy(params=my_paramset, something_else=0.42)
```
(of course there's code that would have to be written to instantiate `my_paramset`).
In fact, we expect you to instantiate the service with some reasonable assumptions about params it expects in a test function at the bottom of the file (more on that below).

Note a few important things about how this service is being instantiated above:

1. All arguments are passed by keyword. Python allows you to just send arguments in order (positional args), but ***don't do this***. It's far better to be explicit than to save a few keystrokes. With a "duck-typed" language like Python, using the names explicitly provides a useful check that what the user sends to a function/method is what the user intends, and that the function/method receives what it expects to get.
1. Arguments like `some_arg_with_default=True` have default values set in the definition of init. These needn't be specified by the user (since they have default values set), but can be, for the user to override the defaults. The arguments without defaults (such as `something_else`) *must* be specified by the user.
1. Since the superclass `Stage` exclusively has keyword arguments, with default values of `None` or `False`, and since the service is not part of a pipeline, `instantiated_dummy.data` returns `None`. (We could in principle choose to pass `data` as an instantiation argument to `osc_dummy`, but it isn't required at `__init__` time).

The init method body is broken down into four parts in the example to hopefully make it easier to see what you have to do to implement your own. For example, you don't *need* to create the temporary variables of steps 1, 2, 3, but this format makes it very clear what's going on and makes the code much cleaner than trying to stuff those objects into step 4 where they are passed to the superclass's init method.
1. Define a temporary variable containing `expected_params`, the parameter names your service requires to be passed via the `params` argument.
1. Define a temporary variable `expected_container_keys`, the container keys expected to be present in each container in the `data` object assigned to this service.
1. If desired, define a temporary variable `supported_reps`, containing the data representations (`calc_mode` and `apply_mode`) you want to allow. If you don't, PISA will populate `supported_reps` with sensible defaults.
1. Call the `__init__` method of the superclass (`Stage`). Don't forget to pass the above arguments. For details of all the arguments it accepts, see the `Stage`'s init method definition (and corresponding class docstring) in [stage.py](https://github.com/icecube/pisa/blob/master/pisa/core/stage.py).
1. Do any custom things that your class requires for setting itself up.

#### Defining service parameters: instantiation argument or `Param` object?
Instantiation arguments (`something_else` and `some_arg_with_default` in the dummy service above) configure the service and will *never* be passed to a fitter (minimizer, sampling algorithm, etc.). Within the corresponding section of a pipeline config file, they are specified simply by `name = value` or `name = value * units.xyz`.

In contrast, `Param` objects *are modified* by a fitter. They correspond to the entries of `expected_params` (`a` and `b` in the dummy service above). They are characterized by (at least) a name, a value with units, a finite range, a flag indicating whether they are to be kept fixed or not, and a prior. In the pipeline config file, these properties are specified with a `param.name` prefix.


### Implementing the physics: three `{...}_function`s

There are three abstract methods provided by the `Stage` class where the actual work of computing things should be implemented:
* `setup_function` : executed upon instantiation
* `compute_function` : executed in the first run and then only *whenever parameters change* (**caching**)
* `apply_function` : executed in every run

The `Stage` class implements some default logic before or after the execution of each of these methods, but for any service at least one of the above will need to be implemented (i.e. implemented in the service you're writing) in order to do anything non-trivial.
More specifically, the `Stage` class by default sets the data representation to `calc_mode` before calling each of the first two, and to `apply_mode` before the third.

Considering which of the abstract methods above should perform a given task for your service to output correct physics results in an efficient manner whenever parameter values change is of paramount importance.
In the following, "transform" shall refer to the result of any computation (e.g., multidimensional function based on external simulation, based on a parameterization, etc.) with which the contents of the `data` object passed to the service are modified (create new arrays representing some physical quantities, manipulate existing ones, etc.).

#### Services that use nominal transforms

🠚 Implement **`setup_function`** and **`apply_function`** in your service's class

Nominal transforms are generated e.g. when all params (if any) are set to their nominal values.
It is useful to generate nominal transforms if it is slow to compute them or read them from a file, and the params modify these in some comparably cheap way to arrive at the final transform.

#### Services that use transforms

🠚 Implement **`compute_function`** and **`apply_function`** in your service's class

Even when there are no nominal transforms, it might be efficient to transfer any universal parts of the calculation to the `setup_function` and to store the results in `data` for use by the other two methods.

#### Services that do not use transforms

🠚 Implement **`apply_function`** in your service's class

Consider an example in which the params of the service are directly applied to some contents of `data`, for instance as scaling factors.

#### `apply_function` example

An example of an `apply_function`&mdash;which performs some non-sensical scaling of 'weights'&mdash;could look like this:
```python
    def apply_function(self):
        a_magnitude = self.params.a.m_as('dimensionless')
        b_magnitude = self.params.b.m_as('dimensionless')
        for container in self.data:
            container['weights'] *= a_magnitude * b_magnitude * container['true_energy'] * container['true_coszen']
            container.mark_changed('weights')
```

```{admonition} Marking changes
:class: important

If you use operations on your containers that do not result in execution of the `Container.__setitem__` dunder method (e.g. `container['weights'][mask] = 0.0`), you need to *mark these changes* via `container.mark_changed('weights')` *manually*. This will ensure that the variable 'weights' is invalidated in all other representations than the current one (namely that given by `apply_mode` in the case of the `apply_function` example). If 'weights' are ever requested in another representation, a transformation from a valid source representation to the target representation will first be performed. If in doubt, there is no harm in performing the manual call as in the example above. Alternatively, look for messages about marked changes in trace-level log outputs.
```

## Testing

At the bottom of the file, add a function called `init_test` which successfully creates and returns an instance of the service (consider this an important part of its documentation).
This will be detected and called by the service-testing script that attempts to run every service existing within PISA.
In case the service instantiation requires reading from some input file: if the file is generally useful, doesn't contain non-public information, and is not too large, consider adding it to PISA; otherwise, consider creating a correctly formatted dummy file on the fly within the function below with which the service is instantiable.

The instantiation test function for our simple dummy service could look like

```python
def init_test(**param_kwargs):
    """Initialisation example"""
    param_set = ParamSet([
        name='a', value=0.0, **param_kwargs),
        name='b', value=1.0, **param_kwargs),
    ])
    return dummy(params=param_set, something_else=0.42)
```
