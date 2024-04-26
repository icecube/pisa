# Units and Uncertainties

## Units

Units were introduced for parameters and binnings in PISA Cake using the [Pint package](https://pint.readthedocs.io/). The package is fully numpy and uncertainties compatible.

Note the terminology used:
* ***Unit*** has no attached value(s). Type: `pint.unit._Unit`
* ***Quantity*** is a number (or array) *with units attached* to them. Type: `pint.quantity._Quantity`
* ***Magnitude*** is the value of a quantity sans units. Type: `float`, `int`, `numpy.ndarray`, etc.

### Pint's quirks
One quirk is that pint has lazy imports, so until you actually instantiate a unit registry, doing things like
```python
isinstance(x, pint.unit._Unit)
isinstance(x, pint.quantity._Quantity)
```
will fail (`pint.unit` and `pint.quantity` effectively don't exist) until you first do
```python
ureg = pint.UnitRegistry()
```
Therefore, and to assure that we alsways use the same instance, the "standard" way of importing pint in PISA is the following line:
```python
from pisa import ureg, Q_
```

### Basic attributes of Pint quantities

* To access the *units* of a quantity:
  ```python
  >>> q = 10.2 * ureg.meter
  >>> q.units
  <Unit('meter')>
  >>> units_2 = q.u
  <Unit('meter')>
  ```

* To access the *magnitude* of a Pint quantity:
  ```python
  >>> q = 12.4 * ureg.foot
  >>> q.magnitude
  12.4
  >>> q.m
  12.4
  ```

## Uncertainties

Support for the handling of errors is available from the python [uncertainties package](https://pythonhosted.org/uncertainties/). This interoperates well with units, too. Note that there *is* a performance penalty for using uncertainties, so hooks are provided to enable or disable this feature at the user's discretion. 

### Basic attributes of numbers with uncertainties

* Nominal value:
  ```python
  >>> x = ufloat(10.0, 0.3)
  >>> x.nominal_value
  10.0
  >>> x.n
  10.0
  ```

* Standard deviation:
  ```python
  >>> x = ufloat(-12.3, 0.6)
  >>> x.std_dev
  0.6
  >>> x.s
  0.6
  ```

*Pint* and *uncertainties* interoperate well. So a quantity can have uncertainty. All of the above attributes work for numbers that are wrapped with both features. See examples below. 

## Examples

More examples of how to use the above packages are given below.

* For ordinary numbers (floats):
  ```python
  >>> from uncertainties import ufloat
  >>> from pisa import ureg, Q_
  
  >>> q = ufloat(1.3, 0.2) * ureg.meter
  
  >>> q.m # magnitude
  1.3+/-0.2
  >>> q.u # unit
  <Unit('meter')>
  >>> q.n # nominal value
  1.3
  >>> q.s # standard deviation
  0.2
  
  >>> q.to('foot') # conversion
  <Quantity(4.3+/-0.7, 'foot')>
  >>> q.dimensionality
  <UnitsContainer({'[length]': 1.0})>
  ```

* For numpy arrays:
  ```python
  >>> from uncertainties import unumpy as unp
  >>> a = unp.uarray([1., 2., 3.], [1., 1.41, 1.73]) # with list of values, and list of errors, or alternatively:
  >>> a = np.array([ufloat(1.,1.),ufloat(2.,1.41), ufloat( 3.,1.73)])
  >>> a *= ureg.seconds
  <Quantity([1.0+/-1.0 2.0+/-1.41 3.0+/-1.73], 'second')>
  >>> a.u
  <Unit('second')>
  >>> a.m
  array([1.0+/-1.0, 2.0+/-1.41, 3.0+/-1.73], dtype=object)
  >>> unp.nominal_values((a)
  array([ 1.,  2.,  3.])
  >>> unp.std_devs(a)
  array([ 1.  ,  1.41,  1.73])
  ```

* Check if units are compatible
  * Explicitly check if `dimensionality` is equal for two units, quantities, or a combination thereof:
    ```python
    >>> units0 = ureg.meter
    >>> units1 = ureg.yard
    >>> units2 = ureg.kg
    >>> quant0 = 1 * units0
    >>> quant1 = 1 * units1
    >>> quant0.dimensionality == quant1.dimensionality
    True
    >>> quant0.dimensionality == units1.dimensionality
    True
    >>> units1.dimensionality == units2.dimensionality
    False
    ```

  * Implicitly check if one quantity is convertible to a different unit (whether a raw units object, the units attached to another quantity, or a string that Pint can convert into units):
    ```python
    >>> q0 = ureg.meter * 10
    >>> q1 = ureg.yard * 2
    >>> u0 = ureg.cm
    >>> u1 = ureg.second
    >>> # Convert q0 from meters to the units of q1 (yards)
    >>> q0.to(q1)
    <Quantity(10.9361329834, 'yard')>
    >>> q0.to(u0)
    <Quantity(1000.0, 'centimeter')>
    >>> q0.to('mile')
    <Quantity(0.00621371192237, 'mile')>
    >>> q0.to(u1)
    DimensionalityError: Cannot convert from 'meter' ([length]) to 'second' ([time])
    ```

* The [`pisa.core.Map`](https://github.com/jllanfranchi/pisa/blob/cake/pisa/core/map.py) object integrates uncertainties, which are turned off by default but can be enabled with either the `set_poisson_errors()` or `set_errors(...)` methods. Once enabled, all subsequent operations on the `Map` will propagate errors (to first order).
