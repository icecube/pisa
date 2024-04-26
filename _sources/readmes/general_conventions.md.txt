# Coding conventions
* General Python conventions should be followed
  * [PEP8](https://www.python.org/dev/peps/pep-0008/)
  * Run [pylint](https://www.pylint.org/) to see how egregious your code is (and make it less so) before submitting a pull request (no hard requirement on making it pass 100%, but at least you'll learn how to make it cleaner in the future). Also, static analysis of code (i.e., running pylint) can find bugs you miss in other ways. (Note that there are some nice GUI interfaces for pylint, including e.g. [Spyder IDE](https://github.com/spyder-ide/spyder).)
  * Call functions and methods using full keyword names. E.g.:<br>`my_function(x=1, y=2, z='blahblah')`
  * Avoid positional arguments wherever possible and reasonable (both for Python methods/functions and for command-line interfaces)
  * Always give the user double-dash (`--three-word-arg`) command-line keyword options. Separate words in the option with single-dashes, *not* underscores. Optionally provide single-dash/single-letter options (`-a`) in addition to the double-sash keyword options.
  * To use matplotlib on a headless server, you can specify at the command line the environment variable `MPLBACKEND=agg` or `=pdf`, etc.

# Naming conventions
* Stage/service naming: see [creating a service](creating_a_service). (Note that the all-lower-case class naming scheme for services is one of the few exceptions we make to the general Python conventions above.)
* Use the following syntax to refer to neutrino signatures `*flavour*_*interaction*_*pid*` where:
  * `flavour` is one of `nue, nuebar, numu, numubar, nutau, nutaubar`
  * `interaction` is one of `cc, nc`
  * `pid` is one of `trck, cscd`

  Omit suffixed fields as necessary to refer to a more general signature for the same flavour, e.g. `nue` refers to all interactions/pids involving electron neutrinos. To refer to multiple signatures separate each signature with a `+` symbol, e.g. `nue_cc+nuebar_cc`

# Documentation conventions

Documentation comes in two forms: ***docstrings*** and ***standalone files*** (either markdown **.md** files or reStricturedText **.rst** files). Docstrings are the most important for physics and framework developers to consider, and can (and should) be quite far-ranging in their scope. Standalone files are reserved for guides (install guide, developer's guide, user's guide, quick-start, etc.) and one README.md file within each directory to document the directory's raison d'être, or what is common to all of the things contained in the directory.

* All documentation is run through [Sphinx](http://www.sphinx-doc.org) using the [Napoleon](http://sphinxcontrib-napoleon.readthedocs.io) (to understand Numpy-style docstrings) and [Recommonmark](http://recommonmark.readthedocs.io) (to understand Markdown syntax) extensions, so the final arbiter of whether a docstring is formatted "correctly" is the output generated using these.
Refer to those links for how to format docstrings / Markdown to produce a desired result.

## Docstrings

Docstrings can document nearly everything within a Python file. The various types of docstrings are:
* **module**: beneath the authorship, at the top of the .py file
  * Why did you create this file (module)? What purpose does it serve? Are there issues that the user should be aware of? Good and bad situations for using the class(es) and function(s) contained? Anything that doesn't fit into the more usage-oriented docstrings below should go here.
* **function**: beneath the function declaration line
  * What the function does, args to the function, notes
* **class**: beneath the class declaration line
  * Purpose of the class, instantiation args for the class, notes to go into more detail, references, ...
* **method**: beneath the method declaration line
  * Purpose of the method, args to the method, notes
* **attribute**: beneath the definition of a variable (in any scope in the file)
  * What purpose it serves. Note that if you add an attribute docstring, the attribute is included in the documentation, while if you do not add such a docstring, the attribute does not appear in the documentation.

Examples of the above can be seen and are discussed further in the [tutorial for creating a service](creating_a_service).

Docstrings should be formatted according to the NumPy/SciPy convention.
* [PEP257: General Python docstring conventions](https://www.python.org/dev/peps/pep-0257/)
* [NumPy/SciPy documentation style guide](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)
* [Example NumPy docstrings in code](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)
* [Recommonmark translates markdown into reST](http://recommonmark.readthedocs.io/en/latest/auto_structify.html)
* Since all documentation will be run through Sphinx using the Napoleon and Recommonmark extensions, the final arbiter on whether a docstring is formatted correctly is the output generated using these. Good information for making nice docstrings can be found in both [Napoleon](http://sphinxcontrib-napoleon.readthedocs.io/)

## Standalone files

### Guides

These are high-level documents addressing one (or multiple) of the three audiences: users, physics developers, and framework developers.

* Install guide
* Quick-start guide
* User guide
* Physics developer guide
* Framework developer guide

### README.md files

There is (or should be) one README.md file per directory in PISA. This should state the raison d'être of the directory, or what is common to all of the "things" contained in the directory. Those that live in stage directories (e.g. `$PISA/pisa/stages/pid/README.md`) are more critical than the others, so are discussed further below.

#### Stage README.md files

* Try to avoid saying anything here that's already (or should be) said in the docstrings within each individual service. Anything like this is only going to get out of sync with the actual implementations, since it's repeating what's already in the docstrings for the individual services. This violates probably the most important [principle of software development](https://en.wikipedia.org/wiki/Don't_repeat_yourself).

* So just make that as high-level as possible. For example, for `$PISA/pisa/stages/pid/README.md`, guidance would be:
  * What is particle ID? I.e., what is the physics it represents, what is the process we generally use to do PID, and what systematics are pertinent to PID?
  * What general categories to the contained service(s) fall into?
  * What are the major difference between services that would lead a user to pick one vs. another?
  * A table of what systematics are implemented by each service would be useful at this level

# Testing
If you add a feature: ***add a test that proves it works***. If you find a bug: ***add a test that it's fixed***. Not only does a test ensure that the functionality continues to work / bug continues to be squashed, through future iterations of the code, but it also is another way of communicating the assumptions that went into the code that might not be obvious from the code itself.

## Low-level testing: Unit tests
Write simple functions named `test_<FUNCNAME>` or `test_<CLASSNAME>` within your modules. These get called automatically from [pytest](http://pytest.org/). Pytest picks up exceptions that occur, so sprinkle `assert` statements liberally in your `test_*` functions, and you should test both things you expect to succeed *and* test things that you expect to fail (wrap the latter in `try:/except:/else:` blocks, where the `assert` occurs in the `else:` clause).

We chose [pytest](http://pytest.org/): We'd rather have more unit tests written because they're dead-easy to write than to use the full-featured (and built-in) framework that comes with Python.

Expect pytest to eventually be configured to run with [doctest](https://pytest.org/latest/doctest.html) functionality enabled, so it's also acceptable to put simple tests in docstrings, Markdown documentation files, or reST documentation files. (Just these won't be exercised until we get pytest fully configured to do look for these tests.)

Finally, until we get pytest configured, the `test_*` functions are called in the `__main__` section of many of the `pisa/core` modules (e.g., `map.py`, `binning.py`, `transform.py`, `param.py`). For now, these must be invoked by calling `python <filename>`.

### How to test a service
Look at the `pisa.core.pipeline.stage_test()` function to implement tests for your service. It can run a stage in isolation (either with maps you supply or with dummy `numpy.ones()` maps) or will run a pipeline from beginning up until the stage you're testing. To keep tests synchronized with the code they are testing, it is recommended that configurations be generated within the test code rather than read from an external file. This is not a hard-and-fast rule, but usually results in better long-term outcomes.

## High-level testing
Here we will supply whatever basic configuration files and example data is necessary to fully test all high-level functionality. Outputs should be able to be compared against known-outputs, either exactly (for deterministic processes or pseudo-random processes) or as a ratio with a know result (in the case of a "new" physics implementation).

There should be both quick tests that just show an analysis can make it through from beginning to end (and produce a known output), as well as more thorough tests that check the correctness of a result.

For comparing non-deterministic results, plotting is essential for understanding differences. Difference plots, ratio plots, fractional-difference plots, side-by-side plots, overlays, correlations, ... 

# Physical Quantities

## Units

All physical quantities with units should have units attached. Parameters with units specified also require that their prior specifications adhere to those same units.

This is essential so that, transparently, the user can input units are most comprehensible while computations can be carried out in whatever units have been defined for computation. Furthermore, it is made explicit in the user's configuration files what units were chosen for input and the code for performing computation makes it explicit what units it uses as well.

## Uncertainties

Likewise, measured quantities with uncertainties should have these specified as well. Since there is a performance penalty for computing with uncertainties, this is a feature that can be enabled or disabled according to the task the user is working on. Certain parts of PISA (e.g., the convolutional likelihood metric) will only work if errors are propagated through the analysis.

## Further documentation
For more information on using units and uncertainties, see the [Units and uncertainties](units_and_uncertainties) page.
