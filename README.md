<img src="images/pisa4_logo_transparent.png" width="250">

[![Unit Tests](https://img.shields.io/github/actions/workflow/status/icecube/pisa/.github/workflows/pythonpackage.yml?label=unit%20tests)](https://github.com/icecube/pisa/actions/workflows/pythonpackage.yml)
[![Pull Requests](https://img.shields.io/github/issues-pr/icecube/pisa)](https://github.com/icecube/pisa/pulls)
[![Activity](https://img.shields.io/github/commit-activity/m/icecube/pisa)](https://github.com/icecube/pisa/pulse)
[![Contributors](https://img.shields.io/github/contributors/icecube/pisa)](https://github.com/icecube/pisa/graphs/contributors)
[![Repo Stars](https://img.shields.io/github/stars/icecube/pisa?style=social)](https://github.com/icecube/pisa/stargazers)

[Introduction](pisa/README.md) |
[Installation](INSTALL.md) |
[Documentation](https://icecube.github.io/pisa/) |
[Terminology](pisa/glossary.md) |
[Conventions](pisa/general_conventions.md) |
[License](LICENSE)

PISA is a software written to analyze the results (or expected results) of an experiment based on Monte Carlo (MC) simulation.

In particular, PISA was written by and for the IceCube Collaboration for analyses employing the [IceCube Neutrino Observatory](https://icecube.wisc.edu/), including the [DeepCore](https://arxiv.org/abs/1109.6096) and the planned [Upgrade](https://arxiv.org/abs/2307.15295) low-energy in-fill arrays.

> [!NOTE]
> However, any experiment can make use of PISA for analyzing expected and actual results.

PISA was originally developed to cope with low-statistics MC by using parameterizations of the MC and operate on histograms of the data rather than directly reweighting the MC. However, PISA's methods apply equally well to high-MC situations, and PISA also performs traditional reweighted-MC analysis as well.

If you use PISA, please cite our publication ([e-Print available here](https://arxiv.org/abs/1803.05390)):
```
Computational Techniques for the Analysis of Small Signals
in High-Statistics Neutrino Oscillation Experiments
IceCube Collaboration - M.G. Aartsen et al.
Mar 14, 2018
Published in: Nucl.Instrum.Meth.A 977 (2020) 164332
```


# Quick start

## Installation

For quick-start and detailed installation instructions see [Installation](INSTALL.md).

## Minimal Example

Producing some oscillograms


```python
from pisa.core import Pipeline
import matplotlib.pyplot as plt
```

    << PISA is running in single precision (FP32) mode; numba is running on CPU (single core) >>


Instantiate a `Pipeline` or multiple pipelines in a `DistributionMaker` using PISA config files


```python
template_maker = Pipeline("settings/pipeline/osc_example.cfg")
```

Run the pipeline with nominal settings


```python
template_maker.run()
```

Get the oscillation probabilities $P_{\nu_\mu\to\nu_\beta}$


```python
outputs = template_maker.data.get_mapset('prob_mu')
```

Plot some results


```python
fig, axes = plt.subplots(figsize=(18, 5), ncols=3)
outputs['nue_cc'].plot(ax=axes[0], cmap='RdYlBu_r', vmin=0, vmax=1);
outputs['numu_cc'].plot(ax=axes[1], cmap='RdYlBu_r', vmin=0, vmax=1);
outputs['nutau_cc'].plot(ax=axes[2], cmap='RdYlBu_r', vmin=0, vmax=1);
```


![png](README_files/README_10_0.png)


# Contributions

Contributors are listed specifically [here](CONTRIBUTORS.md), while the used external software is summarized [here](EXTERNAL_ATTRIBUTION.md).
