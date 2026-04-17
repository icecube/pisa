.. PISA documentation master file, original template created by
   sphinx-quickstart on Sun Mar 29 22:46:48 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. TODO: certain guides currently do not lend themselves to inclusion in these docs, such as the main readme or conventions markdown guide
   -> need to make these docs the authoritative documentation source, and accordingly adapt the above documents (also seems to be in accordance
   with PISA design choices laid out in deactivated wiki)

Welcome to PISA's documentation!
================================

PISA is a software written in Python designed to analyze the results (or expected results) of an experiment based on Monte Carlo simulation.
In particular, PISA was written by and for the IceCube Collaboration for analyses employing the `IceCube Neutrino Observatory <https://icecube.wisc.edu>`__,
including the `DeepCore <https://arxiv.org/abs/1109.6096>`__ and the `Upgrade <https://arxiv.org/abs/2509.13066>`__ low-energy in-fill arrays.
**Any experiment can make use of PISA for analyzing expected and actual results though.**

If you use PISA, **please cite our publication** (`e-Print available here <https://arxiv.org/abs/1803.05390>`__):

.. admonition:: **Computational Techniques for the Analysis of Small Signals in High-Statistics Neutrino Oscillation Experiments**

   IceCube Collaboration - M.G. Aartsen et al.

   Mar 14, 2018

   Published in: Nucl.Instrum.Meth.A 977 (2020) 164332


.. toctree::
   :caption: Navigation
   :hidden:

   Installation <stubs/install_stub>
   Tutorials <tutorials>
   Contributing <contributing>

.. toctree::
   :caption: Reference
   :hidden:

   Python API <pisa>
   Changelog <changelog>
   License <stubs/license_stub>
