# External software and data in PISA

PISA is distributed with some software and data obtained from outside the IceCube Collaboration.
The authors and any pertinent copyrights are listed below.
If you identify any mistakes in the below or find any other such components being distributed with PISA that are not listed here, please [email](jll1062+pisa@phys.psu.edu) or [file an issue](http://github.com/icecubeopensource/pisa).

Unless noted below or in the contents of an individual file, all files distributed with PISA are Copyright (c) 2014-2017, The IceCube Collaboration, and are licensed under the Apache 2.0 license.
See the LICENSE file for details.

## Prob3++

Files in the directory `pisa/stages/osc/prob3` were written by members of the Super-Kamiokande collaboration and are offered for public use. The original source code can be found at
> http://webhome.phy.duke.edu/~raw22/public/Prob3++

and that work is based on the paper
> V. D. Barger, K. Whisnant, S. Pakvasa, and R. J. N. Phillips, Phys. Rev. D 22, 2718 (1980)

## prob3GPU

Files in the directory `pisa/pisa/stages/osc/prob3cuda` were adapted from the CUDA re-implementation of Prob3++ called prob3GPU
> https://github.com/rcalland/probGPU

which is cited by the paper
> R. G. Calland, A. C. Kaboth, and D. Payne, Journal of Instrumentation 9, P04016 (2014).

## Honda et al. flux models

Files in the directory `example_resources/flux` containing the name *honda* are from
> http://www.icrr.u-tokyo.ac.jp/~mhonda

with associated paper
>  M. Honda, M. S. Athar, T. Kajita, K. Kasahara, and S. Midorikawa, Phys. Rev. D 92, 023004 (2015).

## Barr et al. flux models

Files in the directory `example_resources/flux` containing the name *bartol* are modified slightly (to have similar format to the work by Honda et al. cited above) from
>  http://www-pnp.physics.ox.ac.uk/~barr/fluxfiles
with associated paper
   G. D. Barr, T. K. Gaisser, P. Lipari, S. Robbins, and T. Stanev, Phys. Rev. D 70, 023006 (2004).

## PREM

The preliminary reference Earth model data in the `example_resources/osc` directory (named `PREM*`) come from the paper
> A. M. Dziewonski and D. L. Anderson, Physics of the Earth and Planetary Interiors 25, 297 (1981)

## KDE

The file `pisa/pisa/utils/vbwkde.py` contains an implementation of (part of) the paper 
> Z. I. Botev, J. F. Grotowski, and D. P. Kroese, Ann. Statist. 38, 2916 (2010).

The functions `isj_bandwidth` and `fixed_point` therein are adapted directly from the Matlab implementation by Zdravko Botev at
> https://www.mathworks.com/matlabcentral/fileexchange/14034-kernel-density-estimator

and are therefore subject to the following copyright:
```
  Copyright (c) 2007, Zdravko Botev
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
  OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Versioneer

Automatic versioning is provided by public-domain sofware The Versioneer, written by Brian Warner (files `versioneer.py` and `pisa/_version.py`).
This project can be found at
> https://github.com/warner/python-versioneer