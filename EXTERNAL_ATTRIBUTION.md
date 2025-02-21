# External software and data in PISA

PISA is distributed with some software and data obtained from outside the IceCube Collaboration.
The authors and any pertinent copyrights are listed below.
If you identify any mistakes in the below or find any other such components being distributed with PISA that are not listed here, please [email](analysis@icecube.wisc.edu) or [file an issue](https://github.com/icecube/pisa/issues).

Unless noted below or in the contents of an individual file, all files distributed with PISA are Copyright (c) 2014-2025, The IceCube Collaboration, and are licensed under the Apache 2.0 license.
See the LICENSE file for details.


## daemonflux

The service [`flux.daemon_flux`](/pisa/stages/flux/daemon_flux) does not reproduce but calls upon the daemonflux software
> https://github.com/mceq-project/daemonflux

based on the paper
> J. P. Yanez, A. Fedynitch, Phys.Rev. D 107, 123037 (2023)

The software is subject to the following copyright:
```
BSD 3-Clause License

Copyright (c) 2023, Anatoli Fedynitch

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## MCEq

Files in the directory [`pisa_examples/resources/flux`](/pisa_examples/resources/flux) containing the name *MCEq*
have been produced with the MCEq software
> https://github.com/mceq-project/MCEq

The authors of that software / the paper that it is based upon
> A. Fedynitch, R. Engel, T. K. Gaisser, F. Riehn, T. Stanev, EPJ Web Conf. 99 (2015) 08001, arXiv:1503.00544
 request that anyone who uses their work to produce results cite their work. Please do so if you make use of either of the
[`flux.mceq_barr`](/pisa/stages/flux/daemon_flux) or [`flux.mceq_barr_red`](/pisa/stages/flux/daemon_flux) services or of
the script [`create_barr_sys_tables_mceq.py`](/pisa/scripts/create_barr_sys_tables_mceq.py) which calls upon MCEq.

The form of the citation that they request is found in their documentation at
> http://mceq.readthedocs.io/en/latest/citations.html

The software is subject to the following copyright:
```
BSD 3-Clause License

Copyright (c) 2019, Anatoli Fedynitch
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Honda et al. flux models

Files in the directory [`pisa_examples/resources/flux`](/pisa_examples/resources/flux) containing the name *honda* are from
> http://www.icrr.u-tokyo.ac.jp/~mhonda

with associated paper
>  M. Honda, M. S. Athar, T. Kajita, K. Kasahara, and S. Midorikawa, Phys. Rev. D 92, 023004 (2015).

## Barr et al. flux models

Files in the directory [`pisa_examples/resources/flux`](/pisa_examples/resources/flux) containing the name *bartol* are
modified slightly (to have similar format to the work by Honda et al. cited above) from
>  http://www-pnp.physics.ox.ac.uk/~barr/fluxfiles

with associated paper
> G. D. Barr, T. K. Gaisser, P. Lipari, S. Robbins, and T. Stanev, Phys. Rev. D 70, 023006 (2004).

## nuSQuIDS

The service [`osc.nusquids`](/pisa/stages/osc/nusquids.py) does not reproduce but calls upon the nuSQuIDS software
> https://github.com/arguelles/nuSQuIDS
which is subject to the LGPL-3.0 license.

## prob3numba

Files in [`pisa/stages/osc/prob3numba`](/pisa/stages/osc/prob3numba) were adapted from the CUDA re-implementation of Prob3++ called prob3GPU
> https://github.com/rcalland/probGPU

which is cited by the paper
> R. G. Calland, A. C. Kaboth, and D. Payne, Journal of Instrumentation 9, P04016 (2014).

## PREM

The preliminary reference Earth model data in the [`pisa_examples/resources/osc`](/pisa_examples/resources/osc) (named `PREM*`) come from the paper
> A. M. Dziewonski and D. L. Anderson, Physics of the Earth and Planetary Interiors 25, 297 (1981)

## emcee

The files [`llh_client.py`](/pisa/utils/llh_client.py) and [`analysis.py`](/pisa/analysis/analysis.py) call upon the emcee software
> https://github.com/dfm/emcee

which is subject to the following copyright:
```
The MIT License (MIT)

Copyright (c) 2010-2021 Daniel Foreman-Mackey & contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## KDE

The file [`vbwkde.py`](/pisa/pisa/utils/vbwkde.py) contains an implementation of (part of) the paper
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

Automatic versioning is provided by public-domain sofware The Versioneer, written by Brian Warner
(files `versioneer.py` and `pisa/_version.py`). This project can be found at
> https://github.com/python-versioneer/python-versioneer

