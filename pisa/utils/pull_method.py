"""
Pull method tools.
"""


from __future__ import absolute_import, division

__all__ = []

__author__ = 'T. Ehrhardt'

__license__ = '''Copyright (c) 2014-2018, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''



def derivative_from_polycoefficients(coeff, loc):
    """
    Return derivative of a polynomial of the form

        f(x) = coeff[0] + coeff[1]*x + coeff[2]*x**2 + ...

    at x = loc
    """
    derivative = 0.
    for n in range(len(coeff))[1:]: # runs from 1 to len(coeff)
        derivative += n*coeff[n]*loc**(n-1)

    return derivative
