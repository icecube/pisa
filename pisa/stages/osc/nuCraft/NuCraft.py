#!/usr/bin/env python
"""
Main and only module of the nuCraft package that computes neutrino oscillation
probabilities for atmospheric neutrinos by directly solving the Schroedinger
equation.

It contains the main class NuCraft with its main method CalcWeights, as well as
the auxiliary class EarthModel.

For default usage, only CalcWeights and the NuCraft constructor are needed:

nC = NuCraft([1., 7.50e-5, 7.50e-5+2.32e-3], [(1,2,33.89),(1,3,9.12),(2,3,45.00)])
nC.CalcWeights([(type1, energy1, zenith1),(type2, energy2, zenith2),...])

For more information please see the docstrings of the respective classes and
methods, or check out the example script.

################################################################################

Copyright (c) 2013, Marius Wallraff (mwallraff#physik.rwth-aachen.de)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the RWTH Aachen University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL MARIUS WALLRAFF BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

################################################################################

NuCraft heavily relies on NumPy and SciPy. If you want to use nuCraft, you might
want to cite scipy (see www.scipy.org/Citing_SciPy) and the publication related
to the ODE solver ZVODE, http://dx.doi.org/10.1137/0910062.

NuCraft is shipped with a table of data points sampled from the Preliminary
Earth Reference Model PREM, see http://dx.doi.org/10.1016/0031-9201(81)90046-7.

Also, please cite our publication; a link and bibTeX entry will be added to this
file as soon as it is available.
"""

from __future__ import print_function, division

from math import sqrt as ssqrt
from math import cos as scos
from math import fabs as fabs     # more restrictive than abs() (in a good way)
from numpy import *
from scipy import interpolate, integrate
from scipy.stats import lognorm
from ast import literal_eval
import warnings
def CustomWarning(message, category, filename, lineno):
   print("%s:%s: %s:%s" % (filename, lineno, category.__name__, message))
warnings.showwarning = CustomWarning
from os import path

set_printoptions(precision=5, linewidth=150)





class EarthModel:
   """
   Auxiliary class with informations regarding Earth matter properties
   at a given distance from the center of the Earth, for use with nuCraft.
   
   The class can be constructed from an entry of the included dictionary of models,
   in which case the parameter 'name' should just be the name of the model, i.e.,
   the key value of the dictionary entry; or it can be constructed from a file, in
   which case 'name' should be the path to the profile file. An example profile
   file is provided as EarthModelPrem.txt. Names in the dictionary have priority.
   
   In both cases, values for the (relative) electron numbers y and for the radii
   of mantle (including crust), outer and inner core that are given by the model
   can be overwritten with keyword arguments:
         y: tuple of electron numbers, (y_mantle, y_oCore, y_iCore), 0 <= y <= 1
        rE: radius of the Earth in km
    rOCore: radius of the outer core in km
    rICore: radius of the inner core in km,   0 <= rICore <= rOCore <= rE
   """
   # models is a dictionary:  name:(r values, rho values, y, rE, rOCore, rICore),
   # where r are radii in km, rho the corresponding matter density values in kg/cm^3,
   # y the tuple of electron numbers for mantle, outer and inner core, and the last
   # three entries the corresponding radii (in km)
   models = {"prem":
             ([0., 200., 400., 600., 800., 1000., 1200., 1221.5, 1221.5005, 1400.,
               1600., 1800., 2000., 2200., 2400., 2600., 2800., 3000., 3200., 3400.,
               3480., 3480.0005, 3600., 3630., 3630.0005, 3800., 4000., 4200., 4400.,
               4600., 4800., 5000., 5200., 5400., 5600., 5600.0005, 5701., 5701.0005,
               5771., 5771.0005, 5871., 5971., 5971.0005, 6061., 6151., 6151.0005,
               6221., 6291., 6291.0005, 6346.6, 6346.6005, 6356., 6356.0005, 6368.,
               6368.0005, 6371., 6371.0005, 8000.],
              [13.08848, 13.07977, 13.05364, 13.01009, 12.94912, 12.87073, 12.77493,
               12.76360, 12.16634, 12.06924, 11.94682, 11.80900, 11.65478, 11.48311,
               11.29298, 11.08335, 10.85321, 10.60152, 10.32726, 10.02940,  9.90349,
                5.56645,  5.50642,  5.49145,  5.49145,  5.40681,  5.30724,  5.20713,
                5.10590,  5.00299,  4.89783,  4.78983,  4.67844,  4.56307,  4.44317,
                4.44317,  4.38071,  3.99214,  3.97584,  3.97584,  3.84980,  3.72378,
                3.54325,  3.48951,  3.43578,  3.35950,  3.36710,  3.37471,  3.37471,
                3.38076,  2.90000,  2.90000,  2.60000,  2.60000,  1.02000,  1.02000,
                0.,       0.],
              (0.4957, 0.4656, 0.4656),
              6371.,
              3480.,
              1121.5)}
   
   def __init__(self, name, y=None, rE=None, rOCore=None, rICore=None):
      self.name = name
      
      if name in self.models:
         # see doc for models above
         model = self.models[name]
         self.y = model[2]
         self.rE = model[3]
         self.rOCore = model[4]
         self.rICore = model[5]
         # x (radius r) and y (matter density rho) values of the Earth density profile
         profX = model[0]
         profY = model[1]
      elif path.isfile(name):
         profFile = file(name, 'r')
         profLines = profFile.readlines()
         profFile.close()
         # previous verions of nuCraft used the eval function which can be unsafe;
         # there are no known security risks in ast.literal_eval, but the check for underscores
         # in the file has been kept in place to warn the user of files that were possibly intended
         # to exploit security holes of the eval function
         for l in profLines:
            if "_" in l:
               raise Exception("Underscore detected in EarthModel file '%s'; possibly malicious code, see README!" % name)
         self.y = literal_eval(profLines[1])
         self.rE = literal_eval(profLines[2])
         self.rOCore = literal_eval(profLines[3])
         self.rICore = literal_eval(profLines[4])
         profX = [float(prof.split()[0]) for prof in profLines[6:-1]]
         profY = [float(prof.split()[1]) for prof in profLines[6:-1]]
      else:
         raise NotImplementedError("The Earth model name '%s' can not be found!" % name)
      
      assert len(profX) == len(profY), "Density profile must have exactly one density per radius."
      self.profInt = interpolate.InterpolatedUnivariateSpline(profX, profY, k=1)
      
      # self.mod indicates whether this is the profile indicated by its name, or if it has been modified
      self.mod = False
      
      if not y == None:
         if not self.y == y:
            self.mod = True
         self.y = y
      assert len(self.y) == 3 and logical_and(0 <= array(self.y), array(self.y) <= 1).all(), \
             "Electron number tumple must be (y_mantle, y_oCore, y_iCore), with 0 <= y <= 1."
      
      if not rE == None:
         if not self.rE == rE:
            self.mod = True
         self.rE = rE
      if not rOCore == None:
         if not self.rOCore == rOCore:
            self.mod = True
         self.rOCore = rOCore
      if not rICore == None:
         if not self.rICore == rICore:
            self.mod = True
         self.rICore = rICore
      assert 0. <= self.rICore <= self.rOCore <= self.rE, \
             "Radii of Earth's inner and outer core and surface must be positive and monotonically increasing."
      
      # compute induced mass potentials for three flavors of neutrinos; see SetDim() below
      self.SetDim(3)
   
   
   def SetDim(self, dim):
      """
      Set the number of neutrino flavors (self._dim) this instance should take into account.
      
      The dimension of A depends on this quantity. The function also computes the
      dim-dimensional A arrays.
      
      Calling this manually should not be needed because it is called by nuCraft automatically.
      """
      assert isinstance(dim, int) and dim >= 3, "Dimension must be an integer larger than two."
      self._dim = dim
      
      # A and AxCore are constants of the squared mass potentials induced by matter
      # effects, which still need to be multiplied by the neutrino energy, mass
      # density, and by -1 if the particle is an anti-neutrino;
      # the two versions take into account that the elementary composition of the
      # inner and outer Earth cores is very different from that in the mantle, which
      # is important because the relevant quantities are electron and neutron number
      # densities instead of the mass densities given by the PREM profile
      #
      # A = sqrt(2) * G_F * rho / m_N * (2*Y_e,0,0,1-Y_e,...) * E_nu
      #   = sqrt(2)*1.16637e-5 / 0.939 * (2*Y_e,0,0,1-Y_e,...)
      #      * (1/1.783e-27)*(1.973e-15)**3 * rho/(kg/dm^3) * E/GeV  *  1e18 * eV^2
      self.A =      array([15.256e-5*self.y[0], 0., 0.]+[7.6525e-5*(1-self.y[0])]*(dim-3))
      self.AOCore = array([15.256e-5*self.y[1], 0., 0.]+[7.6525e-5*(1-self.y[1])]*(dim-3))
      self.AICore = array([15.256e-5*self.y[2], 0., 0.]+[7.6525e-5*(1-self.y[2])]*(dim-3))
   
   
   def __str__(self):
      if self.mod:
         return "EarthModel('%s', modified)" % self.name
      else:
         return "EarthModel('%s')" % self.name
   
   
   def __repr__(self):
      # assumes that EarthModel was imported into the main namespace (i.e., from NuCraft...),
      # like numpy repr do; do   nc = eval("NuCraft."+repr(nc))   otherwise
      return "EarthModel('%s', y=(%.17f, %.17f, %.17f), rE=%.17f, rOCore=%.17f, rICore=%.17f)" \
             % (self.name, self.y[0], self.y[1], self.y[2], self.rE, self.rOCore, self.rICore)





class NuCraft:
   """
   Main class that calculates atmospheric neutrino oscillation probabilities
   by directly solving the Schroedinger equation. It includes matter effects (using the
   PREM Earth density profile, modified for neutron abundance in the core), and supports
   an arbitrary number of neutrino "flavors". The first flavor is interpreted as electron
   neutrinos, the next two as non-electron non-sterile neutrinos, and all other flavors
   as sterile neutrinos; to modify this, edit self.A and self.ACore.
   
   An instance of the class is created by calling NuCraft(deltaMi1List, angleList),
   where deltaMi1List is a list like described in ConstructMassMatrix, and angleList like
   described in ConstructMixingMatrix. The constructor also accepts keyword arguments that
   can be used to change default values:
       earthModel: an instance of EarthModel class for the computation of matter effects
    detectorDepth: depth of the detector relative to the surface of the Earth in km (>= 0)
        atmHeight: height of the atmosphere relative to the surface of the Earth in km,
                   only used for atmMode 0 (see CalcWeights documentation)
   The last two can also be changed after initialization by modifying self.detectorDepth
   (default 2. km) and self.atmHeight (default 20. km).
   
   The recommended method to calculate the oscillation probabilites is CalcWeights;
   it solves the Schroedinger equation in the interaction basis (i.e., where vacuum
   oscillations are flat), and it supports proper handling of the Earth's atmosphere,
   which is enabled by default (atmMode 3; see docstring).
   
   The other method CalcWeightsLegacy solves the Schroedinger equation in the flavor basis;
   it is mostly kept for comparability, as it is much slower and less precise for
   most problems. It can be suitable for high-energy sterile neutrino problems, where
   vacuum oscillations are reeaaally slow. Its handling of the Earth's atmosphere is
   very basic (atmMode == 0 of the other method).
   
   The code should NEVER throw warnings like "excess work done"; if it does, adjust the
   ODE solver parameters as specified in the warning, or contact the author.
   Seldom warnings saying "the computed unitarity does not meet the specified precision"
   are unproblematic if the inequalities thereafter are not off by much (e.g., 50%).
   """
   
   def __init__(self, deltaMi1List, angleList, earthModel=EarthModel("prem"),
                detectorDepth=2., atmHeight=20.):
      if len(deltaMi1List) == 2 and len(angleList) == 1 and angleList[0][1] == 2:
         deltaMi1List = list(deltaMi1List)
         deltaMi1List.append(0.)
         angleList = list(angleList)
         angleList.append((1,3,0.))
      # mainly used for __repr__, do not modify in instances ("private")
      self._deltaMi1List = deltaMi1List
      self._angleList = angleList
      self.M = self.ConstructMassMatrix(deltaMi1List)
      self.U = self.ConstructMixingMatrix(angleList)
      
      dim = len(self.M)
      assert self.M.shape == self.U.shape == eye(dim).shape, "Shape mismatch between M and U."
      
      if isinstance(earthModel, EarthModel):
         earthModel.SetDim(dim)
         self.earthModel = earthModel
      else:
         raise ValueError("The provided earth model '%s' is not of the EarthModel class." % earthModel)
      
      # tuple entry 0 is 1 for particles and -1 for antiparticles; it became redundant because of the
      # transition from Geant-style particle IDs to PDG-style particle IDs, where IDs < 0 signify
      # antiparticles; kept it because of readability
      self.mcTypeDict = {}
      self.mcTypeDict[ 12] = ( 1, array([1.]        +[0.j]*(dim-1)))   # NuE
      self.mcTypeDict[-12] = (-1, array([1.]        +[0.j]*(dim-1)))   # NuEBar
      self.mcTypeDict[ 14] = ( 1, array([0.j,1.]    +[0.j]*(dim-2)))   # NuMu
      self.mcTypeDict[-14] = (-1, array([0.j,1.]    +[0.j]*(dim-2)))   # NuMuBar
      self.mcTypeDict[ 16] = ( 1, array([0.j,0.j,1.]+[0.j]*(dim-3)))   # NuTau
      self.mcTypeDict[-16] = (-1, array([0.j,0.j,1.]+[0.j]*(dim-3)))   # NuTauBar
      for i in range(1,dim-2):
         self.mcTypeDict[ 80+i] = ( 1, array([0.j]*(2+i)+[1.]+[0.j]*(dim-3-i)))   # NuSterile_i
         self.mcTypeDict[-80-i] = (-1, array([0.j]*(2+i)+[1.]+[0.j]*(dim-3-i)))   # NuSterileBar_i
      
      # depth of the (center of the) detector below the surface of the Earth sphere, in km
      self.detectorDepth = detectorDepth
      if detectorDepth < 0:
         warnings.warn("detectorDepth was set to a value smaller 0; your detector is %.3f km above ground!"
                       % abs(detectorDepth))
      # extension of the atmosphere above the surface of the Earth sphere, in km; ignored in default
      # atmosphere handling mode (uses Gaisser/Stanev model instead, see InteractionAlt method below)
      self.atmHeight = atmHeight
   
   
   def __repr__(self):
      # assumes that nuCraft was imported into the main namespace (i.e., from NuCraft...),
      # like numpy repr do; do   nc = eval("NuCraft."+repr(nc))   otherwise;
      # availability of the class EarthModel is essential
      if (    (self.M == self.ConstructMassMatrix(self._deltaMi1List)).all()
          and (self.U == self.ConstructMixingMatrix(self._angleList)).all()):
         return "NuCraft(%s, %s, earthModel=%s)" % (self._deltaMi1List, self._angleList, repr(self.earthModel))
      else:
         return "NuCraft(%s, %s, mod)" % (self._deltaMi1List, self._angleList)
   
   
   def __str__(self):
      return "nuCraft(n=%d)" % len(self.M)
   
   
   def ConstructMassMatrix(self, parList):
      """
      Construct and return a squared-mass matrix out of the input list;
      the first parameter is the mass of mass state 1, the following parameters are
      the correctly ordered squared mass differences of the other states to state 1,
      all given in units of eV or eV^2, respectively, i.e.,
      parList[i] = m_i^2 - m_1^2   for i > 0; e.g.,
      parList = [1., 7.50e-5, 7.50e-5+2.32e-3]
      """
      # ensure that all masses are positive
      assert -parList[0]**2 <= min(parList[1:]), "All masses have to be positive!"
      
      return diag([parList[0]**2] + [parList[0]**2 + m for m in parList[1:]])
   
   
   def ConstructMixingMatrix(self, parList):
      """
      Construct and return a mixing matrix out of the input list;
      the list should consist of tuples of the format (i,j,theta_ij), i<j, with theta
      in degrees, and the mixing matrix will be constructed in reverse order, e.g.:
      parList = [(1,2,33.89),(1,3,9.12),(2,3,45.00)]
      => U = R_23 . R_13 . R_12
      
      For CP-violating factors, use tuples like (i,j,theta_ij,delta_ij),
      with delta_ij in degrees.
      """
      dim = max([par[1] for par in parList])
      
      def RotMat(dim, i, j, ang, cp):   # actually not rotation matrices, but Gell-Mann-generated matrices
         if not i < j <= dim:
            raise Exception("Missconstructed rotation matrix: "+repr(dim)+", "+repr(i)+", "+repr(j)+", "+repr(ang))
         if cp == 0:
            R = eye(dim)
            R[i,j] = sin(ang)
            R[j,i] = -sin(ang)
         else:
            R = eye(dim, dtype='complex128')
            R[i,j] = sin(ang) * exp(-1j*cp)
            R[j,i] = -sin(ang) * exp(1j*cp)
         R[i,i] = R[j,j] = cos(ang)
         return R
      
      degToRad = pi/180.
      
      U = eye(dim)
      for par in parList:
         if len(par)>3:
            U = dot(RotMat(dim,par[0]-1,par[1]-1,par[2]*degToRad,par[3]*degToRad), U)
         else:
            U = dot(RotMat(dim,par[0]-1,par[1]-1,par[2]*degToRad,0), U)
      return U
   
   
   def InteractionAlt(self, mcType, mcEn, mcZen, mode):
      """
      Return a list of weight-altitude tuples for atmospheric propagation.
      
      Helper method; depending on mode, returns a list of one or more tuples of
      weights and altitudes in which the neutrinos should start to be propagated.
      The weights have to add up to one.
      
      mode 0:
         returns self.earthModel.rE + self.atmHeight with weight 1., which means that the
         interaction is expected to happen at a fixed hight (default 20 km) above ground level
      mode 1:
         samples a single altitude from a parametrization to the atmospheric interaction
         model presented in "Path length distributions of atmospheric neutrinos",
         Gaisser and Stanev, PhysRevD.57.1977
      mode 2 and mode 3:
         returns eight equally probable altitudes from the whole range of values allowed
         by the parametrization also used in mode 1
      
      The interaction height distributions in the paper quoted above are only given
      implicitely as differential equations without closed-form solutions.
      The parametrization was obtained by solving those equations numerically at a fixed
      energy of 2 GeV, as the energy depedence is weak and 2 GeV is the energy where
      oscillation effects start to become significant at the horizon, where the relative
      impact of the atmosphere is large. These numerical solutions were then parameterized
      by log-normal distributions as described in the nuCraft publication.
      As the equations in the paper are only given for six discrete zenith angles down to
      cos(zen) = 0.05, the solutions had to be inter- and extrapolated to other zenith
      angles. The interpolation was done by fitting the two parameters mu and sigma of the
      log-normal distributions as function of cos(zen), using a polynomial for mu and and
      a power function plus linear polynomial for sigma.
      The extrapolation was done by adding a cubically suppressed constant term to cos(zen)
      (see formula below), such that cos(zen) never falls below 0.05, thereby possibly
      underestimating the path length for very horizontal events, but achieving a realistic
      smooth transition between particles above and below the horizon.
      """
      
      if mode == 0:
         return [(1., self.earthModel.rE + self.atmHeight)]
      
      # extrapolation formula described above:
      # get the cosine of the zenith angle, and compensate for not having a parametrization
      # for the effective zenith angle (i.e., the zenith angle relative to the Earth's curvature
      # averaged through the atmosphere) at cos(zen) < 0.05;   0.000125 == 0.05**3
      cosZen = (fabs(scos(mcZen))**3 + 0.000125)**0.333333333
      
      # interpolation part described above, with numbers gained from the parametrizations
      if mcType in (12, -12):   # electron neutrinos, mostly from muon decay
         mu = 1.285e-9*(cosZen-4.677)**14. + 2.581
         sigma = 0.6048*cosZen**0.7667 - 0.5308*cosZen + 0.1823
      else:   # muon neutrinos, from muon and pion/kaon decay
         mu = 1.546e-9*(cosZen-4.618)**14. + 2.553
         sigma = 1.729*cosZen**0.8938 - 1.634*cosZen + 0.1844
      
      # log-normal distribution, shifted 12 km upwards as required by the parametrization
      logn = lognorm(sigma, scale=2*exp(mu), loc=-12)
      
      if mode == 1:
         # draw a random non-negative production height from the parametrization
         z = logn.rvs()*cosZen
         while z < 0:
            z = logn.rvs()*cosZen
         return [(1., z+self.earthModel.rE)]
      elif mode in [2, 3]:
         # get eight equally probable altitudes, using the cumulative distribution
         cdf0 = logn.cdf(0)
         # the array contains the central values of the partition of [0,1] into eight equally-sized
         # intervals, in ascending order
         qList = cdf0 + array([0.0625,0.1875,0.3125,0.4375,0.5625,0.6875,0.8125,0.9375])*(1.-cdf0)
         return list(zip(ones(8)/8., logn.ppf(qList)*cosZen + self.earthModel.rE))
      else:
         raise NotImplementedError("Unrecognized mode for neutrino interaction height estimation!")
   
   
   def CalcWeights(self, inList, vacuum=False, atmMode=3, numPrec=5e-4):
      """
      Calculate neutrino oscillation probabilities in the interaction basis.
      
      inList may be provided in three formats:
      1 - as a tuple of three lists: mcType, neutrino energy and zenith angle,
          i.e., ([type1, type2, ...],[energy1, energy2, ...],[zenith1, zenith2, ...])
      2 - as a list of tuples (mcType, neutrinoEnergy, zenith angle),
          i.e., [(type1, energy1, zenith1),(type2, energy2, zenith2),...]
      3 - a list of particles of a customizable format:
             from collections import namedtuple
             SimPart = namedtuple("SimPart", (..., "zenMC", "eMC", "mcType", "oscProb", ...))
             part = SimPart(..., zenith, energy, mctype, -1., ...)
      
      mcType uses PDG conventions:
         NuE:            mcType =  12
         NuEBar:         mcType = -12
         NuMu:           mcType =  14
         NuMuBar:        mcType = -14
         NuTau:          mcType =  16
         NuTauBar:       mcType = -16
         NuSterile_i:    mcType =  80+i     i = 1,2,3...
         NuSterileBar_i: mcType = -80-i
      
      energy is expected in units of GeV, zenith angles in radian
      
      return format:
      in input format cases 1 and 2, the return format is a list of [P_E, P_Mu, P_Tau, ...];
      in case 3, the list of particles is returned with updated oscProb fields
      
      atmMode controls the handling of the atmospheric interaction altitude:
       0 assumes a fixed height of self.atmHeight (20 km by default)
       1 draws a single altitude from a parametrization described in the InteractionAlt method
       2 calculates the average for eight altitudes distributed over the whole range according
         to the same parametrization as in mode 1; this is SLOW and only meant for debugging
       3 uses the same altitudes as mode 2, but only propagates the lowest-altitude neutrino
         and adds the remaining lengths as vacuum oscillations afterwards; this is pretty fast
         and used by default
      
      numPrec governs the numerical precision with which the Schroedinger equation is solved;
      the unitarity condition (i.e., the fact that the sum of the resulting probabilities
      should be 1.) is used as a simple cross-check, a warning is issued if the deviation from
      1. is larger than numPrec.
      """
      # for the input format checks, assume that inList is homogeneous
      assert type(vacuum) is bool, "Argument vacuum has to be a Boolean."
      assert atmMode in range(4), "Only atmModes 0, 1, 2, and 3 are supported."
      partMode = False
      if not len(inList):   # empty input
         return []
      elif type(inList) is tuple and not type(inList[0]) is tuple:   # input case 1
         assert len(inList[0]) == len(inList[1]) == len(inList[2]), \
                "The lists for type, energy and zenith have to be of the same length!"
         inList = zip(*inList)
      elif not type(inList) is tuple and type(inList[0]) is tuple:   # input case 2
         assert len(inList[0]) == 3, \
                "Input tuples need to have three attributes: type, energy, and zenith angle."
      elif type(inList[0]).__base__ is tuple:   # input case 3
         try:
            if not set(("zenMC", "eMC", "mcType", "oscProb")).issubset(inList[0]._fields):
               raise Exception("NamedTuple in inList does not have the required fields!")
         except AttributeError:
            raise NotImplementedError("Format of the input inList is not supported!")
         partMode = True
      else:
         raise NotImplementedError("Wrong input format!")
      
      VAC = dot(self.U, dot(self.M, conj(self.U).T))
      
      (svdVAC0n, svdVAC1, svdVAC2n) = linalg.svd(VAC)
      svdVAC0n = array(svdVAC0n, order='C')
      svdVAC2n = array(svdVAC2n, order='C')   # svd returns fortran-ordered arrays
      svdVAC0a = conj(svdVAC0n)
      svdVAC2a = conj(svdVAC2n)
      
      # caching of some quantities that will be used often later on
      rE = self.earthModel.rE
      rOCore = self.earthModel.rOCore
      rICore = self.earthModel.rICore
      rDet = self.earthModel.rE - self.detectorDepth
      profInt = self.earthModel.profInt   # matter density profile interpolation
      
      global lCache
      global aMSW
      global modVAC
      lCache = pi
      
      def calcProb(inTuple):
         """
         Calculate the oscillation probabilities of an individual nu.
         
         The constant -2.533866j that appears throughout this function is -2*j*1.266933,
         where j is the imaginary unit and the other factor is GeV*fm/(4*hbar*c), which
         is the factor required to transition from natural units to SI units.
         It is hard-coded as it is not a free parameter and will never change.
         """
         # python 3 does not support tuple parameter unpacking anymore
         mcType, mcEn, mcZen = inTuple
         
         def dscM(l, en, zen):
            """
            Update the MSW-induced mass-squared matrix aMSW with the current density,
            and update the state-transition matrix modVAC to the current time/position.
            """
            global lCache, aMSW, modVAC
            
            # if l did not change, the update is unnecessary
            if l == lCache:
               return
            
            # modVAC is the time-dependent state-transition matrix that brings a state
            # vector to the interaction basis, i.e., to the basis where the vacuum
            # oscillations are flat; see calcProb docstring for magic-number explanation
            if isAnti:
               modVAC = dot(svdVAC0a * exp(-2.533866j/en*svdVAC1*(L-l)), svdVAC2a)
            else:
               modVAC = dot(svdVAC0n * exp(-2.533866j/en*svdVAC1*(L-l)), svdVAC2n)
            # <==> modVAC = dot(dot(svdVAC0, diag(exp(-2.533866j/mcEn*svdVAC1*(L-l)))), svdVAC2)
            
            # distance from the center of Earth
            r = ssqrt( l*l + rDet*rDet - 2*l*rDet*scos(pi - zen) )
            
            if r <= rICore:
               aMSW = aMSWWoRhoICore * profInt(r)
            elif r <= rOCore:
               aMSW = aMSWWoRhoOCore * profInt(r)
            else:
               aMSW = aMSWWoRho * profInt(r)
            lCache = l
         
         def f(l, psi, en, zen):
            dscM(l, en, zen)
            return -2.533866j/en * dot(modVAC, aMSW*dot(psi, conj(modVAC)))
            # <==> return -2.533866j/en * dot(modVAC, dot(diag(aMSW), dot(conj(modVAC).T, psi)))
         
         def jac(l, psi, en, zen):
            dscM(l, en, zen)
            return -2.533866j/en * dot(modVAC*aMSW, conj(modVAC).T)
            # <==> return -2.533866j/en * dot(dot(modVAC, diag(aMSW)), conj(modVAC).T)
         
         try:
            mcType = self.mcTypeDict[mcType]
         except KeyError:
            raise KeyError("The mcType %d is not known to nuCraft!" % mcType)
         isAnti = mcType[0] == -1
         
         # inefficient performance-wise, but nicer code, and vacuum is fast enough anyway
         if vacuum:
            aMSWWoRho = zeros_like(self.earthModel.A)
            aMSWWoRhoOCore = aMSWWoRho
            aMSWWoRhoICore = aMSWWoRho
         else:
            aMSWWoRho = mcType[0] * self.earthModel.A * mcEn
            aMSWWoRhoOCore = mcType[0] * self.earthModel.AOCore * mcEn
            aMSWWoRhoICore = mcType[0] * self.earthModel.AICore * mcEn
         
         # depending on the mode, get a list of interaction altitude tuples, see method doc string;
         # the first number of the tuples is the weight, the second the distance of the interaction
         # point to the center of the Earth; the weights have to add up to 1.
         if atmMode == 3:
            # first get the tuples and propagate only the lowest-altitude neutrino:
            rAtmTuples = self.InteractionAlt(mcType, mcEn, mcZen, 2)
            rAtm = rAtmTuples[0][1]
            
            L = ssqrt( rAtm*rAtm + rDet*rDet - 2*rAtm*rDet*scos( mcZen - arcsin(sin(pi-mcZen)/rAtm*rDet) ) )
            dscM(L, mcEn, mcZen)
            
            solver = integrate.ode(f, jac).set_integrator('zvode', method='adams', order=5, with_jacobian=True,
                                                                   nsteps=1200000, atol=numPrec*2e-3, rtol=numPrec*2e-3)
            solver.set_initial_value(dot(modVAC, mcType[1]), L).set_f_params(mcEn, mcZen).set_jac_params(mcEn, mcZen)
            solver.integrate(0.)
            
            if isAnti:
               endVAC = dot(svdVAC0a * exp(-2.533866j/mcEn*svdVAC1*L), svdVAC2a)
            else:
               endVAC = dot(svdVAC0n * exp(-2.533866j/mcEn*svdVAC1*L), svdVAC2n)
            # <==> endVAC = dot(dot(svdVAC0, diag(exp(-2.533866j/mcEn*svdVAC1*L))), svdVAC2)
            
            results = [rAtmTuples[0][0] * square(absolute( dot(conj(endVAC).T, solver.y) ))]
            
            # now for all the other neutrinos, add the missing lengths as vacuum oscillations
            # at the end of the track; keep in mind that atmosphere is always handled as vacuum
            for rAtmWeight, rAtm in rAtmTuples[1:]:
               L = ssqrt( rAtm*rAtm + rDet*rDet - 2*rAtm*rDet*scos( mcZen - arcsin(sin(pi-mcZen)/rAtm*rDet) ) )
               
               if isAnti:
                  endVAC = dot(svdVAC0a * exp(-2.533866j/mcEn*svdVAC1*L), svdVAC2a)
               else:
                  endVAC = dot(svdVAC0n * exp(-2.533866j/mcEn*svdVAC1*L), svdVAC2n)
               
               results.append( rAtmWeight * square(absolute( dot(conj(endVAC).T, solver.y) )) )
         else:
            # in this case, just stupidly propagate every neutrino in the list...
            results = []
            for rAtmWeight, rAtm in self.InteractionAlt(mcType, mcEn, mcZen, atmMode):
               
               L = ssqrt( rAtm*rAtm + rDet*rDet - 2*rAtm*rDet*scos( mcZen - arcsin(sin(pi-mcZen)/rAtm*rDet) ) )
               dscM(L, mcEn, mcZen)
               
               solver = integrate.ode(f, jac).set_integrator('zvode', method='adams', order=5, with_jacobian=True,
                                                                      nsteps=1200000, atol=numPrec*2e-3, rtol=numPrec*2e-3)
               solver.set_initial_value(dot(modVAC, mcType[1]), L).set_f_params(mcEn, mcZen).set_jac_params(mcEn, mcZen)
               solver.integrate(0.)
               
               if isAnti:
                  endVAC = dot(svdVAC0a * exp(-2.533866j/mcEn*svdVAC1*L), svdVAC2a)
               else:
                  endVAC = dot(svdVAC0n * exp(-2.533866j/mcEn*svdVAC1*L), svdVAC2n)
               # <==> endVAC = dot(dot(svdVAC0, diag(exp(-2.533866j/mcEn*svdVAC1*L))), svdVAC2)
               results.append( rAtmWeight * square(absolute( dot(conj(endVAC).T, solver.y) )) )
         if not solver.successful():
               raise ArithmeticError("ODE solver was not successful, check for warnings about 'excess work done'!")
         prob = sum(results, 0)
         if abs(1-sum(prob)) > numPrec:
            warnings.warn("The computed unitarity does not meet the specified precision: %.2e > %.2e" % (abs(1-sum(prob)), numPrec))
         return prob
      
      if partMode:
         # return the list of input particles with updated oscProb fields
         return [p._replace(oscProb=calcProb((p.mcType, p.eMC, p.zenMC))) for p in inList]
      else:
         # return a list with one oscillation probability array per input tuple
         return [calcProb(t) for t in inList]
   
   
   def CalcWeightsLegacy(self, inList, vacuum=False, numPrec=5e-3):
      """
      Calculate neutrino oscillation probabilities in the flavor basis.
      
      Legacy mode, solves the Schroedinger equation in the flavor basis; better
      use CalcWeights, unless you know what you are doing.
      
      inList may be provided in three formats:
      1 - as a tuple of three lists: mcType, neutrino energy and zenith angle,
          i.e., ([type1, type2, ...],[energy1, energy2, ...],[zenith1, zenith2, ...])
      2 - as a list of tuples (mcType, neutrinoEnergy, zenith angle),
          i.e., [(type1, energy1, zenith1),(type2, energy2, zenith2),...]
      3 - a list of particles of a customizable format
             from collections import namedtuple
             SimPart = namedtuple("SimPart", (..., "zenMC", "eMC", "mcType", "oscProb", ...))
             part = SimPart(..., zenith, energy, mctype, -1., ...)
      
      mcType uses PDG conventions:
         NuE:            mcType =  12
         NuEBar:         mcType = -12
         NuMu:           mcType =  14
         NuMuBar:        mcType = -14
         NuTau:          mcType =  16
         NuTauBar:       mcType = -16
         NuSterile_i:    mcType =  80+i     i = 1,2,3...
         NuSterileBar_i: mcType = -80-i
      
      energy is expected in units of GeV, zenith angles in radian
      
      return format:
      in input format cases 1 and 2, the return format is a list of [P_E, P_Mu, P_Tau, ...];
      in case 3, the list of particles is returned with updated oscProb fields
      
      numPrec governs the numerical precision with which the Schroedinger equation is solved;
      the unitarity condition (i.e., the fact that the sum of the resulting probabilities
      should be 1.) is used as a simple cross-check, a warning is issued if the deviation from
      1. is larger than numPrec.
      
      Does not properly take into account the atmosphere (only atmMode 0 is available).
      """
      warnings.warn("Legacy mode methods are often slower and less precise, use at your own risk")
      
      # for the input format checks, assume that inList is homogeneous
      assert type(vacuum) is bool, "Argument vacuum has to be a Boolean."
      partMode = False
      if not len(inList):   # empty input
         return []
      elif type(inList) is tuple and not type(inList[0]) is tuple:   # input case 1
         assert len(inList[0]) == len(inList[1]) == len(inList[2]), \
                "The lists for type, energy and zenith have to be of the same length!"
         inList = zip(*inList)
      elif not type(inList) is tuple and type(inList[0]) is tuple:   # input case 2
         assert len(inList[0]) == 3, \
                "Input tuples need to have three attributes: type, energy, and zenith angle."
      elif type(inList[0]).__base__ is tuple:   # input case 3
         try:
            if not set(("zenMC", "eMC", "mcType", "oscProb")).issubset(inList[0]._fields):
               raise Exception("NamedTuple in inList does not have the required fields!")
         except AttributeError:
            raise NotImplementedError("Format of the input inList is not supported!")
         partMode = True
      else:
         raise NotImplementedError("Wrong input format!")
      
      VACn = dot(self.U, dot(self.M, conj(self.U).T))
      VACa = conj(VACn)
      
      # radius of the Earth, radius to the center of the detector,
      # and radius including the atmosphere, all in units of km
      rE = self.earthModel.rE
      rDet = self.earthModel.rE - self.detectorDepth
      rAtm = self.earthModel.rE + self.atmHeight
      
      global lCache
      global aMSW
      lCache = pi
      
      def calcProb(inTuple):
         """
         Calculate the oscillation probabilities of an individual nu.
         
         The constant -2.533866j that appears throughout this function is -j*1.266933,
         where j is the imaginary unit and the other factor is GeV*fm/(4*hbar*c), which
         is the factor required to transition from natural units to SI units.
         It is hard-coded as it is not a free parameter and will never change.
         """
         # python 3 does not support tuple parameter unpacking anymore
         mcType, mcEn, mcZen = inTuple
         
         def dscM(l, zen):
            """
            Update the MSW-induced mass-squared matrix aMSW with the current density.
            """
            global lCache, aMSW
            
            # if l did not change, the update is unnecessary
            if l == lCache:
               return
            
            # distance from the center of Earth
            r = ssqrt( l*l + rDet*rDet - 2*l*rDet*scos(pi - zen) )
            
            if r <= self.earthModel.rICore:
               aMSW = aMSWWoRhoICore * self.earthModel.profInt(r)
            elif r <= self.earthModel.rOCore:
               aMSW = aMSWWoRhoOCore * self.earthModel.profInt(r)
            else:
               aMSW = aMSWWoRho * self.earthModel.profInt(r)
            lCache = l
         
         def f(l, psi, en, zen):
            dscM(l, zen)
            # see calcProb docstring for magic-number explanation
            if isAnti:
               return -2.533866j/en * dot((VACa + aMSW), psi)
            else:
               return -2.533866j/en * dot((VACn + aMSW), psi)
         
         def jac(l, psi, en, zen):
            dscM(l, zen)
            if isAnti:
               return -2.533866j/en * (VACa + aMSW)
            else:
               return -2.533866j/en * (VACn + aMSW)
         
         try:
            mcType = self.mcTypeDict[mcType]
         except KeyError:
            raise KeyError("The mcType %d is not known to nuCraft!" % mcType)
         isAnti = mcType[0] == -1
         
         # inefficient performance-wise, but nicer code, and vacuum is fast enough anyway
         if vacuum:
            aMSWWoRho = diag(zeros_like(self.earthModel.A))
            aMSWWoRhoOCore = aMSWWoRho
            aMSWWoRhoICore = aMSWWoRho
         else:
            aMSWWoRho = diag(mcType[0] * self.earthModel.A * mcEn)
            aMSWWoRhoOCore = diag(mcType[0] * self.earthModel.AOCore * mcEn)
            aMSWWoRhoICore = diag(mcType[0] * self.earthModel.AICore * mcEn)
         
         L = ssqrt( rAtm*rAtm + rDet*rDet - 2*rAtm*rDet*scos( mcZen - arcsin(sin(pi-mcZen)/rAtm*rDet) ) )
         dscM(L, mcZen)
         
         solver = integrate.ode(f, jac).set_integrator('zvode', method='adams', order=5, with_jacobian=True,
                                                                nsteps=12000000, atol=numPrec*2e-2, rtol=numPrec*2e-2)
         solver.set_initial_value(mcType[1], L).set_f_params(mcEn, mcZen).set_jac_params(mcEn, mcZen)
         solver.integrate(0.)
         if not solver.successful():
            raise ArithmeticError("ODE solver was not successful, check for warnings about 'excess work done'!")
         
         prob = square(absolute(solver.y))
         if abs(1-sum(prob)) > numPrec:
            warnings.warn("The computed unitarity does not meet the specified precision: %.2e > %.2e" % (abs(1-sum(prob)), numPrec))
         return prob
      
      if partMode:
         # return the list of input particles with updated oscProb fields
         return [p._replace(oscProb=calcProb((p.mcType, p.eMC, p.zenMC))) for p in inList]
      else:
         # return a list with one oscillation probability array per input tuple
         return [calcProb(t) for t in inList]

