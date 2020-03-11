#!/usr/bin/env python

"""Handle Monte Carlo simulation run settings"""


from __future__ import absolute_import, division

import pisa.utils.fileio as fileio
import pisa.utils.flavInt as flavInt
from pisa.utils import resources as resources
from pisa.utils.cross_sections import CrossSections

# Following "import *" is intentionally done so that `eval` called in
# translate_source_dict will execute with direct access to numpy namespace
from numpy import * # pylint: disable=wildcard-import, unused-wildcard-import, redefined-builtin


__all__ = ['MCSimRunSettings', 'DetMCSimRunsSettings']

__author__ = 'J.L. Lanfranchi'

__license__ = '''Copyright (c) 2014-2017, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


# TODO: make ability to serialize instantiated MCSimRunSettings and
# DetMCSimRunsSettings objects back to JSON format from which they were
# generated

# TODO: put more thought into the MCSimRunSettings class

# TODO: make sure user is forced to choose run & detector here

class MCSimRunSettings(dict):
    """Handle Monte Carlo run settings

    Parameters
    ----------
    run_settings : string or dict
    run
    detector : string or None

    Notes
    -----
    run_settings dictionary format (and same for corresponding JSON file, e.g.
    resources/events/mc_sim_run_settings.json); example is for PINGU but should
    generalize to DeepCore, etc. Note that a JSON file will use null and not
    None.

    (Also note that expressions for numerical values utilizing the Python/numpy
    namespace are valid, e.g. "2*pi" will be evaluated within the code to its
    decimal value.)::

        {

          # Specify the detector name, lower case
          "pingu": {

            # Monte Carlo run number for this detector
            "388": {

              # Version of geometry, lower-case. E.g., ic86 for IceCube/DeepCore
              "geom": "v36",

              # A straightforward way of computing aeff/veff/meff is to keep all
              # simulated events and compare the #analysis/#simulated. If all
              # simulated events are kept, then the filename containing these is
              # recorded here.
              "all_gen_events_file": None,

              # Max and min azimuth angle simulated (rad)
              "azimuth_max": "2*pi",
              "azimuth_min": 0,

              # Max and min energy simulated (GeV)
              "energy_max": 80,
              "energy_min": 1,

              # GENIE simulates some un-physica events (interactions that will not
              # occur in nature). The number below was arrived at by Ken Clark, so
              # ask him for more info.
              "physical_events_fract": 0.8095,

              # GENIE has a prescale factor (TODO: generalize or eliminate for
              # other xsec?)
              "genie_prescale_factor": 1.2,

              # Neutrino flavors simulated
              "flavints": "nutau,nutaubar",

              # #nu / (#nu + #nubar) simulated
              "nu_to_total_fract": 0.5,

              # Number of events simulated per I3 file
              "num_events_per_file": 250000,

              # Number of I3 files used
              "num_i3_files": 195,

              # Simulated spectral inde gamma; value of 1 => E*{-1}
              "sim_spectral_index": 1,

              # Version of neutrino/ice cross sections used for the simulation
              "xsec_version": "genie_2.6.4",

              # Max and min zenith angle simulated (rad)
              "zenith_max": "pi",
              "zenith_min": 0
            }
          }
        }


    """
    def __init__(self, run_settings, run=None, detector=None):
        super().__init__()
        # TODO: clean up this constructor!
        #if isinstance(run_settings, str):
        #    rsd = jsons.from_json(resources.find_resource(run_settings))
        if isinstance(run_settings, dict):
            rsd = run_settings
        else:
            raise TypeError('Unhandled run_settings type passed in arg: %s'
                            %type(run_settings))
        #if detector is not None:
        #    try:
        #        rsd = rsd[detector]
        #    except:
        #        pass
        rsd = self.translate_source_dict(rsd)
        if not detector is None:
            detector = str(detector).strip()
        self.detector = detector
        self.run = run
        self.update(rsd)

    @staticmethod
    def translate_source_dict(d):
        d['tot_gen'] = d['num_events_per_file'] * d['num_i3_files']

        # TODO: does the following logic actually work with both old and new
        # conventions?

        # NOTE: the ',' --> '+' mapping is necessary since some data files
        # were saved prior to the convention that commas exclusively separate
        # groups while plus signs indicate flav/ints grouped together

        d['flavints'] = flavInt.NuFlavIntGroup(d['flavints'].replace(',', '+'))

        # Numeric fields are allowed to be expressions that get evaluated
        numeric_fields = [
            'azimuth_max',
            'azimuth_min',
            'energy_max',
            'energy_min',
            'physical_events_fract',
            'genie_prescale_factor',
            'nu_to_total_fract',
            'num_events_per_file',
            'num_i3_files',
            'sim_spectral_index',
            'zenith_max',
            'zenith_min',
        ]
        for f in numeric_fields:
            if isinstance(d[f], str):
                d[f] = eval(d[f])

        return d

    def consistency_checks(self, data, flav=None):
        # TODO: implement!
        pass

    def barnobarfract(self, barnobar=None, is_particle=None,
                      flav_or_flavint=None):
        """Fraction of events generated (either particles or antiparticles).

        Specifying whether you want the fraction for particle or
        antiparticle is done by specifying one (and only one) of the three
        parameters:
            `barnobar`, `is_particle` or `flav_or_flavint`

        Parameters
        ----------
        barnobar : None or int
            -1 for antiparticle, +1 for particle
        is_particle : None or bool
            True for particle, false for antiparticle
        flav_or_flavint : None or convertible to NuFlav or NuFlavInt
            Particle or antiparticles is determined from the flavor or flavint
            passed

        """
        nargs = sum([(not barnobar is None),
                     (not is_particle is None),
                     (not flav_or_flavint is None)])
        if nargs != 1:
            raise ValueError('One and only one of `barnobar`, `is_particle`,'
                             ' and `flav_or_flavint` must be specified. Got'
                             ' %d non-None args instead.' % nargs)

        if flav_or_flavint is not None:
            is_particle = flavInt.NuFlavInt(flav_or_flavint).particle
        elif barnobar is not None:
            is_particle = barnobar > 0

        if is_particle:
            return self['nu_to_total_fract']
        return 1 - self['nu_to_total_fract']

    def get_num_gen(self, barnobar=None, is_particle=None,
                    flav_or_flavint=None, include_physical_fract=True):
        """Return the number of events generated.

        Parameters
        ----------
        barnobar : None or int
            -1 for antiparticle or +1 for particle

        is_particle : None or bool

        flav_or_flavint : None or convertible to NuFlav or NuFlavInt
            If one of `barnobar`, `is_particle`, or `flav_or_flavint` is
            specified, returns only the number of particles or antiparticles
            generated. Otherwise (if none of those is specified), return the
            total number of generated events.

        include_physical_fract : bool
            Whether to include the "GENIE physical fraction", which accounts
            for events that are generated but are un-physical and therefore
            will never be detectable. These are removed to not penalize
            detection efficiency.

        """
        nargs = sum([(not barnobar is None),
                     (not is_particle is None),
                     (not flav_or_flavint is None)])
        if flav_or_flavint is not None:
            if (flav_or_flavint not in self.get_flavs()
                    and flav_or_flavint not in self.get_flavints()):
                return 0
        barnobarfract = 1
        if nargs > 0:
            barnobarfract = self.barnobarfract(
                barnobar=barnobar, is_particle=is_particle,
                flav_or_flavint=flav_or_flavint
            )
        physical_fract = 1
        if include_physical_fract:
            physical_fract = self['physical_events_fract']
        return self['tot_gen'] * barnobarfract * physical_fract

    def get_flavints(self):
        return self['flavints'].get_flavints()

    def get_flavs(self):
        return self['flavints'].get_flavs()

    def get_energy_range(self):
        """(min, max) energy in GeV"""
        return self['energy_min'], self['energy_max']

    def get_spectral_index(self):
        """Spectral index (positive number for negative powers of energy)"""
        return self['sim_spectral_index']

    def get_xsec_version(self):
        """Cross sectons version name used in generating the MC"""
        return self['xsec_version']

    def get_xsec(self, xsec=None):
        """Instantiated CrossSections object"""
        if xsec is None:
            return CrossSections(ver=self['xsec_version'])
        return CrossSections(ver=self['xsec_version'], xsec=xsec)


class DetMCSimRunsSettings(dict):
    """Handle Monte Carlo run settings for a detector (i.e., without specifying
    which run as is required for the MCSimRunSettings object)

    Since run is not specified at instantiation, method calls require the user
    to specify a run ID.

    Parameters
    ----------
    run_settings : string or dict
    detector : string or None

    See Also
    --------
    MCSimRunSettings : Same, but specifies a specific run at instantiation; see
                       class docstring for specification of run_settings dict /
                       JSON file

    """
    def __init__(self, run_settings, detector=None):
        super().__init__()
        if isinstance(run_settings, str):
            rsd = fileio.from_file(resources.find_resource(run_settings))
        elif isinstance(run_settings, dict):
            rsd = run_settings
        else:
            raise TypeError('Unhandled run_settings type passed in arg: ' +
                            type(run_settings))

        if detector:
            detector = str(detector).strip()
        self.detector = detector

        # Determine how deeply nested runs are in the dict to allow for
        # user to specify a dict that has multiple detectors in it OR
        # a dict with just a single detector in it
        if 'flavints' in rsd.values()[0]:
            runs_d = rsd
        elif 'flavints' in rsd.values()[0].values()[0]:
            if self.detector is None:
                if len(rsd) == 1:
                    runs_d = rsd.values()[0]
                else:
                    raise ValueError('Must specify which detector; detectors '
                                     'found: ' + str(rsd.keys()))
            else:
                runs_d = rsd[self.detector.strip()]
        else:
            raise Exception('dict must either be 3 levels: '
                            '{DET:{RUN:{...}}}; or 2 levels: {RUN:{...}}')

        # Force run numbers to be strings (JSON files cannot have an int as
        # a key, so it is a string upon import, and it's safest to keep it as
        # a string considering how non-standardized naming is in IceCube) and
        # convert actual run settings dict to MCSimRunSettings instances
        runs_d = {str(k): MCSimRunSettings(v) for k, v in runs_d.items()}

        # Save the runs_d to this object instance, which behaves like a dict
        self.update(runs_d)

    def consistency_checks(self, data, run, flav=None):
        pass

    def barnobarfract(self, run, barnobar=None, is_particle=None,
                      flav_or_flavint=None):
        return self[str(run)].barnobarfract(barnobar=barnobar,
                                            is_particle=is_particle,
                                            flav_or_flavint=flav_or_flavint)

    def get_num_gen(self, run, barnobar=None, is_particle=None,
                    flav_or_flavint=None, include_physical_fract=True):
        """Return the total number of events generated"""
        return self[str(run)].get_num_gen(
            barnobar=barnobar, is_particle=is_particle,
            flav_or_flavint=flav_or_flavint,
            include_physical_fract=include_physical_fract
        )

    def get_flavints(self, run):
        return self[str(run)].get_flavints()

    def get_flavs(self, run):
        return self[str(run)].get_flavs()

    def get_energy_range(self, run):
        """(min, max) energy in GeV"""
        return self[str(run)].get_energy_range()

    def get_spectral_index(self, run):
        """Spectral index (positive number for negative powers of energy)"""
        return self[str(run)].get_spectral_index()

    def get_xsec_version(self, run):
        """Cross sectons version name used in generating the MC"""
        return self[str(run)].get_xsec_version()

    def get_xsec(self, run, xsec=None):
        """Instantiated CrossSections object"""
        return self[str(run)].get_xsec(xsec)
