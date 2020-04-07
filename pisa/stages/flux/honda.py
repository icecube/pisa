"""
This flux service provides flux values from tables provided by Honda.
The returned values are in a map with binning defined by the user. Currently
this can be either 2D (energy and cosZenith) or 3D (including azimuth) though
the latter is currently rather untested. If you end up using it please keep in
contact with us so we can work through any issues together!
This is either achieved through b-spline interpolation (done in both energy and
cosZenith dimensions simultaneously in log10(flux)) or an integral-preserving
method that manipulates 1 dimensional splines of integrated flux.
Support now also includes the 2D tables from Bartol

Most of the functionality will be ported from PISA with any necessary
changes/improvements applied.
"""


from __future__ import absolute_import, division

from collections.abc import Mapping

import numpy as np
import scipy.interpolate as interpolate

from pisa import numba_jit, ureg
from pisa.core.map import Map, MapSet
from pisa.core.stage import Stage
from pisa.utils.resources import open_resource
from pisa.utils.log import logging
from pisa.utils.profiler import profile


__all__ = ['apply_ratio_scale_2d', 'mmc2d', 'honda']

__author__ = 'S. Boeser, S. Wren'

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


@numba_jit(nopython=True, nogil=True, cache=True)
def apply_ratio_scale_2d(h1, h2, ratio_scale, out1, out2):
    n_i = h1.shape[0]
    n_j = h1.shape[1]
    for i in range(n_i):
        for j in range(n_j):
            orig_map_sum = h1[i, j] + h2[i, j]
            orig_map_ratio = h1[i, j] / h2[i, j]
            out2[i, j] = orig_map_sum / (1 + ratio_scale * orig_map_ratio)
            out1[i, j] = ratio_scale * orig_map_ratio * out2[i, j]
    return out1, out2


# NOTE: the following doesn't speed up the operation, but line_profile
# misidentifies such operations as being more time-consuming than they are.
def mmc2d(m1, m2, c):
    return m1*m2*c


class honda(Stage): #pylint: disable=invalid-name
    """
    Flux Service for performing interpolation on the Honda tables.

    Both 2D and 3D tables can be loaded; the specifics of this are explained
    below the respective load_table function.

    Currently there are two interpolation choices:

      1) 'bisplrep' - A simple b-spline representation. This is quick.
      2) 'integral-preserving' - A slower, but more accurate choice

    These will be explained in more details in the Notes section.

    Parameters
    ----------
    params : ParamSet or sequence with which to instantiate a ParamSet.
        Expected params are:
            atm_delta_index : float
                The systematic describing the shift in the shape of the
                numu(bar) data as a function of energy relative to the
                prediction from the Honda tables.

            energy_scale : float
                A proxy systematic designed to account for any uncertainty on
                the overall energy measurements. i.e. an energy scale of 0.9
                says all energies are systematically reconstructed at 90% of
                the truth (on average). This systematic works by evaluating the
                flux splines at energy E*energy_scale and reading it in to
                energy bin E.

            nue_numu_ratio : float
                The systematic accounting for the uncertainty in the relative
                production of nue(bar) and numu(bar) in the atmosphere. It is
                implemented in a normalising-preserving way i.e. total nue +
                numu will stay constant.

            nu_nubar_ratio : float
                The systematic accounting for the uncertainty in the relative
                production of nu and nubar in the atmosphere. It is implemented
                in a normalising-preserving way i.e. total nu + nubar will stay
                constant.

            oversample_e : int
                An integer by which to oversample the requested binning in the
                energy dimension. I.e. an oversampling of 10 means the flux
                will be calculated 10 times inside each energy bin. This is
                implemented in such a way that the bins become equally split
                up.

            oversample_cz : int
                An integer by which to oversample the requested binning in the
                cosZenith dimension. All the oversampling values will end up
                effectively "multiplied" together. I.e. if you request 10
                oversampling in both energy and cosZenith directions then each
                (E,cZ) bin will have 100 evaluations inside of it.

            flux_file : str
                The name of the file containing the atmospheric neutrino files
                to be read in.

            flux_mode : str
                The name of the interpolation method to be used on these
                atmospheric neutrino flux tables. The current choices are
                'bisplrep' and 'integral-preserving'. The former is fast, but
                the latter represents a more accurate choice.

    output_binning : The binning desired for the output maps

    disk_cache : None, str, or DiskCache
        If None, no disk cache is available.
        If str, represents a path with which to instantiate a utils.DiskCache
        object. Must be concurrent-access-safe (across threads and processes).

    outputs_cache_depth : int >= 0

    memcache_deepcopy : bool

    debug_mode : None, bool, or string
        Whether to store extra debug info for this service.

    Notes
    -----
    The `outputs` container generated by this service will be objects with the
    following `name` attributes:

        * 'numu'
        * 'numubar'
        * 'nue'
        * 'nuebar'

    Corresponding to the 4 conventional atmospheric neutrino flavours.

    This service reads in a table of flux values and then performs some form of
    interpolation on them.
    It is named honda.py since it expects that these input tables come from the
    Honda group and their simulations.
    All of these tables can be found on his website [1].
    These are provided for various locations, the list of which now includes
    the South Pole as of the 2014/15 calculations.
    The details of this latest calculation are published in Physics Reviews D
    [2]:
    These tables are provided over 3 dimensions - neutrino energy, the cosine
    of the neutrino zenith angle and the azimuth of the neutrino.
    This is what is meant by a 3D flux calculation in the context of this
    service.
    Please be aware that the 3D calculations are currently untested in PISA.
    If you find any issues with them please contact the developers.
    One also deals with 2D flux calculations where the azimuth is averaged
    over.
    This is the recommended set of tables to use for the energies one is
    interested in in IceCube.
    The Honda group also provide 1D flux calculations, where all of the angles
    are averaged over i.e. they are provided solely as a function of energy.

    The tables should all follow a consistent naming convention:

    1) The first part of the file name will be which group performed the
       simulation.
       So for us they should all say honda
    2) The second part will be the year the calculations were published.
    3) The third part will be the location.
       For example, spl means the South Pole and frj means Frejus.
       There may also be the word 'mountain' here since tables have been
       produced for some detectors with and without the large matter above them
       (mostly a mountain) that purports to shield them from cosmic rays.
    4) The fourth part will be the part of the solar cycle they were produced
       for.
       Solar activity has an effect on the primaries impacting the Earth and
       therefore on the flux of atmospheric neutrinos.
       This will either be solmax or solmin, which refers to solar maximum and
       solar minimum respectively.
       Solar maximum means the flux is slightly reduced.
    5) The fifth part will be a label whether the tables are 2D or 3D.
       In the former they will have aa in the name, which means
       azimuth-averaged.
       In the latter there will be no fifth part.
       This service has been designed to look for this string specifically to
       validate the input files together with the requested output binning.

    Each of these sections of the file name are separated by a hyphen, and they
    should all have the file extension '.d'.
    All of the tables in PISA can be found in pisa.utils/flux.
    PISA actively supports the following tables from Honda:

    * honda-2015-spl-solmax-aa.d
    * honda-2015-spl-solmin-aa.d
    * honda-2015-spl-solmax.d
    * honda-2015-spl-solmin.d

    As of the time of writing this service, these tables represented the latest
    and greatest output from the Honda group.
    The use of azimuth-averaged tables is recommended for the energies IceCube
    works with, but the ability to read 3D tables has also been included here.

    PISA also contains the following other tables:

    * honda-2012-spl-solmax-aa.d
    * honda-2012-spl-solmin-aa.d
    * honda-2012-spl-solmax.d
    * honda-2012-spl-solmin.d
    * honda-2011-frj-mountain-solmin-aa.d

    These are included for legacy purposes and are NOT recommended to be used.
    If you access the Honda website linked above you will notice that the 2012
    South Pole files listed here are NOT on that website.
    This is because they were given to the IceCube collaboration as an interim
    while Honda was between major releases of the simulation.
    They are known to be buggy and have problems which have been fixed with the
    release of the 2014/15 simulation.
    Please only use these tables when reproducing old results that also used
    them.
    The Frejus file is also included as a legacy since it was used for some of
    the very first PINGU results.
    It is not buggy per se, but since we have a better alternative please don't
    use it.
    One Bartol file is also provided for demonstration.

    * bartol-2004-sno-solmax-aa.d

    This calculation is older but is still useful for cross checks.

    To choose the tables file you wish for your PISA analysis, set 'flux_file'
    in your pipeline settings file to the desired file with the prefix 'flux/'
    so that the find_resource function knows where to find it.

    Once the tables have been read in, an interpolation is then performed on
    them.
    Currently PISA supports two methods for this with the Honda files:

    1) 'bisplrep' - A simple b-spline representation.

       Here, the data is splined in both energy and cosZenith dimensions
       simultaneously.
       This is done in log10 of the flux to make it more stable since the flux
       is mostly linear in this space as a function of energy.
       We use the SciPy module 'scipy.interpolate.bisplrep' to do this (hence
       the name of this interpolation method).
       This stands for Bivariate Spline Representation (Bivariate since it is
       done in two dimensions simultaneously).
       These splines are then just evaluated at the desired points using the
       function 'scipy.interpolate.bisplev'.

    2) 'integral-preserving' - A slower, but more accurate choice.

       Here, the data is manipulated in sets of 1D splines.
       When initialising this method, a spline is produced as a function of
       energy for every table cosZenith value.
       However, in order to be integral-preserving, these splines are of the
       integrated flux.
       This is implemented by summing the bin contents along the chosen
       dimension.
       Thus, the knots of this spline are the bin edges of the tables, so this
       method naturally gives the correct flux up to the boundaries which the
       bisplrep method does not.
       Not only this, but this is in fact the correct method by which to
       interpolate the tables, since the values provided are the average across
       the bin and NOT the value at the bin center.
       The splining is done using the scipy.interpolate.splrep function.

       In any case, these splines as a function of energy are of the integral
       of the product of flux and energy, since this makes the splining more
       stable. To evaluate the integral-preserved flux, the derivative of this
       spline is evaluated.
       This is achieved with the 'scipy.interpolate.splev' function with the
       argument 'der=1' (it defaults to 'der=0' i.e. direct evaluation of the
       spline).
       This is done for every spline in the set over all cosZenith and gives
       another set of points in that dimension at the required energy.
       One then just repeats the method of integration in this dimension (this
       time of just flux since it is stable enough and also makes no sense to
       multiply by cosZenith since we will get sign switches) and evaluates the
       derivative at the required cosZenith value.

       This method only works when the binning is constant.
       Thus, it must actually be done in the energy dimension as log10 of
       energy.
       With this, the Honda tables have a constant binning.
       Since this bin width was not accounted for in the integration, we
       instead account for it once the derivative is taken by multiplying by
       this bin width.
       This is a constant (for each dimension) set by the original tables.
       Since this is always consistent in the Honda tables they are hard-coded
       in this service.
       This is potentially something which may trip people up in the future.

    To choose between these, set 'flux_mode' in your pipeline settings file to
    either 'bisplrep' or 'integral-preserving'.
    Setting it to the latter is slower, but the recommended choice.


    References
    ----------
    .. [1] http://www.icrr.u-tokyo.ac.jp/~mhonda/

    .. [2] M. Honda, M. Sajjad Athar, T. Kajita, K. Kasahara, S. Midorikawa,
           "Atmospheric neutrino flux calculation using the NRLMSISE00
           atmospheric model", Phys. Rev. D92, 023004 (2015), arXiv:1502.03916.

    """
    def __init__(self, params, output_binning, error_method,
                 outputs_cache_depth, memcache_deepcopy, disk_cache=None,
                 debug_mode=None):
        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'atm_delta_index', 'energy_scale', 'nu_nubar_ratio',
            'nue_numu_ratio', 'oversample_e', 'oversample_cz',
            'flux_file', 'flux_mode'
        )

        # Define the names of objects that get produced by this stage
        output_names = (
            'nue', 'numu', 'nuebar', 'numubar'
        )

        # Invoke the init method from the parent class, which does a lot of
        # work for you. Note that we do not specify `input_names` here, since
        # there are no "inputs" used by this stage. (Of course there are
        # parameters, and files with info, but no maps or MC events are used
        # and transformed directly by this stage to produce its output.)
        super().__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning,
            debug_mode=debug_mode,
        )

        # Set the neutrio primaries
        # This has to match the headers in the Honda files
        # When adding new source files please ensure this is respected
        self.primaries = ['numu', 'numubar', 'nue', 'nuebar']

        # Initialisation of this service should load the flux tables
        # Can work with either 2D (E,Z,AA) or 3D (E,Z,A) tables.
        # Also, the splining should only be done once, so do that too
        if set(output_binning.names) == set(['true_energy', 'true_coszen',
                                             'true_azimuth']):
            self.load_3d_table(smooth=0.05)
            logging.warning('!! ############################################ !!')
            logging.warning('')
            logging.warning('You have requested a full 3D calculation from the '
                            'tables.')
            logging.warning('Please be aware that this version is '
                            'currently untested and so you should put your '
                            'results under extra scrutiny.')
            logging.warning('Please also keep in contact with the developers '
                            'about your findings!')
            logging.warning('')
            logging.warning('!! ############################################ !!')
        elif set(self.output_binning.names) == set(['true_energy',
                                                    'true_coszen']):
            self.load_2d_table(smooth=0.05)
            self.compute_2d_binning_constants()
        else:
            raise ValueError(
                'Incompatible `output_binning` for either 2D (requires'
                ' "true_energy" and "true_coszen") or 3D (additionally'
                ' requires "true_azimuth"). Faulty `output_binning`=%s'
                %self.output_binning
            )

        self.previous_energy_scale = None
        self.output_maps = None

    def load_2d_table(self, smooth=0.05):
        """Manipulate 2 dimensional flux tables.

        2D is expected to mean energy and cosZenith, where azimuth is averaged
        over (before being stored in the table) and the zenith range should
        include both hemispheres.

        Parameters
        ----------
        smooth : float
            The smoothing factor for the splining when using bisplrep
            Not changing from 0.05 is strongly recommended
            The integral-preserving has a fixed smoothing of 0.

        """

        flux_file = self.params.flux_file.value
        logging.debug("Loading atmospheric flux table %s", flux_file)

        # columns in Honda files are in the same order
        cols = ['energy'] + self.primaries

        # Load the data table
        table = np.genfromtxt(open_resource(flux_file),
                              usecols=list(range(len(cols))))
        mask = np.all(np.isnan(table) | np.equal(table, 0), axis=1)
        table = table[~mask].T

        flux_dict = dict(zip(cols, table))
        for key in flux_dict.keys():
            # There are 20 lines per zenith range
            flux_dict[key] = np.array(np.split(flux_dict[key], 20))

        # Set the zenith and energy range as they are in the tables
        # The energy may change, but the zenith should always be
        # 20 bins, full sky.
        flux_dict['energy'] = flux_dict['energy'][0]
        flux_dict['coszen'] = np.linspace(0.95, -0.95, 20)

        # Now get a spline representation of the flux table.
        logging.debug('Make spline representation of flux')

        flux_mode = self.params.flux_mode.value

        if flux_mode == 'bisplrep':
            logging.debug('Doing quick bivariate spline interpolation')

            # bisplrep needs this to be transposed.
            # Not exactly sure why, but there you go!
            for key in flux_dict.keys():
                if key not in ('energy', 'coszen'):
                    flux_dict[key] = flux_dict[key].T

            # do this in log of energy and log of flux (more stable)
            log_e_mesh, cz_mesh = np.meshgrid(np.log10(flux_dict['energy']),
                                              flux_dict['coszen'])

            self.spline_dict = {}
            for nutype in self.primaries:
                # Get the logarithmic flux
                log_flux = np.log10(flux_dict[nutype]).T
                # Get a spline representation
                spline = interpolate.bisplrep(log_e_mesh, cz_mesh, log_flux,
                                              s=smooth)
                # and store
                self.spline_dict[nutype] = spline

        elif flux_mode == 'integral-preserving':
            logging.debug('Doing this integral-preserving. Will take longer')

            self.spline_dict = {}

            # Do integral-preserving method as in IceCube's NuFlux
            # This one will be based purely on SciPy rather than ROOT
            # Stored splines will be 1D in integrated flux over energy
            int_flux_dict = {}
            # Energy and CosZenith bins needed for integral-preserving
            # method must be the edges of those of the normal tables
            # This will depend on the tables being loaded
            if 'honda' in self.params.flux_file.value:
                int_flux_dict['logenergy'] = np.linspace(-1.025, 4.025, 102)
            else:
                # Honda did a 3D calculation at the lower energies with
                # smaller bins. Above this the calculation is 1D and the
                # bin width doubles.
                low_log_energy = np.linspace(-1, 1, 41)
                high_log_energy = np.linspace(1.1, 4, 30)
                int_flux_dict['logenergy'] = np.concatenate(
                    [low_log_energy, high_log_energy]
                )
            int_flux_dict['coszen'] = np.linspace(-1, 1, 21)
            for nutype in self.primaries:
                # spline_dict now wants to be a set of splines for
                # every table cosZenith value.
                splines = {}
                cz_iter = 1
                for energyfluxlist in flux_dict[nutype]:
                    int_flux = []
                    tot_flux = 0.0
                    int_flux.append(tot_flux)
                    for energyfluxval, energyval in zip(
                            energyfluxlist,
                            flux_dict['energy']):
                        # Spline works best if you integrate flux * energy
                        # Since this is the integrated flux it must also be
                        # multiplied by bin width.
                        if 'honda' in self.params.flux_file.value:
                            # For honda this is easy, it's constant
                            tot_flux += energyfluxval*energyval*0.05
                        else:
                            # For Bartol there is a switch at 10 GeV where
                            # the bin widths double.
                            if energyval < 10.0:
                                tot_flux += energyfluxval*energyval*0.05
                            else:
                                tot_flux += energyfluxval*energyval*0.1
                        int_flux.append(tot_flux)

                    spline = interpolate.splrep(int_flux_dict['logenergy'],
                                                int_flux, s=0)
                    cz_value = '%.2f'%(1.05-cz_iter*0.1)
                    splines[cz_value] = spline
                    cz_iter += 1

                self.spline_dict[nutype] = splines

    def load_3d_table(self, smooth=0.05):
        """Manipulate 3 dimensional flux tables.

        3D is expected to mean energy, cosZenith, and azimuth.
        The angles coverage should be full sky.

        Parameters
        ----------
        smooth : float
            The smoothing factor for the splining when using bisplrep
            Not changing from 0.05 is strongly recommended
            The integral-preserving has a fixed smoothing of 0.
        """
        flux_file = self.params.flux_file.value
        logging.debug("Loading atmospheric flux table %s", flux_file)

        # columns in Honda files are in the same order
        cols = ['energy'] + self.primaries

        # Load the data table
        table = np.genfromtxt(open_resource(flux_file),
                              usecols=list(range(len(cols))))
        mask = np.all(np.isnan(table) | np.equal(table, 0), axis=1)
        table = table[~mask].T

        flux_dict = dict(zip(cols, table))
        for key in flux_dict.keys():

            # There are 20 lines per zenith range
            coszenith_lists = np.array(np.split(flux_dict[key], 20))
            azimuth_lists = []
            for coszenith_list in coszenith_lists:
                azimuth_lists.append(np.array(np.split(coszenith_list, 12)).T)
            flux_dict[key] = np.array(azimuth_lists)
            if key != 'energy':
                flux_dict[key] = flux_dict[key].T

        # Set the zenith and energy range
        flux_dict['energy'] = flux_dict['energy'][0].T[0]
        flux_dict['coszen'] = np.linspace(0.95, -0.95, 20)
        flux_dict['azimuth'] = np.linspace(15, 345, 12)

        # Now get a spline representation of the flux table.
        logging.debug('Make spline representation of flux')

        flux_mode = self.params.flux_mode.value

        if flux_mode == 'bisplrep':

            logging.debug('Doing quick bsplrep spline interpolation in 3D')
            # do this in log of energy and log of flux (more stable)
            log_e_mesh, cz_mesh = np.meshgrid(np.log10(flux_dict['energy']),
                                              flux_dict['coszen'])

            self.spline_dict = {}

            for nutype in self.primaries:
                self.spline_dict[nutype] = {}
                # Make 1 2D bsplrep in E,CZ for each azimuth value
                for az, f in zip(flux_dict['azimuth'], flux_dict[nutype]):
                    # Get the logarithmic flux
                    log_flux = np.log10(f.T)
                    # Get a spline representation
                    spline = interpolate.bisplrep(log_e_mesh, cz_mesh,
                                                  log_flux, s=smooth*4.)
                    # Found smoothing has to be weaker here. Not sure why.
                    # and store
                    self.spline_dict[nutype][az] = spline

        elif flux_mode == 'integral-preserving':

            logging.debug('Doing this integral-preserving. Will take longer')

            self.spline_dict = {}

            # Do integral-preserving method as in IceCube's NuFlux
            # This one will be based purely on SciPy rather than ROOT
            # Stored splines will be 1D in integrated flux over energy
            int_flux_dict = {}
            # Energy and CosZenith bins needed for integral-preserving
            # method must be the edges of those of the normal tables
            int_flux_dict['logenergy'] = np.linspace(-1.025, 4.025, 102)
            int_flux_dict['coszen'] = np.linspace(-1, 1, 21)
            for nutype in self.primaries:
                # spline_dict now wants to be a set of splines for
                # every table cosZenith value.
                # In 3D mode we have a set of these sets for every
                # table azimuth value.
                az_splines = {}
                for az, f in zip(flux_dict['azimuth'], flux_dict[nutype]):
                    splines = {}
                    cz_iter = 1
                    for energyfluxlist in f.T:
                        int_flux = []
                        tot_flux = 0.0
                        int_flux.append(tot_flux)
                        for energyfluxval, energyval \
                                in zip(energyfluxlist, flux_dict['energy']):
                            # Spline works best if you integrate flux * energy
                            tot_flux += energyfluxval*energyval*0.05
                            int_flux.append(tot_flux)

                        spline = interpolate.splrep(int_flux_dict['logenergy'],
                                                    int_flux, s=0)
                        cz_value = '%.2f'%(1.05-cz_iter*0.1)
                        splines[cz_value] = spline
                        cz_iter += 1

                    az_splines[az] = splines

                self.spline_dict[nutype] = az_splines

    def _compute_outputs(self, inputs=None):
        """Method for computing both 2D and 3D fluxes.

        The appropriate method is called based on the binning.
        This is done by checking the set of names matches what's expexted
        If the binning isn't energy and coszen (and azimuth if 3D) then this
        doesn't know what to do with it and stops.

        """
        output_maps = []

        if self.params.energy_scale.value != self.previous_energy_scale \
                or self.output_maps is None:
            for prim in self.primaries:
                outbnames = set(self.output_binning.names)
                if outbnames == set(['true_energy', 'true_coszen',
                                     'true_azimuth']):
                    output_maps.append(self.compute_3d_outputs(prim))
                elif outbnames == set(['true_energy', 'true_coszen']):
                    output_maps.append(self.compute_2d_outputs(prim))
                else:
                    raise ValueError(
                        'Incompatible `output_binning` for either 2D (requires'
                        ' "energy" and "coszen") or 3D (additionally requires'
                        ' "azimuth"). Faulty `output_binning`=%s'
                        %self.output_binning
                    )
            self.previous_energy_scale = self.params.energy_scale.value
            self.output_maps = output_maps

        # Combine the output maps into a single MapSet object to return.
        # The MapSet contains the varous things that are necessary to make
        # caching work and also provides a nice interface for the user to all
        # of the contained maps

        # Now apply systematics if needed
        # First check what needs doing
        nuenumuratio_toapply = self.params.nue_numu_ratio.value != 1.0
        nunubarratio_toapply = self.params.nu_nubar_ratio.value != 1.0
        atmdeltaindex_toapply = self.params.atm_delta_index.value != 0.0

        scaled_output_maps = MapSet(maps=self.output_maps, name='flux maps')
        if nuenumuratio_toapply:
            scaled_output_maps = self.apply_nue_numu_ratio(scaled_output_maps)
        if nunubarratio_toapply:
            scaled_output_maps = self.apply_nu_nubar_ratio(scaled_output_maps)
        if atmdeltaindex_toapply:
            scaled_output_maps = self.apply_delta_index(scaled_output_maps)

        return scaled_output_maps

    def compute_2d_binning_constants(self):
        self.bin_volumes = self.output_binning.bin_volumes(attach_units=False)
        # Adds/ensures the expected units for the binning
        if self.output_binning['true_energy'].units != ureg.GeV:
            self.all_binning = self.output_binning.to(true_energy='GeV',
                                                      true_coszen=None)
        else:
            self.all_binning = self.output_binning
        # Get bin centers to evaluate splines at
        self.evals = self.all_binning.true_energy.weighted_centers.magnitude
        self.czvals = self.all_binning.true_coszen.weighted_centers.magnitude
        self.proxy_binning = self.all_binning.reorder_dimensions(
            ['true_energy', 'true_coszen']
        )

    @profile
    def compute_2d_outputs(self, prim):
        """
        Method for computing 2 dimensional fluxes.
        Binning always expected in energy and cosZenith.
        Splines are manipulated based on whether
        they were set up as bisplrep or integral-preserving.

        Parameters
        ----------
        prim : string
            The string corresponding to the name of the primary.
            This deals with atmospheric neutrinos so the following primaries
            are expected:
              * 'numu'
              * 'numubar'
              * 'nue'
              * 'nuebar'

        """
        # Energy scale systematic says to read in flux to bin E at
        # energy_scale*E
        energy_scale = self.params.energy_scale.m_as('dimensionless')
        evals = self.evals * energy_scale
        logevals = np.log10(evals)
        czvals = self.czvals
        czkeys = ['%.2f'%x for x in np.linspace(-0.95, 0.95, 20)]
        cz_spline_points = np.linspace(-1, 1, 21)
        flux_mode = self.params.flux_mode.value

        if flux_mode == 'bisplrep':
            # Assert that spline dict matches what is expected
            # i.e. One spline for each primary
            assert not isinstance(self.spline_dict[self.primaries[0]], Mapping)

            # Get the spline interpolation, which is in
            # log(flux) as function of log(E), cos(zenith)
            return_table = interpolate.bisplev(np.log10(evals), czvals,
                                               self.spline_dict[prim])
            return_table = np.power(10., return_table)

        elif flux_mode == 'integral-preserving':
            # Assert that spline dict matches what is expected
            # i.e. One spline for every table cosZenith value
            #      0.95 is used for no particular reason
            #      These keys are strings, despite being numbers
            assert not isinstance(self.spline_dict[self.primaries[0]]['0.95'],
                                  Mapping)
            # 1st derivatives of one spline for each coszen (splines are
            # functions of energy)
            all_spline_vals = [np.zeros_like(logevals)]
            for czkey in czkeys:
                # Have to multiply by bin widths to get correct derivatives
                # Here the bin width is 0.05 (in log energy)
                spvals = interpolate.splev(logevals,
                                           self.spline_dict[prim][czkey],
                                           der=1)
                all_spline_vals.append(spvals)
            all_spline_vals = np.array(all_spline_vals)

            # Cumulative sum over coszen to get "integrated" spline values
            int_spline_vals = np.cumsum(all_spline_vals, axis=0)

            # TODO: the following loop is what takes almost all of the time in
            # this function. This is candidate for obtaining speedups! (Or can
            # algorithm be re-worked to avoid re-splining each time
            # compute_2d_outputs is called?)
            return_table = []
            for n, energyval in enumerate(evals):
                spline = interpolate.splrep(cz_spline_points,
                                            int_spline_vals[:, n], s=0)

                # Have to multiply by bin widths to get correct derivatives
                # Here the bin width is in cosZenith, is 0.1
                return_table.append(interpolate.splev(czvals, spline,
                                                      der=1)*(0.1/energyval))

            return_table = np.array(return_table)

        # Put the flux into a Map object, give it the output_name
        # Need a dummy binning object for this first
        return_map = Map(name=prim, hist=return_table,
                         binning=self.proxy_binning)

        # Now put map in correct dimensionality for user request
        return_map = return_map.reorder_dimensions(self.output_binning.names)

        # * Flux is given per sr and GeV, so we need to multiply
        #   by bin width in both dimensions i.e. the bin volume
        # * Energy scale systematic must be applied again here since it should
        #   come in the bin volume
        # * For 2D we also need to integrate over azimuth
        #   There is no dependency, so this is a multiplication of 2pi
        hist = mmc2d(return_map.hist, self.bin_volumes, 2*np.pi*energy_scale)
        return_map = Map(name=prim, hist=hist, binning=return_map.binning)
        #return_map *= self.bin_volumes*(2*np.pi*self.params.energy_scale.m)
        return return_map

    def compute_3d_outputs(self, prim):
        """Compute 3 dimensional fluxes when binning is specified in energy,
        cosZenith, and azimuth.

        Binning always expected in energy and cosZenith, and the previous
        compute_2d_outputs is called in that case instead.

        For now this just mimics the dummy functionality. Will add it in
        properly later.

        Parameters
        ----------
        prim : string
            The string corresponding to the name of the primary.
            This deals with atmospheric neutrinos so the following primaries
            are expected:
              * 'numu'
              * 'numubar'
              * 'nue'
              * 'nuebar'

        """
        # Adds/ensures the expected units for the binning
        all_binning = self.output_binning.to(true_energy='GeV',
                                             true_coszen=None,
                                             true_azimuth='deg')

        # Get bin centers to evaluate splines at
        evals = all_binning.true_energy.weighted_centers.magnitude
        czvals = all_binning.true_coszen.weighted_centers.magnitude
        azvals = all_binning.true_azimuth.weighted_centers.magnitude

        # Energy scale systematic says to read in flux to bin E at
        # energy_scale*E
        evals *= self.params.energy_scale.magnitude

        flux_mode = self.params.flux_mode.value

        if flux_mode == 'bisplrep':
            # Assert that spline dict matches what is expected
            # i.e. One spline for each primary for every table azimuth value
            #      45.0 is used for no particular reason
            assert not isinstance(self.spline_dict[self.primaries[0]][45.0],
                                  Mapping)

            # Get the spline interpolation, which is in
            # log(flux) as function of log(E), cos(zenith)
            # There is one for every table azimuth value
            az_maps = []
            for azkey in np.linspace(15.0, 345.0, 12):
                intermediate_table = interpolate.bisplev(
                    np.log10(evals), czvals, self.spline_dict[prim][azkey]
                )
                intermediate_table = np.power(10., intermediate_table)
                az_maps.append(intermediate_table)

            # Then have to interpolate in remaining dimension
            # This is done in this stupid manner since splining
            # in all 3 dimensions at once takes FOREVER.
            # This method is still slow, but is at least an improvement.
            return_table = []
            for enit in range(0, len(az_maps[0])):
                cz_vals = []
                for czit in range(0, len(az_maps[0][0])):
                    az_spline_vals = []
                    for azit in range(0, len(az_maps)):
                        az_spline_vals.append(az_maps[azit][enit][czit])
                    # Azimuth spline wants to be cyclic.
                    # Achieve this by adding 15 deg point at 375
                    az_spline_vals.append(az_spline_vals[0])
                    # Do this linearly to avoid issues in that dimension.
                    az_spline = interpolate.splrep(
                        np.linspace(15.0, 375.0, 13), az_spline_vals, k=1, s=0
                    )
                    # Therefore any value requested Az < 15 deg we need
                    # to evaluate at Az + 360 deg
                    proxy_azvals = []
                    for azval in azvals:
                        if azval < 15.0:
                            proxy_azvals.append(azval+360.0)
                        else:
                            proxy_azvals.append(azval)
                    # Thus these fluxes should be cyclic
                    azfluxes = interpolate.splev(proxy_azvals, az_spline)
                    cz_vals.append(azfluxes)
                return_table.append(cz_vals)

        elif flux_mode == 'integral-preserving':
            # Assert that spline dict matches what is expected
            # i.e. A set of splines for every table azimuth value corresponding
            #      to one spline for every table cosZenith value.
            #      45.0 and 0.95 are used for no particular reason
            #      cosZenith keys are strings, despite being numbers
            assert not isinstance(
                self.spline_dict[self.primaries[0]][45.0]['0.95'], Mapping)

            return_table = []
            for energyval in evals:
                logenergyval = np.log10(energyval)
                intermediate_table = []
                for czval in czvals:
                    az_spline_vals = []
                    for azkey in np.linspace(15.0, 345.0, 12):
                        cz_spline_vals = []
                        for czkey in np.linspace(-0.95, 0.95, 20):
                            # Have to multiply by bin widths to get correct
                            # derivatives, Here the bin width is in log
                            # energy is 0.05
                            cz_spline_vals.append(interpolate.splev(
                                logenergyval,
                                self.spline_dict[prim][azkey]['%.2f'%czkey],
                                der=1))
                        int_spline_vals = []
                        tot_val = 0.0
                        int_spline_vals.append(tot_val)
                        for val in cz_spline_vals:
                            tot_val += val
                            int_spline_vals.append(tot_val)

                        cz_spline = interpolate.splrep(np.linspace(-1, 1, 21),
                                                       int_spline_vals, s=0)
                        # Have to multiply by bin widths to get correct
                        # derivatives
                        # Here the bin width is in cosZenith, is 0.1
                        azflux = (interpolate.splev(czval, cz_spline, der=1)
                                  * 0.1/energyval)
                        az_spline_vals.append(azflux)
                    # Azimuth spline wants to be cyclic.
                    # Achieve this by adding 15 deg point at 375
                    az_spline_vals.append(az_spline_vals[0])
                    # Now spline in azimuth.
                    # Do this linearly to avoid issues in that dimension.
                    az_spline = interpolate.splrep(np.linspace(15.0, 375.0, 13),
                                                   az_spline_vals, k=1, s=0)
                    # Therefore any value requested Az < 15 deg we need
                    # to evaluate at Az + 360 deg
                    proxy_azvals = []
                    for azval in azvals:
                        if azval < 15.0:
                            proxy_azvals.append(azval+360.0)
                        else:
                            proxy_azvals.append(azval)
                    # Thus these fluxes should be cyclic
                    # So we evaluate this spline directly
                    # i.e. rather than the first derivative.
                    azfluxes = interpolate.splev(proxy_azvals, az_spline,
                                                 der=0)
                    # Save these fluxes in an intermediate table
                    intermediate_table.append(azfluxes)
                # Save these fluxes in to the return table
                # Dimensionality should be (E, CZ, Az)
                return_table.append(intermediate_table)

        # Need to turn the list in to a numpy array
        return_table = np.array(return_table)

        # Put the flux into a Map object, give it the output_name
        # Need a dummy binning object for this first
        proxy_binning = all_binning.reorder_dimensions(['true_energy',
                                                        'true_coszen',
                                                        'true_azimuth'])
        return_map = Map(name=prim,
                         hist=return_table,
                         binning=proxy_binning)

        # Now put map in correct dimensionality for user request
        return_map = return_map.reorder_dimensions(self.output_binning.names)

        # Flux is given per sr and GeV, so we need to multiply
        # by bin width in all dimensions
        # i.e. the bin volume
        return_map *= self.output_binning.bin_volumes(attach_units=False)

        # Energy scale systematic must be applied again here since it should
        # come in the bin volume
        return_map *= self.params.energy_scale.magnitude

        return return_map

    @staticmethod
    def apply_ratio_scale(map1, map2, ratio_scale):
        """Apply an arbitrary ratio systematic.

        This will be normalisation-preserving to minimise the correlation with
        any normalisation systematics such as aeff_scale.

        This systematic changes map1/map2 to a new value.

        If map1/map2 = orig_ratio then the returned maps will have

            map1/map2 = orig_ratio * ratio_scale.

        Normalisation-preserving means

            original_map1 + original_map2 = returned_map1 + returned_map2.

        Parameters
        ----------
        map1 : Map
            The numerator map in the ratio.
        map2 : Map
            The denominator map in the ratio.
        ratio_scale : float
            The amount by which to scale map1/map2 by.

        Returns
        -------
        scaled_map1, scaled_map2 : re-scaled maps

        """
        out1 = np.empty_like(map1.hist)
        out2 = np.empty_like(map2.hist)
        sh1, sh2 = apply_ratio_scale_2d(map1.hist, map2.hist, ratio_scale, out1, out2)
        scaled_map1 = Map(
            name=map1.name,
            hist=sh1,
            binning=map1.binning,
            tex=map1.tex,
            full_comparison=map1.full_comparison,
        )
        scaled_map2 = Map(
            name=map2.name,
            hist=sh2,
            binning=map2.binning,
            tex=map2.tex,
            full_comparison=map2.full_comparison,
        )

        return scaled_map1, scaled_map2

    def apply_nue_numu_ratio(self, flux_maps):
        """Apply the nue/numu ratio systematic.

        The actual calculation is done by apply_ratio_scale.

        Parameters
        ----------
        flux_maps : MapSet
            The set of maps returned by the Honda interpolation.

        Returns
        -------
        scaled_map_set : map set containing rescaled maps

        """
        scaled_nue_flux, scaled_numu_flux = self.apply_ratio_scale(
            map1=flux_maps['nue'],
            map2=flux_maps['numu'],
            ratio_scale=self.params.nue_numu_ratio.magnitude
        )

        scaled_nuebar_flux, scaled_numubar_flux = self.apply_ratio_scale(
            map1=flux_maps['nuebar'],
            map2=flux_maps['numubar'],
            ratio_scale=self.params.nue_numu_ratio.magnitude
        )

        # Names of maps get messed up by apply_ratio_scale function
        # Just redefine them correctly now to be sure
        scaled_numu_flux = Map(name='numu',
                               hist=scaled_numu_flux.hist,
                               binning=self.output_binning)
        scaled_numubar_flux = Map(name='numubar',
                                  hist=scaled_numubar_flux.hist,
                                  binning=self.output_binning)
        scaled_nue_flux = Map(name='nue',
                              hist=scaled_nue_flux.hist,
                              binning=self.output_binning)
        scaled_nuebar_flux = Map(name='nuebar',
                                 hist=scaled_nuebar_flux.hist,
                                 binning=self.output_binning)

        scaled_flux_maps = []
        scaled_flux_maps.append(scaled_nue_flux)
        scaled_flux_maps.append(scaled_numu_flux)
        scaled_flux_maps.append(scaled_nuebar_flux)
        scaled_flux_maps.append(scaled_numubar_flux)

        return MapSet(maps=scaled_flux_maps, name='flux maps')

    def apply_nu_nubar_ratio(self, flux_maps):
        """Apply the nue/nubar ratio systematic.

        The actual calculation is done by apply_ratio_scale.

        Parameters
        ----------
        flux_maps : MapSet
            The set of maps returned by the Honda interpolation.

        Returns
        -------
        scaled_map_set : map set containing rescaled maps

        """
        scaled_nue_flux, scaled_nuebar_flux = self.apply_ratio_scale(
            map1=flux_maps['nue'],
            map2=flux_maps['nuebar'],
            ratio_scale=self.params.nu_nubar_ratio.magnitude
        )

        scaled_numu_flux, scaled_numubar_flux = self.apply_ratio_scale(
            map1=flux_maps['numu'],
            map2=flux_maps['numubar'],
            ratio_scale=self.params.nu_nubar_ratio.magnitude
        )

        # Names of maps get messed up by apply_ratio_scale function
        # Just redefine them correctly now to be sure
        scaled_numu_flux = Map(name='numu',
                               hist=scaled_numu_flux.hist,
                               binning=self.output_binning)
        scaled_numubar_flux = Map(name='numubar',
                                  hist=scaled_numubar_flux.hist,
                                  binning=self.output_binning)
        scaled_nue_flux = Map(name='nue',
                              hist=scaled_nue_flux.hist,
                              binning=self.output_binning)
        scaled_nuebar_flux = Map(name='nuebar',
                                 hist=scaled_nuebar_flux.hist,
                                 binning=self.output_binning)

        scaled_flux_maps = []
        scaled_flux_maps.append(scaled_nue_flux)
        scaled_flux_maps.append(scaled_numu_flux)
        scaled_flux_maps.append(scaled_nuebar_flux)
        scaled_flux_maps.append(scaled_numubar_flux)

        return MapSet(maps=scaled_flux_maps, name='flux maps')

    def apply_delta_index(self, flux_maps):
        r"""Apply the atmospheric index systematic.

        This is applied as a shift in the E^{-\gamma} spectrum predicted by
        Honda. I.e. A value of the atmospheric index systematic of 0 leaves the
        shape as a function of E unchanged. A value of the atmospheric index
        systematic of 0.5 shifts it to E^{-\gamma+0.5}

        This is applied in a normalisation-preserving way.

        Currently it is only applied to the numu/numubar flux.

        Parameters
        ----------
        flux_maps : MapSet
            The set of maps returned by the Honda interpolation.

        """
        scaled_flux_maps = []

        for flav in ['numu', 'numubar']:
            if len(self.output_binning.names) == 2:
                all_binning = self.output_binning.to(true_energy='GeV',
                                                     true_coszen=None)
            elif len(self.output_binning.names) == 3:
                all_binning = self.output_binning.to(true_energy='GeV',
                                                     true_coszen=None,
                                                     true_azimuth='deg')
            evals = all_binning.true_energy.weighted_centers.magnitude

            # Values are pivoted about the median energy in the chosen range
            median_energy = evals[len(evals)//2]

            # This pivoting will give an energy-dependent scale factor
            scale = np.power(evals/median_energy,
                             self.params.atm_delta_index.value)

            flux_map = flux_maps[flav]

            # Need to multiply along the energy axis, so it must be last
            if 'energy' not in self.output_binning.names[-1]:
                if len(self.output_binning.names) == 2:
                    transposed_map = flux_map.reorder_dimensions(
                        ['true_coszen', 'true_energy']
                    )
                elif len(self.output_binning.names) == 3:
                    transposed_map = flux_map.reorder_dimensions(
                        ['true_azimuth', 'true_coszen', 'true_energy']
                    )
                scaled_transposed_map = transposed_map*scale
                scaled_flux = scaled_transposed_map.reorder_dimensions(
                    self.output_binning.names
                )
            else:
                scaled_flux = flux_map*scale

            # Now perform the renormalisation
            scaled_flux *= (flux_map.hist.sum()/scaled_flux.hist.sum())

            scaled_flux_maps.append(scaled_flux)

        scaled_flux_maps.insert(0, flux_maps['nue'])
        scaled_flux_maps.insert(2, flux_maps['nuebar'])

        return MapSet(maps=scaled_flux_maps, name='flux maps')

    def validate_params(self, params):
        """Do some checks on the parameters"""

        # Currently, these are the only interpolation methods supported
        assert (params.flux_mode.value in ['integral-preserving',
                                           'bisplrep'])

        # This is the Honda service after all...
        if 'honda' not in params.flux_file.value:
            # But it does support the Bartol tables too now
            if 'bartol' not in params.flux_file.value:
                raise ValueError(
                    "Input flux file must be from Honda or possibly "
                    "from Bartol. This is identified by having honda "
                    "or bartol in the name. The file you provided "
                    "was %s."%params.flux_file.value
                )

        # Flux file should have aa (for azimuth-averaged) if binning
        # is energy and cosZenith
        if set(self.output_binning.names) == set(['true_energy',
                                                  'true_coszen']):
            assert 'aa' in params.flux_file.value

        # Flux file should not have aa (for azimuth-averaged) if binning
        # is energy, cosZenith and azimuth
        elif set(self.output_binning.names) == set(['true_energy',
                                                    'true_coszen',
                                                    'true_azimuth']):
            assert 'aa' not in params.flux_file.value

        # Ratio systematics should not be negative.
        # This makes no sense after all
        assert params.nue_numu_ratio.value > 0.0
        assert params.nu_nubar_ratio.value > 0.0

        # Neither should energy scale for that matter
        assert params.energy_scale.value > 0.0
