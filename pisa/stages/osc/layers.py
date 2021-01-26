"""
Calculation of Earth layers and electron densities.
"""


from __future__ import division

import numpy as np
try:
    import numba
except ImportError:
    numba = None

from pisa import FTYPE
from pisa.utils.fileio import from_file
from pisa.utils.log import logging, set_verbosity

__all__ = ['extCalcLayers', 'Layers']

__author__ = 'P. Eller','E. Bourbeau'

__license__ = '''Copyright (c) 2014-2020, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


if numba is None:
    class jit(object):
        """Decorator class to mimic Numba's `jit` when Numba is missing"""
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args):
            return args[0]
else:
    jit = numba.jit
    ftype = numba.typeof(FTYPE(1))



@jit(nopython=True, nogil=True, cache=True)
def extCalcLayers(cz,
        r_detector,
        prop_height,
        detector_depth,
        rhos,
        coszen_limit,
        radii,
        max_layers):
    """Layer density/distance calculator for each coszen specified.

    Accelerated with Numba if present.

    Parameters
    ----------
    cz             : coszen values (array of float)
    r_detector     : radial position of the detector (float)
    prop_height    : height at which neutrinos are assumed to be produced (float)
    detector_depth : depth at which the detector is buried (float)
    rhos           : densities (already weighted by electron fractions) (ndarray)
    radii          : radii defining the Earth's layer (ndarray)
    coszen         : coszen values corresponding to the radii above (ndarray)
    max_layers     : maximum number of layers it is possible to cross (int)

    Returns
    -------
    n_layers : int number of layers
    density : array of densities, flattened from (cz, max_layers)
    distance : array of distances per layer, flattened from (cz, max_layers)
    
    """

    # The densities, distances and number of layers are appended to one long list
    # which is later reshaped into containers of size (# of cz values, max_layers)
    # in the pi_prob3 module

    densities = np.zeros((len(cz), max_layers), dtype=FTYPE)
    distances = np.zeros((len(cz), max_layers), dtype=FTYPE)
    number_of_layers = np.zeros(len(cz))

    # Loop over all CZ values
    for i, coszen in enumerate(cz):

        r_prop = r_detector+detector_depth+prop_height
        # Compute the full path length
        # path_len = -r_detector * coszen + np.sqrt(r_detector**2. * coszen**2 - (r_detector**2. - r_prop**2.))

        # Determine if there will be a crossing of layer
        # idx is the index of the first inner layer
        idx = np.where(radii<r_detector)[0][0]

        # Deal with paths that do not have tangeants
        if coszen >= coszen_limit[idx]: 
            cumulative_distances = -r_detector * coszen + np.sqrt(r_detector**2. * coszen**2. - r_detector**2. + radii[:idx]**2.)
            # a bit of flippy business is done here to order terms
            # such that numpy diff can work
            segments_lengths = np.diff(np.concatenate((np.array([0.]), cumulative_distances[::-1])))
            segments_lengths = segments_lengths[::-1]
            segments_lengths = np.concatenate((segments_lengths, np.zeros(radii.shape[0] - idx)))

            density = rhos*(segments_lengths > 0.)

        else:
            #
            # Figure out how many layers are crossed twice
            # (meaning we calculate the negative and positive roots for these layers)
            #
            
            calculate_small_root = (coszen < coszen_limit) * (coszen_limit <= coszen_limit[idx])
            calculate_large_root = (coszen_limit>coszen)

            small_roots = - r_detector * coszen * calculate_small_root - np.sqrt(r_detector**2 * coszen**2 - r_detector**2 + radii**2) #, where=calculate_small_root, out=np.zeros_like(radii))
            large_roots = - r_detector * coszen * calculate_large_root + np.sqrt(r_detector**2 * coszen**2 - r_detector**2 + radii**2) #, where=calculate_large_root, out=np.zeros_like(radii))

            # Remove the negative root numbers, and the initial zeros distances
            small_roots = small_roots[small_roots>0]
            small_roots = np.concatenate((np.array([0.]), small_roots))

            # Reverse the order of the large roots
            # That should give the segment distances from the furthest layer to
            # the middle layer
            large_roots = large_roots[::-1]

            # concatenate large and small roots together
            #  this gives the cumulative distance from the detector outward
            full_distances = np.concatenate((np.zeros(1),small_roots[small_roots>0], large_roots[large_roots>0]))

            # Diff the distances and reverse the order 
            # such that the path starts away from the detector
            segments_lengths = np.diff(full_distances)
            segments_lengths = segments_lengths[::-1]


            # The last problem is to match back the densities
            # to the proper array elements. 
            # Densities coming out of the earth core are inverted w.r.t the
            # densities of the ones coming in, with the following exception:
            #
            # - the middle layer and the atmosphere must be counted only once
            #
            # - densities corresponding to layers that are not crossed must be removed
            #
            # NOTE: this assumes that the detector is not positioned in the atmosphere
            #
            # start by removing the layers not crossed from rhos
            inner_layer_mask = coszen_limit>coszen
            density = np.concatenate((rhos[inner_layer_mask],rhos[inner_layer_mask][1:-1][::-1]))

            # As an extra precaution, set all densities that are not crossed to zero
            density*=(segments_lengths>0)

        number_of_layers[i] = np.sum(segments_lengths > 0.)
        # append to the large list
        for j in range(len(density)):
            # index j may not run all the way to max_layers, unreached indices stay zero
            densities[i, j] = density[j]
            distances[i, j] = segments_lengths[j]

    return number_of_layers, densities, distances


class Layers(object):
    """
    Calculate the path through earth for a given layer model with densities
    (PREM [1]), the electron fractions (Ye) and an array of coszen values

    Parameters
    ----------
    prem_file : str
        path to PREM file containing layer radii and densities as white space
        separated txt

    detector_depth : float
        depth of detector underground in km

    prop_height : float
        the production height of the neutrinos in the atmosphere in km (?)


    Attributes
    ----------
    max_layers : int
            maximum number of layers (this is important for the shape of the
            output! if less than maximumm number of layers are crossed, it's
            filled up with 0s

    n_layers : 1d int array of length len(cz)
            number of layers crossed for every CZ value

    density : 1d float array of length (max_layers * len(cz))
            containing density values and filled up with 0s otherwise

    distance : 1d float array of length (max_layers * len(cz))
            containing distance values and filled up with 0s otherwise

    References
    ----------
    [1] A.M. Dziewonski and D.L. Anderson (1981) "Preliminary reference
        Earth model," Physics of the Earth and Planetary Interiors, 25(4),
        pp. 297 – 356.
        http://www.sciencedirect.com/science/article/pii/300031920181900467

    """
    def __init__(self, prem_file, detector_depth=1., prop_height=2.):
        # Load earth model
        if prem_file is not None :
            self.using_earth_model = True
            prem = from_file(prem_file, as_array=True)

            # The following radii and densities are extracted in reverse order
            # w.r.t the file. The first elements of the arrays below corresponds
            # the Earth's surface, and the following numbers go deeper toward the 
            # planet's core
            self.rhos = prem[..., 1][::-1].astype(FTYPE)
            self.radii = prem[..., 0][::-1].astype(FTYPE)
            r_earth = prem[-1][0]
            self.default_elec_frac = 0.5

            # Add an external layer corresponding to the atmosphere / production boundary
            self.radii = np.concatenate((np.array([r_earth+prop_height]), self.radii))
            self.rhos  = np.concatenate((np.ones(1, dtype=FTYPE), self.rhos))
            self.max_layers = 2 * (len(self.radii))


        else :
            self.using_earth_model = False
            r_earth = 6371.0 #If no Earth model provided, use a standard Earth radius value


        #
        # Make some checks about the input production height and detector depth
        #
        assert detector_depth > 0, 'ERROR: detector depth must be a positive value'
        assert detector_depth <= r_earth, 'ERROR: detector depth is deeper than one Earth radius!'
        assert prop_height >= 0, 'ERROR: neutrino production height must be positive'

        # Set some other
        self.r_detector = r_earth - detector_depth
        self.prop_height = prop_height
        self.detector_depth = detector_depth

        if self.using_earth_model:
            # Compute the coszen_limits
            self.computeMinLengthToLayers()
            


    def setElecFrac(self, YeI, YeO, YeM):
        """Set electron fractions of inner core, outer core, and mantle.
        Locations of boundaries between each layer come from PREM.

        Parameters
        ----------
        YeI, YeO, YeM : scalars
            Three electron fractions (Ye), where I=inner core, O=outer core,
            and M=mantle

        """
        if not self.using_earth_model :
            raise ValueError("Cannot set electron fraction when not using an Earth model")

        self.YeFrac = np.array([YeI, YeO, YeM], dtype=FTYPE)

        # re-weight the layer densities accordingly
        self.weight_density_to_YeFrac()

    def computeMinLengthToLayers(self):
        '''
        Deterine the coszen values for which a track will 
        be tangeant to a given layer.

        Given the detector radius and the layer radii:

        - A layer will be tangeant if radii<r_detector

        - Given r_detector and r_i, the limit angle 
          will be:

                sin(theta) = r_i / r_detector

        that angle can then be expressed back into a cosine using
        trigonometric identities

        '''
        coszen_limit = []
        # First element of self.radii is largest radius!
        for i, rad in enumerate(self.radii):
            # Using a cosine threshold instead!
            if rad >= self.r_detector:
                x = 1.
            else:
                x = - np.sqrt(1 - (rad**2 / self.r_detector**2))
            coszen_limit.append(x)
        self.coszen_limit = np.array(coszen_limit, dtype=FTYPE)



    def calcLayers(self, cz):
        """

        Parameters
        ----------
        cz : 1d float array
            Array of coszen values

        """

        if not self.using_earth_model:
            raise ValueError("Cannot calculate layers when not using an Earth model")

        # run external function
        self._n_layers, self._density, self._distance = extCalcLayers(
            cz=cz,
            r_detector=self.r_detector,
            prop_height=self.prop_height,
            detector_depth=self.detector_depth,
            rhos=self.rhos,
            coszen_limit=self.coszen_limit,
            radii=self.radii,
            max_layers=self.max_layers,
        )

    @property
    def n_layers(self):
        if not self.using_earth_model:
            raise ValueError("Cannot get layers when not using an Earth model")
        return self._n_layers

    @property
    def density(self):
        if not self.using_earth_model:
            raise ValueError("Cannot get density when not using an Earth model")
        return self._density

    @property
    def distance(self):
        return self._distance


    def calcPathLength(self, cz) :
        """

        Calculate path length of the neutrino through an Earth-sized sphere, given the 
        production height, detector depth and zenith angle.
        Useful if not considering matter effects.

        Parameters
        ----------
        cz : cos(zenith angle), either single float value or an array of float values

        """
        r_prop = self.r_detector + self.detector_depth + self.prop_height

        if not hasattr(cz,"__len__"):
            cz = np.array([cz])
        else:
            cz = np.array(cz)

        pathlength = - self.r_detector * cz + np.sqrt(self.r_detector**2. * cz**2 - (self.r_detector**2. - r_prop**2.))

        self._distance = pathlength

    def weight_density_to_YeFrac(self):
        '''
        Adjust the densities of the provided earth model layers
        for the different electron fractions in the inner core,
        outer core and mantle.
        '''

        # TODO make this generic
        R_INNER = 1221.5
        R_OUTER = 3480.
        R_MANTLE= 6371. # the crust is assumed to have the same electron fraction as the mantle

        assert isinstance(self.YeFrac, np.ndarray) and self.YeFrac.shape[0] == 3, 'ERROR: YeFrac must be an array of size 3'
        #
        # TODO: insert extra radii is the electron density boundaries
        #       don't match the current layer boundaries
        
        #
        # Weight the density properly
        #
        density_inner = self.rhos * self.YeFrac[0] * (self.radii <= R_INNER)
        density_outer = self.rhos * self.YeFrac[1] * (self.radii <= R_OUTER) * (self.radii > R_INNER)
        density_mantle = self.rhos * self.YeFrac[2] * (self.radii <= R_MANTLE) * (self.radii > R_OUTER)

        weighted_densities = density_inner + density_outer + density_mantle
        
        self.rhos = weighted_densities



def test_layers_1():

    logging.info('Test layers calculation:')
    layer = Layers('osc/PREM_4layer.dat')
    layer.setElecFrac(0.4656, 0.4656, 0.4957)
    cz = np.linspace(-1, 1, int(1e5), dtype=FTYPE)
    layer.calcLayers(cz)
    logging.info('n_layers = %s' %layer.n_layers)
    logging.info('density  = %s' %layer.density)
    logging.info('distance = %s' %layer.distance)

    logging.info('Test path length calculation:')
    layer = Layers(None)
    cz = np.array([1.,0.,-1.])
    layer.calcPathLength(cz)
    logging.info('coszen = %s' %cz)
    logging.info('pathlengths = %s' %layer.distance)

    logging.info('<< PASS : test_Layers 1 >>')

def test_layers_2():
    '''
    Validate the total distance travered,
    the number of layers crossed and the distance
    travelled in each of these layers, for 
    neutrinos coming from various zenith angles

    also test separately the calculation of critical
    zenith boundaries for any particular layer, as
    calculated by computeMinLengthToLayers
    '''
    from pisa.utils.comparisons import ALLCLOSE_KW
    #
    # The test file is a 4-layer PREM Earth model. The
    # file contains the following information:
    #
    # Distance to Earth's core [km]     density []
    # -----------------------------     ----------
    #               0.                     13.0
    #             1220.0                   13.0
    #             3480.0                   11.3
    #             5701.0                   5.0
    #             6371.0                   3.3
    #
    # Note that the order of these values is inverted in 
    # layer.radii, so the first element in this object
    # will be 6371

    # TEST I: critical coszen values
    #
    # For each layer, the angle at which a neutrino track will
    # become tangeant to a layer boundary can be calculated as
    # follow:
    #
    # cos(theta) = -np.sqrt(1-r_n**2/R_detector**2)
    #
    # where the negative value is taken because the zenith angle 
    # is larger than pi/2
    #
    # Note that if the layer is above the detector depth,
    # The critical coszen is set to 0.
    #
    layer = Layers('osc/PREM_4layer.dat', detector_depth=1., prop_height=20.)
    logging.info('detector depth = %s km' %layer.detector_depth)
    logging.info('Detector radius = %s km'%layer.r_detector)
    logging.info('Neutrino production height = %s km'%layer.prop_height)
    layer.computeMinLengthToLayers()
    ref_cz_crit = np.array([1., 1., -0.4461133826191877, -0.8375825182106081, -0.9814881717430358,  -1.], dtype=FTYPE)
    logging.debug('Asserting Critical coszen values...')
    assert np.allclose(layer.coszen_limit, ref_cz_crit, **ALLCLOSE_KW), f'test:\n{layer.coszen_limit}\n!= ref:\n{ref_cz_crit}'

    #
    # TEST II: Verify total path length (in vacuum)
    #
    # The total path length is given by:
    #
    # -r_detector*cz + np.sqrt(r_detector**2.*cz**2 - (r_detector**2. - r_prop**2.))
    #
    # where r_detector is the radius distance of
    # the detector, and r_prop is the radius
    # at which neutrinos are produced
    input_cz = np.cos(np.array([0., 36.* np.pi / 180., 63. * np.pi / 180., \
                         np.pi/2., 105.* np.pi / 180., 125. * np.pi / 180., \
                         170 * np.pi / 180., np.pi]))

    correct_length = np.array([21., 25.934954968613056, 45.9673929915939, 517.6688130455607,\
                              3376.716060094899, 7343.854310588515, 12567.773643090592, 12761.])
    layer.calcPathLength(input_cz)
    computed_length = layer.distance
    logging.debug('Testing full path in vacuum calculations...')
    assert np.allclose(computed_length, correct_length, **ALLCLOSE_KW), f'test:\n{computed_length}\n!= ref:\n{correct_length}'
    logging.info('<< PASS : test_Layers 2 >>')

def test_layers_3():
    #
    # TEST III: check the individual path distances crossed
    #           for the previous input cosines
    #
    # For negative values of coszen, the distance crossed in a layer i is:
    #
    # d_i = R_p*cos(alpha) + sqrt(Rp**2cos(alpha)**2 - (Rp**2-r1**2)))
    #
    # where Rp is the production radius, r1 is the outer limit of a layer
    # and alpha is an angle that relates to the total path D and zenith 
    # theta via the sine law:
    #
    # sin(alpha) = sin(pi-theta)*D /Rp
    #
    from pisa.utils.comparisons import ALLCLOSE_KW
    import copy

    logging.debug('Testing Earth layer segments and density computations...')
    layer = Layers('osc/PREM_4layer.dat', detector_depth=1., prop_height=20.)
    logging.info('detector depth = %s km' %layer.detector_depth)
    logging.info('Detector radius = %s km'%layer.r_detector)
    logging.info('Neutrino production height = %s km'%layer.prop_height)
    layer.computeMinLengthToLayers()

    # Define some electron densities
    # (Normally, these would come from some a config
    # file in PISA)
    YeI = 0.4656
    YeM = 0.4957
    YeO = 0.4656
    
    layer.setElecFrac(YeI, YeO, YeM)

    # Define a couple of key coszen directions
    # cz = 1 (path above the detector)
    # cz = 0 (horizontal path)
    # cz = -0.4461133826191877 (tangent to the first inner layer of PREM4)
    # cz = -1 (path below the detector)
    cz_values = np.array([1., 0, -0.4461133826191877, -1.], dtype=FTYPE)

    # Run the layer calculation
    layer.calcLayers(cz=cz_values)

    # Save a copy of the segment information, and reshape them as they
    # are reshaped in pi_prob3
    layers_crossed = copy.deepcopy(layer.n_layers)
    distance_segments = copy.deepcopy(layer.distance.reshape(4,layer.max_layers))
    density_segments  = copy.deepcopy(layer.density.reshape(4,layer.max_layers))

    # Replace the segmented distances by the total path length in vacuum
    # (like in test #2):
    layer.calcPathLength(cz_values)
    vacuum_distances = copy.deepcopy(layer.distance)


    # Print out the outcome of the layer calculations
    # Compare total segmented lengh with total vacuum path length
    logging.info('Down-going neutrino (coszen = 1):\n-------------')
    logging.info("number of layers: {}".format(layers_crossed[0]))
    logging.info("Densities crossed: {}".format(density_segments[0,:]))
    logging.info("Segment lengths: {}\n".format(distance_segments[0,:]))
    correct_path = np.array([20., 1.,0,0,0,0,0,0,0,0,0,0])
    assert np.allclose(distance_segments[0,:], correct_path, **ALLCLOSE_KW), 'ERROR in downgoing neutrino path: {0} vs {1}'.format(distance_segments[0,:],correct_path)

    logging.info('Horizontal neutrino (coszen = 0):\n-------------')
    logging.info("number of layers: {}".format(layers_crossed[1]))
    logging.info("Densities crossed: {}".format(density_segments[1,:]))
    logging.info("Segment lengths: {}\n".format(distance_segments[1,:]))
    correct_path = np.array([404.79277484435556, 112.87603820120549,0,0,0,0,0,0,0,0,0,0])
    assert np.allclose(distance_segments[1,:], correct_path, **ALLCLOSE_KW), 'ERROR in horizontal neutrino path: {0} vs {1}'.format(distance_segments[1,:],correct_path)

    logging.info('Neutrino tangeant to the first inner layer (coszen = -0.4461133826191877):\n-------------\n')
    logging.info("number of layers: {}".format(layer.n_layers[2]))
    logging.info("Densities crossed: {}".format(density_segments[2,:]))
    logging.info("Segment lengths: {}\n".format(distance_segments[2,:]))
    correct_path = np.array([44.525143211129944, 5685.725369597015,0,0,0,0,0,0,0,0,0,0])
    assert np.allclose(distance_segments[2,:], correct_path, **ALLCLOSE_KW), 'ERROR in tangeant neutrino path: {0} vs {1}'.format(distance_segments[2,:],correct_path)

    logging.info('Up-going neutrino (coszen = -1):\n-------------')
    logging.info("number of layers: {}".format(layer.n_layers[3]))
    logging.info("Densities crossed: {}".format(density_segments[3,:]))
    logging.info("Segment lengths: {}\n".format(distance_segments[3,:]))
    correct_path = np.array([20., 670., 2221., 2260., 2440., 2260., 2221., 669., 0, 0,0,0], dtype=FTYPE)
    assert np.allclose(distance_segments[3,:], correct_path, **ALLCLOSE_KW), 'ERROR in upgoing neutrino path: {0} vs {1}'.format(distance_segments[3,:],correct_path)

    logging.info('Comparing the segments sums with the total path in vacuum...')
    assert np.allclose(np.sum(distance_segments, axis=1), vacuum_distances, **ALLCLOSE_KW), 'ERROR: distance mismatch: {0} vs {1}'.format(np.sum(distance_segments, axis=1), vacuum_distances)

    logging.info('<< PASS : test_Layers 3 >>')
    



if __name__ == '__main__':
    set_verbosity(3)
    test_layers_1()
    test_layers_2()
    test_layers_3()
