'''Wrapper around the nuSQuIDS python interface

Note that the nuSQuIDS NSI branch which contains the
full functionality this wrapper can provide is at
https://github.com/thehrh/nuSQuIDS/tree/nusquids_nsi (experimental!),
but the master branch https://github.com/arguelles/nuSQuIDS can
and should be used if no NSI are needed.
'''
from __future__ import absolute_import, print_function, division
import os
import numpy as np
try:
    import nuSQUIDSpy as nsq
except:
    pypath = os.environ.get('PYTHONPATH', 'not set') # pylint: disable=invalid-name
    raise ImportError(
        'Could not import the nuSQuIDS python interface. Your $PYTHONPATH'
        ' is: %s.' % pypath
    )

# Check which nuSQuIDS classes are available
NSI_CLASS = "nuSQUIDSNSIAtm"
DECOH_CLASS = "nuSQUIDSAtmDecoh"
ATM_CLASS = "nuSQUIDSAtm"
NSI_AVAIL = hasattr(nsq, NSI_CLASS)
DECOH_AVAIL = hasattr(nsq, DECOH_CLASS)
ATM_AVAIL = hasattr(nsq, ATM_CLASS)

if not NSI_AVAIL and not ATM_AVAIL and not DECOH_AVAIL:
    raise AttributeError(
        'nuSQuIDS interface does not seem to provide any'
        ' useful classes ("%s", "%s" or "%s").'
        % (NSI_CLASS, DECOH_CLASS, ATM_CLASS)
    )

from pisa import CACHE_DIR, FTYPE, OMP_NUM_THREADS, ureg, OrderedDict
from pisa.core.binning import MultiDimBinning
from pisa.stages.osc.pi_osc_params import OscParams
from pisa.utils.fileio import from_file
from pisa.utils.log import logging

__all__ = [
    # constants
    'NSI_AVAIL', 'DECOH_AVAIL', 'ATM_AVAIL', 'NSI_CLASS', 'DECOH_CLASS', 'ATM_CLASS', 'NSQ_CONST',
    # functions
    'validate_calc_grid', 'compute_binning_constants', 'init_nusquids_prop',
    'evolve_states', 'osc_probs', 'earth_model'
]

__version__ = '0.1'


NSQ_CONST = nsq.Const()

PRIMARIES = ['numu', 'numubar', 'nue', 'nuebar']
# flavor codes used internally by nuSQuIDS
FLAV_INDS = {'nue': 0, 'nuebar': 0, 'numu': 1, 'numubar': 1, 'nutau': 2,
             'nutaubar': 2}
FLAV_INDS = OrderedDict(sorted(FLAV_INDS.items(), key=lambda t: t[0]))

def validate_calc_grid(calc_grid):
    """Check whether a multi-dimensional binning is suitable for use as
    the grid on which oscillations are calculated for event-by-event
    reweighting."""
    calc_grid = MultiDimBinning(calc_grid)
    dim_names = set(calc_grid.names)
    if not dim_names == set(['true_energy', 'true_coszen']):
        raise ValueError('Oscillation grid must contain "true_energy" and'
                         ' "true_coszen" dimensions, and no more! Got "%s".'
                         % dim_names)


def compute_binning_constants(calc_grid):
    """Compute some binning constants used further down."""
    validate_calc_grid(calc_grid)
    binning = calc_grid.basename_binning
    cz_binning = binning['coszen']
    en_binning = binning['energy']

    cz_min = cz_binning.bin_edges.min().m_as('radian')
    cz_max = cz_binning.bin_edges.max().m_as('radian')
    en_min = en_binning.bin_edges.min().m_as('GeV') * NSQ_CONST.GeV
    en_max = en_binning.bin_edges.max().m_as('GeV') * NSQ_CONST.GeV
    cz_centers = cz_binning.weighted_centers.m_as('dimensionless')
    en_centers = en_binning.weighted_centers.m_as('GeV') * NSQ_CONST.GeV
    # if interpolation is used, need to extend the range to beyond the
    # outermost bin centers or nuSQuIDS will be unhappy
    cz_grid = np.array([cz_min] + cz_centers.tolist() + [cz_max])
    en_grid = np.array([en_min] + en_centers.tolist() + [en_max])
    return en_grid, cz_grid


def init_nusquids_prop(
        cz_nodes, en_nodes, nu_flav_no,
        rel_err=1.0e-5, abs_err=1.0e-5, progress_bar=True,
        use_nsi=False,use_decoherence=False,
    ):
    """Set up nuSQuIDs propagators (propagation medium,
    initial states, grid, etc.)."""

    assert not (use_nsi and use_decoherence), "Cannot use `NSI` and `decoherence` together"

    # choose nuSQuIDS class
    if use_nsi :
        assert NSI_AVAIL == True, 'nuSQuIDS atmopsheric NSI class not available'
        assert nu_flav_no == 3, 'nuSQuIDS atmopsheric NSI class only supports 3 neutrino flavors'
        NSQ = getattr(nsq, NSI_CLASS) # pylint: disable=invalid-name
    elif use_decoherence :
        assert DECOH_AVAIL == True, 'nuSQuIDS atmospheric decoherence class not available'
        assert nu_flav_no == 3, 'nuSQuIDS atmopsheric decoherence class only supports 3 neutrino flavors'
        NSQ = getattr(nsq, DECOH_CLASS) # pylint: disable=invalid-name
    else :
        assert ATM_AVAIL == True, 'nuSQuIDS atmospheric class not available'
        NSQ = getattr(nsq, ATM_CLASS) # pylint: disable=invalid-name

    cz_shape = cz_nodes.shape[0]
    en_shape = en_nodes.shape[0]
    shape = (cz_shape, en_shape) + (2, nu_flav_no)

    ini_states = {'nue': {}, 'numu': {}}
    propagators = {'nue': {}, 'numu': {}}

    for input_name in PRIMARIES:
        if input_name.endswith("bar"):
            continue
        # single-flavor initial states to assume for
        # oscillation probabilities
        ini_state = np.zeros(shape)
        ini_state[:, :, 0, FLAV_INDS[input_name]] = np.ones((cz_shape, en_shape), dtype=FTYPE)
        ini_state[:, :, 1, FLAV_INDS[input_name]] = np.ones((cz_shape, en_shape), dtype=FTYPE)

        ini_states[input_name] = ini_state

        # instantiate a nuSQuIDS instance
        nuSQ = NSQ(CosZenith_vector=cz_nodes, E_vector=en_nodes,
                   numneu=nu_flav_no, NT=nsq.NeutrinoType.both,
                   iinteraction=False)

        nuSQ.Set_EvalThreads(0) #OMP_NUM_THREADS) # TODO What to use here? Using 'OMP_NUM_THREADS' seems to make Set_initial_state slow and hnce the minimizer becomes unusable
        nuSQ.Set_ProgressBar(progress_bar)
        nuSQ.Set_rel_error(rel_err)
        nuSQ.Set_abs_error(abs_err)

        propagators[input_name] = nuSQ

    return ini_states, propagators


def evolve_states(cz_shape, propagators, ini_states, nsq_earth_atm, osc_params):
    '''Function that sets the oscillation parameters
    and solves for the neutrino evolution.'''
    if osc_params.nsi_eps.any() and not NSI_AVAIL:
        raise ValueError(
            'nuSQuIDS interface does not seem to support NSI parameters,'
            ' but you have requested to set at least one.'
        )

    for (input_name, nuSQ) in propagators.items():

        nu_flav_no = nuSQ.GetNumNeu()
        nuSQ.Set_EarthModel(nsq_earth_atm)
        nuSQ.Set_MixingAngle(0, 1, osc_params.theta12)
        nuSQ.Set_MixingAngle(0, 2, osc_params.theta13)
        nuSQ.Set_MixingAngle(1, 2, osc_params.theta23)
        if nu_flav_no == 4:
            nuSQ.Set_SquareMassDifference(3, osc_params.dm41)
            nuSQ.Set_MixingAngle(1, 3, osc_params.theta14)
        nuSQ.Set_SquareMassDifference(1, osc_params.dm21)
        nuSQ.Set_SquareMassDifference(2, osc_params.dm31) #TODO Should this be dm32?
        nuSQ.Set_CPPhase(0, 2, osc_params.deltacp)

        # invoke odd mechanism to set NSI parameters
        for icz in range(cz_shape): # pylint: disable=xrange-builtin
            nuSQ_icz = nuSQ.GetnuSQuIDS(icz)
            if hasattr(nuSQ_icz,"Set_epsilon_ee") : #TODO Find a nicer way to do this (maybe make all this into a class?)
                nuSQ_icz.Set_epsilon_ee(osc_params.eps_ee)
                nuSQ_icz.Set_epsilon_emu(osc_params.eps_emu)
                nuSQ_icz.Set_epsilon_etau(osc_params.eps_etau)
                nuSQ_icz.Set_epsilon_mumu(osc_params.eps_mumu)
                nuSQ_icz.Set_epsilon_mutau(osc_params.eps_mutau)
                nuSQ_icz.Set_epsilon_tautau(osc_params.eps_tautau)

        # set decoherence parameters
        if hasattr(nuSQ,"Set_DecoherenceGammaMatrix") : #TODO Find a nicer way to do this (maybe make all this into a class?)
            nuSQ.Set_DecoherenceGammaMatrix(osc_params.gamma21,osc_params.gamma31,osc_params.gamma32)
            nuSQ.Set_DecoherenceGammaEnergyDependence(osc_params.n_energy)

        nuSQ.Set_initial_state(ini_states[input_name], nsq.Basis.flavor)

        nuSQ.EvolveState()


def osc_probs(nuflav, propagators, true_energies, true_coszens, prob_e=None, prob_mu=None ):
    """Evaluate oscillation probs. from nue(bar) and numu(bar) to
    nuflav (of nutype) for given array of energy and cos(zenith).
    Arrays of true energy and zenith are tested to be of the same length.

    Parameters
    ----------
    nuflav : 'nue' or 'nuebar' or 'numu' or 'numubar' or 'nutau' or 'nutaubar'
        string for denoting neutrino flavor and type
    propagators : dict
        Dictionary of neutrino flavors and corresponding
        nuSQuIDS propagators
    true_energies : list or numpy array
        A list of the true energies in GeV
    true_coszens : list or numpy array
        A list of the true cosine zenith values
    prob_e : numpy array
        An array to fill with the probability to oscillate to nue(bar)
        If none is provided one will be created and returned (more efficient to pass an existing one)
        Must be the same shape as `true_energies` and `true_coszens`
    prob_mu : numpy array
        Same as `prob_e` but contains probabilities for oscillation to numu(bar)
    """
    if not isinstance(true_energies, np.ndarray):
        if not isinstance(true_energies, list):
            raise TypeError("`true_energies` must be a list or numpy array."
                            " You passed a '%s'." % type(true_energies))
    else:
        true_energies = np.array(true_energies)
    if not isinstance(true_coszens, np.ndarray):
        if not isinstance(true_coszens, list):
            raise TypeError("`true_coszens` must be a list or numpy array."
                            " You passed a '%s'." % type(true_coszens))
        else:
            true_coszens = np.array(true_coszens)
    if not ((true_coszens >= -1.0).all() and (true_coszens <= 1.0).all()):
        raise ValueError('Not all true coszens found to be between -1 and 1.')
    if not (true_energies >= 0.0).all():
        raise ValueError('Not all true energies found to be positive.')
    if not len(true_energies) == len(true_coszens):
        raise ValueError('Length of energy and coszen arrays must match.')
    if not nuflav in FLAV_INDS:
        raise ValueError('Only `nutype` values accepted are %s.'
                         ' Your choice: %s.' % (FLAV_INDS.keys(), nuflav))
    # needed by `EvalFlavor`:
    nutype = 1 if nuflav.endswith('bar') else 0
    kflav = FLAV_INDS[nuflav]
    #TODO Check nu falvor type in the events... (not handled very consistently with e.g. prob3 right now)

    # create arrays to hold results, unless they already exist
    # initialise with Nan to check whether calculation was performed later on
    if prob_e is None :
        prob_e = np.zeros(len(true_energies), dtype=FTYPE)*np.nan
    if prob_mu is None :
        prob_mu = np.zeros_like(prob_e, dtype=FTYPE)*np.nan

    # evaluate oscillation probabilties
    for (i, (cz, en)) in enumerate(zip(true_coszens, true_energies)):
        for (input_name, nuSQ) in propagators.items():
            if not isinstance(en.dtype,np.float64) : 
                en = np.float64(en)  #TODO Current nuSQuIDS pybindings can only accept double, not float. Fix this (overload) and remove this hack
            if not isinstance(cz.dtype,np.float64) : 
                cz = np.float64(cz)  #TODO Current nuSQuIDS pybindings can only accept double, not float. Fix this (overload) and remove this hack
            chan_prob = nuSQ.EvalFlavor(kflav, cz, en, nutype)
            if input_name == "nue":
                prob_e[i] = chan_prob
            elif input_name == "numu":
                prob_mu[i] = chan_prob
            else:
                raise ValueError(
                    "Input name '%s' not recognized!" % input_name
                )

    if np.isnan(prob_e).any() or np.isnan(prob_mu).any():
        raise ValueError(
            'Nan detected in oscillation probabilities returned by nuSQuIDS!'
        )

    return prob_e, prob_mu


def earth_model(YeI, YeO, YeM, PREM_file='osc/nuSQuIDS_PREM.dat'):  # pylint: disable=invalid-name
    """Return a `nuSQUIDSpy.EarthAtm` object with
    user-defined electron fractions. Note that a
    temporary Earth model file is produced (over-
    written) each time this function is executed.

    Parameters
    ----------
    YeI, YeO, YeM : float
        electron fractions in Earth's inner core,
        outer core, and mantle
        (defined by spherical shells with radii of
         1121.5, 3480.0, and 6371.0 km)
    PREM_file : str
        path to nuSQuIDS PREM Earth Model file whose
        electron fractions will be modified

    Returns
    -------
    earth_atm : nuSQUIDSpy.EarthAtm
        can be passed to `Set_EarthModel` method of
        a nuSQuIDs propagator object
    """
    logging.debug("Regenerating nuSQuIDS Earth Model with electron"
                  " fractions: YeI=%s, YeO=%s, YeM=%s" % (YeI, YeO, YeM))
    earth_radius = 6371.0 # km
    # radii at which main transitions occur according to PREM
    transition_radii = np.array([1121.5, 3480.0, earth_radius]) # km

    fname_tmp = os.path.join(CACHE_DIR, "nuSQuIDS_PREM_TMP.dat")
    PREM_file = from_file(fname=PREM_file, as_array=True)
    for i, (r, _, _) in enumerate(PREM_file):
        # r is fraction of total radius
        current_radius = r*earth_radius
        if current_radius <= transition_radii[0]:
            # inner core region
            Ye_new = YeI
        elif current_radius <= transition_radii[1]:
            # outer core region
            Ye_new = YeO
        elif current_radius <= transition_radii[2]:
            # mantle region
            Ye_new = YeM
        # update electron fraction
        PREM_file[i][2] = Ye_new
    # make temporary file
    np.savetxt(fname=fname_tmp, X=PREM_file)
    # create and return the Earth model from file
    earth_atm = nsq.EarthAtm(fname_tmp)
    return earth_atm


def test_nusquids_osc():
    """Test nuSQuIDS functions."""
    from pisa.core.binning import OneDimBinning
    # define binning for nuSQuIDS nodes (where master eqn. is solved)
    en_calc_binning = OneDimBinning(
        name='true_energy',
        bin_edges=np.logspace(0.99, 2.01, 40)*ureg.GeV,
    )
    cz_calc_binning = OneDimBinning(
        name='true_coszen',
        domain=[-1, 1]*ureg.dimensionless,
        is_lin=True,
        num_bins=21
    )
    # make 2D binning
    binning_2d_calc = en_calc_binning*cz_calc_binning
    # check it has necessary entries
    validate_calc_grid(binning_2d_calc)
    # pad the grid to make sure we can later on evaluate osc. probs.
    # *anywhere* in between of the outermost bin edges
    en_calc_grid, cz_calc_grid = compute_binning_constants(binning_2d_calc)
    # set up initial states, get the nuSQuIDS "propagator" instances
    ini_states, props = init_nusquids_prop(
        cz_nodes=cz_calc_grid,
        en_nodes=en_calc_grid,
        nu_flav_no=3,
        rel_err=1.0e-5,
        abs_err=1.0e-5,
        progress_bar=True
    )
    # make an Earth model
    YeI, YeM, YeO = 0.4656, 0.4957, 0.4656
    earth_atm = earth_model(YeI=YeI, YeM=YeM, YeO=YeO)

    # define some oscillation parameter values
    osc_params = OscParams()
    osc_params.theta23 = np.deg2rad(48.7)
    osc_params.theta12 = np.deg2rad(33.63)
    osc_params.theta13 = np.deg2rad(8.52)
    osc_params.theta14 = np.deg2rad(0.0)
    osc_params.dm21 = 7.40e-5
    osc_params.dm31 = 2.515e-3
    osc_params.dm41 = 0.
    osc_params.eps_ee = 0.
    osc_params.eps_emu = 0.
    osc_params.eps_etau = 0.
    osc_params.eps_mumu = 0.
    osc_params.eps_mutau = 0.005
    osc_params.eps_tautau = 0.
    # evolve the states starting from initial ones
    evolve_states(
        cz_shape=cz_calc_grid.shape[0],
        propagators=props,
        ini_states=ini_states,
        nsq_earth_atm=earth_atm,
        osc_params=osc_params
    )

    # define some points where osc. probs. are to be
    # evaluated
    en_eval = np.logspace(1, 2, 500) * NSQ_CONST.GeV
    cz_eval = np.linspace(-0.95, 0.95, 500)
    # look them up for appearing tau neutrinos
    nuflav = 'nutau'
    # collect the transition probabilities from
    # muon and electron neutrinos
    prob_e, prob_mu = osc_probs(  # pylint: disable=unused-variable
        nuflav=nuflav,
        propagators=props,
        true_energies=en_eval,
        true_coszens=cz_eval,
    )


if __name__ == "__main__":
    test_nusquids_osc()
