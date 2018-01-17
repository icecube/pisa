# pylint: disable = not-callable, bad-whitespace
'''Wrapper around the nuSQuIDS python interface
'''
from __future__ import absolute_import, print_function, division

import numpy as np
import nuSQUIDSpy as nsq

from pisa import CACHE_DIR, FTYPE, OMP_NUM_THREADS, ureg, OrderedDict
from pisa.core.binning import MultiDimBinning
from pisa.stages.osc.pi_osc_params import OscParams
from pisa.utils.fileio import from_file
from pisa.utils.resources import find_resource

__all__ = [
    # constants
    'NSI_AVAIL', 'ATM_AVAIL', 'NSQ', 'NSQ_CONST'
]

__version__ = '0.1'

NSI_CLASS = "nuSQUIDSNSIAtm"
ATM_CLASS = "nuSQUIDSAtm"
NSI_AVAIL = hasattr(nsq, NSI_CLASS)
ATM_AVAIL = hasattr(nsq, ATM_CLASS)
if NSI_AVAIL:
    NSQ = getattr(nsq, NSI_CLASS)
elif ATM_AVAIL:
    NSQ = getattr(nsq, ATM_CLASS)
else:
    raise AttributeError('Could not find classes %s or %s in the nuSQuIDS'
                         ' python interface.'%(NSI_CLASS, ATM_CLASS))
del NSI_CLASS, ATM_CLASS

NSQ_CONST = nsq.Const()

PRIMARIES = ['numu', 'numubar', 'nue', 'nuebar']
FLAV_INDS = {'nue': 0, 'nuebar': 0, 'numu': 1, 'numubar': 1, 'nutau': 2,
             'nutaubar': 2}
FLAV_INDS = OrderedDict(sorted(FLAV_INDS.items(), key=lambda t: t[0]))

def validate_calc_grid(calc_grid):
    """Check whether a multi-dimensional binning is suitable for use as
    the grid on which oscillations are calculated for event-by-event
    reweighting."""
    if calc_grid is None:
        return
    calc_grid = MultiDimBinning(calc_grid)
    dim_names = set(calc_grid.names)
    if not dim_names == set(['true_energy', 'true_coszen']):
        raise ValueError('Oscillation grid must contain "true_energy" and'
                         ' "true_coszen" dimensions, and no more! Got "%s".'
                         % dim_names)


def compute_binning_constants(calc_grid):
    """Compute some binning constants used further down."""
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


def _get_nusquids_ini_prop(cz_nodes,
                           en_nodes,
                           nu_flav_no,
                           rel_err=1.0e-5,
                           abs_err=1.0e-5,
                           progress_bar=True,
                          ):
    """Set up nuSQuIDs propagators (propagation medium,
    initial states, grid, etc.)."""
    #logging.trace('Entering nusquids._init_nusquids_calc_grid')

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
        ini_state[:,:,0,FLAV_INDS[input_name]] = np.ones((cz_shape, en_shape))
        ini_state[:,:,1,FLAV_INDS[input_name]] = np.ones((cz_shape, en_shape))

        ini_states[input_name] = ini_state

        # instantiate a nuSQuIDS instance
        nuSQ = NSQ(cz_nodes, en_nodes, nu_flav_no, nsq.NeutrinoType.both,
                   False)

        nuSQ.Set_EvalThreads(OMP_NUM_THREADS)
        nuSQ.Set_ProgressBar(progress_bar)
        nuSQ.Set_rel_error(rel_err)
        nuSQ.Set_abs_error(abs_err)

        propagators[input_name] = nuSQ

    return ini_states, propagators


def _setup_physics_and_evolve_states(cz_shape,
                                     propagators,
                                     ini_states,
                                     nsq_earth_atm,
                                     osc_params,
                                    ):
    for (input_name, nuSQ) in propagators.iteritems():
        nu_flav_no = nuSQ.GetNumNeu()
        nuSQ.Set_EarthModel(nsq_earth_atm)
        nuSQ.Set_MixingAngle(0, 1, osc_params.theta12)
        nuSQ.Set_MixingAngle(0, 2, osc_params.theta13)
        nuSQ.Set_MixingAngle(1, 2, osc_params.theta23)
        if nu_flav_no == 4:
            raise NotImplementedError("n=3 required")
            # taken from the "atm default" example on github
            # TODO: make configurable
            nuSQ.Set_SquareMassDifference(3, -1.)
            nuSQ.Set_MixingAngle(1, 3, 0.160875)

        nuSQ.Set_SquareMassDifference(1, osc_params.dm21)
        nuSQ.Set_SquareMassDifference(2, osc_params.dm31)

        nuSQ.Set_CPPhase(0, 2, osc_params.deltacp)

        for icz in xrange(cz_shape):
            nuSQ_NSI_icz = nuSQ.GetnuSQuIDS(icz)
            nuSQ_NSI_icz.Set_epsilon_ee(osc_params.eps_ee)
            nuSQ_NSI_icz.Set_epsilon_emu(osc_params.eps_emu)
            nuSQ_NSI_icz.Set_epsilon_etau(osc_params.eps_etau)
            nuSQ_NSI_icz.Set_epsilon_mumu(osc_params.eps_mumu)
            nuSQ_NSI_icz.Set_epsilon_mutau(osc_params.eps_mutau)
            nuSQ_NSI_icz.Set_epsilon_tautau(osc_params.eps_tautau)

        nuSQ.Set_initial_state(ini_states[input_name], nsq.Basis.flavor)
        nuSQ.EvolveState()


def _eval_osc_probs(kNuBar,
                    kFlav,
                    propagators,
                    true_energies,
                    true_coszens,
                   ):
    """Calculate oscillation probs. for given array of energy and cos(zenith).
    Arrays of true energy and zenith are tested to be of the same length.
    Parameters
    ----------
    kNuBar : 1 or -1
        Code for denoting nu or anti-nu
    kFlav : 0, 1, or 2
        Code for denoting neutrino flavor
    propagators : dict
        Dictionary of neutrino flavors and corresponding
        nuSQuIDS propagators
    true_energies : list or numpy array
        A list of the true energies in GeV
    true_coszens : list or numpy array
        A list of the true cosine zenith values
    Example
    -------
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
    if not ((true_energies >= 0.0).all()):
        raise ValueError('Not all true energies found to be positive.')
    if not len(true_energies) == len(true_coszens):
        raise ValueError('Length of energy and coszen arrays must match.')
    if not (kNuBar == 1 or kNuBar == -1):
        raise ValueError('Only `kNuBar` values accepted are 1 and -1. Your'
                         ' choice: %s.' % kNuBar)

    # TODO: initalise with Nan to check whether calculation was performed
    prob_e = np.zeros(len(true_energies), dtype=FTYPE)
    prob_mu = np.zeros_like(prob_e, dtype=FTYPE)
    nutype = 1 if kNuBar==1 else 0
    for (i, (cz, en)) in enumerate(zip(true_coszens, true_energies)):
        for (input_name, nuSQ) in propagators.iteritems():
            chan_prob = nuSQ.EvalFlavor(kFlav, cz, en, nutype)
            if input_name == "nue":
                prob_e[i] = chan_prob
            elif input_name == "numu":
                prob_mu[i] = chan_prob
            else:
                raise ValueError(
                        "Input name '%s' not recognised!"%input_name
                      )

    return prob_e, prob_mu


def _make_EarthAtm_Ye(YeI,
                      YeO,
                      YeM,
                      PREM_file='osc/nuSQuIDS_PREM.dat',
                     ):
    """Return a `nuSQUIDSpy.EarthAtm` object with
    user-defined electron fractions. Note that a
    temporary Earth model file is produced (over-
    written) each time this function is executed.
    Should not be used stand-alone as no sanity checks
    are performed on arguments.
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
                  " fractions: YeI=%s, YeO=%s, YeM=%s"%(YeI, YeO, YeM))
    Ye = np.array([YeI, YeO, YeM])
    earth_radius = 6371.0 # km
    Ye_outer_radius = np.array([1121.5, 3480.0, earth_radius]) # km

    fname_tmp = os.path.join(CACHE_DIR, "nuSQuIDS_PREM_TMP.dat")
    f = from_file(fname=PREM_file, as_array=True)
    for i,(r, _, _) in enumerate(f):
        this_radius = r*earth_radius
        if this_radius <= Ye_outer_radius[0]:
            Ye_new = YeI
        elif this_radius <= Ye_outer_radius[1]:
            Ye_new = YeO
        elif this_radius <= Ye_outer_radius[2]:
            Ye_new = YeM
        f[i][2] = Ye_new

    np.savetxt(fname=fname_tmp, X=f)
    earth_atm = nsq.EarthAtm(fname_tmp)
    return earth_atm


def test_nusquids_osc():
    from pisa.core.binning import OneDimBinning
    # define binning for nuSQuIDS nodes (where master eqn. is solved)
    en_calc_binning = OneDimBinning(name='true_energy',
                                    bin_edges=np.logspace(0.99, 2.01, 40)*ureg.GeV,
                                   )
    cz_calc_binning = OneDimBinning(name='true_coszen',
                                    domain=[-1, 1]*ureg.dimensionless,
                                    is_lin=True,
                                    num_bins=21
                                   )
    # make 2D binning
    binning_2d_calc = en_calc_binning*cz_calc_binning
    # check it has necessary entries
    validate_calc_grid(binning_2d_calc)
    # pad the grid to make sure we can later on evaluate osc. probs.
    # *anywhere* inside the outermost bin edges
    en_calc_grid, cz_calc_grid = compute_binning_constants(binning_2d_calc)
    # set up initial states, get the nuSQuIDS "propagator" instances
    ini_states, props = _get_nusquids_ini_prop(
                            cz_nodes=cz_calc_grid,
                            en_nodes=en_calc_grid,
                            nu_flav_no=3,
                            rel_err=1.0e-5,
                            abs_err=1.0e-5,
                            progress_bar=True
                        )
    # make an Earth model
    PREM_fpath = find_resource(resource='osc/nuSQuIDS_PREM.dat')
    earth_atm = nsq.EarthAtm(PREM_fpath)

    # define some oscillation parameter values
    osc_params = OscParams()
    osc_params.theta23 = np.deg2rad(45.)
    osc_params.theta12 = np.deg2rad(33.)
    osc_params.theta13 = np.deg2rad(8.)
    osc_params.dm21 = 7.5e-5
    osc_params.dm31 = 2.5e-3
    osc_params.eps_ee = 0.
    osc_params.eps_emu = 0.
    osc_params.eps_etau = 0.
    osc_params.eps_mumu = 0.
    osc_params.eps_mutau = 0.
    # evolve the states starting from initial ones
    _setup_physics_and_evolve_states(
        cz_shape=cz_calc_grid.shape[0],
        propagators=props,
        ini_states=ini_states,
        nsq_earth_atm=earth_atm,
        osc_params=osc_params
    )

    # define some points where osc. probs. are to be
    # evaluated
    en_eval = np.logspace(1, 2, 100) * NSQ_CONST.GeV
    cz_eval = np.linspace(-0.95, 0.95, 100)
    # look them up for appearing tau neutrinos
    kFlav = FLAV_INDS['nutau']
    kNuBar = 1
    # collect the transition probabilities from
    # muon and electron neutrinos
    prob_e, prob_mu = _eval_osc_probs(
                           kNuBar=kNuBar,
                           kFlav=kFlav,
                           propagators=props,
                           true_energies=en_eval,
                           true_coszens=cz_eval,
                      )


if __name__ == "__main__":
    test_nusquids_osc()
