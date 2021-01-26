#! /usr/bin/env python

# Calculates the differentials for Jacobian matrix wrt. particle
# production uncertainty.
#
# Author: Anatoli Fedynitch
# August 2017
#
# Update for IceCube oscillation analyss by Ida Storehaug & Tom Stuttard (2019)
# See https://github.com/afedynitch/MCEq/blob/master/examples/KPi_demonstration.ipynb for a related example
#

from __future__ import print_function

import os, sys, gzip, bz2, collections
import numpy as np
import pickle

from scipy.interpolate import RectBivariateSpline

from MCEq.core import MCEqRun
from MCEq.misc import normalize_hadronic_model_name
from mceq_config import config
import crflux.models as crf

# Global Barr parameter table
# format (x_min, x_max, E_min, E_max) | x is x_lab= E_pi/E, E projectile-air interaction energy
barr = {
    'a': [(0.0, 0.5, 0.00, 8.0)],
    'b1': [(0.5, 1.0, 0.00, 8.0)],
    'b2': [(0.6, 1.0, 8.00, 15.0)],
    'c': [(0.2, 0.6, 8.00, 15.0)],
    'd1': [(0.0, 0.2, 8.00, 15.0)],
    'd2': [(0.0, 0.1, 15.0, 30.0)],
    'd3': [(0.1, 0.2, 15.0, 30.0)],
    'e': [(0.2, 0.6, 15.0, 30.0)],
    'f': [(0.6, 1.0, 15.0, 30.0)],
    'g': [(0.0, 0.1, 30.0, 1e11)],
    'h1': [(0.1, 1.0, 30.0, 500.)],
    'h2': [(0.1, 1.0, 500.0, 1e11)],
    'i': [(0.1, 1.0, 500.0, 1e11)],
    'w1': [(0.0, 1.0, 0.00, 8.0)],
    'w2': [(0.0, 1.0, 8.00, 15.0)],
    'w3': [(0.0, 0.1, 15.0, 30.0)],
    'w4': [(0.1, 0.2, 15.0, 30.0)],
    'w5': [(0.0, 0.1, 30.0, 500.)],
    'w6': [(0.0, 0.1, 500., 1e11)],
    'x': [(0.2, 1.0, 15.0, 30.0)],
    'y1': [(0.1, 1.0, 30.0, 500.)],
    'y2': [(0.1, 1.0, 500., 1e11)],
    'z': [(0.1, 1.0, 500., 1e11)],
    'ch_a': [(0.0, 0.1, 0., 1e11)],
    'ch_b': [(0.1, 1.0, 0., 1e11)],
    'ch_e': [(0.1, 1.0, 800., 1e11)],
}


def barr_unc(xmat, egrid, pname, value):
    """Implementation of hadronic uncertainties as in Barr et al. PRD 74 094009 (2006)

    The names of parameters are explained in Fig. 2 and Fig. 3 in the paper."""

    # Energy dependence
    u = lambda E, val, ethr, maxerr, expected_err: val*min(
        maxerr/expected_err,
        0.122/expected_err*np.log10(E / ethr)) if E > ethr else 0.

    modmat = np.ones_like(xmat)
    modmat[np.tril_indices(xmat.shape[0], -1)] = 0.

    for minx, maxx, mine, maxe in barr[pname]:
        eidcs = np.where((mine < egrid) & (egrid <= maxe))[0]
        for eidx in eidcs:
            xsel = np.where((xmat[:eidx + 1, eidx] >= minx) &
                            (xmat[:eidx + 1, eidx] <= maxx))[0]
            if not np.any(xsel):
                continue
            if pname in ['i', 'z']:
                modmat[xsel, eidx] += u(egrid[eidx], value, 500., 0.5, 0.122)
            elif pname in ['ch_e']:
                modmat[xsel, eidx] += u(egrid[eidx], value, 800., 0.3, 0.25)
            else:
                modmat[xsel, eidx] += value

    return modmat


def compute_abs_derivatives(mceq_run, pid, barr_param, zenith_list):
    mceq_run.unset_mod_pprod(dont_fill=False)

    barr_pars = [p for p in barr if p.startswith(barr_param) and 'ch' not in p]
    print('Parameters corresponding to selection', barr_pars)
    dim_res = len(zenith_list), etr.shape[0]
    gs = mceq_run.get_solution
    unit=1e4

    # Solving nominal MCEq flux
    numu, anumu, nue, anue = (np.zeros(dim_res), np.zeros(dim_res),
                              np.zeros(dim_res), np.zeros(dim_res))

    for iz, zen_deg in enumerate(zenith_list):
        mceq_run.set_theta_deg(zen_deg)
        mceq_run.solve()
        numu[iz] = gs('total_numu', 0)[tr]*unit
        anumu[iz] = gs('total_antinumu', 0)[tr]*unit
        nue[iz] = gs('total_nue', 0)[tr]*unit
        anue[iz] = gs('total_antinue', 0)[tr]*unit

    # Solving for plus one sigma
    mceq_run.unset_mod_pprod(dont_fill=True)
    for p in barr_pars:
        mceq_run.set_mod_pprod(primary_particle, pid, barr_unc, (p, delta))

    mceq_run.regenerate_matrices(skip_decay_matrix=True)

    numu_up, anumu_up, nue_up, anue_up = (np.zeros(dim_res), np.zeros(dim_res),
                                          np.zeros(dim_res), np.zeros(dim_res))
    for iz, zen_deg in enumerate(zenith_list):
        mceq_run.set_theta_deg(zen_deg)
        mceq_run.solve()
        numu_up[iz] = gs('total_numu', 0)[tr]*unit
        anumu_up[iz] = gs('total_antinumu', 0)[tr]*unit
        nue_up[iz] = gs('total_nue', 0)[tr]*unit
        anue_up[iz] = gs('total_antinue', 0)[tr]*unit

    # Solving for minus one sigma
    mceq_run.unset_mod_pprod(dont_fill=True)
    for p in barr_pars:
        mceq_run.set_mod_pprod(primary_particle, pid, barr_unc, (p, -delta))

    mceq_run.regenerate_matrices(skip_decay_matrix=True)

    numu_down, anumu_down, nue_down, anue_down = (np.zeros(dim_res),
                                                  np.zeros(dim_res),
                                                  np.zeros(dim_res),
                                                  np.zeros(dim_res))
    for iz, zen_deg in enumerate(zenith_list):
        mceq_run.set_theta_deg(zen_deg)
        mceq_run.solve()
        numu_down[iz] = gs('total_numu', 0)[tr]*unit
        anumu_down[iz] = gs('total_antinumu', 0)[tr]*unit
        nue_down[iz] = gs('total_nue', 0)[tr]*unit
        anue_down[iz] = gs('total_antinue', 0)[tr]*unit

    # calculating derivatives
    fd_derivative = lambda up, down: (up - down) / (2. * delta)

    dnumu = fd_derivative(numu_up, numu_down)
    danumu = fd_derivative(anumu_up, anumu_down)
    dnue = fd_derivative(nue_up, nue_down)
    danue = fd_derivative(anue_up, anue_down)

    result = collections.OrderedDict()
    result_type = ["numu", "dnumu", "numubar", "dnumubar", "nue", "dnue", "nuebar", "dnuebar"]

    for dist, sp in zip([numu, dnumu, anumu, danumu, nue, dnue, anue, danue], result_type):
        result[sp] = RectBivariateSpline(cos_theta, np.log(etr), dist)

    return result

if __name__ == '__main__':

    # Get command line args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( "-i", "--interaction-model", type=str, required=False, default="sibyll23c", help="Hadronic interaction model" )
    parser.add_argument( "-c", "--cosmic-ray-model", type=str, required=False, default="GlobalSplineFitBeta", help="Primary cosmic ray spectrum model" )
    parser.add_argument( "-o", "--output-dir", type=str, required=False, default="", help="Output directory" )
    args = parser.parse_args()

    # Get interaction model
    interaction_model = normalize_hadronic_model_name(args.interaction_model)

    # Get primary cosmic ray spectrum model
    assert hasattr(crf, args.cosmic_ray_model), "Unknown primary cosmic ray spectrum model"
    CRModel = getattr(crf, args.cosmic_ray_model) # Gettting class (NOT instantiating)
    assert issubclass(CRModel, crf.PrimaryFlux), "Unknown primary cosmic ray spectrum model"

    # define CR model parameters
    if args.cosmic_ray_model=="HillasGaisser2012":
        CR_vers = "H3a"
    elif args.cosmic_ray_model=="GaisserStanevTilav":
        CR_vers = "4-gen"
    else:
        CR_vers = None

    mceq_run = MCEqRun(
        #provide the string of the interaction model
        interaction_model=interaction_model,
        #primary cosmic ray flux model
        #support a tuple (primary model class (not instance!), arguments)
        primary_model=(CRModel, CR_vers),
        # Zenith angle in degrees. 0=vertical, 90=horizontal
        theta_deg=0.,
        #GPU device id
        **config)

    # Some global settings. One can play around with them, but there
    # is currently no good reason why

    # Primary proton projectile (neutron is included automatically
    # vie isospin symmetries)
    primary_particle = 2212
    # The parameter delta for finite differences computation
    delta = 0.001
    # Energy grid will be truncated below this value (saves some
    # memory and interpolation speed, but not really needed, I think)
    E_tr = 1e5

    # Set density/atmosphere model:
    # For a yearly average of the global atmosphere, the US Std.
    # atmosphere is sufficiently accurate. It would be wrong to
    # choose here anything related to South Pole, since stuff
    # comes from/from below horizon.
    atm_model = "CORSIKA" #TODO Try varying this...
    atm_model_config = ('BK_USStd', None)
    mceq_run.set_density_model((atm_model,atm_model_config))

    # Define equidistant grid in cos(theta) for 2D interpolation
    # (Can be increased to 20 after debugging is done)
    # The flux without propagation effects and atmospheric variations
    # is up/down symmetric.
    cos_theta = np.linspace(0, 1, 21)
    angles = np.arccos(cos_theta) / np.pi * 180.

    # Report settings
    print("Running with :")
    print("  Interaction model : %s" % interaction_model)
    print("  Primary cosmic ray spectrum model : %s" % args.cosmic_ray_model)

    # Some technical shortcuts
    solution = {}
    tr = np.where(mceq_run.e_grid < E_tr)
    etr = mceq_run.e_grid[tr]


    # Barr variables related to pions
    barr_pivars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    for bp in barr_pivars:
        solution[bp + '+'] = compute_abs_derivatives(mceq_run, 211, bp, angles)
        solution[bp + '-'] = compute_abs_derivatives(mceq_run, -211, bp, angles)

    # Barr variables related to kaons
    barr_kvars = ['w', 'x',  'y', 'z']

    for bp in barr_kvars:
        solution[bp + '+'] = compute_abs_derivatives(mceq_run, 321, bp, angles)
        solution[bp + '-'] = compute_abs_derivatives(mceq_run, -321, bp, angles)

    # Store some metadata
    solution["metadata"] = {
        "primary_particle" : primary_particle,
        "cosmic_ray_model" : args.cosmic_ray_model,
        "interaction_model" : interaction_model,
        "barr_variables": barr_pivars+barr_kvars,
        "atmospheric_model": atm_model
        #TODO atmosphere
    }

    # Write th output file
    output_file = 'MCEq_flux_gradient_splines_{primary_particle}_{cosmic_ray_model}_{interaction_model}.pckl.bz2'.format( #TODO atm model, prod height, etc
        cosmic_ray_model=args.cosmic_ray_model,
        interaction_model=interaction_model,
        primary_particle=primary_particle,
    )
    output_file = os.path.join( args.output_dir, output_file )

    pickle.dump(
        solution,
        bz2.BZ2File(output_file, 'wb'),
        protocol=-1
    )

    #TODO store settings used in pickle file too (and make name more explicit)

    print("\nFinished : Output file is %s\n" % output_file)
