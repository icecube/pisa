"""
nuSQuIDS oscillation probabilities with layered Earth model.

This implementation supports several filtering and interpolation techniques designed
to deal with fast oscillations that occur in the presence of eV-scale sterile neutrinos.

It is required that SQuIDS and nuSQuIDS are updated to include the layered Earth model
class `nuSQUIDSLayers` in nuSQuIDS as well as low-pass filtering and range averaging 
methods in SQuIDS.
"""

import math
import numpy as np
from numba import guvectorize
from scipy.interpolate import RectBivariateSpline

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile, line_profile
from pisa.stages.osc.layers import Layers
from pisa.utils.numba_tools import WHERE
from pisa.core.binning import MultiDimBinning

from pisa.utils.resources import find_resource
from pisa import ureg

import nuSQUIDSpy as nsq

__all__ = ["pi_nusquids"]

__author__ = "T. Stuttard, T. Ehrhardt, A. Trettin"


class pi_nusquids(PiStage):
    """
    PISA Pi stage for weighting events due to the effect of neutrino oscillations, using
    nuSQuIDS as the oscillation probability calculator. One specialty here is that we
    have to specify an additional binning to determine where to place nodes for the
    exact calculation. The points where the actual probability is evaluated is
    determined by calc_specs as usual and may be much finer than node_specs or even
    event-wise since the interpolation step is fast.

    Parameters
    ----------

    Uses the standard parameters as required by a PISA pi stage
    (see `pisa/core/pi_stage.py`)
    
    node_specs : MultiDimBinning
        Binning to determine where to place nodes at which the evaluation of interaction
        states occurs. The nodes are places at the _corners_ of the binning to avoid
        extrapolation.

    use_decoherence : bool
        set to true to include neutrino decoherence in the oscillation probability
        calculation

    num_decoherence_gamma : int
        number of decoherence gamma parameters to be considered in the decoherence model
        must be either 1 or 3

    use_nsi : bool
        set to true to include Non-Standard Interactions (NSI) in the oscillation
        probability calculation

    num_neutrinos : int
        Number of neutrino flavors to include. This stage supports 3 or 4 flavors, but
        nuSQuIDS allows up to 6 such that this stage could easily be expanded.
    
    earth_model : str
        Path to Earth model (PREM) file.
    
    detector_depth : quantity (distance)
    
    prop_height : quantity (distance) or str
        Height at which neutrinos are produced. If a quantity is given, the height is
        assumed to be the same for all neutrinos. An alternative is to pass
        `from_container`. In that case, the stage will search the input container for a
        key `prop_height` and take the height from there on a bin-wise or event-wise
        basis depending on `calc_specs`.
    
    prop_height_min : quantity (distance)
        Minimum production height (optional). If this value is passed probabilities are
        averaged between the maximum production height in `prop_height` and this value
        under the assumption of a uniform production height distribution.

    YeI : quantity (dimensionless)
        Inner electron fraction.
    
    YeO : quantity (dimensionless)
        Outer electron fraction.
        
    YeM : quantity (dimensionless)
        Mantle electron fraction.

    rel_err : float
        Relative error of the numerical integration

    abs_err : float
        Absolute error of the numerical integration
    
    prop_lowpass_cutoff : quantity (1/distance)
        Frequency cutoff for fast oscillations applied during numerical integration
        of the interaction state. The frequency is passed as oscillations per distance.
        A reasonable order of magnitude would allow ~100 oscillations over 12000 km.
    
    prop_lowpass_frac : quantity (dimensionless)
        This number determines how harsh the cut-off of the low-pass filter is applied
        during numerical integration is. A value of 0.1 would mean that the filter would
        begin to kick in when 90% of the cutoff frequency is reached and linearly
        decrease oscillation amplitudes until the cutoff is reached.
    
    eval_lowpass_cutoff : quantity (1/distance)
        Same as `prop_lowpass_cutoff`, but applied during evaluation of interpolated 
        states, not during integration.
    
    eval_lowpass_frac : quantity (1/distance)
        Same as `prop_lowpass_frac`, but applied during evaluation of interpolated 
        states, not during integration.
    
    exact_mode : bool
        With this turned on, the probabilities are evaluated using the exact calculation
        for constant densities in every layer without numerical integration. This method
        is much faster than the numerical integration for a node, but you lose the
        option to project out probabilities from interaction picture states. In this
        mode, nuSQuIDS behaves essentially like GLoBES with the same speed trade-off.
        You cannot apply filters in this mode either. Its only recommended use is for
        pseudo-data generation, where you may want an exact event-by-event calculation
        that is allowed to take several minutes.

    params : ParamSet or sequence with which to instantiate a ParamSet.
        Expected params .. ::
            theta12 : quantity (angle)
            theta13 : quantity (angle)
            theta23 : quantity (angle)
            deltam21 : quantity (mass^2)
            deltam31 : quantity (mass^2)
            deltacp : quantity (angle)
        Additional expected params if `num_neutrinos == 4` .. ::
            theta14 : quantity (angle)
            theta24 : quantity (angle)
            deltam41 : quantity (mass^2)
            deltacp14 : quantity (angle)
            deltacp24 : quantity (angle)

    Additional ParamSet params expected when using the `use_decoherence` argument:
        n_energy : quantity (dimensionless)
        * If using `num_decoherence_gamma` == 1:
            gamma : quantity (energy)
        * If using `num_decoherence_gamma` == 3:
            gamma12 : quantity (energy)
            gamma13 : quantity (energy)
            gamma23 : quantity (energy)

    """
    def __init__(
        self,
        data=None,
        params=None,
        earth_model=None,
        detector_depth=None,
        prop_height=None,
        prop_height_min=None,
        YeI=None,
        YeO=None,
        YeM=None,
        rel_err=None,
        abs_err=None,
        prop_lowpass_cutoff=None,
        prop_lowpass_frac=None,
        eval_lowpass_cutoff=None,
        eval_lowpass_frac=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        input_specs=None,
        node_specs=None,
        calc_specs=None,
        output_specs=None,
        use_decoherence=False,
        num_decoherence_gamma=1,
        use_nsi=False,
        num_neutrinos=3,
        exact_mode=False,
    ):

        if use_nsi:
            raise NotImplementedError("NSI not implemented")
        if use_decoherence:
            raise NotImplementedError("Decoherence not implemented")
        if type(prop_height) is not ureg.Quantity:
            raise NotImplementedError("Getting propagation heights from containers is "
                "not yet implemented")
        self.num_neutrinos = int(num_neutrinos)
        assert self.num_neutrinos < 5, "currently only supports up to 4 flavor oscillations"
        self.use_nsi = use_nsi
        self.use_decoherence = use_decoherence
        self.num_decoherence_gamma = num_decoherence_gamma
        self.node_specs = node_specs

        self.earth_model = earth_model
        self.YeI = YeI.m_as("dimensionless")
        self.YeO = YeO.m_as("dimensionless")
        self.YeM = YeM.m_as("dimensionless")
        self.detector_depth = detector_depth.m_as("km")
        self.prop_height = prop_height.m_as("km")
        self.avg_height = False
        self.prop_height_min = None
        if prop_height_min is not None:  # this is optional
            self.prop_height_min = prop_height_min.m_as("km")
            self.avg_height = True
        
        self.layers = None
        
        self.rel_err = rel_err.m_as("dimensionless") if rel_err is not None else 1.0e-10
        self.abs_err = abs_err.m_as("dimensionless") if abs_err is not None else 1.0e-10
        self.prop_lowpass_cutoff = (prop_lowpass_cutoff.m_as("1/km")
                                    if prop_lowpass_cutoff is not None else 0.)
        self.prop_lowpass_frac = (prop_lowpass_frac.m_as("dimensionless")
                                  if prop_lowpass_frac is not None else 0.)
        self.eval_lowpass_cutoff = (eval_lowpass_cutoff.m_as("1/km")
                                    if eval_lowpass_cutoff is not None else 0.)
        self.eval_lowpass_frac = (eval_lowpass_frac.m_as("dimensionless")
                                  if eval_lowpass_frac is not None else 0.)
        
        if self.prop_lowpass_frac > 1. or self.eval_lowpass_frac > 1.:
            raise ValueError("lowpass filter fraction cannot be greater than one")
        
        if self.prop_lowpass_frac < 0. or self.eval_lowpass_frac < 0.:
            raise ValueError("lowpass filter fraction cannot be smaller than zero")

        self.nus_layer = None
        self.nus_layerbar = None
        
        # Define standard params
        expected_params = [
            "theta12",
            "theta13",
            "theta23",
            "deltam21",
            "deltam31",
            "deltacp",
        ]

        # Add decoherence parameters
        assert self.num_decoherence_gamma in [1, 3], ("Must choose either 1 or 3 "
            "decoherence gamma parameters"
        )
        if self.use_decoherence:
            if self.num_decoherence_gamma == 1:
                expected_params.extend(["gamma"])
            elif self.num_decoherence_gamma == 3:
                expected_params.extend(["gamma21",
                                        "gamma31",
                                        "gamma32"])
            expected_params.extend(["n_energy"])
        
        # We may want to reparametrize this with the difference between deltacp14 and
        # deltacp24, as the absolute value seems to play a small role (see
        # https://arxiv.org/pdf/2010.06321.pdf)
        if self.num_neutrinos == 4:
            expected_params.extend([
                "theta14",
                "theta24",
                "deltam41",
                "deltacp14",
                "deltacp24",
            ])

        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_apply_keys = ("weights")
        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ("prob_e", "prob_mu")
        # what keys are added or altered for the outputs during apply
        output_apply_keys = ("weights")
        
        # init base class
        super().__init__(
            data=data,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            input_specs=input_specs,
            calc_specs=calc_specs,
            output_specs=output_specs,
            input_apply_keys=input_apply_keys,
            output_calc_keys=output_calc_keys,
            output_apply_keys=output_apply_keys,
        )
        
        # This is special: We have an additional "binning" to account for. It is in
        # principle possible to work in event mode even for the nodes, which would mean
        # that the full oscillation problem is solved for all events individually.
        # Together with the constant oscillation mode, this can be used to calculate
        # probabilities in exact mode in a time that is reasonable at least for
        # generating pseudodata.
        if isinstance(self.node_specs, MultiDimBinning):
            self.node_mode = "binned"
        elif self.node_specs == "events":
            self.node_mode = "events"
        elif self.node_specs is None:
            self.node_mode = None
        else:
            raise ValueError("Cannot understand `node_specs` %s" % calc_specs)

        assert not (self.use_nsi and self.use_decoherence), ("NSI and decoherence not "
            "suported together, must use one or the other")

        assert self.input_mode is not None
        assert self.calc_mode is not None
        assert self.output_mode is not None
        
        self.exact_mode = exact_mode
            
        if exact_mode:
            # No interpolation is happening in exact mode so any passed node_specs
            # will be ignored. Probabilities are calculated at calc_specs.
            if self.node_mode is not None:
                logging.warn("nuSQuIDS is configured in exact mode, the passed "
                    f"`node_specs`\n({self.node_specs})\n will be ignored!")
            if self.prop_lowpass_cutoff > 0 or self.eval_lowpass_cutoff > 0:
                logging.warn("nuSQuIDS is configured in exact mode, low-pass filters "
                    "will be ignored")
        else:
            assert self.calc_mode == self.node_mode or self.calc_mode == "events"

        self.e_node_specs = None
        self.e_mesh = None
        self.coszen_node_specs = None
        self.cosz_mesh = None
    
    def set_osc_parameters(self, nus_layer):
        # nuSQuIDS uses zero-index for mixing angles
        nus_layer.Set_MixingAngle(0, 1, self.params.theta12.value.m_as("rad"))
        nus_layer.Set_MixingAngle(0, 2, self.params.theta13.value.m_as("rad"))
        nus_layer.Set_MixingAngle(1, 2, self.params.theta23.value.m_as("rad"))
        
        # mass differences in nuSQuIDS are always w.r.t. m_1
        nus_layer.Set_SquareMassDifference(1, self.params.deltam21.value.m_as("eV**2"))
        nus_layer.Set_SquareMassDifference(2, self.params.deltam31.value.m_as("eV**2"))
    
        nus_layer.Set_CPPhase(0, 2, self.params.deltacp.value.m_as("rad"))
        
        if self.num_neutrinos == 3: return
        
        nus_layer.Set_MixingAngle(0, 3, self.params.theta14.value.m_as("rad"))
        nus_layer.Set_MixingAngle(1, 3, self.params.theta24.value.m_as("rad"))
        nus_layer.Set_SquareMassDifference(3, self.params.deltam41.value.m_as("eV**2"))
        nus_layer.Set_CPPhase(0, 3, self.params.deltacp14.value.m_as("rad"))
        nus_layer.Set_CPPhase(1, 3, self.params.deltacp24.value.m_as("rad"))
        # TODO: Implement NSI, decoherence

    def apply_prop_settings(self, nus_layer):
        nsq_units = nsq.Const()
        nus_layer.Set_rel_error(self.rel_err)
        nus_layer.Set_abs_error(self.abs_err)
        nus_layer.Set_EvolLowPassCutoff(self.prop_lowpass_cutoff / nsq_units.km)
        # The ramp of the low-pass filter starts to drop at (cutoff - scale)
        scale = self.prop_lowpass_frac * self.prop_lowpass_cutoff / nsq_units.km
        nus_layer.Set_EvolLowPassScale(scale)
        nus_layer.Set_AllowConstantDensityOscillationOnlyEvolution(self.exact_mode)

    def setup_function(self):

        earth_model = find_resource(self.earth_model)
        prop_height = self.prop_height
        detector_depth = self.detector_depth
        self.layers = Layers(earth_model, detector_depth, prop_height)
        self.layers.setElecFrac(self.YeI, self.YeO, self.YeM)
        
        nsq_units = nsq.Const()  # natural units for nusquids
        # Because we don't want to extrapolate, we check that all points at which we
        # want to evaluate probabilities are fully contained within the node specs. This
        # is of course not necessary in events mode.
        if self.node_mode == "binned" and not self.exact_mode:
            logging.debug("setting up nuSQuIDS nodes in binned mode")
            # we can prepare the calculator like this only in binned mode, see 
            # compute_function for node_mode == "events"
            self.data.data_specs = self.calc_specs
            for container in self.data:
                for var in ["true_coszen", "true_energy"]:
                    upper_bound = np.max(self.node_specs[var].bin_edges)
                    lower_bound = np.min(self.node_specs[var].bin_edges)
                    err_msg = (
                        "The outer edges of the node_specs must encompass "
                        "the entire range of calc_specs to avoid extrapolation"
                    )
                    assert np.all(container[var].get(WHERE) <= upper_bound), err_msg
                    assert np.all(container[var].get(WHERE) >= lower_bound), err_msg

            # Layers in nuSQuIDS are special: We need all the individual distances and
            # densities for the nodes to solve the interaction picture states, but on
            # the final calculation grid (or events) we only need the *total* traversed
            # distance. Because we are placing nodes at the bin edges rather than the
            # bin middle, this doesn't really fit with how containers store data, so we
            # are making arrays as variables that never go into the container.
            
            # These are stored because we need them later during interpolation
            self.coszen_node_specs = self.node_specs["true_coszen"].bin_edges.m_as("dimensionless")
            self.e_node_specs = self.node_specs["true_energy"].bin_edges.m_as("GeV")
            logging.debug(
                f"Setting up nodes at\n"
                f"cos_zen = \n{self.coszen_node_specs}\n"
                f"energy = \n{self.e_node_specs}\n"
            )
            # things are getting a bit meshy from here...
            self.e_mesh, self.cosz_mesh = np.meshgrid(self.e_node_specs, self.coszen_node_specs)
            e_nodes = self.e_mesh.ravel()
            coszen_nodes = self.cosz_mesh.ravel()
        
            # The lines below should not be necessary because we will always get at
            # least two numbers from the bin edges. However, if either energy or coszen
            # somehow was just a scalar, we would need to broadcast it out to the same
            # size. Keeping the code in here in case you want to use the stage in 1D.
            # convert lists to ndarrays and scalars to ndarrays with length 1
            e_nodes = np.atleast_1d(e_nodes)
            coszen_nodes = np.atleast_1d(coszen_nodes)
            # broadcast against each other and make a copy
            # (see https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html)
            e_nodes, coszen_nodes = [np.array(a) for a in np.broadcast_arrays(e_nodes, coszen_nodes)]
        
            assert len(e_nodes) == len(coszen_nodes)
            assert coszen_nodes.ndim == 1
            assert e_nodes.ndim == 1
            
            self.layers.calcLayers(coszen_nodes)
            distances = np.reshape(self.layers.distance,
                                   (len(e_nodes), self.layers.max_layers))
            densities = np.reshape(self.layers.density,
                                   (len(e_nodes), self.layers.max_layers))
            # electron fraction is already included by multiplying the densities with
            # them in the Layers module, so we pass 1. to nuSQuIDS (unless energies are
            # very high, this should be equivalent).
            ye = np.broadcast_to(np.array([1.]), (len(e_nodes), self.layers.max_layers))
            self.nus_layer = nsq.nuSQUIDSLayers(
                distances * nsq_units.km,
                densities,
                ye,
                e_nodes * nsq_units.GeV,
                self.num_neutrinos,
                nsq.NeutrinoType.both,
            )
            self.apply_prop_settings(self.nus_layer)
        
        # Now that we have our nusquids calculator set up on the node grid, we make 
        # container output space for the probability output which may be on a finer grid
        # than the nodes or even working in events mode.
        self.data.data_specs = self.calc_specs

        # --- calculate the layers ---
        if self.calc_mode == "binned":
            # as layers don't care about flavour
            self.data.link_containers("nu", ["nue_cc", "numu_cc", "nutau_cc",
                                             "nue_nc", "numu_nc", "nutau_nc",
                                             "nuebar_cc", "numubar_cc", "nutaubar_cc",
                                             "nuebar_nc", "numubar_nc", "nutaubar_nc"])
        # calculate the distance difference between minimum and maximum production
        # height, if applicable
        if self.avg_height:
            layers_min = Layers(earth_model, detector_depth, self.prop_height_min)
            layers_min.setElecFrac(self.YeI, self.YeO, self.YeM)
        for container in self.data:
            self.layers.calcLayers(container["true_coszen"].get("host"))
            distances = self.layers.distance.reshape((container.size, self.layers.max_layers))
            tot_distances = np.sum(distances, axis=1)
            if self.avg_height:
                layers_min.calcLayers(container["true_coszen"].get("host"))
                dists_min = layers_min.distance.reshape((container.size, self.layers.max_layers))
                min_tot_dists = np.sum(dists_min, axis=1)
                # nuSQuIDS assumes the original distance is the longest distance and 
                # the averaging range is the difference between the minimum and maximum
                # distance.
                avg_ranges = tot_distances - min_tot_dists
                assert np.all(avg_ranges > 0)
            if self.node_mode == "binned" and not self.exact_mode:
                # To project out probabilities we only need the *total* distance
                container["tot_distances"] = tot_distances
                # for the binned node_mode we already calculated layers above
                if self.avg_height:
                    container["avg_ranges"] = avg_ranges
            elif self.node_mode == "events" or self.exact_mode:
                # in any other mode (events or exact) we store all densities and 
                # distances in the container in calc_specs
                densities = self.layers.density.reshape((container.size, self.layers.max_layers))
                container["densities"] = densities
                container["distances"] = distances
        
        self.data.unlink_containers()
        
        if self.calc_mode == "binned":
            self.data.link_containers("nue", ["nue_cc", "nue_nc"])
            self.data.link_containers("numu", ["numu_cc", "numu_nc"])
            self.data.link_containers("nutau", ["nutau_cc", "nutau_nc"])
            self.data.link_containers("nuebar", ["nuebar_cc", "nuebar_nc"])
            self.data.link_containers("numubar", ["numubar_cc", "numubar_nc"])
            self.data.link_containers("nutaubar", ["nutaubar_cc", "nutaubar_nc"])

        # setup more empty arrays
        for container in self.data:
            container["prob_e"] = np.empty((container.size), dtype=FTYPE)
            container["prob_mu"] = np.empty((container.size), dtype=FTYPE)
        self.data.unlink_containers()
        
        if self.exact_mode: return
        
        # --- containers for interpolated states ---
        # This is not needed in exact mode
        if self.calc_mode == "binned":
            self.data.link_containers("nu", ["nue_cc", "numu_cc", "nutau_cc",
                                             "nue_nc", "numu_nc", "nutau_nc"])
            self.data.link_containers("nubar", ["nuebar_cc", "numubar_cc", "nutaubar_cc",
                                                "nuebar_nc", "numubar_nc", "nutaubar_nc"])
        for container in self.data:
            container["interp_states_e"] = np.empty(
                (container.size, self.num_neutrinos**2),
                dtype=FTYPE,
            )
            container["interp_states_mu"] = np.empty(
                (container.size, self.num_neutrinos**2),
                dtype=FTYPE,
            )
        self.data.unlink_containers()
    
    @line_profile
    def calc_node_probs(self, nus_layer, flav_in, flav_out, n_nodes):
        """
        Evaluate oscillation probabilities at nodes. This does not require any
        interpolation.
        """
        ini_state = np.array([0] * self.num_neutrinos)
        ini_state[flav_in] = 1
        nus_layer.Set_initial_state(ini_state, nsq.Basis.flavor)
        nus_layer.EvolveState()
        prob_nodes = nus_layer.EvalFlavorAtNodes(flav_out)
        return prob_nodes
    
    def calc_interpolated_states(self, evolved_states, e_out, cosz_out):
        """
        Calculate interpolated states at the energies and zenith angles requested.
        """
        nsq_units = nsq.Const()
        interp_states = np.zeros((e_out.size, evolved_states.shape[1]))
        
        assert np.all(e_out <= np.max(self.e_node_specs * nsq_units.GeV))
        assert np.all(e_out >= np.min(self.e_node_specs * nsq_units.GeV))
        assert np.all(cosz_out <= np.max(self.coszen_node_specs))
        assert np.all(cosz_out >= np.min(self.coszen_node_specs))

        for i in range(evolved_states.shape[1]):
            z = evolved_states[:, i].reshape(self.e_mesh.shape).T
            assert np.all(np.isfinite(z))
            # RectBivariateSpline takes in the 1D node position and assumes that they 
            # are on a mesh.
            f = RectBivariateSpline(
                np.log10(self.e_node_specs * nsq_units.GeV),
                self.coszen_node_specs,
                z,
                kx=2,
                ky=2,
            )
            interp_states[..., i] = f(np.log10(e_out), cosz_out, grid=False)
        return interp_states

    def calc_probs_interp(self, flav_out, nubar, interp_states, out_distances,
                          e_out, avg_ranges=0):
        """
        Project out probabilities from interpolated interaction picture states.
        """
        nsq_units = nsq.Const()

        prob_interp = np.zeros(e_out.size)
        scale = self.eval_lowpass_frac * self.eval_lowpass_cutoff / nsq_units.km
        prob_interp = self.nus_layer.EvalWithState(
            flav_out,
            out_distances,
            e_out,
            interp_states,
            avr_scale=0.,
            rho=int(nubar),
            lowpass_cutoff=self.eval_lowpass_cutoff / nsq_units.km,
            lowpass_scale=scale,
            t_range=avg_ranges
        )
        return prob_interp
    
    def compute_function_no_interpolation(self):
        """
        Version of the compute function that does not use any interpolation between
        nodes.
        """
        nsq_units = nsq.Const()
        # it is possible to work in binned calc mode while being in exact mode
        if self.calc_mode == "binned":
            self.data.link_containers("nue", ["nue_cc", "nue_nc"])
            self.data.link_containers("numu", ["numu_cc", "numu_nc"])
            self.data.link_containers("nutau", ["nutau_cc", "nutau_nc"])
            self.data.link_containers("nuebar", ["nuebar_cc", "nuebar_nc"])
            self.data.link_containers("numubar", ["numubar_cc", "numubar_nc"])
            self.data.link_containers("nutaubar", ["nutaubar_cc", "nutaubar_nc"])
        for container in self.data:
            nubar = container["nubar"] < 0
            flav = container["flav"]
            # electron fraction is already included by multiplying the densities
            # with them in the Layers module, so we pass 1. to nuSQuIDS (unless
            # energies are very high, this should be equivalent).
            ye = np.broadcast_to(np.array([1.]),
                                 (container.size, self.layers.max_layers))
            nus_layer = nsq.nuSQUIDSLayers(
                container["distances"].get(WHERE) * nsq_units.km,
                container["densities"].get(WHERE),
                ye,
                container["true_energy"].get(WHERE) * nsq_units.GeV,
                self.num_neutrinos,
                nsq.NeutrinoType.antineutrino if nubar else nsq.NeutrinoType.neutrino,
            )
            self.apply_prop_settings(nus_layer)
            self.set_osc_parameters(nus_layer)
            container["prob_e"] = self.calc_node_probs(nus_layer, 0, flav,
                                                       container.size)
            container["prob_mu"] = self.calc_node_probs(nus_layer, 1, flav,
                                                        container.size)
        
            container["prob_e"].mark_changed(WHERE)
            container["prob_mu"].mark_changed(WHERE)
        self.data.unlink_containers()
    
    @line_profile
    def compute_function_interpolated(self):
        """
        Version of the compute function that does use interpolation between nodes.
        """
        nsq_units = nsq.Const()
        # We need to make two evolutions, one for numu and the other for nue.
        # These produce neutrino and antineutrino states at the same time thanks to
        # the "both" neutrino mode of nuSQuIDS.
        self.apply_prop_settings(self.nus_layer)
        self.set_osc_parameters(self.nus_layer)

        ini_state_nue = np.array([1, 0] + [0] * (self.num_neutrinos - 2))
        ini_state_numu = np.array([0, 1] + [0] * (self.num_neutrinos - 2))
        
        self.nus_layer.Set_initial_state(ini_state_nue, nsq.Basis.flavor)
        self.nus_layer.EvolveState()
        evolved_states_nue = self.nus_layer.GetStates(0)
        evolved_states_nuebar = self.nus_layer.GetStates(1)
        
        self.nus_layer.Set_initial_state(ini_state_numu, nsq.Basis.flavor)
        self.nus_layer.EvolveState()
        evolved_states_numu = self.nus_layer.GetStates(0)
        evolved_states_numubar = self.nus_layer.GetStates(1)

        
        # Now comes the step where we interpolate the interaction picture states
        # and project out oscillation probabilities. This can be done in either events
        # or binned mode.
        if self.calc_mode == "binned":
            self.data.link_containers("nu", ["nue_cc", "numu_cc", "nutau_cc",
                                             "nue_nc", "numu_nc", "nutau_nc"])
            self.data.link_containers("nubar", ["nuebar_cc", "numubar_cc", "nutaubar_cc",
                                                "nuebar_nc", "numubar_nc", "nutaubar_nc"])
        for container in self.data:
            nubar = container["nubar"] < 0
            container["interp_states_e"] = self.calc_interpolated_states(
                evolved_states_nuebar if nubar else evolved_states_nue,
                container["true_energy"].get(WHERE) * nsq_units.GeV,
                container["true_coszen"].get(WHERE)
            )
            container["interp_states_mu"] = self.calc_interpolated_states(
                evolved_states_numubar if nubar else evolved_states_numu,
                container["true_energy"].get(WHERE) * nsq_units.GeV,
                container["true_coszen"].get(WHERE)
            )
        self.data.unlink_containers()
        
        if self.calc_mode == "binned":
            self.data.link_containers("nue", ["nue_cc", "nue_nc"])
            self.data.link_containers("numu", ["numu_cc", "numu_nc"])
            self.data.link_containers("nutau", ["nutau_cc", "nutau_nc"])
            self.data.link_containers("nuebar", ["nuebar_cc", "nuebar_nc"])
            self.data.link_containers("numubar", ["numubar_cc", "numubar_nc"])
            self.data.link_containers("nutaubar", ["nutaubar_cc", "nutaubar_nc"])
        
        for container in self.data:
            nubar = container["nubar"] < 0
            flav_out = container["flav"]
            for flav_in in ["e", "mu"]:
                container["prob_"+flav_in] = self.calc_probs_interp(
                    flav_out,
                    nubar,
                    container["interp_states_"+flav_in].get(WHERE),
                    container["tot_distances"].get(WHERE) * nsq_units.km,
                    container["true_energy"].get(WHERE) * nsq_units.GeV,
                    container["avg_ranges"].get(WHERE) * nsq_units.km if self.avg_height else 0.
                )
            container["prob_e"].mark_changed(WHERE)
            container["prob_mu"].mark_changed(WHERE)
        self.data.unlink_containers()
    
    @profile
    def compute_function(self):
        if self.node_mode == "events" or self.exact_mode:
            self.compute_function_no_interpolation()
        else:
            self.compute_function_interpolated()

    def apply_function(self):

        # update the outputted weights
        for container in self.data:
            apply_probs(container["nu_flux"].get(WHERE),
                        container["prob_e"].get(WHERE),
                        container["prob_mu"].get(WHERE),
                        out=container["weights"].get(WHERE))
            container["weights"].mark_changed(WHERE)

# vectorized function to apply (flux * prob)
# must be outside class
if FTYPE == np.float64:
    signature = '(f8[:], f8, f8, f8[:])'
else:
    signature = '(f4[:], f4, f4, f4[:])'
@guvectorize([signature], '(d),(),()->()', target=TARGET)
def apply_probs(flux, prob_e, prob_mu, out):
    out[0] *= (flux[0] * prob_e) + (flux[1] * prob_mu)
