"""
PISA stage to apply ultrasurface fits from discrete systematics parameterizations
"""

import collections
import os

import numpy as np
from numba import njit

from pisa import FTYPE, CACHE_DIR
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.resources import find_resource

__all__ = [
    "get_us_grouping_from_container_name",
    "ultrasurfaces",
    "init_test"
]

__author__ = "A. Trettin, L. Fischer, T. Ehrhardt"

__license__ = """Copyright (c) 2014-2025, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


def get_us_grouping_from_container_name(name, groupings_set):
    """
    Find the event grouping to which a given event type (of a container)
    belonged during the ultrasurface classification/fitting procedure.
    This function therefore connects this stage with those separate
    scripts. It assumes that groups of CC events have the naming format
    "numu_numubar_cc", and that there is one grouping of all NC events
    (e.g. "nu_nc", fine as long as it ends with "nc").

    Parameters
    ----------
    name : str
        name of a single event type
    groupings_set : set of str
        set of grouping names (assumes e.g. "numu_numubar_cc")

    Returns
    -------
    associated_grouping : str
        the grouping among `groupings_set` which is found to contain the
        input event type `name`

    """
    # require exactly one NC grouping
    assert len([group for group in groupings_set if group.lower().endswith("nc")]) == 1
    # split e.g. numu_cc -> "numu", "cc"
    flav, int_type = name.lower().split("_")
    associated_grouping = None
    for group in groupings_set:
        if int_type == "cc":
            # Detection when e.g. "numu_" is part of group name and if it ends
            # with "cc"
            if (f"{flav}_" in group.lower() and
                group.lower().endswith(int_type)):
                associated_grouping = group
                break
        elif int_type == "nc":
            # Detection if group name ends with "nc"
            if group.lower().endswith(int_type):
                associated_grouping = group
                break
    if associated_grouping is None:
        raise ValueError(
            f"Unable to find event grouping associated with {name}"
            f" among the groups {groupings_set}!"
        )
    return associated_grouping

class ultrasurfaces(Stage):  # pylint: disable=invalid-name
    """
    Service to apply ultrasurface parameterisation stored in a feather file.

    Parameters
    ----------
    fit_results_file : str
        Path to .feather file containing all nominal events with gradients.
    nominal_points : str or dict
        Dictionary (or str that can be evaluated thereto) of the form
        {'parameter_name': <nominal value>} containing the nominal value for each
        parameter that was used to fit the gradients with.
    varnames : list of str
        List of variables to match the pisa events to the pre-fitted events.
    event_grouping_key : str
        The name of the variable under which the name of the grouping for each
        pre-fitted event is found. If `None`, will not restrict gradient lookups
        to event groupings (allows reproducing the service's original behavior).
    approx_exponential : bool
        Approximate the exponential using exp(x) = 1 + x. This is appropriate when
        gradients have been fit with the purely linear `hardmax` activation function.
        (If you don't know what that is, just leave it at `False`.)
    support : str or dict
        Dictionary (or str that can be evaluated thereto) of the form {'parameter_name':
        (lower bound, upper bound)} containing the bounds of the parameter space inside
        which the gradients are valid. If a value outside of these bounds is requested,
        we have to extrapolate using the strategy defined in the `extrapolation`
        parameter.
    extrapolation : str
        Strategy to use for extrapolating beyond the bounds set by the `bounds` option.
        Options are `continue`, `linear` and `constant`. If `continue`, polynomial
        features are simply extended at the risk of weights getting out of control.
        If `linear`, second order features are extrapolated using their derivative at
        the closest bound. If `constant`, the value at the closest boundary is returned.
    distance_tol : float
        Numerical tolerance for distances to nearest neighbors above which a warning
        will be issued. Default is 0.
    params : ParamSet
        Note that the params required to be in `params` are determined from
        those listed in the `systematics`.
    """

    def __init__( # pylint: disable=dangerous-default-value
        self,
        fit_results_file,
        nominal_points,
        varnames=["pid", "true_coszen", "reco_coszen", "true_energy", "reco_energy"],
        event_grouping_key="event_category",
        approx_exponential=False,
        support=None,
        extrapolation="continue",
        distance_tol=0,
        **std_kwargs,
    ):
        # evaluation only works on event-by-event basis
        assert std_kwargs["calc_mode"] == "events"

        # Store args
        self.fit_results_file = find_resource(fit_results_file)
        self.varnames = varnames
        assert isinstance(event_grouping_key, str) or event_grouping_key is None
        self.event_grouping_key = event_grouping_key
        self.approx_exponential = approx_exponential
        assert isinstance(distance_tol, (int, float))
        self.distance_tol = distance_tol

        if isinstance(nominal_points, str):
            self.nominal_points = eval(nominal_points)
        else:
            self.nominal_points = nominal_points
        assert isinstance(self.nominal_points, collections.abc.Mapping)

        if isinstance(support, str):
            self.support = eval(support)
            assert isinstance(self.support, collections.abc.Mapping)
        elif isinstance(support, collections.abc.Mapping):
            self.support = support
        elif support is None:
            self.support = None
        else:
            raise ValueError("Unknown input format for `support`.")

        assert extrapolation in ["continue", "linear", "constant"]
        self.extrapolation = extrapolation

        param_names = list(self.nominal_points.keys())
        for pname in param_names:
            if self.support is not None and pname not in self.support:
                raise ValueError(
                    f"Support range is missing for parameter {pname}"
                )

        expected_container_keys = varnames + ['weights']
        # 'true_energy' is hard-coded to get sample size
        if 'true_energy' not in expected_container_keys:
            expected_container_keys.append('true_energy')

        # -- Initialize base class -- #
        super().__init__(
            expected_params=param_names,
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )

    def setup_function(self):
        """Load the fit results from the file and make some compatibility checks"""

        # make this an optional dependency
        import pandas as pd
        from sklearn.neighbors import KDTree

        self.data.representation = self.calc_mode

        # create containers for scale factors
        for container in self.data:
            container["us_scales"] = np.ones(container.size, dtype=FTYPE)

        # load the feather file and extract gradient names
        df = pd.read_feather(self.fit_results_file)

        self.gradient_names = [key for key in df.keys() if key.startswith("grad")]

        # create containers for gradients
        for container in self.data:
            for gradient_name in self.gradient_names:
                container[gradient_name] = np.empty(container.size, dtype=FTYPE)

        # convert the variable columns as well as the event groupings to an array
        X_pandas = df[self.varnames].to_numpy()
        if self.event_grouping_key is not None:
            groupings_array = df[self.event_grouping_key].to_numpy()
            groupings_set = set(groupings_array) # unique groupings
            logging.debug(
                "Event grouping information for ultrasurfaces evaluation taken"
                " from data frame entry '%s'. Found groupings '%s'.",
                self.event_grouping_key, groupings_set
            )
        else:
            # without groupings, create one tree containing all events
            logging.debug("Events will not be grouped for ultrasurfaces evaluation")
            tree = KDTree(X_pandas)
        # We will use a nearest-neighbor tree to search for matching events in the
        # DataFrame. Ideally, these should actually be the exact same events with a
        # distance of zero. We will raise a warning if we had to approximate an
        # event by its nearest neighbor with a distance > tolerance.
        for container in self.data:
            n_container = len(container["true_energy"])
            # It's important to match the datatype of the loaded DataFrame (single prec.)
            # so that matches will be exact (TODO: but matches aren't necessarily exact)
            X_pisa = np.zeros((n_container, len(self.varnames)), dtype=X_pandas.dtype)
            for i, vname in enumerate(self.varnames):
                X_pisa[:, i] = container[vname]

            if self.event_grouping_key is None:
                logging.debug(
                    "Looking for nearest neighbors of %d '%s' events among all"
                    " %d events in data frame.",
                    container.size, container.name, len(X_pandas)
                )
            else:
                # produce a dedicated KDTree in case of associated event grouping
                assoc_grouping = get_us_grouping_from_container_name(
                    name=container.name,
                    groupings_set=groupings_set
                )
                where = np.where(groupings_array == assoc_grouping)
                tree = KDTree(X_pandas[where])
                logging.debug(
                    "Looking for nearest neighbors of %d '%s' events among all"
                    " %d '%s' events in data frame.",
                    container.size, container.name, len(X_pandas[where]), assoc_grouping
                )
            # Query the tree for the single nearest neighbor
            dists, ind = tree.query( # pylint: disable=possibly-used-before-assignment
                X_pisa, k=1, return_distance=True, dualtree=False,
                breadth_first=False
            )
            n_outside_tol = np.sum(dists > self.distance_tol)
            if n_outside_tol:
                max_dist = np.max(dists)
                frac = float(n_outside_tol) * 100 / n_container
                logging.warning(
                    f"For {n_outside_tol} {container.name} events ({frac:.2g}%),"
                    " the nearest neighbor, from which each gradient will be "
                    "taken, is at a distance beyond the pre-set tolerance of "
                    f"{self.distance_tol:.2g}. The maximum distance to a "
                    f"nearest neighbor is {max_dist:.2g}."
                )

            if self.debug_mode:
                outfile = os.path.join(
                    CACHE_DIR, f"ultrasurfaces_{container.name}_debug_data.npz"
                )
                grads_list = []

            for gradient_name in self.gradient_names:
                grads = df[gradient_name].to_numpy()
                if self.event_grouping_key is not None:
                    # indices apply to the array of events of the grouping
                    grads = grads[where]
                container[gradient_name] = grads[ind.ravel()]
                if self.debug_mode:
                    grads_list.append(container[gradient_name])

            if self.debug_mode:
                np.savez_compressed(
                    file=outfile, dists=dists.ravel(), inds=ind.ravel(),
                    grads=grads_list, fit_results_file=self.fit_results_file,
                    gradient_names=self.gradient_names
                )
                logging.debug("Stored '%s' ultrasurfaces debug data in %s.",
                              container.name, CACHE_DIR)

    @profile
    def compute_function(self):

        self.data.representation = self.calc_mode

        # Calculate the `delta_p` matrix containing the polynomial features.
        # If requested, these feature may be extrapolated using the strategy defined
        # by `self.extrapolation`.

        delta_p_dict = {}

        # The gradients may be of arbitrary order and have interaction
        # terms. For example, if the gradient's name is
        # `grad__dom_eff__hole_ice_p0`, then the corresponding feature is
        # (delta dom_eff) * (delta hole_ice_p0).
        for count, gradient_name in enumerate(self.gradient_names):
            feature = 1.0
            # extract the parameter names from the name of the gradient
            param_names = gradient_name.split("grad")[-1].split("__")[1:]
            grad_order = len(param_names)
            has_interactions = len(set(param_names)) > 1

            for i, pname in enumerate(param_names):
                # If support has been set and a parameter is evaluated outside of those
                # bounds, we evaluate it at the nearest bound.
                if self.support is None:
                    bounded_value = self.params[pname].m
                else:
                    bounded_value = np.clip(self.params[pname].m, *self.support[pname])

                # The bounded value of the parameter shift from nominal
                x_b = bounded_value - self.nominal_points[pname]
                # The unbounded value
                x = self.params[pname].m - self.nominal_points[pname]

                # The extrapolation strategy `continue` is equivalent to just evaluating
                # at the unbounded point.
                if self.extrapolation == "continue":
                    # For a squared parameter, this will be done twice, i.e. the feature
                    # will be (dom_eff)^2 if the gradient is `grad__dom_eff__dom_eff`.
                    feature *= x
                elif self.extrapolation == "constant":
                    # Constant extrapolation simply means that we evaluate the bounded
                    # value.
                    feature *= x_b
                elif self.extrapolation == "linear":
                    # The linear extrapolation of a squared feature is given by
                    #   y = x_b^2 + (2x_b)(x - x_b),
                    # which can be re-written as
                    #   y = x_b (2x - x_b).
                    # We see right away that y = x^2 when x is within the bounds.
                    # We also want to pass through the first order gradients, since
                    # the linear extrapolation of x is trivially x.

                    if grad_order == 1:
                        feature *= x
                        continue

                    if has_interactions:
                        raise RuntimeError(
                            "Cannot calculate linear extrapolation for gradients with "
                            f"interaction terms: {gradient_name}"
                        )

                    if i == 0:
                        feature *= x_b
                    elif i == 1:
                        feature *= (2*x - x_b)
                    else:
                        raise RuntimeError(
                            "Cannot use linear extrapolation for orders > 2"
                        )

            delta_p_dict[gradient_name] = feature

        for container in self.data:

            # The "gradient shift" is the sum of the gradients times the parameter shifts,
            # i.e. grad * delta_p.
            # We allocate this array just once and accumulate the sum over all gradients
            # into it.

            # Also using zeros_like ensures consistent dtype
            grad_shifts = np.zeros_like(container["weights"])

            for count, gradient_name in enumerate(self.gradient_names):
                shift = delta_p_dict[gradient_name]
                grad_shift_inplace(container[gradient_name], shift, grad_shifts)
            # In the end, the equation for the re-weighting scale is
            #    exp(grad_p1 * shift_p1 + grad_p2 * shift_p2 + ...)
            if self.approx_exponential:
                # We can approximate an exponential with exp(x) = 1 + x,
                # but this is not recommended unless the gradients have also been fit
                # using this approximation.
                container["us_scales"] = 1 + grad_shifts
            else:
                container["us_scales"] = np.exp(grad_shifts)

    def apply_function(self):
        for container in self.data:
            container["weights"] *= container["us_scales"]


@njit
def grad_shift_inplace(grads, shift, out):
    for i, g in enumerate(grads):
        out[i] += shift * g


def init_test(**param_kwargs):
    """Instantiation example"""
    import pandas as pd
    from pisa.utils.random_numbers import get_random_state

    p1, p2 = 'opt_eff_overall', 'ice_scattering'
    param_set = ParamSet([
        Param(name=p1, value=1.0, **param_kwargs),
        Param(name=p2, value=0.0, **param_kwargs)
    ])
    # We cannot just use `value` attribute here (service unable to deal with pint quantity)
    nominal_points = {
        p1: param_set[p1].value.m_as('dimensionless'),
        p2: param_set[p2].value.m_as('dimensionless')
    }

    # create a test file containing 100 gradients on the fly
    N = 100
    random_state = get_random_state(0)
    varnames = ['inelasticity', 'reco_energy']
    df = {var: random_state.random(N).astype(dtype=FTYPE) for var in varnames}
    df.update({f'grad_{p}': np.multiply(random_state.random(N), 2).astype(dtype=FTYPE) for p in param_set.names})
    # also add an interaction term
    df[f'grad__{p1}__{p2}'] = np.multiply(random_state.random(N), 2).astype(dtype=FTYPE)

    df = pd.DataFrame.from_dict(data=df, dtype=FTYPE)
    fpath = os.path.join(CACHE_DIR, 'test_us_file.feather')
    df.to_feather(fpath)

    return ultrasurfaces(
        params=param_set, fit_results_file=fpath, varnames=varnames,
        nominal_points=nominal_points, calc_mode='events',
        event_grouping_key=None
    )
