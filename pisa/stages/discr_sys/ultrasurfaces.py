"""
PISA pi stage to apply ultrasurface fits from discrete systematics parameterizations
"""

import collections

import numpy as np
from numba import njit

from pisa import FTYPE
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.resources import find_resource

__all__ = [
    "ultrasurfaces",
    "init_test"
]

__author__ = "A. Trettin, L. Fischer"

__license__ = """Copyright (c) 2014-2022, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


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
    params : ParamSet
        Note that the params required to be in `params` are determined from
        those listed in the `systematics`.
    """

    def __init__(
        self,
        fit_results_file,
        nominal_points,
        varnames=["pid", "true_coszen", "reco_coszen", "true_energy", "reco_energy"],
        approx_exponential=False,
        support=None,
        extrapolation="continue",
        **std_kwargs,
    ):
        # evaluation only works on event-by-event basis
        assert std_kwargs["calc_mode"] == "events"

        # Store args
        self.fit_results_file = find_resource(fit_results_file)
        self.varnames = varnames
        self.approx_exponential = approx_exponential

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

        self.extrapolation = extrapolation

        param_names = list(self.nominal_points.keys())
        for pname in param_names:
            if self.support is not None and pname not in self.support:
                raise ValueError(
                    f"Support range is missing for parameter {pname}"
                )

        # -- Initialize base class -- #
        super().__init__(
            expected_params=param_names,
            expected_container_keys=varnames+["weights"],
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

        # Convert the variable columns to an array
        X_pandas = df[self.varnames].to_numpy()
        # We will use a nearest-neighbor tree to search for matching events in the
        # DataFrame. Ideally, these should actually be the exact same events with a
        # distance of zero. We will raise a warning if we had to approximate an
        # event by its nearest neighbor with a distance > 0.

        # At least in theory, this should always pick out the one exact event from the
        # correct event category. If an event is not exactly matched, however, it's
        # possible that the gradient for a numu_cc event might get picked from
        # a nu_nc event, for example. We don't have any safeguards against that
        # at this time, even though the information is in the DataFrame to do it.

        # TODO: Ensure event category of nearest neighbor matches that of query
        tree = KDTree(X_pandas)
        for container in self.data:
            n_container = len(container["true_energy"])
            # It's important to match the datatype of the loaded DataFrame (single prec.)
            # so that matches will be exact.
            X_pisa = np.zeros((n_container, len(self.varnames)), dtype=X_pandas.dtype)
            for i, vname in enumerate(self.varnames):
                X_pisa[:, i] = container[vname]
            # Query the tree for the single nearest neighbor
            dists, ind = tree.query(X_pisa, k=1)
            if np.any(dists > 0):
                logging.warn(
                    f"Could not find exact match for {np.sum(dists > 0)} {container.name} "
                    f"events ({float(np.sum(dists > 0)) * 100 / n_container:.4f}%) "
                    "in the loaded DataFrame. Their "
                    "gradients will be taken from the nearest neighbor."
                )
            # TODO: since we read out all gradients we could loop over the
            # parameters outside and then over the containers inside
            for gradient_name in self.gradient_names:
                grads = df[gradient_name].to_numpy()
                container[gradient_name] = grads[ind.ravel()]

    @profile
    def compute_function(self):

        self.data.representation = self.calc_mode

        # Calculate the `delta_p` matrix containing the polynomial features.
        # If requested, these feature may be extrapolated using the strategy defined
        # by `self.extrapolation`.

        delta_p_dict = dict()

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
    param_set = ParamSet([
        Param(name='dom_eff', value=1.0, **param_kwargs),
        Param(name='hole_ice_p0', value=0.0, **param_kwargs),
        Param(name='hole_ice_p1', value=0.0, **param_kwargs),
        Param(name='bulk_ice_abs', value=1.0, **param_kwargs),
        Param(name='bulk_ice_scatter', value=1.0, **param_kwargs),
        Param(name='bfr_eff', value=0.0, **param_kwargs),
    ])
    return ultrasurfaces(
        params=param_set,
        fit_results_file='fit_settings/oscNext_pisa_genie_flercnn_knn_grads_all.feather', #TODO
        nominal_points = {
            "dom_eff": 1.0, "hole_ice_p0": 0.101569, "hole_ice_p1": -0.049344,
            "bulk_ice_abs": 1.0, "bulk_ice_scatter": 1.0, "bfr_eff": 0.0
        },
        calc_mode = 'events'
    )
