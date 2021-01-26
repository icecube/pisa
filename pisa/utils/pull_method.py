"""
Pull method tools.
"""


from __future__ import absolute_import, division

import numpy as np

from pisa.utils.log import logging, set_verbosity

__all__ = []

__author__ = 'T. Ehrhardt'

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


def derivative_from_polycoefficients(coeff, loc):
    """
    Return derivative of a polynomial of the form

        f(x) = coeff[0] + coeff[1]*x + coeff[2]*x**2 + ...

    at x = loc
    """
    derivative = 0.
    for n, c in enumerate(coeff):
        if n == 0:
            continue
        derivative += n*c*loc**(n-1)

    return derivative


def get_derivative_map(hypo_maps):
    """
    Get binwise (linear) derivatives of counts w.r.t. a parameter, whose
    variations and associated templates are stored in `hypo_maps`.

    Parameters
    -----------
    hypo_maps : dict
        dictionary of the form
        {
        'test_point1': {
            'params': {},
            {'map': [[],[],...],
             'ebins': [],
             'czbins': []
            },
        }
        'test_point2': ...
        }

    Returns
    -------
    derivative_map : sequence
        Flat array of derivatives (of length = no. of bins)

    """
    test_points = sorted(hypo_maps.keys())

    # flatten data map (for use with polyfit - not employed currently)
    hypo_maps_flat = [hypo_maps[pvalue].flatten() for pvalue in test_points]

    assert len(test_points) == 2
    # we only have 2 test points
    del_x = test_points[1] - test_points[0]
    del_counts = np.subtract(hypo_maps_flat[1], hypo_maps_flat[0])
    derivative_map = np.divide(del_counts, del_x.magnitude)

    # keep flat
    return derivative_map


def get_gradients(param, hypo_maker, test_vals):
    """Use the template maker to create all the templates needed
    to obtain the gradients in a given parameter.

    Parameters
    ----------
    param : str
        Name of parameter w.r.t. which we are calculating binwise
        template changes
    hypo_maker : DistributionMaker
        Needs to hold the parameter `param` in its `ParamSet`
    test_vals :  sequence with units
        Values of the parameter `param` to probe, i.e., generate templates for

    Returns
    -------
    pmaps : dict
        Dictionary with `test_vals` as keys and resulting templates' 'total'
        nominal values
    gradient_map : sequence
        As returned by `get_derivative_map`

    """
    logging.trace("Working on parameter %s."%param)

    pmaps = {}

    # generate one template for each value of the parameter in question
    # and store in pmaps
    for param_value in test_vals:
        hypo_maker.params[param].value = param_value
        # make the template corresponding to the current value of the parameter
        hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)
        pmaps[param_value] = hypo_asimov_dist.nominal_values['total']

    gradient_map = get_derivative_map(
        hypo_maps=pmaps,
    )

    return pmaps, gradient_map


def calculate_pulls(fisher, fid_maps_truth, fid_hypo_asimov_dist, gradient_maps,
                    nonempty):
    """Compute parameter pulls given data distribution, fiducial hypothesis
    distribution, Fisher matrix, and binwise gradients.

    Parameters
    ----------
    fisher : FisherMatrix
    fid_maps_truth : MapSet
    fid_hypo_asimov_dist : MapSet
    gradient_maps : dict
        output of `get_gradients` put into a dict, with one entry for each
        parameter that was varied
    nonempty : sequence
        Indices of nonzero entries of flat list of `fid_hypo_asimov_dist`'s
        'total' entries

    Returns
    -------
    param_pull_list: sequence of tuples
        tuples of parameter name and parameter pull

    """
    fisher = {'total': fisher}
    d = []
    for chan in fisher:
        chan_d = []
        f = fisher[chan]
        # binwise derivatives w.r.t all parameters in this chan
        gm = gradient_maps[chan]
        # binwise differences between truth and model in this chan
        # [d_bin1, d_bin2, ..., d_bin780]
        dm = np.subtract(
            fid_maps_truth[chan].nominal_values,
            fid_hypo_asimov_dist[chan].nominal_values).flatten()
        dm = dm[nonempty]
        # binwise statist. uncertainties for truth
        # [sigma_bin1, sigma_bin2, ..., sigma_bin3]
        # TODO: had sigma of observed counts here, but could be zero
        # (nonempty *expectations* are ensured when fisher matrix is generated)
        #sigma = fid_maps_truth[chan].std_devs.flatten()
        sigma = fid_hypo_asimov_dist[chan].std_devs.flatten()[nonempty]
        for i, param in enumerate(f.parameters):
            chan_d.append([])
            assert param in gm.keys()
            d_p_binwise = np.divide(np.multiply(dm, gm[param].flatten()[nonempty]), sigma)
            # Sum over bins
            d_p = d_p_binwise.sum()
            chan_d[i] = d_p
        d.append(chan_d)
    # Binwise sum over (difference btw. fiducial maps times derivatives of
    # expected bin count / statistical uncertainty of bin count),
    # summed over channels
    # Sum over channels (n-d vector, where n the number of systematics which
    # are linearised)
    d = np.sum(d, axis=0)

    # This only needs to be multiplied by the (overall) Fisher matrix inverse.
    f_tot = fisher['total']
    f_tot.calculateCovariance()
    pulls = np.dot(f_tot.covariance, d)
    param_pull_list = [(pname, pull) for pname, pull in zip(f_tot.parameters, pulls.flat)]
    return param_pull_list
