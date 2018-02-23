"""
Pull method tools.
"""


from __future__ import absolute_import, division

import numpy as np

from pisa.utils.fisher_matrix import build_fisher_matrix, get_fisher_matrix
from pisa.utils.log import logging, set_verbosity

__all__ = []

__author__ = 'T. Ehrhardt'

__license__ = '''Copyright (c) 2014-2018, The IceCube Collaboration

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
    for n in range(len(coeff))[1:]: # runs from 1 to len(coeff)
        derivative += n*coeff[n]*loc**(n-1)

    return derivative


def get_derivative_map(hypo_maps, chan, fiducial=None, take_finite_diffs=False, degree=2):
    """
    Get the approximate derivative of data w.r.t parameter par
    at location loc with polynomic degree of approximation, default: 2.
    Data is a dictionary of the form
    {
    'test_point1': {'params': {},
	     {'map': [[],[],...],
		        'ebins': [],
		        'czbins': []
		      },
	      }
    'test_point2': ...
    }
    """
    test_points = sorted(hypo_maps.keys())

    # Flatten data map (for use with polyfit)
    hypo_maps_flat = [hypo_maps[pvalue].flatten() for pvalue in test_points]

    if take_finite_diffs:
        assert(len(test_points)==2)
        # we only have 2 test points
        del_x = test_points[1] - test_points[0]
        del_counts = np.subtract(hypo_maps_flat[1], hypo_maps_flat[0])
        derivative_map = np.divide(del_counts, del_x.magnitude)

    else:
        # perform a fit, with number of test points specified by grid settings file
        # Polynomial fit of bin counts
        channel_fit_params = np.polyfit(test_points, channel_data, deg=degree)
        # Get partial derivatives at fiducial values
        derivative_map = derivative_from_polycoefficients(channel_fit_params[::-1], fiducial['value'])

    # TODO: keep flat or reshape?
    #derivative_map = np.reshape(derivative_map, hypo_maps.values()[0].shape)

    return derivative_map


def get_gradients(param, hypo_maker, take_finite_diffs=False, grid_settings=None):
    """
    Use the template maker to create all the templates needed to obtain the gradients.
    """
    logging.info("Working on parameter %s."%param)

    fiducial_params = hypo_maker.params

    steps = fiducial_params[param].range #np.array([fiducial_params[param].range[0], fiducial_params[param].range[1]])
    pmaps = {}

    # Generate one template for each value of the parameter in question and store in pmaps
    for param_value in steps:
        hypo_maker.params[param].value = param_value
        # Make the template corresponding to the current value of the parameter
        hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)
        pmaps[param_value] = hypo_asimov_dist.hist['total']

    gradient_map = get_derivative_map(pmaps, fiducial_params[param], take_finite_diffs, 2)

    return pmaps, gradient_map


def calculate_pulls(fisher, fid_maps_truth, fid_hypo_asimov_dist, gradient_maps):
    """Compute parameter pulls given data distribution, fiducial hypothesis
    distribution, Fisher matrix, and binwise gradients.
    """
    fisher = {'total': fisher}
    d = []
    for chan_idx, chan in enumerate(fisher):
        chan_d = []
        f = fisher[chan]
        # binwise derivatives w.r.t all parameters in this chan
        gm = gradient_maps[chan]
        # binwise differences between truth and model in this chan
        # [d_bin1, d_bin2, ..., d_bin780]
        dm = np.subtract(fid_maps_truth[chan].hist, fid_hypo_asimov_dist[chan].hist).flatten()
        # binwise statist. uncertainties for truth
        # [sigma_bin1, sigma_bin2, ..., sigma_bin3]
        sigma = fid_maps_truth[chan].hist.flatten()
        for i, param in enumerate(f.parameters):
            chan_d.append([])
            assert(param in gm.keys())
            d_p_binwise = np.divide(np.multiply(dm, gm[param].flatten()), sigma)
            # Sum over bins
            d_p = d_p_binwise.sum()
            chan_d[i] = d_p
        d.append(chan_d)
    # Binwise sum over (difference btw. fiducial maps times derivatives of
    # expected bin count / statistical uncertainty of bin count), summed over channels
    # Sum over channels (n-d vector, where n the number of systematics which are linearised)
    d = np.sum(d, axis=0)

    # This only needs to be multiplied by the (overall) Fisher matrix inverse.
    f_tot = fisher['total']
    f_tot.calculateCovariance()
    pulls = np.dot(f_tot.covariance, d)
    return [(pname, pull) for pname, pull in zip(f_tot.parameters, pulls.flat)]


def test_pull_method():
    import time
    from pisa import ureg
    from pisa.core.distribution_maker import DistributionMaker

    data_maker = DistributionMaker(
        pipelines="../../pisa_examples/resources/settings/pipeline/example_param.cfg"
    )

    hypo_maker = DistributionMaker(
        pipelines="../../pisa_examples/resources/settings/pipeline/example_param.cfg"
    )


    param_variations = {'aeff_scale': +0.07*ureg.dimensionless,
                        'nue_numu_ratio': -0.04*ureg.dimensionless
    }

    for pname, variation in param_variations.items():
        nominal = data_maker.params[pname].nominal_value
        data_maker.params[pname].value = nominal + variation.to(nominal.units)

    data_dist = data_maker.get_outputs(return_sum=True)

    # we want to test whether we can get back the parameters
    # varied away from nominal above
    for param in hypo_maker.params.free:
        if not param.name in param_variations:
            param.is_fixed = True

    for pname in param_variations:
        param = hypo_maker.params[pname]
        param.is_fixed = False
        # set sensible ranges over which difference quotients are computed
        # (don't have to include the true value of the parameter
        # to fit it back)
        param.range = [0.95*param.nominal_value, 1.05*param.nominal_value]

    # now we run the fit
    start_t = time.time()
    fisher, gradient_maps, hypo_asimov_dist = get_fisher_matrix(
        data_dist=data_dist,
        hypo_maker=hypo_maker,
        take_finite_diffs=True
    )

    pulls = calculate_pulls(
        fisher=fisher,
        fid_maps_truth=data_dist,
        fid_hypo_asimov_dist=hypo_asimov_dist,
        gradient_maps=gradient_maps
    )
    end_t = time.time()
    logging.info('Pull fit took %.2fs.' % (((end_t - start_t) * ureg.sec).m))
    msg = 'fit vs. true parameter pulls:\n'
    for pname, variation in param_variations.items():
        true_pull = variation
        fit_pull = [pull for (name, pull) in pulls if name==pname][0]
        msg += ' '*12
        msg += '%s: %.4f (fit) vs. %.4f (truth)\n' % (pname, fit_pull, true_pull)
    logging.info(msg)

if __name__ == '__main__':
    set_verbosity(1)
    test_pull_method()
