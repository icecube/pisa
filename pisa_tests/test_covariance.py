#!/usr/bin/env python

from pisa.utils.log import Levels, logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.core.pipeline import Pipeline

from argparse import ArgumentParser

import numpy as np

eps = 1e-10

"""
Runs a set of tests on an example PISA pipeline to ensure 
    1. correlated priors can be added
    2. internal conversions between correlated and uncorrelated priors are accurate 
    3. stddev in the uncorrelated basis' priors are accurate
    4. the LLHs in the uncorrelated basis are consistent with the expectation from the correlated basis
"""

__author__ = 'Ben Smithers'



def test_covariance(verbosity=Levels.WARN):
    """
    Run one of the example pipelines, applying a covariance matrix to two of the parameters
    """
    from pisa.utils.log import Levels, logging, set_verbosity
    from pisa.utils.resources import find_resource
    from pisa.core.pipeline import Pipeline

    set_verbosity(verbosity)

    settings = find_resource("settings/pipeline/IceCube_3y_neutrinos.cfg")

    ex_cov = {
        "opt_eff_lateral":{
            "opt_eff_lateral":1.0,
            "opt_eff_headon":0.2
        },
        "opt_eff_headon":{
            "opt_eff_lateral":0.2,
            "opt_eff_headon":1.0
        }
    }

    dim = len(ex_cov.keys())
    cov = np.zeros((dim,dim))
    for k_i, key in enumerate(ex_cov.keys()):
        for k_j, subkey in enumerate(ex_cov[key].keys()):
            cov[k_i][k_j] = ex_cov[key][subkey]

    
    evals, evecs = np.linalg.eig(cov)
    reverse = np.linalg.inv(evecs)

    pipe = Pipeline(settings)

    old_params = [pipe.params[parname] for parname in ex_cov.keys()]
    means = [0.0 for i in range(dim)]
    for i, param in enumerate(old_params):
        if param.prior.kind == "gaussian":
            means[i] = param.prior.mean
        elif param.prior.kind=="uniform":
            means[i] = (param.range[1] + param.range[0])*0.5
        else:
            raise NotImplementedError()

    pipe.add_covariance(ex_cov)

    def vs_from_xs(xs):
        return np.matmul(xs-means,evecs)

    def xs_from_vs(vs):
        xs = np.matmul(vs, reverse) + means
        return xs

    pipe.params.randomize_free()

    # check the Xs are what they should be and the vs are what they should be 

    vs = np.array([ pipe.params["opt_eff_lateral_rotated"].value.m,  pipe.params["opt_eff_headon_rotated"].value.m ])
    xs = np.array([ pipe.params["opt_eff_lateral"].value.m,  pipe.params["opt_eff_headon"].value.m ])

    get_xs = xs_from_vs(vs)
    for i_x, value in enumerate(get_xs):
        assert abs(value-xs[i_x])<eps, "{} is different from {}".format(value, xs[i_x])

    get_vs = vs_from_xs(xs)
    for i_v, value in enumerate(get_vs):
        assert abs(value-vs[i_v])<eps, "{} is different from {}".format(value, xs[i_v])


    true_sigams = np.sqrt(evals)
    sigmas = np.array([pipe.params["opt_eff_lateral_rotated"].prior.stddev,
                        pipe.params["opt_eff_headon_rotated"].prior.stddev  ])
    for i_s, value in enumerate(sigmas):
        assert abs(value - true_sigams[i_s])<eps, "sigma {} is different from {}".format(value, true_sigams[i_s])


    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    def eval_as_xs(xs):
        rhs = (xs-means)
        return np.sum( -0.5*np.matmul(np.transpose(rhs),  np.matmul(inv, rhs)) )


    
    llh_sum = pipe.params["opt_eff_headon_rotated"].prior_llh + pipe.params["opt_eff_lateral_rotated"].prior_llh
    direct = eval_as_xs(xs)
    assert abs(llh_sum-direct)<eps, "Difference in llhs {} vs {}".format(llh_sum, direct)

    logging.info("<< PASS : Correlated Priors >>")    


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-v", action="count", default=Levels.WARN, help="set verbosity level"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    kwargs = vars(args)
    kwargs["verbosity"] = kwargs.pop("v")
    test_covariance(**kwargs)

if __name__=="__main__":
    main()