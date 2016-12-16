# -*- coding: <encoding name> -*-
# author:  J.L. Lanfanchi
#          jll1062@phys.psu.edu
#
# date:    March 28, 2015
#
# Rev 0
#   MLConfInterval class implementation imported from personal code repo:
#   Handy utility for finding simplistic only-single-mode-aware
#   maximum-likelihood confidence intervals for arbitrarily-shaped PDF's
#   using linear interpolation (i.e., trapezoidal numerical integration).
"""
Class implementing numerical routines for finding maximum-likelihood confidence
intervals given a PDF.

"""


from __future__ import division

import numpy as np
import scipy.optimize as optimize


__all__ = ['MLConfInterval']


class MLConfInterval(object):
    """Maximum-likelihood confidence interval of a PDF.

    Parameters
    ----------
    x : sequence of floats
        Sample locations of the random variable at which the PDF is sampled

    y : sequence of floats
        Density values corresponding to the sample locations, hence
        defining a linearly-interpolated approximation to the PDF y =
        PDF(x)

    epsilon : float
        Numerical tolerance for checking numerics

    """
    def __init__(self, x, y, epsilon=1e-10):
        self.epsilon = epsilon

        # Sort all data in ascending-X order
        idx = x.argsort()
        self.x = x[idx]
        self.y = y[idx]

        # Make sure these don't propagate below; can remove this once function
        # is tested to work correctly
        del x, y

        # Append points to left and/or right ends if endpoints don't go to zero
        if self.y[0] != 0:
            self.x = np.concatenate([[self.x[0]], self.x])
            self.y = np.concatenate([[0], self.y])
        if self.y[-1] != 0:
            self.x = np.concatenate([self.x, [self.x[-1]]])
            self.y = np.concatenate([self.y, [0]])

        # Find ML point
        self.max_y_idx = np.argmax(self.y)
        self.max_y_x = self.x[self.max_y_idx]
        self.max_y = self.y[self.max_y_idx]

        # In each group, *include* the ML point for interpolation in bins
        # immediately adjacent to (i.e., both left and right of) ML point
        self.left_x = self.x[0:self.max_y_idx+1]
        self.left_y = self.y[0:self.max_y_idx+1]
        self.right_x = self.x[self.max_y_idx:]
        self.right_y = self.y[self.max_y_idx:]

        # Sort points to left of (& including) ML point in decreasing-X order.
        # In order to re-sort the left data later on, simply index it again
        # with left_sortidx
        #
        # NOTE: Right data is already sorted correctly, i.e., in *increasing-X*
        #       order, and needs no re-sorting here or later on.
        self.left_sortidx = np.argsort(self.left_x)[::-1]
        self.left_x = self.left_x[self.left_sortidx]
        self.left_y = self.left_y[self.left_sortidx]

        self.left_dx = np.diff(self.left_x)
        self.left_dy = np.diff(self.left_y)
        self.left_areas = -self.left_dx * (self.left_dy/2.0 + self.left_y[:-1])

        self.right_dx = np.diff(self.right_x)
        self.right_dy = np.diff(self.right_y)
        self.right_areas = (self.right_dx * (self.right_dy/2.0
                                             + self.right_y[:-1]))

        self.total_area = np.sum(self.left_areas) + np.sum(self.right_areas)

        # Compute total area, trapezoidal rule, using Numpy routine (for check)
        trapz_total_area = np.trapz(y=self.y, x=self.x)

        assert (np.abs(self.total_area-trapz_total_area) < self.epsilon), \
            'Total areas do not match within the specified precision, ' + \
            'epsilon='+str(self.epsilon)

        self.ci_lower_bound = -np.inf
        self.ci_upper_bound = np.inf
        self.ci_prob = 1.0

    # TODO: make this more succinct
    def find_ci_lin(self, conf, maxiter=100):
        """Perform a search in Y to find that Y whose outermost-in-X
        intersection points with the PDF define an X-interval enclosing the
        fraction conf of the total area under that curve. "Outermost" is
        defined as being furthest left and right (i.e., in X) of the location
        of the curve's maximum, so e.g. a multi-modal distribution will "hop"
        to right or left to another mode and *overshoot* the confidence level
        desired. This is a conservative approach to dealing with multi-modal
        distributions but is not the most accurate (most accurate would be to
        draw multiple confidence intervals and only combine them when they
        overlap).

        Parameters
        ----------
        conf : float in [0, 1]
        maxiter : int > 0

        Returns
        -------
        ci_lower_bound, ci_upper_bound, ci_prob, r

        """
        target_area = self.total_area * conf
        y, r = optimize.brentq(
            self.area,
            1e-5,
            self.max_y,
            args=(target_area,),
            maxiter=maxiter,
            full_output=True
        )
        return self.ci_lower_bound, self.ci_upper_bound, self.ci_prob, r

    def area(self, y, area_ref=0):
        """For a given y in the range [0, max(PDF)), compute the total area
        under the PDF between the outermost x-intersection points at that y.

        Parameters
        ----------
        y : float
        area_ref : float

        Returns
        -------
        area

        """
        left_ind, left_x, right_ind, right_x = self.furthestRoots(y)

        # Areas up to but excluding the bin within which the Y-value lies
        left_area = np.sum(self.left_areas[0:left_ind])
        right_area = np.sum(self.right_areas[0:right_ind])

        # Add in the areas of the partial bins
        left_area += (
            -(left_x - self.left_x[left_ind])
            * ((y - self.left_y[left_ind])/2.0 + self.left_y[left_ind])
        )
        right_area += (
            (right_x - self.right_x[right_ind])
            * ((y - self.right_y[right_ind])/2.0 + self.right_y[right_ind])
        )

        # Store the interval values
        self.ci_lower_bound = left_x
        self.ci_upper_bound = right_x
        self.ci_prob = y

        return left_area + right_area - area_ref

    def furthestRoots(self, y):
        """Find the outermost x-intersection points, i.e., PDF(x) = y

        Parameters
        ----------
        y : float

        Returns
        -------
        left_ind, left_x, right_ind, right_x

        """
        # This indexes the edge closest to the ML value of the bin containing
        # the furthest root left of the ML value
        left_inds = list(np.where(np.diff(np.sign(self.left_y - y)) != 0)[0])
        left_inds.extend(list(np.where(self.left_y == y)[0]))
        left_ind = max(left_inds)

        # This indexes the edge closest to the ML value of the bin containing
        # the furthest root right of the ML value
        right_inds = list(np.where(np.diff(np.sign(self.right_y - y)) != 0)[0])
        right_inds.extend(list(np.where(self.right_y == y)[0]))
        right_ind = max(right_inds)

        # Linear interpolation: $ x_i = x_0 + (dx_0)/(dy_0) * (y_i - y_0) $

        # Linear interp to find X coordinates corresponding to Y value
        left_x = (
            self.left_x[left_ind]
            + float(self.left_dx[left_ind])
            / float(self.left_dy[left_ind])
            * (y - self.left_y[left_ind])
        )
        right_x = (
            self.right_x[right_ind]
            + float(self.right_dx[right_ind])
            / float(self.right_dy[right_ind])
            * (y - self.right_y[right_ind])
        )

        return left_ind, left_x, right_ind, right_x
