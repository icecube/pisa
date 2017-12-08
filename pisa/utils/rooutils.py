# author: Shivesh Mandalia
#         s.p.mandalia@qmul.ac.uk
#
# date:   2016-12-30
"""
Convenience functions when interacting with ROOT.
"""


from itertools import product

from uncertainties import unumpy as unp

from ROOT import TH1D, TH2D
from root_numpy import array2hist, hist2array

from pisa.core.map import Map

__all__ = ['convert_to_th1d', 'convert_to_th2d', 'unflatten_thist']


def convert_to_th1d(in_map, errors=False):
    assert isinstance(in_map, Map)
    name = in_map.name
    assert len(in_map.shape) == 1
    n_bins = in_map.shape[0]
    edges = in_map.binning.bin_edges[0].m

    th1d = TH1D(name, name, n_bins, edges)
    array2hist(unp.nominal_values(in_map.hist), th1d)
    if errors:
        map_errors = unp.std_devs(in_map.hist)
        for idx in xrange(n_bins):
            th1d.SetBinError(idx+1, map_errors[idx])
    return th1d


def convert_to_th2d(in_map, errors=False):
    assert isinstance(in_map, Map)
    name = in_map.name
    n_bins = in_map.shape
    assert len(n_bins) == 2
    nbins_a, nbins_b = n_bins
    edges_a, edges_b = [b.m for b in in_map.binning.bin_edges]

    th2d = TH2D(name, name, nbins_a, edges_a, nbins_b, edges_b)
    array2hist(unp.nominal_values(in_map.hist), th2d)
    if errors:
        map_errors = unp.std_devs(in_map.hist)
        for x_idx, y_idx in product(*map(range, n_bins)):
            th2d.SetBinError(x_idx+1, y_idx+1, map_errors[x_idx][y_idx])
    return th2d


def unflatten_thist(in_th1d, binning, name='', errors=False, **kwargs):
    flat_hist = hist2array(in_th1d)
    if errors:
        map_errors = [in_th1d.GetBinError(idx+1)
                      for idx in xrange(len(flat_hist))]
        flat_hist = unp.uarray(flat_hist, map_errors)
    hist = flat_hist.reshape(binning.shape)
    return Map(hist=hist, binning=binning, name=name, **kwargs)
