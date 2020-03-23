"""
Functions to help compare and plot differences between PISA 2 and PISA 3 maps
"""


from __future__ import absolute_import, division

import os
import numpy as np

import matplotlib.pyplot as plt

from pisa import ureg
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map
from pisa.utils.fileio import get_valid_filename, mkdir
from pisa.utils.log import logging


__all__ = ['order', 'order_str', 'check_agreement', 'print_agreement',
           'print_event_rates', 'validate_maps',
           'make_delta_map', 'make_ratio_map',
           'validate_map_objs',
           'baseplot', 'baseplot2',
           'plot_comparisons', 'plot_map_comparisons', 'plot_cmp',
           'pisa2_map_to_pisa3_map']

__author__ = 'S. Wren'

__license__ = '''Copyright (c) 2014-2017, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


# TODO: make functions work transparently (i.e. autodetect) whether it's a
# PISA 2 or PISA 3 style map object, convert to PISA 3 maps, and go from
# there.


def order(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        o = np.ceil(np.log10(np.abs(x)))
    return o


def order_str(x):
    order_float = order(x)
    try:
        return str(int(order_float)).rjust(4)
    except OverflowError:
        pass
    return str(order_float)


def check_agreement(testname, thresh_ratio, ratio, thresh_diff, diff):
    ratio_pass = np.abs(ratio) <= np.abs(thresh_ratio)
    diff_pass = np.abs(diff) <= np.abs(thresh_diff)

    thresh_ratio_str = order_str(thresh_ratio)
    ratio_ord_str = order_str(ratio)
    ratio_pass_str = 'PASS' if ratio_pass else 'FAIL'

    thresh_diff_str = order_str(thresh_diff)
    diff_ord_str = order_str(diff)
    diff_pass_str = 'PASS' if diff_pass else 'FAIL'

    headline = '<< {pass_str:s} : {testname:s}, {kind:s} >>'
    detail_str = ('... agree to (( 10^{level:s} )) ; '
                  'thresh = (( 10^{thresh:s} ))')

    ratio_headline = headline.format(
        testname=testname, kind='fract diff', pass_str=ratio_pass_str
    )
    ratio_detail = detail_str.format(
        level=ratio_ord_str, thresh=thresh_ratio_str
    )

    diff_headline = headline.format(
        testname=testname, kind='diff', pass_str=diff_pass_str,
    )
    diff_detail = detail_str.format(
        level=diff_ord_str, thresh=thresh_diff_str
    )

    err_messages = []
    if ratio_pass:
        logging.info(ratio_headline)
        logging.info(ratio_detail)
    else:
        err_messages += [ratio_headline, ratio_detail]

    if diff_pass:
        logging.info(diff_headline)
        logging.info(diff_detail)
    else:
        err_messages += [diff_headline, diff_detail]

    if not (ratio_pass and diff_pass):
        for m in err_messages:
            logging.error(m)
        raise ValueError('\n    '.join(err_messages))


def print_agreement(testname, ratio):
    ratio_ord_str = order_str(ratio)
    s = '<< {testname:s}, {kind:s} >>\n' \
        ' agreement to 10^{level:s}'

    s_ratio = s.format(
        testname=testname, kind='fract diff', level=ratio_ord_str
    )

    logging.info(s_ratio)


def print_event_rates(testname1, testname2, kind, map1_events, map2_events):
    s = '<< {testname:s} total {kind:s} events >>' \
        ' {events:s}'

    test1_events = s.format(
        testname=testname1, kind=kind, events='%.2f'%(map1_events)
    )
    test2_events = s.format(
        testname=testname2, kind=kind, events='%.2f'%(map2_events)
    )

    logging.info(test1_events)
    logging.info(test2_events)


# TODO: specify `allclose` parameters `rtol` and `atol` excplicitly
def validate_maps(amap, bmap):
    """Validate that two PISA 2 style maps are compatible binning."""
    if not (np.allclose(amap['ebins'], bmap['ebins']) and
            np.allclose(amap['czbins'], bmap['czbins'])):
        raise ValueError("Maps' binnings do not match!")


# TODO: specify `allclose` parameters `rtol` and `atol` excplicitly
def validate_map_objs(amap, bmap):
    """Validate that two PISA 3 style maps are compatible binning."""
    if not all([np.allclose(ae, be) for ae, be
                in zip(amap.binning.bin_edges, bmap.binning.bin_edges)]):
        raise ValueError(
            "Maps' binnings do not match! Got first map as \n%s \nand second "
            " map as \n%s"
            % (amap.binning.hashable_state, bmap.binning.hashable_state)
        )


def make_delta_map(amap, bmap):
    """Get the difference between two PISA 2 style maps (amap-bmap) and return
    as another PISA 2 style map."""
    validate_maps(amap, bmap)
    return {'ebins': amap['ebins'],
            'czbins': amap['czbins'],
            'map': amap['map'] - bmap['map']}


def make_ratio_map(amap, bmap):
    """Get the ratio of two PISA 2 style maps (amap/bmap) and return as another
    PISA 2 style map."""
    validate_maps(amap, bmap)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = {'ebins': amap['ebins'],
                  'czbins': amap['czbins'],
                  'map': amap['map']/bmap['map']}
    return result


def baseplot(m, title, ax, clabel=None, symm=False, evtrate=False,
             vmax=None, cmap=plt.cm.afmhot):
    """Simple plotting of a 2D histogram (map)"""
    hist = np.ma.masked_invalid(m['map'])
    energy = m['ebins']
    coszen = m['czbins']
    if symm:
        cmap = plt.cm.seismic
        extr = np.nanmax(np.abs(hist))
        if vmax is None:
            vmax = extr
        vmin = -extr
    else:
        if evtrate:
            vmin = 0
        else:
            vmin = np.nanmin(hist)
        if vmax is None:
            vmax = np.nanmax(hist)
    cmap.set_bad(color=(0, 1, 0), alpha=1)
    x = coszen
    y = np.log10(energy)
    X, Y = np.meshgrid(x, y)
    pcmesh = ax.pcolormesh(X, Y, hist, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = plt.colorbar(mappable=pcmesh, ax=ax)
    if clabel is not None:
        cbar.set_label(clabel)
    cbar.ax.tick_params(labelsize='large')
    ax.set_xlabel(r'$\cos\theta_Z$')
    ax.set_ylabel(r'Energy (GeV)')
    ax.set_title(title, y=1.03)
    min_e = np.min(energy)
    max_e = np.max(energy)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    lin_yticks = 2**(np.arange(np.ceil(np.log2(min_e)),
                               np.floor(np.log2(max_e))+1))
    ax.set_yticks(np.log10(lin_yticks))
    ax.set_yticklabels([str(int(yt)) for yt in lin_yticks])


def baseplot2(map, title, ax, vmax=None, symm=False, evtrate=False):
    """Simple plotting of a 2D map.

    Parameters
    ----------
    map : Map
    title : str
    ax : axis
    symm : bool
    evtrate : bool

    Returns
    -------
    ax, pcmesh, cbar

    """
    assert len(map.binning) == 2
    hist = np.ma.masked_invalid(map.hist)
    if symm:
        cmap = plt.cm.seismic
        extr = np.nanmax(np.abs(hist))
        vmax = extr
        vmin = -extr
    else:
        cmap = plt.cm.afmhot
        if evtrate:
            vmin = 0
        else:
            vmin = np.nanmin(hist)
        if vmax is None:
            vmax = np.nanmax(hist)
    cmap.set_bad(color=(0, 1, 0), alpha=1)

    x = map.binning.dims[0].bin_edges.magnitude
    y = map.binning.dims[1].bin_edges.magnitude

    if map.binning.dims[0].is_log:
        xticks = 2**(np.arange(np.ceil(np.log2(min(x))),
                               np.floor(np.log2(max(x)))+1))
        x = np.log10(x)
    if map.binning.dims[1].is_log:
        yticks = 2**(np.arange(np.ceil(np.log2(min(y))),
                               np.floor(np.log2(max(y)))+1))
        y = np.log10(y)

    X, Y = np.meshgrid(x, y)
    pcmesh = ax.pcolormesh(X, Y, hist.T, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = plt.colorbar(mappable=pcmesh, ax=ax)
    cbar.ax.tick_params(labelsize='large')
    ax.set_xlabel(map.binning.dims[0].tex)
    ax.set_ylabel(map.binning.dims[1].tex)
    ax.set_title(title, y=1.03)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))

    if map.binning.dims[0].is_log:
        ax.set_xticks(np.log10(xticks))
        ax.set_xticklabels([str(int(xt)) for xt in xticks])
    if map.binning.dims[1].is_log:
        ax.set_yticks(np.log10(yticks))
        ax.set_yticklabels([str(int(yt)) for yt in yticks])

    return ax, pcmesh, cbar


def plot_comparisons(ref_map, new_map, ref_abv, new_abv, outdir, subdir, name,
                     texname, stagename, servicename, shorttitles=False,
                     ftype='png'):
    """Plot comparisons between two identically-binned PISA 2 style maps"""
    path = [outdir]

    if subdir is None:
        subdir = stagename.lower()
    path.append(subdir)

    if outdir is not None:
        mkdir(os.path.join(*path), warn=False)

    if stagename is not None:
        fname = ['%s_%s_comparisons' %(ref_abv.lower(), new_abv.lower()),
                 'stage_'+stagename]
    else:
        fname = ['%s_%s_comparisons' %(ref_abv.lower(), new_abv.lower())]
    if servicename is not None:
        fname.append('service_'+servicename)
    if name is not None:
        fname.append(name.lower())
    fname = '__'.join(fname) + '.' + ftype

    path.append(fname)

    basetitle = []
    if stagename is not None:
        basetitle.append('%s' % stagename)
    if texname is not None:
        basetitle.append(r'$%s$' % texname)
    basetitle = ' '.join(basetitle)

    ratio_map = make_ratio_map(new_map, ref_map)
    diff_map = make_delta_map(new_map, ref_map)
    diff_ratio_map = make_ratio_map(diff_map, ref_map)

    max_diff_ratio = np.nanmax(np.abs(diff_ratio_map['map']))

    # Handle cases where ratio returns infinite
    # This isn't necessarily a fail, since all it means is the referene was
    # zero If the new value is sufficiently close to zero then it's still fine
    if max_diff_ratio == float('inf'):
        logging.warning(
            'Infinite value found in ratio tests. Difference tests '
            'now also being calculated'
        )
        # First find all the finite elements
        finite_map = np.isfinite(diff_ratio_map['map'])
        # Then find the nanmax of this, will be our new test value
        max_diff_ratio = np.nanmax(np.abs(diff_ratio_map['map'][finite_map]))
        # Also find all the infinite elements
        infinite_map = np.logical_not(finite_map)
        # This will be a second test value
        max_diff = np.nanmax(np.abs(diff_map['map'][infinite_map]))
    else:
        # Without any infinite elements we can ignore this second test
        max_diff = 0.0

    if outdir is not None:
        gridspec_kw = dict(left=0.03, right=0.968, wspace=0.32)
        fig, axes = plt.subplots(nrows=1, ncols=5, gridspec_kw=gridspec_kw,
                                 sharex=False, sharey=False, figsize=(20, 5))
        if shorttitles:
            baseplot(m=ref_map,
                     title=basetitle+' '+ref_abv+' (A)',
                     evtrate=True,
                     ax=axes[0])
            baseplot(m=new_map,
                     title=basetitle+' '+new_abv+' (B)',
                     evtrate=True,
                     ax=axes[1])
            baseplot(m=ratio_map,
                     title='A/B',
                     ax=axes[2])
            baseplot(m=diff_map,
                     title='A-B',
                     symm=True, ax=axes[3])
            baseplot(m=diff_ratio_map,
                     title='(A-B)/A',
                     symm=True,
                     ax=axes[4])
        else:
            baseplot(m=ref_map,
                     title=basetitle+' '+ref_abv,
                     evtrate=True,
                     ax=axes[0])
            baseplot(m=new_map,
                     title=basetitle+' '+new_abv,
                     evtrate=True,
                     ax=axes[1])
            baseplot(m=ratio_map,
                     title=basetitle+' %s/%s' %(new_abv, ref_abv),
                     ax=axes[2])
            baseplot(m=diff_map,
                     title=basetitle+' %s-%s' %(new_abv, ref_abv),
                     symm=True, ax=axes[3])
            baseplot(m=diff_ratio_map,
                     title=basetitle+' (%s-%s)/%s' %(new_abv, ref_abv, ref_abv),
                     symm=True,
                     ax=axes[4])
        logging.debug('>>>> Plot for inspection saved at %s'
                      %os.path.join(*path))
        fig.savefig(os.path.join(*path))
        plt.close(fig.number)

    return max_diff_ratio, max_diff


def plot_map_comparisons(ref_map, new_map, ref_abv, new_abv, outdir, subdir,
                         name, texname, stagename, servicename,
                         shorttitles=False, ftype='png'):
    """Plot comparisons between two identically-binned PISA 3 style maps"""
    path = [outdir]

    if subdir is None:
        subdir = stagename.lower()
    path.append(subdir)

    if outdir is not None:
        mkdir(os.path.join(*path), warn=False)

    if stagename is not None:
        fname = ['%s_%s_comparisons' %(ref_abv.lower(), new_abv.lower()),
                 'stage_'+stagename]
    else:
        fname = ['%s_%s_comparisons' %(ref_abv.lower(), new_abv.lower())]
    if servicename is not None:
        fname.append('service_'+servicename)
    if name is not None:
        fname.append(name.lower())
    fname = '__'.join(fname) + '.' + ftype

    path.append(fname)

    basetitle = []
    if stagename is not None:
        basetitle.append('%s' % stagename)
    if texname is not None:
        basetitle.append(r'$%s$' % texname)
    basetitle = ' '.join(basetitle)

    validate_map_objs(new_map, ref_map)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_map = new_map/ref_map
    diff_map = new_map - ref_map
    with np.errstate(divide='ignore', invalid='ignore'):
        diff_ratio_map = diff_map/ref_map

    max_diff_ratio = np.nanmax(np.abs(diff_ratio_map.hist))

    # Handle cases where ratio returns infinite
    # This isn't necessarily a fail, since all it means is the referene was
    # zero If the new value is sufficiently close to zero then it's still fine
    if max_diff_ratio == float('inf'):
        logging.warning(
            'Infinite value found in ratio tests. Difference tests '
            'now also being calculated'
        )
        # First find all the finite elements
        finite_map = np.isfinite(diff_ratio_map.hist)
        # Then find the nanmax of this, will be our new test value
        max_diff_ratio = np.nanmax(np.abs(diff_ratio_map.hist[finite_map]))
        # Also find all the infinite elements
        infinite_map = np.logical_not(finite_map)
        # This will be a second test value
        max_diff = np.nanmax(np.abs(diff_map.hist[infinite_map]))
    else:
        # Without any infinite elements we can ignore this second test
        max_diff = 0.0

    if outdir is not None:
        gridspec_kw = dict(left=0.03, right=0.968, wspace=0.32)
        fig, axes = plt.subplots(nrows=1, ncols=5, gridspec_kw=gridspec_kw,
                                 sharex=False, sharey=False, figsize=(20, 5))
        if shorttitles:
            ref_map.plot(
                fig=fig,
                ax=axes[0],
                title=basetitle+' '+ref_abv+' (A)',
                cmap=plt.cm.afmhot
            )
            new_map.plot(
                fig=fig,
                ax=axes[1],
                title=basetitle+' '+new_abv+' (B)',
                cmap=plt.cm.afmhot
            )
            ratio_map.plot(
                fig=fig,
                ax=axes[2],
                title='A/B',
                cmap=plt.cm.afmhot
            )
            diff_map.plot(
                fig=fig,
                ax=axes[3],
                title='A-B',
                symm=True,
                cmap=plt.cm.seismic
            )
            diff_ratio_map.plot(
                fig=fig,
                ax=axes[4],
                title='(A-B)/A',
                symm=True,
                cmap=plt.cm.seismic
            )
        else:
            ref_map.plot(
                fig=fig,
                ax=axes[0],
                title=basetitle+' '+ref_abv,
                cmap=plt.cm.afmhot
            )
            new_map.plot(
                fig=fig,
                ax=axes[1],
                title=basetitle+' '+new_abv,
                cmap=plt.cm.afmhot
            )
            ratio_map.plot(
                fig=fig,
                ax=axes[2],
                title=basetitle+' %s/%s' %(new_abv, ref_abv),
                cmap=plt.cm.afmhot
            )
            diff_map.plot(
                fig=fig,
                ax=axes[3],
                title=basetitle+' %s-%s' %(new_abv, ref_abv),
                symm=True,
                cmap=plt.cm.seismic
            )
            diff_ratio_map.plot(
                fig=fig,
                ax=axes[4],
                title=basetitle+' (%s-%s)/%s' %(new_abv, ref_abv, ref_abv),
                symm=True,
                cmap=plt.cm.seismic
            )
        logging.debug('>>>> Plot for inspection saved at %s'
                      %os.path.join(*path))
        fig.savefig(os.path.join(*path))
        plt.close(fig.number)

    return max_diff_ratio, max_diff


def plot_cmp(new, ref, new_label, ref_label, plot_label, file_label, outdir,
             ftype='png'):
    """Plot comparisons between two (identically-binned) maps or map sets.

    Parameters
    ----------
    new : Map or MapSet
    ref : Map or MapSet
    new_label : str
    ref_label : str
    plot_label : str
    file_label : str
    outdir : str
    ftype : str

    """
    path = [outdir]

    if isinstance(ref, Map):
        assert isinstance(new, Map)
        ref_maps = [ref]
        new_maps = [new]

    if outdir is not None:
        mkdir(os.path.join(*path), warn=False)

    for ref, new in zip(ref_maps, new_maps):
        assert ref.binning == new.binning
        fname = get_valid_filename(
            '__'.join([
                get_valid_filename(file_label),
                '%s_vs_%s' %(get_valid_filename(new_label.lower()),
                             get_valid_filename(ref_label.lower()))
            ]) + '.' + ftype
        )
        path.append(fname)

        ratio = new / ref
        diff = new - ref
        fract_diff = diff / ref

        finite_ratio = ratio.hist[np.isfinite(ratio.hist)]
        ratio_mean = np.mean(finite_ratio)
        ratio_median = np.median(finite_ratio)

        finite_diff = diff.hist[np.isfinite(diff.hist)]
        diff_mean = np.mean(finite_diff)
        diff_median = np.median(finite_diff)

        finite_fract_diff = fract_diff.hist[np.isfinite(fract_diff.hist)]
        fract_diff_mean = np.mean(finite_fract_diff)
        fract_diff_median = np.median(finite_fract_diff)

        max_diff_ratio = np.nanmax(fract_diff.hist)

        # Handle cases where ratio returns infinite
        # This isn't necessarily a fail, since all it means is the referene was
        # zero. If the new value is sufficiently close to zero then it's stil
        # fine.
        if max_diff_ratio == np.inf:
            logging.warning(
                'Infinite value found in ratio tests. Difference tests'
                ' now also being calculated'
            )
            # First find all the finite elements
            finite_mask = np.isfinite(fract_diff.hist)
            # Then find the nanmax of this, will be our new test value
            max_diff_ratio = np.nanmax(fract_diff.hist[finite_mask])
            # Also find all the infinite elements; compute a second test value
            max_diff = np.nanmax(diff.hist[~finite_mask])
        else:
            # Without any infinite elements we can ignore this second test
            max_diff = 0.0

        if outdir is not None:
            if new.binning.num_dims == 2:
                n_dims = 2
                n_third_dim_bins = 1
            elif new.binning.num_dims == 3:
                n_dims = 3
                odd_dim_idx = new.binning.shape.index(np.min(new.binning.shape))
                logging.debug('odd_dim_idx: %s', odd_dim_idx)
                n_third_dim_bins = new.binning.shape[odd_dim_idx]

            gridspec_kw = dict(left=0.03, right=0.968, wspace=0.32)
            fig, axes = plt.subplots(nrows=n_third_dim_bins, ncols=5,
                                     gridspec_kw=gridspec_kw,
                                     squeeze=False, sharex=False, sharey=False,
                                     figsize=(20, 5))

            refslice = ref
            newslice = new
            bin_names = None
            if n_dims == 3:
                if odd_dim_idx != 0:
                    refslice = np.moveaxis(ref, source=odd_dim_idx,
                                           destination=0)
                    newslice = np.moveaxis(new, source=odd_dim_idx,
                                           destination=0)
                bin_names = new.binning.dims[odd_dim_idx].bin_names

            for odd_bin_idx in range(n_third_dim_bins):
                if n_dims == 2:
                    thisbin_ref = refslice
                    thisbin_new = newslice
                    tmp_ref_label = ref_label
                    tmp_new_label = new_label

                elif n_dims == 3:
                    thisbin_ref = refslice[odd_bin_idx, ...].squeeze()
                    thisbin_new = newslice[odd_bin_idx, ...].squeeze()

                    if bin_names is not None:
                        suffix = bin_names[odd_bin_idx]
                    else:
                        suffix = format(odd_bin_idx, 'd')
                    tmp_new_label = new_label + ' ' + suffix
                    tmp_ref_label = ref_label + ' ' + suffix

                    ratio = thisbin_new / thisbin_ref
                    diff = thisbin_new - thisbin_ref
                    fract_diff = diff / thisbin_ref

                refmax = np.nanmax(thisbin_ref.hist)
                newmax = np.nanmax(thisbin_new.hist)
                vmax = refmax if refmax > newmax else newmax

                baseplot2(map=thisbin_new,
                          title=tmp_new_label,
                          vmax=vmax,
                          evtrate=True,
                          ax=axes[odd_bin_idx][0])

                baseplot2(map=thisbin_ref,
                          title=tmp_ref_label,
                          vmax=vmax,
                          evtrate=True,
                          ax=axes[odd_bin_idx][1])

                ax, _, _ = baseplot2(map=ratio,
                                     title='%s/%s' %(tmp_new_label,
                                                     tmp_ref_label),
                                     ax=axes[odd_bin_idx][2])
                ax.text(0.95, 0.95, "Mean: %.6f"%ratio_mean,
                        horizontalalignment='right',
                        transform=ax.transAxes, color=(0, 0.8, 0.8))
                ax.text(0.95, 0.91, "Median: %.6f"%ratio_median,
                        horizontalalignment='right',
                        transform=ax.transAxes, color=(0, 0.8, 0.8))

                ax, _, _ = baseplot2(map=diff,
                                     title='%s-%s' %(tmp_new_label,
                                                     tmp_ref_label),
                                     symm=True, ax=axes[odd_bin_idx][3])
                ax.text(0.95, 0.95, "Mean: %.6f"%diff_mean,
                        horizontalalignment='right',
                        transform=ax.transAxes)
                ax.text(0.95, 0.91, "Median: %.6f"%diff_median,
                        horizontalalignment='right',
                        transform=ax.transAxes)

                ax, _, _ = baseplot2(map=fract_diff,
                                     title='(%s-%s)/%s' %(tmp_new_label,
                                                          tmp_ref_label,
                                                          tmp_ref_label),
                                     symm=True,
                                     ax=axes[odd_bin_idx][4])
                ax.text(0.95, 0.95, "Mean: %.6f"%fract_diff_mean,
                        horizontalalignment='right',
                        transform=ax.transAxes)
                ax.text(0.95, 0.91, "Median: %.6f"%fract_diff_median,
                        horizontalalignment='right',
                        transform=ax.transAxes)

            logging.debug('>>>> Plot for inspection saved at %s'
                          %os.path.join(*path))
            fig.savefig(os.path.join(*path))
            plt.close(fig.number)

        return max_diff_ratio, max_diff


def pisa2_map_to_pisa3_map(pisa2_map, ebins_name='ebins', czbins_name='czbins',
                           is_log=True, is_lin=True):
    expected_keys = ['map', 'ebins', 'czbins']
    if sorted(pisa2_map.keys()) != sorted(expected_keys):
        raise ValueError('PISA 2 map should be a dict containining entries: %s'
                         %expected_keys)
    ebins = OneDimBinning(
        name=ebins_name,
        bin_edges=pisa2_map['ebins'] * ureg.GeV,
        is_log=is_log,
        tex=r'E_{\nu}'
    )
    czbins = OneDimBinning(
        name=czbins_name,
        bin_edges=pisa2_map['czbins'],
        is_lin=is_lin,
        tex=r'\cos\theta_Z'
    )
    bins = MultiDimBinning([ebins, czbins])
    return Map(
        name='pisa2equivalent',
        hist=pisa2_map['map'],
        binning=bins
    )
