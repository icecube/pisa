#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This script/module computes significances, etc. from the logfiles recorded by
the `hypo_testing.py` script.

"""


from __future__ import division

from argparse import ArgumentParser
from collections import OrderedDict
import os

import numpy as np

from pisa import ureg
from pisa.analysis.hypo_testing import Labels
from pisa.utils.fileio import from_file, nsort
from pisa.utils.log import set_verbosity


__all__ = ['extract_trials', 'extract_fit', 'parse_args', 'main']


def extract_trials(logdir, fluctuate_fid, fluctuate_data=False):
    """Extract and aggregate analysis results.

    Parameters
    ----------
    logdir : string
        Path to logging directory where files are stored. This should contain
        e.g. the "config_summary.json" file.

    fluctuate_fid : bool
        Whether the trials you're interested in applied fluctuations to the
        fiducial-fit Asimov distributions. `fluctuate_fid` False is equivalent
        to specifying an Asimov analysis (so long as the metric used was
        chi-squared).

    fluctuate_data : bool
        Whether the trials you're interested in applied fluctuations to the
        (toy) data. This is invalid if actual data was processed.

    Note that a single `logdir` can have different kinds of analyses run and
    results be logged within, so `fluctuate_fid` and `fluctuate_data` allows
    these to be separated from one another.

    """
    logdir = os.path.expanduser(os.path.expandvars(logdir))
    config_summary_fpath = os.path.join(logdir, 'config_summary.json')
    cfg = from_file(config_summary_fpath)

    data_is_data = cfg['data_is_data']
    if data_is_data and fluctuate_data:
        raise ValueError('Analysis was performed on data, so `fluctuate_data`'
                         ' is not supported.')

    # Get naming scheme
    labels = Labels(
        h0_name=cfg['h0_name'], h1_name=cfg['h1_name'],
        data_name=cfg['data_name'], data_is_data=data_is_data,
        fluctuate_data=fluctuate_data, fluctuate_fid=fluctuate_fid
    )

    # Find all relevant data dirs, and from each extract the fiducial fit(s)
    # information contained
    data_sets = OrderedDict()
    for basename in nsort(os.listdir(logdir)):
        m = labels.subdir_re.match(basename)
        if m is None:
            continue

        if fluctuate_data:
            data_ind = int(m.groupdict()['data_ind'])
            dset_label = data_ind
        else:
            dset_label = labels.data_prefix
            if not labels.data_suffix in [None, '']:
                dset_label += '_' + labels.data_suffix

        lvl2_fits = OrderedDict()
        lvl2_fits['h0_fit_to_data'] = None
        lvl2_fits['h1_fit_to_data'] = None

        subdir = os.path.join(logdir, basename)
        for fnum, fname in enumerate(nsort(os.listdir(subdir))):
            fpath = os.path.join(subdir, fname)
            for x in ['0', '1']:
                k = 'h{x}_fit_to_data'.format(x=x)
                if fname == labels.dict[k]:
                    lvl2_fits[k] = extract_fit(fpath, 'metric_val')
                    break
                for y in ['0','1']:
                    k = 'h{x}_fit_to_h{y}_fid'.format(x=x, y=y)
                    r = labels.dict[k + '_re']
                    #print r.pattern
                    #return labels
                    m = r.match(fname)
                    if m is None:
                        continue
                    #sys.stdout.write('.')
                    if fluctuate_fid:
                        fid_label = int(m.groupdict()['fid_ind'])
                    else:
                        fid_label = labels.fid
                    if k not in lvl2_fits:
                        lvl2_fits[k] = OrderedDict()
                    lvl2_fits[k][fid_label] = extract_fit(fpath, 'metric_val')
                    break
        data_sets[dset_label] = lvl2_fits
    return data_sets


def extract_fit(fpath, keys=None):
    """Extract fit info from a file.

    Parameters
    ----------
    fpath : string
        Path to the file

    keys : None, string, or iterable of strings
        Keys to extract. If None, all keys are extracted.

    """
    info = from_file(fpath)
    if keys is None:
        return info
    if isinstance(keys, basestring):
        keys = [keys]
    for key in info.keys():
        if key not in keys:
            info.pop(key)
    return info


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '-d', '--dir', required=True,
        metavar='DIR', type=str,
        help='Directory into which to store results and metadata.'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--asimov', action='store_true',
        help='''Analyze the Asimov trials in the specified directories.'''
    )
    group.add_argument(
        '--llr', action='store_true',
        help='''Analyze the LLR trials in the specified directories.'''
    )

    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    init_args_d = vars(args)

    # NOTE: Removing extraneous args that won't get passed to instantiate the
    # HypoTesting object via dictionary's `pop()` method.

    set_verbosity(init_args_d.pop('v'))

    if args.asimov:
        data_sets = extract_trials(logdir=args.dir, fluctuate_fid=False,
                                   fluctuate_data=False)
        od = data_sets.values()[0]
        #if od['h1_fit_to_h0_fid']['fid_asimov']['metric_val'] > od['h0_fit_to_h1_fid']['fid_asimov']['metric_val']:
        print np.sqrt(np.abs(od['h1_fit_to_h0_fid']['fid_asimov']['metric_val'] - od['h0_fit_to_h1_fid']['fid_asimov']['metric_val']))

    else:
        raise NotImplementedError('llr-analysis')


if __name__ == '__main__':
    main()
