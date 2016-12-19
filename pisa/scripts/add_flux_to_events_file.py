#! /usr/bin/env python
"""
Add neutrino fluxes (and neutrino weights(osc*flux*sim_weight) if needed) for
each event.
"""


import os
import sys
import numpy as np

import pisa.core.events as events
from pisa.utils.log import logging, set_verbosity
from pisa.utils.fileio import from_file, to_file, mkdir
import pisa.utils.resources as resources
from pisa.utils.flux_weights import load_2D_table, calculate_flux_weights

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


__all__ = ['add_fluxes_to_file', 'main']


def add_fluxes_to_file(data_file_path, flux_table, neutrino_weight_name, outdir):
    data_file, attrs = from_file(resources.find_resource(data_file_path), return_attrs = True)
    data_file_name = os.path.basename(data_file_path)
    mkdir(outdir)
    output_file_name = outdir + '/' + data_file_name.split('.hdf5')[0]+'_with_fluxes.hdf5'
    if not os.path.isfile(output_file_name):
        for prim in data_file.keys():
            for int_type in data_file[prim].keys():
                true_e = data_file[prim][int_type]['true_energy']
                true_cz = data_file[prim][int_type]['true_coszen']
                isbar = 'bar' if 'bar' in prim else ''
                nue_flux = calculate_flux_weights(true_e, true_cz, flux_table['nue'+isbar])
                numu_flux = calculate_flux_weights(true_e, true_cz, flux_table['numu'+isbar])
                # the opposite flavor fluxes( used only in the nu_nubar_ratio systematic)
                oppo_isbar = '' if 'bar' in prim else 'bar'
                oppo_nue_flux = calculate_flux_weights(true_e, true_cz, flux_table['nue'+isbar])
                oppo_numu_flux = calculate_flux_weights(true_e, true_cz, flux_table['numu'+isbar])
                data_file[prim][int_type][neutrino_weight_name+'_nue_flux'] = nue_flux
                data_file[prim][int_type][neutrino_weight_name+'_numu_flux'] = numu_flux
                data_file[prim][int_type][neutrino_weight_name+'_oppo_nue_flux'] = oppo_nue_flux
                data_file[prim][int_type][neutrino_weight_name+'_oppo_numu_flux'] = oppo_numu_flux
                # if need to calculate neutrino weights here
        to_file(data_file, output_file_name, attrs=attrs, overwrite=True)
    else:
        print 'File %s already exists, skipped. Please delete it or rename it.' % output_file_name


def main():
    parser = ArgumentParser(description=__doc__)
    parser_file = parser.add_mutually_exclusive_group(required=True)
    parser_file.add_argument( '-f', '--file', metavar='H5_FILE', type=str, help='input HDF5 file')
    parser_file.add_argument( '--flux-file', metavar='FLUX_FILE', type=str, help='input flux file',default='flux/honda-2015-spl-solmin-aa.d')
    parser.add_argument('-o','--outdir',metavar='DIR',default='',
                        help='Directory to save the output figures.')
    parser.add_argument('-v', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()
    set_verbosity(args.v)
    # flux and osc service
    flux_table = load_2D_table(args.flux_file)
    add_fluxes_to_file(args.file, flux_table, neutrino_weight_name='neutrino', outdir=args.outdir)


if __name__ == '__main__':
    main()
