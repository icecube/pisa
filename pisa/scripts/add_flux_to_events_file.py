#! /usr/bin/env python
"""
Add neutrino fluxes (and neutrino weights(osc*flux*sim_weight) if needed) for
each event.
"""


from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
import glob
from os import listdir
from os.path import basename, isdir, isfile, join, splitext

from pisa.utils.fileio import from_file, to_file, mkdir, nsort
from pisa.utils.flux_weights import load_2d_table, calculate_2d_flux_weights
from pisa.utils.hdf import HDF5_EXTS
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource


__all__ = ['add_fluxes_to_file', 'main']


# TODO: Should output filename include which flux file was used, so that info
#       doesn't get lost? Or is this recorded in some other way?


def add_fluxes_to_file(data_file_path, flux_table, neutrino_weight_name,
                       outdir, overwrite=False):
    """Add fluxes to PISA events file (e.g. for use by an mc stage)

    Parameters
    -----------
    data_file_path
    flux_table
    neutrino_weight_name
    outdir
    overwrite : bool, optional

    """
    data, attrs = from_file(find_resource(data_file_path), return_attrs=True)
    basename, ext = splitext(basename(data_file_path))
    assert ext.lstrip('.') in HDF5_EXTS
    outpath = join(outdir, '{}_with_fluxes{}'.format(basename, ext))

    mkdir(outdir, warn=False)

    if not overwrite and isfile(outpath):
        logging.warning('Output path "%s" already exists, not regenerating',
                        outpath)
        return

    for primary, primary_node in data.items():
        for int_type, int_node in primary_node.items():
            true_e = int_node['true_energy']
            true_cz = int_node['true_coszen']

            # NOTE: The opposite-flavor fluxes are used only in the
            #       nu_nubar_ratio systematic

            for opposite in (False, True):
                if not opposite:
                    bar_label = 'bar' if 'bar' in primary else ''
                    oppo_label = ''
                else:
                    bar_label = '' if 'bar' in primary else 'bar'
                    oppo_label = '_oppo'

                nue_flux = calculate_2d_flux_weights(
                    true_energies=true_e,
                    true_coszens=true_cz,
                    en_splines=flux_table['nue' + bar_label]
                )
                numu_flux = calculate_2d_flux_weights(
                    true_energies=true_e,
                    true_coszens=true_cz,
                    en_splines=flux_table['numu' + bar_label]
                )

                basekey = neutrino_weight_name + oppo_label
                int_node[basekey + '_nue_flux'] = nue_flux
                int_node[basekey + '_numu_flux'] = numu_flux

                # TODO: if need to calculate neutrino weights here

    to_file(data, outpath, attrs=attrs, overwrite=overwrite)


def main():
    """Parse command-line arguments and execute `add_fluxes_to_file` function"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input', metavar='(H5_FILE|DIR)', nargs='+', type=str, required=True,
        help='''Path to a PISA events HDF5 file or a directory containing HDF5
        files; output files are copies of this/these, but with flux fields
        added.''' 
    )
    parser.add_argument(
        '--flux-file', metavar='FLUX_FILE', type=str, required=True,
        help='''Flux file from which to obtain fluxes, e.g.
        "flux/honda-2015-spl-solmin-aa.d"'''
    )
    parser.add_argument(
        '--outdir', metavar='DIR', required=True,
        help='Directory to save the output figures.'
    )
    parser.add_argument(
        '-v', action='count', default=0,
        help='set verbosity level'
    )
    args = parser.parse_args()

    set_verbosity(args.v)

    flux_table = load_2d_table(args.flux_file)

    input_paths = []
    for input_path in args.input:
        if isdir(input_path):
            for filename in listdir(input_path):
                filepath = join(input_path, filename)
                input_paths.append(filepath)

        else:
            input_paths += glob.glob(input_path)

    input_paths = nsort(input_paths)

    paths_to_process = []
    basenames = []
    for input_path in input_paths:
        if isdir(input_path):
            logging.debug('Path "%s" is a directory, skipping', file_path)
            continue

        firstpart, ext = splitext(input_path)
        if ext.lstrip('.') not in HDF5_EXTS:
            logging.debug('Path "%s" is a directory, skipping', file_path)
            continue

        basename = basename(firstpart)
        if basename in basenames:
            raise ValueError(
                'Found files with duplicate basename "%s" (despite files'
                ' having different paths); resolve the ambiguous names and'
                ' re-run. Offending files are:\n  "%s"\n  "%s"'
                % (basename,
                   paths_to_process[basenames.index(basename)],
                   input_path)
            )

        basenames.append(basename)
        paths_to_process.append(input_path)

    logging.info('Will process %d input file(s)...', len(paths_to_process))

    for filepath in paths_to_process:
        logging.info('Working on input file "%s"', filepath)
        add_fluxes_to_file(data_file_path=filepath, flux_table=flux_table,
                           neutrino_weight_name='neutrino', outdir=args.outdir)


if __name__ == '__main__':
    main()
