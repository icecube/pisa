#! /usr/bin/env python

"""
Add neutrino fluxes (and neutrino weights(osc*flux*sim_weight) if needed) for
each event.
"""


from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
import glob
from os import listdir
from os.path import basename, dirname, isdir, isfile, join, splitext

from pisa.utils.fileio import from_file, to_file, mkdir, nsort
from pisa.utils.flux_weights import load_2d_table, calculate_2d_flux_weights
from pisa.utils.hdf import HDF5_EXTS
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource


__all__ = ['add_fluxes_to_file', 'main']

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

def add_fluxes_to_file(data_file_path, flux_table, flux_name,
                       outdir=None, label=None, overwrite=False):
    """Add fluxes to PISA events file (e.g. for use by an mc stage)
    
    Parameters
    -----------
    data_file_path : string
    flux_table
    flux_name
    outdir : string or None
        If None, output is to the same directory as `data_file_path`
    overwrite : bool, optional
    """
    data, attrs = from_file(find_resource(data_file_path), return_attrs=True)
    bname, ext = splitext(basename(data_file_path))
    assert ext.lstrip('.') in HDF5_EXTS

    if outdir is None:
        outdir = dirname(data_file_path)

    if label is None:
        label = ''
    else:
        assert isinstance(label, str)
        label = '_' + label

    outpath = join(outdir, '{}__with_fluxes{}{}'.format(bname, label, ext))

    if not overwrite and isfile(outpath):
        logging.warning('Output path "%s" already exists, not regenerating',
                        outpath)
        return

    mkdir(outdir, warn=False)

    # Loop over the top-level keys
    for primary, primary_node in data.items():

        # Only handling neutrnio fluxes here, skip past e.g. muon or noise MC events
        if primary.startswith("nu") :

            logging.info('Adding fluxes to "%s" events', primary)

            # Input data may have one layer of hierarchy before the event variables (e.g. [numu_cc]), 
            # or for older files there maybe be a second layer (e.g. [numu][cc]).
            # Handling either case here...
            if "true_energy" in primary_node :
                secondary_nodes = [primary_node]
            else :
                secondary_nodes = primary_node.values()

            for secondary_node in secondary_nodes :

                true_e = secondary_node['true_energy']
                true_cz = secondary_node['true_coszen']

                # calculate all 4 fluxes (nue, nuebar, numu and numubar)
                for table in ['nue', 'nuebar', 'numu', 'numubar']:
                    flux = calculate_2d_flux_weights(
                        true_energies=true_e,
                        true_coszens=true_cz,
                        en_splines=flux_table[table]
                    )
                    keyname = flux_name + '_' + table + '_flux'
                    secondary_node[keyname] = flux

    to_file(data, outpath, attrs=attrs, overwrite=overwrite)
    logging.info('--> Wrote file including fluxes to "%s"', outpath)


def parse_args(description=__doc__):
    """Parse command-line arguments"""
    parser = ArgumentParser(description=description)
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
        '--outdir', metavar='DIR', default=None,
        help='''Directory to save the output to; if none is provided, output is
        placed in same dir as --input.'''
    )
    parser.add_argument(
        '--label', type=str, default=None,
        help='''Label to give output files. If a label is not specified,
        default label is the flux file's basename with extension removed.'''
    )
    parser.add_argument(
        '-v', action='count', default=1,
        help='''Increase verbosity level _beyond_ INFO level by specifying -v
        (DEBUG) or -vv (TRACE)'''
    )
    return parser.parse_args()


def main():
    """Run `add_fluxes_to_file` function with arguments from command line"""
    args = parse_args()
    set_verbosity(args.v)

    flux_table = load_2d_table(args.flux_file)
    flux_file_bname, ext = splitext(basename(args.flux_file))

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
            logging.debug('Path "%s" is a directory, skipping', input_path)
            continue

        firstpart, ext = splitext(input_path)
        if ext.lstrip('.') not in HDF5_EXTS:
            logging.debug('Path "%s" is not an HDF5 file, skipping', input_path)
            continue

        bname = basename(firstpart)
        if bname in basenames:
            raise ValueError(
                'Found files with duplicate basename "%s" (despite files'
                ' having different paths); resolve the ambiguous names and'
                ' re-run. Offending files are:\n  "%s"\n  "%s"'
                % (bname,
                   paths_to_process[basenames.index(bname)],
                   input_path)
            )

        basenames.append(bname)
        paths_to_process.append(input_path)

    logging.info('Will process %d input file(s)...', len(paths_to_process))

    for filepath in paths_to_process:
        logging.info('Working on input file "%s"', filepath)
        add_fluxes_to_file(
            data_file_path=filepath,
            flux_table=flux_table,
            flux_name='nominal',
            outdir=args.outdir,
            label=flux_file_bname
        )


if __name__ == '__main__':
    main()
