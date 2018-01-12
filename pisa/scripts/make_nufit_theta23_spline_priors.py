#! /usr/bin/env python

"""
Create splines to the NuFit delta-chi2 surfaces for theta23 and output them in
a format that can be read by PISA to use as a prior on this parameter.
"""


from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
import gzip
import os
import sys

import numpy as np
import scipy.interpolate

from pisa.utils.fileio import to_file


__all__ = ['extract_vals', 'make_prior_dict', 'main']

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


def extract_vals(infile, string_of_interest):
    readout = False
    x = []
    y = []
    for line in infile:
        if line.strip().startswith('#'):
            if line.strip() == string_of_interest:
                readout = True
            else:
                readout = False
        else:
            if readout:
                vals = line.strip().split(' ')
                if len(vals) == 2:
                    x.append(float(vals[0]))
                    y.append(float(vals[1]))

    return x, y


def make_prior_dict(f_io=None, f_no=None, f=None):

    if f is not None:
        priors = {}
        priors["theta23"] = {}
        priors["theta23"]["coeffs"] = f[1]
        priors["theta23"]["deg"] = f[2]
        priors["theta23"]["knots"] = f[0]
        priors["theta23"]["kind"] = "spline"
        priors["theta23"]["units"] = "radian"

    elif f_io is not None and f_no is not None:
        priors = {}
        priors["theta23_ih"] = {}
        priors["theta23_ih"]["coeffs"] = f_io[1]
        priors["theta23_ih"]["deg"] = f_io[2]
        priors["theta23_ih"]["knots"] = f_io[0]
        priors["theta23_ih"]["kind"] = "spline"
        priors["theta23_ih"]["units"] = "radian"
        priors["theta23_nh"] = {}
        priors["theta23_nh"]["coeffs"] = f_no[1]
        priors["theta23_nh"]["deg"] = f_no[2]
        priors["theta23_nh"]["knots"] = f_no[0]
        priors["theta23_nh"]["kind"] = "spline"
        priors["theta23_nh"]["units"] = "radian"

    else:
        raise ValueError('No functions passed to save!')

    return priors


def main():
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-io','--io_chi2_file',type=str,required=True,
                        help="Inverted Ordering Chi2 file from NuFit")
    parser.add_argument('-no','--no_chi2_file',type=str,required=True,
                        help="Inverted Ordering Chi2 file from NuFit")
    parser.add_argument('--shifted', action='store_true', default=False,
                        help='''Flag if wanting prior which attempts to remove
                        the ordering prior by subtracting the delta chi2.''')
    parser.add_argument('--minimised', action='store_true', default=False,
                        help='''Flag if wanting prior which attempts to remove
                        the ordering prior by minimising over both surfaces.''')
    parser.add_argument('--outdir', metavar='DIR', type=str, required=True,
                        help='''Store all output files to this directory. It
                        is recommended you save them in the priors directory
                        in the PISA resources.''')

    args = parser.parse_args()

    io_filename, io_fileext = os.path.splitext(args.io_chi2_file)
    no_filename, no_fileext = os.path.splitext(args.no_chi2_file)

    if io_fileext != '.gz':
        raise ValueError('%s file extension not expected. Please use the file '
                         'as downloaded from the Nu-Fit website.'%io_fileext)
    if no_fileext != '.gz':
        raise ValueError('%s file extension not expected. Please use the file as '
                         'downloaded directly from the Nu-Fit website.'%no_fileext)

    # Get Nu-Fit version from filenames
    NuFitVersion = io_filename.split('/')[-1].split('.')[0]
    if NuFitVersion[0].lower() != 'v':
        raise ValueError('%s%s input file does not allow for discerning the '
                         'Nu-Fit version directly from the filename. Please '
                         'use the file as downloaded directly from the Nu-Fit '
                         'website.'%(io_filename,io_fileext))
    NO_NuFitVersion = no_filename.split('/')[-1].split('.')[0]
    if NuFitVersion != NO_NuFitVersion:
        raise ValueError('The NuFit version extracted from the NO and IO files '
                         'do not match. i.e. %s is not the same as %s. Please '
                         'use the same NuFit version for each of the NO and IO '
                         'chi2 surfaces.'
                         %(NuFitVersion,NO_NuFitVersion))

    # Add special treatment for NuFit 2.1 since it has two releases
    if NuFitVersion == 'v21':
        NuFitVersion += io_filename.split('/')[-1].split('-')[1]

    io_infile = gzip.open(args.io_chi2_file)
    no_infile = gzip.open(args.no_chi2_file)

    io_s2th23, io_dchi2 = extract_vals(
        infile=io_infile,
        string_of_interest='# T23 projection: sin^2(theta23) Delta_chi^2'
    )
    no_s2th23, no_dchi2 = extract_vals(
        infile=no_infile,
        string_of_interest='# T23 projection: sin^2(theta23) Delta_chi^2'
    )

    io_th23 = np.arcsin(np.sqrt(np.array(io_s2th23)))
    no_th23 = np.arcsin(np.sqrt(np.array(no_s2th23)))

    io_dchi2 = np.array(io_dchi2)
    no_dchi2 = np.array(no_dchi2)

    f_io = scipy.interpolate.splrep(io_th23,-io_dchi2/2.0,s=0)
    f_no = scipy.interpolate.splrep(no_th23,-no_dchi2/2.0,s=0)

    priors = make_prior_dict(f_io=f_io,
                             f_no=f_no)

    to_file(priors, os.path.join(args.outdir,
                                 'nufit%sstandardtheta23splines.json'%NuFitVersion))

    if args.shifted:
        # Make priors where the delta chi2 between the orderings is removed.
        # The idea is to remove the prior on the ordering.

        io_shifteddchi2 = io_dchi2 - min(io_dchi2)
        no_shifteddchi2 = no_dchi2 - min(no_dchi2)

        f_shiftedio = scipy.interpolate.splrep(io_th23,-io_shifteddchi2/2.0,s=0)
        f_shiftedno = scipy.interpolate.splrep(no_th23,-no_shifteddchi2/2.0,s=0)

        shiftedpriors = make_prior_dict(f_io=f_shiftedio,
                                        f_no=f_shiftedno)

        to_file(shiftedpriors,
                os.path.join(args.outdir,
                             'nufit%sshiftedtheta23splines.json'%NuFitVersion))

    if args.minimised:
        # Make one prior that is the minimum of both of the original chi2
        # surfaces. The idea is to remove the prior on the ordering.

        minchi2 = np.minimum(io_dchi2, no_dchi2)

        # Now just one prior. X values should be the same for both.
        f_minimised = scipy.interpolate.splrep(io_th23,-minchi2/2.0,s=0)

        minimisedprior = make_prior_dict(f=f_minimised)

        to_file(minimisedprior,
                os.path.join(args.outdir,
                             'nufit%sminimisedtheta23spline.json'%NuFitVersion))

main.__doc__ = __doc__


if __name__ == '__main__':
    main()
