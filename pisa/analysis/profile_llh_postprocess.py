#! /usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import sys

import numpy as np

from pisa.utils.fileio import from_file


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-d', '--dir', metavar='DIR', default='.',
        help='directory containg output json files'
    )
    parser.add_argument(
        '-f', '--file', metavar='FILE', default='',
        help='single json files'
    )
    parser.add_argument(
        '-p', '--plot', default='',
        help='Plot significance vs bdt score',
    )
    args = parser.parse_args()
    res = {}

    if args.file is not '':
        fnames = [args.file]
    else:
        fnames =  os.listdir(args.dir)

    x=[]
    y=[]
    for filename in fnames:
        if filename.endswith('.json'):
            file = from_file(args.dir+'/'+filename)
            name, _ = filename.split('.json')
            if args.plot:
                name = name.split('_')[2]
            assert(not file[0][0]['warnflag'][0] and not file[0][1]['warnflag'])
            if file[0][0].has_key('llh'):
                metric = 'llh'
            elif file[0][0].has_key('barlow_llh'):
                metric = 'barlow_llh'
            elif file[0][0].has_key('conv_llh'):
                metric = 'conv_llh'
            elif file[0][0].has_key('chi2'):
                metric = 'chi2'
            elif file[0][0].has_key('mod_chi2'):
                metric = 'mod_chi2'
            else:
                continue

            cond_llh = file[0][0][metric][0]
            glob_llh = file[0][1][metric]
            if 'chi2' in metric:
                signif = np.sqrt(cond_llh - glob_llh)
            else:
                signif = np.sqrt(2*(cond_llh - glob_llh))
            res[name] = signif
            if args.plot:
                x.append(eval(name))
                y.append(signif)
            #print '%s\t%.4f'%(name, signif)

    if args.plot:
        import matplotlib as mpl
        mpl.use('Agg')
        from matplotlib import pyplot as plt

        print "x, y = ", x, " ", y
        plt.figure()
        plt.plot(x, y, 'bo')
        plt.grid()
        plt.xlabel('BDT score')
        plt.ylabel("Signficance (Asimov data)")
        plt.savefig('signifi_vs_bdt.pdf')
        plt.savefig('signifi_vs_bdt.png')

    if res.has_key('nominal'):
        nominal = res['nominal']
        print '%i systematics'%(len(res)-1)
        print '%-20s\tsign\tdelta\tpercent'%'sys'
    else:
        nominal = None
    for key, val in sorted(res.items(), key=lambda s: s[1], reverse=True):
        if nominal is not None:
            print '%-20s\t%.3f\t%.4f\t%.2f %%' \
                    %(key, val, val-nominal, (val-nominal)/nominal*100)
        else:
            print '%s\t%.4f sigma'%(key, val)


if __name__ == '__main__':
    main()
