from copy import deepcopy

import numpy as np
import pint
from uncertainties import unumpy as unp

from pisa import ureg, Q_
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.utils.log import set_verbosity


if __name__ == '__main__':
    set_verbosity(1)

    # livetimes = [2, 3, 4, 5, 6, 7, 8] * ureg.common_year
    livetimes = [1, 4, 16, 64] * ureg.common_year
    # livetimes = [1, 2] * ureg.common_year

    template_maker = DistributionMaker('settings/pipeline/cfx.cfg')

    template_maker.params.fix(template_maker.params.free)
    template_maker.params.unfix('livetime')

    re_param = template_maker.params['regularisation']
    sf_param = template_maker.params['stat_fluctuations']
    lt_param = template_maker.params['livetime']

    mean = []
    for idx, lt in enumerate(livetimes):
        print '==========='
        print 'livetime = {0}'.format(lt)
        print '==========='
        mean.append([])

        lt_param.value = lt
        template_maker.update_params(lt_param)

        re_param.value = 0 * ureg.dimensionless
        template_maker.update_params(re_param)
        nom_out = template_maker.get_outputs(return_sum=True)[0].pop()

        re_param.value = 4 * ureg.dimensionless
        sf_param.value = 1234 * ureg.dimensionless
        template_maker.update_params(re_param)
        template_maker.update_params(sf_param)
        fe = []
        for x in xrange(200):
        # for x in xrange(2):
            temp_out = template_maker.get_outputs(return_sum=True)[0].pop()
            nan_mask = nom_out.hist < 0.0001
            div = temp_out.hist[~nan_mask] / nom_out.hist[~nan_mask]
            fe.append(div)
        for f in fe:
            mean[idx].append(np.mean(f))

    fe = zip(*mean)
    print fe

    import matplotlib as mpl
    # headless mode
    mpl.use('Agg')
    # fonts
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    from matplotlib import pyplot as plt
    from matplotlib.offsetbox import AnchoredText

    binning = livetimes.m
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.set_xlim(np.min(binning)-1, np.max(binning)+1)
    ax.set_ylim(0.5, 1.5)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_xlabel('livetime (years)', fontsize=18)
    ax.set_ylabel('mean ratio unfolded vs. truth (200 trials)', fontsize=15)
    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(y=ymaj, ls=':', color='gray', alpha=0.7, linewidth=1)
    for xmaj in ax.xaxis.get_majorticklocs():
        ax.axvline(x=xmaj, ls=':', color='gray', alpha=0.7, linewidth=1)

    def get_edges_from_cen(bincen):
        hwidth = 0.5*(bincen[1] - bincen[0])
        return np.append([bincen[0]-hwidth], bincen[:]+hwidth)

    for f in fe:
        fe_0 = np.concatenate([[f[0]], f])
        ax.errorbar(
            binning, unp.nominal_values(f), xerr=0,
            yerr=unp.std_devs(f), capsize=3, alpha=0.5, linestyle='--',
            markersize=2, linewidth=1
        )
    fig.savefig('./images/cfx/livetime_4.png', bbox_inches='tight', dpi=150)
