from BargerPropagator import BargerPropagator
import matplotlib.ticker
import matplotlib as mpl
from matplotlib.offsetbox import AnchoredText
mpl.use('Agg')
from math import pi
from matplotlib import pyplot as plt
#from matplotlib import rcParams
#rcParams['font.family'] = 'sans-serif'
#rcParams['font.sans-serif'] = ['Tahoma']
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rc('font', **{'sans-serif': 'Computer Modern Sans Serif'})
mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{sansmath}',
    r'\SetSymbolFont{operators}   {sans}{OT1}{cmss} {m}{n}'
    r'\SetSymbolFont{letters}     {sans}{OML}{cmbrm}{m}{it}'
    r'\SetSymbolFont{symbols}     {sans}{OMS}{cmbrs}{m}{n}'
    r'\SetSymbolFont{largesymbols}{sans}{OMX}{iwona}{m}{n}'
    r'\sansmath'
]

kSquared  = True

#kNuBar  =  -1
kNuBar  = 1
#dm2     =  7.59e-5#7.50e-5
#DM2     =  2.42e-3#2.457e-3-dm2
#Theta23 =  0.49#0.452
#Theta13 =  0.02513#0.0218
#Theta12 =  0.314# 0.304
#delta   =  0# 306 * (pi/180.0)

dm2     =  7.50e-5#7.50e-5
DM2     =  2.524e-3#2.457e-3-dm2
Theta23 =  0.441#0.587
Theta13 =  0.02166#0.0218
Theta12 =  0.306# 0.306
delta   =  261*pi/180.0 # 306 * (pi/180.0)

#energy = 5.46281496742
#coszen = -0.414607620378
energy = 9.0
coszen = -1.0

# electron densities
YeI = 0.4656
YeM = 0.4957
YeO = 0.4656
# height
detector_depth = 2.0
prop_height = 20.0



total_prob=0.0



print 'Using          '
print '      DM2      ', DM2
print '      Theta23  ', Theta23
print '      Theta13  ', Theta13
print '      dm2      ', dm2
print '      Theta12  ', Theta12
print '      deltaCP  ', delta
print '---------------'
print '      type     ', kNuBar
print '      energy   ', energy
print '      coszen   ', coszen

#bNu = BargerPropagator('PREM_vac.dat' )
bNu = BargerPropagator('../../../resources/osc/PREM_12layer.dat', detector_depth )
bNu.UseMassEigenstates( False )
bNu.SetOneMassScaleMode( False )
bNu.SetWarningSuppression( True )

import numpy as np

arr_e = np.zeros(1000)
arr_nue = np.zeros(1000)
arr_numu = np.zeros(1000)
arr_nutau = np.zeros(1000)

i = 0

for energy in np.logspace(np.log10(5), np.log10(500), 1000):
    bNu.SetMNS( Theta12,  Theta13, Theta23, dm2, DM2, delta , energy, kSquared, kNuBar )
    #bNu.DefinePath( coszen , prop_height, YeI, YeO, YeM, 20.00 )
    bNu.DefinePath( coszen , prop_height, YeI, YeO, YeM)
    bNu.propagate( 1*kNuBar )

    total_prob = 0.0
    for m in xrange(0,3):
        total_prob += bNu.GetProb(1, m)

    if ( total_prob >1.00001 or total_prob<0.99998 ):
        print 'ERROR Prob: Energy: ', energy , ' ', total_prob
        raise Exception('step 1, total_prob not 1')

    arr_e[i] = energy
    arr_nue[i]   = bNu.GetProb(1,0)
    arr_numu[i]  = bNu.GetProb(1,1)
    arr_nutau[i] = bNu.GetProb(1,2)
    i += 1

fig, ax = plt.subplots()
labels={}
labels['nue']=r'$\nu_e'
labels['numu']=r'$\nu_{\mu}'
labels['nutau']=r'$\nu_{\tau}'
labels['nue'] = r'$\nu_\mu \rightarrow \nu_e$'
labels['numu'] = r'$\nu_\mu \rightarrow \nu_{\mu}$'
labels['nutau'] = r'$\nu_\mu \rightarrow \nu_{\tau}$'
ax.plot(arr_e, arr_numu , color='blue', label=labels['numu'])
ax.plot(arr_e, arr_nutau, color='magenta', label=labels['nutau'])
ax.plot(arr_e, arr_nue  , color='green', label=labels['nue'])
ax.set_xscale('log')
tick_x = [5, 10, 20, 30, 50, 100, 200, 500]
tick_x_label = ['5', '10', '20', '30', '50', '100', '200', '500']
plt.xticks(tick_x, tick_x_label)
plt.title(r'$\rm{Upgoing \ \nu \ [\cos(zen) = -1]}$', fontsize=14)
ax.tick_params(axis='x', which='both', labelsize=12)
ax.tick_params(axis='y', which='both', labelsize=12)
ax.legend(loc=(0.765, 0.72),fancybox=True,ncol=1,frameon=True, columnspacing=0.8, handlelength=1.5, prop={'size':12}, bbox_transform=ax.transAxes)

labelsize=14
ax.set_xlim([5,500])
ax.set_ylim([0,1])
ax.set_ylabel(r'$\rm{P (\nu_\mu \rightarrow \nu_x)}$', fontsize=labelsize)
ax.set_xlabel(r'$\rm{True \ \nu \ energy \ (GeV)}$', fontsize=labelsize)

params_text = r'$\rm{{{\Delta m}^2}_{31} = 2.524 \times 10^{-3} \ eV^2}$'+'\n'
params_text += r'$\rm{{{\Delta m}^2}_{21} = 7.50 \times 10^{-5} \ eV^2}$'+'\n'
params_text += r'$\rm{{{\sin ^2\theta_{23}}} = 0.441}$'+'\n'
params_text += r'$\rm{{{\sin ^2\theta_{13}}} = 0.02166}$'+'\n'
params_text += r'$\rm{{{\sin ^2\theta_{12}}} = 0.306}$'+'\n'
params_text += r'$\rm{{\delta_{CP}} = 261^{\circ}}$'

#ax.text(0.59,0.3 , r'$\rm{{{\Delta m}^2}_{31}    = 2.524 \times 10^{-3} \ eV^2}$'+'\n'+
#    r'$\rm{{{\Delta m}^2}_{21}    = 7.50 \times 10^{-5} \ eV^2}$'+'\n'+
#    r'$\rm{{{\sin ^2\theta_{23}}} = 0.441}$'+'\n'+
#    r'$\rm{{{\sin ^2\theta_{13}}} = 0.02166}$'+'\n'+
#    r'$\rm{{{\sin ^2\theta_{12}}} = 0.306}$'+'\n'+
#    r'$\rm{{\delta_{CP}}          = 261^{\circ}}$',
#    verticalalignment='center',
#    horizontalalignment='center')

#a_text = AnchoredText(params_text, loc=3, prop={'color':'k','alpha':0.4,'size':12}, frameon=True,bbox_to_anchor=(0.59, 0.3), bbox_transform=ax.transAxes)
a_text = AnchoredText(params_text, loc=3, prop={'color':'k','size':12}, frameon=False,bbox_to_anchor=(0.59, 0.3), bbox_transform=ax.transAxes)
a_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(a_text)

#ax.legend(loc=0,ncol=1,frameon=True, columnspacing=0.8, handlelength=1.5, prop={'size':10}, bbox_transform=ax.transAxes)

plt.savefig('P_mu_to_x.pdf')
plt.savefig('P_mu_to_x.png')

arr_e2 = np.zeros(1000)
arr_nue_numu_NH  = np.zeros(1000)
arr_numu_numu_NH = np.zeros(1000)
arr_nue_numu_IH  = np.zeros(1000)
arr_numu_numu_IH = np.zeros(1000)
arr_nue_nutau_NH  = np.zeros(1000)
arr_numu_nutau_NH = np.zeros(1000)
arr_nutau_nutau_NH  = np.zeros(1000)

i = 0

#for energy in np.linspace(1, 20, 1000):
for energy in np.logspace(np.log10(5), np.log10(500), 1000):
    bNu.SetMNS( Theta12,  Theta13, Theta23, dm2, DM2, delta , energy, kSquared, kNuBar )
    #bNu.DefinePath( coszen , 25.00 )
    #bNu.DefinePath( coszen , prop_height, YeI, YeO, YeM, 25.00 )
    bNu.DefinePath( coszen , prop_height, YeI, YeO, YeM)
    bNu.propagate( 1*kNuBar )

    total_prob = 0.0
    #for m in xrange(1,4):
    for m in xrange(0,3):
        total_prob += bNu.GetProb(1, m)

    if ( total_prob >1.00001 or total_prob<0.99998 ):
        print 'ERROR Prob: Energy: ', energy , ' ', total_prob
        raise Exception('step 2, total_prob not 1')

    arr_e2[i] = energy
    arr_nue_numu_NH[i]   = bNu.GetProb(0,1)
    arr_numu_numu_NH[i]  = bNu.GetProb(1,1)
    arr_nue_nutau_NH[i]   = bNu.GetProb(0,2)
    arr_numu_nutau_NH[i]  = bNu.GetProb(1,2)
    arr_nutau_nutau_NH[i]   = bNu.GetProb(2,2)

    bNu.SetMNS( Theta12,  Theta13, Theta23, dm2, -DM2, delta , energy, kSquared, kNuBar )
    #bNu.DefinePath( coszen , 25.00 )
    #bNu.DefinePath( coszen , prop_height, YeI, YeO, YeM, 25.00 )
    bNu.DefinePath( coszen , prop_height, YeI, YeO, YeM)
    bNu.propagate( 1*kNuBar )

    total_prob = 0.0
    #for m in xrange(1,4):
    for m in xrange(0,3):
        total_prob += bNu.GetProb(1, m)

    if ( total_prob >1.00001 or total_prob<0.99998 ):
        print 'ERROR Prob: Energy: ', energy , ' ', total_prob
        continue
        raise Exception('step 3, total_prob not 1')
        #raise
        #raise Exception('total_prob not 1')

    arr_nue_numu_IH[i]   = bNu.GetProb(0,1)
    arr_numu_numu_IH[i]  = bNu.GetProb(1,1)

    i += 1

fig, ax = plt.subplots()
ax.plot(arr_e2, arr_nue_numu_NH  , 'r-')
ax.plot(arr_e2, arr_numu_numu_NH , 'b-')
ax.plot(arr_e2, arr_nue_numu_IH  , 'r--')
ax.plot(arr_e2, arr_numu_numu_IH , 'b--')
labelsize=14
#ax.set_xlim([1,20])
ax.set_ylim([0,1])
ax.set_ylabel(r'$\rm{P (\nu_{x} \rightarrow \nu_{\mu})}$', fontsize=labelsize)
ax.set_xlabel(r'Energy (GeV)', fontsize=labelsize)

#plt.show()
plt.savefig('./P_x_to_mu.pdf')
plt.savefig('./P_x_to_mu.png')
plt.clf()
#print '- 1\t2'
#for j in xrange(0,3):
#	print j+1, bNu.GetProb(1,j+1), bNu.GetProb(2,j+1)


fig, ax = plt.subplots()
labels['numu'] = r'$\nu_\mu \rightarrow \nu_{\tau}$'
labels['nue'] = r'$\nu_e \rightarrow \nu_{\tau}$'
labels['nutau'] = r'$\nu_\tau \rightarrow \nu_{\tau}$'
ax.plot(arr_e2, arr_nue_nutau_NH  , 'g-', label=labels['nue'])
ax.plot(arr_e2, arr_numu_nutau_NH , 'r-', label=labels['numu'])
#ax.plot(arr_e2, arr_nutau_nutau_NH  , 'b-', label=labels['nutau'])

ax.set_xscale('log')
tick_x = [5, 10, 20, 30, 50, 100, 200, 500]
tick_x_label = ['5', '10', '20', '30', '50', '100', '200', '500']
plt.xticks(tick_x, tick_x_label)
ax.tick_params(axis='x', which='both', labelsize=12)
ax.tick_params(axis='y', which='both', labelsize=12)
ax.legend(loc=(0.765, 0.72),fancybox=True,ncol=1,frameon=True, columnspacing=0.8, handlelength=1.5, prop={'size':12}, bbox_transform=ax.transAxes)


labelsize=20
#ax.set_xlim([1,20])
ax.set_xlim([5,500])
ax.set_ylim([0,1])
ax.set_ylabel(r'$\rm{P (\nu_{x} \rightarrow \nu_{\tau})}$', fontsize=labelsize)
ax.set_xlabel(r'$\rm{True \ \nu \ energy \ (GeV)}$', fontsize=labelsize)
plt.title(r'$\rm{Upgoing \ \nu \ [\cos(zen) = -1]}$', fontsize=labelsize)
fig.subplots_adjust(hspace=0.1, wspace=0.1, top=0.88, bottom=0.12, left=0.12, right=0.95)
a_text = AnchoredText(params_text, loc=3, prop={'color':'k','size':12}, frameon=False,bbox_to_anchor=(0.59, 0.3), bbox_transform=ax.transAxes)
a_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(a_text)

#plt.show()
plt.savefig('./P_x_to_tau_2.pdf')
plt.savefig('./P_x_to_tau_2.png')

