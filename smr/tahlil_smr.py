import numpy as np
import matplotlib.pyplot as plt

systems = ['Pn60','Pn120','Pn180','Hn60','Hn120','Hn180']

methods =  ['rand','dbsa','drg']
precs = [10,5,4,2,1] 
num_DRG = 24

for sys in systems:
    plt.clf()
    DRG_vol = np.zeros((len(methods),len(precs)))
    sasa_DRG = np.loadtxt(f'vol_comp/sasa/vol_l10/{sys}-DRG.xvg', unpack = True ,  skiprows=25, usecols=1)
    sasa_DRG_av = np.average(sasa_DRG)
    sasa_DRG_std = np.std(sasa_DRG)
    for i, method in enumerate(methods):
        for j , prec in enumerate(precs):
            arra = np.loadtxt(f'nats_smr/{method}/d{prec}{sys}.txt', unpack = True ,  skiprows=1, usecols=0)
            DRG_vol[i,j] = arra * num_DRG 
    plt.errorbar(range(len(methods)),np.average(DRG_vol,axis=1),yerr=np.std(DRG_vol,axis=1),elinewidth=2,linewidth =0,marker='o',capsize=5,markersize=7)
    plt.plot([sasa_DRG_av]*len(methods),'k-',label='gmx_sasa')
    plt.plot([sasa_DRG_av + sasa_DRG_std ]*len(methods),'r--')
    plt.plot([sasa_DRG_av - sasa_DRG_std ]*len(methods),'r--')
    plt.legend(loc='upper right')
    plt.xlabel('Method')
    plt.ylabel('DRG Volume')
    plt.xticks(range(len(methods)),methods)
    plt.title(f'System {sys}')
    plt.savefig(f'tahlil_figs/{sys}.png')
    
# plt.show()
# print('sas',sasa_DRG)
# print('vol',DRG_vol)
# print('av',np.average(DRG_vol,axis=1))
# print('std',np.std(DRG_vol,axis=1))