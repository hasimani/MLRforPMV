import numpy as np
from numpy.linalg import pinv
from numpy.random import seed
from numpy.random import rand
import matplotlib.pyplot as plt
import sys
import time


N = int(sys.argv[2])

DRG = 24
nocv = 5
nocv3 = nocv**3
eps = 1e-4

excd = [' CBM',' CBN',' CBO',' CBP',' CBQ',' CBR',' CBS',' CBT',' CBU',' CBV',' CBW',' CBX',' CBY',' CBZ',' CCA',' CCB']

def rt_st(x,y,b): # return r2 stat of linearly regressing y into x using parameters b
    yh = np.matmul(x,b) #computing the estimated values
    rss = sum(np.square(y-yh)) # residual sum of squares
    yb = np.mean(y)
    tss = sum(np.square(y-yb)) #total sum of squares
    return 1 - (rss/tss)

def boxCheck(cors,dims): #checks if an atom with position cors (strings) is inside the box defined by boundaries in dims
    c1 = float(cors[:6])
    c2 = float(cors[8:14])
    c3 = float(cors[16:])
    if max((c1 - dims[0])*(c1 - dims[3]),(c2 - dims[1])*(c2 - dims[4]),(c3 - dims[2])*(c3- dims[5])) < 0:
        return 1
    else:
        return 0

def cv_make(abd,tedad,rpa,haj):
    seed(rpa)
    leni = (0.6 + rand()/5)*abd

    sft = (abd - leni)/(tedad-1)
    dil = []
    for i in range(0,tedad):
        for j in range(0, tedad):
            for k in range(0, tedad):
                dil.append([i*sft,j*sft,k*sft,i*sft+leni,j*sft+leni,k*sft+leni])

    haj[rpa] = leni**3
    return dil

group_list = ['DRG','SOL','DBSA']
mol_dict = {'DRG':0,'dbs':2,'PEN':1,'HEP':1}
mol_num_dict= {'DRG':52,'dbs':27,'PEN':5,'HEP':7}

with open(sys.argv[1], 'r') as file:
    data = file.readlines()
    nums = int(data[1])

    M = int(len(data)/(nums+3))
    d = (M - 1) // int(sys.argv[3])
    Xs = np.zeros(d+1)
    numl = np.zeros((d+1, N , nocv3), float)

    Ns = np.zeros((d+1,N))

    for i in range(0,M,M//d):
        ip = i//(M//d)
        ins_dims = data[(i+1)*(nums+3)-1] #instant dimensions
        abad = float(ins_dims[:10])
        dim = cv_make(abad,nocv,ip,Xs) #creating the list of control volumes
        for k, s in enumerate(dim):

            for dat in data[i*(nums+3)+2:(i+1)*(nums+3)-1]:
                if boxCheck(dat[22:], s) and dat[11:15] not in excd:
                    numl[ip][mol_dict[dat[5:8]]][k] += 1/mol_num_dict[dat[5:8]]
        seed(int(time.time()))
        weis = np.multiply(numl[ip,0,:] > DRG/nocv3,np.ones(nocv3))
        weis /= np.sum(weis)

        Ns[ip,:] = np.average(numl[ip,:,:], axis=1,weights=weis)

    N_std = np.amax(Ns, axis=0)

    Ns /= np.dot(np.ones((d+1,1)),np.amax(Ns, axis=0,keepdims=True))

    Xbar = np.matmul(pinv(np.matmul(Ns.T,Ns)+eps*np.identity(N)), np.matmul(Ns.T,Xs))
    # Xbar = np.matmul(pinv(Ns), Xs)

    
    # print('samples:',len(Xs),'r2 stat',rt_st(Ns,Xs,Xbar))
    Xbar1 = Xbar / N_std 
    with open(sys.argv[4]+'.txt', 'w') as f:
        f.write(f'samples: {len(Xs)} - r2 stat: {rt_st(Ns,Xs,Xbar)}\n')
        f.write(' '.join(map(str, Xbar1)))
    f.close()
    # print('Partial Molar Volumes:\n')
    # print(Xbar1)
    # print('Matrix of selected instances')
    # print(Ns)
    # print('N . X_bar')
    # print(np.matmul(Ns,Xbar))
    # print('X')
    # print(Xs)
    for j in range(N):
        plt.hist(Ns[:,j],bins=10,label=group_list[j],histtype='step')

    plt.legend(loc='upper right')

    # plt.xlabel('Simulation Time')


    # plt.ylabel('DBSA PMV')
    # plt.yticks(np.arange(0.18,0.24,0.02))
    # plt.xticks(DBSA_nums)
    # plt.title('Heptane Systems')
    plt.savefig(sys.argv[4]+'.png')
    plt.clf()


file.close()