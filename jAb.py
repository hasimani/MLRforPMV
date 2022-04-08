import numpy as np
from numpy.linalg import pinv
from numpy.random import seed
from numpy.random import rand
import sys
import time


N = int(sys.argv[2])

nocv = 5
nocv3 = nocv**3

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
    leni = (0.4 + rand()/2)*abd

    sft = (abd - leni)/(tedad-1)
    dil = []
    for i in range(0,tedad):
        for j in range(0, tedad):
            for k in range(0, tedad):
                dil.append([i*sft,j*sft,k*sft,i*sft+leni,j*sft+leni,k*sft+leni])

    haj[rpa] = leni**3
    return dil

mol_dict = {'DRG':0,'dbs':2,'PEN':1,'HEP':1,'SOL':2}
mol_num_dict= {'DRG':68,'dbs':27,'PEN':5,'HEP':7,'SOL':3}

with open(sys.argv[1], 'r') as file:
    data = file.readlines()
    nums = int(data[1])

    M = int(len(data)/(nums+3))

    Xs = np.zeros(M)
    numl = np.zeros((M, N , nocv3), float)

    for i in range(0,M): #iterating over time-steps
        ins_dims = data[(i+1)*(nums+3)-1] #instant dimensions
        abad = float(ins_dims[:10])
        dim = cv_make(abad,nocv,i,Xs) #creating the list of control volumes with random dimensions and creating X vector
        for k, s in enumerate(dim): #iterating over all CV's in frame i

            for dat in data[i*(nums+3)+2:(i+1)*(nums+3)-1]: #iterating over all atoms in frame i

                if boxCheck(dat[22:], s): #checking if atom dat is inside CV
                    numl[i][mol_dict[dat[5:8]]][k] += 1/mol_num_dict[dat[5:8]] #increasing the related cell by 1/M
    seed(int(time.time()))
    weis = np.multiply(rand(nocv3) > 0.4,np.ones(nocv3))
    weis /= np.sum(weis) #creating a random sampling array
    Ns = np.average(numl, axis=2,weights=weis) #sampling a random number of CVs in each frame

    Xbar = np.matmul(pinv(Ns), Xs) # calculating X_bar
    print('samples:',len(Xs),'r2 stat',rt_st(Ns,Xs,Xbar))
    print('Partial Molar Volumes:\n')
    print(Xbar)
    print('Matrix of selected instances')
    print(Ns)
    print('N . X_bar')
    print(np.matmul(Ns,Xbar))
    print('X')
    print(Xs)

file.close()