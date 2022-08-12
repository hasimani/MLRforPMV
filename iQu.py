import numpy as np
from numpy.linalg import pinv
from numpy.random import seed
from numpy.random import rand
import sys
import time

Time_intervals = int(sys.argv[3])#19#95
num_comp = int(sys.argv[2])

num_cv = 5
num_cv3 = num_cv**3
eps = 1e-5

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

mol_dict = {'DRG':0,'dbs':2,'PEN':1,'HEP':1}
mol_num_dict= {'DRG':68,'dbs':27,'PEN':5,'HEP':7}

with open(sys.argv[1], 'r') as file:
    data = file.readlines()
    nums = int(data[1])

    timeSteps = int(len(data)/(nums+3))

    steps = int(timeSteps/Time_intervals)

    
    
    for t in range(Time_intervals):
        Xs = np.zeros(steps)
        numl = np.zeros((steps, num_comp , num_cv3), float)
        for i in range(steps):
            ins_dims = data[(t*steps+i+1)*(nums+3)-1] #instant dimensions
            abad = float(ins_dims[:10])
            dim = cv_make(abad,num_cv,i,Xs) #creating the list of control volumes
            for k, s in enumerate(dim):

                for dat in data[(t*steps+i)*(nums+3)+2:(t*steps+i+1)*(nums+3)-1]:

                    if boxCheck(dat[22:], s):
                        numl[i][mol_dict[dat[5:8]]][k] += 1/mol_num_dict[dat[5:8]]
        seed(int(time.time()))
        weis = np.multiply(rand(num_cv3) > 0.4,np.ones(num_cv3))
        weis /= np.sum(weis)

        Ns = np.average(numl, axis=2,weights=weis)

        Xbar = np.matmul(pinv(np.matmul(Ns.T,Ns)+eps*np.identity(num_comp)), np.matmul(Ns.T,Xs))
        #print('samples:',len(Xs),'r2 stat',rt_st(Ns,Xs,Xbar))
        #print('Partial Molar Volumes:\n')
        print(Xbar)
        #print('Matrix of selected instances')
        #print(Ns)
        # print('N . X_bar')
        # print(np.matmul(Ns,Xbar))
        # print('X')
        # print(Xs)

file.close()