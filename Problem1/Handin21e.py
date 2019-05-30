import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
import functions21 as f
samp = np.loadtxt('samp.txt')
r1c = np.zeros(16)
r1c[1:] = 10.**np.linspace(1,5,15)
r1c = np.asarray(r1c,dtype='int')
n = len(r1c) - 1
#-------------------- 1(e) ----------------

#Function performing the steps for 1e, same idea as with 1c and 1d, but not 
#with a KS test for two datasets.
def ex1e(data):
    #Create array of sample sizes
    ds, ps = [np.zeros(n) for el in range(2)]
    #For each sample size, find the KS-statistic d and the p-value according 
    #to my own implementation and that of scipy.
    for i in range(n):
        rands = samp[r1c[i]:r1c[i]+r1c[i+1]:1]
        datas = data[::len(data)//r1c[i+1]]
        ds[i], ps[i] = f.kstwo(datas, rands)
    return ds, ps

#Perform the KStest for the 10 given datasets and plot in a figure with 
#subplots.
D = np.loadtxt('randomnumbers.txt')
fig, axs = plt.subplots(4,5,figsize=(30,20),dpi=100)
for i in range(10):
    dse, pse = ex1e(D[:,i])
    if i<5:
        k=0
    else:
        k=2
    axs[k,i%5].plot(r1c[1:],dse)
    axs[k,i%5].set_xscale('log')
    axs[k,i%5].set_yscale('log')
    axs[k,i%5].set_xlabel('sample size')
    axs[k,i%5].set_ylabel('KS-statistic')
    axs[k,i%5].set_title('Value of d for set'+str(i+1))
    k += 1
    axs[k,i%5].plot(r1c[1:],pse)
    axs[k,i%5].plot([r1c[1],r1c[-1]],[5e-2, 5e-2], linestyle='dashed', 
                       c='red')
    axs[k,i%5].set_xscale('log')
    axs[k,i%5].set_yscale('log')
    axs[k,i%5].set_xlabel('sample size')
    axs[k,i%5].set_ylabel('p-value')
    axs[k,i%5].set_title('p-value for set'+str(i+1))
plt.savefig('e.png',format='png')
plt.clf()