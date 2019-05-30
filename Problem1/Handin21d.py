import numpy as np
from matplotlib import pyplot as plt
import functions21 as f
from astropy.stats import kuiper

samp = np.loadtxt('samp.txt')
r1c = np.zeros(41)
r1c[1:] = 10.**np.linspace(1,5,40)
r1c = np.asarray(r1c, dtype='int')
n = len(r1c) - 1
#-------------------- 1(d) ----------------

#Function that performs the steps for 1d, same idea as ex1c but with Kuiper test
def ex1d():
    #Create array of sample-sizes.
    ds, ps, dsci, psci = [np.zeros(n) for el in range(4)]
    #For each sample size, find the Kuiper-statistic d and the p-value according 
    #to my own implementation and that of scipy.
    for i in range(n):
        rands = samp[r1c[i]:r1c[i+1]+r1c[i]:1]
        ds[i], ps[i] = f.kp(f.CDGaussian, rands)
        dsci[i], psci[i] = kuiper(rands, f.CDGaussian)
    return ds, ps, dsci, psci

dsk, psk, dscik, pscik = ex1d()

#Create the asked figures.
plt.plot(r1c[1:],dsk,label='implemented')
plt.plot(r1c[1:],dscik,label='astropy Kuiper', linestyle='dashed')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('sample size')
plt.ylabel('Kuiper-statistic for the BoxMuller random numbers')
plt.savefig('ds1d.png',format='png')
plt.clf()

plt.plot(r1c[1:],psk,label='implemented')
plt.plot(r1c[1:],pscik,label='astropy Kuiper', linestyle='dashed')
plt.plot([r1c[1],r1c[-1]],[5e-2, 5e-2], linestyle='dashed', c='red')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('sample size')
plt.ylabel('p-value')
plt.savefig('ps1d.png',format='png')
plt.clf()