import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
import functions21 as f
from scipy.stats import kstest
#-------------------- 1(c) ----------------
R = f.RNG(np.loadtxt('../seed.txt'))

r1c = np.zeros(41)
r1c[1:] = 10.**np.linspace(1,5,40)
r1c = np.asarray(r1c, dtype='int')
rands = R.BoxMuller(sum(r1c)+1)
np.savetxt('samp.txt',rands)
n = len(r1c) - 1
#function that performs the steps for 1c
def ex1c():
    #Create an array with sample sizes and make them integers
    ds, ps, dsci, psci = [np.zeros(n) for el in range(4)]
    #For each sample size, find the KS-statistic d and the p-value according 
    #to my own implementation and that of scipy.
    for i in range(n):
        r = rands[r1c[i]:r1c[i]+r1c[i+1]:1]
        ds[i], ps[i] = f.ks(f.CDGaussian, r)
        dsci[i], psci[i] = kstest(r, 'norm')
    #Save the sample to use the same in d and e to save some time.

    return ds, ps, dsci, psci

ds, ps, dsci, psci= ex1c()

#Plot the results
plt.plot(r1c[1:],ds,label='implemented')
plt.plot(r1c[1:],dsci,label='scipy', linestyle='dashed')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('sample size')
plt.ylabel('KS-statistic')
plt.title('KS-statistic for the BoxMuller random numbers')
plt.savefig('ds1c.png',format='png')
plt.clf()

plt.plot(r1c[1:],ps,label='implemented')
plt.plot(r1c[1:],psci,label='scipy',linestyle='dashed')
plt.plot([r1c[1],r1c[-1]],[5e-2, 5e-2], linestyle='dashed', c='red')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('sample size')
plt.ylabel('p-value')
plt.savefig('ps1c.png',format='png')
plt.clf()

np.savetxt('../seed.txt',R.state)