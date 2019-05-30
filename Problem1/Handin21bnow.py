import numpy as np
from matplotlib import pyplot as plt
import functions21 as f

R = f.RNG(np.loadtxt('../seed.txt'))
#---------------------------- 1(b) --------------------------
#For the asked distribution values plot both the NormedGaussian and the 
#histogram of a sample.
m = 3
s = 2.4
y1 = R.BoxMuller(1000, mu = m, sigma = s)
xsG = np.linspace(m - 5 * s, m + 5 * s, 10000)
ysG = f.NormedGaussian(xsG, mu = m, sig = s)
plt.hist(y1, bins=21, density=True, label='RNG')
plt.plot(xsG,ysG, label='Theoretical')
for i in range(6):
    dx = i * s
    fval = f.NormedGaussian(m - dx, mu = m, sig = s)
    #Add the +-n\sigma-lines.
    plt.plot([m - dx, m - dx], [0,fval], c='black')
    plt.plot([m + dx, m + dx], [0,fval], c='black')
    
#Plot the y-axis in 'log' to be able to see the lines clearly
plt.yscale('log')
plt.legend()
plt.title(r'Normal distribution with $\mu=3$ and $\sigma=2.4$')
plt.xlabel(r'$x$')
plt.ylabel(r'$\mathbb{P}(x)$')
plt.savefig('1b.png',format='png')
plt.clf()

np.savetxt('../seed.txt',R.state)