import matplotlib
matplotlib.use('Agg')
import functions22 as f2
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../Problem1')
import functions21 as f1

#Initialize the RNG again
R = f1.RNG(np.loadtxt('../seed.txt'))
size = 1024
rands = R.BoxMuller(size**2)

#Generate the three Gaussian Random Fields and plot them.
B = f2.grf(rands,-3, size=size)

plt.imshow(B[0].real,cmap='plasma')
plt.colorbar()
plt.title('Gaussian Random Field for n=-3')
plt.savefig('n3.png',format='png')
plt.clf()

B = f2.grf(rands,-2, size=size)

plt.imshow(B[0].real,cmap='plasma')
plt.colorbar()
plt.title('Gaussian Random Field for n=-2')
plt.savefig('n2.png',format='png')
plt.clf()

B = f2.grf(rands,-1, size=size)

plt.imshow(B[0].real,cmap='plasma')
plt.colorbar()
plt.title('Gaussian Random Field for n=-1')
plt.savefig('n1.png',format='png')
plt.clf()

#Save the state as new seed
np.savetxt('../seed.txt',R.state)





