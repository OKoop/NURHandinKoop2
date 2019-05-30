import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
size = 16
#------------------------------------5(c)-----------------------------
cellvals = np.zeros((size,size,size))
pos = np.loadtxt('randpos.txt')
posc = pos.T.reshape((1024,3))
#For each position, find the grid point in the corner at the bottom left of the
#grid cell that contains it. Find thus the distance dx, dy, and dz of the
#particle to that corner, and using these we can assign parts of the mass to
#each corner of the grid cell.
#This is just done by dividing the cloud of the particle into rectangles that
#are closest to each point. These sizes depend only on dx, dy and dz.
for p in posc:
    i, j, k = int(p[0]), int(p[1]), int(p[2])
    dx, dy, dz = p[0] - i, p[1] - j, p[2] - k
    mdx, mdy, mdz = 1 - dx, 1 - dy, 1 - dz
    #Get the % of each index to ensure periodic boundaries. Assume total
    #particle mass to be 1. And the grid size to be 1.
    cellvals[i%size,j%size,k%size] += mdx * mdy * mdz
    cellvals[(i+1)%size,j%size,k%size] += dx * mdy * mdz
    cellvals[i%size,(j+1)%size,k%size] += mdx * dy * mdz
    cellvals[i%size,j%size,(k+1)%size] += mdx * mdy * dz
    cellvals[(i+1)%size,(j+1)%size,k%size] += dx * dy * mdz
    cellvals[(i+1)%size,j%size,(k+1)%size] += dx * mdy * dz
    cellvals[i%size,(j+1)%size,(k+1)%size] += mdx * dy * dz
    cellvals[(i+1)%size,(j+1)%size,(k+1)%size] += dx * dy * dz

#Plot the asked slices.
plt.imshow(cellvals[:,:,4])
plt.colorbar()
plt.title('Slice of CIC-mesh z=4')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('c4')
plt.clf()

plt.imshow(cellvals[:,:,9])
plt.colorbar()
plt.title('Slice of CIC-mesh z=9')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('c9')
plt.clf()

plt.imshow(cellvals[:,:,11])
plt.colorbar()
plt.title('Slice of CIC-mesh z=11')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('c11')
plt.clf()

plt.imshow(cellvals[:,:,14])
plt.colorbar()
plt.title('Slice of CIC-mesh z=14')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('c14')
plt.clf()

#Save for use in f.
np.savetxt('CIC.txt',cellvals.reshape((16,-1)))

#Perform the same idea as above but in 1D for many possible x-positions.
lensim = 1000
xpos = np.linspace(0,size,lensim)
counts = np.zeros((size,lensim))
for i in range(lensim):
    xi = xpos[i]
    ix = int(xpos[i])
    dx = xi - ix
    mdx = 1 - dx
    counts[ix%size,i] += mdx
    counts[(ix+1)%size,i] += dx

plt.plot(xpos,counts[4,:])
plt.title('Value of bin 4 for given x-position.')
plt.xlabel('x')
plt.ylabel('Value of bin 4')
plt.savefig('c4c')
plt.clf()

plt.plot(xpos,counts[0,:])
plt.title('Value of bin 0 for given x-position.')
plt.xlabel('x')
plt.ylabel('Value of bin 0')
plt.savefig('c0c')
plt.clf()