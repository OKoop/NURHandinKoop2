import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(121)
size = 16
#------------------------------------5(a)-----------------------------
pos = np.random.uniform(0,size,(3,1024))

#Save the generated positions for use in c.
np.savetxt('randpos.txt',pos)

#Create empty array with counts per cell.
counts = np.zeros((size,size,size))
#The NGP for each mass has indices equal to the rounded off coordinates.
#Rounding off can be done by int(x+.5), because int rounds down per default.
#We then reshape the random positions into triples for counting.
posa = np.array(pos+.5,dtype='int').T.reshape((1024,3))

#For each particle count it in the appropriate bin, using % for the periodic
#boundary conditions.
for p in posa:
    counts[p[0]%size,p[1]%size,p[2]%size] += 1

#Plot the needed slices.
plt.imshow(counts[:,:,4])
plt.colorbar()
plt.title('Slice of NGP-mesh z=4')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('a4')
plt.clf()

plt.imshow(counts[:,:,9])
plt.colorbar()
plt.title('Slice of NGP-mesh z=9')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('a9')
plt.clf()

plt.imshow(counts[:,:,11])
plt.colorbar()
plt.title('Slice of NGP-mesh z=11')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('a11')
plt.clf()

plt.imshow(counts[:,:,14])
plt.colorbar()
plt.title('Slice of NGP-mesh z=14')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('a14')
plt.clf()

#------------------------------------5(b)-----------------------------
lensim = 1000
xpos = np.linspace(0,size,lensim)
counts = np.zeros((size,lensim))
#Perform the same idea as for (a) per position along one dimension.
for i in range(lensim):
    counts[int(xpos[i]+.5)%size,i] += 1

#Plot the results for bins 4 and 0.
plt.plot(xpos,counts[4,:])
plt.title('Value of bin 4 for given x-position.')
plt.xlabel('x')
plt.ylabel('Value of bin 4')
plt.savefig('b4')
plt.clf()

plt.plot(xpos,counts[0,:])
plt.title('Value of bin 0 for given x-position.')
plt.xlabel('x')
plt.ylabel('Value of bin 0')
plt.savefig('b0')
plt.clf()




