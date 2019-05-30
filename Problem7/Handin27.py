import h5py
import functions27 as f7
import numpy as np

#Read in the file
filename = 'colliding.hdf5'
f = h5py.File(filename, 'r')

#Get the coorinates from the file
coords = np.asarray(f['PartType4']['Coordinates'])[:,0:2]
#As found in the file:
mass = 0.0125

#Initialize the quadtree
BHtree = f7.Tree(12, coords, mass)
#Build the quadtree
BHtree.build()
#Plot it
BHtree.plottree([0,150],[0,150])

#The 0th multipole is just the sum of all particle-masses in the cell
BHtree.calcm0s()

#Get the coordinates at i=100
xi, yi = coords[100]
#Get the list of m0's for all nodes containing i=100.
m0s = BHtree.givem0i(xi,yi)

print('The n=0 multipole moment of each node (leaf to root) containing i=100 is:',
      m0s)