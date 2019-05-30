import numpy as np
from matplotlib import pyplot as plt
import functions25 as f5
size = 16
cellvals = np.loadtxt('CIC.txt').reshape((size,size,size))
#------------------------------------5(f)-----------------------------
m = sum(cellvals)/(16*16*16)
ave = (cellvals-m)/m

#This is the FFT of the average (\delta)
fftave = f5.d3fft(ave)

#We need to divide by k^2
fftfreq = f5.fftfreq(size)
ksarr = np.zeros((3,size,size,size))
ksarr[0] = fftfreq[:,None,None]
ksarr[1] = fftfreq[None,:,None]
ksarr[2] = fftfreq[None,None,:]
ksarr = ksarr
k2 = ksarr[0]*ksarr[0] + ksarr[1]*ksarr[1] + ksarr[2]*ksarr[2]
#To make sure not to divide by 0:
k2[0,0,0]=1

fftavek2 = fftave/k2

#And then take the IFFT (where i --> -i)
pot = np.abs(f5.d3fft(fftavek2,True))

#Plot the results:
plt.imshow(pot[:,:,4])
plt.colorbar()
plt.title('Slice of absolute value of CIC-potential z=4')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('f4')
plt.clf()

plt.imshow(pot[:,:,9])
plt.colorbar()
plt.title('Slice of absolute value of CIC-potential z=9')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('f9')
plt.clf()

plt.imshow(pot[:,:,11])
plt.colorbar()
plt.title('Slice of absolute value of CIC-potential z=11')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('f11')
plt.clf()

plt.imshow(pot[:,:,14])
plt.colorbar()
plt.title('Slice of absolute value of CIC-potential z=14')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('f14')
plt.clf()

plt.imshow(pot[:,7,:])
plt.colorbar()
plt.title('Slice of absolute value of CIC-potential y=7')
plt.xlabel('x')
plt.ylabel('z')
plt.savefig('f7xz')
plt.clf()

plt.imshow(pot[7,:,:])
plt.colorbar()
plt.title('Slice of absolute value of CIC-potential x=7')
plt.xlabel('y')
plt.ylabel('z')
plt.savefig('f7yz')
plt.clf()

#-----------------------5(g)---------------------
#Here we find the gradients of the first ten generated random particles.
pos = np.loadtxt('randpos.txt').T.reshape((1024,3))
grads = [None]*10
for i in range(10):
    grads[i] = f5.d3grad(pot,*pos[i])
print('The gradients are:',grads)
