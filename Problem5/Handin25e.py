import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
import functions25 as f5
from scipy import fftpack as fp

#------------------------------------5(e)-----------------------------
#Use a 2D sine to test.
def testfunc2d(x,y):
    return np.sin(2. * np.pi * (x+y)/5.)

xs, ys = np.linspace(0,20,16), np.linspace(0,20,16)
xs, ys = np.meshgrid(xs,ys)
zs = testfunc2d(xs, ys)

scipys = fp.fft2(zs)
myown = f5.d2fft(zs)

#The analytical fft should be two distinct points at $e^{\pm i\pi/5(x+y)}$.
#So it should only be the two yellow points.
plt.imshow(np.abs(scipys))
plt.title('scipyFFT of a given sine')
plt.savefig('e2dscipy')
plt.clf()

plt.imshow(np.abs(myown))
plt.title('My own FFT of a given sine')
plt.savefig('e2down')
plt.clf()

#A multivariate Gaussian for testing, as asked.
def testfunc3d(x,y,z,mux=0,muy=0,muz=0,sx=1,sy=1,sz=1):
    A = (x - mux)**2/(2 * sx * sx)
    B = (y - muy)**2/(2 * sy * sy)
    C = (z - muz)**2/(2 * sz * sz)
    return np.exp(-(A + B + C))

#Prepare a grid and calculate the values.
xs, ys, zs = np.linspace(0,20,16), np.linspace(0,20,16), np.linspace(0,20,16)
xs, ys, zs = np.meshgrid(xs,ys,zs)
ws = testfunc3d(xs, ys, zs)
#ws = testfunc3d(xs, ys, zs, 0, .2, .4, 1, 2, 4)

scipys = fp.fftn(ws)
myown = f5.d3fft(ws)

#To plot the slices centered in the array
n=7

plt.imshow(np.abs(scipys[:,:,n]))
plt.title('scipyFFT x-y of a 3D Gaussian')
plt.savefig('e3dxyscipy')
plt.clf()

plt.imshow(np.abs(myown[:,:,n]))
plt.title('My own FFT x-y of a 3D Gaussian')
plt.savefig('e3dxyown')
plt.clf()

plt.imshow(np.abs(scipys[:,n,:]))
plt.title('scipyFFT x-z of a 3D Gaussian')
plt.savefig('e3dxzscipy')
plt.clf()

plt.imshow(np.abs(myown[:,n,:]))
plt.title('My own FFT x-z of a 3D Gaussian')
plt.savefig('e3dxzown')
plt.clf()

plt.imshow(np.abs(scipys[n,:,:]))
plt.title('scipyFFT y-z of a 3D Gaussian')
plt.savefig('e3dyzscipy')
plt.clf()

plt.imshow(np.abs(myown[n,:,:]))
plt.title('My own FFT y-z of a 3D Gaussian')
plt.savefig('e3dyzown')
plt.clf()