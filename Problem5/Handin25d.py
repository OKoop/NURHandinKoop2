import numpy as np
from matplotlib import pyplot as plt
import functions25 as f5
from scipy import fftpack as fp

#------------------------------------5(d)-----------------------------

#We test the FFT with a sine.
def testfunc(x):
    return np.sin(2. * np.pi * x/5.)

#Get values and calculate the fft-frequencies.
xs = np.linspace(0,20,64)
ys = testfunc(xs)
myown = f5.d1fft(ys)
fs = f5.fftfreq(len(xs))/(len(xs)*(xs[1]-xs[0]))

#Get the same from scipy to check
scipys = fp.fft(ys)
fs2 = fp.fftfreq(len(xs),d=xs[1]-xs[0])

#The analytical fft should be two delta-peaks at the frequency of the sine 
#k = (1/5)
x0, x1 = .5j, -.5j
k0 = 1/5.

#Scatter my own fft, the scipy fft and the two analytical delta points along
#with many zeros.
plt.scatter(fs[1:],abs(myown[1:]),label='My own',s=60)
plt.scatter(fs2[1:],abs(scipys[1:]), label='Scipy',s=25)
l = list(fs)
l.remove(fs[4])
l.remove(-fs[4])
plt.scatter([-.2,.2]+l,np.abs(64*np.array([.5j,.5j]+[0 for i in range(62)])),
            label='Analytical',s=10)
plt.yscale('log')
plt.legend()
plt.title('FFT\'s of the given sine')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.savefig('1d')
plt.clf()

