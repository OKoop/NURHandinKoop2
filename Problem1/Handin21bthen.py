import numpy as np
from matplotlib import pyplot as plt
import functions21 as f

#---------------- 1(b) ---------------------------
def gausintegrd(x, a):
    return np.exp(-a * x**2.)

def line(x, a, b):
    return np.exp(a * np.log(x) + b)

#Find the exact values and the numerical values. I've chosen to use Simpson 
#integration after checking both that one and Romberg, and both were similar in
#result, so I used the fastest one.
alphs = 2.**np.arange(-10,11,1)
alphse = 2. ** np.linspace(-10,10,10000)
ext = f.Gausexct(alphse)
simp = np.zeros(len(alphs))
for i in range(len(alphs)):  
    simp[i] = 2. * f.simpson(gausintegrd, 0, 100, 100000, args=[alphs[i]])


plt.plot(alphse, ext, label='Exact')
plt.plot(alphs, simp, label='Simpson', linestyle='dashed')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('x',fontsize=15)
plt.ylabel('f(x)',fontsize=15)
plt.title('A Gaussian function integrator',fontsize=18)
plt.savefig('fits.png',format='png')
plt.clf()

#As seen, in log-log space this is a line, so we will just calculate a and b 
#from there
la, ls = np.log(alphs), np.log(simp)

def chi21b(x):
    models = x[0] * la + x[1]
    return sum((models-ls)**2.)

#Based on the first and last points
ag = (ls[-1]-ls[0])/(la[-1]-la[0])
#Where np.log(alphs)=0
bg = ls[10]
opt = f.DHS(chi21b, np.array([ag,bg]), 10**-10, 100)

print('My parameters for the line in log-space: [a,b] = ',opt)
print('The analytical values are not in here anymore...')

#I made this exercise until this point before it got cut from the Handin.
