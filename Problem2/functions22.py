import numpy as np
from scipy import fftpack as ft

def genran(rands,i,j,ampl,A,k):
    sigma2 = ampl[i][j]
    a = sigma2 * rands[k] + 1j * sigma2 * rands[k+1]
    #We need to get a complex number so we transform to 'cartesian' coordinates
    A[i][j] = a
    A[-i][-j] = a.conjugate()
    return

def grf(rands,n=-1,size=1024):
    #First generate, without library functions, an array of k_x and k_y to be 
    #used for the ifft, thus with frequencies in the range 
    #[0,...,f_c,-f_c+1,...,-1] for both dimensions.
    #Not the most efficient way.

    n2 = int((size+1)/2)
    ks, ks2d, ks2d2 = np.zeros(size), [None]*size, [None]*size
    for i in range(0,n2):
        ks[i] = i
    for i in range(-n2,0):
        ks[i+size] = i
    for i in range(size):
        ks2d[i] = ks
        ks2d2[i] = ([ks[i] for j in range(size)])
    ksarr = np.array([ks2d,ks2d2])
  
    #Turn the frequencies into amplitudes depending on the given power spectrum
    #index
    ampl = (ksarr[0]**2. + ksarr[1]**2. + 1e-10)**(n/2.)
    ampl[0,0] = 0
    
    #turn the amplitudes into random complex numbers, and allow for the ifft to
    #be real by the given rule.
    A = np.zeros((size,size),dtype='complex')
    k = 0
    
    #For each element in the top half (with the Nyquist-band), generate the
    #random numbers needed.
    for i in range(n2+1):
        for j in range(size):
            if (size-i)%size==i and (size-j)%size==j:
                if i==0 and j==0:
                    A[i][j] = 0
                else:
                    A[i][j] = rands[k] * ampl[i][j]
                k += 1
            elif A[i][j]==0 and A[-i][-j]==0:
                genran(rands,i,j,ampl,A,k)
                k += 2
                
    B = ft.ifft2(A)
    
    return B, A