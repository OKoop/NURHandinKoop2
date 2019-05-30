import numpy as np

#This function I wrote to create a mask to bit-reverse an array, but it was not
#needed in the algorithm I wrote below (I do however not know why...)
def maskbitrev(n):
    mask = np.zeros(n, dtype='int')
    for i in range(n):
        mask[i] = int(bin(i + n)[:1:-1],2)//2
    return list(mask)

#This function gives a one-dimensional (i)fft.
def d1fft(ys,inv=False):
    n = len(ys)
    #ys = ys[maskbitrev(n)]
    if n==1:
        return ys
    else:
        #Get the e-power needed for later puposes, we'll update it per k
        #to spare calculations.
        if inv:
            Wn = np.exp(-2j * np.pi/n)
        else:
            Wn = np.exp(2j * np.pi/n)
        W = 1
        #Split the array into an even- and odd part, and fft those.
        yseven = d1fft(ys[0::2])
        ysodd = d1fft(ys[1::2])
        #Combine even and odd parts into a full array again according to the
        #rules from Lecture 9.
        Y = np.zeros(n,dtype='complex')
        for j in range(0,int(np.ceil(n/2))):
            Y[j] = yseven[j] + W * ysodd[j]
            Y[j + int(n/2)] = yseven[j] - W * ysodd[j]
            W *= Wn
    return Y

#Function that returns an array (0,...,fc,-fc+1,...,-1).
def fftfreq(size):
    n2 = int((size+1)/2)
    fftfreq = np.zeros(size)
    for i in range(0,n2):
        fftfreq[i] = i
    for i in range(-n2,0):
        fftfreq[i+size] = i
    return fftfreq

#Function that uses 1dffts to get a 2dfft.
def d2fft(ys,inv=False):
    shape = ys.shape
    res = np.zeros(shape,dtype='complex')
    #First fft the rows.
    for row in range(shape[0]):
        res[row,:] = d1fft(ys[row,:],inv)
    #Then fft the hereby created columns
    for col in range(shape[1]):
        res[:,col] = d1fft(res[:,col],inv)
    return res

#Function that uses 1dffts to get a 3dfft.
def d3fft(ys,inv=False):
    shape = ys.shape
    res = np.zeros(shape,dtype='complex')
    #As in d2fft, first fft all rows along a direction
    for row in range(shape[0]):
        for col in range(shape[1]):
            res[row,col,:] = d1fft(ys[row,col,:],inv)
    #Then all created rows along another direction
    for row in range(shape[1]):
        for col in range(shape[2]):
            res[:,row,col] = d1fft(res[:,row,col],inv)
    #Then along the last direction.
    for row in range(shape[0]):
        for col in range(shape[2]):
            res[row,:,col] = d1fft(res[row,:,col],inv)
    return res

#Really simple gradient calculation. We find in which cell a particle subsides
#and find the slope of a line through the values of the next grid points
#along all directions. I do not expect great results.
def d3grad(pot,xc,yc,zc,size=16):
    ix, iy, iz = int(xc), int(yc), int(zc)
    gx = (pot[(ix+1)%size, iy%size, iz%size]-pot[ix%size, iy%size, iz%size])/2
    gy = (pot[ix%size, (iy+1)%size, iz%size]-pot[ix%size, iy%size, iz%size])/2
    gz = (pot[ix%size, iy%size, (iz+1)%size]-pot[ix%size, iy%size, iz%size])/2
    return gx, gy, gz
