import numpy as np
from scipy import fftpack as ft

Om = 0.3
Ol = 0.7
H0 = 70 #km/s/Mpc

#Function to find H^2 for this exercise.
def H2(z):
    return (Om * (1 + z)**3 + Ol)

#Function that defines the integrand from formula (11) from the sheet.
def integrand(z):
    return (1 + z)/(H2(z)**(3./2.))

#Function to find H from H2.
def H(z):
    return H2(z)**(1./2.)

#Function to find \dot{D(z)}, analytically calculated.
def ddot(a, integralr):
    A = 5 * Om/2. * H0
    B = (-3 * Om * a**-3)/2.
    C = 1/(a*a * H(1/a - 1))
    return A * (B * integralr + C)

#Function to find D for exercise 4(b).
def D1(z):
    A = 5 * Om/2.
    B = H(z)
    C = romberg(integrand, z, 100000, 15)
    return A * B * C

#Function to find D for 4(c) and 4(d).
def D(z):
    A = 5 * Om/2.
    B = H(z)
    C = romberg(integrand,z,10000,12)
    return A * B * C, C

#Integration by Simpsons rule
def simpson(func,a,b,N,args=[]):
    #Prepare an array with x-values and the corresponding y-values
    xs = np.linspace(a,b,N+1)
    ys = func(xs, *args)
    #Sum over the y-values and multply by h/3 after adding the begin- and end-value.
    intsum = 4.*sum(ys[1:-1:2]) + 2.*sum(ys[2:-1:2])
    h = (b-a)/N
    intsum = (h/3.)*(intsum + ys[0] + ys[N])
    return intsum

#Romberg-algorithm for integration.
#We use this algorithm without any extra methods. It is good enough as is, and
#the order is manageable. It is the best method discussed in lecture 
#(in my opinion)
def romberg(func,a,b,order,args=[]):
    #Start by creating an array to save values
    res = np.zeros((order,order), dtype=np.float64)
    #Create an array with powers of 4 to be used.
    pow_4 = 4**np.arange(order)
    
    #The first 'interval size' is just the size of the starting interval
    h = (b-a)
    #The first value is given by h*(mean of f(a) and f(b))
    res[0,0] = h*(func(a, *args) + func(b, *args))/2.
    
    #Per step decrease the interval size and update the array with results by
    #combining previously known values and using the new points and thus values
    for i in range(1,order):
        h /= 2.
        
        res[i,0] = res[i-1,0]/2.
        #Including new values at only the new points.
        res[i,0] += h*sum(func(a+j*h, *args) for j in range(1,2**i+1,2))
        
        #Update new results as specified by the formula in Lecture 3 
        #(as analogue to Neville's algorithm).
        for k in range(1, i+1):
            res[i,k] = (pow_4[k]*res[i,k-1] - res[i-1,k-1])/(pow_4[k]-1)
    
    return res[-1,-1]

#Function to determine central difference for derivatives in one point
def ysh(xs, func, h, args=[]):
    A = func(xs + h, *args) - func(xs - h, *args)
    ys = float(A/(2. * h))
    return ys

#Implementation of Ridders method based on the Romberg method, with terminology
#From the slides. This is the algorithm of which I expect the best result. Also
#it was easiest to implement for I already had Romberg implemented above.
def Ridders(func, xs, hstart, d, m, args=[]):
    #Create arrays to store values.
    results = [[0.0 for i in range(m)] for j in range(m)]
    #Create array with powers of d in advance.
    ds = d**np.arange(m)
    
    h=hstart
    #The first value is just the central difference for the point.
    results[0][0] = ysh(xs, func, h, args)
    
    for i in range(1, m):
        h /= 2.
        #Get the next central difference with the new interval size.
        results[i][0] = ysh(xs, func, h, args)
        
        #Include the new results as found in the Slides (as in Nevilles 
        #algorithm.)
        for k in range(1, i + 1):
            results[i][k] = ((ds[k] * results[i][k-1] - 
                             results[i-1][k-1])/(ds[k] - 1))
    
    return results[-1][-1]

#Function to use in grf2d to get the random numbers from a pre-generated array
#and symmetrize the array which will be ifft'd.
def action2d(rands,ks,i,j,A,n,ind):
    #The k for the given element:
    k = ks[i][j]
    #Tranform the random numbers as given in (10)
    sp = k**(n/2. - 2)
    a, b = sp * rands[ind], sp * rands[ind+1]
    c = (a - 1j * b)/2. 
    #We need to get symmetry which we do as follows:
    A[i][j] = c
    A[-i][-j] = c.conjugate()
    return

#Function to generate the x- and y- values of S(q).
def grf2d(R,n=-2,size=64):
    #First generate, without library functions, an array of k_x and k_y to be 
    #used for the ifft, thus with frequencies in the range 0 to fc and -fc+1 to
    #-1 in all dimensions.
    #Not the most efficient way.
    n2 = int((size+1)/2)
    ks, ks2d, ks2d2 = [[] for i in range(3)]
    for i in range(0,n2):
        ks.append(i)
    for i in range(-n2,0):
        ks.append(i)
    for i in range(size):
        ks2d.append(ks)
    for i in ks:
        ks2d2.append([i for j in range(size)])
    ksarr = np.array([ks2d,ks2d2]) * (2. * np.pi)/size
    
    #Turn the frequencies into amplitudes depending on the given power spectrum
    #index
    ks = (ksarr[0]**2. + ksarr[1]**2. + 1e-10)**(1/2.)
    ks[0,0] = 0
    #turn the amplitudes into random complex numbers, and allow for the ifft to
    #be real by the given rule.
    A = np.zeros((size,size), dtype='complex')
    
    #Generate the 4 real numbers for the pixels that need to get only real 
    #entries
    rands = R.BoxMuller(size*size)
    ind = 0
    
    #Loop over only the top half (including the Nyquist band), and enter the
    #random numbers (and conjugates at -k). Check if -k = k and put in a real
    #number to ensure the symmetry.
    for i in range(n2+1):
        for j in range(size):
            if (size-i)%size==i and (size-j)%size==j:
                if i==0 and j==0:
                    A[i][j] = 0
                else:
                    A[i][j] = rands[ind] * ks[i][j]**(n/2. - 2)
                ind += 1
            elif A[i][j]==0 and A[-i][-j]==0:
                action2d(rands,ks,i,j,A,n,ind)
                ind += 2
    
    #Perform the inverse Fourier transforms.
    Bx = ft.ifft2(1j * A * ksarr[0])
    By = ft.ifft2(1j * A * ksarr[1])
    
    return Bx, By

#Function to use in grf3d to generate the random numbers and symmetrize the
#array. Same idea as in action2d.
def action3d(rands,ks,i,j,l,A,n,ind):
    #find the k that belongs for the given indices.
    k = ks[i][j][l]
    #Transform according to (10)
    sp = k**(n/2. - 2)
    a, b = sp * rands[ind], sp * rands[ind + 1]
    c = (a - 1j * b)/2. 
    A[i][j][l] = c
    A[-i][-j][-l] = c.conjugate()
    return

#Function to get the x- y- and z-values for a 3D S(q).
def grf3d(R,n=-2,size=64):
    #First generate, without library functions, an array of k_x and k_y and k_z
    #to be used for the ifft, thus with frequencies in the range 0 to fc and from
    # -fc+1 to -1 in all the dimensions.
    n2 = int((size+1)/2)
    fftfreq = np.zeros(size)
    for i in range(0,n2):
        fftfreq[i] = i
    for i in range(-n2,0):
        fftfreq[i+size] = i
    ksarr = np.zeros((3,size,size,size))
    ksarr[0] = fftfreq[:,None,None]
    ksarr[1] = fftfreq[None,:,None]
    ksarr[2] = fftfreq[None,None,:]
    ksarr = ksarr * (2. * np.pi)/size
    
    #Turn the frequencies into amplitudes depending on the given power spectrum
    #index
    ks = (ksarr[0]**2. + ksarr[1]**2. + ksarr[2]**2. + 1e-10)**(1/2.)
    ks[0,0,0] = 0
    
    #turn the amplitudes into random complex numbers, and allow for the ifft to
    #be real by the given rule.
    
    #First generate random numbers for the entire array
    A = np.zeros((size,size,size), dtype='complex')
    rands = R.BoxMuller(size*size*size)
    ind = 0
    
    #Only loop over the 'front half' of the cube (including the Nyquist band)
    #Then check for if -k = k, if so, only give a real number. If not
    #get a complex number and add its conjugate to -k.
    for i in range(size):
        for j in range(size):
            for l in range(n2+1):
                if (size-i)%size==i and (size-j)%size==j and (size-l)%size==l:
                    if i==0 and j==0 and l==0:
                        A[i][j][l] = 0
                    else:
                        A[i][j][l] = rands[ind] * ks[i][j][l]**(n/2. - 2)
                    ind += 1
                elif A[i][j][l]==0 and A[-i][-j][-l]==0:
                    action3d(rands,ks,i,j,l,A,n,ind)
                    ind += 2
    
    #Do the inverse Fourier transform for each direction.
    Bx = ft.ifftn(1j * A * ksarr[0])
    By = ft.ifftn(1j * A * ksarr[1])
    Bz = ft.ifftn(1j * A * ksarr[2])
    
    return Bx, By, Bz










