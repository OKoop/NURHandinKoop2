import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
import sys
#Defining the Random Number Generator used in this exercise. This one is based
#on the lectures and values from there. It implements the Ranq1-generator from
#'Numerical Recipes'
class RNG(object):
    
    #Initialize the LCG-parameters and XOR-parameters and the state equal to 
    #the seed.
    def __init__(self, seed, a=4294957665, x1=21, x2=35, x3=4):
        self.state = np.uint64(seed)
        self.a = np.uint64(a)
        self.x1, self.x2, self.x3 = np.uint64(x1), np.uint64(x2), np.uint64(x3)
    
    #A 'multiply with carry' as found in the slides/book, replacing the state 
    #with the next value
    #The standard 'a' implemented is the value proposed in the slides.
    def MWC(self):
        x = self.state
        self.state = np.uint64(self.a * (x & np.uint64([2**32 - 1])) + 
                               (x >> np.uint64(32)))
    
    #An XOR-shift, replacing the state with the next value
    #Based on slides and 'Numerical Recipes'
    def XOR(self):
        x = self.state
        x ^= x >> self.x1
        x ^= x << self.x2
        x ^= x >> self.x3
        self.state = np.uint64(x)
    
    #Function that returns one random number
    def randn(self):
        self.MWC()
        self.XOR()
        return self.state
    
    #Function that returns a random number or a sample of size 'size'
    #in a given interval given by [mini, maxi]
    def sample(self, size, mini=0, maxi=1):
        samp = np.zeros(size, dtype=np.uint64)
        for i in range(size):
            samp[i] = self.randn()
        #This factor 2 included here is needed to find numbers between 0 and 1
        #because np.uints can be twice the sys.maxsize.
        samp = samp/(sys.maxsize * 2.)
        return samp * (maxi - mini) + mini
    
    def BoxMuller(self, size, mu=0, sigma=1):
        #Generate two samples of uniform numbers between 0 and 1 with half the 
        #size.
        x1, x2 = self.sample(size//2), self.sample(size//2)
        #Perform the transformation as seen in the slides/book.
        r = (-2. * np.log(x1))**(1./2.)
        y1, y2 = r * np.cos(2. * np.pi * x2), r * np.sin(2. * np.pi * x2)
        #Return all generated numbers.
        return(np.append(y1,y2) * sigma + mu)
    
    #Returns the state.
    def state(self):
        return self.state

#Plots a simple scatterplot and saves it.
def scatter(x, y, t, xl, yl, svn=' ', sv=False, loglog=False):
    plt.scatter(x,y)
    plt.title(t, fontsize=18)
    plt.xlabel(xl, fontsize=15)
    plt.ylabel(yl, fontsize=15)
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    if sv:
        plt.savefig(svn+'.png', format='png')
    plt.close()
    return

#plots a simple histogram (density) and saves it.
def plthist(x, bina, t, xl, yl, svn=' ', sv=False):
    plt.hist(x, bins=bina, density=True)
    plt.title(t, fontsize=18)
    plt.xlabel(xl, fontsize=15)
    plt.ylabel(yl, fontsize=15)
    if sv:
        plt.savefig(svn+'.png', format='png')
    plt.close()
    return

#The exact result of Gaussian from -infty to +infty
def Gausexct(a):
    return (np.pi/a)**(1./2.)

#The Gaussian distribution function, already normalized.
def NormedGaussian(x, mu=0, sig=1):
    A = 1./((2. * np.pi * sig**2.)**(1./2.))
    B = -(x - mu)**2./(2. * sig**2.)
    return A * np.exp(B)

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

#Function to order three 2D-vectors on argument when filled into the given 
#function.
def order(func, a, b, c):
    fa, fb, fc = func(a), func(b), func(c)
    if fa <= fb <= fc:
        return a, b, c
    elif fb <= fa <= fc:
        return b, a, c
    elif fb <= fc <= fa:
        return b, c, a
    elif fa <= fc <= fb:
        return a, c, b
    elif fc <= fa <= fb:
        return c, a, b
    else:
        return c, b, a

#The Downhill Simplex Algorithm from the slides, only works in 2D. Start needs 
#to be a 2D vector
def DHS(func, start, taracc, maxit):
    #Prepare three starting vectors around the starting point.
    x0 = np.array([start[0], start[1]+.1])
    x1 = np.array([start[0]+.1, start[1]])
    x2 = np.array([start[0]-.1, start[1]-.1])
    
    #Order the vectors w.r.t. their function values
    x0, x1, x2 = order(func, x0, x1, x2)
    #Calculate the centroid of the lowest function values.
    cent = (x0 + x1)/2.
    
    i=0
    
    #While the target accuracy and maximal amount of steps are not reached, 
    #search for new points
    while ((abs(func(x2)-func(x0))/(abs(func(x2)+func(x0))/2.)) > taracc 
           and i < maxit):
        #First try reflecting x2 in the centroid
        xtry = 2. * cent - x2
        
        f0, ftry, f2 = func(x0), func(xtry), func(x2)
        if f0 <= ftry < f2:
            x2 = xtry
        elif ftry < f0:
            #Propose extension if the reflected point was good.
            xtry2 = 2. * xtry - cent
            if func(xtry2) < ftry:
                x2 = xtry2
            else:
                x2 = xtry
        #If we did not accept the reflecton or extension, try contracting.
        else:
            xtry = (cent + x2)/2.
            if func(xtry) < f2:
                x2 = xtry
            else:
                #If contracting did not work, decrease the size of the simplex.
                x1 = (cent + x1)/2.
                x2 = (cent + x2)/2.
        
        #Order the resulting points and calculate the new centroid to prepare 
        #for the next step.
        x0, x1, x2 = order(func, x0, x1, x2)
        cent = (x1 + x0)/2.
        i += 1
        
    return x0

#Function to merge two sorted arrays to be used in mergesort below.
def merge(left, right):
    result = []
    l, r = 0, 0
    lenl, lenr = len(left), len(right)
    #Iterate through both left and right and append the lowest value to 
    #'result'
    while l < lenl and r < lenr:
        if left[l] <= right[r]:
            result.append(left[l])
            l += 1
        else:
            result.append(right[r])
            r += 1
    #Append the 'residue' of the list that was not completely appended yet.
    result.extend(left[l:])
    result.extend(right[r:])
    return result

#Mergesort algorithm for sorting an array. I implemented mergesort because it
#is quite fast, and was easy to implement.
def mergesort(a):
    N = len(a)
    #If not already 1 element long, split the array and sort both parts. This
    #is done recursively.
    if N > 1:
        midi = N//2
        left, right = a[:midi], a[midi:]
        left, right = mergesort(left), mergesort(right)
        a = merge(left, right)
    return a

#Function to determine the error function of a value. isint should be False if
#entering an entire array of arguments, if only a single one, it should be True.
#It is used to find the Cumulative Gaussian Distribution.
def erf(x, isint=False):
    #Store the sign(s) of x
    if isint:
        if x >=0:
            signs = 1
        else:
            signs = -1
        x = abs(x)
    
    else:
        x = np.array(x)
        n = len(x)
        signs = np.zeros(n)
        for i in range(n):
            if x[i] >= 0:
                signs[i] = 1
            else:
                signs[i] = -1
            x[i] = abs(x[i])

    #Found online an efficient way to calculate erf(x) with the given constants
    x1, x2, x3, x4, x5, x6 =  [0.254829592, -0.284496736, 1.421413741, 
                               -1.453152027, 1.061405429, 0.3275911]
    A = 1./(1. + x6 * x)
    y = 1. - ((((((x5 * A + x4) * A) + x3) * A + x2) * A + x1) * A * 
              np.exp(-x * x))
    
    return signs * y

#Finds the value of the cumulative distribution of a Gaussian with mu and sigma 
#given in x. isint is as in erf above.
def CDGaussian(x, mu=0, sig=1, isint=False):
    B = (x - mu)/(sig * 2.**(1./2.))
    A = 1. + erf(B, isint)
    return (1./2.) * A

#A class that gives a Kolmogorov-Smirnov-distribution
class KSdist(object):
    
    def __init__(self):
        return
    
    #returns the probability
    def pks(self, z):
        if z < 0.:
            return 'error'
        elif z==0:
            return 0.
        #As stated in the book/slides, z=1.18 is a turning point for the 
        #approximation we should use.
        elif z < 1.18:
            A = (2. * np.pi)**(1./2.)/z
            B = np.exp(-np.pi**2./(8. * z**2.))
            return A * (B + B**9. + B**25.)
        else:
            x = np.exp(-2. * z**2.)
            return 1. - 2. * (x - x**4. + x**9.)
       
    #returns 1-pks, but then more efficiently, by not allowing a 1-(1-...) to 
    #occur.
    def qks(self, z):
        if z < 0.:
            return 'error'
        elif z == 0.:
            return 1.
        elif z < 1.18:
            return 1. - self.pks(z)
        else:
            x = np.exp(-2. * z**2.)
            return 2. * (x - x**4. + x**9.)

#Implements the KS-test from the book, and returns the statistic as well as the
# p-value.
def ks(func, data, args=[0, 1, True]):
    #Initialize parameters and sort the data-array
    n = len(data)
    fo, d, en = 0., 0., n
    data = mergesort(data)
    #For each datapoint, calculate its corresponding cumulative-distribution 
    #value were it to occur in the 'true' function 'func'
    #Also fn is the cumulative value after having added this datapoint, where 
    #fo is that before, so we find the maximal
    #distance to 'func' between fo and ff and fn and ff. If this is bigger than
    #the stored maximum, store it.
    for j in range(n):
        fn = (j + 1)/en
        ff = func(data[j], *args)
        dt = max(abs(fo - ff), abs(fn - ff))
        if dt > d:
            d = dt
        fo = fn
    #Use the above defined object to generate a KS-distribution and find the 
    #p-value.
    ksdist = KSdist()
    en = en**(1./2.)
    p = ksdist.qks((en + .12 + .11/en) * d)
    return d, p

#A class like the KSdist object, to return the Kuiper distribution. 
#Only the 1-probability is implemented here.
class KPdist(object):
    
    def __init__(self):
        return
       
    #returns 1-pks, but then more efficiently.
    def qks(self, z):
        if z < 0.:
            return 'error'
        elif z == 0.:
            return 1.
        #It says in the book that we do not need to bother with z < .4
        elif z < .4:
            return 1.
        #I approximate the value with the first four terms from the 
        #distribution found in the book.
        else:
            s = 0
            for j in range(1,4):
                a = 2. * j**2. * z**2.
                A = 2. * a - 1.
                B = np.exp(-a)
                s += A * B
            return 2. * s

#Implements the Kuiper-test by altering the function ks above, and returns
#the statistic as well as the p-value.
def kp(func, data, args=[0, 1, True]):
    #Initialize parameters and sort the data-array
    n = len(data)
    fo, d1, d2, en = 0., 0., 0., n
    data = mergesort(data)
    #For each datapoint, just as in KS, but now do both a maximal 
    #over-estimation and a maximal underestimation
    for j in range(n):
        fn = (j + 1)/en
        ff = func(data[j], *args)
        d1t = max(fo - ff, fn - ff)
        d2t = max(ff - fo, ff - fn)
        if d1t > d1:
            d1 = d1t
        if d2t > d2:
            d2 = d2t
        fo = fn
    kpdist = KPdist()
    en = en**(1./2.)
    #Add the maximal over- and underestimation and find the p-value as 
    #specified in the book.
    d = d1 + d2
    p = kpdist.qks((en + .155 + .24/en) * d)
    return d, p

#Compares two dists for equality using the KS-test. Implemented from the book.
def kstwo(data1, data2):
    #Initialize parameters and sort the data-arrays
    n1, n2 = len(data1), len(data2)
    fn1, fn2, j1, j2, d = [0 for i in range(5)]
    en1, en2 = n1, n2
    data1, data2 = mergesort(data1), mergesort(data2)
    #While we have not gone through both data-arrays, find which of them is at 
    #a higher value. While this is still so,
    #add to the cumulative value of the lower one until they meet or the value 
    #of the lowest one changes. Then find the absolute
    #difference between the values, and if it is the highest yet, update.
    while (j1 < n1 and j2 < n2):
        d1, d2 = data1[j1], data2[j2]
        if d1 <= d2:
            while (j1 < n1 and d1 == data1[j1]):
                j1 += 1
                fn1 = j1/en1
        if d2 <= d1:
            while (j2 < n2 and d2 == data2[j2]):
                j2 += 1
                fn2 = j2/en2
        dt = abs(fn2 - fn1)
        if dt > d:
            d = dt
    ksdist = KSdist()
    #The effective N is different in this case w.r.t. KS-test for one 
    #distribution, the p-value computation is done the same as in KS.
    en = ((en1 * en2)/(en1 + en2))**(1./2.)
    p = ksdist.qks((en + .12 + .11/en) * d)
    return d, p

#Compares two dists for equality using the KP-test. Implemented from the book.
def kptwo(data1, data2):
    #Initialize parameters and sort the data-arrays
    n1, n2 = len(data1), len(data2)
    fn1, fn2, j1, j2, dp, dm = [0 for i in range(6)]
    en1, en2 = n1, n2
    data1, data2 = mergesort(data1), mergesort(data2)
    
    #While we have not gone through both data-arrays, find which of them is at 
    #a higher value. While this is still so,
    #add to the cumulative value of the lower one until they meet or the value 
    #of the lowest one changes. Then find the
    #difference between the values on both sides, and if it is the highest yet, 
    #update.
    while (j1 < n1 and j2 < n2):
        d1, d2 = data1[j1], data2[j2]
        if d1 <= d2:
            while (j1 < n1 and d1 == data1[j1]):
                j1 += 1
                fn1 = j1/en1
        if d2 <= d1:
            while (j2 < n2 and d2 == data2[j2]):
                j2 += 1
                fn2 = j2/en2
        dpt, dmt = fn2 - fn1, fn1 - fn2
        if dpt > dp:
            dp = dpt
        if dmt > dm:
            dm = dmt
    kpdist = KPdist()
    #um the highest over- and underestimation
    d = dp + dm
    #The effective N is different in this case w.r.t. KP-test for one 
    #distribution, the p-value computation is done the same as in KP.
    en = ((en1 * en2)/(en1 + en2))**(1./2.)
    p = kpdist.qks((en + .155 + .24/en) * d)
    return d, p













