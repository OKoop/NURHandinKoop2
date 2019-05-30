import numpy as np

#This is a second-order ODE-solver using the RungeKutta 4th order algorithm.
class ODEer2(object):
    def __init__(self,func1,func2):
        self.func1 = func1
        self.func2 = func2
    
    #Runge-Kutta algorithm
    def RK4(self, x0, y0, yp0, stop, n=100, step=None):
        if step==None:
            step = stop/n
        
        #The 2nd order ODE should be rewritten into two coupled 1st order ODEs
        #Then we can perform the RK-steps from LEcture 10 for each
        f = self.func1
        fp = self.func2
        
        xs,zs,ys = [np.zeros(n) for i in range(3)]
        
        xs[0] = x0
        zs[0] = yp0
        ys[0] = y0
        #k's are the k_i for the first function, l's the k_i for the second.
        #We need to include the k's and l's of the previous steps in the
        #calculations of the next k's and l's.
        for i in range(1,n):
            k1 = step * f(xs[i-1], ys[i-1], zs[i-1])
            l1 = step * fp(xs[i-1], ys[i-1], zs[i-1])
            k2 = step * f(xs[i-1] + step/2., ys[i-1] + k1/2., zs[i-1] + l1/2.)
            l2 = step * fp(xs[i-1] + step/2., ys[i-1] + k1/2., zs[i-1] + l1/2.)
            k3 = step * f(xs[i-1] + step/2., ys[i-1] + k2/2., zs[i-1] + l2/2.)
            l3 = step * fp(xs[i-1] + step/2., ys[i-1] + k2/2., zs[i-1] + l2/2.)
            k4 = step * f(xs[i-1] + step, ys[i-1] + k3, zs[i-1] + l3)
            l4 = step * fp(xs[i-1] + step, ys[i-1] + k3, zs[i-1] + l3)
            ys[i] = (ys[i-1] + (k1 + 2. * k2 + 2. * k3 + k4)/6.)
            zs[i] = (zs[i-1] + (l1 + 2. * l2 + 2. * l3 + l4)/6.)
            xs[i] = (xs[i-1] + step)
        return xs, ys, zs


