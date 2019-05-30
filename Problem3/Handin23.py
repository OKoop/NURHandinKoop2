from matplotlib import pyplot as plt
import functions23 as f

O0 = 1.

#Define the functions that we are using for solving the ODE, and the
#analytical solution.
def f1(t, D, z):
    return z
def f2(t, D, z):
    A = (2./3.) * O0 * (1./t**2.) * D
    B = (4./(3. * t)) * z
    return A - B
def sol(t, D1, Dp1):
    c1 = (3./5.) * (D1 + Dp1)
    c2 = (2./5.) * (D1 - (3./2.) * Dp1)
    return c1 * t**(2./3.) + c2/t

#Initialize the ODE-class and get the solutions for the given initial
#conditions
G = f.ODEer2(f1, f2)
tsG, DsG, zsG = G.RK4(1, 3, 2, 1000, 100000)
tsG2, DsG2, zsG2 = G.RK4(1, 10, -10, 1000, 100000)
tsG3, DsG3, zsG3 = G.RK4(1, 5, 0, 1000, 100000)

#Plot the solutions (analytical and from the RK4-algorithm)
plt.plot(tsG, DsG, label='Numerical')
#plt.plot(tsG, zsG)
plt.plot(tsG, sol(tsG, 3, 2), linestyle='dashed',label='Analytical')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('32.png',format='png')
plt.clf()

plt.plot(tsG2, DsG2, label='Numerical')
#plt.plot(tsG2, zsG2)
plt.plot(tsG, sol(tsG, 10, -10), linestyle='dashed',label='Analytical')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('1010.png',format='png')
plt.clf()

plt.plot(tsG3, DsG3, label='Numerical')
#plt.plot(tsG3, zsG3)
plt.plot(tsG, sol(tsG, 5, 0), linestyle='dashed',label='Analytical')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('50.png',format='png')
plt.clf()
