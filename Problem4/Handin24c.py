import matplotlib
matplotlib.use('Agg')
import functions24 as f4
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../Problem1')
import functions21 as f1

#Initialize an RNG and positions if 64 * 64 particles.
R = f1.RNG(1246578543)
print('seed for c =',1246578543)
xso, yso = [np.linspace(0,63,64) for i in range(2)]
xso, yso = np.meshgrid(xso, yso)

#Get the x- and y- components of S(q) from a Gaussian Random Field.
Bx, By = f4.grf2d(R)
#Counteract the normalization from scipy.fftpack.
Bx, By = Bx.real * 64, By.real * 64

#Initialize the a-values and arrays to store x, y, p_x and p_y for the first
#ten particles along the y-axis (for x=0 thus).
aval = np.linspace(0.0025,1,30*3)
da = aval[1]-aval[0]
x10, y10, px10, py10 = [np.zeros((10,90)) for i in range(4)]

#Plot the initial positions and store the initial positions and momenta.
fig = plt.figure(dpi=50, figsize=[6.4,6.4])
plt.scatter(xso, yso, s=1, c='black')
plt.xlim([-1,65])
plt.ylim([-1,65])
plt.title('Positions, a='+str(aval[0]))
plt.xlabel('x (Mpc)')
plt.ylabel('y (Mpc)')
plt.savefig('./UF1/step00.png',format='png')
plt.clf()
plt.close(fig)

x10[:,0], y10[:,0], px10[:,0], py10[:,0] = [xso[0:10,0], yso[0:10,0], 
                                            np.zeros(10), np.zeros(10)]

#For each a (such that the movie can be 3 seconds) calculate the new positions
#according to (8). We use a less dedicated integration here then in a, b, for
#speed.
for i in range(1,3*30):
    a = aval[i]
    ap = a - da/2
    Da, inte = f4.D(1/a-1)
    Dpa = f4.ddot(ap, inte)
    xs = Da * Bx + xso
    ys = Da * By + yso
    #For the periodic boundary conditions:
    xs %= 64
    ys %= 64
    temp = (ap)*(ap) * Dpa
    px = -temp * Bx
    py = -temp * By
    #Store positions and momenta
    x10[:,i], y10[:,i], px10[:,i], py10[:,i] = [xs[0:10,0], ys[0:10,0], 
                                                px[0:10,0], py[0:10,0]]
    #Save the next frame.
    fig = plt.figure(dpi=50, figsize=[6.4,6.4])
    plt.scatter(xs, ys, s=1, c='black')
    plt.xlim([-1,65])
    plt.ylim([-1,65])
    plt.title('Positions, a='+str(a))
    plt.xlabel('x (Mpc)')
    plt.ylabel('y (Mpc)')
    plt.savefig('./UF1/step{:02d}.png'.format(i),format='png')
    plt.clf()
    plt.close(fig)

#Plot the positions and momenta.
for i in range(10):
    plt.plot(aval,x10[i],label=str(i))
plt.legend()
plt.title('x-coordinate of 10 particles in 2D')
plt.xlabel('a')
plt.ylabel('x (Mpc)')
plt.savefig('2dposx')
plt.clf()

for i in range(10):
    plt.plot(aval,y10[i],label=str(i))
plt.legend()
plt.title('y-coordinate of 10 particles in 2D')
plt.xlabel('a')
plt.ylabel('y (Mpc)')
plt.savefig('2dposy')
plt.clf()

for i in range(10):
    plt.plot(aval,np.sqrt(px10[i]**2+py10[i]**2),label=str(i))
plt.legend()
plt.title('Absolute value of the momentum of 10 particles in 2D')
plt.xlabel('a')
plt.ylabel('p (Units unknown)')
plt.savefig('2dmom')
plt.clf()

for i in range(10):
    plt.plot(aval,px10[i],label=str(i))
plt.legend()
plt.title('Momentum along the x-direction of 10 particles in 2D')
plt.xlabel('a')
plt.ylabel(r'$p_x$ (Units unknown)')
plt.savefig('2dmomx')
plt.clf()

for i in range(10):
    plt.plot(aval,py10[i],label=str(i))
plt.legend()
plt.title('Momentum along the y-direction of 10 particles in 2D')
plt.xlabel('a')
plt.ylabel(r'$p_y$ (Units unknown)')
plt.savefig('2dmomy')
plt.clf()