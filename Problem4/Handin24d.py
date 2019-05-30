import functions24 as f4
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../Problem1')
import functions21 as f1

#Initialize the RNG with the same seed and get the random fields (S(q)'s).
R = f1.RNG(15860573632797531096)
print('seed for d =',15860573632797531096)

Bx, By, Bz = f4.grf3d(R)

#Counteract the normalization from scipy.fftpack.
n = 64*64
Bx, By, Bz = (Bx).real*n, (By).real*n, (Bz).real*n

#Initialize positions, a-values and position- and momenta-array.
xso, yso, zso = [np.linspace(0,63,64) for i in range(3)]
xso, yso, zso = np.meshgrid(xso, yso, zso)
aval = np.linspace(1/(1+50), 1, 30*3)
da = aval[1]-aval[0]
x10, y10, z10, px10, py10, pz10 = [np.zeros((10,90)) for i in range(6)]
x10[:,0], y10[:,0], z10[:,0], px10[:,0], py10[:,0], pz10[:,0] = [xso[0,0,0:10], 
    yso[0,0,0:10], zso[0,0,0:10], np.zeros(10), np.zeros(10), np.zeros(10)]

#These are masks for the slices in x-y, y-z, and x-z.
maskz = (zso>=31.5) & (zso<=32.5)
maskx = (xso>=31.5) & (xso<=32.5)
masky = (yso>=31.5) & (yso<=32.5)

#Plot the initial positions.
fig, axs = plt.subplots(nrows=1,ncols=3, figsize=(12,4))
plt.suptitle('Positions, a='+str(aval[0]))
axs[0].scatter(xso[maskz], yso[maskz], s=1)
axs[0].set_xlim([-1,65])
axs[0].set_ylim([-1,65])
axs[0].set_xlabel('x (Mpc)')
axs[0].set_ylabel('y (Mpc)')
axs[1].scatter(xso[masky], zso[masky], s=1)
axs[1].set_xlim([-1,65])
axs[1].set_ylim([-1,65])
axs[1].set_xlabel('x (Mpc)')
axs[1].set_ylabel('z (Mpc)')
axs[2].scatter(yso[maskx], zso[maskx], s=1)
axs[2].set_xlim([-1,65])
axs[2].set_ylim([-1,65])
axs[2].set_xlabel('y (Mpc)')
axs[2].set_ylabel('z (Mpc)')
plt.savefig('./UF2/step00.png',format='png')
plt.clf()
plt.close(fig)

#For each a calculate the new positions and momenta according to (8)
for i in range(1,3*30):
    a = aval[i]
    ap = a - da/2
    Da, inte = f4.D(1/a-1)
    Dpa = f4.ddot(ap, inte)
    xs = Da * Bx + xso
    ys = Da * By + yso
    zs = Da * Bz + zso
    #Periodic boundary conditions.
    xs %= 64
    ys %= 64
    zs %= 64
    temp = (ap)*(ap) * Dpa
    px = -temp * Bx
    py = -temp * By
    pz = -temp * Bz
    x10[:,i], y10[:,i], z10[:,i], px10[:,i], py10[:,i], pz10[:,i] = [xs[0,0,0:10],
        ys[0,0,0:10], zs[0,0,0:10], px[0,0,0:10], py[0,0,0:10], pz[0,0,0:10]]
    
    #Again create the masks.
    maskz = (zs>=31.5) & (zs<=32.5)
    maskx = (xs>=31.5) & (xs<=32.5)
    masky = (ys>=31.5) & (ys<=32.5)

    fig, axs = plt.subplots(nrows=1,ncols=3, figsize=(12,4))
    plt.suptitle('Positions, a='+str(a))
    axs[0].scatter(xs[maskz], ys[maskz], s=1)
    axs[0].set_xlim([-1,65])
    axs[0].set_ylim([-1,65])
    axs[0].set_xlabel('x (Mpc)')
    axs[0].set_ylabel('y (Mpc)')
    axs[1].scatter(xs[masky], zs[masky], s=1)
    axs[1].set_xlim([-1,65])
    axs[1].set_ylim([-1,65])
    axs[1].set_xlabel('x (Mpc)')
    axs[1].set_ylabel('z (Mpc)')
    axs[2].scatter(ys[maskx], zs[maskx], s=1)
    axs[2].set_xlim([-1,65])
    axs[2].set_ylim([-1,65])
    axs[2].set_xlabel('y (Mpc)')
    axs[2].set_ylabel('z (Mpc)')
    plt.savefig('./UF2/step{:02d}.png'.format(i),format='png')
    plt.clf()
    plt.close(fig)


for i in range(10):
    plt.plot(aval,x10[i],label=str(i))
plt.legend()
plt.title('x-coordinate of 10 particles in 3D')
plt.xlabel('a')
plt.ylabel('x (Mpc)')
plt.savefig('3dposx')
plt.clf()

for i in range(10):
    plt.plot(aval,y10[i],label=str(i))
plt.legend()
plt.title('y-coordinate of 10 particles in 3D')
plt.xlabel('a')
plt.ylabel('y (Mpc)')
plt.savefig('3dposy')
plt.clf()

for i in range(10):
    plt.plot(aval,z10[i],label=str(i))
plt.legend()
plt.title('z-coordinate of 10 particles in 3D')
plt.xlabel('a')
plt.ylabel('z (Mpc)')
plt.savefig('3dposz')
plt.clf()

for i in range(10):
    plt.plot(aval,np.sqrt(px10[i]**2+py10[i]**2+pz10[i]**2),label=str(i))
plt.legend()
plt.title('Absolute value of the momentum of 10 particles in 3D')
plt.xlabel('a')
plt.ylabel('p (Units unknown)')
plt.savefig('3dmom')
plt.clf()

for i in range(10):
    plt.plot(aval,px10[i],label=str(i))
plt.legend()
plt.title('Momentum along the x-direction of 10 particles in 3D')
plt.xlabel('a')
plt.ylabel(r'$p_x$ (Units unknown)')
plt.savefig('3dmomx')
plt.clf()

for i in range(10):
    plt.plot(aval,py10[i],label=str(i))
plt.legend()
plt.title('Momentum along the y-direction of 10 particles in 3D')
plt.xlabel('a')
plt.ylabel(r'$p_y$ (Units unknown)')
plt.savefig('3dmomy')
plt.clf()

for i in range(10):
    plt.plot(aval,pz10[i],label=str(i))
plt.legend()
plt.title('Momentum along the z-direction of 10 particles in 3D')
plt.xlabel('a')
plt.ylabel(r'$p_z$ (Units unknown)')
plt.savefig('3dmomz')
plt.clf()