import functions21 as f
import numpy as np

#----------------- 1(a) ----------------------------
seed = 1094801294865
print('The seed will be:',seed)

#Initialize the RNG and plot the asked figures.
R = f.RNG(seed)
samp = R.sample(1000000)

samp1000 = samp[:1000]
sshift = samp[1:1001:1]

f.scatter(sshift, samp1000, 'Sequential Random Numbers', r'$x_{i+1}$', r'$x_i$'
        ,'scatter1a1',True)
f.scatter(np.linspace(0,1000,1000), samp1000, ' Random Numbers', 'i', r'$x_i$'
        ,'scatter1a2',True)

#plot the histogram of the sample.
f.plthist(samp, 20, 'Distribution of the RNG', 'Result', 'Probability','hist1a',
        True)

np.savetxt('../seed.txt',R.state)














