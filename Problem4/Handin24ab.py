import functions24 as f4
Om = 0.3
Ol = 0.7
#----------------------------4(a)------------------------

#The result according to Wolfram is 0.000131053, so this result is
#quite close to the value wanted. Furthermore, Romberg integration until
#20th order should be way better than a relative accuracy of 10^-5, even on
#an interval as big as the given.
integralr = f4.romberg(f4.integrand,50,1000000,20)

#Wolfram gives 0.019607780428266..., and it is independent of H0
D50 = integralr * Om * (5/2.) * f4.H2(50)**(1./2.)

print('D(z=50) =',D50)

#----------------------------4(b)------------------------

#Analytical expression seen in pdf
H0 = 70 #km/s/Mpc, as stated in 3.

ans2 = f4.ddot(1/51, integralr)

#Trying to find it using numerical methods as well.
dDdz = f4.Ridders(f4.D1, 50, .1, 2., 20)
dDdt = -dDdz * 51 * H0 * f4.H(50)

print('Analytical dDdt =',ans2)
print('Numerical dDdt =',dDdt)




