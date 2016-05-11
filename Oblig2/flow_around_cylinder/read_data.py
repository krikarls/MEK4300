from numpy import *
import matplotlib.pyplot as plt

M = loadtxt('full_simulation_fine.txt')

t = M[:,0]; Cd = M[:,1]; Cl = M[:,2]; dP = M[:,3]
U_m = 1.5; U_ = (2/3.)*U_m; D = .1


plt.figure(1)
plt.plot(t,Cd)
plt.title('$C_D(t)$',fontsize=20)
plt.xlabel('t',fontsize=18)
plt.ylim((2,3.5))

plt.figure(2)
plt.plot(t,Cl)
plt.title('$C_L(t)$',fontsize=20)
plt.xlabel('t',fontsize=18)
#plt.ylim((2,5))

plt.figure(3)
plt.plot(t,dP)
plt.title('$\Delta P(t)$',fontsize=20)
plt.xlabel('t',fontsize=18)
plt.ylim((1.5,3))


plt.figure(4)
plt.plot(t,Cd,label='$C_D(t)$')
plt.plot(t,Cl,label='$C_L(t)$')
plt.plot(t,dP,label='$\Delta P(t)$')
plt.title('Simulated quantities for $t \in [0,8]$ ',fontsize=20)
plt.xlabel('t',fontsize=18)
plt.legend()
plt.ylim((-1.5,4))


plt.figure(5)
plt.plot(t[10000:],Cd[10000:],label='$C_D(t)$')
plt.plot(t[10000:],0.1*Cl[10000:]+3.205,label='$A+C_L(t)$')
plt.title('Time trace of $C_D$ and $C_L$ ',fontsize=20)
plt.axvline(x=7.305,  linewidth=1.0, color='r')
plt.axvline(x=7.6351,  linewidth=1.0, color='r')
plt.xlabel('t',fontsize=18)
plt.legend()
plt.ylim((3.,3.4))


def find_maxima(a):
	maxima_indices = []
	for i in range(6000,len(a)-1):
		if a[i] > a[i-1] and a[i]>a[i+1]:
			maxima_indices.append(i)
	return maxima_indices

Cl_maxima = find_maxima(Cl)

"""
for j in range(0,len(Cl_maxima)-1):
	T = t[Cl_maxima[j+1]]-t[Cl_maxima[j]]
	print 'Period: ', T
	f = T**-1
	print 'St. number: ' , D*f/U_
	print 'Pressure difference at C_L maxima: ', dP[Cl_maxima[j]]
"""

t0 = 8 # pick start of period refering to eigth maxima of C_L
print 'CD_max :' , amax(Cd[4000:])
print 'CL_max :' , amax(Cl[4000:])
T = t[Cl_maxima[t0+1]]-t[Cl_maxima[t0]]
print 'Period: ', T
f = T**-1
print 'St. number: ' , D*f/U_
px = int((Cl_maxima[t0]+Cl_maxima[t0+1])/2.0)  #index to evaluate dP in
print 'Pressure difference at t0 + 1/2*T: ', dP[px]

plt.show()