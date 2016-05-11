from fenics import *
import numpy as np
import matplotlib.pyplot as plt

mesh = IntervalMesh(1100,0,1)

V = FunctionSpace(mesh,'Lagrange',1)
u = Function(V)
v = TestFunction(V)

bndry = lambda x, on_boundary: on_boundary and near(x[0],0)
bcs = DirichletBC(V,Constant(0),bndry)	

# constant parameters
h = 2.0
K = 0.5
A = 26.0
v_str = 0.05
nu = v_str/1000
dpdx = Constant(2*0.05*0.05/h)

l = Expression(('K*x[0]*(1-exp(-(1000.0*x[0])/A))'), K=K, A=A)

F = -nu*inner(u.dx(0),v.dx(0))*dx + dpdx*v*dx -l*l*inner(abs(u.dx(0))*u.dx(0),v.dx(0))*dx

solve(F == 0,u,bcs)


u_array = u.vector().array()
y = np.linspace(0,1,len(u_array))
u_full = np.zeros(2*len(u_array))
u_full[:len(u_array)] = u_array[::-1]
u_full[len(u_array):] = u_array
y_full = np.linspace(0,2,len(u_full))
u_poiseuille = 1.2*y_full*(h-y_full)


plt.figure(1)
plt.title('Turbulent plane channel flow', fontsize=18)
plt.xlabel('$y$', fontsize=18)
plt.plot(y_full,u_full,label='Turbulent flow')
#plt.plot(y_full,u_poiseuille,label='Poiseuille flow')
#plt.legend()

plt.figure(2)
plt.plot(y,u_array[::-1],label='numerical')
u1 = v_str*1000*y[:6]
plt.plot(y[:6],u1,'r',linewidth=2.0 ,label='exact')
u2 = (v_str/K)*np.log(1000*y[33:]) + 5.5*v_str
plt.plot(y[33:],u2,'r')
plt.title('Mixed length model vs exact, $\kappa=0.5$',fontsize=18)
plt.xlabel('$y$', fontsize=18)
plt.legend(loc=4)
"""
plt.figure(3)
plt.plot(y,u_array[::-1],label='numerical')
u1 = v_str*1000*y[:6]
plt.plot(y[:6],u1,'r',linewidth=2.0 ,label='exact')
u2 = (v_str/K)*np.log(1000*y[33:]) + 5.2*v_str
plt.plot(y[33:],u2,'r')
plt.title('Mixed length model vs exact, B=5.2',fontsize=18)
plt.xlabel('$y$', fontsize=18)
plt.legend(loc=4)
"""
plt.show()


