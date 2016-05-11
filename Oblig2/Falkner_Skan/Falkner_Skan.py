
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

L = 6
N = 1000
mesh = IntervalMesh(N,0,L)

def Newtons_met(beta):
	V = FunctionSpace(mesh,'CG',1)
	Q = FunctionSpace(mesh,'CG',1)
	W = V*Q

	def left(x,on_boundary):
	    return on_boundary and near(x[0],0)

	def right(x,on_boundary):
	    return on_boundary and near(x[0],L)

	HF = Function(W)  
	H,F = split(HF)   

	HF_ = interpolate(Expression(('1.0','1.0'),L=L), W) 

	HFt = TestFunction(W) 
	vh,vf = split(HFt)

	BC_H0 = DirichletBC(W.sub(0),0,left)
	BC_H1 = DirichletBC(W.sub(0),1,right)
	BC_F  = DirichletBC(W.sub(1),0,left)

	H_, F_ = split(HF_)

	beta = Constant(beta)

	Eq1 = H_*vf*dx - F_.dx(0)*vf*dx
	Eq2 = - inner(grad(H_),grad(vh))*dx + F_*H_.dx(0)*vh*dx + beta*(vh*dx - H_*H_*vh*dx) 
	
	Eq = Eq1+Eq2

	solve(Eq == 0,HF_,bcs=[BC_H0,BC_H1,BC_F])

	H_, F_ = HF_.split(True)

	U = FunctionSpace(mesh,'CG',1)
	H2 = project(H_.dx(0), U)

	#plot(H2,interactive=True)

	h2 = H2.vector().array()

	F_newton = F_.vector().array()
	H_newton = H_.vector().array()


	#plot(F_,interactive=True,title='Newtons method')

	return H_newton, h2

X = np.linspace(0,L,N+1)

#beta = -0.198838

Y1, Z1 = Newtons_met(1.0)
Y2, Z2 = Newtons_met(0.3)
Y3, Z3 = Newtons_met(0.0)
Y4, Z4 = Newtons_met(-0.1)
Y5, Z5 = Newtons_met(-0.18)
Y6, Z6 = Newtons_met(-0.198838)

plt.figure(1)
plt.plot(X,Y1[::-1],label='$\\beta = 1.0$')
plt.plot(X,Y2[::-1],label='$\\beta = 0.3$')
plt.plot(X,Y3[::-1],label='$\\beta = 0.0$')
plt.plot(X,Y4[::-1],label='$\\beta = -0.1$')
plt.plot(X,Y5[::-1],label='$\\beta = -0.18$')
plt.plot(X,Y6[::-1],label='$\\beta = -0.198838$')
plt.title('$f\'(\eta)$', fontsize=22)
plt.xlabel('$\eta$',fontsize=18)
plt.legend(loc='southeast')
plt.figure(2)
plt.plot(X,Z1[::-1],label='$\\beta = 1.0$')
plt.plot(X,Z2[::-1],label='$\\beta = 0.3$')
plt.plot(X,Z3[::-1],label='$\\beta = 0.0$')
plt.plot(X,Z4[::-1],label='$\\beta = -0.1$')
plt.plot(X,Z5[::-1],label='$\\beta = -0.18$')
plt.plot(X,Z6[::-1],label='$\\beta = -0.198838$')
plt.title('$f\'\'(\eta)$ ',fontsize=22)
plt.xlabel('$\eta$',fontsize=18)
plt.legend()

plt.show()

