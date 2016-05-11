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


def Newtons_met2(beta):
	V = FunctionSpace(mesh,'CG',1)
	Q = FunctionSpace(mesh,'CG',1)
	W = V*Q

	def left(x,on_boundary):
	    return on_boundary and near(x[0],0)

	def right(x,on_boundary):
	    return on_boundary and near(x[0],L)

	HF = Function(W)  
	H,F = split(HF)   

	HF_ = interpolate(Expression(('x[0]/L','0'),L=L), W) 

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


Y4, Z4 = Newtons_met(-0.1)
Y5, Z5 = Newtons_met2(-0.1)


plt.figure(1)
plt.plot(X,Y4[::-1],label='$\\beta = -0.1$')
plt.plot(X,Y5[::-1],label='$\\beta = -0.1$')
plt.title('$f\'(\eta)$', fontsize=22)
plt.xlabel('$\eta$',fontsize=18)
plt.legend(loc='northeast')
plt.figure(2)
plt.plot(X,Z4[::-1],label='$\\beta = -0.1$')
plt.plot(X,Z5[::-1],label='$\\beta = -0.1$')
plt.title('$f\'\'(\eta)$ ',fontsize=22)
plt.xlabel('$\eta$',fontsize=18)
plt.legend(loc='northwest')


plt.show()
