
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

print 'Give method: newton/picard/compare'
method = raw_input()

L = 1
N = 1000
mesh = IntervalMesh(N,0,L)

def Newtons_met():
	V = FunctionSpace(mesh,'CG',1)
	Q = FunctionSpace(mesh,'CG',1)
	W = V*Q

	def left(x,on_boundary):
	    return on_boundary and near(x[0],0)

	def right(x,on_boundary):
	    return on_boundary and near(x[0],L)

	HF = Function(W)  
	H,F = split(HF)   

	HF_ = Function(W) 

	HFt = TestFunction(W) 
	vh,vf = split(HFt)

	BC_H0 = DirichletBC(W.sub(0),0,left)
	BC_H1 = DirichletBC(W.sub(0),1,right)
	BC_F  = DirichletBC(W.sub(1),0,left)

	H_, F_ = split(HF_)

	Eq1 = H_*vf*dx - F_.dx(0)*vf*dx
	Eq2 = - inner(grad(H_),grad(vh))*dx + 2*F_*H_.dx(0)*vh*dx + vh*dx - H_*H_*vh*dx 
	
	Eq = Eq1+Eq2

	solve(Eq == 0,HF_,bcs=[BC_H0,BC_H1,BC_F])

	H_, F_ = HF_.split(True)
	F_newton = F_.vector().array()

	#plot(F_,interactive=True,title='Newtons method')

	return F_newton



def Picard_met():
	V = FunctionSpace(mesh,'Lagrange',1)
	Q = FunctionSpace(mesh,'Lagrange',1)
	W = V*Q

	def left(x,on_boundary):
	    return on_boundary and near(x[0],0)

	def right(x,on_boundary):
	    return on_boundary and near(x[0],L)

	bcH0 = DirichletBC(W.sub(0),0,left)
	bcH1 = DirichletBC(W.sub(0),1,right)
	bcF  = DirichletBC(W.sub(1),0,left)

	HF = TrialFunction(W)  
	H,F = split(HF)   

	HF_ = Function(W) 

	HFt = TestFunction(W) 
	vh,vf = split(HFt)

	HFstar = Function(W)
	Hstar,Fstar = split(HFstar)

	Hstar = interpolate(Expression('x[0]'),FunctionSpace(mesh,'Lagrange',3))

	epsilon = 1
	i = 0

	Eq1 = H*vf*dx - F.dx(0)*vf*dx
	Eq2	 = - inner(grad(H),grad(vh))*dx + 2*F*Hstar.dx(0)*vh*dx + vh*dx - H*Hstar*vh*dx 
	Eq = Eq1+Eq2

	while epsilon > 1e-12 and i < 100:
		i += 1
	
		solve(lhs(Eq)==rhs(Eq),HF_,bcs=[bcH0,bcH1,bcF])
		epsilon = errornorm(HF_,HFstar)
		HFstar.assign(HF_)

	print 'Error after ', i, ' iterations is: ', epsilon

	H_, F_ = HF_.split(True)
	F_picard = F_.vector().array()
	#plot(F_,interactive=True,title='Picard')

	return F_picard


import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(0,L,N+1)

if method == 'newton':
	Y_n = Newtons_met()
	plt.plot(X,Y_n[::-1])
	plt.title('Solution using Newtons method with {} elements'.format(N), {'color': 'k', 'fontsize': 18})
	plt.xlabel('x')
	plt.show()
elif method == 'picard':
	Y_p = Picard_met()
	plt.plot(X,Y_p[::-1])
	plt.title('Solution using Picard with {} elements'.format(N), {'color': 'k', 'fontsize': 18})
	plt.xlabel('x')
	plt.show()
elif method == 'compare':
	Y_n = Newtons_met()
	Y_p = Picard_met()
	plt.plot(X,Y_n[::-1],label='Newtons method')
	plt.plot(X,Y_p[::-1],label='Picard')
	plt.legend(loc=4)
	plt.title('Axisymmetric stagnation flow \n Solution with {} elements '.format(N), {'color': 'k', 'fontsize': 18})
	plt.xlabel('x')
	plt.show()
else:
	print 'No valid option given'
