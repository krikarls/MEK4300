
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

print 'Give method: newton/picard'
method = raw_input()

L = 1
N = 100
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

	Eq1 = - inner(grad(H_),grad(vh))*dx + F_*H_.dx(0)*vh*dx - H_*H_*vh*dx + 1*vh*dx
	Eq2 = H_*vf*dx - F_.dx(0)*vf*dx
	Eq = Eq1+Eq2

	solve(G == 0,HF_,bcs=[BC_H0,BC_H1,BC_F])

	#F_newton = F_.vector().array()
	#print F_newton

	plot(F_,interactive=True,title='Newtons method')

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

	Hstar = interpolate(Expression("x[0]"),V)

	epsilon = 1
	i = 0

	while epsilon > 1e-8 and i < 100:
		i += 1
		G1 = - inner(grad(H),grad(vh))*dx + F*Hstar.dx(0)*vh*dx \
		     - H*Hstar*vh*dx + 1*vh*dx

		G2 = H*vf*dx - F.dx(0)*vf*dx
		G = G1+G2

		solve(lhs(G)==rhs(G),HF_,bcs=[bcH0,bcH1,bcF])
		H_, F_ = HF_.split()
		epsilon = errornorm(H_,Hstar,"h1")
		HFstar.assign(HF_)
		Hstar,Fstar = HFstar.split()

	print 'Error after ', i, ' iterations is: ', epsilon

	plot(F_,interactive=True,title='Picard')



if method == 'newton':
	Newtons_met()
elif method == 'picard':
	Picard_met()
else:
	Picard_met()
	Newtons_met()
