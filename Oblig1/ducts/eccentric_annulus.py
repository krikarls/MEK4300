from dolfin import *
import numpy as np
from math import log as ln, sinh, pi

A = 5.0   # big circle radius
B = 1.0	  # small circle radius
C = 1.0	  # distance between centers 

# Create mesh
import mshr as mshh
big_circle = mshh.Circle(Point(0,0,0),A)
small_circle = mshh.Circle(Point(-1.0,0,0),B)
geometry = big_circle - small_circle
mesh1 = mshh.generate_mesh(geometry, 10)
mesh2 = mshh.generate_mesh(geometry, 20)
mesh3 = mshh.generate_mesh(geometry, 40)
mesh4 = mshh.generate_mesh(geometry, 80)

def solver(mesh,deg): 
	# Physical parameters
	dpdx = Constant(-1)
	mu = Constant(100)

	V = FunctionSpace(mesh, "Lagrange", deg)
	u = TrialFunction(V)
	v = TestFunction(V)

	# Mark boundary subdomians
	class Sides(SubDomain):
		def inside(self, x, on_boundry):
			return on_boundry

	side = Sides()

	mf = FacetFunction("size_t", mesh)
	mf.set_all(2)

	side.mark(mf, 1)
	noslip = DirichletBC(V, Constant(0), mf, 1)

	a = inner( grad(u), grad(v) )*dx 
	L = -1.0/mu*dpdx*v*dx

	u_ = Function(V)
	solve(a == L, u_, bcs=noslip)

	# Compute the flux 
	Q = assemble(u_*dx)

	# Flux from analytical expression
	mu = 100; dpdx = -1
	F = (A**2 - B**2 + C**2)/(2*C)
	M = sqrt(F**2 - A**2)
	alpha = 0.5*ln((F + M)/(F - M))
	beta = 0.5*ln((F - C + M)/(F - C - M))
	s = 0
	for n in range(1,100):
		s += (n*exp(-n*(beta + alpha)))/sinh(n*beta - n*alpha)

	Q_analytical = (pi/(8*mu)) * (-dpdx) * (A**4 - B**4 - (4*C*C*M*M)/(beta - alpha) - 8*C*C*M*M*s)

	Q_error = abs(Q-Q_analytical)

	print 'Flux computed numerically : ', Q
	print 'Flux computed using (3-52): ', Q_analytical

	return mesh.hmin(), Q_error


deg = 1

h = [0,0,0,0]; E = [0,0,0,0]

h[0],E[0] = solver(mesh1,deg)
h[1],E[1] = solver(mesh2,deg)
h[2],E[2] = solver(mesh3,deg)	
h[3],E[3] = solver(mesh4,deg)

print E

print ''
print 'Polynomial degree = ', deg
print '        h     ', '        E      ', '     r'
for i in range(1,4):
	print h[i], E[i] , ln(E[i]/E[i-1])/ln(h[i]/h[i-1])


#plot(u_,title="Numerical")
#interactive()
