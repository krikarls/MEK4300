import mshr as mshh 
import numpy as np
import pylab
from dolfin import *

####  Rotating cylinders  ####

# physical values
u0 = Constant(0)
u1 = Constant(1)

# Create mesh
c1 = mshh.Circle(Point(0, 0, 0),  1.0)
c2 = mshh.Circle(Point(0, 0, 0),  3.0)

geometry = c2-c1
mesh = mshh.generate_mesh(geometry, 8)

# function spaces and functions
V = FunctionSpace(mesh, "Lagrange", 1)
u = TrialFunction(V)
v = TestFunction(V)

class Inner(SubDomain):
	def inside(self, x, on_boundry):
		return on_boundry

class Outer(SubDomain):
	def inside(self, x, on_boundry):
		return (sqrt(x[0]*x[0] + x[1]*x[1]) > 1.5) and on_boundry

out_circ = Outer()
inn_circ = Inner()
mf = FacetFunction("size_t", mesh)
mf.set_all(3)

inn_circ.mark(mf, 0)
out_circ.mark(mf, 2)
plot(mf, interactive=True)

bc0 = DirichletBC(V, u0, inn_circ)
bc1 = DirichletBC(V, u1, out_circ)

# variational problem
r = Expression("sqrt(x[0]*x[0]+x[1]*x[1])")		# define spatial variable
f = Constant(0)									# define constant

F = inner(grad(v), grad(u))*r*dx + u*v/r*dx == f*v*r*dx

u_ = Function(V)	# to store solution

solve(F, u_, bcs=[bc0, bc1])	# solving

plot(u_, interactive = True)





