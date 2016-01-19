
# MEK4300 - Couette flow(moving upper plate)

from dolfin import *

N = 10	 # number of elements
h = 1.0	 # channel hight
U = 1.0  # upper plate velocity

mesh = IntervalMesh(N, -h, h)			# create mesh
V = FunctionSpace(mesh, 'Lagrange', 1)	# set up function space

u = TrialFunction(V)					# define trail- and test function
v = TestFunction(V)

# define boundary conditions as python functions
def bottom(x, on_boundary):
	return near(x[0],-h) and on_boundary

def top(x, on_boundary):
	return near(x[0],h) and on_boundary

BCs = [DirichletBC(V, 0, bottom), DirichletBC(V, U, top)]

u_ = Function(V)

solve(-inner(grad(u), grad(v))*dx ==  Constant(0)*v*dx, u_ , bcs=BCs)	# solving the variational problem

u_exact = project(Expression("U/2*(1+x[0]/h)", U=U, h=h), V)

plot(u_ , title="Couette flow")
plot(u_ - u_exact , title="Absolute error")
interactive()   
