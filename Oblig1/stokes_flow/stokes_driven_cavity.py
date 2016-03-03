
### Plate driven Stokes flow in closed square ###
from dolfin import *
import numpy as np

# Create mesh
import mshr as mshh
geometry = mshh.Rectangle(Point(0,0,0),Point(1,1,0))
mesh = mshh.generate_mesh(geometry, 55)

# Set BC values
u_plate = Expression(("1", "0.0"))
zero_velocity = Expression(("0.0", "0.0"))

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

class NoSlip(SubDomain):
	def inside(self, x, on_boundry):
		return on_boundry

class UpperPlate(SubDomain):
	def inside(self, x, on_boundry):
		return (x[1]-1 < 1e-6) and (x[1]-1 > -1e-6) and on_boundry

noslip = NoSlip()
plate = UpperPlate()

mf = FacetFunction("size_t", mesh)
mf.set_all(2)

noslip.mark(mf, 0)
plate.mark(mf, 1)

bc0 = DirichletBC(W.sub(0), u_plate, mf, 1)
bc1 = DirichletBC(W.sub(0), zero_velocity, mf, 0)
bcs = [bc1, bc0]

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

mu = Constant(100)
a = (mu*inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
L = rhs(a) # to make system solvable for FEniCS without a source term f 

# Compute solution
w = Function(W)
solve(a == L, w, bcs)

(u, p) = w.split()

def stream_function(u):
  V = u.function_space().sub(0).collapse()

  psi = TrialFunction(V)
  phi = TestFunction(V)

  a = inner(grad(psi), grad(phi))*dx
  L = inner(u[1].dx(0) - u[0].dx(1), phi)*dx
  bc = DirichletBC(V, Constant(1.), DomainBoundary())

  psi = Function(V)
  solve(a==L,psi,bcs=bc)

  PSI_min = psi.vector().array().argmin()

  V2 = FunctionSpace(mesh,'CG',2)
  X = interpolate(Expression("x[0]"),V2)
  Y = interpolate(Expression("x[1]"),V2)

  print 'Location of vortex [x,y]: ', X.vector()[PSI_min], Y.vector()[PSI_min]

  return psi

PSI = stream_function(u)
plot(PSI,interactive=True)

"""
file = File('psi_square.pvd')
file << psi_

# Plot solution
plot(u)
plot(p)
plot(psi_)
interactive()
"""