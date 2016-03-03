from dolfin import *
import numpy as np

A = 5.0   # big circle radius
B = 1.0	  # small circle radius
C = 2.0	  # distance between centers 

# Create mesh
import mshr as mshh
big_circle = mshh.Circle(Point(0,0,0),A)
small_circle = mshh.Circle(Point(-2.0,0,0),B)
geometry = big_circle - small_circle
mesh = mshh.generate_mesh(geometry, 60)

# Physical parameters
dpdx = Constant(-1)
mu = Constant(1)

V = FunctionSpace(mesh, "Lagrange", 1)
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
mu = 1; dpdx = -1
F = (A**2-B**2+C**2)/(2*C)
M = np.sqrt(F**2-A**2)
alpha = 0.5*ln((F+M)/(F-M)) 
beta = 0.5*ln((F-C+M)/(F-C-M))
K = (4*C**2*M**2)/(beta-alpha)

n = 10
for n in range(1,n+1):
	S = 0
	S += (n*exp(-n*(beta+alpha)))/sinh(n*beta-n*alpha) 
	print S
S = S*8*C**2*M**2

Q_analytical = pi/(8*mu)*(-dpdx)*( A**4 - B**4 - K - S) 

print 'Flux computed numerically : ', Q
print 'Flux computed using (3-52): ', Q_analytical

"""
file = File('annulus_velocity.pvd')
file << u_
"""

plot(u_,title="Numerical")
interactive()

