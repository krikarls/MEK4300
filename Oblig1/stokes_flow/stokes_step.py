from dolfin import *

mesh = Mesh("step.xml")
mesh = refine(mesh)
mesh = refine(mesh)

zero_velocity = Expression(("0.0", "0.0"))
upper_velocity = Expression(("1.0", "0.0"))
#upper_velocity = Expression(("-1.0", "0.0"))  # use for reversed velocity

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

class NoSlip(SubDomain):
	def inside(self, x, on_boundry):
		return on_boundry

class Inflow(SubDomain):
	def inside(self, x, on_boundry):
		return (x[0] < 1e-6) and on_boundry

class UpperPlate(SubDomain):
	def inside(self, x, on_boundry):
		return (x[1]-0.5 < 1e-6) and (x[1]-0.5 > -1e-6) and on_boundry

class Outflow(SubDomain):
	def inside(self, x, on_boundry):
		return (x[0]-1 < 1e-6) and (x[0]-1 > -1e-6) and on_boundry

noslip = NoSlip()
inflow = Inflow()
upper = UpperPlate()
outflow = Outflow()

mf = FacetFunction("size_t", mesh)
mf.set_all(4)

noslip.mark(mf, 0)
inflow.mark(mf, 1)
outflow.mark(mf, 2)
upper.mark(mf, 3)

bc2 = DirichletBC(W.sub(0), zero_velocity, mf, 0)
bc3 = DirichletBC(W.sub(0), upper_velocity, mf, 3)

# Collect boundary conditions
bcs = [bc2, bc3]

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

mu = Constant(100)
a = (mu*inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
L = rhs(a) # to make system solvable for FEniCS without a source term f 

# Compute solution
w = Function(W)
solve(a == L, w, bcs)

# Split the mixed solution 
(u, p) = w.split()

# Compute the flux over the inlet and the outlet
n = FacetNormal(mesh)
ds = ds[mf]

inlet_flux = dot(u,n)*ds(1) 
in_flux = assemble(inlet_flux) 

outlet_flux = dot(u,n)*ds(2) 
out_flux = assemble(outlet_flux) 

print 'Inlet flux: ', in_flux
print 'Outlet flux: ',out_flux
print 'Difference in influx/outflux:', in_flux+out_flux

# Normal stress 
pressure = -Identity(2)*p
normal_stress = dot(dot(pressure,n),n)*ds(0) 
print 'Normal stress: ',assemble(normal_stress)

def stream_function(u):
  V = u.function_space().sub(0).collapse()

  psi = TrialFunction(V)
  phi = TestFunction(V)

  grad_psi = as_vector((-u[1],u[0]))

  a = inner(grad(psi), grad(phi))*dx
  L = inner(u[1].dx(0) - u[0].dx(1), phi)*dx + phi*dot(grad_psi,n)*ds

  psi = Function(V)
  solve(a==L,psi)

  PSI_min = psi.vector().array().argmin()
  # PSI_min = psi.vector().array().argmax() # use if reversed

  V2 = FunctionSpace(mesh,'CG',2)
  X = interpolate(Expression("x[0]"),V2)
  Y = interpolate(Expression("x[1]"),V2)

  print 'Location of vortex [x,y]: ', X.vector()[PSI_min], Y.vector()[PSI_min]


  return psi

PSI = stream_function(u)
plot(PSI,interactive=True)

""" to visualize results
PSI = stream_function(u)

file = File('psi_step.pvd')
file << PSI

file = File('velocity_step.pvd')
file << u

file = File('pressure_step.pvd')
file << p

plot(PSI)
plot(u,range_min=0.0,range_max=2.0)
plot(p)
interactive()
"""
