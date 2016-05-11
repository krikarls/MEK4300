from fenics import *
import numpy as np

mesh = Mesh("fine_karman.xml")

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define functions and function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
VQ = V * Q

up = Function(VQ)
u, p = split(up)
v, q = TestFunctions(VQ)
up_ = Function(VQ)

# Set parameter values
rho = Constant(1.0); nu = Constant(1.0e-3); U_m = 0.3;U_ = (2/3.)*U_m; D = .1

p_out = Constant(0)
u_in = Expression(('4.0*U_m*x[1]*(0.41-x[1])/(0.41*0.41)','0.0'),U_m=U_m)
zero_velocity = Expression(("0.0", "0.0"))

# Define boundary conditions
class NoSlip(SubDomain):
    def inside(self, x, on_boundry):
        return on_boundry

class Inflow(SubDomain):
    def inside(self, x, on_boundry):
        return (x[0] < 1e-6) and on_boundry

class Outflow(SubDomain):
    def inside(self, x, on_boundry):
        return (x[0]-2.2 < 1e-6) and (x[0]-2.2 > -1e-6) and on_boundry

class Cylinder(SubDomain):
    def inside(self, x, on_boundry):
        return ( (x[0]-0.2)*(x[0]-0.2)+(x[1]-0.2)*(x[1]-0.2) < 0.051*0.051) and on_boundry

mf = FacetFunction("size_t", mesh)
mf.set_all(4)

noslip = NoSlip(); noslip.mark(mf, 0)
inflow = Inflow(); inflow.mark(mf, 1)
outflow = Outflow(); outflow.mark(mf, 2)
cylinder = Cylinder(); cylinder.mark(mf, 3)
plot(mf, interactive=True)

BC_u  = DirichletBC(VQ.sub(0), zero_velocity, mf, 0)
U_in  = DirichletBC(VQ.sub(0), u_in, mf, 1)
BC_cyl = DirichletBC(VQ.sub(0), zero_velocity, mf, 3)

bcs = [BC_u, U_in, BC_cyl]

n = FacetNormal(mesh); ds=ds[mf] 
f = Constant((0, 0))

# Create files for storing solution
ufile = File("results/stdy_velocity.pvd")
pfile = File("results/stdy_pressure.pvd")

F = inner(dot(grad(u),u),v)*dx + nu*inner(grad(u),grad(v))*dx - inner(p,div(v))*dx - inner(q,div(u))*dx

# Compute velocity and pressure
solve(F == 0, up, bcs, solver_parameters={'newton_solver':{'maximum_iterations': 15}})

# Split the mixed solution 
u, p = up.split()

# Print size of system
print 'Number of unknowns: ', VQ.dim()

# Compute forces 
tau = -p*Identity(2)+nu*(grad(u)+grad(u).T)
total_force = dot(tau, n)

Drag = -assemble(total_force[0]*ds(3))
Lift = -assemble(total_force[1]*ds(3))

CD = 2*Drag / (U_**2*D)
CL = 2*Lift / (U_**2*D)
print 'Drag coefficient:',CD
print 'Lift coefficient:',CL

circ_area = np.linspace(0.25,1.0,1000)
for i in range(0,len(circ_area)):
	u_x = u[0](np.array([circ_area[i],0.20]))
	if u_x > 0.0 :
		x_r = circ_area[i]; break

print 'Length of circulation: ', x_r-0.25
print 'Pressure difference: ', p(np.array([0.15,0.20]))-p(np.array([0.25,0.20]))

plot(u,interactive=True)
plot(p,interactive=True)
	